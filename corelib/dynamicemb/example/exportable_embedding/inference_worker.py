# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""One eager or AOTI runtime with a passive, process-local update consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from multiprocessing.connection import Listener
from pathlib import Path
from threading import Event, Thread

import torch
import torch.distributed as dist

# Register the native ops and metadata before importing the runtime APIs.
# isort: off
import workflow_utils as workflow
from dynamicemb.exportable_embedding import (
    EmbeddingCollectionUpdateSubscriber,
    embedding_collection_indexers_from_model,
    export_embedding_collection_aot,
    load_embedding_collection_aot,
)  # noqa: E402

# isort: on


def serve(args, updates, control):
    device = torch.device("cuda", torch.cuda.current_device())
    inference_stream = torch.cuda.default_stream(device)
    if args.runtime == "eager":
        # Export the actual serving eager model so all runtimes share its mapping.
        # This startup finishes before any AOTI worker is launched.
        rendezvous = args.fixtures / "eager-rendezvous"
        rendezvous.unlink(missing_ok=True)
        dist.init_process_group(
            "nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1
        )
        names = workflow.BASE_COLLECTIONS + (
            () if args.initial_only else workflow.REDIS_COLLECTIONS
        )
        checkpoint = args.fixtures / "checkpoint" / "model.pt"
        runtime = workflow.construct_inference_model(
            workflow.TrainingSparseModel(names, device),
            workflow.collection_configs(
                enable_redis=not args.initial_only, redis_address=args.redis_address
            ),
            checkpoint,
            args.fixtures / "checkpoint" / "dynamicemb",
        )
        indexers = embedding_collection_indexers_from_model(runtime)
        if not args.initial_only:
            bindings = workflow.live_layer_bindings(runtime, indexers)
        torch.save(
            workflow.snapshot_zero_expected(
                workflow.checkpoint_weights(checkpoint, names), runtime
            ),
            args.fixtures / "initial.pt",
        )
        keys = torch.export.Dim("base_key_count", min=2, max=514)
        redis_keys = torch.export.Dim("redis_key_count", min=2, max=514)
        export_embedding_collection_aot(
            runtime,
            workflow.full_inputs(device),
            args.package_dir,
            dynamic_shapes=({0: keys}, None, {0: redis_keys}, None),
        )
    else:
        runtime, layers, indexers = load_embedding_collection_aot(
            args.package_dir, device
        )
        if not args.initial_only:
            bindings = workflow.loaded_layer_bindings(layers)
    if not args.initial_only:
        subscriber = EmbeddingCollectionUpdateSubscriber(
            str(args.package_dir),
            indexers,
            bindings,
            indexers.device_index,
        )
    control.send({"phase": "ready", "pid": os.getpid(), "runtime": args.runtime})
    applied = [Event(), Event()]
    failures = []
    received = []
    concurrent_calls = [0]

    def receive_updates():
        """Only update JSON enters this thread; only native ACK JSON leaves."""
        try:
            with torch.cuda.device(device):
                for batch in range(2):
                    payloads = updates.recv()
                    if not isinstance(payloads, list) or not payloads:
                        raise ValueError(
                            "Expected a nonempty batch of update JSON strings"
                        )
                    before = concurrent_calls[0]
                    for payload in payloads:
                        subscriber.apply_incremental_load(
                            payload, inference_stream.cuda_stream
                        )
                    after_apply = concurrent_calls[0]
                    # Resume round 2 before waiting for snapshot adoption/retirement.
                    # Round 3 already runs continuously when its batch arrives.
                    applied[batch].set()
                    acknowledgements = []
                    for payload in payloads:
                        update = workflow.EmbeddingCollectionUpdate.from_json(payload)
                        acknowledgements.append(
                            subscriber.wait_for_retirement(
                                update.collection_id, update.snapshot_id
                            )
                        )
                    received.append(
                        {
                            "payload_hashes": [
                                hashlib.sha256(p.encode()).hexdigest() for p in payloads
                            ],
                            "calls_at_receive": before,
                            "calls_at_apply_return": after_apply,
                            "calls_at_ack": concurrent_calls[0],
                        }
                    )
                    updates.send(acknowledgements)
        except BaseException as error:
            failures.append(error)
            try:
                updates.send({"error": repr(error)})
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            for event in applied:
                event.set()

    def check_update_error():
        if failures:
            raise RuntimeError("Update receiver failed") from failures[0]

    worker = None
    if not args.initial_only:
        worker = Thread(target=receive_updates, name="update-receiver", daemon=True)
        worker.start()
    reports = []

    def lookup(inputs):
        if args.runtime == "eager":
            return runtime(*inputs)
        return runtime.run(list(inputs))

    def verify(round_id, inputs, fixture, eviction_inputs=None):
        expected = torch.load(fixture, map_location="cpu", weights_only=True)
        for _ in range(workflow.REQUESTS_PER_VERIFIED_ROUND):
            outputs = lookup(inputs)
            if not workflow.outputs_match(
                outputs, expected, f"{args.worker_id}_{round_id}"
            ):
                raise AssertionError(
                    f"{args.worker_id}: {round_id} differs from reference"
                )
        if eviction_inputs is not None:
            query = tuple(
                t.to(device)
                for t in torch.load(
                    eviction_inputs, map_location="cpu", weights_only=True
                )
            )
            if not workflow.linear_evictions_miss(
                runtime, 2, query, f"{args.worker_id}_evictions"
            ):
                raise AssertionError("Evicted LinearHashMap keys must miss")
        # Verification boundary; continuous round 3 has no per-lookup sync.
        torch.cuda.synchronize(device)
        actual = [t.detach().cpu() for t in outputs]
        report = {
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "phase": round_id,
            "requests": workflow.REQUESTS_PER_VERIFIED_ROUND,
            "output_hashes": [
                hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in actual
            ],
            "output_sums": [float(t.sum()) for t in actual],
            "max_abs_error": max(
                float((a - b).abs().max()) for a, b in zip(actual, expected)
            ),
        }
        reports.append(report)
        control.send(report)

    with torch.cuda.stream(inference_stream), torch.inference_mode():
        if control.recv() != "start_initial":
            raise ValueError("Expected start_initial test command")
        verify("initial", workflow.full_inputs(device), args.fixtures / "initial.pt")
        if not args.initial_only:
            applied[0].wait()
            check_update_error()
            verify(
                "first_update",
                workflow.post_update_inputs(device, 1),
                args.fixtures / "first_update.pt",
                args.fixtures / "evictions.pt",
            )
            # Test-control traffic is separate from the payload/ACK update channel.
            if control.recv() != "start_concurrent":
                raise ValueError("Expected start_concurrent test command")
            inputs = workflow.post_update_inputs(device, 2)
            for iteration in range(workflow.ROUND_THREE_MIN_INFERENCES):
                lookup(inputs)
                concurrent_calls[0] += 1
                if iteration == 0:
                    control.send({"phase": "concurrent_started", "pid": os.getpid()})
            while not applied[1].is_set():
                check_update_error()
                lookup(inputs)
                concurrent_calls[0] += 1
            for _ in range(workflow.ROUND_THREE_POST_STAGE_INFERENCES):
                lookup(inputs)
                concurrent_calls[0] += 1
            torch.cuda.synchronize(device)
            # Final verification follows all ACKs, including C++; this gate
            # is after the continuous round-3 loop.
            if control.recv() != "updates_done":
                raise ValueError("Expected updates_done test command")
            check_update_error()
            verify("second_update", inputs, args.fixtures / "second_update.pt")

    if worker is not None:
        worker.join()
    result = {
        "worker_id": args.worker_id,
        "runtime": args.runtime,
        "pid": os.getpid(),
        "rounds": reports,
        "received": received,
        "concurrent_calls": concurrent_calls[0],
    }
    (args.fixtures / f"{args.worker_id}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--runtime", choices=("eager", "aoti"), default="aoti")
    parser.add_argument("--redis-address", help="Used only for eager setup/export")
    parser.add_argument("--initial-only", action="store_true")
    parser.add_argument("--updates-socket", required=True)
    parser.add_argument("--control-socket", required=True)
    args = parser.parse_args()
    with Listener(
        args.updates_socket, family="AF_UNIX", authkey=workflow.COORDINATOR_AUTHKEY
    ) as update_listener:
        with Listener(
            args.control_socket, family="AF_UNIX", authkey=workflow.COORDINATOR_AUTHKEY
        ) as control_listener:
            with update_listener.accept() as updates, control_listener.accept() as control:
                try:
                    serve(args, updates, control)
                except BaseException as error:
                    control.send({"error": repr(error), "pid": os.getpid()})
                    raise
                finally:
                    if dist.is_initialized():
                        dist.destroy_process_group()


if __name__ == "__main__":
    main()
