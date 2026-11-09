# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""One AOTI runtime with a passive, process-local update consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path
from threading import Event, Thread

import torch

# Reuse the reference demo's registration order, inputs and numerical checks.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# isort: off
import export_and_verify as reference  # noqa: E402
from dynamicemb.exportable_embedding import (
    EmbeddingCollectionUpdateSubscriber,
)  # noqa: E402

# isort: on


def serve(args, updates, control):
    device = torch.device("cuda", torch.cuda.current_device())
    inference_stream = torch.cuda.default_stream(device)
    loader, layers, indexers = reference.load_embedding_collection_aot(
        args.package_dir, device
    )
    subscriber = EmbeddingCollectionUpdateSubscriber(
        str(args.package_dir),
        indexers,
        reference.loaded_layer_bindings(layers),
        indexers.device_index,
    )
    applied = [Event(), Event()]
    update_done = Event()
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
                        update = reference.EmbeddingCollectionUpdate.from_json(payload)
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
            update_done.set()

    def check_update_error():
        if failures:
            raise RuntimeError("Update receiver failed") from failures[0]

    worker = Thread(target=receive_updates, name="update-receiver", daemon=True)
    worker.start()
    reports = []

    def verify(round_id, inputs, fixture, eviction_inputs=None):
        expected = torch.load(fixture, map_location="cpu", weights_only=True)
        for _ in range(reference.REQUESTS_PER_VERIFIED_ROUND):
            outputs = loader.run(list(inputs))
            if not reference.outputs_match(
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
            if not reference.linear_evictions_miss(
                loader, 2, query, f"{args.worker_id}_evictions"
            ):
                raise AssertionError("Evicted LinearHashMap keys must miss")
        # Same phase boundary as the reference demo; no per-lookup sync is added.
        torch.cuda.synchronize(device)
        actual = [t.detach().cpu() for t in outputs]
        report = {
            "worker_id": args.worker_id,
            "pid": os.getpid(),
            "phase": round_id,
            "requests": reference.REQUESTS_PER_VERIFIED_ROUND,
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
        verify("initial", reference.full_inputs(device), args.fixtures / "initial.pt")
        applied[0].wait()
        check_update_error()
        verify(
            "first_update",
            reference.post_update_inputs(device, 1),
            args.fixtures / "first_update.pt",
            args.fixtures / "evictions.pt",
        )
        # Test-control traffic is separate from the payload/ACK update channel.
        if control.recv() != "start_concurrent":
            raise ValueError("Expected start_concurrent test command")
        inputs = reference.post_update_inputs(device, 2)
        for iteration in range(reference.ROUND_THREE_MIN_INFERENCES):
            loader.run(list(inputs))
            concurrent_calls[0] += 1
            if iteration == 0:
                control.send({"phase": "concurrent_started", "pid": os.getpid()})
        while not applied[1].is_set():
            check_update_error()
            loader.run(list(inputs))
            concurrent_calls[0] += 1
        for _ in range(reference.ROUND_THREE_POST_STAGE_INFERENCES):
            loader.run(list(inputs))
            concurrent_calls[0] += 1
        torch.cuda.synchronize(device)
        update_done.wait()
        check_update_error()
        verify("second_update", inputs, args.fixtures / "second_update.pt")

    worker.join()
    result = {
        "worker_id": args.worker_id,
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
    parser.add_argument("--updates-socket", required=True)
    parser.add_argument("--control-socket", required=True)
    args = parser.parse_args()
    with Listener(
        args.updates_socket, family="AF_UNIX", authkey=reference.COORDINATOR_AUTHKEY
    ) as update_listener:
        with Listener(
            args.control_socket, family="AF_UNIX", authkey=reference.COORDINATOR_AUTHKEY
        ) as control_listener:
            with update_listener.accept() as updates, control_listener.accept() as control:
                try:
                    serve(args, updates, control)
                except BaseException as error:
                    control.send({"error": repr(error), "pid": os.getpid()})
                    raise


if __name__ == "__main__":
    main()
