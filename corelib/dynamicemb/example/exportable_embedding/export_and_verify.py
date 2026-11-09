# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepare each delta once; deliver its update to separate inference runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from multiprocessing.connection import Client
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

# Register the native ops and metadata before importing the runtime APIs.
# isort: off
import workflow_utils as workflow
from dynamicemb.exportable_embedding import imported_nve_generation

# isort: on


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def connect(path, process):
    deadline = time.monotonic() + 180
    while process.poll() is None:
        try:
            return Client(
                str(path), family="AF_UNIX", authkey=workflow.COORDINATOR_AUTHKEY
            )
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise RuntimeError(f"Worker {process.pid} did not open {path}; inspect its log")


def receive(connection, process):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if connection.poll(0.1):
            value = connection.recv()
            if isinstance(value, dict) and "error" in value:
                raise RuntimeError(value["error"])
            return value
        if process.poll() is not None:
            raise RuntimeError(f"Worker {process.pid} exited with {process.returncode}")
    raise TimeoutError(f"Worker {process.pid} did not respond; inspect its log")


def local_redis(stack, work_dir):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [
            "redis-server",
            "--bind",
            "127.0.0.1",
            "--protected-mode",
            "yes",
            "--port",
            str(port),
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(work_dir),
            "--logfile",
            str(work_dir / "redis.log"),
        ]
    )
    stack.callback(stop_process, process)
    for _ in range(100):
        result = subprocess.run(
            ["redis-cli", "-p", str(port), "ping"], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip() == "PONG":
            return f"127.0.0.1:{port}"
        if process.poll() is not None:
            break
        time.sleep(0.05)
    raise RuntimeError("Private Redis did not start")


def prepare(work_dir, device, enable_redis):
    """Own the source/delta producers; the eager worker builds and exports its model."""
    names = workflow.BASE_COLLECTIONS + (
        workflow.REDIS_COLLECTIONS if enable_redis else ()
    )
    training = workflow.TrainingSparseModel(names, device)
    workflow.fill_stable_random_weights(training)
    checkpoint, producers = workflow.dump_training_artifacts(
        training, work_dir / "checkpoint", names, device
    )
    weights = workflow.checkpoint_weights(checkpoint, names)
    return names, weights, producers


def finish(work_dir, workers, publications, cpp_result, coordinator_process=None):
    for _, process, _, _ in workers:
        if process.wait(timeout=30) != 0:
            raise RuntimeError(f"Inference process {process.pid} failed")

    results = [json.loads((work_dir / f"{w[0]}.json").read_text()) for w in workers]
    for result in results:
        if [r["payload_hashes"] for r in result["received"]] != [
            p["payload_hashes"] for p in publications
        ]:
            raise AssertionError("A replica did not consume the published payloads")
        if publications:
            if result["received"][1]["calls_at_receive"] < 1:
                raise AssertionError(
                    "BitConcat update arrived before concurrent inference started"
                )
            if (
                result["concurrent_calls"]
                < workflow.ROUND_THREE_MIN_INFERENCES
                + workflow.ROUND_THREE_POST_STAGE_INFERENCES
            ):
                raise AssertionError("Concurrent inference coverage was shortened")
        if [r["output_hashes"] for r in result["rounds"]] != [
            r["output_hashes"] for r in results[0]["rounds"]
        ]:
            raise AssertionError("Inference replicas returned different vectors")
    summary = {
        "orchestrator_pid": os.getpid(),
        "coordinator_pid": coordinator_process.pid if coordinator_process else None,
        "coordinator_apply_delta_calls": sum(len(p["updates"]) for p in publications),
        "publications": publications,
        "replicas": results,
        "cpp_aoti": cpp_result,
        "result": "PASS",
    }
    (work_dir / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    mode = (
        "paused and concurrent updates"
        if publications
        else "initial export/replay only"
    )
    print(
        f"PASS: eager + {len(workers) - 1} Python AOTI + {int(cpp_result is not None)} C++ AOTI processes; "
        f"{len(results[0]['rounds'][0]['output_hashes'])} collections; {mode}",
        flush=True,
    )


def run(args):
    if args.replicas < 1:
        raise ValueError("--replicas must be positive")
    enable_redis = (
        imported_nve_generation() >= (26, 6) and not args.disable_redis_incremental
    )
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=0, world_size=1)
    with ExitStack() as stack:
        stack.callback(dist.destroy_process_group)
        redis_address = args.redis_address
        if enable_redis and redis_address is None:
            redis_address = local_redis(stack, work_dir)
        device = torch.device("cuda", torch.cuda.current_device())
        names, weights, producers = prepare(work_dir, device, enable_redis)
        # Short-lived local IPC paths; model/update artifacts remain in work_dir.
        sockets = Path(stack.enter_context(TemporaryDirectory(prefix="demb-workers-")))
        workers = []
        worker_specs = [("eager", "eager")] + [
            (f"aoti_{index}", "aoti") for index in range(args.replicas)
        ]
        for index, (worker_id, runtime) in enumerate(worker_specs):
            log = stack.enter_context((work_dir / f"{worker_id}.log").open("w"))
            update_socket, control_socket = sockets / f"u{index}", sockets / f"c{index}"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("inference_worker.py")),
                    "--package-dir",
                    str(work_dir / "package"),
                    "--fixtures",
                    str(work_dir),
                    "--worker-id",
                    worker_id,
                    "--runtime",
                    runtime,
                    *(["--redis-address", redis_address] if redis_address else []),
                    *([] if enable_redis else ["--initial-only"]),
                    "--updates-socket",
                    str(update_socket),
                    "--control-socket",
                    str(control_socket),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            stack.callback(stop_process, process)
            updates = stack.enter_context(connect(update_socket, process))
            control = stack.enter_context(connect(control_socket, process))
            workers.append((worker_id, process, updates, control))
            ready = receive(control, process)
            if ready.get("phase") != "ready" or ready.get("runtime") != runtime:
                raise AssertionError(f"Unexpected startup response from {worker_id}")
            print(
                f"READY {worker_id}: pid={process.pid}, runtime={runtime}", flush=True
            )

        def collect(phase):
            for worker_id, process, _, control in workers:
                result = receive(control, process)
                if result.get("phase") != phase:
                    raise AssertionError(
                        f"{worker_id}: expected {phase}, received {result}"
                    )

        cpp_process = None
        cpp_result = None
        subscriber_ids = {w[0] for w in workers}
        if args.cpp_replayer is not None:
            initial = torch.load(work_dir / "initial.pt", weights_only=True)
            cpp_result = {
                "worker_id": "cpp_aoti",
                "runtime": "cpp_aoti",
                "verification": "output-count and sum checks",
                "rounds_passed": [1],
                "acknowledgements": [],
            }
            if enable_redis:
                cpp_process, ok = workflow.start_incremental_cpp_replayer(
                    args.cpp_replayer.resolve(), work_dir / "package", initial
                )
                stack.callback(stop_process, cpp_process)
                subscriber_ids.add("cpp_aoti")
                cpp_result["pid"] = cpp_process.pid
                print(f"READY cpp_aoti: pid={cpp_process.pid}", flush=True)
            else:
                ok = workflow.run_cpp_replayer(
                    args.cpp_replayer.resolve(), work_dir / "package", initial
                )
            if not ok:
                raise AssertionError("C++ initial verification failed")

        if not enable_redis:
            for _, _, _, control in workers:
                control.send("start_initial")
            collect("initial")
            finish(work_dir, workers, [], cpp_result)
            return

        coordinator_process, coordinator = workflow.start_update_coordinator(
            work_dir / "package", work_dir / "updates", subscriber_ids
        )
        stack.callback(stop_process, coordinator_process)
        stack.enter_context(coordinator)
        publications = []

        def publish(deltas):
            # Exactly one coordinator call per logical delta, irrespective of replicas.
            batch = [
                workflow.coordinator_delta(coordinator, f"collections.{name}", delta)
                for name, delta in deltas
            ]
            payloads = [update.to_json() for update in batch]
            publications.append(
                {
                    "payload_hashes": [
                        hashlib.sha256(p.encode()).hexdigest() for p in payloads
                    ],
                    "updates": [
                        {"collection_id": u.collection_id, "snapshot_id": u.snapshot_id}
                        for u in batch
                    ],
                }
            )
            # Send to every replica BEFORE receiving ACKs. A mapping change needs
            # inference to resume and adopt it before retirement can complete.
            for _, _, updates, _ in workers:
                updates.send(payloads)
            return batch

        def acknowledge(batch, expected_values):
            expected = [(u.collection_id, u.snapshot_id) for u in batch]
            for worker_id, process, updates, _ in workers:
                acknowledgements = [
                    workflow.EmbeddingCollectionUpdateAck.from_json(p)
                    for p in receive(updates, process)
                ]
                if [
                    (a.collection_id, a.snapshot_id) for a in acknowledgements
                ] != expected:
                    raise AssertionError(f"Unexpected ACKs from {worker_id}")
                for ack in acknowledgements:
                    workflow.coordinator_acknowledge(coordinator, worker_id, ack)
            if cpp_process is not None:
                # After Python ACKs, the C++ process applies the same batch
                # and runs its own verification/inference loop.
                acknowledgements, ok = workflow.run_incremental_cpp_round(
                    cpp_process, batch, expected_values
                )
                if (
                    not ok
                    or [(a.collection_id, a.snapshot_id) for a in acknowledgements]
                    != expected
                ):
                    raise AssertionError("C++ verification or acknowledgement failed")
                for ack in acknowledgements:
                    workflow.coordinator_acknowledge(coordinator, "cpp_aoti", ack)
                cpp_result["acknowledgements"].append(
                    [json.loads(a.to_json()) for a in acknowledgements]
                )
                cpp_result["rounds_passed"].extend(
                    [2] if len(cpp_result["acknowledgements"]) == 1 else [3, 4]
                )

        for _, _, _, control in workers:
            control.send("start_initial")
        collect("initial")
        first_changed, deltas = {}, []
        for name in workflow.REDIS_COLLECTIONS:
            delta, first_changed[name] = workflow.produce_delta(
                name, producers[name], 1
            )
            deltas.append((name, delta))
            if name == workflow.LINEAR_HASH_REDIS:
                torch.save(
                    tuple(
                        t.cpu() for t in workflow.linear_eviction_inputs(delta, device)
                    ),
                    work_dir / "evictions.pt",
                )
        expected_one = workflow.expected_after_update(weights, names, first_changed, 1)
        torch.save(expected_one, work_dir / "first_update.pt")
        first_batch = publish(deltas)
        acknowledge(first_batch, expected_one)
        collect("first_update")

        for _, _, _, control in workers:
            control.send("start_concurrent")
        collect("concurrent_started")
        second_changed = {
            workflow.LINEAR_HASH_REDIS: {
                (table, 1): first_changed[workflow.LINEAR_HASH_REDIS][(table, 1)]
                for table in range(2)
            }
        }
        name = workflow.BIT_CONCAT_REDIS
        delta, second_changed[name] = workflow.produce_delta(name, producers[name], 2)
        expected_two = workflow.expected_after_update(weights, names, second_changed, 2)
        torch.save(expected_two, work_dir / "second_update.pt")
        second_batch = publish([(name, delta)])
        acknowledge(second_batch, expected_two)
        # Round 4 follows all update ACKs, including the optional C++ replayer.
        for _, _, _, control in workers:
            control.send("updates_done")
        collect("second_update")
        if cpp_process is not None and not workflow.stop_incremental_cpp_replayer(
            cpp_process
        ):
            raise AssertionError("C++ replayer failed")
        coordinator.send({"op": "stop"})
        if coordinator_process.wait(timeout=30) != 0:
            raise RuntimeError("Coordinator failed")
        finish(work_dir, workers, publications, cpp_result, coordinator_process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--replicas", type=int, default=1, help="Number of Python AOTI processes"
    )
    parser.add_argument("--cpp-replayer", type=Path, help="C++ AOTI replay executable")
    parser.add_argument("--redis-address", help="Omit to start an isolated local Redis")
    parser.add_argument("--disable-redis-incremental", action="store_true")
    run(parser.parse_args())
