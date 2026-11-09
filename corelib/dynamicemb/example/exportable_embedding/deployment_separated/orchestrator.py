# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Prepare each delta once and deliver it to independent AOTI inference processes."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_and_verify as reference  # noqa: E402


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
                str(path), family="AF_UNIX", authkey=reference.COORDINATOR_AUTHKEY
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


def prepare(work_dir, redis_address, device):
    """Offline setup reuses the unchanged demo's source, loading and export."""
    names = reference.BASE_COLLECTIONS + reference.REDIS_COLLECTIONS
    training = reference.TrainingSparseModel(names, device)
    reference.fill_stable_random_weights(training)
    checkpoint, producers = reference.dump_training_artifacts(
        training, work_dir / "checkpoint", names, device
    )
    weights = reference.checkpoint_weights(checkpoint, names)
    model = reference.construct_inference_model(
        training,
        reference.collection_configs(enable_redis=True, redis_address=redis_address),
        checkpoint,
        work_dir / "checkpoint" / "dynamicemb",
    )
    torch.save(
        reference.snapshot_zero_expected(weights, model), work_dir / "initial.pt"
    )
    keys = torch.export.Dim("base_key_count", min=2, max=514)
    redis_keys = torch.export.Dim("redis_key_count", min=2, max=514)
    reference.export_embedding_collection_aot(
        model,
        reference.full_inputs(device),
        work_dir / "package",
        dynamic_shapes=({0: keys}, None, {0: redis_keys}, None),
    )
    return names, weights, producers


def run(args):
    if args.replicas < 1:
        raise ValueError("--replicas must be positive")
    if reference.imported_nve_generation() < (26, 6):
        raise ValueError("Redis incremental updates require NVE 26.06 or later")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=0, world_size=1)
    with ExitStack() as stack:
        stack.callback(dist.destroy_process_group)
        redis_address = args.redis_address or local_redis(stack, work_dir)
        device = torch.device("cuda", torch.cuda.current_device())
        names, weights, producers = prepare(work_dir, redis_address, device)
        # Short-lived local IPC paths; model/update artifacts remain in work_dir.
        sockets = Path(stack.enter_context(TemporaryDirectory(prefix="demb-workers-")))
        workers = []
        for index in range(args.replicas):
            worker_id = f"aoti_{index}"
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
        coordinator_process, coordinator = reference.start_update_coordinator(
            work_dir / "package", work_dir / "updates", {w[0] for w in workers}
        )
        stack.callback(stop_process, coordinator_process)
        stack.enter_context(coordinator)
        publications = []

        def collect(phase):
            for worker_id, process, _, control in workers:
                result = receive(control, process)
                if result.get("phase") != phase:
                    raise AssertionError(
                        f"{worker_id}: expected {phase}, received {result}"
                    )

        def publish(deltas):
            # Exactly one coordinator call per logical delta, irrespective of replicas.
            batch = [
                reference.coordinator_delta(coordinator, f"collections.{name}", delta)
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

        def acknowledge(batch):
            expected = [(u.collection_id, u.snapshot_id) for u in batch]
            for worker_id, process, updates, _ in workers:
                acknowledgements = [
                    reference.EmbeddingCollectionUpdateAck.from_json(p)
                    for p in receive(updates, process)
                ]
                if [
                    (a.collection_id, a.snapshot_id) for a in acknowledgements
                ] != expected:
                    raise AssertionError(f"Unexpected ACKs from {worker_id}")
                for ack in acknowledgements:
                    reference.coordinator_acknowledge(coordinator, worker_id, ack)

        collect("initial")
        first_changed, deltas = {}, []
        for name in reference.REDIS_COLLECTIONS:
            delta, first_changed[name] = reference.produce_delta(
                name, producers[name], 1
            )
            deltas.append((name, delta))
            if name == reference.LINEAR_HASH_REDIS:
                torch.save(
                    tuple(
                        t.cpu() for t in reference.linear_eviction_inputs(delta, device)
                    ),
                    work_dir / "evictions.pt",
                )
        torch.save(
            reference.expected_after_update(weights, names, first_changed, 1),
            work_dir / "first_update.pt",
        )
        first_batch = publish(deltas)
        acknowledge(first_batch)
        collect("first_update")

        for _, _, _, control in workers:
            control.send("start_concurrent")
        collect("concurrent_started")
        second_changed = {
            reference.LINEAR_HASH_REDIS: {
                (table, 1): first_changed[reference.LINEAR_HASH_REDIS][(table, 1)]
                for table in range(2)
            }
        }
        name = reference.BIT_CONCAT_REDIS
        delta, second_changed[name] = reference.produce_delta(name, producers[name], 2)
        torch.save(
            reference.expected_after_update(weights, names, second_changed, 2),
            work_dir / "second_update.pt",
        )
        second_batch = publish([(name, delta)])
        acknowledge(second_batch)
        collect("second_update")
        coordinator.send({"op": "stop"})
        if coordinator_process.wait(timeout=30) != 0:
            raise RuntimeError("Coordinator failed")
        for _, process, _, _ in workers:
            if process.wait(timeout=30) != 0:
                raise RuntimeError(f"Inference process {process.pid} failed")

        results = [json.loads((work_dir / f"{w[0]}.json").read_text()) for w in workers]
        for result in results:
            if [r["payload_hashes"] for r in result["received"]] != [
                p["payload_hashes"] for p in publications
            ]:
                raise AssertionError("A replica did not consume the published payloads")
            if result["received"][1]["calls_at_receive"] < 1:
                raise AssertionError(
                    "BitConcat update arrived before concurrent inference started"
                )
            if (
                result["concurrent_calls"]
                < reference.ROUND_THREE_MIN_INFERENCES
                + reference.ROUND_THREE_POST_STAGE_INFERENCES
            ):
                raise AssertionError("Concurrent inference coverage was shortened")
            if [r["output_hashes"] for r in result["rounds"]] != [
                r["output_hashes"] for r in results[0]["rounds"]
            ]:
                raise AssertionError("Inference replicas returned different vectors")
        summary = {
            "orchestrator_pid": os.getpid(),
            "coordinator_pid": coordinator_process.pid,
            "coordinator_apply_delta_calls": sum(
                len(p["updates"]) for p in publications
            ),
            "publications": publications,
            "replicas": results,
            "result": "PASS",
        }
        (work_dir / "verification.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            f"PASS: {args.replicas} independent AOTI runtimes, four collections, three shared payloads, paused and concurrent updates",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--redis-address", help="Omit to start an isolated local Redis")
    run(parser.parse_args())
