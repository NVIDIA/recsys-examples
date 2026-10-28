# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run retirement scenarios in bounded subprocesses so a regression cannot hang CI."""

import itertools
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_redis_incremental_removals as removal_tests
import torch
from test_redis_incremental_removals import (
    CACHE_BYTES,
    DIM,
    ROWS,
    Indexer,
    cold_ps_rows,
    lookup,
    redis_address,
    runtime,
)

from dynamicemb.exportable_embedding import EmbeddingCollectionUpdateAck

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA snapshot retirement"
)
CASES = [
    "oldest_first",
    "newest_first",
    "chain",
    "waiting",
    "already_adopted",
    "live_lookup",
    "duplicate_id",
]


def removal_delta(rt, removed, updated, value):
    return SimpleNamespace(
        table_names=rt.names,
        keys=[torch.tensor([updated]) for _ in rt.names],
        values=[torch.full((1, DIM), value + 10 * i) for i in range(len(rt.names))],
        evicted_keys=[None] * len(rt.names),
        erased_keys=[torch.tensor([removed]) for _ in rt.names],
    )


def stage(rt, delta):
    update = rt.coordinator.apply_delta(rt.id, delta)
    assert update.indexer_snapshot_path
    rt.subscriber.apply_incremental_load(
        update.to_json(), torch.cuda.current_stream().cuda_stream
    )
    return update


def acknowledge(rt, update):
    ack = EmbeddingCollectionUpdateAck.from_json(
        rt.subscriber.wait_for_retirement(rt.id, update.snapshot_id)
    )
    rt.coordinator.acknowledge("reader", ack)


def set_row(rt, values, key, value):
    for i in range(len(rt.names)):
        values[ROWS * i + key] = value + 10 * i


def pending_cases(rt, case):
    # First delta removes A; second removes B. Both upserts must survive even
    # when inference never adopts the intermediate mapping.
    first = stage(rt, removal_delta(rt, 1, 0, 20.0))
    expected_first = rt.weights.clone()
    expected_first[1::ROWS] = 0
    set_row(rt, expected_first, 0, 20.0)
    if case == "already_adopted":
        torch.testing.assert_close(lookup(rt), expected_first)
    result, errors = [], []
    started, done = threading.Event(), threading.Event()
    waiter = None
    if case == "waiting":

        def wait_first():
            try:
                torch.cuda.set_device(0)
                started.set()
                result.append(
                    rt.subscriber.wait_for_retirement(rt.id, first.snapshot_id)
                )
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        waiter = threading.Thread(target=wait_first, daemon=True)
        waiter.start()
        assert started.wait(5)
        assert not done.wait(0.05), "Cannot retire S0 before a replacement is adopted"
    second = stage(rt, removal_delta(rt, 2, 3, 30.0))
    expected = expected_first.clone()
    expected[2::ROWS] = 0
    set_row(rt, expected, 3, 30.0)
    updates = [first, second]
    if case == "chain":
        updates.append(stage(rt, removal_delta(rt, 3, 0, 40.0)))
        expected[3::ROWS] = 0
        set_row(rt, expected, 0, 40.0)
    if case == "waiting":
        assert not done.wait(
            0.05
        ), "Superseding alone must not acknowledge the old snapshot"
    if case == "already_adopted":
        # S1 was adopted; retiring S0 must not depend on pending S2 adoption.
        acknowledge(rt, first)
        assert cold_ps_rows(rt)[1::ROWS].count_nonzero() == 0
    else:
        # No acknowledgement yet: retain rows still reachable from active S0.
        torch.testing.assert_close(cold_ps_rows(rt)[1::ROWS], rt.weights[1::ROWS])
    torch.testing.assert_close(lookup(rt), expected)
    if waiter is not None:
        assert done.wait(
            5
        ), "Superseded waiter did not complete after successor adoption"
        waiter.join()
        assert not errors, errors
        rt.coordinator.acknowledge(
            "reader", EmbeddingCollectionUpdateAck.from_json(result[0])
        )
        updates = [second]
    elif case == "already_adopted":
        updates = [second]
    elif case == "newest_first":
        updates.reverse()
    elif case == "chain":
        updates = [updates[2], updates[0], updates[1]]
    for update in updates:
        acknowledge(rt, update)
    torch.testing.assert_close(cold_ps_rows(rt), expected)
    torch.testing.assert_close(lookup(rt), expected)
    return {"staged": 3 if case == "chain" else 2}


def live_lookup(rt):
    count = 16
    keys = torch.tensor(
        ([0] + list(range(100, 100 + count))) * len(rt.names), device="cuda"
    )
    offsets = torch.arange(len(rt.names) + 1, device="cuda") * (count + 1)
    expected = torch.zeros((keys.numel(), DIM))
    for i in range(len(rt.names)):
        expected[i * (count + 1)] = rt.weights[i * ROWS]
        expected[i * (count + 1) + count] = 50.0 + count - 1 + 10 * i
    started, staged, latest_seen, stop = (threading.Event() for _ in range(4))
    errors, iterations = [], [0]
    stream = torch.cuda.current_stream()

    def infer():
        try:
            torch.cuda.set_device(0)
            with torch.cuda.stream(stream), torch.inference_mode():
                while not stop.is_set():
                    output = rt.model(keys, offsets).cpu()
                    assert torch.isfinite(output).all()
                    torch.testing.assert_close(
                        output[:: count + 1], expected[:: count + 1]
                    )
                    iterations[0] += 1
                    started.set()
                    if staged.is_set() and torch.equal(output, expected):
                        latest_seen.set()
        except BaseException as error:
            errors.append(error)
            started.set()
            latest_seen.set()

    worker = threading.Thread(target=infer, daemon=True)
    worker.start()
    updates = []
    try:
        assert started.wait(5)
        before = iterations[0]
        previous = 1
        for i in range(count):
            updates.append(stage(rt, removal_delta(rt, previous, 100 + i, 50.0 + i)))
            previous = 100 + i
        during = iterations[0] - before
        assert during > 0, "Lookup made no progress during publication"
        staged.set()
        assert latest_seen.wait(10), "Inference never adopted the final mapping"
        assert not errors, errors
        # Reverse acknowledgement order also exercises completion-event ownership.
        for update in reversed(updates):
            acknowledge(rt, update)
    finally:
        stop.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors, errors
    torch.testing.assert_close(rt.model(keys, offsets).cpu(), expected)
    assert not rt.coordinator._pending_deletes
    return {
        "staged": count,
        "lookup_iterations": iterations[0],
        "lookups_during_publication": during,
    }


def duplicate_id(rt):
    update = stage(rt, removal_delta(rt, 1, 0, 20.0))
    with pytest.raises(RuntimeError, match="already outstanding"):
        rt.subscriber.apply_incremental_load(
            update.to_json(), torch.cuda.current_stream().cuda_stream
        )
    expected = rt.weights.clone()
    expected[1::ROWS] = 0
    set_row(rt, expected, 0, 20.0)
    torch.testing.assert_close(lookup(rt), expected)
    acknowledge(rt, update)
    torch.testing.assert_close(cold_ps_rows(rt), expected)
    return {"staged": 1}


@pytest.mark.parametrize("host_cache", [0, CACHE_BYTES], ids=["gpu_only", "host_gpu"])
@pytest.mark.parametrize("case", CASES)
def test_snapshot_retirement(tmp_path, redis_address, host_cache, case):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        case,
        str(tmp_path),
        redis_address,
        str(host_cache),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


if __name__ == "__main__":
    case, path, address, host_cache = sys.argv[1:]
    removal_tests.NAMESPACES = itertools.count(os.getpid())
    rt = runtime(
        Path(path), address, Indexer.LINEAR_HASH_MAP, torch.float32, int(host_cache)
    )
    if case == "live_lookup":
        details = live_lookup(rt)
    elif case == "duplicate_id":
        details = duplicate_id(rt)
    else:
        details = pending_cases(rt, case)
    print(
        json.dumps(
            {"case": case, "host_cache": int(host_cache), "result": "passed", **details}
        )
    )
