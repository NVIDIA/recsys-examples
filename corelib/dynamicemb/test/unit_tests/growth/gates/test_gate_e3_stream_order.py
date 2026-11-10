# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GATE E3 -- table growth against the streams that use the table, under a TorchRec-like loop.

With ``ExtendableBuffer.extend`` no longer waiting for the device, what orders a growth
against the other streams is ``GrowthOrder``: the growth stream waits for what the others
queued (H1), they wait for the growth (H2), the replaced key map is not reused early (H3).
This gate drives real modules through the host order of the prefetch pipeline of TorchRec.
``PrefetchTrainPipelineSparseDist.progress`` (``torchrec/distributed/train_pipeline/
train_pipelines.py`` of the TorchRec clone of this repository, read when the gate was written)
does, in this order: ``_wait_for_batch(batch_i, prefetch_stream)`` (F waits for P), the forward of
batch i, ``_prefetch(batch_i+1)`` on P, the backward of batch i, the optimizer step. The loop below
is that order, with the modules of DynamicEmb and none of TorchRec (no input dist, no sharding)::

    F waits for P  ->  forward(i) on F  ->  prefetch(i + 1) on P  ->  backward(i) on F

where the prefetch is the call that makes the table grow (fresh keys at every step) and P
does not wait for F. What this gate does NOT do is run the real pipeline: see ``run_order.py`` of
this directory for the run of the HSTU pipeline test with a table that grows. ``torch.cuda._sleep`` is put at random in front of the forward, the
backward and the prefetch: a stream that is late is what turns a missing edge into a wrong
result. The run is cut into epochs of a few dozen steps, each on a module of its own, so
that growths stay frequent however long the run is (a table grows by doubling).

Per step the output must be what the DEBUG initializer makes of the keys (``key %
DEBUG_EMB_INITIALIZER_MOD``: a forward that read a half-built key map gets something else).
At the end of an epoch every reference counter must be back at zero (a decrement that was
lost leaves a slot pinned for good), the table must hold exactly the keys it was given, and
a table that can grow must have grown. Configurations (the layout each one gets is
checked):

* ``hbm_no_eviction``: values in HBM, never evicts, the key map is twice the value buffer;
* ``hbm_evicting_with_erase``: values in HBM, evicts, with an erase (by a score below zero:
  nothing goes) running beside the training on its own stream;
* ``host_only``: values on the host, the forward inserts;
* ``hybrid``: an HBM tier of 2048 rows and a host tier that grows.

``GATE_E3_DISABLE_ORDER=1`` is the negative control: H1 and H2 are switched off (the
module-level ``_growth_section`` is a no-op and ``wait_for_growth`` registers nothing) and
the run is expected to FAIL; it prints ``E3 CONTROL <config> sensitive`` when it did. The
control of ``hbm_no_eviction`` must fail, or the gate cannot tell a missing edge from a
present one and the verdict of the real run says nothing (that test fails). The others
print their result only, because a table without reference counters has fewer ways to show
a lost edge. A failed control can leave the CUDA context unusable (the run then ends with
its verdict printed, without tearing the context down): run one configuration
per process (``-k``).

Run it (one H100; ``GATE_E3_STEPS`` is the total, ``GATE_E3_EPOCH_STEPS`` at most 64)::

    DEMB_GROWTH_GATES=1 GATE_E3_STEPS=5000 python -m pytest -s -q \\
        corelib/dynamicemb/test/unit_tests/growth/gates/test_gate_e3_stream_order.py
    # the memory checker, a few steps (it is slow); its report must be empty for all four
    # (it finds a stale pointer, not a wrong order: that is what the value checks are for)
    DEMB_GROWTH_GATES=1 GATE_E3_STEPS=96 compute-sanitizer --tool memcheck \\
        --error-exitcode 99 python -m pytest -s -q <the file>
    # the control, one process per configuration, on the tree without the synchronize
    DEMB_GROWTH_GATES=1 GATE_E3_STEPS=1000 GATE_E3_DISABLE_ORDER=1 python -m pytest -s -q \\
        <the file> -k hbm_no_eviction

Every verdict is one printed line: ``E3 PASS ...`` or ``E3 FAIL ...``. The line also says how
much memory the epochs peaked at over what their module held (torch's allocated and reserved,
the value buffers apart): informational, for the peak-memory question of the design, which
says a growth holds the replaced key map until the growth stream has passed it and nothing
more than the old and new maps that coexisted anyway. The normal mode of this file also runs
on the tree before the lane (it only needs ``HybridStorage``, ``key_map_generation`` and
``erase_by_score_async``): run both and compare ``peak_reserved``.
"""

import contextlib
import gc
import io
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List

import dynamicemb
import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb import key_value_table as kvt
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import DEBUG_EMB_INITIALIZER_MOD
from dynamicemb.key_value_table import HybridStorage
from dynamicemb.optimizer import get_optimizer_state_dim

pytestmark = pytest.mark.skipif(
    os.environ.get("DEMB_GROWTH_GATES") != "1" or not torch.cuda.is_available(),
    reason="a hardware gate: set DEMB_GROWTH_GATES=1 on a GPU",
)

STEPS = int(os.environ.get("GATE_E3_STEPS", "300"))
EPOCH_STEPS = int(os.environ.get("GATE_E3_EPOCH_STEPS", "48"))
SEED = int(os.environ.get("GATE_E3_SEED", "7"))
DISABLE = os.environ.get("GATE_E3_DISABLE_ORDER") == "1"
SLEEP_CYCLES = int(os.environ.get("GATE_E3_SLEEP_CYCLES", "3000000"))  # ~1.5 ms
KEYS_PER_STEP = 200
DIM = 8
INIT_CAPACITY = 1024
HYBRID_HBM_ROWS = 2048
ERASE_EVERY = 4

# Value rows per table (a never-evicting table starts at INIT_CAPACITY and has no cap).
# An epoch holds at most (64 + 1) * KEYS_PER_STEP keys, a quarter of what the smallest
# table can keep (the host tier of the hybrid layout), so nothing is ever evicted and the
# count of keys at the end of an epoch is exact.
MAX_CAPACITY = {
    "hbm_no_eviction": 1 << 21,
    "hbm_evicting_with_erase": 1 << 21,
    "host_only": 1 << 21,
    "hybrid": 1 << 16,
}
CONFIGS = list(MAX_CAPACITY)
CONTROL_MUST_FAIL = {"hbm_no_eviction"}


def options_for(config: str) -> DynamicEmbTableOptions:
    row_bytes = 4 * (DIM + get_optimizer_state_dim(EmbOptimType.SGD, DIM))
    kwargs = dict(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=torch.cuda.current_device(),
        dim=DIM,
        max_capacity=MAX_CAPACITY[config],
        init_capacity=INIT_CAPACITY,
        bucket_capacity=128,
        max_load_factor=0.5,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=1024**3,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    if config == "hbm_no_eviction":
        kwargs["score_strategy"] = DynamicEmbScoreStrategy.NO_EVICTION
    elif config == "host_only":
        kwargs["local_hbm_for_values"] = 0
    elif config == "hybrid":
        # the HBM tier gets this share of max_capacity; the rest of it is the host tier
        kwargs["local_hbm_for_values"] = HYBRID_HBM_ROWS * row_bytes
    return DynamicEmbTableOptions(**kwargs)


def build_model(config: str) -> BatchedDynamicEmbeddingTablesV2:
    with contextlib.redirect_stdout(io.StringIO()):  # the memory report of the module
        return BatchedDynamicEmbeddingTablesV2(
            table_options=[options_for(config)],
            output_dtype=torch.float32,
            table_names=["t_0"],
            feature_table_map=[0],
            pooling_mode=DynamicEmbPoolingMode.SUM,
            use_index_dedup=False,
            optimizer=EmbOptimType.SGD,
            learning_rate=0.0,
            stochastic_rounding=False,
        )


def all_states(model) -> list:
    storage = model._storage
    if isinstance(storage, HybridStorage):
        return [storage._hbm, storage._host]
    return [storage._state]


def growable_states(model) -> list:
    """The table states whose key map and value buffers grow (the HBM tier of the hybrid
    layout is made as large as it gets)."""
    storage = model._storage
    return [storage._host] if isinstance(storage, HybridStorage) else all_states(model)


def layout_problems(config: str, model) -> List[str]:
    """The gate means what it says only if the module got the layout it names."""
    states = all_states(model)
    expected_hybrid = config == "hybrid"
    problems = []
    if isinstance(model._storage, HybridStorage) != expected_hybrid:
        problems.append("%s: the storage is a %s" % (config, type(model._storage)))
        return problems
    device_values = [s.tables[0].is_device_buffer() for s in states]
    want = {
        "hbm_no_eviction": [True],
        "hbm_evicting_with_erase": [True],
        "host_only": [False],
        "hybrid": [True, False],
    }[config]
    if device_values != want:
        problems.append(
            "%s: value buffers on device = %s, wanted %s"
            % (config, device_values, want)
        )
    pinned = [s.key_index_map._ref_counter is not None for s in states]
    want_pinned = [config.startswith("hbm")] * len(states)
    if pinned != want_pinned:
        problems.append(
            "%s: reference counters = %s, wanted %s" % (config, pinned, want_pinned)
        )
    return problems


@contextlib.contextmanager
def order_disabled():
    """The negative control: H1 and H2 off (no registration, no waits, no events)."""
    real_wait = kvt.DynamicEmbTableState.wait_for_growth
    real_section = kvt._growth_section
    kvt.DynamicEmbTableState.wait_for_growth = lambda self: None
    kvt._growth_section = lambda state, stream: contextlib.nullcontext()
    try:
        yield
    finally:
        kvt.DynamicEmbTableState.wait_for_growth = real_wait
        kvt._growth_section = real_section


def batch_on(stream, step: int, device, other):
    """The keys of one step, made on ``stream`` (the stream that prefetches them) and
    marked as used by ``other`` (the stream that runs their forward and backward)."""
    base = 1_000 + step * KEYS_PER_STEP
    with torch.cuda.stream(stream):
        keys = torch.arange(
            base, base + KEYS_PER_STEP, dtype=torch.int64, device=device
        )
        offsets = torch.arange(0, KEYS_PER_STEP + 1, dtype=torch.int64, device=device)
    keys.record_stream(other)
    offsets.record_stream(other)
    return keys, offsets


@dataclass
class Report:
    problems: List[str] = field(default_factory=list)
    epochs: int = 0
    steps: int = 0
    rehashes: int = 0  # the key maps that were replaced (``key_map_generation``)
    # memory, in MiB, over what the module held when its epoch began (the largest of the
    # epochs): torch's peak allocated and peak reserved (a block that a stream still uses
    # is reserved, not allocated), and the value buffers at the end (mapped by the driver,
    # which torch does not count)
    peak_allocated_mib: float = 0.0
    peak_reserved_mib: float = 0.0
    values_mib: float = 0.0
    # A CUDA error (an illegal access, a device assert) leaves the context unusable: its
    # tensors cannot be freed (the destructor raises and the process aborts), so nothing is
    # torn down and the process ends with the verdict.
    faulted: bool = False


def is_cuda_fault(exc: BaseException) -> bool:
    text = str(exc)
    return "CUDA error" in text or "illegal" in text or "device-side assert" in text


def run_epoch(config: str, steps: int, rng, forward, prefetch, report: Report) -> None:
    """One module, ``steps`` steps; what went wrong goes to ``report.problems``."""
    device = torch.device("cuda", torch.cuda.current_device())
    problems = report.problems
    model, outputs, erase, erased = None, [], None, 0
    try:
        model = build_model(config)
        model.train()
        problems.extend(layout_problems(config, model))
        if problems:
            return
        value_rows_before = [s.tables[0].shape[0] for s in growable_states(model)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_allocated = torch.cuda.memory_allocated()
        start_reserved = torch.cuda.memory_reserved()

        def maybe_sleep():
            if rng.random() < 0.3:
                torch.cuda._sleep(int(SLEEP_CYCLES * rng.random()))

        # prefetch(0), as the pipeline does before its first progress()
        batches = {0: batch_on(prefetch, 0, device, forward)}
        with torch.cuda.stream(prefetch):
            model.prefetch(batches[0][0], batches[0][1], forward)
        for step in range(steps):
            keys, offsets = batches.pop(step)
            forward.wait_stream(prefetch)  # _wait_for_batch: this batch is prefetched
            with torch.cuda.stream(forward):
                maybe_sleep()
                out = model(keys, offsets)
            next_keys, next_offsets = batch_on(prefetch, step + 1, device, forward)
            batches[step + 1] = (next_keys, next_offsets)
            with torch.cuda.stream(prefetch):
                maybe_sleep()
                # P does not wait for F
                model.prefetch(next_keys, next_offsets, forward)
            with torch.cuda.stream(forward):
                maybe_sleep()
                out.sum().backward()
            outputs.append((keys, out.detach()))
            if config == "hbm_evicting_with_erase" and step % ERASE_EVERY == 0:
                if erase is None or erase.done():
                    if erase is not None:
                        erased += erase.result()[0]
                    # nothing has a score below 0: it scans every chunk of every table
                    erase = dynamicemb.erase_by_score_async(
                        model, DynamicEmbScoreStrategy.TIMESTAMP, 0
                    )
        # the batch that was prefetched last holds its keys' pins until it runs
        keys, offsets = batches.pop(steps)
        forward.wait_stream(prefetch)
        with torch.cuda.stream(forward):
            model(keys, offsets).sum().backward()
        torch.cuda.synchronize()
        if erase is not None:
            erased += erase.result()[0]
        if erased != 0:
            problems.append("the erase below score 0 removed %d keys" % erased)
        for step, (step_keys, out) in enumerate(outputs):
            expected = (step_keys % DEBUG_EMB_INITIALIZER_MOD).to(out.dtype)
            if not bool((out == expected.unsqueeze(1)).all()):
                problems.append(
                    "epoch step %d: the output is not the initializer's value of its key"
                    % step
                )
                break
        for state in all_states(model):
            counter = state.key_index_map._ref_counter
            if counter is not None and bool((counter != 0).any()):
                problems.append(
                    "a reference counter is not back at zero: a decrement was lost"
                )
        total = sum(int(s.key_index_map.size()) for s in all_states(model))
        given = (steps + 1) * KEYS_PER_STEP
        if total != given:
            problems.append("the table holds %d keys, %d were given" % (total, given))
        rehashes = sum(s.key_map_generation for s in growable_states(model))
        value_rows_after = [s.tables[0].shape[0] for s in growable_states(model)]
        if rehashes == 0 and value_rows_after == value_rows_before:
            problems.append("no table grew: the epoch did not exercise a growth")
        report.rehashes += rehashes
        mib = 1 << 20
        report.peak_allocated_mib = max(
            report.peak_allocated_mib,
            (torch.cuda.max_memory_allocated() - start_allocated) / mib,
        )
        report.peak_reserved_mib = max(
            report.peak_reserved_mib,
            (torch.cuda.max_memory_reserved() - start_reserved) / mib,
        )
        report.values_mib = max(
            report.values_mib,
            sum(t.allocated_bytes() for s in all_states(model) for t in s.tables) / mib,
        )
    except Exception as exc:  # noqa: BLE001
        # an error (an IMA, a device assert) is the verdict
        problems.append("%s: %s" % (type(exc).__name__, str(exc)[:300]))
        report.faulted |= is_cuda_fault(exc)
    finally:
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            problems.append("synchronize: %s" % str(exc)[:300])
            report.faulted |= is_cuda_fault(exc)
        if not report.faulted:
            del outputs, model
            gc.collect()


def run_config(config: str, steps: int) -> Report:
    assert (
        1 <= EPOCH_STEPS <= 64
    ), "GATE_E3_EPOCH_STEPS is at most 64 (see MAX_CAPACITY)"
    rng = random.Random(SEED)
    forward, prefetch = torch.cuda.Stream(), torch.cuda.Stream()
    report = Report()
    # Epochs of (nearly) one size, none longer than EPOCH_STEPS: a short last one (5000 steps
    # in epochs of 48 leave one of 8) is too short for the host tier of the hybrid layout to
    # fill and grow, and an epoch that does not grow is a problem of the run.
    epochs = -(-steps // EPOCH_STEPS)
    done = 0
    for index in range(epochs):
        if report.problems:
            break
        n = steps // epochs + (1 if index < steps % epochs else 0)
        run_epoch(config, n, rng, forward, prefetch, report)
        done += n
        report.epochs += 1
        report.steps = done
        if not report.problems:
            torch.cuda.empty_cache()
    return report


@pytest.mark.parametrize("config", CONFIGS)
def test_e3_growth_under_the_pipeline_order(config):
    context = order_disabled() if DISABLE else contextlib.nullcontext()
    with context:
        report = run_config(config, STEPS)
    line = (
        "%s epochs=%d steps=%d rehashes=%d peak_allocated=+%.1fMiB peak_reserved=+%.1fMiB "
        "values=%.1fMiB"
        % (
            config,
            report.epochs,
            report.steps,
            report.rehashes,
            report.peak_allocated_mib,
            report.peak_reserved_mib,
            report.values_mib,
        )
    )
    if DISABLE:
        if report.problems:
            print(
                "E3 CONTROL %s sensitive: %s (%s)" % (config, report.problems[0], line)
            )
        else:
            print("E3 CONTROL %s INSENSITIVE: nothing failed (%s)" % (config, line))
            if config in CONTROL_MUST_FAIL:
                pytest.fail(
                    "the negative control did not fail: the gate cannot see a missing edge"
                )
        end_if_faulted(report, 0)
        return
    if report.problems:
        print("E3 FAIL %s: %s (%s)" % (config, "; ".join(report.problems), line))
        end_if_faulted(report, 1)
        pytest.fail("; ".join(report.problems))
    print("E3 PASS %s" % line)


def end_if_faulted(report: Report, code: int) -> None:
    """A faulted context cannot be torn down (see ``Report.faulted``): the verdict is
    printed, and the process ends here, with ``code``."""
    if report.faulted:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    os.environ.setdefault("DEMB_GROWTH_GATES", "1")
    sys.exit(pytest.main(["-s", "-q", __file__]))
