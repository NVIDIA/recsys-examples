# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a table has to grow is decided from an estimate that never waits for the
device (``SizeSnapshots``): the newest snapshot of the table sizes that has landed,
plus the bound of every batch issued since it was taken.

What this checks is that the estimate is never too low -- a table never grows later
than the exact rule would grow it, and a table that never evicts never runs out of
value rows, even when the GPU is far behind the host -- that it does not drift for a
key set that stays the same, and that deciding costs the host no wait and no
allocation. The exact rule, which a checkpoint load applies (``exact=True`` of
``expand_if_need``, one blocking read), is what the estimate is compared to.
"""

import itertools
import warnings
from typing import List, Sequence

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
from dynamicemb import sync_free as sf
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)
from dynamicemb.key_value_table import (
    can_grow,
    create_table_state,
    expand_if_need_impl,
    request_size_snapshot,
)
from dynamicemb.optimizer import AdamDynamicEmbeddingOptimizer, OptimizerArgs
from dynamicemb.scored_hashtable import ScoreArg, ScorePolicy

# The DEBUG initializer writes `key % DEBUG_MOD` into every element of a row.
DEBUG_MOD = 100_000
DIM = 8
HBM = 1024**3  # every value in HBM: HBM-direct storage
HOST = 0  # no HBM for values: host storage, inserted into by the forward


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


@pytest.fixture
def enqueued(monkeypatch):
    """The snapshots enqueued so far, as a list that grows with each one."""
    calls = []
    real = kvt.SizeSnapshots.enqueue

    def spy(self, *args, **kwargs):
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(kvt.SizeSnapshots, "enqueue", spy)
    return calls


def options_of(current_device: int, **overrides) -> DynamicEmbTableOptions:
    kwargs = dict(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=1 << 20,
        init_capacity=1024,
        bucket_capacity=128,
        max_load_factor=0.5,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        local_hbm_for_values=HBM,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
    )
    kwargs.update(overrides)
    return DynamicEmbTableOptions(**kwargs)


def build_model(current_device: int, num_tables: int = 1, **overrides):
    """``num_tables`` identical tables, one feature each; a training that never
    changes a value, so the DEBUG pattern survives it."""
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[
            options_of(current_device, **overrides) for _ in range(num_tables)
        ],
        output_dtype=torch.float32,
        table_names=[f"t_{i}" for i in range(num_tables)],
        feature_table_map=list(range(num_tables)),
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )


def make_batch(keys_per_table: Sequence[Sequence[int]], batch: int, device):
    """One feature per table; each of its ``batch`` bags holds an equal share of the
    table's keys. Returns ``(indices, offsets)``."""
    lengths: List[int] = []
    indices: List[int] = []
    for keys in keys_per_table:
        assert len(keys) % batch == 0
        lengths += [len(keys) // batch] * batch
        indices += list(keys)
    offsets = [0] + list(itertools.accumulate(lengths))
    return (
        torch.tensor(indices, dtype=torch.int64, device=device),
        torch.tensor(offsets, dtype=torch.int64, device=device),
    )


def one_table_batch(keys: Sequence[int], device):
    return make_batch([keys], len(keys), device)


def train_step(model, batch):
    out = model(*batch)
    out.sum().backward()
    torch.cuda.synchronize()
    return out


def capacities(model) -> List[int]:
    return list(model._storage._state.key_index_map.per_table_capacity_)


# --- who gets an estimate -----------------------------------------------------------


def test_a_table_at_its_max_capacity_has_no_estimate(current_device):
    """``init_capacity == max_capacity`` (the default): nothing can grow, so there is
    no ring and no bookkeeping, and the decision costs nothing."""
    model = build_model(current_device, init_capacity=1 << 20)
    assert model._storage._state.size_snapshots is None
    assert not can_grow(model._storage._state)
    assert not model.needs_key_bounds


def test_a_table_below_its_max_capacity_has_one(current_device):
    model = build_model(current_device)
    state = model._storage._state
    assert state.size_snapshots is not None
    assert can_grow(state)
    # NO_EVICTION grows without a max_capacity to stop it, and has rows to count.
    no_eviction = build_model(
        current_device,
        init_capacity=1 << 20,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
    )
    assert no_eviction._storage._state.size_snapshots.with_rows
    assert can_grow(no_eviction._storage._state)


def test_a_cache_has_no_estimate(current_device):
    rows = 64 * 1024
    model = build_model(
        current_device,
        max_capacity=rows,
        init_capacity=None,
        local_hbm_for_values=rows * DIM * 4 // 16,
        caching=True,
    )
    assert model._storage_layout == StorageLayout.CACHING
    assert model.cache._state.size_snapshots is None
    assert model._storage._state.size_snapshots is None


# --- the snapshots follow the inserts -----------------------------------------------


def table_state(current_device: int, **overrides):
    option = options_of(current_device, local_hbm_for_values=0, **overrides)
    return create_table_state([option], AdamDynamicEmbeddingOptimizer(OptimizerArgs()))


def insert_keys(state, first: int, count: int, device):
    keys = torch.arange(first, first + count, dtype=torch.int64, device=device)
    score = ScoreArg(
        name=state.score_policy.name,
        value=torch.ones_like(keys, dtype=torch.uint64),
        policy=ScorePolicy.ASSIGN,
    )
    state.key_index_map.insert(keys, torch.zeros_like(keys), score)


def test_a_snapshot_counts_the_inserts_before_it_and_the_bounds_after(current_device):
    device = torch.device("cuda", current_device)
    state = table_state(current_device)
    snapshots = state.size_snapshots

    insert_keys(state, 0, 100, device)
    request_size_snapshot(state)
    insert_keys(state, 1000, 50, device)  # enqueued after the snapshot: not in it
    snapshots.issued += torch.tensor([50])  # ... and a bound issued for them
    torch.cuda.synchronize()
    snapshots.poll()

    assert snapshots.base_sizes.tolist() == [100]
    keys, _ = snapshots.estimate()
    assert keys.tolist() == [150]  # the 100 it saw plus the 50 issued since

    # A blocking read of the exact sizes starts the estimate over from them.
    state.collect_table_sizes_flag = True
    kvt.collect_table_sizes_for_state(state, non_blocking=False)
    assert snapshots.base_sizes.tolist() == [150]
    assert not snapshots.pending
    assert snapshots.estimate()[0].tolist() == [150]


def test_a_full_ring_never_waits_and_only_loosens_the_estimate(current_device):
    device = torch.device("cuda", current_device)
    state = table_state(current_device)
    snapshots = state.size_snapshots
    insert_keys(state, 0, 10, device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(3 * snapshots.slots):  # nobody polls: the ring fills up
            request_size_snapshot(state)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert len(snapshots.pending) == snapshots.slots
    assert snapshots.ring.free_slots() == 0
    torch.cuda.synchronize()
    snapshots.poll()  # everything that landed is taken, the ring is free again
    assert snapshots.ring.free_slots() == snapshots.slots
    assert snapshots.base_sizes.tolist() == [10]


def test_deciding_reads_nothing_back_and_allocates_nothing(current_device, monkeypatch):
    """Fifty batches through the snapshot and the decision of a table that has room:
    no wait (sync-debug mode raises on one), no ``sf.read_now``, no pinned allocation.
    """
    state = table_state(current_device, init_capacity=8192)
    pinned = []
    for name in ("empty", "zeros"):
        real = getattr(torch, name)

        def spy(*args, __real=real, __name=name, **kwargs):
            if kwargs.get("pin_memory"):
                pinned.append(__name)
            return __real(*args, **kwargs)

        monkeypatch.setattr(torch, name, spy)
    bound = torch.tensor([8], dtype=torch.int64)
    reads = getattr(sf, "READ_NOW_CALLS", 0)

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(50):
            request_size_snapshot(state)
            expand_if_need_impl(state, bound)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert not pinned, pinned
    assert getattr(sf, "READ_NOW_CALLS", 0) == reads
    assert state.key_index_map.per_table_capacity_ == [8192]  # nothing needed to grow


# --- the exact rule is the reference ------------------------------------------------


def capacity_trace(current_device, steps: int, new: int, repeated: int) -> List[int]:
    """The capacity of a growing table after each step of ``steps``, each of which
    brings ``new`` keys it never had and ``repeated`` that the first step brought."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device)
    trace = []
    for step in range(steps):
        fresh = range(1_000 + step * new, 1_000 + (step + 1) * new)
        keys = list(fresh) + list(range(1_000, 1_000 + repeated))
        train_step(model, one_table_batch(keys, device))
        trace.append(capacities(model)[0])
    return trace


def test_the_estimate_grows_a_table_never_later_than_the_exact_rule(
    current_device, monkeypatch
):
    """The estimate counts the bound of the batches the device has not reported yet,
    so it may grow a table a little before an exact read would; never after it. With
    batches that are small against the table, a little is under a doubling."""
    steps, new, repeated = 80, 64, 64
    estimated = capacity_trace(current_device, steps, new, repeated)
    real = kvt.expand_if_need_impl
    monkeypatch.setattr(
        kvt,
        "expand_if_need_impl",
        lambda state, unique, *, exact=False, inserts_in_forward=False: real(
            state, unique, exact=True, inserts_in_forward=inserts_in_forward
        ),
    )
    exact = capacity_trace(current_device, steps, new, repeated)

    assert exact[-1] > exact[0], "the trace never made the table grow"
    for step, (est, ref) in enumerate(zip(estimated, exact)):
        assert est >= ref, f"step {step}: grew to {est}, the exact rule had {ref}"
        assert est <= 2 * ref, f"step {step}: grew to {est}, the exact rule had {ref}"


def test_a_key_set_that_stays_the_same_stops_growing(current_device):
    """The same 64 keys again and again are never more than 64 keys. An estimate that
    only added up the batches would reach the growth threshold of a 1024-slot table
    (512 keys) after eight of them and grow it for nothing, over and over."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device)
    batch = one_table_batch(list(range(5_000, 5_064)), device)
    for _ in range(200):
        train_step(model, batch)
    assert capacities(model) == [1024]


def test_an_exact_decision_reads_the_sizes_once_and_drops_the_snapshots(
    current_device, read_now_calls
):
    """The decision a checkpoint load makes: one blocking read of the device sizes when
    an estimate says a table might grow (``read_table_sizes_now``), which restarts the
    estimate from the numbers read and drops the snapshots still in flight."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device)
    state = model._storage._state
    snapshots = state.size_snapshots
    train_step(model, one_table_batch(list(range(10, 20)), device))
    torch.cuda.synchronize()
    before = read_now_calls.delta

    model._storage.expand_if_need(torch.tensor([4_000], dtype=torch.int64), exact=True)

    assert read_now_calls.delta - before == 1
    assert capacities(model)[0] > 1024
    assert not snapshots.pending
    assert snapshots.ring.free_slots() == snapshots.slots


# --- a table that never evicts must never run out of rows ---------------------------


@pytest.mark.parametrize("depth", [0, 2], ids=["plain_loop", "prefetch_depth_2"])
@pytest.mark.parametrize("hbm", [HBM, HOST], ids=["hbm", "host"])
def test_no_eviction_never_trips_the_row_limit_with_a_lagging_gpu(
    current_device, hbm, depth
):
    """Every batch brings 200 keys a 256-row table does not have, so it grows again and
    again, while the GPU is held back by a sleep before every forward: the snapshots
    land late, or not at all, and the decision has only the bounds to go on. A growth
    that came late would hand out a row past the end of the value buffer, which a
    device assert turns into an error at the synchronize below."""
    device = torch.device("cuda", current_device)
    model = build_model(
        current_device,
        max_capacity=1 << 18,
        init_capacity=256,
        local_hbm_for_values=hbm,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
    )
    batches = [list(range(1_001 + 200 * i, 1_001 + 200 * (i + 1))) for i in range(10)]
    initial = capacities(model)[0]
    queued, outputs = [], []
    for step in range(len(batches) + depth):
        if step < len(batches):
            batch = one_table_batch(batches[step], device)
            model.prefetch(*batch)
            queued.append((batches[step], batch))
        if step >= depth:
            keys, batch = queued.pop(0)
            torch.cuda._sleep(50_000_000)
            out = model(*batch)
            out.sum().backward()
            outputs.append((keys, out.detach()))
    torch.cuda.synchronize()

    assert capacities(model)[0] > initial
    for keys, out in outputs:
        expected = torch.tensor(
            [k % DEBUG_MOD for k in keys], dtype=out.dtype, device=out.device
        )
        torch.testing.assert_close(out, expected.unsqueeze(1).expand(-1, out.size(1)))


# --- where the snapshot is taken ----------------------------------------------------


def test_hbm_storage_snapshots_at_the_prefetch(current_device, enqueued):
    """An HBM storage is inserted into by the prefetch: its snapshot is enqueued there,
    and nothing is ever in flight."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, local_hbm_for_values=HBM)
    snapshots = model._storage._state.size_snapshots
    batch = one_table_batch(list(range(100)), device)

    model.prefetch(*batch)
    assert enqueued == [snapshots]
    assert snapshots.inflight.tolist() == [0]
    model(*batch).sum().backward()
    torch.cuda.synchronize()
    assert enqueued == [snapshots]  # the forward of an HBM storage adds none


def test_host_storage_snapshots_at_the_end_of_the_forward(current_device, enqueued):
    """A host storage is inserted into by the forward: no snapshot at the prefetch,
    where the batch is in flight; one at the end of the forward, which counts it."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, local_hbm_for_values=HOST)
    assert model._storage_layout == StorageLayout.HOST_ONLY
    snapshots = model._storage._state.size_snapshots
    batch = one_table_batch(list(range(100)), device)

    model.prefetch(*batch)
    assert enqueued == []
    assert snapshots.inflight.tolist() == [100]
    model(*batch).sum().backward()
    torch.cuda.synchronize()
    assert snapshots.inflight.tolist() == [0]
    assert enqueued == [snapshots]


# --- several tables: each is bounded by its own keys --------------------------------


def test_the_key_bounds_of_the_tables_are_those_of_their_features(current_device):
    model = build_model(current_device, num_tables=2)
    assert model.needs_key_bounds
    assert model._table_key_bounds([300, 40], 340).tolist() == [300, 40]
    assert model._table_key_bounds([300, 40]).tolist() == [300, 40]
    # Not known, not matching the call, or not worth knowing: no bound, so the prefetch
    # bounds every table by the whole batch.
    assert model._table_key_bounds(None, 340) is None
    assert model._table_key_bounds([300, 40], 100) is None
    assert model._table_key_bounds([340], 340) is None

    one_table = build_model(current_device)
    assert not one_table.needs_key_bounds
    assert one_table._table_key_bounds([100], 100) is None

    full = build_model(current_device, num_tables=2, init_capacity=1 << 20)
    assert not full.needs_key_bounds
    assert full._table_key_bounds([300, 40], 340) is None


def test_a_table_with_few_keys_does_not_grow_for_a_table_with_many(current_device):
    """Table 0 gets 3,000 keys, table 1 gets 40. Bounding table 1 by the whole batch
    of 3,040 would grow it, and a module of T tables would grow every one of them
    T-fold; its own 40 keys leave it alone."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, num_tables=2)
    keys = [list(range(10_000, 13_000)), list(range(40))]
    batch = make_batch(keys, 40, device)
    before = capacities(model)

    model.prefetch(*batch, feature_key_counts=[3_000, 40])
    state = model._prefetch_states[0]
    # What the growth was told is never below what the tables get.
    assert state.table_key_bounds[0] >= 3_000
    assert state.table_key_bounds[1] >= 40
    after = capacities(model)
    assert after[0] > before[0]
    assert after[1] == before[1]
    model(*batch).sum().backward()
    torch.cuda.synchronize()


# --- the metadata of a growth is uploaded without waiting ----------------------------


def pointers_of(state) -> List[int]:
    return [buffer.tensor().data_ptr() for buffer in state.tables]


@pytest.mark.parametrize("hbm", [HBM, HOST], ids=["hbm", "host"])
def test_the_pointer_table_follows_the_buffers_and_is_uploaded_only_when_one_moved(
    current_device, hbm, monkeypatch
):
    """A device buffer keeps its address when it grows, and with it the table the
    kernels read: nothing is uploaded. A buffer that does move is published by one
    queued copy. Either way the table on the device ends up as the buffers are."""
    option = options_of(current_device, local_hbm_for_values=hbm, init_capacity=1024)
    state = create_table_state([option], AdamDynamicEmbeddingOptimizer(OptimizerArgs()))
    assert state.table_ptrs_host == pointers_of(state)
    assert state.table_ptrs_dev.tolist() == pointers_of(state)
    uploads = []
    real = sf.copy_i64_into
    monkeypatch.setattr(
        sf,
        "copy_i64_into",
        lambda dst, values: (uploads.append(dst), real(dst, values)),
    )
    before = pointers_of(state)

    kvt._expand_tables_impl(state, [True], [4096])

    moved = pointers_of(state) != before
    published = [u for u in uploads if u is state.table_ptrs_dev]
    assert len(published) == (1 if moved else 0)
    torch.cuda.synchronize()
    assert state.table_ptrs_dev.tolist() == pointers_of(state)
    assert state.table_ptrs_host == pointers_of(state)
    if hbm:
        assert not moved, "a device value buffer must keep its address when it grows"


def test_small_uploads_do_not_wait_and_do_not_allocate_pinned_memory(
    current_device, monkeypatch
):
    """The bucket offsets, the row limits and the pointer table of a growth are a few
    int64 each, uploaded one after the other: through the ring of pinned buffers, with
    no wait and no pinned allocation, however many of them are in flight."""
    device = torch.device("cuda", current_device)
    sf.upload_i64([1, 2, 3], device)  # the ring is made here, not in the window
    torch.cuda.synchronize()
    pinned = []
    for name in ("empty", "zeros", "tensor"):
        real = getattr(torch, name)

        def spy(*args, __real=real, __name=name, **kwargs):
            if kwargs.get("pin_memory"):
                pinned.append(__name)
            return __real(*args, **kwargs)

        monkeypatch.setattr(torch, name, spy)

    torch.cuda._sleep(50_000_000)  # the copies queue behind it: every buffer in flight
    torch.cuda.set_sync_debug_mode("error")
    try:
        small = [
            sf.upload_i64([i, i + 1, i + 2], device) for i in range(sf._UPLOAD_SLOTS)
        ]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert not pinned, pinned
    torch.cuda.synchronize()
    assert [t.tolist() for t in small] == [
        [i, i + 1, i + 2] for i in range(sf._UPLOAD_SLOTS)
    ]

    # More uploads than buffers while the first are still in flight, and a list that
    # does not fit a buffer, take a pinned tensor of their own: still no wait.
    torch.cuda._sleep(50_000_000)
    wide = list(range(sf._UPLOAD_SLOT_NUMEL + 5))
    torch.cuda.set_sync_debug_mode("error")
    try:
        many = [sf.upload_i64([i], device) for i in range(2 * sf._UPLOAD_SLOTS)]
        long = sf.upload_i64(wide, device)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert [t.tolist() for t in many] == [[i] for i in range(2 * sf._UPLOAD_SLOTS)]
    assert long.tolist() == wide


def test_a_key_map_is_built_without_waiting(current_device):
    from dynamicemb.scored_hashtable import ScoreSpec, get_scored_table

    device = torch.device("cuda", current_device)
    get_scored_table(  # the first one allocates the ring and its pinned buffers
        capacity=[256, 256],
        bucket_capacity=128,
        score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)],
        use_rows=True,
    )
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        table = get_scored_table(
            capacity=[512, 1024],
            bucket_capacity=128,
            score_specs=[ScoreSpec(name="s", policy=ScorePolicy.ASSIGN)],
            use_rows=True,
        )
        table.set_row_limits([100, 200])
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert table.table_bucket_offsets_.tolist() == [0, 4, 12]
    assert table.bucket_offsets_host_ == [0, 4, 12]
    assert table.table_bucket_offsets_cpu_.tolist() == [0, 4, 12]
    assert table.row_limits_.tolist() == [100, 200]
    assert table.device == device


@pytest.mark.parametrize("no_eviction", [False, True], ids=["evicting", "no_eviction"])
def test_the_exact_sizes_are_one_read(current_device, no_eviction, read_now_calls):
    device = torch.device("cuda", current_device)
    option = options_of(
        current_device,
        init_capacity=1 << 15,
        local_hbm_for_values=0,
        score_strategy=DynamicEmbScoreStrategy.NO_EVICTION
        if no_eviction
        else DynamicEmbScoreStrategy.TIMESTAMP,
    )
    state = create_table_state(
        [option], AdamDynamicEmbeddingOptimizer(OptimizerArgs()), use_rows=True
    )
    insert_keys(state, 0, 100, device)
    if no_eviction:
        state.key_index_map.reserve_rows(0, 150)  # rows handed out beyond the keys
    torch.cuda.synchronize()

    sizes, rows = kvt.read_table_sizes_now(state)

    assert read_now_calls.delta == 1
    assert sizes.tolist() == [100]
    assert (rows.tolist() == [150]) if no_eviction else (rows is None)
    assert not sizes.is_cuda


def test_the_removed_exact_growth_switch_warns_and_changes_nothing(monkeypatch):
    """DYNAMICEMB_EXACT_GROWTH made every decision wait for the device; no environment
    variable may, so it is gone. A script that sets it is told, once, when the module is
    imported, and keeps running on the estimate."""
    monkeypatch.setenv("DYNAMICEMB_EXACT_GROWTH", "1")
    with pytest.warns(UserWarning, match="DYNAMICEMB_EXACT_GROWTH was removed"):
        kvt._warn_if_exact_growth_is_set()
    for quiet in ("0", ""):
        monkeypatch.setenv("DYNAMICEMB_EXACT_GROWTH", quiet)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kvt._warn_if_exact_growth_is_set()
    monkeypatch.delenv("DYNAMICEMB_EXACT_GROWTH")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kvt._warn_if_exact_growth_is_set()
