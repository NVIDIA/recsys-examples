# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module-level tests for ``replay_increment``: writing an ``incremental_dump``
delta back into another model's tables."""

from typing import List, Optional

import pytest
import torch
from dynamicemb import (
    BATCH_SIZE_PER_DUMP,
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    EvictedItemMode,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import ReplayContent

TABLE_NAME = "t_0"
DIM = 8
DEFAULT_CAPACITY = BATCH_SIZE_PER_DUMP * 2


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def make_model(
    current_device: int,
    score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    max_capacity: int = DEFAULT_CAPACITY,
    init_capacity: Optional[int] = None,
    bucket_capacity: int = 128,
    caching: bool = False,
    local_hbm_for_values: int = 1024**3,
    evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD,
    optimizer: EmbOptimType = EmbOptimType.SGD,
    **opt_params,
) -> BatchedDynamicEmbeddingTablesV2:
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=max_capacity,
        bucket_capacity=bucket_capacity,
        safe_check_mode=DynamicEmbCheckMode.IGNORE,
        # A cache only exists when the values do not fit in HBM
        # (``total_memory > local_hbm``), so a caching test has to shrink this --
        # ``caching=True`` alone silently yields a plain HBM_ONLY table.
        local_hbm_for_values=local_hbm_for_values,
        score_strategy=score_strategy,
        caching=caching,
        evicted_item_mode=evicted_item_mode,
        # Only when asked: pinning init to max disables rehash growth, which the
        # other tests do not want.
        **({} if init_capacity is None else {"init_capacity": init_capacity}),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        output_dtype=torch.float32,
        table_names=[TABLE_NAME],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=optimizer,
        **opt_params,
    )


def feature_from_keys(keys: List[int], device: torch.device):
    """One SUM-pooled feature, one key per bag."""
    indices = torch.tensor(keys, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) + 1, dtype=torch.int64, device=device)
    return indices, offsets


def touch(model, keys: List[int], device: torch.device, backward: bool = False):
    indices, offsets = feature_from_keys(keys, device)
    out = model(indices, offsets)
    if backward:
        out.sum().backward()
    torch.cuda.synchronize()
    return out


def dump_all(model):
    """Everything in the table: threshold 0 matches every score."""
    return model.incremental_dump({TABLE_NAME: 0})


def sorted_view(keys: torch.Tensor, *columns: torch.Tensor):
    """Sort a delta's columns by key so two dumps can be compared row by row."""
    order = torch.argsort(keys)
    return (keys[order],) + tuple(c[order] for c in columns)


def _mask_delta(delta, mask: torch.Tensor, table_id: int = 0) -> None:
    """Keep only ``mask``'s rows, across every column-aligned list at once.

    A delta has five per-key columns plus ``meta["slot_index"]``; masking them
    one by one in a test is how a stale column slips through unnoticed.
    """
    delta.keys[table_id] = delta.keys[table_id][mask]
    delta.values[table_id] = delta.values[table_id][mask]
    delta.scores[table_id] = delta.scores[table_id][mask]
    if delta.optimizer_states[table_id] is not None:
        delta.optimizer_states[table_id] = delta.optimizer_states[table_id][mask]
    delta.meta[table_id]["slot_index"] = delta.meta[table_id]["slot_index"][mask]


def export_optimizer_state(model, table_id: int = 0) -> dict:
    """{key: optimizer state row} straight out of the storage."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    out = {}
    for keys, _, opt_states, _ in model._storage.export_keys_values(
        device, 65536, table_id
    ):
        if opt_states is None:
            continue
        for k, o in zip(keys.cpu().tolist(), opt_states.cpu()):
            out[k] = o.clone()
    return out


@pytest.mark.parametrize(
    "score_strategy",
    [
        DynamicEmbScoreStrategy.TIMESTAMP,
        DynamicEmbScoreStrategy.STEP,
        DynamicEmbScoreStrategy.LFU,
        (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU),
        # NO_EVICTION exercises the packed slot_index (key slot in the high 32
        # bits, value row in the low 32) and the auto-increment row counter.
        DynamicEmbScoreStrategy.NO_EVICTION,
    ],
)
def test_replay_round_trip(current_device, score_strategy):
    """A delta replayed into an identically configured model reproduces it
    key-for-key, value-for-value, at the same slots."""
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, score_strategy=score_strategy)
    dst = make_model(current_device, score_strategy=score_strategy)

    keys = list(range(1001, 1301))
    touch(src, keys, device)
    delta = dump_all(src)
    assert set(delta.keys[0].tolist()) == set(keys)

    stats = dst.replay_increment(delta)[TABLE_NAME]
    assert stats.upserted == len(keys), stats
    assert stats.skipped == 0

    replayed = dump_all(dst)
    src_keys, src_vals = sorted_view(delta.keys[0], delta.values[0])
    dst_keys, dst_vals = sorted_view(replayed.keys[0], replayed.values[0])
    assert torch.equal(src_keys, dst_keys)
    torch.testing.assert_close(src_vals, dst_vals)

    # Precise replay means the same slot, so the source's slot_index round-trips.
    _, src_slots = sorted_view(delta.keys[0], delta.meta[0]["slot_index"])
    _, dst_slots = sorted_view(replayed.keys[0], replayed.meta[0]["slot_index"])
    assert torch.equal(src_slots, dst_slots)

    # A lookup on the replica returns the replayed embeddings. Sort the lookup
    # into the same key order as the delta rather than permuting the delta back
    # into lookup order -- one comparison, no dependence on `keys` being sorted.
    out = touch(dst, keys, device)
    _, out_vals = sorted_view(torch.tensor(keys), out.cpu())
    torch.testing.assert_close(out_vals, src_vals)


@pytest.mark.parametrize(
    "optimizer", [EmbOptimType.SGD, EmbOptimType.EXACT_ROWWISE_ADAGRAD]
)
def test_dump_splits_the_value_row(current_device, optimizer):
    """``values`` and ``optimizer_states`` are the two halves of one stored row.

    They are dumped as separate tensors, so the thing worth pinning down is that
    nothing is lost or duplicated between them: the widths must add up to the
    table's value row, and a table with no per-row state must say so with
    ``None`` rather than a zero-width tensor.
    """
    device = torch.device(f"cuda:{current_device}")
    kwargs = (
        {"learning_rate": 0.1, "eps": 1e-8}
        if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else {"learning_rate": 0.1}
    )
    model = make_model(current_device, optimizer=optimizer, **kwargs)
    keys = list(range(1001, 1051))
    touch(model, keys, device, backward=True)

    delta = dump_all(model)
    n = delta.keys[0].numel()
    assert n == len(keys)
    assert delta.values[0].shape == (n, DIM)

    value_dim = model._storage.value_dim(0)
    opt = delta.optimizer_states[0]
    if value_dim == DIM:
        assert opt is None, "a table with no per-row state must report None"
        return

    # The width is the file checkpoint's, not the runtime row's. Rowwise Adagrad
    # differs between the two -- 16 bytes reserved per row, one scalar used -- so
    # asserting against value_dim - DIM would pin the padding instead.
    optimizer_obj = model._storage._state.optimizer
    ckpt_dim = optimizer_obj.get_ckpt_state_dim(DIM)
    assert opt is not None and opt.shape == (n, ckpt_dim)
    if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        assert ckpt_dim < value_dim - DIM, (
            "this parametrisation is meant to cover the case where the dumped "
            "width is narrower than the runtime row"
        )


def test_dump_scores_are_column_aligned(current_device):
    """``scores`` carries every score word, one row per key, in logical order."""
    device = torch.device(f"cuda:{current_device}")
    strategy = (DynamicEmbScoreStrategy.TIMESTAMP, DynamicEmbScoreStrategy.LFU)
    model = make_model(current_device, score_strategy=strategy)
    hot = list(range(1001, 1051))
    cold = list(range(2001, 2051))
    touch(model, cold, device)
    for _ in range(5):
        touch(model, hot, device)

    delta = dump_all(model)
    keys, scores = sorted_view(delta.keys[0], delta.scores[0])
    assert scores.shape == (keys.numel(), len(strategy))

    # Column 1 is the LFU frequency, carried verbatim: the keys touched six
    # times must outrank those touched once. Column 0 is a timestamp held as an
    # age, so it is not compared against a raw clock here.
    hot_set = set(hot)
    is_hot = torch.tensor([int(k) in hot_set for k in keys.tolist()])
    assert int(scores[is_hot, 1].min()) > int(scores[~is_hot, 1].max())


@pytest.mark.parametrize(
    "content, replica_ranks_like_source",
    [(ReplayContent.EMBEDDING, False), (ReplayContent.ALL, True)],
)
def test_replay_scores_follow_the_content_flag(
    current_device, content, replica_ranks_like_source
):
    """Whether the replica inherits the source's ranking is ``SCORE``'s call.

    Same setup either way -- hot keys accessed six times, cold keys once, and a
    threshold drawn between them that splits the source's table exactly in half.
    Without ``SCORE`` every restored key carries the same fresh score, so the
    same threshold splits nothing; with it the frequency column is carried
    verbatim and the split is reproduced. The pair is what makes the limitation
    legible: neither half means much alone.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.LFU)
    dst = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.LFU)

    hot = list(range(1001, 1051))
    cold = list(range(2001, 2051))
    touch(src, cold, device)
    for _ in range(5):  # drive the two groups' LFU counts apart
        touch(src, hot, device)

    all_keys = set(hot) | set(cold)
    threshold = 3  # between the cold keys' 1 access and the hot keys' 6
    assert set(src.incremental_dump({TABLE_NAME: threshold}).keys[0].tolist()) == set(
        hot
    ), "the source must rank hot keys above cold ones, or there is nothing to inherit"

    dst.replay_increment(dump_all(src), content=content)
    assert set(dump_all(dst).keys[0].tolist()) == all_keys

    split = set(dst.incremental_dump({TABLE_NAME: threshold}).keys[0].tolist())
    if replica_ranks_like_source:
        assert split == set(hot), "the replica must rank the way the source did"
    else:
        # Every key scored alike, so the threshold takes all of them or none --
        # which of the two depends on where the fresh score falls, and that is
        # an implementation detail this deliberately does not pin down.
        assert split in (set(), all_keys), f"got {len(split)} of {len(all_keys)}"


def test_replay_spans_multiple_batches(current_device):
    """Replay is chunked, and every column has to be chunked the same way.

    The columns are not all the same rank -- ``keys`` / ``slot_index`` are 1-D,
    ``values`` / ``optimizer_states`` / ``scores`` are ``[N, ...]`` -- so the
    slicing has to cut dimension 0 in each case, or a batch would pair one key's
    row with another key's slot. Nothing else here notices: batches are sized to
    the device (``threads_in_wave``, hundreds of thousands of keys), so every
    other test in this file fits in one and never crosses a boundary. Shrink it
    so the chunking actually runs.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(
        current_device,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )
    dst = make_model(
        current_device,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )

    keys = list(range(1001, 1101))
    touch(src, keys, device, backward=True)
    delta = dump_all(src)
    assert delta.values[0].dim() == 2 and delta.scores[0].dim() == 2
    assert delta.optimizer_states[0] is not None

    # 7 keys per batch: not a divisor of 100, so the last batch is short too.
    dst._storage._state.threads_in_wave = 7
    dst.replay_increment(delta, content=ReplayContent.ALL)

    out = dump_all(dst)
    src_k, src_v = sorted_view(delta.keys[0], delta.values[0])
    dst_k, dst_v = sorted_view(out.keys[0], out.values[0])
    assert torch.equal(src_k, dst_k)
    torch.testing.assert_close(src_v, dst_v)
    # Slicing the wrong dimension would still produce the right key set, so the
    # per-key payload is what actually pins it down.
    got, want = export_optimizer_state(dst), export_optimizer_state(src)
    for k in keys:
        torch.testing.assert_close(got[k], want[k])


def test_replay_rejects_layout_mismatch(current_device):
    """A target whose layout differs cannot take the source's slots, so the whole
    table is rejected -- and nothing is written."""
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, max_capacity=DEFAULT_CAPACITY)
    dst = make_model(current_device, max_capacity=DEFAULT_CAPACITY * 2)

    keys = list(range(1001, 1201))
    touch(src, keys, device)
    delta = dump_all(src)

    with pytest.raises(ValueError, match="capacity mismatch"):
        dst.replay_increment(delta)

    # Rejection must be total: no partial write is left behind.
    assert dump_all(dst).keys[0].numel() == 0


def test_replay_rejects_score_strategy_mismatch(current_device):
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.TIMESTAMP)
    dst = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.LFU)
    touch(src, list(range(1001, 1051)), device)
    delta = dump_all(src)
    with pytest.raises(ValueError, match="score_strategy mismatch"):
        dst.replay_increment(delta)
    assert dump_all(dst).keys[0].numel() == 0, "rejection must be total"


def test_replay_accepts_swapped_score_order(current_device):
    """Same two score words, opposite configured order, must still replay.

    The tuple order is only ever the checkpoint column order: ``(TIMESTAMP, LFU)``
    and ``(LFU, TIMESTAMP)`` are the same physical layout, so a source slot means
    the same thing in both. With no scores in flight there is nothing left for
    the order to break, and rejecting the pair would be over-strict.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(
        current_device,
        score_strategy=(
            DynamicEmbScoreStrategy.TIMESTAMP,
            DynamicEmbScoreStrategy.LFU,
        ),
    )
    dst = make_model(
        current_device,
        score_strategy=(
            DynamicEmbScoreStrategy.LFU,
            DynamicEmbScoreStrategy.TIMESTAMP,
        ),
    )
    keys = list(range(1001, 1051))
    touch(src, keys, device)
    delta = dump_all(src)

    dst.replay_increment(delta)
    src_keys, src_vals = sorted_view(delta.keys[0], delta.values[0])
    out = dump_all(dst)
    dst_keys, dst_vals = sorted_view(out.keys[0], out.values[0])
    assert torch.equal(src_keys, dst_keys)
    torch.testing.assert_close(src_vals, dst_vals)


def test_replay_rejects_missing_table_options(current_device):
    """A delta without table_options must be rejected, not replayed with the
    score-order / dim / dist_type checks quietly skipped."""
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device)
    dst = make_model(current_device)
    touch(src, list(range(1001, 1051)), device)
    delta = dump_all(src)
    delta.meta[0].pop("table_options")

    with pytest.raises(ValueError, match="carries no 'table_options'"):
        dst.replay_increment(delta)
    assert dump_all(dst).keys[0].numel() == 0


def test_replay_applies_erasures_and_ignores_evictions(current_device):
    """``erased_keys`` is replayed; ``evicted_keys`` never is.

    An eviction needs no action on the replica -- the key that took the evicted
    one's slot is in this very delta and overwrites it. An erase does, because
    nothing takes over that slot. Feeding a non-empty ``evicted_keys`` here would
    delete live keys if the two were ever confused.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device)
    dst = make_model(current_device)

    keys = list(range(1001, 1101))
    touch(src, keys, device)
    delta = dump_all(src)
    dst.replay_increment(delta)
    assert set(dump_all(dst).keys[0].tolist()) == set(keys)

    # A delta that erases half the keys and re-adds one of them.
    dropped = keys[:50]
    readmitted = dropped[0]
    keep = keys[50:]
    next_delta = dump_all(src)
    mask = torch.tensor([int(k) == readmitted for k in next_delta.keys[0].tolist()])
    _mask_delta(next_delta, mask)
    next_delta.erased_keys[0] = torch.tensor(dropped, dtype=torch.int64)
    # The surviving keys, offered as evictions: replay must not touch them.
    next_delta.evicted_keys[0] = torch.tensor(keep, dtype=torch.int64)

    stats = dst.replay_increment(next_delta)[TABLE_NAME]
    assert stats.erased == len(dropped)
    left = set(dump_all(dst).keys[0].tolist())
    assert readmitted in left, "an erased-then-readmitted key must survive"
    assert left == set(keep) | {
        readmitted
    }, "evicted_keys must not have removed anything"

    # An absent list is not an error: a table may never have been erased from,
    # and under row-wise sharding a rank may own none of the keys that were.
    # ``None`` rather than an empty tensor, since a hand-built delta may do that.
    next_delta.erased_keys[0] = None
    next_delta.evicted_keys[0] = None
    assert dst.replay_increment(next_delta)[TABLE_NAME].erased == 0
    assert set(dump_all(dst).keys[0].tolist()) == left


def test_replay_erased_counts_only_keys_actually_present(current_device):
    """``erased`` reports removals that really happened, not removals asked for.

    A delta may name keys this replica never held -- e.g. it was built from a
    wider source, or the key was already gone. Counting the request instead of
    the effect would silently overstate what the replica did.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device)
    dst = make_model(current_device)

    keys = list(range(1001, 1051))
    touch(src, keys, device)
    delta = dump_all(src)
    dst.replay_increment(delta)

    present = keys[:20]
    absent = list(range(90001, 90031))  # never inserted anywhere
    delta.erased_keys[0] = torch.tensor(present + absent, dtype=torch.int64)
    # Drop the upsert side so the removals are not immediately undone.
    _mask_delta(delta, torch.zeros(delta.keys[0].numel(), dtype=torch.bool))

    stats = dst.replay_increment(delta)[TABLE_NAME]
    assert stats.erased == len(present), (
        f"expected {len(present)} real removals out of "
        f"{len(present) + len(absent)} requested, got {stats.erased}"
    )
    assert set(dump_all(dst).keys[0].tolist()) == set(keys) - set(present)


def test_replay_optimizer_state(current_device):
    """Without ``ReplayContent.OPTIMIZER_STATE``, the state is preserved for a key
    already sitting at its target slot and initialised for one that is not."""
    device = torch.device(f"cuda:{current_device}")
    src = make_model(
        current_device,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )
    dst = make_model(
        current_device,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )

    keys = list(range(1001, 1101))
    touch(src, keys, device, backward=True)
    delta = dump_all(src)

    # First replay: brand-new rows, so the optimizer state starts at its initial.
    dst.replay_increment(delta, content=ReplayContent.EMBEDDING)
    fresh_state = export_optimizer_state(dst)
    assert fresh_state, "expected an optimizer state region for ROWWISE_ADAGRAD"
    for key in keys:
        assert float(fresh_state[key].abs().max()) == 0.0

    # Train the replica so its optimizer state diverges from the initial value.
    touch(dst, keys, device, backward=True)
    trained_state = export_optimizer_state(dst)
    assert any(float(trained_state[k].abs().max()) > 0.0 for k in keys)

    # Second replay of the same delta: every key is already in its target slot,
    # so the embedding is overwritten but the optimizer state must survive.
    dst.replay_increment(delta, content=ReplayContent.EMBEDDING)
    after_state = export_optimizer_state(dst)
    for key in keys:
        torch.testing.assert_close(after_state[key], trained_state[key])


def test_replay_content_restores_optimizer_state(current_device):
    """With ``OPTIMIZER_STATE``, the source's state is written -- even onto a row
    the key is taking over, where the default would have initialised it."""
    device = torch.device(f"cuda:{current_device}")
    kwargs = dict(
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )
    src = make_model(current_device, **kwargs)
    dst = make_model(current_device, **kwargs)

    keys = list(range(1001, 1101))
    touch(src, keys, device, backward=True)
    src_state = export_optimizer_state(src)
    assert any(float(src_state[k].abs().max()) > 0.0 for k in keys), (
        "the source must have trained, or there is no state to distinguish "
        "from the initial value"
    )

    # Brand-new rows on the target: without the flag these would be initialised.
    dst.replay_increment(dump_all(src), content=ReplayContent.ALL)
    got = export_optimizer_state(dst)
    for key in keys:
        torch.testing.assert_close(got[key], src_state[key])


def test_replay_without_embedding_needs_aligned_rows(current_device):
    """Omitting ``EMBEDDING`` is only legal for an already-aligned replica.

    A key landing on a row it does not already own has no embedding to keep, and
    the row still holds the previous occupant's vector -- serving that under the
    new key would be silent corruption, so replay refuses.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device)
    dst = make_model(current_device)
    keys = list(range(1001, 1051))
    touch(src, keys, device)
    delta = dump_all(src)

    score_only = ReplayContent.SCORE
    with pytest.raises(ValueError, match="do not already occupy their target row"):
        dst.replay_increment(delta, content=score_only)

    # Align the replica first; then the same call is fine.
    dst.replay_increment(delta)
    dst.replay_increment(delta, content=score_only)
    assert set(dump_all(dst).keys[0].tolist()) == set(keys)


def test_replay_does_not_leave_displaced_keys_in_the_cache(current_device):
    """A key whose slot a delta key takes over must not survive in the cache.

    Writing a key at its source slot reproduces an eviction in the *storage* --
    the slot's new owner overwrites the old one. It does not reach the cache,
    which is a separate index the slot write knows nothing about. So the
    displaced key is in neither ``keys`` nor ``erased_keys``, and an
    invalidation working from those two lists alone leaves it behind.

    Letting it survive is worse than a stale read: ``flush_cache`` pushes every
    cached key back down, so the next dump would *resurrect* it into storage and
    it would reappear in the dump -- which is what this asserts against.
    """
    device = torch.device(f"cuda:{current_device}")
    # One bucket, so any key may legally occupy any slot (see below), and room
    # to spare so that nothing here depends on the table being full.
    # 1024 rows * 8 dims * 4 bytes = 32 KiB of values; half in HBM leaves a real
    # GPU cache in front of a host-backed store.
    cfg = dict(
        max_capacity=1024,
        init_capacity=1024,
        bucket_capacity=1024,
        caching=True,
        local_hbm_for_values=16384,
    )
    src = make_model(current_device, **cfg)
    dst = make_model(current_device, **cfg)
    assert dst._cache is not None, "this test is meaningless without a cache"

    keys = list(range(1001, 1129))
    touch(src, keys, device)
    delta = dump_all(src)
    dst.replay_increment(delta)
    assert set(dump_all(dst).keys[0].tolist()) == set(keys), "must start converged"

    # Pull one key into the replica's cache while it is still resident there.
    victim = int(delta.keys[0][0].item())
    touch(dst, [victim], device)

    # Now hand the replica a delta in which a *different* key claims the
    # victim's slot -- what the source produces when it evicts the victim and
    # admits someone else in its place.
    #
    # Constructed rather than provoked: which key a table evicts, and whether it
    # evicts at all, depends on the eviction policy, on ref-counter pinning and
    # on how full the table happens to be -- none of which this test is about,
    # and all of which vary by GPU. The single bucket makes the substitution
    # legal, since a slot only has to lie in its key's home bucket.
    taker = 9001
    delta.keys[0] = delta.keys[0].clone()
    delta.keys[0][0] = taker
    delta.evicted_keys[0] = torch.tensor([victim], dtype=delta.keys[0].dtype)

    dst.replay_increment(delta)

    # dump_all flushes the cache into storage first, so a surviving cached copy
    # of the victim would reappear here.
    left = set(dump_all(dst).keys[0].tolist())
    assert taker in left, "the taker must hold the slot it was given"
    assert (
        victim not in left
    ), f"key {victim} lost its slot to {taker} but survived in the cache"


def test_replay_with_caching(current_device):
    """With a cache in front of the storage, a lookup after replay must see the
    replayed values, not the stale cached ones.

    Only the cached layout is worth a case here: without a cache this is
    test_replay_round_trip's closing lookup. The HBM budget has to be shrunk or
    the values fit and no cache is built at all.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, caching=True, local_hbm_for_values=2048)
    dst = make_model(current_device, caching=True, local_hbm_for_values=2048)
    assert dst._cache is not None, "this test is meaningless without a cache"

    keys = list(range(1001, 1101))
    touch(src, keys, device)
    delta = dump_all(src)

    # Warm the target's cache with different values for the same keys.
    touch(dst, keys, device)
    dst.replay_increment(delta)

    out = touch(dst, keys, device)
    _, src_vals = sorted_view(delta.keys[0], delta.values[0])
    _, out_vals = sorted_view(torch.tensor(keys), out.cpu())
    torch.testing.assert_close(out_vals, src_vals)
