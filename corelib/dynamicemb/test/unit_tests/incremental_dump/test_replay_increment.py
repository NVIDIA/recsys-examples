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
    """``scores`` is one row per key, one column per configured strategy.

    Configured as ``(LFU, TIMESTAMP)`` on purpose. The physical layout of the
    compound policy is always ``(TIMESTAMP, LFU)``, so the *reversed* order is
    what makes the columns tell logical apart from physical: with the natural
    order the permutation is the identity and the test would pass even with
    ``score_dump_permutation`` deleted.
    """
    device = torch.device(f"cuda:{current_device}")
    strategy = (DynamicEmbScoreStrategy.LFU, DynamicEmbScoreStrategy.TIMESTAMP)
    model = make_model(current_device, score_strategy=strategy)
    hot = list(range(1001, 1051))
    cold = list(range(2001, 2051))
    touch(model, cold, device)
    for _ in range(5):
        touch(model, hot, device)

    delta = dump_all(model)
    keys, scores = sorted_view(delta.keys[0], delta.scores[0])
    assert scores.shape == (keys.numel(), len(strategy))

    hot_set = set(hot)
    is_hot = torch.tensor([int(k) in hot_set for k in keys.tolist()])

    # Column 0 is the frequency, carried verbatim -- exact counts, not an
    # ordering: one access each for the cold keys, five for the hot ones. A
    # comparison would still hold if every count were off by the same amount.
    assert set(scores[is_hot, 0].tolist()) == {5}
    assert set(scores[~is_hot, 0].tolist()) == {1}

    # Column 1 is the timestamp, dumped as an age (cur_ts - score). The hot keys
    # were touched last, so theirs is the smaller -- which pins the direction of
    # the conversion, the part a raw clock cannot be compared against.
    assert int(scores[is_hot, 1].max()) < int(scores[~is_hot, 1].min())


@pytest.mark.parametrize(
    "content, keeps_source_scores",
    [(ReplayContent.EMBEDDING_ONLY, False), (ReplayContent.ALL, True)],
)
def test_replay_scores_follow_the_content_flag(
    current_device, content, keeps_source_scores
):
    """Whether the replica inherits the source's ranking is ``SCORE``'s call.

    Same setup either way -- cold keys accessed once, hot keys five times -- and
    the replica's own scores are read back to see which it got. With ``SCORE``
    the frequencies arrive verbatim; without it every restored key is scored as
    if freshly inserted here, so they all score alike and the replica ranks by
    when it received a key rather than by how the source ranked it. The pair is
    what makes that a visible choice: neither half says much alone.
    """
    device = torch.device(f"cuda:{current_device}")
    src = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.LFU)
    dst = make_model(current_device, score_strategy=DynamicEmbScoreStrategy.LFU)

    hot = list(range(1001, 1051))
    cold = list(range(2001, 2051))
    touch(src, cold, device)
    for _ in range(5):  # drive the two groups' LFU counts apart
        touch(src, hot, device)

    delta = dump_all(src)
    hot_set = set(hot)
    src_keys, src_freq = sorted_view(delta.keys[0], delta.scores[0][:, 0])
    src_is_hot = torch.tensor([k in hot_set for k in src_keys.tolist()])
    assert set(src_freq[src_is_hot].tolist()) == {5}
    assert set(src_freq[~src_is_hot].tolist()) == {
        1
    }, "the source must rank hot above cold, or there is nothing to inherit"

    dst.replay_increment(delta, content=content)

    # Read the replica's scores directly rather than inferring them from what a
    # threshold selects: the frequency is what the flag does or does not carry.
    replayed = dump_all(dst)
    assert set(replayed.keys[0].tolist()) == set(hot) | set(cold)
    keys, freq = sorted_view(replayed.keys[0], replayed.scores[0][:, 0])
    is_hot = torch.tensor([k in hot_set for k in keys.tolist()])

    if keeps_source_scores:
        assert set(freq[is_hot].tolist()) == {5}
        assert set(freq[~is_hot].tolist()) == {1}
    else:
        # One value for every key, whatever it is -- which value a fresh insert
        # assigns is not this test's business, only that the source's ranking
        # did not come along.
        assert len(set(freq.tolist())) == 1, f"got {sorted(set(freq.tolist()))}"


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


def test_replay_rejects_swapped_score_order(current_device):
    """Same two score words, opposite configured order, must be rejected.

    The two agree physically -- ``(TIMESTAMP, LFU)`` and ``(LFU, TIMESTAMP)``
    both store the timestamp at word 0 -- so a source slot does mean the same
    thing in both, and matching on physical layout alone lets the pair through.
    The score block does not survive it: a dump emits its columns in the source's
    configured order, and replay decides which columns are ages to rebase and how
    to permute them by position in the *target's* order. Swapped, that rebases
    the frequency as though it were a timestamp and puts each word in the other's
    slot -- and since the embeddings still land correctly, nothing downstream
    would show it.
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
    touch(src, list(range(1001, 1051)), device)
    delta = dump_all(src)

    with pytest.raises(ValueError, match="score_strategy mismatch"):
        dst.replay_increment(delta)
    assert dump_all(dst).keys[0].numel() == 0, "rejection must be total"


def _two_table_model(current_device, capacities):
    """A module whose one storage holds two logical tables, sized separately.

    Rowwise Adagrad rather than SGD so the rows carry optimizer state: a table
    without any lets a malformed optimizer column through untouched, and these
    tests are about columns being rejected.
    """
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=current_device,
            dim=DIM,
            max_capacity=cap,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        )
        for cap in capacities
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        output_dtype=torch.float32,
        table_names=["t_0", "t_1"],
        feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        use_index_dedup=False,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=0.1,
        initial_accumulator_value=0.0,
    )


def test_replay_rejects_a_multi_table_delta_without_writing_any(current_device):
    """A mismatch on one table must leave the others untouched.

    A delta normally spans a whole collection, so validating each table only when
    its turn comes would let the last one's mismatch raise after the earlier ones
    had already been written -- a partly applied delta, harder to recover from
    than a rejected one. Here the first table matches and the second does not.
    """
    device = torch.device(f"cuda:{current_device}")
    src = _two_table_model(current_device, [DEFAULT_CAPACITY, DEFAULT_CAPACITY])
    dst = _two_table_model(current_device, [DEFAULT_CAPACITY, DEFAULT_CAPACITY * 2])

    keys = list(range(1001, 1101))
    indices = torch.tensor(keys * 2, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) * 2 + 1, dtype=torch.int64, device=device)
    src(indices, offsets)
    torch.cuda.synchronize()

    delta = src.incremental_dump({"t_0": 0, "t_1": 0})
    assert all(k.numel() == len(keys) for k in delta.keys), "both tables must dump"

    with pytest.raises(ValueError, match="capacity mismatch"):
        dst.replay_increment(delta)

    after = dst.incremental_dump({"t_0": 0, "t_1": 0})
    assert all(
        k.numel() == 0 for k in after.keys
    ), "t_0 must not have been written before t_1 was rejected"


@pytest.mark.parametrize(
    "broken", ["scores", "optimizer_states", "no_optimizer_states", "slot_index"]
)
def test_replay_rejects_a_malformed_column_before_writing_any_table(
    current_device, broken
):
    """A bad column on a later table must not let an earlier one through.

    Row counts are not the only shape the write path depends on: the score block
    and the optimizer-state block each have a width, and slot_index a length.
    Those are checked again down in the storage layer, which is callable on its
    own -- but by then this table's predecessors have been written, so the
    collection ends up half advanced. The check has to happen up front.
    """
    device = torch.device(f"cuda:{current_device}")
    src = _two_table_model(current_device, [DEFAULT_CAPACITY, DEFAULT_CAPACITY])
    dst = _two_table_model(current_device, [DEFAULT_CAPACITY, DEFAULT_CAPACITY])

    keys = list(range(1001, 1101))
    indices = torch.tensor(keys * 2, dtype=torch.int64, device=device)
    offsets = torch.arange(0, len(keys) * 2 + 1, dtype=torch.int64, device=device)
    src(indices, offsets)
    torch.cuda.synchronize()

    delta = src.incremental_dump({"t_0": 0, "t_1": 0})
    # Damage the second table only: the first is well formed and would be
    # written first if the check came too late.
    if broken == "scores":
        delta.scores[1] = delta.scores[1].repeat(1, 3)
    elif broken == "optimizer_states":
        delta.optimizer_states[1] = torch.zeros(
            delta.keys[1].numel(), 7, dtype=delta.values[1].dtype
        )
    elif broken == "no_optimizer_states":
        # What a delta from a stateless optimizer looks like -- SGD dumped,
        # rowwise Adagrad replayed. The layout check does not compare
        # optimizers, so only this catches it before the write path does.
        delta.optimizer_states[1] = None
    else:
        delta.meta[1]["slot_index"] = delta.meta[1]["slot_index"][:-1]

    with pytest.raises(ValueError):
        dst.replay_increment(delta)

    after = dst.incremental_dump({"t_0": 0, "t_1": 0})
    assert all(
        k.numel() == 0 for k in after.keys
    ), "t_0 must not have been written before t_1's column was rejected"


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
    dst.replay_increment(delta, content=ReplayContent.EMBEDDING_ONLY)
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
    dst.replay_increment(delta, content=ReplayContent.EMBEDDING_ONLY)
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


def test_replay_rechecks_layout_after_cache_flush_expansion(current_device):
    """A cache flush may rehash storage, invalidating the source slot plan."""
    device = torch.device(f"cuda:{current_device}")
    cfg = dict(
        max_capacity=4096,
        init_capacity=512,
        bucket_capacity=128,
        caching=True,
        local_hbm_for_values=65536,
    )
    src = make_model(current_device, **cfg)
    dst = make_model(current_device, **cfg)

    delta_keys = list(range(1001, 1101))
    extra_keys = list(range(5001, 5451))
    touch(src, delta_keys, device)
    delta = dump_all(src)

    # Keep the target's cache dirty and force its backing table to grow when
    # replay flushes it. The pre-flush plan still sees the original capacity.
    all_keys = delta_keys + extra_keys
    for start in range(0, len(all_keys), 50):
        touch(dst, all_keys[start : start + 50], device)

    with pytest.raises(ValueError, match="capacity mismatch"):
        dst.replay_increment(delta)
