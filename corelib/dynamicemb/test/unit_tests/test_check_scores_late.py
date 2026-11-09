# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A table that takes a score per key (``PRECISE_CUSTOMIZED``) can be asked to look for
a negative score, which the table would read as a very large one
(``safe_check_mode``). Looking waits for the device if the answer is read at once, so
it is read late: the check is started by the call that carries the scores, its answer is
copied back without waiting, and the next call that finds it landed -- or ``flush`` --
raises (``ERROR``) or warns (``WARNING``). A call that finds every slot of the ring of
answers still in flight waits for the oldest of them rather than skip its own check.
"""

import warnings

import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb import sync_free as sf
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

DIM = 8
KEYS = 64


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def build_model(
    current_device: int,
    safe_check_mode: DynamicEmbCheckMode,
    score_strategy=DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
):
    options = DynamicEmbTableOptions(
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=current_device,
        dim=DIM,
        max_capacity=1 << 16,
        safe_check_mode=safe_check_mode,
        local_hbm_for_values=1024**3,
        score_strategy=score_strategy,
        caching=False,
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_options=[options],
        table_names=["t_0"],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.0,
        stochastic_rounding=False,
    )


def batch(first_key: int, device):
    """``KEYS`` keys, one per bag, and a positive score for each."""
    indices = torch.arange(
        first_key, first_key + KEYS, dtype=torch.int64, device=device
    )
    offsets = torch.arange(KEYS + 1, dtype=torch.int64, device=device)
    scores = torch.arange(1, KEYS + 1, dtype=torch.int64, device=device)
    return indices, offsets, scores


@pytest.mark.parametrize(
    "mode", [DynamicEmbCheckMode.ERROR, DynamicEmbCheckMode.WARNING]
)
def test_the_call_that_carries_a_negative_score_does_not_report_it(
    current_device, mode
):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, mode)
    indices, offsets, scores = batch(0, device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here would be an error
        model.prefetch(indices, offsets, scores=-scores)
    torch.cuda.synchronize()


def test_error_is_raised_by_the_next_call(current_device):
    """Within two calls of the one that carried the score: here, the next prefetch."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, offsets, scores = batch(0, device)
    model.prefetch(indices, offsets, scores=-scores)
    torch.cuda.synchronize()  # the answer has landed

    indices, offsets, scores = batch(1000, device)
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.prefetch(indices, offsets, scores=scores)


def test_error_is_raised_by_the_next_forward(current_device):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, offsets, scores = batch(0, device)
    model(indices, offsets, scores=-scores).sum().backward()
    torch.cuda.synchronize()

    indices, offsets, scores = batch(1000, device)
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model(indices, offsets, scores=scores)


def test_warning_is_given_by_the_next_call(current_device):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.WARNING)
    indices, offsets, scores = batch(0, device)
    model.prefetch(indices, offsets, scores=-scores)
    torch.cuda.synchronize()

    indices, offsets, scores = batch(1000, device)
    with pytest.warns(UserWarning, match="scores must be non-negative"):
        model.prefetch(indices, offsets, scores=scores)


@pytest.mark.parametrize(
    "mode,outcome",
    [
        (DynamicEmbCheckMode.ERROR, "raises"),
        (DynamicEmbCheckMode.WARNING, "warns"),
    ],
)
def test_flush_reports_what_was_not_reported_yet(current_device, mode, outcome):
    """No call after the one with the negative score: ``flush`` waits for the answer,
    and reports it. Once."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, mode)
    indices, offsets, scores = batch(0, device)
    model.prefetch(indices, offsets, scores=-scores)

    if outcome == "raises":
        with pytest.raises(ValueError, match="scores must be non-negative"):
            model.flush()
    else:
        with pytest.warns(UserWarning, match="scores must be non-negative"):
            model.flush()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.flush()  # nothing is left to report


def build_lagging_model(current_device: int, mode: DynamicEmbCheckMode):
    """A model whose device is held back by a sleep: none of the answers of the checks
    made from here on has landed."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, mode)
    indices, _, scores = batch(0, device)
    model._check_scores(scores, indices)  # the first launch of the kernels
    torch.cuda.synchronize()
    model.flush()
    torch.cuda._sleep(100_000_000)
    return model, indices, scores


@pytest.mark.parametrize(
    "mode,outcome",
    [
        (DynamicEmbCheckMode.ERROR, "raises"),
        (DynamicEmbCheckMode.WARNING, "warns"),
    ],
)
def test_a_check_is_not_dropped_when_the_ring_is_full(current_device, mode, outcome):
    """The device is a whole ring of checks behind, so the call with the negative score
    finds no free slot. It waits for the oldest answer and takes its slot, instead of
    leaving its own check unmade."""
    model, indices, scores = build_lagging_model(current_device, mode)
    slots = model._score_ring.slots
    for _ in range(slots):
        model._check_scores(scores, indices)
    assert len(model._score_pending) == slots  # none has landed: the ring is full

    model._check_scores(-scores, indices)
    assert len(model._score_pending) == slots

    if outcome == "raises":
        with pytest.raises(ValueError, match="scores must be non-negative"):
            model.flush()
    else:
        with pytest.warns(UserWarning, match="scores must be non-negative"):
            model.flush()


@pytest.mark.parametrize(
    "mode,outcome",
    [
        (DynamicEmbCheckMode.ERROR, "raises"),
        (DynamicEmbCheckMode.WARNING, "warns"),
    ],
)
def test_the_call_that_waits_for_a_slot_reports_the_oldest_answer(
    current_device, mode, outcome
):
    """The answer it waits for is reported like any other: by that call, once."""
    model, indices, scores = build_lagging_model(current_device, mode)
    slots = model._score_ring.slots
    model._check_scores(-scores, indices)  # the oldest answer
    for _ in range(slots - 1):
        model._check_scores(scores, indices)
    assert len(model._score_pending) == slots

    if outcome == "raises":
        with pytest.raises(ValueError, match="scores must be non-negative"):
            model._check_scores(scores, indices)
    else:
        with pytest.warns(UserWarning, match="scores must be non-negative"):
            model._check_scores(scores, indices)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.flush()  # nothing is left to report


def test_scores_that_are_not_negative_report_nothing(current_device):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    for step in range(3):
        indices, offsets, scores = batch(step * 1000, device)
        model(indices, offsets, scores=scores).sum().backward()
    torch.cuda.synchronize()
    model.flush()


def test_a_table_that_does_not_check_has_no_ring(current_device):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.IGNORE)
    assert model._score_ring is None
    indices, offsets, scores = batch(0, device)
    model.prefetch(indices, offsets, scores=-scores)
    model.flush()  # a negative score is not looked for, so none is reported

    # Tables that take no score per key have nothing to check either.
    plain = build_model(
        current_device,
        DynamicEmbCheckMode.ERROR,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
    )
    assert plain._score_ring is None


def test_checking_waits_for_nothing(current_device):
    """Sync-debug mode raises on anything that waits for the device."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, offsets, scores = batch(0, device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        model._check_scores(-scores, indices)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def test_a_training_forward_checks_once_and_an_evaluation_forward_checks(
    current_device, monkeypatch
):
    """In training the prefetch of the batch checks its scores, so the forward does not
    check them again; evaluation prefetches nothing, so the forward does."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    asked = []
    real = sf.DeferredRead.enqueue

    def spy(self, dev):
        if self is model._score_ring:
            asked.append(dev)
        return real(self, dev)

    monkeypatch.setattr(sf.DeferredRead, "enqueue", spy)
    indices, offsets, scores = batch(0, device)

    model(indices, offsets, scores=scores).sum().backward()
    assert len(asked) == 1

    model.eval()
    with torch.no_grad():
        model(indices, offsets, scores=scores)
    assert len(asked) == 2
