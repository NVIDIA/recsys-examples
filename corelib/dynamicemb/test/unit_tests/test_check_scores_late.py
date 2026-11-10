# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A table that takes a score per key (``PRECISE_CUSTOMIZED``) can be asked to look for
a negative score, which the table would read as a very large one
(``safe_check_mode``). Looking waits for the device if the answer is read at once, so
it is read late: the device counts the calls that carried a negative score, the call that
carries the scores copies a snapshot of the count back without waiting, and the next call
that finds a snapshot landed with a larger count -- or ``flush`` -- raises (``ERROR``) or
warns (``WARNING``). A call that finds every slot of the ring of snapshots still in flight
does not wait for one: its check is in the count, and the next snapshot (or ``flush``)
carries it.
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


def count_waits(monkeypatch):
    """The tickets that the host blocked on (``DeferredRead.wait``) from here on."""
    waited = []
    real = sf.DeferredRead.wait

    def spy(self, ticket):
        waited.append(ticket)
        return real(self, ticket)

    monkeypatch.setattr(sf.DeferredRead, "wait", spy)
    return waited


@pytest.mark.parametrize(
    "mode,outcome",
    [
        (DynamicEmbCheckMode.ERROR, "raises"),
        (DynamicEmbCheckMode.WARNING, "warns"),
    ],
)
def test_a_check_is_not_dropped_when_the_ring_is_full(
    current_device, mode, outcome, monkeypatch
):
    """The device is a whole ring of checks behind, so the call with the negative score
    finds no free slot. It does not wait for one, and its check is not lost: the count
    has it, and ``flush`` reports it."""
    model, indices, scores = build_lagging_model(current_device, mode)
    slots = model._score_ring.slots
    waited = count_waits(monkeypatch)
    for _ in range(slots):
        model._check_scores(scores, indices)
    assert len(model._score_pending) == slots  # none has landed: the ring is full

    model._check_scores(-scores, indices)
    assert len(model._score_pending) == slots  # no slot, so no snapshot of its own
    assert not waited, "a check waited for the device"

    if outcome == "raises":
        with pytest.raises(ValueError, match="scores must be non-negative"):
            model.flush()
    else:
        with pytest.warns(UserWarning, match="scores must be non-negative"):
            model.flush()
    assert waited, "flush is an explicit call: it waits for the answers"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.flush()  # nothing is left to report


@pytest.mark.parametrize(
    "mode,outcome",
    [
        (DynamicEmbCheckMode.ERROR, "raises"),
        (DynamicEmbCheckMode.WARNING, "warns"),
    ],
)
def test_the_first_call_that_finds_the_oldest_answer_landed_reports_it(
    current_device, mode, outcome, monkeypatch
):
    """Nothing waits for the oldest answer: once the device catches up, the next call
    reports it, once, however many calls came in between."""
    model, indices, scores = build_lagging_model(current_device, mode)
    slots = model._score_ring.slots
    waited = count_waits(monkeypatch)
    model._check_scores(-scores, indices)  # the oldest answer
    for _ in range(slots):  # the last of them finds the ring full
        model._check_scores(scores, indices)
    assert len(model._score_pending) == slots
    assert not waited, "a check waited for the device"

    torch.cuda.synchronize()  # the device catches up: every answer has landed
    if outcome == "raises":
        with pytest.raises(ValueError, match="scores must be non-negative"):
            model._check_scores(scores, indices)
    else:
        with pytest.warns(UserWarning, match="scores must be non-negative"):
            model._check_scores(scores, indices)
    assert not waited
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.flush()  # nothing is left to report


def test_negative_batches_that_found_no_slot_are_reported_once(current_device):
    """Each negative batch that gets a snapshot of its own is reported by it; those that
    find the ring full share the one that ``flush`` takes, and are reported once."""
    model, indices, scores = build_lagging_model(
        current_device, DynamicEmbCheckMode.WARNING
    )
    for _ in range(model._score_ring.slots):
        model._check_scores(scores, indices)  # the ring is full
    for _ in range(3):
        model._check_scores(-scores, indices)  # no snapshot of their own
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.flush()
    reported = [w for w in caught if "non-negative" in str(w.message)]
    assert len(reported) == 1, caught


def test_a_check_whose_call_raised_an_earlier_report_is_still_counted(current_device):
    """The call that raises the report of an earlier check has counted its own check
    already: ``flush`` reports it, and then nothing is left."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, _, scores = batch(0, device)
    model._check_scores(-scores, indices)
    torch.cuda.synchronize()  # its answer has landed
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model._check_scores(-scores, indices)  # raises the first, carries a second
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()  # the second
    model.flush()  # nothing is left


def test_flush_on_another_stream_sees_a_check_that_found_no_slot(current_device):
    """The checks run on a side stream that is a whole ring behind, and the last call,
    which carries the negative score, finds no slot. ``flush`` on the default stream,
    with nothing in between that waits for the side stream, must still report it: its
    snapshot has to come after the side stream's count."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, _, scores = batch(0, device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        model._check_scores(scores, indices)  # the first launch of the kernels
    side.synchronize()
    model.flush()

    slots = model._score_ring.slots
    with torch.cuda.stream(side):
        torch.cuda._sleep(100_000_000)
        for _ in range(slots):
            model._check_scores(scores, indices)
        model._check_scores(-scores, indices)
    assert len(model._score_pending) == slots  # the last check has no snapshot
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()


def test_checks_that_alternate_between_two_streams_count_each_negative_once(
    current_device,
):
    """Checks that alternate between two streams, one of them with a negative score:
    the count ends at one, and ``flush`` reports it. (Two adds that overlap in time
    and lose one of the increments cannot be made to happen on demand: the test below
    pins the order that rules them out.)"""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, _, scores = batch(0, device)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for i in range(12):
        with torch.cuda.stream(streams[i % 2]):
            model._check_scores(scores, indices)
    with torch.cuda.stream(streams[1]):
        model._check_scores(-scores, indices)
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()
    torch.cuda.synchronize()
    assert int(model._score_neg_total) == 1


def test_a_snapshot_taken_on_another_stream_counts_the_checks_before_it(
    current_device,
):
    """Every add to the count is ordered after the one before it, whichever stream made
    it, so that a snapshot taken after a check has all of them. Stream A is a sleep behind
    and carries a negative check; the next call, from stream B, runs on the device at once
    and takes its snapshot: it has to wait for A's add, on the device. A snapshot that did
    not would read the count before the negative check had been added to it."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    indices, _, scores = batch(0, device)
    a, b = torch.cuda.Stream(), torch.cuda.Stream()
    for stream in (a, b):
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(a):
        model._check_scores(scores, indices)  # the first launch of the kernels
    a.synchronize()
    model.flush()

    with torch.cuda.stream(a):
        torch.cuda._sleep(200_000_000)
        model._check_scores(-scores, indices)  # its add comes after the sleep
    with torch.cuda.stream(b):
        model._check_scores(scores, indices)  # clean; its snapshot is taken at once
    (ticket, _) = model._score_pending[-1]
    torch.cuda.synchronize()
    assert int(model._score_ring.value(ticket)[0]) == 1
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()


def test_the_first_check_on_another_stream_is_ordered_after_the_count_was_made(
    current_device,
):
    """The count is zero-filled on the stream that makes it, and the first check may run
    on another (the prefetch stream, say) right after the module is built. The fill is
    queued behind a sleep here, and the check runs on a side stream with nothing that
    makes it wait for the default stream but the module itself: an add that ran before
    the fill would be wiped out by it."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.IGNORE)
    for option in model._dynamicemb_options:
        option.safe_check_mode = DynamicEmbCheckMode.ERROR
    indices, _, scores = batch(0, device)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()

    torch.cuda._sleep(200_000_000)
    model._make_score_ring(device)  # the fill of the count waits for the sleep
    with torch.cuda.stream(side):
        model._check_scores(-scores, indices)
    torch.cuda.synchronize()
    assert int(model._score_neg_total) == 1, "the fill came after the add"
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()


def test_a_ring_made_inside_inference_mode_takes_the_checks_of_a_training_step(
    current_device,
):
    """A module whose check mode was switched on later makes its ring on its first check,
    which can be an evaluation forward under ``torch.inference_mode()``. What it made
    there has to be updated by the training steps after it: an inference tensor may not
    be updated in place outside inference mode."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.IGNORE)
    for option in model._dynamicemb_options:
        option.safe_check_mode = DynamicEmbCheckMode.ERROR
    indices, _, scores = batch(0, device)
    with torch.inference_mode():
        model._check_scores(scores, indices)  # makes the ring, here
    assert not model._score_neg_total.is_inference()
    model._check_scores(scores, indices)  # updates the count, copies into the ring
    torch.cuda.synchronize()
    model.flush()
    model._check_scores(-scores, indices)
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="scores must be non-negative"):
        model.flush()


def test_scores_that_are_not_negative_report_nothing(current_device):
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.ERROR)
    for step in range(3):
        indices, offsets, scores = batch(step * 1000, device)
        model(indices, offsets, scores=scores).sum().backward()
    torch.cuda.synchronize()
    model.flush()


def test_a_check_mode_switched_on_later_makes_its_ring_and_waits_for_nothing(
    current_device,
):
    """The options are mutable: a module built without a check that is asked to check
    afterwards has no ring yet. It makes one on its first check instead of reading the
    scores at once, which would wait for the device."""
    device = torch.device("cuda", current_device)
    model = build_model(current_device, DynamicEmbCheckMode.IGNORE)
    assert model._score_ring is None
    for option in model._dynamicemb_options:
        option.safe_check_mode = DynamicEmbCheckMode.ERROR
    indices, _, scores = batch(0, device)
    model._check_scores(scores, indices)  # makes the ring, once
    assert model._score_ring is not None
    torch.cuda.synchronize()
    model.flush()
    torch.cuda.set_sync_debug_mode("error")
    try:
        model._check_scores(-scores, indices)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="scores must be non-negative"):
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
