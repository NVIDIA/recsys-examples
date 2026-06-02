import importlib.util

import pytest


def test_select_initial_topk_from_prefill_logits() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk

    logits = torch.tensor([[[0.0, 5.0, 1.0, 3.0, 2.0]]])

    selection = select_initial_topk(logits, beam_width=3)

    assert selection.token_ids == (1, 3, 4)
    assert selection.parent_beams == (0, 0, 0)
    assert selection.width == 3


def test_select_initial_topk_applies_item_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk

    logits = torch.tensor([[0.0, 5.0, 1.0, 3.0, 2.0]])
    item_mask = torch.tensor([True, False, True, True, False])

    selection = select_initial_topk(logits, beam_width=2, item_mask=item_mask)

    assert selection.token_ids == (3, 2)


def test_select_initial_topk_rejects_too_small_item_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk

    logits = torch.tensor([[0.0, 5.0, 1.0, 3.0, 2.0]])
    item_mask = torch.tensor([False, False, True, False, False])

    with pytest.raises(ValueError, match="fewer valid tokens"):
        select_initial_topk(logits, beam_width=2, item_mask=item_mask)


def test_select_initial_topk_rejects_batch_gt_one() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk

    with pytest.raises(ValueError, match="batch_size=1"):
        select_initial_topk(torch.zeros(2, 4), beam_width=2)


def test_select_next_topk_uses_parent_scores() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk

    logits = torch.tensor([[[0.0, 1.0, 2.0], [10.0, 0.0, 0.5]]])

    selection = select_next_topk(
        logits,
        previous_scores=(100.0, 0.0),
        beam_width=2,
    )

    assert selection.parent_beams == (0, 0)
    assert selection.token_ids == (2, 1)
    assert selection.scores == (102.0, 101.0)


def test_select_next_topk_supports_logprob_ranking() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk

    logits = torch.tensor([[[10.0, 9.0], [1.0, -100.0]]])

    raw = select_next_topk(
        logits,
        previous_scores=(0.0, 0.0),
        beam_width=1,
        score_mode="raw_logits",
    )
    logprob = select_next_topk(
        logits,
        previous_scores=(0.0, 0.0),
        beam_width=1,
        score_mode="logprob",
    )

    assert raw.parent_beams == (0,)
    assert raw.token_ids == (0,)
    assert logprob.parent_beams == (1,)
    assert logprob.token_ids == (0,)


def test_select_next_topk_applies_per_beam_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk

    logits = torch.tensor([[[0.0, 1.0, 2.0], [10.0, 0.0, 0.5]]])
    item_mask = torch.tensor([[True, True, False], [False, True, True]])

    selection = select_next_topk(
        logits,
        previous_scores=(0.0, 0.0),
        beam_width=2,
        item_mask=item_mask,
    )

    assert selection.token_ids == (1, 2)


def test_item_mask_limited_beam_width_clamps_to_valid_candidates() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import (
        batched_item_mask_limited_beam_width,
        item_mask_limited_beam_width,
    )

    assert item_mask_limited_beam_width(4, torch.tensor([False, True, True])) == 2
    assert (
        batched_item_mask_limited_beam_width(
            4,
            torch.tensor(
                [
                    [[True, False, False], [False, True, False]],
                    [[False, True, False], [False, False, False]],
                ]
            ),
        )
        == 1
    )


def test_item_mask_limited_beam_width_rejects_empty_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import item_mask_limited_beam_width

    with pytest.raises(ValueError, match="no valid candidates"):
        item_mask_limited_beam_width(2, torch.tensor([False, False]))
