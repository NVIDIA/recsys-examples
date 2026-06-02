import importlib.util

import pytest


def test_select_initial_topk_batched() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk_batched

    logits = torch.tensor(
        [
            [[0.0, 5.0, 1.0, 3.0]],
            [[4.0, 0.0, 2.0, 6.0]],
        ]
    )

    selection = select_initial_topk_batched(logits, beam_width=2)

    assert selection.batch_size == 2
    assert selection.beam_width == 2
    assert selection.token_ids == ((1, 3), (3, 0))
    assert selection.parent_beams == ((0, 0), (0, 0))


def test_select_initial_topk_batched_applies_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk_batched

    logits = torch.tensor([[0.0, 5.0, 1.0, 3.0], [4.0, 0.0, 2.0, 6.0]])
    mask = torch.tensor(
        [
            [True, False, True, True],
            [False, False, True, True],
        ]
    )

    selection = select_initial_topk_batched(logits, beam_width=2, item_mask=mask)

    assert selection.token_ids == ((3, 2), (3, 2))


def test_select_initial_topk_batched_rejects_too_small_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk_batched

    logits = torch.tensor([[0.0, 5.0, 1.0, 3.0], [4.0, 0.0, 2.0, 6.0]])
    mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, True],
        ]
    )

    with pytest.raises(ValueError, match="fewer valid tokens"):
        select_initial_topk_batched(logits, beam_width=2, item_mask=mask)


def test_select_initial_topk_batched_logprob_matches_log_softmax() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_initial_topk_batched

    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 3.0],
            [4.0, 0.0, 2.0, 6.0],
        ],
        dtype=torch.bfloat16,
    )
    mask = torch.tensor(
        [
            [True, False, True, True],
            [False, True, True, True],
        ]
    )

    selection = select_initial_topk_batched(
        logits,
        beam_width=2,
        item_mask=mask,
        score_mode="logprob",
    )

    masked = logits.float().masked_fill(~mask, -torch.inf)
    expected_values, expected_indices = torch.topk(
        torch.log_softmax(masked, dim=-1),
        k=2,
        dim=-1,
    )

    assert selection.token_ids == tuple(
        tuple(int(token) for token in row.tolist()) for row in expected_indices
    )
    for selected_row, expected_row in zip(selection.scores, expected_values):
        assert selected_row == pytest.approx(
            tuple(float(value) for value in expected_row.tolist()),
            abs=1e-6,
        )


def test_select_next_topk_batched_supports_logprob_ranking() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk_batched

    logits = torch.tensor([[[10.0, 9.0], [1.0, -100.0]]])

    raw = select_next_topk_batched(
        logits,
        previous_scores=((0.0, 0.0),),
        beam_width=1,
        score_mode="raw_logits",
    )
    logprob = select_next_topk_batched(
        logits,
        previous_scores=((0.0, 0.0),),
        beam_width=1,
        score_mode="logprob",
    )

    assert raw.parent_beams == ((0,),)
    assert raw.token_ids == ((0,),)
    assert logprob.parent_beams == ((1,),)
    assert logprob.token_ids == ((0,),)


def test_select_next_topk_batched_logprob_matches_log_softmax_with_mask() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk_batched

    logits = torch.tensor(
        [
            [
                [0.5, 3.0, -1.0, 2.0, 0.0, 1.5],
                [2.5, -0.5, 1.0, 0.1, 4.0, 0.2],
                [-1.0, 2.2, 2.1, 0.0, 0.3, 1.9],
            ],
            [
                [1.0, 0.1, 2.0, -2.0, 0.3, 1.2],
                [-1.0, 3.0, 0.5, 2.5, 0.0, 0.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            ],
        ],
        dtype=torch.bfloat16,
    )
    item_mask = torch.tensor(
        [
            [True, True, True, False, True, True],
            [True, False, True, True, True, True],
        ]
    )
    previous_scores = ((0.0, 1.5, -0.2), (0.3, -0.4, 2.0))

    selection = select_next_topk_batched(
        logits,
        previous_scores=previous_scores,
        beam_width=3,
        item_mask=item_mask,
        score_mode="logprob",
    )

    masked = logits.float().masked_fill(~item_mask[:, None, :], -torch.inf)
    scores = torch.log_softmax(masked, dim=-1)
    scores = scores + torch.tensor(previous_scores, dtype=scores.dtype)[:, :, None]
    values, flat_indices = torch.topk(scores.reshape(2, -1), k=3, dim=-1)
    expected_parent_beams = flat_indices // logits.shape[-1]
    expected_token_ids = flat_indices % logits.shape[-1]

    assert selection.parent_beams == tuple(
        tuple(int(parent) for parent in row.tolist()) for row in expected_parent_beams
    )
    assert selection.token_ids == tuple(
        tuple(int(token) for token in row.tolist()) for row in expected_token_ids
    )
    for selected_row, expected_row in zip(selection.scores, values):
        assert selected_row == pytest.approx(
            tuple(float(value) for value in expected_row.tolist()),
            abs=1e-6,
        )


def test_select_next_topk_batched_two_stage_matches_flat_topk() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk_batched

    logits = torch.tensor(
        [
            [
                [0.5, 3.0, -1.0, 2.0, 0.0, 1.5],
                [2.5, -0.5, 1.0, 0.1, 4.0, 0.2],
                [-1.0, 2.2, 2.1, 0.0, 0.3, 1.9],
            ],
            [
                [1.0, 0.1, 2.0, -2.0, 0.3, 1.2],
                [-1.0, 3.0, 0.5, 2.5, 0.0, 0.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            ],
        ]
    )
    previous_scores = ((0.0, 1.5, -0.2), (0.3, -0.4, 2.0))

    selection = select_next_topk_batched(
        logits,
        previous_scores=previous_scores,
        beam_width=3,
        score_mode="logprob",
    )

    scores = torch.log_softmax(logits.float(), dim=-1)
    scores = scores + torch.tensor(previous_scores, dtype=scores.dtype)[:, :, None]
    values, flat_indices = torch.topk(scores.reshape(2, -1), k=3, dim=-1)
    expected_parent_beams = flat_indices // logits.shape[-1]
    expected_token_ids = flat_indices % logits.shape[-1]

    assert selection.parent_beams == tuple(
        tuple(int(parent) for parent in row.tolist()) for row in expected_parent_beams
    )
    assert selection.token_ids == tuple(
        tuple(int(token) for token in row.tolist()) for row in expected_token_ids
    )
    for selected_row, expected_row in zip(selection.scores, values):
        assert selected_row == pytest.approx(
            tuple(float(value) for value in expected_row.tolist())
        )


def test_select_next_topk_batched_tensor_backed_materializes_to_default() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import select_next_topk_batched

    logits = torch.tensor(
        [
            [
                [0.5, 3.0, -1.0, 2.0, 0.0, 1.5],
                [2.5, -0.5, 1.0, 0.1, 4.0, 0.2],
                [-1.0, 2.2, 2.1, 0.0, 0.3, 1.9],
            ],
            [
                [1.0, 0.1, 2.0, -2.0, 0.3, 1.2],
                [-1.0, 3.0, 0.5, 2.5, 0.0, 0.2],
                [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            ],
        ]
    )
    previous_scores = torch.tensor([[0.0, 1.5, -0.2], [0.3, -0.4, 2.0]])

    eager = select_next_topk_batched(
        logits,
        previous_scores=tuple(
            tuple(float(value) for value in row) for row in previous_scores
        ),
        beam_width=3,
        score_mode="logprob",
    )
    tensor_backed = select_next_topk_batched(
        logits,
        previous_scores_tensor=previous_scores,
        beam_width=3,
        score_mode="logprob",
        materialize=False,
    ).materialize()

    assert tensor_backed.token_ids == eager.token_ids
    assert tensor_backed.parent_beams == eager.parent_beams
    for tensor_row, eager_row in zip(tensor_backed.scores, eager.scores):
        assert tensor_row == pytest.approx(eager_row, abs=1e-6)
