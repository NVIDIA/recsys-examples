# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


def test_make_batched_topk_indices_fast_path_decode_nums_one() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import BatchedBeamSelection, make_batched_topk_indices

    path = BatchedBeamPath.create(
        batch_size=2,
        max_decode_steps=2,
        max_beam_width=3,
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((10, 11, 12), (20, 21, 22)),
            scores=((1.0, 0.9, 0.8), (2.0, 1.9, 1.8)),
            parent_beams=((0, 0, 0), (0, 0, 0)),
        )
    )

    topk = make_batched_topk_indices(
        path,
        num_q_heads=4,
        decode_nums=1,
        beam_width=3,
    )

    expected = torch.tensor([0, 1, 2], dtype=torch.int32)
    assert tuple(topk.shape) == (2, 1, 4, 1, 3)
    assert topk.dtype == torch.int32
    assert torch.equal(topk[0, 0, 0, 0], expected)
    assert torch.equal(topk[1, 0, 3, 0], expected)
    assert topk.is_contiguous()


def test_make_batched_topk_indices_follows_parent_chain() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import BatchedBeamSelection, make_batched_topk_indices

    path = BatchedBeamPath.create(
        batch_size=1,
        max_decode_steps=3,
        max_beam_width=2,
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((10, 11),),
            scores=((1.0, 0.9),),
            parent_beams=((0, 0),),
        )
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((20, 21),),
            scores=((2.0, 1.9),),
            parent_beams=((1, 0),),
        )
    )

    topk = make_batched_topk_indices(
        path,
        num_q_heads=4,
        decode_nums=2,
        beam_width=2,
    )

    assert tuple(topk.shape) == (1, 1, 4, 2, 2)
    assert topk.dtype == torch.int32
    assert torch.equal(topk[0, 0, 0], torch.tensor([[1, 0], [2, 3]], dtype=torch.int32))


def test_make_compacted_batched_topk_indices_uses_query_slot_per_step() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import make_compacted_batched_topk_indices

    topk = make_compacted_batched_topk_indices(
        batch_size=2,
        num_q_heads=4,
        decode_nums=3,
        beam_width=2,
    )

    expected = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    assert tuple(topk.shape) == (2, 1, 4, 3, 2)
    assert topk.dtype == torch.int32
    assert torch.equal(topk[0, 0, 0], expected)
    assert torch.equal(topk[1, 0, 3], expected)
    assert topk.is_contiguous()
