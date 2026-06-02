# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


def test_make_batched_beam_token_ids() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_runtime import (
        BatchedBeamSelection,
        make_batched_beam_token_ids,
    )

    selection = BatchedBeamSelection(
        token_ids=((1, 3), (4, 2)),
        scores=((0.9, 0.8), (0.7, 0.6)),
        parent_beams=((0, 0), (0, 0)),
    )

    inputs = make_batched_beam_token_ids(selection)

    assert inputs.batch_size == 2
    assert inputs.beam_width == 2
    assert tuple(inputs.beam_token_ids.shape) == (2, 2)
    assert torch.equal(inputs.beam_token_ids, torch.tensor([[1, 3], [4, 2]]))


def test_make_batched_beam_token_ids_rejects_ragged_rows() -> None:
    from gr_inference.gr_runtime import (
        BatchedBeamSelection,
        make_batched_beam_token_ids,
    )

    selection = BatchedBeamSelection(
        token_ids=((1, 2), (3,)),
        scores=((0.0, 0.0), (0.0,)),
        parent_beams=((0, 0), (0,)),
    )

    with pytest.raises(ValueError, match="equal beam width"):
        make_batched_beam_token_ids(selection)
