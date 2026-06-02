# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for preparing batched decode-step inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gr_inference.gr_runtime.batched_beam_search import BatchedBeamSelection


@dataclass(frozen=True)
class BatchedDecodeInputs:
    """Inputs shared by a batched decode step."""

    beam_token_ids: Any
    batch_size: int
    beam_width: int


def make_batched_beam_token_ids(
    selection: BatchedBeamSelection,
    *,
    device: Any | None = None,
) -> BatchedDecodeInputs:
    """Build [B, W] beam token ids from a batched beam selection."""

    if selection.batch_size <= 0:
        raise ValueError("selection must contain at least one batch item")
    beam_width = selection.beam_width
    if beam_width <= 0:
        raise ValueError("selection beam_width must be positive")

    import torch

    if selection.token_ids_tensor is not None:
        tensor = selection.token_ids_tensor
        if tuple(tensor.shape) != (selection.batch_size, beam_width):
            raise ValueError(
                "selection token_ids_tensor must match "
                f"{(selection.batch_size, beam_width)}, got {tuple(tensor.shape)}"
            )
        tensor = tensor.to(device=device, dtype=torch.long)
    else:
        for row in selection.token_ids:
            if len(row) != beam_width:
                raise ValueError("all batch rows must have equal beam width")
        tensor = torch.tensor(selection.token_ids, dtype=torch.long, device=device)
    return BatchedDecodeInputs(
        beam_token_ids=tensor,
        batch_size=selection.batch_size,
        beam_width=beam_width,
    )
