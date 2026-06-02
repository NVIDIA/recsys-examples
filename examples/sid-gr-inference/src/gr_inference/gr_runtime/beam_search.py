# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Beam search helpers for GR decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BeamSelection:
    """One step of selected beam tokens and scores."""

    token_ids: tuple[int, ...]
    scores: tuple[float, ...]
    parent_beams: tuple[int, ...]

    @property
    def width(self) -> int:
        return len(self.token_ids)


class InitialTopKBeamSelector:
    """Select fixed-width initial beams from prefill logits.

    MVP scope: batch=1. Batch-aware BeamPath will come later.
    """

    def __init__(self, beam_width: int, *, score_mode: str = "raw_logits") -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        _validate_score_mode(score_mode)
        self.beam_width = beam_width
        self.score_mode = score_mode

    def __call__(self, logits: Any, *, item_mask: Any | None = None) -> BeamSelection:
        return select_initial_topk(
            logits,
            beam_width=self.beam_width,
            item_mask=item_mask,
            score_mode=self.score_mode,
        )


def select_initial_topk(
    logits: Any,
    *,
    beam_width: int,
    item_mask: Any | None = None,
    score_mode: str = "raw_logits",
) -> BeamSelection:
    """Select top-k tokens from final prefill logits.

    ``logits`` may be shaped [B, S, V] or [B, V]. The current BeamPath is
    request-local rather than batch-aware, so B must be 1.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    _validate_score_mode(score_mode)
    if not hasattr(logits, "dim"):
        raise TypeError("select_initial_topk currently requires a torch-like tensor")

    import torch

    if logits.dim() == 3:
        scores = logits[:, -1, :]
    elif logits.dim() == 2:
        scores = logits
    else:
        raise ValueError(
            f"logits expects [B, S, V] or [B, V], got {tuple(logits.shape)}"
        )

    if scores.shape[0] != 1:
        raise ValueError("initial beam selection currently supports batch_size=1")
    if beam_width > scores.shape[-1]:
        raise ValueError("beam_width exceeds vocabulary size")

    if item_mask is not None:
        mask = item_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != tuple(scores.shape):
            raise ValueError(
                f"item_mask shape must match scores {tuple(scores.shape)}, got {tuple(mask.shape)}"
            )
        scores = scores.masked_fill(~mask.bool(), -torch.inf)

    if score_mode == "logprob":
        scores = torch.log_softmax(scores.float(), dim=-1)
    values, indices = torch.topk(scores[0], k=beam_width)
    if item_mask is not None and not torch.isfinite(values).all():
        raise ValueError("item_mask leaves fewer valid tokens than beam_width")

    token_ids = tuple(int(token) for token in indices.detach().cpu().tolist())
    selected_scores = tuple(float(score) for score in values.detach().cpu().tolist())
    parent_beams = tuple(0 for _ in token_ids)
    return BeamSelection(
        token_ids=token_ids,
        scores=selected_scores,
        parent_beams=parent_beams,
    )


def select_next_topk(
    logits: Any,
    *,
    previous_scores: tuple[float, ...],
    beam_width: int,
    item_mask: Any | None = None,
    score_mode: str = "raw_logits",
) -> BeamSelection:
    """Select next fixed-width beams from per-beam decode logits.

    ``logits`` is shaped [B, W_prev, V]. MVP scope remains batch=1.
    Scores are cumulative: previous beam score + current token logit/score.
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    _validate_score_mode(score_mode)
    if not hasattr(logits, "dim"):
        raise TypeError("select_next_topk currently requires a torch-like tensor")

    import torch

    if logits.dim() != 3:
        raise ValueError(f"logits expects [B, W, V], got {tuple(logits.shape)}")
    if logits.shape[0] != 1:
        raise ValueError("next beam selection currently supports batch_size=1")
    if logits.shape[1] != len(previous_scores):
        raise ValueError(
            f"logits beam width={logits.shape[1]} does not match "
            f"previous_scores={len(previous_scores)}"
        )
    if beam_width > logits.shape[1] * logits.shape[2]:
        raise ValueError("beam_width exceeds candidate count")

    candidate_logits = logits[0]
    if item_mask is not None:
        mask = item_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand_as(candidate_logits)
        if tuple(mask.shape) != tuple(candidate_logits.shape):
            raise ValueError(
                f"item_mask shape must match scores {tuple(candidate_logits.shape)}, got {tuple(mask.shape)}"
            )
        candidate_logits = candidate_logits.masked_fill(~mask.bool(), -torch.inf)

    scores = candidate_logits
    if score_mode == "logprob":
        scores = torch.log_softmax(scores.float(), dim=-1)
    prev = torch.tensor(previous_scores, device=scores.device, dtype=scores.dtype)
    scores = scores + prev[:, None]

    flat = scores.reshape(-1)
    values, flat_indices = torch.topk(flat, k=beam_width)
    if item_mask is not None and not torch.isfinite(values).all():
        raise ValueError("item_mask leaves fewer valid candidates than beam_width")

    vocab_size = scores.shape[1]
    flat_indices_cpu = flat_indices.detach().cpu().tolist()
    values_cpu = values.detach().cpu().tolist()
    parent_beams = tuple(int(index // vocab_size) for index in flat_indices_cpu)
    token_ids = tuple(int(index % vocab_size) for index in flat_indices_cpu)
    selected_scores = tuple(float(score) for score in values_cpu)
    return BeamSelection(
        token_ids=token_ids,
        scores=selected_scores,
        parent_beams=parent_beams,
    )


def item_mask_limited_beam_width(beam_width: int, item_mask: Any | None) -> int:
    """Clamp width to the number of valid candidates exposed by an item mask."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if item_mask is None:
        return beam_width
    valid_candidates = _valid_item_mask_candidates(item_mask)
    if valid_candidates <= 0:
        raise ValueError("item_mask leaves no valid candidates")
    return min(beam_width, valid_candidates)


def _validate_score_mode(score_mode: str) -> None:
    if score_mode not in {"raw_logits", "logprob"}:
        raise ValueError("score_mode must be 'raw_logits' or 'logprob'")


def _valid_item_mask_candidates(item_mask: Any) -> int:
    mask = item_mask.bool()
    return int(mask.sum().item())
