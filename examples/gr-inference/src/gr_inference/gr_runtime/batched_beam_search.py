"""Batch-aware beam search helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchedBeamSelection:
    """Top-k beam selection for multiple requests."""

    token_ids: tuple[tuple[int, ...], ...]
    scores: tuple[tuple[float, ...], ...]
    parent_beams: tuple[tuple[int, ...], ...]
    token_ids_tensor: Any | None = None
    scores_tensor: Any | None = None
    parent_beams_tensor: Any | None = None

    @property
    def batch_size(self) -> int:
        if not self.token_ids and self.token_ids_tensor is not None:
            return int(self.token_ids_tensor.shape[0])
        return len(self.token_ids)

    @property
    def beam_width(self) -> int:
        if not self.token_ids and self.token_ids_tensor is not None:
            return int(self.token_ids_tensor.shape[1])
        if not self.token_ids:
            return 0
        return len(self.token_ids[0])

    def materialize(self) -> "BatchedBeamSelection":
        """Return a selection with CPU tuple fields populated."""

        if self.token_ids and self.scores and self.parent_beams:
            return self
        if (
            self.token_ids_tensor is None
            or self.scores_tensor is None
            or self.parent_beams_tensor is None
        ):
            raise ValueError("tensor-backed selection is missing tensor fields")
        token_rows = self.token_ids_tensor.detach().cpu().tolist()
        score_rows = self.scores_tensor.detach().cpu().tolist()
        parent_rows = self.parent_beams_tensor.detach().cpu().tolist()
        return BatchedBeamSelection(
            token_ids=tuple(tuple(int(token) for token in row) for row in token_rows),
            scores=tuple(tuple(float(score) for score in row) for row in score_rows),
            parent_beams=tuple(
                tuple(int(parent) for parent in row) for row in parent_rows
            ),
            token_ids_tensor=self.token_ids_tensor,
            scores_tensor=self.scores_tensor,
            parent_beams_tensor=self.parent_beams_tensor,
        )


def select_initial_topk_batched(
    logits: Any,
    *,
    beam_width: int,
    item_mask: Any | None = None,
    score_mode: str = "raw_logits",
) -> BatchedBeamSelection:
    """Select initial beams from final prefill logits for B requests."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    _validate_score_mode(score_mode)
    if not hasattr(logits, "dim"):
        raise TypeError("select_initial_topk_batched requires a torch-like tensor")

    import torch

    if logits.dim() == 3:
        scores = logits[:, -1, :]
    elif logits.dim() == 2:
        scores = logits
    else:
        raise ValueError(
            f"logits expects [B, S, V] or [B, V], got {tuple(logits.shape)}"
        )
    if beam_width > scores.shape[-1]:
        raise ValueError("beam_width exceeds vocabulary size")

    if item_mask is not None:
        mask = item_mask
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand_as(scores)
        if tuple(mask.shape) != tuple(scores.shape):
            raise ValueError(
                f"item_mask shape must match scores {tuple(scores.shape)}, got {tuple(mask.shape)}"
            )
        scores = scores.masked_fill(~mask.bool(), -torch.inf)

    if score_mode == "logprob":
        scores = torch.log_softmax(scores.float(), dim=-1)
    values, indices = torch.topk(scores, k=beam_width, dim=-1)
    if item_mask is not None and not torch.isfinite(values).all():
        raise ValueError("item_mask leaves fewer valid tokens than beam_width")

    token_rows = indices.detach().cpu().tolist()
    score_rows = values.detach().cpu().tolist()
    token_ids = tuple(tuple(int(token) for token in row) for row in token_rows)
    selected_scores = tuple(tuple(float(score) for score in row) for row in score_rows)
    parent_beams = tuple(tuple(0 for _ in row) for row in token_ids)
    return BatchedBeamSelection(
        token_ids=token_ids,
        scores=selected_scores,
        parent_beams=parent_beams,
    )


def select_next_topk_batched(
    logits: Any,
    *,
    previous_scores: tuple[tuple[float, ...], ...] | None = None,
    previous_scores_tensor: Any | None = None,
    beam_width: int,
    item_mask: Any | None = None,
    score_mode: str = "raw_logits",
    materialize: bool = True,
    validate_finite: bool = True,
) -> BatchedBeamSelection:
    """Select next beams from batched decode logits [B, W_prev, V]."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    _validate_score_mode(score_mode)
    if not hasattr(logits, "dim"):
        raise TypeError("select_next_topk_batched requires a torch-like tensor")

    import torch

    if logits.dim() != 3:
        raise ValueError(f"logits expects [B, W, V], got {tuple(logits.shape)}")
    batch_size, previous_width, vocab_size = logits.shape
    if previous_scores is None and previous_scores_tensor is None:
        raise ValueError("previous_scores or previous_scores_tensor must be provided")
    if previous_scores is not None and batch_size != len(previous_scores):
        raise ValueError("previous_scores batch size must match logits")
    if beam_width > previous_width * vocab_size:
        raise ValueError("beam_width exceeds candidate count")

    candidate_logits = logits
    if item_mask is not None:
        mask = item_mask
        if mask.dim() == 2:
            mask = mask[:, None, :].expand_as(candidate_logits)
        if tuple(mask.shape) != tuple(candidate_logits.shape):
            raise ValueError(
                f"item_mask shape must match scores {tuple(candidate_logits.shape)}, got {tuple(mask.shape)}"
            )
        candidate_logits = candidate_logits.masked_fill(~mask.bool(), -torch.inf)

    local_k = min(beam_width, vocab_size)
    if score_mode == "logprob":
        candidate_scores = candidate_logits.float()
        local_logits, local_token_ids = torch.topk(candidate_scores, k=local_k, dim=-1)
        local_values = local_logits - torch.logsumexp(
            candidate_scores,
            dim=-1,
            keepdim=True,
        )
    else:
        local_values, local_token_ids = torch.topk(candidate_logits, k=local_k, dim=-1)

    if previous_scores_tensor is None:
        prev = torch.tensor(
            previous_scores, device=local_values.device, dtype=local_values.dtype
        )
    else:
        prev = previous_scores_tensor.to(
            device=local_values.device, dtype=local_values.dtype
        )
    if tuple(prev.shape) != (batch_size, previous_width):
        raise ValueError(
            f"previous_scores must be shaped {(batch_size, previous_width)}, got {tuple(prev.shape)}"
        )
    local_values = local_values + prev[:, :, None]

    flat = local_values.reshape(batch_size, -1)
    values, flat_indices = torch.topk(flat, k=beam_width, dim=-1)
    if validate_finite and item_mask is not None and not torch.isfinite(values).all():
        raise ValueError("item_mask leaves fewer valid candidates than beam_width")

    parent_beams_tensor = flat_indices // local_k
    token_ids_tensor = local_token_ids.reshape(batch_size, -1).gather(1, flat_indices)
    if not materialize:
        return BatchedBeamSelection(
            token_ids=(),
            scores=(),
            parent_beams=(),
            token_ids_tensor=token_ids_tensor,
            scores_tensor=values,
            parent_beams_tensor=parent_beams_tensor,
        )

    token_rows = token_ids_tensor.detach().cpu().tolist()
    score_rows = values.detach().cpu().tolist()
    parent_rows = parent_beams_tensor.detach().cpu().tolist()
    return BatchedBeamSelection(
        token_ids=tuple(tuple(int(token) for token in row) for row in token_rows),
        scores=tuple(tuple(float(score) for score in row) for row in score_rows),
        parent_beams=tuple(tuple(int(parent) for parent in row) for row in parent_rows),
        token_ids_tensor=token_ids_tensor,
        scores_tensor=values,
        parent_beams_tensor=parent_beams_tensor,
    )


def batched_item_mask_limited_beam_width(beam_width: int, item_mask: Any | None) -> int:
    """Clamp common batched width to every row's valid item-mask candidate count."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if item_mask is None:
        return beam_width
    mask = item_mask.bool()
    if mask.dim() == 1:
        counts = mask.reshape(1, -1).sum(dim=1)
    else:
        counts = mask.reshape(mask.shape[0], -1).sum(dim=1)
    valid_candidates = int(counts.min().item())
    if valid_candidates <= 0:
        raise ValueError("item_mask leaves no valid candidates")
    return min(beam_width, valid_candidates)


def _validate_score_mode(score_mode: str) -> None:
    if score_mode not in {"raw_logits", "logprob"}:
        raise ValueError("score_mode must be 'raw_logits' or 'logprob'")
