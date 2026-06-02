"""Shared helpers for serving beam metadata and stop checks."""

from __future__ import annotations

from typing import Any, Mapping

from gr_inference.gr_kv import BatchedBeamPath
from gr_inference.gr_serving.request import GRServingRequest


def request_stop_token_ids(request: GRServingRequest) -> tuple[int, ...]:
    provider_stop = tuple(getattr(request.item_mask_provider, "stop_token_ids", ()))
    return tuple(dict.fromkeys((*request.stop_token_ids, *provider_stop)))


def request_selection_all_stop(
    request: GRServingRequest,
    token_ids: tuple[int, ...],
) -> bool:
    stop_token_ids = request_stop_token_ids(request)
    if not stop_token_ids:
        return False
    stop_tokens = set(stop_token_ids)
    return bool(token_ids) and all(token in stop_tokens for token in token_ids)


def request_selection_all_item_complete(
    request: GRServingRequest,
    beam_path: Any,
    beam_width: int,
) -> bool:
    is_complete = getattr(request.item_mask_provider, "is_complete", None)
    if is_complete is None or getattr(beam_path, "steps_done", 0) == 0:
        return False
    return beam_width > 0 and all(
        bool(is_complete(beam_path.token_trace(beam))) for beam in range(beam_width)
    )


def request_selection_stop_reason(
    request: GRServingRequest,
    beam_path: Any,
    token_ids: tuple[int, ...],
) -> str | None:
    if request_selection_all_stop(request, token_ids):
        return "stop_token"
    if request_selection_all_item_complete(request, beam_path, len(token_ids)):
        return "item_complete"
    return None


def batched_selection_stop_reason(
    requests: tuple[GRServingRequest, ...],
    batched_beam_path: BatchedBeamPath,
    token_ids: tuple[tuple[int, ...], ...],
) -> str | None:
    reasons = tuple(
        request_selection_stop_reason(
            request,
            batched_beam_path.paths[index],
            row,
        )
        for index, (request, row) in enumerate(zip(requests, token_ids))
    )
    if reasons and all(reason == "stop_token" for reason in reasons):
        return "stop_token"
    if reasons and all(reason in {"stop_token", "item_complete"} for reason in reasons):
        return "item_complete"
    return None


def beam_width_policy_metadata(policy: Any) -> dict[str, Any]:
    return {
        "type": type(policy).__name__,
        "widths": dict(getattr(policy, "widths", {}) or {}),
    }


def attach_item_results(
    metadata: dict[str, Any],
    *,
    request: GRServingRequest,
    beam_path: Any | None,
    beam_width: int,
) -> None:
    if beam_path is None:
        return
    resolver = getattr(request.item_mask_provider, "beam_item_results", None)
    if resolver is None:
        return
    metadata["item_results"] = resolver(beam_path, beam_width=beam_width)


def beam_results(
    beam_path: Any,
    *,
    beam_width: int,
    stop_reason: str,
    max_new_tokens: int | None = None,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        single_beam_result(
            beam_path,
            beam=beam,
            stop_reason=stop_reason,
            max_new_tokens=max_new_tokens,
        )
        for beam in range(beam_width)
    )


def normalized_beam_results_from_metadata(
    metadata: Mapping[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], ...]:
    existing_results = metadata.get("beam_results")
    if isinstance(existing_results, (list, tuple)):
        normalized = []
        for row in existing_results:
            if not isinstance(row, Mapping):
                continue
            output_ids = row.get("output_ids")
            if not isinstance(output_ids, (list, tuple)):
                continue
            normalized.append(
                {
                    "output_ids": tuple(
                        int(token) for token in output_ids[:max_new_tokens]
                    ),
                    "text": str(row.get("text", "")) if "text" in row else "",
                    "meta_info": dict(row.get("meta_info", {}) or {}),
                }
            )
        return tuple(normalized)

    beam_path = metadata.get("_beam_path")
    beam_width = int(metadata.get("active_beam_width") or 0)
    stop_reason = str(metadata.get("stop_reason", "max_decode_steps"))
    if beam_path is None or beam_width <= 0:
        return ()
    return beam_results(
        beam_path,
        beam_width=beam_width,
        max_new_tokens=max_new_tokens,
        stop_reason=stop_reason,
    )


def single_beam_result(
    beam_path: Any,
    *,
    beam: int,
    stop_reason: str,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    current_beam = beam
    token_ids: list[int] = []
    scores: list[float] = []
    for step in range(beam_path.steps_done - 1, -1, -1):
        entry = beam_path.entries[step]
        token_ids.append(entry.token_ids[current_beam])
        scores.append(entry.scores[current_beam])
        current_beam = entry.parent_beams[current_beam]
    token_ids.reverse()
    scores.reverse()
    if max_new_tokens is not None:
        token_ids = token_ids[:max_new_tokens]
    return {
        "output_ids": tuple(int(token) for token in token_ids),
        "text": "",
        "meta_info": {
            "finish_reason": stop_reason,
            "sequence_score": scores[-1] if scores else 0.0,
        },
    }


def beam_details(
    beam_path: Any,
    *,
    beam_width: int,
    token_logprob_steps: list[tuple[float, ...]] | None = None,
    score_type: str = "beam_score_raw_logits_cumulative",
) -> tuple[dict[str, Any], ...]:
    return tuple(
        single_beam_detail(
            beam_path,
            beam=beam,
            rank=beam,
            token_logprob_steps=token_logprob_steps,
            score_type=score_type,
        )
        for beam in range(beam_width)
    )


def batched_beam_details(
    batched_beam_path: BatchedBeamPath,
    *,
    batch_idx: int,
    beam_width: int,
    token_logprob_steps: list[tuple[tuple[float, ...], ...]] | None = None,
    score_type: str = "beam_score_raw_logits_cumulative",
) -> tuple[dict[str, Any], ...]:
    return beam_details(
        batched_beam_path.paths[batch_idx],
        beam_width=beam_width,
        token_logprob_steps=_batch_token_logprob_steps(
            token_logprob_steps,
            batch_idx=batch_idx,
        ),
        score_type=score_type,
    )


def single_beam_detail(
    beam_path: Any,
    *,
    beam: int,
    rank: int,
    token_logprob_steps: list[tuple[float, ...]] | None = None,
    score_type: str = "beam_score_raw_logits_cumulative",
) -> dict[str, Any]:
    current_beam = beam
    token_ids: list[int] = []
    parent_beams: list[int] = []
    score_trace: list[float] = []
    token_logprobs: list[float] = []
    for step in range(beam_path.steps_done - 1, -1, -1):
        entry = beam_path.entries[step]
        token_ids.append(entry.token_ids[current_beam])
        parent_beams.append(entry.parent_beams[current_beam])
        score_trace.append(entry.scores[current_beam])
        if token_logprob_steps is not None:
            token_logprobs.append(token_logprob_steps[step][current_beam])
        current_beam = entry.parent_beams[current_beam]
    token_ids.reverse()
    parent_beams.reverse()
    score_trace.reverse()
    detail = {
        "rank": rank,
        "token_ids": tuple(token_ids),
        "parent_beams": tuple(parent_beams),
        "score_trace": tuple(score_trace),
        "cumulative_score": score_trace[-1] if score_trace else 0.0,
        "score_type": score_type,
    }
    if token_logprob_steps is not None:
        token_logprobs.reverse()
        detail.update(
            {
                "token_logprobs": tuple(token_logprobs),
                "logprob_sum": sum(token_logprobs),
                "logprob_type": "token_logsoftmax",
            }
        )
    return detail


def beam_score_type(score_mode: str) -> str:
    if score_mode == "logprob":
        return "beam_score_logprob_cumulative"
    return "beam_score_raw_logits_cumulative"


def selected_initial_token_logprobs(
    logits: Any,
    token_ids: tuple[int, ...],
) -> tuple[float, ...]:
    import torch

    scores = logits[:, -1, :] if logits.dim() == 3 else logits
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    return tuple(float(log_probs[0, token].item()) for token in token_ids)


def selected_initial_token_logprobs_batched(
    logits: Any,
    selection: Any,
) -> tuple[tuple[float, ...], ...]:
    import torch

    scores = logits[:, -1, :] if logits.dim() == 3 else logits
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    rows: list[tuple[float, ...]] = []
    for batch_idx, token_row in enumerate(selection.token_ids):
        rows.append(
            tuple(float(log_probs[batch_idx, token].item()) for token in token_row)
        )
    return tuple(rows)


def selected_decode_token_logprobs(
    logits: Any,
    selection: Any,
) -> tuple[tuple[float, ...], ...]:
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    rows: list[tuple[float, ...]] = []
    for batch_idx, (parents, tokens) in enumerate(
        zip(selection.parent_beams, selection.token_ids)
    ):
        rows.append(
            tuple(
                float(log_probs[batch_idx, parent, token].item())
                for parent, token in zip(parents, tokens)
            )
        )
    return tuple(rows)


def _batch_token_logprob_steps(
    token_logprob_steps: list[tuple[tuple[float, ...], ...]] | None,
    *,
    batch_idx: int,
) -> list[tuple[float, ...]] | None:
    if token_logprob_steps is None:
        return None
    return [step[batch_idx] for step in token_logprob_steps]
