# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score fixed Qwen3 candidate continuations with a HuggingFace reference model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tool_utils import iter_jsonl, read_json


def main() -> None:
    args = build_parser().parse_args()
    workload = _load_workload_row(Path(args.workload_jsonl), args.request_id)
    candidates = _candidate_token_ids(args, workload_id=args.request_id)
    report = score_candidates(args, workload["input_ids"], candidates)
    report.update(
        {
            "model_dir": args.model_dir,
            "request_id": args.request_id,
            "prompt_len": len(workload["input_ids"]),
        }
    )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def score_candidates(
    args: argparse.Namespace,
    prompt_ids: list[int],
    candidates: list[list[int]],
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - diagnostic dependency
        raise RuntimeError(
            "This diagnostic requires torch and transformers in the active environment."
        ) from exc

    device = _resolve_device(args.device, torch)
    dtype = _resolve_dtype(args.dtype, device, torch)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    scored = []
    with torch.inference_mode():
        for candidate in candidates:
            input_ids = torch.tensor(
                [prompt_ids + candidate], dtype=torch.long, device=device
            )
            logits = model(input_ids).logits.float()
            prompt_len = len(prompt_ids)
            candidate_logits = logits[
                :, prompt_len - 1 : prompt_len + len(candidate) - 1, :
            ]
            logprobs = candidate_logits.log_softmax(dim=-1)
            target_ids = torch.tensor(candidate, dtype=torch.long, device=device).view(
                1, -1, 1
            )
            token_logprobs = (
                logprobs.gather(dim=-1, index=target_ids).squeeze(0).squeeze(-1)
            )
            token_logprobs_list = [float(value) for value in token_logprobs.cpu()]
            cumulative = sum(token_logprobs_list)
            scored.append(
                {
                    "token_ids": candidate,
                    "token_logprobs": token_logprobs_list,
                    "cumulative_logprob": cumulative,
                    "average_logprob": cumulative / max(len(candidate), 1),
                }
            )

    scored.sort(key=lambda row: row["cumulative_logprob"], reverse=True)
    for rank, row in enumerate(scored):
        row["rank"] = rank
    return {"device": device, "dtype": str(dtype), "candidates": scored}


def _load_workload_row(path: Path, request_id: str) -> dict[str, Any]:
    for line_no, row in iter_jsonl(path):
        if str(row.get("request_id")) == request_id:
            input_ids = row.get("input_ids")
            if not input_ids:
                raise ValueError(f"{path}:{line_no} row has no input_ids")
            return {"input_ids": [int(token) for token in input_ids]}
    raise ValueError(f"request_id={request_id!r} not found in {path}")


def _candidate_token_ids(
    args: argparse.Namespace, *, workload_id: str
) -> list[list[int]]:
    if args.candidate_token_ids:
        return [_parse_token_ids(row) for row in args.candidate_token_ids]
    candidates = []
    if args.gr_json:
        gr_output = _output_by_workload_id(Path(args.gr_json), workload_id)
        if gr_output.get("beam_details"):
            candidates.append(
                [int(token) for token in gr_output["beam_details"][0]["token_ids"]]
            )
    if args.sglang_json:
        sglang_output = _sglang_output_by_workload_id(
            Path(args.sglang_json), workload_id
        )
        if sglang_output.get("beams"):
            candidates.append(
                [int(token) for token in sglang_output["beams"][0]["token_ids"]]
            )
    deduped = []
    seen = set()
    for candidate in candidates:
        key = tuple(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    if not deduped:
        raise ValueError(
            "Provide --candidate-token-ids, or provide --gr-json/--sglang-json "
            "artifacts with beam details."
        )
    return deduped


def _output_by_workload_id(path: Path, workload_id: str) -> dict[str, Any]:
    payload = read_json(path)
    for output in payload.get("outputs", ()) or ():
        if str(output.get("workload_id") or output.get("request_id")) == workload_id:
            return output
    raise ValueError(f"workload_id={workload_id!r} not found in {path}")


def _sglang_output_by_workload_id(path: Path, workload_id: str) -> dict[str, Any]:
    payload = read_json(path)
    runs = payload.get("runs", ()) or ()
    if not runs:
        raise ValueError(f"{path} has no runs")
    for output in runs[-1].get("outputs", ()) or ():
        if str(output.get("workload_id") or output.get("request_id")) == workload_id:
            return output
    raise ValueError(f"workload_id={workload_id!r} not found in {path}")


def _parse_token_ids(value: str) -> list[int]:
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def _resolve_device(device: str, torch) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(dtype: str, device: str, torch):
    if dtype == "auto":
        return torch.bfloat16 if device == "cuda" else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument(
        "--candidate-token-ids",
        action="append",
        help="Candidate continuation token IDs as comma-separated integers. Repeatable.",
    )
    parser.add_argument(
        "--gr-json", help="Optional GR artifact; uses top1 if candidates omitted."
    )
    parser.add_argument(
        "--sglang-json",
        help="Optional SGLang artifact; uses last run top1 if candidates omitted.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--output-json")
    return parser


if __name__ == "__main__":
    main()
