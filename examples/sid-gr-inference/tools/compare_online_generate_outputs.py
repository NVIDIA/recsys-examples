# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare GR and SGLang /generate outputs through their online HTTP servers."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping
from urllib import request as urllib_request

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__, include_tools=True)

from make_qwen3_beam_workload import build_workload  # noqa: E402


def main() -> None:
    args = build_parser().parse_args()
    rows = build_workload(args)
    _warmup(args, rows)
    results = compare(args, rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_markdown:
        Path(args.output_markdown).write_text(_markdown(results), encoding="utf-8")
    print(json.dumps(_summary(results), indent=2, sort_keys=True))


def compare(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        gr_outputs = list(pool.map(lambda row: _send(args.gr_url, args, row), rows))
        sglang_outputs = list(
            pool.map(lambda row: _send(args.sglang_url, args, row), rows)
        )

    comparisons = []
    for row, gr_output, sglang_output in zip(
        rows, gr_outputs, sglang_outputs, strict=True
    ):
        gr_top1 = _extract_top1_ids(gr_output)
        sglang_top1 = _extract_top1_ids(sglang_output)
        comparisons.append(
            {
                "request_id": row["request_id"],
                "gr_top1_ids": gr_top1,
                "sglang_top1_ids": sglang_top1,
                "top1_exact": gr_top1 == sglang_top1,
                "gr_text": _extract_text(gr_output),
                "sglang_text": _extract_text(sglang_output),
                "gr_raw": gr_output,
                "sglang_raw": sglang_output,
            }
        )

    return {
        "schema_version": "online_generate_compare_v1",
        "workload": {
            "context_len": args.context_len,
            "requests": args.requests,
            "beam_width": args.beam_width,
            "max_new_tokens": args.max_new_tokens,
            "max_concurrency": args.max_concurrency,
        },
        "summary": _comparison_summary(comparisons),
        "comparisons": comparisons,
    }


def _send(url: str, args: argparse.Namespace, row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "request_id": row["request_id"],
        "input_ids": row["input_ids"],
        "stream": False,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "ignore_eos": True,
            "n": args.beam_width,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=args.timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _warmup(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    if args.warmup_requests <= 0:
        return
    warmup_rows = rows[: min(args.warmup_requests, len(rows))]
    for row in warmup_rows:
        _send(args.gr_url, args, row)
        _send(args.sglang_url, args, row)


def _extract_top1_ids(output: Mapping[str, Any]) -> list[int]:
    output_ids = output.get("output_ids")
    if isinstance(output_ids, list):
        return [int(token) for token in output_ids]
    meta = output.get("meta_info")
    if isinstance(meta, Mapping):
        token_ids = meta.get("token_ids")
        if isinstance(token_ids, list):
            return [int(token) for token in token_ids]
    return []


def _extract_text(output: Mapping[str, Any]) -> str:
    text = output.get("text")
    return text if isinstance(text, str) else ""


def _comparison_summary(comparisons: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(comparisons)
    exact = sum(1 for row in comparisons if row.get("top1_exact"))
    return {
        "requests": total,
        "top1_exact": exact,
        "top1_exact_rate": exact / total if total else 0.0,
    }


def _summary(results: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "workload": results["workload"],
        "summary": results["summary"],
        "output_json": results.get("output_json"),
    }


def _markdown(results: Mapping[str, Any]) -> str:
    lines = [
        "# Online /generate Correctness",
        "",
        "## Workload",
        "",
    ]
    for key, value in results["workload"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- top1 exact rate: `{results['summary']['top1_exact_rate']}`",
            f"- exact requests: `{results['summary']['top1_exact']} / {results['summary']['requests']}`",
            "",
            "## Requests",
            "",
            "| request | top1 exact | GR top1 ids | SGLang top1 ids |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in results["comparisons"]:
        lines.append(
            "| {request_id} | {top1_exact} | `{gr_top1_ids}` | `{sglang_top1_ids}` |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--gr-url", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000/generate")
    parser.add_argument("--context-len", type=int, default=5000)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--request-prefix", default="onlinecmp")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--no-tokenizer", action="store_true", default=True)
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--extra-phrase", nargs="*", default=())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    return parser


if __name__ == "__main__":
    main()
