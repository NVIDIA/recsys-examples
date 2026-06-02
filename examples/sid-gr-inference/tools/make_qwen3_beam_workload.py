# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic Qwen3 workloads for GR/SGLang beam comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tool_utils import load_optional_tokenizer

BASE_PARAGRAPH = (
    "User profile: long history recommendation context with repeated item "
    "signals, category preferences, timestamps, and behavioral evidence. "
    "The model should rank candidate continuations for a generative "
    "recommendation beam-search workload. "
)


def build_workload(args) -> list[dict]:
    tokenizer = _load_tokenizer(args)
    rows = []
    for idx in range(args.requests):
        if tokenizer is None:
            input_ids = _deterministic_token_ids_for_request(
                idx,
                context_len=args.context_len,
                vocab_size=args.vocab_size,
                shared_prefix_len=args.shared_prefix_len,
            )
            prompt_text = None
            retokenized = ()
        else:
            seed_text = (
                f"Request {idx}. " + BASE_PARAGRAPH + " ".join(args.extra_phrase)
            ).strip()
            text = _repeat_to_token_length(
                tokenizer,
                seed_text,
                context_len=args.context_len,
            )
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(input_ids) < args.context_len:
                raise RuntimeError(
                    f"failed to build request {idx}: tokenized length {len(input_ids)}"
                )
            input_ids = input_ids[: args.context_len]
            prompt_text = tokenizer.decode(input_ids, skip_special_tokens=False)
            retokenized = tokenizer.encode(prompt_text, add_special_tokens=False)
        rows.append(
            {
                "request_id": f"{args.request_prefix}-{idx}",
                "text": prompt_text,
                "input_ids": input_ids,
                "context_len": len(input_ids),
                "retokenized_len": len(retokenized),
                "retokenized_prefix_matches": (
                    retokenized[: len(input_ids)] == input_ids
                    if tokenizer is not None
                    else None
                ),
                "metadata": {
                    "source": "make_qwen3_beam_workload.py",
                    "model_dir": args.model_dir,
                    "mode": "tokenizer"
                    if tokenizer is not None
                    else "deterministic_ids",
                    "shared_prefix_len": args.shared_prefix_len
                    if tokenizer is None
                    else None,
                },
            }
        )
    return rows


def _load_tokenizer(args):
    return load_optional_tokenizer(args)


def _deterministic_token_ids(
    idx: int, *, context_len: int, vocab_size: int
) -> list[int]:
    if vocab_size <= 1024:
        raise ValueError("--vocab-size must be > 1024 for deterministic workload")
    start = 1024 + idx * 17
    span = vocab_size - 1024
    return [1024 + ((start + pos * 13) % span) for pos in range(context_len)]


def _deterministic_token_ids_for_request(
    idx: int,
    *,
    context_len: int,
    vocab_size: int,
    shared_prefix_len: int = 0,
) -> list[int]:
    if shared_prefix_len <= 0:
        return _deterministic_token_ids(
            idx,
            context_len=context_len,
            vocab_size=vocab_size,
        )
    if shared_prefix_len >= context_len:
        raise ValueError("--shared-prefix-len must be smaller than --context-len")
    shared = _deterministic_token_ids(
        0,
        context_len=shared_prefix_len,
        vocab_size=vocab_size,
    )
    suffix = _deterministic_token_ids(
        idx + 1,
        context_len=context_len - shared_prefix_len,
        vocab_size=vocab_size,
    )
    return shared + suffix


def _repeat_to_token_length(tokenizer, text: str, *, context_len: int) -> str:
    repeated = text
    while len(tokenizer.encode(repeated, add_special_tokens=False)) < context_len:
        repeated = repeated + "\n" + text
    return repeated


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context-len", type=int, default=5000)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--request-prefix", default="beamcmp")
    parser.add_argument("--extra-phrase", nargs="*", default=())
    parser.add_argument(
        "--shared-prefix-len",
        type=int,
        default=0,
        help=(
            "For --no-tokenizer deterministic workloads, make all requests share "
            "this many leading token ids and vary the suffix."
        ),
    )
    parser.add_argument("--no-tokenizer", action="store_true")
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--output-jsonl", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.shared_prefix_len < 0:
        raise ValueError("--shared-prefix-len must be non-negative")
    if args.shared_prefix_len and not args.no_tokenizer:
        raise ValueError("--shared-prefix-len currently requires --no-tokenizer")
    rows = build_workload(args)
    write_jsonl(rows, Path(args.output_jsonl))
    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "requests": len(rows),
                "context_len": args.context_len,
                "retokenized_prefix_matches": [
                    row["retokenized_prefix_matches"] for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
