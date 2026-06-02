# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny synchronous serving demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gr_inference import (  # noqa: E402
    GRDecodeAttention,
    GRDecodeEngine,
    GRServingConfig,
    GRServingEngine,
    GRServingRequest,
    PrefillAttention,
    Qwen3GRConfig,
    Qwen3GRModel,
    SchedulerPolicy,
    SyncGRScheduler,
    TorchSDPAPrefillBackend,
)


def run_demo(args) -> dict:
    import torch

    config = Qwen3GRConfig(
        model_name="tiny-serving-qwen3-gr",
        num_layers=args.layers,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=args.context_len,
        max_seq_len=args.context_len + args.decode_steps,
        max_decode_steps=args.decode_steps,
        max_beam_width=args.beam_width,
        intermediate_size=64,
        vocab_size=args.vocab_size,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=args.beam_width,
    )
    engine = GRServingEngine(
        model=model,
        decode_engine=decode_engine,
        config=GRServingConfig(
            max_decode_steps=args.decode_steps,
            max_beam_width=args.beam_width,
            enable_batched_decode=getattr(args, "batched_decode", False),
            return_beam_details=getattr(args, "return_beam_details", False),
            beam_score_mode=getattr(args, "beam_score_mode", "logprob"),
        ),
    )
    if args.warmup:
        engine.warmup(
            GRServingRequest(
                request_id="warmup",
                input_ids=torch.randint(0, config.vocab_size, (1, args.context_len)),
                max_decode_steps=args.decode_steps,
                beam_width=args.beam_width,
            )
        )
    scheduler = SyncGRScheduler(
        engine,
        policy=SchedulerPolicy(max_batch_size=args.max_batch_size),
    )
    for idx in range(args.requests):
        scheduler.submit(
            GRServingRequest(
                request_id=f"req-{idx}",
                input_ids=torch.randint(0, config.vocab_size, (1, args.context_len)),
                max_decode_steps=args.decode_steps,
                beam_width=args.beam_width,
            )
        )
    responses = scheduler.run_until_empty()
    return {
        "responses": len(responses),
        "first_request_id": responses[0].request_id if responses else None,
        "first_token_ids": responses[0].token_ids if responses else (),
        "first_metadata": responses[0].metadata if responses else {},
        "engine_status": engine.status(),
        "scheduler_status": scheduler.status(),
        "scheduler_metrics": scheduler.metrics(),
        "batch_history": scheduler.batch_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--batched-decode", action="store_true")
    parser.add_argument("--return-beam-details", action="store_true")
    parser.add_argument(
        "--beam-score-mode", choices=["raw_logits", "logprob"], default="logprob"
    )
    args = parser.parse_args()

    summary = run_demo(args)
    print("Tiny serving demo")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
