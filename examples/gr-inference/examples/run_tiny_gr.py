"""Run a tiny end-to-end GR inference skeleton."""

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
    GRGenerationState,
    PrefillAttention,
    Qwen3GRConfig,
    Qwen3GRModel,
    ScheduledBeamPolicy,
    TorchSDPAPrefillBackend,
    TokenTrie,
    TrieItemMaskProvider,
)


def build_tiny_config(
    *,
    layers: int,
    context_len: int,
    decode_steps: int,
    beam_width: int,
    vocab_size: int,
) -> Qwen3GRConfig:
    return Qwen3GRConfig(
        model_name="tiny-qwen3-gr-demo",
        num_layers=layers,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=context_len,
        max_seq_len=context_len + decode_steps,
        max_decode_steps=decode_steps,
        max_beam_width=beam_width,
        intermediate_size=64,
        vocab_size=vocab_size,
    )


def run_demo(args) -> dict:
    import torch

    torch.manual_seed(args.seed)
    config = build_tiny_config(
        layers=args.layers,
        context_len=args.context_len,
        decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        vocab_size=args.vocab_size,
    )
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    input_ids = torch.randint(0, config.vocab_size, (1, args.context_len))

    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="tiny-demo",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
        fixed_beam_width=args.beam_width,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=args.beam_width,
    )
    item_mask_provider = None
    if args.constraint_demo:
        sequences = []
        for root in range(1, args.beam_width + 1):
            sequence = [root]
            for depth in range(args.decode_steps):
                sequence.append((10 * root + depth) % args.vocab_size)
            sequences.append(sequence)
        item_mask_provider = TrieItemMaskProvider(
            TokenTrie.from_sequences(sequences),
            vocab_size=config.vocab_size,
        )
    beam_width_policy = None
    if args.beam_schedule:
        schedule = {}
        for item in args.beam_schedule.split(","):
            step, width = item.split(":")
            schedule[int(step)] = int(width)
        beam_width_policy = ScheduledBeamPolicy(schedule)
    result = model.generate_fixed_beam(
        generation,
        decode_engine,
        max_steps=args.decode_steps,
        item_mask_provider=item_mask_provider,
        beam_width_policy=beam_width_policy,
    )
    return {
        "input_shape": tuple(input_ids.shape),
        "context_kv_shape": generation.prefill.context_kv.key_shape,
        "beam_kv_shape": generation.beam_kv.key_shape,
        "beam_path_steps": generation.beam_path.steps_done,
        "steps": len(result.steps),
        "final_token_ids": result.final_token_ids,
        "final_scores": result.steps[-1].scores if result.steps else (),
        "constraint_demo": args.constraint_demo,
        "beam_schedule": args.beam_schedule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constraint-demo", action="store_true")
    parser.add_argument(
        "--beam-schedule",
        default=None,
        help="Comma separated depth:width schedule, e.g. 0:3,1:2,2:1",
    )
    args = parser.parse_args()

    summary = run_demo(args)
    print("Tiny GR demo")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
