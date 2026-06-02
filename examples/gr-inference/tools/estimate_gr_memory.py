"""Estimate GR serving KV/workspace memory for a target shape."""

from __future__ import annotations

import argparse
import json

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference.gr_models import HFCheckpointLoader  # noqa: E402
from gr_inference.gr_models.qwen3 import Qwen3GRConfig  # noqa: E402
from gr_inference.gr_serving import estimate_gr_kv_memory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        help="Optional HF checkpoint directory. When set, Qwen3 KV shape fields default from config.json.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--max-decode-steps", type=int, required=True)
    parser.add_argument("--max-beam-width", type=int, required=True)
    parser.add_argument("--active-beam-width", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    model_config = _load_model_config(args.model_dir) if args.model_dir else None
    num_layers = _required_or_model(
        args.num_layers,
        model_config,
        "num_layers",
        "--num-layers",
    )
    num_kv_heads = _required_or_model(
        args.num_kv_heads,
        model_config,
        "num_kv_heads",
        "--num-kv-heads",
    )
    head_dim = _required_or_model(
        args.head_dim,
        model_config,
        "head_dim",
        "--head-dim",
    )
    vocab_size = args.vocab_size
    if vocab_size is None and model_config is not None:
        vocab_size = model_config.vocab_size

    estimate = estimate_gr_kv_memory(
        batch_size=args.batch_size,
        num_layers=num_layers,
        context_len=args.context_len,
        max_decode_steps=args.max_decode_steps,
        max_beam_width=args.max_beam_width,
        active_beam_width=args.active_beam_width,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bytes_per_element=args.bytes_per_element,
        vocab_size=vocab_size,
    )
    print(
        json.dumps(
            estimate.metadata(), indent=2 if args.pretty else None, sort_keys=True
        )
    )
    return 0


def _load_model_config(model_dir: str) -> Qwen3GRConfig:
    manifest = HFCheckpointLoader(model_dir).manifest()
    return Qwen3GRConfig.from_hf_config(manifest.config)


def _required_or_model(
    value: int | None,
    model_config: Qwen3GRConfig | None,
    attr: str,
    flag: str,
) -> int:
    if value is not None:
        return value
    if model_config is not None:
        return int(getattr(model_config, attr))
    raise SystemExit(f"{flag} is required unless --model-dir is provided")


if __name__ == "__main__":
    raise SystemExit(main())
