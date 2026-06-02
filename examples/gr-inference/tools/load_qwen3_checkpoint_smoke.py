"""Smoke-load a Qwen3 checkpoint into Qwen3GRModel."""

from __future__ import annotations

import argparse

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference import PrefillAttention, Qwen3GRModel, TorchSDPAPrefillBackend  # noqa: E402
from gr_inference.gr_models import HFCheckpointLoader, resolve_model_dir  # noqa: E402
from gr_inference.gr_models.qwen3 import (  # noqa: E402
    DEFAULT_QWEN3_MODEL_ID,
    Qwen3GRConfig,
    materialize_qwen3_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help=(
            "Model reference. Accepts either a local checkpoint directory or a "
            "HuggingFace repo id such as Qwen/Qwen3-1.7B."
        ),
    )
    parser.add_argument(
        "--model-dir",
        help="Explicit local checkpoint directory. Takes precedence over --model.",
    )
    parser.add_argument(
        "--revision",
        help="Optional HuggingFace branch, tag, or commit for --model repo ids.",
    )
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--run-prefill", action="store_true")
    args = parser.parse_args()

    import torch

    args.model_dir = resolve_model_dir(
        model_dir=args.model_dir,
        model=args.model,
        default_model=DEFAULT_QWEN3_MODEL_ID,
        revision=args.revision,
    )
    manifest = HFCheckpointLoader(args.model_dir).manifest()
    config = Qwen3GRConfig.from_hf_config(
        manifest.config,
        max_context_len=max(args.context_len, 1),
        max_seq_len=max(args.context_len + 1, 2),
    )
    print("Materializing checkpoint logical tensors...")
    weights = materialize_qwen3_checkpoint(args.model_dir)
    print(f"Loaded logical tensors: {len(weights)}")

    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    )
    model.load_logical_weights(weights)
    print("Loaded weights into Qwen3GRModel.")

    if args.run_prefill:
        input_ids = torch.randint(0, config.vocab_size, (1, args.context_len))
        logits, context_kv = model.forward_prefill(input_ids)
        print(f"prefill logits: {tuple(logits.shape)}")
        print(f"context_kv: {context_kv.key_shape}")


if __name__ == "__main__":
    main()
