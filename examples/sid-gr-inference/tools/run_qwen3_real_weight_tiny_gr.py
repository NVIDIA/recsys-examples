# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run tiny GR inference with a real Qwen3 checkpoint."""

from __future__ import annotations

import argparse

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference import (  # noqa: E402
    GRDecodeAttention,
    GRDecodeEngine,
    GRGenerationState,
    PrefillAttention,
    Qwen3GRModel,
    TorchSDPAPrefillBackend,
)
from gr_inference.gr_kernels.attention import (  # noqa: E402
    ExistingGRDecodeAttentionBackend,
)
from gr_inference.gr_models import HFCheckpointLoader, resolve_model_dir  # noqa: E402
from gr_inference.gr_models.qwen3 import (  # noqa: E402
    DEFAULT_QWEN3_MODEL_ID,
    Qwen3GRConfig,
    materialize_qwen3_checkpoint,
)


def choose_dtype(torch, device: str):
    return torch.bfloat16 if device == "cuda" else torch.float32


def build_identity_topk_indices(
    torch, *, batch: int, head_q: int, decode_nums: int, beam_width: int, device
):
    indices = torch.arange(beam_width, dtype=torch.int32, device=device)
    indices = indices.view(1, 1, 1, 1, beam_width)
    indices = indices.expand(batch, 1, head_q, decode_nums, beam_width).contiguous()
    for decode_idx in range(decode_nums):
        indices[:, :, :, decode_idx, :] = decode_idx * beam_width + torch.arange(
            beam_width,
            dtype=torch.int32,
            device=device,
        )
    return indices


def load_model(args, torch):
    args.model_dir = resolve_model_dir(
        model_dir=getattr(args, "model_dir", None),
        model=getattr(args, "model", None),
        default_model=DEFAULT_QWEN3_MODEL_ID,
        revision=getattr(args, "revision", None),
    )
    manifest = HFCheckpointLoader(args.model_dir).manifest()
    config = Qwen3GRConfig.from_hf_config(
        manifest.config,
        max_context_len=max(args.context_len, 1),
        max_seq_len=max(args.context_len + args.decode_steps, 2),
        max_decode_steps=max(args.decode_steps, 1),
        max_beam_width=args.beam_width,
    )
    device = (
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = choose_dtype(torch, device)

    print("Materializing checkpoint logical tensors...")
    weights = materialize_qwen3_checkpoint(args.model_dir)
    print(f"Loaded logical tensors: {len(weights)}")

    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
        dtype=dtype,
    ).to(device)
    model.load_logical_weights(weights)
    model.eval()
    print(f"Loaded Qwen3GRModel on {device} with dtype={dtype}.")
    return model, config, device


def run_fake_decode(args, torch, model, config, device):
    input_ids = torch.randint(
        0, config.vocab_size, (1, args.context_len), device=device
    )
    with torch.no_grad():
        prefill = model.forward_prefill(input_ids, return_result=True)
        generation = GRGenerationState.from_prefill(
            request_id="real-weight-tiny-fake",
            prefill=prefill,
            max_decode_steps=config.max_decode_steps,
            max_beam_width=config.max_beam_width,
            fixed_beam_width=args.beam_width,
        )
        decode_engine = GRDecodeEngine(
            attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
            fixed_beam_width=args.beam_width,
        )
        result = model.generate_fixed_beam(
            generation, decode_engine, max_steps=args.decode_steps
        )
    return {
        "backend": "fake",
        "prefill_logits": tuple(prefill.logits.shape),
        "context_kv": generation.prefill.context_kv.key_shape,
        "beam_kv": generation.beam_kv.key_shape,
        "steps": len(result.steps),
        "final_token_ids": result.final_token_ids,
    }


def run_real_decode(args, torch, model, config, device):
    if device != "cuda":
        raise RuntimeError("--decode-backend real requires CUDA")
    if args.decode_steps != 1:
        raise RuntimeError("--decode-backend real currently supports --decode-steps 1")

    input_ids = torch.randint(
        0, config.vocab_size, (1, args.context_len), device=device
    )
    with torch.no_grad():
        prefill = model.forward_prefill(input_ids, return_result=True)
        generation = GRGenerationState.from_prefill(
            request_id="real-weight-tiny-real",
            prefill=prefill,
            max_decode_steps=config.max_decode_steps,
            max_beam_width=config.max_beam_width,
            fixed_beam_width=args.beam_width,
        )
        selection = generation.initialize_beams()
        beam_token_ids = torch.tensor(
            [selection.token_ids], dtype=torch.long, device=device
        )
        topk_indices = build_identity_topk_indices(
            torch,
            batch=1,
            head_q=config.num_attention_heads,
            decode_nums=1,
            beam_width=args.beam_width,
            device=device,
        )
        decode_engine = GRDecodeEngine(
            attention=GRDecodeAttention(backend=ExistingGRDecodeAttentionBackend()),
            fixed_beam_width=args.beam_width,
        )
        logits = model.forward_decode_step(
            beam_token_ids,
            generation,
            decode_engine,
            step=0,
            topk_indices=topk_indices,
            decode_nums=1,
            return_lse=True,
            backend_name=args.kernel_backend,
        )
        generation.update_beams_from_logits(logits)
    return {
        "backend": "real",
        "prefill_logits": tuple(prefill.logits.shape),
        "decode_logits": tuple(logits.shape),
        "context_kv": generation.prefill.context_kv.key_shape,
        "beam_kv": generation.beam_kv.key_shape,
        "beam_path_steps": generation.beam_path.steps_done,
        "final_token_ids": generation.beam_path.entries[-1].token_ids,
    }


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
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--kernel-backend", choices=["dsl", "3kernel"], default="dsl")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    import torch

    model, config, device = load_model(args, torch)
    if args.decode_backend == "real":
        summary = run_real_decode(args, torch, model, config, device)
    else:
        summary = run_fake_decode(args, torch, model, config, device)

    print("Real-weight tiny GR")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
