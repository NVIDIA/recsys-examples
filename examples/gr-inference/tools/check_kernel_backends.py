"""Inspect available kernel backend libraries."""

from __future__ import annotations

import argparse
import json

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference.gr_kernels import (  # noqa: E402
    CAP_FUSED_ADD_RMSNORM,
    CAP_FUSED_MLP,
    CAP_GR_DECODE_ATTENTION,
    CAP_PACKED_GEMM,
    CAP_PREFILL_ATTENTION,
    CAP_QK_NORM_ROPE,
    CAP_RMSNORM,
    CAP_ROPE,
    CAP_SAMPLING_TOPK,
    build_default_kernel_registry,
    default_kernel_selection_policy,
)


INSTALL_HINTS = {
    "flashinfer": 'cd /cb/gr-inference && python -m pip install -U ".[kernels]"',
    "flash_attn": "Install flash-attn in the current Python environment.",
    "gr_decode_atten": (
        "cd /cb/gr_inference/gr-decode_atten && "
        "python -m pip install -r requirements.txt"
    ),
    "trtllm_aligned": (
        'python -m pip install -e ".[trtllm-kernels]" to install standalone '
        "TRT-LLM-style kernel ops, or ensure the in-repo gr_trtllm ops are "
        "importable. Then verify torch.ops.trtllm.{fused_qk_norm_rope,gated_mlp} or "
        "torch.ops.gr_trtllm.{fused_qk_norm_rope,gated_mlp} is available."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = build_default_kernel_registry()
    summary = registry.summary()
    if args.json:
        print(json.dumps(_json_summary(registry), indent=2))
        return

    print("Kernel backend availability")
    print("=" * 72)
    for name, info in summary.items():
        status = "available" if info["available"] else "missing"
        capabilities = ", ".join(info["capabilities"])
        print(f"{name}: {status} [{capabilities}]")
        metadata = info.get("metadata") or {}
        missing_requirements = [
            key
            for key, value in metadata.items()
            if isinstance(value, bool) and not value
        ]
        if missing_requirements:
            print(f"  missing: {', '.join(missing_requirements)}")
        details = [
            f"{key}={value}"
            for key, value in metadata.items()
            if not isinstance(value, bool) and value is not None
        ]
        if details:
            print(f"  details: {', '.join(details)}")
        if not info["available"] and name in INSTALL_HINTS:
            print(f"  hint: {INSTALL_HINTS[name]}")

    print()
    print("Kernel backend selection")
    print("=" * 72)
    for capability, backend_name in _selected_backends(registry).items():
        print(f"{capability}: {backend_name}")


def _json_summary(registry) -> dict[str, object]:
    return {
        "availability": registry.summary(),
        "selection": _selected_backends(registry),
    }


def _selected_backends(registry) -> dict[str, str | None]:
    policy = default_kernel_selection_policy()
    selected: dict[str, str | None] = {}
    for capability in (
        CAP_RMSNORM,
        CAP_FUSED_ADD_RMSNORM,
        CAP_ROPE,
        CAP_QK_NORM_ROPE,
        CAP_PREFILL_ATTENTION,
        CAP_GR_DECODE_ATTENTION,
        CAP_PACKED_GEMM,
        CAP_FUSED_MLP,
        CAP_SAMPLING_TOPK,
    ):
        backend = policy.select(capability, registry)
        selected[capability] = backend.name if backend is not None else None
    return selected


if __name__ == "__main__":
    main()
