"""Experimental TRT-LLM-aligned custom ops for GR-Inference.

This package follows the TokenSpeed-style dependency shape: importing it
registers PyTorch custom ops without taking over the inference runtime.

The first implementation is a reference op for the Qwen3 fused Q/K RMSNorm +
RoPE boundary. It is intentionally gated by backend selection before use; the
native CUDA implementation can replace this registration surface later.
"""

from gr_inference_trtllm_kernels.qwen3 import (
    call_counts,
    register_ops,
    reset_call_counts,
)

register_ops()

__all__ = ["call_counts", "register_ops", "reset_call_counts"]
