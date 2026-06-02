"""Qwen3 TRT-LLM-aligned custom op registrations."""

from __future__ import annotations

import os

try:  # pragma: no cover - optional at import time
    import torch
except ImportError:  # pragma: no cover
    torch = None

_REGISTERED = False
_LIBRARY = None
_CUDA_EXTENSION = None
_CUDA_EXTENSION_FAILED = False
_CALLS: dict[str, int] = {}


def register_ops() -> None:
    """Register experimental Qwen3 custom ops under ``torch.ops.gr_trtllm``."""

    global _LIBRARY, _REGISTERED
    if _REGISTERED or torch is None:
        return

    _LIBRARY = torch.library.Library("gr_trtllm", "DEF")
    _LIBRARY.define(
        "fused_qk_norm_rope("
        "Tensor(a!) qkv, "
        "int num_heads, "
        "int num_kv_heads, "
        "int num_kv_heads_for_cache, "
        "int head_dim, "
        "int rotary_dim, "
        "float eps, "
        "Tensor q_weight, "
        "Tensor k_weight, "
        "float rope_theta, "
        "bool is_neox, "
        "Tensor position_ids, "
        "float yarn_factor, "
        "int yarn_low, "
        "int yarn_high, "
        "float attention_factor, "
        "bool is_qk_norm"
        ") -> ()"
    )
    _LIBRARY.define(
        "gated_mlp("
        "Tensor hidden_states, "
        "Tensor gate_up_weight, "
        "Tensor down_weight"
        ") -> Tensor"
    )
    _LIBRARY.define(
        "packed_gemm("
        "Tensor input, "
        "Tensor weight, "
        "Tensor? bias=None"
        ") -> Tensor"
    )
    _LIBRARY.impl(
        "fused_qk_norm_rope",
        _fused_qk_norm_rope_reference,
        "CompositeExplicitAutograd",
    )
    _LIBRARY.impl(
        "gated_mlp",
        _gated_mlp_reference,
        "CompositeExplicitAutograd",
    )
    _LIBRARY.impl(
        "packed_gemm",
        _packed_gemm_reference,
        "CompositeExplicitAutograd",
    )
    _REGISTERED = True


def write_beam_kv_step(
    beam_key,
    beam_value,
    k,
    v,
    *,
    layer_idx: int,
    step: int,
    active_beam_width: int,
) -> bool:
    """Write one decode layer step into BeamKV using one CUDA launch.

    Returns ``False`` when the CUDA extension is unavailable or the tensors do
    not match the supported hot-path shape, allowing callers to keep the
    regular PyTorch copy fallback.
    """

    if (
        torch is None
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "1") != "1"
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_BEAM_KV_WRITE_JIT", "0") != "1"
    ):
        _record_call("beam_kv_write_jit_disabled")
        return False
    if not (
        getattr(beam_key, "is_cuda", False)
        and getattr(beam_value, "is_cuda", False)
        and getattr(k, "is_cuda", False)
        and getattr(v, "is_cuda", False)
    ):
        _record_call("beam_kv_write_jit_non_cuda")
        return False
    if (
        beam_key.dtype != beam_value.dtype
        or k.dtype != beam_key.dtype
        or v.dtype != beam_key.dtype
    ):
        _record_call("beam_kv_write_jit_dtype_mismatch")
        return False
    extension = _cuda_extension()
    if extension is None:
        _record_call("beam_kv_write_jit_unavailable")
        return False
    try:
        extension.write_beam_kv_step_cuda(
            beam_key,
            beam_value,
            k,
            v,
            int(layer_idx),
            int(step),
            int(active_beam_width),
        )
        _record_call("beam_kv_write_cuda")
        return True
    except Exception:
        _record_call("beam_kv_write_cuda_failed")
        return False


def write_packed_qkv_prefill_kv(
    k,
    qkv,
    context_key,
    context_value,
    *,
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    """Write prefill K/V into ContextKV while reading V from packed QKV."""

    if (
        torch is None
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "1") != "1"
    ):
        _record_call("packed_qkv_kv_write_jit_disabled")
        return False
    if not (
        getattr(k, "is_cuda", False)
        and getattr(qkv, "is_cuda", False)
        and getattr(context_key, "is_cuda", False)
        and getattr(context_value, "is_cuda", False)
    ):
        _record_call("packed_qkv_kv_write_jit_non_cuda")
        return False
    if (
        k.dtype != qkv.dtype
        or k.dtype != context_key.dtype
        or k.dtype != context_value.dtype
    ):
        _record_call("packed_qkv_kv_write_jit_dtype_mismatch")
        return False
    extension = _cuda_extension()
    if extension is None:
        _record_call("packed_qkv_kv_write_jit_unavailable")
        return False
    try:
        extension.write_packed_qkv_prefill_kv_cuda(
            k,
            qkv,
            context_key,
            context_value,
            int(layer_idx),
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
        )
        _record_call("packed_qkv_kv_write_cuda")
        return True
    except Exception:
        _record_call("packed_qkv_kv_write_cuda_failed")
        return False


def exact_fused_add_rmsnorm(input_tensor, residual_tensor, weight, eps: float):
    """Exact add + RMSNorm for decode lookahead fusion.

    FlashInfer's fused add+rmsnorm is fast, but on the MLP residual path it can
    skip the baseline bf16 residual-add materialization point. This helper keeps
    that rounding order while still replacing add + RMSNorm with one launch.
    The CUDA path mutates ``residual_tensor`` into the residual sum and reuses
    ``input_tensor`` as the normalized output so it can be captured in CUDA
    graphs without an internal allocation.
    """

    if (
        torch is None
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "1") != "1"
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_EXACT_ADD_RMSNORM_JIT", "1") != "1"
    ):
        _record_call("exact_add_rmsnorm_jit_disabled")
        return None
    if not (
        getattr(input_tensor, "is_cuda", False)
        and getattr(residual_tensor, "is_cuda", False)
        and getattr(weight, "is_cuda", False)
    ):
        _record_call("exact_add_rmsnorm_jit_non_cuda")
        return None
    if input_tensor.shape != residual_tensor.shape:
        _record_call("exact_add_rmsnorm_jit_shape_mismatch")
        return None
    if (
        input_tensor.dtype != residual_tensor.dtype
        or weight.dtype != residual_tensor.dtype
    ):
        _record_call("exact_add_rmsnorm_jit_dtype_mismatch")
        return None
    extension = _cuda_extension()
    if extension is None:
        _record_call("exact_add_rmsnorm_jit_unavailable")
        return None
    try:
        result = extension.exact_fused_add_rmsnorm_cuda(
            input_tensor,
            residual_tensor,
            weight,
            float(eps),
        )
        _record_call("exact_add_rmsnorm_cuda")
        return result
    except Exception:
        _record_call("exact_add_rmsnorm_cuda_failed")
        return None


def _fused_qk_norm_rope_reference(
    qkv,
    num_heads: int,
    num_kv_heads: int,
    num_kv_heads_for_cache: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    q_weight,
    k_weight,
    rope_theta: float,
    is_neox: bool,
    position_ids,
    yarn_factor: float,
    yarn_low: int,
    yarn_high: int,
    attention_factor: float,
    is_qk_norm: bool,
) -> None:
    """Reference implementation matching TRT-LLM's fused op boundary.

    The op mutates the packed QKV tensor in place and leaves V unchanged.
    ``num_kv_heads_for_cache`` and YaRN parameters are accepted for signature
    compatibility with TRT-LLM; this first Qwen3 path uses standard RoPE.
    """

    del num_kv_heads_for_cache, yarn_factor, yarn_low, yarn_high
    if qkv.dim() != 2:
        raise RuntimeError(
            "gr_trtllm.fused_qk_norm_rope expects qkv shaped [N, packed]"
        )
    if rotary_dim != head_dim:
        raise RuntimeError("partial rotary_dim is not supported by the reference op")
    if head_dim % 2 != 0:
        raise RuntimeError("RoPE requires an even head_dim")

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    if qkv.shape[-1] < q_size + kv_size:
        raise RuntimeError("packed qkv tensor is smaller than q/k sizes")

    if _try_cuda_fused_qk_norm_rope(
        qkv,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        rope_theta,
        is_neox,
        position_ids,
        attention_factor,
        is_qk_norm,
    ):
        return

    _record_call("fused_qk_norm_rope_reference")
    q = qkv[:, :q_size].reshape(-1, num_heads, head_dim)
    k = qkv[:, q_size : q_size + kv_size].reshape(-1, num_kv_heads, head_dim)

    if is_qk_norm:
        q.copy_(_rmsnorm(q, q_weight, eps))
        k.copy_(_rmsnorm(k, k_weight, eps))

    q_rope = _apply_rope(q, position_ids, rope_theta, attention_factor, is_neox=is_neox)
    k_rope = _apply_rope(k, position_ids, rope_theta, attention_factor, is_neox=is_neox)
    q.copy_(q_rope)
    k.copy_(k_rope)


def _rmsnorm(x, weight, eps: float):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + eps)
    return (normalized * weight.reshape(1, 1, -1).float()).to(x.dtype)


def _gated_mlp_reference(hidden_states, gate_up_weight, down_weight):
    """TRT-LLM-style GatedMLP op boundary using torch GEMM fallbacks."""

    if hidden_states.shape[-1] != gate_up_weight.shape[-1]:
        raise RuntimeError("hidden size does not match gate_up_weight")
    if gate_up_weight.shape[0] % 2 != 0:
        raise RuntimeError("gate_up_weight output dimension must be even")
    intermediate_size = gate_up_weight.shape[0] // 2
    if down_weight.shape[-1] != intermediate_size:
        raise RuntimeError(
            "down_weight input dimension does not match intermediate size"
        )

    cuda_output = _try_cuda_gated_mlp(
        hidden_states,
        gate_up_weight,
        down_weight,
        intermediate_size,
    )
    if cuda_output is not None:
        return cuda_output

    _record_call("gated_mlp_reference")
    gate_up = torch.matmul(hidden_states, gate_up_weight.transpose(0, 1))
    gate, up = gate_up.split([intermediate_size, intermediate_size], dim=-1)
    intermediate = torch.nn.functional.silu(gate) * up
    return torch.matmul(intermediate, down_weight.transpose(0, 1))


def _packed_gemm_reference(input, weight, bias=None):
    """TRT-LLM-style unquantized Linear boundary using torch GEMM fallback."""

    if input.shape[-1] != weight.shape[-1]:
        raise RuntimeError("input hidden size does not match packed_gemm weight")
    cuda_output = _try_cuda_packed_gemm(input, weight, bias)
    if cuda_output is not None:
        return cuda_output
    _record_call("packed_gemm_reference")
    original_shape = input.shape[:-1]
    flat_input = input.reshape(-1, input.shape[-1])
    flat_output = _packed_gemm_flatten_mm(flat_input, weight, bias)
    return flat_output.reshape(*original_shape, weight.shape[0])


def _packed_gemm_flatten_mm(flat_input, weight, bias=None):
    """Use a 2D GEMM path for serving projections.

    TRT-LLM's Qwen Linear path is fundamentally a 2D GEMM over flattened tokens.
    Keeping this fallback in the same shape makes the ABI probe closer to the
    native cublasLt/CUTLASS/Triton kernel that will replace it.
    """

    _record_call("packed_gemm_flatten_mm")
    weight_t = weight.transpose(0, 1)
    if bias is not None:
        return torch.addmm(bias, flat_input, weight_t)
    return torch.mm(flat_input, weight_t)


def _apply_rope(
    x, position_ids, rope_theta: float, attention_factor: float, *, is_neox: bool
):
    compute_dtype = torch.float32
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, device=x.device, dtype=compute_dtype)
            / head_dim
        )
    )
    positions = position_ids.to(device=x.device, dtype=compute_dtype).reshape(-1)
    if positions.numel() != x.shape[0]:
        raise RuntimeError("position_ids must flatten to the qkv token count")
    freqs = positions[:, None] * inv_freq[None, :]
    cos = freqs.cos()[:, None, :] * float(attention_factor)
    sin = freqs.sin()[:, None, :] * float(attention_factor)

    x_float = x.float()
    out = torch.empty_like(x_float)
    if is_neox:
        half_dim = head_dim // 2
        x_first = x_float[..., :half_dim]
        x_second = x_float[..., half_dim:]
        out[..., :half_dim] = x_first * cos - x_second * sin
        out[..., half_dim:] = x_second * cos + x_first * sin
    else:
        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
    return out.to(x.dtype)


def _try_cuda_fused_qk_norm_rope(
    qkv,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    q_weight,
    k_weight,
    rope_theta: float,
    is_neox: bool,
    position_ids,
    attention_factor: float,
    is_qk_norm: bool,
) -> bool:
    if os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "1") != "1":
        _record_call("fused_qk_norm_rope_jit_disabled")
        return False
    if not qkv.is_cuda:
        _record_call("fused_qk_norm_rope_jit_non_cuda")
        return False
    extension = _cuda_extension()
    if extension is None:
        _record_call("fused_qk_norm_rope_jit_unavailable")
        return False
    try:
        pos_ids = (
            position_ids.to(device=qkv.device, dtype=torch.int32)
            .reshape(-1)
            .contiguous()
        )
        extension.fused_qk_norm_rope_cuda(
            qkv,
            int(num_heads),
            int(num_kv_heads),
            int(head_dim),
            int(rotary_dim),
            float(eps),
            q_weight.contiguous(),
            k_weight.contiguous(),
            float(rope_theta),
            bool(is_neox),
            pos_ids,
            float(attention_factor),
            bool(is_qk_norm),
        )
        _record_call("fused_qk_norm_rope_cuda")
        return True
    except Exception:
        _record_call("fused_qk_norm_rope_cuda_failed")
        return False


def _try_cuda_gated_mlp(
    hidden_states, gate_up_weight, down_weight, intermediate_size: int
):
    if (
        os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "0") != "1"
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_GATED_MLP_JIT", "0") != "1"
    ):
        _record_call("gated_mlp_jit_disabled")
        return None
    if not hidden_states.is_cuda:
        _record_call("gated_mlp_jit_non_cuda")
        return None
    extension = _cuda_extension()
    if extension is None:
        _record_call("gated_mlp_jit_unavailable")
        return None
    try:
        gate_up = torch.matmul(hidden_states, gate_up_weight.transpose(0, 1))
        intermediate = extension.silu_and_mul_packed_cuda(gate_up)
        output = torch.matmul(intermediate, down_weight.transpose(0, 1))
        _record_call("gated_mlp_packed_silu_mul_cuda")
        return output
    except Exception:
        _record_call("gated_mlp_packed_silu_mul_cuda_failed")
        return None


def _try_cuda_packed_gemm(input, weight, bias=None):
    if (
        os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "0") != "1"
        or os.environ.get("GR_INFERENCE_GR_TRTLLM_PACKED_GEMM_JIT", "0") != "1"
    ):
        _record_call("packed_gemm_jit_disabled")
        return None
    if bias is not None:
        _record_call("packed_gemm_jit_bias_unsupported")
        return None
    if not input.is_cuda or not weight.is_cuda:
        _record_call("packed_gemm_jit_non_cuda")
        return None
    extension = _cuda_extension()
    if extension is None:
        _record_call("packed_gemm_jit_unavailable")
        return None
    try:
        original_shape = input.shape[:-1]
        flat_input = input.reshape(-1, input.shape[-1]).contiguous()
        output = extension.packed_gemm_cuda(flat_input, weight.contiguous())
        _record_call("packed_gemm_cuda")
        return output.reshape(*original_shape, weight.shape[0])
    except Exception:
        _record_call("packed_gemm_cuda_failed")
        return None


def _cuda_extension():
    global _CUDA_EXTENSION, _CUDA_EXTENSION_FAILED
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION
    if _CUDA_EXTENSION_FAILED:
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _CUDA_EXTENSION = load_inline(
            name="gr_trtllm_qwen3_kernels",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "fused_qk_norm_rope_cuda",
                "silu_and_mul_packed_cuda",
                "packed_gemm_cuda",
                "exact_fused_add_rmsnorm_cuda",
                "write_beam_kv_step_cuda",
                "write_packed_qkv_prefill_kv_cuda",
            ],
            extra_cuda_cflags=["-O3"],
            extra_ldflags=["-lcublas"],
            with_cuda=True,
            verbose=os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_VERBOSE") == "1",
        )
        return _CUDA_EXTENSION
    except Exception:
        _CUDA_EXTENSION_FAILED = True
        return None


def call_counts() -> dict[str, int]:
    """Return custom-op call counters for profiling/debugging."""

    return dict(_CALLS)


def reset_call_counts() -> None:
    _CALLS.clear()


def _record_call(name: str) -> None:
    _CALLS[name] = _CALLS.get(name, 0) + 1
    if os.environ.get("GR_INFERENCE_DEBUG_GR_TRTLLM") == "1":
        print(f"gr_trtllm_{name}_calls={_CALLS[name]}")


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

void fused_qk_norm_rope_cuda_launcher(
    torch::Tensor qkv,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double eps,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double rope_theta,
    bool is_neox,
    torch::Tensor position_ids,
    double attention_factor,
    bool is_qk_norm);

torch::Tensor silu_and_mul_packed_cuda_launcher(
    torch::Tensor gate_up);

torch::Tensor packed_gemm_cuda_launcher(
    torch::Tensor input,
    torch::Tensor weight);

std::vector<torch::Tensor> exact_fused_add_rmsnorm_cuda_launcher(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps);

void write_beam_kv_step_cuda_launcher(
    torch::Tensor beam_key,
    torch::Tensor beam_value,
    torch::Tensor k,
    torch::Tensor v,
    int64_t layer_idx,
    int64_t step,
    int64_t active_beam_width);

void write_packed_qkv_prefill_kv_cuda_launcher(
    torch::Tensor k,
    torch::Tensor qkv,
    torch::Tensor context_key,
    torch::Tensor context_value,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim);

void fused_qk_norm_rope_cuda(
    torch::Tensor qkv,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double eps,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double rope_theta,
    bool is_neox,
    torch::Tensor position_ids,
    double attention_factor,
    bool is_qk_norm) {
  fused_qk_norm_rope_cuda_launcher(
      qkv, num_heads, num_kv_heads, head_dim, rotary_dim, eps,
      q_weight, k_weight, rope_theta, is_neox, position_ids,
      attention_factor, is_qk_norm);
}

torch::Tensor silu_and_mul_packed_cuda(
    torch::Tensor gate_up) {
  return silu_and_mul_packed_cuda_launcher(gate_up);
}

torch::Tensor packed_gemm_cuda(
    torch::Tensor input,
    torch::Tensor weight) {
  return packed_gemm_cuda_launcher(input, weight);
}

std::vector<torch::Tensor> exact_fused_add_rmsnorm_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  return exact_fused_add_rmsnorm_cuda_launcher(input, residual, weight, eps);
}

void write_beam_kv_step_cuda(
    torch::Tensor beam_key,
    torch::Tensor beam_value,
    torch::Tensor k,
    torch::Tensor v,
    int64_t layer_idx,
    int64_t step,
    int64_t active_beam_width) {
  write_beam_kv_step_cuda_launcher(
      beam_key, beam_value, k, v, layer_idx, step, active_beam_width);
}

void write_packed_qkv_prefill_kv_cuda(
    torch::Tensor k,
    torch::Tensor qkv,
    torch::Tensor context_key,
    torch::Tensor context_value,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  write_packed_qkv_prefill_kv_cuda_launcher(
      k, qkv, context_key, context_value, layer_idx,
      num_heads, num_kv_heads, head_dim);
}
"""


_CUDA_SOURCE = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdlib>
#include <cublas_v2.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <torch/extension.h>
#include <vector>

inline void check_cublas_status(cublasStatus_t status, const char* expr) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed: ", expr);
}

#define GR_TRTLLM_CUBLAS_CHECK(expr) check_cublas_status((expr), #expr)

template <typename scalar_t>
__device__ inline float load_as_float(const scalar_t* ptr, int64_t offset) {
  return static_cast<float>(ptr[offset]);
}

template <typename scalar_t>
__device__ inline void store_from_float(scalar_t* ptr, int64_t offset, float value) {
  ptr[offset] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__device__ inline float round_to_scalar_float(float value) {
  return static_cast<float>(static_cast<scalar_t>(value));
}

template <typename scalar_t>
__global__ void fused_qk_norm_rope_kernel(
    scalar_t* __restrict__ qkv,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const int32_t* __restrict__ position_ids,
    int64_t tokens,
    int64_t packed_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    float rope_theta,
    float attention_factor,
    bool is_neox,
    bool is_qk_norm) {
  extern __shared__ float shared[];
  int token = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x;
  int total_heads = num_heads + num_kv_heads;
  if (token >= tokens || head >= total_heads) {
    return;
  }

  bool is_q = head < num_heads;
  int local_head = is_q ? head : head - num_heads;
  int64_t q_size = static_cast<int64_t>(num_heads) * head_dim;
  int64_t kv_size = static_cast<int64_t>(num_kv_heads) * head_dim;
  int64_t base = static_cast<int64_t>(token) * packed_size
      + (is_q ? static_cast<int64_t>(local_head) * head_dim
              : q_size + static_cast<int64_t>(local_head) * head_dim);
  const scalar_t* weight = is_q ? q_weight : k_weight;

  float sum = 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float value = load_as_float(qkv, base + d);
    sum += value * value;
  }
  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float inv_rms = rsqrtf(shared[0] / static_cast<float>(head_dim) + eps);

  int pos = position_ids[token];
  int half_dim = head_dim / 2;
  for (int pair = tid; pair < half_dim; pair += blockDim.x) {
    int first = is_neox ? pair : 2 * pair;
    int second = is_neox ? pair + half_dim : first + 1;
    float first_value = load_as_float(qkv, base + first);
    float second_value = load_as_float(qkv, base + second);
    if (is_qk_norm) {
      first_value = round_to_scalar_float<scalar_t>(
          first_value * inv_rms * load_as_float(weight, first));
      second_value = round_to_scalar_float<scalar_t>(
          second_value * inv_rms * load_as_float(weight, second));
    }
    float exponent = is_neox ? static_cast<float>(2 * pair) : static_cast<float>(first);
    float inv_freq = powf(rope_theta, -exponent / static_cast<float>(head_dim));
    float angle = static_cast<float>(pos) * inv_freq;
    float s, c;
    sincosf(angle, &s, &c);
    c *= attention_factor;
    s *= attention_factor;
    store_from_float(qkv, base + first, first_value * c - second_value * s);
    store_from_float(qkv, base + second, first_value * s + second_value * c);
  }
}

template <typename scalar_t>
__global__ void silu_and_mul_packed_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ output,
    int64_t elements,
    int64_t intermediate_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= elements) {
    return;
  }
  int64_t token = idx / intermediate_size;
  int64_t dim = idx - token * intermediate_size;
  int64_t packed_base = token * intermediate_size * 2;
  float gate_value = load_as_float(gate_up, packed_base + dim);
  float up_value = load_as_float(gate_up, packed_base + intermediate_size + dim);
  float sigmoid = 1.0f / (1.0f + expf(-gate_value));
  float silu_value = round_to_scalar_float<scalar_t>(gate_value * sigmoid);
  store_from_float(output, idx, silu_value * up_value);
}

template <typename scalar_t, int VecSize>
struct alignas(16) PackedVec {
  scalar_t values[VecSize];
};

template <typename scalar_t, int VecSize>
__global__ void silu_and_mul_packed_vec_kernel(
    const scalar_t* __restrict__ gate_up,
    scalar_t* __restrict__ output,
    int64_t tokens,
    int64_t intermediate_size) {
  int64_t token = blockIdx.x;
  int64_t thread_idx = threadIdx.x;
  int64_t stride = blockDim.x;
  if (token >= tokens) {
    return;
  }

  int64_t packed_base = token * intermediate_size * 2;
  int64_t output_base = token * intermediate_size;
  int64_t vec_cols = intermediate_size / VecSize;
  for (int64_t vec_idx = thread_idx; vec_idx < vec_cols; vec_idx += stride) {
    int64_t dim = vec_idx * VecSize;
    PackedVec<scalar_t, VecSize> gate_vec =
        *reinterpret_cast<const PackedVec<scalar_t, VecSize>*>(gate_up + packed_base + dim);
    PackedVec<scalar_t, VecSize> up_vec =
        *reinterpret_cast<const PackedVec<scalar_t, VecSize>*>(
            gate_up + packed_base + intermediate_size + dim);
    PackedVec<scalar_t, VecSize> out_vec;
#pragma unroll
    for (int i = 0; i < VecSize; ++i) {
      float gate_value = static_cast<float>(gate_vec.values[i]);
      float up_value = static_cast<float>(up_vec.values[i]);
      float sigmoid = 1.0f / (1.0f + expf(-gate_value));
      float silu_value = round_to_scalar_float<scalar_t>(gate_value * sigmoid);
      out_vec.values[i] = static_cast<scalar_t>(silu_value * up_value);
    }
    *reinterpret_cast<PackedVec<scalar_t, VecSize>*>(output + output_base + dim) = out_vec;
  }
}

template <typename scalar_t>
__global__ void exact_fused_add_rmsnorm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    int64_t rows,
    int64_t hidden_size,
    float eps) {
  extern __shared__ float shared[];
  int64_t row = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }
  int64_t base = row * hidden_size;

  float sum = 0.0f;
  for (int64_t dim = tid; dim < hidden_size; dim += blockDim.x) {
    int64_t offset = base + dim;
    float value = round_to_scalar_float<scalar_t>(
        load_as_float(input, offset) + load_as_float(residual, offset));
    residual[offset] = static_cast<scalar_t>(value);
    sum += value * value;
  }
  shared[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);
  for (int64_t dim = tid; dim < hidden_size; dim += blockDim.x) {
    int64_t offset = base + dim;
    float value = load_as_float(residual, offset);
    float weight_value = load_as_float(weight, dim);
    input[offset] = static_cast<scalar_t>(value * inv_rms * weight_value);
  }
}

inline bool is_aligned_to_16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) % 16) == 0;
}

template <typename scalar_t>
__global__ void write_beam_kv_step_kernel(
    scalar_t* __restrict__ beam_key,
    scalar_t* __restrict__ beam_value,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    int64_t elements,
    int64_t batch,
    int64_t active_beam_width,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t dst_layer_offset,
    int64_t dst_step_offset,
    int64_t dst_stride_batch,
    int64_t dst_stride_beam,
    int64_t dst_stride_head,
    int64_t dst_stride_dim,
    int64_t dst_value_layer_offset,
    int64_t dst_value_step_offset,
    int64_t dst_value_stride_batch,
    int64_t dst_value_stride_beam,
    int64_t dst_value_stride_head,
    int64_t dst_value_stride_dim,
    int64_t src_stride_batch,
    int64_t src_stride_beam,
    int64_t src_stride_head,
    int64_t src_stride_dim,
    int64_t value_src_stride_batch,
    int64_t value_src_stride_beam,
    int64_t value_src_stride_head,
    int64_t value_src_stride_dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= elements) {
    return;
  }

  int64_t dim = linear % head_dim;
  int64_t rest = linear / head_dim;
  int64_t head = rest % num_kv_heads;
  rest /= num_kv_heads;
  int64_t beam = rest % active_beam_width;
  int64_t batch_idx = rest / active_beam_width;
  if (batch_idx >= batch) {
    return;
  }

  int64_t src_offset = batch_idx * src_stride_batch
      + beam * src_stride_beam
      + head * src_stride_head
      + dim * src_stride_dim;
  int64_t value_src_offset = batch_idx * value_src_stride_batch
      + beam * value_src_stride_beam
      + head * value_src_stride_head
      + dim * value_src_stride_dim;
  int64_t dst_offset = dst_layer_offset
      + batch_idx * dst_stride_batch
      + dst_step_offset
      + beam * dst_stride_beam
      + head * dst_stride_head
      + dim * dst_stride_dim;
  int64_t value_dst_offset = dst_value_layer_offset
      + batch_idx * dst_value_stride_batch
      + dst_value_step_offset
      + beam * dst_value_stride_beam
      + head * dst_value_stride_head
      + dim * dst_value_stride_dim;
  beam_key[dst_offset] = k[src_offset];
  beam_value[value_dst_offset] = v[value_src_offset];
}

template <typename scalar_t>
__global__ void write_packed_qkv_prefill_kv_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ context_key,
    scalar_t* __restrict__ context_value,
    int64_t total_elements,
    int64_t kv_elements,
    int64_t seq_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t q_size,
    int64_t kv_size,
    int64_t k_stride_batch,
    int64_t k_stride_seq,
    int64_t k_stride_head,
    int64_t k_stride_dim,
    int64_t qkv_stride_batch,
    int64_t qkv_stride_seq,
    int64_t qkv_stride_col,
    int64_t key_layer_offset,
    int64_t key_stride_batch,
    int64_t key_stride_seq,
    int64_t key_stride_head,
    int64_t key_stride_dim,
    int64_t value_layer_offset,
    int64_t value_stride_batch,
    int64_t value_stride_seq,
    int64_t value_stride_head,
    int64_t value_stride_dim) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_elements) {
    return;
  }

  bool is_value = linear >= kv_elements;
  int64_t kv_linear = is_value ? linear - kv_elements : linear;
  int64_t dim = kv_linear % head_dim;
  int64_t rest = kv_linear / head_dim;
  int64_t head = rest % num_kv_heads;
  rest /= num_kv_heads;
  int64_t seq = rest % seq_len;
  int64_t batch_idx = rest / seq_len;

  if (is_value) {
    int64_t packed_col = q_size + kv_size + head * head_dim + dim;
    int64_t src_offset = batch_idx * qkv_stride_batch
        + seq * qkv_stride_seq
        + packed_col * qkv_stride_col;
    int64_t dst_offset = value_layer_offset
        + batch_idx * value_stride_batch
        + seq * value_stride_seq
        + head * value_stride_head
        + dim * value_stride_dim;
    context_value[dst_offset] = qkv[src_offset];
  } else {
    int64_t src_offset = batch_idx * k_stride_batch
        + seq * k_stride_seq
        + head * k_stride_head
        + dim * k_stride_dim;
    int64_t dst_offset = key_layer_offset
        + batch_idx * key_stride_batch
        + seq * key_stride_seq
        + head * key_stride_head
        + dim * key_stride_dim;
    context_key[dst_offset] = k[src_offset];
  }
}

template <typename scalar_t, int VecSize>
__global__ void write_packed_qkv_prefill_kv_vec_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ qkv,
    scalar_t* __restrict__ context_key,
    scalar_t* __restrict__ context_value,
    int64_t total_vec_elements,
    int64_t kv_vec_elements,
    int64_t batch,
    int64_t seq_len,
    int64_t kv_size,
    int64_t packed_size,
    int64_t q_size,
    int64_t key_layer_offset,
    int64_t value_layer_offset) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_vec_elements) {
    return;
  }

  bool is_value = linear >= kv_vec_elements;
  int64_t vec_linear = is_value ? linear - kv_vec_elements : linear;
  int64_t vec_cols = kv_size / VecSize;
  int64_t vec_col = vec_linear % vec_cols;
  int64_t row = vec_linear / vec_cols;
  if (row >= batch * seq_len) {
    return;
  }

  int64_t elem_col = vec_col * VecSize;
  if (is_value) {
    int64_t src_offset = row * packed_size + q_size + kv_size + elem_col;
    int64_t dst_offset = value_layer_offset + row * kv_size + elem_col;
    PackedVec<scalar_t, VecSize> value_vec =
        *reinterpret_cast<const PackedVec<scalar_t, VecSize>*>(qkv + src_offset);
    *reinterpret_cast<PackedVec<scalar_t, VecSize>*>(context_value + dst_offset) = value_vec;
  } else {
    int64_t src_offset = row * kv_size + elem_col;
    int64_t dst_offset = key_layer_offset + row * kv_size + elem_col;
    PackedVec<scalar_t, VecSize> key_vec =
        *reinterpret_cast<const PackedVec<scalar_t, VecSize>*>(k + src_offset);
    *reinterpret_cast<PackedVec<scalar_t, VecSize>*>(context_key + dst_offset) = key_vec;
  }
}

void fused_qk_norm_rope_cuda_launcher(
    torch::Tensor qkv,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double eps,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    double rope_theta,
    bool is_neox,
    torch::Tensor position_ids,
    double attention_factor,
    bool is_qk_norm) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be shaped [N, packed]");
  TORCH_CHECK(q_weight.is_cuda() && k_weight.is_cuda(), "q/k weights must be CUDA tensors");
  TORCH_CHECK(position_ids.is_cuda(), "position_ids must be a CUDA tensor");
  TORCH_CHECK(position_ids.scalar_type() == at::ScalarType::Int, "position_ids must be int32");
  TORCH_CHECK(rotary_dim == head_dim, "partial rotary_dim is not supported");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");

  int64_t tokens = qkv.size(0);
  int64_t packed_size = qkv.size(1);
  int threads = 256;
  dim3 grid(tokens, num_heads + num_kv_heads);
  size_t smem = threads * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      qkv.scalar_type(),
      "gr_trtllm_fused_qk_norm_rope",
      [&] {
        fused_qk_norm_rope_kernel<scalar_t><<<grid, threads, smem, stream>>>(
            qkv.data_ptr<scalar_t>(),
            q_weight.data_ptr<scalar_t>(),
            k_weight.data_ptr<scalar_t>(),
            position_ids.data_ptr<int32_t>(),
            tokens,
            packed_size,
            static_cast<int>(num_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            static_cast<float>(eps),
            static_cast<float>(rope_theta),
            static_cast<float>(attention_factor),
            is_neox,
            is_qk_norm);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor silu_and_mul_packed_cuda_launcher(
    torch::Tensor gate_up) {
  TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a CUDA tensor");
  TORCH_CHECK(gate_up.is_contiguous(), "gate_up must be contiguous");
  TORCH_CHECK(gate_up.dim() >= 1, "gate_up must have at least one dimension");
  TORCH_CHECK(gate_up.size(-1) % 2 == 0, "gate_up last dimension must be even");
  TORCH_CHECK(
      gate_up.scalar_type() == at::ScalarType::Half ||
      gate_up.scalar_type() == at::ScalarType::BFloat16 ||
      gate_up.scalar_type() == at::ScalarType::Float,
      "silu_and_mul_packed_cuda supports fp32/fp16/bf16 only");
  int64_t packed_size = gate_up.size(-1);
  int64_t intermediate_size = packed_size / 2;
  auto output_shape = gate_up.sizes().vec();
  output_shape.back() = intermediate_size;
  auto output = torch::empty(output_shape, gate_up.options());
  int64_t elements = output.numel();
  if (elements == 0) {
    return output;
  }
  int64_t tokens = elements / intermediate_size;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate_up.scalar_type(),
      "gr_trtllm_silu_and_mul_packed",
      [&] {
        constexpr int vec_size = 16 / sizeof(scalar_t);
        const char* vectorize_env = std::getenv("GR_INFERENCE_TRTLLM_PACKED_SILU_MUL_VECTORIZE");
        bool vectorize_enabled = vectorize_env == nullptr || std::string(vectorize_env) != "0";
        bool vectorizable =
            vectorize_enabled &&
            intermediate_size % vec_size == 0 &&
            is_aligned_to_16(gate_up.data_ptr<scalar_t>()) &&
            is_aligned_to_16(output.data_ptr<scalar_t>());
        if (vectorizable) {
          int vec_cols = static_cast<int>(intermediate_size / vec_size);
          int threads = std::max(1, std::min(vec_cols, 1024));
          silu_and_mul_packed_vec_kernel<scalar_t, vec_size>
              <<<static_cast<unsigned int>(tokens), threads, 0, stream>>>(
                  gate_up.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(),
                  tokens,
                  intermediate_size);
        } else {
          int threads = 256;
          int blocks = static_cast<int>((elements + threads - 1) / threads);
          silu_and_mul_packed_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              gate_up.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              elements,
              intermediate_size);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> exact_fused_add_rmsnorm_cuda_launcher(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps) {
  TORCH_CHECK(input.is_cuda() && residual.is_cuda() && weight.is_cuda(),
      "input, residual, and weight must be CUDA tensors");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual shapes must match");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous(),
      "input and residual must be contiguous");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type(),
      "input and residual dtypes must match");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
      "weight dtype must match input dtype");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
      input.scalar_type() == at::ScalarType::BFloat16 ||
      input.scalar_type() == at::ScalarType::Float,
      "exact_fused_add_rmsnorm_cuda supports fp32/fp16/bf16 only");

  int64_t hidden_size = input.size(-1);
  TORCH_CHECK(weight.size(0) == hidden_size, "weight size must match hidden size");
  int64_t elements = input.numel();
  if (elements == 0) {
    return {residual, input};
  }
  int64_t rows = elements / hidden_size;
  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "gr_trtllm_exact_fused_add_rmsnorm",
      [&] {
        exact_fused_add_rmsnorm_kernel<scalar_t>
            <<<static_cast<unsigned int>(rows), threads, threads * sizeof(float), stream>>>(
                input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                rows,
                hidden_size,
                static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual, input};
}

torch::Tensor packed_gemm_cuda_launcher(
    torch::Tensor input,
    torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input and weight must be CUDA tensors");
  TORCH_CHECK(input.dim() == 2, "input must be flattened to [M, K]");
  TORCH_CHECK(weight.dim() == 2, "weight must be shaped [N, K]");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "input and weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input and weight dtypes must match");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half ||
      input.scalar_type() == at::ScalarType::BFloat16,
      "packed_gemm_cuda supports fp16/bf16 only");
  TORCH_CHECK(input.size(1) == weight.size(1), "input K must match weight K");

  int64_t m_tokens = input.size(0);
  int64_t n_out = weight.size(0);
  int64_t k_hidden = input.size(1);
  auto output = torch::empty({m_tokens, n_out}, input.options());
  if (m_tokens == 0 || n_out == 0 || k_hidden == 0) {
    return output;
  }

  cudaDataType_t data_type = input.scalar_type() == at::ScalarType::Half
      ? CUDA_R_16F
      : CUDA_R_16BF;
  float alpha = 1.0f;
  float beta = 0.0f;

  // Row-major C[M,N] = input[M,K] @ weight[N,K]^T.
  // cuBLAS is column-major, so compute C^T[N,M] = weight @ input^T.
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  auto stream = at::cuda::getCurrentCUDAStream();
  GR_TRTLLM_CUBLAS_CHECK(cublasSetStream(handle, stream));
  GR_TRTLLM_CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      static_cast<int>(n_out),
      static_cast<int>(m_tokens),
      static_cast<int>(k_hidden),
      &alpha,
      weight.data_ptr(),
      data_type,
      static_cast<int>(k_hidden),
      input.data_ptr(),
      data_type,
      static_cast<int>(k_hidden),
      &beta,
      output.data_ptr(),
      data_type,
      static_cast<int>(n_out),
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void write_packed_qkv_prefill_kv_cuda_launcher(
    torch::Tensor k,
    torch::Tensor qkv,
    torch::Tensor context_key,
    torch::Tensor context_value,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  TORCH_CHECK(k.is_cuda() && qkv.is_cuda(), "k and qkv must be CUDA tensors");
  TORCH_CHECK(context_key.is_cuda() && context_value.is_cuda(), "ContextKV tensors must be CUDA tensors");
  TORCH_CHECK(k.dim() == 4, "k must be [B,S,H,D]");
  TORCH_CHECK(qkv.dim() == 3, "qkv must be [B,S,packed]");
  TORCH_CHECK(context_key.dim() == 5 && context_value.dim() == 5, "ContextKV tensors must be [L,B,S,H,D]");
  TORCH_CHECK(k.scalar_type() == qkv.scalar_type(), "k/qkv dtype mismatch");
  TORCH_CHECK(k.scalar_type() == context_key.scalar_type(), "k/key dtype mismatch");
  TORCH_CHECK(k.scalar_type() == context_value.scalar_type(), "k/value dtype mismatch");
  TORCH_CHECK(
      k.scalar_type() == at::ScalarType::Half ||
      k.scalar_type() == at::ScalarType::BFloat16 ||
      k.scalar_type() == at::ScalarType::Float,
      "write_packed_qkv_prefill_kv_cuda supports fp32/fp16/bf16 only");
  TORCH_CHECK(context_key.sizes() == context_value.sizes(), "ContextKV key/value shapes must match");
  TORCH_CHECK(layer_idx >= 0 && layer_idx < context_key.size(0), "layer_idx out of range");
  TORCH_CHECK(k.size(0) == qkv.size(0), "k/qkv batch mismatch");
  TORCH_CHECK(k.size(1) == qkv.size(1), "k/qkv seq_len mismatch");
  TORCH_CHECK(k.size(0) == context_key.size(1), "k batch must match ContextKV batch");
  TORCH_CHECK(k.size(1) <= context_key.size(2), "k seq_len exceeds ContextKV context length");
  TORCH_CHECK(k.size(2) == num_kv_heads, "k num_kv_heads mismatch");
  TORCH_CHECK(k.size(3) == head_dim, "k head_dim mismatch");
  TORCH_CHECK(context_key.size(3) == num_kv_heads, "ContextKV num_kv_heads mismatch");
  TORCH_CHECK(context_key.size(4) == head_dim, "ContextKV head_dim mismatch");
  TORCH_CHECK(num_heads > 0 && num_kv_heads > 0 && head_dim > 0, "invalid QKV shape metadata");

  int64_t q_size = num_heads * head_dim;
  int64_t kv_size = num_kv_heads * head_dim;
  TORCH_CHECK(qkv.size(2) >= q_size + 2 * kv_size, "packed qkv last dimension is too small");

  int64_t kv_elements = k.numel();
  int64_t total_elements = 2 * kv_elements;
  if (total_elements == 0) {
    return;
  }

  int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      k.scalar_type(),
      "gr_trtllm_write_packed_qkv_prefill_kv",
      [&] {
        constexpr int vec_size = 16 / sizeof(scalar_t);
        bool vectorizable =
            k.is_contiguous() &&
            qkv.is_contiguous() &&
            context_key.is_contiguous() &&
            context_value.is_contiguous() &&
            kv_size % vec_size == 0 &&
            is_aligned_to_16(k.data_ptr<scalar_t>()) &&
            is_aligned_to_16(qkv.data_ptr<scalar_t>()) &&
            is_aligned_to_16(context_key.data_ptr<scalar_t>() + layer_idx * context_key.stride(0)) &&
            is_aligned_to_16(context_value.data_ptr<scalar_t>() + layer_idx * context_value.stride(0));
        if (vectorizable) {
          int64_t kv_vec_elements = k.size(0) * k.size(1) * kv_size / vec_size;
          int64_t total_vec_elements = 2 * kv_vec_elements;
          int blocks = static_cast<int>((total_vec_elements + threads - 1) / threads);
          write_packed_qkv_prefill_kv_vec_kernel<scalar_t, vec_size>
              <<<blocks, threads, 0, stream>>>(
                  k.data_ptr<scalar_t>(),
                  qkv.data_ptr<scalar_t>(),
                  context_key.data_ptr<scalar_t>(),
                  context_value.data_ptr<scalar_t>(),
                  total_vec_elements,
                  kv_vec_elements,
                  k.size(0),
                  k.size(1),
                  kv_size,
                  qkv.size(2),
                  q_size,
                  layer_idx * context_key.stride(0),
                  layer_idx * context_value.stride(0));
        } else {
          int blocks = static_cast<int>((total_elements + threads - 1) / threads);
          write_packed_qkv_prefill_kv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              k.data_ptr<scalar_t>(),
              qkv.data_ptr<scalar_t>(),
              context_key.data_ptr<scalar_t>(),
              context_value.data_ptr<scalar_t>(),
              total_elements,
              kv_elements,
              k.size(1),
              num_kv_heads,
              head_dim,
              q_size,
              kv_size,
              k.stride(0),
              k.stride(1),
              k.stride(2),
              k.stride(3),
              qkv.stride(0),
              qkv.stride(1),
              qkv.stride(2),
              layer_idx * context_key.stride(0),
              context_key.stride(1),
              context_key.stride(2),
              context_key.stride(3),
              context_key.stride(4),
              layer_idx * context_value.stride(0),
              context_value.stride(1),
              context_value.stride(2),
              context_value.stride(3),
              context_value.stride(4));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void write_beam_kv_step_cuda_launcher(
    torch::Tensor beam_key,
    torch::Tensor beam_value,
    torch::Tensor k,
    torch::Tensor v,
    int64_t layer_idx,
    int64_t step,
    int64_t active_beam_width) {
  TORCH_CHECK(beam_key.is_cuda() && beam_value.is_cuda(), "BeamKV tensors must be CUDA tensors");
  TORCH_CHECK(k.is_cuda() && v.is_cuda(), "k/v tensors must be CUDA tensors");
  TORCH_CHECK(beam_key.dim() == 6 && beam_value.dim() == 6, "BeamKV tensors must be [L,B,S,W,H,D]");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "k/v must be [B,W,H,D]");
  TORCH_CHECK(beam_key.scalar_type() == beam_value.scalar_type(), "BeamKV dtypes must match");
  TORCH_CHECK(k.scalar_type() == beam_key.scalar_type(), "k dtype must match BeamKV dtype");
  TORCH_CHECK(v.scalar_type() == beam_key.scalar_type(), "v dtype must match BeamKV dtype");
  TORCH_CHECK(
      k.scalar_type() == at::ScalarType::Half ||
      k.scalar_type() == at::ScalarType::BFloat16 ||
      k.scalar_type() == at::ScalarType::Float,
      "write_beam_kv_step_cuda supports fp32/fp16/bf16 only");
  TORCH_CHECK(beam_key.sizes() == beam_value.sizes(), "BeamKV key/value shapes must match");
  TORCH_CHECK(k.sizes() == v.sizes(), "k/v shapes must match");
  TORCH_CHECK(layer_idx >= 0 && layer_idx < beam_key.size(0), "layer_idx out of range");
  TORCH_CHECK(step >= 0 && step < beam_key.size(2), "step out of range");
  TORCH_CHECK(active_beam_width > 0 && active_beam_width <= beam_key.size(3), "active_beam_width out of range");
  TORCH_CHECK(k.size(0) == beam_key.size(1), "k batch must match BeamKV batch");
  TORCH_CHECK(k.size(1) == active_beam_width, "k beam dimension must match active_beam_width");
  TORCH_CHECK(k.size(2) == beam_key.size(4), "k head count must match BeamKV");
  TORCH_CHECK(k.size(3) == beam_key.size(5), "k head dim must match BeamKV");

  int64_t elements = k.numel();
  if (elements == 0) {
    return;
  }
  int threads = 256;
  int blocks = static_cast<int>((elements + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      k.scalar_type(),
      "gr_trtllm_write_beam_kv_step",
      [&] {
        write_beam_kv_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            beam_key.data_ptr<scalar_t>(),
            beam_value.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            elements,
            k.size(0),
            active_beam_width,
            k.size(2),
            k.size(3),
            layer_idx * beam_key.stride(0),
            step * beam_key.stride(2),
            beam_key.stride(1),
            beam_key.stride(3),
            beam_key.stride(4),
            beam_key.stride(5),
            layer_idx * beam_value.stride(0),
            step * beam_value.stride(2),
            beam_value.stride(1),
            beam_value.stride(3),
            beam_value.stride(4),
            beam_value.stride(5),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
