# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Per-token FP8 activation quantization in groups of 128 channels, for DeepGEMM.

``triton_quant_1x128`` quantizes a bf16 tensor on its own. The fused variants quantize
the result of the preceding op in the same pass, so the bf16 intermediate is never
written to memory: ``triton_layernorm_quant_1x128`` after a LayerNorm and
``triton_bias_gelu_quant_1x128`` after a bias add and GELU.
"""

import torch
import triton
import triton.language as tl

GROUP_SIZE = 128


@triton.jit
def _quantize_group(values, ue8m0: tl.constexpr):
    """Scale each row of a [block_m, 128] fp32 tile into float8_e4m3fn: (q, scale)."""
    # The floor of DeepGEMM's own casts: an all-zero group still gets a finite scale.
    amax = tl.maximum(tl.max(tl.abs(values), axis=1), 1e-4)
    scale = amax / 448.0  # largest float8_e4m3fn value
    if ue8m0:
        # Round the scale up to a power of two, the only scale format SM100+ accepts.
        bits = scale.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exponent = tl.minimum(tl.maximum(exponent, 1), 254)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
    return (values / scale[:, None]).to(tl.float8e4nv), scale


@triton.jit
def _quant_1x128_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    num_rows,
    num_cols,
    num_groups,
    block_m: tl.constexpr,
    ue8m0: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    rows = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    cols = pid_g * 128 + tl.arange(0, 128)
    row_mask = rows < num_rows
    offsets = rows[:, None] * num_cols + cols[None, :]
    x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    q, scale = _quantize_group(x, ue8m0)
    tl.store(q_ptr + offsets, q, mask=row_mask[:, None])
    tl.store(scale_ptr + rows * num_groups + pid_g, scale, mask=row_mask)


@triton.jit
def _layernorm_quant_1x128_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    q_ptr,
    scale_ptr,
    num_rows,
    eps,
    num_cols: tl.constexpr,
    num_groups: tl.constexpr,
    block_m: tl.constexpr,
    has_affine: tl.constexpr,
    ue8m0: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = (pid * block_m + tl.arange(0, block_m)).to(tl.int64)
    lane = tl.arange(0, 128)
    row_mask = rows < num_rows

    # The row is read three times (mean, variance, normalize and quantize) one 128
    # channel group at a time, so any width that is a multiple of 128 works and the
    # repeated reads are served from cache.
    total = tl.zeros([block_m], dtype=tl.float32)
    for group in range(num_groups):
        offsets = rows[:, None] * num_cols + group * 128 + lane[None, :]
        x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        total += tl.sum(x, axis=1)
    mean = total / num_cols

    squares = tl.zeros([block_m], dtype=tl.float32)
    for group in range(num_groups):
        offsets = rows[:, None] * num_cols + group * 128 + lane[None, :]
        x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        centered = tl.where(row_mask[:, None], x - mean[:, None], 0.0)
        squares += tl.sum(centered * centered, axis=1)
    rstd = 1.0 / tl.sqrt(squares / num_cols + eps)

    for group in range(num_groups):
        offsets = rows[:, None] * num_cols + group * 128 + lane[None, :]
        x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * rstd[:, None]
        if has_affine:
            weight = tl.load(weight_ptr + group * 128 + lane).to(tl.float32)
            bias = tl.load(bias_ptr + group * 128 + lane).to(tl.float32)
            y = y * weight[None, :] + bias[None, :]
        q, scale = _quantize_group(y, ue8m0)
        tl.store(q_ptr + offsets, q, mask=row_mask[:, None])
        tl.store(scale_ptr + rows * num_groups + group, scale, mask=row_mask)


@triton.jit
def _bias_gelu_quant_1x128_kernel(
    x_ptr,
    bias_ptr,
    q_ptr,
    scale_ptr,
    num_rows,
    num_cols,
    num_groups,
    block_m: tl.constexpr,
    ue8m0: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    rows = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)
    cols = pid_g * 128 + tl.arange(0, 128)
    row_mask = rows < num_rows
    offsets = rows[:, None] * num_cols + cols[None, :]
    x = tl.load(x_ptr + offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
    x += tl.load(bias_ptr + cols).to(tl.float32)[None, :]
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))  # exact (erf) GELU
    q, scale = _quantize_group(y, ue8m0)
    tl.store(q_ptr + offsets, q, mask=row_mask[:, None])
    tl.store(scale_ptr + rows * num_groups + pid_g, scale, mask=row_mask)


def _outputs(x):
    num_rows, num_cols = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        (num_rows, num_cols // GROUP_SIZE), dtype=torch.float32, device=x.device
    )
    return q, scale


def triton_quant_1x128(x: torch.Tensor, ue8m0: bool, block_m: int = 32):
    """Quantize a [M, K] tensor to float8_e4m3fn with one scale per 128 channels of a row.

    Returns (q, scale): q is float8_e4m3fn [M, K], scale is float32 [M, K // 128]. With
    ue8m0 the scales are rounded up to powers of two (SM100+ block-scaled GEMMs).
    """
    assert x.dim() == 2 and x.shape[1] % GROUP_SIZE == 0
    x = x.contiguous()
    num_rows, num_cols = x.shape
    q, scale = _outputs(x)
    if num_rows > 0:
        num_groups = num_cols // GROUP_SIZE
        grid = (triton.cdiv(num_rows, block_m), num_groups)
        _quant_1x128_kernel[grid](
            x, q, scale, num_rows, num_cols, num_groups, block_m=block_m, ue8m0=ue8m0
        )
    return q, scale


def triton_layernorm_quant_1x128(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    ue8m0: bool,
    block_m: int = 8,
):
    """LayerNorm over the last dim of [M, K], then ``triton_quant_1x128`` of the result.

    ``weight`` and ``bias`` are the affine parameters, or both None for a LayerNorm
    without them. The normalized tensor is never written to memory.
    """
    assert x.dim() == 2 and x.shape[1] % GROUP_SIZE == 0
    assert (weight is None) == (bias is None)
    x = x.contiguous()
    num_rows, num_cols = x.shape
    q, scale = _outputs(x)
    if num_rows > 0:
        has_affine = weight is not None
        _layernorm_quant_1x128_kernel[(triton.cdiv(num_rows, block_m),)](
            x,
            weight if has_affine else x,
            bias if has_affine else x,
            q,
            scale,
            num_rows,
            eps,
            num_cols=num_cols,
            num_groups=num_cols // GROUP_SIZE,
            block_m=block_m,
            has_affine=has_affine,
            ue8m0=ue8m0,
        )
    return q, scale


def triton_bias_gelu_quant_1x128(
    x: torch.Tensor, bias: torch.Tensor, ue8m0: bool, block_m: int = 32
):
    """Add ``bias`` to a [M, K] tensor, apply the exact (erf) GELU, then quantize."""
    assert x.dim() == 2 and x.shape[1] % GROUP_SIZE == 0
    x = x.contiguous()
    num_rows, num_cols = x.shape
    q, scale = _outputs(x)
    if num_rows > 0:
        num_groups = num_cols // GROUP_SIZE
        grid = (triton.cdiv(num_rows, block_m), num_groups)
        _bias_gelu_quant_1x128_kernel[grid](
            x,
            bias,
            q,
            scale,
            num_rows,
            num_cols,
            num_groups,
            block_m=block_m,
            ue8m0=ue8m0,
        )
    return q, scale
