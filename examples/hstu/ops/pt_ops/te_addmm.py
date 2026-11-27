# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import nvtx
import torch
import torch.nn.functional as F


def _quantize(
    tensor: torch.Tensor,
    *,
    columnwise: bool,
    gradient: bool = False,
) -> torch.Tensor:
    with nvtx.annotate("fp8 projection quantizer setup"):
        from transformer_engine.pytorch import DType, Float8CurrentScalingQuantizer

        dtype = DType.kFloat8E5M2 if gradient else DType.kFloat8E4M3
        # TE current scaling needs rowwise storage even for columnwise GEMM operands.
        quantizer = Float8CurrentScalingQuantizer(
            dtype, tensor.device, rowwise=True, columnwise=columnwise
        )
    with nvtx.annotate("fp8 projection contiguous operand"):
        contiguous = tensor.contiguous()
    with nvtx.annotate("fp8 projection quantize operand"):
        return quantizer(contiguous)


def te_fp8_addmm_silu_fwd(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, silu: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute a BF16 projection and optional SiLU using an FP8 GEMM."""
    rows = x.size(0)
    if rows == 0:
        z = x.new_empty((0, w.size(1)))
        return z, F.silu(z) if silu else None

    from transformer_engine.pytorch.cpp_extensions import general_gemm

    # Pad within the projection so neither attention offsets nor loss rows change.
    padding = (-rows) % 16
    if padding:
        with nvtx.annotate("fp8 projection pad forward input"):
            padded_x = F.pad(x, (0, 0, 0, padding))
        x_fp8 = _quantize(padded_x, columnwise=False)
        del padded_x
    else:
        x_fp8 = _quantize(x, columnwise=False)
    w_fp8 = _quantize(w, columnwise=True)
    # TE uses column-major GEMM ordering: NN computes x @ w for [in, out] weights.
    if y.ndim == 1:
        z, *_ = general_gemm(w_fp8, x_fp8, layout="NN", out_dtype=x.dtype, bias=y)
    else:
        if padding:
            with nvtx.annotate("fp8 projection pad residual"):
                residual = F.pad(y, (0, 0, 0, padding))
        else:
            with nvtx.annotate("fp8 projection clone residual"):
                residual = y.contiguous().clone()
        z, *_ = general_gemm(
            w_fp8, x_fp8, layout="NN", out_dtype=x.dtype, out=residual, accumulate=True
        )
    z = z[:rows]
    return z, F.silu(z) if silu else None


def te_fp8_linear_bwd(
    x: torch.Tensor, w: torch.Tensor, dz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BF16 input and weight gradients using FP8 GEMMs."""
    rows = x.size(0)
    if rows == 0:
        return torch.empty_like(x), torch.zeros_like(w)

    from transformer_engine.pytorch.cpp_extensions import general_gemm

    padding = (-rows) % 16
    if padding:
        with nvtx.annotate("fp8 projection pad backward operands"):
            x = F.pad(x, (0, 0, 0, padding))
            dz = F.pad(dz, (0, 0, 0, padding))
    x_fp8 = _quantize(x, columnwise=True)
    w_fp8 = _quantize(w, columnwise=False)
    dz_fp8 = _quantize(dz, columnwise=True, gradient=True)
    # TN computes dz @ w.T and NT computes x.T @ dz.
    dx, *_ = general_gemm(
        w_fp8, dz_fp8, layout="TN", out_dtype=x.dtype, use_split_accumulator=True
    )
    dw, *_ = general_gemm(
        dz_fp8, x_fp8, layout="NT", out_dtype=w.dtype, use_split_accumulator=True
    )
    return dx[:rows], dw
