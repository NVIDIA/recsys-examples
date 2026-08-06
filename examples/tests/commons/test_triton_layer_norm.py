# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from ops.triton_ops.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
@pytest.mark.parametrize("shape", [(7, 64), (33, 257)])
@pytest.mark.parametrize("accumulate", [False, True])
def test_non_learnable_layer_norm_backward_residual_gradient(
    dtype, rtol, atol, shape, accumulate
):
    torch.manual_seed(1234)
    device = torch.device("cuda")
    eps = 1e-5
    rows, columns = shape

    # Slicing keeps the feature dimension contiguous while exercising a row
    # stride larger than the logical feature dimension.
    x = torch.randn((rows, columns + 11), device=device, dtype=dtype)[:, :columns]
    dy = torch.randn((rows, columns + 7), device=device, dtype=dtype)[:, :columns]
    dx_accumulate = None
    if accumulate:
        dx_accumulate = torch.randn(
            (rows, columns + 5), device=device, dtype=dtype
        )[:, :columns]

    x_float = x.float()
    mean = x_float.mean(dim=-1)
    rstd = torch.rsqrt(x_float.var(dim=-1, unbiased=False) + eps)
    block_d = 1 << (columns - 1).bit_length()
    num_warps = min(max(block_d // 256, 1), 8)
    actual, dweight, dbias = triton_weighted_layer_norm_bwd(
        dy=dy,
        x=x,
        weight=None,
        bias=None,
        mean=mean,
        rstd=rstd,
        learnable=False,
        eps=eps,
        BLOCK_D=block_d,
        num_warps=num_warps,
        dx_accumulate=dx_accumulate,
    )

    reference_x = x.detach().clone().requires_grad_(True)
    reference_y = F.layer_norm(reference_x, (columns,), eps=eps)
    expected = torch.autograd.grad(reference_y, reference_x, dy)[0]
    if dx_accumulate is not None:
        expected = expected + dx_accumulate

    absolute_error = (actual.float() - expected.float()).abs()
    relative_error = absolute_error / expected.float().abs().clamp_min(
        torch.finfo(torch.float32).eps
    )
    print(
        f"dtype={dtype}, shape={shape}, accumulate={accumulate}, "
        f"max_abs_error={absolute_error.max().item():.8e}, "
        f"mean_abs_error={absolute_error.mean().item():.8e}, "
        f"max_rel_error={relative_error.max().item():.8e}"
    )

    assert dweight is None
    assert dbias is None
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
