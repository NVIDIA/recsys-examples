# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CUDA tests for the Triton FP8 quantization kernels; DeepGEMM is not needed.

Each kernel is compared with a plain PyTorch computation of the same values followed by
a PyTorch reference quantizer.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

HSTU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HSTU_ROOT))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

from ops.triton_ops.triton_fp8_quant import (
    GROUP_SIZE,
    triton_bias_gelu_quant_1x128,
    triton_layernorm_quant_1x128,
    triton_quant_1x128,
)

SHAPES = [(1, 128), (7, 512), (130, 1536), (130, 2048)]


def reference_quant(values, ue8m0):
    groups = values.float().reshape(values.shape[0], -1, GROUP_SIZE)
    scale = groups.abs().amax(-1).clamp(min=1e-4) / 448.0
    if ue8m0:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    return q.reshape(values.shape), scale


def dequantize(q, scale):
    groups = q.float().reshape(q.shape[0], -1, GROUP_SIZE)
    return (groups * scale[..., None]).reshape(q.shape)


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def mismatch_fraction(a, b):
    return (a != b).float().mean().item()


def assert_matches_reference(q, scale, expected_values, ue8m0):
    ref_q, ref_scale = reference_quant(expected_values, ue8m0)
    # A fused kernel sums in a different order than torch and divides with the fast fp32
    # division, so scales agree to about 1e-6 and a rare value can land one float8 step
    # away. The scales, the codes and the dequantized values must still agree.
    scale_mismatch = (~torch.isclose(scale, ref_scale, rtol=1e-3)).float().mean()
    assert scale_mismatch.item() < 1e-3
    assert mismatch_fraction(q.view(torch.uint8), ref_q.view(torch.uint8)) < 1e-3
    assert cosine(dequantize(q, scale), dequantize(ref_q, ref_scale)) > 0.99999
    assert cosine(dequantize(q, scale), expected_values) > 0.999


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("m,k", SHAPES)
def test_quant_matches_reference(m, k, ue8m0):
    torch.manual_seed(0)
    x = 3 * torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q, scale = triton_quant_1x128(x, ue8m0)
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (m, k // GROUP_SIZE)
    assert_matches_reference(q, scale, x.float(), ue8m0)


def test_ue8m0_scales_are_powers_of_two():
    x = torch.randn(33, 512, device="cuda", dtype=torch.bfloat16)
    _, scale = triton_quant_1x128(x, ue8m0=True)
    assert torch.equal(torch.exp2(torch.log2(scale).round()), scale)


@pytest.mark.parametrize("has_affine", [True, False])
@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("m,k", SHAPES)
def test_layernorm_quant_matches_reference(m, k, ue8m0, has_affine):
    torch.manual_seed(0)
    x = 2 * torch.randn(m, k, device="cuda", dtype=torch.bfloat16) + 1
    weight = (0.5 * torch.randn(k, device="cuda") + 1).to(torch.bfloat16)
    bias = (0.1 * torch.randn(k, device="cuda")).to(torch.bfloat16)
    if not has_affine:
        weight = bias = None
    q, scale = triton_layernorm_quant_1x128(x, weight, bias, 1e-5, ue8m0)
    expected = F.layer_norm(
        x.float(),
        (k,),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        1e-5,
    )
    assert_matches_reference(q, scale, expected, ue8m0)


@pytest.mark.parametrize("ue8m0", [False, True])
@pytest.mark.parametrize("m,k", SHAPES)
def test_bias_gelu_quant_matches_reference(m, k, ue8m0):
    torch.manual_seed(0)
    x = 2 * torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    bias = (0.5 * torch.randn(k, device="cuda")).to(torch.bfloat16)
    q, scale = triton_bias_gelu_quant_1x128(x, bias, ue8m0)
    expected = F.gelu(x.float() + bias.float())
    assert_matches_reference(q, scale, expected, ue8m0)


def test_empty_input():
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    for q, scale in (
        triton_quant_1x128(x, False),
        triton_layernorm_quant_1x128(x, None, None, 1e-5, False),
        triton_bias_gelu_quant_1x128(x, bias, False),
    ):
        assert q.shape == (0, 256) and scale.shape == (0, 2)
