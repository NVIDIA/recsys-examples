# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import triton
import triton.language as tl

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef

from commons.ops.triton_ops.common import triton_autotune


def silu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [2, 4, 8, 16]:
            config = triton.Config({"x_block_size": x_block_size}, num_warps)
            configs.append(config)
    return configs


@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_forward(
    output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    output_block_ptr = output_ptr + x_offset
    input_block_ptr = input_ptr + x_offset
    cols = tl.arange(0, x_block_size)

    input = tl.load(input_block_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    output = fast_dividef(input, 1.0 + tl.exp(-input)).to(output_ptr.dtype.element_ty)

    tl.store(output_block_ptr + cols, output, mask=mask)


@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_backward(
    grad_input_ptr: tl.tensor,
    grad_output_ptr: tl.tensor,
    input_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    grad_input_ptr += x_offset
    grad_output_ptr += x_offset
    input_ptr += x_offset
    cols = tl.arange(0, x_block_size)
    grad_output = tl.load(grad_output_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    input = tl.load(input_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    sigma = tl.sigmoid(input)
    grad_input = grad_output * (sigma + input * sigma * (1 - sigma))
    tl.store(
        grad_input_ptr + cols, grad_input.to(grad_input_ptr.dtype.element_ty), mask=mask
    )


def triton_silu_fwd(input: torch.Tensor) -> torch.Tensor:
    x_size = input.numel()
    input_1d = input.view(-1).contiguous()
    output = torch.empty_like(input_1d)

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_forward[grid](
        output,
        input_1d,
        input_1d.numel(),
    )
    return output.view(input.shape)


# TODO: compare performance with torch.ops.aten.silu_backward
def triton_silu_bwd(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    shape = input.shape
    input_1d = input.view(-1).contiguous()
    grad_input = torch.empty_like(input_1d)
    grad_output_1d = grad_output.view(-1).contiguous()
    x_size = input.numel()

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_backward[grid](
        grad_input,
        grad_output_1d,
        input_1d,
        input_1d.numel(),
    )
    return grad_input.view(shape)


class TritonSilu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        output = triton_silu_fwd(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors[0]
        return triton_silu_bwd(grad_output, input)


def triton_silu(input: torch.Tensor) -> torch.Tensor:
    input = input.contiguous()
    return TritonSilu.apply(input)
