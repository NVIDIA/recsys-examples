#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Grouped MLP as PyTorch Custom Ops

Structure
[1] Triton Kernels      - SiLU*Up kernel
[2] Custom Ops          - strided_bmm, silu_mul, grouped_mlp_gated_fwd/bwd
[3] nn.Module Wrappers  - GroupedMLP_CustomOp, ReferenceGroupedMLP
[4] Correctness & Benchmark
"""

import sys

sys.path.insert(
    0, "/home/scratch.runchuz_gpu/repos-github/recsys-examples/examples/hstu"
)

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        from triton.language.math import fast_dividef

from ops.triton_ops.common import triton_autotune

# =============================================================================
# [1] Triton Kernel Configurations
# =============================================================================


def silu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [2, 4, 8, 16]:
            config = triton.Config({"x_block_size": x_block_size}, num_warps)
            configs.append(config)
    return configs


# =============================================================================
# [1] Triton Kernels - SiLU * Up (SwiGLU pattern)
# =============================================================================


@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_mul_forward_kernel(
    output_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    up_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    """Fused forward: output = silu(gate) * up"""
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    cols = tl.arange(0, x_block_size)

    gate = tl.load(gate_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)

    silu_gate = fast_dividef(gate, 1.0 + tl.exp(-gate))
    output = (silu_gate * up).to(output_ptr.dtype.element_ty)

    tl.store(output_ptr + x_offset + cols, output, mask=mask)


@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_mul_backward_kernel(
    grad_gate_ptr: tl.tensor,
    grad_up_ptr: tl.tensor,
    grad_output_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    up_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    """Fused backward for output = silu(gate) * up"""
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    cols = tl.arange(0, x_block_size)

    grad_output = tl.load(grad_output_ptr + x_offset + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gate = tl.load(gate_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)

    sigma = tl.sigmoid(gate)
    silu_gate = gate * sigma
    dsilu_dgate = sigma + gate * sigma * (1.0 - sigma)

    grad_gate = grad_output * up * dsilu_dgate
    grad_up = grad_output * silu_gate

    tl.store(
        grad_gate_ptr + x_offset + cols,
        grad_gate.to(grad_gate_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        grad_up_ptr + x_offset + cols,
        grad_up.to(grad_up_ptr.dtype.element_ty),
        mask=mask,
    )


# =============================================================================
# [1] Triton Kernel Launchers (Internal)
# =============================================================================


def _launch_silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Internal: launch forward kernel"""
    x_size = gate.numel()
    gate_1d = gate.reshape(-1).contiguous()
    up_1d = up.reshape(-1).contiguous()
    output = torch.empty_like(gate_1d)

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_mul_forward_kernel[grid](output, gate_1d, up_1d, x_size)
    return output.view(gate.shape)


def _launch_silu_mul_bwd(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Internal: launch backward kernel"""
    shape = gate.shape
    x_size = gate.numel()
    gate_1d = gate.reshape(-1).contiguous()
    up_1d = up.reshape(-1).contiguous()
    grad_output_1d = grad_output.reshape(-1).contiguous()
    grad_gate = torch.empty_like(gate_1d)
    grad_up = torch.empty_like(up_1d)

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_mul_backward_kernel[grid](
        grad_gate, grad_up, grad_output_1d, gate_1d, up_1d, x_size
    )
    return grad_gate.view(shape), grad_up.view(shape)


# =============================================================================
# [2] Custom Op: silu_mul
# =============================================================================


@torch.library.custom_op("grouped_mlp::silu_mul", mutates_args=(), device_types="cuda")
def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU multiplication: output = silu(gate) * up

    This is the SwiGLU activation pattern used in modern LLMs.
    """
    torch._check(
        gate.shape == up.shape, lambda: f"Shape mismatch: {gate.shape} vs {up.shape}"
    )
    return _launch_silu_mul_fwd(gate.contiguous(), up.contiguous())


@silu_mul.register_fake
def _(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    torch._check(gate.shape == up.shape)
    return torch.empty_like(gate)


@torch.library.custom_op(
    "grouped_mlp::silu_mul_backward", mutates_args=(), device_types="cuda"
)
def silu_mul_backward(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward for silu_mul: returns (grad_gate, grad_up)"""
    return _launch_silu_mul_bwd(
        grad_output.contiguous(), gate.contiguous(), up.contiguous()
    )


@silu_mul_backward.register_fake
def _(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(gate), torch.empty_like(up)


def _silu_mul_bwd_fn(ctx, grad_output):
    gate, up = ctx.saved_tensors
    return silu_mul_backward(grad_output, gate, up)


def _silu_mul_setup_ctx(ctx, inputs, output):
    gate, up = inputs
    ctx.save_for_backward(gate, up)


silu_mul.register_autograd(_silu_mul_bwd_fn, setup_context=_silu_mul_setup_ctx)


# =============================================================================
# [2] Custom Op: strided_bmm (Strided Batched Matrix Multiply)
# =============================================================================


@torch.library.custom_op(
    "grouped_mlp::strided_bmm", mutates_args=(), device_types="cuda"
)
def strided_bmm(
    x: torch.Tensor,  # (B, N, K)
    weight: torch.Tensor,  # (N, K, M)
) -> torch.Tensor:
    """
    Strided BMM: output[b, n, :] = x[b, n, :] @ weight[n, :, :]

    Input:  x      - (batch_size, num_groups, input_dim)
    Weight: weight - (num_groups, input_dim, output_dim)
    Output:        - (batch_size, num_groups, output_dim)

    Internally uses permute + torch.bmm for efficiency.
    """
    batch_size, num_groups, _ = x.shape
    output_dim = weight.shape[2]

    output = torch.empty(
        batch_size, num_groups, output_dim, device=x.device, dtype=x.dtype
    )
    # x: (B, N, K) -> (N, B, K)
    # weight: (N, K, M)
    # out: (N, B, M) -> (B, N, M)
    torch.bmm(x.permute(1, 0, 2), weight, out=output.permute(1, 0, 2))
    return output


@strided_bmm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    batch_size, num_groups, _ = x.shape
    output_dim = weight.shape[2]
    return x.new_empty(batch_size, num_groups, output_dim)


@torch.library.custom_op(
    "grouped_mlp::strided_bmm_backward", mutates_args=(), device_types="cuda"
)
def strided_bmm_backward(
    grad_output: torch.Tensor,  # (B, N, M)
    x: torch.Tensor,  # (B, N, K)
    weight: torch.Tensor,  # (N, K, M)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward for strided_bmm.

    grad_x      = grad_output @ weight.T  => (B, N, K)
    grad_weight = x.T @ grad_output       => (N, K, M)
    """
    grad_output_t = grad_output.permute(1, 0, 2)  # (N, B, M)

    # grad_x: (N, B, M) @ (N, M, K) -> (N, B, K) -> (B, N, K)
    grad_x = torch.empty_like(x)
    torch.bmm(grad_output_t, weight.transpose(-1, -2), out=grad_x.permute(1, 0, 2))

    # grad_weight: (N, K, B) @ (N, B, M) -> (N, K, M)
    x_t = x.permute(1, 0, 2)  # (N, B, K)
    grad_weight = torch.bmm(x_t.transpose(-1, -2), grad_output_t)

    return grad_x, grad_weight


@strided_bmm_backward.register_fake
def _(
    grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(weight)


def _strided_bmm_bwd_fn(ctx, grad_output):
    x, weight = ctx.saved_tensors
    return strided_bmm_backward(grad_output, x, weight)


def _strided_bmm_setup_ctx(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


strided_bmm.register_autograd(_strided_bmm_bwd_fn, setup_context=_strided_bmm_setup_ctx)


def grouped_mlp_gated_forward(
    x: torch.Tensor,  # (B*N, D_in)
    gate_weight: torch.Tensor,  # (N, D_in, D_hidden)
    up_weight: torch.Tensor,  # (N, D_in, D_hidden)
    down_weight: torch.Tensor,  # (N, D_hidden, D_out)
    num_groups: int,
) -> torch.Tensor:
    """
    Grouped MLP forward with gating (SwiGLU pattern).

    This is a composition of custom ops (strided_bmm, silu_mul).
    Autograd is handled automatically through the registered backward of each op.

    For each group n:
        gate = x @ gate_weight[n]
        up = x @ up_weight[n]
        hidden = silu(gate) * up
        output = hidden @ down_weight[n]

    Args:
        x: Input tensor, shape (B*N, D_in)
        gate_weight: Gate projection weights, shape (N, D_in, D_hidden)
        up_weight: Up projection weights, shape (N, D_in, D_hidden)
        down_weight: Down projection weights, shape (N, D_hidden, D_out)
        num_groups: Number of groups (N)

    Returns:
        Output tensor, shape (B*N, D_out)
    """
    batch_size = x.shape[0] // num_groups
    input_dim = gate_weight.shape[1]
    output_dim = down_weight.shape[2]

    # Reshape: (B*N, D_in) -> (B, N, D_in)
    x_3d = x.reshape(batch_size, num_groups, input_dim)

    # Gate BMM: (B, N, D_in) @ (N, D_in, D_hidden) -> (B, N, D_hidden)
    gate = strided_bmm(x_3d, gate_weight)

    # Up BMM: (B, N, D_in) @ (N, D_in, D_hidden) -> (B, N, D_hidden)
    up = strided_bmm(x_3d, up_weight)

    # Fused SiLU activation: hidden = silu(gate) * up
    hidden = silu_mul(gate, up)

    # Down BMM: (B, N, D_hidden) @ (N, D_hidden, D_out) -> (B, N, D_out)
    output = strided_bmm(hidden, down_weight)

    # Reshape: (B, N, D_out) -> (B*N, D_out)
    return output.reshape(-1, output_dim)


# =============================================================================
# [3] nn.Module: GroupedMLP_CustomOp
# =============================================================================


class GroupedMLP(nn.Module):
    """
    Grouped MLP using custom ops.

    This implementation uses registered custom ops instead of autograd.Function,
    making it compatible with torch.compile and scenarios where autograd is not available.

    Forward computation:
        For each group n:
            gate = x @ gate_weight[n]
            up = x @ up_weight[n]
            hidden = silu(gate) * up
            output = hidden @ down_weight[n]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_groups: int,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weights: (N, D_in, D_hidden) or (N, D_hidden, D_out)
        self.gate_weight = nn.Parameter(
            torch.empty(num_groups, input_dim, hidden_dim, device=device, dtype=dtype)
        )
        self.up_weight = nn.Parameter(
            torch.empty(num_groups, input_dim, hidden_dim, device=device, dtype=dtype)
        )
        self.down_weight = nn.Parameter(
            torch.empty(num_groups, hidden_dim, output_dim, device=device, dtype=dtype)
        )

        self._init_weights()

    def _init_weights(self):
        for i in range(self.num_groups):
            nn.init.xavier_normal_(self.gate_weight[i], gain=1.0)
            nn.init.xavier_normal_(self.up_weight[i], gain=1.0)
            nn.init.xavier_normal_(self.down_weight[i], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using custom ops.

        Args:
            x: Input tensor, shape (B*N, D_in)

        Returns:
            Output tensor, shape (B*N, D_out)
        """
        return grouped_mlp_gated_forward(
            x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            self.num_groups,
        )

    def forward_decomposed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with explicit steps (for debugging/profiling).

        Same computation as forward(), but with explicit intermediate tensors.
        """
        batch_size = x.shape[0] // self.num_groups

        # Reshape
        x_3d = x.reshape(batch_size, self.num_groups, self.input_dim)

        # 3 BMMs + 1 fused activation
        gate = strided_bmm(x_3d, self.gate_weight)
        up = strided_bmm(x_3d, self.up_weight)
        hidden = silu_mul(gate, up)
        output = strided_bmm(hidden, self.down_weight)

        return output.reshape(-1, self.output_dim)


# =============================================================================
# [3] nn.Module: ReferenceGroupedMLP
# =============================================================================


class ReferenceGroupedMLP(nn.Module):
    """
    Reference implementation using loop over groups with nn.Linear.

    This is the baseline for correctness verification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_groups: int,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gate_proj = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ]
        )
        self.up_proj = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ]
        )
        self.down_proj = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dim, output_dim, bias=False, device=device, dtype=dtype
                )
                for _ in range(num_groups)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        for module_list in [self.gate_proj, self.up_proj, self.down_proj]:
            for layer in module_list:
                nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with loop over groups."""
        x = x.reshape(-1, self.num_groups, self.input_dim)
        x_split = torch.split(x, 1, dim=1)

        out_list = []
        for i in range(self.num_groups):
            x_i = x_split[i].squeeze(1)
            gate_i = self.gate_proj[i](x_i)
            up_i = self.up_proj[i](x_i)
            hidden_i = F.silu(gate_i) * up_i
            out_i = self.down_proj[i](hidden_i)
            out_list.append(out_i)

        return torch.stack(out_list, dim=1).reshape(-1, self.output_dim)


# =============================================================================
# [4] Weight Copy Utilities
# =============================================================================


def copy_weights_ref_to_customop(
    ref_model: ReferenceGroupedMLP,
    customop_model: GroupedMLP,
):
    """Copy weights from Reference model to CustomOp model."""
    with torch.no_grad():
        for i in range(ref_model.num_groups):
            # nn.Linear weight is (out, in), we need (in, out)
            customop_model.gate_weight[i].copy_(ref_model.gate_proj[i].weight.T)
            customop_model.up_weight[i].copy_(ref_model.up_proj[i].weight.T)
            customop_model.down_weight[i].copy_(ref_model.down_proj[i].weight.T)


# =============================================================================
# [5] Correctness Verification
# =============================================================================


def check_forward_correctness(
    ref_model: ReferenceGroupedMLP,
    customop_model: GroupedMLP,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, float]:
    """Check forward correctness."""
    num_groups = ref_model.num_groups
    input_dim = ref_model.input_dim

    x = torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)

    with torch.no_grad():
        ref_out = ref_model(x)
        customop_out = customop_model(x)

    max_diff = (ref_out - customop_out).abs().max().item()
    # bf16 has limited precision, use looser tolerance
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    passed = torch.allclose(ref_out, customop_out, atol=atol, rtol=rtol)

    return passed, max_diff


def check_backward_correctness(
    ref_model: ReferenceGroupedMLP,
    customop_model: GroupedMLP,
    batch_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[bool, float]:
    """Check backward correctness (input gradient)."""
    num_groups = ref_model.num_groups
    input_dim = ref_model.input_dim

    x_ref = torch.randn(
        batch_size * num_groups,
        input_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    x_customop = x_ref.detach().clone().requires_grad_(True)

    ref_out = ref_model(x_ref)
    customop_out = customop_model(x_customop)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)
    customop_out.backward(grad_output)

    max_diff = (x_ref.grad - x_customop.grad).abs().max().item()
    # bf16 has ~3 significant digits precision, multiple BMM ops accumulate error
    # Use looser tolerance: 5e-2 for bf16 (vs 1e-2 for fp32)
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    rtol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    passed = torch.allclose(x_ref.grad, x_customop.grad, atol=atol, rtol=rtol)

    return passed, max_diff


def check_torch_compile(customop_model: GroupedMLP, batch_size: int) -> bool:
    """Check torch.compile compatibility."""
    num_groups = customop_model.num_groups
    input_dim = customop_model.input_dim

    @torch.compile(fullgraph=True)
    def compiled_forward(model, x):
        return model(x)

    x = torch.randn(
        batch_size * num_groups, input_dim, device="cuda", dtype=torch.bfloat16
    )

    try:
        out = compiled_forward(customop_model, x)
        return out.shape == (batch_size * num_groups, customop_model.output_dim)
    except Exception as e:
        print(f"torch.compile failed: {e}")
        return False


def run_opcheck(customop_model: GroupedMLP):
    """Run opcheck on individual custom ops."""
    print("\nRunning opcheck on custom ops...")

    # Test silu_mul
    examples_silu = [
        [
            torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(32, 64, device="cuda", dtype=torch.bfloat16),
        ],
        [
            torch.randn(
                16, 12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(
                16, 12, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
        ],
    ]
    for i, ex in enumerate(examples_silu):
        try:
            torch.library.opcheck(silu_mul, ex)
            print(f"  silu_mul example {i+1}: PASSED")
        except Exception as e:
            print(f"  silu_mul example {i+1}: FAILED - {e}")

    # Test strided_bmm
    examples_bmm = [
        [
            torch.randn(32, 8, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(8, 64, 128, device="cuda", dtype=torch.bfloat16),
        ],
        [
            torch.randn(
                16, 12, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
            torch.randn(
                12, 256, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
            ),
        ],
    ]
    for i, ex in enumerate(examples_bmm):
        try:
            torch.library.opcheck(strided_bmm, ex)
            print(f"  strided_bmm example {i+1}: PASSED")
        except Exception as e:
            print(f"  strided_bmm example {i+1}: FAILED - {e}")


# =============================================================================
# [6] Benchmark Functions
# =============================================================================


def warmup_gpu():
    """Warmup GPU."""
    x = torch.randn(1000, 1000, device="cuda")
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()


def benchmark_forward(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
) -> float:
    """Benchmark forward pass."""
    for i in range(num_warmup):
        _ = model(x_list[i % len(x_list)])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_iterations):
        _ = model(x_list[i % len(x_list)])
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iterations


def benchmark_forward_backward(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
) -> float:
    """Benchmark forward + backward pass."""
    output_dim = model.output_dim
    grad_outputs = [
        torch.randn(xi.shape[0], output_dim, device="cuda", dtype=xi.dtype)
        for xi in x_list
    ]
    x_with_grad = [xi.requires_grad_(True) for xi in x_list]
    params = list(model.parameters())

    for i in range(num_warmup):
        xi = x_with_grad[i % len(x_list)]
        out = model(xi)
        torch.autograd.grad(out, [xi] + params, grad_outputs[i % len(x_list)])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(num_iterations):
        xi = x_with_grad[i % len(x_list)]
        out = model(xi)
        torch.autograd.grad(out, [xi] + params, grad_outputs[i % len(x_list)])
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iterations


def benchmark_silu_mul_bandwidth(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
    num_iterations: int = 100,
) -> Tuple[float, float, float, float]:
    """
    Benchmark silu_mul kernel bandwidth.

    Returns: (fwd_time_ms, fwd_bw_gb_s, bwd_time_ms, bwd_bw_gb_s)
    """
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn(shape, device="cuda", dtype=dtype)
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)

    numel = gate.numel()
    bytes_per_elem = gate.element_size()
    fwd_bytes = 3 * numel * bytes_per_elem  # read gate, up; write output
    bwd_bytes = (
        5 * numel * bytes_per_elem
    )  # read grad_out, gate, up; write grad_gate, grad_up

    # Warmup
    for _ in range(10):
        _ = silu_mul(gate, up)
        _ = silu_mul_backward(grad_output, gate, up)
    torch.cuda.synchronize()

    # Forward benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        _ = silu_mul(gate, up)
    end.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(end) / num_iterations
    fwd_bw = fwd_bytes / (fwd_ms / 1000) / 1e9

    # Backward benchmark
    start.record()
    for _ in range(num_iterations):
        _ = silu_mul_backward(grad_output, gate, up)
    end.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(end) / num_iterations
    bwd_bw = bwd_bytes / (bwd_ms / 1000) / 1e9

    return fwd_ms, fwd_bw, bwd_ms, bwd_bw


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Grouped MLP as PyTorch Custom Ops")
    print("=" * 70)

    torch.cuda.init()
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")

    # Configuration
    batch_size = 2560
    num_groups = 12
    input_dim = 1024
    hidden_dim = 3072
    output_dim = 1024
    dtype = torch.bfloat16
    num_iterations = 100

    print(
        f"""
Config:
  Batch size:  {batch_size}
  Num groups:  {num_groups}
  Dimensions:  {input_dim} -> {hidden_dim} -> {output_dim}
  Dtype:       {dtype}
"""
    )

    print("Warming up GPU...")
    warmup_gpu()

    # Create models
    print("Creating models...")
    ref_model = ReferenceGroupedMLP(
        input_dim, hidden_dim, output_dim, num_groups, dtype=dtype
    ).cuda()

    customop_model = GroupedMLP(
        input_dim, hidden_dim, output_dim, num_groups, dtype=dtype
    ).cuda()

    # Copy weights
    copy_weights_ref_to_customop(ref_model, customop_model)

    # =========================
    # Correctness Verification
    # =========================
    print("\n" + "-" * 70)
    print("Correctness Verification")
    print("-" * 70)

    fwd_ok, fwd_diff = check_forward_correctness(
        ref_model, customop_model, batch_size, dtype
    )
    print(f"\nForward:  {'✓ PASS' if fwd_ok else '✗ FAIL'}  (max_diff: {fwd_diff:.2e})")

    bwd_ok, bwd_diff = check_backward_correctness(
        ref_model, customop_model, batch_size, dtype
    )
    print(f"Backward: {'✓ PASS' if bwd_ok else '✗ FAIL'}  (max_diff: {bwd_diff:.2e})")

    compile_ok = check_torch_compile(customop_model, batch_size)
    print(f"torch.compile(fullgraph=True): {'✓ PASS' if compile_ok else '✗ FAIL'}")

    # =========================
    # opcheck
    # =========================
    print("\n" + "-" * 70)
    print("Op Registration Check")
    print("-" * 70)
    run_opcheck(customop_model)

    # =========================
    # Performance Benchmark
    # =========================
    print("\n" + "-" * 70)
    print("Performance Benchmark")
    print("-" * 70)

    x_list = [
        torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)
        for _ in range(10)
    ]

    # Forward
    print("\n>>> Forward Pass <<<")
    ref_fwd = benchmark_forward(ref_model, x_list, num_iterations)
    customop_fwd = benchmark_forward(customop_model, x_list, num_iterations)

    print(f"\n{'Model':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 52)
    print(f"{'Reference (loop)':<30} {ref_fwd:<12.4f} {'1.00x':<10}")
    print(
        f"{'CustomOp (BMM)':<30} {customop_fwd:<12.4f} {ref_fwd/customop_fwd:<10.2f}x"
    )

    # Forward + Backward
    print("\n>>> Forward + Backward <<<")
    ref_fwdbwd = benchmark_forward_backward(ref_model, x_list, num_iterations)
    customop_fwdbwd = benchmark_forward_backward(customop_model, x_list, num_iterations)

    print(f"\n{'Model':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 52)
    print(f"{'Reference (loop)':<30} {ref_fwdbwd:<12.4f} {'1.00x':<10}")
    print(
        f"{'CustomOp (BMM)':<30} {customop_fwdbwd:<12.4f} {ref_fwdbwd/customop_fwdbwd:<10.2f}x"
    )

    # =========================
    # SiLU*Up Kernel Bandwidth
    # =========================
    print("\n" + "-" * 70)
    print("Fused SiLU*Up Kernel Bandwidth")
    print("-" * 70)

    shape = (batch_size, num_groups, hidden_dim)
    fwd_ms, fwd_bw, bwd_ms, bwd_bw = benchmark_silu_mul_bandwidth(shape, dtype)

    print(f"\nShape: {shape}")
    print(f"\n{'Kernel':<15} {'Time (ms)':<12} {'BW (GB/s)':<12}")
    print("-" * 40)
    print(f"{'Forward':<15} {fwd_ms:<12.4f} {fwd_bw:<12.1f}")
    print(f"{'Backward':<15} {bwd_ms:<12.4f} {bwd_bw:<12.1f}")

    # =========================
    # Summary
    # =========================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"""
  Forward Speedup:  {ref_fwd/customop_fwd:.2f}x
  Fwd+Bwd Speedup:  {ref_fwdbwd/customop_fwdbwd:.2f}x
  SiLU*Up BW:       {fwd_bw:.1f} GB/s (fwd), {bwd_bw:.1f} GB/s (bwd)
"""
    )
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
