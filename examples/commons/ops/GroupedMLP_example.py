#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Grouped MLP Benchmark: Reference vs Plan A

================================================================================
Problem
================================================================================
Apply num_groups different MLP transformations with GLU gating (SwiGLU/GeGLU):

    For each group n: y[b, n, :] = down(act(gate(x)) * up(x))

================================================================================
Implementations
================================================================================
Reference: Loop over groups with separate nn.Linear layers
Plan A:    3 independent strided BMMs (gate, up, down)

================================================================================
"""
import sys
sys.path.insert(0, '/home/scratch.runchuz_gpu/repos-github/recsys-examples/examples/hstu')

import argparse
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
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

from ops.triton_ops.common import triton_autotune

def silu_configs():
    configs = []
    for x_block_size in [256, 512, 1024, 2048]:
        for num_warps in [2, 4, 8, 16]:
            config = triton.Config({"x_block_size": x_block_size}, num_warps)
            configs.append(config)
    return configs





# =============================================================================
# Fused SiLU * Up (SwiGLU pattern): output = silu(gate) * up
# =============================================================================

@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_mul_forward(
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

    # silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
    silu_gate = fast_dividef(gate, 1.0 + tl.exp(-gate))
    output = (silu_gate * up).to(output_ptr.dtype.element_ty)

    tl.store(output_ptr + x_offset + cols, output, mask=mask)


@triton_autotune(silu_configs(), key=["x_size"])
@triton.jit
def _silu_mul_backward(
    grad_gate_ptr: tl.tensor,
    grad_up_ptr: tl.tensor,
    grad_output_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    up_ptr: tl.tensor,
    x_size: tl.int32,
    x_block_size: tl.constexpr,
):
    """
    Fused backward for output = silu(gate) * up
    
    grad_gate = grad_output * up * d(silu)/d(gate)
              = grad_output * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
    grad_up   = grad_output * silu(gate)
    """
    x_offset = tl.program_id(0) * x_block_size
    mask = x_offset + tl.arange(0, x_block_size) < x_size
    cols = tl.arange(0, x_block_size)

    grad_output = tl.load(grad_output_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)

    sigma = tl.sigmoid(gate)
    silu_gate = gate * sigma
    
    # d(silu)/d(gate) = sigma + gate * sigma * (1 - sigma)
    dsilu_dgate = sigma + gate * sigma * (1.0 - sigma)
    
    grad_gate = grad_output * up * dsilu_dgate
    grad_up = grad_output * silu_gate

    tl.store(grad_gate_ptr + x_offset + cols, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_up_ptr + x_offset + cols, grad_up.to(grad_up_ptr.dtype.element_ty), mask=mask)


def triton_silu_mul_fwd(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Forward: output = silu(gate) * up"""
    assert gate.shape == up.shape, f"Shape mismatch: gate {gate.shape} vs up {up.shape}"
    x_size = gate.numel()
    gate_1d = gate.view(-1).contiguous()
    up_1d = up.view(-1).contiguous()
    output = torch.empty_like(gate_1d)

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_mul_forward[grid](
        output,
        gate_1d,
        up_1d,
        x_size,
    )
    return output.view(gate.shape)


def triton_silu_mul_bwd(
    grad_output: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward: returns (grad_gate, grad_up)"""
    shape = gate.shape
    x_size = gate.numel()
    gate_1d = gate.view(-1).contiguous()
    up_1d = up.view(-1).contiguous()
    grad_output_1d = grad_output.view(-1).contiguous()
    grad_gate = torch.empty_like(gate_1d)
    grad_up = torch.empty_like(up_1d)

    def grid(meta):
        return (triton.cdiv(x_size, meta["x_block_size"]),)

    _silu_mul_backward[grid](
        grad_gate,
        grad_up,
        grad_output_1d,
        gate_1d,
        up_1d,
        x_size,
    )
    return grad_gate.view(shape), grad_up.view(shape)


class TritonSiluMul(torch.autograd.Function):
    """Autograd function for fused silu(gate) * up"""
    
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        output = triton_silu_mul_fwd(gate, up)
        ctx.save_for_backward(gate, up)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        grad_gate, grad_up = triton_silu_mul_bwd(grad_output, gate, up)
        return grad_gate, grad_up


def triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SiLU multiplication (SwiGLU pattern).
    
    Computes: output = silu(gate) * up
    
    Args:
        gate: Input tensor that goes through SiLU activation
        up: Input tensor that multiplies with activated gate
        
    Returns:
        output: silu(gate) * up
    """
    gate = gate.contiguous()
    up = up.contiguous()
    return TritonSiluMul.apply(gate, up)


def warmup_gpu():
    """Warmup GPU to get stable timing."""
    x = torch.randn(1000, 1000, device="cuda")
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()


def get_activation_fn(activation: Optional[str]) -> Optional[Callable]:
    """Get activation function by name."""
    if activation is None:
        return None
    activation_map = {
        "silu": F.silu,
        "swish": F.silu,
        "gelu": F.gelu,
        "relu": F.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "swiglu": triton_silu_mul,
    }
    if activation.lower() not in activation_map:
        raise ValueError(f"Unknown activation: {activation}")
    return activation_map[activation.lower()]


# =============================================================================
# Reference Implementation
# =============================================================================

class ReferenceGroupedMLP(nn.Module):
    """Reference implementation using loop over groups."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_groups: int,
        use_gating: bool = True,
        activation: Optional[str] = "swiglu",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gating = use_gating
        self.act_fn = get_activation_fn(activation)

        if use_gating:
            self.gate_proj = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ])
            self.up_proj = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ])
        else:
            self.proj = nn.ModuleList([
                nn.Linear(input_dim, hidden_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ])

        self.down_proj = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim, bias=False, device=device, dtype=dtype)
            for _ in range(num_groups)
        ])

        self._init_weights()

    def _init_weights(self):
        for module_list in [getattr(self, 'gate_proj', []),
                           getattr(self, 'up_proj', []),
                           getattr(self, 'proj', []),
                           self.down_proj]:
            for layer in module_list:
                nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x: torch.Tensor, enable_nvtx: bool = False) -> torch.Tensor:
        if enable_nvtx:
            with nvtx.range("Ref_reshape"):
                x = x.reshape(-1, self.num_groups, self.input_dim)
            
            with nvtx.range("Ref_split"):
                x_split = torch.split(x, 1, dim=1)
            
            with nvtx.range("Ref_loop_gemm"):
                out_list = []
                for i in range(self.num_groups):
                    x_i = x_split[i].squeeze(1)
                    if self.use_gating:
                        gate_i = self.gate_proj[i](x_i)
                        up_i = self.up_proj[i](x_i)
                        if self.act_fn is not None:
                            hidden_i = self.act_fn(gate_i) * up_i
                        else:
                            hidden_i = gate_i * up_i
                    else:
                        hidden_i = self.proj[i](x_i)
                        if self.act_fn is not None:
                            hidden_i = self.act_fn(hidden_i)
                    out_i = self.down_proj[i](hidden_i)
                    out_list.append(out_i)
            
            with nvtx.range("Ref_stack"):
                output = torch.stack(out_list, dim=1).reshape(-1, self.output_dim)
        else:
            x = x.reshape(-1, self.num_groups, self.input_dim)
            x_split = torch.split(x, 1, dim=1)
            
            out_list = []
            for i in range(self.num_groups):
                x_i = x_split[i].squeeze(1)
                if self.use_gating:
                    gate_i = self.gate_proj[i](x_i)
                    up_i = self.up_proj[i](x_i)
                    if self.act_fn is not None:
                        hidden_i = self.act_fn(gate_i) * up_i
                    else:
                        hidden_i = gate_i * up_i
                else:
                    hidden_i = self.proj[i](x_i)
                    if self.act_fn is not None:
                        hidden_i = self.act_fn(hidden_i)
                out_i = self.down_proj[i](hidden_i)
                out_list.append(out_i)
            
            output = torch.stack(out_list, dim=1).reshape(-1, self.output_dim)
        
        return output

    def forward_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to hidden (excluding down projection)."""
        x = x.reshape(-1, self.num_groups, self.input_dim)
        x_split = torch.split(x, 1, dim=1)
        
        hidden_list = []
        for i in range(self.num_groups):
            x_i = x_split[i].squeeze(1)
            if self.use_gating:
                gate_i = self.gate_proj[i](x_i)
                up_i = self.up_proj[i](x_i)
                if self.act_fn is not None:
                    hidden_i = self.act_fn(gate_i) * up_i
                else:
                    hidden_i = gate_i * up_i
            else:
                hidden_i = self.proj[i](x_i)
                if self.act_fn is not None:
                    hidden_i = self.act_fn(hidden_i)
            hidden_list.append(hidden_i)
        
        return torch.stack(hidden_list, dim=1).reshape(-1, self.hidden_dim)


# =============================================================================
# Strided BMM Function
# =============================================================================

class StridedBmmFunction(torch.autograd.Function):
    """Custom autograd function for BMM with strided output."""

    @staticmethod
    def forward(ctx, x, weight, batch_size, num_groups, output_dim):
        ctx.save_for_backward(x, weight)
        
        output = torch.empty(batch_size, num_groups, output_dim,
                            device=x.device, dtype=x.dtype)
        torch.bmm(x.permute(1, 0, 2), weight, out=output.permute(1, 0, 2))
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = None
        
        grad_output_t = grad_output.permute(1, 0, 2)
        
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            torch.bmm(grad_output_t, weight.transpose(-1, -2), out=grad_x.permute(1, 0, 2))
        
        if ctx.needs_input_grad[1]:
            x_t = x.permute(1, 0, 2)
            grad_weight = torch.bmm(x_t.transpose(-1, -2), grad_output_t)
        
        return grad_x, grad_weight, None, None, None


# =============================================================================
# Plan A: 3 Independent BMMs
# =============================================================================

class GroupedMLP_PlanA(nn.Module):
    """Plan A: 3 independent strided BMMs (gate, up, down)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_groups: int,
        use_gating: bool = True,
        activation: Optional[str] = "silu",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gating = use_gating
        self.act_fn = get_activation_fn(activation)

        if use_gating:
            self.gate_weight = nn.Parameter(
                torch.empty(num_groups, input_dim, hidden_dim, device=device, dtype=dtype)
            )
            self.up_weight = nn.Parameter(
                torch.empty(num_groups, input_dim, hidden_dim, device=device, dtype=dtype)
            )
        else:
            self.proj_weight = nn.Parameter(
                torch.empty(num_groups, input_dim, hidden_dim, device=device, dtype=dtype)
            )

        self.down_weight = nn.Parameter(
            torch.empty(num_groups, hidden_dim, output_dim, device=device, dtype=dtype)
        )

        self._init_weights()

    def _init_weights(self):
        for i in range(self.num_groups):
            if self.use_gating:
                nn.init.xavier_normal_(self.gate_weight[i], gain=1.0)
                nn.init.xavier_normal_(self.up_weight[i], gain=1.0)
            else:
                nn.init.xavier_normal_(self.proj_weight[i], gain=1.0)
            nn.init.xavier_normal_(self.down_weight[i], gain=1.0)

    def forward(self, x: torch.Tensor, enable_nvtx: bool = False) -> torch.Tensor:
        batch_size = x.shape[0] // self.num_groups
        
        if enable_nvtx:
            with nvtx.range("PlanA_reshape"):
                x = x.reshape(batch_size, self.num_groups, self.input_dim)
            
            if self.use_gating:
                with nvtx.range("PlanA_gate_bmm"):
                    gate = StridedBmmFunction.apply(
                        x, self.gate_weight, batch_size, self.num_groups, self.hidden_dim
                    )
                
                with nvtx.range("PlanA_up_bmm"):
                    up = StridedBmmFunction.apply(
                        x, self.up_weight, batch_size, self.num_groups, self.hidden_dim
                    )
                
                with nvtx.range("PlanA_activation"):
                    if self.act_fn is not None:
                        hidden = self.act_fn(gate) * up
                    else:
                        hidden = gate * up
            else:
                with nvtx.range("PlanA_proj_bmm"):
                    hidden = StridedBmmFunction.apply(
                        x, self.proj_weight, batch_size, self.num_groups, self.hidden_dim
                    )
                with nvtx.range("PlanA_activation"):
                    if self.act_fn is not None:
                        hidden = self.act_fn(hidden)
            
            with nvtx.range("PlanA_down_bmm"):
                output = StridedBmmFunction.apply(
                    hidden, self.down_weight, batch_size, self.num_groups, self.output_dim
                )
            
            with nvtx.range("PlanA_view"):
                return output.view(-1, self.output_dim)
        else:
            x = x.reshape(batch_size, self.num_groups, self.input_dim)
            
            if self.use_gating:
                gate = StridedBmmFunction.apply(
                    x, self.gate_weight, batch_size, self.num_groups, self.hidden_dim
                )
                up = StridedBmmFunction.apply(
                    x, self.up_weight, batch_size, self.num_groups, self.hidden_dim
                )
                if self.act_fn is not None:
                    # hidden = self.act_fn(gate) * up
                    hidden = triton_silu_mul(gate, up)
                else:
                    hidden = gate * up
            else:
                hidden = StridedBmmFunction.apply(
                    x, self.proj_weight, batch_size, self.num_groups, self.hidden_dim
                )
                if self.act_fn is not None:
                    hidden = self.act_fn(hidden)
            
            output = StridedBmmFunction.apply(
                hidden, self.down_weight, batch_size, self.num_groups, self.output_dim
            )
            
            return output.view(-1, self.output_dim)

    def forward_to_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to hidden (excluding down projection)."""
        batch_size = x.shape[0] // self.num_groups
        x = x.reshape(batch_size, self.num_groups, self.input_dim)
        
        if self.use_gating:
            gate = StridedBmmFunction.apply(
                x, self.gate_weight, batch_size, self.num_groups, self.hidden_dim
            )
            up = StridedBmmFunction.apply(
                x, self.up_weight, batch_size, self.num_groups, self.hidden_dim
            )
            if self.act_fn is not None:
                hidden = triton_silu_mul(gate, up)
            else:
                hidden = gate * up
        else:
            hidden = StridedBmmFunction.apply(
                x, self.proj_weight, batch_size, self.num_groups, self.hidden_dim
            )
            if self.act_fn is not None:
                hidden = self.act_fn(hidden)
        
        return hidden.view(-1, self.hidden_dim)


# =============================================================================
# Weight Copy Utilities
# =============================================================================

def copy_weights_to_plan_a(ref_model: ReferenceGroupedMLP, opt_model: GroupedMLP_PlanA):
    with torch.no_grad():
        num_groups = ref_model.num_groups
        if ref_model.use_gating:
            for i in range(num_groups):
                opt_model.gate_weight[i].copy_(ref_model.gate_proj[i].weight.T)
                opt_model.up_weight[i].copy_(ref_model.up_proj[i].weight.T)
        else:
            for i in range(num_groups):
                opt_model.proj_weight[i].copy_(ref_model.proj[i].weight.T)
        for i in range(num_groups):
            opt_model.down_weight[i].copy_(ref_model.down_proj[i].weight.T)


# =============================================================================
# Correctness Check
# =============================================================================

def check_correctness(
    ref_model: nn.Module,
    opt_model: nn.Module,
    batch_size: int,
    num_groups: int,
    input_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float]:
    """Check forward and backward correctness."""
    x = torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)
    with torch.no_grad():
        ref_out = ref_model(x)
        opt_out = opt_model(x)
    fwd_diff = (ref_out - opt_out).abs().max().item()

    x_ref = torch.randn(
        batch_size * num_groups, input_dim,
        device="cuda", dtype=dtype, requires_grad=True
    )
    x_opt = x_ref.detach().clone().requires_grad_(True)

    ref_out = ref_model(x_ref)
    opt_out = opt_model(x_opt)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)
    opt_out.backward(grad_output)

    bwd_x_diff = (x_ref.grad - x_opt.grad).abs().max().item()

    return fwd_diff, bwd_x_diff


# =============================================================================
# Benchmark Functions (same as benchmark_batched_gemm.py)
# =============================================================================

def benchmark_forward(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
    enable_nvtx: bool = False,
) -> float:
    """Benchmark forward pass using CUDA events for accurate GPU timing."""
    model_name = model.__class__.__name__
    
    # Warmup
    for i in range(num_warmup):
        _ = model(x_list[i % len(x_list)], enable_nvtx=False)
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Benchmark
    start_event.record()
    if enable_nvtx:
        for i in range(num_iterations):
            with nvtx.range(f"{model_name}_fwd_iter{i}"):
                _ = model(x_list[i % len(x_list)], enable_nvtx=True)
    else:
        for i in range(num_iterations):
            _ = model(x_list[i % len(x_list)], enable_nvtx=False)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / num_iterations


def benchmark_forward_to_hidden(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
) -> float:
    """Benchmark forward pass up to hidden (excluding down projection)."""
    # Warmup
    for i in range(num_warmup):
        _ = model.forward_to_hidden(x_list[i % len(x_list)])
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Benchmark
    start_event.record()
    for i in range(num_iterations):
        _ = model.forward_to_hidden(x_list[i % len(x_list)])
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / num_iterations


def benchmark_forward_backward(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
    enable_nvtx: bool = False,
) -> float:
    """Benchmark forward + backward pass using CUDA events."""
    model_name = model.__class__.__name__
    output_dim = model.output_dim
    
    grad_outputs = [
        torch.randn(xi.shape[0], output_dim, device="cuda", dtype=xi.dtype)
        for xi in x_list
    ]
    
    x_with_grad = [xi.requires_grad_(True) for xi in x_list]
    params = list(model.parameters())
    
    # Warmup
    for i in range(num_warmup):
        xi = x_with_grad[i % len(x_list)]
        out = model(xi, enable_nvtx=False)
        grads = torch.autograd.grad(
            outputs=out,
            inputs=[xi] + params,
            grad_outputs=grad_outputs[i % len(x_list)],
        )
    torch.cuda.synchronize()
    
    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Benchmark
    start_event.record()
    if enable_nvtx:
        for i in range(num_iterations):
            with nvtx.range(f"{model_name}_fwdbwd_iter{i}"):
                xi = x_with_grad[i % len(x_list)]
                grad_out = grad_outputs[i % len(x_list)]
                with nvtx.range("forward"):
                    out = model(xi, enable_nvtx=True)
                # Separate backward into two parts for clearer profiling
                with nvtx.range("backward_activation"):
                    # dL/dx (activation gradient)
                    grad_x = torch.autograd.grad(
                        outputs=out,
                        inputs=xi,
                        grad_outputs=grad_out,
                        retain_graph=True,  # Keep graph for weight gradient
                    )
                with nvtx.range("backward_weight"):
                    # dL/dW (weight gradient)
                    grad_w = torch.autograd.grad(
                        outputs=out,
                        inputs=params,
                        grad_outputs=grad_out,
                        retain_graph=False,
                    )
    else:
        for i in range(num_iterations):
            xi = x_with_grad[i % len(x_list)]
            out = model(xi, enable_nvtx=False)
            grads = torch.autograd.grad(
                outputs=out,
                inputs=[xi] + params,
                grad_outputs=grad_outputs[i % len(x_list)],
            )
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / num_iterations


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Grouped MLP Benchmark: Reference vs Plan A"
    )
    parser.add_argument("--batch-size", type=int, default=2560)
    parser.add_argument("--num-groups", type=int, default=12)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=3072)
    parser.add_argument("--output-dim", type=int, default=1024)
    parser.add_argument("--activation", type=str, default="silu",
                       choices=["silu", "gelu", "relu", "tanh", "none"])
    parser.add_argument("--no-gating", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--enable-nvtx", action="store_true",
                       help="Enable NVTX markers (use with nsys profile)")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile() to optimize models")
    args = parser.parse_args()

    torch.cuda.init()

    batch_size = args.batch_size
    num_groups = args.num_groups
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    activation = None if args.activation == "none" else args.activation
    use_gating = not args.no_gating
    dtype = torch.bfloat16
    num_iterations = args.iterations

    print("=" * 80)
    print("Grouped MLP Benchmark: Reference vs Plan A")
    print("=" * 80)
    
    if args.enable_nvtx:
        print("\n*** NVTX PROFILING MODE ***")
        print("Run with: nsys profile -o <output> --trace=cuda,nvtx python ...")
    
    print(f"""
Config:
  Batch size:  {batch_size}
  Num groups:  {num_groups}
  Dimensions:  {input_dim} -> {hidden_dim} -> {output_dim}
  Mode:        {"GLU (SwiGLU)" if use_gating else "Simple MLP"}
  Activation:  {activation if activation else "None"}
  Dtype:       {dtype}
  Device:      {torch.cuda.get_device_name(0)}
  Iterations:  {num_iterations}
""")

    print("Warming up GPU...")
    warmup_gpu()

    # Create models
    print("Creating models...")
    ref_model = ReferenceGroupedMLP(
        input_dim, hidden_dim, output_dim, num_groups,
        use_gating=use_gating, activation=activation, dtype=dtype
    ).cuda()

    plan_a_model = GroupedMLP_PlanA(
        input_dim, hidden_dim, output_dim, num_groups,
        use_gating=use_gating, activation=activation, dtype=dtype
    ).cuda()

    copy_weights_to_plan_a(ref_model, plan_a_model)

    # Apply torch.compile() if requested
    if args.compile:
        print("\nApplying torch.compile() to all models...")
        ref_model = torch.compile(ref_model)
        plan_a_model = torch.compile(plan_a_model)
        print("Compilation complete (will JIT compile on first run).")

    # Correctness check
    print("-" * 60)
    print("Correctness Check")
    print("-" * 60)
    
    fwd_a, bwd_a = check_correctness(ref_model, plan_a_model, batch_size, num_groups, input_dim, dtype)
    print(f"Plan A - Forward diff: {fwd_a:.2e}, Backward diff: {bwd_a:.2e}")

    # Prepare test data
    x_list = [
        torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)
        for _ in range(10)
    ]

    # Benchmark (NEVER use NVTX for timing - NVTX adds Python overhead)
    print("\n" + "-" * 60)
    print("Performance Benchmark (NVTX disabled for accurate timing)")
    print("-" * 60)

    # Forward - always benchmark without NVTX
    print("\n>>> Forward Pass <<<")
    ref_fwd = benchmark_forward(ref_model, x_list, num_iterations, enable_nvtx=False)
    plan_a_fwd = benchmark_forward(plan_a_model, x_list, num_iterations, enable_nvtx=False)

    print(f"\n{'Model':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 52)
    print(f"{'Reference (loop)':<30} {ref_fwd:<12.4f} {'1.00x':<10}")
    print(f"{'Plan A (batched BMM)':<30} {plan_a_fwd:<12.4f} {ref_fwd/plan_a_fwd:<10.2f}x")

    # Forward to Hidden (excluding down projection)
    print("\n>>> Forward to Hidden (excluding down_proj) <<<")
    ref_hidden = benchmark_forward_to_hidden(ref_model, x_list, num_iterations)
    plan_a_hidden = benchmark_forward_to_hidden(plan_a_model, x_list, num_iterations)

    print(f"\n{'Model':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 52)
    print(f"{'Reference (loop)':<30} {ref_hidden:<12.4f} {'1.00x':<10}")
    print(f"{'Plan A (batched BMM)':<30} {plan_a_hidden:<12.4f} {ref_hidden/plan_a_hidden:<10.2f}x")

    # Forward + Backward - always benchmark without NVTX
    print("\n>>> Forward + Backward <<<")
    ref_fwdbwd = benchmark_forward_backward(ref_model, x_list, num_iterations, enable_nvtx=False)
    plan_a_fwdbwd = benchmark_forward_backward(plan_a_model, x_list, num_iterations, enable_nvtx=False)

    print(f"\n{'Model':<30} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 52)
    print(f"{'Reference (loop)':<30} {ref_fwdbwd:<12.4f} {'1.00x':<10}")
    print(f"{'Plan A (batched BMM)':<30} {plan_a_fwdbwd:<12.4f} {ref_fwdbwd/plan_a_fwdbwd:<10.2f}x")

    # NVTX profiling run (separate from benchmark)
    if args.enable_nvtx:
        print("\n" + "-" * 60)
        print("NVTX Profiling Run (for nsys analysis only)")
        print("-" * 60)
        torch.cuda.profiler.start()
        
        # Run a few iterations with NVTX for profiling
        nvtx_iterations = min(10, num_iterations)
        _ = benchmark_forward(ref_model, x_list, nvtx_iterations, enable_nvtx=True)
        _ = benchmark_forward(plan_a_model, x_list, nvtx_iterations, enable_nvtx=True)
        _ = benchmark_forward_backward(ref_model, x_list, nvtx_iterations, enable_nvtx=True)
        _ = benchmark_forward_backward(plan_a_model, x_list, nvtx_iterations, enable_nvtx=True)
        
        torch.cuda.profiler.stop()
        print("NVTX profiling complete.")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"""
Implementation Details:
  Reference:  Loop over {num_groups} groups, uses nn.Linear (C++ autograd)
  Plan A:     Batched BMM with custom StridedBmmFunction

Forward Speedup (full MLP):
  Plan A vs Reference: {ref_fwd/plan_a_fwd:.2f}x

Forward to Hidden (excluding down_proj):
  Plan A vs Reference: {ref_hidden/plan_a_hidden:.2f}x

Fwd+Bwd Speedup:
  Plan A vs Reference: {ref_fwdbwd/plan_a_fwdbwd:.2f}x
""")
    print("=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
