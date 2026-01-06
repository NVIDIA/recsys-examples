#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Grouped Linear Layer with Strided BMM Optimization

================================================================================
Problem
================================================================================
Apply num_groups different linear transformations to corresponding slices of input:

    Input:  x of shape (B * num_groups, input_dim)
    Output: y of shape (B * num_groups, output_dim)
    
    For each group n: y[b, n, :] = x[b, n, :] @ W[n, :, :]

================================================================================
Reference Implementation
================================================================================
The straightforward approach uses a loop over groups:

    x = x.reshape(B, num_groups, D_in)
    x_split = torch.split(x, 1, dim=1)
    
    out_list = []
    for i in range(num_groups):
        x_i = x_split[i].squeeze(1)           # (B, D_in)
        out_i = linear_layers[i](x_i)         # (B, D_out)
        out_list.append(out_i)
    
    output = torch.stack(out_list, dim=1).reshape(-1, D_out)

================================================================================
Optimized Implementation
================================================================================
Use torch.bmm with strided output to fuse all GEMMs into one kernel:

    x = x.reshape(B, num_groups, D_in)
    output = torch.empty(B, num_groups, D_out, ...)   # pre-allocate final layout
    torch.bmm(x.permute(1,0,2), weight,
              out=output.permute(1,0,2))              # cuBLAS writes to strided memory
    return output.view(-1, D_out)                     # O(1) view, no copy.

Key feature: cuBLAS strided batched GEMM supports strided output via ldc/strideC
parameters, allowing direct write to the transposed memory layout.

================================================================================
Performance Results
================================================================================
Config: batch_size=2560, num_groups=12, input_dim=1024, output_dim=3072, dtype=bf16
Device: NVIDIA H100
BMM_Opt Forward:         1.46x
BMM_Opt Forward+Backward:1.41x

================================================================================
"""

import argparse
from typing import List, Tuple

import torch
import torch.nn as nn


def warmup_gpu():
    """Warmup GPU to get stable timing"""
    x = torch.randn(1000, 1000, device="cuda")
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()


class ReferenceImpl(nn.Module):
    """
    Reference implementation using reshape + split + loop + stack.
    Simple but slow due to multiple kernel launches.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_groups: int,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, output_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_groups)
            ]
        )

        for layer in self.linear_layers:
            nn.init.xavier_normal_(layer.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape: (B*ns, D) -> (B, ns, D)
        x = x.reshape(-1, self.num_groups, self.input_dim)

        # split and loop
        x_split = torch.split(x, 1, dim=1)
        out_list = []
        for i in range(self.num_groups):
            x_i = x_split[i].squeeze(1)  # (B, D)
            out_i = self.linear_layers[i](x_i)  # (B, D_out)
            out_list.append(out_i)

        # stack: ns * (B, D_out) -> (B, ns, D_out) -> (B*ns, D_out)
        return torch.stack(out_list, dim=1).reshape(-1, self.output_dim)


class StridedBmmFunction(torch.autograd.Function):
    """Custom autograd function for BMM with strided output."""

    @staticmethod
    def forward(ctx, x, weight, batch_size, num_groups, output_dim):
        ctx.save_for_backward(x, weight)
        output = torch.empty(
            batch_size, num_groups, output_dim, device=x.device, dtype=x.dtype
        )
        torch.bmm(x.permute(1, 0, 2), weight, out=output.permute(1, 0, 2))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = None
        grad_output_t = grad_output.permute(1, 0, 2)

        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            torch.bmm(
                grad_output_t, weight.transpose(-1, -2), out=grad_x.permute(1, 0, 2)
            )

        if ctx.needs_input_grad[1]:
            grad_weight = torch.bmm(x.permute(1, 0, 2).transpose(-1, -2), grad_output_t)

        return grad_x, grad_weight, None, None, None


class BmmImpl(nn.Module):
    """
    Optimized implementation using strided BMM.
    Single kernel launch with fused permute via strided output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_groups: int,
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = nn.Parameter(
            torch.empty(num_groups, input_dim, output_dim, device=device, dtype=dtype)
        )
        for i in range(num_groups):
            nn.init.xavier_normal_(self.weight[i], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] // self.num_groups
        x = x.reshape(batch_size, self.num_groups, self.input_dim)
        output = StridedBmmFunction.apply(
            x, self.weight, batch_size, self.num_groups, self.output_dim
        )
        return output.view(-1, self.output_dim)


def copy_weights(ref_model: ReferenceImpl, opt_model: BmmImpl):
    """Copy weights from reference to optimized model."""
    with torch.no_grad():
        for i in range(ref_model.num_groups):
            opt_model.weight[i].copy_(ref_model.linear_layers[i].weight.T)


def check_correctness(
    ref_model: ReferenceImpl,
    opt_model: BmmImpl,
    batch_size: int,
    num_groups: int,
    input_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[float, float, float]:
    """Check forward and backward correctness."""
    # Forward check
    x = torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)
    with torch.no_grad():
        ref_out = ref_model(x)
        opt_out = opt_model(x)
    fwd_diff = (ref_out - opt_out).abs().max().item()

    # Backward check
    x_ref = torch.randn(
        batch_size * num_groups,
        input_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    x_opt = x_ref.detach().clone().requires_grad_(True)

    ref_out = ref_model(x_ref)
    opt_out = opt_model(x_opt)

    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)
    opt_out.backward(grad_output)

    # Input gradient
    bwd_x_diff = (x_ref.grad - x_opt.grad).abs().max().item()

    # Weight gradient
    ref_weight_grad = torch.stack(
        [ref_model.linear_layers[i].weight.grad.T for i in range(num_groups)]
    )
    bwd_w_diff = (ref_weight_grad - opt_model.weight.grad).abs().max().item()

    return fwd_diff, bwd_x_diff, bwd_w_diff


def benchmark(
    model: nn.Module,
    x_list: List[torch.Tensor],
    num_iterations: int = 100,
    num_warmup: int = 10,
    with_backward: bool = False,
) -> float:
    """Benchmark forward or forward+backward pass using CUDA events for accurate GPU timing."""
    if with_backward:
        x_list = [xi.requires_grad_(True) for xi in x_list]
        grad_outputs = [
            torch.randn(xi.shape[0], model.output_dim, device="cuda", dtype=xi.dtype)
            for xi in x_list
        ]
        params = list(model.parameters())

    # Warmup
    for i in range(num_warmup):
        xi = x_list[i % len(x_list)]
        out = model(xi)
        if with_backward:
            torch.autograd.grad(out, [xi] + params, grad_outputs[i % len(x_list)])
    torch.cuda.synchronize()

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark
    start_event.record()
    for i in range(num_iterations):
        xi = x_list[i % len(x_list)]
        out = model(xi)
        if with_backward:
            torch.autograd.grad(out, [xi] + params, grad_outputs[i % len(x_list)])
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_iterations  # ms


def main():
    parser = argparse.ArgumentParser(
        description="Grouped GEMM: Reference vs Strided BMM"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2560, help="Batch size to test"
    )
    parser.add_argument("--num-groups", type=int, default=12, help="Number of groups")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension")
    parser.add_argument("--output-dim", type=int, default=3072, help="Output dimension")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations for timing"
    )
    args = parser.parse_args()

    torch.cuda.init()

    # Configuration from args
    batch_size = args.batch_size
    num_groups = args.num_groups
    input_dim = args.input_dim
    output_dim = args.output_dim
    dtype = torch.bfloat16
    num_iterations = args.iterations

    print("=" * 60)
    print("Grouped GEMM: Reference vs Strided BMM")
    print("=" * 60)
    print(
        f"\nConfig: B={batch_size}, groups={num_groups}, D_in={input_dim}, D_out={output_dim}"
    )
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Warmup GPU
    print("\nWarming up GPU...")
    warmup_gpu()

    # Create models
    ref_model = ReferenceImpl(input_dim, output_dim, num_groups, dtype=dtype).cuda()
    opt_model = BmmImpl(input_dim, output_dim, num_groups, dtype=dtype).cuda()
    copy_weights(ref_model, opt_model)

    # Correctness check
    print("\n" + "-" * 40)
    print("Correctness Check")
    print("-" * 40)
    fwd_diff, bwd_x_diff, bwd_w_diff = check_correctness(
        ref_model, opt_model, batch_size, num_groups, input_dim, dtype
    )
    print(f"Forward max diff:    {fwd_diff:.2e} {'✓' if fwd_diff < 1e-3 else '✗'}")
    print(f"Backward dL/dx diff: {bwd_x_diff:.2e} {'✓' if bwd_x_diff < 1e-3 else '✗'}")
    print(f"Backward dL/dW diff: {bwd_w_diff:.2e} {'✓' if bwd_w_diff < 1e-3 else '✗'}")

    # Benchmark
    print("\n" + "-" * 40)
    print("Performance Benchmark")
    print("-" * 40)

    x_list = [
        torch.randn(batch_size * num_groups, input_dim, device="cuda", dtype=dtype)
        for _ in range(10)
    ]

    # Forward only
    ref_fwd = benchmark(ref_model, x_list, num_iterations, with_backward=False)
    opt_fwd = benchmark(opt_model, x_list, num_iterations, with_backward=False)

    print(f"\nForward pass (ms):")
    print(f"  Reference (loop): {ref_fwd:.4f}")
    print(f"  Optimized (BMM):  {opt_fwd:.4f}")
    print(f"  Speedup:          {ref_fwd/opt_fwd:.2f}x")

    # Forward + Backward
    ref_fwdbwd = benchmark(ref_model, x_list, num_iterations, with_backward=True)
    opt_fwdbwd = benchmark(opt_model, x_list, num_iterations, with_backward=True)

    print(f"\nForward + Backward (ms):")
    print(f"  Reference (loop): {ref_fwdbwd:.4f}")
    print(f"  Optimized (BMM):  {opt_fwdbwd:.4f}")
    print(f"  Speedup:          {ref_fwdbwd/opt_fwdbwd:.2f}x")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
