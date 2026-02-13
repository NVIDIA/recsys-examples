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

"""
Performance utilities for GPU device specifications and MFU calculation.
"""

import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch


@dataclass
class DeviceSpec:
    """GPU device specification."""

    # Device identification
    device_index: int
    device_name: str
    compute_capability: Tuple[int, int]  # (major, minor)
    architecture: str  # e.g., "Hopper", "Ampere", "Ada Lovelace"

    # Compute specs
    num_sms: int
    gpu_clock_mhz: float  # Max GPU clock in MHz

    # Memory specs
    memory_total_gb: float
    memory_bandwidth_gb_s: float  # GB/s

    # Fields with default values must come after fields without defaults
    # Peak TFLOPS for different data types (Tensor Core operations)
    # Keys: "fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8"
    peak_tflops: Dict[str, float] = field(default_factory=dict)

    # Network specs (if available)
    nvlink_bandwidth_gb_s: Optional[float] = None
    ib_bandwidth_gb_s: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            f"Device: {self.device_name} (index: {self.device_index})",
            f"Architecture: {self.architecture} (SM {self.compute_capability[0]}{self.compute_capability[1]})",
            f"SMs: {self.num_sms}, Clock: {self.gpu_clock_mhz:.0f} MHz",
            f"Memory: {self.memory_total_gb:.1f} GB, Bandwidth: {self.memory_bandwidth_gb_s:.0f} GB/s",
            "Peak TFLOPS (Tensor Core):",
        ]
        for dtype, tflops in sorted(self.peak_tflops.items()):
            lines.append(f"  {dtype}: {tflops:.1f} TFLOPS")
        if self.nvlink_bandwidth_gb_s:
            lines.append(f"NVLink Bandwidth: {self.nvlink_bandwidth_gb_s:.0f} GB/s")
        if self.ib_bandwidth_gb_s:
            lines.append(f"InfiniBand Bandwidth: {self.ib_bandwidth_gb_s:.0f} GB/s")
        return "\n".join(lines)


# GPU architecture mapping based on compute capability
_ARCH_NAMES = {
    (7, 0): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada Lovelace",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
}

# Tensor Core FMA operations per SM per cycle for different architectures.
# The formula to compute peak dense TFLOPS is:
#   tflops = num_sms * clock_mhz * 1e6 * fma_per_sm_per_cycle * 2 / 1e12
# where the *2 accounts for FMA counting as 2 FLOPS (multiply + add).
#
# Values are calibrated against NVIDIA datasheet *dense* (without 2:4
# structured sparsity) peak TFLOPS at the reference boost clock.
# Reference: NVIDIA Architecture Whitepapers & Datasheets
_TC_FLOPS_PER_SM_PER_CYCLE = {
    # Ampere (A100): 108 SMs, ~1410 MHz boost
    # Dense peaks: FP64 TC 19.5 TF, TF32 156 TF, BF16/FP16 312 TF, INT8 624 TF
    (8, 0): {
        "fp64": 64,  # FP64 Tensor Core
        "fp32": 512,  # TF32 (actually uses TF32)
        "tf32": 512,
        "fp16": 1024,
        "bf16": 1024,
        "int8": 2048,
    },
    # Ampere (A10, A30, etc.): fewer TCs per SM than A100
    (8, 6): {
        "fp32": 256,
        "tf32": 256,
        "fp16": 512,
        "bf16": 512,
        "int8": 1024,
    },
    # Ada Lovelace (L40, RTX 4090)
    (8, 9): {
        "fp32": 256,
        "tf32": 256,
        "fp16": 512,
        "bf16": 512,
        "fp8": 1024,
        "int8": 1024,
    },
    # Hopper (H100, H200): 132 SMs, ~1830 MHz boost
    # Dense peaks: FP64 TC 67 TF, TF32 989 TF, BF16/FP16 989 TF,
    #              FP8/INT8 1979 TF
    (9, 0): {
        "fp64": 128,  # FP64 TC (correct at ~1980 MHz max boost)
        "fp32": 2048,  # TF32
        "tf32": 2048,
        "fp16": 2048,
        "bf16": 2048,
        "fp8": 4096,
        "int8": 4096,
    },
}

# Known GPU peak specs — **DENSE** (without 2:4 structured sparsity).
# Sparse throughput is exactly 2× dense for the same dtype.
# All values below are the dense Tensor Core peaks from NVIDIA datasheets.
# Format: {gpu_name_pattern: {dtype: peak_tflops_dense}}
_KNOWN_GPU_SPECS: Dict[str, Dict[str, float]] = {
    # H100 SXM5: 132 SMs, boost ~1830 MHz
    "H100": {
        "fp64": 67,
        "tf32": 989,
        "fp16": 989,  # dense; sparse = 1979
        "bf16": 989,  # dense; sparse = 1979
        "fp8": 1979,  # dense; sparse = 3958
        "int8": 1979,  # dense; sparse = 3958
    },
    "H200": {
        "fp64": 67,
        "tf32": 989,
        "fp16": 989,  # dense; sparse = 1979
        "bf16": 989,  # dense; sparse = 1979
        "fp8": 1979,  # dense; sparse = 3958
        "int8": 1979,  # dense; sparse = 3958
    },
    # A100 SXM4: 108 SMs, boost ~1410 MHz
    "A100": {
        "fp64": 19.5,
        "tf32": 156,
        "fp16": 312,
        "bf16": 312,
        "int8": 624,
    },
    "A10": {
        "tf32": 62.5,
        "fp16": 125,
        "bf16": 125,
        "int8": 250,
    },
    "L40": {
        "tf32": 181,
        "fp16": 362,
        "bf16": 362,
        "fp8": 724,
        "int8": 724,
    },
}


# Known GPU clock frequencies (MHz)
_GPU_CLOCK_MHZ = {
    "H100": 1980,
    "H200": 1980,
    "A100": 1410,
    "A10": 1695,
    "L40": 2490,
}

# Known GPU memory bandwidth (GB/s)
# Format: {gpu_pattern: (sxm_bandwidth, pcie_bandwidth)} or {gpu_pattern: bandwidth}
_GPU_MEMORY_BANDWIDTH: Dict[str, Union[float, Tuple[float, float]]] = {
    "H100": (3350, 2000),  # HBM3: SXM vs PCIe
    "H200": 4800,  # HBM3e
    "A100": (2039, 1555),  # HBM2e: SXM vs PCIe
    "A10": 600,
    "L40": 864,
}

# Known NVLink bandwidth (GB/s, bidirectional)
_NVLINK_BANDWIDTH = {
    "H100": 900,  # NVLink 4.0
    "H200": 900,  # NVLink 4.0
    "A100": 600,  # NVLink 3.0
}


def _match_gpu_pattern(name: str, specs_dict: Dict) -> Optional[str]:
    """Find matching GPU pattern in specs dictionary."""
    name_upper = name.upper()
    for pattern in specs_dict:
        if pattern in name_upper:
            return pattern
    return None


def _run_nvidia_smi(query: str) -> Optional[str]:
    """Run nvidia-smi query and return result."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _get_gpu_clock_mhz(device_index: int = 0) -> float:
    """Get max GPU clock frequency in MHz."""
    # Try nvidia-smi first
    result = _run_nvidia_smi("clocks.max.graphics")
    if result:
        lines = result.split("\n")
        if device_index < len(lines):
            try:
                return float(lines[device_index])
            except ValueError:
                pass

    # Fallback: estimate from known GPUs
    props = torch.cuda.get_device_properties(device_index)
    pattern = _match_gpu_pattern(props.name, _GPU_CLOCK_MHZ)
    if pattern:
        return _GPU_CLOCK_MHZ[pattern]
    return 1500  # Default fallback


def _get_memory_bandwidth_gb_s(device_index: int = 0) -> float:
    """Calculate memory bandwidth in GB/s from device properties."""
    props = torch.cuda.get_device_properties(device_index)

    # memory_clock_rate is in kHz, memory_bus_width is in bits
    # Bandwidth = clock_rate * bus_width * 2 (DDR) / 8 (bits to bytes)
    if hasattr(props, "memory_clock_rate") and hasattr(props, "memory_bus_width"):
        clock_khz = props.memory_clock_rate
        bus_width_bits = props.memory_bus_width
        bandwidth_gb_s = clock_khz * 1000 * bus_width_bits * 2 / 8 / 1e9
        return bandwidth_gb_s

    # Fallback: use known specs
    name = props.name.upper()
    pattern = _match_gpu_pattern(name, _GPU_MEMORY_BANDWIDTH)
    if pattern:
        bandwidth = _GPU_MEMORY_BANDWIDTH[pattern]
        if isinstance(bandwidth, tuple):
            # (SXM, PCIe) - check if SXM in name
            return bandwidth[0] if "SXM" in name else bandwidth[1]
        return bandwidth
    return 1000  # Default fallback


def _get_nvlink_bandwidth(device_index: int = 0) -> Optional[float]:
    """Get NVLink bandwidth if available."""
    result = _run_nvidia_smi("pcie.link.gen.max,pcie.link.width.max")
    if not result:
        return None

    # Check for NVLink via nvidia-smi nvlink status
    try:
        nvlink_result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status", "-i", str(device_index)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if nvlink_result.returncode == 0 and "Link" in nvlink_result.stdout:
            # Parse NVLink info
            props = torch.cuda.get_device_properties(device_index)
            pattern = _match_gpu_pattern(props.name, _NVLINK_BANDWIDTH)
            if pattern:
                return _NVLINK_BANDWIDTH[pattern]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def _get_ib_bandwidth() -> Optional[float]:
    """Get InfiniBand bandwidth if available."""
    # Check for IB devices
    try:
        result = subprocess.run(
            ["ibstat", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # IB devices found, try to get speed
            speed_result = subprocess.run(
                ["ibstat"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Rate:" in speed_result.stdout:
                # Parse rate (e.g., "Rate: 400" for HDR)
                for line in speed_result.stdout.split("\n"):
                    if "Rate:" in line:
                        try:
                            rate = float(line.split(":")[1].strip())
                            # Rate is in Gb/s, convert to GB/s
                            return rate / 8
                        except (ValueError, IndexError):
                            pass
            # Fallback: assume HDR (400 Gb/s = 50 GB/s per port)
            return 50
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def _calculate_peak_tflops(
    compute_capability: Tuple[int, int],
    num_sms: int,
    clock_mhz: float,
    device_name: str,
) -> Dict[str, float]:
    """Calculate peak TFLOPS for different data types."""
    peak_tflops = {}

    # Try to use per-SM-per-cycle data
    tc_specs = _TC_FLOPS_PER_SM_PER_CYCLE.get(compute_capability)
    if tc_specs:
        for dtype, flops_per_sm_per_cycle in tc_specs.items():
            # TFLOPS = SMs * clock_MHz * 1e6 * flops_per_sm_per_cycle * 2 / 1e12
            # The *2 is for FMA (fused multiply-add counts as 2 ops)
            tflops = num_sms * clock_mhz * 1e6 * flops_per_sm_per_cycle * 2 / 1e12
            peak_tflops[dtype] = tflops

    # Validate/override with known specs if available
    name_upper = device_name.upper()
    for pattern, known_specs in _KNOWN_GPU_SPECS.items():
        if pattern in name_upper:
            # Use known specs as reference (they account for real boost behavior)
            for dtype, known_tflops in known_specs.items():
                # If calculated value is significantly different, use known value
                if (
                    dtype not in peak_tflops
                    or abs(peak_tflops[dtype] - known_tflops) / known_tflops > 0.2
                ):
                    peak_tflops[dtype] = known_tflops
            break

    return peak_tflops


def get_current_device_spec(device_index: Optional[int] = None) -> DeviceSpec:
    """
    Get the specification of the current CUDA device.

    This function queries hardware information through CUDA API and nvidia-smi
    to provide accurate device specifications including:
    - GPU architecture and compute capability
    - Number of SMs and clock frequency
    - Peak TFLOPS for various data types (Tensor Core operations)
    - Memory capacity and bandwidth
    - Network bandwidth (NVLink, InfiniBand) if available

    Args:
        device_index: CUDA device index. If None, uses current device.

    Returns:
        DeviceSpec: Dataclass containing device specifications.

    Example:
        >>> spec = get_current_device_spec()
        >>> print(spec)
        Device: NVIDIA H100 80GB HBM3 (index: 0)
        Architecture: Hopper (SM 90)
        SMs: 132, Clock: 1980 MHz
        Memory: 80.0 GB, Bandwidth: 3350 GB/s
        Peak TFLOPS (Tensor Core):
          bf16: 1979.0 TFLOPS
          fp16: 1979.0 TFLOPS
          fp8: 3958.0 TFLOPS
          ...

        >>> # Get peak BF16 TFLOPS for MFU calculation
        >>> bf16_peak = spec.peak_tflops.get("bf16", spec.peak_tflops.get("fp16"))
    """
    if device_index is None:
        device_index = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device_index)

    compute_capability = (props.major, props.minor)
    architecture = _ARCH_NAMES.get(
        compute_capability, f"Unknown (SM {props.major}{props.minor})"
    )

    gpu_clock_mhz = _get_gpu_clock_mhz(device_index)
    memory_bandwidth = _get_memory_bandwidth_gb_s(device_index)
    nvlink_bandwidth = _get_nvlink_bandwidth(device_index)
    ib_bandwidth = _get_ib_bandwidth()

    peak_tflops = _calculate_peak_tflops(
        compute_capability=compute_capability,
        num_sms=props.multi_processor_count,
        clock_mhz=gpu_clock_mhz,
        device_name=props.name,
    )

    return DeviceSpec(
        device_index=device_index,
        device_name=props.name,
        compute_capability=compute_capability,
        architecture=architecture,
        num_sms=props.multi_processor_count,
        gpu_clock_mhz=gpu_clock_mhz,
        peak_tflops=peak_tflops,
        memory_total_gb=props.total_memory / (1024**3),
        memory_bandwidth_gb_s=memory_bandwidth,
        nvlink_bandwidth_gb_s=nvlink_bandwidth,
        ib_bandwidth_gb_s=ib_bandwidth,
    )


def cal_mfu(
    achieved_tflops: float,
    world_size: int = 1,
    dtype: str = "bf16",
    device_spec: Optional[DeviceSpec] = None,
) -> float:
    """
    Calculate Model FLOPs Utilization (MFU).

    MFU measures how efficiently the model utilizes the GPU's peak compute
    capability. It is defined as:
        MFU = achieved_tflops / (peak_tflops * world_size)

    Args:
        achieved_tflops: Actual TFLOPS achieved during training (global, across all GPUs).
        world_size: Number of GPUs in the distributed training.
        dtype: Data type used for training. Options: "fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8".
               Default is "bf16" which is common for modern training.
        device_spec: Optional DeviceSpec. If None, will query current device.

    Returns:
        MFU as a percentage (0-100).

    Example:
        >>> # During training loop
        >>> flops = cal_flops(model_config, seqlens, num_contextuals, num_candidates)
        >>> elapsed_ms = timer.elapsed_time()
        >>> achieved_tflops = flops / elapsed_ms / 1e9
        >>>
        >>> world_size = torch.distributed.get_world_size()
        >>> mfu = cal_mfu(achieved_tflops, world_size, dtype="bf16")
        >>> print(f"MFU: {mfu:.2f}%")

        >>> # With custom device spec
        >>> spec = get_current_device_spec()
        >>> mfu = cal_mfu(achieved_tflops, world_size, dtype="bf16", device_spec=spec)
    """
    if device_spec is None:
        device_spec = get_current_device_spec()

    # Get peak TFLOPS for the specified dtype
    peak_tflops = device_spec.peak_tflops.get(dtype)

    if peak_tflops is None:
        # Fallback: try similar dtypes
        fallback_order = {
            "bf16": ["fp16", "tf32"],
            "fp16": ["bf16", "tf32"],
            "tf32": ["fp32", "fp16"],
            "fp32": ["tf32", "fp16"],
            "fp8": ["int8", "bf16"],
            "int8": ["fp8", "bf16"],
        }
        for fallback in fallback_order.get(dtype, []):
            if fallback in device_spec.peak_tflops:
                peak_tflops = device_spec.peak_tflops[fallback]
                break

    if peak_tflops is None:
        raise ValueError(
            f"Cannot determine peak TFLOPS for dtype '{dtype}' on device '{device_spec.device_name}'. "
            f"Available dtypes: {list(device_spec.peak_tflops.keys())}"
        )

    # Calculate global peak TFLOPS
    global_peak_tflops = peak_tflops * world_size

    # Calculate MFU percentage
    mfu = (achieved_tflops / global_peak_tflops) * 100

    return mfu


def cal_hstu_flops_single_rank(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    dim_per_head: int,
    seqlens: torch.Tensor,
    num_contextuals: Optional[torch.Tensor],
    num_candidates: Optional[torch.Tensor],
    has_bwd: bool = True,
    is_causal: bool = True,
    residual: bool = True,
) -> torch.Tensor:
    if num_contextuals is None:
        num_contextuals = torch.zeros_like(seqlens)
    if num_candidates is None:
        num_candidates = torch.zeros_like(seqlens)
    with torch.inference_mode():
        seqlens = seqlens.to(torch.float)
        num_contextuals = num_contextuals.to(torch.float)
        num_candidates = num_candidates.to(torch.float)
        num_history = seqlens - num_contextuals - num_candidates
        # reference: https://github.com/Dao-AILab/flash-attention/blob/9c0e9ee86d0e0022b60deddb405c20ab77481582/benchmarks/benchmark_flash_attention.py#L27-L30
        # flops between seq and contextual + history
        attn_flops_per_layer = (
            4 * num_heads * seqlens * (num_contextuals + num_history) * dim_per_head
        )
        if is_causal:
            # remove upper triangular flops between history and history
            attn_flops_per_layer -= (
                2 * num_heads * num_history * num_history * dim_per_head
            )
        # flops between candidates
        attn_flops_per_layer += 4 * num_heads * num_candidates * dim_per_head
        if has_bwd:
            attn_flops_per_layer *= 3.5

        gemm_flops_per_layer = (
            2 * seqlens * 4 * num_heads * dim_per_head * hidden_size
        )  # qkvu proj fwd
        gemm_flops_per_layer += 2 * seqlens * num_heads * hidden_size  # proj fwd
        if has_bwd:
            gemm_flops_per_layer *= 3

        other_ops_flops_per_layer = seqlens * num_heads * dim_per_head  # mul fwd
        if has_bwd:
            other_ops_flops_per_layer *= 2  # bwd
        if residual:
            other_ops_flops_per_layer += (
                seqlens * num_heads * hidden_size
            )  # add fwd, bwd is no-op

        return (
            torch.sum(
                gemm_flops_per_layer + attn_flops_per_layer + other_ops_flops_per_layer
            )
            * num_layers
        )


def cal_hstu_flops(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    dim_per_head: int,
    seqlens: List[torch.Tensor],
    num_contextuals: List[torch.Tensor],
    num_candidates: List[torch.Tensor],
    has_bwd: bool = True,
    is_causal: bool = True,
    residual: bool = True,
    dp_pg: Optional[torch.distributed.ProcessGroup] = None,
) -> int:
    """
    Calculate total FLOPS for HSTU model across all data parallel ranks.

    Args:
        dp_pg: Data Parallel process group. When using TP, pass the DP process
               group to gather only from DP ranks (avoiding duplicate data from
               TP ranks that process the same batch). If None, uses the default
               world process group.
    """
    seqlens_tensor = torch.cat(seqlens)
    num_contextuals_tensor = torch.cat(num_contextuals)
    num_candidates_tensor = torch.cat(num_candidates)

    # Use DP process group if provided, otherwise use default world group
    if dp_pg is not None:
        dp_world_size = torch.distributed.get_world_size(group=dp_pg)
        dp_rank = torch.distributed.get_rank(group=dp_pg)
    else:
        dp_world_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()

    # Gather to rank 0 in the DP group
    gathered_seqlens = (
        [torch.empty_like(seqlens_tensor) for _ in range(dp_world_size)]
        if dp_rank == 0
        else None
    )
    gathered_num_contextuals = (
        [torch.empty_like(num_contextuals_tensor) for _ in range(dp_world_size)]
        if dp_rank == 0
        else None
    )
    gathered_num_candidates = (
        [torch.empty_like(num_candidates_tensor) for _ in range(dp_world_size)]
        if dp_rank == 0
        else None
    )

    torch.distributed.gather(seqlens_tensor, gathered_seqlens, dst=0, group=dp_pg)
    torch.distributed.gather(
        num_contextuals_tensor, gathered_num_contextuals, dst=0, group=dp_pg
    )
    torch.distributed.gather(
        num_candidates_tensor, gathered_num_candidates, dst=0, group=dp_pg
    )

    if dp_rank == 0:
        flops = (
            cal_hstu_flops_single_rank(
                num_layers,
                hidden_size,
                num_heads,
                dim_per_head,
                torch.cat(gathered_seqlens),
                torch.cat(gathered_num_contextuals),
                torch.cat(gathered_num_candidates),
                has_bwd,
                is_causal,
                residual,
            )
            .cpu()
            .item()
        )
    else:
        flops = 0
    return flops


def get_mfu_summary(
    achieved_tflops: float,
    world_size: int = 1,
    dtype: str = "bf16",
    device_spec: Optional[DeviceSpec] = None,
) -> str:
    """
    Get a formatted summary string of MFU calculation.

    Args:
        achieved_tflops: Actual TFLOPS achieved during training.
        world_size: Number of GPUs.
        dtype: Data type used for training.
        device_spec: Optional DeviceSpec.

    Returns:
        Formatted string with MFU breakdown.

    Example:
        >>> summary = get_mfu_summary(achieved_tflops=500, world_size=8, dtype="bf16")
        >>> print(summary)
        MFU Summary:
          Achieved TFLOPS: 500.00 (global)
          Peak TFLOPS per GPU: 1979.00 (bf16)
          World size: 8
          Global Peak TFLOPS: 15832.00
          MFU: 3.16%
    """
    if device_spec is None:
        device_spec = get_current_device_spec()

    peak_tflops = device_spec.peak_tflops.get(dtype)
    if peak_tflops is None:
        for fallback in ["fp16", "tf32", "fp32"]:
            if fallback in device_spec.peak_tflops:
                peak_tflops = device_spec.peak_tflops[fallback]
                dtype = fallback
                break

    global_peak = peak_tflops * world_size if peak_tflops else 0
    mfu = cal_mfu(achieved_tflops, world_size, dtype, device_spec) if peak_tflops else 0

    return (
        f"MFU Summary:\n"
        f"  Device: {device_spec.device_name}\n"
        f"  Achieved TFLOPS: {achieved_tflops:.2f} (global)\n"
        f"  Peak TFLOPS per GPU: {peak_tflops:.2f} ({dtype})\n"
        f"  World size: {world_size}\n"
        f"  Global Peak TFLOPS: {global_peak:.2f}\n"
        f"  MFU: {mfu:.2f}%"
    )


def _compute_attn_fwd_flops(
    offsets: torch.Tensor,
    num_heads: int,
    attn_dim: int,
    linear_dim: int,
    is_causal: bool,
    num_candidates: Optional[torch.Tensor],
    num_contextuals: Optional[Union[int, torch.Tensor]],
) -> float:
    """Compute **forward-only** attention FLOPs for one layer.

    Uses the same formula as ``cal_flops_single_rank`` in trainer/utils.py but
    restricted to the attention component (QK + PV).
    """
    with torch.inference_mode():
        seqlens = (offsets[1:] - offsets[:-1]).float()  # (B,)
        B = seqlens.shape[0]

        if num_contextuals is None:
            ctx = torch.zeros(B, device=seqlens.device)
        elif isinstance(num_contextuals, int):
            ctx = torch.full(
                (B,), num_contextuals, device=seqlens.device, dtype=torch.float
            )
        else:
            ctx = num_contextuals.float()

        if num_candidates is None:
            cand = torch.zeros(B, device=seqlens.device)
        else:
            cand = num_candidates.float()

        num_history = seqlens - ctx - cand

        # reference: flash-attention benchmark formula
        # QK + PV over (contextual + history) region
        flops = 4.0 * num_heads * seqlens * (ctx + num_history) * attn_dim
        if is_causal:
            # subtract upper-triangular portion of history×history
            flops = flops - 2.0 * num_heads * num_history * num_history * attn_dim
        # candidates contribution
        flops = flops + 4.0 * num_heads * cand * attn_dim

        return flops.sum().item()
