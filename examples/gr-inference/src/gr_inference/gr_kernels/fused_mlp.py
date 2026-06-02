"""Fused MLP backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TorchFusedMLPBackend:
    """Torch fallback for Qwen gate/up + SiLU + down projection."""

    def __call__(self, hidden_states: Any, ops: Any) -> Any:
        gate, up = ops.gate_up(hidden_states)
        intermediate = ops.silu_mul(gate, up)
        return ops.down_proj_only(intermediate)


@dataclass
class FusedMLP:
    """Dispatch wrapper for fused MLP implementations."""

    backend: Any

    def __call__(self, hidden_states: Any, ops: Any) -> Any:
        return self.backend(hidden_states, ops)
