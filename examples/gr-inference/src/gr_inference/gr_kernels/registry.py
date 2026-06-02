"""Kernel backend capability registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KernelCapability:
    """A named optimized operator capability."""

    name: str


@dataclass
class KernelBackendInfo:
    """Metadata for a kernel backend library."""

    name: str
    capabilities: frozenset[str]
    available: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        return self.available and capability in self.capabilities


class KernelBackendRegistry:
    """Small capability registry for optional kernel libraries."""

    def __init__(self) -> None:
        self._backends: dict[str, KernelBackendInfo] = {}

    def register(self, backend: KernelBackendInfo) -> None:
        self._backends[backend.name] = backend

    def get(self, name: str) -> KernelBackendInfo | None:
        return self._backends.get(name)

    def available_for(self, capability: str) -> list[KernelBackendInfo]:
        return [
            backend
            for backend in self._backends.values()
            if backend.supports(capability)
        ]

    def prefer(self, capability: str, order: tuple[str, ...]) -> KernelBackendInfo | None:
        for name in order:
            backend = self._backends.get(name)
            if backend is not None and backend.supports(capability):
                return backend
        candidates = self.available_for(capability)
        return candidates[0] if candidates else None

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "available": backend.available,
                "capabilities": sorted(backend.capabilities),
                "metadata": dict(backend.metadata),
            }
            for name, backend in sorted(self._backends.items())
        }


CAP_RMSNORM = "rmsnorm"
CAP_FUSED_ADD_RMSNORM = "fused_add_rmsnorm"
CAP_ROPE = "rope"
CAP_ROPE_WITH_CACHE = "rope_with_cache"
CAP_QK_NORM_ROPE = "qk_norm_rope"
CAP_PREFILL_ATTENTION = "prefill_attention"
CAP_GR_DECODE_ATTENTION = "gr_decode_attention"
CAP_PACKED_GEMM = "packed_gemm"
CAP_FUSED_MLP = "fused_mlp"
CAP_SAMPLING_TOPK = "sampling_topk"

