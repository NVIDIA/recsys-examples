"""Shape-only tensor layout helpers.

These helpers intentionally avoid importing torch or CUDA libraries. The first
MVP can validate framework contracts anywhere, then swap in real tensors when
the existing kernel binding is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

Shape = tuple[int, ...]


def normalize_shape(shape: Iterable[int]) -> Shape:
    normalized = tuple(int(dim) for dim in shape)
    if not normalized:
        raise ValueError("shape must not be empty")
    if any(dim <= 0 for dim in normalized):
        raise ValueError(f"shape dimensions must be positive: {normalized}")
    return normalized


def shape_of(tensor: Any) -> Shape:
    """Return a normalized shape from a TensorSpec or tensor-like object."""

    if isinstance(tensor, TensorSpec):
        return tensor.shape
    if hasattr(tensor, "shape"):
        return normalize_shape(getattr(tensor, "shape"))
    raise TypeError(f"object has no shape attribute: {type(tensor)!r}")


def dtype_of(tensor: Any) -> str | None:
    if isinstance(tensor, TensorSpec):
        return tensor.dtype
    dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


@dataclass(frozen=True)
class TensorSpec:
    """Small tensor descriptor used by tests and wrapper validation."""

    name: str
    shape: Shape
    dtype: str = "bf16"
    device: str = "cuda"
    layout: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", normalize_shape(self.shape))

    def with_name(self, name: str) -> "TensorSpec":
        return TensorSpec(
            name=name,
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
        )

    def with_shape(self, shape: Iterable[int]) -> "TensorSpec":
        return TensorSpec(
            name=self.name,
            shape=normalize_shape(shape),
            dtype=self.dtype,
            device=self.device,
            layout=self.layout,
        )
