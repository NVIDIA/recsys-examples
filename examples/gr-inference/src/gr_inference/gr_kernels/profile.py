"""Kernel selection profile schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelProfile:
    """Serializable kernel backend selection profile."""

    schema_version: int
    model: dict[str, Any]
    target: dict[str, Any]
    selected: dict[str, str]
    benchmarks: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "KernelProfile":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelProfile":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            model=dict(data.get("model", {})),
            target=dict(data.get("target", {})),
            selected=dict(data.get("selected", {})),
            benchmarks=dict(data.get("benchmarks", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": dict(self.model),
            "target": dict(self.target),
            "selected": dict(self.selected),
            "benchmarks": dict(self.benchmarks),
        }

    def save(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

