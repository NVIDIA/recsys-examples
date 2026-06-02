"""Checkpoint discovery and optional tensor loading.

The loader layer is intentionally model-family agnostic. Model adapters own
logical tensor names; this module only discovers files, config, and tensor
locations for HuggingFace-style checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TensorLocation:
    """Where a tensor lives inside a checkpoint directory."""

    name: str
    filename: str


@dataclass(frozen=True)
class CheckpointManifest:
    """HuggingFace-style checkpoint manifest."""

    model_dir: Path
    config: dict[str, Any]
    tensor_map: dict[str, TensorLocation]
    weight_files: tuple[str, ...]

    def has_tensor(self, name: str) -> bool:
        return name in self.tensor_map

    def require(self, names: list[str] | tuple[str, ...]) -> None:
        missing = [name for name in names if name not in self.tensor_map]
        if missing:
            raise KeyError("missing checkpoint tensors: " + ", ".join(missing))


@dataclass(frozen=True)
class TensorLoadRequest:
    """One logical framework tensor and the checkpoint tensors needed for it.

    This is inspired by vLLM's per-parameter weight loaders and packed parameter
    mappings. The request records intent without requiring torch at discovery
    time.
    """

    logical_name: str
    source_names: tuple[str, ...]
    transform: str = "identity"
    dim: int = 0
    required: bool = True

    def __post_init__(self) -> None:
        if not self.logical_name:
            raise ValueError("logical_name must not be empty")
        if not self.source_names:
            raise ValueError("source_names must not be empty")
        if self.transform not in {"identity", "concat"}:
            raise ValueError(f"unsupported tensor transform: {self.transform}")
        if self.transform == "identity" and len(self.source_names) != 1:
            raise ValueError("identity transform expects exactly one source tensor")


@dataclass(frozen=True)
class CheckpointLoadPlan:
    """Model-adapter-produced plan for loading a checkpoint into GR runtime."""

    requests: tuple[TensorLoadRequest, ...]

    def source_names(self, *, required_only: bool = False) -> tuple[str, ...]:
        names: list[str] = []
        for request in self.requests:
            if required_only and not request.required:
                continue
            names.extend(request.source_names)
        return tuple(dict.fromkeys(names))

    def validate(self, manifest: CheckpointManifest) -> None:
        manifest.require(self.source_names(required_only=True))

    def grouped_by_file(
        self, manifest: CheckpointManifest
    ) -> dict[str, tuple[str, ...]]:
        """Group required source tensor names by checkpoint file.

        This is the hook for SGLang/vLLM-style streaming, buffering, and
        per-rank file reads.
        """

        groups: dict[str, list[str]] = {}
        for name in self.source_names(required_only=True):
            location = manifest.tensor_map[name]
            groups.setdefault(location.filename, []).append(name)
        return {
            filename: tuple(tensor_names)
            for filename, tensor_names in sorted(groups.items())
        }

    def materialize(self, tensor_getter) -> dict[str, Any]:
        """Materialize logical tensors using an injected source tensor getter.

        ``tensor_getter`` receives a source tensor name and returns an object that
        supports the required transform. For torch tensors, concat uses
        ``torch.cat``; for simple fake tensors in tests, objects can expose a
        ``concat`` classmethod.
        """

        materialized: dict[str, Any] = {}
        for request in self.requests:
            source_tensors = [tensor_getter(name) for name in request.source_names]
            if any(tensor is None for tensor in source_tensors):
                if request.required:
                    missing = [
                        name
                        for name, tensor in zip(
                            request.source_names, source_tensors, strict=True
                        )
                        if tensor is None
                    ]
                    raise KeyError(
                        "missing source tensors for "
                        f"{request.logical_name}: {', '.join(missing)}"
                    )
                continue
            if request.transform == "identity":
                materialized[request.logical_name] = source_tensors[0]
            elif request.transform == "concat":
                materialized[request.logical_name] = concat_tensors(
                    source_tensors,
                    dim=request.dim,
                )
            else:  # pragma: no cover - guarded by TensorLoadRequest
                raise ValueError(f"unsupported transform: {request.transform}")
        return materialized


def concat_tensors(tensors: list[Any], *, dim: int) -> Any:
    """Concatenate tensor-like objects without forcing a torch dependency."""

    if not tensors:
        raise ValueError("cannot concat an empty tensor list")
    first = tensors[0]
    concat = getattr(type(first), "concat", None)
    if callable(concat):
        return concat(tensors, dim=dim)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "concat transform requires torch or a concat-capable fake tensor"
        ) from exc
    return torch.cat(tensors, dim=dim)


class HFCheckpointLoader:
    """Discover and optionally load tensors from a HF checkpoint directory."""

    CONFIG_NAME = "config.json"
    SAFETENSORS_INDEX = "model.safetensors.index.json"
    PYTORCH_INDEX = "pytorch_model.bin.index.json"

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

    def manifest(self) -> CheckpointManifest:
        config_path = self.model_dir / self.CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found under {self.model_dir}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        tensor_map: dict[str, TensorLocation] = {}
        weight_files: list[str] = []

        index_path = self._find_index_file()
        if index_path is not None:
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            for tensor_name, filename in index.get("weight_map", {}).items():
                tensor_map[tensor_name] = TensorLocation(tensor_name, filename)
            weight_files = sorted(set(index.get("weight_map", {}).values()))
        else:
            weight_files = self._find_weight_files()
            tensor_map = self._discover_unindexed_safetensors(weight_files)

        return CheckpointManifest(
            model_dir=self.model_dir,
            config=config,
            tensor_map=tensor_map,
            weight_files=tuple(weight_files),
        )

    def load_tensor(self, manifest: CheckpointManifest, name: str) -> Any:
        """Load one tensor if optional runtime dependencies are installed."""

        location = manifest.tensor_map[name]
        path = manifest.model_dir / location.filename
        if location.filename.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("loading safetensors requires safetensors") from exc
            return load_file(str(path))[name]

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("loading PyTorch checkpoints requires torch") from exc
        return torch.load(path, map_location="cpu")[name]

    def _find_index_file(self) -> Path | None:
        for filename in (self.SAFETENSORS_INDEX, self.PYTORCH_INDEX):
            path = self.model_dir / filename
            if path.is_file():
                return path
        return None

    def _find_weight_files(self) -> list[str]:
        patterns = ("*.safetensors", "*.bin", "*.pt")
        files: list[str] = []
        for pattern in patterns:
            files.extend(path.name for path in sorted(self.model_dir.glob(pattern)))
        return files

    def _discover_unindexed_safetensors(
        self,
        weight_files: list[str],
    ) -> dict[str, TensorLocation]:
        tensor_map: dict[str, TensorLocation] = {}
        safetensor_files = [
            filename for filename in weight_files if filename.endswith(".safetensors")
        ]
        if not safetensor_files:
            return tensor_map

        try:
            from safetensors import safe_open
        except ImportError:
            return tensor_map

        for filename in safetensor_files:
            path = self.model_dir / filename
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for tensor_name in handle.keys():
                    tensor_map[tensor_name] = TensorLocation(tensor_name, filename)
        return tensor_map
