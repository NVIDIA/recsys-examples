# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-IPC weight-tensor serialization for colocate weight sync.

This mirrors SGLang's colocate ``update_weights_from_tensor`` wire format so a
slime trainer (separate process, same GPU) can push weights to this engine --
which plays the rollout-engine role that SGLang plays in slime's native setup:

- The trainer flattens same-dtype tensors into one buffer, then serializes the
  bucket with ``ForkingPickler``. CUDA tensors reduce to a **CUDA IPC handle**
  (a pointer), not their bytes — so the HTTP payload is small and the real
  weight memory is shared zero-copy via CUDA IPC.
- The engine deserializes the IPC handle and reconstructs the CUDA tensor in its
  own context. Because trainer and engine may map the GPU to different device
  indices, the device is carried as its UUID and remapped on receive
  (SGLang's ``monkey_patch_torch_reductions``).

The SGLang originals are Apache-2.0:
  - ``sglang/srt/utils/patch_torch.py`` (``monkey_patch_torch_reductions``)
  - ``sglang/srt/utils/common.py`` (``MultiprocessingSerializer``)
  - ``sglang/srt/weight_sync/tensor_bucket.py`` (``FlattenedTensorBucket``)

Only the CUDA receive path is implemented; NPU/MUSA branches are omitted.
"""

from __future__ import annotations

import base64
import io
import pickle
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - import availability depends on runtime container
    import torch
    from torch.multiprocessing import reductions as _torch_reductions
except ImportError:  # pragma: no cover
    torch = None
    _torch_reductions = None

# ForkingPickler lives at multiprocessing.reduction in current torch; reductions
# register CUDA-IPC rebuild functions against it via init_reductions().
from multiprocessing.reduction import ForkingPickler  # type: ignore[import-not-found]


# --------------------------------------------------------------------------- #
# Device-UUID monkey patch (SGLang patch_torch.py, CUDA branch)
# --------------------------------------------------------------------------- #

# reduce_tensor's tuple has the CUDA device at this fixed index.
_REDUCE_TENSOR_ARG_DEVICE_INDEX = 6
_PATCHED = False


def monkey_patch_torch_reductions() -> None:
    """Patch torch multiprocessing reductions to carry the device as a UUID.

    Idempotent. Mirrors SGLang's ``monkey_patch_torch_reductions`` so that
    tensors serialized by a slime trainer (device encoded as UUID) can be
    reconstructed here against this process's own device index.

    Process-global and effectively irreversible: it mutates
    ``torch.multiprocessing.reductions`` in place and re-runs
    ``init_reductions()``. The ``hasattr(..., "_reduce_tensor_original")`` guard
    keeps it a no-op if a co-resident SGLang already patched (or we already did),
    so the two can coexist, but the patch itself cannot be cleanly undone in a
    running process.
    """

    global _PATCHED
    if _PATCHED or _torch_reductions is None:
        return
    if hasattr(_torch_reductions, "_reduce_tensor_original"):
        # Already patched (e.g. by a co-installed sglang).
        _PATCHED = True
        return
    _torch_reductions._reduce_tensor_original = _torch_reductions.reduce_tensor
    _torch_reductions._rebuild_cuda_tensor_original = (
        _torch_reductions.rebuild_cuda_tensor
    )
    _torch_reductions.reduce_tensor = _reduce_tensor_modified
    _torch_reductions.rebuild_cuda_tensor = _rebuild_cuda_tensor_modified
    _torch_reductions.init_reductions()
    _PATCHED = True


def _device_to_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


def _device_from_maybe_uuid(device: Any) -> int:
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        for candidate in range(torch.cuda.device_count()):
            if str(torch.cuda.get_device_properties(candidate).uuid) == device:
                return candidate
        raise ValueError(f"unknown cuda device uuid: {device}")
    raise TypeError(f"unexpected device specifier: {device!r}")


def _modify_tuple(t: tuple, index: int, modifier):
    return *t[:index], modifier(t[index]), *t[index + 1 :]


def _reduce_tensor_modified(*args, **kwargs):
    output_fn, output_args = _torch_reductions._reduce_tensor_original(
        *args, **kwargs
    )
    # Only CUDA tensors carry the device at this index; CPU/other reductions use
    # a different (shorter) tuple, so leave them untouched. Guard with an
    # isinstance check so a future PyTorch reduce-protocol change (reordered or
    # added args) fails loud instead of silently UUID-encoding the wrong element
    # and misrouting the CUDA device.
    if len(output_args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:
        device_arg = output_args[_REDUCE_TENSOR_ARG_DEVICE_INDEX]
        if not isinstance(device_arg, int):
            raise RuntimeError(
                f"reduce_tensor protocol drift: expected int CUDA device at "
                f"index {_REDUCE_TENSOR_ARG_DEVICE_INDEX}, got "
                f"{type(device_arg).__name__} ({device_arg!r})"
            )
        output_args = _modify_tuple(
            output_args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_to_uuid
        )
    return output_fn, output_args


def _rebuild_cuda_tensor_modified(*args):
    if len(args) > _REDUCE_TENSOR_ARG_DEVICE_INDEX:
        args = _modify_tuple(
            args, _REDUCE_TENSOR_ARG_DEVICE_INDEX, _device_from_maybe_uuid
        )
    return _torch_reductions._rebuild_cuda_tensor_original(*args)


# --------------------------------------------------------------------------- #
# Flattened tensor bucket (SGLang tensor_bucket.py)
# --------------------------------------------------------------------------- #


@dataclass
class FlattenedTensorMetadata:
    name: str
    shape: Any
    dtype: Any
    start_idx: int
    end_idx: int
    numel: int


class FlattenedTensorBucket:
    """Flatten same-dtype tensors into one uint8 buffer for one IPC transfer."""

    supports_multi_dtypes = True

    def __init__(self, flattened_tensor=None, metadata=None, named_tensors=None):
        if named_tensors is not None:
            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")
            current = 0
            flats = []
            self.metadata = []
            for name, tensor in named_tensors:
                flat = tensor.flatten().view(torch.uint8)
                flats.append(flat)
                numel = flat.numel()
                self.metadata.append(
                    FlattenedTensorMetadata(
                        name=name,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        start_idx=current,
                        end_idx=current + numel,
                        numel=numel,
                    )
                )
                current += numel
            self.flattened_tensor = torch.cat(flats, dim=0)
        else:
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "provide named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata

    def reconstruct_tensors(self):
        out = []
        for meta in self.metadata:
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )
            out.append((meta.name, tensor))
        return out


# --------------------------------------------------------------------------- #
# MultiprocessingSerializer (SGLang common.py) + allowlisted, metadata-remapping
# unpickler
# --------------------------------------------------------------------------- #

# The wire payload is untrusted (any caller with the API key can POST). Restrict
# unpickling to exactly the modules a legitimate weight bucket needs: torch
# tensors + their multiprocessing CUDA-IPC rebuild functions, plain builtins /
# collections, and our metadata class (remapped by name below). Anything else --
# e.g. os.system, subprocess -- is rejected so a crafted payload cannot trigger
# arbitrary class instantiation (pickle RCE). Mirrors SGLang's SafeUnpickler.
# Roots: a module is allowed if it equals a root or is a submodule
# (root + "."), so "torch" covers "torch.Tensor" and "torch.storage.UntypedStorage".
_ALLOWED_UNPICKLE_ROOTS = frozenset(
    {
        "builtins",
        "collections",
        "copyreg",
        "functools",
        "itertools",
        "operator",
        "types",
        "weakref",
        "pickletools",
        "torch",
        "numpy",
        "multiprocessing.reduction",
        "multiprocessing.resource_sharer",
    }
)


def _module_allowed(module: str) -> bool:
    return module in _ALLOWED_UNPICKLE_ROOTS or any(
        module.startswith(root + ".") for root in _ALLOWED_UNPICKLE_ROOTS
    )


class _WeightUnpickler(pickle.Unpickler):
    """Allowlisted unpickler for trainer payloads.

    Maps any ``FlattenedTensorMetadata`` (pickled by qualified name; the class
    lives at different module paths across SGLang versions) to ours, and rejects
    every class whose module is not on the allowlist -- so we never depend on
    SGLang being installed, and a crafted payload cannot instantiate arbitrary
    classes.
    """

    def find_class(self, module: str, name: str):
        if name == "FlattenedTensorMetadata":
            return FlattenedTensorMetadata
        # The CUDA rebuild callable lives in whatever module patched reductions
        # (ours, SGLang's, or a slime-vendored copy). Always resolve it to OUR
        # reviewed version, so the producer's module is irrelevant and a crafted
        # payload cannot substitute a malicious rebuild callable. The unpatched
        # rebuild_cuda_tensor is in torch.multiprocessing.reductions (allowed).
        if name == "_rebuild_cuda_tensor_modified":
            return _rebuild_cuda_tensor_modified
        if _module_allowed(module):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"blocked unpickle of {module}.{name}: module not in the "
            "weight-payload allowlist"
        )


class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        monkey_patch_torch_reductions()
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        data = buf.getvalue()
        if output_str:
            return base64.b64encode(data).decode("utf-8")
        return data

    @staticmethod
    def deserialize(data):
        monkey_patch_torch_reductions()
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        return _WeightUnpickler(io.BytesIO(data)).load()


def reconstruct_named_tensors(
    serialized_named_tensors,
    *,
    load_format: str | None,
    tp_rank: int = 0,
):
    """Deserialize a SGLang ``serialized_named_tensors`` payload to (name, tensor).

    ``serialized_named_tensors`` is a list with one entry per TP rank; we take
    this rank's entry (single-GPU -> index 0). For ``load_format="flattened_bucket"``
    the entry deserializes to ``{flattened_tensor, metadata}``; otherwise it is a
    ``[(name, tensor)]`` list (``load_format`` None/"direct").
    """

    monkey_patch_torch_reductions()
    if not serialized_named_tensors:
        return []
    index = tp_rank if tp_rank < len(serialized_named_tensors) else 0
    decoded = MultiprocessingSerializer.deserialize(serialized_named_tensors[index])
    if load_format == "flattened_bucket":
        bucket = FlattenedTensorBucket(
            flattened_tensor=decoded["flattened_tensor"],
            metadata=decoded["metadata"],
        )
        return bucket.reconstruct_tensors()
    return list(decoded)
