"""Adapter for the external ``gr-decode_atten`` CuTe DSL kernel."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from gr_inference.gr_kernels.attention.gr_decode_attention import (
    GRDecodeAttentionInputs,
    MissingKernelBackend,
)

DEFAULT_KERNEL_ROOTS = (
    Path(__file__).resolve().parents[4] / "third_party/gr-decode-attention",
)


class ExistingGRDecodeAttentionBackend:
    """Call the existing ``interface.beam_decode_attn`` implementation.

    Framework-facing Q is compact ``[B, W, Hq, D]``. The external kernel expects
    decode form ``[B, 1, W, Hq, D]`` and layer-local KV tensors:

    - ``k_context/v_context``: ``[B, S_ctx, Hkv, D]``
    - ``k_beam/v_beam``: ``[B, decode_nums * W, Hkv, D]``
    - ``topk_indices``: ``[B, 1, Hq, max_decode_nums, W]``
    """

    def __init__(self, kernel_root: str | os.PathLike[str] | None = None) -> None:
        self.kernel_root = self._resolve_kernel_root(kernel_root)
        self._beam_decode_attn = None

    def ensure_available(self) -> "ExistingGRDecodeAttentionBackend":
        """Eagerly import the external kernel so optional tests can skip cleanly."""
        self._load_kernel()
        return self

    def __call__(self, inputs: GRDecodeAttentionInputs) -> Any:
        if inputs.topk_indices is None:
            raise ValueError("Existing kernel backend requires topk_indices")

        beam_decode_attn = self._load_kernel()
        decode_nums = inputs.step if inputs.decode_nums is None else inputs.decode_nums

        q = self._ensure_decode_q(inputs.q)
        k_context = self._select_layer(inputs.context_kv.key, inputs.layer_idx)
        v_context = self._select_layer(inputs.context_kv.value, inputs.layer_idx)
        k_beam = self._select_beam_history(
            inputs.beam_kv.key,
            inputs.layer_idx,
            decode_nums,
            inputs.active_beam_width,
        )
        v_beam = self._select_beam_history(
            inputs.beam_kv.value,
            inputs.layer_idx,
            decode_nums,
            inputs.active_beam_width,
        )

        output = beam_decode_attn(
            q,
            k_context,
            v_context,
            k_beam,
            v_beam,
            inputs.topk_indices,
            decode_nums,
            return_lse=inputs.return_lse,
            backend=inputs.backend_name,
        )
        return self._normalize_kernel_output(output, return_lse=inputs.return_lse)

    def _load_kernel(self):
        if self._beam_decode_attn is not None:
            return self._beam_decode_attn

        root = str(self.kernel_root)
        if root not in sys.path:
            sys.path.insert(0, root)

        try:
            module = importlib.import_module("interface")
        except Exception as exc:  # pragma: no cover - depends on external env
            raise MissingKernelBackend(
                "Failed to import gr-decode_atten interface.py from "
                f"{self.kernel_root}. Install its requirements and set "
                "GR_DECODE_ATTEN_ROOT if needed."
            ) from exc

        self._beam_decode_attn = module.beam_decode_attn
        return self._beam_decode_attn

    @staticmethod
    def _resolve_kernel_root(kernel_root: str | os.PathLike[str] | None) -> Path:
        candidates: list[str | os.PathLike[str]] = []
        if kernel_root is not None:
            candidates.append(kernel_root)
        env_root = os.environ.get("GR_DECODE_ATTEN_ROOT")
        if env_root:
            candidates.append(env_root)
        candidates.extend(DEFAULT_KERNEL_ROOTS)

        for candidate in candidates:
            path = Path(candidate)
            if (path / "interface.py").is_file():
                return path

        formatted = ", ".join(str(candidate) for candidate in candidates)
        raise MissingKernelBackend(
            "Cannot find gr-decode_atten interface.py. Checked: " + formatted
        )

    @staticmethod
    def _ensure_decode_q(q: Any) -> Any:
        if not hasattr(q, "dim"):
            return q
        if q.dim() == 5:
            return q
        if q.dim() == 4:
            return q.unsqueeze(1)
        raise ValueError(f"Q must be 4-D or 5-D, got {q.dim()}-D")

    @staticmethod
    def _normalize_kernel_output(output: Any, *, return_lse: bool) -> Any:
        if not isinstance(output, tuple):
            return output
        if return_lse:
            return output
        return output[0]

    @staticmethod
    def _select_layer(tensor: Any, layer_idx: int) -> Any:
        if hasattr(tensor, "dim") and tensor.dim() == 5:
            return tensor[layer_idx]
        return tensor

    @staticmethod
    def _select_beam_history(
        tensor: Any,
        layer_idx: int,
        decode_nums: int,
        active_beam_width: int,
    ) -> Any:
        if not hasattr(tensor, "dim"):
            return tensor
        layer_tensor = tensor[layer_idx] if tensor.dim() == 6 else tensor
        if layer_tensor.dim() == 5:
            # [B, S_dec_max, W_max, Hkv, D] -> [B, decode_nums * W, Hkv, D]
            layer_tensor = layer_tensor[:, :decode_nums, :active_beam_width]
            batch, steps, width, kv_heads, head_dim = layer_tensor.shape
            return layer_tensor.reshape(batch, steps * width, kv_heads, head_dim)
        return layer_tensor
