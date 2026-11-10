# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""FP8 DeepGEMM variant of the reference Transformer inference layer.

``Fp8TransformerInferLayer`` inherits ``TransformerInferLayer`` and replaces its four
``nn.Linear`` projections (``qkv``, ``proj``, ``ffn.0``, ``ffn.2``) with ``Fp8Linear``.
LayerNorm, GELU, residuals, attention, the KV-cache append and the CUDA graph entry
points are inherited unchanged, so the checkpoint schema stays the same.

``Fp8Linear`` quantizes its bf16 input to FP8 with a Triton kernel, runs
``deep_gemm.fp8_gemm_nt`` against a pre-quantized weight and adds the bias in a separate
elementwise op because DeepGEMM has no bias epilogue. The quantization and the bias add
are standalone kernels in ``Fp8TransformerInferLayer``.

``Fp8FusedTransformerInferLayer`` moves the quantization into the neighbouring kernels
instead: the two LayerNorms and the bias + GELU quantize their own output, and only the
attention output is quantized by a standalone kernel.

``HSTUBlockInference`` builds ``Fp8FusedTransformerInferLayer`` layers when the inference
config sets ``transformer_fp8_linear`` (gin: ``NetworkArgs.transformer_fp8_linear``).
Loading a state dict requantizes their FP8 weights in place, which also reaches CUDA
graphs captured before the checkpoint was loaded.

DeepGEMM is an optional dependency and is JIT compiled at the first call of each shape,
so run a warmup forward before capturing a CUDA graph. Neither DeepGEMM nor the Triton
quantizer can be exported with torch.export or AOTI.
"""

import functools
import warnings

import torch
from modules.transformer_infer_layer import TransformerInferLayer
from ops.triton_ops.triton_fp8_quant import (
    GROUP_SIZE,
    triton_bias_gelu_quant_1x128,
    triton_layernorm_quant_1x128,
    triton_quant_1x128,
)
from torch import nn

try:
    import deep_gemm
except ImportError as error:
    deep_gemm = None
    _DEEP_GEMM_IMPORT_ERROR = error


@functools.lru_cache(maxsize=None)
def _uses_ue8m0(device):
    # SM100 and SM120 only take power-of-two (UE8M0) scales; SM90 takes fp32 scales.
    return torch.cuda.get_device_capability(device)[0] >= 10


def _supports_fp8(linear):
    return (
        linear.weight.is_cuda
        and linear.weight.dtype == torch.bfloat16
        and linear.in_features % GROUP_SIZE == 0
        and linear.out_features % GROUP_SIZE == 0
    )


def _requantize_after_load(module, incompatible_keys):
    module.refresh_fp8_weights()


class Fp8Linear(nn.Linear):
    """``nn.Linear`` that multiplies through DeepGEMM FP8 (bf16 in, bf16 out).

    ``weight`` and ``bias`` stay the bf16 checkpoint parameters, so state-dict keys and
    loading are unchanged. The FP8 weight is derived from ``weight`` by
    ``refresh_fp8_weights`` and rewritten in place, which keeps its address valid for
    captured CUDA graphs. The first forward builds it, and a ``load_state_dict`` post-hook
    rebuilds it, so a checkpoint loaded into a captured model reaches the graphs. After
    changing ``weight`` any other way, call ``refresh_fp8_weights``.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # Plain attributes, not buffers: Module.bfloat16() would cast registered fp8
        # buffers, and they do not belong in the state dict.
        self._fp8_weight = None
        self._fp8_scale = None
        # Every load_state_dict reaches it, also through a parent module's load.
        self.register_load_state_dict_post_hook(_requantize_after_load)

    @classmethod
    def from_linear(cls, linear):
        """Wrap ``linear``'s parameters without copying them."""
        fp8_linear = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device="meta",
            dtype=linear.weight.dtype,
        )
        fp8_linear.weight = linear.weight
        fp8_linear.bias = linear.bias
        return fp8_linear

    @torch.inference_mode()
    def refresh_fp8_weights(self):
        """Quantize ``weight`` to FP8 (128x128 blocks) into the cached FP8 tensors.

        Runs in inference mode: the FP8 tensors are first built during the CUDA graph
        warmup, which runs in inference mode, and PyTorch only lets inference tensors be
        updated in place inside it, while checkpoints are loaded outside.
        """
        weight = self.weight.detach()
        ue8m0 = _uses_ue8m0(weight.device)
        qweight, scale = deep_gemm.per_block_cast_to_fp8(weight, use_ue8m0=ue8m0)
        if ue8m0:
            # Broadcast the per-block scales to per-row and pack them into DeepGEMM's
            # layout once, instead of letting every GEMM redo it.
            rows = torch.arange(self.out_features, device=weight.device)
            scale = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(
                scale.index_select(0, rows // GROUP_SIZE)
            )
        if self._fp8_weight is None or self._fp8_weight.device != weight.device:
            self._fp8_weight, self._fp8_scale = qweight, scale
        else:
            self._fp8_weight.copy_(qweight)
            self._fp8_scale.copy_(scale)

    def gemm(self, qx, scale_x):
        """FP8 GEMM of an already quantized [M, in_features] input: bf16 out, no bias."""
        if self._fp8_weight is None or self._fp8_weight.device != self.weight.device:
            self.refresh_fp8_weights()
        out = torch.empty(
            (qx.shape[0], self.out_features), dtype=torch.bfloat16, device=qx.device
        )
        if qx.shape[0] > 0:
            deep_gemm.fp8_gemm_nt(
                (qx, scale_x),
                (self._fp8_weight, self._fp8_scale),
                out,
                disable_ue8m0_cast=not _uses_ue8m0(qx.device),
            )
        return out

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.bfloat16):
            return super().forward(x)
        lead_shape = x.shape[:-1]
        qx, scale_x = triton_quant_1x128(
            x.reshape(-1, self.in_features), _uses_ue8m0(x.device)
        )
        out = self.gemm(qx, scale_x)
        if self.bias is not None:
            out.add_(self.bias)
        return out.reshape(*lead_shape, self.out_features)


class Fp8TransformerInferLayer(TransformerInferLayer):
    """``TransformerInferLayer`` whose four projections run as FP8 DeepGEMM GEMMs.

    Requires bf16 and a layer that is not built for export. A projection whose input or
    output width is not a multiple of 128 keeps its bf16 ``nn.Linear`` with a warning.

    Args:
        config: Same inference configuration as ``TransformerInferLayer``.
        layer_idx: Index into the shared per-layer KV-cache table collection.
        device: Parameter device; defaults to the current CUDA device.
    """

    def __init__(self, config, layer_idx, device=None):
        self._check_supported(config.bf16, config.export_mode)
        super().__init__(config, layer_idx, device)
        self._swap_linears()

    @staticmethod
    def _check_supported(is_bf16, export_mode):
        if deep_gemm is None:
            raise ImportError(
                "Fp8TransformerInferLayer needs DeepGEMM (deep_gemm), which the Docker "
                "image builds from third_party/DeepGEMM. Import failed with: "
                f"{_DEEP_GEMM_IMPORT_ERROR}"
            )
        if not is_bf16:
            raise ValueError("Fp8TransformerInferLayer requires bf16")
        if export_mode:
            raise ValueError("Fp8TransformerInferLayer cannot be exported")

    def _swap_linears(self):
        self.qkv = self._to_fp8(self.qkv, "qkv")
        self.proj = self._to_fp8(self.proj, "proj")
        self.ffn[0] = self._to_fp8(self.ffn[0], "ffn.0")
        self.ffn[2] = self._to_fp8(self.ffn[2], "ffn.2")

    @staticmethod
    def _to_fp8(linear, name):
        if not _supports_fp8(linear):
            warnings.warn(
                f"{name} ({linear.in_features} -> {linear.out_features}) is not FP8 "
                f"eligible (CUDA bf16, widths multiples of {GROUP_SIZE}); keeping bf16"
            )
            return linear
        return Fp8Linear.from_linear(linear)

    def refresh_fp8_weights(self):
        """Re-quantize every FP8 projection after its bf16 weight was updated."""
        for module in self.modules():
            if isinstance(module, Fp8Linear):
                module.refresh_fp8_weights()


class Fp8FusedTransformerInferLayer(Fp8TransformerInferLayer):
    """``Fp8TransformerInferLayer`` that quantizes inside the neighbouring kernels.

    The LayerNorm in front of ``qkv`` and the one in front of ``ffn.0`` quantize their
    own output, and the GELU in front of ``ffn.2`` quantizes its output together with the
    ``ffn.0`` bias. Only the attention output in front of ``proj`` is quantized by a
    standalone kernel, because attention is a separate kernel that writes bf16. The
    biases of ``qkv``, ``proj`` and ``ffn.2`` are still separate elementwise adds. If a
    projection is not FP8 eligible the layer falls back to ``Fp8TransformerInferLayer``.
    """

    @property
    def _fused(self):
        projections = (self.qkv, self.proj, self.ffn[0], self.ffn[2])
        return all(isinstance(p, Fp8Linear) for p in projections)

    def _mixed_qkv(self, hidden):
        if not self._fused:
            return super()._mixed_qkv(hidden)
        norm = self.input_norm
        ue8m0 = _uses_ue8m0(hidden.device)
        qx, scale_x = triton_layernorm_quant_1x128(
            hidden, norm.weight, norm.bias, norm.eps, ue8m0
        )
        return self.qkv.gemm(qx, scale_x).add_(self.qkv.bias)

    def finish(self, hidden, attention):
        if not self._fused:
            return super().finish(hidden, attention)
        ue8m0 = _uses_ue8m0(hidden.device)
        qa, scale_a = triton_quant_1x128(attention, ue8m0)
        projected = self.proj.gemm(qa, scale_a).add_(self.proj.bias)
        hidden = hidden + projected if self._residual else projected
        norm = self.ffn_norm
        qh, scale_h = triton_layernorm_quant_1x128(
            hidden, norm.weight, norm.bias, norm.eps, ue8m0
        )
        qg, scale_g = triton_bias_gelu_quant_1x128(
            self.ffn[0].gemm(qh, scale_h), self.ffn[0].bias, ue8m0
        )
        output = self.ffn[2].gemm(qg, scale_g).add_(self.ffn[2].bias)
        return hidden + output if self._residual else output
