"""Qwen3 single-layer prefill skeleton.

The classes in this module are optional torch modules. They are intentionally
kept separate from metadata/loader modules so the package can still be imported
in environments without torch.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

from gr_inference.gr_kernels import (
    CAP_FUSED_ADD_RMSNORM,
    CAP_FUSED_MLP,
    CAP_PACKED_GEMM,
    CAP_QK_NORM_ROPE,
    CAP_RMSNORM,
    CAP_ROPE,
    FusedMLP,
    TorchFusedMLPBackend,
    default_kernel_selection_policy,
)
from gr_inference.gr_kernels.package_imports import (
    load_trtllm_kernel_registration as _load_trtllm_kernel_registration,
)
from gr_inference.gr_kernels.prefill import PrefillAttention
from gr_inference.gr_kv import ContextKV
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_runtime.decode_kv import BeamKVWriter
from gr_inference.gr_runtime.engine import GRDecodeEngine
from gr_inference.gr_runtime.generation import GRGenerationState
from gr_inference.gr_runtime.prefill import GRPrefillRunner

try:  # pragma: no cover - import availability depends on runtime container
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("Qwen3 layer modules require torch")


if nn is not None:

    class Qwen3RMSNorm(nn.Module):
        """RMSNorm used by Qwen-family decoder layers."""

        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, hidden_states):
            flashinfer_rmsnorm = _flashinfer_rmsnorm()
            if (
                _selected_kernel_backend(CAP_RMSNORM) == "flashinfer"
                and flashinfer_rmsnorm is not None
                and _is_cuda_tensor(hidden_states)
            ):
                _record_flashinfer_call("rmsnorm")
                original_shape = hidden_states.shape
                norm_input = _reshape_for_flashinfer_rmsnorm(hidden_states)
                norm_output = flashinfer_rmsnorm(norm_input, self.weight, self.eps)
                return norm_output.reshape(original_shape)

            functional_rms_norm = getattr(torch.nn.functional, "rms_norm", None)
            if functional_rms_norm is not None:
                return functional_rms_norm(
                    hidden_states,
                    (hidden_states.shape[-1],),
                    self.weight,
                    self.eps,
                )

            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
            variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return (self.weight * hidden_states).to(input_dtype)

    class Qwen3LayerOps(ABC, nn.Module):
        """Backend interface for replaceable Qwen3 layer operations."""

        @abstractmethod
        def input_norm(self, hidden_states):
            raise NotImplementedError

        @abstractmethod
        def qkv(self, hidden_states):
            raise NotImplementedError

        @abstractmethod
        def qk_norm_rope(self, q, k, *, position_ids=None):
            raise NotImplementedError

        def prepare_prefill_attention_inputs(
            self,
            q,
            k,
            v,
            context_kv: ContextKV,
            *,
            layer_idx: int,
        ):
            return None

        @abstractmethod
        def o_proj(self, attention_output):
            raise NotImplementedError

        @abstractmethod
        def post_attention_norm(self, hidden_states):
            raise NotImplementedError

        def post_attention_residual_norm(self, residual, attention_output):
            projected = self.o_proj(attention_output)
            return self.post_attention_residual_norm_projected(residual, projected)

        def post_attention_residual_norm_projected(self, residual, projected):
            hidden_states = residual + projected
            return hidden_states, self.post_attention_norm(hidden_states)

        @abstractmethod
        def mlp(self, hidden_states):
            raise NotImplementedError

    class TorchQwen3LayerOps(Qwen3LayerOps):
        """Torch eager implementation of Qwen3 layer operations.

        This class is the baseline backend. Fused RMSNorm, packed QKV GEMM,
        fused q/k norm + RoPE, and fused MLP can replace this interface later.
        """

        def __init__(
            self,
            config: Qwen3GRConfig,
            *,
            dtype: Any | None = None,
        ) -> None:
            super().__init__()
            self.config = config
            hidden_size = config.hidden_size
            intermediate_size = config.intermediate_size or hidden_size * 4
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_kv_heads * config.head_dim
            self.q_size = q_size
            self.kv_size = kv_size
            self.intermediate_size = intermediate_size
            linear_kwargs = {"bias": False}
            if dtype is not None:
                linear_kwargs["dtype"] = dtype

            self.input_layernorm = Qwen3RMSNorm(hidden_size, config.rms_norm_eps)
            self.qkv_proj = nn.Linear(
                hidden_size,
                q_size + 2 * kv_size,
                **linear_kwargs,
            )
            self.q_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
            self.out_proj = nn.Linear(q_size, hidden_size, **linear_kwargs)
            self.post_attention_layernorm = Qwen3RMSNorm(
                hidden_size, config.rms_norm_eps
            )
            self.gate_up_proj = nn.Linear(
                hidden_size,
                2 * intermediate_size,
                **linear_kwargs,
            )
            self.down_proj = nn.Linear(intermediate_size, hidden_size, **linear_kwargs)
            self.fused_mlp = FusedMLP(TorchFusedMLPBackend())
            self._last_qkv_for_trtllm_qk_norm_rope = None

            if dtype is not None:
                self.input_layernorm.to(dtype=dtype)
                self.q_norm.to(dtype=dtype)
                self.k_norm.to(dtype=dtype)
                self.post_attention_layernorm.to(dtype=dtype)

        def input_norm(self, hidden_states):
            return self.input_layernorm(hidden_states)

        def qkv(self, hidden_states):
            # Current packed GEMM backend is torch. The capability lookup keeps
            # this call site ready for CUTLASS/TRT-LLM/Triton replacement.
            selected_backend = _selected_kernel_backend(CAP_PACKED_GEMM)
            batch, seq_len, _hidden = hidden_states.shape
            qkv = (
                _apply_trtllm_packed_gemm(
                    hidden_states,
                    self.qkv_proj.weight,
                    self.qkv_proj.bias,
                )
                if selected_backend == "trtllm_aligned"
                and _trtllm_packed_gemm_scope_enabled("attention")
                else None
            )
            if qkv is None:
                qkv = _linear_project(self.qkv_proj, hidden_states)
            q_raw, k_raw, v_raw = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            q = q_raw.view(
                batch,
                seq_len,
                self.config.num_attention_heads,
                self.config.head_dim,
            )
            k = k_raw.view(
                batch,
                seq_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            )
            v = v_raw.view(
                batch,
                seq_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            )
            self._last_qkv_for_trtllm_qk_norm_rope = qkv
            return q, k, v

        def qk_norm_rope(self, q, k, *, position_ids=None):
            selected_backend = _selected_kernel_backend(CAP_QK_NORM_ROPE)
            qk_phase = "decode" if position_ids is not None else "prefill"
            fused_qk_norm_rope = (
                _trtllm_fused_qk_norm_rope()
                if selected_backend == "trtllm_aligned"
                and _trtllm_qk_norm_rope_phase_enabled(qk_phase)
                else None
            )
            if (
                fused_qk_norm_rope is not None
                and _is_cuda_tensor(q)
                and _is_cuda_tensor(k)
                and self._last_qkv_for_trtllm_qk_norm_rope is not None
            ):
                fused_output = _apply_trtllm_fused_qk_norm_rope(
                    fused_qk_norm_rope,
                    self._last_qkv_for_trtllm_qk_norm_rope,
                    q,
                    k,
                    num_attention_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_kv_heads,
                    head_dim=self.config.head_dim,
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    q_norm_weight=self.q_norm.weight,
                    k_norm_weight=self.k_norm.weight,
                    eps=self.q_norm.eps,
                    rope_theta=self.config.rope_theta,
                    position_ids=position_ids,
                )
                if fused_output is not None:
                    return fused_output
            if _apply_sglang_fused_qknorm(
                q,
                k,
                self.q_norm.weight,
                self.k_norm.weight,
                head_dim=self.config.head_dim,
                eps=self.q_norm.eps,
            ):
                return self.rope_only(q, k, position_ids=position_ids)
            q = self.q_norm(q)
            k = self.k_norm(k)
            return self.rope_only(q, k, position_ids=position_ids)

        def prepare_prefill_attention_inputs(
            self,
            q,
            k,
            v,
            context_kv: ContextKV,
            *,
            layer_idx: int,
        ):
            if self._last_qkv_for_trtllm_qk_norm_rope is None:
                return None
            return _apply_packed_qkv_prefill_kv_write(
                q,
                k,
                self._last_qkv_for_trtllm_qk_norm_rope,
                context_kv,
                layer_idx=layer_idx,
                num_attention_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_kv_heads,
                head_dim=self.config.head_dim,
            )

        def q_norm_only(self, q):
            return self.q_norm(q)

        def k_norm_only(self, k):
            return self.k_norm(k)

        def rope_only(self, q, k, *, position_ids=None):
            return apply_qwen3_rope(
                q,
                k,
                rope_theta=self.config.rope_theta,
                position_ids=position_ids,
            )

        def o_proj(self, attention_output):
            selected_backend = _selected_kernel_backend(CAP_PACKED_GEMM)
            batch, seq_len = attention_output.shape[:2]
            flattened = attention_output.reshape(batch, seq_len, -1)
            projected = (
                _apply_trtllm_packed_gemm(
                    flattened,
                    self.out_proj.weight,
                    self.out_proj.bias,
                )
                if selected_backend == "trtllm_aligned"
                and _trtllm_packed_gemm_scope_enabled("attention")
                else None
            )
            return (
                projected
                if projected is not None
                else _linear_project(
                    self.out_proj,
                    flattened,
                )
            )

        def post_attention_norm(self, hidden_states):
            return self.post_attention_layernorm(hidden_states)

        def post_attention_residual_norm(self, residual, attention_output):
            projected = self.o_proj(attention_output)
            return self.post_attention_residual_norm_projected(residual, projected)

        def post_attention_residual_norm_projected(self, residual, projected):
            fused_add_rmsnorm = _flashinfer_fused_add_rmsnorm()
            if (
                _selected_kernel_backend(CAP_FUSED_ADD_RMSNORM) == "flashinfer"
                and fused_add_rmsnorm is not None
                and _is_cuda_tensor(residual)
                and _is_cuda_tensor(projected)
            ):
                fused_output = _apply_flashinfer_fused_add_rmsnorm(
                    fused_add_rmsnorm,
                    projected,
                    residual,
                    self.post_attention_layernorm.weight,
                    self.post_attention_layernorm.eps,
                )
                if fused_output is not None:
                    return fused_output
            return super().post_attention_residual_norm_projected(residual, projected)

        def mlp(self, hidden_states):
            selected_backend = _selected_kernel_backend(CAP_FUSED_MLP)
            if selected_backend == "sgl_kernel":
                fused_silu_mul = _sgl_kernel_silu_and_mul()
                if fused_silu_mul is not None and _is_cuda_tensor(hidden_states):
                    fused_output = _apply_sgl_kernel_gated_mlp(
                        fused_silu_mul,
                        hidden_states,
                        self,
                    )
                    if fused_output is not None:
                        return fused_output
            if selected_backend == "torch_compile":
                compiled_output = _apply_torch_compile_gated_mlp(
                    hidden_states,
                    self.gate_up_proj.weight,
                    self.down_proj.weight,
                    intermediate_size=self.intermediate_size,
                )
                if compiled_output is not None:
                    return compiled_output
            if selected_backend == "trtllm_aligned":
                packed_silu_mul = _gr_trtllm_packed_silu_mul()
                if packed_silu_mul is not None and _is_cuda_tensor(hidden_states):
                    fused_output = _apply_packed_silu_mul_mlp(
                        packed_silu_mul,
                        hidden_states,
                        self,
                    )
                    if fused_output is not None:
                        return fused_output
            return self.fused_mlp(hidden_states, self)

        def gate_up_packed(self, hidden_states):
            selected_backend = _selected_kernel_backend(CAP_PACKED_GEMM)
            gate_up = (
                _apply_trtllm_packed_gemm(
                    hidden_states,
                    self.gate_up_proj.weight,
                    self.gate_up_proj.bias,
                )
                if selected_backend == "trtllm_aligned"
                and _trtllm_packed_gemm_scope_enabled("mlp")
                else None
            )
            return (
                gate_up
                if gate_up is not None
                else _linear_project(
                    self.gate_up_proj,
                    hidden_states,
                )
            )

        def gate_up(self, hidden_states):
            gate_up = self.gate_up_packed(hidden_states)
            return gate_up.split(
                [self.intermediate_size, self.intermediate_size],
                dim=-1,
            )

        def silu_mul(self, gate, up):
            _selected_kernel_backend(CAP_FUSED_MLP)
            return torch.nn.functional.silu(gate) * up

        def down_proj_only(self, hidden_states):
            selected_backend = _selected_kernel_backend(CAP_PACKED_GEMM)
            output = (
                _apply_trtllm_packed_gemm(
                    hidden_states,
                    self.down_proj.weight,
                    self.down_proj.bias,
                )
                if selected_backend == "trtllm_aligned"
                and _trtllm_packed_gemm_scope_enabled("mlp")
                else None
            )
            return (
                output
                if output is not None
                else _linear_project(
                    self.down_proj,
                    hidden_states,
                )
            )

        def load_logical_weights(
            self,
            weights: dict[str, Any],
            *,
            layer_idx: int,
            strict: bool = True,
        ) -> None:
            """Load logical tensors produced by Qwen3HFAdapter.load_plan()."""

            prefix = f"layers.{layer_idx}"
            required = {
                f"{prefix}.input_layernorm.weight": self.input_layernorm.weight,
                f"{prefix}.self_attn.o_proj.weight": self.out_proj.weight,
                f"{prefix}.post_attention_layernorm.weight": (
                    self.post_attention_layernorm.weight
                ),
                f"{prefix}.mlp.down_proj.weight": self.down_proj.weight,
            }
            for name, param in required.items():
                self._copy_weight(name, weights, param, strict=strict)

            qkv_name = f"{prefix}.self_attn.qkv_proj.weight"
            if qkv_name in weights:
                self._copy_tensor(self.qkv_proj.weight, weights[qkv_name], qkv_name)
            else:
                self._copy_weight(
                    f"{prefix}.self_attn.q_proj.weight",
                    weights,
                    self.qkv_proj.weight[: self.q_size],
                    strict=strict,
                )
                self._copy_weight(
                    f"{prefix}.self_attn.k_proj.weight",
                    weights,
                    self.qkv_proj.weight[self.q_size : self.q_size + self.kv_size],
                    strict=strict,
                )
                self._copy_weight(
                    f"{prefix}.self_attn.v_proj.weight",
                    weights,
                    self.qkv_proj.weight[self.q_size + self.kv_size :],
                    strict=strict,
                )

            gate_up_name = f"{prefix}.mlp.gate_up_proj.weight"
            if gate_up_name in weights:
                self._copy_tensor(
                    self.gate_up_proj.weight,
                    weights[gate_up_name],
                    gate_up_name,
                )
            else:
                self._copy_weight(
                    f"{prefix}.mlp.gate_proj.weight",
                    weights,
                    self.gate_up_proj.weight[: self.intermediate_size],
                    strict=strict,
                )
                self._copy_weight(
                    f"{prefix}.mlp.up_proj.weight",
                    weights,
                    self.gate_up_proj.weight[self.intermediate_size :],
                    strict=strict,
                )

            self._copy_weight(
                f"{prefix}.self_attn.q_norm.weight",
                weights,
                self.q_norm.weight,
                strict=False,
            )
            self._copy_weight(
                f"{prefix}.self_attn.k_norm.weight",
                weights,
                self.k_norm.weight,
                strict=False,
            )

        def _copy_weight(
            self, name: str, weights: dict[str, Any], param, *, strict: bool
        ) -> None:
            if name not in weights:
                if strict:
                    raise KeyError(f"missing logical tensor: {name}")
                return
            self._copy_tensor(param, weights[name], name)

        @staticmethod
        def _copy_tensor(param, tensor, name: str) -> None:
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    class Qwen3SingleLayerPrefill(nn.Module):
        """One full Qwen3 decoder layer wired to GR prefill runtime.

        This is the first executable bridge from model code into ContextKV and
        the selected prefill attention backend. Fused kernels can replace the
        ``ops`` backend without changing runtime/KV ownership.
        """

        def __init__(
            self,
            config: Qwen3GRConfig,
            *,
            layer_idx: int,
            prefill_attention: PrefillAttention,
            ops: Qwen3LayerOps | None = None,
            dtype: Any | None = None,
        ) -> None:
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.prefill_runner = GRPrefillRunner(prefill_attention)
            self.ops = (
                ops if ops is not None else TorchQwen3LayerOps(config, dtype=dtype)
            )

        def forward_prefill(
            self,
            hidden_states,
            context_kv: ContextKV,
            *,
            timing_recorder: Any | None = None,
            normed_hidden_states: Any | None = None,
            next_input_norm: Any | None = None,
            return_next_normed: bool = False,
        ):
            batch, seq_len, hidden_size = hidden_states.shape
            if hidden_size != self.config.hidden_size:
                raise ValueError(
                    f"hidden size mismatch: expected {self.config.hidden_size}, got {hidden_size}"
                )

            prefix = f"prefill.layer{self.layer_idx}"
            with _timed(timing_recorder, f"{prefix}.total"):
                residual = hidden_states
                with _timed(timing_recorder, f"{prefix}.qkv"):
                    if normed_hidden_states is None:
                        with _timed_fine(timing_recorder, f"{prefix}.input_norm"):
                            normed = self.ops.input_norm(hidden_states)
                    else:
                        normed = normed_hidden_states
                    with _timed_fine(timing_recorder, f"{prefix}.qkv_proj"):
                        q, k, v = self.ops.qkv(normed)
                    with _timed(timing_recorder, f"{prefix}.qk_norm_rope"):
                        q, k = self.ops.qk_norm_rope(q, k)
                    with _timed(timing_recorder, f"{prefix}.qkv_finalize"):
                        prepared_qkv = self.ops.prepare_prefill_attention_inputs(
                            q,
                            k,
                            v,
                            context_kv,
                            layer_idx=self.layer_idx,
                        )
                        write_context_kv = prepared_qkv is None
                        if prepared_qkv is not None:
                            q, k, v = prepared_qkv

                with _timed(timing_recorder, f"{prefix}.attention"):
                    attn_out = self.prefill_runner.run_layer(
                        q=q,
                        k=k,
                        v=v,
                        context_kv=context_kv,
                        layer_idx=self.layer_idx,
                        causal=True,
                        write_context_kv=write_context_kv,
                    )
                with _timed(timing_recorder, f"{prefix}.post_attention"):
                    residual, hidden_states = self.ops.post_attention_residual_norm(
                        residual,
                        attn_out,
                    )
                    with _timed(timing_recorder, f"{prefix}.mlp"):
                        mlp_out = self.ops.mlp(hidden_states)
                        if next_input_norm is not None:
                            fused = _apply_prefill_next_input_norm(
                                mlp_out,
                                residual,
                                next_input_norm,
                            )
                            if fused is not None:
                                hidden_states, next_normed = fused
                            else:
                                hidden_states = residual + mlp_out
                                next_normed = next_input_norm(hidden_states)
                        else:
                            hidden_states = residual + mlp_out
                            next_normed = None
            if return_next_normed:
                return hidden_states, next_normed
            return hidden_states

        def forward_prefill_extend(
            self,
            hidden_states,
            context_kv: ContextKV,
            *,
            prefix_len: int,
            timing_recorder: Any | None = None,
        ):
            batch, suffix_len, hidden_size = hidden_states.shape
            if hidden_size != self.config.hidden_size:
                raise ValueError(
                    f"hidden size mismatch: expected {self.config.hidden_size}, got {hidden_size}"
                )
            if prefix_len < 0 or prefix_len + suffix_len > context_kv.context_len:
                raise ValueError("prefix/suffix lengths exceed ContextKV capacity")

            prefix = f"prefill_extend.layer{self.layer_idx}"
            with _timed(timing_recorder, f"{prefix}.total"):
                residual = hidden_states
                with _timed(timing_recorder, f"{prefix}.qkv"):
                    with _timed_fine(timing_recorder, f"{prefix}.input_norm"):
                        normed = self.ops.input_norm(hidden_states)
                    with _timed_fine(timing_recorder, f"{prefix}.qkv_proj"):
                        q, k, v = self.ops.qkv(normed)
                    with _timed(timing_recorder, f"{prefix}.qk_norm_rope"):
                        position_ids = _suffix_position_ids(
                            q,
                            batch=batch,
                            suffix_len=suffix_len,
                            prefix_len=prefix_len,
                        )
                        q, k = _qk_norm_rope_prefill_extend(
                            self.ops,
                            q,
                            k,
                            position_ids=position_ids,
                        )

                with _timed(timing_recorder, f"{prefix}.kv_write"):
                    _write_context_kv_suffix(
                        context_kv,
                        layer_idx=self.layer_idx,
                        prefix_len=prefix_len,
                        k=k,
                        v=v,
                    )

                with _timed(timing_recorder, f"{prefix}.attention"):
                    attn_out = _prefill_extend_attention(
                        q,
                        context_kv.key[self.layer_idx, :, : prefix_len + suffix_len],
                        context_kv.value[self.layer_idx, :, : prefix_len + suffix_len],
                        prefix_len=prefix_len,
                    )

                with _timed(timing_recorder, f"{prefix}.post_attention"):
                    residual, hidden_states = self.ops.post_attention_residual_norm(
                        residual,
                        attn_out,
                    )
                    with _timed(timing_recorder, f"{prefix}.mlp"):
                        hidden_states = residual + self.ops.mlp(hidden_states)
            return hidden_states

        def forward_decode(
            self,
            hidden_states,
            generation: GRGenerationState,
            decode_engine: GRDecodeEngine,
            *,
            step: int,
            active_beam_width: int | None = None,
            topk_indices: Any | None = None,
            decode_nums: int | None = None,
            return_lse: bool = False,
            backend_name: str = "dsl",
            timing_recorder: Any | None = None,
            position_ids: Any | None = None,
            normed_hidden_states: Any | None = None,
            next_input_norm: Any | None = None,
            return_next_normed: bool = False,
        ):
            batch, beam_width, hidden_size = hidden_states.shape
            if hidden_size != self.config.hidden_size:
                raise ValueError(
                    f"hidden size mismatch: expected {self.config.hidden_size}, got {hidden_size}"
                )
            if active_beam_width is None:
                active_beam_width = generation.fixed_beam_width
            if beam_width != active_beam_width:
                raise ValueError(
                    f"beam width mismatch: expected {active_beam_width}, got {beam_width}"
                )

            with _timed(timing_recorder, f"layer{self.layer_idx}.decode_total"):
                residual = hidden_states
                with _timed(timing_recorder, f"layer{self.layer_idx}.qkv"):
                    if normed_hidden_states is None:
                        with _timed_fine(
                            timing_recorder,
                            f"layer{self.layer_idx}.input_norm",
                        ):
                            normed = self.ops.input_norm(hidden_states)
                    else:
                        normed = normed_hidden_states
                    with _timed_fine(
                        timing_recorder, f"layer{self.layer_idx}.qkv_proj"
                    ):
                        q, k, v = self.ops.qkv(normed)
                    decode_position = generation.prefill.context_len + step
                    if position_ids is None:
                        position_ids = decode_position
                    with _timed(timing_recorder, f"layer{self.layer_idx}.qk_norm_rope"):
                        q, k = self.ops.qk_norm_rope(q, k, position_ids=position_ids)

                with _timed(timing_recorder, f"layer{self.layer_idx}.beam_kv_write"):
                    BeamKVWriter(generation.beam_kv).write_layer_step(
                        layer_idx=self.layer_idx,
                        step=step,
                        active_beam_width=active_beam_width,
                        k=k,
                        v=v,
                    )

                with _timed(timing_recorder, f"layer{self.layer_idx}.decode_attention"):
                    decode_output = decode_engine.decode_attention_step(
                        generation.request_state(),
                        q,
                        layer_idx=self.layer_idx,
                        step=step,
                        active_beam_width=active_beam_width,
                        topk_indices=topk_indices,
                        decode_nums=decode_nums
                        if decode_nums is not None
                        else step + 1,
                        return_lse=return_lse,
                        backend_name=backend_name,
                    )

                with _timed(timing_recorder, f"layer{self.layer_idx}.post_attention"):
                    attn_out = normalize_decode_attention_output(
                        decode_output.attention_output
                    )
                    if _is_fine_timing(timing_recorder):
                        with _timed_fine(
                            timing_recorder, f"layer{self.layer_idx}.o_proj"
                        ):
                            projected = self.ops.o_proj(attn_out)
                        with _timed_fine(
                            timing_recorder, f"layer{self.layer_idx}.post_norm"
                        ):
                            (
                                residual,
                                hidden_states,
                            ) = self.ops.post_attention_residual_norm_projected(
                                residual,
                                projected,
                            )
                    else:
                        residual, hidden_states = self.ops.post_attention_residual_norm(
                            residual,
                            attn_out,
                        )
                    with _timed(timing_recorder, f"layer{self.layer_idx}.mlp"):
                        if (
                            _is_fine_timing(timing_recorder)
                            and isinstance(
                                self.ops,
                                TorchQwen3LayerOps,
                            )
                            and _selected_kernel_backend(CAP_FUSED_MLP)
                            != "trtllm_aligned"
                        ):
                            with _timed_fine(
                                timing_recorder,
                                f"layer{self.layer_idx}.gate_up_proj",
                            ):
                                gate, up = self.ops.gate_up(hidden_states)
                            with _timed_fine(
                                timing_recorder,
                                f"layer{self.layer_idx}.silu_mul",
                            ):
                                intermediate = self.ops.silu_mul(gate, up)
                            with _timed_fine(
                                timing_recorder,
                                f"layer{self.layer_idx}.down_proj",
                            ):
                                mlp_out = self.ops.down_proj_only(intermediate)
                        else:
                            mlp_out = self.ops.mlp(hidden_states)
                        if next_input_norm is not None:
                            fused = _apply_decode_next_input_norm(
                                mlp_out,
                                residual,
                                next_input_norm,
                            )
                            if fused is not None:
                                hidden_states, next_normed = fused
                            else:
                                hidden_states = residual + mlp_out
                                next_normed = next_input_norm(hidden_states)
                        else:
                            hidden_states = residual + mlp_out
                            next_normed = None
            if return_next_normed:
                return hidden_states, next_normed
            return hidden_states

else:

    class Qwen3RMSNorm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    class Qwen3LayerOps:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    class TorchQwen3LayerOps:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    class Qwen3SingleLayerPrefill:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def normalize_decode_attention_output(attention_output):
    """Normalize decode attention output to [B, W, Hq, D]."""

    if isinstance(attention_output, tuple):
        attention_output = attention_output[0]
    if hasattr(attention_output, "dim"):
        if attention_output.dim() == 5:
            if attention_output.shape[1] != 1:
                raise ValueError(
                    "decode attention output with rank 5 must have seqlen_q=1"
                )
            return attention_output[:, 0]
        if attention_output.dim() == 4:
            return attention_output
    return attention_output


def _suffix_position_ids(q, *, batch: int, suffix_len: int, prefix_len: int):
    positions = torch.arange(
        prefix_len,
        prefix_len + suffix_len,
        device=q.device,
        dtype=torch.int32,
    )
    return positions.unsqueeze(0).expand(batch, suffix_len)


def _qk_norm_rope_prefill_extend(ops, q, k, *, position_ids):
    if all(hasattr(ops, name) for name in ("q_norm_only", "k_norm_only", "rope_only")):
        q = ops.q_norm_only(q)
        k = ops.k_norm_only(k)
        return ops.rope_only(q, k, position_ids=position_ids)
    return ops.qk_norm_rope(q, k, position_ids=position_ids)


def _write_context_kv_suffix(
    context_kv: ContextKV,
    *,
    layer_idx: int,
    prefix_len: int,
    k,
    v,
) -> None:
    suffix_len = int(k.shape[1])
    key_slice = context_kv.key[
        layer_idx,
        :,
        prefix_len : prefix_len + suffix_len,
    ]
    value_slice = context_kv.value[
        layer_idx,
        :,
        prefix_len : prefix_len + suffix_len,
    ]
    key_slice.copy_(k)
    value_slice.copy_(v)


def _prefill_extend_attention(q, k, v, *, prefix_len: int):
    if _env_flag("GR_INFERENCE_PREFILL_EXTEND_FLASH"):
        flash_hidden = _flash_attn_prefill_extend(q, k, v)
        if flash_hidden is not None:
            return flash_hidden
    return _torch_prefill_extend_attention(q, k, v, prefix_len=prefix_len)


def _flash_attn_prefill_extend(q, k, v):
    if not _is_cuda_tensor(q):
        return None
    try:
        from flash_attn import flash_attn_func
    except Exception:
        return None
    try:
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    except Exception:
        return None


def _torch_prefill_extend_attention(q, k, v, *, prefix_len: int):
    import torch.nn.functional as F

    batch, suffix_len, num_q_heads, _head_dim = q.shape
    total_len = k.shape[1]
    num_kv_heads = k.shape[2]
    qhead_per_kv = num_q_heads // num_kv_heads
    if qhead_per_kv != 1:
        k = k.repeat_interleave(qhead_per_kv, dim=2)
        v = v.repeat_interleave(qhead_per_kv, dim=2)

    q_sdpa = q.permute(0, 2, 1, 3)
    k_sdpa = k.permute(0, 2, 1, 3)
    v_sdpa = v.permute(0, 2, 1, 3)
    mask = _prefill_extend_attention_mask(
        device=q.device,
        suffix_len=suffix_len,
        total_len=total_len,
        prefix_len=prefix_len,
    )
    out = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return (
        out.permute(0, 2, 1, 3)
        .contiguous()
        .view(
            batch,
            suffix_len,
            num_q_heads,
            q.shape[-1],
        )
    )


def _prefill_extend_attention_mask(
    *,
    device,
    suffix_len: int,
    total_len: int,
    prefix_len: int,
):
    query_positions = torch.arange(suffix_len, device=device).unsqueeze(1)
    key_positions = torch.arange(total_len, device=device).unsqueeze(0)
    return key_positions <= (prefix_len + query_positions)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _linear_project(linear, hidden_states):
    return _functional_linear_project(
        hidden_states,
        linear.weight,
        linear.bias,
    )


def _functional_linear_project(hidden_states, weight, bias=None):
    if (
        _env_flag("GR_INFERENCE_DISABLE_FLATTEN_LINEAR")
        or not hasattr(hidden_states, "dim")
        or hidden_states.dim() <= 2
    ):
        return torch.nn.functional.linear(hidden_states, weight, bias)
    original_shape = tuple(hidden_states.shape[:-1])
    flat_input = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_output = torch.nn.functional.linear(flat_input, weight, bias)
    return flat_output.reshape(*original_shape, flat_output.shape[-1])


def _timed(timing_recorder, name: str):
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.section(name)


def _is_fine_timing(timing_recorder) -> bool:
    return getattr(timing_recorder, "detail", "coarse") == "fine"


def _timed_fine(timing_recorder, name: str):
    if not _is_fine_timing(timing_recorder):
        return nullcontext()
    return _timed(timing_recorder, name)


_FLASHINFER_NORM = None
_FLASHINFER_NORM_CHECKED = False
_FLASHINFER_FUSED_ADD_RMSNORM = None
_FLASHINFER_FUSED_ADD_RMSNORM_CHECKED = False
_FLASHINFER_ROPE = None
_FLASHINFER_ROPE_CHECKED = False
_TRTLLM_FUSED_QK_NORM_ROPE = None
_TRTLLM_FUSED_QK_NORM_ROPE_CHECKED = False
_GR_TRTLLM_PACKED_SILU_MUL = None
_GR_TRTLLM_PACKED_SILU_MUL_CHECKED = False
_GR_TRTLLM_PACKED_SILU_MUL_FAILED = False
_TRTLLM_PACKED_GEMM = None
_TRTLLM_PACKED_GEMM_CHECKED = False
_TRTLLM_PACKED_GEMM_KIND = None
_SGLANG_FUSED_QKNORM = None
_SGLANG_CAN_USE_FUSED_QKNORM = None
_SGLANG_FUSED_QKNORM_CHECKED = False
_SGLANG_FUSED_QKNORM_FAILED = False
_SGLANG_ROPE_INPLACE = None
_SGLANG_ROPE_INPLACE_CHECKED = False
_SGLANG_ROPE_INPLACE_FAILED = False
_SGL_KERNEL_SILU_AND_MUL = None
_SGL_KERNEL_SILU_AND_MUL_CHECKED = False
_SGL_KERNEL_SILU_AND_MUL_FAILED = False
_TORCH_COMPILE_GATED_MLP = None
_TORCH_COMPILE_GATED_MLP_FAILED = False
_FLASHINFER_CALLS: dict[str, int] = {}
_ROPE_COS_SIN_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}
_ROPE_POSITION_IDS_CACHE: dict[tuple[Any, ...], Any] = {}
_SGLANG_ROPE_COS_SIN_CACHE: dict[tuple[Any, ...], Any] = {}


def apply_qwen3_rope(q, k, *, rope_theta: float = 1_000_000.0, position_ids=None):
    """Apply rotary position embedding to q/k tensors shaped [B, S, H, D]."""

    _require_torch()
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k head_dim mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    if _apply_sglang_rope_inplace(
        q,
        k,
        rope_theta=rope_theta,
        position_ids=position_ids,
    ):
        return q, k

    flashinfer_rope = _flashinfer_apply_rope_pos_ids()
    if (
        _selected_kernel_backend(CAP_ROPE) == "flashinfer"
        and flashinfer_rope is not None
        and _is_cuda_tensor(q)
        and _is_cuda_tensor(k)
    ):
        rope_output = _apply_flashinfer_rope_pos_ids(
            flashinfer_rope,
            q,
            k,
            rope_theta=rope_theta,
            position_ids=position_ids,
        )
        if rope_output is not None:
            return rope_output

    cos, sin = _rope_cos_sin(
        q,
        head_dim=head_dim,
        rope_theta=rope_theta,
        position_ids=position_ids,
    )
    return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _flashinfer_rmsnorm():
    global _FLASHINFER_NORM, _FLASHINFER_NORM_CHECKED
    if not _FLASHINFER_NORM_CHECKED:
        _FLASHINFER_NORM_CHECKED = True
        try:
            norm_module = importlib.import_module("flashinfer.norm")
            _FLASHINFER_NORM = getattr(norm_module, "rmsnorm", None)
        except Exception:
            _FLASHINFER_NORM = None
    return _FLASHINFER_NORM


def _flashinfer_fused_add_rmsnorm():
    global _FLASHINFER_FUSED_ADD_RMSNORM, _FLASHINFER_FUSED_ADD_RMSNORM_CHECKED
    if not _FLASHINFER_FUSED_ADD_RMSNORM_CHECKED:
        _FLASHINFER_FUSED_ADD_RMSNORM_CHECKED = True
        try:
            norm_module = importlib.import_module("flashinfer.norm")
            _FLASHINFER_FUSED_ADD_RMSNORM = getattr(
                norm_module,
                "fused_add_rmsnorm",
                None,
            )
        except Exception:
            _FLASHINFER_FUSED_ADD_RMSNORM = None
    return _FLASHINFER_FUSED_ADD_RMSNORM


def _flashinfer_apply_rope_pos_ids():
    global _FLASHINFER_ROPE, _FLASHINFER_ROPE_CHECKED
    if not _FLASHINFER_ROPE_CHECKED:
        _FLASHINFER_ROPE_CHECKED = True
        try:
            rope_module = importlib.import_module("flashinfer.rope")
            _FLASHINFER_ROPE = getattr(rope_module, "apply_rope_pos_ids", None)
        except Exception:
            _FLASHINFER_ROPE = None
    return _FLASHINFER_ROPE


def _trtllm_fused_qk_norm_rope():
    global _TRTLLM_FUSED_QK_NORM_ROPE, _TRTLLM_FUSED_QK_NORM_ROPE_CHECKED
    if not _TRTLLM_FUSED_QK_NORM_ROPE_CHECKED:
        _TRTLLM_FUSED_QK_NORM_ROPE_CHECKED = True
        try:
            _load_trtllm_kernel_registration()
            gr_trtllm_ops = getattr(torch.ops, "gr_trtllm", None)
            if gr_trtllm_ops is not None:
                _TRTLLM_FUSED_QK_NORM_ROPE = getattr(
                    gr_trtllm_ops,
                    "fused_qk_norm_rope",
                    None,
                )
            if _TRTLLM_FUSED_QK_NORM_ROPE is not None:
                return _TRTLLM_FUSED_QK_NORM_ROPE
            _TRTLLM_FUSED_QK_NORM_ROPE = getattr(
                torch.ops.trtllm,
                "fused_qk_norm_rope",
                None,
            )
        except Exception:
            _TRTLLM_FUSED_QK_NORM_ROPE = None
    return _TRTLLM_FUSED_QK_NORM_ROPE


def _trtllm_qk_norm_rope_phase_enabled(phase: str) -> bool:
    setting = os.environ.get("GR_INFERENCE_TRTLLM_QK_NORM_ROPE_PHASE", "all")
    enabled = {part.strip().lower() for part in setting.split(",") if part.strip()}
    if not enabled:
        enabled = {"decode"}
    return "all" in enabled or phase.lower() in enabled


def _trtllm_packed_gemm_scope_enabled(scope: str) -> bool:
    setting = os.environ.get("GR_INFERENCE_TRTLLM_PACKED_GEMM_SCOPE", "all")
    enabled = {part.strip().lower() for part in setting.split(",") if part.strip()}
    if not enabled:
        enabled = {"all"}
    return "all" in enabled or scope.lower() in enabled


def _gr_trtllm_packed_silu_mul():
    global _GR_TRTLLM_PACKED_SILU_MUL
    global _GR_TRTLLM_PACKED_SILU_MUL_CHECKED
    if _GR_TRTLLM_PACKED_SILU_MUL_FAILED:
        return None
    if not _GR_TRTLLM_PACKED_SILU_MUL_CHECKED:
        _GR_TRTLLM_PACKED_SILU_MUL_CHECKED = True
        if os.environ.get("GR_INFERENCE_GR_TRTLLM_KERNELS_JIT", "1") != "1":
            return None
        try:
            module = importlib.import_module("gr_inference_trtllm_kernels.qwen3")
            extension = module._cuda_extension()
            _GR_TRTLLM_PACKED_SILU_MUL = getattr(
                extension,
                "silu_and_mul_packed_cuda",
                None,
            )
        except Exception:
            _GR_TRTLLM_PACKED_SILU_MUL = None
    return _GR_TRTLLM_PACKED_SILU_MUL


def _trtllm_packed_gemm():
    global _TRTLLM_PACKED_GEMM, _TRTLLM_PACKED_GEMM_CHECKED, _TRTLLM_PACKED_GEMM_KIND
    if not _TRTLLM_PACKED_GEMM_CHECKED:
        _TRTLLM_PACKED_GEMM_CHECKED = True
        try:
            _load_trtllm_kernel_registration()
            gr_trtllm_ops = getattr(torch.ops, "gr_trtllm", None)
            if gr_trtllm_ops is not None:
                _TRTLLM_PACKED_GEMM = getattr(
                    gr_trtllm_ops,
                    "packed_gemm",
                    None,
                )
            if _TRTLLM_PACKED_GEMM is not None:
                _TRTLLM_PACKED_GEMM_KIND = "gr_trtllm"
                return _TRTLLM_PACKED_GEMM
            trtllm_ops = getattr(torch.ops, "trtllm", None)
            _TRTLLM_PACKED_GEMM = (
                getattr(trtllm_ops, "cublas_mm", None)
                if trtllm_ops is not None
                else None
            )
            if _TRTLLM_PACKED_GEMM is not None:
                _TRTLLM_PACKED_GEMM_KIND = "trtllm_cublas_mm"
        except Exception:
            _TRTLLM_PACKED_GEMM = None
            _TRTLLM_PACKED_GEMM_KIND = None
    return _TRTLLM_PACKED_GEMM


def _apply_sglang_fused_qknorm(
    q,
    k,
    q_weight,
    k_weight,
    *,
    head_dim: int,
    eps: float,
) -> bool:
    global _SGLANG_FUSED_QKNORM_FAILED
    if not _is_cuda_tensor(q) or not _is_cuda_tensor(k):
        return False
    ops = _sglang_fused_qknorm_ops()
    if ops is None:
        return False
    can_use, fused = ops
    try:
        if not can_use(int(head_dim), q.dtype):
            return False
        q_heads = _view_qk_heads_by_token(q, head_dim=int(head_dim))
        k_heads = _view_qk_heads_by_token(k, head_dim=int(head_dim))
        fused(
            q=q_heads,
            k=k_heads,
            q_weight=q_weight,
            k_weight=k_weight,
            head_dim=int(head_dim),
            eps=float(eps),
        )
        return True
    except Exception:
        _SGLANG_FUSED_QKNORM_FAILED = True
        return False


def _sglang_fused_qknorm_ops():
    global _SGLANG_FUSED_QKNORM
    global _SGLANG_CAN_USE_FUSED_QKNORM
    global _SGLANG_FUSED_QKNORM_CHECKED
    if _SGLANG_FUSED_QKNORM_FAILED:
        return None
    if not _SGLANG_FUSED_QKNORM_CHECKED:
        _SGLANG_FUSED_QKNORM_CHECKED = True
        try:
            module = importlib.import_module("sglang.jit_kernel.norm")
            _SGLANG_CAN_USE_FUSED_QKNORM = getattr(
                module,
                "can_use_fused_inplace_qknorm",
                None,
            )
            _SGLANG_FUSED_QKNORM = getattr(module, "fused_inplace_qknorm", None)
        except Exception:
            _SGLANG_CAN_USE_FUSED_QKNORM = None
            _SGLANG_FUSED_QKNORM = None
    if _SGLANG_CAN_USE_FUSED_QKNORM is None or _SGLANG_FUSED_QKNORM is None:
        return None
    return _SGLANG_CAN_USE_FUSED_QKNORM, _SGLANG_FUSED_QKNORM


def _apply_sglang_rope_inplace(q, k, *, rope_theta: float, position_ids=None) -> bool:
    global _SGLANG_ROPE_INPLACE_FAILED
    if not _is_cuda_tensor(q) or q.dtype != torch.bfloat16:
        return False
    if position_ids is not None and not isinstance(position_ids, int):
        return False
    rope = _sglang_rope_inplace()
    if rope is None:
        return False
    try:
        batch, seq_len = q.shape[:2]
        head_dim = int(q.shape[-1])
        max_position = seq_len if position_ids is None else int(position_ids) + 1
        cos_sin_cache = _sglang_rope_cos_sin_cache(
            q,
            head_dim=head_dim,
            rope_theta=rope_theta,
            max_position=max_position,
        )
        positions = _flatten_rope_position_ids(
            q,
            batch=batch,
            seq_len=seq_len,
            position_ids=position_ids,
        )
        q_heads = _view_qk_heads_by_token(q, head_dim=head_dim)
        k_heads = _view_qk_heads_by_token(k, head_dim=head_dim)
        rope(
            q_heads,
            k_heads,
            cos_sin_cache,
            positions,
            is_neox=True,
            rope_dim=head_dim,
        )
        return True
    except Exception:
        _SGLANG_ROPE_INPLACE_FAILED = True
        return False


def _sglang_rope_inplace():
    global _SGLANG_ROPE_INPLACE
    global _SGLANG_ROPE_INPLACE_CHECKED
    if _SGLANG_ROPE_INPLACE_FAILED:
        return None
    if not _SGLANG_ROPE_INPLACE_CHECKED:
        _SGLANG_ROPE_INPLACE_CHECKED = True
        try:
            module = importlib.import_module("sglang.jit_kernel.rope")
            _SGLANG_ROPE_INPLACE = getattr(module, "apply_rope_inplace", None)
        except Exception:
            _SGLANG_ROPE_INPLACE = None
    return _SGLANG_ROPE_INPLACE


def _sglang_rope_cos_sin_cache(
    q, *, head_dim: int, rope_theta: float, max_position: int
):
    device = q.device
    device_key = (device.type, device.index)
    cache_key = (
        device_key,
        head_dim,
        float(rope_theta),
        max_position,
    )
    cached = _SGLANG_ROPE_COS_SIN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    inv_freq = _rope_inv_freq(
        head_dim=head_dim,
        rope_theta=rope_theta,
        device=device,
        dtype=torch.float32,
    )
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()
    _SGLANG_ROPE_COS_SIN_CACHE[cache_key] = cache
    return cache


def _view_qk_heads_by_token(x, *, head_dim: int):
    if x.dim() == 4:
        batch, seq_len, heads, _dim = x.shape
        return x.view(batch * seq_len, heads, head_dim)
    return x.view(x.shape[0], -1, head_dim)


def _sgl_kernel_silu_and_mul():
    global _SGL_KERNEL_SILU_AND_MUL
    global _SGL_KERNEL_SILU_AND_MUL_CHECKED
    if _SGL_KERNEL_SILU_AND_MUL_FAILED:
        return None
    if not _SGL_KERNEL_SILU_AND_MUL_CHECKED:
        _SGL_KERNEL_SILU_AND_MUL_CHECKED = True
        try:
            module = importlib.import_module("sgl_kernel")
            _SGL_KERNEL_SILU_AND_MUL = getattr(module, "silu_and_mul", None)
        except Exception:
            _SGL_KERNEL_SILU_AND_MUL = None
    return _SGL_KERNEL_SILU_AND_MUL


def _apply_sgl_kernel_gated_mlp(fused_silu_mul, hidden_states, ops):
    global _SGL_KERNEL_SILU_AND_MUL_FAILED
    try:
        gate_up = ops.gate_up_packed(hidden_states)
        if gate_up.shape[-1] != 2 * int(ops.intermediate_size):
            return None
        intermediate = fused_silu_mul(gate_up)
        return ops.down_proj_only(intermediate)
    except Exception:
        _SGL_KERNEL_SILU_AND_MUL_FAILED = True
        return None


def _apply_packed_silu_mul_mlp(packed_silu_mul, hidden_states, ops):
    global _GR_TRTLLM_PACKED_SILU_MUL_FAILED
    try:
        gate_up = ops.gate_up_packed(hidden_states)
        if gate_up.shape[-1] != 2 * int(ops.intermediate_size):
            return None
        intermediate = packed_silu_mul(gate_up)
        return ops.down_proj_only(intermediate)
    except Exception:
        _GR_TRTLLM_PACKED_SILU_MUL_FAILED = True
        return None


def _apply_trtllm_packed_gemm(hidden_states, weight, bias=None):
    packed_gemm = _trtllm_packed_gemm()
    if packed_gemm is None or not _is_cuda_tensor(hidden_states):
        return None
    try:
        if _TRTLLM_PACKED_GEMM_KIND == "trtllm_cublas_mm":
            return packed_gemm(
                hidden_states, weight.transpose(0, 1), bias, out_dtype=None
            )
        return packed_gemm(hidden_states, weight, bias)
    except Exception:
        return None


def _apply_packed_qkv_prefill_kv_write(
    q,
    k,
    qkv,
    context_kv: ContextKV,
    *,
    layer_idx: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    if not (_is_cuda_tensor(k) and _is_cuda_tensor(qkv)):
        return None
    try:
        module = importlib.import_module("gr_inference_trtllm_kernels.qwen3")
        write_kv = getattr(module, "write_packed_qkv_prefill_kv", None)
        if write_kv is None:
            return None
        if not write_kv(
            k,
            qkv,
            context_kv.key,
            context_kv.value,
            layer_idx=int(layer_idx),
            num_heads=int(num_attention_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        ):
            return None
        return q, context_kv.key[layer_idx], context_kv.value[layer_idx]
    except Exception:
        return None


def _apply_torch_compile_gated_mlp(
    hidden_states,
    gate_up_weight,
    down_weight,
    *,
    intermediate_size: int,
):
    global _TORCH_COMPILE_GATED_MLP_FAILED
    compiled = _torch_compile_gated_mlp()
    if compiled is None:
        return None
    try:
        return compiled(
            hidden_states, gate_up_weight, down_weight, int(intermediate_size)
        )
    except Exception:
        _TORCH_COMPILE_GATED_MLP_FAILED = True
        return None


def _torch_compile_gated_mlp():
    global _TORCH_COMPILE_GATED_MLP, _TORCH_COMPILE_GATED_MLP_FAILED
    if _TORCH_COMPILE_GATED_MLP is not None:
        return _TORCH_COMPILE_GATED_MLP
    if _TORCH_COMPILE_GATED_MLP_FAILED:
        return None
    if torch is None or not hasattr(torch, "compile"):
        _TORCH_COMPILE_GATED_MLP_FAILED = True
        return None
    try:
        _TORCH_COMPILE_GATED_MLP = torch.compile(
            _torch_compile_gated_mlp_impl,
            mode=os.environ.get(
                "GR_INFERENCE_TORCH_COMPILE_MLP_MODE", "reduce-overhead"
            ),
            dynamic=False,
        )
        return _TORCH_COMPILE_GATED_MLP
    except Exception:
        _TORCH_COMPILE_GATED_MLP_FAILED = True
        return None


def _torch_compile_gated_mlp_impl(
    hidden_states,
    gate_up_weight,
    down_weight,
    intermediate_size: int,
):
    gate_up = torch.matmul(hidden_states, gate_up_weight.transpose(0, 1))
    gate, up = gate_up.split([intermediate_size, intermediate_size], dim=-1)
    intermediate = torch.nn.functional.silu(gate) * up
    return torch.matmul(intermediate, down_weight.transpose(0, 1))


def _is_cuda_tensor(tensor) -> bool:
    return bool(getattr(tensor, "is_cuda", False))


def _selected_kernel_backend(capability: str) -> str | None:
    backend = default_kernel_selection_policy().select(capability)
    return backend.name if backend is not None else None


def flashinfer_call_counts() -> dict[str, int]:
    return dict(_FLASHINFER_CALLS)


def reset_flashinfer_call_counts() -> None:
    _FLASHINFER_CALLS.clear()


def _record_flashinfer_call(name: str) -> None:
    _FLASHINFER_CALLS[name] = _FLASHINFER_CALLS.get(name, 0) + 1
    if os.environ.get("GR_INFERENCE_DEBUG_FLASHINFER") == "1":
        print(f"flashinfer_{name}_calls={_FLASHINFER_CALLS[name]}")


def _reshape_for_flashinfer_rmsnorm(hidden_states):
    if hidden_states.dim() <= 2:
        return hidden_states
    return hidden_states.reshape(-1, hidden_states.shape[-1])


def _apply_flashinfer_fused_add_rmsnorm(
    fused_add_rmsnorm,
    input_tensor,
    residual_tensor,
    weight,
    eps: float,
):
    original_shape = input_tensor.shape
    try:
        norm_input = _reshape_for_flashinfer_rmsnorm(input_tensor)
        residual_input = _reshape_for_flashinfer_rmsnorm(residual_tensor)
        result = fused_add_rmsnorm(norm_input, residual_input, weight, eps)
        _record_flashinfer_call("fused_add_rmsnorm")
        if result is None:
            return residual_input.reshape(original_shape), norm_input.reshape(
                original_shape
            )
        if isinstance(result, tuple):
            if len(result) == 2:
                norm_output, residual_output = result
                return (
                    residual_output.reshape(original_shape),
                    norm_output.reshape(original_shape),
                )
            return None
        return residual_input.reshape(original_shape), result.reshape(original_shape)
    except Exception:
        return None


def _apply_prefill_next_input_norm(mlp_out, residual, next_input_norm):
    fused_add_rmsnorm = _flashinfer_fused_add_rmsnorm()
    if (
        _selected_kernel_backend(CAP_FUSED_ADD_RMSNORM) != "flashinfer"
        or fused_add_rmsnorm is None
        or not _is_cuda_tensor(mlp_out)
        or not _is_cuda_tensor(residual)
    ):
        return None
    return _apply_flashinfer_fused_add_rmsnorm(
        fused_add_rmsnorm,
        mlp_out,
        residual,
        next_input_norm.weight,
        next_input_norm.eps,
    )


def _apply_decode_next_input_norm(mlp_out, residual, next_input_norm):
    exact_fused = _apply_gr_trtllm_exact_fused_add_rmsnorm(
        mlp_out,
        residual,
        next_input_norm,
    )
    if exact_fused is not None:
        return exact_fused
    return _apply_prefill_next_input_norm(mlp_out, residual, next_input_norm)


def _apply_gr_trtllm_exact_fused_add_rmsnorm(mlp_out, residual, next_input_norm):
    if os.environ.get("GR_INFERENCE_DECODE_NEXT_INPUT_NORM_EXACT", "1") != "1":
        return None
    if not (_is_cuda_tensor(mlp_out) and _is_cuda_tensor(residual)):
        return None
    if not getattr(mlp_out, "is_contiguous", lambda: False)():
        return None
    if not getattr(residual, "is_contiguous", lambda: False)():
        return None
    try:
        module = importlib.import_module("gr_inference_trtllm_kernels.qwen3")
        exact_add_rmsnorm = getattr(module, "exact_fused_add_rmsnorm", None)
        if exact_add_rmsnorm is None:
            return None
        result = exact_add_rmsnorm(
            mlp_out,
            residual,
            next_input_norm.weight,
            next_input_norm.eps,
        )
        if isinstance(result, (tuple, list)) and len(result) == 2:
            hidden_states, next_normed = result
            return hidden_states.reshape_as(residual), next_normed.reshape_as(residual)
        return None
    except Exception:
        return None


def _apply_flashinfer_rope_pos_ids(
    flashinfer_rope,
    q,
    k,
    *,
    rope_theta: float,
    position_ids=None,
):
    try:
        batch, seq_len = q.shape[:2]
        q_flat = q.reshape(batch * seq_len, q.shape[-2], q.shape[-1]).contiguous()
        k_flat = k.reshape(batch * seq_len, k.shape[-2], k.shape[-1]).contiguous()
        pos_ids = _flatten_rope_position_ids(
            q,
            batch=batch,
            seq_len=seq_len,
            position_ids=position_ids,
        )
        q_rope, k_rope = flashinfer_rope(
            q_flat,
            k_flat,
            pos_ids,
            rotary_dim=q.shape[-1],
            interleave=False,
            rope_scale=1,
            rope_theta=rope_theta,
        )
        _record_flashinfer_call("rope")
        return q_rope.reshape_as(q), k_rope.reshape_as(k)
    except Exception:
        return None


def _apply_trtllm_fused_qk_norm_rope(
    fused_qk_norm_rope,
    qkv,
    q,
    k,
    *,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    q_norm_weight,
    k_norm_weight,
    eps: float,
    rope_theta: float,
    position_ids=None,
):
    try:
        batch, seq_len = q.shape[:2]
        qkv_flat = qkv.reshape(batch * seq_len, qkv.shape[-1]).contiguous()
        pos_ids = _flatten_rope_position_ids(
            q,
            batch=batch,
            seq_len=seq_len,
            position_ids=position_ids,
        )
        fused_qk_norm_rope(
            qkv_flat,
            num_attention_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            eps,
            q_norm_weight,
            k_norm_weight,
            rope_theta,
            True,
            pos_ids.view(-1),
            1.0,
            0,
            0,
            1.0,
            True,
        )
        qkv_out = qkv_flat.reshape(*qkv.shape[:-1], qkv.shape[-1])
        q_raw, k_raw, _v_raw = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
        return q_raw.reshape_as(q), k_raw.reshape_as(k)
    except Exception:
        return None


def _flatten_rope_position_ids(q, *, batch: int, seq_len: int, position_ids=None):
    if isinstance(position_ids, int):
        device_key = (q.device.type, q.device.index)
        cache_key = (device_key, batch, seq_len, int(position_ids))
        if cache_key not in _ROPE_POSITION_IDS_CACHE:
            _ROPE_POSITION_IDS_CACHE[cache_key] = torch.full(
                (batch * seq_len,),
                int(position_ids),
                dtype=torch.int32,
                device=q.device,
            )
        return _ROPE_POSITION_IDS_CACHE[cache_key]
    if position_ids is None:
        device_key = (q.device.type, q.device.index)
        cache_key = (device_key, batch, seq_len, "seq")
        if cache_key not in _ROPE_POSITION_IDS_CACHE:
            _ROPE_POSITION_IDS_CACHE[cache_key] = torch.arange(
                batch * seq_len,
                dtype=torch.int32,
                device=q.device,
            ).remainder_(seq_len)
        return _ROPE_POSITION_IDS_CACHE[cache_key]
    return position_ids.to(device=q.device, dtype=torch.int32).reshape(-1).contiguous()


def _rope_cos_sin(q, *, head_dim: int, rope_theta: float, position_ids=None):
    device = q.device
    compute_dtype = torch.float32
    device_key = (device.type, device.index)

    if isinstance(position_ids, int):
        cache_key = (device_key, head_dim, float(rope_theta), "pos", int(position_ids))
        if cache_key not in _ROPE_COS_SIN_CACHE:
            inv_freq = _rope_inv_freq(
                head_dim=head_dim,
                rope_theta=rope_theta,
                device=device,
                dtype=compute_dtype,
            )
            freqs = int(position_ids) * inv_freq
            _ROPE_COS_SIN_CACHE[cache_key] = (
                freqs.cos()[None, None, None, :],
                freqs.sin()[None, None, None, :],
            )
        return _ROPE_COS_SIN_CACHE[cache_key]

    inv_freq = _rope_inv_freq(
        head_dim=head_dim,
        rope_theta=rope_theta,
        device=device,
        dtype=compute_dtype,
    )
    if position_ids is None:
        cache_key = (device_key, head_dim, float(rope_theta), "seq", q.shape[1])
        if cache_key not in _ROPE_COS_SIN_CACHE:
            positions = torch.arange(q.shape[1], device=device, dtype=compute_dtype)
            freqs = torch.outer(positions, inv_freq)
            _ROPE_COS_SIN_CACHE[cache_key] = (
                freqs.cos()[None, :, None, :],
                freqs.sin()[None, :, None, :],
            )
        return _ROPE_COS_SIN_CACHE[cache_key]

    positions = position_ids.to(device=device, dtype=compute_dtype)
    freqs = positions[..., None] * inv_freq
    return freqs.cos()[:, :, None, :], freqs.sin()[:, :, None, :]


def _rope_inv_freq(*, head_dim: int, rope_theta: float, device, dtype):
    return 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )


def _apply_rope(x, cos, sin):
    x_float = x.float()
    half = x_float.shape[-1] // 2
    x_first = x_float[..., :half]
    x_second = x_float[..., half:]
    out = torch.empty_like(x_float)
    out[..., :half] = x_first * cos - x_second * sin
    out[..., half:] = x_first * sin + x_second * cos
    return out.to(x.dtype)
