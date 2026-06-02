"""Qwen3 GR model skeleton."""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

from gr_inference.gr_kernels.prefill import PrefillAttention
from gr_inference.gr_kv import ContextKV
from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_models.qwen3.layers import Qwen3RMSNorm, Qwen3SingleLayerPrefill
from gr_inference.gr_runtime import (
    FixedBeamDecodeLoop,
    GRDecodeEngine,
    GRGenerationState,
    PrefillResult,
)

try:  # pragma: no cover - import availability depends on runtime container
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("Qwen3 model modules require torch")


def _copy_context_prefix(context_kv: ContextKV, prefix_context_kv: ContextKV) -> None:
    prefix_len = prefix_context_kv.context_len
    context_kv.key[:, :, :prefix_len].copy_(prefix_context_kv.key)
    context_kv.value[:, :, :prefix_len].copy_(prefix_context_kv.value)


def _decode_next_input_norm_fusion_enabled() -> bool:
    return os.environ.get("GR_INFERENCE_DECODE_NEXT_INPUT_NORM_FUSION") == "1"


if nn is not None:

    class Qwen3GRModel(nn.Module):
        """Minimal multi-layer Qwen3 GR model for prefill path validation."""

        def __init__(
            self,
            config: Qwen3GRConfig,
            *,
            prefill_attention: PrefillAttention,
            dtype: Any | None = None,
        ) -> None:
            super().__init__()
            if config.vocab_size is None:
                raise ValueError("Qwen3GRModel requires config.vocab_size")
            self.config = config
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList(
                [
                    Qwen3SingleLayerPrefill(
                        config,
                        layer_idx=layer_idx,
                        prefill_attention=prefill_attention,
                        dtype=dtype,
                    )
                    for layer_idx in range(config.num_layers)
                ]
            )
            self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if config.tie_word_embeddings:
                self.lm_head.weight = self.embed_tokens.weight
            if dtype is not None:
                self.to(dtype=dtype)

        def allocate_context_kv(
            self,
            *,
            batch_size: int,
            context_len: int,
            device: Any | None = None,
            dtype: Any | None = None,
        ) -> ContextKV:
            if context_len <= 0:
                raise ValueError("context_len must be positive")
            if context_len > self.config.max_context_len:
                raise ValueError(
                    f"context_len={context_len} exceeds "
                    f"max_context_len={self.config.max_context_len}"
                )
            if device is None:
                device = self.embed_tokens.weight.device
            if dtype is None:
                dtype = self.embed_tokens.weight.dtype
            shape = (
                self.config.num_layers,
                batch_size,
                context_len,
                self.config.num_kv_heads,
                self.config.head_dim,
            )
            key = torch.empty(shape, device=device, dtype=dtype)
            value = torch.empty_like(key)
            return ContextKV(key, value)

        @torch.no_grad()
        def forward_prefill_embed(
            self,
            input_ids,
            *,
            timing_recorder: Any | None = None,
        ):
            with _timed(timing_recorder, "prefill.embed_tokens"):
                return self.embed_tokens(input_ids)

        @torch.no_grad()
        def forward_prefill_layer(
            self,
            layer_idx: int,
            hidden_states,
            context_kv: ContextKV,
            *,
            normed_hidden_states: Any | None = None,
            timing_recorder: Any | None = None,
        ):
            if layer_idx < 0 or layer_idx >= len(self.layers):
                raise IndexError(f"prefill layer_idx out of range: {layer_idx}")
            layer = self.layers[layer_idx]
            next_layer = (
                self.layers[layer_idx + 1] if layer_idx + 1 < len(self.layers) else None
            )
            next_input_norm = (
                next_layer.ops.input_layernorm if next_layer is not None else None
            )
            return layer.forward_prefill(
                hidden_states,
                context_kv,
                timing_recorder=timing_recorder,
                normed_hidden_states=normed_hidden_states,
                next_input_norm=next_input_norm,
                return_next_normed=True,
            )

        @torch.no_grad()
        def forward_prefill_output(
            self,
            hidden_states,
            context_kv: ContextKV,
            *,
            return_hidden_states: bool = False,
            timing_recorder: Any | None = None,
            last_token_logits_only: bool = False,
        ) -> PrefillResult:
            with _timed(timing_recorder, "prefill.final_norm"):
                hidden_states = self.norm(hidden_states)
            with _timed(timing_recorder, "prefill.lm_head"):
                logits_input = (
                    hidden_states[:, -1, :] if last_token_logits_only else hidden_states
                )
                logits = _linear_project(self.lm_head, logits_input)
            return PrefillResult(
                logits=logits,
                context_kv=context_kv,
                hidden_states=hidden_states if return_hidden_states else None,
            )

        @torch.no_grad()
        def forward_prefill(
            self,
            input_ids,
            context_kv: ContextKV | None = None,
            *,
            return_result: bool = False,
            return_hidden_states: bool = False,
            timing_recorder: Any | None = None,
            last_token_logits_only: bool = False,
        ):
            batch_size, context_len = input_ids.shape
            if context_kv is None:
                context_kv = self.allocate_context_kv(
                    batch_size=batch_size,
                    context_len=context_len,
                    device=input_ids.device,
                    dtype=self.embed_tokens.weight.dtype,
                )
            with _timed(timing_recorder, "model.forward_prefill"):
                hidden_states = self.forward_prefill_embed(
                    input_ids,
                    timing_recorder=timing_recorder,
                )
                normed_hidden_states = None
                for layer_idx in range(len(self.layers)):
                    hidden_states, normed_hidden_states = self.forward_prefill_layer(
                        layer_idx,
                        hidden_states,
                        context_kv,
                        normed_hidden_states=normed_hidden_states,
                        timing_recorder=timing_recorder,
                    )
                prefill = self.forward_prefill_output(
                    hidden_states,
                    context_kv,
                    return_hidden_states=return_hidden_states,
                    timing_recorder=timing_recorder,
                    last_token_logits_only=last_token_logits_only,
                )
            if return_result:
                return prefill
            return prefill.logits, prefill.context_kv

        @torch.no_grad()
        def forward_prefill_extend(
            self,
            suffix_input_ids,
            prefix_context_kv: ContextKV,
            context_kv: ContextKV | None = None,
            *,
            return_result: bool = False,
            return_hidden_states: bool = False,
            timing_recorder: Any | None = None,
            last_token_logits_only: bool = False,
        ):
            batch_size, suffix_len = suffix_input_ids.shape
            prefix_len = prefix_context_kv.context_len
            context_len = prefix_len + suffix_len
            if suffix_len <= 0:
                raise ValueError("suffix_input_ids must contain at least one token")
            if prefix_context_kv.batch_size != batch_size:
                raise ValueError("prefix ContextKV batch must match suffix batch")
            if context_kv is None:
                context_kv = self.allocate_context_kv(
                    batch_size=batch_size,
                    context_len=context_len,
                    device=suffix_input_ids.device,
                    dtype=self.embed_tokens.weight.dtype,
                )
            elif context_kv.context_len != context_len:
                raise ValueError("context_kv length must equal prefix + suffix length")
            _copy_context_prefix(context_kv, prefix_context_kv)
            with _timed(timing_recorder, "model.forward_prefill_extend"):
                with _timed(timing_recorder, "prefill_extend.embed_tokens"):
                    hidden_states = self.embed_tokens(suffix_input_ids)
                for layer in self.layers:
                    hidden_states = layer.forward_prefill_extend(
                        hidden_states,
                        context_kv,
                        prefix_len=prefix_len,
                        timing_recorder=timing_recorder,
                    )
                with _timed(timing_recorder, "prefill_extend.final_norm"):
                    hidden_states = self.norm(hidden_states)
                with _timed(timing_recorder, "prefill_extend.lm_head"):
                    logits_input = (
                        hidden_states[:, -1, :]
                        if last_token_logits_only
                        else hidden_states
                    )
                    logits = _linear_project(self.lm_head, logits_input)
            if return_result:
                return PrefillResult(
                    logits=logits,
                    context_kv=context_kv,
                    hidden_states=hidden_states if return_hidden_states else None,
                )
            return logits, context_kv

        @torch.no_grad()
        def forward_decode_step(
            self,
            beam_token_ids,
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
        ):
            if beam_token_ids.shape[0] != generation.prefill.batch_size:
                raise ValueError("beam_token_ids batch must match prefill batch")
            if active_beam_width is None:
                active_beam_width = generation.fixed_beam_width
            if beam_token_ids.shape[1] != active_beam_width:
                raise ValueError("beam_token_ids width must match active_beam_width")

            hidden_states = self.embed_tokens(beam_token_ids)
            decode_position = generation.prefill.context_len + step
            normed_hidden_states = None
            use_next_norm_fusion = _decode_next_input_norm_fusion_enabled()
            for layer_idx, layer in enumerate(self.layers):
                if use_next_norm_fusion:
                    next_layer = (
                        self.layers[layer_idx + 1]
                        if layer_idx + 1 < len(self.layers)
                        else None
                    )
                    next_input_norm = (
                        next_layer.ops.input_layernorm
                        if next_layer is not None
                        else self.norm
                    )
                else:
                    next_input_norm = None
                hidden_states, normed_hidden_states = layer.forward_decode(
                    hidden_states,
                    generation,
                    decode_engine,
                    step=step,
                    active_beam_width=active_beam_width,
                    topk_indices=topk_indices,
                    decode_nums=decode_nums,
                    return_lse=return_lse,
                    backend_name=backend_name,
                    timing_recorder=timing_recorder,
                    position_ids=decode_position,
                    normed_hidden_states=normed_hidden_states,
                    next_input_norm=next_input_norm,
                    return_next_normed=True,
                )
            hidden_states = (
                normed_hidden_states
                if normed_hidden_states is not None
                else self.norm(hidden_states)
            )
            return _linear_project(self.lm_head, hidden_states)

        def generate_fixed_beam(
            self,
            generation: GRGenerationState,
            decode_engine: GRDecodeEngine,
            *,
            max_steps: int,
            initial_item_mask: Any | None = None,
            item_mask_provider: Any | None = None,
            beam_width_policy: Any | None = None,
            stop_token_ids: tuple[int, ...] = (),
            logits_processors: tuple[Any, ...] = (),
        ):
            loop = FixedBeamDecodeLoop(
                model=self,
                decode_engine=decode_engine,
                item_mask_provider=item_mask_provider,
                beam_width_policy=beam_width_policy,
                stop_token_ids=stop_token_ids,
                logits_processors=logits_processors,
            )
            return loop.run(
                generation,
                max_steps=max_steps,
                initial_item_mask=initial_item_mask,
            )

        def load_logical_weights(
            self, weights: dict[str, Any], *, strict: bool = True
        ) -> None:
            """Load model-level logical tensors produced by Qwen3HFAdapter."""

            self._copy_tensor(
                self.embed_tokens.weight, weights, "embed_tokens.weight", strict=strict
            )
            self._copy_tensor(
                self.norm.weight, weights, "final_norm.weight", strict=strict
            )
            if not self.config.tie_word_embeddings:
                self._copy_tensor(
                    self.lm_head.weight, weights, "lm_head.weight", strict=strict
                )
            for layer_idx, layer in enumerate(self.layers):
                layer.ops.load_logical_weights(
                    weights, layer_idx=layer_idx, strict=strict
                )

        @staticmethod
        def _copy_tensor(
            param, weights: dict[str, Any], name: str, *, strict: bool
        ) -> None:
            if name not in weights:
                if strict:
                    raise KeyError(f"missing logical tensor: {name}")
                return
            tensor = weights[name]
            if tuple(param.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            with torch.no_grad():
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))

else:

    class Qwen3GRModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def _timed(timing_recorder, name: str):
    if timing_recorder is None:
        return nullcontext()
    return timing_recorder.section(name)


def _linear_project(linear, hidden_states):
    if (
        _flatten_linear_disabled()
        or not hasattr(hidden_states, "dim")
        or hidden_states.dim() <= 2
    ):
        return linear(hidden_states)
    original_shape = tuple(hidden_states.shape[:-1])
    flat_input = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_output = linear(flat_input)
    return flat_output.reshape(*original_shape, flat_output.shape[-1])


def _flatten_linear_disabled() -> bool:
    return os.environ.get("GR_INFERENCE_DISABLE_FLATTEN_LINEAR", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
