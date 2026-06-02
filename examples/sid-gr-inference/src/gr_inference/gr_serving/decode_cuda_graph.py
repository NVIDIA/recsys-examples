# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph runner for continuous GR decode microbatches."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv import BeamKV, ContextKV
from gr_inference.gr_runtime import GRGenerationState
from gr_inference.gr_runtime.generation import PrefillResult
from gr_inference.gr_serving.cuda_graph_utils import CudaGraphCacheMixin
from gr_inference.gr_serving.cuda_graph_utils import env_int as _env_int
from gr_inference.gr_serving.cuda_graph_utils import is_cuda_tensor as _is_cuda_tensor
from gr_inference.gr_serving.cuda_graph_utils import tensor_data_ptr as _tensor_data_ptr
from gr_inference.gr_serving.cuda_graph_utils import tensor_nbytes as _tensor_nbytes
from gr_inference.gr_serving.cuda_graph_utils import tensor_view_key as _tensor_view_key


@dataclass
class _DecodeGraphEntry:
    graph: Any
    output: Any
    beam_token_ids: Any
    topk_indices: Any
    context_kv: ContextKV
    beam_kv: BeamKV
    generation: GRGenerationState
    step: int
    active_beam_width: int


class GRDecodeCudaGraphRunner(CudaGraphCacheMixin):
    """Capture/replay ``model.forward_decode_step`` for fixed decode shapes.

    The runner relies on stable ``ContextKV``/``BeamKV`` pool slices, so replay
    can reuse request-owned KV addresses without staging KV copies.
    """

    def __init__(self, model: Any, decode_engine: Any) -> None:
        self.model = model
        self.decode_engine = decode_engine
        self.max_entries = _env_int("GR_INFERENCE_DECODE_CUDA_GRAPH_MAX_ENTRIES", 128)
        self.allow_captures = True
        self._graphs: OrderedDict[tuple[Any, ...], _DecodeGraphEntry] = OrderedDict()
        self.replays = 0
        self.captures = 0
        self.failures = 0
        self.evictions = 0
        self.requests = 0
        self.hit_count = 0
        self.miss_count = 0
        self.fallback_eager_count = 0
        self.capture_latency_ms_total = 0.0
        self.replay_cpu_ms_total = 0.0
        self.context_kv_copy_bytes = 0
        self.beam_kv_copy_bytes = 0
        self.beam_kv_writeback_bytes = 0
        self.input_copy_bytes = 0
        self.direct_beam_kv_replays = 0
        self.reuse_pointer_misses = 0
        self.reuse_pointer_miss_reasons: dict[str, int] = {}
        self.miss_reasons: dict[str, int] = {}

    def status(self) -> dict[str, int | float]:
        return {
            "decode_cuda_graph_entries": len(self._graphs),
            "decode_cuda_graph_captures_enabled": int(self.allow_captures),
            "decode_cuda_graph_captures": self.captures,
            "decode_cuda_graph_replays": self.replays,
            "decode_cuda_graph_failures": self.failures,
            "decode_cuda_graph_evictions": self.evictions,
            "decode_cuda_graph_max_entries": self.max_entries,
            "decode_cuda_graph_requests": self.requests,
            "decode_cuda_graph_hit_count": self.hit_count,
            "decode_cuda_graph_miss_count": self.miss_count,
            "decode_cuda_graph_fallback_eager_count": self.fallback_eager_count,
            "decode_cuda_graph_capture_latency_ms_total": self.capture_latency_ms_total,
            "decode_cuda_graph_replay_cpu_ms_total": self.replay_cpu_ms_total,
            "decode_cuda_graph_context_kv_copy_bytes": self.context_kv_copy_bytes,
            "decode_cuda_graph_beam_kv_copy_bytes": self.beam_kv_copy_bytes,
            "decode_cuda_graph_beam_kv_writeback_bytes": self.beam_kv_writeback_bytes,
            "decode_cuda_graph_input_copy_bytes": self.input_copy_bytes,
            "decode_cuda_graph_direct_beam_kv_replays": self.direct_beam_kv_replays,
            "decode_cuda_graph_reuse_pointer_misses": self.reuse_pointer_misses,
            **{
                f"decode_cuda_graph_miss_reason_{reason}": count
                for reason, count in sorted(self.miss_reasons.items())
            },
            **{
                f"decode_cuda_graph_reuse_pointer_miss_{reason}": count
                for reason, count in sorted(self.reuse_pointer_miss_reasons.items())
            },
        }

    def can_run(
        self,
        *,
        beam_token_ids: Any,
        generation: GRGenerationState,
        topk_indices: Any | None,
    ) -> bool:
        return (
            self._can_run_miss_reason(
                beam_token_ids=beam_token_ids,
                generation=generation,
                topk_indices=topk_indices,
            )
            is None
        )

    def freeze_captures(self) -> None:
        self.allow_captures = False

    def forward_decode_step(
        self,
        beam_token_ids: Any,
        generation: GRGenerationState,
        *,
        step: int,
        active_beam_width: int,
        topk_indices: Any | None,
        decode_nums: int,
    ) -> Any | None:
        self.requests += 1
        unsupported_reason = self._can_run_miss_reason(
            beam_token_ids=beam_token_ids,
            generation=generation,
            topk_indices=topk_indices,
        )
        if unsupported_reason is not None:
            self._record_miss(unsupported_reason, fallback=True)
            return None

        key = self._key(
            beam_token_ids=beam_token_ids,
            generation=generation,
            topk_indices=topk_indices,
            step=step,
            active_beam_width=active_beam_width,
            decode_nums=decode_nums,
        )
        entry = self._graphs.get(key)
        if entry is not None:
            miss_reason = self._reuse_miss_reason(entry, generation)
        else:
            miss_reason = "shape"
        if entry is None or miss_reason is not None:
            if entry is not None and miss_reason is not None:
                self.reuse_pointer_misses += 1
                self.reuse_pointer_miss_reasons[miss_reason] = (
                    self.reuse_pointer_miss_reasons.get(miss_reason, 0) + 1
                )
            if not self.allow_captures:
                self._record_miss(f"{miss_reason}_capture_disabled", fallback=True)
                return None
            self._record_miss(miss_reason)
            try:
                capture_start = time.perf_counter()
                entry = self._capture(
                    key,
                    beam_token_ids,
                    generation,
                    step=step,
                    active_beam_width=active_beam_width,
                    topk_indices=topk_indices,
                    decode_nums=decode_nums,
                )
                self.capture_latency_ms_total += (
                    time.perf_counter() - capture_start
                ) * 1000.0
            except RuntimeError:
                self.failures += 1
                self._record_fallback("capture_failure")
                return None
            replay_start = time.perf_counter()
            entry.graph.replay()
            self.replay_cpu_ms_total += (time.perf_counter() - replay_start) * 1000.0
            self.replays += 1
        else:
            self._graphs.move_to_end(key)
            self._copy_inputs(entry, beam_token_ids, generation, topk_indices)
            self.hit_count += 1
            replay_start = time.perf_counter()
            entry.graph.replay()
            self.replay_cpu_ms_total += (time.perf_counter() - replay_start) * 1000.0
            self.replays += 1

        self._copy_written_beam_step(entry, generation)
        return entry.output

    def _capture(
        self,
        key: tuple[Any, ...],
        beam_token_ids: Any,
        generation: GRGenerationState,
        *,
        step: int,
        active_beam_width: int,
        topk_indices: Any,
        decode_nums: int,
    ) -> _DecodeGraphEntry:
        import torch

        context_kv = generation.prefill.context_kv
        beam_kv = generation.beam_kv
        static_generation = GRGenerationState(
            request_id="cuda-graph-decode",
            prefill=PrefillResult(
                logits=generation.prefill.logits,
                context_kv=context_kv,
                hidden_states=None,
            ),
            beam_kv=beam_kv,
            beam_path=generation.beam_path,
            fixed_beam_width=generation.fixed_beam_width,
            beam_score_mode=generation.beam_score_mode,
        )
        entry = _DecodeGraphEntry(
            graph=torch.cuda.CUDAGraph(),
            output=None,
            beam_token_ids=beam_token_ids.detach().clone(),
            topk_indices=topk_indices.detach().clone(),
            context_kv=context_kv,
            beam_kv=beam_kv,
            generation=static_generation,
            step=step,
            active_beam_width=active_beam_width,
        )

        def run_once():
            return self.model.forward_decode_step(
                entry.beam_token_ids,
                entry.generation,
                self.decode_engine,
                step=step,
                active_beam_width=active_beam_width,
                topk_indices=entry.topk_indices,
                decode_nums=decode_nums,
                timing_recorder=None,
            )

        # Warm up kernels and lazy imports outside capture.
        side_stream = torch.cuda.Stream(device=beam_token_ids.device)
        side_stream.wait_stream(torch.cuda.current_stream(device=beam_token_ids.device))
        with torch.cuda.stream(side_stream):
            run_once()
        torch.cuda.current_stream(device=beam_token_ids.device).wait_stream(side_stream)
        torch.cuda.synchronize(device=beam_token_ids.device)

        with torch.cuda.graph(entry.graph):
            entry.output = run_once()

        self._store_graph(key, entry)
        self.captures += 1
        return entry

    def _copy_inputs(
        self,
        entry: _DecodeGraphEntry,
        beam_token_ids: Any,
        generation: GRGenerationState,
        topk_indices: Any,
    ) -> None:
        entry.beam_token_ids.copy_(beam_token_ids)
        entry.topk_indices.copy_(topk_indices)
        self.input_copy_bytes += _tensor_nbytes(beam_token_ids)
        self.input_copy_bytes += _tensor_nbytes(topk_indices)

    def _copy_written_beam_step(
        self,
        entry: _DecodeGraphEntry,
        generation: GRGenerationState,
    ) -> None:
        if _tensor_data_ptr(entry.beam_kv.key) == _tensor_data_ptr(
            generation.beam_kv.key
        ) and _tensor_data_ptr(entry.beam_kv.value) == _tensor_data_ptr(
            generation.beam_kv.value
        ):
            self.direct_beam_kv_replays += 1
            return

    @staticmethod
    def _can_run_miss_reason(
        *,
        beam_token_ids: Any,
        generation: GRGenerationState,
        topk_indices: Any | None,
    ) -> str | None:
        if topk_indices is None:
            return "missing_topk_indices"
        if not _is_cuda_tensor(beam_token_ids):
            return "beam_token_ids_not_cuda"
        if not _is_cuda_tensor(topk_indices):
            return "topk_indices_not_cuda"
        if not _is_cuda_tensor(generation.prefill.context_kv.key):
            return "context_kv_key_not_cuda"
        if not _is_cuda_tensor(generation.prefill.context_kv.value):
            return "context_kv_value_not_cuda"
        if not _is_cuda_tensor(generation.beam_kv.key):
            return "beam_kv_key_not_cuda"
        if not _is_cuda_tensor(generation.beam_kv.value):
            return "beam_kv_value_not_cuda"
        return None

    @staticmethod
    def _reuse_miss_reason(
        entry: _DecodeGraphEntry,
        generation: GRGenerationState,
    ) -> str | None:
        checks = (
            ("context_key", entry.context_kv.key, generation.prefill.context_kv.key),
            (
                "context_value",
                entry.context_kv.value,
                generation.prefill.context_kv.value,
            ),
            ("beam_key", entry.beam_kv.key, generation.beam_kv.key),
            ("beam_value", entry.beam_kv.value, generation.beam_kv.value),
        )
        for name, left, right in checks:
            if _tensor_data_ptr(left) != _tensor_data_ptr(right):
                return name
            if int(getattr(left, "storage_offset", lambda: 0)()) != int(
                getattr(right, "storage_offset", lambda: 0)()
            ):
                return f"{name}_offset"
        return None

    @staticmethod
    def _key(
        *,
        beam_token_ids: Any,
        generation: GRGenerationState,
        topk_indices: Any,
        step: int,
        active_beam_width: int,
        decode_nums: int,
    ) -> tuple[Any, ...]:
        return (
            str(beam_token_ids.device),
            beam_token_ids.dtype,
            tuple(beam_token_ids.shape),
            generation.prefill.context_len,
            tuple(generation.prefill.context_kv.key.shape),
            tuple(generation.beam_kv.key.shape),
            _tensor_view_key(generation.prefill.context_kv.key),
            _tensor_view_key(generation.prefill.context_kv.value),
            _tensor_view_key(generation.beam_kv.key),
            _tensor_view_key(generation.beam_kv.value),
            tuple(topk_indices.shape),
            step,
            active_beam_width,
            decode_nums,
        )
