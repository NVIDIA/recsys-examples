# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph runner for continuous GR prefill microbatches."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from gr_inference.gr_kv import ContextKV
from gr_inference.gr_runtime.generation import PrefillResult
from gr_inference.gr_serving.cuda_graph_utils import CudaGraphCacheMixin
from gr_inference.gr_serving.cuda_graph_utils import env_flag as _env_flag
from gr_inference.gr_serving.cuda_graph_utils import env_int as _env_int
from gr_inference.gr_serving.cuda_graph_utils import is_cuda_tensor as _is_cuda_tensor
from gr_inference.gr_serving.cuda_graph_utils import tensor_data_ptr as _tensor_data_ptr
from gr_inference.gr_serving.cuda_graph_utils import tensor_nbytes as _tensor_nbytes
from gr_inference.gr_serving.cuda_graph_utils import tensor_view_key as _tensor_view_key


@dataclass
class _PrefillGraphSegment:
    graph: Any
    name: str


@dataclass
class _PrefillGraphEntry:
    graphs: tuple[_PrefillGraphSegment, ...]
    output: PrefillResult | None
    input_ids: Any
    context_kv: ContextKV
    last_token_logits_only: bool
    mode: str


class GRPrefillCudaGraphRunner(CudaGraphCacheMixin):
    """Capture/replay ``model.forward_prefill`` for fixed prefill shapes.

    Replay is guarded by the ``ContextKV`` pointer.  This keeps graph writes tied
    to the same pool slice that downstream decode will read.
    """

    _separate_pools_env = "GR_INFERENCE_PREFILL_CUDA_GRAPH_SEPARATE_POOLS"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.max_entries = _env_int("GR_INFERENCE_PREFILL_CUDA_GRAPH_MAX_ENTRIES", 32)
        self.mode = _prefill_cuda_graph_mode()
        self.piecewise_layer_chunk_size = max(
            1,
            _env_int(
                "GR_INFERENCE_PREFILL_CUDA_GRAPH_LAYER_CHUNK_SIZE",
                _default_piecewise_layer_chunk_size(model),
            ),
        )
        self._graph_pool: Any | None = None
        self.allow_captures = True
        self._graphs: OrderedDict[tuple[Any, ...], _PrefillGraphEntry] = OrderedDict()
        self.replays = 0
        self.captures = 0
        self.full_captures = 0
        self.piecewise_captures = 0
        self.piecewise_fallback_full_captures = 0
        self.failures = 0
        self.evictions = 0
        self.requests = 0
        self.hit_count = 0
        self.miss_count = 0
        self.fallback_eager_count = 0
        self.capture_latency_ms_total = 0.0
        self.replay_cpu_ms_total = 0.0
        self.input_copy_bytes = 0
        self.reuse_pointer_misses = 0
        self.reuse_pointer_miss_reasons: dict[str, int] = {}
        self.miss_reasons: dict[str, int] = {}

    def status(self) -> dict[str, int | float | str]:
        piecewise_entries = sum(
            1 for entry in self._graphs.values() if entry.mode == "piecewise"
        )
        full_entries = sum(1 for entry in self._graphs.values() if entry.mode == "full")
        piecewise_graphs = sum(
            len(entry.graphs)
            for entry in self._graphs.values()
            if entry.mode == "piecewise"
        )
        return {
            "prefill_cuda_graph_mode": self.mode,
            "prefill_cuda_graph_piecewise_layer_chunk_size": (
                self.piecewise_layer_chunk_size
            ),
            "prefill_cuda_graph_entries": len(self._graphs),
            "prefill_cuda_graph_full_entries": full_entries,
            "prefill_cuda_graph_piecewise_entries": piecewise_entries,
            "prefill_cuda_graph_piecewise_graphs": piecewise_graphs,
            "prefill_cuda_graph_captures_enabled": int(self.allow_captures),
            "prefill_cuda_graph_captures": self.captures,
            "prefill_cuda_graph_full_captures": self.full_captures,
            "prefill_cuda_graph_piecewise_captures": self.piecewise_captures,
            "prefill_cuda_graph_piecewise_fallback_full_captures": (
                self.piecewise_fallback_full_captures
            ),
            "prefill_cuda_graph_replays": self.replays,
            "prefill_cuda_graph_failures": self.failures,
            "prefill_cuda_graph_evictions": self.evictions,
            "prefill_cuda_graph_max_entries": self.max_entries,
            "prefill_cuda_graph_requests": self.requests,
            "prefill_cuda_graph_hit_count": self.hit_count,
            "prefill_cuda_graph_miss_count": self.miss_count,
            "prefill_cuda_graph_fallback_eager_count": self.fallback_eager_count,
            "prefill_cuda_graph_capture_latency_ms_total": self.capture_latency_ms_total,
            "prefill_cuda_graph_replay_cpu_ms_total": self.replay_cpu_ms_total,
            "prefill_cuda_graph_input_copy_bytes": self.input_copy_bytes,
            "prefill_cuda_graph_reuse_pointer_misses": self.reuse_pointer_misses,
            **{
                f"prefill_cuda_graph_miss_reason_{reason}": count
                for reason, count in sorted(self.miss_reasons.items())
            },
            **{
                f"prefill_cuda_graph_reuse_pointer_miss_{reason}": count
                for reason, count in sorted(self.reuse_pointer_miss_reasons.items())
            },
        }

    def forward_prefill(
        self,
        input_ids: Any,
        *,
        context_kv: ContextKV | None,
        last_token_logits_only: bool,
    ) -> PrefillResult | None:
        self.requests += 1
        unsupported_reason = self._can_run_miss_reason(
            input_ids=input_ids,
            context_kv=context_kv,
        )
        if unsupported_reason is not None:
            self._record_miss(unsupported_reason, fallback=True)
            return None

        assert context_kv is not None
        key = self._key(
            input_ids=input_ids,
            context_kv=context_kv,
            last_token_logits_only=last_token_logits_only,
        )
        entry = self._graphs.get(key)
        if entry is not None:
            miss_reason = self._reuse_miss_reason(entry, context_kv)
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
                    input_ids,
                    context_kv=context_kv,
                    last_token_logits_only=last_token_logits_only,
                )
                self.capture_latency_ms_total += (
                    time.perf_counter() - capture_start
                ) * 1000.0
            except RuntimeError:
                self.failures += 1
                self._record_fallback("capture_failure")
                return None
        else:
            self._graphs.move_to_end(key)
            entry.input_ids.copy_(input_ids)
            self.input_copy_bytes += _tensor_nbytes(input_ids)
            self.hit_count += 1

        replay_start = time.perf_counter()
        for segment in entry.graphs:
            segment.graph.replay()
        self.replay_cpu_ms_total += (time.perf_counter() - replay_start) * 1000.0
        self.replays += 1
        return entry.output

    def freeze_captures(self) -> None:
        self.allow_captures = False

    def _capture(
        self,
        key: tuple[Any, ...],
        input_ids: Any,
        *,
        context_kv: ContextKV,
        last_token_logits_only: bool,
    ) -> _PrefillGraphEntry:
        if self.mode == "piecewise" and self._can_capture_piecewise():
            return self._capture_piecewise(
                key,
                input_ids,
                context_kv=context_kv,
                last_token_logits_only=last_token_logits_only,
            )
        if self.mode == "piecewise":
            self.piecewise_fallback_full_captures += 1
        return self._capture_full(
            key,
            input_ids,
            context_kv=context_kv,
            last_token_logits_only=last_token_logits_only,
        )

    def _capture_full(
        self,
        key: tuple[Any, ...],
        input_ids: Any,
        *,
        context_kv: ContextKV,
        last_token_logits_only: bool,
    ) -> _PrefillGraphEntry:
        import torch

        entry = _PrefillGraphEntry(
            graphs=(),
            output=None,
            input_ids=input_ids.detach().clone(),
            context_kv=context_kv,
            last_token_logits_only=last_token_logits_only,
            mode="full",
        )
        graph = torch.cuda.CUDAGraph()

        def run_once() -> PrefillResult:
            return self.model.forward_prefill(
                entry.input_ids,
                context_kv=entry.context_kv,
                return_result=True,
                timing_recorder=None,
                last_token_logits_only=entry.last_token_logits_only,
            )

        side_stream = torch.cuda.Stream(device=input_ids.device)
        side_stream.wait_stream(torch.cuda.current_stream(device=input_ids.device))
        with torch.cuda.stream(side_stream):
            run_once()
        torch.cuda.current_stream(device=input_ids.device).wait_stream(side_stream)
        torch.cuda.synchronize(device=input_ids.device)

        with torch.cuda.graph(graph):
            entry.output = run_once()
        entry.graphs = (_PrefillGraphSegment(graph=graph, name="full"),)

        self._store_graph(key, entry)
        self.captures += 1
        self.full_captures += 1
        return entry

    def _capture_piecewise(
        self,
        key: tuple[Any, ...],
        input_ids: Any,
        *,
        context_kv: ContextKV,
        last_token_logits_only: bool,
    ) -> _PrefillGraphEntry:
        import torch

        entry = _PrefillGraphEntry(
            graphs=(),
            output=None,
            input_ids=input_ids.detach().clone(),
            context_kv=context_kv,
            last_token_logits_only=last_token_logits_only,
            mode="piecewise",
        )
        segments: list[_PrefillGraphSegment] = []
        side_stream = torch.cuda.Stream(device=input_ids.device)

        def capture_segment(name: str, run_once):
            graph = torch.cuda.CUDAGraph()
            side_stream.wait_stream(torch.cuda.current_stream(device=input_ids.device))
            with torch.cuda.stream(side_stream):
                run_once()
            torch.cuda.current_stream(device=input_ids.device).wait_stream(side_stream)
            torch.cuda.synchronize(device=input_ids.device)
            graph_kwargs = self._graph_capture_kwargs(torch)
            with torch.cuda.graph(graph, **graph_kwargs):
                output = run_once()
            segments.append(_PrefillGraphSegment(graph=graph, name=name))
            return output

        hidden_states = capture_segment(
            "embed",
            lambda: self.model.forward_prefill_embed(
                entry.input_ids,
                timing_recorder=None,
            ),
        )
        normed_hidden_states = None
        layer_count = len(self.model.layers)
        for chunk_start in range(0, layer_count, self.piecewise_layer_chunk_size):
            chunk_end = min(chunk_start + self.piecewise_layer_chunk_size, layer_count)

            def run_layer_chunk(
                *,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                hidden_states=hidden_states,
                normed_hidden_states=normed_hidden_states,
            ):
                current_hidden_states = hidden_states
                current_normed_hidden_states = normed_hidden_states
                for layer_idx in range(chunk_start, chunk_end):
                    (
                        current_hidden_states,
                        current_normed_hidden_states,
                    ) = self.model.forward_prefill_layer(
                        layer_idx,
                        current_hidden_states,
                        entry.context_kv,
                        normed_hidden_states=current_normed_hidden_states,
                        timing_recorder=None,
                    )
                return current_hidden_states, current_normed_hidden_states

            hidden_states, normed_hidden_states = capture_segment(
                f"layers{chunk_start}-{chunk_end - 1}",
                run_layer_chunk,
            )

        entry.output = capture_segment(
            "output",
            lambda: self.model.forward_prefill_output(
                hidden_states,
                entry.context_kv,
                return_hidden_states=False,
                timing_recorder=None,
                last_token_logits_only=entry.last_token_logits_only,
            ),
        )
        entry.graphs = tuple(segments)

        self._store_graph(key, entry)
        self.captures += 1
        self.piecewise_captures += 1
        return entry

    def _can_capture_piecewise(self) -> bool:
        return (
            callable(getattr(self.model, "forward_prefill_embed", None))
            and callable(getattr(self.model, "forward_prefill_layer", None))
            and callable(getattr(self.model, "forward_prefill_output", None))
            and hasattr(self.model, "layers")
        )

    @staticmethod
    def _can_run_miss_reason(
        *,
        input_ids: Any,
        context_kv: ContextKV | None,
    ) -> str | None:
        if context_kv is None:
            return "missing_context_kv"
        if not _is_cuda_tensor(input_ids):
            return "input_ids_not_cuda"
        if not _is_cuda_tensor(context_kv.key):
            return "context_kv_key_not_cuda"
        if not _is_cuda_tensor(context_kv.value):
            return "context_kv_value_not_cuda"
        return None

    @staticmethod
    def _reuse_miss_reason(
        entry: _PrefillGraphEntry,
        context_kv: ContextKV,
    ) -> str | None:
        checks = (
            ("context_key", entry.context_kv.key, context_kv.key),
            ("context_value", entry.context_kv.value, context_kv.value),
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
        input_ids: Any,
        context_kv: ContextKV,
        last_token_logits_only: bool,
    ) -> tuple[Any, ...]:
        return (
            str(input_ids.device),
            input_ids.dtype,
            tuple(input_ids.shape),
            tuple(context_kv.key.shape),
            tuple(context_kv.value.shape),
            _tensor_view_key(context_kv.key),
            _tensor_view_key(context_kv.value),
            bool(last_token_logits_only),
        )


def _prefill_cuda_graph_mode() -> str:
    import os

    raw = os.environ.get("GR_INFERENCE_PREFILL_CUDA_GRAPH_MODE", "").strip().lower()
    if not raw and _env_flag("GR_INFERENCE_DISABLE_PIECEWISE_PREFILL_CUDA_GRAPH"):
        raw = "full"
    if raw in {"", "piecewise", "pcg"}:
        return "piecewise"
    if raw in {"full", "whole", "single"}:
        return "full"
    return "piecewise"


def _default_piecewise_layer_chunk_size(model: Any) -> int:
    layers = getattr(model, "layers", None)
    try:
        layer_count = len(layers)
    except TypeError:
        layer_count = 0
    if layer_count <= 0:
        return 4
    target_chunks = max(
        1,
        _env_int("GR_INFERENCE_PREFILL_CUDA_GRAPH_TARGET_LAYER_CHUNKS", 4),
    )
    return max(1, (layer_count + target_chunks - 1) // target_chunks)
