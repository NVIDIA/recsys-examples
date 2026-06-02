# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a synchronous serving smoke with real Qwen3 weights."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__)

from gr_inference import (  # noqa: E402
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRDecodeAttention,
    GRDecodeEngine,
    GRDenseBeamKVPool,
    GRDenseContextKVPool,
    GRServingConfig,
    GRServingEngine,
    GRServingRequest,
    PrefillAttention,
    ScheduledBeamPolicy,
    SchedulerPolicy,
    SyncGRScheduler,
    TokenSuppressLogitsProcessor,
)
from gr_inference.gr_kernels import (  # noqa: E402
    CAP_FUSED_MLP,
    CAP_GR_DECODE_ATTENTION,
    CAP_PACKED_GEMM,
    CAP_PREFILL_ATTENTION,
    CAP_QK_NORM_ROPE,
    CAP_RMSNORM,
    CAP_ROPE,
    build_default_kernel_registry,
    default_kernel_selection_policy,
)
from gr_inference.gr_kernels.attention import (  # noqa: E402
    ExistingGRDecodeAttentionBackend,
)
from gr_inference.gr_kernels.prefill import (  # noqa: E402
    AutoPrefillBackend,
    FlashAttentionPrefillBackend,
    SGLangFlashAttentionPrefillBackend,
    TorchSDPAPrefillBackend,
)
from gr_inference.gr_models import HFCheckpointLoader, resolve_model_dir  # noqa: E402
from gr_inference.gr_models.qwen3 import (  # noqa: E402
    DEFAULT_QWEN3_MODEL_ID,
    Qwen3GRConfig,
    Qwen3GRModel,
    flashinfer_call_counts,
    materialize_qwen3_checkpoint,
    reset_flashinfer_call_counts,
)
from gr_inference.gr_runtime import TimingRecorder  # noqa: E402
from gr_inference.gr_serving.beam_metadata import (  # noqa: E402
    normalized_beam_results_from_metadata,
)
from gr_inference.gr_serving.cli import parse_unique_int_list  # noqa: E402
from tool_utils import iter_jsonl
from tool_utils import numeric_median as _median  # noqa: E402
from tool_utils import write_json

try:  # noqa: E402
    from gr_inference_trtllm_kernels import call_counts as gr_trtllm_call_counts
    from gr_inference_trtllm_kernels import (
        reset_call_counts as reset_gr_trtllm_call_counts,
    )
except Exception:  # pragma: no cover - optional kernel package

    def gr_trtllm_call_counts() -> dict[str, int]:
        return {}

    def reset_gr_trtllm_call_counts() -> None:
        return None


def choose_dtype(torch, device: str):
    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(args, torch):
    args.model_dir = resolve_model_dir(
        model_dir=getattr(args, "model_dir", None),
        model=getattr(args, "model", None),
        default_model=DEFAULT_QWEN3_MODEL_ID,
        revision=getattr(args, "revision", None),
    )
    manifest = HFCheckpointLoader(args.model_dir).manifest()
    config = Qwen3GRConfig.from_hf_config(
        manifest.config,
        max_context_len=max(args.context_len, 1),
        max_seq_len=max(args.context_len + args.decode_steps, 2),
        max_decode_steps=max(args.decode_steps, 1),
        max_beam_width=args.beam_width,
    )
    device = (
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    dtype = choose_dtype(torch, device)

    print("Materializing checkpoint logical tensors...")
    weights = materialize_qwen3_checkpoint(args.model_dir)
    print(f"Loaded logical tensors: {len(weights)}")

    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(_make_prefill_backend()),
        dtype=dtype,
    ).to(device)
    model.load_logical_weights(weights)
    model.eval()
    print(f"Loaded Qwen3GRModel on {device} with dtype={dtype}.")
    return model, config, device


def _make_prefill_backend():
    backend = os.environ.get("GR_INFERENCE_PREFILL_BACKEND", "auto").lower()
    if backend == "auto":
        return AutoPrefillBackend()
    if backend in {"sglang_flash_attn", "sglang_fa3", "fa3"}:
        return SGLangFlashAttentionPrefillBackend()
    if backend == "flash_attn":
        return FlashAttentionPrefillBackend()
    if backend in {"torch", "torch_sdpa", "sdpa"}:
        return TorchSDPAPrefillBackend()
    raise ValueError(f"unknown GR_INFERENCE_PREFILL_BACKEND={backend!r}")


def make_decode_backend(args, device: str):
    if args.decode_backend == "fake":
        return lambda inputs: inputs.q
    if device != "cuda":
        raise RuntimeError("--decode-backend real requires CUDA")
    if not args.batched_decode:
        raise RuntimeError("--decode-backend real currently requires --batched-decode")
    return ExistingGRDecodeAttentionBackend()


def run_serving(args) -> dict:
    import torch

    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    arrival_stagger_ticks = getattr(args, "arrival_stagger_ticks", 0)
    arrival_burst_size = getattr(args, "arrival_burst_size", 1)
    if arrival_stagger_ticks < 0:
        raise ValueError("--arrival-stagger-ticks must be non-negative")
    if arrival_burst_size <= 0:
        raise ValueError("--arrival-burst-size must be positive")
    if arrival_stagger_ticks and not args.continuous:
        raise ValueError("--arrival-stagger-ticks requires --continuous")

    model, config, device = load_model(args, torch)
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=make_decode_backend(args, device)),
        fixed_beam_width=args.beam_width,
    )
    engine = GRServingEngine(
        model=model,
        decode_engine=decode_engine,
        config=GRServingConfig(
            max_decode_steps=args.decode_steps,
            max_beam_width=args.beam_width,
            enable_batched_decode=args.batched_decode,
            return_beam_details=args.return_beam_details,
            beam_score_mode=args.beam_score_mode,
        ),
    )
    shared_beam_kv_pool = (
        _make_beam_kv_pool(args, torch, config, device) if args.continuous else None
    )
    shared_context_kv_pool = (
        _make_context_kv_pool(args, torch, config, device) if args.continuous else None
    )

    warmup_runs = [
        _run_once(
            args,
            torch,
            config,
            device,
            engine,
            request_prefix=f"warmup-{idx}",
            beam_kv_pool=shared_beam_kv_pool,
            context_kv_pool=shared_context_kv_pool,
        )
        for idx in range(args.warmup_runs)
    ]
    _cuda_profiler_start(torch, enabled=getattr(args, "cuda_profiler_range", False))
    try:
        measured_runs = [
            _run_once(
                args,
                torch,
                config,
                device,
                engine,
                request_prefix=f"req-{idx}",
                beam_kv_pool=shared_beam_kv_pool,
                context_kv_pool=shared_context_kv_pool,
            )
            for idx in range(args.repeat)
        ]
    finally:
        _cuda_profiler_stop(torch, enabled=getattr(args, "cuda_profiler_range", False))
    primary = (
        measured_runs[-1] if args.verbose_metadata else _compact_run(measured_runs[-1])
    )
    summary = {
        "decode_backend": args.decode_backend,
        "serving_mode": "continuous" if args.continuous else "sync",
        "warmup_runs": len(warmup_runs),
        "repeat": len(measured_runs),
        "wall_ms_samples": [run["wall_ms"] for run in measured_runs],
        "wall_ms_median": _median(run["wall_ms"] for run in measured_runs),
        "arrival_stagger_ticks": arrival_stagger_ticks,
        "arrival_burst_size": arrival_burst_size,
    }
    summary.update(
        _metadata_metric_summaries(measured_runs, "prefill_ms", "decode_ms", "total_ms")
    )
    summary.update(
        _scheduler_metric_summaries(
            measured_runs,
            (
                ("scheduler_ms", "total_scheduler_ms", 0.0),
                ("continuous_ticks", "ticks", 0),
                ("planned_decode_batches", "planned_decode_batches", 0),
                ("avg_decode_batch_size", "avg_decode_batch_size", 0.0),
                ("inflight_admission_ticks", "inflight_admission_ticks", 0),
                (
                    "inflight_mixed_decode_step_ticks",
                    "inflight_mixed_decode_step_ticks",
                    0,
                ),
                ("inflight_max_decode_batch_size", "inflight_max_decode_batch_size", 0),
            ),
        )
    )
    summary.update(primary)
    return summary


def _compact_run(run: dict) -> dict:
    compact = dict(run)
    token_ids = compact.get("first_token_ids", ())
    compact["first_token_ids_count"] = len(token_ids)
    compact["first_token_ids_sample"] = tuple(token_ids[:8])
    compact.pop("first_token_ids", None)
    compact["first_metadata"] = _compact_metadata(compact.get("first_metadata", {}))
    return compact


def _compact_metadata(metadata: dict) -> dict:
    compact = dict(metadata)
    compact.pop("_beam_path", None)
    details = compact.pop("beam_details", None)
    if details is not None:
        compact["beam_details_count"] = len(details)
        compact["beam_details_sample"] = tuple(details[:4])
    return compact


def _metadata_metric_samples(runs: list[dict], key: str) -> list[float]:
    samples = []
    for run in runs:
        value = run.get("first_metadata", {}).get(key)
        if value is not None:
            samples.append(float(value))
    return samples


def _metadata_metric_median(runs: list[dict], key: str) -> float | None:
    samples = _metadata_metric_samples(runs, key)
    if not samples:
        return None
    return _median(samples)


def _metadata_metric_summaries(runs: list[dict], *keys: str) -> dict[str, Any]:
    return {
        output_key: value
        for key in keys
        for output_key, value in (
            (f"{key}_samples", _metadata_metric_samples(runs, key)),
            (f"{key}_median", _metadata_metric_median(runs, key)),
        )
    }


def _scheduler_metric_summaries(
    runs: list[dict],
    specs: tuple[tuple[str, str, float | int], ...],
) -> dict[str, Any]:
    return {
        output_key: value
        for output_name, metric_name, default in specs
        for samples in (
            [run["scheduler_metrics"].get(metric_name, default) for run in runs],
        )
        for output_key, value in (
            (f"{output_name}_samples", samples),
            (f"{output_name}_median", _median(samples)),
        )
    }


def _run_once(
    args,
    torch,
    config,
    device: str,
    engine,
    *,
    request_prefix: str,
    beam_kv_pool=None,
    context_kv_pool=None,
) -> dict:
    if args.continuous:
        return _run_continuous_once(
            args,
            torch,
            config,
            device,
            engine,
            request_prefix=request_prefix,
            beam_kv_pool=beam_kv_pool,
            context_kv_pool=context_kv_pool,
        )
    return _run_scheduler_once(
        args,
        torch,
        config,
        device,
        engine,
        request_prefix=request_prefix,
    )


def _run_scheduler_once(
    args, torch, config, device: str, engine, *, request_prefix: str
) -> dict:
    scheduler = SyncGRScheduler(
        engine,
        policy=SchedulerPolicy(max_batch_size=args.max_batch_size),
    )
    workload = _load_workload_inputs(args, torch, device)
    reset_flashinfer_call_counts()
    reset_gr_trtllm_call_counts()
    with torch.no_grad():
        for idx in range(args.requests):
            scheduler.submit(
                _serving_request(
                    args,
                    torch,
                    config,
                    device,
                    workload,
                    idx,
                    request_id=f"{request_prefix}-{idx}",
                )
            )
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        responses = scheduler.run_until_empty()
        if device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000.0

    return _run_result(
        args,
        engine,
        responses,
        status=scheduler.status(),
        metrics=scheduler.metrics(),
        batch_history=scheduler.batch_history,
        wall_ms=wall_ms,
    )


def _run_continuous_once(
    args,
    torch,
    config,
    device: str,
    engine,
    *,
    request_prefix: str,
    beam_kv_pool=None,
    context_kv_pool=None,
) -> dict:
    arrival_stagger_ticks = getattr(args, "arrival_stagger_ticks", 0)
    if beam_kv_pool is None:
        beam_kv_pool = _make_beam_kv_pool(args, torch, config, device)
    workload = _load_workload_inputs(args, torch, device)
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=args.max_batch_size,
            max_decode_batch_size=args.max_batch_size,
            max_running_requests=(
                args.beam_kv_pool_capacity if beam_kv_pool is not None else None
            ),
        )
    )
    recorder = (
        TimingRecorder(
            sync_module=torch,
            detail=args.profile_detail,
            sync_timing=args.profile_sync,
        )
        if args.profile_continuous_decode
        else None
    )
    executor = GRContinuousServingExecutor(
        engine=engine,
        scheduler=scheduler,
        synchronize=torch.cuda.synchronize if device == "cuda" else None,
        sync_timing=args.executor_sync_timing,
        timing_recorder=recorder,
        beam_kv_pool=beam_kv_pool,
        context_kv_pool=context_kv_pool,
        prefill_cache_enabled=args.enable_prefill_cache,
        max_prefill_cache_entries=args.prefill_cache_max_entries,
        max_prefill_cache_tokens=args.prefill_cache_max_tokens or None,
        prefill_cache_page_size=args.prefill_cache_page_size,
        min_prefill_cache_prefix_tokens=args.prefill_cache_min_prefix_tokens,
        max_prefill_cache_decode_extend_tokens=args.prefill_cache_max_decode_extend_tokens,
    )
    reset_flashinfer_call_counts()
    reset_gr_trtllm_call_counts()
    with torch.no_grad():
        if arrival_stagger_ticks > 0:
            next_request = _submit_staggered_arrivals(
                args,
                torch,
                config,
                device,
                executor,
                request_prefix=request_prefix,
                next_request=0,
                current_tick=0,
                workload=workload,
            )
        else:
            next_request = args.requests
            for idx in range(args.requests):
                executor.submit(
                    _serving_request(
                        args,
                        torch,
                        config,
                        device,
                        workload,
                        idx,
                        request_id=f"{request_prefix}-{idx}",
                    )
                )
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if arrival_stagger_ticks > 0:
            current_tick = 0
            while (
                next_request < args.requests
                or len(executor.scheduler.waiting_prefill)
                or executor.scheduler.decoding
            ):
                executor.tick()
                current_tick += 1
                next_request = _submit_staggered_arrivals(
                    args,
                    torch,
                    config,
                    device,
                    executor,
                    request_prefix=request_prefix,
                    next_request=next_request,
                    current_tick=current_tick,
                    workload=workload,
                )
            responses = tuple(executor.scheduler.finished.values())
        else:
            responses = executor.run_until_empty()
        if device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000.0

    scheduler_metrics = dict(executor.metrics())
    scheduler_metrics["total_scheduler_ms"] = wall_ms
    scheduler_metrics.update(
        _inflight_arrival_metrics(
            executor.scheduler.tick_history,
            arrival_stagger_ticks=arrival_stagger_ticks,
            submitted_after_start=max(0, args.requests - 1)
            if arrival_stagger_ticks > 0
            else 0,
        )
    )
    decode_profile = recorder.summary() if recorder is not None else None
    return _run_result(
        args,
        engine,
        responses,
        status=executor.status(),
        metrics=scheduler_metrics,
        batch_history=executor.scheduler.tick_history,
        wall_ms=wall_ms,
        extra=(
            {
                "decode_profile": decode_profile,
                "decode_profile_aggregate": _aggregate_decode_profile(
                    decode_profile or {}
                ),
            }
            if decode_profile is not None
            else None
        ),
    )


def _serving_request(
    args,
    torch,
    config,
    device: str,
    workload: tuple[dict, ...],
    workload_index: int,
    *,
    request_id: str,
) -> GRServingRequest:
    input_ids, workload_id = _request_input_ids(
        args,
        torch,
        config,
        device,
        workload,
        workload_index,
    )
    return GRServingRequest(
        request_id=request_id,
        input_ids=input_ids,
        max_decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        beam_width_policy=_make_beam_width_policy(args),
        metadata={"workload_id": workload_id} if workload_id else {},
        logits_processors=_make_logits_processors(args),
    )


def _run_result(
    args,
    engine,
    responses,
    *,
    status: dict,
    metrics: dict,
    batch_history: list,
    wall_ms: float,
    extra: dict | None = None,
) -> dict:
    result = {
        "responses": len(responses),
        "first_request_id": responses[0].request_id if responses else None,
        "first_token_ids": responses[0].token_ids if responses else (),
        "first_metadata": responses[0].metadata if responses else {},
        "engine_status": engine.status(),
        "scheduler_status": status,
        "scheduler_metrics": metrics,
        "kernel_backend_selection": _selected_kernel_backends(),
        "flashinfer_call_counts": flashinfer_call_counts(),
        "gr_trtllm_call_counts": gr_trtllm_call_counts(),
        "wall_ms": wall_ms,
        "batch_history": batch_history,
    }
    if args.record_outputs:
        result["outputs"] = _response_output_records(responses)
    if extra:
        result.update(extra)
    return result


def _submit_staggered_arrivals(
    args,
    torch,
    config,
    device: str,
    executor,
    *,
    request_prefix: str,
    next_request: int,
    current_tick: int,
    workload: tuple[dict, ...],
) -> int:
    arrival_stagger_ticks = getattr(args, "arrival_stagger_ticks", 0)
    arrival_burst_size = getattr(args, "arrival_burst_size", 1)
    while (
        next_request < args.requests
        and (next_request // arrival_burst_size) * arrival_stagger_ticks <= current_tick
    ):
        executor.submit(
            _serving_request(
                args,
                torch,
                config,
                device,
                workload,
                next_request,
                request_id=f"{request_prefix}-{next_request}",
            )
        )
        next_request += 1
    return next_request


def _inflight_arrival_metrics(
    tick_history: list[dict],
    *,
    arrival_stagger_ticks: int,
    submitted_after_start: int,
) -> dict[str, float | int]:
    inflight_admission_ticks = 0
    mixed_decode_step_ticks = 0
    max_decode_batch_size = 0
    for tick in tick_history:
        prefill_ids = set(tick.get("prefill_request_ids", []))
        decode_batches = tick.get("decode_batches", [])
        decode_ids = {
            request_id
            for batch in decode_batches
            for request_id in batch.get("request_ids", [])
        }
        if prefill_ids and (decode_ids - prefill_ids):
            inflight_admission_ticks += 1
        decode_steps = {batch.get("step") for batch in decode_batches}
        if len(decode_steps) > 1:
            mixed_decode_step_ticks += 1
        for batch in decode_batches:
            max_decode_batch_size = max(
                max_decode_batch_size, int(batch.get("size", 0))
            )
    return {
        "arrival_stagger_ticks": arrival_stagger_ticks,
        "inflight_submitted_after_start": submitted_after_start,
        "inflight_admission_ticks": inflight_admission_ticks,
        "inflight_mixed_decode_step_ticks": mixed_decode_step_ticks,
        "inflight_max_decode_batch_size": max_decode_batch_size,
    }


def _load_workload_inputs(args, torch, device: str) -> tuple[dict, ...]:
    workload_jsonl = getattr(args, "workload_jsonl", None)
    if not workload_jsonl:
        return ()
    rows: list[dict] = []
    path = Path(workload_jsonl)
    for line_no, row in iter_jsonl(path):
        token_ids = row.get("input_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"{path}:{line_no} must contain non-empty input_ids list")
        if len(token_ids) != args.context_len:
            raise ValueError(
                f"{path}:{line_no} has input_ids length {len(token_ids)}, "
                f"expected --context-len {args.context_len}"
            )
        rows.append(
            {
                "workload_id": str(row.get("request_id", f"workload-{len(rows)}")),
                "input_ids": torch.tensor([token_ids], dtype=torch.long, device=device),
                "text": row.get("text"),
            }
        )
    if len(rows) < args.requests:
        raise ValueError(
            f"workload has {len(rows)} rows but --requests is {args.requests}"
        )
    return tuple(rows)


def _request_input_ids(
    args, torch, config, device: str, workload: tuple[dict, ...], idx: int
):
    if workload:
        row = workload[idx]
        return row["input_ids"], row["workload_id"]
    return (
        torch.randint(
            0,
            config.vocab_size,
            (1, args.context_len),
            device=device,
        ),
        None,
    )


def _response_output_records(responses) -> tuple[dict, ...]:
    records = []
    for response in responses:
        raw_metadata = dict(response.metadata)
        max_new_tokens = int(
            raw_metadata.get(
                "requested_max_new_tokens",
                int(raw_metadata.get("decode_steps", 0)) + 1,
            )
        )
        records.append(
            {
                "request_id": response.request_id,
                "workload_id": raw_metadata.get("workload_id"),
                "token_ids": tuple(int(token) for token in response.token_ids),
                "scores": tuple(float(score) for score in response.scores),
                "beam_results": normalized_beam_results_from_metadata(
                    raw_metadata,
                    max_new_tokens=max_new_tokens,
                ),
                "beam_details": tuple(raw_metadata.get("beam_details", ()) or ()),
                "metadata": _compact_metadata(raw_metadata),
            }
        )
    return tuple(records)


def _make_beam_kv_pool(args, torch, config, device: str):
    capacity = getattr(args, "beam_kv_pool_capacity", 0)
    if capacity <= 0:
        return None
    if not getattr(args, "continuous", False):
        raise RuntimeError("--beam-kv-pool-capacity requires --continuous")
    dtype = choose_dtype(torch, device)
    shape = (
        config.num_layers,
        capacity,
        args.decode_steps,
        args.beam_width,
        config.num_kv_heads,
        config.head_dim,
    )
    return GRDenseBeamKVPool(
        key=torch.empty(shape, device=device, dtype=dtype),
        value=torch.empty(shape, device=device, dtype=dtype),
    )


def _make_context_kv_pool(args, torch, config, device: str):
    capacity = getattr(args, "context_kv_pool_capacity", 0)
    if capacity <= 0:
        return None
    if not getattr(args, "continuous", False):
        raise RuntimeError("--context-kv-pool-capacity requires --continuous")
    dtype = choose_dtype(torch, device)
    shape = (
        config.num_layers,
        capacity,
        args.context_len,
        config.num_kv_heads,
        config.head_dim,
    )
    return GRDenseContextKVPool(
        key=torch.empty(shape, device=device, dtype=dtype),
        value=torch.empty(shape, device=device, dtype=dtype),
    )


def _make_beam_width_policy(args):
    schedule = getattr(args, "beam_schedule", None)
    if not schedule:
        return None
    parsed = _parse_beam_schedule(schedule)
    max_width = max(parsed.values())
    if max_width > args.beam_width:
        raise ValueError(
            f"beam schedule width {max_width} exceeds --beam-width {args.beam_width}"
        )
    return ScheduledBeamPolicy(parsed)


def _make_logits_processors(args) -> tuple[object, ...]:
    token_ids = parse_unique_int_list(getattr(args, "suppress_token_ids", ""))
    if not token_ids:
        return ()
    return (TokenSuppressLogitsProcessor(token_ids),)


def _parse_beam_schedule(schedule: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for raw_entry in schedule.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError("--beam-schedule entries must be formatted as step:width")
        raw_step, raw_width = entry.split(":", 1)
        try:
            step = int(raw_step.strip())
            width = int(raw_width.strip())
        except ValueError as exc:
            raise ValueError(
                "--beam-schedule entries must contain integer step and width"
            ) from exc
        if step in parsed:
            raise ValueError(f"duplicate beam schedule step: {step}")
        parsed[step] = width
    if not parsed:
        raise ValueError("--beam-schedule must not be empty")
    if 0 not in parsed:
        raise ValueError("--beam-schedule must include step 0")
    return parsed


def _aggregate_decode_profile(summary: dict) -> dict[str, float]:
    groups = {
        "continuous_prefill_ms": "continuous.prefill",
        "continuous_decode_microbatch_total_ms": "continuous.decode_microbatch_total",
        "continuous_decode_batch_build_ms": "continuous.decode_batch_build",
        "continuous_topk_indices_ms": "continuous.topk_indices",
        "continuous_beam_selection_ms": "continuous.beam_selection",
        "continuous_beam_kv_scatter_ms": "continuous.beam_kv_scatter",
        "prefill_layer_total_ms": ("prefill.layer", ".total"),
        "prefill_qkv_ms": ("prefill.layer", ".qkv"),
        "prefill_input_norm_ms": ("prefill.layer", ".input_norm"),
        "prefill_qkv_proj_ms": ("prefill.layer", ".qkv_proj"),
        "prefill_qk_norm_rope_ms": ("prefill.layer", ".qk_norm_rope"),
        "prefill_attention_ms": ("prefill.layer", ".attention"),
        "prefill_post_attention_ms": ("prefill.layer", ".post_attention"),
        "prefill_mlp_ms": ("prefill.layer", ".mlp"),
        "qkv_ms": ".qkv",
        "input_norm_ms": ".input_norm",
        "qkv_proj_ms": ".qkv_proj",
        "qk_norm_rope_ms": ".qk_norm_rope",
        "q_norm_ms": ".q_norm",
        "k_norm_ms": ".k_norm",
        "rope_ms": ".rope",
        "beam_kv_write_ms": ".beam_kv_write",
        "decode_attention_ms": ".decode_attention",
        "post_attention_mlp_ms": ".post_attention",
        "o_proj_ms": ".o_proj",
        "post_norm_ms": ".post_norm",
        "mlp_ms": ".mlp",
        "gate_up_proj_ms": ".gate_up_proj",
        "silu_mul_ms": ".silu_mul",
        "down_proj_ms": ".down_proj",
        "layer_total_ms": ".decode_total",
    }
    aggregate = {}
    for output_name, suffix in groups.items():
        if isinstance(suffix, tuple):
            prefix, ending = suffix
            aggregate[output_name] = sum(
                stats["total_ms"]
                for name, stats in summary.items()
                if name.startswith(prefix) and name.endswith(ending)
            )
        else:
            aggregate[output_name] = sum(
                stats["total_ms"]
                for name, stats in summary.items()
                if name == suffix
                or (name.startswith("layer") and name.endswith(suffix))
            )
    aggregate["model_forward_decode_step_ms"] = summary.get(
        "model.forward_decode_step",
        {},
    ).get("total_ms", 0.0)
    aggregate["model_forward_prefill_ms"] = summary.get(
        "model.forward_prefill",
        {},
    ).get("total_ms", 0.0)
    for key in ("prefill.embed_tokens", "prefill.final_norm", "prefill.lm_head"):
        aggregate[key.replace(".", "_") + "_ms"] = summary.get(key, {}).get(
            "total_ms",
            0.0,
        )
    return aggregate


def _selected_kernel_backends() -> dict[str, str | None]:
    registry = build_default_kernel_registry()
    policy = default_kernel_selection_policy()
    selected: dict[str, str | None] = {}
    for capability in (
        CAP_RMSNORM,
        CAP_ROPE,
        CAP_QK_NORM_ROPE,
        CAP_PREFILL_ATTENTION,
        CAP_GR_DECODE_ATTENTION,
        CAP_PACKED_GEMM,
        CAP_FUSED_MLP,
    ):
        backend = policy.select(capability, registry)
        selected[capability] = backend.name if backend is not None else None
    selected["prefill_attention_actual"] = os.environ.get(
        "GR_INFERENCE_PREFILL_BACKEND",
        "auto",
    )
    return selected


def _write_summary_json(summary: dict, output_json: str) -> None:
    write_json(output_json, summary, trailing_newline=False)


def _cuda_profiler_start(torch, *, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop(torch, *, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help=(
            "Model reference. Accepts either a local checkpoint directory or a "
            "HuggingFace repo id such as Qwen/Qwen3-1.7B."
        ),
    )
    parser.add_argument(
        "--model-dir",
        help="Explicit local checkpoint directory. Takes precedence over --model.",
    )
    parser.add_argument(
        "--revision",
        help="Optional HuggingFace branch, tag, or commit for --model repo ids.",
    )
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument(
        "--beam-schedule",
        help="Scheduled dynamic beam widths, for example: 0:128,1:64,2:32",
    )
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--batched-decode", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument(
        "--arrival-stagger-ticks",
        type=int,
        default=0,
        help="For continuous serving, submit later requests every N scheduler ticks",
    )
    parser.add_argument(
        "--arrival-burst-size",
        type=int,
        default=1,
        help="With staggered arrivals, submit this many requests per arrival tick",
    )
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--workload-jsonl",
        help="Optional JSONL workload with request_id, text, and exact input_ids.",
    )
    parser.add_argument(
        "--record-outputs",
        action="store_true",
        help="Record response token IDs, scores, and beam details in output JSON.",
    )
    parser.add_argument("--return-beam-details", action="store_true")
    parser.add_argument(
        "--suppress-token-ids",
        default="",
        help="Comma/space separated token IDs to suppress before beam selection",
    )
    parser.add_argument(
        "--beam-score-mode", choices=["raw_logits", "logprob"], default="logprob"
    )
    parser.add_argument("--profile-continuous-decode", action="store_true")
    parser.add_argument(
        "--executor-sync-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize CUDA around executor prefill/decode timers. Disabled by "
            "default so benchmark/nsys runs avoid per-microbatch sync bubbles; "
            "wall_ms still synchronizes around the measured run."
        ),
    )
    parser.add_argument(
        "--profile-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize CUDA around profile sections for accurate wall timings. "
            "Use --no-profile-sync for low-overhead NVTX attribution under nsys."
        ),
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap measured runs with cudaProfilerStart/Stop for nsys capture ranges.",
    )
    parser.add_argument(
        "--beam-kv-pool-capacity",
        type=int,
        default=0,
        help="Preallocate this many dense BeamKV slots for continuous serving",
    )
    parser.add_argument(
        "--context-kv-pool-capacity",
        type=int,
        default=0,
        help="Preallocate this many dense ContextKV slots for continuous serving",
    )
    parser.add_argument(
        "--enable-prefill-cache",
        action="store_true",
        help=(
            "Reuse exact prompt prefill results and eligible long-prefix/short-suffix "
            "matches across warmup/repeat runs."
        ),
    )
    parser.add_argument(
        "--prefill-cache-min-prefix-tokens",
        type=int,
        help="Minimum shared prefix length before partial-prefix suffix extend is used.",
    )
    parser.add_argument(
        "--prefill-cache-max-entries",
        type=int,
        help="Maximum cached prompt entries. Defaults to GR_INFERENCE_PREFILL_CACHE_MAX_ENTRIES or 16.",
    )
    parser.add_argument(
        "--prefill-cache-max-tokens",
        type=int,
        default=0,
        help="Maximum cached prompt tokens across entries. 0 keeps the token budget unbounded.",
    )
    parser.add_argument(
        "--prefill-cache-page-size",
        type=int,
        help="Page-align partial-prefix matches to this token multiple. Defaults to 1.",
    )
    parser.add_argument(
        "--prefill-cache-max-decode-extend-tokens",
        type=int,
        help=(
            "Maximum suffix length to extend from a cached prefix before falling "
            "back to full prefill. Qwen3 uses prefill-extend; legacy decode-extend "
            "requires GR_INFERENCE_PREFILL_CACHE_ENABLE_DECODE_EXTEND=1. The "
            "executor default is 0, which keeps partial-prefix reuse disabled "
            "unless this option or the matching env var enables it."
        ),
    )
    parser.add_argument(
        "--profile-detail",
        choices=["coarse", "fine"],
        default="coarse",
        help="coarse avoids extra syncs; fine reports per-operator sub-stages",
    )
    parser.add_argument("--verbose-metadata", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    summary = run_serving(args)
    if args.output_json:
        _write_summary_json(summary, args.output_json)
    print("Real-weight serving smoke")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
