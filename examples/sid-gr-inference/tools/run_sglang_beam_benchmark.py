# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run SGLang PR beam-search benchmark on a shared JSONL workload."""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from tool_utils import json_dumps, load_optional_tokenizer
from tool_utils import numeric_median as _median
from tool_utils import read_jsonl, write_json


def run_benchmark(args) -> dict[str, Any]:
    import sglang as sgl
    import torch

    tokenizer = _load_tokenizer(args)
    workload = _load_workload(Path(args.workload_jsonl), limit=args.requests)
    engine = _create_engine(sgl, args)
    module_profiler = (
        _SGLangModuleProfiler(torch, engine) if args.profile_modules else None
    )
    try:
        if module_profiler is not None:
            module_profiler.install()
        for _ in range(args.warmup_runs):
            _run_once(engine, workload, tokenizer, args, record_outputs=False)
        if module_profiler is not None:
            module_profiler.reset()
        _cuda_profiler_start(torch, enabled=args.cuda_profiler_range)
        try:
            measured = [
                _run_once(engine, workload, tokenizer, args, record_outputs=True)
                for _ in range(args.repeat)
            ]
        finally:
            _cuda_profiler_stop(torch, enabled=args.cuda_profiler_range)
            if module_profiler is not None:
                module_profiler.remove()
        if not args.allow_missing_beam_scores:
            _validate_beam_score_outputs(measured, expected_beam_width=args.beam_width)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()

    return {
        "framework": "sglang",
        "model_dir": args.model_dir,
        "workload_jsonl": args.workload_jsonl,
        "context_len": args.context_len,
        "decode_steps": args.decode_steps,
        "beam_width": args.beam_width,
        "requests": len(workload),
        "arrival_mode": args.arrival_mode,
        "arrival_stagger_ms": args.arrival_stagger_ms,
        "arrival_burst_size": args.arrival_burst_size,
        "warmup_runs": args.warmup_runs,
        "repeat": args.repeat,
        "use_input_ids": args.use_input_ids,
        "disable_radix_cache": args.disable_radix_cache,
        "disable_piecewise_cuda_graph": args.disable_piecewise_cuda_graph,
        "disable_cuda_graph": args.disable_cuda_graph,
        "profile_modules": args.profile_modules,
        "sampling_params": _sampling_params(args),
        **_run_metric_summaries(
            measured,
            (
                ("wall_ms", "wall_ms", None, False, True),
                (
                    "output_normalization_ms",
                    "output_normalization_ms",
                    0.0,
                    False,
                    True,
                ),
                ("request_latency_ms_p50", "request_latency_ms_p50", None, True, False),
                ("request_latency_ms_p95", "request_latency_ms_p95", None, True, False),
                ("qps", "qps", None, False, False),
                (
                    "generated_tokens_per_s",
                    "generated_tokens_per_s",
                    None,
                    False,
                    False,
                ),
                ("beam_candidates_per_s", "beam_candidates_per_s", None, False, False),
            ),
        ),
        "runs": measured,
        "environment": _environment(torch),
        "sglang_commit": _git_commit(args.sglang_repo),
        **(
            {
                "sglang_module_profile": module_profiler.summary(),
                "sglang_module_profile_aggregate": module_profiler.aggregate(),
                "sglang_module_profile_model_path": module_profiler.model_path,
                "sglang_module_profile_visited_types": module_profiler.visited_types[
                    :120
                ],
            }
            if module_profiler is not None
            else {}
        ),
    }


def _engine_kwargs(args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.tp_size is not None:
        kwargs["tp_size"] = args.tp_size
    if args.mem_fraction_static is not None:
        kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.disable_radix_cache:
        kwargs["disable_radix_cache"] = True
    if args.disable_piecewise_cuda_graph:
        kwargs["disable_piecewise_cuda_graph"] = True
    if args.disable_cuda_graph:
        kwargs["disable_cuda_graph"] = True
    if args.profile_modules:
        kwargs["enable_layerwise_nvtx_marker"] = True
        kwargs["disable_piecewise_cuda_graph"] = True
    return kwargs


def _create_engine(sgl, args):
    kwargs = {
        "model_path": args.model_dir,
        "enable_beam_search": True,
        **_engine_kwargs(args),
    }
    try:
        return sgl.Engine(**kwargs)
    except TypeError as exc:
        if "disable_radix_cache" in str(exc):
            kwargs.pop("disable_radix_cache", None)
            return sgl.Engine(**kwargs)
        if "enable_beam_search" not in str(exc):
            raise
        kwargs.pop("enable_beam_search", None)
        return sgl.Engine(**kwargs)


def _run_once(
    engine, workload: list[dict], tokenizer, args, *, record_outputs: bool
) -> dict[str, Any]:
    if args.arrival_mode == "staggered":
        return _run_staggered_once(
            engine, workload, tokenizer, args, record_outputs=record_outputs
        )
    return _run_batch_once(
        engine, workload, tokenizer, args, record_outputs=record_outputs
    )


def _run_batch_once(
    engine,
    workload: list[dict],
    tokenizer,
    args,
    *,
    record_outputs: bool,
) -> dict[str, Any]:
    prompts = [row.get("text") for row in workload]
    input_ids = [row["input_ids"] for row in workload]
    sampling_params = _sampling_params(args)
    start = time.perf_counter()
    outputs = _engine_generate(
        engine,
        prompts,
        input_ids,
        sampling_params,
        use_input_ids=args.use_input_ids,
    )
    wall_ms = (time.perf_counter() - start) * 1000.0
    normalize_start = time.perf_counter()
    normalized = (
        _normalize_outputs(outputs, workload, tokenizer) if record_outputs else ()
    )
    output_normalization_ms = (time.perf_counter() - normalize_start) * 1000.0
    request_count = len(workload)
    generated_tokens = request_count * args.decode_steps
    beam_candidates = generated_tokens * args.beam_width
    elapsed_s = wall_ms / 1000.0
    return {
        "wall_ms": wall_ms,
        "qps": request_count / elapsed_s if elapsed_s > 0 else None,
        "generated_tokens_per_s": generated_tokens / elapsed_s
        if elapsed_s > 0
        else None,
        "beam_candidates_per_s": beam_candidates / elapsed_s if elapsed_s > 0 else None,
        "arrival_mode": "batch",
        "request_latencies_ms": [wall_ms for _ in workload],
        "request_latency_ms_p50": wall_ms,
        "request_latency_ms_p95": wall_ms,
        "output_normalization_ms": output_normalization_ms,
        "outputs": normalized,
    }


def _run_staggered_once(
    engine,
    workload: list[dict],
    tokenizer,
    args,
    *,
    record_outputs: bool,
) -> dict[str, Any]:
    sampling_params = _sampling_params(args)
    start = time.perf_counter()

    def run_request(index: int, row: dict) -> dict[str, Any]:
        target_submit = start + _arrival_delay_s(index, args)
        remaining = target_submit - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        submit_time = time.perf_counter()
        output = _engine_generate_single(
            engine,
            row.get("text"),
            row["input_ids"],
            sampling_params,
            use_input_ids=args.use_input_ids,
        )
        complete_time = time.perf_counter()
        return {
            "index": index,
            "request_id": row["request_id"],
            "submit_ms": (submit_time - start) * 1000.0,
            "complete_ms": (complete_time - start) * 1000.0,
            "latency_ms": (complete_time - submit_time) * 1000.0,
            "output": output,
        }

    max_workers = max(1, min(args.staggered_workers, len(workload)))
    if max_workers == 1:
        request_results = [
            run_request(index, row) for index, row in enumerate(workload)
        ]
        client_mode = "serial"
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            request_results = list(
                executor.map(lambda item: run_request(*item), enumerate(workload))
            )
        client_mode = "threaded"

    wall_ms = (time.perf_counter() - start) * 1000.0
    request_results.sort(key=lambda row: row["index"])
    outputs = [row["output"] for row in request_results]
    normalize_start = time.perf_counter()
    normalized = (
        _normalize_outputs(outputs, workload, tokenizer) if record_outputs else ()
    )
    output_normalization_ms = (time.perf_counter() - normalize_start) * 1000.0
    request_count = len(workload)
    generated_tokens = request_count * args.decode_steps
    beam_candidates = generated_tokens * args.beam_width
    elapsed_s = wall_ms / 1000.0
    latencies = [float(row["latency_ms"]) for row in request_results]
    return {
        "wall_ms": wall_ms,
        "qps": request_count / elapsed_s if elapsed_s > 0 else None,
        "generated_tokens_per_s": generated_tokens / elapsed_s
        if elapsed_s > 0
        else None,
        "beam_candidates_per_s": beam_candidates / elapsed_s if elapsed_s > 0 else None,
        "arrival_mode": "staggered",
        "client_mode": client_mode,
        "arrival_stagger_ms": args.arrival_stagger_ms,
        "arrival_burst_size": args.arrival_burst_size,
        "request_latencies_ms": latencies,
        "request_latency_ms_p50": _percentile(latencies, 0.50),
        "request_latency_ms_p95": _percentile(latencies, 0.95),
        "output_normalization_ms": output_normalization_ms,
        "request_timeline": tuple(
            {
                "request_id": row["request_id"],
                "submit_ms": row["submit_ms"],
                "complete_ms": row["complete_ms"],
                "latency_ms": row["latency_ms"],
            }
            for row in request_results
        ),
        "outputs": normalized,
    }


def _sampling_params(args) -> dict[str, int]:
    return {
        "max_new_tokens": args.decode_steps,
        "n": args.beam_width,
    }


def _arrival_delay_s(index: int, args) -> float:
    arrival_group = index // args.arrival_burst_size
    return arrival_group * args.arrival_stagger_ms / 1000.0


def _engine_generate(
    engine,
    prompts: list[str | None],
    input_ids: list[list[int]],
    sampling_params: Mapping[str, Any],
    *,
    use_input_ids: bool,
):
    generate = engine.generate
    if use_input_ids:
        return generate(input_ids=input_ids, sampling_params=sampling_params)
    try:
        return generate(prompts, sampling_params=sampling_params)
    except TypeError:
        signature = inspect.signature(generate)
        if "prompt" in signature.parameters:
            return [
                generate(prompt=prompt, sampling_params=sampling_params)
                for prompt in prompts
            ]
        return [generate(prompt, sampling_params) for prompt in prompts]


def _engine_generate_single(
    engine,
    prompt: str | None,
    input_ids: list[int],
    sampling_params: Mapping[str, Any],
    *,
    use_input_ids: bool,
):
    generate = engine.generate
    if use_input_ids:
        output = generate(input_ids=[input_ids], sampling_params=sampling_params)
    else:
        try:
            output = generate([prompt], sampling_params=sampling_params)
        except TypeError:
            signature = inspect.signature(generate)
            if "prompt" in signature.parameters:
                output = generate(prompt=prompt, sampling_params=sampling_params)
            else:
                output = generate(prompt, sampling_params)
    if isinstance(output, Mapping):
        return output
    output_rows = list(output)
    if len(output_rows) != 1:
        return output_rows
    return output_rows[0]


def _normalize_outputs(
    outputs: Any, workload: list[dict], tokenizer
) -> tuple[dict[str, Any], ...]:
    if isinstance(outputs, Mapping):
        output_rows = [outputs]
    else:
        output_rows = list(outputs)
    if len(output_rows) == len(workload):
        return tuple(
            _normalize_request_output(workload[idx], output, tokenizer)
            for idx, output in enumerate(output_rows)
        )
    if workload and len(output_rows) % len(workload) == 0:
        beams_per_request = len(output_rows) // len(workload)
        records = []
        for idx, row in enumerate(workload):
            start = idx * beams_per_request
            end = start + beams_per_request
            candidates = output_rows[start:end]
            records.append(_normalize_request_output(row, candidates, tokenizer))
        return tuple(records)
    raise ValueError(
        f"cannot map {len(output_rows)} SGLang output rows to {len(workload)} workload rows"
    )


def _normalize_request_output(row: dict, output: Any, tokenizer) -> dict[str, Any]:
    beams = _extract_beams(output, tokenizer)
    if not beams:
        candidates = output if isinstance(output, list | tuple) else [output]
        beams = tuple(
            _candidate_to_beam(candidate, rank, tokenizer)
            for rank, candidate in enumerate(candidates)
        )
    return {
        "request_id": row["request_id"],
        "workload_id": row["request_id"],
        "prompt_tokens": len(row["input_ids"]),
        "beams": beams,
        "raw_output": output,
    }


def _candidate_to_beam(candidate: Any, rank: int, tokenizer) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        return {
            "rank": rank,
            "text": None,
            "token_ids": (),
            "score": None,
            "raw": candidate,
        }
    text = _candidate_text(candidate)
    token_ids = _candidate_token_ids(candidate)
    if token_ids is None and text is not None and tokenizer is not None:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    return {
        "rank": rank,
        "text": text,
        "token_ids": tuple(token_ids or ()),
        "score": _candidate_score(candidate),
        "raw": candidate,
    }


def _extract_beams(output: Any, tokenizer) -> tuple[dict[str, Any], ...]:
    if not isinstance(output, Mapping):
        return ()
    meta = _mapping(output.get("meta_info"))
    candidates = (
        meta.get("beam_results")
        or output.get("beam_results")
        or output.get("choices")
        or []
    )
    beams = []
    for rank, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        beams.append(_candidate_to_beam(candidate, rank, tokenizer))
    return tuple(beams)


def _candidate_text(candidate: Mapping[str, Any]) -> str | None:
    if isinstance(candidate.get("text"), str):
        return candidate["text"]
    message = _mapping(candidate.get("message"))
    if isinstance(message.get("content"), str):
        return message["content"]
    return None


def _candidate_token_ids(candidate: Mapping[str, Any]) -> list[int] | None:
    for key in (
        "token_ids",
        "output_ids",
        "input_ids",
        "output_token_ids",
        "generated_token_ids",
    ):
        value = candidate.get(key)
        if isinstance(value, list):
            return [int(token) for token in value]
    meta = _mapping(candidate.get("meta_info"))
    for key in ("token_ids", "output_ids", "output_token_ids", "generated_token_ids"):
        value = meta.get(key)
        if isinstance(value, list):
            return [int(token) for token in value]
    token_ids = _token_ids_from_logprobs(meta.get("output_token_logprobs"))
    if token_ids is not None:
        return token_ids
    return None


def _token_ids_from_logprobs(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    token_ids = []
    for entry in value:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            token_ids.append(int(entry[1]))
        elif isinstance(entry, Mapping):
            for key in ("token_id", "id"):
                if key in entry:
                    token_ids.append(int(entry[key]))
                    break
    return token_ids or None


def _load_tokenizer(args):
    return load_optional_tokenizer(args)


def _candidate_score(candidate: Mapping[str, Any]) -> float | None:
    for source in (
        candidate,
        _mapping(candidate.get("meta_info")),
        _mapping(candidate.get("sglext")),
    ):
        for key in ("sequence_score", "score", "cumulative_score"):
            value = source.get(key)
            if value is not None:
                return float(value)
    return None


def _validate_beam_score_outputs(
    runs: list[dict[str, Any]], *, expected_beam_width: int
) -> None:
    if not runs:
        return
    outputs = runs[-1].get("outputs") or ()
    for record in outputs:
        beams = record.get("beams") or ()
        if len(beams) != expected_beam_width:
            raise RuntimeError(
                f"SGLang returned {len(beams)} candidates; expected {expected_beam_width}. "
                "This does not look like the beam-search output shape."
            )
        if not any(beam.get("score") is not None for beam in beams):
            raise RuntimeError(
                "SGLang returned beam-width candidates but no candidate scores. "
                "This usually means current SGLang is doing ordinary multi-sampling "
                "instead of beam search. Use the SGLang large-beam branch/API or pass "
                "--allow-missing-beam-scores for performance-only diagnostics."
            )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_workload(path: Path, *, limit: int) -> list[dict]:
    rows = read_jsonl(path, limit=limit)
    if len(rows) < limit:
        raise ValueError(f"{path} has {len(rows)} rows, expected {limit}")
    return rows


def _environment(torch) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_capability": torch.cuda.get_device_capability(0)
        if cuda_available
        else None,
    }


def _git_commit(repo: str | None) -> str | None:
    if not repo:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _median_optional(values) -> float | None:
    return _median(value for value in values if value is not None)


def _run_metric_summaries(
    runs: list[dict[str, Any]],
    specs: tuple[tuple[str, str, float | None, bool, bool], ...],
) -> dict[str, Any]:
    summary = {}
    for output_name, run_key, default, optional, include_samples in specs:
        samples = [run.get(run_key, default) for run in runs]
        if include_samples:
            summary[f"{output_name}_samples"] = samples
        summary[f"{output_name}_median"] = (
            _median_optional(samples) if optional else _median(samples)
        )
    return summary


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    index = min(len(sorted_values) - 1, int(quantile * (len(sorted_values) - 1)))
    return sorted_values[index]


class _SGLangModuleProfiler:
    TARGET_CLASS_TO_BUCKET = {
        "Qwen3Attention": "attention_block_ms",
        "Qwen2MLP": "mlp_block_ms",
        "Qwen3DecoderLayer": "decoder_layer_ms",
        "Qwen3ForCausalLM": "model_forward_ms",
        "Qwen3Model": "model_body_ms",
        "QKVParallelLinear": "qkv_proj_ms",
        "MergedColumnParallelLinear": "gate_up_proj_ms",
        "RowParallelLinear": "row_parallel_linear_ms",
        "SiluAndMul": "silu_mul_ms",
        "RMSNorm": "rmsnorm_ms",
        "RadixAttention": "radix_attention_ms",
        "LogitsProcessor": "logits_processor_ms",
    }

    TARGET_NAME_RULES = (
        ("q_norm", "q_norm_ms"),
        ("k_norm", "k_norm_ms"),
        ("rotary_emb", "rope_ms"),
        ("o_proj", "o_proj_ms"),
        ("down_proj", "down_proj_ms"),
        ("gate_up_proj", "gate_up_proj_ms"),
        ("qkv_proj", "qkv_proj_ms"),
        ("input_layernorm", "input_norm_ms"),
        ("post_attention_layernorm", "post_norm_ms"),
        ("lm_head", "lm_head_ms"),
        ("logits_processor", "logits_processor_ms"),
    )

    def __init__(self, torch, engine) -> None:
        self.torch = torch
        self.engine = engine
        self.handles = []
        self.starts = {}
        self.records: dict[str, dict[str, float | int]] = {}
        self.model_path = None
        self.visited_types = []

    def install(self) -> None:
        model = self._find_model(self.engine)
        self.model_path = self._model_path(model)
        if model is None or not hasattr(model, "named_modules"):
            return
        seen = set()
        for name, module in model.named_modules():
            bucket = self._bucket_for(name, module)
            if bucket is None:
                continue
            key = (id(module), bucket)
            if key in seen:
                continue
            seen.add(key)
            self.handles.append(
                module.register_forward_pre_hook(self._pre_hook(bucket))
            )
            self.handles.append(module.register_forward_hook(self._post_hook(bucket)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def reset(self) -> None:
        self.records.clear()
        self.starts.clear()

    def summary(self) -> dict[str, dict[str, float | int]]:
        return dict(sorted(self.records.items()))

    def aggregate(self) -> dict[str, float]:
        aggregate = {
            bucket: float(stats["total_ms"]) for bucket, stats in self.records.items()
        }
        aggregate["attention_ms"] = aggregate.get("radix_attention_ms", 0.0)
        aggregate["mlp_ms"] = aggregate.get("mlp_block_ms", 0.0)
        aggregate["qk_norm_rope_ms"] = (
            aggregate.get("q_norm_ms", 0.0)
            + aggregate.get("k_norm_ms", 0.0)
            + aggregate.get("rope_ms", 0.0)
        )
        aggregate["qkv_ms"] = (
            aggregate.get("input_norm_ms", 0.0)
            + aggregate.get("qkv_proj_ms", 0.0)
            + aggregate["qk_norm_rope_ms"]
        )
        aggregate["post_attention_ms"] = (
            aggregate.get("o_proj_ms", 0.0)
            + aggregate.get("post_norm_ms", 0.0)
            + aggregate.get("mlp_block_ms", 0.0)
        )
        return dict(sorted(aggregate.items()))

    def _pre_hook(self, bucket: str):
        def hook(module, inputs):
            self._sync()
            self.starts[id(module), bucket] = time.perf_counter()

        return hook

    def _post_hook(self, bucket: str):
        def hook(module, inputs, output):
            self._sync()
            start = self.starts.pop((id(module), bucket), None)
            if start is None:
                return
            elapsed = (time.perf_counter() - start) * 1000.0
            record = self.records.setdefault(
                bucket,
                {"total_ms": 0.0, "count": 0, "avg_ms": 0.0},
            )
            record["total_ms"] = float(record["total_ms"]) + elapsed
            record["count"] = int(record["count"]) + 1
            record["avg_ms"] = float(record["total_ms"]) / int(record["count"])

        return hook

    def _sync(self) -> None:
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def _bucket_for(self, name: str, module) -> str | None:
        class_name = module.__class__.__name__
        for suffix, bucket in self.TARGET_NAME_RULES:
            if name.endswith(suffix):
                return bucket
        return self.TARGET_CLASS_TO_BUCKET.get(class_name)

    def _find_model(self, obj, *, depth: int = 0, seen=None):
        if seen is None:
            seen = set()
        if obj is None or id(obj) in seen or depth > 6:
            return None
        seen.add(id(obj))
        if len(self.visited_types) < 300:
            self.visited_types.append(
                obj.__class__.__module__ + "." + obj.__class__.__name__
            )
        if hasattr(obj, "named_modules") and obj.__class__.__name__.startswith("Qwen"):
            return obj
        for attr in (
            "model",
            "model_runner",
            "tp_model_worker",
            "worker",
            "server",
            "engine",
            "tokenizer_manager",
            "scheduler",
        ):
            child = getattr(obj, attr, None)
            found = self._find_model(child, depth=depth + 1, seen=seen)
            if found is not None:
                return found
        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = self._find_model(item, depth=depth + 1, seen=seen)
                if found is not None:
                    return found
        if isinstance(obj, dict):
            for item in obj.values():
                found = self._find_model(item, depth=depth + 1, seen=seen)
                if found is not None:
                    return found
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for name, child in list(attrs.items()):
                if name.startswith("__") or callable(child):
                    continue
                found = self._find_model(child, depth=depth + 1, seen=seen)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _model_path(model) -> str | None:
        if model is None:
            return None
        return model.__class__.__module__ + "." + model.__class__.__name__


def _cuda_profiler_start(torch, *, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()


def _cuda_profiler_stop(torch, *, enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--workload-jsonl", required=True)
    parser.add_argument("--context-len", type=int, default=5000)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=256)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--arrival-mode",
        choices=("batch", "staggered"),
        default="batch",
        help=(
            "batch submits all requests in one Engine.generate call; staggered "
            "submits per-request Engine.generate calls after offsets. Use "
            "--staggered-workers > 1 only with an Engine/API that supports "
            "concurrent sync calls."
        ),
    )
    parser.add_argument(
        "--arrival-stagger-ms",
        type=float,
        default=0.0,
        help="For --arrival-mode staggered, submit each arrival group after this delay.",
    )
    parser.add_argument(
        "--arrival-burst-size",
        type=int,
        default=1,
        help="For --arrival-mode staggered, submit this many requests per arrival group.",
    )
    parser.add_argument(
        "--staggered-workers",
        type=int,
        default=1,
        help=(
            "Maximum client threads used by --arrival-mode staggered. The embedded "
            "SGLang sync Engine reuses one event loop, so the safe default is 1."
        ),
    )
    parser.add_argument(
        "--use-input-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call SGLang Engine.generate(input_ids=...) instead of prompt text.",
    )
    parser.add_argument("--no-tokenizer", action="store_true")
    parser.add_argument("--require-tokenizer", action="store_true")
    parser.add_argument("--tp-size", type=int)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument(
        "--disable-radix-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable SGLang prefix/radix cache for fair no-prefix-cache comparisons.",
    )
    parser.add_argument(
        "--disable-piecewise-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable SGLang piecewise CUDA Graph while leaving classic decode CUDA Graph available.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable SGLang classic CUDA Graph.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Wrap measured runs with cudaProfilerStart/Stop for nsys capture ranges.",
    )
    parser.add_argument("--sglang-repo")
    parser.add_argument(
        "--allow-missing-beam-scores",
        action="store_true",
        help=(
            "Allow score-less beam-width outputs. This is useful only for "
            "diagnosing SGLang performance, not for GR-vs-SGLang beam correctness."
        ),
    )
    parser.add_argument(
        "--profile-modules",
        action="store_true",
        help="Install lightweight forward hooks for SGLang Qwen module timing.",
    )
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.arrival_burst_size <= 0:
        raise ValueError("--arrival-burst-size must be positive")
    if args.arrival_stagger_ms < 0:
        raise ValueError("--arrival-stagger-ms must be non-negative")
    if args.arrival_mode == "batch" and args.arrival_stagger_ms:
        raise ValueError("--arrival-stagger-ms requires --arrival-mode staggered")
    if args.staggered_workers <= 0:
        raise ValueError("--staggered-workers must be positive")
    summary = run_benchmark(args)
    write_json(args.output_json, summary, trailing_newline=False)
    print(json_dumps(summary))


if __name__ == "__main__":
    main()
