"""Run a real-weight Qwen3 GR HTTP serving demo."""

from __future__ import annotations

import argparse
from typing import Any, Mapping

from tool_utils import bootstrap_repo_paths

bootstrap_repo_paths(__file__, include_tools=True)

from gr_inference import (  # noqa: E402
    GRContinuousBatchingPolicy,
    GRContinuousScheduler,
    GRContinuousServingExecutor,
    GRDecodeAttention,
    GRDecodeEngine,
    GRHTTPServingAdapter,
    GRHTTPValidationPolicy,
    GRInProcessServingFacade,
    GRServingConfig,
    GRServingEngine,
    GRServingRequest,
    GRServingWorker,
    TrieItemMaskProviderStore,
)
from gr_inference.gr_runtime import TimingRecorder, logits_processors_from_specs  # noqa: E402
from run_qwen3_real_weight_serving import (  # noqa: E402
    _make_beam_kv_pool,
    _make_context_kv_pool,
    load_model,
    make_decode_backend,
)
from gr_inference.gr_serving.cli import parse_unique_int_list  # noqa: E402
from gr_inference.gr_serving.http import beam_width_policy_from_payload  # noqa: E402
from gr_inference.gr_serving.payload import (  # noqa: E402
    optional_int as _optional_int,
    required_field as _required_field,
    required_str as _required_str,
)


def build_http_serving_adapter(args) -> GRHTTPServingAdapter:
    import torch

    _normalize_args(args)
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
            enable_batched_decode=True,
            return_beam_details=args.return_beam_details,
            beam_score_mode=args.beam_score_mode,
        ),
    )
    beam_kv_pool = _make_beam_kv_pool(args, torch, config, device)
    context_kv_pool = _make_context_kv_pool(args, torch, config, device)
    if args.warmup_online_shapes:
        _warmup_online_shapes(
            args,
            torch=torch,
            engine=engine,
            beam_kv_pool=beam_kv_pool,
            context_kv_pool=context_kv_pool,
            device=device,
        )
        if args.freeze_cuda_graphs_after_warmup:
            _freeze_cuda_graph_captures(engine)
    scheduler = GRContinuousScheduler(
        policy=GRContinuousBatchingPolicy(
            max_prefill_batch_size=args.max_batch_size,
            max_decode_batch_size=args.max_batch_size,
            max_running_requests=(
                args.beam_kv_pool_capacity if beam_kv_pool is not None else None
            ),
            max_finished_requests=args.max_finished_requests,
        )
    )
    recorder = (
        TimingRecorder(sync_module=torch, detail=args.profile_detail)
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
        decode_cuda_graph_batch_buckets=tuple(args.decode_cuda_graph_batch_buckets),
    )
    facade = GRInProcessServingFacade(
        executor,
        item_mask_provider_store=_make_catalog_store(args, config),
    )
    serving = (
        facade
        if args.disable_background_worker
        else GRServingWorker(
            facade,
            tick_interval_s=args.worker_tick_interval_s,
            idle_sleep_s=args.worker_idle_sleep_s,
            decode_log_interval=args.decode_log_interval,
            autostart=True,
        )
    )
    return GRHTTPServingAdapter(
        serving,
        request_factory=make_torch_request_factory(
            torch,
            device="cpu",
            default_decode_steps=args.decode_steps,
            default_beam_width=args.beam_width,
            suppress_token_ids_on_ignore_eos=_ignore_eos_suppress_token_ids(args),
        ),
        validation_policy=_make_http_validation_policy(args),
        api_key=args.api_key,
        build_info=_make_build_info(
            args,
            torch,
            config,
            device,
            beam_kv_pool,
            context_kv_pool,
        ),
        enable_request_logging=args.enable_log_requests,
        log_requests_level=args.log_requests_level,
    )


def make_torch_request_factory(
    torch,
    *,
    device: str,
    default_decode_steps: int,
    default_beam_width: int,
    suppress_token_ids_on_ignore_eos: tuple[int, ...] = (),
):
    def factory(payload: Mapping[str, Any]) -> GRServingRequest:
        request_id = _required_str(payload, "request_id")
        input_ids = torch.as_tensor(
            _required_field(payload, "input_ids"),
            dtype=torch.long,
            device=device,
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must be shaped [S] or [1, S]")
        metadata = dict(payload.get("metadata", {}))
        if "timeout_ticks" in payload:
            metadata["timeout_ticks"] = _optional_int(payload, "timeout_ticks")
        logits_processor_specs = list(payload.get("logits_processors", ()) or ())
        if metadata.get("ignore_eos") and suppress_token_ids_on_ignore_eos:
            logits_processor_specs.append(
                {
                    "type": "token_suppress",
                    "token_ids": suppress_token_ids_on_ignore_eos,
                }
            )
        return GRServingRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_decode_steps=int(payload.get("max_decode_steps", default_decode_steps)),
            beam_width=int(payload.get("beam_width", default_beam_width)),
            metadata=metadata,
            beam_width_policy=beam_width_policy_from_payload(
                payload,
                max_beam_width=int(payload.get("beam_width", default_beam_width)),
            ),
            stop_token_ids=tuple(int(token) for token in payload.get("stop_token_ids", ())),
            logits_processors=logits_processors_from_specs(logits_processor_specs),
        )

    return factory


def _warmup_online_shapes(
    args,
    *,
    torch,
    engine,
    beam_kv_pool,
    context_kv_pool,
    device: str,
) -> None:
    """Warm common continuous serving shapes before accepting HTTP traffic."""

    executor = GRContinuousServingExecutor(
        engine=engine,
        scheduler=GRContinuousScheduler(
            policy=GRContinuousBatchingPolicy(
                max_prefill_batch_size=args.max_batch_size,
                max_decode_batch_size=args.max_batch_size,
                max_running_requests=(
                    args.beam_kv_pool_capacity if beam_kv_pool is not None else None
                ),
                max_finished_requests=args.max_batch_size * args.max_batch_size,
            )
        ),
        synchronize=torch.cuda.synchronize if device == "cuda" else None,
        sync_timing=args.executor_sync_timing,
        beam_kv_pool=beam_kv_pool,
        context_kv_pool=context_kv_pool,
        prefill_cache_enabled=args.enable_prefill_cache,
        max_prefill_cache_entries=args.prefill_cache_max_entries,
        max_prefill_cache_tokens=args.prefill_cache_max_tokens or None,
        prefill_cache_page_size=args.prefill_cache_page_size,
        min_prefill_cache_prefix_tokens=args.prefill_cache_min_prefix_tokens,
        max_prefill_cache_decode_extend_tokens=args.prefill_cache_max_decode_extend_tokens,
        decode_cuda_graph_batch_buckets=tuple(args.decode_cuda_graph_batch_buckets),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    pool_capacity = _online_warmup_pool_capacity(args, beam_kv_pool, context_kv_pool)
    blocker_beam_width = 1 if args.beam_width > 1 else args.beam_width
    logits_processors = _warmup_online_logits_processors(args)
    for case_index, (first_slot, batch_size) in enumerate(
        _online_warmup_cases(args, pool_capacity)
    ):
        if first_slot > 0:
            for idx in range(first_slot):
                input_ids = torch.randint(
                    0,
                    engine.model.config.vocab_size,
                    (1, args.context_len),
                    dtype=torch.long,
                    device="cpu",
                    generator=generator,
                )
                executor.submit(
                    GRServingRequest(
                        request_id=(
                            f"startup-warmup-c{case_index}-s{first_slot}"
                            f"-b{batch_size}-blocker{idx}"
                        ),
                        input_ids=input_ids,
                        max_decode_steps=args.decode_steps,
                        beam_width=blocker_beam_width,
                        metadata={
                            "warmup": True,
                            "startup_shape_warmup": True,
                            "warmup_pool_first_slot": first_slot,
                            "warmup_target_batch_size": batch_size,
                            "warmup_target_request": False,
                            "ignore_eos": True,
                        },
                        logits_processors=logits_processors,
                    )
                )
            # Admit blockers through prefill only, so their pool leases are live
            # when the target batch is prefilling the offset window.
            executor.scheduler.tick(
                prefill_executor=executor._run_prefill,
                decode_executor=lambda _decode_batches: (),
            )

        for idx in range(batch_size):
            input_ids = torch.randint(
                0,
                engine.model.config.vocab_size,
                (1, args.context_len),
                dtype=torch.long,
                device="cpu",
                generator=generator,
            )
            executor.submit(
                GRServingRequest(
                    request_id=(
                        f"startup-warmup-c{case_index}-s{first_slot}"
                        f"-b{batch_size}-target{idx}"
                    ),
                    input_ids=input_ids,
                    max_decode_steps=args.decode_steps,
                    beam_width=args.beam_width,
                    metadata={
                        "warmup": True,
                        "startup_shape_warmup": True,
                        "warmup_pool_first_slot": first_slot,
                        "warmup_target_batch_size": batch_size,
                        "warmup_target_request": True,
                        "ignore_eos": True,
                    },
                    logits_processors=logits_processors,
                )
            )
        executor.run_until_empty(max_ticks=args.decode_steps + 2)
    if device == "cuda":
        torch.cuda.synchronize()


def _warmup_online_logits_processors(args) -> tuple[Any, ...]:
    suppress_token_ids = _ignore_eos_suppress_token_ids(args)
    if not suppress_token_ids:
        return ()
    return logits_processors_from_specs(
        [
            {
                "type": "token_suppress",
                "token_ids": suppress_token_ids,
            }
        ]
    )


def _freeze_cuda_graph_captures(engine) -> None:
    for attribute in ("_prefill_cuda_graph_runner", "_decode_cuda_graph_runner"):
        runner = getattr(engine, attribute, None)
        freeze = getattr(runner, "freeze_captures", None)
        if callable(freeze):
            freeze()


def _online_warmup_pool_capacity(args, beam_kv_pool, context_kv_pool) -> int:
    capacities: list[int] = []
    for pool in (beam_kv_pool, context_kv_pool):
        capacity = getattr(pool, "capacity", None)
        if capacity is not None:
            capacities.append(int(capacity))
    if capacities:
        return max(1, min(capacities))
    return max(1, int(args.max_batch_size))


def _online_warmup_cases(args, pool_capacity: int) -> tuple[tuple[int, int], ...]:
    max_batch_size = max(1, min(int(args.max_batch_size), int(pool_capacity)))
    if not args.warmup_online_pool_windows or int(args.beam_width) <= 1:
        cases = [(0, batch_size) for batch_size in range(1, max_batch_size + 1)]
    else:
        cases = [
            (first_slot, batch_size)
            for batch_size in range(1, max_batch_size + 1)
            for first_slot in range(0, int(pool_capacity) - batch_size + 1)
        ]
    max_cases = int(args.warmup_online_max_cases or 0)
    if max_cases > 0:
        cases = cases[:max_cases]
    return tuple(cases)


def _make_catalog_store(args, config) -> TrieItemMaskProviderStore | None:
    if args.catalog_jsonl is None:
        return None
    return TrieItemMaskProviderStore.from_jsonl(
        args.catalog_jsonl,
        vocab_size=args.catalog_vocab_size or config.vocab_size,
        eos_token_id=args.catalog_eos_token_id,
        allow_eos_for_terminal=args.catalog_allow_eos_for_terminal,
        item_id_field=args.catalog_item_id_field,
        token_ids_field=args.catalog_token_ids_field,
        metadata_field=args.catalog_metadata_field,
        allow_duplicate_item_ids=args.catalog_allow_duplicate_item_ids,
        allow_duplicate_token_paths=args.catalog_allow_duplicate_token_paths,
    )


def _normalize_args(args) -> None:
    args.continuous = True
    args.batched_decode = True
    if not hasattr(args, "warmup_online_shapes"):
        args.warmup_online_shapes = True
    if not hasattr(args, "warmup_online_pool_windows"):
        args.warmup_online_pool_windows = True
    if not hasattr(args, "warmup_online_max_cases"):
        args.warmup_online_max_cases = 64
    if not hasattr(args, "freeze_cuda_graphs_after_warmup"):
        args.freeze_cuda_graphs_after_warmup = True
    if not hasattr(args, "context_kv_pool_capacity"):
        args.context_kv_pool_capacity = 0
    if not hasattr(args, "suppress_token_ids"):
        args.suppress_token_ids = ""
    if not hasattr(args, "suppress_special_tokens_on_ignore_eos"):
        args.suppress_special_tokens_on_ignore_eos = True
    if not hasattr(args, "enable_prefill_cache"):
        args.enable_prefill_cache = False
    if not hasattr(args, "prefill_cache_max_entries"):
        args.prefill_cache_max_entries = None
    if not hasattr(args, "prefill_cache_max_tokens"):
        args.prefill_cache_max_tokens = 0
    if not hasattr(args, "prefill_cache_page_size"):
        args.prefill_cache_page_size = None
    if not hasattr(args, "prefill_cache_min_prefix_tokens"):
        args.prefill_cache_min_prefix_tokens = None
    if not hasattr(args, "prefill_cache_max_decode_extend_tokens"):
        args.prefill_cache_max_decode_extend_tokens = None
    if not hasattr(args, "decode_cuda_graph_batch_buckets"):
        args.decode_cuda_graph_batch_buckets = (1, 2, 4, 8)
    if not hasattr(args, "executor_sync_timing"):
        args.executor_sync_timing = False
    if not hasattr(args, "requests"):
        args.requests = max(args.max_batch_size, 2)
    else:
        args.requests = max(args.requests, args.max_batch_size, 2)
    if isinstance(args.decode_cuda_graph_batch_buckets, str):
        args.decode_cuda_graph_batch_buckets = parse_unique_int_list(
            args.decode_cuda_graph_batch_buckets
        ) or (1, 2, 4, 8)


def _make_http_validation_policy(args) -> GRHTTPValidationPolicy:
    return GRHTTPValidationPolicy(
        max_request_bytes=args.max_http_request_bytes,
        max_context_len=args.max_http_context_len or args.context_len,
        max_decode_steps=args.max_http_decode_steps or args.decode_steps,
        max_beam_width=args.max_http_beam_width or args.beam_width,
        max_submit_many=args.max_http_submit_many,
        max_waiting_requests=args.max_http_waiting_requests,
        max_timeout_ticks=args.max_http_timeout_ticks,
        allow_manual_tick=args.allow_manual_tick,
        allow_catalog_reload=args.allow_catalog_reload,
    )


def _make_build_info(
    args,
    torch,
    config,
    device: str,
    beam_kv_pool,
    context_kv_pool,
) -> dict[str, Any]:
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    info = {
        "framework": "gr-inference",
        "entrypoint": "tools/serve_qwen3_gr_http.py",
        "model": getattr(args, "model", None),
        "model_dir": getattr(args, "model_dir", None),
        "revision": getattr(args, "revision", None),
        "model_name": getattr(config, "model_name", None),
        "device": device,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_device_count": int(cuda.device_count()) if cuda_available else 0,
        "decode_backend": args.decode_backend,
        "prefill_backend": "torch_sdpa",
        "beam_width": args.beam_width,
        "decode_steps": args.decode_steps,
        "context_len": args.context_len,
        "max_batch_size": args.max_batch_size,
        "beam_kv_pool_enabled": beam_kv_pool is not None,
        "beam_kv_pool_capacity": args.beam_kv_pool_capacity,
        "context_kv_pool_enabled": context_kv_pool is not None,
        "context_kv_pool_capacity": args.context_kv_pool_capacity,
        "warmup_online_shapes": args.warmup_online_shapes,
        "warmup_online_pool_windows": args.warmup_online_pool_windows,
        "warmup_online_max_cases": args.warmup_online_max_cases,
        "freeze_cuda_graphs_after_warmup": args.freeze_cuda_graphs_after_warmup,
        "enable_prefill_cache": args.enable_prefill_cache,
        "prefill_cache_max_entries": args.prefill_cache_max_entries,
        "prefill_cache_max_tokens": args.prefill_cache_max_tokens,
        "prefill_cache_page_size": args.prefill_cache_page_size,
        "decode_cuda_graph_batch_buckets": tuple(args.decode_cuda_graph_batch_buckets),
        "catalog_configured": args.catalog_jsonl is not None,
        "background_worker_enabled": not args.disable_background_worker,
        "profile_continuous_decode": args.profile_continuous_decode,
        "executor_sync_timing": args.executor_sync_timing,
        "enable_log_requests": args.enable_log_requests,
        "decode_log_interval": args.decode_log_interval,
    }
    if cuda_available:
        current_device = int(cuda.current_device())
        info.update(
            {
                "cuda_current_device": current_device,
                "cuda_device_name": cuda.get_device_name(current_device),
                "cuda_device_capability": tuple(cuda.get_device_capability(current_device)),
                "cuda_memory_allocated_bytes": int(cuda.memory_allocated(current_device)),
                "cuda_memory_reserved_bytes": int(cuda.memory_reserved(current_device)),
                "cuda_max_memory_reserved_bytes": int(cuda.max_memory_reserved(current_device)),
            }
        )
    return info


def _ignore_eos_suppress_token_ids(args) -> tuple[int, ...]:
    cached = getattr(args, "_ignore_eos_suppress_token_ids_cache", None)
    if cached is not None:
        return cached
    explicit = parse_unique_int_list(args.suppress_token_ids)
    if explicit:
        args._ignore_eos_suppress_token_ids_cache = explicit
        return explicit
    if not args.suppress_special_tokens_on_ignore_eos:
        args._ignore_eos_suppress_token_ids_cache = ()
        return ()
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        token_ids = tuple(dict.fromkeys(int(token) for token in tokenizer.all_special_ids))
        args._ignore_eos_suppress_token_ids_cache = token_ids
        return token_ids
    except Exception:
        args._ignore_eos_suppress_token_ids_cache = ()
        return ()


def _startup_config_lines(adapter: GRHTTPServingAdapter) -> tuple[str, ...]:
    config = adapter._config_payload()
    validation = config["validation_policy"]
    scheduler_policy = config.get("scheduler_policy") or {}
    build = config.get("build") or {}
    return (
        "Production limits: "
        f"max_waiting_requests={validation.get('max_waiting_requests')} "
        f"max_timeout_ticks={validation.get('max_timeout_ticks')} "
        f"max_finished_requests={scheduler_policy.get('max_finished_requests')}",
        f"Auth: enabled={config.get('auth', {}).get('enabled', False)}",
        "Runtime: "
        f"model={build.get('model_name') or build.get('model_dir')} "
        f"device={build.get('device')} "
        f"decode_backend={build.get('decode_backend')} "
        f"torch={build.get('torch_version')}",
        "HTTP limits: "
        f"max_request_bytes={validation.get('max_request_bytes')} "
        f"max_context_len={validation.get('max_context_len')} "
        f"max_decode_steps={validation.get('max_decode_steps')} "
        f"max_beam_width={validation.get('max_beam_width')} "
        f"max_submit_many={validation.get('max_submit_many')}",
    )


def build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--return-beam-details", action="store_true")
    parser.add_argument("--beam-score-mode", choices=["raw_logits", "logprob"], default="logprob")
    parser.add_argument("--profile-continuous-decode", action="store_true")
    parser.add_argument(
        "--executor-sync-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize CUDA around executor prefill/decode timers. Disabled by "
            "default to avoid per-microbatch latency bubbles in serving."
        ),
    )
    parser.add_argument(
        "--profile-detail",
        choices=["coarse", "fine"],
        default="coarse",
        help="coarse avoids extra syncs; fine reports per-operator sub-stages",
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
        help="Reuse exact prompt prefill results in HTTP continuous serving",
    )
    parser.add_argument(
        "--prefill-cache-max-entries",
        type=int,
        help="Maximum cached prompt entries. Defaults to executor env/default.",
    )
    parser.add_argument(
        "--prefill-cache-max-tokens",
        type=int,
        default=0,
        help="Maximum cached prompt tokens across entries. 0 keeps it unbounded.",
    )
    parser.add_argument(
        "--prefill-cache-page-size",
        type=int,
        help="Page-align partial-prefix cache matches to this token multiple.",
    )
    parser.add_argument(
        "--prefill-cache-min-prefix-tokens",
        type=int,
        help="Minimum shared prefix length before partial-prefix suffix extend is considered.",
    )
    parser.add_argument(
        "--prefill-cache-max-decode-extend-tokens",
        type=int,
        help="Maximum suffix length for partial-prefix extend. Default executor value is 0.",
    )
    parser.add_argument(
        "--decode-cuda-graph-batch-buckets",
        default="1,2,4,8",
        help="Comma/space separated decode CUDA graph batch buckets, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--suppress-token-ids",
        default="",
        help=(
            "Optional comma/space separated token IDs to suppress when /generate "
            "sets ignore_eos=true. Defaults to tokenizer all_special_ids."
        ),
    )
    parser.add_argument(
        "--suppress-special-tokens-on-ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress tokenizer special tokens automatically when /generate sets ignore_eos=true",
    )
    parser.add_argument("--catalog-jsonl")
    parser.add_argument("--catalog-vocab-size", type=int)
    parser.add_argument("--catalog-eos-token-id", type=int)
    parser.add_argument("--catalog-item-id-field", default="item_id")
    parser.add_argument("--catalog-token-ids-field", default="token_ids")
    parser.add_argument("--catalog-metadata-field", default="metadata")
    parser.add_argument(
        "--catalog-allow-eos-for-terminal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--catalog-allow-duplicate-item-ids", action="store_true")
    parser.add_argument("--catalog-allow-duplicate-token-paths", action="store_true")
    parser.add_argument("--disable-background-worker", action="store_true")
    parser.add_argument(
        "--warmup-online-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm batch sizes 1..max_batch_size before accepting HTTP traffic",
    )
    parser.add_argument(
        "--warmup-online-pool-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm pool slot windows as well as batch sizes before accepting HTTP traffic",
    )
    parser.add_argument(
        "--warmup-online-max-cases",
        type=int,
        default=64,
        help="Maximum startup online warmup cases; <=0 means no limit",
    )
    parser.add_argument(
        "--freeze-cuda-graphs-after-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After startup online warmup, replay known CUDA graphs but do not capture new ones",
    )
    parser.add_argument("--worker-tick-interval-s", type=float, default=0.001)
    parser.add_argument("--worker-idle-sleep-s", type=float, default=0.005)
    parser.add_argument("--decode-log-interval", type=int, default=0)
    parser.add_argument("--max-http-request-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-http-context-len", type=int)
    parser.add_argument("--max-http-decode-steps", type=int)
    parser.add_argument("--max-http-beam-width", type=int)
    parser.add_argument("--max-http-submit-many", type=int, default=32)
    parser.add_argument(
        "--max-http-waiting-requests",
        type=int,
        help="Reject new HTTP submissions once the waiting-prefill queue exceeds this cap",
    )
    parser.add_argument(
        "--max-http-timeout-ticks",
        type=int,
        help="Maximum client-provided timeout_ticks accepted by HTTP submit routes",
    )
    parser.add_argument(
        "--max-finished-requests",
        type=int,
        help="Retain at most this many finished responses for result polling",
    )
    parser.add_argument("--allow-manual-tick", action="store_true")
    parser.add_argument("--allow-catalog-reload", action="store_true")
    parser.add_argument(
        "--api-key",
        help="Optional API key. When set, non-probe routes require Bearer or X-GR-API-Key auth.",
    )
    parser.add_argument("--enable-log-requests", action="store_true")
    parser.add_argument(
        "--log-requests-level",
        choices=["summary"],
        default="summary",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = build_http_serving_adapter(args)
    server = adapter.serve(host=args.host, port=args.port)
    print(f"Qwen3 GR HTTP serving on http://{args.host}:{args.port}")
    print(
        "Routes: /health /ready /config /status /metrics /generate /submit "
        "/poll/{id} /result/{id} /drain /shutdown"
    )
    for line in _startup_config_lines(adapter):
        print(line)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if hasattr(adapter.facade, "stop"):
            adapter.facade.stop()
        server.server_close()


if __name__ == "__main__":
    main()
