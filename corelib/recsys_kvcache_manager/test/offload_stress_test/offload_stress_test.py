
import argparse
import csv
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

# Hook / NVTX / timing dict keys — one entry per wrapped call site.
LAYER_METRICS: Tuple[str, ...] = (
    "kvcache_manager.offload_try_wait",
    "host_kvstorage_manager.offload_kvcache_wait",
    "host_kvstorage_manager.finish_task",
    "flexkv_client.try_wait",
    "flexkv_client.wait",
)

_PROFILER_NUM_LAYERS = 3
_PROFILER_NUM_KV_HEADS = 4
_PROFILER_HEAD_SIZE = 128
_PROFILER_TOKENS_PER_BLOCK = 32

_FIXED_HOST_CAPACITY_SCALE = 8.0
# Start stage debug logs near the known burst hang point.
_DEFAULT_DEBUG_LAUNCH_STAGE_FROM = 55


# Same sizing as FlexKV ``convert_to_block_num(cpu_cache_gb, block_size)``.
def flexkv_cpu_blocks_from_cache_gb(
    cpu_cache_gb: float,
    *,
    num_layers: int = _PROFILER_NUM_LAYERS,
    num_kv_heads: int = _PROFILER_NUM_KV_HEADS,
    head_size: int = _PROFILER_HEAD_SIZE,
    tokens_per_block: int = _PROFILER_TOKENS_PER_BLOCK,
) -> int:
    if cpu_cache_gb <= 0:
        raise ValueError("cpu_cache_gb must be positive")
    kv_dim = 2
    token_size_bytes = (
        num_layers * num_kv_heads * head_size * kv_dim * 2
    )  # torch.bfloat16
    block_size_bytes = token_size_bytes * tokens_per_block
    return int(cpu_cache_gb * (1024**3) / block_size_bytes)

_current_timing_lists: ContextVar[Optional[Dict[str, List[float]]]] = ContextVar(
    "current_timing_lists", default=None
)
_current_stage_marker: ContextVar[Optional[Callable[[str, Optional[float]], float]]] = (
    ContextVar("current_stage_marker", default=None)
)


# Convert a metric name into a CSV-safe column prefix.
def _metric_csv_prefix(metric: str) -> str:
    return metric.replace(".", "_")


# Format per-call millisecond values for console and CSV output.
def _format_ms_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


# Return call count, total latency, and average latency for one metric.
def layer_timing_stats(each_ms: Sequence[float]) -> Tuple[int, float, float]:
    calls = len(each_ms)
    total_ms = float(sum(each_ms))
    avg_ms = total_ms / calls if calls else 0.0
    return calls, total_ms, avg_ms


# Build one human-readable timing summary line for a metric.
def format_layer_timing_line(metric: str, each_ms: Sequence[float]) -> str:
    calls, total_ms, avg_ms = layer_timing_stats(each_ms)
    return (
        f"  {metric}: calls={calls}, total_ms={total_ms:.3f}, "
        f"avg_ms={avg_ms:.3f}, each_ms={_format_ms_list(each_ms)}"
    )

# Store one measured burst repeat and its hook timing lists.
@dataclass
class RepeatTimingRow:
    launch_count: int
    iteration: int
    batch_size: int
    len_per_seq: int
    total_burst_once_wait_ms: float
    each_ms_by_layer: Dict[str, List[float]] = field(default_factory=dict)

    # Return a copy of the per-call timings for one metric.
    def each_ms(self, metric: str) -> List[float]:
        return list(self.each_ms_by_layer.get(metric, []))

    # Return how many times one metric was observed.
    def calls(self, metric: str) -> int:
        return len(self.each_ms(metric))

    # Return total elapsed milliseconds for one metric.
    def total_ms(self, metric: str) -> float:
        return float(sum(self.each_ms(metric)))

    # Return average elapsed milliseconds for one metric.
    def avg_ms(self, metric: str) -> float:
        c = self.calls(metric)
        return self.total_ms(metric) / c if c else 0.0


# Build the per-repeat CSV header.
def origin_csv_header() -> List[str]:
    cols = [
        "launch_count",
        "iteration",
        "request_batch_size",
        "len_per_seq",
        "total_burst_once_wait_ms",
    ]
    for metric in LAYER_METRICS:
        prefix = _metric_csv_prefix(metric)
        cols.extend(
            [
                f"{prefix}_calls",
                f"{prefix}_total_ms",
                f"{prefix}_avg_ms",
                f"{prefix}_each_ms",
            ]
        )
    return cols


# Serialize one repeat timing row for the origin CSV.
def origin_csv_row(row: RepeatTimingRow) -> List:
    out: List = [
        row.launch_count,
        row.iteration,
        row.batch_size,
        row.len_per_seq,
        row.total_burst_once_wait_ms,
    ]
    for metric in LAYER_METRICS:
        each_ms = row.each_ms(metric)
        calls, total_ms, avg_ms = layer_timing_stats(each_ms)
        out.extend([calls, total_ms, avg_ms, _format_ms_list(each_ms)])
    return out


# Build the per-scenario summary CSV header.
def summarization_csv_header() -> List[str]:
    cols = [
        "launch_count",
        "request_batch_size",
        "len_per_seq",
        "num_bursts_measured",
        "sum_total_burst_once_wait_ms",
        "avg_total_burst_once_wait_ms",
    ]
    for metric in LAYER_METRICS:
        prefix = _metric_csv_prefix(metric)
        cols.extend(
            [
                f"sum_{prefix}_total_ms",
                f"sum_{prefix}_calls",
                f"avg_{prefix}_avg_ms",
            ]
        )
    cols.append("scenario_wall_ms")
    return cols


# Aggregate measured repeats into one summary CSV row.
def summarization_csv_row(
    offload_batch_count: int,
    batch_size: int,
    len_per_seq: int,
    metrics_list: Sequence[RepeatTimingRow],
    scenario_wall_ms: float,
) -> List:
    n = len(metrics_list)
    if n == 0:
        raise ValueError("cannot summarize empty metrics list")
    sum_total_burst_once_wait_ms = sum(m.total_burst_once_wait_ms for m in metrics_list)
    out: List = [
        offload_batch_count,
        batch_size,
        len_per_seq,
        n,
        sum_total_burst_once_wait_ms,
        sum_total_burst_once_wait_ms / n,
    ]
    for metric in LAYER_METRICS:
        sum_total_ms = sum(m.total_ms(metric) for m in metrics_list)
        sum_calls = sum(m.calls(metric) for m in metrics_list)
        avg_avg_ms = sum(m.avg_ms(metric) for m in metrics_list) / n
        out.extend([sum_total_ms, sum_calls, avg_avg_ms])
    out.append(scenario_wall_ms)
    return out


# Extract total burst time from a summary row for sorting.
def summary_row_sum_total_burst(summary_row: Sequence) -> float:
    return float(summary_row[4])

# Track one wrapped call with NVTX and optional timing collection.
@contextmanager
def track_flexkv_metric(name: str, print_nvtx: bool):
    start = time.perf_counter()
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.nvtx.range_pop()
        timing_lists = _current_timing_lists.get()
        if timing_lists is not None and name in timing_lists:
            timing_lists[name].append(elapsed_ms)
        if print_nvtx:
            print(f"[NVTX] {name:<52} {elapsed_ms:9.3f} ms")


# Wrap a method to record NVTX and per-call latency metrics.
def wrap_method_with_nvtx(obj, method_name: str, nvtx_name: str, print_nvtx: bool) -> None:
    if obj is None or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

    # Run the original method inside an NVTX timing range.
    @wraps(original)
    def wrapped(*args, **kwargs):
        with track_flexkv_metric(nvtx_name, print_nvtx=print_nvtx):
            return original(*args, **kwargs)

    wrapped.__nvtx_wrapped__ = True
    try:
        setattr(obj, method_name, wrapped)
    except Exception as e:  # noqa: BLE001
        print(
            f"[WARN] Failed to wrap {obj}.{method_name} "
            f"with NVTX ({nvtx_name}): {e}"
        )


# Install timing hooks for the FlexKV offload path.
def install_flexkv_offload_hooks(kvcache_mgr: KVCacheManager, print_nvtx: bool) -> None:
    flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    if flexkv_mgr is None or getattr(flexkv_mgr, "backend_name", "") != "flexkv":
        raise RuntimeError("host_kvstorage_manager must be flexkv backend")

    wrap_method_with_nvtx(
        kvcache_mgr,
        "offload_try_wait",
        "kvcache_manager.offload_try_wait",
        print_nvtx,
    )
    wrap_method_with_nvtx(
        flexkv_mgr,
        "offload_kvcache_wait",
        "host_kvstorage_manager.offload_kvcache_wait",
        print_nvtx,
    )
    wrap_method_with_nvtx(
        flexkv_mgr,
        "finish_task",
        "host_kvstorage_manager.finish_task",
        print_nvtx,
    )

    client = getattr(flexkv_mgr, "_client", None)
    wrap_method_with_nvtx(client, "try_wait", "flexkv_client.try_wait", print_nvtx)
    wrap_method_with_nvtx(client, "wait", "flexkv_client.wait", print_nvtx)


# Normalize index metadata fields used by the offload API.
def normalize_index_meta(index_meta) -> None:
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]


# Parse a comma-separated positive integer list.
def parse_int_list(value: str) -> List[int]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("launch counts cannot be empty")
    counts = [int(x) for x in parts]
    if any(c <= 0 for c in counts):
        raise ValueError("all launch counts must be positive")
    return counts


# Resolve origin and summary CSV output paths.
def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"offload_flexkv_bs{args.batch_size}_len{args.len_per_seq}"
    origin_path = output_root / "origin_data" / f"{run_name}.csv"
    summary_path = output_root / "summarization" / f"{run_name}.csv"
    return origin_path, summary_path


# Parse command-line options for the burst-once offload profiler.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Breakdown_2 pressure benchmark: launch x N then offload_try_wait loop "
            "until pending clears. Per-layer hook timings: "
            "kvcache_manager.offload_try_wait, "
            "host_kvstorage_manager.offload_kvcache_wait/finish_task, "
            "flexkv_client.try_wait/wait."
        )
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--len-per-seq", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Request batch size for one offload request",
    )
    parser.add_argument(
        "--offload-batch-counts",
        type=str,
        default="50,100,150,200",
        help="Comma separated launch_count (N) values per scenario",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Measured repeats per launch_count scenario",
    )
    parser.add_argument(
        "--flexkv-cpu-cache-gb",
        type=int,
        default=8,
        help=(
            "FlexKV CPU cache size in GB (aligned with benchmarks/example_config.yml). "
            "Used when --flexkv-num-cpu-blocks is 0. Set 0 to fall back to per-scenario "
            "auto blocks from launch_count."
        ),
    )
    parser.add_argument(
        "--flexkv-num-cpu-blocks",
        type=int,
        default=0,
        help="Override FlexKV CPU blocks; 0 uses --flexkv-cpu-cache-gb or auto-scaled value.",
    )
    parser.add_argument(
        "--flexkv-num-local-blocks",
        type=int,
        default=0,
        help="Override FlexKV local blocks; 0 follows CPU blocks.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=".",
        help=(
            "Root directory; writes origin_data/<run_name>.csv and "
            "summarization/<run_name>.csv"
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help=(
            "Base filename without extension. Default: "
            "offload_flexkv_bs<batch_size>_len<len_per_seq>"
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="First N repeats per scenario are warmup and not written to CSV",
    )
    parser.add_argument(
        "--print-nvtx",
        action="store_true",
        help="Print per-call [NVTX] lines to console",
    )
    parser.add_argument(
        "--launch-progress-every",
        type=int,
        default=10,
        help=(
            "During burst-once launch loop, print progress every N launches "
            "(0 disables). Helps locate hangs under large launch_count."
        ),
    )
    parser.add_argument(
        "--debug-launch-stages",
        action="store_true",
        help=(
            "Print before/after markers for lookup, allocate, GPU put, and "
            "offload_launch. Use with --debug-launch-stage-from near the hang point."
        ),
    )
    parser.add_argument(
        "--debug-launch-stage-from",
        type=int,
        default=_DEFAULT_DEBUG_LAUNCH_STAGE_FROM,
        help="First 1-based launch index that prints stage markers. Default: 55.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


# Create a FlexKV-backed KVCacheManager for one stress scenario.
def create_stress_kvcache_manager(
    max_batch_size: int,
    max_seq_len: int,
    print_nvtx: bool,
    flexkv_num_cpu_blocks: int,
    flexkv_num_local_blocks: int,
    flexkv_num_tmp_cpu_blocks: int,
) -> KVCacheManager:
    base_host_capacity = max_seq_len * max_batch_size * 32 * 4 * 128 * 2
    host_capacity_per_layer = int(base_host_capacity * _FIXED_HOST_CAPACITY_SCALE)
    kvcache_config = get_kvcache_config(
        num_layers=_PROFILER_NUM_LAYERS,
        num_heads=_PROFILER_NUM_KV_HEADS,
        head_dim=_PROFILER_HEAD_SIZE,
        page_size=_PROFILER_TOKENS_PER_BLOCK,
        offload_chunksize=128,
        num_primary_cache_pages=512,
        num_buffer_pages=0,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs={
            "flexkv_mode": "direct",
            "flexkv_host_kvstorage_fail_policy": "fail_open",
            "flexkv_enable_mps": 0,
            "flexkv_num_cpu_blocks": int(flexkv_num_cpu_blocks),
            "flexkv_num_local_blocks": int(flexkv_num_local_blocks),
            "flexkv_num_tmp_cpu_blocks": int(flexkv_num_tmp_cpu_blocks),
        },
    )
    gpu_gib = (
        kvcache_config.num_layers
        * kvcache_config.num_primary_cache_pages
        * kvcache_config.page_size
        * 2
        * kvcache_config.num_heads
        * kvcache_config.head_dim
        * 2
    ) / (1024.0**3)
    host_gib = (
        kvcache_config.num_layers * kvcache_config.host_capacity_per_layer
    ) / (1024.0**3)
    print(f"[DEBUG] KVCache GPU Memory Usage: {gpu_gib:.3f} GiB")
    print(f"[DEBUG] KVCache Host Memory Usage: {host_gib:.3f} GiB")
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    install_flexkv_offload_hooks(
        kvcache_mgr,
        print_nvtx=print_nvtx,
    )
    return kvcache_mgr


# Build one uniform request batch from preallocated KV tensors.
def build_uniform_batch_for_user_range(
    all_keys,
    all_values,
    user_start: int,
    batch_size: int,
    len_per_seq: int,
):
    user_ids = torch.arange(user_start, user_start + batch_size, dtype=torch.int64)
    sequence_lengths = torch.full((batch_size,), len_per_seq, dtype=torch.int32)
    keys = [all_keys[i][:, :len_per_seq, ...] for i in range(batch_size)]
    values = [all_values[i][:, :len_per_seq, ...] for i in range(batch_size)]
    return user_ids, sequence_lengths, keys, values

# Store launch and wait results for one burst-once run.
@dataclass
class BurstRunResult:
    launch_succeeded: int
    launch_rejected: int
    pending_before_try_wait: int
    pending_after_try_wait: int
    wait_rounds: int
    total_burst_once_wait_ms: float
    each_ms_by_layer: Dict[str, List[float]]


# Launch all offloads first, then drain them with offload_try_wait.
def run_one_burst_once_wait(
    kvcache_mgr: KVCacheManager,
    launch_count: int,
    repeat_idx: int,
    batch_size: int,
    len_per_seq: int,
    all_keys,
    all_values,
    launch_progress_every: int = 10,
    debug_launch_stages: bool = False,
    debug_launch_stage_from: int = _DEFAULT_DEBUG_LAUNCH_STAGE_FROM,
) -> BurstRunResult:
    timing_lists: Dict[str, List[float]] = {name: [] for name in LAYER_METRICS}
    timing_token = _current_timing_lists.set(timing_lists)
    launch_succeeded = 0
    launch_rejected = 0
    wait_rounds = 0
    uid_base = repeat_idx * launch_count * batch_size
    try:
        total_t0 = time.perf_counter()
        for i in range(launch_count):
            launch_idx = i + 1
            should_debug_stage = (
                debug_launch_stages and launch_idx >= debug_launch_stage_from
            )

            # Print a launch-stage marker when stage debugging is enabled.
            def stage_marker(stage: str, stage_t0: Optional[float] = None) -> float:
                now = time.perf_counter()
                if should_debug_stage:
                    suffix = ""
                    if stage_t0 is not None:
                        suffix = f", elapsed_ms={(now - stage_t0) * 1000.0:.3f}"
                    print(
                        f"[STAGE] launch {launch_idx}/{launch_count} {stage}"
                        f"{suffix}, ongoing={len(kvcache_mgr.ongoing_offload_tasks)}, "
                        f"succeeded={launch_succeeded}, rejected={launch_rejected}",
                        flush=True,
                    )
                return now

            if launch_progress_every > 0 and (
                launch_idx == 1
                or launch_idx == launch_count
                or launch_idx % launch_progress_every == 0
            ):
                print(
                    f"[PROGRESS] launch {launch_idx}/{launch_count}, "
                    f"ongoing_offload_tasks={len(kvcache_mgr.ongoing_offload_tasks)}",
                    flush=True,
                )
            stage_t0 = stage_marker("before_build_batch")
            user_ids, sequence_lengths, keys, values = build_uniform_batch_for_user_range(
                all_keys=all_keys,
                all_values=all_values,
                user_start=uid_base + i * batch_size,
                batch_size=batch_size,
                len_per_seq=len_per_seq,
            )
            stage_t0 = stage_marker("after_build_batch", stage_t0)
            stage_t0 = stage_marker("before_lookup")
            index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
            stage_t0 = stage_marker("after_lookup", stage_t0)
            stage_t0 = stage_marker("before_normalize")
            normalize_index_meta(index_meta)
            stage_t0 = stage_marker("after_normalize", stage_t0)
            stage_t0 = stage_marker("before_allocate")
            kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
            stage_t0 = stage_marker("after_allocate", stage_t0)
            stage_t0 = stage_marker("before_gpu_put")
            for layer_idx in range(3):
                layer_t0 = stage_marker(f"before_gpu_put_layer{layer_idx}")
                kvcache_mgr.gpu_kvcache_mgr.put(
                    torch.cat([k[layer_idx] for k in keys], dim=0),
                    torch.cat([v[layer_idx] for v in values], dim=0),
                    layer_idx,
                    kvcache_metadata,
                )
                stage_marker(f"after_gpu_put_layer{layer_idx}", layer_t0)
            stage_t0 = stage_marker("after_gpu_put", stage_t0)
            stage_t0 = stage_marker("before_offload_launch")
            stage_token = (
                _current_stage_marker.set(stage_marker)
                if should_debug_stage
                else None
            )
            try:
                task_handle = kvcache_mgr.offload_launch(
                    index_meta=index_meta,
                    kvcache_metadata=kvcache_metadata,
                )
            finally:
                if stage_token is not None:
                    _current_stage_marker.reset(stage_token)
            stage_marker("after_offload_launch", stage_t0)
            if bool(task_handle is not None and task_handle.handle is not None):
                launch_succeeded += 1
            else:
                launch_rejected += 1
            stage_marker("after_record_launch_result")

        pending_before_try_wait = len(kvcache_mgr.ongoing_offload_tasks)
        print(
            f"[PROGRESS] burst-once: {launch_count} launches done, "
            f"pending={pending_before_try_wait}, starting offload_try_wait",
            flush=True,
        )
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            wait_rounds += 1
            kvcache_mgr.offload_try_wait()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            time.sleep(1)
        pending_after_try_wait = len(kvcache_mgr.ongoing_offload_tasks)
        total_burst_once_wait_ms = (time.perf_counter() - total_t0) * 1000.0
    finally:
        _current_timing_lists.reset(timing_token)

    return BurstRunResult(
        launch_succeeded=launch_succeeded,
        launch_rejected=launch_rejected,
        pending_before_try_wait=pending_before_try_wait,
        pending_after_try_wait=pending_after_try_wait,
        wait_rounds=wait_rounds,
        total_burst_once_wait_ms=total_burst_once_wait_ms,
        each_ms_by_layer={name: list(timing_lists[name]) for name in LAYER_METRICS},
    )


# Print one burst result and its per-layer timing breakdown.
def print_burst_timing_report(tag: str, burst: BurstRunResult, launch_count: int, iteration: int, repeat: int) -> None:
    print(
        f"[{tag}] launch_count={launch_count:>4}, repeat={iteration:>4}/{repeat:<4} | "
        f"launch_succeeded={burst.launch_succeeded}/{launch_count}, "
        f"launch_rejected={burst.launch_rejected}, "
        f"pending={burst.pending_before_try_wait}->{burst.pending_after_try_wait}, "
        f"wait_rounds={burst.wait_rounds}, "
        f"total_burst_once_wait_ms={burst.total_burst_once_wait_ms:.3f}"
    )
    print("[TIMING] per-layer (calls / total_ms / avg_ms / each_ms):")
    for metric in LAYER_METRICS:
        print(format_layer_timing_line(metric, burst.each_ms_by_layer.get(metric, [])))


# Try to shut down the FlexKV client without failing the benchmark.
def best_effort_shutdown_kvcache_mgr(
    kvcache_mgr: KVCacheManager,
) -> None:
    host_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    client = getattr(host_mgr, "_client", None)
    shutdown_fn = getattr(client, "shutdown", None)
    if shutdown_fn is None:
        return

    try:
        shutdown_fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] client.shutdown failed: {exc}")


# Run one launch_count scenario and return measured repeat rows.
def run_scenario(
    offload_batch_count: int,
    args: argparse.Namespace,
    csv_writer: Optional[csv.writer],
) -> Tuple[List[RepeatTimingRow], float]:
    print("\n" + "=" * 88)
    print(
        f"[SCENARIO] launch_count={offload_batch_count}, "
        f"request_batch_size={args.batch_size}, len_per_seq={args.len_per_seq}, "
        f"repeat={args.repeat}, warmup_repeats={args.warmup_iterations}"
    )
    print("=" * 88)

    blocks_per_seq = (args.len_per_seq + 31) // 32
    estimated_blocks = offload_batch_count * args.batch_size * blocks_per_seq
    estimated_tmp_blocks = estimated_blocks * 2
    auto_cpu_blocks = max(4096, int(estimated_blocks * 1.5))
    auto_local_blocks = auto_cpu_blocks
    auto_tmp_cpu_blocks = max(256, estimated_tmp_blocks)

    if args.flexkv_num_cpu_blocks > 0:
        flexkv_num_cpu_blocks = args.flexkv_num_cpu_blocks
    elif args.flexkv_cpu_cache_gb > 0:
        flexkv_num_cpu_blocks = flexkv_cpu_blocks_from_cache_gb(
            float(args.flexkv_cpu_cache_gb)
        )
    else:
        flexkv_num_cpu_blocks = auto_cpu_blocks
    flexkv_num_local_blocks = (
        args.flexkv_num_local_blocks
        if args.flexkv_num_local_blocks > 0
        else auto_local_blocks
    )
    flexkv_num_tmp_cpu_blocks = auto_tmp_cpu_blocks

    print(
        f"[SCENARIO][FLEXKV_BLOCKS] estimated={estimated_blocks}, "
        f"estimated_tmp={estimated_tmp_blocks}, "
        f"cpu_cache_gb={args.flexkv_cpu_cache_gb}, "
        f"cpu={flexkv_num_cpu_blocks}, local={flexkv_num_local_blocks}, "
        f"tmp_cpu={flexkv_num_tmp_cpu_blocks}"
    )

    kvcache_mgr = create_stress_kvcache_manager(
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        print_nvtx=args.print_nvtx,
        flexkv_num_cpu_blocks=flexkv_num_cpu_blocks,
        flexkv_num_local_blocks=flexkv_num_local_blocks,
        flexkv_num_tmp_cpu_blocks=flexkv_num_tmp_cpu_blocks,
    )
    all_keys = [
        torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
        for _ in range(args.batch_size)
    ]
    all_values = [
        torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
        for _ in range(args.batch_size)
    ]

    measured: List[RepeatTimingRow] = []
    scenario_t0 = time.perf_counter()
    for i in range(args.repeat):
        iteration = i + 1
        is_warmup = iteration <= args.warmup_iterations

        burst = run_one_burst_once_wait(
            kvcache_mgr=kvcache_mgr,
            launch_count=offload_batch_count,
            repeat_idx=i,
            batch_size=args.batch_size,
            len_per_seq=args.len_per_seq,
            all_keys=all_keys,
            all_values=all_values,
            launch_progress_every=args.launch_progress_every,
            debug_launch_stages=args.debug_launch_stages,
            debug_launch_stage_from=args.debug_launch_stage_from,
        )

        tag = "WARMUP" if is_warmup else "BURST"
        print_burst_timing_report(
            tag=tag,
            burst=burst,
            launch_count=offload_batch_count,
            iteration=iteration,
            repeat=args.repeat,
        )

        if is_warmup:
            continue

        row_data = RepeatTimingRow(
            launch_count=offload_batch_count,
            iteration=iteration,
            batch_size=args.batch_size,
            len_per_seq=args.len_per_seq,
            total_burst_once_wait_ms=burst.total_burst_once_wait_ms,
            each_ms_by_layer=burst.each_ms_by_layer,
        )
        measured.append(row_data)
        if csv_writer is not None:
            csv_writer.writerow(origin_csv_row(row_data))

    scenario_wall_ms = (time.perf_counter() - scenario_t0) * 1000.0
    best_effort_shutdown_kvcache_mgr(kvcache_mgr=kvcache_mgr)
    return measured, scenario_wall_ms


# Run all configured burst-once scenarios and write CSV outputs.
def main() -> None:
    args = parse_args()

    offload_batch_counts = parse_int_list(args.offload_batch_counts)
    torch.manual_seed(args.seed)

    origin_path, summary_path = resolve_output_paths(args)
    origin_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[INFO] launch_counts={offload_batch_counts}, batch_size={args.batch_size}, "
        f"len_per_seq={args.len_per_seq}, repeat={args.repeat}, "
        f"warmup={args.warmup_iterations}, "
    )
    print(f"[INFO] origin csv: {origin_path}")
    print(f"[INFO] summary csv: {summary_path}")
    print(f"[INFO] layer hooks: {', '.join(LAYER_METRICS)}")
    print(
        f"[INFO] flexkv_cpu_cache_gb={args.flexkv_cpu_cache_gb}, "
        f"launch_progress_every={args.launch_progress_every}, "
        f"debug_launch_stages={args.debug_launch_stages}, "
        f"debug_launch_stage_from={args.debug_launch_stage_from}"
    )

    summary_rows: List[List] = []

    # Run one scenario and append its summary row.
    def process_scenario(offload_batch_count: int, csv_writer) -> None:
        measured, scenario_wall_ms = run_scenario(
            offload_batch_count=offload_batch_count,
            args=args,
            csv_writer=csv_writer,
        )
        if not measured:
            print(
                f"[WARN] launch_count={offload_batch_count}: no measured repeats "
                "(all warmup?)"
            )
            return
        summary_rows.append(
            summarization_csv_row(
                offload_batch_count=offload_batch_count,
                batch_size=args.batch_size,
                len_per_seq=args.len_per_seq,
                metrics_list=measured,
                scenario_wall_ms=scenario_wall_ms,
            )
        )
        sum_total_burst = summary_row_sum_total_burst(summary_rows[-1])
        print(
            f"[SUMMARY] launch_count={offload_batch_count}, "
            f"measured {len(measured)} bursts, "
            f"sum_total_burst_once_wait={sum_total_burst:.1f} ms, "
            f"scenario_wall={scenario_wall_ms:.1f} ms"
        )

    with origin_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(origin_csv_header())
        for offload_batch_count in offload_batch_counts:
            process_scenario(offload_batch_count, csv_writer=writer)

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summarization_csv_header())
        writer.writerows(summary_rows)

    print(f"[DONE] origin_data: {origin_path}")
    print(f"[DONE] summarization: {summary_path}")


if __name__ == "__main__":
    main()
