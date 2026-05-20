"""
Repeated FlexKV offload micro-benchmark (Micro-bench 2).

Measures: lookup → allocate → gpu.put → offload_launch → offload_try_wait until
this round completes (while loop in run_one_offload; not the single-shot sub-exp).

FlexKV client.shutdown() after a scenario is omitted — it can hang; rely on process exit.

CSV schemas / aggregation: utils.py (perf_counter on hooks, not nsys export).
"""

import argparse
import csv
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import (  # noqa: E402
    FLEXKV_CSV_METRICS,
    IterationMetrics,
    origin_csv_header,
    origin_csv_row,
    summarization_csv_header,
    summarization_csv_row,
    summary_row_sum_effective,
)

from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

_current_metrics: ContextVar[Optional[IterationMetrics]] = ContextVar(
    "current_metrics", default=None
)


@contextmanager
def track_flexkv_metric(name: str, print_nvtx: bool):
    start = time.perf_counter()
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.nvtx.range_pop()
        metrics = _current_metrics.get()
        if metrics is not None and name in FLEXKV_CSV_METRICS:
            metrics.record(name, elapsed_ms)
        if print_nvtx:
            print(f"[NVTX] {name:<44} {elapsed_ms:9.3f} ms")


def wrap_method_with_nvtx(obj, method_name: str, nvtx_name: str, print_nvtx: bool) -> None:
    if obj is None or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "__nvtx_wrapped__", False):
        return

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


def install_flexkv_offload_hooks(kvcache_mgr: KVCacheManager, print_nvtx: bool) -> None:
    flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    if flexkv_mgr is None or getattr(flexkv_mgr, "backend_name", "") != "flexkv":
        raise RuntimeError("host_kvstorage_manager must be flexkv backend")

    wrap_method_with_nvtx(
        flexkv_mgr, "offload_kvcache_launch", "flexkv.offload_kvcache_launch", print_nvtx
    )
    wrap_method_with_nvtx(
        flexkv_mgr, "offload_kvcache_wait", "flexkv.offload_kvcache_wait", print_nvtx
    )
    wrap_method_with_nvtx(flexkv_mgr, "finish_task", "flexkv.finish_task", print_nvtx)
    wrap_method_with_nvtx(flexkv_mgr, "cancel_task", "flexkv.cancel_task", print_nvtx)

    client = getattr(flexkv_mgr, "_client", None)
    wrap_method_with_nvtx(client, "put_async", "flexkv.client.put_async", print_nvtx)
    wrap_method_with_nvtx(client, "try_wait", "flexkv.client.try_wait", print_nvtx)
    wrap_method_with_nvtx(client, "wait", "flexkv.client.wait", print_nvtx)


def normalize_index_meta(index_meta) -> None:
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]


def parse_int_list(value: str) -> List[int]:
    parts = [x.strip() for x in value.split(",") if x.strip()]
    if not parts:
        raise ValueError("offload batch counts cannot be empty")
    counts = [int(x) for x in parts]
    if any(c <= 0 for c in counts):
        raise ValueError("all offload batch counts must be positive")
    return counts


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"offload_flexkv_bs{args.batch_size}_len{args.len_per_seq}"
    origin_path = output_root / "origin_data" / f"{run_name}.csv"
    summary_path = output_root / "summarization" / f"{run_name}.csv"
    return origin_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously send offload requests and record FlexKV launch/wait "
            "breakdown per iteration into CSV."
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
        default="50,100,150,200,250,300",
        help="Comma separated offload iteration counts per scenario",
    )
    parser.add_argument(
        "--host-capacity-scale",
        type=float,
        default=8.0,
        help="Multiply host capacity per layer to avoid early capacity bottleneck",
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
        help="First N iterations per scenario are warmup and not written to CSV",
    )
    parser.add_argument(
        "--no-origin-data",
        action="store_true",
        help="Skip per-iteration origin_data CSV (summarization totals only)",
    )
    parser.add_argument(
        "--print-nvtx",
        action="store_true",
        help="Print per-call [NVTX] lines to console",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def create_stress_kvcache_manager(
    max_batch_size: int,
    max_seq_len: int,
    host_capacity_scale: float,
    print_nvtx: bool,
) -> KVCacheManager:
    if host_capacity_scale <= 0:
        raise ValueError("host_capacity_scale must be positive")
    base_host_capacity = max_seq_len * max_batch_size * 32 * 4 * 128 * 2
    host_capacity_per_layer = int(base_host_capacity * host_capacity_scale)
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
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
    install_flexkv_offload_hooks(kvcache_mgr, print_nvtx=print_nvtx)
    return kvcache_mgr


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


def run_one_offload(kvcache_mgr, user_ids, sequence_lengths, keys, values) -> IterationMetrics:
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    metrics = IterationMetrics()
    token = _current_metrics.set(metrics)
    try:
        task_handle = kvcache_mgr.offload_launch(
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
        if task_handle is None or task_handle.handle is None:
            raise RuntimeError("offload_launch did not return a valid handle")

        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            kvcache_mgr.offload_try_wait()
    finally:
        _current_metrics.reset(token)

    return metrics


def format_metrics_line(metrics: IterationMetrics) -> str:
    parts = [
        f"effective={metrics.flexkv_offload_effective_total_ms():.3f}ms",
        f"raw_total={metrics.flexkv_offload_total_ms():.3f}ms",
    ]
    for name in FLEXKV_CSV_METRICS:
        ms = metrics.get_sum_ms(name)
        cnt = metrics.get_count(name)
        if cnt > 0:
            short = name.replace("flexkv.", "")
            parts.append(f"{short}={ms:.3f}ms/x{cnt}")
    return ", ".join(parts)


def run_scenario(
    offload_batch_count: int,
    args: argparse.Namespace,
    csv_writer: Optional[csv.writer],
) -> Tuple[List[IterationMetrics], float]:
    print("\n" + "=" * 88)
    print(
        f"[SCENARIO] offload_batch_count={offload_batch_count}, "
        f"request_batch_size={args.batch_size}, len_per_seq={args.len_per_seq}, "
        f"warmup_iterations={args.warmup_iterations}"
    )
    print("=" * 88)

    kvcache_mgr = create_stress_kvcache_manager(
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        host_capacity_scale=args.host_capacity_scale,
        print_nvtx=args.print_nvtx,
    )
    all_keys = [
        torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
        for _ in range(args.batch_size)
    ]
    all_values = [
        torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
        for _ in range(args.batch_size)
    ]

    measured: List[IterationMetrics] = []
    scenario_t0 = time.perf_counter()
    for i in range(offload_batch_count):
        iteration = i + 1
        is_warmup = iteration <= args.warmup_iterations

        user_ids, sequence_lengths, keys, values = build_uniform_batch_for_user_range(
            all_keys=all_keys,
            all_values=all_values,
            user_start=i * args.batch_size,
            batch_size=args.batch_size,
            len_per_seq=args.len_per_seq,
        )

        metrics = run_one_offload(
            kvcache_mgr=kvcache_mgr,
            user_ids=user_ids,
            sequence_lengths=sequence_lengths,
            keys=keys,
            values=values,
        )

        tag = "WARMUP" if is_warmup else "OFFLOAD"
        print(
            f"[{tag}] scenario={offload_batch_count:>4}, "
            f"iter={iteration:>4}/{offload_batch_count:<4} | "
            f"{format_metrics_line(metrics)}"
        )

        if is_warmup:
            continue

        measured.append(metrics)
        if csv_writer is not None:
            csv_writer.writerow(
                origin_csv_row(
                    offload_batch_count=offload_batch_count,
                    iteration=iteration,
                    batch_size=args.batch_size,
                    len_per_seq=args.len_per_seq,
                    metrics=metrics,
                )
            )

    # No FlexKV client.shutdown() here — it can hang after many offloads.

    scenario_wall_ms = (time.perf_counter() - scenario_t0) * 1000.0
    return measured, scenario_wall_ms


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")
    if args.len_per_seq <= 0:
        raise ValueError("--len-per-seq must be a positive integer")
    if args.max_seq_len < args.len_per_seq:
        raise ValueError("--max-seq-len must be >= --len-per-seq")
    if args.host_capacity_scale <= 0:
        raise ValueError("--host-capacity-scale must be positive")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be >= 0")

    offload_batch_counts = parse_int_list(args.offload_batch_counts)
    torch.manual_seed(args.seed)

    origin_path, summary_path = resolve_output_paths(args)
    origin_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] origin csv: {origin_path}")
    print(f"[INFO] summary csv: {summary_path}")
    print(f"[INFO] offload_batch_counts={offload_batch_counts}")
    print(
        f"[INFO] request_batch_size={args.batch_size}, "
        f"len_per_seq={args.len_per_seq}, host_capacity_scale={args.host_capacity_scale}"
    )
    print(
        f"[INFO] warmup: first {args.warmup_iterations} iteration(s) per scenario "
        "excluded from CSV and scenario totals"
    )
    if args.no_origin_data:
        print("[INFO] --no-origin-data: skipping origin_data CSV")

    summary_rows: List[List] = []

    def process_scenario(offload_batch_count: int, csv_writer) -> None:
        measured, scenario_wall_ms = run_scenario(
            offload_batch_count=offload_batch_count,
            args=args,
            csv_writer=csv_writer,
        )
        if not measured:
            print(
                f"[WARN] scenario={offload_batch_count}: no measured iterations "
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
        sum_effective = summary_row_sum_effective(summary_rows[-1])
        print(
            f"[SUMMARY] offload_batch_count={offload_batch_count}, "
            f"measured {len(measured)} offloads, "
            f"sum_effective={sum_effective:.1f} ms, "
            f"scenario_wall={scenario_wall_ms:.1f} ms"
        )

    if args.no_origin_data:
        for offload_batch_count in offload_batch_counts:
            process_scenario(offload_batch_count, csv_writer=None)
    else:
        with origin_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(origin_csv_header())
            for offload_batch_count in offload_batch_counts:
                process_scenario(offload_batch_count, csv_writer=writer)

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summarization_csv_header())
        writer.writerows(summary_rows)

    if not args.no_origin_data:
        print(f"[DONE] origin_data: {origin_path}")
    print(f"[DONE] summarization: {summary_path}")


if __name__ == "__main__":
    main()
