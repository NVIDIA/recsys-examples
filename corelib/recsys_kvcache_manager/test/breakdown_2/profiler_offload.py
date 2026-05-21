

import argparse
import csv
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

_current_timing_lists: ContextVar[Optional[Dict[str, List[float]]]] = ContextVar(
    "current_timing_lists", default=None
)
_TIMING_LIST_METRICS: Tuple[str, ...] = (
    "flexkv.offload_kvcache_wait",
    "flexkv.client.try_wait",
    "flexkv.client.wait",
)


class RepeatTimingRow:
    def __init__(
        self,
        launch_count: int,
        iteration: int,
        batch_size: int,
        len_per_seq: int,
        total_burst_once_wait_ms: float,
        offload_try_wait_each_ms: List[float],
        client_try_wait_each_ms: List[float],
        client_wait_each_ms: List[float],
    ) -> None:
        self.launch_count = launch_count
        self.iteration = iteration
        self.batch_size = batch_size
        self.len_per_seq = len_per_seq
        self.total_burst_once_wait_ms = total_burst_once_wait_ms
        self.offload_try_wait_each_ms = offload_try_wait_each_ms
        self.client_try_wait_each_ms = client_try_wait_each_ms
        self.client_wait_each_ms = client_wait_each_ms

    @staticmethod
    def _sum(values: Sequence[float]) -> float:
        return float(sum(values))

    @property
    def offload_try_wait_total_ms(self) -> float:
        return self._sum(self.offload_try_wait_each_ms)

    @property
    def client_try_wait_total_ms(self) -> float:
        return self._sum(self.client_try_wait_each_ms)

    @property
    def client_wait_total_ms(self) -> float:
        return self._sum(self.client_wait_each_ms)

    @property
    def offload_try_wait_calls(self) -> int:
        return len(self.offload_try_wait_each_ms)

    @property
    def client_try_wait_calls(self) -> int:
        return len(self.client_try_wait_each_ms)

    @property
    def client_wait_calls(self) -> int:
        return len(self.client_wait_each_ms)


def _format_ms_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


def origin_csv_header() -> List[str]:
    return [
        "launch_count",
        "iteration",
        "request_batch_size",
        "len_per_seq",
        "total_burst_once_wait_ms",
        "offload_try_wait_total_ms",
        "offload_try_wait_calls",
        "offload_try_wait_each_ms",
        "client_try_wait_total_ms",
        "client_try_wait_calls",
        "client_try_wait_each_ms",
        "client_wait_total_ms",
        "client_wait_calls",
        "client_wait_each_ms",
    ]


def origin_csv_row(row: RepeatTimingRow) -> List:
    return [
        row.launch_count,
        row.iteration,
        row.batch_size,
        row.len_per_seq,
        row.total_burst_once_wait_ms,
        row.offload_try_wait_total_ms,
        row.offload_try_wait_calls,
        _format_ms_list(row.offload_try_wait_each_ms),
        row.client_try_wait_total_ms,
        row.client_try_wait_calls,
        _format_ms_list(row.client_try_wait_each_ms),
        row.client_wait_total_ms,
        row.client_wait_calls,
        _format_ms_list(row.client_wait_each_ms),
    ]


def summarization_csv_header() -> List[str]:
    return [
        "launch_count",
        "request_batch_size",
        "len_per_seq",
        "num_bursts_measured",
        "sum_total_burst_once_wait_ms",
        "avg_total_burst_once_wait_ms",
        "sum_offload_try_wait_total_ms",
        "sum_offload_try_wait_calls",
        "sum_client_try_wait_total_ms",
        "sum_client_try_wait_calls",
        "sum_client_wait_total_ms",
        "sum_client_wait_calls",
        "scenario_wall_ms",
    ]


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
    sum_offload_try_wait_total_ms = sum(m.offload_try_wait_total_ms for m in metrics_list)
    sum_offload_try_wait_calls = sum(m.offload_try_wait_calls for m in metrics_list)
    sum_client_try_wait_total_ms = sum(m.client_try_wait_total_ms for m in metrics_list)
    sum_client_try_wait_calls = sum(m.client_try_wait_calls for m in metrics_list)
    sum_client_wait_total_ms = sum(m.client_wait_total_ms for m in metrics_list)
    sum_client_wait_calls = sum(m.client_wait_calls for m in metrics_list)
    return [
        offload_batch_count,
        batch_size,
        len_per_seq,
        n,
        sum_total_burst_once_wait_ms,
        sum_total_burst_once_wait_ms / n,
        sum_offload_try_wait_total_ms,
        sum_offload_try_wait_calls,
        sum_client_try_wait_total_ms,
        sum_client_try_wait_calls,
        sum_client_wait_total_ms,
        sum_client_wait_calls,
        scenario_wall_ms,
    ]


def summary_row_sum_total_burst(summary_row: Sequence) -> float:
    return float(summary_row[4])


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
        raise ValueError("launch counts cannot be empty")
    counts = [int(x) for x in parts]
    if any(c <= 0 for c in counts):
        raise ValueError("all launch counts must be positive")
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
            "Breakdown_2 pressure benchmark: launch x N then one offload_try_wait "
            "per measured repeat."
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
        "--host-capacity-scale",
        type=float,
        default=8.0,
        help="Multiply host capacity per layer to avoid early capacity bottleneck",
    )
    parser.add_argument(
        "--flexkv-num-cpu-blocks",
        type=int,
        default=0,
        help="Override FlexKV CPU blocks; 0 uses auto-scaled value.",
    )
    parser.add_argument(
        "--flexkv-num-local-blocks",
        type=int,
        default=0,
        help="Override FlexKV local blocks; 0 follows CPU blocks.",
    )
    parser.add_argument(
        "--flexkv-num-tmp-cpu-blocks",
        type=int,
        default=0,
        help="Override FlexKV tmp CPU blocks; 0 uses auto-scaled value.",
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
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def create_stress_kvcache_manager(
    max_batch_size: int,
    max_seq_len: int,
    host_capacity_scale: float,
    print_nvtx: bool,
    flexkv_num_cpu_blocks: int,
    flexkv_num_local_blocks: int,
    flexkv_num_tmp_cpu_blocks: int,
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


def run_one_burst_once_wait(
    kvcache_mgr: KVCacheManager,
    launch_count: int,
    repeat_idx: int,
    batch_size: int,
    len_per_seq: int,
    all_keys,
    all_values,
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    float,
    List[float],
    List[float],
    List[float],
]:
    timing_lists = {name: [] for name in _TIMING_LIST_METRICS}
    timing_token = _current_timing_lists.set(timing_lists)
    launch_succeeded = 0
    launch_rejected = 0
    wait_rounds = 0
    uid_base = repeat_idx * launch_count * batch_size
    try:
        total_t0 = time.perf_counter()
        for i in range(launch_count):
            user_ids, sequence_lengths, keys, values = build_uniform_batch_for_user_range(
                all_keys=all_keys,
                all_values=all_values,
                user_start=uid_base + i * batch_size,
                batch_size=batch_size,
                len_per_seq=len_per_seq,
            )
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
            task_handle = kvcache_mgr.offload_launch(
                index_meta=index_meta,
                kvcache_metadata=kvcache_metadata,
            )
            if bool(task_handle is not None and task_handle.handle is not None):
                launch_succeeded += 1
            else:
                launch_rejected += 1

        pending_before_try_wait = len(kvcache_mgr.ongoing_offload_tasks)
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            wait_rounds += 1
            kvcache_mgr.offload_try_wait()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            time.sleep(0.1)
        pending_after_try_wait = len(kvcache_mgr.ongoing_offload_tasks)
        total_burst_once_wait_ms = (time.perf_counter() - total_t0) * 1000.0
    finally:
        _current_timing_lists.reset(timing_token)

    return (
        launch_succeeded,
        launch_rejected,
        pending_before_try_wait,
        pending_after_try_wait,
        wait_rounds,
        total_burst_once_wait_ms,
        list(timing_lists["flexkv.offload_kvcache_wait"]),
        list(timing_lists["flexkv.client.try_wait"]),
        list(timing_lists["flexkv.client.wait"]),
    )


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
    # FlexKV offload in this path materializes both K/V payloads; tmp blocks sized
    # by a single-stream estimate can under-allocate and cause NOTFOUND under burst.
    estimated_tmp_blocks = estimated_blocks * 2
    auto_cpu_blocks = max(4096, int(estimated_blocks * 1.5))
    auto_local_blocks = auto_cpu_blocks
    auto_tmp_cpu_blocks = max(256, estimated_tmp_blocks)

    flexkv_num_cpu_blocks = (
        args.flexkv_num_cpu_blocks
        if args.flexkv_num_cpu_blocks > 0
        else auto_cpu_blocks
    )
    flexkv_num_local_blocks = (
        args.flexkv_num_local_blocks
        if args.flexkv_num_local_blocks > 0
        else auto_local_blocks
    )
    flexkv_num_tmp_cpu_blocks = (
        args.flexkv_num_tmp_cpu_blocks
        if args.flexkv_num_tmp_cpu_blocks > 0
        else auto_tmp_cpu_blocks
    )

    print(
        f"[SCENARIO][FLEXKV_BLOCKS] estimated={estimated_blocks}, "
        f"estimated_tmp={estimated_tmp_blocks}, "
        f"cpu={flexkv_num_cpu_blocks}, local={flexkv_num_local_blocks}, "
        f"tmp_cpu={flexkv_num_tmp_cpu_blocks}"
    )

    kvcache_mgr = create_stress_kvcache_manager(
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        host_capacity_scale=args.host_capacity_scale,
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

        (
            launch_succeeded,
            launch_rejected,
            pending_before_try_wait,
            pending_after_try_wait,
            wait_rounds,
            total_burst_once_wait_ms,
            offload_try_wait_each_ms,
            client_try_wait_each_ms,
            client_wait_each_ms,
        ) = run_one_burst_once_wait(
            kvcache_mgr=kvcache_mgr,
            launch_count=offload_batch_count,
            repeat_idx=i,
            batch_size=args.batch_size,
            len_per_seq=args.len_per_seq,
            all_keys=all_keys,
            all_values=all_values,
        )

        tag = "WARMUP" if is_warmup else "BURST"
        print(
            f"[{tag}] launch_count={offload_batch_count:>4}, "
            f"repeat={iteration:>4}/{args.repeat:<4} | "
            f"launch_succeeded={launch_succeeded}/{offload_batch_count}, "
            f"launch_rejected={launch_rejected}, "
            f"pending={pending_before_try_wait}->{pending_after_try_wait}, "
            f"wait_rounds={wait_rounds}, "
            f"total_burst_once_wait={total_burst_once_wait_ms:.3f}ms, "
            f"offload_try_wait_calls={len(offload_try_wait_each_ms)}, "
            f"client_try_wait_calls={len(client_try_wait_each_ms)}, "
            f"client_wait_calls={len(client_wait_each_ms)}"
        )

        if is_warmup:
            continue

        row_data = RepeatTimingRow(
            launch_count=offload_batch_count,
            iteration=iteration,
            batch_size=args.batch_size,
            len_per_seq=args.len_per_seq,
            total_burst_once_wait_ms=total_burst_once_wait_ms,
            offload_try_wait_each_ms=offload_try_wait_each_ms,
            client_try_wait_each_ms=client_try_wait_each_ms,
            client_wait_each_ms=client_wait_each_ms,
        )
        measured.append(row_data)
        if csv_writer is not None:
            csv_writer.writerow(origin_csv_row(row_data))

    scenario_wall_ms = (time.perf_counter() - scenario_t0) * 1000.0
    best_effort_shutdown_kvcache_mgr(kvcache_mgr=kvcache_mgr)
    return measured, scenario_wall_ms


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

    summary_rows: List[List] = []

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
