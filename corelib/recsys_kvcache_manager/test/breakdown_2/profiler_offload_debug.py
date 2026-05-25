import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

_PROFILER_NUM_LAYERS = 3
_PROFILER_NUM_KV_HEADS = 4
_PROFILER_HEAD_SIZE = 128
_PROFILER_TOKENS_PER_BLOCK = 32
_DEFAULT_FLEXKV_CPU_CACHE_GB = 8


def parse_int_list(value: str) -> List[int]:
    counts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not counts:
        raise ValueError("launch counts cannot be empty")
    if any(c <= 0 for c in counts):
        raise ValueError("all launch counts must be positive")
    return counts


def _format_ms_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in values) + "]"


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
    token_size_bytes = num_layers * num_kv_heads * head_size * kv_dim * 2
    block_size_bytes = token_size_bytes * tokens_per_block
    return int(cpu_cache_gb * (1024**3) / block_size_bytes)


def status_is_success(status) -> bool:
    return (
        str(status).endswith("SUCCESS")
        or getattr(status, "name", "") == "SUCCESS"
        or getattr(status, "value", "") == "success"
    )


def count_mask_tokens(mask) -> int:
    if mask is None:
        return 0
    if isinstance(mask, list):
        return int(sum(torch.as_tensor(m).sum().item() for m in mask))
    return int(torch.as_tensor(mask).sum().item())


class RepeatTimingRow(NamedTuple):
    launch_count: int
    iteration: int
    batch_size: int
    len_per_seq: int
    submitted_tasks: int
    done_tasks: int
    pending_tasks: int
    success_tokens: int
    failed_tasks: int
    timeout_tasks: int
    total_burst_once_wait_ms: float
    launch_total_ms: float
    try_wait_total_ms: float
    put_async_each_ms: List[float]
    try_wait_each_ms: List[float]

    @property
    def put_async_total_ms(self) -> float:
        return float(sum(self.put_async_each_ms))

    @property
    def try_wait_calls(self) -> int:
        return len(self.try_wait_each_ms)


def origin_csv_header() -> List[str]:
    return [
        "launch_count",
        "iteration",
        "request_batch_size",
        "len_per_seq",
        "submitted_tasks",
        "done_tasks",
        "pending_tasks",
        "success_tokens",
        "failed_tasks",
        "timeout_tasks",
        "total_burst_once_wait_ms",
        "launch_total_ms",
        "flexkv_client_put_async_total_ms",
        "flexkv_client_put_async_calls",
        "flexkv_client_put_async_each_ms",
        "try_wait_total_ms",
        "try_wait_calls",
        "try_wait_each_ms",
    ]


def origin_csv_row(row: RepeatTimingRow) -> List:
    return [
        row.launch_count,
        row.iteration,
        row.batch_size,
        row.len_per_seq,
        row.submitted_tasks,
        row.done_tasks,
        row.pending_tasks,
        row.success_tokens,
        row.failed_tasks,
        row.timeout_tasks,
        row.total_burst_once_wait_ms,
        row.launch_total_ms,
        row.put_async_total_ms,
        len(row.put_async_each_ms),
        _format_ms_list(row.put_async_each_ms),
        row.try_wait_total_ms,
        row.try_wait_calls,
        _format_ms_list(row.try_wait_each_ms),
    ]


def summarization_csv_header() -> List[str]:
    return [
        "launch_count",
        "request_batch_size",
        "len_per_seq",
        "num_bursts_measured",
        "sum_submitted_tasks",
        "sum_done_tasks",
        "sum_pending_tasks",
        "sum_success_tokens",
        "sum_failed_tasks",
        "sum_timeout_tasks",
        "sum_total_burst_once_wait_ms",
        "sum_launch_total_ms",
        "sum_flexkv_client_put_async_total_ms",
        "sum_flexkv_client_put_async_calls",
        "sum_try_wait_total_ms",
        "sum_try_wait_calls",
        "scenario_wall_ms",
    ]


def summarization_csv_row(
    launch_count: int,
    batch_size: int,
    len_per_seq: int,
    rows: Sequence[RepeatTimingRow],
    scenario_wall_ms: float,
) -> List:
    return [
        launch_count,
        batch_size,
        len_per_seq,
        len(rows),
        sum(r.submitted_tasks for r in rows),
        sum(r.done_tasks for r in rows),
        sum(r.pending_tasks for r in rows),
        sum(r.success_tokens for r in rows),
        sum(r.failed_tasks for r in rows),
        sum(r.timeout_tasks for r in rows),
        sum(r.total_burst_once_wait_ms for r in rows),
        sum(r.launch_total_ms for r in rows),
        sum(r.put_async_total_ms for r in rows),
        sum(len(r.put_async_each_ms) for r in rows),
        sum(r.try_wait_total_ms for r in rows),
        sum(r.try_wait_calls for r in rows),
        scenario_wall_ms,
    ]


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name or f"direct_flexkv_bs{args.batch_size}_len{args.len_per_seq}"
    return (
        output_root / "origin_data" / f"{run_name}.csv",
        output_root / "summarization" / f"{run_name}.csv",
    )


def create_kvcache_manager(args: argparse.Namespace, launch_count: int) -> KVCacheManager:
    blocks_per_seq = (args.len_per_seq + args.tokens_per_block - 1) // args.tokens_per_block
    estimated_blocks = launch_count * args.batch_size * blocks_per_seq
    flexkv_num_cpu_blocks = (
        args.flexkv_num_cpu_blocks
        if args.flexkv_num_cpu_blocks > 0
        else flexkv_cpu_blocks_from_cache_gb(float(args.flexkv_cpu_cache_gb))
    )
    flexkv_num_local_blocks = (
        args.flexkv_num_local_blocks
        if args.flexkv_num_local_blocks > 0
        else flexkv_num_cpu_blocks
    )
    flexkv_num_tmp_cpu_blocks = (
        args.flexkv_num_tmp_cpu_blocks
        if args.flexkv_num_tmp_cpu_blocks > 0
        else max(256, estimated_blocks * 2)
    )
    num_primary_cache_pages = (
        args.num_primary_cache_pages
        if args.num_primary_cache_pages > 0
        else max(512, args.batch_size * blocks_per_seq)
    )
    host_capacity_per_layer = int(
        args.max_seq_len
        * args.batch_size
        * args.tokens_per_block
        * _PROFILER_NUM_KV_HEADS
        * _PROFILER_HEAD_SIZE
        * 2
    )
    kvcache_config = get_kvcache_config(
        num_layers=_PROFILER_NUM_LAYERS,
        num_heads=_PROFILER_NUM_KV_HEADS,
        head_dim=_PROFILER_HEAD_SIZE,
        page_size=args.tokens_per_block,
        offload_chunksize=128,
        num_primary_cache_pages=num_primary_cache_pages,
        num_buffer_pages=0,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
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
    print(
        f"[SCENARIO][FLEXKV_BLOCKS] estimated={estimated_blocks}, "
        f"cpu_cache_gb={args.flexkv_cpu_cache_gb}, cpu={flexkv_num_cpu_blocks}, "
        f"local={flexkv_num_local_blocks}, tmp_cpu={flexkv_num_tmp_cpu_blocks}, "
        f"gpu_pages={num_primary_cache_pages}"
    )
    return KVCacheManager.from_config(kvcache_config)


def build_direct_put_inputs(args: argparse.Namespace) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.manual_seed(args.seed)
    batch_sequences = [
        torch.randint(0, 100000, (args.len_per_seq,), dtype=torch.int64)
        for _ in range(args.batch_size)
    ]
    batch_slot_mappings = [
        torch.arange(
            i * args.len_per_seq,
            (i + 1) * args.len_per_seq,
            dtype=torch.int64,
        )
        for i in range(args.batch_size)
    ]
    return batch_sequences, batch_slot_mappings


def run_one_burst_once_wait(
    client,
    args: argparse.Namespace,
    launch_count: int,
    repeat_idx: int,
    batch_sequences: Sequence[torch.Tensor],
    batch_slot_mappings: Sequence[torch.Tensor],
) -> RepeatTimingRow:
    all_task_ids: List[int] = []
    put_async_each_ms: List[float] = []
    try_wait_each_ms: List[float] = []
    total_t0 = time.perf_counter()

    launch_t0 = time.perf_counter()
    for launch_idx in range(launch_count):
        if args.launch_progress_every > 0 and (
            launch_idx == 0
            or launch_idx + 1 == launch_count
            or (launch_idx + 1) % args.launch_progress_every == 0
        ):
            print(
                f"[PROGRESS] launch {launch_idx + 1}/{launch_count}, "
                f"submitted={len(all_task_ids)}",
                flush=True,
            )

        for batch_idx in range(args.batch_size):
            put_t0 = time.perf_counter()
            task_id = client.put_async(
                token_ids=batch_sequences[batch_idx],
                token_mask=None,
                slot_mapping=batch_slot_mappings[batch_idx],
                namespace=[f"uid:{batch_idx}:launch:{launch_idx}:repeat:{repeat_idx}"],
            )
            put_async_each_ms.append((time.perf_counter() - put_t0) * 1000.0)
            all_task_ids.append(int(task_id))
    launch_total_ms = (time.perf_counter() - launch_t0) * 1000.0

    print(
        f"[PROGRESS] burst-once direct put_async done, submitted={len(all_task_ids)}, "
        "starting try_wait",
        flush=True,
    )

    pending = set(all_task_ids)
    done: Dict[int, object] = {}
    wait_t0 = time.perf_counter()
    while pending:
        if time.perf_counter() - wait_t0 > args.try_wait_timeout_seconds:
            break
        try_t0 = time.perf_counter()
        responses = client.try_wait(list(pending))
        try_wait_each_ms.append((time.perf_counter() - try_t0) * 1000.0)
        for task_id, response in responses.items():
            done[task_id] = response
            pending.discard(task_id)
        if pending:
            time.sleep(args.poll_sleep_seconds)
    try_wait_total_ms = (time.perf_counter() - wait_t0) * 1000.0

    if pending:
        client.cancel(list(pending))

    success_tokens = 0
    failed_tasks = 0
    for response in done.values():
        if status_is_success(response.status):
            success_tokens += count_mask_tokens(response.return_mask)
        else:
            failed_tasks += 1

    total_ms = (time.perf_counter() - total_t0) * 1000.0
    print(
        f"[BURST] launch_count={launch_count:>4}, repeat={repeat_idx + 1:>4}/{args.repeat:<4} | "
        f"submitted={len(all_task_ids)}, done={len(done)}, pending={len(pending)}, "
        f"failed={failed_tasks}, launch_total={launch_total_ms:.3f}ms, "
        f"try_wait_total={try_wait_total_ms:.3f}ms, try_wait_calls={len(try_wait_each_ms)}"
    )

    return RepeatTimingRow(
        launch_count=launch_count,
        iteration=repeat_idx + 1,
        batch_size=args.batch_size,
        len_per_seq=args.len_per_seq,
        submitted_tasks=len(all_task_ids),
        done_tasks=len(done),
        pending_tasks=len(pending),
        success_tokens=success_tokens,
        failed_tasks=failed_tasks,
        timeout_tasks=len(pending),
        total_burst_once_wait_ms=total_ms,
        launch_total_ms=launch_total_ms,
        try_wait_total_ms=try_wait_total_ms,
        put_async_each_ms=put_async_each_ms,
        try_wait_each_ms=try_wait_each_ms,
    )


def best_effort_shutdown_kvcache_mgr(kvcache_mgr: KVCacheManager) -> None:
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
    launch_count: int,
    args: argparse.Namespace,
    csv_writer: csv.writer,
) -> Tuple[List[RepeatTimingRow], float]:
    print("\n" + "=" * 88)
    print(
        f"[SCENARIO] direct FlexKV _client.put_async only, launch_count={launch_count}, "
        f"batch_size={args.batch_size}, len_per_seq={args.len_per_seq}, "
        f"repeat={args.repeat}, warmup_repeats={args.warmup_iterations}"
    )
    print("=" * 88)

    kvcache_mgr = create_kvcache_manager(args, launch_count)
    client = getattr(kvcache_mgr.host_kvstorage_manager, "_client", None)
    if client is None:
        raise RuntimeError("FlexKV client is not initialized")

    batch_sequences, batch_slot_mappings = build_direct_put_inputs(args)
    measured: List[RepeatTimingRow] = []
    scenario_t0 = time.perf_counter()
    try:
        for i in range(args.repeat):
            row = run_one_burst_once_wait(
                client=client,
                args=args,
                launch_count=launch_count,
                repeat_idx=i,
                batch_sequences=batch_sequences,
                batch_slot_mappings=batch_slot_mappings,
            )
            if row.iteration <= args.warmup_iterations:
                continue
            measured.append(row)
            csv_writer.writerow(origin_csv_row(row))
            clear_fn = getattr(client, "_clear_cpu_cache", None)
            if args.clear_cpu_cache and clear_fn is not None:
                clear_fn()
    finally:
        best_effort_shutdown_kvcache_mgr(kvcache_mgr)
    return measured, (time.perf_counter() - scenario_t0) * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Breakdown_2 direct FlexKV put_async profiler: align with profiler_offload.py "
            "CLI/output, but skip recsys lookup/allocate/gpu_put/offload_launch."
        )
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--len-per-seq", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--offload-batch-counts", type=str, default="50,100")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--try-wait-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-sleep-seconds", type=float, default=0.01)
    parser.add_argument("--tokens-per-block", type=int, default=_PROFILER_TOKENS_PER_BLOCK)
    parser.add_argument("--flexkv-cpu-cache-gb", type=float, default=_DEFAULT_FLEXKV_CPU_CACHE_GB)
    parser.add_argument("--flexkv-num-cpu-blocks", type=int, default=0)
    parser.add_argument("--flexkv-num-local-blocks", type=int, default=0)
    parser.add_argument("--flexkv-num-tmp-cpu-blocks", type=int, default=0)
    parser.add_argument("--num-primary-cache-pages", type=int, default=512)
    parser.add_argument("--launch-progress-every", type=int, default=10)
    parser.add_argument("--clear-cpu-cache", action="store_true")
    parser.add_argument("--output-root", type=str, default=".")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.len_per_seq = (
        (args.len_per_seq - 1) // args.tokens_per_block + 1
    ) * args.tokens_per_block
    launch_counts = parse_int_list(args.offload_batch_counts)
    torch.manual_seed(args.seed)
    origin_path, summary_path = resolve_output_paths(args)
    origin_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] direct FlexKV _client.put_async only, launch_counts={launch_counts}, "
        f"batch_size={args.batch_size}, len_per_seq={args.len_per_seq}, "
        f"repeat={args.repeat}, warmup={args.warmup_iterations}"
    )
    print(f"[INFO] origin csv: {origin_path}")
    print(f"[INFO] summary csv: {summary_path}")

    summary_rows: List[List] = []
    with origin_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(origin_csv_header())
        for launch_count in launch_counts:
            measured, scenario_wall_ms = run_scenario(
                launch_count=launch_count,
                args=args,
                csv_writer=writer,
            )
            if not measured:
                print(f"[WARN] launch_count={launch_count}: no measured repeats")
                continue
            summary_rows.append(
                summarization_csv_row(
                    launch_count=launch_count,
                    batch_size=args.batch_size,
                    len_per_seq=args.len_per_seq,
                    rows=measured,
                    scenario_wall_ms=scenario_wall_ms,
                )
            )

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summarization_csv_header())
        writer.writerows(summary_rows)

    print(f"[DONE] origin_data: {origin_path}")
    print(f"[DONE] summarization: {summary_path}")


if __name__ == "__main__":
    main()
