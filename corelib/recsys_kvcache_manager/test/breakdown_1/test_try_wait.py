"""
Benchmark offload wait path for FlexKV phase-1 style requests.

Goal:
1) Baseline path (current KVCacheManager.offload_try_wait):
   multiple client.try_wait + at most one client.wait
   => T_try_wait x N + T_wait_1
2) Patched path:
   monkey patch KVCacheManager.offload_try_wait to skip client.try_wait
   and directly use client.wait only
   => T_wait_2
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Dict, List, Sequence, Tuple

import torch

from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from recsys_kvcache_manager.kvcache_manager import KVCacheManager

PHASE1_SEQLEN = [700, 128, 336, 624, 486, 358, 716, 537]
PAGE_SIZE = 32
NUM_LAYERS = 3
NUM_HEADS = 4
HEAD_DIM = 128


@dataclass
class TrialStat:
    mode: str
    trial_idx: int
    manager_offload_try_wait_times_ms: List[float]
    client_try_wait_times_ms: List[float]
    client_wait_times_ms: List[float]
    offload_loop_wall_ms: float

    @property
    def manager_offload_try_wait_total_ms(self) -> float:
        return sum(self.manager_offload_try_wait_times_ms)

    @property
    def client_try_wait_total_ms(self) -> float:
        return sum(self.client_try_wait_times_ms)

    @property
    def client_wait_total_ms(self) -> float:
        return sum(self.client_wait_times_ms)

    @property
    def decomposition_ms(self) -> float:
        return self.client_try_wait_total_ms + self.client_wait_total_ms


def normalize_index_meta(index_meta) -> None:
    if not hasattr(index_meta, "sequence_lengths"):
        index_meta.sequence_lengths = index_meta.seq_lengths
    if not hasattr(index_meta, "slot_mappings"):
        index_meta.slot_mappings = None
    if hasattr(index_meta, "namespaces") and index_meta.namespaces is not None:
        index_meta.namespaces = [
            ns if isinstance(ns, list) else [ns] for ns in index_meta.namespaces
        ]


def parse_seqlens(raw: str, batch_size: int) -> List[int]:
    if raw.strip().lower() == "phase1":
        seqlens = PHASE1_SEQLEN.copy()
    else:
        seqlens = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(seqlens) != batch_size:
        raise ValueError(
            f"seqlens size mismatch: got {len(seqlens)}, expected {batch_size}. "
            "Use --seqlens phase1 with --batch-size 8, or provide a list with "
            "exactly batch_size integers."
        )
    if any(s <= 0 for s in seqlens):
        raise ValueError("all sequence lengths must be positive")
    return seqlens


def create_testing_kvcache_manager(
    batch_size: int,
    max_seq_len: int,
    num_primary_cache_pages: int,
    host_capacity_scale: float,
    flexkv_num_cpu_blocks: int,
    flexkv_num_local_blocks: int,
    flexkv_num_tmp_cpu_blocks: int,
) -> KVCacheManager:
    if host_capacity_scale <= 0:
        raise ValueError("host_capacity_scale must be positive")

    base_host_capacity = max_seq_len * batch_size * PAGE_SIZE * NUM_HEADS * HEAD_DIM * 2
    host_capacity_per_layer = int(base_host_capacity * host_capacity_scale)

    kvcache_config = get_kvcache_config(
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        offload_chunksize=128,
        num_primary_cache_pages=num_primary_cache_pages,
        num_buffer_pages=0,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=batch_size,
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
    return KVCacheManager.from_config(kvcache_config)


def build_phase1_like_batch(
    batch_size: int,
    seqlens: Sequence[int],
    uid_base: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    user_ids = torch.arange(uid_base, uid_base + batch_size, dtype=torch.int64)
    sequence_lengths = torch.tensor(seqlens, dtype=torch.int32)
    keys = [
        torch.randn((NUM_LAYERS, seqlens[i], NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
        .cuda()
        .contiguous()
        for i in range(batch_size)
    ]
    values = [
        torch.randn((NUM_LAYERS, seqlens[i], NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
        .cuda()
        .contiguous()
        for i in range(batch_size)
    ]
    return user_ids, sequence_lengths, keys, values


def install_wait_only_offload_try_wait(kvcache_mgr: KVCacheManager) -> None:
    def _offload_wait_only(self: KVCacheManager) -> None:
        for task_handle in self.ongoing_offload_tasks:
            offload_success = self.host_kvstorage_manager.finish_task(task_handle)
            self.gpu_kvcache_mgr.release_offload_pages(
                *(self.host_kvstorage_manager.get_offload_handle_metadata(task_handle)),
                offloaded=offload_success,
            )
        self.ongoing_offload_tasks = []

    kvcache_mgr.offload_try_wait = MethodType(_offload_wait_only, kvcache_mgr)


def run_one_trial(
    kvcache_mgr: KVCacheManager,
    mode: str,
    trial_idx: int,
    batch_size: int,
    seqlens: Sequence[int],
    sleep_ms_between_try_wait: float,
    uid_base: int,
) -> TrialStat:
    user_ids, sequence_lengths, keys, values = build_phase1_like_batch(
        batch_size=batch_size,
        seqlens=seqlens,
        uid_base=uid_base,
    )

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)

    if not torch.allclose(
        lookup_res.cached_lengths,
        torch.zeros((batch_size,), dtype=torch.int32),
    ):
        raise AssertionError(
            f"Trial {trial_idx}: expected fresh users with zero cached_lengths, "
            f"actual={lookup_res.cached_lengths}"
        )

    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)
    if getattr(kvcache_metadata, "new_history_nnz_cuda", None) is not None:
        kvcache_metadata.new_history_nnz = int(kvcache_metadata.new_history_nnz_cuda.item())

    for layer_idx in range(NUM_LAYERS):
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
    if task_handle is None or task_handle.handle is None:
        raise RuntimeError("offload_launch did not return a valid handle")

    flexkv_mgr = kvcache_mgr.host_kvstorage_manager
    client = getattr(flexkv_mgr, "_client", None)
    if client is None:
        raise RuntimeError("FlexKV client not found")

    client_try_wait_times_ms: List[float] = []
    client_wait_times_ms: List[float] = []
    manager_offload_try_wait_times_ms: List[float] = []

    original_client_try_wait = client.try_wait
    original_client_wait = client.wait

    def _timed_try_wait(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_client_try_wait(*args, **kwargs)
        client_try_wait_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    def _timed_wait(*args, **kwargs):
        t0 = time.perf_counter()
        out = original_client_wait(*args, **kwargs)
        client_wait_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    client.try_wait = _timed_try_wait
    client.wait = _timed_wait

    try:
        loop_t0 = time.perf_counter()
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            call_t0 = time.perf_counter()
            kvcache_mgr.offload_try_wait()
            manager_offload_try_wait_times_ms.append(
                (time.perf_counter() - call_t0) * 1000.0
            )
            if (
                sleep_ms_between_try_wait > 0
                and len(kvcache_mgr.ongoing_offload_tasks) > 0
            ):
                time.sleep(sleep_ms_between_try_wait / 1000.0)
        offload_loop_wall_ms = (time.perf_counter() - loop_t0) * 1000.0
    finally:
        client.try_wait = original_client_try_wait
        client.wait = original_client_wait

    expected_host_lens = [s // PAGE_SIZE * PAGE_SIZE for s in seqlens]
    _, post_lookup = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    host_lens = [int(x.item()) for x in post_lookup.host_cached_lengths]
    if host_lens != expected_host_lens:
        raise AssertionError(
            "host_cached_lengths mismatch after offload. "
            f"expected={expected_host_lens}, actual={host_lens}"
        )

    return TrialStat(
        mode=mode,
        trial_idx=trial_idx,
        manager_offload_try_wait_times_ms=manager_offload_try_wait_times_ms,
        client_try_wait_times_ms=client_try_wait_times_ms,
        client_wait_times_ms=client_wait_times_ms,
        offload_loop_wall_ms=offload_loop_wall_ms,
    )


def summarize_mode(stats: Sequence[TrialStat], mode: str) -> None:
    if not stats:
        return
    total_try_wait_calls = sum(len(s.client_try_wait_times_ms) for s in stats)
    total_wait_calls = sum(len(s.client_wait_times_ms) for s in stats)
    total_try_wait_ms = sum(s.client_try_wait_total_ms for s in stats)
    total_wait_ms = sum(s.client_wait_total_ms for s in stats)
    total_decomposition_ms = sum(s.decomposition_ms for s in stats)
    total_wall_ms = sum(s.offload_loop_wall_ms for s in stats)
    trial_count = len(stats)

    avg_n = total_try_wait_calls / trial_count
    avg_t_try_wait = (
        total_try_wait_ms / total_try_wait_calls if total_try_wait_calls > 0 else 0.0
    )
    avg_t_wait = total_wait_ms / trial_count
    avg_t_wait2 = total_wait_ms / trial_count

    print("\n" + "-" * 96)
    print(f"[SUMMARY][{mode}] trials={trial_count}")
    print(
        f"[SUMMARY][{mode}] total client.try_wait calls={total_try_wait_calls}, "
        f"total client.wait calls={total_wait_calls}"
    )
    print(
        f"[SUMMARY][{mode}] total try_wait={total_try_wait_ms:.3f} ms, "
        f"total wait={total_wait_ms:.3f} ms, "
        f"total (try_wait + wait)={total_decomposition_ms:.3f} ms"
    )
    print(
        f"[SUMMARY][{mode}] total offload loop wall={total_wall_ms:.3f} ms "
        "(around KVCacheManager.offload_try_wait loop)"
    )
    if mode == "baseline":
        print(
            f"[SUMMARY][{mode}] avg formula: T_try_wait x N + T_wait_1 "
            f"= ({avg_t_try_wait:.3f} ms) x ({avg_n:.2f}) + ({avg_t_wait:.3f} ms)"
        )
    else:
        print(
            f"[SUMMARY][{mode}] avg formula: T_wait_2 = {avg_t_wait2:.3f} ms "
            "(client.wait only path)"
        )
    print("-" * 96)


def write_csv(path: Path, all_stats: Sequence[TrialStat]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "trial_idx",
                "manager_offload_try_wait_calls",
                "manager_offload_try_wait_total_ms",
                "manager_offload_try_wait_times_ms",
                "client_try_wait_calls",
                "client_try_wait_total_ms",
                "client_try_wait_times_ms",
                "client_wait_calls",
                "client_wait_total_ms",
                "client_wait_times_ms",
                "decomposition_ms",
                "offload_loop_wall_ms",
            ]
        )
        for stat in all_stats:
            writer.writerow(
                [
                    stat.mode,
                    stat.trial_idx,
                    len(stat.manager_offload_try_wait_times_ms),
                    f"{stat.manager_offload_try_wait_total_ms:.6f}",
                    ";".join(f"{x:.6f}" for x in stat.manager_offload_try_wait_times_ms),
                    len(stat.client_try_wait_times_ms),
                    f"{stat.client_try_wait_total_ms:.6f}",
                    ";".join(f"{x:.6f}" for x in stat.client_try_wait_times_ms),
                    len(stat.client_wait_times_ms),
                    f"{stat.client_wait_total_ms:.6f}",
                    ";".join(f"{x:.6f}" for x in stat.client_wait_times_ms),
                    f"{stat.decomposition_ms:.6f}",
                    f"{stat.offload_loop_wall_ms:.6f}",
                ]
            )


def run_mode(
    mode: str,
    args: argparse.Namespace,
    seqlens: Sequence[int],
    uid_offset_base: int,
) -> List[TrialStat]:
    kvcache_mgr = create_testing_kvcache_manager(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_primary_cache_pages=args.num_primary_cache_pages,
        host_capacity_scale=args.host_capacity_scale,
        flexkv_num_cpu_blocks=args.flexkv_num_cpu_blocks,
        flexkv_num_local_blocks=args.flexkv_num_local_blocks,
        flexkv_num_tmp_cpu_blocks=args.flexkv_num_tmp_cpu_blocks,
    )
    if mode == "wait_only":
        install_wait_only_offload_try_wait(kvcache_mgr)

    stats: List[TrialStat] = []
    try:
        for trial_idx in range(args.trials):
            uid_base = uid_offset_base + trial_idx * args.batch_size
            stat = run_one_trial(
                kvcache_mgr=kvcache_mgr,
                mode=mode,
                trial_idx=trial_idx,
                batch_size=args.batch_size,
                seqlens=seqlens,
                sleep_ms_between_try_wait=args.sleep_ms_between_try_wait,
                uid_base=uid_base,
            )
            stats.append(stat)
            print(
                f"[{mode}] trial={trial_idx} | "
                f"manager.offload_try_wait calls={len(stat.manager_offload_try_wait_times_ms)}, "
                f"client.try_wait calls={len(stat.client_try_wait_times_ms)}, "
                f"client.wait calls={len(stat.client_wait_times_ms)}, "
                f"try_wait_total={stat.client_try_wait_total_ms:.3f} ms, "
                f"wait_total={stat.client_wait_total_ms:.3f} ms, "
                f"decomposition={stat.decomposition_ms:.3f} ms, "
                f"loop_wall={stat.offload_loop_wall_ms:.3f} ms"
            )
    finally:
        flexkv_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
        client = getattr(flexkv_mgr, "_client", None)
        if client is not None and hasattr(client, "shutdown"):
            try:
                client.shutdown()
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] FlexKV client shutdown failed: {exc}")

    summarize_mode(stats=stats, mode=mode)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark A/B for offload waiting path.\n"
            "A) baseline: current offload_try_wait -> client.try_wait multiple times "
            "+ at most one client.wait\n"
            "B) wait_only: patch offload_try_wait to skip client.try_wait and use "
            "client.wait only"
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "wait_only", "both"],
        default="both",
        help="Which mode(s) to run.",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seqlens",
        type=str,
        default="phase1",
        help="Use 'phase1' or comma-separated lengths, size must equal batch-size.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Shape upper bound for KVCache manager config.",
    )
    parser.add_argument("--num-primary-cache-pages", type=int, default=512)
    parser.add_argument("--host-capacity-scale", type=float, default=1.0)
    parser.add_argument(
        "--flexkv-num-cpu-blocks",
        type=int,
        default=0,
        help=(
            "FlexKV CPU blocks. 0 means auto estimate based on "
            "trials*batch_size*max(seqlen)/page_size."
        ),
    )
    parser.add_argument(
        "--flexkv-num-local-blocks",
        type=int,
        default=0,
        help=(
            "FlexKV local blocks. 0 means follow CPU blocks (auto mode) "
            "or use --flexkv-num-cpu-blocks (manual mode)."
        ),
    )
    parser.add_argument(
        "--flexkv-num-tmp-cpu-blocks",
        type=int,
        default=256,
        help="FlexKV tmp CPU blocks.",
    )
    parser.add_argument(
        "--sleep-ms-between-try-wait",
        type=float,
        default=0.0,
        help="Optional sleep between offload_try_wait loop iterations.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional CSV output path for per-trial details.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive")

    seqlens = parse_seqlens(args.seqlens, args.batch_size)
    if max(seqlens) > args.max_seq_len:
        raise ValueError(
            f"--max-seq-len={args.max_seq_len} is smaller than max(seqlens)={max(seqlens)}"
        )

    torch.manual_seed(args.seed)

    blocks_per_seq = (max(seqlens) + PAGE_SIZE - 1) // PAGE_SIZE
    estimated_blocks_needed = args.trials * args.batch_size * blocks_per_seq

    if args.flexkv_num_cpu_blocks == 0:
        # Add 25% headroom and a floor to avoid frequent OOM at medium sweep scales.
        flexkv_num_cpu_blocks = max(4096, int(estimated_blocks_needed * 1.25))
    else:
        flexkv_num_cpu_blocks = args.flexkv_num_cpu_blocks

    if args.flexkv_num_local_blocks == 0:
        flexkv_num_local_blocks = flexkv_num_cpu_blocks
    else:
        flexkv_num_local_blocks = args.flexkv_num_local_blocks

    if args.flexkv_num_tmp_cpu_blocks <= 0:
        raise ValueError("--flexkv-num-tmp-cpu-blocks must be positive")

    print(
        f"[INFO] mode={args.mode}, trials={args.trials}, batch_size={args.batch_size}, "
        f"seqlens={seqlens}, max_seq_len={args.max_seq_len}"
    )
    print(
        "[INFO] flexkv blocks: "
        f"cpu={flexkv_num_cpu_blocks}, local={flexkv_num_local_blocks}, "
        f"tmp_cpu={args.flexkv_num_tmp_cpu_blocks} "
        f"(estimated_needed={estimated_blocks_needed})"
    )

    run_args_dict = dict(vars(args))
    run_args_dict["flexkv_num_cpu_blocks"] = flexkv_num_cpu_blocks
    run_args_dict["flexkv_num_local_blocks"] = flexkv_num_local_blocks
    run_args = argparse.Namespace(**run_args_dict)

    all_stats: List[TrialStat] = []
    if args.mode in ("baseline", "both"):
        all_stats.extend(
            run_mode(
                "baseline",
                run_args,
                seqlens,
                uid_offset_base=0,
            )
        )
    if args.mode in ("wait_only", "both"):
        all_stats.extend(
            run_mode(
                "wait_only",
                run_args,
                seqlens,
                uid_offset_base=args.trials * args.batch_size * 2,
            )
        )

    if args.output_csv.strip():
        output_path = Path(args.output_csv).expanduser().resolve()
        write_csv(output_path, all_stats)
        print(f"[DONE] wrote csv: {output_path}")


if __name__ == "__main__":
    main()
