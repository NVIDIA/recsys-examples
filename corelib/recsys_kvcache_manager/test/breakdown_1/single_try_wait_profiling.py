"""
Micro-bench 1 sub-experiment — one offload_try_wait() call (NVTX), then exit.

Imports shared hooks/pipeline helpers from test_flexkv_profile_fine.py.
Post-wait drain/shutdown is skipped because it can hang.
"""

import argparse
import os

import torch

from test_flexkv_profile_fine import (
    PROFILE_MODE_FINE,
    build_uniform_batch,
    create_testing_kvcache_manager,
    install_cpu_cpp_hooks,
    install_gpu_cpp_kernel_hooks,
    normalize_index_meta,
    nvtx_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one single offload_try_wait call with NVTX, then exit. "
            "Post-wait drain/shutdown is skipped because it can hang."
        )
    )
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--len-per-seq", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--base-profile-mode",
        choices=[PROFILE_MODE_FINE],
        default=PROFILE_MODE_FINE,
        help="Must match five-level breakdown (flexkv_fine hooks).",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def run_one_case(args: argparse.Namespace) -> None:
    with nvtx_range("init.create_kvcache_manager"):
        kvcache_mgr = create_testing_kvcache_manager(
            max_batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            profile_mode=args.base_profile_mode,
        )
        install_gpu_cpp_kernel_hooks(kvcache_mgr)
        install_cpu_cpp_hooks(kvcache_mgr)

    with nvtx_range("init.prepare_inputs"):
        all_keys = [
            torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
            for _ in range(args.batch_size)
        ]
        all_values = [
            torch.randn((3, args.max_seq_len, 4, 128), dtype=torch.bfloat16).cuda()
            for _ in range(args.batch_size)
        ]

    with nvtx_range("step1.input"):
        user_ids, sequence_lengths, keys, values, _ = build_uniform_batch(
            all_keys=all_keys,
            all_values=all_values,
            len_per_seq=args.len_per_seq,
            batch_size=args.batch_size,
        )

    with nvtx_range("step1.lookup"):
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
    normalize_index_meta(index_meta)

    with nvtx_range("step1.allocate"):
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    for layer_idx in range(3):
        kvcache_mgr.gpu_kvcache_mgr.put(
            torch.cat([k[layer_idx] for k in keys], dim=0),
            torch.cat([v[layer_idx] for v in values], dim=0),
            layer_idx,
            kvcache_metadata,
        )

    with nvtx_range("step1.offload_launch"):
        task_handle = kvcache_mgr.offload_launch(
            index_meta=index_meta,
            kvcache_metadata=kvcache_metadata,
        )
    if task_handle is None or task_handle.handle is None:
        raise RuntimeError("offload_launch did not return a valid handle")

    before = len(kvcache_mgr.ongoing_offload_tasks)
    with nvtx_range("step1.offload_try_wait_once"):
        kvcache_mgr.offload_try_wait()
    after = len(kvcache_mgr.ongoing_offload_tasks)
    print(f"[INFO] ongoing_offload_tasks before={before}, after={after}")
    print(
        "[INFO] Exiting after step1.offload_try_wait_once "
        "(drain/shutdown skipped — extra drain can hang)."
    )
    os._exit(0)


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

    torch.manual_seed(args.seed)
    print(
        f"[INFO] single_try_wait profile: len_per_seq={args.len_per_seq}, "
        f"batch_size={args.batch_size}, base_profile_mode={args.base_profile_mode}"
    )
    run_one_case(args)


if __name__ == "__main__":
    main()
