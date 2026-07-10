#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import math
import random
import time
from dataclasses import replace

import torch

from inference_benchmark_flexkv_only_onboard import (
    BENCHMARK_CONFIG,
    build_model,
    build_request,
    shutdown_flexkv_client,
)


def forward_with_kvcache_only_onboard_layerwise(
    model_predict,
    batch,
    user_ids: torch.Tensor,
    total_history_lengths: torch.Tensor,
    skip_offload: bool = False,
):
    """Only-onboard forward using standard launch timing and layerwise waits."""
    with torch.inference_mode():
        dense_module = model_predict.dense_module
        kvcache_mgr = dense_module.kvcache

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids,
            total_history_lengths,
        )
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

        torch.cuda.nvtx.range_push("recsys.only_onboard_layerwise.onboard_launch")
        try:
            kvcache_mgr.onboard_launch(
                index_meta,
                lookup_res,
                kvcache_metadata,
            )
        finally:
            torch.cuda.nvtx.range_pop()

        striped_batch = model_predict.strip_cached_tokens(
            batch,
            lookup_res.cached_lengths,
        )

        torch.cuda.nvtx.range_push("HSTU embedding")
        try:
            embeddings = model_predict.sparse_module(striped_batch.features)
        finally:
            torch.cuda.nvtx.range_pop()

        kvcache_info = (index_meta, lookup_res, kvcache_metadata)
        return dense_module.forward_with_kvcache(
            striped_batch,
            embeddings,
            user_ids,
            total_history_lengths,
            kvcache_info,
        )


def wait_offload_queue(kvcache_mgr, range_name: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    torch.cuda.nvtx.range_push(range_name)
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(f"{range_name}.try_wait")
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()


def run_scenario_gpu_hit_layerwise(
    model_predict,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
    timed_iters: int,
    skip_offload: bool,
) -> None:
    base_user_id = 10
    timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    req_timed = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]

    print("warmup")
    for batch, user_ids, total_history_lengths in req_prepare:
        forward_with_kvcache_only_onboard_layerwise(
            model_predict,
            batch,
            user_ids,
            total_history_lengths,
            skip_offload=skip_offload,
        )

    kvcache_mgr = model_predict.dense_module.kvcache
    precheck_user_ids = torch.tensor(timed_user_ids, dtype=torch.int64)
    lookup_res = kvcache_mgr.gpu_kvcache_mgr.lookup(precheck_user_ids)
    gpu_lengths = lookup_res.gpu_cached_lengths.cpu()
    print(f"[Scenario1 precheck] gpu={gpu_lengths.tolist()}")
    expected = history_len * 2
    if any(int(length.item()) != expected for length in gpu_lengths):
        raise RuntimeError(
            f"Scenario1 expects GPU prefix hit ({expected}), got {gpu_lengths.tolist()}"
        )

    print("timed run")
    for iter_idx, (batch, user_ids, total_history_lengths) in enumerate(req_timed):
        torch.cuda.nvtx.range_push(
            f"scenario1_only_onboard_layerwise_timed_run_{iter_idx}"
        )
        try:
            forward_with_kvcache_only_onboard_layerwise(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
                skip_offload=skip_offload,
            )
        finally:
            torch.cuda.nvtx.range_pop()
    print(f"[Scenario1 only_onboard_layerwise] timed run completed, iters={timed_iters}")


def run_scenario_gpu_miss_host_hit_layerwise(
    model_predict,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
    skip_offload: bool,
) -> None:
    base_user_id = 20
    timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    req_timed = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    kvcache_mgr = model_predict.dense_module.kvcache

    print("warmup")
    for batch, user_ids, total_history_lengths in req_prepare:
        forward_with_kvcache_only_onboard_layerwise(
            model_predict,
            batch,
            user_ids,
            total_history_lengths,
            skip_offload=skip_offload,
        )
    wait_offload_queue(
        kvcache_mgr,
        "scenario2.only_onboard_layerwise.warmup.offload_wait_all",
        offload_wait_timeout_s,
    )

    print("timed run")
    for iter_idx, (batch, user_ids, total_history_lengths) in enumerate(req_timed):
        kvcache_mgr.evict(user_ids, for_gpu=True)
        torch.cuda.nvtx.range_push(
            f"scenario2_only_onboard_layerwise_timed_run_{iter_idx}"
        )
        try:
            forward_with_kvcache_only_onboard_layerwise(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
                skip_offload=skip_offload,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    wait_offload_queue(
        kvcache_mgr,
        "scenario2.only_onboard_layerwise.timed.offload_wait_all",
        offload_wait_timeout_s,
    )
    print(f"[Scenario2 only_onboard_layerwise] timed run completed, iters={timed_iters}")


def run_scenario_gpu_cpu_miss_ssd_hit_layerwise(
    model_predict,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
    page_size: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
    ssd_pressure_users: int,
    ssd_pressure_batch_size: int,
    ssd_pressure_batch_sleep_s: float,
    skip_offload: bool,
) -> None:
    base_user_id = 30
    timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    req_targets = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    kvcache_mgr = model_predict.dense_module.kvcache
    host_mgr = kvcache_mgr.host_kvstorage_manager
    cache_cfg = getattr(host_mgr, "_client", None)
    cache_cfg = getattr(cache_cfg, "cache_config", None)

    print("warmup")
    for batch, user_ids, total_history_lengths in req_targets:
        forward_with_kvcache_only_onboard_layerwise(
            model_predict,
            batch,
            user_ids,
            total_history_lengths,
            skip_offload=skip_offload,
        )
    wait_offload_queue(
        kvcache_mgr,
        "scenario3.only_onboard_layerwise.warmup.offload_wait_all",
        offload_wait_timeout_s,
    )

    num_cpu_blocks = int(getattr(cache_cfg, "num_cpu_blocks", 0))
    num_ssd_blocks = int(getattr(cache_cfg, "num_ssd_blocks", 0))
    target_blocks = math.ceil((history_len * 2) / page_size)
    if target_blocks <= 0:
        raise RuntimeError(f"Invalid target_blocks={target_blocks}")
    if ssd_pressure_users <= 0:
        ssd_pressure_users = max(
            128,
            math.ceil(num_cpu_blocks / target_blocks) + timed_iters + 4,
        )

    pressure_base_user_id = base_user_id + 10000
    print(
        "[Scenario3 only_onboard_layerwise pressure] "
        f"num_cpu_blocks={num_cpu_blocks}, num_ssd_blocks={num_ssd_blocks}, "
        f"target_blocks={target_blocks}, pressure_users={ssd_pressure_users}, "
        f"pressure_batch_size={ssd_pressure_batch_size}, "
        f"pressure_batch_sleep_s={ssd_pressure_batch_sleep_s}",
        flush=True,
    )

    torch.cuda.nvtx.range_push("scenario3.only_onboard_layerwise.pressure_fill")
    try:
        for pressure_idx in range(ssd_pressure_users):
            pressure_user_id = pressure_base_user_id + pressure_idx
            batch, user_ids, total_history_lengths = build_request(
                pressure_user_id,
                history_len,
                num_candidates,
                max_seqlen,
            )
            forward_with_kvcache_only_onboard_layerwise(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
                skip_offload=skip_offload,
            )
            if (
                ssd_pressure_batch_size > 0
                and (pressure_idx + 1) % ssd_pressure_batch_size == 0
            ):
                wait_offload_queue(
                    kvcache_mgr,
                    "scenario3.only_onboard_layerwise.pressure.batch_offload_wait_all",
                    offload_wait_timeout_s,
                )
                if ssd_pressure_batch_sleep_s > 0:
                    time.sleep(ssd_pressure_batch_sleep_s)
    finally:
        torch.cuda.nvtx.range_pop()

    wait_offload_queue(
        kvcache_mgr,
        "scenario3.only_onboard_layerwise.pressure.offload_wait_all",
        offload_wait_timeout_s,
    )
    if ssd_pressure_batch_sleep_s > 0:
        time.sleep(ssd_pressure_batch_sleep_s)

    print("timed run")
    for iter_idx in range(timed_iters):
        user_id = timed_user_ids[iter_idx]
        user_ids = torch.tensor([user_id], dtype=torch.int64)
        kvcache_mgr.evict(user_ids, for_gpu=True)

        batch, user_ids, total_history_lengths = build_request(
            user_id,
            history_len,
            num_candidates,
            max_seqlen,
        )

        torch.cuda.nvtx.range_push(
            f"scenario3_only_onboard_layerwise_timed_run_{iter_idx}"
        )
        try:
            forward_with_kvcache_only_onboard_layerwise(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
                skip_offload=skip_offload,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    wait_offload_queue(
        kvcache_mgr,
        "scenario3.only_onboard_layerwise.timed.offload_wait_all",
        offload_wait_timeout_s,
    )
    print(f"[Scenario3 only_onboard_layerwise] timed run completed, iters={timed_iters}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--append-history-len", type=int, default=None)
    parser.add_argument("--ssd-pressure-users", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-size", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-sleep-s", type=float, default=None)
    parser.add_argument("--flexkv-config-path", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default="2,3")
    parser.add_argument("--disable-cudagraph", action="store_true")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--force-skip-offload", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = BENCHMARK_CONFIG
    flexkv_config_path = args.flexkv_config_path or __import__("os").environ.get(
        "RECSYS_FLEXKV_CONFIG_PATH"
    )
    if flexkv_config_path:
        cfg = replace(cfg, flexkv_config_path=flexkv_config_path)
    if args.timed_iters is not None:
        cfg = replace(cfg, timed_iters=args.timed_iters)
    if args.append_history_len is not None:
        cfg = replace(cfg, append_history_len=args.append_history_len)
    if args.ssd_pressure_users is not None:
        cfg = replace(cfg, ssd_pressure_users=args.ssd_pressure_users)
    if args.ssd_pressure_batch_size is not None:
        cfg = replace(cfg, ssd_pressure_batch_size=args.ssd_pressure_batch_size)
    if args.ssd_pressure_batch_sleep_s is not None:
        cfg = replace(cfg, ssd_pressure_batch_sleep_s=args.ssd_pressure_batch_sleep_s)
    if args.disable_cudagraph:
        cfg = replace(cfg, disable_cudagraph=True)
    if args.force_skip_offload:
        cfg = replace(cfg, force_skip_offload=True)
    if args.ablation not in (None, "baseline"):
        raise ValueError("This benchmark currently supports only --ablation baseline.")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    scenarios = {
        scenario.strip() for scenario in args.scenarios.split(",") if scenario.strip()
    }
    print("[Mode] only_onboard_layerwise=True, timed forwards use history_len")
    print(
        f"[Config] history_len={history_len}, append_history_len={cfg.append_history_len}, "
        f"num_candidates={cfg.num_candidates}, disable_cudagraph={cfg.disable_cudagraph}, "
        f"force_skip_offload={cfg.force_skip_offload}, "
        f"scenarios={','.join(sorted(scenarios))}"
    )
    model_predict, page_size, max_seqlen = build_model(cfg, history_len)
    print(f"[Config] page_size={page_size}, max_seqlen={max_seqlen}")

    try:
        with torch.inference_mode():
            if "1" in scenarios:
                run_scenario_gpu_hit_layerwise(
                    model_predict=model_predict,
                    history_len=history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    timed_iters=cfg.timed_iters,
                    skip_offload=cfg.force_skip_offload,
                )
            if "2" in scenarios:
                run_scenario_gpu_miss_host_hit_layerwise(
                    model_predict=model_predict,
                    history_len=history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                    skip_offload=cfg.force_skip_offload,
                )
            if "3" in scenarios:
                run_scenario_gpu_cpu_miss_ssd_hit_layerwise(
                    model_predict=model_predict,
                    history_len=history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    page_size=page_size,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                    ssd_pressure_users=cfg.ssd_pressure_users,
                    ssd_pressure_batch_size=cfg.ssd_pressure_batch_size,
                    ssd_pressure_batch_sleep_s=cfg.ssd_pressure_batch_sleep_s,
                    skip_offload=cfg.force_skip_offload,
                )
    finally:
        shutdown_flexkv_client(model_predict)
