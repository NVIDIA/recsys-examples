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

from inference_benchmark_flexkv import (
    BENCHMARK_CONFIG,
    build_model,
    build_request,
    run_scenario_gpu_hit,
    shutdown_flexkv_client,
)


def forward_with_kvcache_layerwise_attention_overlap(
    model_predict,
    batch,
    user_ids: torch.Tensor,
    total_history_lengths: torch.Tensor,
):
    """Run the standard kvcache path with layerwise wait inside HSTU layers."""
    with torch.inference_mode():
        dense_module = model_predict.dense_module
        kvcache_mgr = dense_module.kvcache

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids,
            total_history_lengths,
        )
        kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

        torch.cuda.nvtx.range_push("recsys.layerwise_overlap.onboard_launch")
        try:
            kvcache_mgr.onboard_launch(
                index_meta,
                lookup_res,
                kvcache_metadata,
            )
        finally:
            torch.cuda.nvtx.range_pop()

        old_cached_lengths = lookup_res.cached_lengths
        striped_batch = model_predict.strip_cached_tokens(
            batch,
            old_cached_lengths,
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


def run_scenario_gpu_miss_host_hit_layerwise(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
) -> None:
    base_user_id = 20
    timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    req_timed = [
        build_request(
            user_id,
            history_len + append_history_len,
            num_candidates,
            max_seqlen,
        )
        for user_id in timed_user_ids
    ]
    kvcache_mgr = model_predict.dense_module.kvcache

    print("warmup")
    for batch, user_ids, total_history_lengths in req_prepare:
        forward_with_kvcache_layerwise_attention_overlap(
            model_predict,
            batch,
            user_ids,
            total_history_lengths,
        )

    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario2.layerwise.warmup.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(
                "scenario2.layerwise.warmup.offload_wait_all.try_wait"
            )
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()

    print("timed run")
    for iter_idx, (batch, user_ids, total_history_lengths) in enumerate(req_timed):
        kvcache_mgr.evict(user_ids, for_gpu=True)

        torch.cuda.nvtx.range_push(f"scenario2_layerwise_timed_run_{iter_idx}")
        try:
            forward_with_kvcache_layerwise_attention_overlap(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario2.layerwise.timed.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(
                "scenario2.layerwise.timed.offload_wait_all.try_wait"
            )
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()
    print(f"[Scenario2 layerwise] timed run completed, iters={timed_iters}")


def run_scenario_gpu_cpu_miss_ssd_hit_layerwise(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    page_size: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
    ssd_pressure_users: int,
    ssd_pressure_batch_size: int,
    ssd_pressure_batch_sleep_s: float,
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
        forward_with_kvcache_layerwise_attention_overlap(
            model_predict,
            batch,
            user_ids,
            total_history_lengths,
        )

    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.layerwise.warmup.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(
                "scenario3.layerwise.warmup.offload_wait_all.try_wait"
            )
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()

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
        "[Scenario3 layerwise pressure] "
        f"num_cpu_blocks={num_cpu_blocks}, num_ssd_blocks={num_ssd_blocks}, "
        f"target_blocks={target_blocks}, pressure_users={ssd_pressure_users}, "
        f"pressure_batch_size={ssd_pressure_batch_size}, "
        f"pressure_batch_sleep_s={ssd_pressure_batch_sleep_s}",
        flush=True,
    )

    torch.cuda.nvtx.range_push("scenario3.layerwise.pressure_fill")
    try:
        for pressure_idx in range(ssd_pressure_users):
            pressure_user_id = pressure_base_user_id + pressure_idx
            batch, user_ids, total_history_lengths = build_request(
                pressure_user_id,
                history_len,
                num_candidates,
                max_seqlen,
            )
            forward_with_kvcache_layerwise_attention_overlap(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
            )

            if (
                ssd_pressure_batch_size > 0
                and (pressure_idx + 1) % ssd_pressure_batch_size == 0
            ):
                deadline = time.time() + offload_wait_timeout_s
                torch.cuda.nvtx.range_push(
                    "scenario3.layerwise.pressure.batch_offload_wait_all"
                )
                try:
                    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
                        torch.cuda.nvtx.range_push(
                            "scenario3.layerwise.pressure.batch_offload_wait_all.try_wait"
                        )
                        try:
                            kvcache_mgr.offload_try_wait()
                        finally:
                            torch.cuda.nvtx.range_pop()
                        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                            break
                        if time.time() > deadline:
                            raise TimeoutError(
                                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                                f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                            )
                        time.sleep(0.001)
                finally:
                    torch.cuda.nvtx.range_pop()
                if ssd_pressure_batch_sleep_s > 0:
                    time.sleep(ssd_pressure_batch_sleep_s)
    finally:
        torch.cuda.nvtx.range_pop()

    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.layerwise.pressure.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(
                "scenario3.layerwise.pressure.offload_wait_all.try_wait"
            )
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()
    if ssd_pressure_batch_sleep_s > 0:
        time.sleep(ssd_pressure_batch_sleep_s)

    print("timed run")
    for iter_idx in range(timed_iters):
        user_id = timed_user_ids[iter_idx]
        user_ids = torch.tensor([user_id], dtype=torch.int64)
        kvcache_mgr.evict(user_ids, for_gpu=True)

        batch, user_ids, total_history_lengths = build_request(
            user_id,
            history_len + append_history_len,
            num_candidates,
            max_seqlen,
        )

        torch.cuda.nvtx.range_push(f"scenario3_layerwise_timed_run_{iter_idx}")
        try:
            forward_with_kvcache_layerwise_attention_overlap(
                model_predict,
                batch,
                user_ids,
                total_history_lengths,
            )
        finally:
            torch.cuda.nvtx.range_pop()

    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.layerwise.timed.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push(
                "scenario3.layerwise.timed.offload_wait_all.try_wait"
            )
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()
    print(f"[Scenario3 layerwise] timed run completed, iters={timed_iters}")


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
    args, _ = parser.parse_known_args()

    cfg = BENCHMARK_CONFIG
    if args.flexkv_config_path:
        cfg = replace(cfg, flexkv_config_path=args.flexkv_config_path)
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

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    scenarios = {
        scenario.strip() for scenario in args.scenarios.split(",") if scenario.strip()
    }
    print(
        f"[Layerwise Config] history_len={history_len}, append_history_len={cfg.append_history_len}, "
        f"num_candidates={cfg.num_candidates}, disable_cudagraph={cfg.disable_cudagraph}, "
        f"scenarios={','.join(sorted(scenarios))}"
    )
    model_predict, page_size, max_seqlen = build_model(cfg, history_len)
    print(f"[Layerwise Config] page_size={page_size}, max_seqlen={max_seqlen}")

    try:
        with torch.inference_mode():
            if "1" in scenarios:
                run_scenario_gpu_hit(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    warmup_iters=cfg.warmup_iters,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
            if "2" in scenarios:
                run_scenario_gpu_miss_host_hit_layerwise(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
            if "3" in scenarios:
                run_scenario_gpu_cpu_miss_ssd_hit_layerwise(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    page_size=page_size,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                    ssd_pressure_users=cfg.ssd_pressure_users,
                    ssd_pressure_batch_size=cfg.ssd_pressure_batch_size,
                    ssd_pressure_batch_sleep_s=cfg.ssd_pressure_batch_sleep_s,
                )
    finally:
        shutdown_flexkv_client(model_predict)
