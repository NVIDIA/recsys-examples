#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import math
import random
import time
from dataclasses import dataclass, replace
from typing import Tuple

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceEmbeddingConfig, RankingConfig, get_inference_hstu_config
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

import sys

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr


ITEM_FEATURE_NAME = "item_feat"
ACTION_FEATURE_NAME = "act_feat"
ITEM_VOCAB_SIZE = 10000
ACTION_VOCAB_SIZE = 128


InferenceRequest = Tuple[HSTUBatch, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class BenchmarkConfig:
    history_len: int = 1024
    append_history_len: int = 1024
    num_candidates: int = 256
    warmup_iters: int = 1
    timed_iters: int = 10
    seed: int = 20260624
    max_batch_size: int = 16
    disable_cudagraph: bool = False
    flexkv_config_path: str = ""
    flexkv_num_cpu_blocks: int = 4096
    flexkv_num_local_blocks: int = 4096
    ssd_pressure_users: int = 0
    ssd_pressure_batch_size: int = 8
    ssd_pressure_batch_sleep_s: float = 1.0
    offload_wait_timeout_s: float = 60.0
    force_skip_offload: bool = False


BENCHMARK_CONFIG = BenchmarkConfig()


def build_request(
    user_id: int,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
) -> InferenceRequest:
    item_seq = torch.randint(
        low=0,
        high=ITEM_VOCAB_SIZE,
        size=(history_len + num_candidates,),
        dtype=torch.long,
    )
    action_seq = torch.randint(
        low=0,
        high=ACTION_VOCAB_SIZE,
        size=(history_len,),
        dtype=torch.long,
    )
    features = KeyedJaggedTensor.from_jt_dict(
        {
            ITEM_FEATURE_NAME: JaggedTensor.from_dense([item_seq]),
            ACTION_FEATURE_NAME: JaggedTensor.from_dense([action_seq]),
        }
    )
    batch = HSTUBatch(
        features=features,
        batch_size=1,
        feature_to_max_seqlen={
            ITEM_FEATURE_NAME: max_seqlen,
            ACTION_FEATURE_NAME: max_seqlen,
        },
        contextual_feature_names=[],
        item_feature_name=ITEM_FEATURE_NAME,
        action_feature_name=ACTION_FEATURE_NAME,
        max_num_candidates=num_candidates,
        num_candidates=torch.full((1,), num_candidates, dtype=torch.long),
    ).to(device=torch.cuda.current_device())
    return (
        batch,
        torch.tensor([user_id], dtype=torch.int64),
        torch.tensor([history_len * 2], dtype=torch.int32),
    )


def build_model(cfg: BenchmarkConfig, history_len: int):
    max_num_history = max(2048, history_len + cfg.append_history_len)
    max_num_candidates = cfg.num_candidates
    max_seqlen = max_num_history * 2 + max_num_candidates

    hidden_dim_size = 1024
    num_heads = 4
    head_dim = 256
    num_layers = 8
    inference_dtype = torch.bfloat16
    hstu_cudagraph_configs = {
        "batch_size": [1, 2, 4, 8],
        "length_per_sequence": [i * 256 for i in range(2, 18)],
    }
    hstu_config = get_inference_hstu_config(
        hidden_size=hidden_dim_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        max_batch_size=cfg.max_batch_size,
        max_seq_len=max_seqlen,
        dtype=inference_dtype,
    )

    sm_major = torch.cuda.get_device_capability()[0]
    page_size = 128 if sm_major >= 10 else 32
    offload_chunksize = 8192
    base_cache_tokens = 10240 * 32
    num_primary_cache_pages = math.ceil(base_cache_tokens / page_size)
    host_capacity_per_layer = (
        num_primary_cache_pages * 2 * page_size * (num_heads * head_dim) * 2
    )

    extra_configs = {
        "flexkv_mode": "direct",
        "flexkv_host_kvstorage_fail_policy": "fail_open",
        "flexkv_enable_mps": 0,
        "flexkv_as_batch": 1,
        "flexkv_num_cpu_blocks": int(cfg.flexkv_num_cpu_blocks),
        "flexkv_num_local_blocks": int(cfg.flexkv_num_local_blocks),
    }
    if cfg.flexkv_config_path:
        extra_configs["flexkv_config_path"] = cfg.flexkv_config_path

    kv_cache_config = get_kvcache_config(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        offload_chunksize=offload_chunksize,
        num_primary_cache_pages=num_primary_cache_pages,
        num_buffer_pages=0,
        host_capacity_per_layer=host_capacity_per_layer,
        max_batch_size=cfg.max_batch_size,
        max_seq_len=math.ceil(max_seqlen / page_size) * page_size,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        host_kvstorage_backend="flexkv",
        offload_timeout_ms=100.0,
        offload_mode="lazy",
        extra_configs=extra_configs,
    )

    emb_configs = [
        InferenceEmbeddingConfig(
            feature_names=[ITEM_FEATURE_NAME],
            table_name="item",
            vocab_size=ITEM_VOCAB_SIZE,
            dim=hidden_dim_size,
            use_dynamicemb=True,
        ),
        InferenceEmbeddingConfig(
            feature_names=[ACTION_FEATURE_NAME],
            table_name="act",
            vocab_size=ACTION_VOCAB_SIZE,
            dim=hidden_dim_size,
            use_dynamicemb=False,
        ),
    ]
    task_config = RankingConfig(
        embedding_configs=emb_configs,
        prediction_head_arch=[128, 10, 1],
        num_tasks=3,
    )

    model_predict = get_inference_ranking_gr(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=not cfg.disable_cudagraph,
        cudagraph_configs=hstu_cudagraph_configs,
    )
    model_predict.bfloat16()
    model_predict.eval()
    return model_predict, page_size, max_seqlen

def run_scenario_gpu_hit(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
) -> None:
    base_user_id = 10
    timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for user_id in timed_user_ids
    ]
    req_timed = [
        build_request(
            user_id,
            history_len,
            num_candidates,
            max_seqlen,
        )
        for user_id in timed_user_ids
    ]

    # warmup
    print("warmup")
    for batch, user_ids, total_history_lengths in req_prepare:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )

    # precheck
    kvcache_mgr = model_predict.dense_module.kvcache
    precheck_user_ids = torch.tensor(timed_user_ids, dtype=torch.int64)
    lookup_res = kvcache_mgr.gpu_kvcache_mgr.lookup(precheck_user_ids)
    gpu_lengths = lookup_res.gpu_cached_lengths.cpu()
    print(
        f"[Scenario1 precheck] gpu={gpu_lengths.tolist()}"
    )
    expected = history_len * 2
    if any(int(length.item()) != expected for length in gpu_lengths):
        raise RuntimeError(
            f"Scenario1 expects GPU prefix hit ({expected}), got {gpu_lengths.tolist()}"
        )

    # timed run
    print("timed run")
    for iter_idx, (batch, user_ids, total_history_lengths) in enumerate(req_timed):
        torch.cuda.nvtx.range_push(f"scenario1_timed_run_{iter_idx}")
        forward_kwargs = {"skip_offload": True} if BENCHMARK_CONFIG.force_skip_offload else {}
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
            **forward_kwargs,
        )
        torch.cuda.nvtx.range_pop()
    print(f"[Scenario1] timed run completed, iters={timed_iters}")


def run_scenario_gpu_miss_host_hit(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
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
            history_len,
            num_candidates,
            max_seqlen,
        )
        for user_id in timed_user_ids
    ]
    kvcache_mgr = model_predict.dense_module.kvcache

    # warmup
    print("warmup")
    for batch, user_ids, total_history_lengths in req_prepare:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario2.warmup.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario2.warmup.offload_wait_all.try_wait")
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

    # timed run
    print("timed run")
    for iter_idx, (batch, user_ids, total_history_lengths) in enumerate(req_timed):
        # Each forward onboards the KV back to GPU. Evict before every timed
        # iteration so the measured prefix stays CPU-hit instead of becoming GPU-hit.

        # evict GPU
        kvcache_mgr.evict(user_ids, for_gpu=True)

        #timed run
        torch.cuda.nvtx.range_push(f"scenario2_timed_run_{iter_idx}")
        forward_kwargs = {"skip_offload": True} if BENCHMARK_CONFIG.force_skip_offload else {}
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
            **forward_kwargs,
        )
        torch.cuda.nvtx.range_pop()
    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario2.timed.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario2.timed.offload_wait_all.try_wait")
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
    print(f"[Scenario2] timed run completed, iters={timed_iters}")


def run_scenario_gpu_cpu_miss_ssd_hit(
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
    # Previous clear_cpu_cache() based version, kept here for reference.
    # It is intentionally commented out instead of deleted so the two Scenario3
    # designs can be compared in-place.
    #
    # base_user_id = 30
    # timed_user_ids = [base_user_id + i for i in range(timed_iters)]
    # req_targets = [
    #     build_request(user_id, history_len, num_candidates, max_seqlen)
    #     for user_id in timed_user_ids
    # ]
    # kvcache_mgr = model_predict.dense_module.kvcache
    # host_mgr = kvcache_mgr.host_kvstorage_manager
    # cache_cfg = getattr(host_mgr, "_client", None)
    # cache_cfg = getattr(cache_cfg, "cache_config", None)
    # enable_ssd = bool(getattr(cache_cfg, "enable_ssd", False))
    #
    # print("warmup")
    # # Prime targets into GPU + CPU + SSD. Timed requests append a new tail so
    # # offload cannot be fully eliminated by put_match.
    # for batch, user_ids, total_history_lengths in req_targets:
    #     model_predict.forward_with_kvcache(
    #         batch,
    #         user_ids,
    #         total_history_lengths,
    #     )
    # deadline = time.time() + offload_wait_timeout_s
    # torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all")
    # try:
    #     while len(kvcache_mgr.ongoing_offload_tasks) > 0:
    #         torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all.try_wait")
    #         try:
    #             kvcache_mgr.offload_try_wait()
    #         finally:
    #             torch.cuda.nvtx.range_pop()
    #         if len(kvcache_mgr.ongoing_offload_tasks) == 0:
    #             break
    #         if time.time() > deadline:
    #             raise TimeoutError(
    #                 f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
    #                 f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
    #             )
    #         time.sleep(0.001)
    # finally:
    #     torch.cuda.nvtx.range_pop()
    # num_cpu_blocks = int(getattr(cache_cfg, "num_cpu_blocks", 0))
    # num_ssd_blocks = int(getattr(cache_cfg, "num_ssd_blocks", 0))
    #
    # flexkv_client = getattr(host_mgr, "_client", None)
    # clear_cpu_cache = getattr(flexkv_client, "_clear_cpu_cache", None)
    # if not callable(clear_cpu_cache):
    #     raise RuntimeError("Scenario3 expects FlexKV client to expose _clear_cpu_cache().")
    #
    # print("timed run")
    # for iter_idx in range(timed_iters):
    #     # Clear local CPU cache explicitly before every measured target forward.
    #     # SSD entries should remain; proof is DISK2H+H2D (or DISK2D for GDS).
    #     print(f"[Scenario3] clear CPU cache before timed iter {iter_idx}", flush=True)
    #     clear_cpu_cache()
    #     user_id = timed_user_ids[iter_idx]
    #     user_ids = torch.tensor([user_id], dtype=torch.int64)
    #     kvcache_mgr.evict(user_ids, for_gpu=True)
    #
    #     batch, user_ids, total_history_lengths = build_request(
    #         user_id,
    #         history_len + append_history_len,
    #         num_candidates,
    #         max_seqlen,
    #     )
    #
    #     # timed run
    #     torch.cuda.nvtx.range_push(f"scenario3_timed_run_{iter_idx}")
    #     model_predict.forward_with_kvcache(
    #         batch,
    #         user_ids,
    #         total_history_lengths,
    #     )
    #     torch.cuda.nvtx.range_pop()
    # deadline = time.time() + offload_wait_timeout_s
    # torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all")
    # try:
    #     while len(kvcache_mgr.ongoing_offload_tasks) > 0:
    #         torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all.try_wait")
    #         try:
    #             kvcache_mgr.offload_try_wait()
    #         finally:
    #             torch.cuda.nvtx.range_pop()
    #         if len(kvcache_mgr.ongoing_offload_tasks) == 0:
    #             break
    #         if time.time() > deadline:
    #             raise TimeoutError(
    #                 f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
    #                 f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
    #             )
    #         time.sleep(0.001)
    # finally:
    #     torch.cuda.nvtx.range_pop()
    # print(f"[Scenario3] timed run completed, iters={timed_iters}")
    #
    # New pressure based version starts here. It avoids clear_cpu_cache() and
    # evicts CPU residency through FlexKV's normal cache path.

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
    enable_ssd = bool(getattr(cache_cfg, "enable_ssd", False))

    print("warmup")
    # Prime targets into GPU + CPU + SSD. Timed requests reuse this same prefix
    # so the measured range isolates onboard; --force-skip-offload additionally
    # removes the offload path itself.
    for batch, user_ids, total_history_lengths in req_targets:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all.try_wait")
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
        # Enough pressure users to exceed CPU cache after target priming, plus
        # a margin so the older target users are stable LRU eviction candidates.
        ssd_pressure_users = max(
            128,
            math.ceil(num_cpu_blocks / target_blocks) + timed_iters + 4,
        )
    pressure_base_user_id = base_user_id + 10000
    print(
        "[Scenario3 pressure] "
        f"num_cpu_blocks={num_cpu_blocks}, num_ssd_blocks={num_ssd_blocks}, "
        f"target_blocks={target_blocks}, pressure_users={ssd_pressure_users}, "
        f"pressure_batch_size={ssd_pressure_batch_size}, "
        f"pressure_batch_sleep_s={ssd_pressure_batch_sleep_s}",
        flush=True,
    )

    torch.cuda.nvtx.range_push("scenario3.pressure_fill")
    try:
        for pressure_idx in range(ssd_pressure_users):
            pressure_user_id = pressure_base_user_id + pressure_idx
            batch, user_ids, total_history_lengths = build_request(
                pressure_user_id,
                history_len,
                num_candidates,
                max_seqlen,
            )
            model_predict.forward_with_kvcache(
                batch,
                user_ids,
                total_history_lengths,
            )
            if (
                ssd_pressure_batch_size > 0
                and (pressure_idx + 1) % ssd_pressure_batch_size == 0
            ):
                deadline = time.time() + offload_wait_timeout_s
                torch.cuda.nvtx.range_push("scenario3.pressure.batch_offload_wait_all")
                try:
                    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
                        torch.cuda.nvtx.range_push(
                            "scenario3.pressure.batch_offload_wait_all.try_wait"
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
    torch.cuda.nvtx.range_push("scenario3.pressure.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.pressure.offload_wait_all.try_wait")
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
            history_len,
            num_candidates,
            max_seqlen,
        )

        # timed run
        torch.cuda.nvtx.range_push(f"scenario3_timed_run_{iter_idx}")
        forward_kwargs = {"skip_offload": True} if BENCHMARK_CONFIG.force_skip_offload else {}
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
            **forward_kwargs,
        )
        torch.cuda.nvtx.range_pop()
    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all")
    try:
        while len(kvcache_mgr.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all.try_wait")
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
    print(f"[Scenario3] timed run completed, iters={timed_iters}")


def shutdown_flexkv_client(model_predict) -> None:
    kvcache_mgr = getattr(getattr(model_predict, "dense_module", None), "kvcache", None)
    host_mgr = getattr(kvcache_mgr, "host_kvstorage_manager", None)
    client = getattr(host_mgr, "_client", None)
    if client is not None and hasattr(client, "shutdown"):
        try:
            client.shutdown()
            print("[Cleanup] FlexKV client shutdown completed", flush=True)
        except Exception as exc:
            print(f"[WARN] FlexKV client shutdown failed: {exc}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--append-history-len", type=int, default=None)
    parser.add_argument("--ssd-pressure-users", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-size", type=int, default=None)
    parser.add_argument("--ssd-pressure-batch-sleep-s", type=float, default=None)
    parser.add_argument("--flexkv-config-path", type=str, default=None)
    parser.add_argument("--scenarios", type=str, default="1,2,3")
    parser.add_argument("--disable-cudagraph", action="store_true")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--force-skip-offload", action="store_true")
    args, _ = parser.parse_known_args()

    cfg = BENCHMARK_CONFIG
    flexkv_config_path = args.flexkv_config_path or os.environ.get(
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
        raise ValueError("This restored benchmark currently supports only --ablation baseline.")
    BENCHMARK_CONFIG = cfg

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    scenarios = {scenario.strip() for scenario in args.scenarios.split(",") if scenario.strip()}
    print("[Mode] only_onboard=True, timed forwards use history_len")
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
                run_scenario_gpu_miss_host_hit(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    warmup_iters=cfg.warmup_iters,
                    timed_iters=cfg.timed_iters,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
            if "3" in scenarios:
                run_scenario_gpu_cpu_miss_ssd_hit(
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
