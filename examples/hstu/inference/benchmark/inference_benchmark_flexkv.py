#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import math
import random
import time
from dataclasses import dataclass
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
    num_candidates: int = 256
    warmup_iters: int = 10
    timed_iters: int = 50
    seed: int = 20260624
    max_batch_size: int = 16
    disable_cudagraph: bool = False
    flexkv_config_path: str = ""
    flexkv_num_cpu_blocks: int = 4096
    flexkv_num_local_blocks: int = 4096
    ssd_pressure_users: int = 0
    offload_wait_timeout_s: float = 60.0


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
    max_num_history = max(2048, history_len)
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
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
) -> None:
    user_id = 10
    prepare_iters = max(1, 1 + warmup_iters)
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for _ in range(prepare_iters)
    ]
    req_timed = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for _ in range(timed_iters)
    ]

    # warmup
    _, prime_user_ids, prime_total_history_lengths = req_prepare[0]
    for batch, user_ids, total_history_lengths in req_prepare:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )

    # precheck
    _, lookup_res = model_predict.dense_module.kvcache.lookup_kvcache(
        prime_user_ids,
        prime_total_history_lengths,
    )
    lengths = {
        "cached": lookup_res.cached_lengths.cpu(),
        "gpu": lookup_res.gpu_cached_lengths.cpu(),
        "host": lookup_res.host_cached_lengths.cpu(),
    }
    print(
        f"[Scenario1 precheck] cached={lengths['cached'].tolist()} "
        f"gpu={lengths['gpu'].tolist()} host={lengths['host'].tolist()}"
    )
    expected = int(prime_total_history_lengths[0].item())
    if int(lengths["gpu"][0].item()) != expected:
        raise RuntimeError(
            f"Scenario1 expects GPU 100% hit ({expected}), got {int(lengths['gpu'][0].item())}"
        )

    # timed run
    torch.cuda.nvtx.range_push("scenario1_timed_run")
    for batch, user_ids, total_history_lengths in req_timed:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    torch.cuda.nvtx.range_pop()
    print(f"[Scenario1] timed run completed, iters={timed_iters}")


def run_scenario_gpu_miss_host_hit(
    model_predict,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
) -> None:
    user_id = 20
    prepare_iters = max(1, 1 + warmup_iters)
    req_prepare = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for _ in range(prepare_iters)
    ]
    req_timed = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for _ in range(timed_iters)
    ]
    kvcache_mgr = model_predict.dense_module.kvcache

    # warmup
    _, prime_user_ids, prime_total_history_lengths = req_prepare[0]
    for batch, user_ids, total_history_lengths in req_prepare:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    deadline = time.time() + offload_wait_timeout_s
    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
        kvcache_mgr.offload_try_wait()
        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
            )
        time.sleep(0.002)

    # evict GPU
    kvcache_mgr.evict(prime_user_ids, for_gpu=True)
    
    # precheck
    _, lookup_res = model_predict.dense_module.kvcache.lookup_kvcache(
        prime_user_ids,
        prime_total_history_lengths,
    )
    lengths = {
        "cached": lookup_res.cached_lengths.cpu(),
        "gpu": lookup_res.gpu_cached_lengths.cpu(),
        "host": lookup_res.host_cached_lengths.cpu(),
    }
    print(
        f"[Scenario2 precheck] cached={lengths['cached'].tolist()} "
        f"gpu={lengths['gpu'].tolist()} host={lengths['host'].tolist()}"
    )
    if int(lengths["gpu"][0].item()) != 0:
        raise RuntimeError("Scenario2 expects GPU 100% miss after evict(for_gpu=True).")
    if int(lengths["host"][0].item()) <= 0:
        raise RuntimeError("Scenario2 expects host hit, but host_cached_lengths is zero.")

    # timed run
    torch.cuda.nvtx.range_push("scenario2_timed_run")
    for batch, user_ids, total_history_lengths in req_timed:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    torch.cuda.nvtx.range_pop()
    print(f"[Scenario2] timed run completed, iters={timed_iters}")


def run_scenario_gpu_cpu_miss_ssd_hit_best_effort(
    model_predict,
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
    page_size: int,
    timed_iters: int,
    offload_wait_timeout_s: float,
    ssd_pressure_users: int,
) -> None:
    user_id = 30
    req_target = build_request(user_id, history_len, num_candidates, max_seqlen)
    _, target_user_ids, target_total_history_lengths = req_target
    kvcache_mgr = model_predict.dense_module.kvcache
    host_mgr = kvcache_mgr.host_kvstorage_manager
    cache_cfg = getattr(host_mgr, "_client", None)
    cache_cfg = getattr(cache_cfg, "cache_config", None)
    enable_ssd = bool(getattr(cache_cfg, "enable_ssd", False))

    # warmup
    for batch, user_ids, total_history_lengths in [req_target]:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    deadline = time.time() + offload_wait_timeout_s
    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
        kvcache_mgr.offload_try_wait()
        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
            )
        time.sleep(0.002)

    pages_per_user = math.ceil(int(target_total_history_lengths[0].item()) / page_size)
    num_cpu_blocks = int(getattr(cache_cfg, "num_cpu_blocks", 0))
    if ssd_pressure_users > 0:
        pressure_users = ssd_pressure_users
    else:
        pressure_users = max(8, (num_cpu_blocks // max(pages_per_user, 1)) + 8)
        pressure_users = min(pressure_users, 1024)

    for uid in range(1000, 1000 + pressure_users):
        req_fill = build_request(uid, history_len, num_candidates, max_seqlen)
        for batch, user_ids, total_history_lengths in [req_fill]:
            model_predict.forward_with_kvcache(
                batch,
                user_ids,
                total_history_lengths,
            )
    deadline = time.time() + offload_wait_timeout_s
    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
        kvcache_mgr.offload_try_wait()
        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
            )
        time.sleep(0.002)

    # evict GPU
    kvcache_mgr.evict(target_user_ids, for_gpu=True)

    # precheck  
    _, lookup_res = model_predict.dense_module.kvcache.lookup_kvcache(
        target_user_ids,
        target_total_history_lengths,
    )
    lengths = {
        "cached": lookup_res.cached_lengths.cpu(),
        "gpu": lookup_res.gpu_cached_lengths.cpu(),
        "host": lookup_res.host_cached_lengths.cpu(),
    }
    print(
        f"[Scenario3 precheck] cached={lengths['cached'].tolist()} "
        f"gpu={lengths['gpu'].tolist()} host={lengths['host'].tolist()}"
    )
    if int(lengths["gpu"][0].item()) != 0:
        raise RuntimeError("Scenario3 expects GPU miss after GPU evict.")
    if int(lengths["host"][0].item()) <= 0:
        print(
            "[Scenario3] host miss observed. This means no host-tier hit can be validated "
            "for current pressure/settings."
        )
        return

    req_timed = [
        build_request(user_id, history_len, num_candidates, max_seqlen)
        for _ in range(timed_iters)
    ]
    torch.cuda.nvtx.range_push("scenario3_timed_run")
    for batch, user_ids, total_history_lengths in req_timed:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    torch.cuda.nvtx.range_pop()
    deadline = time.time() + offload_wait_timeout_s
    while len(kvcache_mgr.ongoing_offload_tasks) > 0:
        kvcache_mgr.offload_try_wait()
        if len(kvcache_mgr.ongoing_offload_tasks) == 0:
            break
        if time.time() > deadline:
            raise TimeoutError(
                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                f"pending={len(kvcache_mgr.ongoing_offload_tasks)}"
            )
        time.sleep(0.002)
    print(f"[Scenario3] timed run completed, iters={timed_iters}")
    print(
        "[Scenario3] note: current Recsys FlexKV API does not expose CPU-tier vs SSD-tier "
        "hit counters; this case is best-effort rather than strict 100% proof."
    )


def main() -> None:
    cfg = BENCHMARK_CONFIG
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    if history_len <= 0:
        raise ValueError("Hardcoded history_len must be > 0.")
    print(f"[Config] history_len={history_len}, num_candidates={cfg.num_candidates}")
    model_predict, page_size, max_seqlen = build_model(cfg, history_len)
    print(f"[Config] page_size={page_size}, max_seqlen={max_seqlen}")

    with torch.inference_mode():
        run_scenario_gpu_hit(
            model_predict=model_predict,
            history_len=history_len,
            num_candidates=cfg.num_candidates,
            max_seqlen=max_seqlen,
            warmup_iters=cfg.warmup_iters,
            timed_iters=cfg.timed_iters,
            offload_wait_timeout_s=cfg.offload_wait_timeout_s,
        )
        run_scenario_gpu_miss_host_hit(
            model_predict=model_predict,
            history_len=history_len,
            num_candidates=cfg.num_candidates,
            max_seqlen=max_seqlen,
            warmup_iters=cfg.warmup_iters,
            timed_iters=cfg.timed_iters,
            offload_wait_timeout_s=cfg.offload_wait_timeout_s,
        )
        run_scenario_gpu_cpu_miss_ssd_hit_best_effort(
            model_predict=model_predict,
            history_len=history_len,
            num_candidates=cfg.num_candidates,
            max_seqlen=max_seqlen,
            page_size=page_size,
            timed_iters=cfg.timed_iters,
            offload_wait_timeout_s=cfg.offload_wait_timeout_s,
            ssd_pressure_users=cfg.ssd_pressure_users,
        )


if __name__ == "__main__":
    main()
