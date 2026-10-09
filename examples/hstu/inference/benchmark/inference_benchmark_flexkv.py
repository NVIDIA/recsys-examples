#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceEmbeddingConfig, RankingConfig, get_inference_hstu_config
from recsys_kvcache_manager.kvcache_config import get_kvcache_config
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

sys.path.append("./model/")
from inference_ranking_gr import get_inference_ranking_gr

ITEM_FEATURE_NAME = "item_feat"
ACTION_FEATURE_NAME = "act_feat"
ITEM_VOCAB_SIZE = 10000
ACTION_VOCAB_SIZE = 128
SSD_PRESSURE_USERS = 0
SSD_PRESSURE_BATCH_SIZE = 8
SSD_PRESSURE_BATCH_SLEEP_S = 1.0
SUPPORTED_SCENARIOS = frozenset({"gpu_hit", "cpu_hit", "ssd_hit"})


InferenceRequest = Tuple[HSTUBatch, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class BenchmarkConfig:
    history_len: int = 1024
    append_history_len: int = 1024
    num_candidates: int = 256
    warmup_iters: int = 1
    timed_iters: int = 10
    batch_size: int = 1
    seed: int = 20260624
    max_batch_size: int = 16
    disable_cudagraph: bool = False
    flexkv_config_path: str = ""
    flexkv_num_cpu_blocks: int = 4096
    flexkv_num_local_blocks: int = 4096
    offload_wait_timeout_s: float = 60.0
    only_onboard: bool = False


BENCHMARK_CONFIG = BenchmarkConfig()


def parse_scenarios(scenarios_arg: str) -> set[str]:
    scenarios = set()
    for scenario in scenarios_arg.split(","):
        scenario = scenario.strip()
        if not scenario:
            continue
        if scenario not in SUPPORTED_SCENARIOS:
            raise ValueError(
                f"Unsupported scenario '{scenario}'. " "Use gpu_hit,cpu_hit,ssd_hit."
            )
        scenarios.add(scenario)
    return scenarios


def get_timed_history_len(history_len: int, append_history_len: int) -> int:
    if BENCHMARK_CONFIG.only_onboard:
        return history_len
    return history_len + append_history_len


def build_user_batches(
    base_user_id: int,
    num_batches: int,
    batch_size: int,
) -> list[list[int]]:
    return [
        [
            base_user_id + batch_idx * batch_size + batch_offset
            for batch_offset in range(batch_size)
        ]
        for batch_idx in range(num_batches)
    ]


def build_request(
    user_id: int | Sequence[int],
    history_len: int,
    num_candidates: int,
    max_seqlen: int,
) -> InferenceRequest:
    user_ids = [user_id] if isinstance(user_id, int) else list(user_id)
    batch_size = len(user_ids)
    if batch_size <= 0:
        raise ValueError("build_request expects at least one user id.")
    max_history_per_feature, remainder = divmod(
        max_seqlen - num_candidates,
        2,
    )
    if max_history_per_feature < history_len or remainder != 0:
        raise ValueError(
            "max_seqlen must equal 2 * max_history_per_feature + "
            f"num_candidates; got max_seqlen={max_seqlen}, "
            f"history_len={history_len}, num_candidates={num_candidates}"
        )

    item_seqs = [
        torch.randint(
            low=0,
            high=ITEM_VOCAB_SIZE,
            size=(history_len + num_candidates,),
            dtype=torch.long,
        )
        for _ in user_ids
    ]
    action_seqs = [
        torch.randint(
            low=0,
            high=ACTION_VOCAB_SIZE,
            size=(history_len + num_candidates,),
            dtype=torch.long,
        )
        for _ in user_ids
    ]
    features = KeyedJaggedTensor.from_jt_dict(
        {
            ITEM_FEATURE_NAME: JaggedTensor.from_dense(item_seqs),
            ACTION_FEATURE_NAME: JaggedTensor.from_dense(action_seqs),
        }
    )
    batch = HSTUBatch(
        features=features,
        batch_size=batch_size,
        feature_to_max_seqlen={
            ITEM_FEATURE_NAME: max_history_per_feature + num_candidates,
            ACTION_FEATURE_NAME: max_history_per_feature + num_candidates,
        },
        contextual_feature_names=[],
        item_feature_name=ITEM_FEATURE_NAME,
        action_feature_name=ACTION_FEATURE_NAME,
        max_num_candidates=num_candidates,
        num_candidates=torch.full((batch_size,), num_candidates, dtype=torch.long),
    ).to(device=torch.cuda.current_device())
    return (
        batch,
        torch.tensor(user_ids, dtype=torch.int64),
        torch.full((batch_size,), history_len * 2, dtype=torch.int32),
    )


def build_model(cfg: BenchmarkConfig, history_len: int, layerwise: bool = False):
    max_num_history = max(2048, history_len + cfg.append_history_len)
    max_num_candidates = cfg.num_candidates
    max_seqlen = max_num_history * 2 + max_num_candidates

    hidden_dim_size = 1024
    num_heads = 4
    head_dim = 256
    num_layers = 8
    inference_dtype = torch.bfloat16
    hstu_cudagraph_configs = {
        "batch_size": sorted({1, 2, 4, 8, cfg.batch_size}),
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
    # The BF16 HSTU paged-attention kernel supports 32/64-token pages. A
    # 128-token page silently selects the non-paged fallback in the Ampere
    # family kernel and causes out-of-bounds reads once batch_size > 1.
    page_size = 64 if sm_major >= 10 else 32
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
        "flexkv_as_batch": int(os.environ.get("RECSYS_FLEXKV_AS_BATCH", "1")),
        "flexkv_enable_layerwise": layerwise,
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


def wait_offload_queue(kvcache_mgr, range_name: str, timeout_s: float) -> None:
    offload_task_owner = getattr(kvcache_mgr, "backend", kvcache_mgr)
    deadline = time.time() + timeout_s
    torch.cuda.nvtx.range_push(range_name)
    try:
        while len(offload_task_owner.ongoing_offload_tasks) > 0:
            kvcache_mgr.offload_try_wait()
            if len(offload_task_owner.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({timeout_s}s), "
                    f"pending={len(offload_task_owner.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()


def forward_with_kvcache_layerwise(
    model_predict,
    batch,
    user_ids: torch.Tensor,
    total_history_lengths: torch.Tensor,
    skip_offload: bool = False,
):
    """Run a full forward with layerwise KV onboard enabled."""
    dense_module = model_predict.dense_module
    kvcache_mgr = dense_module.kvcache

    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
        user_ids,
        total_history_lengths,
    )
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    torch.cuda.nvtx.range_push("recsys.layerwise.onboard_launch")
    try:
        kvcache_mgr.onboard_launch(
            index_meta,
            lookup_res,
            kvcache_metadata,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    stripped_batch = model_predict.strip_cached_tokens(
        batch,
        lookup_res.cached_lengths,
    )
    torch.cuda.nvtx.range_push("HSTU embedding")
    try:
        embeddings = model_predict.sparse_module(stripped_batch.features)
    finally:
        torch.cuda.nvtx.range_pop()

    return dense_module.forward_with_kvcache(
        stripped_batch,
        embeddings,
        user_ids,
        total_history_lengths,
        (index_meta, lookup_res, kvcache_metadata),
        skip_offload=skip_offload,
    )


def run_scenario_cpu_hit_layerwise(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    timed_iters: int,
    batch_size: int,
    offload_wait_timeout_s: float,
) -> None:
    timed_user_ids = build_user_batches(20, 1, batch_size)[0]
    prefix_request = build_request(
        timed_user_ids,
        history_len,
        num_candidates,
        max_seqlen,
    )
    timed_requests = [
        build_request(
            timed_user_ids,
            history_len + append_history_len,
            num_candidates,
            max_seqlen,
        )
        for _ in range(timed_iters)
    ]
    kvcache_mgr = model_predict.dense_module.kvcache

    print("warmup")
    forward_with_kvcache_layerwise(
        model_predict,
        *prefix_request,
        skip_offload=False,
    )
    wait_offload_queue(
        kvcache_mgr,
        "scenario2.layerwise.warmup.offload_wait_all",
        offload_wait_timeout_s,
    )

    _, user_ids, _ = prefix_request
    kvcache_mgr.evict(user_ids, for_gpu=True)
    warmup_request = build_request(
        timed_user_ids,
        history_len + append_history_len,
        num_candidates,
        max_seqlen,
    )
    forward_with_kvcache_layerwise(
        model_predict,
        *warmup_request,
        skip_offload=True,
    )
    torch.cuda.synchronize()
    kvcache_mgr.evict(user_ids, for_gpu=True)

    print("timed run")
    wall_ms = []
    for iter_idx, request in enumerate(timed_requests):
        _, request_user_ids, _ = request
        kvcache_mgr.evict(request_user_ids, for_gpu=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(
            f"scenario2_layerwise_timed_run_{iter_idx}"
        )
        try:
            forward_with_kvcache_layerwise(
                model_predict,
                *request,
                skip_offload=True,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wall_ms.append(elapsed_ms)
        print(
            f"[latency_sample] scenario=cpu_hit_layerwise "
            f"batch_size={batch_size} iter={iter_idx} ms={elapsed_ms:.4f}",
            flush=True,
        )
    print(f"[CPU-hit layerwise] timed run completed, iters={timed_iters}")
    print(
        f"[latency] scenario=cpu_hit_layerwise batch_size={batch_size} "
        f"avg_ms={sum(wall_ms) / len(wall_ms):.4f} "
        f"min_ms={min(wall_ms):.4f} max_ms={max(wall_ms):.4f}",
        flush=True,
    )


def run_scenario_gpu_hit(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
    timed_iters: int,
    batch_size: int,
    offload_wait_timeout_s: float,
) -> None:
    base_user_id = 10
    warmup_user_ids = build_user_batches(base_user_id, 1, batch_size)[0]
    timed_user_batches = build_user_batches(
        base_user_id + batch_size,
        timed_iters,
        batch_size,
    )
    kvcache_mgr = model_predict.dense_module.kvcache

    # warmup
    print("warmup")
    model_predict.forward_with_kvcache(
        *build_request(warmup_user_ids, history_len, num_candidates, max_seqlen),
        skip_offload=True,
    )
    kvcache_mgr.evict(
        torch.tensor(warmup_user_ids, dtype=torch.int64),
        for_gpu=True,
    )

    # timed run
    print("timed run")
    wall_ms = []
    for iter_idx, timed_user_ids in enumerate(timed_user_batches):
        timed_user_ids_tensor = torch.tensor(timed_user_ids, dtype=torch.int64)
        model_predict.forward_with_kvcache(
            *build_request(
                timed_user_ids,
                history_len,
                num_candidates,
                max_seqlen,
            ),
            skip_offload=True,
        )
        lookup_res = kvcache_mgr.gpu_kvcache_mgr.lookup(timed_user_ids_tensor)
        gpu_lengths = lookup_res.gpu_cached_lengths.cpu()
        expected = history_len * 2
        if any(int(length.item()) != expected for length in gpu_lengths):
            raise RuntimeError(
                f"Scenario1 expects GPU prefix hit ({expected}), "
                f"got {gpu_lengths.tolist()}"
            )
        if iter_idx == 0:
            print(f"[Scenario1 precheck] gpu={gpu_lengths.tolist()}")
        request = build_request(
            timed_user_ids,
            get_timed_history_len(history_len, append_history_len),
            num_candidates,
            max_seqlen,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(f"scenario1_timed_run_{iter_idx}")
        try:
            model_predict.forward_with_kvcache(*request, skip_offload=True)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wall_ms.append(elapsed_ms)
        print(
            f"[latency_sample] scenario=gpu_hit batch_size={batch_size} "
            f"iter={iter_idx} ms={elapsed_ms:.4f}",
            flush=True,
        )
        kvcache_mgr.evict(timed_user_ids_tensor, for_gpu=True)
    print(f"[Scenario1] timed run completed, iters={timed_iters}")
    print(
        f"[latency] scenario=gpu_hit batch_size={batch_size} "
        f"avg_ms={sum(wall_ms) / len(wall_ms):.4f} "
        f"min_ms={min(wall_ms):.4f} max_ms={max(wall_ms):.4f}",
        flush=True,
    )


def run_scenario_cpu_hit(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    warmup_iters: int,
    timed_iters: int,
    batch_size: int,
    offload_wait_timeout_s: float,
) -> None:
    base_user_id = 20
    timed_user_ids = build_user_batches(base_user_id, 1, batch_size)[0]
    timed_user_ids_tensor = torch.tensor(timed_user_ids, dtype=torch.int64)
    kvcache_mgr = model_predict.dense_module.kvcache

    # warmup
    print("warmup")
    model_predict.forward_with_kvcache(
        *build_request(timed_user_ids, history_len, num_candidates, max_seqlen)
    )
    wait_offload_queue(
        kvcache_mgr,
        "scenario2.warmup.offload_wait_all",
        offload_wait_timeout_s,
    )

    # timed run
    print("timed run")
    wall_ms = []
    for iter_idx in range(timed_iters):
        # Each forward onboards the KV back to GPU. Evict before every timed
        # iteration so the measured prefix stays CPU-hit instead of becoming GPU-hit.
        kvcache_mgr.evict(timed_user_ids_tensor, for_gpu=True)
        request = build_request(
            timed_user_ids,
            get_timed_history_len(history_len, append_history_len),
            num_candidates,
            max_seqlen,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.cuda.nvtx.range_push(f"scenario2_timed_run_{iter_idx}")
        try:
            model_predict.forward_with_kvcache(*request, skip_offload=True)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        wall_ms.append(elapsed_ms)
        print(
            f"[latency_sample] scenario=cpu_hit batch_size={batch_size} "
            f"iter={iter_idx} ms={elapsed_ms:.4f}",
            flush=True,
        )
    print(f"[Scenario2] timed run completed, iters={timed_iters}")
    print(
        f"[latency] scenario=cpu_hit batch_size={batch_size} "
        f"avg_ms={sum(wall_ms) / len(wall_ms):.4f} "
        f"min_ms={min(wall_ms):.4f} max_ms={max(wall_ms):.4f}",
        flush=True,
    )


def run_scenario_ssd_hit(
    model_predict,
    history_len: int,
    append_history_len: int,
    num_candidates: int,
    max_seqlen: int,
    page_size: int,
    timed_iters: int,
    batch_size: int,
    offload_wait_timeout_s: float,
) -> None:
    ssd_pressure_users = SSD_PRESSURE_USERS
    ssd_pressure_batch_size = SSD_PRESSURE_BATCH_SIZE
    ssd_pressure_batch_sleep_s = SSD_PRESSURE_BATCH_SLEEP_S
    base_user_id = 30
    timed_user_batches = build_user_batches(base_user_id, timed_iters, batch_size)
    req_targets = [
        build_request(user_ids, history_len, num_candidates, max_seqlen)
        for user_ids in timed_user_batches
    ]
    kvcache_mgr = model_predict.dense_module.kvcache
    host_mgr = kvcache_mgr.host_kvstorage_manager
    cache_cfg = getattr(host_mgr, "_client", None)
    cache_cfg = getattr(cache_cfg, "cache_config", None)

    print("warmup")
    # Prime targets into GPU + CPU + SSD. Timed requests append a new tail so
    # offload cannot be fully eliminated by put_match.
    for batch, user_ids, total_history_lengths in req_targets:
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
    deadline = time.time() + offload_wait_timeout_s
    offload_task_owner = getattr(kvcache_mgr, "backend", kvcache_mgr)
    torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all")
    try:
        while len(offload_task_owner.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.warmup.offload_wait_all.try_wait")
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(offload_task_owner.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(offload_task_owner.ongoing_offload_tasks)}"
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
                    while len(offload_task_owner.ongoing_offload_tasks) > 0:
                        torch.cuda.nvtx.range_push(
                            "scenario3.pressure.batch_offload_wait_all.try_wait"
                        )
                        try:
                            kvcache_mgr.offload_try_wait()
                        finally:
                            torch.cuda.nvtx.range_pop()
                        if len(offload_task_owner.ongoing_offload_tasks) == 0:
                            break
                        if time.time() > deadline:
                            raise TimeoutError(
                                f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                                f"pending={len(offload_task_owner.ongoing_offload_tasks)}"
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
        while len(offload_task_owner.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.pressure.offload_wait_all.try_wait")
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(offload_task_owner.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(offload_task_owner.ongoing_offload_tasks)}"
                )
            time.sleep(0.001)
    finally:
        torch.cuda.nvtx.range_pop()
    if ssd_pressure_batch_sleep_s > 0:
        time.sleep(ssd_pressure_batch_sleep_s)

    print("timed run")
    for iter_idx, user_batch in enumerate(timed_user_batches):
        user_ids = torch.tensor(user_batch, dtype=torch.int64)
        kvcache_mgr.evict(user_ids, for_gpu=True)

        batch, user_ids, total_history_lengths = build_request(
            user_batch,
            get_timed_history_len(history_len, append_history_len),
            num_candidates,
            max_seqlen,
        )

        # timed run
        torch.cuda.nvtx.range_push(f"scenario3_timed_run_{iter_idx}")
        model_predict.forward_with_kvcache(
            batch,
            user_ids,
            total_history_lengths,
        )
        torch.cuda.nvtx.range_pop()
    deadline = time.time() + offload_wait_timeout_s
    torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all")
    try:
        while len(offload_task_owner.ongoing_offload_tasks) > 0:
            torch.cuda.nvtx.range_push("scenario3.timed.offload_wait_all.try_wait")
            try:
                kvcache_mgr.offload_try_wait()
            finally:
                torch.cuda.nvtx.range_pop()
            if len(offload_task_owner.ongoing_offload_tasks) == 0:
                break
            if time.time() > deadline:
                raise TimeoutError(
                    f"offload queue not drained within timeout ({offload_wait_timeout_s}s), "
                    f"pending={len(offload_task_owner.ongoing_offload_tasks)}"
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--append-history-len", type=int, default=None)
    parser.add_argument("--flexkv-config-path", type=str, default=None)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--only-onboard",
        action="store_true",
        help="Measure only onboard by reusing the warmed prefix in timed forwards.",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Enable layerwise KV onboard for the CPU-hit full-forward benchmark.",
    )
    parser.add_argument("--disable-cudagraph", action="store_true")
    parser.add_argument("--ablation", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = BENCHMARK_CONFIG
    flexkv_config_path = args.flexkv_config_path or os.environ.get(
        "RECSYS_FLEXKV_CONFIG_PATH"
    )
    if flexkv_config_path:
        cfg = replace(cfg, flexkv_config_path=flexkv_config_path)
    if args.timed_iters is not None:
        cfg = replace(cfg, timed_iters=args.timed_iters)
    if args.batch_size is not None:
        cfg = replace(cfg, batch_size=args.batch_size)
    if args.history_len is not None:
        cfg = replace(cfg, history_len=args.history_len)
    if args.append_history_len is not None:
        cfg = replace(cfg, append_history_len=args.append_history_len)
    if args.disable_cudagraph:
        cfg = replace(cfg, disable_cudagraph=True)
    if args.only_onboard:
        cfg = replace(cfg, only_onboard=True)
    if args.ablation not in (None, "baseline"):
        raise ValueError(
            "This restored benchmark currently supports only --ablation baseline."
        )
    if cfg.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {cfg.batch_size}.")
    if cfg.batch_size > cfg.max_batch_size:
        raise ValueError(
            f"--batch-size ({cfg.batch_size}) cannot exceed max_batch_size "
            f"({cfg.max_batch_size})."
        )
    BENCHMARK_CONFIG = cfg

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    history_len = cfg.history_len
    scenarios_arg = args.scenarios or (
        "cpu_hit" if args.layerwise else "gpu_hit,cpu_hit,ssd_hit"
    )
    scenarios = parse_scenarios(scenarios_arg)
    if args.layerwise and scenarios != {"cpu_hit"}:
        raise ValueError("--layerwise can only be used with --scenarios cpu_hit.")
    mode = (
        "layerwise_only_onboard"
        if args.layerwise
        else "non_layerwise_only_onboard"
        if cfg.only_onboard
        else "with_offload"
    )
    print(
        f"[Config] history_len={history_len}, append_history_len={cfg.append_history_len}, "
        f"num_candidates={cfg.num_candidates}, batch_size={cfg.batch_size}, "
        f"disable_cudagraph={cfg.disable_cudagraph}, "
        f"mode={mode}, "
        f"scenarios={','.join(sorted(scenarios))}"
    )
    model_predict, page_size, max_seqlen = build_model(
        cfg,
        history_len,
        layerwise=args.layerwise,
    )
    print(f"[Config] page_size={page_size}, max_seqlen={max_seqlen}")

    try:
        with torch.inference_mode():
            if "gpu_hit" in scenarios:
                run_scenario_gpu_hit(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    warmup_iters=cfg.warmup_iters,
                    timed_iters=cfg.timed_iters,
                    batch_size=cfg.batch_size,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
            if "cpu_hit" in scenarios:
                cpu_hit_runner = (
                    run_scenario_cpu_hit_layerwise
                    if args.layerwise
                    else run_scenario_cpu_hit
                )
                cpu_hit_kwargs = dict(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    timed_iters=cfg.timed_iters,
                    batch_size=cfg.batch_size,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
                if not args.layerwise:
                    cpu_hit_kwargs["warmup_iters"] = cfg.warmup_iters
                cpu_hit_runner(**cpu_hit_kwargs)
            if "ssd_hit" in scenarios:
                run_scenario_ssd_hit(
                    model_predict=model_predict,
                    history_len=history_len,
                    append_history_len=cfg.append_history_len,
                    num_candidates=cfg.num_candidates,
                    max_seqlen=max_seqlen,
                    page_size=page_size,
                    timed_iters=cfg.timed_iters,
                    batch_size=cfg.batch_size,
                    offload_wait_timeout_s=cfg.offload_wait_timeout_s,
                )
    finally:
        shutdown_flexkv_client(model_predict)
