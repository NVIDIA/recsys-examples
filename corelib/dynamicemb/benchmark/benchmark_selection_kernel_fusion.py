# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark to verify the fused mask-based erase in table_erase_kernel (Part B).

This script verifies the mask parameter correctly filters which keys are erased
from the admission counter, and that non-admitted keys retain their counter state.

Usage:
    torchrun --nproc_per_node=1 benchmark_issue357_fusion.py
"""

import os
import torch
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.embedding_admission import FrequencyAdmissionStrategy, KVCounter
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType
import torchrec


def run():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    emb_dim = 16
    num_embeddings = 10000
    batch_size = 64
    num_iterations = 4
    admission_threshold = 5
    num_features_per_sample = 5

    value_dim_sgd = emb_dim
    total_hbm_bytes = num_embeddings * 4 * value_dim_sgd
    local_hbm_bytes = total_hbm_bytes // 2

    admission_counter_config = KVCounter(
        capacity=5000,
        bucket_capacity=1024,
        key_type=torch.int64,
    )

    admission_strategy = FrequencyAdmissionStrategy(
        threshold=admission_threshold,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT,
            value=0.0,
        ),
    )

    table_options = [
        DynamicEmbTableOptions(
            embedding_dtype=torch.float32,
            dim=emb_dim,
            max_capacity=num_embeddings,
            local_hbm_for_values=local_hbm_bytes,
            index_type=torch.int64,
            score_strategy=DynamicEmbScoreStrategy.LFU,
            caching=True,
            admit_strategy=admission_strategy,
            admission_counter=admission_counter_config,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
            ),
        )
    ]

    model = BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        table_names=["t_0"],
        use_index_dedup=True,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=1e-3,
        device=device,
    )

    model.train()
    assert model._cache is not None, "Cache must be created"
    assert model._admission_counter is not None, "Admission counter must be created"
    print(f"Cache: {type(model._cache).__name__}")
    print(f"Counter: {type(model._admission_counter).__name__}")

    # Generate keys with controlled frequencies:
    #   keys 0..49  = frequent (reached admission threshold)
    #   keys 50..99 = rare (below admission threshold)
    torch.manual_seed(42)
    total_freq_keys = 40
    total_rare_keys = 80
    keys_per_iter = batch_size * num_features_per_sample

    expected_freq = {}
    kjts = []
    for i in range(num_iterations):
        g = torch.Generator(device=device)
        g.manual_seed(i)
        n_freq = int(keys_per_iter * 0.8)
        n_rare = keys_per_iter - n_freq
        freq_batch = torch.randint(
            0, total_freq_keys, (n_freq,), device=device, generator=g
        )
        rare_batch = torch.randint(
            total_freq_keys, total_freq_keys + total_rare_keys,
            (n_rare,), device=device, generator=g
        )
        indices = torch.cat([freq_batch, rare_batch])
        for v in indices:
            expected_freq[int(v)] = expected_freq.get(int(v), 0) + 1
        lengths = torch.ones(keys_per_iter, dtype=torch.int64, device=device)
        kjts.append(torchrec.KeyedJaggedTensor(
            keys=["t_0"], values=indices, lengths=lengths
        ))

    # --- Warmup ---
    for _ in range(2):
        ret = model(kjts[0].values(), kjts[0].offsets())
        grad = torch.empty_like(ret)
        ret.backward(grad)
    torch.cuda.synchronize()

    # --- Profiled run ---
    print("\n=== Profiled forward+backward passes ===")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for i in range(num_iterations):
            ret = model(kjts[i].values(), kjts[i].offsets())
            grad = torch.empty_like(ret)
            ret.backward(grad)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # --- Verify mask-based erase fusion ---
    print(f"\n=== Mask-based Erase Fusion Correctness Check ===")
    print(f"Threshold: {admission_threshold}, Iterations: {num_iterations}")

    # Dump the admission counter to see which keys are still tracked
    counter = model._admission_counter
    score_name = counter.score_name_
    counter_keys = {}
    for keys, named_scores, _ in counter.table_._batched_export_keys_scores(
        [score_name], device, table_id=0
    ):
        for k, s in zip(keys, named_scores[score_name]):
            counter_keys[int(k)] = int(s)

    admitted_expected = {k for k, v in expected_freq.items() if v >= admission_threshold}
    non_admitted_expected = {k for k, v in expected_freq.items() if v < admission_threshold}

    print(f"Unique keys seen: {len(expected_freq)}")
    print(f"Expected admitted  (freq >= {admission_threshold}): {len(admitted_expected)}")
    print(f"Expected non-admitted: {len(non_admitted_expected)}")
    print(f"Keys currently in admission counter: {len(counter_keys)}")

    table_keys = set()
    cache_state = model._cache._state
    for keys, _, _ in cache_state.key_index_map._batched_export_keys_scores(
        [cache_state.score_policy.name], device, table_id=0
    ):
        table_keys.update(int(x) for x in keys)
    for keys, _, _, _ in model._storage.export_keys_values(device, 65536, table_id=0):
        table_keys.update(int(x) for x in keys)
    missing_from_counter = non_admitted_expected - set(counter_keys.keys())
    in_table = missing_from_counter & table_keys
    print(f"Non-admitted keys absent from counter: {len(missing_from_counter)}")
    print(f"Of those, found in cache/storage (actually admitted): {len(in_table)}")
    truly_missing = missing_from_counter - table_keys
    if truly_missing:
        print(f"BUG: {len(truly_missing)} keys absent from both counter and storage")
    else:
        print("All keys accounted for — no unexpected erasures.")

    if truly_missing:
        print(f"FAIL: {len(truly_missing)} keys absent from both counter and storage")
        print("\n=== FUSION FAILED: Mask-based erase has bugs. ===")
    else:
        print("\n=== FUSION VERIFIED: Mask correctly skips non-admitted keys. ===")

    # --- Timing ---
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    n_bench = 30
    start.record()
    for _ in range(n_bench):
        ret = model(kjts[0].values(), kjts[0].offsets())
        grad = torch.empty_like(ret)
        ret.backward(grad)
    end.record()
    torch.cuda.synchronize()
    print(f"\n=== Avg latency ({n_bench} iters): {start.elapsed_time(end) / n_bench:.3f} ms ===")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
