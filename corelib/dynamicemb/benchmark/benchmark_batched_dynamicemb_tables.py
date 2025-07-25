# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os

import numpy as np
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
    DynamicEmbScoreStrategy
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
from torch.distributed.elastic.multiprocessing.errors import record
from benchmark_utils import GPUTimer

from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    EmbeddingLocation,
    PoolingMode,
    RecordCacheMetrics,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType
from fbgemm_gpu.runtime_monitor import (
    AsyncSeriesTimer,
    TBEStatsReporter,
    TBEStatsReporterConfig,
    StdLogStatsReporterConfig,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheState,
    construct_cache_state,
    EmbeddingLocation,
    MAX_PREFETCH_DEPTH,
    PoolingMode,
    RecordCacheMetrics,
    SplitState,
)
from dynamicemb_extensions import insert_or_assign, clear

report_interval = 10
warmup_repeat=100


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_emb_precision(precision_str):
    if precision_str == "fp32":
        return torch.float32
    elif precision_str == "fp16":
        return torch.float16
    elif precision_str == "bf16":
        return torch.bfloat16
    else:
        raise ValueError("unknown embedding precision type")

def get_fbgemm_precision(precision_str):
    if precision_str == "fp32":
        return SparseType.FP32
    elif precision_str == "fp16":
        return SparseType.FP16
    elif precision_str == "bf16":
        return SparseType.BF16
    else:
        raise ValueError("unknown embedding precision type")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark BatchedDynamicEmbeddingTables in dynamicemb."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size used for training",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="1",
        help="Comma separated max_ind_size(MB) per sparse feature. The number of embeddings in each embedding table.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="number of iterations",
    )
    parser.add_argument(
        "--hbm_for_embeddings",
        type=str,
        default="1",
        help="HBM reserved for values in GB.",
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        choices=["sgd", "adam", "exact_adagrad", "exact_row_wise_adagrad"],
        help="optimizer type.",
    )
    parser.add_argument(
        "--feature_distribution",
        type=str,
        default="random",
        choices=["random", "pow-law"],
        help="Distribution of sparse features.",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.05, help="Exponent of power-law distribution."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed used for initialization"
    )
    parser.add_argument(
        "--use_index_dedup",
        action="store_true",
        help="Use index deduplication, using to select the codepath.",
    )
    parser.add_argument("--caching", action="store_true")
    parser.add_argument("--cache_metrics", action="store_true")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Size of each embedding."
    )
    parser.add_argument(
        "--emb_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--cache_algorithm",
        type=str,
        default="lru",
        choices=["lru", "lfu"],
    )
    parser.add_argument(
        "--gpu_ratio",
        type=float,
        default=0.125,
        help="cache how many embeddings to HBM",
    )

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta1.")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay.")

    args = parser.parse_args()
    args.num_embeddings_per_feature = [
        int(v) * 1024 * 1024 for v in args.num_embeddings_per_feature.split(",")
    ]
    args.num_embedding_table = len(args.num_embeddings_per_feature)
    args.hbm_for_embeddings = [
        int(v) * (1024**3) for v in args.hbm_for_embeddings.split(",")
    ]

    return args


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def get_dynamicemb_optimizer(optimizer_type):
    if optimizer_type == "sgd":
        return EmbOptimType.EXACT_SGD
    elif optimizer_type == "exact_sgd":
        return EmbOptimType.EXACT_SGD
    elif optimizer_type == "adam":
        return EmbOptimType.ADAM
    elif optimizer_type == "exact_adagrad":
        return EmbOptimType.EXACT_ADAGRAD
    elif optimizer_type == "exact_row_wise_adagrad":
        return EmbOptimType.EXACT_ROWWISE_ADAGRAD
    else:
        raise ValueError("unknown optimizer type")

def get_fbgemm_optimizer(optimizer_type):
    if optimizer_type == "sgd":
        return OptimType.EXACT_SGD
    elif optimizer_type == "exact_sgd":
        return OptimType.EXACT_SGD
    elif optimizer_type == "adam":
        return OptimType.ADAM
    elif optimizer_type == "exact_adagrad":
        return OptimType.EXACT_ADAGRAD
    elif optimizer_type == "exact_row_wise_adagrad":
        return OptimType.EXACT_ROWWISE_ADAGRAD
    else:
        raise ValueError("unknown optimizer type")

#TODO: collect input distribution theory hit_rate
# if existed gap, then improve hit rate.
# if low theory hit_rate is low, then change alpha.
def zipf(min_val, max_val, exponent, size, device):
    """
    Generates Zipf-like random variables in the inclusive range [min_val, max_val).

    Args:
        min_val (int): Minimum value (inclusive, must be ≥0).
        max_val (int): Maximum value (exclusive).
        exponent (float): Exponent parameter (a > 0).
        size (int): Output shape.

    Returns:
        torch.Tensor: Sampled values of specified size.
    """

    # Generate integer values and probabilities
    values = torch.arange(min_val + 1, max_val + 1, dtype=torch.long, device=device)
    probs = 1.0 / (values.float() ** exponent)
    probs_normalized = probs / probs.sum()

    # k = np.arange(min_val, max_val)
    # np.random.shuffle(k)

    k = torch.arange(min_val, max_val, dtype=torch.long, device=device)
    perm = torch.randperm(k.size(0), device=device)
    k_shuffled = k[perm]
    
    probs_np = probs_normalized.cpu().numpy()
    samples = np.random.choice(k_shuffled.cpu().numpy(), size=size, replace=True, p=probs_np)
    samples = torch.tensor(samples, device=probs_normalized.device)

    return samples


def generate_sequence_sparse_feature(args, device):
    feature_names = [
        feature_idx_to_name(feature_idx)
        for feature_idx in range(args.num_embedding_table)
    ]
    if args.feature_distribution == "random":
        res = []
        for x in range(args.num_iterations):
            indices_list = []
            lengths_list = []
            for i in range(args.num_embedding_table):
                indices_list.append(
                    torch.randint(low=0, high=(2**63)-1, size=(args.batch_size,))
                )
            indices = torch.cat(indices_list, dim=0)
            indices = indices.to(dtype=torch.int64, device="cuda")
            lengths_list.extend([1] * args.batch_size * args.num_embedding_table)
            lengths = torch.tensor(lengths_list, dtype=torch.int64).cuda()

            res.append(
                torchrec.KeyedJaggedTensor(
                    keys=feature_names,
                    values=indices,
                    lengths=lengths,
                )
            )
        return res
    elif args.feature_distribution == "pow-law":
        assert args.num_embedding_table == 1
        from dataset_generator import gen_jagged_key
        res = [
            gen_jagged_key(
                args.batch_size,
                1, 
                args.alpha,
                args.num_embeddings_per_feature[0],
                device,
                feature_names) for i in range(args.num_iterations)]
        return res
    elif args.feature_distribution == "zipf":
        assert args.num_embedding_table == 1
        total_indices = zipf(
            min_val=0,
            max_val=args.num_embeddings_per_feature[0],
            exponent=args.alpha,
            size=args.batch_size * args.num_iterations,
            device=device,
        )
        total_indices = total_indices.to(dtype=torch.int64, device="cuda")
        res = []
        for x in range(args.num_iterations):  
            indices = total_indices[x * args.batch_size: (x + 1) * args.batch_size]
            lengths_list = []
            lengths_list.extend([1] * args.batch_size * args.num_embedding_table)
            lengths = torch.tensor(lengths_list, dtype=torch.int64).cuda()
            feature_names = [
                feature_idx_to_name(feature_idx)
                for feature_idx in range(args.num_embedding_table)
            ]

            res.append(
                torchrec.KeyedJaggedTensor(
                    keys=feature_names,
                    values=indices,
                    lengths=lengths,
                )
            )
        return res
    else:
        raise ValueError(
            f"Not support distribution {args.feature_distribution} of sparse features."
        )

def create_dynamic_embedding_tables(args, device):
    table_options = []
    table_num = args.num_embedding_table
    for i in range(table_num):
        table_options.append(
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=get_emb_precision(args.emb_precision),
                dim=args.embedding_dim,
                max_capacity=args.num_embeddings_per_feature[i],
                local_hbm_for_values=args.hbm_for_embeddings[i],
                bucket_capacity=128,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                ),
                score_strategy=DynamicEmbScoreStrategy.LFU if args.cache_algorithm == "lfu" else DynamicEmbScoreStrategy.TIMESTAMP,
                caching=args.caching,
            )
        )

    var = BatchedDynamicEmbeddingTables(
        table_options=table_options,
        table_names=[table_idx_to_name(i) for i in range(table_num)],
        use_index_dedup=args.use_index_dedup,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=get_emb_precision(args.output_dtype),
        device=device,
        optimizer=get_dynamicemb_optimizer(args.optimizer_type),
        learning_rate=args.learning_rate,
        eps=args.eps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    for table_id in range(table_num):
        if args.caching:
            cur_hkv_table = var.host_tables[table_id]
        else:
            cur_hkv_table = var.tables[table_id]

        num_embeddings = args.num_embeddings_per_feature[table_id]
        fill_batch = 1024*1024
        i = 0
        while i < num_embeddings:
            start = i
            end = min(i + fill_batch, num_embeddings)
            i += fill_batch
            unique_indices = torch.arange(start, end, device=device, dtype=torch.int64)
            unique_values = torch.rand(unique_indices.numel(), args.embedding_dim, device=device, dtype=torch.float32)

            optstate_dim = cur_hkv_table.optstate_dim()
            initial_accumulator = cur_hkv_table.get_initial_optstate()
            optstate = (
                torch.rand(
                    unique_values.size(0),
                    optstate_dim,
                    dtype=unique_values.dtype,
                    device=unique_values.device,
                )
                * initial_accumulator
            )
            unique_values = torch.cat((unique_values, optstate), dim=1).contiguous()
            unique_values = unique_values.reshape(-1)

            n = unique_indices.shape[0]
            if args.cache_algorithm == "lfu":
                scores = torch.ones(n, dtype=torch.uint64, device=unique_indices.device)
                insert_or_assign(cur_hkv_table, n, unique_indices, unique_values, scores)
            else:
                insert_or_assign(cur_hkv_table, n, unique_indices, unique_values)
    return var

def create_split_table_batched_embeddings(args, device):    
    optimizer = get_fbgemm_optimizer(args.optimizer_type)
    D = args.embedding_dim
    Es = args.num_embeddings_per_feature
    cache_alg = CacheAlgorithm.LRU if args.cache_algorithm == "lru" else CacheAlgorithm.LFU
    
    if args.caching:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    e,
                    D,
                    EmbeddingLocation.MANAGED_CACHING,
                    ComputeDevice.CUDA,
                )
                for e in Es
            ],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(args.emb_precision),
            stochastic_rounding=False,
            cache_load_factor=args.gpu_ratio,
            cache_algorithm=cache_alg,
            pooling_mode=PoolingMode.NONE,
            output_dtype=get_fbgemm_precision(args.output_dtype),
            device=device,
            learning_rate=args.learning_rate,
            eps=args.eps,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
            stats_reporter_config=StdLogStatsReporterConfig(report_interval),
            record_cache_metrics=RecordCacheMetrics(True, False),
        ).cuda()
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    e,
                    D,
                    EmbeddingLocation.MANAGED,
                    ComputeDevice.CUDA,
                )
                for e in Es
            ],
            optimizer=optimizer,
            weights_precision=get_fbgemm_precision(args.emb_precision),
            stochastic_rounding=False,
            pooling_mode=PoolingMode.NONE,
            output_dtype=get_fbgemm_precision(args.output_dtype),
            device=device,
            learning_rate=args.learning_rate,
            eps=args.eps,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            bounds_check_mode=BoundsCheckMode.NONE,
        ).cuda()
    return emb


def warmup_gpu(device="cuda"):
    # 1. compute unit
    a = torch.randn(10, 16384, 2048, device=device)
    b = torch.randn(10, 2048, 16384, device=device)
    for _ in range(5):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    # 2. copy engine
    d_cpu = torch.randn(10, 1024, 1024)
    d_gpu = torch.empty_like(d_cpu, device=device)
    for _ in range(5):
        # CPU -> GPU
        d_gpu.copy_(d_cpu, non_blocking=True)
        torch.cuda.synchronize()
        # GPU -> CPU
        d_cpu.copy_(d_gpu, non_blocking=True)
        torch.cuda.synchronize()

def benchmark_one_iteration(model, sparse_feature):
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output = model(sparse_feature.values(), sparse_feature.offsets())
    mid_event.record()
    grad = torch.empty_like(output)
    output.backward(grad)
    end_event.record()

    torch.cuda.synchronize()
    forward_latency = start_event.elapsed_time(mid_event)
    backward_latency = mid_event.elapsed_time(end_event)
    iteration_latency = start_event.elapsed_time(end_event)
    return forward_latency, backward_latency, iteration_latency

def append_to_json(file_path, data):
    try:
        with open(file_path, "r") as f:
            exist_data = json.load(f)
            if isinstance(exist_data, list):
                exist_data.append(data)
            elif isinstance(exist_data, dict):
                exist_data.update(data)
            else:
                raise ValueError("Invalid JSON data type")
    except FileNotFoundError:
        exist_data = [data] if isinstance(data, dict) else data

    with open(file_path, "w") as f:
        json.dump(exist_data, f, indent=4)

def input_distribution(tensor_list, n, max_val, batch_size):
    counts = torch.zeros(max_val + 1, dtype=torch.long, device=tensor_list[0].values().device)
    counts_res = torch.zeros(max_val + 1, dtype=torch.long, device=tensor_list[0].values().device)
    for i in range(n):
        tensor_ = tensor_list[i]
        indices = tensor_.values()
        unique_vals_, cnts = torch.unique(indices, return_counts=True)
        counts[unique_vals_] = 1
    tensor = tensor_list[n]
    indices = tensor.values()
    print(indices.size(0))
    unique_vals, cnts = torch.unique(indices, return_counts=True)
    counts_res[unique_vals] = 1
    print(unique_vals.size(0))
    equal_mask = (counts == 1) & (counts_res == 1) 
    num_equal = equal_mask.sum().item()
    return num_equal, (num_equal / unique_vals.size(0)) * 100

def warmup_tables(tensor_list, n, max_val, batch_size, dynamic_emb, torchrec_emb):
    counts = torch.zeros(max_val + 1, dtype=torch.long, device=tensor_list[0].values().device)
    for tensor in tensor_list:
        indices = tensor.values()
        unique_vals, cnts = torch.unique(indices, return_counts=True)
        counts[unique_vals] += cnts
    top_counts, top_indices = torch.topk(counts, n)
    total_unique_num = (counts != 0).sum().item()
    print("Totol unique input number:", total_unique_num)
    length = torch.ones(batch_size, dtype=torch.int64, device=top_indices.device)
    batches = torch.split(top_indices, batch_size, dim=0)
    for i, batch in enumerate(reversed(batches)):
        features = torchrec.KeyedJaggedTensor(
            keys=["t0"],
            values=batch,
            lengths=length,
        )
        for j in range(warmup_repeat):
            dynamic_emb(features.values(), features.offsets())
            torchrec_emb(features.values(), features.offsets())

def clear_cache(args, dynamic_emb, torchrec_emb):
    table_num = args.num_embedding_table
    if args.caching:
        for table_id in range(table_num):
            clear(dynamic_emb.tables[table_id])
    torchrec_emb.reset_cache_states()
    
@record
def main():
    args = parse_args()
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    backend = "nccl"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    timer = GPUTimer()
    timer.start()
    var = create_dynamic_embedding_tables(args, device)
    timer.stop()
    print(f"Create dynamic embedding done in {timer.elapsed_time() / 1000:.3f} s.")

    timer.start()
    num_embs = [f"{num}" for num in args.num_embeddings_per_feature]
    features_file = f"{args.num_iterations}-{args.feature_distribution}-{num_embs}-{args.batch_size}-{args.alpha}.pt"
    try:
        with open(features_file, 'rb') as f:
            sparse_features = torch.load(f, map_location=f"cuda:{local_rank}")
    except FileNotFoundError:
        sparse_features = []
        for i in range(args.num_iterations):
            sparse_features= generate_sequence_sparse_feature(args, device)
        torch.save(sparse_features, features_file)
    timer.stop()
    print(f"Generate sparse features done in {timer.elapsed_time() / 1000:.3f} s.")

    warmup_gpu(device)
    torchrec_emb = create_split_table_batched_embeddings(args, device)
    cache_miss_counter_torchrec = None
    
    # for i in range(5):
    #     hit_cnt, hit_rate = input_distribution(sparse_features, 200 * i, args.num_embeddings_per_feature[0], args.batch_size)
    #     print(f"Iteration{i}, hit_cnt {hit_cnt}, hit_rate {hit_rate:.5f}")
    
    var.set_record_cache_metrics(True)
    clear_cache(args, var, torchrec_emb)
    # warmup_tables(sparse_features, int(args.gpu_ratio * args.num_embeddings_per_feature[0]), 
    #               args.num_embeddings_per_feature[0], args.batch_size, var, torchrec_emb)
    # os.environ["DISABLE_UPDATE_CACHE"] = "1"

    for i in range(0, args.num_iterations, report_interval):
        for j in range(report_interval):   
            forward_latency, backward_latency, iteration_latency = benchmark_one_iteration(
                var, sparse_features[i + j]
            )
            load_factors = [t.load_factor() for t in var.tables]
            load_factors.extend([t.load_factor() for t in var.host_tables if t is not None])
            load_factors = [f"{lf}" for lf in load_factors]
            cache_info = ""
            if var.host_tables[0] is not None:
                cache_metrics = var.cache_metrics
                unique_num = cache_metrics[0].item()
                cache_hit = cache_metrics[1].item()
                storage_hit = cache_metrics[2].item()
                storage_evict = cache_metrics[3].item()
                hit_rate = 1.0 * cache_hit / unique_num
                cache_info = f"miss: {unique_num - cache_hit} unique: {unique_num}, hit_rate: {hit_rate:.8f}, need_init: {unique_num-cache_hit-storage_hit}, storage_evict: {storage_evict}"
            print(
                f"Iteration {i + j}, forward: {forward_latency:.3f} ms,   backward: {backward_latency:.3f} ms,  "
                f"total: {iteration_latency:.3f} ms,  load factors: {load_factors}  cache info: {cache_info}"
            )

        # table_wise_cache_miss = None
        for j in range(report_interval):   
            forward_latency, backward_latency, iteration_latency = benchmark_one_iteration(
                torchrec_emb, sparse_features[i + j]
            )
            cache_miss_counter_ = torchrec_emb.get_cache_miss_counter().clone()
            # table_wise_cache_miss_ = torchrec_emb.get_table_wise_cache_miss().clone()
            if cache_miss_counter_torchrec is not None:
                cache_miss_counter_incerment = cache_miss_counter_ - cache_miss_counter_torchrec
            else:
                cache_miss_counter_incerment = torch.tensor([0,0])
            # if table_wise_cache_miss is not None:
            #     table_wise_cache_miss_increment = table_wise_cache_miss_ - table_wise_cache_miss
            # else:
            #     table_wise_cache_miss_increment = torch.tensor([0])
            cache_info = f"miss: {cache_miss_counter_incerment[1].item()}"
            print(
                f"Iteration {i + j}, forward: {forward_latency:.3f} ms,   backward: {backward_latency:.3f} ms,  "
                f"total: {iteration_latency:.3f} ms, cache info {cache_info}"
            )
            cache_miss_counter_torchrec = cache_miss_counter_
            # table_wise_cache_miss = table_wise_cache_miss_

    # benchmark TorchRec
    # torchrec_emb = create_split_table_batched_embeddings(args, device)
    # for i in range(args.num_iterations):
            
    #     forward_latency, backward_latency, iteration_latency = benchmark_one_iteration(
    #         torchrec_emb, sparse_features[i]
    #     )
    #     print(
    #         f"Iteration {i}, forward: {forward_latency:.3f} ms,   backward: {backward_latency:.3f} ms,  "
    #         f"total: {iteration_latency:.3f} ms"
    #     )
    
    var.set_record_cache_metrics(False)
    torchrec_emb.record_cache_metrics = RecordCacheMetrics(False, False)
    clear_cache(args, var, torchrec_emb)
    # warmup_tables(sparse_features, int(args.gpu_ratio * args.num_embeddings_per_feature[0]), 
    #               args.num_embeddings_per_feature[0], args.batch_size, var, torchrec_emb)

    torch.cuda.profiler.start()

    timer.start()
    for i in range(args.num_iterations):
        sparse_feature = sparse_features[i]
        output = var(sparse_feature.values(), sparse_feature.offsets())
        grad = torch.empty_like(output)
        output.backward(grad)
    timer.stop()
    latency = timer.elapsed_time() / args.num_iterations
    print(f"Latency(dynamicemb): {latency:.4f}")

    timer.start()
    for i in range(args.num_iterations):
        sparse_feature = sparse_features[i]
        output = torchrec_emb(sparse_feature.values(), sparse_feature.offsets())
        grad = torch.empty_like(output)
        output.backward(grad)
    timer.stop()
    latency = timer.elapsed_time() / args.num_iterations
    print(f"Latency(torchrec): {latency:.4f}")

    torch.cuda.profiler.stop()
    

    # test_result = {
    #     "use_index_dedup": args.use_index_dedup,
    #     "batch_size": args.batch_size,
    #     "num_embeddings_per_feature": args.num_embeddings_per_feature,
    #     "hbm_for_embeddings": args.hbm_for_embeddings,
    #     "optimizer_type": args.optimizer_type,
    #     "forward_overhead": average_iteration_time_fw,
    #     "backward_overhead": average_iteration_time - average_iteration_time_fw,
    #     "totoal_overhead": average_iteration_time,
    # }
    # append_to_json("benchmark_results.json", test_result)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
