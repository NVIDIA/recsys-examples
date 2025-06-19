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
import random
import sys
import time
from typing import List

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTables
from torch.distributed.elastic.multiprocessing.errors import record


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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def get_optimizer(optimizer_type):
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


def generate_dynamic_sequence_sparse_feature(
    batch_size,
):
    indices = []
    lengths = []

    indices_set = set({})
    while len(indices_set) < batch_size:
        indices_set.add(random.randint(0, (2**63) - 1))
    indices.extend(list(indices_set))
    lengths.extend([1] * batch_size)

    return torchrec.KeyedJaggedTensor(
        keys=[feature_idx_to_name(0)],
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
    )


def count_tensor_to_dict(x, d):
    x = x.to("cpu")
    for i in x:
        key = i.item()
        if key not in d:
            d[key] = 1
        else:
            d[key] += 1


def test(args):
    backend = "nccl"
    dist.init_process_group(backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    table_num = args.num_embedding_table
    total_hbm_in_byte = args.hbm_for_embeddings * (1024**3)
    dim = args.embedding_dim
    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=dim,
            max_capacity=num_emb,
            local_hbm_for_values=total_hbm_in_byte,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.NORMAL,
            ),
            caching=True,
        )
        for num_emb in args.num_embeddings_per_feature
    ]

    var = BatchedDynamicEmbeddingTables(
        table_options=table_options,
        table_names=[table_idx_to_name(i) for i in range(table_num)],
        use_index_dedup=args.use_index_dedup,
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32,
        device=device,
        optimizer=get_optimizer(args.optimizer_type),
        learning_rate=args.learning_rate,
        eps=args.eps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    num_iterations = args.num_iterations

    warm_iters = 10
    sparse_features = []
    for i in range(num_iterations * 2 + warm_iters):
        sparse_features.append(
            generate_dynamic_sequence_sparse_feature(args.batch_size)
        )

    for i in range(warm_iters):
        sparse_feature = sparse_features[i]
        res = var(sparse_feature.values(), sparse_feature.offsets())
        grad = torch.ones_like(res)
        res.backward(grad)

    # forward
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for i in range(num_iterations):
        sparse_feature = sparse_features[i + warm_iters]
        res = var(sparse_feature.values(), sparse_feature.offsets())

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    average_iteration_time_fw = (end_time - start_time) / args.num_iterations * 1000
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print(f"Average time per iteration(forward): {average_iteration_time_fw:.4f} ms")

    # forward + backward
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    torch.cuda.profiler.start()
    for i in range(num_iterations):
        sparse_feature = sparse_features[i + warm_iters + num_iterations]
        res = var(sparse_feature.values(), sparse_feature.offsets())
        grad = torch.empty_like(res)
        res.backward(grad)

    torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    average_iteration_time = (end_time - start_time) / args.num_iterations * 1000
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print(
        f"Average time per iteration(forward + backward): {average_iteration_time:.4f} ms"
    )

    test_result = {
        "use_index_dedup": args.use_index_dedup,
        "batch_size": args.batch_size,
        "num_embeddings_per_feature": args.num_embeddings_per_feature,
        "hbm_for_embeddings": args.hbm_for_embeddings,
        "optimizer_type": args.optimizer_type,
        "forward_overhead": average_iteration_time_fw,
        "backward_overhead": average_iteration_time - average_iteration_time_fw,
        "totoal_overhead": average_iteration_time,
    }
    append_to_json("benchmark_results.json", test_result)

    dist.barrier()
    dist.destroy_process_group()


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="1",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="1",
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--emb_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--hbm_for_embeddings",
        type=int,
        default=1,
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
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="beta1.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="beta1.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="weight_decay.",
    )
    parser.add_argument(
        "--use_index_dedup",
        type=str2bool,
        default=True,
        help="Use index deduplication (default: True).",
    )

    args = parser.parse_args()
    args.num_embeddings_per_feature = [
        int(v) * 1024 * 1024 for v in args.num_embeddings_per_feature.split(",")
    ]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.num_embedding_table = len(args.num_embeddings_per_feature)
    args.hbm_for_embeddings = int(args.hbm_for_embeddings)

    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    test(args)


if __name__ == "__main__":
    main(sys.argv[1:])
