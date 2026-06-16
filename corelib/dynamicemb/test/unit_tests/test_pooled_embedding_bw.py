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
import os
import random
import sys
from typing import Dict, List

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
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from torch.distributed.elastic.multiprocessing.errors import record


def str2poolingmode(v):
    if v.lower() in ("sum"):
        return DynamicEmbPoolingMode.SUM
    elif v.lower() in ("mean"):
        return DynamicEmbPoolingMode.MEAN
    else:
        raise argparse.ArgumentTypeError(
            "Only sum and mean is supported for pooled embedding"
        )


def table_idx_to_name(i):
    return f"t_{i}"


def feature_idx_to_name(i):
    return f"cate_{i}"


def generate_sparse_feature(
    feature_num, num_embeddings_list, multi_hot_sizes, local_batch_size=50
):
    feature_batch = feature_num * local_batch_size

    indices = []
    lengths = []

    for i in range(feature_batch):
        f = i // local_batch_size
        cur_bag_size = random.randint(0, multi_hot_sizes[f])
        cur_bag = set({})
        while len(cur_bag) < cur_bag_size:
            cur_bag.add(random.randint(0, num_embeddings_list[f] - 1))

        indices.extend(list(cur_bag))
        lengths.append(cur_bag_size)

    return torchrec.KeyedJaggedTensor(
        keys=[feature_idx_to_name(feature_idx) for feature_idx in range(feature_num)],
        values=torch.tensor(
            indices, dtype=torch.int64
        ).cuda(),  # key [0,1] on rank0, [2] on rank 1
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
    total_hbm_in_byte = 1023**3
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
                mode=DynamicEmbInitializerMode.DEBUG,
            ),
        )
        for num_emb in args.num_embeddings_per_feature
    ]

    var = BatchedDynamicEmbeddingTablesV2(
        table_options=table_options,
        output_dtype=torch.float32,
        table_names=[table_idx_to_name(i) for i in range(table_num)],
        pooling_mode=args.pooling_mode,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
    )

    num_iterations = args.num_iterations

    for i in range(num_iterations):
        sparse_feature = generate_sparse_feature(
            feature_num=table_num,
            num_embeddings_list=args.num_embeddings_per_feature,
            multi_hot_sizes=args.multi_hot_sizes,
            local_batch_size=args.batch_size,
        )
        indices = sparse_feature.values()
        res = var(indices, sparse_feature.offsets())
        grad = torch.ones_like(res)
        res.backward(grad)

        with torch.no_grad():
            ref_dicts: List[Dict[int, int]] = [{} for _ in range(table_num)]
            segment = [1] * table_num
            lengths = []
            for indices_per_table, dict_ in zip(
                [x.values() for x in sparse_feature.split(segment)], ref_dicts
            ):
                count_tensor_to_dict(indices_per_table, dict_)
                lengths.append(indices_per_table.size()[0])
                assert indices_per_table.dim() == 1

            assert res.dim() == 2
            assert res.size()[0] == args.batch_size
            assert res.size()[1] == table_num * dim
    dist.barrier()
    dist.destroy_process_group()


@record
def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic sequence embedding's backward"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=32,
        help="number of iterations",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default="196608,131072,9437184,851968",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default="20,17,101,49",
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
        "--optimizer_type",
        type=str,
        default="adagrad",
        choices=["sgd", "adagrad", "rowwise_adagrad"],
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
        "--pooling_mode",
        type=str2poolingmode,
        default=DynamicEmbPoolingMode.SUM,
        help="Pooling mode of dynamic embedding bag.",
    )

    args = parser.parse_args()
    args.num_embeddings_per_feature = [
        int(v) for v in args.num_embeddings_per_feature.split(",")
    ]
    args.multi_hot_sizes = [int(v) for v in args.multi_hot_sizes.split(",")]
    args.num_embedding_table = len(args.num_embeddings_per_feature)

    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    test(args)


if __name__ == "__main__":
    main(sys.argv[1:])
