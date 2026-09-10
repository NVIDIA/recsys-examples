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

import copy
import os
import random
from typing import List

import pytest
import torch
import torch.distributed as dist
import torchrec
from dynamicemb.construct_twin_module import ConstructTwinModule
from dynamicemb.shard.embedding import DynamicEmbeddingCollectionContext
from torchrec.modules.embedding_configs import PoolingType


def init_fn(x: torch.Tensor):
    with torch.no_grad():
        x.uniform_(0, 1)


def generate_sparse_feature(
    feature_names,
    num_embeddings_list,
    multi_hot_sizes,
    local_batch_size,
    weighted=False,
):
    feature_batch = len(feature_names) * local_batch_size

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

    weights = None
    if weighted:
        weights = torch.rand(len(indices), dtype=torch.float32).cuda() * 1.5 + 0.5

    return torchrec.KeyedJaggedTensor(
        keys=feature_names,
        values=torch.tensor(indices, dtype=torch.int64).cuda(),
        lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
        weights=weights,
    )


def embedding_lookup_backward(ret, is_pooled):
    """
    Perform backward pass on embeddings for both pooled and non-pooled cases
    """
    if is_pooled:
        # For pooled embeddings (EmbeddingBagCollection output)
        # The output is a KeyedTensor - directly sum values
        reduced_tensor = ret.values().sum()
        reduced_tensor.backward()
    else:
        # For non-pooled embeddings (EmbeddingCollection output)
        # The output is a dict of KeyedJaggedTensors
        jagged_tensors = []
        for k, v in ret.items():
            jagged_tensors.append(v.values())

        if jagged_tensors:  # Check if list is not empty
            concatenated_tensor = torch.cat(jagged_tensors, dim=0)
            reduced_tensor = concatenated_tensor.sum()
            reduced_tensor.backward()


optimizer_dict = {
    "sgd": {
        "optimizer": "sgd",
        "learning_rate": 0.01,
    },
    "exact_adagrad": {
        "optimizer": "exact_adagrad",
        "learning_rate": 0.01,
        "eps": 1e-10,
    },
    "exact_row_wise_adagrad": {
        "optimizer": "exact_row_wise_adagrad",
        "learning_rate": 0.01,
        "eps": 1e-10,
    },
    "adam": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-12,
        "weight_decay": 0,
    },
    "adamw": {
        "optimizer": "adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-12,
        "weight_decay": 0.1,
    },
}


def test_dynamic_embedding_collection_context_initializes_fresh_lists():
    field_names = (
        "sharding_contexts",
        "input_features",
        "reverse_indices",
        "seq_vbe_ctx",
        "frequency_counters",
    )

    first = DynamicEmbeddingCollectionContext()
    second = DynamicEmbeddingCollectionContext()

    for field_name in field_names:
        assert getattr(first, field_name) == []
        assert getattr(first, field_name) is not getattr(second, field_name)


@pytest.fixture
def tolerance():
    return {
        "rtol": 1e-7,  # Relative tolerance (5 significant digits)
        "atol": 1e-5,  # Absolute tolerance for values near zero
    }


@pytest.fixture
def seed():
    return 1234


@pytest.fixture(scope="session")
def backend_session():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    yield
    # dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "table_num, num_embeddings",
    [
        pytest.param(1, [128 * 1024]),
        pytest.param(4, [i * 128 * 1024 for i in [1, 2, 3, 4]]),
    ],
)
# The bag size belongs with the pooling mode: a sequence lookup has one
# embedding per key by definition, while SUM has to actually accumulate several
# of them -- with a bag of at most one key the reduction being tested never runs.
@pytest.mark.parametrize(
    "is_pooled, pooling_mode, max_bag_size",
    [(True, PoolingType.SUM, 8), (False, None, 1)],
)
@pytest.mark.parametrize("batch_size", [32, 2048])
@pytest.mark.parametrize("num_iteration", [10])
@pytest.mark.parametrize("dim", [128])
@pytest.mark.parametrize(
    "optimizer_name",
    ["sgd", "exact_row_wise_adagrad", "exact_adagrad"],
)
@pytest.mark.parametrize("use_index_dedup", [True, False])
@pytest.mark.parametrize("is_weighted", [False, True])
def test_twin_module(
    table_num: int,
    dim: int,
    num_embeddings: List[int],
    is_pooled: bool,
    pooling_mode,
    max_bag_size: int,
    batch_size,
    num_iteration,
    tolerance,
    seed,
    optimizer_name,
    backend_session,
    use_index_dedup,
    is_weighted,
):
    if is_weighted:
        if not (is_pooled and pooling_mode == PoolingType.SUM):
            pytest.skip("weighted pooling only exists for pooled SUM collections")
        if optimizer_name != "exact_row_wise_adagrad":
            # The weights only scale what reaches the gradient reduction; which
            # optimizer consumes the result afterwards is orthogonal. Covering
            # every optimizer again would triple this axis for no extra signal.
            pytest.skip("weighted path is covered once, on exact_row_wise_adagrad")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    [f"t_{t}" for t in range(table_num)]
    feature_names = [f"f_{t}" for t in range(table_num)]

    optimizer_kwargs = copy.deepcopy(optimizer_dict[optimizer_name])

    dims = [dim] * table_num
    # randint(0, max_bag_size) per bag, so lengths still vary and empty bags
    # still occur; this only sets the upper end.
    multi_hot_sizes = [max_bag_size] * table_num

    construct = ConstructTwinModule(
        table_num,
        dims,
        num_embeddings,
        pooling_mode,
        is_pooled,
        multi_hot_sizes,
        optimizer_kwargs=optimizer_kwargs,
        use_index_dedup=use_index_dedup,
        is_weighted=is_weighted,
        rank=local_rank,
        world_size=world_size,
    )
    construct.init_twin_embedding_model()
    dynamicemb_model = construct.dynamicemb_model
    torchrec_model = construct.torchrec_model

    for i in range(num_iteration):
        if i % 2 == 0:
            sparse_feature = generate_sparse_feature(
                feature_names,
                num_embeddings,
                multi_hot_sizes,
                batch_size,
                weighted=is_weighted,
            )
        ret = dynamicemb_model(sparse_feature)
        ret_compare = torchrec_model(sparse_feature)

        embedding_lookup_backward(ret, is_pooled)
        embedding_lookup_backward(ret_compare, is_pooled)

        if is_pooled:
            # Check for equivalence with relative tolerance
            assert ret_compare.keys() == ret.keys()

            # Get the values
            torchrec_values = ret_compare.values()
            dynamicemb_values = ret.values()
            torch.testing.assert_close(torchrec_values, dynamicemb_values)
        else:
            for key, value in ret_compare.items():
                torchrec_values = value.values()
                dynamicemb_values = ret[key].values()
                torch.testing.assert_close(torchrec_values, dynamicemb_values)
