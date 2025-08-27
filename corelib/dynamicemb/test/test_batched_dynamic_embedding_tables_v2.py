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

from typing import Dict

import pytest
import torch
from dynamicemb import (
    DynamicEmbEvictStrategy,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import DynamicEmbTable
from dynamicemb.key_value_table import KeyValueTable, insert_or_assign
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    BoundsCheckMode,
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    ComputeDevice,
    SplitTableBatchedEmbeddingBagsCodegen,
)


def create_split_table_batched_embedding(
    table_names,
    feature_table_map,
    optimizer_type,
    opt_params,
    dims,
    num_embs,
    pooling_mode,
    device,
):
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                e,
                d,
                EmbeddingLocation.DEVICE,
                ComputeDevice.CUDA,
            )
            for (e, d) in zip(num_embs, dims)
        ],
        optimizer=optimizer_type,
        weights_precision=SparseType.FP32,
        stochastic_rounding=False,
        pooling_mode=pooling_mode,
        output_dtype=SparseType.FP32,
        device=device,
        table_names=table_names,
        feature_table_map=feature_table_map,
        **opt_params,
        bounds_check_mode=BoundsCheckMode.FATAL,
    ).cuda()
    return emb


def init_embedding_tables(stbe, bdet):
    stbe.init_embedding_weights_uniform(0, 1)
    for split, table in zip(stbe.split_embedding_weights(), bdet.tables):
        num_emb = split.size(0)
        emb_dim = split.size(1)
        indices = torch.arange(num_emb, device=split.device, dtype=torch.long)
        if isinstance(table, DynamicEmbTable):
            val_dim = table.optstate_dim() + emb_dim
            values = torch.empty(
                num_emb, val_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            values[:, emb_dim:val_dim] = table.get_initial_optstate()
            insert_or_assign(table, num_emb, indices, values)
        elif isinstance(table, KeyValueTable):
            val_dim = table.value_dim()
            assert emb_dim == table.embedding_dim()
            values = torch.empty(
                num_emb, val_dim, dtype=split.dtype, device=split.device
            )
            values[:, :emb_dim] = split
            values[:, emb_dim:val_dim] = table.init_optimizer_state()
            table.insert(indices, values)
        else:
            raise ValueError("Not support table type")
    # for states_per_table in stbe.split_optimizer_states():
    #     for state in states_per_table:
    #           pass


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize("caching", [True, False])
def test_forward_train_eval(opt_type, opt_params, caching):
    print(
        f"step in test_forward_train_eval , opt_type = {opt_type} opt_params = {opt_params}"
    )
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    dims = [8, 8, 8]
    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    init_capacity = 1024
    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=init_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            evict_strategy=DynamicEmbEvictStrategy.LRU,
            caching=caching,
            local_hbm_for_values=1024**3,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    bdebt = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=[0, 0, 1, 2],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=opt_type,
        use_index_dedup=True,
        **opt_params,
    )
    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    indices = torch.tensor(
        [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], dtype=key_type, device=device
    )
    offsets = torch.tensor(
        [0, 2, 3, 5, 6, 8, 10, 10, 11], dtype=key_type, device=device
    )

    embs_train = bdebt(indices, offsets)
    torch.cuda.synchronize()

    with torch.no_grad():
        bdebt.eval()
        embs_eval = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # non-exist key
    indices = torch.tensor([777, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device).to(
        key_type
    )
    offsets = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(key_type)
    embs_non_exist = bdebt(indices, offsets)
    torch.cuda.synchronize()

    # train
    bdebt.train()
    embs_train_non_exist = bdebt(indices, offsets)
    torch.cuda.synchronize()

    assert torch.equal(embs_train, embs_eval)
    assert torch.equal(embs_train[1:, :], embs_non_exist[1:, :])
    assert torch.all(embs_non_exist[0, :] == 0)
    assert torch.all(embs_train_non_exist[0, :] != 0)
    assert torch.equal(embs_train_non_exist[1:, :], embs_non_exist[1:, :])

    print("all check passed")


"""
For torchrec's adam optimizer, it will increment the optimizer_step in every forward,
    which will affect the weights update, pay attention to it or try to use `set_optimizer_step()` 
    to control(not verified) it.
"""


@pytest.mark.parametrize(
    "opt_type,opt_params",
    [
        (EmbOptimType.SGD, {"learning_rate": 0.3}),
        (
            EmbOptimType.ADAM,
            {
                "learning_rate": 0.3,
                "weight_decay": 0.06,
                "eps": 3e-5,
                "beta1": 0.8,
                "beta2": 0.888,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "caching, pooling_mode, dims",
    [
        (True, DynamicEmbPoolingMode.NONE, [8, 8, 8]),
        (False, DynamicEmbPoolingMode.NONE, [16, 16, 16]),
        (False, DynamicEmbPoolingMode.SUM, [128, 32, 16]),
        (False, DynamicEmbPoolingMode.MEAN, [4, 8, 16]),
    ],
)
def test_backward(opt_type, opt_params, caching, pooling_mode, dims):
    print(f"step in test_backward , opt_type = {opt_type} opt_params = {opt_params}")
    assert torch.cuda.is_available()
    device_id = 0
    device = torch.device(f"cuda:{device_id}")

    table_names = ["table0", "table1", "table2"]
    key_type = torch.int64
    value_type = torch.float32

    max_capacity = 2048

    dyn_emb_table_options_list = []
    for dim in dims:
        dyn_emb_table_options = DynamicEmbTableOptions(
            dim=dim,
            init_capacity=max_capacity,
            max_capacity=max_capacity,
            index_type=key_type,
            embedding_dtype=value_type,
            device_id=device_id,
            evict_strategy=DynamicEmbEvictStrategy.LRU,
            caching=caching,
            local_hbm_for_values=1024**3,
        )
        dyn_emb_table_options_list.append(dyn_emb_table_options)

    feature_table_map = [0, 0, 1, 2]
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=table_names,
        table_options=dyn_emb_table_options_list,
        feature_table_map=feature_table_map,
        pooling_mode=pooling_mode,
        optimizer=opt_type,
        **opt_params,
    )
    POOLING_MODE: Dict[DynamicEmbPoolingMode, PoolingMode] = {
        DynamicEmbPoolingMode.NONE: PoolingMode.NONE,
        DynamicEmbPoolingMode.MEAN: PoolingMode.MEAN,
        DynamicEmbPoolingMode.SUM: PoolingMode.SUM,
    }
    OPTIM_TYPE: Dict[EmbOptimType, OptimType] = {
        EmbOptimType.SGD: OptimType.EXACT_SGD,
        EmbOptimType.ADAM: OptimType.ADAM,
    }
    num_embs = [max_capacity // 2 for d in dims]
    stbe = create_split_table_batched_embedding(
        table_names,
        feature_table_map,
        OPTIM_TYPE[opt_type],
        opt_params,
        dims,
        num_embs,
        POOLING_MODE[pooling_mode],
        device,
    )
    init_embedding_tables(stbe, bdeb)
    """
    feature number = 4, batch size = 2

    f0  [0,1],      [12],
    f1  [64,8],     [12],
    f2  [15, 2],    [7,105],
    f3  [],         [0]
    """
    for i in range(10):
        indices = torch.tensor(
            [0, 1, 12, 64, 8, 12, 15, 2, 7, 105, 0], device=device
        ).to(key_type)
        offsets = torch.tensor([0, 2, 3, 5, 6, 8, 10, 10, 11], device=device).to(
            key_type
        )

        embs_bdeb = bdeb(indices, offsets)
        embs_stbe = stbe(indices, offsets)

        torch.cuda.synchronize()
        with torch.no_grad():
            torch.testing.assert_close(embs_bdeb, embs_stbe, rtol=1e-06, atol=1e-06)

        loss = embs_bdeb.mean()
        loss.backward()
        loss_stbe = embs_stbe.mean()
        loss_stbe.backward()

        torch.cuda.synchronize()
        torch.testing.assert_close(loss, loss_stbe)

        print(f"Passed iteration {i}")
