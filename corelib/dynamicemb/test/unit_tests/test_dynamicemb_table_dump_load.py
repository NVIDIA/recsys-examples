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

import sys

import pytest
import torch
import torch.distributed as dist
from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
from dynamicemb.dump_load import (
    export_keys_values,
    load_key_values,
    local_export,
    local_load,
)
from dynamicemb.dynamicemb_config import DynamicEmbCheckMode
from dynamicemb.key_value_table import KeyValueTable
from dynamicemb.optimizer import *
from dynamicemb_extensions import EvictStrategy, OptimizerType, find
from torchrec.distributed.comm import get_local_rank

backend = "nccl"
dist.init_process_group(backend=backend)

local_rank = get_local_rank()
dist.get_world_size()
torch.cuda.set_device(local_rank)

device = torch.device(f"cuda:{local_rank}")


def init_dynamicemb_table(
    dynamicemb_table: KeyValueTable, num_embeddings: int, embedding_dim: int
):
    # Generate exactly num_embeddings unique random keys in range 0 to sys.maxsize
    # Use torch.randint to sample from the large range without creating a massive tensor
    keys = torch.randint(
        0, sys.maxsize, (num_embeddings,), dtype=torch.int64, device=device
    )
    # Ensure uniqueness by removing duplicates and adding more if needed
    keys = torch.unique(keys)
    while len(keys) < num_embeddings:
        additional_keys = torch.randint(
            0,
            sys.maxsize,
            (num_embeddings - len(keys),),
            dtype=torch.int64,
            device=device,
        )
        keys = torch.cat([keys, additional_keys])
        keys = torch.unique(keys)
    keys = keys[:num_embeddings]  # Take exactly num_embeddings keys

    embeddings = torch.randn(num_embeddings, embedding_dim, device=device)

    scores = (
        torch.randint(
            0, sys.maxsize, (num_embeddings,), dtype=torch.uint64, device=device
        )
        if dynamicemb_table.evict_strategy() != EvictStrategy.KLru
        else None
    )

    opt_states = (
        torch.randn(num_embeddings, dynamicemb_table.optim_state_dim(), device=device)
        if dynamicemb_table.optim_state_dim() > 0
        else None
    )

    load_key_values(
        dynamicemb_table,
        keys=keys,
        embeddings=embeddings,
        scores=scores,
        opt_states=opt_states,
    )
    return keys, embeddings, scores, opt_states


def assert_two_dynamicemb_table_equal(
    reference_table: KeyValueTable,
    reference_table_optimizer_type: str,
    table: KeyValueTable,
    table_optimizer_type: str,
):
    table_data_iterator = export_keys_values(table, device)
    for keys, embeddings, opt_states, scores in table_data_iterator:
        dim = reference_table.embedding_dim()
        optstate_dim = reference_table.optim_state_dim()
        value_type = reference_table.value_type()
        values = torch.empty(
            keys.numel() * (dim + optstate_dim),
            device=device,
            dtype=value_type,
        )
        founds = torch.empty(keys.numel(), device=device, dtype=torch.bool)
        scores = torch.empty(keys.numel(), device=device, dtype=torch.uint64)

        find(
            reference_table.table,
            keys.numel(),
            keys,
            values,
            founds,
            scores,
        )
        assert torch.allclose(
            founds, torch.ones_like(founds)
        ), "missing keys in reference table"

        reference_values = values.reshape(-1, dim + optstate_dim).to(embeddings.dtype)
        reference_embeddings = reference_values[:, :dim]
        torch.testing.assert_close(embeddings, reference_embeddings)

        if (
            reference_table_optimizer_type == table_optimizer_type
            and table.optim_state_dim() > 0
            and optstate_dim > 0
        ):
            reference_opt_states = reference_values[:, dim:]
            torch.testing.assert_close(opt_states, reference_opt_states)


def create_dummy_optimizer(optimizer_type, embedding_dtype):
    optimizer_args = OptimizerArgs(
        learning_rate=0.3,
        weight_decay=0.06,
        eps=3e-5,
        beta1=0.8,
        beta2=0.888,
        initial_accumulator_value=0.0,
    )
    if optimizer_type == "sgd":
        return SGDDynamicEmbeddingOptimizerV2(
            optimizer_args,
        )
    elif optimizer_type == "adam":
        return AdamDynamicEmbeddingOptimizerV2(
            optimizer_args,
        )
    elif optimizer_type == "adagrad":
        return AdaGradDynamicEmbeddingOptimizerV2(
            optimizer_args,
        )
    elif optimizer_type == "rowwise_adagrad":
        return RowWiseAdaGradDynamicEmbeddingOptimizerV2(
            optimizer_args,
            embedding_dtype,
        )
    else:
        raise ValueError(f"Not supported optimizer type.")


def create_table_options(
    key_type: str,
    value_type: str,
    score_type: str,
    optimizer_type: str,
    score_strategy: str,
    mode: str,
    num_embeddings: int,
    embedding_dim: int,
):
    if key_type == "int64":
        key_type = torch.int64
    elif key_type == "uint64":
        key_type = torch.uint64
    else:
        raise ValueError(f"Invalid key type: {key_type}")

    if value_type == "float32":
        value_type = torch.float32
    elif value_type == "float16":
        value_type = torch.float16
    elif value_type == "bfloat16":
        value_type = torch.bfloat16
    else:
        raise ValueError(f"Invalid value type: {value_type}")

    if score_type == "uint64":
        score_type = torch.uint64
    else:
        raise ValueError(f"Invalid score type: {score_type}")

    if optimizer_type == "sgd":
        optimizer_type = OptimizerType.SGD
    elif optimizer_type == "adam":
        optimizer_type = OptimizerType.Adam
    elif optimizer_type == "adagrad":
        optimizer_type = OptimizerType.AdaGrad
    elif optimizer_type == "rowwise_adagrad":
        optimizer_type = OptimizerType.RowWiseAdaGrad
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

    if score_strategy == "timestamp":
        score_strategy = DynamicEmbScoreStrategy.TIMESTAMP
    elif score_strategy == "step":
        score_strategy = DynamicEmbScoreStrategy.STEP
    elif score_strategy == "custimized":
        score_strategy = DynamicEmbScoreStrategy.CUSTOMIZED
    else:
        raise ValueError(f"Invalid score strategy: {score_strategy}")

    if mode == "training":
        training = True
    elif mode == "evaluation":
        training = False
    else:
        raise ValueError(f"Invalid mode: {mode}")

    table_options = DynamicEmbTableOptions(
        index_type=key_type,
        embedding_dtype=value_type,
        score_type=score_type,
        optimizer_type=optimizer_type,
        dim=embedding_dim,
        init_capacity=num_embeddings * 100,
        max_capacity=num_embeddings * 100,
        score_strategy=score_strategy,
        training=training,
        local_hbm_for_values=0,
        device_id=0,
        safe_check_mode=DynamicEmbCheckMode.ERROR,
    )
    return table_options


@pytest.mark.parametrize("key_type", ["int64"])
@pytest.mark.parametrize("value_type", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("score_type", ["uint64"])
@pytest.mark.parametrize("score_strategy", ["timestamp", "step"])
@pytest.mark.parametrize(
    "dump_optimizer_type, load_optimizer_type",
    [
        ("sgd", "sgd"),
        ("adam", "adam"),
        ("adagrad", "adagrad"),
        ("rowwise_adagrad", "rowwise_adagrad"),
        ("sgd", "adam"),
        ("adam", "rowwise_adagrad"),
    ],
)
@pytest.mark.parametrize("dump_mode", ["training", "evaluation"])
@pytest.mark.parametrize("load_mode", ["training", "evaluation"])
@pytest.mark.parametrize("num_embeddings", [10])
@pytest.mark.parametrize("embedding_dim", [16])
def test_dynamic_table_load_dump(
    key_type: str,
    value_type: str,
    score_type: str,
    score_strategy: str,
    dump_optimizer_type: str,
    load_optimizer_type: str,
    dump_mode: str,
    load_mode: str,
    num_embeddings: int,
    embedding_dim: int,
) -> None:
    dump_table_options = create_table_options(
        key_type,
        value_type,
        score_type,
        dump_optimizer_type,
        score_strategy,
        dump_mode,
        num_embeddings,
        embedding_dim,
    )
    load_table_options = create_table_options(
        key_type,
        value_type,
        score_type,
        load_optimizer_type,
        score_strategy,
        load_mode,
        num_embeddings,
        embedding_dim,
    )

    print("dump_table_options", dump_table_options)
    print("load_table_options", load_table_options)
    print("num_embeddings", num_embeddings)
    print("embedding_dim", embedding_dim)

    dump_dummy_optimizer = create_dummy_optimizer(
        dump_optimizer_type if dump_mode == "training" else "sgd",
        dump_table_options.embedding_dtype,
    )
    dynamicemb_table = KeyValueTable(dump_table_options, dump_dummy_optimizer)
    keys, embeddings, scores, opt_states = init_dynamicemb_table(
        dynamicemb_table, num_embeddings, embedding_dim
    )

    local_export(
        dynamicemb_table,
        "emb_key_path",
        "embedding_file_path",
        "score_file_path" if scores is not None else None,
        "opt_file_path" if opt_states is not None else None,
    )

    load_dummy_optimizer = create_dummy_optimizer(
        load_optimizer_type if load_mode == "training" else "sgd",
        load_table_options.embedding_dtype,
    )
    new_dynamicemb_table = KeyValueTable(load_table_options, load_dummy_optimizer)

    need_load_optimizer = (
        opt_states is not None
        and dump_optimizer_type == load_optimizer_type
        and load_mode == "training"
    )

    local_load(
        new_dynamicemb_table,
        "emb_key_path",
        "embedding_file_path",
        "score_file_path" if scores is not None else None,
        "opt_file_path" if need_load_optimizer else None,
    )

    assert_two_dynamicemb_table_equal(
        dynamicemb_table, dump_optimizer_type, new_dynamicemb_table, load_optimizer_type
    )


dist.destroy_process_group()
