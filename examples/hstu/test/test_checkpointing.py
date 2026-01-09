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

import os
import shutil

import commons.checkpoint as checkpoint
import commons.utils as init
import pytest
import torch
import torch.distributed as dist
from commons.distributed.finalize_model_grads import finalize_model_grads
from commons.utils.distributed_utils import collective_assert
from test_utils import assert_equal_two_state_dict, create_model


@pytest.mark.parametrize(
    "task_type",
    ["ranking", "retrieval"],
)
@pytest.mark.parametrize("contextual_feature_names", [["user0", "user1"], []])
@pytest.mark.parametrize("max_num_candidates", [10, 0])
@pytest.mark.parametrize("optimizer_type_str", ["adam", "sgd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpoint_model(
    task_type: str,
    contextual_feature_names,
    max_num_candidates,
    optimizer_type_str: str,
    dtype: torch.dtype,
):
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    if task_type == "retrieval" and max_num_candidates > 0:
        pytest.skip("skipping retrieval with no candidates")
    model, dense_optimizer, history_batches = create_model(
        task_type=task_type,
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        seed=1234,
    )
    # only train the model for 10 steps with the same batch
    model.train()
    for i in range(10):
        model.module.zero_grad_buffer()
        dense_optimizer.zero_grad()
        loss, _ = model(history_batches[i])
        collective_assert(not torch.isnan(loss).any(), f"iter {i} loss has nan: {loss}")

        loss.sum().backward()
        finalize_model_grads([model.module], None)
        dense_optimizer.step()

    new_model, new_dense_optimizer, _ = create_model(
        task_type=task_type,
        contextual_feature_names=contextual_feature_names,
        max_num_candidates=max_num_candidates,
        optimizer_type_str=optimizer_type_str,
        dtype=dtype,
        seed=2345,
    )

    save_path = "./gr_checkpoint"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    os.makedirs(save_path, exist_ok=True)

    checkpoint.save(save_path, model, dense_optimizer=dense_optimizer)

    checkpoint.load(save_path, new_model, dense_optimizer=new_dense_optimizer)

    model.eval()
    new_model.eval()
    from commons.checkpoint import get_unwrapped_module

    mp_ec = get_unwrapped_module(
        model
    )._embedding_collection  # ._model_parallel_embedding_collection
    new_model_unwrapped = get_unwrapped_module(new_model)._embedding_collection
    # check item embedding export
    item_keys_before, item_values_before = mp_ec.export_local_embedding("item")
    item_keys_after, item_values_after = new_model_unwrapped.export_local_embedding(
        "item"
    )
    item_keys_before = torch.from_numpy(item_keys_before).cuda()
    item_values_before = torch.from_numpy(item_values_before).cuda()
    item_keys_after = torch.from_numpy(item_keys_after).cuda()
    item_values_after = torch.from_numpy(item_values_after).cuda()

    sorted_item_keys_values_before, sorted_item_keys_indices_before = torch.sort(
        item_keys_before, dim=0
    )
    sorted_item_keys_values_after, sorted_item_keys_indices_after = torch.sort(
        item_keys_after, dim=0
    )

    all_emb_before, all_emb_after = (
        item_values_before[sorted_item_keys_indices_before],
        item_values_after[sorted_item_keys_indices_after],
    )
    assert torch.allclose(
        all_emb_before, all_emb_after
    ), f"[rank{dist.get_rank()}] item values should be the same"
    assert torch.allclose(
        sorted_item_keys_values_before, sorted_item_keys_values_after
    ), f"[rank{dist.get_rank()}] item keys should be the same"
    # batches are replicated
    batch = history_batches[0]
    kjt = batch.features
    output_before = mp_ec._model_parallel_embedding_collection(kjt).wait()["item_feat"]
    output_after = new_model_unwrapped._model_parallel_embedding_collection(kjt).wait()[
        "item_feat"
    ]

    assert torch.allclose(
        output_before.values(), output_after.values()
    ), f"[rank{dist.get_rank()}] output should be the same"

    for batch in history_batches:
        with torch.random.fork_rng():  # randomness negative sampling
            loss, _ = model(batch)
        with torch.random.fork_rng():
            new_loss, _ = new_model(batch)
        assert torch.allclose(
            loss, new_loss
        ), f"loaded model should have same output with original model {loss} vs. {new_loss}"

    assert_equal_two_state_dict(
        dense_optimizer.state_dict(), new_dense_optimizer.state_dict()
    )
    assert_equal_two_state_dict(
        new_dense_optimizer.state_dict(), dense_optimizer.state_dict()
    )

    # from commons.checkpoint import get_unwrapped_module
    # eval_module = get_unwrapped_module(model)
    # new_eval_module = get_unwrapped_module(new_model)
    # for batch in history_batches:
    #     eval_module.evaluate_one_batch(batch)
    #     new_eval_module.evaluate_one_batch(batch)
    # eval_result = eval_module.compute_metric()
    # new_eval_result = new_eval_module.compute_metric()

    # assert (
    #     eval_result == new_eval_result
    # ), "loaded model should have same eval result with original model"
    init.destroy_global_state()


from commons.modules.embedding import DataParallelEmbeddingCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.types import BoundsCheckMode, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def test_data_parallel_embedding_collection():
    init.initialize_distributed()
    init.initialize_model_parallel(1)

    embedding_configs = [
        EmbeddingConfig(
            name="item",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["item_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
        EmbeddingConfig(
            name="context",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["context_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
        EmbeddingConfig(
            name="action",
            embedding_dim=128,
            num_embeddings=10000,
            feature_names=["action_feat"],
            data_type=dtype_to_data_type(torch.float32),
        ),
    ]

    embedding_collection = EmbeddingCollection(
        tables=embedding_configs,
        device=torch.device("meta"),
    )
    constraints = {}
    for config in embedding_configs:
        constraints[config.name] = ParameterConstraints(
            sharding_types=[ShardingType.DATA_PARALLEL.value],
            bounds_check_mode=BoundsCheckMode.NONE,
        )
    planner = EmbeddingShardingPlanner(constraints=constraints)

    plan = planner.collective_plan(embedding_collection)
    sharding_plan = plan.plan[""]
    data_parallel_embedding_collection = DataParallelEmbeddingCollection(
        data_parallel_embedding_collection=embedding_collection,
        data_parallel_sharding_plan=sharding_plan,
        env=ShardingEnv.from_process_group(dist.group.WORLD),
        device=torch.device("cuda"),
    )

    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["item_feat", "action_feat", "user0", "user1", "context_feat"],
        lengths=torch.tensor([5, 10, 15, 20, 25]),
        values=torch.randint(0, 10000, (135,)),
    ).to(torch.device("cuda"))
    output = data_parallel_embedding_collection(kjt)
    assert "item_feat" in output, "item_feat should be in output"
    assert "action_feat" in output, "action_feat should be in output"
    assert "user0" not in output, "user0 should not be in output"
    assert "user1" not in output, "user1 should not be in output"
    assert "context_feat" in output, "context_feat should be in output"
    for feature_name, table_name in zip(
        ["item_feat", "action_feat", "context_feat"], ["item", "action", "context"]
    ):
        weights = data_parallel_embedding_collection.embedding_weights[table_name].data
        feature = kjt[feature_name]
        res_embedding = output[feature_name]
        ref_embedding = weights[feature.values().long(), :]
        assert torch.allclose(
            feature.lengths(), res_embedding.lengths()
        ), f"lengths of {feature_name} should be the same"
        assert torch.allclose(
            ref_embedding, res_embedding.values()
        ), f"values of {feature_name} should be the same"
