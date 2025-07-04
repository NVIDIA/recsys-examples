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


from typing import Optional

import commons.utils as init
import configs
import dataset
import model
import torch
from configs import HSTULayerType, OptimizerParam
from distributed.sharding import make_optimizer_and_shard
from dynamicemb import DynamicEmbTableOptions
from megatron.core import parallel_state, tensor_parallel
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.fused_hstu_layer import FusedHSTULayer
from modules.native_hstu_layer import HSTULayer as NativeHSTULayer
from torch.distributed._shard.sharded_tensor import ShardedTensor

debug_module_path_to_tpN_module_path = {
    "_output_layernorm_weight": "_output_ln_dropout_mul.weight",
    "_output_layernorm_bias": "_output_ln_dropout_mul.bias",
}


def collective_assert_tensor(
    tensor: torch.Tensor,
    compare_type: str = "equal",
    pg: Optional[torch.distributed.ProcessGroup] = None,
):
    cur_rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_tensors, tensor.contiguous(), group=pg)
    torch.distributed.barrier(group=pg)

    for i in range(world_size):
        if i == cur_rank:
            continue

        if compare_type == "equal":
            assert torch.equal(
                tensor, gathered_tensors[i]
            ), f"rank {cur_rank} and rank {i} tensor are not equal"
        elif compare_type == "not_equal":
            assert not torch.equal(
                tensor, gathered_tensors[i]
            ), f"rank {cur_rank} and rank {i} tensor are equal"
        elif compare_type == "close":
            assert torch.allclose(
                tensor, gathered_tensors[i]
            ), f"rank {cur_rank} and rank {i} tensor are not close"
        elif compare_type == "not_close":
            assert not torch.allclose(
                tensor, gathered_tensors[i]
            ), f"rank {cur_rank} and rank {i} tensor are close"
        else:
            raise ValueError(f"compare_type {compare_type} is not supported")


def init_fused_weights_from_debug(
    debug_module: DebugHSTULayer, fused_module: FusedHSTULayer
):
    import re

    for name, param in debug_module.named_parameters():
        # linear layer weight is transposed in the fused module
        fused_accessor = name.replace(".weight", "_weight").replace(".bias", "_bias")
        src_data = (
            param.data.t()
            if re.match(r".*linear\w*_weight$", fused_accessor)
            else param.data
        )
        if param.requires_grad:
            fused_module.state_dict()[fused_accessor].data.copy_(src_data)


# allgather weights from tp1 to tpN (slice tp1 to tpN)
def init_tpN_weights_from_debug(
    legacy_module: DebugHSTULayer, tpN_module: NativeHSTULayer
):
    import re

    for name, param in legacy_module.state_dict().items():
        src = param.data
        # col parallel linear weight
        if re.match(r".*_linear_uvqk.weight$", name):
            # TP u,v,q,k layout is different from native
            # (output_size, input_size) => (num_heads, sum(split_arg_list), input_size) => transpose(0, 1) => (sum(split_arg_list), num_heads, input_size) => reshape(-1, input_size)
            # split_arg_list = legacy_module._split_arg_list
            output_size = src.size(0)
            input_size = src.size(1)
            src = (
                src.view(4, legacy_module._num_heads, -1, input_size)
                .transpose(0, 1)
                .reshape(output_size, input_size)
            )
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_slice = tp_rank * output_size // tp_size
            tp_slice_end = (tp_rank + 1) * output_size // tp_size
            src = src[tp_slice:tp_slice_end, :]
        # row wise linear weight
        elif re.match(r".*_linear_proj.weight$", name):
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            input_size = src.size(1)
            tp_slice = tp_rank * input_size // tp_size
            tp_slice_end = (tp_rank + 1) * input_size // tp_size
            src = src[:, tp_slice:tp_slice_end]
        # output layernorm weight and bias are TP split
        # colparallel linear bias is also TP split when config.use_cpu_initialization is True
        # see https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1104, https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1037
        elif re.match(r".*_linear_uvqk.bias$", name):
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_slice = tp_rank * param.shape[0] // tp_size
            tp_slice_end = (tp_rank + 1) * param.shape[0] // tp_size
            src = src[tp_slice:tp_slice_end, ...]
        elif re.match(r".*_output_layernorm.*$", name):
            child_name = name.split(".")[-1]
            name = name.replace(
                child_name, debug_module_path_to_tpN_module_path[child_name]
            )
        dst = tpN_module.state_dict()[name]
        dst.data.copy_(src)


def _flatten_state_dict(state_dict):
    search_list = [("", state_dict)]

    while len(search_list) > 0:
        prefix, s = search_list.pop()
        if isinstance(s, list):
            search_list.extend([(i, v) for i, v in enumerate(s)])
            continue
        if isinstance(s, dict):
            for name, v in s.items():
                subname = str(prefix) + ("." if prefix else "") + str(name)
                search_list.append((subname, v))
            continue
        yield prefix, s


def assert_equal_two_state_dict(a_state_dict, b_state_dict):
    flatten_a_state_dict = dict(_flatten_state_dict(a_state_dict))
    flatten_b_state_dict = dict(_flatten_state_dict(b_state_dict))
    for k, v in flatten_a_state_dict.items():
        assert k in flatten_b_state_dict, f"{k} not loadded"
        r = flatten_b_state_dict[k]
        if isinstance(v, torch.Tensor):
            if isinstance(v, ShardedTensor):
                v = v.local_tensor()
                r = r.local_tensor()
            assert torch.allclose(v, r), f"for {k}, tensor {v} != {r}"
        else:
            assert v == r, f"for {k}, value {v} != {r}"


def create_model(
    task_type,
    contextual_feature_names,
    max_num_candidates,
    optimizer_type_str: str,
    dtype: torch.dtype,
    pipeline_type: str = "none",
    use_dynamic_emb: bool = True,
    *,
    seed: int,
    hstu_layer_type: HSTULayerType = HSTULayerType.DEBUG,
):
    init.set_random_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    embdim = 8
    hstu_config = configs.get_hstu_config(
        hidden_size=embdim,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=1,
        hidden_dropout=0.0,  # disable dropout
        dtype=dtype,
        hstu_layer_type=hstu_layer_type,
    )

    item_feature_name = "item_feat"
    action_feature_name = "action_feat"
    contextual_emb_size = 1000
    item_emb_size = 1024 * 1024
    action_vocab_size = 1000
    emb_configs = [
        configs.ShardedEmbeddingConfig(
            feature_names=[action_feature_name],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=embdim,
            sharding_type="data_parallel",
        ),
        configs.ShardedEmbeddingConfig(
            feature_names=[item_feature_name],
            table_name="item",
            vocab_size=item_emb_size,
            dim=embdim,
            sharding_type="model_parallel",
        ),
    ]
    feature_configs = [
        dataset.utils.FeatureConfig(
            feature_names=[item_feature_name, action_feature_name],
            max_item_ids=[
                max(item_emb_size // 2, 1),
                action_vocab_size,
            ],  # halve the max ids to `minimize` eviction
            max_sequence_length=100,
            is_jagged=True,
        )
    ]
    if len(contextual_feature_names) > 0:
        feature_configs.append(
            dataset.utils.FeatureConfig(
                feature_names=contextual_feature_names,
                max_item_ids=[
                    contextual_emb_size for _ in range(len(contextual_feature_names))
                ],
                max_sequence_length=10,
                is_jagged=True,
            )
        )
        emb_configs.append(
            configs.ShardedEmbeddingConfig(
                feature_names=contextual_feature_names,
                table_name="context",
                vocab_size=contextual_emb_size,
                dim=embdim,
                sharding_type="model_parallel",
            )
        )

    batch_kwargs = dict(
        batch_size=32,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        device=device,
    )
    if task_type == "ranking":
        num_tasks = 1
        task_config = configs.RankingConfig(
            embedding_configs=emb_configs,
            prediction_head_arch=[128, 10, num_tasks],
        )
        model_train = model.RankingGR(hstu_config=hstu_config, task_config=task_config)

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = dataset.utils.RankingBatch.random(
                num_tasks=num_tasks, **batch_kwargs
            )
            for i in range(10):
                history_batches.append(batch)
    else:
        assert task_type == "retrieval"
        task_config = configs.RetrievalConfig(embedding_configs=emb_configs)
        model_train = model.RetrievalGR(
            hstu_config=hstu_config, task_config=task_config
        )

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            batch = dataset.utils.RetrievalBatch.random(**batch_kwargs)
            for i in range(10):
                history_batches.append(batch)
    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3,
    )
    from dynamicemb import DynamicEmbScoreStrategy

    model_train, dense_optimizer = make_optimizer_and_shard(
        model_train,
        config=hstu_config,
        dynamicemb_options_dict={
            "item": DynamicEmbTableOptions(
                global_hbm_for_values=0,
                score_strategy=DynamicEmbScoreStrategy.STEP,
            )
        }
        if use_dynamic_emb
        else {},
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        pipeline_type=pipeline_type,
        device=device,
    )
    return model_train, dense_optimizer, history_batches
