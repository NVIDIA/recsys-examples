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


from typing import List, Optional, Union

import commons.utils as init
import configs
import dataset
import model
import torch
from commons.utils.distributed_utils import collective_assert
from commons.utils.hstu_assert_close import hstu_close
from configs import HSTULayerType, KernelBackend, OptimizerParam
from distributed.sharding import apply_megatron_ddp, make_optimizer_and_shard
from dynamicemb import DynamicEmbTableOptions
from megatron.core import parallel_state, tensor_parallel
from modules.debug.debug_hstu_layer import HSTULayer as DebugHSTULayer
from modules.jagged_data import JaggedData
from modules.native_hstu_layer import HSTULayer
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)

debug_module_path_to_tpN_module_path = {
    "_output_layernorm_weight": "_output_ln_dropout_mul.weight",
    "_output_layernorm_bias": "_output_ln_dropout_mul.bias",
}


def get_batch_on_this_tp_rank(batch: JaggedData):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    _broadcast(batch.values)
    _broadcast(batch.seqlen)
    _broadcast(batch.seqlen_offsets)
    _broadcast(batch.max_seqlen)
    _broadcast(batch.num_candidates)
    _broadcast(batch.num_candidates_offsets)
    _broadcast(batch.contextual_seqlen)
    _broadcast(batch.contextual_seqlen_offsets)
    return batch


def get_diff_tensor(tensor1, tensor2, threshold=1e-5):
    diff_abs = torch.abs(tensor1 - tensor2)
    diff_index = torch.nonzero(diff_abs > threshold, as_tuple=False)
    diff_tensor = tensor1[diff_abs > threshold] - tensor2[diff_abs > threshold]
    return diff_index, diff_tensor


def init_fused_weights_from_debug(
    debug_module,
    fused_module,
    num_heads,
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
        # fused module has different layout for linear_uvqk weight
        if re.match(r".*_linear_uvqk.weight$", name):
            input_size = src_data.size(0)
            output_size = src_data.size(1)
            src_data = (
                src_data.reshape(input_size, num_heads, 4, -1)
                .transpose(1, 2)
                .reshape(input_size, output_size)
            )
        if param.requires_grad:
            fused_module.state_dict()[fused_accessor].data.copy_(src_data)


def get_tp_slice(tensor: Optional[torch.Tensor], mode="row"):
    if tensor is None:
        return None
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    if mode == "row":
        tp_slice_start = tp_rank * tensor.size(0) // tp_size
        tp_slice_end = (tp_rank + 1) * tensor.size(0) // tp_size
        return tensor[tp_slice_start:tp_slice_end, ...]
    elif mode == "col":
        tp_slice_start = tp_rank * tensor.size(1) // tp_size
        tp_slice_end = (tp_rank + 1) * tensor.size(1) // tp_size
        return tensor[:, tp_slice_start:tp_slice_end]
    else:
        raise ValueError(f"mode {mode} is not supported")


# TODO: Add get_tp_slice for optimizer state.
def compare_tpN_to_debug_optimizer_state(
    tpN_optimizer, debug_optimizer, debug_fp32_optimizer
):
    tp_param_groups = tpN_optimizer.chained_optimizers[0].state_dict()["optimizer"][
        "param_groups"
    ]
    debug_param_groups = debug_optimizer.chained_optimizers[0].state_dict()[
        "optimizer"
    ]["param_groups"]
    debug_fp32_param_groups = debug_fp32_optimizer.chained_optimizers[0].state_dict()[
        "param_groups"
    ]
    for i in range(len(tp_param_groups)):
        tp_param_group = tp_param_groups[i]
        debug_param_group = debug_param_groups[i]
        debug_fp32_param_group = debug_fp32_param_groups[i]

        assert (
            tp_param_group["step"] == debug_param_group["step"]
        ), f'step mismatch {tp_param_group["step"]} vs {debug_param_group["step"]}'
        assert (
            tp_param_group["step"] == debug_fp32_param_group["step"]
        ), f'step mismatch {tp_param_group["step"]} vs {debug_fp32_param_group["step"]}'


def compare_tpN_to_debug_weights(
    tpN_module, debug_module, debug_fp32_module, include_grad: bool = True
):
    import re

    tpN_module_params_map = dict(tpN_module.named_parameters())
    tpN_module.state_dict()

    debug_module_params_map = dict(debug_module.named_parameters())
    debug_module_state_dict = debug_module.state_dict()

    debug_fp32_module_state_dict = debug_fp32_module.state_dict()
    debug_fp32_module_params_map = dict(debug_fp32_module.named_parameters())
    for name, param in debug_module_params_map.items():
        src = param if not isinstance(param, ShardedTensor) else param.local_tensor()
        src_fp32 = debug_fp32_module_params_map[name]
        src_fp32 = (
            src_fp32
            if not isinstance(src_fp32, ShardedTensor)
            else src_fp32.local_tensor()
        )
        src_grad = None
        src_grad_fp32 = None
        # col parallel linear weight, weight is sliced along row
        if re.match(r".*_linear_uvqk.weight$", name):
            src_grad = get_tp_slice(getattr(src, "main_grad", None), "row")
            src = get_tp_slice(src, "row")
            src_grad_fp32 = get_tp_slice(getattr(src_fp32, "main_grad", None), "row")
            src_fp32 = get_tp_slice(src_fp32, "row")
        # row wise linear weight, weight is sliced along col
        elif re.match(r".*_linear_proj.weight$", name):
            src_grad = get_tp_slice(getattr(src, "main_grad", None), "col")
            src = get_tp_slice(src, "col")
            src_grad_fp32 = get_tp_slice(getattr(src_fp32, "main_grad", None), "col")
            src_fp32 = get_tp_slice(src_fp32, "col")
        # output layernorm weight and bias are TP split
        # colparallel linear bias is also TP split when config.use_cpu_initialization is True
        # see https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1104, https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1037
        elif re.match(r".*_linear_uvqk.bias$", name):
            src_grad = get_tp_slice(getattr(src, "main_grad", None), "row")
            src = get_tp_slice(src, "row")
            src_grad_fp32 = get_tp_slice(getattr(src_fp32, "main_grad", None), "row")
            src_fp32 = get_tp_slice(src_fp32, "row")
        else:
            src_grad = getattr(src, "main_grad", None)
            src_grad_fp32 = getattr(src_fp32, "main_grad", None)

        if re.match(r".*_output_layernorm.*$", name):
            child_name = name.split(".")[-1]
            name = name.replace(
                child_name, debug_module_path_to_tpN_module_path[child_name]
            )

        dst = tpN_module_params_map[name]
        dst_grad = getattr(dst, "main_grad", None)
        # model parallel embedding table weight is a TableBatchedEmbeddingSlice, which has no grad
        if isinstance(dst, TableBatchedEmbeddingSlice):
            src_grad = None
            src_grad_fp32 = None
            dst_grad = None
            src = debug_module_state_dict[name].local_tensor()
            src_fp32 = debug_fp32_module_state_dict[name].local_tensor()
            dst = dst.data
        if include_grad and all(
            x is not None for x in [dst_grad, src_grad, src_grad_fp32]
        ):
            collective_assert(
                hstu_close(dst_grad, src_grad, src_grad_fp32, multiplier=5),
                f"grad mismatch at {name}, multiplier {(dst_grad - src_grad_fp32).abs().max() / (src_grad - src_grad_fp32).abs().max()}",
                group=parallel_state.get_data_parallel_group(),
            )
        collective_assert(
            hstu_close(dst, src, src_fp32, try_allclose=True, multiplier=5),
            f"weight mismatch at {name}  multiplier {(dst - src_fp32).abs().max() / (src - src_fp32).abs().max()}",
            group=parallel_state.get_data_parallel_group(),
        )  # weight


# allgather weights from tp1 to tpN (slice tp1 to tpN)
# num_heads is required to do the transpose correctly
def init_tpN_weights_from_debug(
    debug_module,
    tpN_module,
):
    import re

    for name, param in debug_module.state_dict().items():
        src = (
            param.data if not isinstance(param, ShardedTensor) else param.local_tensor()
        )
        # col parallel linear weight
        if re.match(r".*_linear_uvqk.weight$", name):
            src = get_tp_slice(src, "row")
        # row wise linear weight
        elif re.match(r".*_linear_proj.weight$", name):
            src = get_tp_slice(src, "col")
        # output layernorm weight and bias are TP split
        # colparallel linear bias is also TP split when config.use_cpu_initialization is True
        # see https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1104, https://github.com/NVIDIA/TransformerEngine/blob/v2.4/transformer_engine/pytorch/module/linear.py#L1037
        elif re.match(r".*_linear_uvqk.bias$", name):
            src = get_tp_slice(src, "row")
        elif re.match(r".*_output_layernorm.*$", name):
            child_name = name.split(".")[-1]
            name = name.replace(
                child_name, debug_module_path_to_tpN_module_path[child_name]
            )
        dst = tpN_module.state_dict()[name]
        # embedding table weight is a ShardedTensor
        if isinstance(dst, ShardedTensor):
            dst.local_tensor().data.copy_(src)
        else:
            dst.data.copy_(src)


def init_module_from(src_module, dst_module):
    for name, param in src_module.state_dict().items():
        src = param if not isinstance(param, ShardedTensor) else param.local_tensor()
        dst = dst_module.state_dict()[name]
        if isinstance(dst, ShardedTensor):
            dst.local_tensor().data.copy_(src)
        else:
            dst.data.copy_(src)


def zero_bias(modules: Union[torch.nn.Module, List[torch.nn.Module]]):
    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    for module in modules:
        for name, param in module.named_parameters():
            if name.endswith("bias"):
                param.data.zero_()


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
    num_heads: int = 4,
    *,
    seed: int,
    hstu_layer_type: HSTULayerType = HSTULayerType.DEBUG,
    kernel_backend: KernelBackend = KernelBackend.CUTLASS,
    num_batches: int = 10,
    replicate_batches: bool = True,
):
    init.set_random_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    embdim = 128
    batch_size = 128
    if dtype == torch.float32:
        assert kernel_backend == KernelBackend.PYTORCH, "only pytorch supports float32"
    hstu_config = configs.get_hstu_config(
        hidden_size=embdim,
        kv_channels=32,
        num_attention_heads=num_heads,
        num_layers=1,
        hidden_dropout=0.0,  # disable dropout
        dtype=dtype,
        hstu_layer_type=hstu_layer_type,
        kernel_backend=kernel_backend,
        add_uvqk_bias=False,  # disable bias for better debugging
        fuse_norm_mul_dropout=False,  # disable fusion for better debugging
        learnable_input_layernorm=False,  # disable bias for better debugging
    )

    item_feature_name = "item_feat"
    action_feature_name = "action_feat"
    contextual_emb_size = 1000
    item_emb_size = 1024 * 1024
    action_vocab_size = 16
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
        batch_size=batch_size,
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
            prediction_head_arch=[num_tasks],  # single Linear for better debugging
            prediction_head_bias=False,  # disable bias for better debugging
        )
        model_train = model.RankingGR(hstu_config=hstu_config, task_config=task_config)

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            if replicate_batches:
                history_batches = [
                    dataset.utils.RankingBatch.random(
                        num_tasks=num_tasks, **batch_kwargs
                    )
                ] * num_batches
            else:
                history_batches = [
                    dataset.utils.RankingBatch.random(
                        num_tasks=num_tasks, **batch_kwargs
                    )
                    for _ in range(num_batches)
                ]
    else:
        assert task_type == "retrieval"
        task_config = configs.RetrievalConfig(embedding_configs=emb_configs)
        model_train = model.RetrievalGR(
            hstu_config=hstu_config, task_config=task_config
        )

        history_batches = []
        with tensor_parallel.get_cuda_rng_tracker().fork():
            if replicate_batches:
                history_batches = [
                    dataset.utils.RetrievalBatch.random(**batch_kwargs)
                ] * num_batches
            else:
                history_batches = [
                    dataset.utils.RetrievalBatch.random(**batch_kwargs)
                    for _ in range(num_batches)
                ]
    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3 if optimizer_type_str == "adam" else 1e-1,
        adam_beta1=0.5,  # larger beta1 for better debugging!
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,  # decay is off for better debugging
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


def create_hstu_layer_and_optimizer(
    dtype: torch.dtype,
    hidden_size: int,
    num_attention_heads: int,
    kv_channels: int,
    optimizer_type_str: str,
    hstu_layer_type: HSTULayerType = HSTULayerType.DEBUG,
    kernel_backend: KernelBackend = KernelBackend.CUTLASS,
    learnable_input_layernorm: bool = False,
):
    hstu_config = configs.get_hstu_config(
        hidden_size=hidden_size,
        kv_channels=kv_channels,
        num_attention_heads=num_attention_heads,
        num_layers=1,
        dtype=dtype,
        hidden_dropout=0.0,
        norm_epsilon=1e-5,
        is_causal=True,
        kernel_backend=kernel_backend,  # attn_backend
        target_group_size=1,
        hstu_layer_type=hstu_layer_type,
        learnable_input_layernorm=learnable_input_layernorm,
        residual=True,
        add_uvqk_bias=False,  # disable bias for better debugging
        fuse_norm_mul_dropout=False,  # disable fusion for better debugging
    )
    torch.cuda.current_device()
    if hstu_layer_type == HSTULayerType.DEBUG:
        hstu_layer = DebugHSTULayer(hstu_config).cuda()
    else:
        hstu_layer = HSTULayer(hstu_config).cuda()

    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3 if optimizer_type_str == "adam" else 1e-1,
        adam_beta1=0.5,  # larger beta1 for better debugging!
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,  # decay is off for better debugging
    )
    model, dense_optimizer = apply_megatron_ddp(
        hstu_layer,
        hstu_config,
        optimizer_param,
        torch.device("cuda", torch.cuda.current_device()),
    )

    return model, dense_optimizer
