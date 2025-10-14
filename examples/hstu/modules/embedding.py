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

# pyre-strict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.fx
import torch.nn as nn
from commons.utils.nvtx_op import output_nvtx_hook, register_setter_and_getter_for_nvtx
from configs.task_config import ShardedEmbeddingConfig
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.distributed.utils import (
    add_params_from_parameter_sharding,
    convert_to_fbgemm_types,
    merge_fused_params,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    EmbeddingTableConfig,
    PoolingType,
    dtype_to_data_type,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def create_data_parallel_sharding_infos_by_sharding(
    module: EmbeddingCollectionInterface,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    fused_params: Optional[Dict[str, Any]],
) -> List[EmbeddingShardingInfo]:
    if fused_params is None:
        fused_params = {}

    sharding_type_to_sharding_infos: List[EmbeddingShardingInfo] = []
    # state_dict returns parameter.Tensor, which loses parameter level attributes
    parameter_by_name = dict(module.named_parameters())
    # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it    there
    state_dict = module.state_dict()

    for (
        config,
        embedding_names,
    ) in zip(module.embedding_configs(), module.embedding_names_by_table()):
        table_name = config.name
        assert table_name in table_name_to_parameter_sharding

        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel != EmbeddingComputeKernel.DENSE.value:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )

        param_name = "embeddings." + config.name + ".weight"
        assert param_name in parameter_by_name or param_name in state_dict
        param = parameter_by_name.get(param_name, state_dict[param_name])

        optimizer_params = getattr(param, "_optimizer_kwargs", [{}])
        optimizer_classes = getattr(param, "_optimizer_classes", [None])

        assert (
            len(optimizer_classes) == 1 and len(optimizer_params) == 1
        ), f"Only support 1 optimizer, given {len(optimizer_classes)}"

        optimizer_class = optimizer_classes[0]
        optimizer_params = optimizer_params[0]
        if optimizer_class:
            optimizer_params["optimizer"] = optimizer_type_to_emb_opt_type(
                optimizer_class
            )
        per_table_fused_params = merge_fused_params(fused_params, optimizer_params)
        per_table_fused_params = add_params_from_parameter_sharding(
            per_table_fused_params, parameter_sharding
        )
        per_table_fused_params = convert_to_fbgemm_types(per_table_fused_params)

        sharding_type_to_sharding_infos.append(
            (
                EmbeddingShardingInfo(
                    embedding_config=EmbeddingTableConfig(
                        num_embeddings=config.num_embeddings,
                        embedding_dim=config.embedding_dim,
                        name=config.name,
                        data_type=config.data_type,
                        feature_names=copy.deepcopy(config.feature_names),
                        pooling=PoolingType.NONE,
                        is_weighted=False,
                        has_feature_processor=False,
                        embedding_names=embedding_names,
                        weight_init_max=config.weight_init_max,
                        weight_init_min=config.weight_init_min,
                    ),
                    param_sharding=parameter_sharding,
                    param=param,
                    fused_params=per_table_fused_params,
                )
            )
        )
    return sharding_type_to_sharding_infos


# TODO add wgrad allreduce sum across tp ranks in finalize model grads!
class DataParallelEmbeddingCollection(torch.nn.Module):
    """
    Sharded implementation of `EmbeddingCollection`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        data_parallel_embedding_collection: EmbeddingCollection,
        data_parallel_sharding_plan,
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = data_parallel_embedding_collection.embedding_dim()
        self._embedding_configs: List[
            EmbeddingConfig
        ] = data_parallel_embedding_collection.embedding_configs()
        self._table_names: List[str] = [
            config.name for config in self._embedding_configs
        ]
        self._table_name_to_config: Dict[str, EmbeddingConfig] = {
            config.name: config for config in self._embedding_configs
        }
        self._feature_names: List[str] = [
            feature
            for config in self._embedding_configs
            for feature in config.feature_names
        ]

        data_parallel_sharding_infos = create_data_parallel_sharding_infos_by_sharding(
            data_parallel_embedding_collection,
            data_parallel_sharding_plan,
            fused_params,
        )
        assert (
            len(data_parallel_sharding_infos) > 0
        ), "data_parallel_sharding_infos should not be empty"
        dp_sharding = DpSequenceEmbeddingSharding(
            sharding_infos=data_parallel_sharding_infos,
            env=env,
            device=device,
        )
        self._dp_lookups = [
            dp_sharding.create_lookup(
                device=device,
                fused_params=fused_params,
            )
        ]

        self._env = env
        self._device = device

        self._initialize_torch_state()
        self._has_uninitialized_input_dist: bool = True

    def _initialize_torch_state(self) -> None:  # noqa
        """
        This provides consistency between this class and the EmbeddingCollection's
        nn.Module API calls (state_dict, named_modules, etc)
        """
        self.embeddings: nn.ModuleDict = nn.Module()
        assert len(self._dp_lookups[0]._emb_modules) == 1
        param_name = f"{'/'.join(self._table_names)}_weights"
        self.embeddings.register_parameter(
            param_name,
            self._dp_lookups[0]._emb_modules[0].emb_module.weights,
        )
        setattr(
            self._dp_lookups[0]._emb_modules[0].emb_module.weights,
            "need_tp_allreduce",
            True,
        )
        self.embedding_weights: Dict[str, torch.Tensor] = {}

        for (
            table_name,
            tbe_slice,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `named_parameters_by_table`.
        ) in self._dp_lookups[0].named_parameters_by_table():
            # for virtual table, currently we don't expose id tensor and bucket tensor
            # because they are not updated in real time, and they are created on the fly
            # whenever state_dict is called
            # reference: ƒbgs _gen_named_parameters_by_table_ssd_pmt
            self.embedding_weights[table_name] = tbe_slice

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self._device and self._device.type == "meta":
            return
        # Initialize embedding weights with init_fn
        for table_config in self._embedding_configs:
            assert table_config.init_fn is not None
            param = self.embedding_weights[f"{table_config.name}"]
            # pyre-ignore
            table_config.init_fn(param)

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        self._features_order: List[int] = []
        for f in self._feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=self._device, dtype=torch.int32),
            persistent=False,
        )
        self._feature_splits = [len(self._feature_names)]

    # return Tensor! Not awaitable!
    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys())
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor,
                )
            features = features.split(self._feature_splits)[0]
        embeddings = self._dp_lookups[0](features).view(-1, self._embedding_dim)
        kjt = KeyedJaggedTensor(
            values=embeddings,
            keys=features.keys(),
            lengths=features.lengths(),
            offsets=features.offsets(),
        )
        return kjt.to_dict()


class ShardedEmbedding(torch.nn.Module):
    """
    ShardedEmbedding is a module for handling sharded embeddings in a distributed setting.

    Args:
        embedding_configs (List[ShardedEmbeddingConfig]): Configuration for the sharded embedding.
    """

    def __init__(
        self,
        embedding_configs: List[ShardedEmbeddingConfig],
    ):
        super(ShardedEmbedding, self).__init__()

        def create_embedding_collection(configs):
            return EmbeddingCollection(
                tables=[
                    EmbeddingConfig(
                        name=config.table_name,
                        embedding_dim=config.dim,
                        num_embeddings=config.vocab_size,
                        feature_names=config.feature_names,
                        data_type=dtype_to_data_type(torch.float32),
                    )
                    for config in configs
                ],
                device=torch.device("meta"),
            )

        model_parallel_embedding_configs = []
        data_parallel_embedding_configs = []
        for config in embedding_configs:
            if config.sharding_type == "data_parallel":
                data_parallel_embedding_configs.append(config)
            else:
                model_parallel_embedding_configs.append(config)

        self._model_parallel_embedding_collection = (
            create_embedding_collection(configs=model_parallel_embedding_configs)
            if len(model_parallel_embedding_configs) > 0
            else None
        )

        if len(data_parallel_embedding_configs) > 0:
            self._data_parallel_embedding_collection = (
                create_embedding_collection(configs=data_parallel_embedding_configs)
                if len(data_parallel_embedding_configs) > 0
                else None
            )
            self._side_stream = torch.cuda.Stream()
        else:
            self._data_parallel_embedding_collection = None
            self._side_stream = None
        self.freeze_embedding = os.environ.get("FREEZE_EMBEDDING", "0")
        # for nvtx setting, we need to get the tensor from the output dict and set it back to the output dict
        register_setter_and_getter_for_nvtx(
            ShardedEmbedding.forward,
            key_or_attr_name=[embedding_configs[0].feature_names[0], "_values"],
        )

    def _maybe_detach(self, embeddings):
        """
        Detach the embeddings if the freeze_embedding is 1. For debugging purpose.
        Args:
            embeddings (Dict[str, JaggedTensor]): The embeddings to be detached.
        Returns:
            Dict[str, JaggedTensor]: The detached embeddings.
        """
        if self.freeze_embedding == "1":
            for key, embedding in embeddings.items():
                embedding._values = embedding._values.detach()
        return embeddings

    @output_nvtx_hook(nvtx_tag="ShardedEmbedding")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.
        Must be symbolic-traceable!

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """
        mp_embeddings_awaitables = self._model_parallel_embedding_collection(kjt)
        if self._data_parallel_embedding_collection is not None:
            with torch.cuda.stream(self._side_stream):
                dp_embeddings = self._data_parallel_embedding_collection(kjt)
            torch.cuda.current_stream().wait_stream(self._side_stream)
            embeddings = {**mp_embeddings_awaitables.wait(), **dp_embeddings}
        else:
            embeddings = mp_embeddings_awaitables.wait()
        return embeddings

    def export_local_embedding(self, table_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exports the local embeddings, i.e., the embeddings stored on the local rank.

        Args:
            table_name (str): The table name to be exported.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the keys and values of the local embeddings.

        Raises:
            ValueError: If the sharding type does not support exporting local embeddings.


        Example:
            >>> # assume we have 2 ranks
            >>> import torch
            >>> from modules.embedding import ShardedEmbedding
            >>> from configs.task_config import ShardedEmbeddingConfig
            >>> from commons.utils.initialize as init
            >>> from commons.utils.logger import print_rank_0
            >>> init.initialize_model_parallel(1) # dp size is 1
            >>> config = ShardedEmbeddingConfig(
            ...     feature_names=["test"],
            ...     table_name="test_table",
            ...     dim=32,
            ...     vocab_size=100,
            ...     sharding_type="model_parallel",
            ... )
            >>> embedding = ShardedEmbedding(embedding_configs=[config])
            >>> keys, values = embedding.export_local_embedding("test_table")
            >>> print(f"rank {torch.distributed.get_rank()}; keys: {keys.shape}, values: {values.shape}")
            rank 0: keys: (50,), values: (50, 32)
            rank 1: keys: (50,), values: (50, 32)
        """
        from dynamicemb.dump_load import get_dynamic_emb_module

        dynamicemb_modules = get_dynamic_emb_module(
            self._model_parallel_embedding_collection
        )
        if len(dynamicemb_modules) > 0:
            for m in dynamicemb_modules:
                if table_name not in set(m.table_names):
                    continue
                keys_tensor, values_tensor = m.export_keys_values(
                    table_name, device=torch.device(f"cpu")
                )
                return keys_tensor.numpy(), values_tensor.numpy()

        keys_list = []
        values_list = []
        for (
            name,
            t,
        ) in self._model_parallel_embedding_collection.state_dict().items():
            if table_name not in name:
                continue
            if not hasattr(t, "local_shards"):
                raise ValueError(
                    "export_local_embedding is not compatible with a data_parallel sharding table"
                )
            for shard in t.local_shards():
                # [row_start, col_start]
                offsets = shard.metadata.shard_offsets
                # [row_length, col_length]
                lengths = shard.metadata.shard_sizes
                keys_list.append(np.arange(offsets[0], offsets[0] + lengths[0]))
                values_list.append(shard.tensor.cpu().numpy())
        return np.concatenate(keys_list), np.concatenate(values_list)


def get_nonfused_embedding_optimizer(
    module: torch.nn.Module,
) -> Iterator[torch.optim.Optimizer]:
    """
    Retrieves non-fused embedding optimizers from a PyTorch module. Non-fused embedding optimizers are used by torchrec data-parallel sharded embedding collection.

    Args:
        module (torch.nn.Module): The PyTorch module to search for non-fused embedding optimizers.

    Yields:
        torch.optim.Optimizer: An iterator over the non-fused embedding optimizers found in the module.
    """
    for module in module.modules():
        if hasattr(module, "_nonfused_embedding_optimizer"):
            yield module._nonfused_embedding_optimizer
