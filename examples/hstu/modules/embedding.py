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
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.distributed as dist
from commons.utils.tensor_initializer import BaseInitializer
from configs.task_config import (
    DynamicShardedEmbeddingConfig,
    EmbeddingOptimizerParam,
    ShardedEmbeddingConfig,
)
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.planner import DynamicEmbeddingEnumerator
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from dynamicemb.planner import DynamicEmbParameterConstraints
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from megatron.core import tensor_parallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    ModuleSharder,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig, dtype_to_data_type
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

_optimizer_str_to_optim_type = {
    "adam": EmbOptimType.ADAM,
    "sgd": EmbOptimType.EXACT_SGD,
    # 'adamw': EmbOptimType.ADAMW,
}


def _extract_initializer_args(
    initializer: BaseInitializer,
) -> DynamicEmbInitializerArgs:
    """
    Extracts the initializer arguments from a BaseInitializer instance.

    Args:
        initializer (BaseInitializer): The initializer instance.

    Returns:
        DynamicEmbInitializerArgs: The extracted initializer arguments.

    Raises:
        ValueError: If the initializer type is unsupported.
    """
    assert isinstance(
        initializer, BaseInitializer
    ), "dynamicemb only support BaseInitializer"
    init_params = initializer.get_params()
    if initializer.get_type_str() == "normal":
        return DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.NORMAL,
            mean=init_params["mean"],
            std_dev=init_params["std"],
        )
    if initializer.get_type_str() == "constant":
        return DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=init_params["value"]
        )
    elif initializer.get_type_str() == "uniform":
        return DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.UNIFORM,
            lower=init_params["low"],
            upper=init_params["high"],
        )
    raise ValueError(f"unsupported init method {initializer}")


class ShardedEmbedding(torch.nn.Module):
    """
    ShardedEmbedding is a module for handling sharded embeddings in a distributed setting.

    Args:
        embedding_configs (List[Union[ShardedEmbeddingConfig, DynamicShardedEmbeddingConfig]]): Configuration for the sharded embedding.
        batch_size_per_rank (int, optional): Batch size per rank, used as a hint for sharder. Defaults to 512.
        pg (Optional[dist.ProcessGroup], optional): Process group for distributed embedding. Defaults to None.
    """

    def __init__(
        self,
        embedding_configs: List[
            Union[ShardedEmbeddingConfig, DynamicShardedEmbeddingConfig]
        ],
        batch_size_per_rank: int = 512,
        pg: Optional[dist.ProcessGroup] = None,
    ):
        super(ShardedEmbedding, self).__init__()
        # torchrec DP embedding itself will handle broadcast
        with tensor_parallel.get_cuda_rng_tracker().fork(
            "sharded-embedding-group-seed"
        ):
            self._configs = embedding_configs
            self._device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._batch_size_per_rank = (
                batch_size_per_rank if batch_size_per_rank else 512
            )
            self._pg = pg if pg else dist.group.WORLD

            # fused_params is used to create FBGEMM TBE object, it includes learning rate, beta1 & beta2
            opt_param: EmbeddingOptimizerParam = self._configs[0].optimizer_param
            assert (
                opt_param.optimizer_str in _optimizer_str_to_optim_type
            ), f"embedding optimizer only support {list(_optimizer_str_to_optim_type.keys())}"
            fused_params = {
                "optimizer": _optimizer_str_to_optim_type[opt_param.optimizer_str],
                "learning_rate": opt_param.learning_rate,
                "beta1": opt_param.adam_beta1,
                "beta2": opt_param.adam_beta2,
                "eps": opt_param.adam_eps,
                # 'weight_decay_mode' : WeightDecayMode.NONE,
                "output_dtype": SparseType.FP32,
            }

            """
            Shards the embedding collection based on the provided configuration. Memory allocation happens during this step.
            """
            self._has_dynamic_embedding = False
            constraints = {}
            for config in self._configs:
                if isinstance(config, DynamicShardedEmbeddingConfig):
                    sharding_types = [ShardingType.ROW_WISE.value]
                    init_args = _extract_initializer_args(config.initializer)
                    # use default value in dynamicemb
                    dynamicemb_options = DynamicEmbTableOptions(
                        global_hbm_for_values=config.global_hbm_for_values,
                        initializer_args=init_args,
                        evict_strategy=config.evict_strategy,
                        bucket_capacity=config.bucket_capacity,
                        safe_check_mode=config.safe_check_mode,
                    )
                    constraint = DynamicEmbParameterConstraints(
                        sharding_types=sharding_types,
                        bounds_check_mode=BoundsCheckMode.NONE,  # dynamic embedding has no bounding!
                        enforce_hbm=True,
                        use_dynamicemb=True,
                        dynamicemb_options=dynamicemb_options,
                    )
                    self._has_dynamic_embedding = True
                else:
                    sharding_types = [
                        ShardingType.ROW_WISE.value
                        if config.sharding_type == "model_parallel"
                        else ShardingType.DATA_PARALLEL.value
                    ]
                    constraint = DynamicEmbParameterConstraints(
                        sharding_types=sharding_types,
                        bounds_check_mode=BoundsCheckMode.NONE,  # dynamic embedding has no bounding!
                        enforce_hbm=True,
                        use_dynamicemb=False,
                    )
                constraints.update({config.table_name: constraint})

            topology = Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(self._pg),
                compute_device=self._device.type,
            )
            estimated_global_batch_size = (
                self._batch_size_per_rank * dist.get_world_size(self._pg)
            )

            eb_configs = []
            for config in self._configs:
                eb_configs.append(
                    EmbeddingConfig(
                        name=config.table_name,
                        embedding_dim=config.dim,
                        num_embeddings=config.vocab_size,  # To
                        feature_names=config.feature_names,
                        init_fn=config.initializer,
                        data_type=dtype_to_data_type(
                            torch.float32
                        ),  # weight storage precision is alrways float32
                    )
                )

            enumerator = DynamicEmbeddingEnumerator(
                topology=topology,
                batch_size=estimated_global_batch_size,
                constraints=constraints,
            )

            planner = DynamicEmbeddingShardingPlanner(
                eb_configs=eb_configs,
                batch_size=estimated_global_batch_size,
                topology=topology,
                constraints=constraints,
                enumerator=enumerator,
                storage_reservation=HeuristicalStorageReservation(percentage=0.05),
            )

            sharders = [
                cast(
                    ModuleSharder[torch.nn.Module],
                    DynamicEmbeddingCollectionSharder(
                        use_index_dedup=True, fused_params=fused_params
                    ),
                ),
            ]

            embedding_collection = EmbeddingCollection(
                tables=eb_configs,
                device=torch.device("meta"),  # do not allocate memory right now
            )
            self._plan = planner.collective_plan(
                embedding_collection, sharders, self._pg
            )

            self._embedding_collection = DMP(
                module=embedding_collection,
                env=ShardingEnv.from_process_group(self._pg),
                device=self._device,
                sharders=sharders,
                plan=self._plan,
            )
            data_parallel_parameters = list(
                dict(in_backward_optimizer_filter(self.named_parameters())).values()
            )
            if len(data_parallel_parameters) > 0:
                if fused_params["optimizer"] == EmbOptimType.ADAM:
                    optimizer_fn = partial(
                        torch.optim.Adam,
                        lr=fused_params["learning_rate"],
                        betas=(
                            fused_params["beta1"],
                            fused_params["beta2"],
                        ),
                        eps=fused_params["eps"],
                        weight_decay=0,
                    )
                elif fused_params["optimizer"] == EmbOptimType.EXACT_SGD:
                    optimizer_fn = partial(
                        torch.optim.SGD,
                        lr=fused_params["learning_rate"],
                        weight_decay=0,
                    )
                else:
                    raise ValueError(f"{opt_param.optimizer_str} not support.")
                self._nonfused_embedding_optimizer = optimizer_fn(
                    data_parallel_parameters
                )

    @property
    def module(self):
        """
        Returns the underlying embedding collection module.

        Returns:
            torch.nn.Module: The embedding collection module.
        """
        return self._embedding_collection.module

    # @output_nvtx_hook(nvtx_tag="ShardedEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """
        return self._embedding_collection(kjt).wait()

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
            >>> from commons.utils.logging import print_rank_0
            >>> init.initialize_model_parallel(1) # dp size is 1
            >>> config = ShardedEmbeddingConfig(
            ...     feature_names=["test"],
            ...     table_name="test_table",
            ...     dim=32,
            ...     vocab_size=100,
            ...     sharding_type="model_parallel",
            ...     optimizer_param=EmbeddingOptimizerParam(
            ...       optimizer_str='sgd', learning_rate=1e-5
            ...     )
            ... )
            >>> embedding = ShardedEmbedding(embedding_configs=[config])
            >>> keys, values = embedding.export_local_embedding("test_table")
            >>> print(f"rank {torch.distributed.get_rank()}; keys: {keys.shape}, values: {values.shape}")
            rank 0: keys: (50,), values: (50, 32)
            rank 1: keys: (50,), values: (50, 32)
        """
        keys_list = []
        values_list = []

        dynamic_table_names = set()
        if self._has_dynamic_embedding:
            from dynamicemb.dump_load import export_keys_values, get_dynamic_emb_module
            from dynamicemb_extensions import (
                dyn_emb_capacity,
                dyn_emb_cols,
                dyn_emb_rows,
            )

            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            dynamicemb_modules = get_dynamic_emb_module(
                self._embedding_collection.module
            )

            for m in dynamicemb_modules:
                for dynamic_table_name, dynamic_table in zip(m.table_names, m.tables):
                    dynamic_table_names.add(dynamic_table_name)
                    if table_name != dynamic_table_name:
                        continue

                    local_max_rows = dyn_emb_rows(dynamic_table)
                    dim = dyn_emb_cols(dynamic_table)
                    search_capacity = dyn_emb_capacity(dynamic_table)

                    offset = 0
                    batchsize = 65536
                    accumulated_counts = 0
                    while offset < search_capacity:
                        keys, values, _, d_counter = export_keys_values(
                            dynamic_table, offset, device, batchsize
                        )
                        acutual_counts = d_counter.cpu().item()
                        keys_list.append(keys.cpu().numpy()[:acutual_counts])
                        values_list.append(
                            values.cpu().numpy().reshape(-1, dim)[:acutual_counts, :]
                        )
                        offset += batchsize
                        accumulated_counts += acutual_counts
                    if local_max_rows != accumulated_counts:
                        raise ValueError(
                            f"Rank {dist.get_rank()} has accumulated count {accumulated_counts} which is different from expected {local_max_rows}, "
                            f"difference: {accumulated_counts - local_max_rows}"
                        )
        if table_name not in dynamic_table_names:
            for name, t in self._embedding_collection.state_dict().items():
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
