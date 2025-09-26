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

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTables,
    BatchedDynamicEmbeddingTablesV2,
)
from dynamicemb.dynamicemb_config import (
    DEFAULT_INDEX_TYPE,
    DTYPE_NUM_BYTES,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    get_optimizer_state_dim,
)
from dynamicemb.optimizer import (
    EmbOptimType,
    convert_optimizer_type,
    string_to_opt_type,
)
from dynamicemb_extensions import OptimizerType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import PoolingMode
from torch import nn
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
)
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_kernel import get_state_dict
from torchrec.distributed.embedding_types import (  # EmbeddingComputeKernel,; GroupedEmbeddingConfig,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
    compute_kernel_to_embedding_location,
)
from torchrec.distributed.types import Shard, ShardedTensor, ShardedTensorMetadata
from torchrec.modules.embedding_configs import DataType, data_type_to_dtype
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizer


def pooling_mode_to_dynamicemb(pooling: PoolingMode) -> DynamicEmbPoolingMode:
    if pooling == PoolingMode.MEAN.value:
        return DynamicEmbPoolingMode.MEAN
    elif pooling == PoolingMode.SUM.value:
        return DynamicEmbPoolingMode.SUM
    elif pooling == PoolingMode.NONE.value:
        return DynamicEmbPoolingMode.NONE
    else:
        raise Exception(f"Invalid pooling type {pooling}")


def get_state_dict(
    embedding_tables: List[ShardedEmbeddingTable],
    params: Union[
        nn.ModuleList,
        List[Union[nn.Module, torch.Tensor]],
        List[torch.Tensor],
        List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    ],
    pg: Optional[dist.ProcessGroup] = None,
    destination: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    if destination is None:
        destination = OrderedDict()
        # pyre-ignore [16]
        destination._metadata = OrderedDict()
    """
    It is possible for there to be multiple shards from a table on a single rank.
    We accumulate them in key_to_local_shards. Repeat shards should have identical
    global ShardedTensorMetadata.
    """
    key_to_local_shards: Dict[str, List[Shard]] = defaultdict(list)
    key_to_global_metadata: Dict[str, ShardedTensorMetadata] = {}

    def get_key_from_embedding_table(embedding_table: ShardedEmbeddingTable) -> str:
        return prefix + f"{embedding_table.name}.weight"

    for embedding_table, param in zip(embedding_tables, params):
        key = get_key_from_embedding_table(embedding_table)

        # param is a dummy tensor
        if embedding_table.global_metadata is not None and pg is not None:
            # set additional field of sharded tensor based on local tensor properties
            embedding_table.global_metadata.tensor_properties.dtype = (
                param.dtype  # pyre-ignore[16]
            )
            embedding_table.global_metadata.tensor_properties.requires_grad = (
                param.requires_grad  # pyre-ignore[16]
            )

            local_metadatas = deepcopy(embedding_table.global_metadata.shards_metadata)

            world_size = pg.size()
            assert world_size == len(local_metadatas)
            for i, local_metadata in enumerate(local_metadatas):
                local_metadata.shard_offsets = [i, 0]
                local_metadata.shard_sizes = [1, 1]

            global_metadata = ShardedTensorMetadata(
                shards_metadata=local_metadatas,
                size=torch.Size(
                    [
                        world_size,
                        1,
                    ]
                ),
                tensor_properties=deepcopy(
                    embedding_table.global_metadata.tensor_properties
                ),
            )
            key_to_global_metadata[key] = global_metadata

            key_to_local_shards[key].append(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Module, Tensor]`.
                # pyre-fixme[6]: For 2nd argument expected `ShardMetadata` but got
                #  `Optional[ShardMetadata]`.
                Shard(param, local_metadatas[pg.rank()])
            )
        else:
            destination[key] = param

    if pg is not None:
        # Populate the remaining destinations that have a global metadata
        for key in key_to_local_shards:
            global_metadata = key_to_global_metadata[key]
            destination[
                key
            ] = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards=key_to_local_shards[key],
                sharded_tensor_metadata=global_metadata,
                process_group=pg,
            )

    return destination


def _gen_named_parameters_by_table_fused(
    emb_module: BatchedDynamicEmbeddingTables,
    table_name_to_count: Dict[str, int],
    config: GroupedEmbeddingConfig,
    pg: Optional[dist.ProcessGroup] = None,
) -> Iterator[Tuple[str, TableBatchedEmbeddingSlice]]:
    # TODO: move logic to FBGEMM to avoid accessing fbgemm internals
    for t_idx, _ in enumerate(emb_module._dynamicemb_options):
        table_name = config.embedding_tables[t_idx].name
        if table_name not in table_name_to_count:
            continue
        table_name_to_count.pop(table_name)

        # weight = nn.Parameter(torch.empty((table_option.max_capacity, table_option.dim),
        #                                   device=torch.device("meta"),
        #                                   dtype=emb_module.embedding_dtype))
        weight = nn.Parameter(
            torch.empty(
                (1, 1), device=torch.device("meta"), dtype=emb_module.embedding_dtype
            )
        )
        # this reuses logic in EmbeddingFusedOptimizer but is per table
        # pyre-ignore
        # weight._in_backward_optimizers = [
        #     EmbeddingFusedOptimizer(
        #         config=config,
        #         emb_module=emb_module,
        #         pg=pg,
        #         create_for_table=table_name,
        #         param_weight_for_table=weight,
        #     )
        # ]
        weight._in_backward_optimizers = EmptyFusedOptimizer()
        yield (table_name, weight)


class DynamicEmbeddingFusedOptimizer(FusedOptimizer):
    def __init__(  # noqa C901
        self,
        emb_module: BatchedDynamicEmbeddingTables,
        lr: float,
    ) -> None:
        state: Dict[Any, Any] = {}
        param_group: Dict[str, Any] = {
            "params": [],
            "lr": lr,
        }

        params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}
        self._emb_module = [emb_module]
        super().__init__(params, state, [param_group])

    def zero_grad(self, set_to_none: bool = False) -> None:
        # pyre-ignore [16]
        self._emb_module[0].set_learning_rate(self.param_groups[0]["lr"])
        return

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        # pyre-ignore [16]
        self._emb_module[0].set_learning_rate(self.param_groups[0]["lr"])
        return


def _clean_grouped_fused_params(fused_params: Dict[str, Any]):
    for f in {
        "customized_compute_kernel",
        # "ComputeKernel",
        "dist_type",
        # "Distributor",
        "dynamicemb_options",
    }:
        fused_params.pop(f)

    if "output_dtype" in fused_params:
        # Convert SparseType in fbgemm_gpu to torch.dtype
        fused_params["output_dtype"] = fused_params["output_dtype"].as_dtype()

    if "betas" in fused_params:
        beta1, beta2 = fused_params["betas"]
        del fused_params["betas"]
        fused_params["beta1"] = beta1
        fused_params["beta2"] = beta2

    # TODO: need to discuss if we need this?
    if "optimizer" in fused_params:
        dyn_emb_opt_type = string_to_opt_type(fused_params["optimizer"].value)
        fused_params["optimizer"] = dyn_emb_opt_type


def _get_dynamicemb_options_per_table(
    local_row,
    local_col,
    data_type: DataType,
    optimizer: EmbOptimType,
    table: ShardedEmbeddingTable,
) -> DynamicEmbTableOptions:
    # User-configured
    dynamicemb_options = table.fused_params["dynamicemb_options"]

    # Internal-configured
    if dynamicemb_options.index_type is None:
        dynamicemb_options.index_type = DEFAULT_INDEX_TYPE
    if dynamicemb_options.embedding_dtype is None:
        dynamicemb_options.embedding_dtype = data_type_to_dtype(data_type)
    if dynamicemb_options.training:
        dynamicemb_options.optimizer_type = convert_optimizer_type(optimizer)
    else:
        dynamicemb_options.optimizer_type = OptimizerType.Null
    dynamicemb_options.dim = local_col
    # Make capacity aligned and improve the HBM budget.
    if dynamicemb_options.num_aligned_embedding_per_rank is not None:
        dynamicemb_options.max_capacity = (
            dynamicemb_options.num_aligned_embedding_per_rank
        )
        dynamicemb_options.local_hbm_for_values += (
            (dynamicemb_options.num_aligned_embedding_per_rank - local_row)
            * (
                local_col
                + get_optimizer_state_dim(
                    dynamicemb_options.optimizer_type,
                    local_col,
                    dynamicemb_options.embedding_dtype,
                )
            )
            * DTYPE_NUM_BYTES[dynamicemb_options.embedding_dtype]
        )
    else:
        dynamicemb_options.max_capacity = local_row

    return dynamicemb_options


class BatchedDynamicEmbeddingBag(
    BaseBatchedEmbeddingBag[torch.Tensor],  # FusedOptimizerModule
    # BaseBatchedEmbeddingBag[BatchedDynamicEmbeddingTables, torch.Tensor], # FusedOptimizerModule
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        # for table in config.embedding_tables:
        #    assert table.local_cols % 4 == 0, (
        #        f"table {table.name} has local_cols={table.local_cols} "
        #        "not divisible by 4. "
        #    )

        _clean_grouped_fused_params(config.fused_params)

        dynamicemb_options_list: List[Dict[str, Any]] = []
        for local_row, local_col, table in zip(
            self._local_rows, self._local_cols, config.embedding_tables
        ):
            dynamicemb_options_list.append(
                _get_dynamicemb_options_per_table(
                    local_row,
                    local_col,
                    config.data_type,
                    config.fused_params["optimizer"],
                    table,
                )
            )

        fused_params = config.fused_params or {}

        self._emb_module: BatchedDynamicEmbeddingTables = (
            BatchedDynamicEmbeddingTablesV2(
                table_options=dynamicemb_options_list,
                pooling_mode=pooling_mode_to_dynamicemb(self._pooling),
                feature_table_map=self._feature_table_map,
                table_names=[t.name for t in config.embedding_tables],
                device=device,
                **fused_params,
            )
        )

        if "learning_rate" in fused_params:
            lr: float = fused_params["learning_rate"]
        else:
            lr: float = 0.01

        self._optim = DynamicEmbeddingFusedOptimizer(self._emb_module, lr)

        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )

    @property
    def emb_module(
        self,
    ) -> BatchedDynamicEmbeddingTables:
        return self._emb_module

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # Our implementation is no cache, so no need to flush.
        # self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after PEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()


class BatchedDynamicEmbedding(BaseBatchedEmbedding[torch.Tensor]):
    # FusedOptimizerModule):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        _clean_grouped_fused_params(config.fused_params)

        dynamicemb_options_list: List[Dict[str, Any]] = []
        for local_row, local_col, table in zip(
            self._local_rows, self._local_cols, config.embedding_tables
        ):
            dynamicemb_options_list.append(
                _get_dynamicemb_options_per_table(
                    local_row,
                    local_col,
                    config.data_type,
                    config.fused_params["optimizer"],
                    table,
                )
            )

        fused_params = config.fused_params or {}

        self._emb_module: BatchedDynamicEmbeddingTables = (
            BatchedDynamicEmbeddingTablesV2(
                table_options=dynamicemb_options_list,
                pooling_mode=DynamicEmbPoolingMode.NONE,
                feature_table_map=self._feature_table_map,
                table_names=[t.name for t in config.embedding_tables],
                device=device,
                **fused_params,
            )
        )

        if "learning_rate" in fused_params:
            lr: float = fused_params["learning_rate"]
        else:
            lr: float = 0.01
        self._optim = DynamicEmbeddingFusedOptimizer(self._emb_module, lr)

        self._param_per_table: Dict[str, TableBatchedEmbeddingSlice] = dict(
            _gen_named_parameters_by_table_fused(
                emb_module=self._emb_module,
                table_name_to_count=self.table_name_to_count.copy(),
                config=self._config,
                pg=pg,
            )
        )

    @property
    def emb_module(
        self,
    ) -> BatchedDynamicEmbeddingTables:
        return self._emb_module

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # Our implementation is no cache, so no need to flush.
        # self.flush()
        return get_state_dict(
            self._config.embedding_tables,
            # pyre-ignore
            self.split_embedding_weights(),
            self._pg,
            destination,
            prefix,
        )

    @property
    def fused_optimizer(self) -> FusedOptimizer:
        return self._optim

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        By convention, fused parameters are designated as buffers because they no longer
        have gradients available to external optimizers.
        """
        # TODO can delete this override once SEA is removed
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, tensor in self.named_split_embedding_weights(
            prefix, recurse, remove_duplicate
        ):
            # hack before we support optimizer on sharded parameter level
            # can delete after SEA deprecation
            param = nn.Parameter(tensor)
            # pyre-ignore
            param._in_backward_optimizers = [EmptyFusedOptimizer()]
            yield name, param

    def flush(self) -> None:
        self._emb_module.flush()

    def purge(self) -> None:
        self._emb_module.reset_cache_states()
