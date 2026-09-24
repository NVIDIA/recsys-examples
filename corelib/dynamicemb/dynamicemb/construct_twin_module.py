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

"""
Functions in this file are usually used in the tests internally.
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchrec
from dynamicemb import DynamicEmbTableOptions
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.key_ownership import owned_key_mask
from dynamicemb.key_value_table import Storage
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from torch import nn
from torchrec import EmbeddingBagCollection, EmbeddingCollection
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import BaseEmbeddingConfig


def _init_fn(x: torch.Tensor):
    with torch.no_grad():
        x.uniform_(0, 1)


def generate_sparse_feature(
    feature_names, num_embeddings_list, local_batch_size, lookup_iter=0, weighted=False
):
    """
    Generate a KeyedJaggedTensor for one lookup iteration across all embedding tables.

    Parameters:
    - feature_names: List of N embedding table names
    - num_embeddings_list: List of N embedding table sizes
    - local_batch_size: Batch size per rank
    - lookup_iter: Current lookup iteration (0, 1, 2, ...)
    - weighted: Attach all-ones per-sample weights. A weighted collection
      unconditionally reads features.weights(), which raises on a KJT that has
      none; all-ones leaves a SUM pooling result equal to the row itself, which
      is what this sweep relies on to read values back out.

    Returns:
    - A KeyedJaggedTensor for the current lookup iteration
    """
    indices = []
    lengths = []

    # Start index for this lookup iteration

    start_idx = lookup_iter * local_batch_size

    # For each embedding table
    for f_idx, table_size in enumerate(num_embeddings_list):
        # For each sample in local batch
        for b_idx in range(local_batch_size):
            # Calculate global index for this lookup
            global_idx = start_idx + b_idx

            # If index exceeds table size, use 1 as fallback
            if global_idx >= table_size:
                global_idx = 1

            indices.append(global_idx)
            lengths.append(1)  # Single lookup

    return (
        torchrec.KeyedJaggedTensor(
            keys=feature_names,
            values=torch.tensor(indices, dtype=torch.int64).cuda(),
            lengths=torch.tensor(lengths, dtype=torch.int64).cuda(),
            weights=(
                torch.ones(len(indices), dtype=torch.float32).cuda()
                if weighted
                else None
            ),
        ),
        torch.tensor(indices, dtype=torch.int64).cuda(),
    )


dynamicemb_bucket_capacity = 128


class Platform:
    def __init__(self, device):
        device_id = device.index
        gpu_name = torch.cuda.get_device_name(device_id)
        if "A100" in gpu_name:
            self.platform = "a100"
            self.intra_host_bw = 300e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H100" in gpu_name:
            self.platform = "h100"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 25e9  # TODO: need check
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H200" in gpu_name:
            self.platform = "h200"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 450e9
            self.hbm_cap = 140 * 1024 * 1024 * 1024
        else:
            raise RuntimeError(f"Not plan for {gpu_name}")


def get_planner(
    eb_configs: List[BaseEmbeddingConfig],
    batch_size: int,
    multi_hot_sizes: List[int],
    device,
    use_dynamicemb,
    dist_type: str = "roundrobin",
):
    dict_const = {}
    for i, config in enumerate(eb_configs):
        if use_dynamicemb:
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                ],
                pooling_factors=[multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=True,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1024**3,
                    dist_type=dist_type,
                ),
            )
        else:
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                ],
                pooling_factors=[multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
            )

        dict_const[config.name] = const

    platform = Platform(device)
    topology = Topology(
        local_world_size=torchrec.distributed.comm.get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=platform.hbm_cap,
        # One node's host memory, not one rank's: Topology replicates
        # whatever it is given to every device instead of dividing it.
        ddr_cap=1024 * 1024 * 1024 * 1024 // torchrec.distributed.comm.get_local_size(),
        intra_host_bw=platform.intra_host_bw,
        inter_host_bw=platform.inter_host_bw,
    )
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=dict_const,
        batch_size=batch_size,
        debug=True,
    )


class ConstructTwinModule:
    def __init__(
        self,
        table_num: int,
        dims: List[int],
        num_embeddings: List[int],
        pooling_mode,
        is_pooled: bool,
        multi_hot_sizes: List[int],
        batch_size: int = 65536,
        output_dtype: SparseType = SparseType.FP32,
        fwd_a2a_precision: CommType = CommType.FP32,
        bwd_a2a_precision: CommType = CommType.FP32,
        optimizer_kwargs: Dict[str, Any] = None,
        use_index_dedup: bool = False,
        is_weighted: bool = False,
        init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = _init_fn,
        rank: int = 0,
        world_size: int = 1,
        scale_factor: int = 2,
        dist_type: str = "roundrobin",
    ) -> None:
        super().__init__()

        self._table_num = table_num
        self._dims = dims
        self._num_embeddings = num_embeddings
        self._pooling_mode = pooling_mode
        self._is_pooled = is_pooled

        self._batch_size = batch_size
        self._multi_hot_sizes = multi_hot_sizes
        self._output_dtype = output_dtype
        self._fwd_a2a_precision = fwd_a2a_precision
        self._bwd_a2a_precision = bwd_a2a_precision
        self._optimizer_kwargs = optimizer_kwargs
        self._use_index_dedup = use_index_dedup
        # Weighted pooling is SUM-only, and only EmbeddingBagCollection has it.
        assert not is_weighted or is_pooled, "is_weighted requires a pooled collection"
        self._is_weighted = is_weighted
        self._rank = rank
        self._world_size = world_size
        self._scale_factor = scale_factor
        self._init_fn = init_fn
        # Used twice: to configure the DynamicEmb tables, and to decide which
        # keys this rank owns when seeding them (``init_twin_embedding_model``).
        # One value for both, so the seeding cannot disagree with the sharding.
        self._dist_type = dist_type

        self._table_names = [f"t_{t}" for t in range(self._table_num)]
        self._feature_names = [f"f_{t}" for t in range(self._table_num)]

    def _construct_dynamicemb_module(self) -> None:
        assert self._rank == torch.cuda.current_device()
        if self._is_pooled:
            configs = [
                torchrec.EmbeddingBagConfig(
                    name=self._table_names[feature_idx],
                    embedding_dim=self._dims[feature_idx],
                    num_embeddings=self._num_embeddings[feature_idx]
                    * self._scale_factor,
                    feature_names=[self._feature_names[feature_idx]],
                    pooling=self._pooling_mode,
                )
                for feature_idx in range(self._table_num)
            ]
            collection = torchrec.EmbeddingBagCollection(
                device=torch.device("meta"),
                tables=configs,
                is_weighted=self._is_weighted,
            )
        else:
            configs = [
                torchrec.EmbeddingConfig(
                    name=self._table_names[feature_idx],
                    embedding_dim=self._dims[feature_idx],
                    num_embeddings=self._num_embeddings[feature_idx]
                    * self._scale_factor,
                    feature_names=[self._feature_names[feature_idx]],
                )
                for feature_idx in range(self._table_num)
            ]
            collection = torchrec.EmbeddingCollection(
                device=torch.device("meta"),
                tables=configs,
            )

        optimizer_kwargs_copy = copy.deepcopy(self._optimizer_kwargs)
        fused_params = {"output_dtype": self._output_dtype}

        optimizer_type = optimizer_kwargs_copy.pop("optimizer")

        if optimizer_type == "sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "exact_sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "adam":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.ADAM
        elif optimizer_type == "exact_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ADAGRAD
        elif optimizer_type == "exact_row_wise_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else:
            raise ValueError(
                f"unknown optimizer type {optimizer_type} type = {type(optimizer_type)}"
            )
        fused_params.update(optimizer_kwargs_copy)

        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=self._fwd_a2a_precision,
                backward_precision=self._bwd_a2a_precision,
            )
        )

        if isinstance(collection, EmbeddingBagCollection):
            sharder = DynamicEmbeddingBagCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            )
        elif isinstance(collection, EmbeddingCollection):
            sharder = DynamicEmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
                use_index_dedup=self._use_index_dedup,
            )
        else:
            raise ValueError(
                f"Don't support to construct DistributedModelParallel from {type(collection)}"
            )

        device = torch.device(f"cuda:{self._rank}")
        planner = get_planner(
            configs,
            self._batch_size,
            self._multi_hot_sizes,
            device,
            use_dynamicemb=True,
            dist_type=self._dist_type,
        )
        plan = planner.collective_plan(collection, [sharder], dist.GroupMember.WORLD)
        self._dynamicemb_model = DistributedModelParallel(
            module=collection,
            device=device,
            sharders=[sharder],
            plan=plan,
        )

        return

    def _construct_torchrec_module(self) -> None:
        assert self._rank == torch.cuda.current_device()
        if self._is_pooled:
            configs = [
                torchrec.EmbeddingBagConfig(
                    name=self._table_names[feature_idx],
                    embedding_dim=self._dims[feature_idx],
                    num_embeddings=self._num_embeddings[feature_idx],
                    feature_names=[self._feature_names[feature_idx]],
                    pooling=self._pooling_mode,
                    init_fn=self._init_fn,
                )
                for feature_idx in range(self._table_num)
            ]
            collection = torchrec.EmbeddingBagCollection(
                device=torch.device("meta"),
                tables=configs,
                is_weighted=self._is_weighted,
            )
        else:
            configs = [
                torchrec.EmbeddingConfig(
                    name=self._table_names[feature_idx],
                    embedding_dim=self._dims[feature_idx],
                    num_embeddings=self._num_embeddings[feature_idx],
                    feature_names=[self._feature_names[feature_idx]],
                    init_fn=self._init_fn,
                )
                for feature_idx in range(self._table_num)
            ]
            collection = torchrec.EmbeddingCollection(
                device=torch.device("meta"),
                tables=configs,
            )

        optimizer_kwargs_copy = copy.deepcopy(self._optimizer_kwargs)
        fused_params = {"output_dtype": self._output_dtype}

        optimizer_type = optimizer_kwargs_copy.pop("optimizer")

        if optimizer_type == "sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "exact_sgd":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_SGD
        elif optimizer_type == "adam":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.ADAM
        elif optimizer_type == "exact_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ADAGRAD
        elif optimizer_type == "exact_row_wise_adagrad":
            optimizer_kwargs_copy["optimizer"] = EmbOptimType.EXACT_ROWWISE_ADAGRAD
        else:
            raise ValueError(
                f"unknown optimizer type {optimizer_type} type = {type(optimizer_type)}"
            )
        fused_params.update(optimizer_kwargs_copy)

        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=self._fwd_a2a_precision,
                backward_precision=self._bwd_a2a_precision,
            )
        )

        if isinstance(collection, EmbeddingBagCollection):
            sharder = DynamicEmbeddingBagCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
            )
        elif isinstance(collection, EmbeddingCollection):
            sharder = DynamicEmbeddingCollectionSharder(
                qcomm_codecs_registry=qcomm_codecs_registry,
                fused_params=fused_params,
                use_index_dedup=self._use_index_dedup,
            )
        else:
            raise ValueError(
                f"Don't support to construct DistributedModelParallel from {type(collection)}"
            )

        # Note: 'device' is undefined in the original code
        # Assuming it should be torch.cuda.current_device()
        device = torch.device(f"cuda:{self._rank}")
        planner = get_planner(
            configs,
            self._batch_size,
            self._multi_hot_sizes,
            device,
            use_dynamicemb=False,
        )
        plan = planner.collective_plan(collection, [sharder], dist.GroupMember.WORLD)
        self._torchrec_model = DistributedModelParallel(
            module=collection,
            device=device,
            sharders=[sharder],
            plan=plan,
        )

        return

    def init_twin_embedding_model(self):
        self._construct_dynamicemb_module()
        self._construct_torchrec_module()
        max_table_size = max(self._num_embeddings)
        total_iterations = (max_table_size + self._batch_size - 1) // self._batch_size

        feature_nums = len(self._feature_names)

        all_indices = {feature: [] for feature in self._feature_names}
        all_values = {feature: [] for feature in self._feature_names}

        # now it can do only one collection
        collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(
            self._dynamicemb_model, ""
        )
        _, _, tmp_collection_module = collections_list[0]

        tmp_dynamic_emb_module_list = get_dynamic_emb_module(tmp_collection_module)
        table_name_map_hkv_table = {}
        table_name_map_table_id = {}
        # Seeding a row's optimizer state is the optimizer's call, not the
        # storage's, so keep a handle on it alongside the storage.
        table_name_map_optimizer = {}
        for dynamic_emb_module in tmp_dynamic_emb_module_list:
            tmp_table_names = dynamic_emb_module.table_names
            tmp_storage = dynamic_emb_module.tables
            tmp_optimizer = dynamic_emb_module.optimizer

            for i, tmp_table_name in enumerate(tmp_table_names):
                table_name_map_hkv_table[tmp_table_name] = tmp_storage
                table_name_map_table_id[tmp_table_name] = i
                table_name_map_optimizer[tmp_table_name] = tmp_optimizer

        # Perform all lookup iterations
        for iter_idx in range(total_iterations):
            # Generate sparse feature for current iteration
            sparse_feature, sparse_indices = generate_sparse_feature(
                self._feature_names,
                self._num_embeddings,
                self._batch_size,
                iter_idx,
                weighted=self._is_weighted,
            )

            # Forward pass through model
            output = self._torchrec_model(sparse_feature)

            # Process output based on pooling type
            if self._is_pooled:
                # For pooled case, output is already pooled values
                values = output.values()
                dim_offset = 0
                for feature_idx in range(feature_nums):
                    tmp_feature_name = self._feature_names[feature_idx]
                    tmp_dim = self._dims[feature_idx]
                    tmp_indices = sparse_indices[
                        feature_idx
                        * self._batch_size : (feature_idx + 1)
                        * self._batch_size
                    ]
                    tmp_values = values[:, dim_offset : dim_offset + tmp_dim]
                    dim_offset += tmp_dim
                    all_indices[tmp_feature_name].append(tmp_indices)
                    all_values[tmp_feature_name].append(tmp_values)
            else:
                # For non-pooled case, output contains indices and values
                feature_idx = 0
                for feature, sparse_data in output.items():
                    tmp_feature_name = self._feature_names[feature_idx]
                    tmp_indices = sparse_indices[
                        feature_idx
                        * self._batch_size : (feature_idx + 1)
                        * self._batch_size
                    ]
                    all_indices[tmp_feature_name].append(tmp_indices)
                    all_values[tmp_feature_name].append(sparse_data.values())
                    feature_idx += 1

        final_results = {}
        for feature in self._feature_names:
            indices = torch.cat(all_indices[feature], dim=0)
            values = torch.cat(all_values[feature], dim=0)

            # Create a sparse tensor representation
            final_results[feature] = {"indices": indices, "values": values}

        for feature, result in final_results.items():
            indices = result["indices"]
            values = result["values"]

            mask = owned_key_mask(
                indices, self._rank, self._world_size, self._dist_type
            )
            if mask is None:
                mask = torch.ones_like(indices, dtype=torch.bool)
            filtered_indices = indices[mask]

            # Get the dimension for this feature
            dim = self._dims[self._feature_names.index(feature)]

            # Reshape values to [num_indices, dim]
            values = values.reshape(-1, dim)

            # Select values based on filtered indices
            filtered_values = values[mask, :]

            # Remove duplicates from filtered indices and values
            sorted_key, idx = torch.sort(filtered_indices)
            sorted_value = filtered_values[idx, :]

            uniq_key, inv, counts = torch.unique(
                sorted_key, sorted=True, return_inverse=True, return_counts=True
            )
            # start pos in sorted_key
            first_pos = torch.cumsum(
                torch.cat([torch.tensor([0], device=sorted_key.device), counts[:-1]]),
                dim=0,
            )
            chosen_value = sorted_value[first_pos, :]

            unique_indices = uniq_key
            unique_values = chosen_value

            tmp_table_name = feature.replace("f_", "t_")
            cur_hkv_table: Storage = table_name_map_hkv_table[tmp_table_name]
            table_id = table_name_map_table_id[tmp_table_name]
            max_emb_dim = cur_hkv_table.max_embedding_dim()
            max_value_dim = cur_hkv_table.max_value_dim()
            # The value row's layout is the table's to describe, so take every
            # width from it rather than mixing in the twin's own `dim` -- the
            # two are the same table's width and disagreeing would corrupt the
            # row silently.
            emb_dim = cur_hkv_table.embedding_dim(table_id)
            optstate_dim = cur_hkv_table.value_dim(table_id) - emb_dim
            if emb_dim != dim:
                raise ValueError(
                    f"table '{tmp_table_name}' is {emb_dim}-wide in dynamicemb "
                    f"but {dim}-wide in the twin model; the feature-to-table "
                    "mapping and the configured dims disagree."
                )

            padded_values = torch.zeros(
                unique_values.size(0),
                max_value_dim,
                dtype=unique_values.dtype,
                device=unique_values.device,
            )
            padded_values[:, :emb_dim] = unique_values
            if optstate_dim > 0:
                table_name_map_optimizer[tmp_table_name].reset_optimizer_states(
                    padded_values[:, max_emb_dim : max_emb_dim + optstate_dim],
                    emb_dims=emb_dim,
                )

            table_ids = torch.full(
                (unique_indices.numel(),),
                table_id,
                dtype=torch.int64,
                device=unique_indices.device,
            )
            cur_hkv_table.insert(unique_indices, table_ids, padded_values)
        # In TorchREC, once a forward lookup occurs, the iteration in the module gets updated(even you don't do backward).
        # This makes it difficult to accurately test ADAM since the iteration count affects the optimizer's behavior.
        # So we reset the optimizer_step in here
        self._torchrec_model.fused_optimizer.set_optimizer_step(0)

    @property
    def dynamicemb_model(self):
        return self._dynamicemb_model

    @property
    def torchrec_model(self):
        return self._torchrec_model
