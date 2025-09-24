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

# pyre-strict

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import distributed as dist
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingsLookup as _GroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup as _GroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseSparseFeaturesDist,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingLookup,
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    RwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import RwPooledEmbeddingSharding
from torchrec.distributed.types import QuantizedCommCodecs, ShardingEnv, ShardingType
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..batched_dynamicemb_compute_kernel import (
    BatchedDynamicEmbedding,
    BatchedDynamicEmbeddingBag,
)
from ..input_dist import RwSparseFeaturesDist


class GroupedEmbeddingsLookup(_GroupedEmbeddingsLookup):
    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
    ) -> BaseEmbedding:
        if config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL:
            """
            fallback to base class
            """
            return super()._create_embedding_kernel(config=config, pg=pg, device=device)
        else:
            self._need_prefetch = True
            return BatchedDynamicEmbedding(
                config=config,
                pg=pg,
                device=device,
            )


class RwSequenceDynamicEmbeddingSharding(RwSequenceEmbeddingSharding):
    """
    Shards sequence (unpooled) row-wise, i.e.. a given embedding table is evenly
    distributed by rows and table slices are placed on all ranks.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            need_pos=need_pos,
            qcomm_codecs_registry=qcomm_codecs_registry,
            device_type_from_sharding_infos=device_type_from_sharding_infos,
        )

        self._init_customized_distributor(sharding_infos)

    def _init_customized_distributor(
        self, sharding_infos: List[EmbeddingShardingInfo]
    ) -> None:
        common_dist_type = None

        self._dist_type_per_feature: Dict[str, str] = {}
        for sharding_info in sharding_infos:
            fused_params = sharding_info.fused_params
            if fused_params is not None and "dist_type" in fused_params:
                dist_type = fused_params["dist_type"]
                if common_dist_type is None:
                    common_dist_type = dist_type
                else:
                    assert (
                        dist_type == common_dist_type
                    ), "Customized distributor type must keep the same."
            else:
                dist_type = "continuous"
            feature_names = sharding_info.embedding_config.feature_names
            for f in feature_names:
                self._dist_type_per_feature[f] = dist_type

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            pg=self._pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=True,
            has_feature_processor=self._has_feature_processor,
            need_pos=False,
            dist_type_per_feature=self._dist_type_per_feature,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
        )


class GroupedPooledEmbeddingsLookup(_GroupedPooledEmbeddingsLookup):
    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        device: Optional[torch.device],
        pg: Optional[dist.ProcessGroup],
        sharding_type: Optional[ShardingType],
    ) -> BaseEmbedding:
        if config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL:
            """
            fallback to base class
            """
            return super()._create_embedding_kernel(config, device, pg, sharding_type)
        else:
            return BatchedDynamicEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
            )


class RwPooledDynamicEmbeddingSharding(RwPooledEmbeddingSharding):
    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            need_pos=need_pos,
            qcomm_codecs_registry=qcomm_codecs_registry,
            device_type_from_sharding_infos=device_type_from_sharding_infos,
        )

        self._init_customized_distributor(sharding_infos)

    def _init_customized_distributor(
        self, sharding_infos: List[EmbeddingShardingInfo]
    ) -> None:
        common_dist_type = None

        self._dist_type_per_feature: Dict[str, str] = {}
        for sharding_info in sharding_infos:
            fused_params = sharding_info.fused_params
            if fused_params is not None and "dist_type" in fused_params:
                dist_type = fused_params["dist_type"]
                if common_dist_type is None:
                    common_dist_type = dist_type
                else:
                    assert (
                        dist_type == common_dist_type
                    ), "Customized distributor type must keep the same."
            else:
                dist_type = "continuous"
            feature_names = sharding_info.embedding_config.feature_names
            for f in feature_names:
                self._dist_type_per_feature[f] = dist_type

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            pg=self._pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
            dist_type_per_feature=self._dist_type_per_feature,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.ROW_WISE,
        )
