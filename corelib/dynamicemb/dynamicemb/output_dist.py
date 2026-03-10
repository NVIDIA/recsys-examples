
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

"""
Output distribution classes for DynamicEmb row-wise sharding.

This module provides optimized output distribution implementations for:
- RwSequenceEmbeddingDist: for sequence (unpooled) embeddings
- RwPooledEmbeddingDist: for pooled embeddings

The key optimization is in the unbucketize_permute operation, which is slow
in the original TorchRec implementation, especially for non-contiguous
distribution patterns (e.g., round-robin).
"""

from typing import Dict, List, Optional, Union, cast
import torch
from torch import distributed as dist
from torchrec.distributed.types import CommOp
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    SequenceEmbeddingsAllToAll,
    VariableBatchPooledEmbeddingsReduceScatter,
)

from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.embedding_sharding import BaseEmbeddingDist, EmbeddingShardingContext
from torchrec.distributed.types import QuantizedCommCodecs


class RwSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes sequence embedding tensor in RW fashion with an AlltoAll operation.
    
    This is a customized version for DynamicEmb that can be optimized for
    non-contiguous distribution patterns (e.g., round-robin).
    
    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        num_features (int): total number of features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]): 
            quantized communication codecs registry.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._num_features = num_features
        self._device = device
        
        self._dist = SequenceEmbeddingsAllToAll(
            pg,
            [num_features] * pg.size(),
            device,
            codecs=(
                qcomm_codecs_registry.get(
                    CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name, None
                )
                if qcomm_codecs_registry
                else None
            ),
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> torch.Tensor:
        """
        Performs AlltoAll operation on sequence embeddings tensor.
        
        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            sharding_ctx (SequenceShardingContext): shared context from KJTAllToAll
                operation.
                
        Returns:
            torch.Tensor: sequence embeddings after distribution.
        """
        if sharding_ctx is None:
            raise ValueError(
                "RwSequenceEmbeddingDist.forward requires a non-None sharding_ctx."
            )
        
        # TODO: Optimize unbucketize_permute operation here
        # The unbucketize_permute_tensor is used in SequenceEmbeddingsAwaitable
        # to reorder the output. For non-contiguous distribution (round-robin),
        # this operation is slow and can be optimized with custom CUDA kernels.
        
        result = self._dist(
            local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            batch_size_per_rank=sharding_ctx.batch_size_per_rank,
            sparse_features_recat=sharding_ctx.sparse_features_recat,
            unbucketize_permute_tensor=sharding_ctx.unbucketize_permute_tensor,
        )
        
        return result


class RwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion by performing a reduce-scatter
    operation.
    
    Args:
        pg (dist.ProcessGroup): ProcessGroup for reduce-scatter communication.
        embedding_dims (List[int]): embedding dimensions per feature.
        qcomm_codecs_registry (Optional[Dict[str, QuantizedCommCodecs]]):
            quantized communication codecs registry.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        embedding_dims: List[int],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._embedding_dims = embedding_dims
        self._qcomm_codecs_registry = qcomm_codecs_registry
        
        self._dist: Optional[
            Union[
                PooledEmbeddingsReduceScatter,
                VariableBatchPooledEmbeddingsReduceScatter,
            ]
        ] = None
        
        self._codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None
        )
        self._dist_type: Optional[str] = None

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> torch.Tensor:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor.
        
        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.
            sharding_ctx (Optional[EmbeddingShardingContext]): shared context from
                KJTAllToAll operation.
                
        Returns:
            torch.Tensor: pooled embeddings tensor after distribution.
        """
        if self._dist is None:
            self._create_output_dist_module(sharding_ctx)

        self._validate_sharding_ctx_consistency(sharding_ctx)
        
        if sharding_ctx is None:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(local_embs)
        elif sharding_ctx.variable_batch_per_feature:
            return cast(VariableBatchPooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                batch_size_per_rank_per_feature=sharding_ctx.batch_size_per_rank_per_feature,
                embedding_dims=self._embedding_dims,
            )
        else:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                input_splits=sharding_ctx.batch_size_per_rank,
            )

    def _create_output_dist_module(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        """Create the appropriate output distribution module based on context."""
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._dist = VariableBatchPooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
            self._dist_type = "variable_batch"
        else:
            self._dist = PooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
            self._dist_type = "normal"

    def _validate_sharding_ctx_consistency(self, sharding_ctx):

        if self._dist_type is None:
            return
    
        current_is_variable_batch = (
            sharding_ctx is not None and sharding_ctx.variable_batch_per_feature
        )
    
        if self._dist_type == "variable_batch" and not current_is_variable_batch:
            raise RuntimeError(
                "RwPooledEmbeddingDist was initialized for variable batch mode, "
                "but current call is not using variable batch. This indicates "
                "inconsistent usage of the output distribution module."
            )
        elif self._dist_type == "normal" and current_is_variable_batch:
            raise RuntimeError(
                "RwPooledEmbeddingDist was initialized for normal batch mode, "
                "but current call is using variable batch. This indicates "
                "inconsistent usage of the output distribution module."
            )

