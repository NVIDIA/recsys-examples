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

"""Table-row-wise sharding for DynamicEmb tables.

A table lives on one node and is split row-wise across that node's ranks. The
two collectives that makes possible -- a reduce-scatter inside the node and an
all-to-all between nodes -- are TorchRec's and are used unchanged, as row-wise
uses its own unchanged. What differs is the same two things row-wise changes:
keys are placed by DynamicEmb's rule rather than by row block, and the lookup
has to reach a hash table rather than a TBE.
"""

from typing import Any, Dict, List, Optional

import torch
import torchrec
from packaging.version import Version
from torch import distributed as dist
from torchrec.distributed.embedding_sharding import (
    BaseSparseFeaturesDist,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingLookup,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.sharding.twrw_sharding import (
    TwRwPooledEmbeddingSharding,
    TwRwSparseFeaturesDist as _TwRwSparseFeaturesDist,
)
from torchrec.distributed.types import (
    Awaitable,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from .input_dist import bucketize_kjt_before_all2all, dist_type_codes
from .rw_sharding import dist_type_per_feature, GroupedPooledEmbeddingsLookup


class TwRwSparseFeaturesDist(_TwRwSparseFeaturesDist):
    """TorchRec's table-row-wise distributor, bucketizing by DynamicEmb's rule.

    The same one call differs as in row-wise (see
    :class:`~dynamicemb.shard.input_dist.RwSparseFeaturesDist`): TorchRec places
    a key by the block of rows it falls in and rewrites it to its offset inside
    that block, while two of DynamicEmb's three rules leave the key global,
    because a hash table has no row index to rewrite it into.

    Everything around it is inherited. In particular the staggered shuffle is
    not rewritten here: ``_staggered_shuffle`` and the permute it drives are
    arithmetic over feature counts, not over keys, so they are correct whatever
    the bucketizer did with the values. That is worth stating because an earlier
    reading of this had DynamicEmb owning that permute -- it does not; it only
    pays for it (§6 R2).

    Args:
        dist_types: the rule per feature, in the same order as
            ``feature_hash_sizes`` -- the kernel indexes both with the same
            ``t``, and building them together is what keeps them aligned.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        local_size: int,
        features_per_rank: List[int],
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        dist_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            pg=pg,
            local_size=local_size,
            features_per_rank=features_per_rank,
            feature_hash_sizes=feature_hash_sizes,
            device=device,
            has_feature_processor=has_feature_processor,
            need_pos=need_pos,
        )
        self.register_buffer(
            "_dist_type_tensor",
            dist_type_codes(dist_types, device=device),
            persistent=False,
        )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """Bucketize by ``dist_type``, shuffle, then TorchRec's AlltoAll unchanged.

        ``num_buckets`` is ``local_size``, not the world: a table-row-wise table
        is split across one node's ranks, so that is how many pieces its keys
        are sorted into. This is the same number the planner sized the table's
        per-rank capacity by (`table_fanout`), and they have to agree -- a key
        the bucketizer sends to bucket `b` has to be a key the table on rank
        `b` of that node believes it owns.
        """
        bucketized_features, _ = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._local_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=False,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            dist_type_per_feature=self._dist_type_tensor,
        )
        bucketized_features = bucketized_features.permute(
            self._sf_staggered_shuffle,
            self._sf_staggered_shuffle_tensor,
        )
        return self._dist(bucketized_features)


class TwRwPooledDynamicEmbeddingSharding(TwRwPooledEmbeddingSharding):
    """Shards pooled DynamicEmb tables table-wise then row-wise.

    Two methods differ from TorchRec's, the same two that differ under row-wise.
    ``create_output_dist`` is deliberately not one of them: the intra-node
    reduce-scatter and the cross-node all-to-all are used as they are, which is
    also what keeps the gradient division right without us accounting for it --
    it arrives as `1/L` from one stage times `1/N` from the other (§3.3).
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            need_pos=need_pos,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
        self._dist_type_per_feature: Dict[str, str] = dist_type_per_feature(
            sharding_infos
        )

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        feature_hash_sizes = self._get_feature_hash_sizes()
        assert self._pg is not None
        assert self._intra_pg is not None
        return TwRwSparseFeaturesDist(
            pg=self._pg,
            local_size=self._intra_pg.size(),
            features_per_rank=features_per_rank,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
            dist_types=[
                self._dist_type_per_feature[name] for name in self.feature_names()
            ],
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        """DynamicEmb's lookup, told it is table-row-wise.

        ``sharding_type`` is not decoration: it is what downgrades MEAN pooling
        to SUM in the kernel, because a rank only holds part of a table's rows
        and cannot divide by a count it does not have -- the real mean is an
        output callback on the EBC (§3.1). Passing ROW_WISE here would be
        arithmetically identical today and wrong the moment TorchRec treats the
        two differently.
        """
        kwargs = {}
        if Version(torchrec.__version__) >= Version("1.5.0"):
            kwargs["env"] = self._env
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._rank],
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.TABLE_ROW_WISE,
            **kwargs,
        )
