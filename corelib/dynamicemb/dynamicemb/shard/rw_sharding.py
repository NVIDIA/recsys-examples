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

import logging
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchrec
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.ssd.training import SSDTableBatchedEmbeddingBags
from packaging.version import Version
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
from ..batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from ..dynamicemb_config import (
    AUTO_OUTPUT_DIST,
    FUSED_OUTPUT_DIST,
    TORCHREC_OUTPUT_DIST,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    get_eviction_score_strategy,
)
from .hier_output_dist import HierSharding, RwSequenceHierEmbeddingDist
from .input_dist import RwSparseFeaturesDist


def dist_type_per_feature(
    sharding_infos: List[EmbeddingShardingInfo],
) -> Dict[str, str]:
    """The key -> rank rule each feature's keys are distributed by.

    Per feature, not per sharding, because the bucketize kernel reads it per
    feature (``dist_type_per_feature[t]`` in
    ``src/sparse_block_bucketize_features.cu``). Two tables grouped into one
    sharding may legitimately pick different rules, and nothing downstream
    needs them to agree.

    A DynamicEmb table always carries its rule, inside the
    ``DynamicEmbTableOptions`` the planner puts into ``fused_params`` (via
    ``DynamicEmbParameterSharding.get_additional_fused_params``). A plain
    TorchRec table carries none and gets ``continuous``, which is what TorchRec's
    own bucketizer does -- but a DynamicEmb table must never land there, so one
    arriving without options raises instead. ``continuous`` places a key by range
    and rewrites the index on the way in (``new_idx = idx % blk_size``), so what
    the table stores is not a global key; `incremental_dump`, `replay_increment`
    and the checkpoint loader all refuse such a table. Falling back to it would
    turn a missing setting into a table that silently cannot be dumped
    incrementally or reloaded.
    """
    per_feature: Dict[str, str] = {}
    for info in sharding_infos:
        fused_params = info.fused_params
        options = fused_params.get("dynamicemb_options") if fused_params else None
        if options is not None:
            dist_type = options.dist_type
        elif (
            info.param_sharding.compute_kernel
            == EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
        ):
            raise ValueError(
                f"DynamicEmb table {info.embedding_config.name!r} reached "
                "sharding without DynamicEmbTableOptions, so its dist_type is "
                "unknown. DynamicEmbeddingShardingPlanner puts the options on "
                "the DynamicEmbParameterSharding it builds; a plan built "
                "another way has to carry them too. dist_type must be "
                "'roundrobin' or 'hash_roundrobin'."
            )
        else:
            dist_type = "continuous"
        for feature_name in info.embedding_config.feature_names:
            per_feature[feature_name] = dist_type
    return per_feature


def built_output_dist(kind: str) -> str:
    """Output dist ``kind`` as this build of DynamicEmb runs it: ``"auto"`` is
    ``"fused"`` when the build has it (``HierOutputDist``, which needs NCCL >=
    2.30.7), else TorchRec's dists (with a warning); ``"fused"`` raises
    ``ImportError`` when it does not."""
    import dynamicemb_extensions

    built = hasattr(dynamicemb_extensions, "HierOutputDist")
    if kind == AUTO_OUTPUT_DIST:
        if built:
            return FUSED_OUTPUT_DIST
        warnings.warn(
            'output_dist="auto": DynamicEmb was built without NCCL >= 2.30.7, so '
            "without its hierarchical dist; using TorchRec's dists"
        )
        return TORCHREC_OUTPUT_DIST
    if kind == FUSED_OUTPUT_DIST and not built:
        raise ImportError(
            'output_dist="fused" needs dynamicemb built against NCCL >= 2.30.7 '
            "(this build has no hierarchical dist; set NCCL_HOME to a newer NCCL "
            'and rebuild, or use "auto")'
        )
    return kind


def output_dist_options(
    sharding_infos: List[EmbeddingShardingInfo],
) -> Optional[DynamicEmbTableOptions]:
    """The options of the sharding's DynamicEmb tables, which must agree on
    ``output_dist``, ``output_dist_capacity`` and ``node_dedup``: one output
    dist serves the whole sharding. None when the sharding has no DynamicEmb
    table (plain TorchRec tables do not vote)."""
    voters = [
        info.fused_params["dynamicemb_options"]
        for info in sharding_infos
        if (info.fused_params or {}).get("dynamicemb_options") is not None
    ]
    choices = [
        (options.output_dist, options.output_dist_capacity, options.node_dedup)
        for options in voters
    ]
    for choice in choices[1:]:
        if choice != choices[0]:
            raise ValueError(
                "The DynamicEmb tables of one row-wise sharding must agree on "
                "output_dist, output_dist_capacity and node_dedup; one table has "
                f"{choices[0]}, another {choice}."
            )
    return voters[0] if voters else None


def counts_frequencies(options) -> bool:
    """Whether a DynamicEmb table counts its keys' frequencies (LFU scores)."""
    return (
        options is not None
        and options.score_strategy is not None
        and get_eviction_score_strategy(options.score_strategy)
        == DynamicEmbScoreStrategy.LFU
    )


def _without_weights(features: KeyedJaggedTensor) -> KeyedJaggedTensor:
    """``features`` without its weights (no copy)."""
    return KeyedJaggedTensor(
        keys=features.keys(),
        values=features.values(),
        lengths=features.lengths(),
        offsets=features.offsets(),
        stride=features.stride(),
        length_per_key=features._length_per_key,
        offset_per_key=features._offset_per_key,
    )


class GroupedEmbeddingsLookup(_GroupedEmbeddingsLookup):
    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        env: Optional[ShardingEnv] = None,
    ) -> BaseEmbedding:
        if config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL:
            """
            fallback to base class
            """
            if Version(torchrec.__version__) < Version("1.5.0"):
                return super()._create_embedding_kernel(
                    config=config, pg=pg, device=device
                )
            return super()._create_embedding_kernel(
                config=config, pg=pg, device=device, env=env
            )
        else:
            self._need_prefetch = True
            return BatchedDynamicEmbedding(
                config=config,
                pg=pg,
                device=device,
            )

    def prefetch(
        self,
        sparse_features: KeyedJaggedTensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """TorchRec's, but a DynamicEmb group's tables also get its keys'
        weights, their frequency counters, as its forward gives them. TorchRec
        passes keys and offsets alone. Under the prefetch pipelines the LFU
        scores and admission counters would then differ from the plain loop's
        wherever weights count occurrences: index dedup's (whatever the
        output dist), and across nodes the hierarchical dist's for LFU."""
        if not self._need_prefetch:
            return
        for emb_op, features in zip(
            self._emb_modules, self._group_features(sparse_features)
        ):
            emb = emb_op.emb_module
            if isinstance(emb, BatchedDynamicEmbeddingTablesV2):
                emb.prefetch(
                    features.values().to(emb.index_type),
                    features.offsets().long(),
                    forward_stream=forward_stream,
                    frequency_counters=features.weights_or_none(),
                )
                continue
            if (
                isinstance(
                    emb,
                    (
                        SplitTableBatchedEmbeddingBagsCodegen,
                        SSDTableBatchedEmbeddingBags,
                    ),
                )
                and not emb.prefetch_pipeline
            ):
                logging.error(
                    f"Invalid setting on {type(emb)} modules. prefetch_pipeline must "
                    "be set to True.\nIf you don't turn on prefetch_pipeline, cache "
                    "locations might be wrong in backward and can cause wrong results.\n"
                )
            if hasattr(emb, "prefetch"):
                emb.prefetch(
                    indices=features.values(),
                    offsets=features.offsets(),
                    forward_stream=forward_stream,
                )

    def _group_features(
        self, sparse_features: KeyedJaggedTensor
    ) -> List[KeyedJaggedTensor]:
        """The features of each lookup group."""
        return sparse_features.split(self._feature_splits)


class HierGroupedEmbeddingsLookup(GroupedEmbeddingsLookup):
    """The lookup of a sharding on the hierarchical dist. A call takes that
    path when its KJT is the one its input dist returned, whatever the
    module's mode. For such a call each group writes its rows straight into
    the call's rows of ``op.lookup_rows``, through the tables'
    ``output_alloc``. The groups' outputs come back as those rows, without a
    copy (``HierSharding.lookup``). When the KJT's weights are occurrence
    counts (LFU across nodes without input weights), only the groups of LFU
    tables get them, in the forward and in the prefetch. TorchRec's owner
    gives the other groups no frequency counters either: there an admission
    strategy counts each key once. See docs/hier_output_dist.md, section 7.4.
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        hier: HierSharding,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(grouped_configs=grouped_configs, pg=pg, device=device)
        self._hier = hier
        self._lfu = []  # per group: its tables count frequencies
        for module in self._emb_modules:
            emb = module.emb_module
            if emb.output_dtype != hier.dtype:
                raise ValueError(
                    f"A lookup group outputs {emb.output_dtype} and the "
                    f"hierarchical dist's rows are {hier.dtype}: give the tables "
                    "of the sharding one output_dtype."
                )
            emb.output_alloc = hier.lookup_buffer
            self._lfu.append(counts_frequencies(emb._dynamicemb_options[0]))

    def _group_features(
        self, sparse_features: KeyedJaggedTensor
    ) -> List[KeyedJaggedTensor]:
        groups = super()._group_features(sparse_features)
        if not (self._hier.owns(sparse_features) and self._hier.counts_as_weights):
            return groups
        return [
            group if lfu else _without_weights(group)
            for group, lfu in zip(groups, self._lfu)
        ]

    def forward(self, sparse_features: KeyedJaggedTensor) -> torch.Tensor:
        if not self._hier.owns(sparse_features):
            return super().forward(sparse_features)
        look_ups = [
            partial(self._look_up, emb_op, features)
            for emb_op, features in zip(
                self._emb_modules, self._group_features(sparse_features)
            )
        ]
        return self._hier.lookup(sparse_features, look_ups)

    def _look_up(self, emb_op, features: KeyedJaggedTensor) -> torch.Tensor:
        lookup = emb_op(features)
        # TorchRec's model delta tracker, from TorchRec 1.5 on.
        tracker = getattr(self, "optim_state_tracker_fn", None)
        if tracker is not None:
            tracker(features, lookup.view(-1), emb_op)
        return lookup


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
        self._dist_type_per_feature: Dict[str, str] = dist_type_per_feature(
            sharding_infos
        )
        self._hier: Optional[HierSharding] = None
        options = output_dist_options(sharding_infos)
        if (
            options is None
            or built_output_dist(options.output_dist) != FUSED_OUTPUT_DIST
        ):
            return
        if options.output_dist_capacity is None:
            # "auto" alone gets here without one: "fused" needs it
            # (DynamicEmbTableOptions).
            warnings.warn(
                f'output_dist="{options.output_dist}" without output_dist_capacity; '
                "using TorchRec's dists"
            )
            return
        self._hier = self._hier_sharding(sharding_infos, options)

    def _hier_sharding(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        options: DynamicEmbTableOptions,
    ) -> Optional[HierSharding]:
        """The hierarchical dist of ``output_dist="fused"`` (or ``"auto"``),
        or None (with a warning) where it does not apply and the sharding
        stays on TorchRec's dists. Decided from the sharding alone, so the
        same on every rank."""
        asked = options.output_dist
        why = None
        if self._pg.size() == 1:
            why = "the sharding has one rank"
        elif self._has_feature_processor:
            why = "the sharding has feature processors"
        elif any(
            info.param_sharding.compute_kernel
            != EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value
            for info in sharding_infos
        ):
            why = "the sharding has tables that are not DynamicEmb tables"
        if why is not None:
            warnings.warn(
                f'output_dist="{asked}" does not apply here ({why}); '
                "using TorchRec's dists"
            )
            return None
        # The rows of the lookup output: an EmbeddingCollection's tables share
        # one dim, and the lookup casts to the sharder's output_dtype (fp32 by
        # default).
        info = sharding_infos[0]
        output_dtype = (info.fused_params or {}).get("output_dtype", torch.float32)
        if hasattr(output_dtype, "as_dtype"):  # an fbgemm SparseType
            output_dtype = output_dtype.as_dtype()
        # The table of each feature, in the sharding's feature order: across
        # nodes a key is its owner node, table and value.
        tables = [
            table_idx
            for table_idx, table in enumerate(self.embedding_tables())
            for _ in table.feature_names
        ]
        # A clipping table clamps each looked-up row's gradient before the
        # per-key sum, so every occurrence must cross on its own: node dedup
        # off.
        clipping = any(
            (sharding_info.fused_params or {}).get("gradient_clipping", False)
            for sharding_info in sharding_infos
        )
        # LFU tables count occurrences: across nodes the owner gets them as
        # weights.
        lfu = any(
            counts_frequencies(
                (sharding_info.fused_params or {}).get("dynamicemb_options")
            )
            for sharding_info in sharding_infos
        )
        return HierSharding(
            self._pg,
            self._get_num_features(),
            info.embedding_config.embedding_dim,
            output_dtype,
            options.output_dist_capacity,
            asked=asked,
            tables=tables,
            node_dedup=options.node_dedup and not clipping,
            lfu=lfu,
        )

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
            dist_types=[
                self._dist_type_per_feature[name] for name in self.feature_names()
            ],
            hier=self._hier,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        device = device if device is not None else self._device
        if self._hier is not None:
            return HierGroupedEmbeddingsLookup(
                self._grouped_embedding_configs, self._hier, self._pg, device
            )
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ):
        if self._hier is not None:
            return RwSequenceHierEmbeddingDist(
                self._hier,
                self._pg,
                self._get_num_features(),
                device if device is not None else self._device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
        return super().create_output_dist(device)


class GroupedPooledEmbeddingsLookup(_GroupedPooledEmbeddingsLookup):
    def _create_embedding_kernel(
        self,
        config: GroupedEmbeddingConfig,
        device: Optional[torch.device],
        pg: Optional[dist.ProcessGroup],
        sharding_type: Optional[ShardingType],
        env: Optional[ShardingEnv] = None,
    ) -> BaseEmbedding:
        if config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL:
            """
            fallback to base class
            """
            if Version(torchrec.__version__) < Version("1.5.0"):
                return super()._create_embedding_kernel(
                    config, device, pg, sharding_type
                )
            return super()._create_embedding_kernel(
                config, device, pg, sharding_type, env
            )
        else:
            return BatchedDynamicEmbeddingBag(
                config=config,
                pg=pg,
                device=device,
                sharding_type=sharding_type,
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
        self._dist_type_per_feature: Dict[str, str] = dist_type_per_feature(
            sharding_infos
        )

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
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
            sharding_type=ShardingType.ROW_WISE,
        )
