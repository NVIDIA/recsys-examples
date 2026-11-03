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
from itertools import accumulate
from typing import Dict, List, Optional

import torch
from dynamicemb_extensions import (
    compute_dedup_lengths_cuda,
    get_table_range,
    segmented_unique_cuda,
)
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding import (
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
    pad_vbe_kjt_lengths,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    KJTList,
    ShardingType,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import (
    Awaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..dynamicemb_config import (
    DynamicEmbKernel,
    DynamicEmbScoreStrategy,
    ScoreStrategy,
    get_eviction_score_strategy,
)
from .hier_output_dist import DynamicEmbSequenceShardingContext
from .rw_sharding import RwSequenceDynamicEmbeddingSharding


class DynamicEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Extended EmbeddingCollectionContext that includes frequency_counters for LFU strategy.

    ``reverse_indices[i]`` is None when the output dist expands the index dedup
    of sharding ``i`` itself, as the hierarchical dist's does for a call that
    took it (see ``ShardedDynamicEmbeddingCollection.input_dist``).
    """

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        seq_vbe_ctx: Optional[List] = None,
        frequency_counters: Optional[List[torch.Tensor]] = None,
    ) -> None:
        # TorchRec may retain None for these fields; dedup and VBE append to them.
        super().__init__(
            sharding_contexts=[] if sharding_contexts is None else sharding_contexts,
            input_features=[] if input_features is None else input_features,
            reverse_indices=[] if reverse_indices is None else reverse_indices,
            seq_vbe_ctx=[] if seq_vbe_ctx is None else seq_vbe_ctx,
        )
        self.frequency_counters: List[torch.Tensor] = (
            [] if frequency_counters is None else frequency_counters
        )

    def record_stream(self, stream: torch.Stream) -> None:
        # TorchRec's, except that it skips a None reverse index: that sharding's
        # plan (DynamicEmbSequenceShardingContext) holds its fan-out and records it.
        for sharding_ctx in self.sharding_contexts:
            sharding_ctx.record_stream(stream)
        for features in self.input_features:
            features.record_stream(stream)
        for reverse in self.reverse_indices:
            if reverse is not None:
                reverse.record_stream(stream)
        for vbe_ctx in self.seq_vbe_ctx:
            vbe_ctx.record_stream(stream)


def _expanded_by_output_dist(
    ctx: DynamicEmbeddingCollectionContext, index: int
) -> None:
    """Sharding ``index``'s output dist expands its index dedup itself (a
    call that took the hierarchical dist): TorchRec must not do it again."""
    ctx.reverse_indices[index] = None


class ShardedDynamicEmbeddingCollection(ShardedEmbeddingCollection):
    supported_compute_kernels: List[str] = [
        kernel.value for kernel in EmbeddingComputeKernel
    ] + [DynamicEmbKernel]

    def __init__(
        self,
        *args,
        score_strategy: Optional[ScoreStrategy] = None,
        has_admit_strategy: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the global score strategy
        self._score_strategy = score_strategy
        # LFU is enabled when the strategy's eviction column is LFU -- for a single
        # LFU strategy or for a compound strategy such as (TIMESTAMP, LFU).
        self._is_lfu_enabled = (
            get_eviction_score_strategy(score_strategy) == DynamicEmbScoreStrategy.LFU
            if score_strategy is not None
            else False
        )
        self._has_admit_strategy = has_admit_strategy

    @classmethod
    def create_embedding_sharding(
        cls,
        sharding_type: str,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> EmbeddingSharding[
        SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]:
        """
        override this function to provide customized EmbeddingSharding
        """
        if sharding_type == ShardingType.ROW_WISE.value:
            return RwSequenceDynamicEmbeddingSharding(
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            # Refused rather than fallen through. Table-row-wise is a pooled
            # arrangement: its saving comes from reducing inside a node before
            # crossing between them, and a sequence output has nothing to
            # reduce -- TorchRec's own sequence side has no TwRw sharding
            # either. Falling through would raise, but somewhere that does not
            # say why.
            raise NotImplementedError(
                f"Table {[i.embedding_config.name for i in sharding_infos]} asks "
                f"for {sharding_type} in an EmbeddingCollection. DynamicEmb "
                "supports table-row-wise for pooled tables only "
                "(EmbeddingBagCollection); use row_wise here."
            )
        else:
            return super().create_embedding_sharding(
                sharding_type=sharding_type,
                sharding_infos=sharding_infos,
                env=env,
                device=device,
                qcomm_codecs_registry=qcomm_codecs_registry,
            )

    def _create_lookups(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            if isinstance(sharding, RwSequenceDynamicEmbeddingSharding):
                for config in sharding._grouped_embedding_configs:
                    if (
                        config.compute_kernel
                        is EmbeddingComputeKernel.CUSTOMIZED_KERNEL
                        and config.pooling is not None
                    ):
                        config.fused_params["use_index_dedup"] = self._use_index_dedup
            self._lookups.append(sharding.create_lookup())

    def _create_hash_size_info(
        self,
        feature_names: List[str],
        ctx: Optional[EmbeddingCollectionContext] = None,
    ) -> None:
        super()._create_hash_size_info(feature_names)

        # _is_lfu_enabled is already set in __init__ from score_strategy parameter

        for i, sharding in enumerate(self._sharding_type_to_sharding.values()):
            nonfuse_table_feature_offsets: List[int] = []
            for table in sharding.embedding_tables():
                nonfuse_table_feature_offsets.append(table.num_features())

            nonfuse_table_feature_offsets_cumsum: List[int] = [0] + list(
                accumulate(nonfuse_table_feature_offsets)
            )

            # Register buffers for this shard
            self.register_buffer(
                f"_nonfuse_table_feature_offsets_host_{i}",
                torch.tensor(
                    nonfuse_table_feature_offsets_cumsum,
                    device="cpu",
                    dtype=torch.int64,
                ),
                persistent=False,
            )

            self.register_buffer(
                f"_nonfuse_table_feature_offsets_device_{i}",
                torch.tensor(
                    nonfuse_table_feature_offsets_cumsum,
                    device=self._device,
                    dtype=torch.int64,
                ),
                persistent=False,
            )

    def _on_hier_path(self) -> List[bool]:
        """Per sharding, whether this call takes the hierarchical dist
        (``shard/hier_output_dist.py``): a training call of a sharding that
        takes it. The sharding decides at its first call (in any mode),
        collectively, so the answer is the same on every rank. Eval calls run
        on TorchRec's dists. The keys carry weights (frequency counters) under
        index dedup with LFU scores or an admission strategy."""
        weights = self._use_index_dedup and (
            self._is_lfu_enabled or self._has_admit_strategy
        )
        return [
            getattr(input_dist, "hier", None) is not None
            and input_dist.hier_setup(weights)
            and self.training
            for input_dist in self._input_dists
        ]

    def _dedup_indices(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        input_feature_splits: List[KeyedJaggedTensor],
        on_hier: List[bool],
    ) -> List[KeyedJaggedTensor]:
        """Deduplicate indices using segmented_unique_cuda.

        A sharding on the hierarchical dist (``on_hier[i]``) keeps its
        deduplicated keys sized on the device. Its ``values`` keep their full
        length, and the lengths say how many are valid. So it needs neither
        the count (a host sync) nor a slice. TorchRec's dists send all of
        ``values``, so theirs are sliced to the count.

        ``segmented_unique_cuda`` takes ``int64`` (or ``uint64``) keys, so
        TorchRec's dists here take no ``int32`` keys. A sharding on the
        hierarchical dist deduplicates its ``int32`` keys as ``int64`` and
        gets the deduplicated keys back in the caller's dtype. Its part 1
        places each key by the rule of the key's own width, as DynamicEmb's
        bucketize does.
        """
        with record_function("## dedup_ec_indices ##"):
            features_by_shards = []
            for i, input_feature in enumerate(input_feature_splits):
                hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                d_table_offset = self.get_buffer(
                    f"_nonfuse_table_feature_offsets_device_{i}"
                )
                input_feature._values = input_feature._values.contiguous()

                table_num = d_table_offset.numel() - 1
                total_B = input_feature.offsets().numel() - 1
                features = hash_size_offset.numel() - 1
                local_batchsize = total_B // features

                values = input_feature.values()
                indices = values.long() if on_hier[i] else values
                offsets = input_feature.offsets().to(torch.int64)
                num_elements = indices.numel()

                # Handle empty input
                if num_elements == 0:
                    dedup_features = KeyedJaggedTensor(
                        keys=input_feature.keys(),
                        lengths=input_feature.lengths().to(torch.int64),
                        offsets=offsets,
                        values=values,
                    )
                    ctx.input_features.append(input_feature)
                    ctx.reverse_indices.append(
                        torch.empty(0, dtype=torch.int64, device=self._device)
                    )
                    # keep ctx.frequency_counters[i] this sharding's, so its
                    # keys carry (empty) weights like every other rank's
                    if self._is_lfu_enabled or self._has_admit_strategy:
                        ctx.frequency_counters.append(
                            torch.empty(0, dtype=torch.uint64, device=self._device)
                        )
                    features_by_shards.append(dedup_features)
                    continue

                # Compute table boundary offsets from jagged offsets (O(T), fully on GPU)
                segmented_range = get_table_range(offsets, d_table_offset)

                # Prepare input_frequencies tensor to control frequency counting
                input_frequencies = None
                if self._is_lfu_enabled or self._has_admit_strategy:
                    input_frequencies = torch.empty(
                        0, dtype=torch.int64, device=self._device
                    )

                # Call segmented_unique_cuda
                (
                    num_uniques,
                    unique_keys,
                    reverse_idx,
                    table_offsets,
                    freq_counters,
                ) = segmented_unique_cuda(
                    indices,
                    segmented_range,
                    table_num,
                    input_frequencies,
                )

                # Compute new lengths and offsets using GPU kernel
                # new_lengths_size = total_B (total number of feature/batch buckets)
                new_lengths, new_offsets = compute_dedup_lengths_cuda(
                    table_offsets,
                    d_table_offset,
                    table_num,
                    local_batchsize,
                    total_B,
                )

                if on_hier[i]:
                    unique_keys = unique_keys.to(values.dtype)
                else:
                    # Get unique values for the KJT
                    # .item() implicitly syncs GPU to CPU
                    total_unique = num_uniques.item()
                    unique_keys = unique_keys[:total_unique]
                    freq_counters = freq_counters[:total_unique]

                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=new_lengths,
                    offsets=new_offsets,
                    values=unique_keys,
                )
                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_idx)

                if self._is_lfu_enabled or self._has_admit_strategy:
                    ctx.frequency_counters.append(freq_counters.to(torch.uint64))

                features_by_shards.append(dedup_features)
        return features_by_shards

    def input_dist(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys(), ctx=ctx)
            self._has_uninitialized_input_dist = False
        # Before index dedup: it slices the keys of TorchRec's dists only.
        on_hier = self._on_hier_path()
        with torch.no_grad():
            unpadded_features = None
            if features.variable_stride_per_key():
                unpadded_features = features
                features = pad_vbe_kjt_lengths(unpadded_features)

            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor,
                )
            # split() reads length_per_key (a host sync) and returns the input
            # itself when one sharding takes all of its features. The
            # hierarchical input dist skips it then (its second wait sets
            # those totals from its one copy); TorchRec's dists keep it, sync
            # included: construct_jagged_tensors reads the totals it leaves on
            # the input.
            if (
                len(self._feature_splits) == 1
                and self._feature_splits[0] == len(features.keys())
                and on_hier[0]
            ):
                features_by_shards = [features]
            else:
                features_by_shards = features.split(self._feature_splits)
            if self._use_index_dedup:
                features_by_shards = self._dedup_indices(
                    ctx, features_by_shards, on_hier
                )

            awaitables = []
            for i, (input_dist, features) in enumerate(
                zip(self._input_dists, features_by_shards)
            ):
                # Attach frequency counters as weights if LFU strategy is enabled
                if self._use_index_dedup and (
                    self._is_lfu_enabled or self._has_admit_strategy
                ):
                    frequency_counters = ctx.frequency_counters[i]
                    features._weights = frequency_counters.float()
                else:
                    features._weights = None

                sharding_ctx = DynamicEmbSequenceShardingContext(
                    features_before_input_dist=features
                )
                if on_hier[i]:
                    input_features, reverse_indices, on_ok = features, None, None
                    if self._use_index_dedup:
                        input_features = ctx.input_features[i]
                        reverse_indices = ctx.reverse_indices[i]
                        # The output dist then expands the index dedup, from
                        # the plan's fan-out table.
                        on_ok = partial(_expanded_by_output_dist, ctx, i)
                    awaitables.append(
                        input_dist.hier_forward(
                            features,
                            input_features,
                            reverse_indices,
                            sharding_ctx,
                            on_ok,
                        )
                    )
                else:
                    awaitables.append(input_dist(features))
                    sharding_ctx.unbucketize_permute_tensor = (
                        input_dist.unbucketize_permute_tensor
                        if hasattr(input_dist, "unbucketize_permute_tensor")
                        else None
                    )
                ctx.sharding_contexts.append(sharding_ctx)
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)
        return KJTListSplitsAwaitable(awaitables, ctx)

    # def create_context(self) -> DynamicEmbeddingCollectionContext:
    #     return DynamicEmbeddingCollectionContext(sharding_contexts=[])

    def create_context(self) -> DynamicEmbeddingCollectionContext:
        # pre-allocate frequency_counters list, ensure all ranks have the same structure
        frequency_counters = (
            [] if not (self._is_lfu_enabled or self._has_admit_strategy) else None
        )
        return DynamicEmbeddingCollectionContext(
            sharding_contexts=[], frequency_counters=frequency_counters
        )


class DynamicEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    """
    DynamicEmbeddingCollectionSharder extends the EmbeddingCollectionSharder class from the TorchREC repo.

    TorchREC performs deduplication in static embedding collections using fuse unique, but fuse unique is not
    suitable for dynamic embedding. Therefore, DynamicEmbeddingCollectionSharder inherits from the
    EmbeddingCollectionSharder class and overrides the shard method to create ShardedDynamicEmbeddingCollection
    and override its input_dist method.

    The usage is completely consistent with TorchREC's EmbeddingCollectionSharder.
    """

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedEmbeddingCollection:
        # Extract global score_strategy from params (only once, as it's a global configuration)
        # Strategy is expected to be consistent across all tables
        global_score_strategy = None
        has_admit_strategy = False
        if global_score_strategy is None:
            for param_name, param_sharding in params.items():
                if (
                    hasattr(param_sharding, "dynamicemb_options")
                    and param_sharding.dynamicemb_options
                ):
                    if param_sharding.dynamicemb_options.score_strategy is not None:
                        global_score_strategy = (
                            param_sharding.dynamicemb_options.score_strategy
                        )

                    if param_sharding.dynamicemb_options.admit_strategy is not None:
                        has_admit_strategy = True

                    break

        # Pass score_strategy directly as a parameter to ShardedDynamicEmbeddingCollection
        return ShardedDynamicEmbeddingCollection(
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            score_strategy=global_score_strategy,  # Pass as direct parameter
            has_admit_strategy=has_admit_strategy,
        )
