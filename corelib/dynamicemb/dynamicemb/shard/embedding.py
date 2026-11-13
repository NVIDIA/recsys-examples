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

from itertools import accumulate
from typing import Dict, List, Optional, Tuple

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

from ..batched_dynamicemb_function import largest_score_per_unique_key
from ..dynamicemb_config import (
    DynamicEmbKernel,
    DynamicEmbScoreStrategy,
    get_eviction_score_strategy,
)
from ..scored_kjt import ScoredKeyedJaggedTensor, without_scores
from .per_key_score import collect_per_key_score, require_scores
from .rw_sharding import RwSequenceDynamicEmbeddingSharding


class DynamicEmbeddingCollectionContext(EmbeddingCollectionContext):
    """EmbeddingCollectionContext with the frequency counters of index dedup.

    With index dedup, ``frequency_counters`` holds one entry per sharding, in
    sharding order. For a sharding that needs counters (LFU or admission) it is
    how often each deduplicated key occurs, or an empty tensor when the
    sharding's input on this rank is empty; for any other, None.
    """

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        seq_vbe_ctx: Optional[List] = None,
        frequency_counters: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> None:
        # TorchRec may retain None for these fields; dedup and VBE append to them.
        super().__init__(
            sharding_contexts=[] if sharding_contexts is None else sharding_contexts,
            input_features=[] if input_features is None else input_features,
            reverse_indices=[] if reverse_indices is None else reverse_indices,
            seq_vbe_ctx=[] if seq_vbe_ctx is None else seq_vbe_ctx,
        )
        # One entry per sharding, in sharding order; None for a sharding that
        # needs no counters.
        self.frequency_counters: List[Optional[torch.Tensor]] = (
            [] if frequency_counters is None else frequency_counters
        )


def collect_frequency_requirements(
    params: Dict[str, ParameterSharding],
) -> Dict[str, Tuple[bool, bool]]:
    """Which sharding types need frequency counters, as ``(is_lfu, has_admit)``.

    The EC dedups and counts frequency once per sharding, over the features of
    the tables that sharding serves. A table belongs to exactly one sharding --
    TorchRec groups the tables by ``ParameterSharding.sharding_type`` -- so this
    groups the tables the same way and answers per sharding type:

    - ``is_lfu``: some table of it evicts by LFU (a single LFU strategy, or a
      compound one such as ``(TIMESTAMP, LFU)``);
    - ``has_admit``: some table of it has an admission strategy.

    The answer is "some table", not the first table's: reading one table's
    options would starve an LFU or admitting table that is not first. A sharding
    type with no DynamicEmb table is absent from the result, i.e. needs nothing.
    Within a sharding the tables may still differ; the ones that do not use the
    counters ignore them.
    """
    requirements: Dict[str, Tuple[bool, bool]] = {}
    for param_sharding in params.values():
        options = getattr(param_sharding, "dynamicemb_options", None)
        if not options:
            continue
        sharding_type = param_sharding.sharding_type
        is_lfu, has_admit = requirements.get(sharding_type, (False, False))
        if options.score_strategy is not None and (
            get_eviction_score_strategy(options.score_strategy)
            == DynamicEmbScoreStrategy.LFU
        ):
            is_lfu = True
        if options.admit_strategy is not None:
            has_admit = True
        requirements[sharding_type] = (is_lfu, has_admit)
    return requirements


class ShardedDynamicEmbeddingCollection(ShardedEmbeddingCollection):
    supported_compute_kernels: List[str] = [
        kernel.value for kernel in EmbeddingComputeKernel
    ] + [DynamicEmbKernel]

    def __init__(
        self,
        *args,
        frequency_requirements: Optional[Dict[str, Tuple[bool, bool]]] = None,
        per_key_score: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Whether this collection's tables take a score per key; the same on every
        # rank. See :func:`collect_per_key_score`.
        self._per_key_score = per_key_score
        # Per sharding, in the order of ``_sharding_type_to_sharding`` -- the order
        # the input dist iterates the shardings in, so ``[i]`` is sharding ``i``.
        # See :func:`collect_frequency_requirements`.
        requirements = frequency_requirements or {}
        self._sharding_is_lfu: List[bool] = []
        self._sharding_has_admit: List[bool] = []
        for sharding_type in self._sharding_type_to_sharding:
            is_lfu, has_admit = requirements.get(sharding_type, (False, False))
            self._sharding_is_lfu.append(is_lfu)
            self._sharding_has_admit.append(has_admit)

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

    def _dedup_indices(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        input_feature_splits: List[KeyedJaggedTensor],
    ) -> List[KeyedJaggedTensor]:
        """Deduplicate indices using segmented_unique_cuda.

        The deduplicated KJT keeps the dtype of the input indices. int32
        indices are deduplicated as int64, the keys the tables store.
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

                indices = input_feature.values()
                offsets = input_feature.offsets().to(torch.int64)
                num_elements = indices.numel()

                # Handle empty input
                if num_elements == 0:
                    dedup_features = KeyedJaggedTensor(
                        keys=input_feature.keys(),
                        lengths=input_feature.lengths().to(torch.int64),
                        offsets=offsets,
                        values=indices,
                    )
                    if isinstance(input_feature, ScoredKeyedJaggedTensor):
                        dedup_features = ScoredKeyedJaggedTensor.from_kjt(
                            dedup_features, input_feature.scores()
                        )
                    ctx.input_features.append(input_feature)
                    ctx.reverse_indices.append(
                        torch.empty(0, dtype=torch.int64, device=self._device)
                    )
                    # No keys, so no counts. A sharding that needs counters still
                    # gets an entry, empty, so input_dist sends empty weights as
                    # the ranks with keys send theirs.
                    ctx.frequency_counters.append(
                        torch.empty(0, dtype=torch.uint64, device=self._device)
                        if self._sharding_is_lfu[i] or self._sharding_has_admit[i]
                        else None
                    )
                    features_by_shards.append(dedup_features)
                    continue

                # Compute table boundary offsets from jagged offsets (O(T), fully on GPU)
                segmented_range = get_table_range(offsets, d_table_offset)

                # Prepare input_frequencies tensor to control frequency counting
                needs_counters = self._sharding_is_lfu[i] or self._sharding_has_admit[i]
                input_frequencies = None
                if needs_counters:
                    input_frequencies = torch.empty(
                        0, dtype=torch.int64, device=self._device
                    )

                # segmented_unique_cuda takes 64-bit keys. The lookup widens
                # int32 indices to int64, so the dedup does the same.
                indices_64bit = (
                    indices.to(torch.int64) if indices.dtype == torch.int32 else indices
                )

                # Call segmented_unique_cuda
                (
                    num_uniques,
                    unique_keys,
                    reverse_idx,
                    table_offsets,
                    freq_counters,
                ) = segmented_unique_cuda(
                    indices_64bit,
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

                # Get unique values for the KJT
                # .item() implicitly syncs GPU to CPU
                total_unique = num_uniques.item()
                # Return the keys in the input dtype. The input dist then sees
                # the same dtype as without index dedup, and as on a rank whose
                # input is empty (handled above).
                unique_keys = unique_keys[:total_unique].to(indices.dtype)

                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=new_lengths,
                    offsets=new_offsets,
                    values=unique_keys,
                )
                # Evaluation may go without scores (see ``input_dist``), so what
                # follows the keys is what the input carries.
                if isinstance(input_feature, ScoredKeyedJaggedTensor):
                    # A key that repeats keeps the largest of its scores -- the
                    # rule the tables apply to repeats within a batch, so
                    # merging here and again there gives what merging once would.
                    dedup_features = ScoredKeyedJaggedTensor.from_kjt(
                        dedup_features,
                        largest_score_per_unique_key(
                            input_feature.scores(), reverse_idx, total_unique
                        ),
                    )
                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_idx)

                ctx.frequency_counters.append(
                    freq_counters[:total_unique].to(torch.uint64)
                    if needs_counters
                    else None
                )

                features_by_shards.append(dedup_features)
        return features_by_shards

    def input_dist(
        self,
        ctx: DynamicEmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        if self._per_key_score:
            if self.training:
                require_scores(features)
            else:
                # Evaluation reads no score, so it does not have to be given
                # any; one given is dropped here, on every rank alike, so the
                # ranks agree on whether the distribution sends scores.
                features = without_scores(features)
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys(), ctx=ctx)
            self._has_uninitialized_input_dist = False
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
            features_by_shards = features.split(self._feature_splits)
            if self._use_index_dedup:
                features_by_shards = self._dedup_indices(ctx, features_by_shards)

            awaitables = []
            for i, (input_dist, features) in enumerate(
                zip(self._input_dists, features_by_shards)
            ):
                # Attach frequency counters as weights if LFU strategy is enabled
                frequency_counters = (
                    ctx.frequency_counters[i] if self._use_index_dedup else None
                )
                if frequency_counters is not None:
                    features._weights = frequency_counters.float()
                else:
                    features._weights = None

                awaitables.append(input_dist(features))
                ctx.sharding_contexts.append(
                    SequenceShardingContext(
                        features_before_input_dist=features,
                        unbucketize_permute_tensor=(
                            input_dist.unbucketize_permute_tensor
                            if hasattr(input_dist, "unbucketize_permute_tensor")
                            else None
                        ),
                    )
                )
            if unpadded_features is not None:
                self._compute_sequence_vbe_context(ctx, unpadded_features)
        return KJTListSplitsAwaitable(awaitables, ctx)

    def create_context(self) -> DynamicEmbeddingCollectionContext:
        return DynamicEmbeddingCollectionContext(sharding_contexts=[])

    def prefetch(
        self,
        dist_input: KJTList,
        forward_stream: Optional[torch.cuda.Stream] = None,
        ctx: Optional[EmbeddingCollectionContext] = None,
    ) -> None:
        """Keep pipeline prefetch idempotent for input_dist-resolved groups."""
        for lookup, features in zip(self._lookups, dist_input):
            lookup.prefetch(features, forward_stream)


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
        frequency_requirements = collect_frequency_requirements(params)
        per_key_score = collect_per_key_score(params)

        return ShardedDynamicEmbeddingCollection(
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            frequency_requirements=frequency_requirements,
            per_key_score=per_key_score,
        )
