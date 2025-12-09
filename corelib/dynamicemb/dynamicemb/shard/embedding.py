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
from typing import Dict, List, Optional

import torch
from dynamicemb_extensions import dedup_input_indices
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

from ..dynamicemb_config import DynamicEmbKernel, DynamicEmbScoreStrategy
from ..planner.rw_sharding import RwSequenceDynamicEmbeddingSharding
from ..unique_op import UniqueOp


class DynamicEmbeddingCollectionContext(EmbeddingCollectionContext):
    """Extended EmbeddingCollectionContext that includes frequency_counters for LFU strategy."""

    def __init__(
        self,
        sharding_contexts: Optional[List[SequenceShardingContext]] = None,
        input_features: Optional[List[KeyedJaggedTensor]] = None,
        reverse_indices: Optional[List[torch.Tensor]] = None,
        seq_vbe_ctx: Optional[List] = None,
        frequency_counters: Optional[List[torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            sharding_contexts, input_features, reverse_indices, seq_vbe_ctx
        )
        self.frequency_counters: List[torch.Tensor] = frequency_counters or []


class ShardedDynamicEmbeddingCollection(ShardedEmbeddingCollection):
    supported_compute_kernels: List[str] = [
        kernel.value for kernel in EmbeddingComputeKernel
    ] + [DynamicEmbKernel]

    def __init__(
        self,
        *args,
        score_strategy: Optional[DynamicEmbScoreStrategy] = None,
        has_admit_strategy: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the global score strategy
        self._score_strategy = score_strategy
        self._is_lfu_enabled = (
            (score_strategy == DynamicEmbScoreStrategy.LFU) if score_strategy else False
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

        if self._use_index_dedup:
            reserve_keys = torch.tensor(2, dtype=torch.int64, device=self._device)
            reserve_vals = torch.tensor(2, dtype=torch.uint64, device=self._device)
            counter = torch.tensor(1, dtype=torch.uint64, device=self._device)
            self._unique_op = UniqueOp(reserve_keys, reserve_vals, counter, 2)
            device_properties = torch.cuda.get_device_properties(self._device.index)
            self._device_num_sms = device_properties.multi_processor_count

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
        with record_function("## dedup_ec_indices ##"):
            features_by_shards = []
            for i, input_feature in enumerate(input_feature_splits):
                hash_size_offset = self.get_buffer(f"_hash_size_offset_tensor_{i}")
                h_table_offset = self.get_buffer(
                    f"_nonfuse_table_feature_offsets_host_{i}"
                )
                d_table_offset = self.get_buffer(
                    f"_nonfuse_table_feature_offsets_device_{i}"
                )
                input_feature._values = input_feature._values.contiguous()
                # for debug
                # hash_size_cumsum = self.get_buffer(
                #    f"_hash_size_cumsum_tensor_{i}"
                # )

                # (
                #    debug_lengths,
                #    debug_offsets,
                #    debug_unique_indices,
                #    debug_reverse_indices,
                # ) = torch.ops.fbgemm.jagged_unique_indices(
                #    hash_size_cumsum,
                #    hash_size_offset,
                #    input_feature.offsets().to(torch.int64),
                #    input_feature.values().to(torch.int64),
                # )

                table_num = h_table_offset.shape[0] - 1
                total_B = input_feature.offsets().shape[0] - 1
                features = hash_size_offset.shape[0] - 1
                local_batchsize = total_B // features

                indices = input_feature.values()
                offsets = input_feature.offsets()
                lengths = input_feature.lengths()
                dtype_convert = False
                if indices.dtype != torch.int64:
                    indices.dtype
                    indices_input = indices.to(torch.int64)
                    dtype_convert = True
                else:
                    indices_input = indices
                reverse_idx = torch.empty_like(
                    indices, dtype=torch.uint64, device=self._device
                )
                unique_idx_list = [
                    torch.empty_like(indices, dtype=torch.int64, device=self._device)
                    for i in range(table_num)
                ]
                h_unique_nums = torch.empty(table_num, dtype=torch.uint64, device="cpu")
                d_unique_nums = torch.empty(
                    table_num, dtype=torch.uint64, device=self._device
                )

                h_unique_offsets = torch.empty(
                    table_num + 1, dtype=torch.uint64, device="cpu"
                )
                d_unique_offsets = torch.zeros(
                    table_num + 1, dtype=torch.uint64, device=self._device
                )

                new_offsets = torch.empty_like(offsets, device=self._device)
                new_lengths = torch.empty_like(lengths, device=self._device)

                # Only create frequency_counters if LFU strategy is enabled
                # For non-LFU strategies, pass empty tensor (C++ extension will check size)
                if self._is_lfu_enabled or self._has_admit_strategy:
                    # TODO: use only one frequency_counters tensor for all tables
                    # frequency_counters = torch.zeros_like(
                    #     indices_input, device=self._device, dtype=torch.uint64
                    # )
                    frequency_counters_list = [
                        torch.zeros_like(
                            indices_input, dtype=torch.uint64, device=self._device
                        )
                        for i in range(table_num)
                    ]
                    dedup_input_indices(
                        indices_input,
                        offsets,
                        h_table_offset,
                        d_table_offset,
                        table_num,
                        local_batchsize,
                        reverse_idx,
                        h_unique_nums,
                        d_unique_nums,
                        h_unique_offsets,
                        d_unique_offsets,
                        unique_idx_list,
                        new_offsets,
                        new_lengths,
                        self._device_num_sms,
                        self._unique_op,
                        frequency_counters_list,
                    )
                else:
                    # Empty tensor for non-LFU and non-admit strategies
                    dedup_input_indices(
                        indices_input,
                        offsets,
                        h_table_offset,
                        d_table_offset,
                        table_num,
                        local_batchsize,
                        reverse_idx,
                        h_unique_nums,
                        d_unique_nums,
                        h_unique_offsets,
                        d_unique_offsets,
                        unique_idx_list,
                        new_offsets,
                        new_lengths,
                        self._device_num_sms,
                        self._unique_op,
                    )

                unique_num = h_unique_offsets[-1].item()
                unique_idx = torch.empty(
                    unique_num, dtype=torch.int64, device=indices.device
                )
                frequency_counters = torch.empty(
                    unique_num, device=self._device, dtype=torch.uint64
                )
                # TODO: check non_blocking=True is valid for device tensor to device tensor
                for i in range(table_num):
                    start_pos = h_unique_offsets[i].item()
                    end_pos = h_unique_offsets[i + 1].item()
                    length = end_pos - start_pos
                    unique_idx[start_pos:end_pos].copy_(
                        unique_idx_list[i][:length], non_blocking=True
                    )
                    if self._is_lfu_enabled or self._has_admit_strategy:
                        frequency_counters[start_pos:end_pos].copy_(
                            frequency_counters_list[i][:length], non_blocking=True
                        )

                if dtype_convert:
                    unique_idx_out = torch.empty(
                        unique_num, dtype=indices.dtype, device=indices.device
                    )
                    unique_idx_out.copy_(unique_idx, non_blocking=True)
                else:
                    unique_idx_out = unique_idx

                dedup_features = KeyedJaggedTensor(
                    keys=input_feature.keys(),
                    lengths=new_lengths,
                    offsets=new_offsets,
                    values=unique_idx_out,
                )
                ctx.input_features.append(input_feature)
                ctx.reverse_indices.append(reverse_idx)
                # Only store frequency_counters if LFU or admit strategy is enabled
                if self._is_lfu_enabled or self._has_admit_strategy:
                    ctx.frequency_counters.append(frequency_counters)
                    assert frequency_counters.size(0) == unique_idx_out.size(
                        0
                    ), f"Frequency counters size {frequency_counters.size(0)} doesn't match unique indices size {unique_idx_out.size(0)}"

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
                if (
                    self._use_index_dedup
                    and (self._is_lfu_enabled or self._has_admit_strategy)
                    and len(ctx.frequency_counters) > i
                ):
                    frequency_counters = ctx.frequency_counters[i]
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
