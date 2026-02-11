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

from typing import List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DynamicEmbPoolingMode
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.key_value_table import (
    Cache,
    KeyValueTableCachingFunction,
    KeyValueTableFunction,
    Storage,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.types import Counter
from dynamicemb_extensions import (
    EvictStrategy,
    expand_table_ids_cuda,
    gather_embedding,
    gather_embedding_pooled,
    get_table_range,
    reduce_grads,
    segmented_unique_cuda,
)


def segmented_unique(
    keys: torch.Tensor,
    segment_range: torch.Tensor,
    evict_strategy: Optional[EvictStrategy] = None,
    frequency_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform segmented unique operation on keys with segment_range.

    This function deduplicates keys within each table segment, using the
    GPU-accelerated segmented_unique_cuda kernel.

    Args:
        keys: Input key tensor (int64 or uint64)
        segment_range: Table boundary offsets where segment_range[i] is the
                       start index for table i (int64)
        evict_strategy: Optional eviction strategy (for LFU mode)
        frequency_counts: Optional input frequency counts per key

    Returns:
        Tuple of (unique_keys, reverse_indices, unique_keys_table_range,
                  h_unique_keys_table_range, output_scores)
    """
    num_keys = keys.size(0)
    num_tables = segment_range.size(0) - 1
    device = keys.device

    # Handle empty input
    if num_keys == 0:
        empty_keys = torch.empty(0, dtype=keys.dtype, device=device)
        empty_reverse_indices = torch.empty(0, dtype=torch.int64, device=device)
        d_table_range = torch.zeros(num_tables + 1, dtype=torch.int64, device=device)
        h_table_range = torch.zeros(num_tables + 1, dtype=torch.int64, device="cpu")
        return (
            empty_keys,
            empty_reverse_indices,
            d_table_range,
            h_table_range,
            torch.Tensor(),
        )

    # Determine if we need frequency output
    is_lfu_enabled = evict_strategy == EvictStrategy.KLfu if evict_strategy else False
    need_frequency_output = is_lfu_enabled or frequency_counts is not None

    # Generate table_ids from segment_range
    # segment_range has size (num_tables + 1), treating each table as one feature
    # with local_batch_size=1. When table_offsets_in_feature=None, each feature
    # maps to a separate table.
    table_ids = expand_table_ids_cuda(
        segment_range,
        None,  # table_offsets_in_feature=None -> each feature = separate table
        num_tables,  # ignored when table_offsets_in_feature is None
        1,  # local_batch_size
        num_keys,
    )

    # Prepare input_frequencies tensor
    input_frequencies = None
    if frequency_counts is not None:
        input_frequencies = frequency_counts
    elif need_frequency_output:
        # Enable frequency counting with count=1 per key
        input_frequencies = torch.empty(0, dtype=torch.int64, device=device)

    # Call segmented_unique_cuda
    (
        num_uniques,
        unique_keys,
        reverse_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, table_ids, num_tables, input_frequencies)

    # Get total unique count and slice the output
    # .item() implicitly syncs
    total_unique = num_uniques.item()
    unique_keys_out = unique_keys[:total_unique]

    # Prepare output tensors in the expected format
    h_table_offsets = table_offsets.cpu()

    output_scores = torch.Tensor()
    if need_frequency_output and total_unique > 0:
        output_scores = freq_counters[:total_unique]

    return (
        unique_keys_out,
        reverse_indices,
        table_offsets,
        h_table_offsets,
        output_scores,
    )


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    caches: List[Optional[Cache]],
    storages: List[Storage],
    feature_offsets: torch.Tensor,
    initializers: List[BaseDynamicEmbInitializer],
    training: bool = True,
    forward_stream: Optional[torch.cuda.Stream] = None,
):
    table_num = len(storages)
    assert table_num != 0
    caching = caches[0] is not None

    indices_table_range = get_table_range(offsets, feature_offsets)
    if training or caching:
        (
            unique_indices,
            reverse_indices,
            unique_indices_table_range,
            h_unique_indices_table_range,
            _,
        ) = segmented_unique(indices, indices_table_range)
        # TODO: only return device unique_indices_table_range
        # h_unique_indices_table_range = unique_indices_table_range.cpu()
    else:
        h_unique_indices_table_range = indices_table_range.cpu()
        unique_indices = indices

    for i in range(table_num):
        begin = h_unique_indices_table_range[i]
        end = h_unique_indices_table_range[i + 1]
        unique_indices_per_table = unique_indices[begin:end]

        KeyValueTableCachingFunction.prefetch(
            caches[i],
            storages[i],
            unique_indices_per_table,
            initializers[i],
            training,
            forward_stream,
        )


class DynamicEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        caches: List[Optional[Cache]],
        storages: List[Storage],
        feature_offsets: torch.Tensor,
        output_dtype: torch.dtype,
        initializers: List[BaseDynamicEmbInitializer],
        optimizer: BaseDynamicEmbeddingOptimizer,
        enable_prefetch: bool = False,
        training: bool = True,
        admit_strategy=None,
        evict_strategy=None,
        frequency_counters: Optional[torch.Tensor] = None,
        admission_counter: Optional[list[Counter]] = None,
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.NONE,
        total_D: int = 0,
        batch_size: int = 0,
        dims: Optional[List[int]] = None,
        max_D: int = 0,
        D_offsets: Optional[torch.Tensor] = None,
        *args,
    ):
        table_num = len(storages)
        assert table_num != 0
        emb_dtype = storages[0].embedding_dtype()
        emb_dim = storages[0].embedding_dim()
        caching = caches[0] is not None

        # Determine if we have mixed dimensions (pooling multi-dim)
        is_pooling = pooling_mode != DynamicEmbPoolingMode.NONE
        mixed_D = is_pooling and dims is not None and max_D > min(dims)

        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        lfu_accumulated_frequency = None
        indices_table_range = get_table_range(offsets, feature_offsets)
        if training or caching:
            (
                unique_indices,
                reverse_indices,
                unique_indices_table_range,
                h_unique_indices_table_range,
                lfu_accumulated_frequency,
            ) = segmented_unique(
                indices,
                indices_table_range,
                EvictStrategy(evict_strategy.value) if evict_strategy else None,
                frequency_counts_int64,
            )
        else:
            h_unique_indices_table_range = indices_table_range.cpu()
            unique_indices = indices

        # Allocate unique_embs buffer.
        # For mixed_D pooling: pad to max_D so all rows share the same stride.
        # For uniform dim: allocate with emb_dim directly.
        out_dim = max_D if mixed_D else emb_dim
        unique_embs = torch.empty(
            unique_indices.shape[0],
            out_dim,
            dtype=emb_dtype,
            device=indices.device,
        )

        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            lfu_accumulated_frequency_per_table = (
                lfu_accumulated_frequency[begin:end]
                if lfu_accumulated_frequency is not None
                and lfu_accumulated_frequency.numel() > 0
                else None
            )

            dim_i = dims[i] if dims is not None else emb_dim
            unique_embs_per_table = unique_embs[begin:end, :dim_i]
            if caching:
                KeyValueTableCachingFunction.lookup(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    enable_prefetch,
                    training,
                    EvictStrategy(evict_strategy.value) if evict_strategy else None,
                    lfu_accumulated_frequency_per_table,
                    admit_strategy,
                    admission_counter[i] if admission_counter else None,
                )
            else:
                KeyValueTableFunction.lookup(
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    training,
                    EvictStrategy(evict_strategy.value) if evict_strategy else None,
                    lfu_accumulated_frequency_per_table,
                    admit_strategy,
                    admission_counter[i] if admission_counter else None,
                )

        if is_pooling:
            combiner = (
                0 if pooling_mode == DynamicEmbPoolingMode.SUM else 1
            )  # 0=SUM, 1=MEAN
        else:
            combiner = -1  # sequence (no pooling)

        if is_pooling:
            output_embs = torch.empty(
                batch_size, total_D, dtype=output_dtype, device=indices.device
            )
            if not (training or caching):
                reverse_indices = torch.arange(
                    indices.numel(), device=indices.device, dtype=torch.int64
                )
            gather_embedding_pooled(
                unique_embs,
                output_embs,
                reverse_indices,
                offsets,
                combiner,
                total_D,
                batch_size,
                D_offsets,
                max_D,
            )
        else:
            if training or caching:
                output_embs = torch.empty(
                    indices.shape[0],
                    emb_dim,
                    dtype=output_dtype,
                    device=indices.device,
                )
                gather_embedding(unique_embs, output_embs, reverse_indices)
            else:
                output_embs = unique_embs

        if training:
            ctx.unique_indices = unique_indices
            ctx.reverse_indices = reverse_indices
            ctx.h_unique_indices_table_range = h_unique_indices_table_range
            ctx.caches = caches
            ctx.storages = storages
            ctx.optimizer = optimizer
            ctx.pooling_mode = pooling_mode
            ctx.combiner = combiner
            ctx.offsets = offsets
            ctx.batch_size = batch_size
            ctx.total_D = total_D
            ctx.emb_dim = emb_dim
            ctx.mixed_D = mixed_D
            ctx.dims = dims
            ctx.max_D = max_D
            ctx.D_offsets = D_offsets
            ctx.num_features = (
                (offsets.shape[0] - 1) // batch_size if batch_size > 0 else 0
            )

        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        h_unique_indices_table_range = ctx.h_unique_indices_table_range
        caches = ctx.caches
        storages = ctx.storages
        optimizer = ctx.optimizer
        caching = caches[0] is not None
        grads = grads.contiguous()

        # clip the gradient before reduction
        if optimizer.need_gradient_clipping():
            optimizer.clip_gradient(grads)

        is_pooling = ctx.pooling_mode != DynamicEmbPoolingMode.NONE
        if is_pooling:
            # Pooling backward: grads is [B, total_D].  D_offsets is always
            # available (uniform: [0, D, 2D, ...]; mixed: per-feature).
            # The kernel uses D_offsets for per-feature source addressing and
            # fuses MEAN scaling.  No reshape needed for either case.
            out_dim = ctx.max_D if ctx.mixed_D else ctx.emb_dim
            unique_grads = reduce_grads(
                ctx.reverse_indices,
                grads,
                ctx.unique_indices.numel(),
                ctx.batch_size,
                out_dim,
                ctx.offsets,
                ctx.D_offsets,
                ctx.combiner,
                ctx.total_D,
            )
        else:
            # Sequence: no offsets -> arange gather_ids internally.
            unique_grads = reduce_grads(
                ctx.reverse_indices,
                grads,
                ctx.unique_indices.numel(),
                ctx.batch_size,
                ctx.emb_dim,
            )

        optimizer.step()
        table_num = len(storages)
        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = ctx.unique_indices[begin:end]

            # Slice to the actual dim for this table (for uniform-dim,
            # dim_i == max_D so this is a full-row slice — no copy).
            dim_i = ctx.dims[i] if ctx.dims is not None else ctx.emb_dim
            unique_grads_per_table = unique_grads[begin:end, :dim_i]
            if dim_i < ctx.max_D:
                unique_grads_per_table = unique_grads_per_table.contiguous()

            if caching:
                KeyValueTableCachingFunction.update(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_grads_per_table,
                    optimizer,
                )
            else:
                KeyValueTableFunction.update(
                    storages[i],
                    unique_indices_per_table,
                    unique_grads_per_table,
                    optimizer,
                )

        return (None,) * 21
