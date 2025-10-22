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

from typing import List, Optional

import torch
from dynamicemb.dynamicemb_config import (
    DynamicEmbInitializerArgs,
    DynamicEmbPoolingMode,
    dyn_emb_to_torch,
)
from dynamicemb.initializer import BaseDynamicEmbInitializer
from dynamicemb.key_value_table import (
    Cache,
    KeyValueTable,
    KeyValueTableCachingFunction,
    KeyValueTableFunction,
    Storage,
)
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.unique_op import UniqueOp
from dynamicemb_extensions import (
    DynamicEmbTable,
    EvictStrategy,
    find_and_initialize,
    find_or_insert,
    gather_embedding,
    get_table_range,
    lookup_backward,
    lookup_forward,
    reduce_grads,
    segmented_unique,
)


# TODO: BatchedDynamicEmbeddingFunction is more concrete.
class DynamicEmbeddingBagFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: torch.Tensor,
        offsets: torch.Tensor,  # [feature_num * batch_size]
        use_index_dedup: bool,
        table_offsets_in_feature: List[int],
        tables: List[DynamicEmbTable],
        scores: List[int],
        total_D: int,
        dims: List[int],
        feature_table_map: List[int],
        embedding_dtype: torch.dtype,
        output_dtype: torch.dtype,
        pooling_mode: DynamicEmbPoolingMode,
        device_num_sms: int,
        unique_op: UniqueOp,
        device: torch.device,
        optimizer: BaseDynamicEmbeddingOptimizer,
        training: bool,
        eval_initializers: List[DynamicEmbInitializerArgs],
        *args,
    ):
        # TODO: remove unnecessary params.
        # TODO:need check dimension is right
        table_num = len(tables)
        assert table_num == len(table_offsets_in_feature) - 1
        feature_num = len(feature_table_map)

        # split indices, offsets by table.
        indices_list: List[torch.Tensor] = []
        biased_offsets_list: List[torch.Tensor] = []

        feature_num = table_offsets_in_feature[-1]
        feature_batch_size = offsets.shape[0] - 1
        batch_size = feature_batch_size // feature_num
        assert feature_batch_size % feature_num == 0
        # The offsets is on device in torchrec, however, the unique_op and lookup_op are done table by table.
        # So we need to know one index belong to which table, to let op know the boundary.
        # Therefore, copy offsets to cpu is necessary, otherwise, many things will be coupled together.
        # For example, UniqueOp have to accept (indices, offsets, table_offsets_in_feature, table_id) as inputs,
        #   and we have to copy table_offsets_in_feature from cpu to gpu.

        # TODO: if the batch size is large, we can develop a kernel to get: indices boundary.

        h_offsets = offsets.to("cpu")

        for i in range(table_num):
            feature_id_begin, feature_id_end = (
                table_offsets_in_feature[i],
                table_offsets_in_feature[i + 1],
            )
            offset_begin, offset_end = (
                feature_id_begin * batch_size,
                feature_id_end * batch_size,
            )
            # include offset_end to know the boundary of the last feature.
            biased_offsets_list.append(offsets[offset_begin : offset_end + 1])

            indices_begin, indices_end = h_offsets[offset_begin], h_offsets[offset_end]
            indices_list.append(indices[indices_begin:indices_end])

        unique_indices_list = []
        inverse_indices_list = []
        unique_count_list = []
        for i in range(table_num):
            unique_indices, inverse_indices = torch.unique(
                indices_list[i], sorted=False, return_inverse=True
            )
            unique_indices_list.append(unique_indices)
            inverse_indices_list.append(
                inverse_indices.to(biased_offsets_list[i].dtype)
            )
            unique_count_list.append(inverse_indices.shape[0])

        unique_embedding_list = []
        for i in range(table_num):
            unique_indices = unique_indices_list[i]
            num_unique_indices = unique_indices.shape[0]
            tmp_value_type_torch = dyn_emb_to_torch(tables[i].value_type())
            tmp_unique_embs = torch.empty(
                num_unique_indices, dims[i], dtype=tmp_value_type_torch, device=device
            )

            if training:
                find_or_insert(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    tmp_unique_embs,
                    scores[i],
                )
            else:
                find_and_initialize(
                    tables[i],
                    num_unique_indices,
                    unique_indices,
                    tmp_unique_embs,
                    eval_initializers[i].as_ctype(),
                )

            unique_embedding_list.append(tmp_unique_embs)

        if pooling_mode == DynamicEmbPoolingMode.NONE:
            combiner = -1
            # total_embs_num = indices.shape[0]
            total_embs_num = indices.numel()
            # All tables have the same dim.
            embs = torch.empty(
                total_embs_num, dims[0], dtype=output_dtype, device=device
            )
        else:
            if pooling_mode == DynamicEmbPoolingMode.SUM:
                combiner = 0
            elif pooling_mode == DynamicEmbPoolingMode.MEAN:
                combiner = 1
            else:
                raise ValueError("Not support pooling mode.")
            total_embs_num = offsets.shape[0] - 1
            embs = torch.empty(batch_size, total_D, dtype=output_dtype, device=device)

        # TODO:To combine all the table's combiner kernel together, we first need to merge the indices. This may require developing a customized kernel to achieve this.
        accum_D = 0
        for i in range(table_num):
            num_embeddings = biased_offsets_list[i].shape[0] - 1
            lookup_forward(
                unique_embedding_list[i],
                embs,
                biased_offsets_list[i],
                inverse_indices_list[i],
                combiner,
                total_D,
                accum_D,
                dims[i],
                num_embeddings,
                batch_size,
                device_num_sms,
            )
            accum_D += dims[i] * (num_embeddings // batch_size)
            assert num_embeddings % batch_size == 0

        if training:
            backward_tensors = [indices, offsets]
            ctx.save_for_backward(*backward_tensors)
            ctx.tables = tables
            ctx.unique_indices_list = unique_indices_list
            ctx.inverse_indices_list = inverse_indices_list
            ctx.biased_offsets_list = biased_offsets_list
            ctx.dims = dims
            ctx.batch_size = batch_size
            ctx.feature_num = feature_num
            ctx.feature_table_map = feature_table_map
            ctx.device = device
            ctx.optimizer = optimizer
            ctx.scores = scores
            ctx.combiner = combiner

        return embs

    @staticmethod
    def backward(ctx, grad):
        # if we want to do the value check, we shouldn't to update the embeddings ].
        tables = ctx.tables
        unique_indices_list = ctx.unique_indices_list
        inverse_indices_list = ctx.inverse_indices_list
        biased_offsets_list = ctx.biased_offsets_list
        dims = ctx.dims
        batch_size = ctx.batch_size
        ctx.feature_num
        feature_table_map_list = ctx.feature_table_map
        indices, offsets = ctx.saved_tensors
        device = ctx.device
        optimizer = ctx.optimizer
        table_num = len(tables)
        combiner = ctx.combiner

        offsets_list_per_table = []
        for i in range(table_num):
            offsets_list_per_table.append(
                biased_offsets_list[i] - biased_offsets_list[i][0]
            )

        feature_num_per_table = [0] * table_num
        for i in range(len(feature_table_map_list)):
            feature_num_per_table[feature_table_map_list[i]] += 1

        dim_offset_per_table = [0]
        for i in range(table_num):
            dim_offset_per_table.append(
                feature_num_per_table[i] * dims[i] + dim_offset_per_table[i]
            )

        dyn_emb_to_torch(tables[0].value_type())
        dyn_emb_to_torch(tables[0].key_type())

        unique_count_list = []
        for i in range(table_num):
            unique_count_list.append(unique_indices_list[i].shape[0])

        unique_backward_grads_per_table = []
        for i in range(table_num):
            unique_backward_grads_per_table.append(
                torch.zeros(
                    unique_count_list[i] * dims[i], dtype=grad.dtype, device=device
                )
            )

        # dims_tensor = torch.tensor(dims_list,dtype=torch.int32,device=device)
        for i in range(table_num):
            grad_for_table = grad[
                :, dim_offset_per_table[i] : dim_offset_per_table[i + 1]
            ]

            splits = torch.split(grad_for_table, dims[i], dim=-1)
            result = torch.cat(splits, dim=0)
            grad_for_table = result.reshape(-1, dims[i]).contiguous()
            lookup_backward(
                grad_for_table,
                unique_backward_grads_per_table[i],
                unique_indices_list[i],
                inverse_indices_list[i],
                offsets_list_per_table[i],
                dims[i],
                table_num,
                batch_size,
                feature_num_per_table[i],
                offsets_list_per_table[i][-1].item(),
                combiner,
            )

        unique_grads_per_table = []
        for i, unique_grad in enumerate(unique_backward_grads_per_table):
            unique_grads_per_table.append(unique_grad.reshape(-1, dims[i]))

        optimizer.update(tables, unique_indices_list, unique_grads_per_table)

        return (None,) * 19


def dynamicemb_prefetch(
    indices: torch.Tensor,
    offsets: torch.Tensor,
    caches: List[Optional[Cache]],
    storages: List[Storage],
    feature_offsets: torch.Tensor,
    initializers: List[BaseDynamicEmbInitializer],
    unique_op,
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
            inverse,
            unique_indices_table_range,
            h_unique_indices_table_range,
            _,
        ) = segmented_unique(indices, indices_table_range, unique_op)
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


class DynamicEmbeddingFunctionV2(torch.autograd.Function):
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
        unique_op,
        enable_prefetch: bool = False,
        input_dist_dedup: bool = False,
        training: bool = True,
        frequency_counters: Optional[torch.Tensor] = None,
        *args,
    ):
        table_num = len(storages)
        assert table_num != 0
        emb_dtype = storages[0].embedding_dtype()
        emb_dim = storages[0].embedding_dim()
        caching = caches[0] is not None

        is_lfu_enabled = False
        if isinstance(storages[0], KeyValueTable):
            is_lfu_enabled = storages[0].evict_strategy() == EvictStrategy.KLfu

        frequency_counts_int64 = None
        if frequency_counters is not None:
            frequency_counts_int64 = frequency_counters.long()

        # TODO: Use frequency_counts_uint64 for LFU strategy in pooled embeddings

        lfu_accumulated_frequency = None
        indices_table_range = get_table_range(offsets, feature_offsets)
        if training or caching:
            (
                unique_indices,
                inverse,
                unique_indices_table_range,
                h_unique_indices_table_range,
                lfu_accumulated_frequency,
            ) = segmented_unique(
                indices,
                indices_table_range,
                unique_op,
                is_lfu_enabled,
                frequency_counts_int64,
            )
            # TODO: only return device unique_indices_table_range
            # h_unique_indices_table_range = unique_indices_table_range.cpu()
        else:
            h_unique_indices_table_range = indices_table_range.cpu()
            unique_indices = indices

        unique_embs = torch.empty(
            unique_indices.shape[0], emb_dim, dtype=emb_dtype, device=indices.device
        )

        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]
            # Slice lfu_accumulated_frequency to match the table
            lfu_accumulated_frequency_per_table = (
                lfu_accumulated_frequency[begin:end]
                if lfu_accumulated_frequency is not None
                and lfu_accumulated_frequency.numel() > 0
                else None
            )

            if caching:
                KeyValueTableCachingFunction.lookup(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    enable_prefetch,
                    training,
                    lfu_accumulated_frequency_per_table,
                )
            else:
                KeyValueTableFunction.lookup(
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    initializers[i],
                    training,
                    lfu_accumulated_frequency_per_table,
                )

        if training or caching:
            output_embs = torch.empty(
                indices.shape[0], emb_dim, dtype=output_dtype, device=indices.device
            )
            gather_embedding(unique_embs, output_embs, inverse)
        else:
            output_embs = unique_embs

        if training:
            # save context
            backward_tensors = [
                indices,
            ]
            ctx.save_for_backward(*backward_tensors)
            ctx.input_dist_dedup = input_dist_dedup
            if input_dist_dedup:
                ctx.unique_indices = unique_indices
                ctx.unique_embs = unique_embs
                ctx.inverse = inverse
            ctx.indices_table_range = indices_table_range
            ctx.h_indices_table_range = indices_table_range.cpu()
            ctx.h_unique_indices_table_range = h_unique_indices_table_range
            ctx.unique_indices_table_range = unique_indices_table_range
            ctx.caches = caches
            ctx.storages = storages
            ctx.optimizer = optimizer
            ctx.enable_prefetch = enable_prefetch

        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        (indices,) = ctx.saved_tensors
        indices_table_range = ctx.indices_table_range
        h_indices_table_range = ctx.h_indices_table_range
        h_unique_indices_table_range = ctx.h_unique_indices_table_range
        ctx.unique_indices_table_range
        caches = ctx.caches
        storages = ctx.storages
        optimizer = ctx.optimizer
        caching = caches[0] is not None

        input_dist_dedup = ctx.input_dist_dedup
        if input_dist_dedup:
            unique_indices = ctx.unique_indices
            unique_embs = ctx.unique_embs
            ctx.inverse
        unique_indices, unique_embs = reduce_grads(
            indices, grads, indices_table_range, h_indices_table_range
        )
        optimizer.step()
        table_num = len(storages)
        for i in range(table_num):
            begin = h_unique_indices_table_range[i]
            end = h_unique_indices_table_range[i + 1]
            unique_indices_per_table = unique_indices[begin:end]
            unique_embs_per_table = unique_embs[begin:end, :]

            if caching:
                KeyValueTableCachingFunction.update(
                    caches[i],
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    optimizer,
                )
            else:
                KeyValueTableFunction.update(
                    storages[i],
                    unique_indices_per_table,
                    unique_embs_per_table,
                    optimizer,
                )

        return (None,) * 14
