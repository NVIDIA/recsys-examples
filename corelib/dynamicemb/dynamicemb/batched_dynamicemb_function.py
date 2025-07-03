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

from typing import List

import torch
from dynamicemb.dynamicemb_config import DynamicEmbPoolingMode, dyn_emb_to_torch
from dynamicemb.optimizer import BaseDynamicEmbeddingOptimizer
from dynamicemb.unique_op import UniqueOp
from dynamicemb_extensions import (
    DynamicEmbTable,
    find_or_insert,
    lookup_backward,
    lookup_backward_dense,
    lookup_backward_dense_dedup,
    lookup_forward,
    lookup_forward_dense,
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
        *args
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

            find_or_insert(
                tables[i],
                num_unique_indices,
                unique_indices,
                tmp_unique_embs,
                scores[i],
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
            )

        unique_grads_per_table = []
        for i, unique_grad in enumerate(unique_backward_grads_per_table):
            unique_grads_per_table.append(unique_grad.reshape(-1, dims[i]))

        scores = ctx.scores

        optimizer.update(
            tables, unique_indices_list, unique_grads_per_table, [], scores
        )

        return (None,) * 17


class DynamicEmbeddingFunction(torch.autograd.Function):
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
        dim: int,
        feature_table_map: List[int],
        embedding_dtype: torch.dtype,
        output_dtype: torch.dtype,
        pooling_mode: DynamicEmbPoolingMode,
        device_num_sms: int,
        unique_op: UniqueOp,
        device: torch.device,
        optimizer: BaseDynamicEmbeddingOptimizer,
        host_tables: List[DynamicEmbTable],
        cache_metrics: torch.Tensor,
        *args
    ):
        # TODO:need check dimension is right
        table_num = len(tables)
        assert table_num == len(table_offsets_in_feature) - 1
        feature_num = table_offsets_in_feature[-1]
        feature_batch_size = offsets.shape[0] - 1
        batch_size = feature_batch_size // feature_num
        assert feature_batch_size % feature_num == 0

        d_unique_offsets = torch.zeros(table_num + 1, dtype=torch.uint64, device=device)
        h_unique_offsets = torch.empty(table_num + 1, dtype=torch.uint64, device="cpu")
        table_offsets = torch.empty(table_num + 1, dtype=offsets.dtype, device=device)
        # table' dtype
        unique_embs = torch.empty(
            indices.shape[0], dim, dtype=embedding_dtype, device=device
        )
        # output' dtype
        output_embs = torch.empty(
            indices.shape[0], dim, dtype=output_dtype, device=device
        )

        # #TODO: if global dedup is done:
        # if use_index_dedup:
        #     lookup_forward_dense(
        #         tables,
        #         indices,
        #         offsets,
        #         table_offsets_in_feature,
        #         table_num,
        #         batch_size,
        #         dim,
        #         h_unique_offsets, # used in backward, actually plays a role as table_offsets.
        #         unique_embs, # serve as a tmp buffer.
        #         output_embs)
        # else:
        # TODO:in our case , maybe uint32 is enough for reverse_idx
        reverse_idx = torch.empty_like(indices, dtype=torch.uint64, device=device)
        unique_idx = torch.empty_like(indices, dtype=indices.dtype, device=device)
        h_unique_nums = torch.empty(table_num, dtype=torch.uint64, device="cpu")
        d_unique_nums = torch.empty(table_num, dtype=torch.uint64, device=device)
        if host_tables is not None:
            lookup_forward_dense(
                tables,
                indices,
                offsets,
                scores,
                table_offsets_in_feature,
                table_offsets,
                table_num,
                batch_size,
                dim,
                use_index_dedup,
                unique_idx,
                reverse_idx,
                h_unique_nums,
                d_unique_nums,
                h_unique_offsets,
                d_unique_offsets,
                unique_embs,
                output_embs,
                device_num_sms,
                unique_op,
                cache_metrics,
                host_tables,
            )
        else:
            lookup_forward_dense(
                tables,
                indices,
                offsets,
                scores,
                table_offsets_in_feature,
                table_offsets,
                table_num,
                batch_size,
                dim,
                use_index_dedup,
                unique_idx,
                reverse_idx,
                h_unique_nums,
                d_unique_nums,
                h_unique_offsets,
                d_unique_offsets,
                unique_embs,
                output_embs,
                device_num_sms,
                unique_op,
            )
        if use_index_dedup:
            unique_idx_forback = torch.empty(
                h_unique_offsets[-1], dtype=indices.dtype, device=device
            )
            unique_idx_forback.copy_(
                unique_idx[: h_unique_offsets[-1]], non_blocking=True
            )
            unique_emb_forback = unique_embs[: h_unique_offsets[-1], :]

        backward_tensors = [indices, offsets]
        ctx.save_for_backward(*backward_tensors)
        ctx.tables = tables
        ctx.dim = dim
        ctx.device = device
        ctx.optimizer = optimizer

        # optimize need
        ctx.h_unique_offsets = h_unique_offsets
        ctx.table_offsets = table_offsets
        ctx.use_index_dedup = use_index_dedup
        ctx.device_num_sms = device_num_sms
        if use_index_dedup:
            ctx.reverse_idx = reverse_idx
            ctx.unique_idx_forback = unique_idx_forback
            ctx.unique_emb_forback = unique_emb_forback
        ctx.scores = scores

        return output_embs

    @staticmethod
    def backward(ctx, grads):
        # parse context
        indices, _ = ctx.saved_tensors
        h_unique_offsets = ctx.h_unique_offsets
        table_offsets = ctx.table_offsets

        dim = ctx.dim
        device = ctx.device
        tables = ctx.tables
        optimizer = ctx.optimizer
        use_index_dedup = ctx.use_index_dedup
        device_num_sms = ctx.device_num_sms
        if use_index_dedup:
            reverse_idx = ctx.reverse_idx
            unique_idx_forback = ctx.unique_idx_forback
            unique_emb_forback = ctx.unique_emb_forback

        table_num = len(tables)
        unique_indices_list = []
        unique_grads_list = []
        unique_embs_list = []

        if use_index_dedup:
            unique_grads = torch.zeros(
                h_unique_offsets[-1], dim, dtype=grads.dtype, device=device
            )
            lookup_backward_dense_dedup(
                grads,
                unique_idx_forback,
                reverse_idx,
                dim,
                unique_grads,
                device_num_sms,
            )

            for i in range(table_num):
                unique_indices_list.append(
                    unique_idx_forback[h_unique_offsets[i] : h_unique_offsets[i + 1]]
                )
                unique_grads_list.append(
                    unique_grads[h_unique_offsets[i] : h_unique_offsets[i + 1], :]
                )
                unique_embs_list.append(
                    unique_emb_forback[h_unique_offsets[i] : h_unique_offsets[i + 1], :]
                )
        else:
            # backward: reduce the grad.
            unique_indices = torch.empty(
                h_unique_offsets[-1], dtype=indices.dtype, device=device
            )
            unique_grads = torch.empty(
                h_unique_offsets[-1], dim, dtype=grads.dtype, device=device
            )
            lookup_backward_dense(
                indices,
                grads,
                dim,
                table_offsets,
                unique_indices,
                unique_grads,
            )
            for i in range(table_num):
                unique_indices_list.append(
                    unique_indices[h_unique_offsets[i] : h_unique_offsets[i + 1]]
                )
                unique_grads_list.append(
                    unique_grads[h_unique_offsets[i] : h_unique_offsets[i + 1], :]
                )

        scores = ctx.scores
        # optimizer: update tables.
        optimizer.update(
            tables, unique_indices_list, unique_grads_list, unique_embs_list, scores
        )
        return (None,) * 19
