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
import os

import torch
from commons.datasets.hstu_batch import HSTUBatch
from modules.inference_dense_module import InferenceDenseModule
from recsys_kvcache_manager import load_kvcache_manager_ops
from recsys_kvcache_manager.kvcache_config import KVCacheConfig
from recsys_kvcache_manager.kvcache_metadata import KVCacheMetadata
from recsys_kvcache_manager.kvcache_utils import KVLookupResult
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from model.inference_ranking_gr import _STRIP_CACHED_TOKENS_OP


# Export kvcache flow calls torch.ops.kvcache_manager_ops.* directly in forward,
# so the runtime op library must be present for this module to be usable.
load_kvcache_manager_ops(strict=True)


class ExportKVCachedInferenceRankingGR(torch.nn.Module):
    def __init__(
        self,
        sparse_module: torch.nn.Module,
        dense_module: InferenceDenseModule,
        kvcache_config: KVCacheConfig,
    ):
        super().__init__()
        self.sparse_module = sparse_module
        self.dense_module = dense_module
        self.kvcache_config = kvcache_config

        self.register_buffer(
            "_offload_slot_mapping_sentinel",
            torch.full((1,), -1, dtype=torch.int64),
            persistent=False,
        )

    def bfloat16(self):
        self.dense_module.bfloat16()
        return self

    def half(self):
        self.dense_module.half()
        return self

    def get_num_class(self):
        return self.dense_module.get_num_class()

    def get_num_tasks(self):
        return self.dense_module.get_num_tasks()

    def get_metric_types(self):
        return self.dense_module.get_metric_types()

    def load_checkpoint(self, checkpoint_dir):
        if checkpoint_dir is None:
            return

        model_state_dict_path = os.path.join(
            checkpoint_dir, "torch_module", "model.0.pth"
        )
        model_state_dict = torch.load(model_state_dict_path)["model_state_dict"]

        self.sparse_module.load_checkpoint(checkpoint_dir, model_state_dict)
        self.dense_module.load_state_dict(model_state_dict, strict=False)

    def _lookup_result_from_ops(
        self,
        user_ids: torch.Tensor,
        lookup_res,
    ) -> KVLookupResult:
        return KVLookupResult(
            user_ids=user_ids,
            cached_start_indices=lookup_res[0],
            cached_lengths=lookup_res[1],
            gpu_cached_start_indices=lookup_res[2],
            gpu_cached_lengths=lookup_res[3],
            host_cached_start_indices=lookup_res[4],
            host_cached_lengths=lookup_res[5],
            extra={"onboard_task_ids": lookup_res[6]},
        )

    def _kvcache_metadata_from_ops(self, alloc_result) -> KVCacheMetadata:
        return KVCacheMetadata(
            page_ids_gpu_buffer=alloc_result[0],
            metadata_gpu_buffer=alloc_result[1],
            kv_indices=alloc_result[0],
            kv_indptr=alloc_result[2],
            kv_last_page_len=alloc_result[3],
            total_history_lengths=alloc_result[4],
            total_history_offsets=alloc_result[5],
            new_history_offsets=alloc_result[6],
            batch_indices=alloc_result[7],
            position=alloc_result[8],
            new_history_nnz=alloc_result[9],
            new_history_nnz_cuda=alloc_result[10],
            kv_seqlens=alloc_result[11],
            kv_seqlen_offsets=alloc_result[12],
            kv_cache_table=None,
            kv_onload_handle=None,
            max_seqlen=self.kvcache_config.max_seq_len,
        )

    def _strip_cached_tokens(self, batch: HSTUBatch, cached_lengths: torch.Tensor):
        num_context = len(batch.contextual_feature_names)
        feature_order = list(range(num_context + 2))
        lengths = batch.features.lengths()
        cached_i64 = cached_lengths.to(device=lengths.device, dtype=torch.int64)
        new_values, new_lengths = _STRIP_CACHED_TOKENS_OP(
            batch.features.values(),
            lengths,
            batch.features.offsets(),
            cached_i64,
            feature_order,
        )
        batch.features = KeyedJaggedTensor(
            values=new_values,
            lengths=new_lengths,
            keys=batch.features.keys(),
        )
        return batch

    def forward(
        self,
        batch: HSTUBatch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            completed_offloads = torch.ops.kvcache_manager_ops.offload_reap_completed(
                user_ids
            )
            lookup_res = torch.ops.kvcache_manager_ops.lookup(
                user_ids,
                total_history_lengths,
                completed_offloads,
            )
            alloc_result = torch.ops.kvcache_manager_ops.allocate(
                user_ids,
                total_history_lengths,
                lookup_res[1],
                lookup_res[5],
            )
            kvcache_metadata = self._kvcache_metadata_from_ops(alloc_result)

            onboard_slot_mappings = torch.ops.kvcache_manager_ops.onboard_launch(
                user_ids,
                total_history_lengths,
                lookup_res,
                alloc_result[0],
                alloc_result[2],
            )
            onboard_task_ids = onboard_slot_mappings[2]

            lookup_result = self._lookup_result_from_ops(user_ids, lookup_res)
            striped_batch = self._strip_cached_tokens(batch, lookup_result.cached_lengths)
            embeddings = self.sparse_module(striped_batch.features)

            jagged_data = self.dense_module._hstu_block._preprocessor(
                embeddings=embeddings,
                batch=striped_batch,
                seq_start_position=lookup_result.cached_lengths.cuda(),
            )
            jagged_data.scaling_seqlen = self.dense_module._scaling_seqlen
            kvcache_metadata.kv_seqlen_offsets = (
                kvcache_metadata.total_history_offsets
                + jagged_data.num_candidates_offsets
            )
            kvcache_metadata.kv_seqlens = (
                kvcache_metadata.total_history_lengths + jagged_data.num_candidates
            )
            kvcache_metadata.kv_cache_table = torch.ops.kvcache_manager_ops.onboard_wait(
                onboard_task_ids,
                jagged_data.values,
            )

            num_tokens = striped_batch.features.values().shape[0]
            jagged_data.values = self.dense_module._hstu_block.predict(
                striped_batch.batch_size,
                num_tokens,
                jagged_data.values,
                jagged_data,
                kvcache_metadata,
                use_cudagraph=False,
            )

            offload_task_ids = torch.ops.kvcache_manager_ops.offload_launch(
                user_ids,
                total_history_lengths,
                lookup_res[1],
                lookup_res[5],
                lookup_res[2],
                lookup_res[3],
                alloc_result[0],
                alloc_result[2],
                [self._offload_slot_mapping_sentinel.cpu()],
                jagged_data.values,
            )

            jagged_data = self.dense_module._hstu_block._postprocessor(jagged_data)
            return self.dense_module._mlp(jagged_data.values), offload_task_ids