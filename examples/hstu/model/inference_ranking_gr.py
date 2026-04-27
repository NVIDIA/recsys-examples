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
from typing import Dict, List, Tuple

import torch
from commons.datasets.hstu_batch import HSTUBatch
from configs import InferenceHSTUConfig, KVCacheConfig, RankingConfig
from modules.inference_dense_module import InferenceDenseModule
from modules.inference_embedding import InferenceEmbedding


class InferenceRankingGR(torch.nn.Module):
    """
    A class representing the ranking model inference.

    Args:
        sparse_module (InferenceHSTUConfig): The HSTU configuration.
        dense_module (RankingConfig): The ranking task configuration.
    """

    def __init__(
        self,
        sparse_module: torch.nn.Module,
        dense_module: torch.nn.Module,
    ):
        super().__init__()
        self.sparse_module = sparse_module
        self.dense_module = dense_module

    def bfloat16(self):
        """
        Convert the model to use bfloat16 precision. Only affects the dense module.

        Returns:
            RankingGR: The model with bfloat16 precision.
        """
        self.dense_module.bfloat16()
        return self

    def half(self):
        """
        Convert the model to use half precision. Only affects the dense module.

        Returns:
            RankingGR: The model with half precision.
        """
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

    def _build_lookup_tokens_from_batch(
        self,
        batch: HSTUBatch,
        total_history_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-user history token_ids/token_mask for FlexKV get_match.
        Token order is aligned with HSTU inference preprocessor:
        [contextual..., interleaved(item, action) history...], excluding candidates.
        """
        batch_size = int(batch.batch_size)
        history_lengths_cpu = total_history_lengths.detach().cpu().to(torch.int64)
        max_history_len = int(history_lengths_cpu.max().item()) if batch_size > 0 else 0
        token_ids = torch.zeros((batch_size, max_history_len), dtype=torch.int64)
        token_mask = torch.zeros((batch_size, max_history_len), dtype=torch.bool)

        feature_order: List[str] = list(batch.contextual_feature_names)
        feature_order.append(batch.item_feature_name)
        if batch.action_feature_name is not None:
            feature_order.append(batch.action_feature_name)
        feature_tag = {name: idx + 1 for idx, name in enumerate(feature_order)}
        tag_shift = 48
        tag_mask = (1 << tag_shift) - 1

        def _encode_feature_tokens(values: torch.Tensor, feat_name: str) -> torch.Tensor:
            if values.numel() == 0:
                return values.to(torch.int64)
            tag = int(feature_tag.get(feat_name, 0))
            encoded = values.to(torch.int64) & tag_mask
            if tag > 0:
                encoded = encoded | (tag << tag_shift)
            return encoded

        # Contextual features (if any)
        contextual_cpu: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for feat_name in batch.contextual_feature_names:
            feat_jt = batch.features[feat_name]
            contextual_cpu[feat_name] = (
                feat_jt.values().detach().cpu(),
                feat_jt.offsets().detach().cpu().to(torch.int64),
            )

        item_jt = batch.features[batch.item_feature_name]
        item_values = item_jt.values().detach().cpu()
        item_offsets = item_jt.offsets().detach().cpu().to(torch.int64)

        action_values = None
        action_offsets = None
        if batch.action_feature_name is not None:
            action_jt = batch.features[batch.action_feature_name]
            action_values = action_jt.values().detach().cpu()
            action_offsets = action_jt.offsets().detach().cpu().to(torch.int64)

        if batch.num_candidates is None:
            num_candidates = torch.zeros((batch_size,), dtype=torch.int64)
        else:
            num_candidates = batch.num_candidates.detach().cpu().to(torch.int64)
            if num_candidates.numel() < batch_size:
                num_candidates = torch.nn.functional.pad(
                    num_candidates, (0, batch_size - num_candidates.numel())
                )

        for idx in range(batch_size):
            target_len = int(history_lengths_cpu[idx].item())
            if target_len <= 0:
                continue

            seq_chunks: List[torch.Tensor] = []

            for feat_name in batch.contextual_feature_names:
                feat_values, feat_offsets = contextual_cpu[feat_name]
                left = int(feat_offsets[idx].item())
                right = int(feat_offsets[idx + 1].item())
                seq_chunks.append(
                    _encode_feature_tokens(feat_values[left:right], feat_name)
                )

            item_l = int(item_offsets[idx].item())
            item_r = int(item_offsets[idx + 1].item())
            item_seq = item_values[item_l:item_r]
            cand_cnt = int(num_candidates[idx].item())
            cand_cnt = max(0, min(cand_cnt, item_seq.numel()))
            item_hist = item_seq[: item_seq.numel() - cand_cnt]
            item_hist = _encode_feature_tokens(item_hist, batch.item_feature_name)

            if (
                batch.action_feature_name is not None
                and action_values is not None
                and action_offsets is not None
            ):
                action_l = int(action_offsets[idx].item())
                action_r = int(action_offsets[idx + 1].item())
                action_seq = _encode_feature_tokens(
                    action_values[action_l:action_r], batch.action_feature_name
                )
                interleave_len = min(item_hist.numel(), action_seq.numel())
                if interleave_len > 0:
                    interleaved = torch.empty(
                        (interleave_len * 2,), dtype=torch.int64
                    )
                    interleaved[0::2] = item_hist[:interleave_len]
                    interleaved[1::2] = action_seq[:interleave_len]
                    seq_chunks.append(interleaved)
                if item_hist.numel() > interleave_len:
                    seq_chunks.append(item_hist[interleave_len:])
                if action_seq.numel() > interleave_len:
                    seq_chunks.append(action_seq[interleave_len:])
            else:
                seq_chunks.append(item_hist)

            if len(seq_chunks) == 0:
                continue
            seq_tokens = torch.cat(seq_chunks, dim=0)
            if seq_tokens.numel() == 0:
                continue
            clipped = seq_tokens[:target_len]
            valid_len = clipped.numel()
            token_ids[idx, :valid_len] = clipped
            token_mask[idx, :valid_len] = True

        return token_ids, token_mask

    def forward_with_kvcache(
        self,
        batch: HSTUBatch,
        user_ids: torch.Tensor,
        total_history_lengths: torch.Tensor,
    ):
        with torch.inference_mode():
            lookup_token_ids, lookup_token_mask = self._build_lookup_tokens_from_batch(
                batch=batch,
                total_history_lengths=total_history_lengths,
            )
            lookup_result = self.dense_module.async_kvcache.lookup_kvcache(
                user_ids,
                total_history_lengths,
                token_ids=lookup_token_ids,
                token_mask=lookup_token_mask,
            )
            self.dense_module.async_kvcache.finish_or_cancel_kvcache_ops()
            kv_index_meta, prepare_result = (
                self.dense_module.async_kvcache.allocate_kvcache(
                    lookup_result,
                )
            )

            old_cached_lengths = torch.tensor(
                lookup_result.old_cached_lengths, dtype=torch.int32
            )
            striped_batch = self.dense_module.async_kvcache.strip_cached_tokens(
                batch,
                old_cached_lengths,
            )

            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(striped_batch.features)
            torch.cuda.nvtx.range_pop()

            logits = self.dense_module.forward_with_kvcache(
                striped_batch,
                embeddings,
                user_ids,
                total_history_lengths,
                prepare_result,
                kv_index_meta,
                lookup_result,
            )
            self.dense_module.async_kvcache.lazy_offload_kvcache(
                kv_index_meta
            )

        return logits

    def forward_nokvcache(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module.forward_nokvcache(batch, embeddings)

        return logits

    def forward(
        self,
        batch: HSTUBatch,
    ):
        with torch.inference_mode():
            torch.cuda.nvtx.range_push("HSTU embedding")
            embeddings = self.sparse_module(batch.features)
            torch.cuda.nvtx.range_pop()
            logits = self.dense_module(batch, embeddings)
        return logits


def get_inference_ranking_gr(
    hstu_config: InferenceHSTUConfig,
    kvcache_config: KVCacheConfig,
    task_config: RankingConfig,
    use_cudagraph=False,
    cudagraph_configs=None,
    sparse_shareables=None,
):
    for ebc_config in task_config.embedding_configs:
        assert (
            ebc_config.dim == hstu_config.hidden_size
        ), "hstu layer hidden size should equal to embedding dim"

    inference_sparse = InferenceEmbedding(
        task_config.embedding_configs,
        sparse_shareables,
    )
    inference_dense = InferenceDenseModule(
        hstu_config,
        kvcache_config,
        task_config,
        use_cudagraph,
        cudagraph_configs,
    )

    return InferenceRankingGR(inference_sparse, inference_dense)
