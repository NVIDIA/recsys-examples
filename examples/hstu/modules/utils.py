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
import itertools
from typing import Dict, Optional

import torch
from dataset.utils import RankingBatch
from modules.jagged_data import JaggedData
from ops.cuda_ops.JaggedTensorOpFunction import jagged_2D_tensor_concat
from ops.length_to_offsets import length_to_complete_offsets
from ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from torchrec.sparse.jagged_tensor import JaggedTensor


def init_mlp_weights_optional_bias(
    m: torch.nn.Module,
) -> None:
    """
    Initialize the weights of a linear layer and optionally the bias.

    Args:
        m: The module to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # Always initialize bias to zero.
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def hstu_preprocess_embeddings(
    embeddings: Dict[str, JaggedTensor],
    batch: RankingBatch,
    is_inference: bool,
    dtype: Optional[torch.dtype] = None,
) -> JaggedData:
    """
    Preprocesses the embeddings for use in the HSTU architecture.

    This method performs the following steps:
    1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
                         During inference, action embeddings are only for the history sequence, and
                         they will be interleaved with item embeddings of the history part, while
                         the embeddings of candidates need no interleaving.
    2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample,
                          following the order specified in the batch.
                          During inference, we concatenate three parts:
                          1) contextual embeedings,
                          2) interleaved *item & action* history embeddings, and
                          3) (item) candidates embeddings
                          for each sample, following the order specified in the batch.

    Args:
        embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
        batch (RankingBatch): The batch of ranking data.
        is_inference (bool): Whether is for inference
        dtype (dtype, optional): The output data type of the embeddings.
    Returns:
        JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
    """
    item_jt = embeddings[batch.item_feature_name]  # history + candidate
    dtype = item_jt.values().dtype if dtype is None else dtype
    sequence_embeddings = item_jt.values().to(dtype)
    sequence_embeddings_lengths = item_jt.lengths()
    sequence_embeddings_lengths_offsets = item_jt.offsets()
    sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

    if batch.action_feature_name is not None:
        action_jt = embeddings[batch.action_feature_name]
        jagged_size = sequence_embeddings.size(0)
        embedding_dim = sequence_embeddings.size(1)

        if not is_inference:
            sequence_embeddings = torch.cat(
                [sequence_embeddings, action_jt.values().to(dtype)], dim=1
            ).view(2 * jagged_size, embedding_dim)
            sequence_embeddings_lengths = sequence_embeddings_lengths * 2
            sequence_embeddings_lengths_offsets = (
                sequence_embeddings_lengths_offsets * 2
            )
            sequence_max_seqlen = sequence_max_seqlen * 2
        else:
            action_offsets = action_jt.offsets()
            item_offsets = item_jt.offsets()
            candidates_indptr = item_offsets[: batch.batch_size] + action_jt.lengths()

            item_embs = item_jt.values().to(dtype)
            action_embs = action_jt.values().to(dtype)
            interleaved_embeddings = [
                (
                    torch.cat(
                        [
                            item_embs[item_offsets[idx] : candidates_indptr[idx]],
                            action_embs[action_offsets[idx] : action_offsets[idx + 1]],
                        ],
                        dim=1,
                    ).view(-1, embedding_dim),
                    item_embs[candidates_indptr[idx] : item_offsets[idx + 1]],
                )
                for idx in range(batch.batch_size)
            ]
            interleaved_embeddings = list(itertools.chain(*interleaved_embeddings))
            sequence_embeddings = torch.cat(interleaved_embeddings, dim=0).view(
                -1, embedding_dim
            )
            sequence_embeddings_lengths = item_jt.lengths() + action_jt.lengths()
            sequence_embeddings_lengths_offsets = (
                item_jt.offsets() + action_jt.offsets()
            )
            sequence_max_seqlen += batch.feature_to_max_seqlen[
                batch.action_feature_name
            ]

    if (
        batch.num_candidates is not None
        and batch.action_feature_name is not None
        and not is_inference
    ):
        num_candidates = batch.num_candidates * 2
        max_num_candidates = batch.max_num_candidates * 2
    else:
        num_candidates = batch.num_candidates
        max_num_candidates = batch.max_num_candidates

    contextual_max_seqlen = 0
    contextual_seqlen = None
    contextual_seqlen_offsets = None
    if len(batch.contextual_feature_names) > 0:
        contextual_max_seqlens = [
            batch.feature_to_max_seqlen[name] for name in batch.contextual_feature_names
        ]
        contextual_jts = [embeddings[name] for name in batch.contextual_feature_names]
        all_values = [jt.values().to(dtype) for jt in contextual_jts] + [
            sequence_embeddings
        ]
        all_offsets = [jt.offsets() for jt in contextual_jts] + [
            sequence_embeddings_lengths_offsets
        ]
        all_max_seqlens = contextual_max_seqlens + [sequence_max_seqlen]
        (
            sequence_embeddings,
            sequence_embeddings_lengths_after_concat,
        ) = jagged_2D_tensor_concat(
            all_values,
            all_offsets,
            all_max_seqlens,
        )
        contextual_max_seqlen = max(
            len(batch.contextual_feature_names), sum(contextual_max_seqlens)
        )

        contextual_seqlen = (
            sequence_embeddings_lengths_after_concat - sequence_embeddings_lengths
        )
        sequence_embeddings_lengths = sequence_embeddings_lengths_after_concat

        sequence_embeddings_lengths_offsets = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(sequence_embeddings_lengths)
        )

        contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            contextual_seqlen
        )

        sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

    return JaggedData(
        values=sequence_embeddings,
        seqlen=sequence_embeddings_lengths.to(
            torch.int32
        ),  # contextual + history + candidate
        seqlen_offsets=sequence_embeddings_lengths_offsets.to(torch.int32),
        max_seqlen=sequence_max_seqlen,
        max_num_candidates=max_num_candidates,
        num_candidates=num_candidates.to(torch.int32)
        if num_candidates is not None
        else None,
        num_candidates_offsets=length_to_complete_offsets(num_candidates).to(
            torch.int32
        )
        if num_candidates is not None
        else None,
        contextual_max_seqlen=contextual_max_seqlen,
        contextual_seqlen=contextual_seqlen.to(torch.int32)
        if contextual_seqlen is not None
        else None,
        contextual_seqlen_offsets=contextual_seqlen_offsets.to(torch.int32)
        if contextual_seqlen_offsets is not None
        else None,
        has_interleaved_action=batch.action_feature_name is not None,
    )


def hstu_postprocess_embeddings(jd: JaggedData, is_inference: bool) -> JaggedData:
    sequence_embeddings: torch.Tensor
    seqlen_offsets: torch.Tensor
    max_seqlen: int
    if jd.max_num_candidates > 0:
        seqlen_offsets = jd.num_candidates_offsets
        max_seqlen = jd.max_num_candidates
        _, sequence_embeddings = triton_split_2D_jagged(
            jd.values,
            jd.max_seqlen,
            offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
            offsets_b=seqlen_offsets,
        )
    elif jd.contextual_max_seqlen > 0:
        seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
        max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
        _, sequence_embeddings = triton_split_2D_jagged(
            jd.values,
            jd.max_seqlen,
            offsets_a=jd.contextual_seqlen_offsets,
            offsets_b=seqlen_offsets,
        )
    else:
        sequence_embeddings = jd.values
        seqlen_offsets = jd.seqlen_offsets
        max_seqlen = jd.max_seqlen

    if jd.has_interleaved_action and not is_inference:
        sequence_embeddings = sequence_embeddings[0::2, ...]
        seqlen_offsets = seqlen_offsets // 2
        max_seqlen = max_seqlen // 2

    sequence_embeddings = sequence_embeddings / torch.linalg.norm(
        sequence_embeddings, ord=2, dim=-1, keepdim=True
    ).clamp(min=1e-6)

    return JaggedData(
        values=sequence_embeddings,
        seqlen=(seqlen_offsets[1:] - seqlen_offsets[:-1]).to(jd.seqlen.dtype),
        seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
        max_seqlen=max_seqlen,
        has_interleaved_action=False,
    )
