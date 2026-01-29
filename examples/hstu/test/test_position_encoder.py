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
import pytest
import torch
from commons.ops.length_to_offsets import length_to_complete_offsets

# from.datasets.hstu_batch import Batch, FeatureConfig
from modules.position_encoder import HSTUPositionalEncoder
from ops.triton_ops.triton_jagged import triton_concat_2D_jagged


def merge_jagged_list(
    jagged_values,
    jagged_seqlens,
):
    assert len(jagged_values) == len(jagged_seqlens)

    if len(jagged_values) == 1:
        return (jagged_values[0], jagged_seqlens[0])

    merge_embeddings = triton_concat_2D_jagged(
        max_seq_len=torch.max(jagged_seqlens[0]).item()
        + torch.max(jagged_seqlens[1]).item(),
        values_a=jagged_values[0],
        values_b=jagged_values[1],
        offsets_a=length_to_complete_offsets(jagged_seqlens[0]),
        offsets_b=length_to_complete_offsets(jagged_seqlens[1]),
    )
    merge_seqlen = jagged_seqlens[0] + jagged_seqlens[1]

    for idx in range(2, len(jagged_values)):
        merge_embeddings = triton_concat_2D_jagged(
            max_seq_len=torch.max(merge_seqlen).item()
            + torch.max(jagged_seqlens[idx]).item(),
            values_a=merge_embeddings,
            values_b=jagged_values[idx],
            offsets_a=length_to_complete_offsets(merge_seqlen),
            offsets_b=length_to_complete_offsets(jagged_seqlens[idx]),
        )
        merge_seqlen += jagged_seqlens[idx]

    return (merge_embeddings, merge_seqlen)


def get_test_input(
    batch_size,
    max_seq_len,
    max_num_candidates,
    embedding_dim,
):
    num_splits = 4
    max_split_len = (max_seq_len - max_num_candidates) // num_splits

    split_embeddings = list()
    split_seqlen = list()
    split_start_position = list()

    for idx in range(num_splits):
        seqlen = (
            torch.randint(low=1, high=max_split_len, size=(batch_size,)).long().cuda()
        )
        if idx == num_splits - 1:
            num_candidates = (
                torch.randint(low=1, high=max_num_candidates, size=(batch_size,))
                .long()
                .cuda()
            )
            seqlen += num_candidates
        split_seqlen.append(seqlen)

    for idx in range(num_splits):
        split_start_position.append(
            torch.zeros((batch_size,)).long().cuda()
            if idx == 0
            else (split_start_position[idx - 1] + split_seqlen[idx - 1])
        )

    for idx in range(num_splits):
        split_embeddings.append(
            torch.randn(
                (torch.sum(split_seqlen[idx]), embedding_dim), dtype=torch.bfloat16
            ).cuda()
        )

    merge_embeddings, merge_seqlen = merge_jagged_list(split_embeddings, split_seqlen)

    # print("[[debug]]", merge_embeddings.shape, merge_seqlen.tolist())
    # print("---")
    # [
    #     print("[[debug]]", embs.shape, seq_len.tolist()) for (embs, seq_len) in zip(split_embeddings, split_seqlen)
    # ]
    # print("---")
    # print("[[debug]]", num_candidates.tolist())
    # print("---")
    # print("[[debug]]", length_to_complete_offsets(merge_seqlen))
    # print()

    return (
        merge_embeddings,
        merge_seqlen,
        num_candidates,
        split_embeddings,
        split_seqlen,
        split_start_position,
    )


@pytest.mark.parametrize("max_seq_len", [4096])
@pytest.mark.parametrize("max_num_candidates", [100, 256, 500])
def test_hstu_position_encoder_with_offsets(
    max_seq_len,
    max_num_candidates,
    batch_size=1,
    embedding_dim=512,
):
    kwargs = {
        "num_position_buckets": 8192,
        "num_time_buckets": 2048,
        "embedding_dim": embedding_dim,
        "training_dtype": torch.bfloat16,
        "is_inference": True,
        "use_time_encoding": False,
    }
    position_encoder = HSTUPositionalEncoder(**kwargs)
    device = torch.cuda.current_device()
    position_encoder = position_encoder.to(device)

    with torch.inference_mode():
        (
            seq_embeddings,
            seqlen,
            num_candidates,
            split_embeddings,
            split_seqlen,
            split_start_position,
        ) = get_test_input(
            batch_size,
            max_seq_len,
            max_num_candidates,
            embedding_dim,
        )

        split_posed_embedding = list()
        for idx in range(len(split_embeddings)):
            split_posed_embedding.append(
                position_encoder(
                    max_seq_len=max_seq_len,
                    seq_lengths=split_seqlen[idx],
                    seq_offsets=length_to_complete_offsets(split_seqlen[idx]),
                    seq_embeddings=split_embeddings[idx],
                    num_targets=num_candidates
                    if idx == len(split_embeddings) - 1
                    else None,
                    seq_timestamps=None,
                    seq_start_position=split_start_position[idx],
                )
            )

        merged_embeddings, merged_seqlen = merge_jagged_list(
            split_posed_embedding,
            split_seqlen,
        )
        assert torch.allclose(merged_seqlen, seqlen)

        ref_embeddings = position_encoder(
            max_seq_len=max_seq_len,
            seq_lengths=seqlen,
            seq_offsets=length_to_complete_offsets(seqlen),
            seq_timestamps=None,
            seq_embeddings=seq_embeddings,
            num_targets=num_candidates,
        )
        assert torch.allclose(merged_embeddings, ref_embeddings)
