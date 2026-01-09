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
import commons.utils.initialize as init
import pytest
import torch
from configs import get_hstu_config
from datasets.utils import Batch, FeatureConfig
from modules.hstu_block import HSTUBlock
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.parametrize(
    "contextual_feature_names", [[], ["user_feature0", "user_feature1"]]
)
@pytest.mark.parametrize("action_feature_name", ["action", None])
@pytest.mark.parametrize("max_num_candidates", [10, 0])
def test_hstu_preprocess(
    contextual_feature_names,
    action_feature_name,
    max_num_candidates,
    dim_size=8,
    batch_size=32,
    max_seqlen=20,
):
    init.initialize_distributed()
    init.set_random_seed(1234)
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        return
    device = torch.cuda.current_device()

    item_feature_name = "item"
    item_and_action_feature_names = (
        [item_feature_name]
        if action_feature_name is None
        else [item_feature_name, action_feature_name]
    )
    feature_configs = [
        FeatureConfig(
            feature_names=item_and_action_feature_names,
            max_item_ids=[1000 for _ in item_and_action_feature_names],
            max_sequence_length=max_seqlen,
            is_jagged=True,
        )
    ]
    for n in contextual_feature_names:
        feature_configs.append(
            FeatureConfig(
                feature_names=[n],
                max_item_ids=[1000],
                max_sequence_length=max_seqlen,
                is_jagged=True,
            )
        )

    batch = Batch.random(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        device=device,
    )

    hstu_config = get_hstu_config(
        hidden_size=dim_size,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=1,
        position_encoding_config=None,
        dtype=torch.float,
    )
    hstu_block = HSTUBlock(hstu_config)
    hstu_block = hstu_block.eval()

    seqlen_sum = torch.sum(batch.features.lengths()).cpu().item()
    embeddings = KeyedJaggedTensor.from_lengths_sync(
        keys=batch.features.keys(),
        values=torch.rand((seqlen_sum, dim_size), device=device),
        lengths=batch.features.lengths(),
    )
    embedding_dict = embeddings.to_dict()
    item_embedding = embedding_dict[item_feature_name].values()
    item_embedding_offests_cpu = embedding_dict[item_feature_name].offsets().cpu()
    if action_feature_name is not None:
        action_embedding = embedding_dict[action_feature_name].values()

    jd = hstu_block._preprocessor(embeddings=embeddings, batch=batch)
    for sample_id in range(batch_size):
        start, end = jd.seqlen_offsets[sample_id], jd.seqlen_offsets[sample_id + 1]
        cur_sequence_embedding = jd.values[start:end, :]
        idx = 0
        for contextual_feature_name in contextual_feature_names:
            contextual_embedding = embedding_dict[contextual_feature_name].values()
            contextual_embedding_offsets_cpu = (
                embedding_dict[contextual_feature_name].offsets().cpu()
            )
            cur_start, cur_end = (
                contextual_embedding_offsets_cpu[sample_id],
                contextual_embedding_offsets_cpu[sample_id + 1],
            )

            for i in range(cur_start, cur_end):
                assert torch.allclose(
                    cur_sequence_embedding[idx, :], contextual_embedding[i, :]
                ), "contextual embedding not match"
                idx += 1
        cur_start, cur_end = (
            item_embedding_offests_cpu[sample_id],
            item_embedding_offests_cpu[sample_id + 1],
        )
        for i in range(cur_start, cur_end):
            assert torch.allclose(
                cur_sequence_embedding[idx, :], item_embedding[i, :]
            ), "item embedding not match"
            idx += 1
            if action_feature_name is not None:
                assert torch.allclose(
                    cur_sequence_embedding[idx, :], action_embedding[i, :]
                ), "action embedding not match"
                idx += 1

    result_jd = hstu_block._postprocessor(jd)
    for sample_id in range(batch_size):
        start, end = (
            result_jd.seqlen_offsets[sample_id],
            result_jd.seqlen_offsets[sample_id + 1],
        )
        result_embedding = result_jd.values[start:end, :]
        cur_start, cur_end = (
            item_embedding_offests_cpu[sample_id],
            item_embedding_offests_cpu[sample_id + 1],
        )
        if max_num_candidates > 0:
            num_candidates_cpu = batch.num_candidates.cpu()
            cur_num_candidates_cpu = num_candidates_cpu[sample_id].item()
            candidate_embedding = item_embedding[
                cur_end - cur_num_candidates_cpu : cur_end, :
            ]
            candidate_embedding = candidate_embedding / torch.linalg.norm(
                candidate_embedding, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-6)
            assert torch.allclose(
                result_embedding, candidate_embedding
            ), "candidate embedding not match"
        else:
            all_item_embedding = item_embedding[cur_start:cur_end, :]
            all_item_embedding = all_item_embedding / torch.linalg.norm(
                all_item_embedding, ord=2, dim=-1, keepdim=True
            ).clamp(min=1e-6)
            assert torch.allclose(
                result_embedding, all_item_embedding
            ), "all item embedding not match"
