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
from commons.datasets.hstu_batch import FeatureConfig, HSTUBatch
from configs import get_hstu_config
from modules.hstu_block import HSTUBlock
from modules.hstu_processor import hstu_preprocess_embeddings
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class _ExportBatch:
    def __init__(
        self,
        *,
        item_feature_name,
        action_feature_name,
        contextual_feature_names,
        feature_to_max_seqlen,
        max_num_candidates,
        num_candidates,
    ):
        self.item_feature_name = item_feature_name
        self.action_feature_name = action_feature_name
        self.contextual_feature_names = contextual_feature_names
        self.feature_to_max_seqlen = feature_to_max_seqlen
        self.max_num_candidates = max_num_candidates
        self.num_candidates = num_candidates


class _ExportInferencePreprocessWrapper(torch.nn.Module):
    def __init__(self, feature_to_max_seqlen, max_num_candidates):
        super().__init__()
        self._feature_to_max_seqlen = feature_to_max_seqlen
        self._max_num_candidates = max_num_candidates

    def forward(
        self,
        item_values,
        item_lengths,
        action_values,
        action_lengths,
        contextual_values,
        contextual_lengths,
        num_candidates,
    ):
        batch = _ExportBatch(
            item_feature_name="item",
            action_feature_name="action",
            contextual_feature_names=["user"],
            feature_to_max_seqlen=self._feature_to_max_seqlen,
            max_num_candidates=self._max_num_candidates,
            num_candidates=num_candidates,
        )
        jd = hstu_preprocess_embeddings(
            {
                "item": JaggedTensor(values=item_values, lengths=item_lengths),
                "action": JaggedTensor(values=action_values, lengths=action_lengths),
                "user": JaggedTensor(
                    values=contextual_values, lengths=contextual_lengths
                ),
            },
            batch,
            is_inference=True,
        )
        return jd.values, jd.seqlen, jd.seqlen_offsets, jd.num_candidates_offsets


def _make_values(num_tokens, embedding_dim, *, device, start=0):
    return torch.arange(
        start,
        start + num_tokens * embedding_dim,
        dtype=torch.float32,
        device=device,
    ).view(num_tokens, embedding_dim)


def _offsets(lengths):
    return torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)


def _reference_inference_preprocess(
    *,
    item_values,
    item_lengths,
    action_values,
    contextual_values,
    contextual_lengths,
    num_candidates,
):
    item_offsets = _offsets(item_lengths).cpu()
    action_offsets = _offsets(item_lengths).cpu()
    contextual_offsets = _offsets(contextual_lengths).cpu()
    num_candidates_cpu = num_candidates.cpu()

    rows = []
    lengths = []
    for batch_idx in range(item_lengths.numel()):
        contextual_start = contextual_offsets[batch_idx].item()
        contextual_end = contextual_offsets[batch_idx + 1].item()
        item_start = item_offsets[batch_idx].item()
        item_end = item_offsets[batch_idx + 1].item()
        action_start = action_offsets[batch_idx].item()
        num_candidate = num_candidates_cpu[batch_idx].item()
        history_len = item_end - item_start - num_candidate

        sample_rows = [contextual_values[contextual_start:contextual_end]]
        history_item = item_values[item_start : item_start + history_len]
        history_action = action_values[action_start : action_start + history_len]
        sample_rows.append(
            torch.cat([history_item, history_action], dim=1).view(
                -1, item_values.size(1)
            )
        )
        sample_rows.append(item_values[item_start + history_len : item_end])
        rows.extend(sample_rows)
        lengths.append(contextual_end - contextual_start + history_len * 2 + num_candidate)

    expected_lengths = torch.tensor(
        lengths, dtype=torch.int32, device=item_values.device
    )
    return torch.cat(rows, dim=0), expected_lengths, _offsets(expected_lengths)


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

    batch = HSTUBatch.random(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hstu_preprocess_inference_drops_candidate_actions():
    device = torch.device("cuda")
    embedding_dim = 4
    item_lengths = torch.tensor([3, 1, 4], dtype=torch.int64, device=device)
    action_lengths = item_lengths.clone()
    contextual_lengths = torch.tensor([1, 2, 1], dtype=torch.int64, device=device)
    num_candidates = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)

    item_values = _make_values(
        int(item_lengths.sum().item()), embedding_dim, device=device, start=0
    )
    action_values = _make_values(
        int(action_lengths.sum().item()), embedding_dim, device=device, start=1000
    )
    contextual_values = _make_values(
        int(contextual_lengths.sum().item()), embedding_dim, device=device, start=2000
    )
    batch = HSTUBatch(
        features=KeyedJaggedTensor.from_lengths_sync(
            keys=["user", "item", "action"],
            values=torch.arange(
                int(
                    contextual_lengths.sum().item()
                    + item_lengths.sum().item()
                    + action_lengths.sum().item()
                ),
                dtype=torch.int64,
                device=device,
            ),
            lengths=torch.cat([contextual_lengths, item_lengths, action_lengths]),
        ),
        batch_size=item_lengths.numel(),
        feature_to_max_seqlen={"user": 2, "item": 5, "action": 5},
        contextual_feature_names=["user"],
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=2,
        num_candidates=num_candidates,
    )
    jd = hstu_preprocess_embeddings(
        {
            "item": JaggedTensor(values=item_values, lengths=item_lengths),
            "action": JaggedTensor(values=action_values, lengths=action_lengths),
            "user": JaggedTensor(values=contextual_values, lengths=contextual_lengths),
        },
        batch,
        is_inference=True,
    )

    expected_values, expected_lengths, expected_offsets = _reference_inference_preprocess(
        item_values=item_values,
        item_lengths=item_lengths,
        action_values=action_values,
        contextual_values=contextual_values,
        contextual_lengths=contextual_lengths,
        num_candidates=num_candidates,
    )
    torch.testing.assert_close(jd.values, expected_values)
    torch.testing.assert_close(jd.seqlen, expected_lengths)
    torch.testing.assert_close(jd.seqlen_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(
        jd.num_candidates_offsets, _offsets(num_candidates).to(torch.int32)
    )
    assert jd.max_seqlen == 10
    assert jd.max_num_candidates == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hstu_preprocess_inference_export_dynamic_batch():
    device = torch.device("cuda")
    embedding_dim = 4
    wrapper = _ExportInferencePreprocessWrapper(
        feature_to_max_seqlen={"user": 2, "item": 5, "action": 5},
        max_num_candidates=2,
    ).to(device)

    item_lengths = torch.tensor([3, 1, 4], dtype=torch.int64, device=device)
    action_lengths = item_lengths.clone()
    contextual_lengths = torch.tensor([1, 2, 1], dtype=torch.int64, device=device)
    num_candidates = torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
    args = (
        _make_values(int(item_lengths.sum().item()), embedding_dim, device=device),
        item_lengths,
        _make_values(int(action_lengths.sum().item()), embedding_dim, device=device, start=1000),
        action_lengths,
        _make_values(int(contextual_lengths.sum().item()), embedding_dim, device=device, start=2000),
        contextual_lengths,
        num_candidates,
    )

    batch_dim = torch.export.Dim("batch_size", min=1, max=32)
    token_dim = torch.export.Dim("tokens", min=1, max=64)
    contextual_token_dim = torch.export.Dim("contextual_tokens", min=1, max=64)
    exported = torch.export.export(
        wrapper,
        args,
        dynamic_shapes={
            "item_values": {0: token_dim},
            "item_lengths": {0: batch_dim},
            "action_values": {0: token_dim},
            "action_lengths": {0: batch_dim},
            "contextual_values": {0: contextual_token_dim},
            "contextual_lengths": {0: batch_dim},
            "num_candidates": {0: batch_dim},
        },
    )

    new_item_lengths = torch.tensor([2, 5], dtype=torch.int64, device=device)
    new_action_lengths = new_item_lengths.clone()
    new_contextual_lengths = torch.tensor([1, 1], dtype=torch.int64, device=device)
    new_num_candidates = torch.tensor([1, 2], dtype=torch.int64, device=device)
    new_args = (
        _make_values(int(new_item_lengths.sum().item()), embedding_dim, device=device),
        new_item_lengths,
        _make_values(
            int(new_action_lengths.sum().item()), embedding_dim, device=device, start=1000
        ),
        new_action_lengths,
        _make_values(
            int(new_contextual_lengths.sum().item()), embedding_dim, device=device, start=2000
        ),
        new_contextual_lengths,
        new_num_candidates,
    )
    actual_values, actual_lengths, actual_offsets, actual_candidate_offsets = (
        exported.module()(*new_args)
    )
    expected_values, expected_lengths, expected_offsets = _reference_inference_preprocess(
        item_values=new_args[0],
        item_lengths=new_item_lengths,
        action_values=new_args[2],
        contextual_values=new_args[4],
        contextual_lengths=new_contextual_lengths,
        num_candidates=new_num_candidates,
    )
    torch.testing.assert_close(actual_values, expected_values)
    torch.testing.assert_close(actual_lengths, expected_lengths)
    torch.testing.assert_close(actual_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(
        actual_candidate_offsets, _offsets(new_num_candidates).to(torch.int32)
    )
