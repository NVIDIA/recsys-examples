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
from configs.inference_config import EmbeddingBackend, InferenceEmbeddingConfig
from modules.inference_embedding import InferenceEmbedding
from modules.nve_embeddingcollection import InferenceNVEEmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("max_seq_len", [20, 50, 100])
@pytest.mark.parametrize("embedding_dim", [512])
@pytest.mark.parametrize("action_vocab_size", [256])
@pytest.mark.parametrize("item_vocab_size", [10000])
@pytest.mark.parametrize(
    "embedding_backend",
    [
        EmbeddingBackend.NVEMB,
    ],
)
def test_embedding(
    batch_size,
    max_seq_len,
    embedding_dim,
    action_vocab_size,
    item_vocab_size,
    embedding_backend,
):
    if (
        embedding_backend == EmbeddingBackend.NVEMB
        and InferenceNVEEmbeddingCollection is None
    ):
        pytest.skip("NV-Embeddings is not installed.")

    embeddding_configs = [
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["item_feat"],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=True,
        ),
    ]

    embedding_collection = InferenceEmbedding(embeddding_configs, embedding_backend)
    embedding_collection = embedding_collection.to(torch.device("cuda:0"))

    act_features_lengths = torch.randint(
        max_seq_len, (batch_size,), device=torch.device("cuda:0")
    )
    item_features_lengths = torch.randint(
        max_seq_len, (batch_size,), device=torch.device("cuda:0")
    )
    act_features = torch.randint(
        action_vocab_size - 1, (torch.sum(act_features_lengths),)
    )
    item_features = torch.randint(
        item_vocab_size - 1, (torch.sum(item_features_lengths),)
    )

    features = KeyedJaggedTensor.from_lengths_sync(
        keys=["act_feat", "item_feat"],
        values=torch.concat([act_features, item_features]).to(torch.device("cuda:0")),
        lengths=torch.concat([act_features_lengths, item_features_lengths])
        .to(torch.device("cuda:0"))
        .long(),
    ).to(torch.device("cuda:0"))

    embeddings = embedding_collection(features)

    features_dict = features.to_dict()
    for key, item in embeddings.items():
        assert torch.allclose(item.lengths(), features_dict[key].lengths())
