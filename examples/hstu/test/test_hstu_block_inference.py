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
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

HSTU_ROOT = Path(__file__).resolve().parents[1]
for path in (HSTU_ROOT, HSTU_ROOT.parent):
    sys.path.insert(0, str(path))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires the CUDA recommendation preprocessing operators",
)
@pytest.mark.parametrize("backbone", ["hstu", "transformer"])
@torch.inference_mode()
def test_hstu_process_inference(backbone):
    # Exercise the current processor/block interfaces directly. The old test
    # used a removed RandomInferenceDataGenerator and obsolete cache APIs even
    # though it was testing only preprocessing and candidate extraction.
    from commons.datasets.hstu_batch import HSTUBatch
    from configs import get_inference_hstu_config
    from modules.hstu_block_inference import HSTUBlockInference
    from modules.transformer_infer_layer import TransformerInferLayer
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    device = torch.device("cuda")
    cfg = get_inference_hstu_config(
        hidden_size=128,
        num_layers=1,
        num_attention_heads=2,
        head_dim=64,
        max_batch_size=2,
        max_seq_len=32,
        dtype=torch.float32,
        contextual_max_seqlen=1,
        backbone=backbone,
    )
    block = HSTUBlockInference(cfg).to(device)
    assert isinstance(block._attention_layers[0], TransformerInferLayer) == (
        backbone == "transformer"
    )
    row_lengths = {"context": [1, 1], "item": [3, 3], "action": [2, 1]}
    embeddings = {}
    for name, lengths in row_lengths.items():
        embeddings[name] = JaggedTensor(
            values=torch.randn(sum(lengths), 128, device=device),
            lengths=torch.tensor(lengths, device=device),
        )
    features = KeyedJaggedTensor(
        keys=list(row_lengths),
        values=torch.arange(11, device=device),
        lengths=torch.tensor([1, 1, 3, 3, 2, 1], device=device),
    )
    batch = HSTUBatch(
        features=features,
        batch_size=2,
        # The shared schema bounds include candidate slots for item and action,
        # although inference action values themselves contain history only.
        feature_to_max_seqlen={"context": 1, "item": 4, "action": 4},
        contextual_feature_names=["context"],
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=2,
        num_candidates=torch.tensor([1, 2], device=device),
    )
    jd = block._preprocessor(embeddings, batch)
    ctx, items, actions = (
        embeddings[k].values() for k in ("context", "item", "action")
    )
    expected = torch.stack(
        [
            ctx[0],
            items[0],
            actions[0],
            items[1],
            actions[1],
            items[2],
            ctx[1],
            items[3],
            actions[2],
            items[4],
            items[5],
        ]
    )
    torch.testing.assert_close(jd.values, expected)
    torch.testing.assert_close(
        jd.seqlen, torch.tensor([6, 5], dtype=torch.int32, device=device)
    )
    post = block._postprocessor(jd)
    expected_candidates = F.normalize(
        items[torch.tensor([2, 4, 5], device=device)], dim=-1, eps=1e-6
    )
    torch.testing.assert_close(post.values, expected_candidates)
    if backbone == "transformer":
        output = block(embeddings, batch)
        assert output.values.shape == (3, 128)
        assert torch.isfinite(output.values).all()
        from configs import InferenceEmbeddingConfig, RankingConfig
        from modules.inference_dense_module import InferenceDenseModule

        task = RankingConfig(
            embedding_configs=[
                InferenceEmbeddingConfig(["item"], "item", 16, 128, False)
            ],
            prediction_head_arch=[128, 2],
            num_tasks=2,
        )
        dense = InferenceDenseModule(cfg, None, task, hstu_block=block).eval()
        logits = dense(batch, embeddings)
        expected_logits = dense._mlp(output.values)
        torch.testing.assert_close(logits, expected_logits)
        state = {k: v.clone() for k, v in dense.state_dict().items()}
        dense.load_state_dict(state, strict=True)
        torch.testing.assert_close(dense(batch, embeddings), logits)
