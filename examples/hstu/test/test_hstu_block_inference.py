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
from functools import partial
from pathlib import Path
from types import SimpleNamespace

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
        dtype=torch.bfloat16,
        contextual_max_seqlen=1,
        backbone=backbone,
    )
    block = HSTUBlockInference(cfg).to(device)
    assert isinstance(block._attention_layers[0], TransformerInferLayer) == (
        backbone == "transformer"
    )
    row_lengths = {"context": [1, 1], "item": [3, 3], "action": [3, 3]}
    embeddings = {}
    for name, lengths in row_lengths.items():
        embeddings[name] = JaggedTensor(
            values=torch.randn(sum(lengths), 128, device=device, dtype=torch.bfloat16),
            lengths=torch.tensor(lengths, device=device),
        )
    features = KeyedJaggedTensor(
        keys=list(row_lengths),
        values=torch.arange(14, device=device),
        lengths=torch.tensor([1, 1, 3, 3, 3, 3], device=device),
    )
    batch = HSTUBatch(
        features=features,
        batch_size=2,
        # Inputs include candidate slots for item and action. The preprocessing
        # operator removes candidate actions and interleaves history only.
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
            actions[3],
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
    # The block's own forward, which runs the layers through predict, and the dense
    # module on top of it, for both backbones.
    output = block(embeddings, batch)
    assert output.values.shape == (3, 128)
    assert torch.isfinite(output.values).all()
    from configs import InferenceEmbeddingConfig, RankingConfig
    from modules.inference_dense_module import InferenceDenseModule

    task = RankingConfig(
        embedding_configs=[InferenceEmbeddingConfig(["item"], "item", 16, 128, False)],
        prediction_head_arch=[128, 2],
        num_tasks=2,
    )
    dense = InferenceDenseModule(cfg, None, task, hstu_block=block).bfloat16().eval()
    logits = dense(batch, embeddings)
    torch.testing.assert_close(logits, dense._mlp(output.values))
    # Load the checkpoint of another dense module. Transformer checkpoints keep the
    # module's parameter names; HSTU ones use the training layout, with transposed
    # UVQK and projection weights. The sparse tables of a model checkpoint are skipped.
    other = InferenceDenseModule(
        cfg, None, task, hstu_block=HSTUBlockInference(cfg).to(device)
    )
    other = other.bfloat16().eval()
    expected = other(batch, embeddings)
    assert not torch.allclose(expected, logits)
    state = {k: v.clone() for k, v in other.state_dict().items()}
    if backbone == "hstu":
        for key in list(state):
            for name, transpose in (
                ("_linear_uvqk.weight", True),
                ("_linear_uvqk.bias", False),
                ("_linear_proj.weight", True),
            ):
                if key.endswith(name):
                    value = state.pop(key)
                    training_key = key.removesuffix(name) + name.replace(".", "_")
                    state[training_key] = value.T if transpose else value
    sparse = "_embedding_collection._data_parallel_embedding_collection.embeddings."
    state[sparse + "item.weight"] = torch.zeros(1)
    dense.load_state_dict(state, strict=False)
    torch.testing.assert_close(dense(batch, embeddings), expected)
    if backbone == "hstu":
        # Large requests (1024+ tokens with CUDA graphs, 2048+ eagerly) multiply by
        # these transposed copies instead of the nn.Linear weights.
        for layer in dense._hstu_block._attention_layers:
            torch.testing.assert_close(
                layer._linear_uvqk_weight, layer._linear_uvqk.weight.T
            )
            torch.testing.assert_close(
                layer._linear_proj_weight, layer._linear_proj.weight.T
            )
    # Missing or unexpected dense keys fail even with strict=False.
    with pytest.raises(RuntimeError, match=f"Checkpoint does not match {backbone}"):
        dense.load_state_dict({"_hstu_block.unknown": torch.zeros(1)}, strict=False)


@pytest.mark.parametrize(
    "precision,error",
    [
        ({"up_projection_fp8": True}, "require rowwise scaling"),
        ({"attention_quant_mode": 5}, "without CUDA graphs"),
    ],
)
def test_hstu_rejects_unsupported_fp8_cuda_graphs(precision, error):
    from modules.hstu_block_inference import HSTUBlockInference

    options = dict(
        up_projection_fp8=False,
        down_projection_fp8=False,
        linear_fp8_scaling="per-tensor",
        attention_quant_mode=-1,
    )
    options.update(precision)
    block = SimpleNamespace(config=SimpleNamespace(**options), _hstu_graph=None)
    block.capture_graph = partial(HSTUBlockInference.capture_graph, block)

    with pytest.raises(ValueError, match=error):
        HSTUBlockInference.set_cudagraph(block, 1, 32, None, None, None)
    assert block._hstu_graph is None


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FP8 HSTU inference requires Hopper",
)
@pytest.mark.parametrize(
    "fp8_projections, attention_quant_mode",
    [(True, -1), (False, 4)],
    ids=["rowwise-projections", "fp8-attention"],
)
@torch.inference_mode()
def test_hstu_fp8_inference(monkeypatch, fp8_projections, attention_quant_mode):
    from configs import get_inference_hstu_config
    from modules.hstu_block_inference import HSTUBlockInference
    from modules.jagged_data import JaggedData

    device = torch.cuda.current_device()
    dtype = torch.bfloat16
    hidden = 128
    config_args = dict(
        hidden_size=hidden,
        num_layers=1,
        num_attention_heads=2,
        head_dim=64,
        max_batch_size=2,
        max_seq_len=64,
        dtype=dtype,
        residual=False,
        export_mode=False,
    )
    with torch.random.fork_rng(devices=[device]):
        torch.manual_seed(1234)
        baseline = HSTUBlockInference(get_inference_hstu_config(**config_args))
        torch.manual_seed(1234)
        block = HSTUBlockInference(
            get_inference_hstu_config(
                **config_args,
                up_projection_fp8=fp8_projections,
                down_projection_fp8=fp8_projections,
                linear_fp8_scaling="rowwise",
                attention_quant_mode=attention_quant_mode,
            )
        )
        jd = JaggedData.random(
            torch.tensor([23, 42], dtype=torch.int32, device=device),
            hidden,
            num_candidates=torch.tensor([1, 2], dtype=torch.int32, device=device),
            device=device,
            dtype=dtype,
        )

    expected = baseline.predict(2, jd.values.size(0), jd.values, jd, None)
    projection_widths = []
    scaled_mm = F.scaled_mm

    def capture_projection(a, b, *args, **kwargs):
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        projection_widths.append(b.size(1))
        return scaled_mm(a, b, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_mm", capture_projection)
    attention_modes = []
    attention = torch.ops.fbgemm.hstu_varlen_fwd_90

    def capture_attention(*args, **kwargs):
        mode = args[18] if len(args) > 18 else kwargs["quant_mode"]
        if mode >= 0:
            assert args[0].dtype == torch.float8_e4m3fn
            attention_modes.append(mode)
        return attention(*args, **kwargs)

    monkeypatch.setattr(torch.ops.fbgemm, "hstu_varlen_fwd_90", capture_attention)
    actual = block.predict(2, jd.values.size(0), jd.values, jd, None)
    assert projection_widths == ([4 * hidden, hidden] if fp8_projections else [])
    assert attention_modes == (
        [attention_quant_mode] if attention_quant_mode >= 0 else []
    )
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == dtype
    assert torch.isfinite(actual).all()
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error < 0.18
