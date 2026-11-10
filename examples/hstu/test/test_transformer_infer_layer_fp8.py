# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FP8 DeepGEMM Transformer inference layers.

The reference is the bf16 ``TransformerInferLayer`` holding identical weights. Needs
DeepGEMM (the Docker image builds it from third_party/DeepGEMM) and an SM90+ GPU;
otherwise the whole module is skipped, and a skip is not validation of the FP8 path.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

HSTU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HSTU_ROOT))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 DeepGEMM needs an SM90+ GPU",
)
pytest.importorskip("deep_gemm", reason="DeepGEMM is not installed")

from modules import transformer_infer_layer_fp8
from modules.transformer_infer_layer import TransformerInferLayer
from modules.transformer_infer_layer_fp8 import (
    Fp8FusedTransformerInferLayer,
    Fp8Linear,
    Fp8TransformerInferLayer,
)
from recsys_kvcache_manager.kvcache_metadata import copy_kvcache_metadata
from transformer_test_utils import (
    CACHE_LAYER,
    cache_inputs,
    cache_metadata,
    config,
    cosine,
    count_graph_replays,
    dense_batch,
    dense_config,
    dense_module,
    layer_inputs,
    layer_pair,
    randomize_linears,
)

LAYER_CLASSES = [Fp8TransformerInferLayer, Fp8FusedTransformerInferLayer]
layer_classes = pytest.mark.parametrize("layer_cls", LAYER_CLASSES)


@pytest.mark.parametrize("m", [1, 130, 1024])
@pytest.mark.parametrize("k,n", [(512, 1536), (512, 512), (512, 2048), (2048, 512)])
def test_fp8_linear_matches_bf16(k, n, m):
    torch.manual_seed(0)
    linear = nn.Linear(k, n, device="cuda", dtype=torch.bfloat16)
    fp8_linear = Fp8Linear.from_linear(linear)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    actual = fp8_linear(x)
    assert actual.shape == (m, n) and actual.dtype == torch.bfloat16
    assert cosine(actual, linear(x)) > 0.998


def test_fp8_linear_keeps_leading_dims_and_empty_input():
    linear = nn.Linear(512, 256, device="cuda", dtype=torch.bfloat16)
    fp8_linear = Fp8Linear.from_linear(linear)
    x = torch.randn(2, 5, 512, device="cuda", dtype=torch.bfloat16)
    assert fp8_linear(x).shape == (2, 5, 256)
    assert fp8_linear(x[:0, :0]).shape == (0, 0, 256)


@layer_classes
def test_layer_matches_bf16_layer(layer_cls):
    base, fp8 = layer_pair(layer_cls)
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        expected = base(x, offsets, candidates)
        actual = fp8(x, offsets, candidates)
    relative_error = (actual.float() - expected.float()).abs().mean() / (
        expected.float().abs().mean()
    )
    assert cosine(actual, expected) > 0.999
    # e4m3 rounding alone gives about 3.7% mean relative error per GEMM.
    assert relative_error < 0.06


@layer_classes
def test_split_entry_points_match_forward(layer_cls):
    _, layer = layer_pair(layer_cls)
    x, offsets, candidates = layer_inputs()
    num_users, num_tokens = len(candidates), x.shape[0]
    jd = SimpleNamespace(seqlen_offsets=offsets, num_candidates=candidates)
    with torch.inference_mode():
        expected = layer(x, offsets, candidates)
        naive = layer.forward_naive(num_users, num_tokens, x, jd, None)
        layer.forward_input(num_users, num_tokens, x, jd, None)
        split = layer.forward_output(num_users, num_tokens, x, jd, None)
    assert cosine(naive, expected) > 0.99999
    assert cosine(split, expected) > 0.99999


@layer_classes
def test_checkpoint_schema_is_unchanged(layer_cls):
    base, fp8 = layer_pair(layer_cls)
    assert list(fp8.state_dict()) == list(base.state_dict())
    projections = (fp8.qkv, fp8.proj, fp8.ffn[0], fp8.ffn[2])
    assert all(isinstance(p, Fp8Linear) for p in projections)


@layer_classes
def test_updated_weights_are_picked_up(layer_cls):
    base, fp8 = layer_pair(layer_cls)
    x, offsets, candidates = layer_inputs()
    with torch.no_grad():
        base.qkv.weight.mul_(1.5)
        base.ffn[0].weight.mul_(0.5)
    fresh = layer_cls(config(), 0, "cuda")
    fresh.load_state_dict(base.state_dict())
    with torch.inference_mode():
        stale = fp8(x, offsets, candidates)  # builds FP8 weights from the old values
        fp8.load_state_dict(base.state_dict())  # in-place copy bumps the weight version
        updated = fp8(x, offsets, candidates)
        expected = fresh(x, offsets, candidates)
    assert not torch.equal(stale, updated)
    assert cosine(updated, expected) > 0.99999


@layer_classes
def test_unsupported_widths_keep_bf16_linears(layer_cls):
    kwargs = dict(hidden=192, heads=2, head_dim=96, ffn=384)
    base = TransformerInferLayer(config(**kwargs), 0, "cuda")
    randomize_linears(base)
    with pytest.warns(UserWarning, match="not FP8 eligible"):
        fp8 = layer_cls(config(**kwargs), 0, "cuda")
    fp8.load_state_dict(base.state_dict())
    assert not any(isinstance(m, Fp8Linear) for m in fp8.modules())
    x, offsets, candidates = layer_inputs(hidden=192)
    with torch.inference_mode():
        torch.testing.assert_close(
            fp8(x, offsets, candidates), base(x, offsets, candidates), rtol=0, atol=0
        )


def test_partially_eligible_fused_layer_falls_back():
    with pytest.warns(UserWarning, match="not FP8 eligible"):
        base, layer = layer_pair(Fp8FusedTransformerInferLayer, ffn=200)
    assert isinstance(layer.qkv, Fp8Linear) and not isinstance(layer.ffn[0], Fp8Linear)
    assert not layer._fused
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        actual = layer(x, offsets, candidates)
        expected = base(x, offsets, candidates)
    assert cosine(actual, expected) > 0.999


@layer_classes
def test_rejects_unsupported_configs(layer_cls):
    with pytest.raises(ValueError, match="bf16"):
        layer_cls(config(bf16=False), 0, "cuda")
    with pytest.raises(ValueError, match="exported"):
        layer_cls(config(export_mode=True), 0, "cuda")


def count_quant_kernels(monkeypatch, layer, inputs):
    counts = {}
    for name in (
        "triton_quant_1x128",
        "triton_layernorm_quant_1x128",
        "triton_bias_gelu_quant_1x128",
    ):
        original = getattr(transformer_infer_layer_fp8, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(transformer_infer_layer_fp8, name, counted)
    with torch.inference_mode():
        layer(*inputs)
    return counts


def test_quantization_kernels_per_layer(monkeypatch):
    inputs = layer_inputs()
    _, unfused = layer_pair(Fp8TransformerInferLayer)
    _, fused = layer_pair(Fp8FusedTransformerInferLayer)
    # Four quantization points per layer. Unfused, each is a standalone kernel. Fused,
    # both LayerNorms and the bias + GELU quantize their own output, and only the
    # attention output in front of proj is quantized on its own.
    assert count_quant_kernels(monkeypatch, unfused, inputs) == {
        "triton_quant_1x128": 4
    }
    assert count_quant_kernels(monkeypatch, fused, inputs) == {
        "triton_quant_1x128": 1,
        "triton_layernorm_quant_1x128": 2,
        "triton_bias_gelu_quant_1x128": 1,
    }


@layer_classes
def test_cuda_graph_replay_matches_eager(layer_cls):
    _, fp8 = layer_pair(layer_cls)
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        # Warm up on a side stream so DeepGEMM and Triton compile outside the capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fp8(x, offsets, candidates)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fp8(x, offsets, candidates)
        x.copy_(layer_inputs(seed=2)[0])
        graph.replay()
        torch.cuda.synchronize()
        assert cosine(captured, fp8(x, offsets, candidates)) > 0.9999


@layer_classes
def test_paged_cache_forward_matches_bf16_layer(layer_cls):
    pytest.importorskip("paged_kvcache_ops")
    base, layer = layer_pair(layer_cls, **CACHE_LAYER)
    x, jd = cache_inputs()
    # Total history per user; the part not in this call is already cached.
    for seed, histories in enumerate(([2, 2], [3, 2], [4, 3])):
        expected_cache = cache_metadata(histories, seed)
        actual_cache = cache_metadata(histories, seed)
        with torch.inference_mode():
            expected = base.forward_naive(2, 7, x, jd, expected_cache)
            actual = layer.forward_naive(2, 7, x, jd, actual_cache)
        # Only 7 rows: the cosine varies by about 1e-3 from draw to draw around 0.9991.
        assert cosine(actual, expected) > 0.995
        if histories == [2, 2]:
            # Nothing was cached before this call, so the cached path must reproduce the
            # layer's plain forward; this is exact and independent of the FP8 noise.
            with torch.inference_mode():
                plain = layer(x, jd.seqlen_offsets, jd.num_candidates)
            assert cosine(actual, plain) > 0.99999
        # The K and V the FP8 layer appended match the bf16 layer's.
        assert (
            cosine(actual_cache.kv_cache_table[0], expected_cache.kv_cache_table[0])
            > 0.999
        )


@layer_classes
def test_cuda_graph_with_paged_cache_follows_new_requests(layer_cls):
    pytest.importorskip("paged_kvcache_ops")
    base, layer = layer_pair(layer_cls, **CACHE_LAYER)
    x, jd = cache_inputs()
    static = cache_metadata([2, 2], seed=0)
    with torch.inference_mode():
        # Warm up on a side stream so DeepGEMM and Triton compile outside the capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                layer.forward_input(2, 7, x, jd, static)
                layer.forward_output(2, 7, x, jd, static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.forward_input(2, 7, x, jd, static)
            output = layer.forward_output(2, 7, x, jd, static)
        # Like a new request: same graph shapes, new cached prefix lengths, positions and
        # physical pages.
        for seed, histories in enumerate(([3, 2], [4, 3], [2, 2]), start=1):
            source = cache_metadata(histories, seed)
            static.kv_cache_table[0].copy_(source.kv_cache_table[0])
            expected_bf16 = base.forward_naive(
                2, 7, x, jd, cache_metadata(histories, seed)
            )
            expected = layer.forward_naive(2, 7, x, jd, source)
            copy_kvcache_metadata(static, source)
            graph.replay()
            assert cosine(output, expected) > 0.99999
            assert cosine(output, expected_bf16) > 0.995


@layer_classes
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("learnable_input_layernorm", [True, False])
def test_config_variants_match_bf16_layer(
    layer_cls, residual, learnable_input_layernorm
):
    base, layer = layer_pair(
        layer_cls,
        residual=residual,
        learnable_input_layernorm=learnable_input_layernorm,
    )
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        actual = layer(x, offsets, candidates)
        expected = base(x, offsets, candidates)
    # Without the residual nothing dilutes the FP8 noise of the four projections.
    assert cosine(actual, expected) > 0.995


@layer_classes
def test_padded_tail_rows_do_not_affect_valid_rows(layer_cls):
    # A CUDA graph bucket packs more rows than the request has; its tail holds stale data.
    _, layer = layer_pair(layer_cls)
    x, offsets, candidates = layer_inputs()
    jd = SimpleNamespace(seqlen_offsets=offsets, num_candidates=candidates)
    tail = torch.full((9, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    padded = torch.cat((x, tail))
    with torch.inference_mode():
        expected = layer(x, offsets, candidates)
        actual = layer.forward_naive(len(candidates), padded.shape[0], padded, jd, None)
    assert torch.isfinite(actual[: x.shape[0]]).all()
    assert cosine(actual[: x.shape[0]], expected) > 0.99999


def test_config_selects_fp8_layers():
    from modules.hstu_block_inference import HSTUBlockInference
    from utils.gin_config_args import NetworkArgs

    fp8 = HSTUBlockInference(
        dense_config(transformer_fp8_linear=True)
    )._attention_layers
    bf16 = HSTUBlockInference(dense_config())
    assert all(type(layer) is Fp8FusedTransformerInferLayer for layer in fp8)
    assert all(type(layer) is TransformerInferLayer for layer in bf16._attention_layers)
    with pytest.raises(ValueError, match="transformer backbone"):
        dense_config(backbone="hstu", transformer_fp8_linear=True)
    with pytest.raises(ValueError, match="transformer backbone"):
        NetworkArgs(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            kv_channels=64,
            transformer_fp8_linear=True,
        )


def test_checkpoint_loaded_after_graph_capture_reaches_the_graphs(monkeypatch):
    # The Triton model's order: the dense module captures its per-layer graphs while
    # it is built, then it is cast to bf16 and the checkpoint is loaded.
    source = dense_module(0, dense_config())
    fp8 = dense_config(transformer_fp8_linear=True)
    graphs = {"batch_size": [2], "length_per_sequence": [64]}
    served = dense_module(1, fp8, use_cudagraph=True, cudagraph_configs=graphs)
    eager = dense_module(2, fp8)
    replays = count_graph_replays(monkeypatch, served)
    batch, embeddings = dense_batch()
    before = served(batch, embeddings)
    for model in (served, eager):
        model.load_state_dict(source.state_dict(), strict=False)
    after = served(batch, embeddings)
    expected = eager(batch, embeddings)
    assert len(replays) == 2
    # The loaded weights change the output, so stale FP8 weights would show up.
    assert cosine(before, expected) < 0.9
    assert cosine(after, expected) > 0.99999
    assert cosine(after, source(batch, embeddings)) > 0.999
