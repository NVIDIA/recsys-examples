# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU numerical tests for the production Transformer inference implementation.

No CUDA modules, monkeypatched operators or extracted AST are used. The manual
reference computes each user's unpadded attention independently with softmax.
GPU-only integration is covered separately at the end of this file.
"""

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

HSTU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HSTU_ROOT))

from modules.inference_checkpoint import load_dense_state_dict
from modules.transformer_infer_layer import TransformerInferLayer


def config(**overrides):
    args = dict(
        num_heads=2,
        head_dim=4,
        hidden_size=12,
        max_seq_len=24,
        max_batch_size=4,
        export_mode=False,
        residual=True,
        bf16=False,
        fp16=False,
        layernorm_epsilon=1e-5,
        transformer_ffn_dim=20,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_backbone_configuration_defaults_and_validation():
    from configs import get_inference_hstu_config

    args = dict(
        hidden_size=12,
        num_layers=2,
        num_attention_heads=2,
        head_dim=4,
        max_batch_size=4,
        max_seq_len=24,
    )
    assert get_inference_hstu_config(**args).backbone == "hstu"
    cfg = get_inference_hstu_config(
        **args, backbone="transformer", transformer_ffn_dim=20, dtype=torch.float32
    )
    assert cfg.hstu_preprocessing_config is None
    layer = TransformerInferLayer(cfg, 0, "cpu")
    assert layer.ffn[0].out_features == 20
    with pytest.raises(ValueError, match="Unknown inference backbone"):
        get_inference_hstu_config(**args, backbone="typo")
    with pytest.raises(ValueError, match="positive"):
        get_inference_hstu_config(**args, backbone="transformer", transformer_ffn_dim=0)


def offsets(lengths):
    return torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32
    )


def reference(layer, x, candidates):
    """Independent unpadded Pre-LN Transformer arithmetic for one user."""
    normed = F.layer_norm(
        x,
        (x.shape[-1],),
        layer.input_norm.weight,
        layer.input_norm.bias,
        layer.input_norm.eps,
    )
    q, k, v = F.linear(normed, layer.qkv.weight, layer.qkv.bias).chunk(3, -1)
    q, k, v = [
        z.reshape(-1, layer.num_heads, layer.head_dim).transpose(0, 1)
        for z in (q, k, v)
    ]
    n = x.shape[0]
    allowed = torch.tensor(
        [
            [j <= i and (j < n - candidates or j == i) for j in range(n)]
            for i in range(n)
        ],
        device=x.device,
    )
    logits = (q @ k.transpose(-1, -2)) / layer.head_dim**0.5
    weights = logits.masked_fill(~allowed, -float("inf")).softmax(-1)
    out = (weights @ v).transpose(0, 1).reshape(n, -1)
    out = F.linear(out, layer.proj.weight, layer.proj.bias)
    x = x + out if layer._residual else out
    normed = F.layer_norm(
        x,
        (x.shape[-1],),
        layer.ffn_norm.weight,
        layer.ffn_norm.bias,
        layer.ffn_norm.eps,
    )
    ffn = F.linear(
        F.gelu(F.linear(normed, layer.ffn[0].weight, layer.ffn[0].bias)),
        layer.ffn[2].weight,
        layer.ffn[2].bias,
    )
    return x + ffn if layer._residual else ffn


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("seed", range(5))
def test_ragged_forward_matches_unpadded_reference(dtype, residual, seed):
    torch.manual_seed(seed)
    layer = TransformerInferLayer(config(residual=residual), 0, "cpu").to(dtype)
    lengths, candidates = [1, 5, 11, 20], [1, 2, 0, 4]
    x = torch.randn(sum(lengths), 12, dtype=dtype)
    expected = torch.cat(
        [reference(layer, user, c) for user, c in zip(x.split(lengths), candidates)]
    )
    actual = layer(x, offsets(lengths), torch.tensor(candidates))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)


def make_metadata(histories, layers, heads, dim, dtype, page_size=4):
    counts = [(h + page_size - 1) // page_size for h in histories]
    pages = sum(counts)
    return SimpleNamespace(
        kv_cache_table=[
            torch.full(
                (max(1, pages), 2, page_size, heads, dim), float("nan"), dtype=dtype
            )
            for _ in range(layers)
        ],
        kv_indices=torch.randperm(pages).to(torch.int32),
        kv_indptr=offsets(counts),
        total_history_lengths=torch.tensor(histories, dtype=torch.int32),
        kv_onload_handle=None,
    )


def jagged(lengths, candidates):
    return SimpleNamespace(
        seqlen_offsets=offsets(lengths), num_candidates=torch.tensor(candidates)
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("page_size", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("split_path", [False, True])
@torch.no_grad()
def test_two_layer_paged_cache_matches_full_reference(
    seed, page_size, dtype, split_path
):
    torch.manual_seed(seed)
    layers = nn.ModuleList(
        [TransformerInferLayer(config(), i, "cpu").to(dtype) for i in range(2)]
    )
    histories, candidates = [0, 3, 8, 16], [1, 2, 3, 4]
    prefixes = [0, seed % 4, seed % 9, 16 if seed % 2 else 7]
    lengths = [h + c for h, c in zip(histories, candidates)]
    users = [torch.randn(n, 12, dtype=dtype) for n in lengths]
    full = users
    for layer in layers:
        full = [reference(layer, x, c) for x, c in zip(full, candidates)]
    metadata = make_metadata(histories, 2, 2, 4, dtype, page_size)

    # Populate cache using independent full-history projections, layer by layer.
    # Prefix K/V are invariant to future history with the causal history mask.
    history_inputs = [x[:h] for x, h in zip(users, histories)]
    for i, layer in enumerate(layers):
        next_history = []
        for u, (x, prefix) in enumerate(zip(history_inputs, prefixes)):
            if x.shape[0]:
                q, k, v = layer.project_qkv(x)
                for p in range(prefix):
                    page = metadata.kv_indices[metadata.kv_indptr[u] + p // page_size]
                    metadata.kv_cache_table[i][page, 0, p % page_size] = k[p]
                    metadata.kv_cache_table[i][page, 1, p % page_size] = v[p]
                next_history.append(reference(layer, x, 0))
            else:
                next_history.append(x)
        history_inputs = next_history

    delta_lengths = [n - p for n, p in zip(lengths, prefixes)]
    jd = jagged(delta_lengths, candidates)
    packed = torch.cat([x[p:] for x, p in zip(users, prefixes)])
    for layer in layers:
        if split_path:
            layer.forward_input(4, packed.shape[0], packed, jd, metadata)
            packed = layer.forward_output(
                4, packed.shape[0], packed, jd, metadata
            ).clone()
        else:
            packed = layer.forward_naive(4, packed.shape[0], packed, jd, metadata)
    expected = torch.cat([x[p:] for x, p in zip(full, prefixes)])
    torch.testing.assert_close(packed, expected, atol=2e-6, rtol=2e-5)
    assert torch.isfinite(packed).all()
    # Candidate slots and unused page tails must never be persisted.
    for table in metadata.kv_cache_table:
        for u, h in enumerate(histories):
            if h and h % page_size:
                last_page = metadata.kv_indices[metadata.kv_indptr[u + 1] - 1]
                assert torch.isnan(table[last_page, :, h % page_size :]).all()


def test_candidates_are_independent_and_permutation_equivariant():
    torch.manual_seed(9)
    layer = TransformerInferLayer(config(), 0, "cpu").double()
    x = torch.randn(9, 12, dtype=torch.float64)
    off, c = offsets([9]), torch.tensor([3])
    before = layer(x, off, c)
    changed = x.clone()
    changed[6] += torch.arange(12)
    after = layer(changed, off, c)
    torch.testing.assert_close(before[7:], after[7:], rtol=0, atol=0)
    perm = torch.tensor([0, 1, 2, 3, 4, 5, 8, 6, 7])
    torch.testing.assert_close(layer(x[perm], off, c), before[perm])


def test_empty_history_and_empty_users():
    layer = TransformerInferLayer(config(), 0, "cpu")
    x = torch.randn(3, 12)
    jd = jagged([0, 2, 1], [0, 2, 1])
    metadata = make_metadata([0, 0, 0], 1, 2, 4, torch.float32)
    got = layer.forward_naive(3, 3, x, jd, metadata)
    torch.testing.assert_close(got, layer(x, jd.seqlen_offsets, jd.num_candidates))
    assert torch.isnan(metadata.kv_cache_table[0]).all()


def test_padded_packed_tail_is_ignored():
    layer = TransformerInferLayer(config(), 0, "cpu")
    x = torch.randn(7, 12)
    jd = jagged([3, 4], [1, 2])
    padded = torch.cat((x, torch.randn(9, 12)))
    actual = layer.forward_naive(2, 16, padded, jd, None)
    torch.testing.assert_close(
        actual[:7], layer(x, jd.seqlen_offsets, jd.num_candidates)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_low_precision_cached_and_full(dtype):
    torch.manual_seed(42)
    layer = TransformerInferLayer(
        config(bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16), 0, "cpu"
    )
    x = torch.randn(9, 12).to(dtype)
    jd = jagged([9], [3])
    metadata = make_metadata([6], 1, 2, 4, dtype)
    got = layer.forward_naive(1, 9, x, jd, metadata)
    expected = reference(layer.double(), x.double(), 3)
    torch.testing.assert_close(got.double(), expected, atol=0.025, rtol=0.02)


def test_invalid_lengths_are_rejected():
    layer = TransformerInferLayer(config(), 0, "cpu")
    with pytest.raises(RuntimeError, match="max_seq_len"):
        layer(torch.randn(25, 12), offsets([25]), torch.tensor([1]))
    with pytest.raises(RuntimeError, match="candidate count"):
        layer(torch.randn(5, 12), offsets([5]), torch.tensor([6]))


def test_non_affine_input_norm_and_empty_batch_tokens():
    layer = TransformerInferLayer(config(learnable_input_layernorm=False), 0, "cpu")
    x = torch.randn(5, 12)
    torch.testing.assert_close(
        layer(x, offsets([5]), torch.tensor([2])), reference(layer, x, 2)
    )
    empty = layer(torch.empty(0, 12), offsets([0, 0]), torch.tensor([0, 0]))
    assert empty.shape == (0, 12)


def test_matching_transformer_checkpoint_and_wrong_backbone_rejection():
    class Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self._backbone, self._use_exportable = "transformer", False
            self._hstu_block = nn.Module()
            self._hstu_block._attention_layers = nn.ModuleList(
                [TransformerInferLayer(config(), 0, "cpu")]
            )

    original, loaded = Dense(), Dense()
    state = copy.deepcopy(original.state_dict())
    state[
        "_embedding_collection._data_parallel_embedding_collection.embeddings.item.weight"
    ] = torch.randn(2, 3)
    load_dense_state_dict(loaded, state, strict=False)
    for a, b in zip(original.parameters(), loaded.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Checkpoint does not match transformer"):
        load_dense_state_dict(
            loaded,
            {"_hstu_block._attention_layers.0._linear_uvqk_weight": torch.randn(1)},
            strict=False,
        )


def test_hstu_checkpoint_transposition_is_preserved():
    dense = nn.Module()
    dense._backbone, dense._use_exportable = "hstu", False
    dense._hstu_block = nn.Module()
    layer = nn.Module()
    layer._linear_uvqk = nn.Linear(3, 8)
    layer._linear_proj = nn.Linear(2, 3, bias=False)
    layer._linear_uvqk_weight = torch.empty(3, 8)
    layer._linear_proj_weight = torch.empty(2, 3)
    dense._hstu_block._attention_layers = nn.ModuleList([layer])
    prefix = "_hstu_block._attention_layers.0."
    uvqk, proj, bias = torch.randn(3, 8), torch.randn(2, 3), torch.randn(8)
    load_dense_state_dict(
        dense,
        {
            prefix + "_linear_uvqk_weight": uvqk,
            prefix + "_linear_uvqk_bias": bias,
            prefix + "_linear_proj_weight": proj,
        },
    )
    torch.testing.assert_close(layer._linear_uvqk.weight, uvqk.T)
    torch.testing.assert_close(layer._linear_uvqk_weight, uvqk)
    torch.testing.assert_close(layer._linear_proj_weight, proj)


def test_export_real_layer_dynamic_batch_and_tokens(tmp_path):
    layer = TransformerInferLayer(config(), 0, "cpu").eval()
    t = torch.export.Dim("tokens", min=2, max=40)
    b = torch.export.Dim("batch", min=1, max=4)
    args = (torch.randn(9, 12), offsets([4, 5]), torch.tensor([1, 2]))
    ep = torch.export.export(layer, args, dynamic_shapes=({0: t}, {0: b + 1}, {0: b}))
    path = tmp_path / "transformer.pt2"
    torch.export.save(ep, path)
    replay = torch.export.load(path).module()
    for lengths, candidates in [
        ([2], [1]),
        ([3, 5, 8], [1, 2, 3]),
        ([0, 4, 6, 12], [0, 0, 1, 2]),
        ([10, 11, 12], [3, 4, 5]),
    ]:
        inputs = (
            torch.randn(sum(lengths), 12),
            offsets(lengths),
            torch.tensor(candidates),
        )
        torch.testing.assert_close(replay(*inputs), layer(*inputs))


class CachedAttention(nn.Module):
    """Expose the production paged reader through tensor-only export inputs."""

    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, x, off, candidates, table, ids, indptr, history):
        """Run attention against an already populated cache."""
        q, k, v = self._layer.project_qkv(x)
        return self._layer.finish(
            x,
            self._layer.attention(
                q, k, v, off, candidates, table, ids, indptr, history
            ),
        )


@dataclass
class CacheExportMetadata:
    """Hold cache tensors used by the CPU export fixture."""

    kv_cache_table: list
    kv_indices: torch.Tensor
    kv_indptr: torch.Tensor
    total_history_lengths: torch.Tensor
    kv_onload_handle: object = None


@dataclass
class JaggedExportMetadata:
    """Hold packed sequence offsets and candidate counts for export."""

    seqlen_offsets: torch.Tensor
    num_candidates: torch.Tensor


class CachedLayer(nn.Module):
    """Expose production cache append and attention as tensor-only inputs."""

    def __init__(self, layer):
        super().__init__()
        self._layer = layer

    def forward(self, x, off, candidates, table, ids, indptr, history):
        """Append new history to the input cache and compute layer outputs."""
        metadata = CacheExportMetadata(
            kv_cache_table=[table],
            kv_indices=ids,
            kv_indptr=indptr,
            total_history_lengths=history,
            kv_onload_handle=None,
        )
        jd = JaggedExportMetadata(seqlen_offsets=off, num_candidates=candidates)
        return self._layer.forward_naive(
            candidates.shape[0], x.shape[0], x, jd, metadata
        )


def test_export_cpu_cache_append_and_replay(tmp_path):
    layer = TransformerInferLayer(config(export_mode=True), 0, "cpu").eval()
    wrapper = CachedLayer(layer)
    md = make_metadata([4, 7], 1, 2, 4, torch.float32)
    md.kv_cache_table[0].normal_()
    args = (
        torch.randn(7, 12),
        offsets([3, 4]),
        torch.tensor([1, 2]),
        md.kv_cache_table[0],
        md.kv_indices,
        md.kv_indptr,
        md.total_history_lengths,
    )
    ep = torch.export.export(wrapper, args)
    torch.export.save(ep, tmp_path / "cache_append.pt2")
    replay = torch.export.load(tmp_path / "cache_append.pt2").module()
    for _ in range(2):
        args[0].normal_()
        eager_args = tuple(x.clone() for x in args)
        replay_args = tuple(x.clone() for x in args)
        expected = wrapper(*eager_args)
        actual = replay(*replay_args)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(replay_args[3], eager_args[3])


def test_export_real_paged_reader_attention(tmp_path):
    layer = TransformerInferLayer(config(), 0, "cpu").eval()
    wrapper = CachedAttention(layer)
    md = make_metadata([4, 7], 1, 2, 4, torch.float32)
    md.kv_cache_table[0].normal_()
    args = (
        torch.randn(7, 12),
        offsets([3, 4]),
        torch.tensor([1, 2]),
        md.kv_cache_table[0],
        md.kv_indices,
        md.kv_indptr,
        md.total_history_lengths,
    )
    ep = torch.export.export(wrapper, args)
    torch.export.save(ep, tmp_path / "cached.pt2")
    replay = torch.export.load(tmp_path / "cached.pt2").module()
    torch.testing.assert_close(replay(*args), wrapper(*args))
    # Cache content is a runtime input, not baked into the exported program.
    args[3].normal_()
    torch.testing.assert_close(replay(*args), wrapper(*args))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA and compiled paged_kvcache_ops"
)
def test_cuda_append_and_graph_matches_eager():
    import paged_kvcache_ops  # noqa: F401

    layer = TransformerInferLayer(
        config(hidden_size=64, head_dim=32, fp16=True), 0
    ).eval()
    x = torch.randn(7, 64, device="cuda", dtype=torch.float16)
    jd = jagged([3, 4], [1, 2])
    jd.seqlen_offsets, jd.num_candidates = (
        jd.seqlen_offsets.cuda(),
        jd.num_candidates.cuda(),
    )
    md = make_metadata([2, 2], 1, 2, 32, torch.float16)
    md.kv_cache_table = [md.kv_cache_table[0].cuda()]
    for attr in ("kv_indices", "kv_indptr", "total_history_lengths"):
        setattr(md, attr, getattr(md, attr).cuda())
    md.batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    md.position = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    md.kv_last_page_len = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    md.new_history_nnz_cuda = torch.tensor([4], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        expected = layer(x, jd.seqlen_offsets, jd.num_candidates)
        eager = layer.forward_naive(2, 7, x, jd, md)
        torch.testing.assert_close(eager, expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layer.forward_input(2, 7, x, jd, md)
                layer.forward_output(2, 7, x, jd, md)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.forward_input(2, 7, x, jd, md)
            output = layer.forward_output(2, 7, x, jd, md)
        graph.replay()
        torch.testing.assert_close(output, expected)
