# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the FA4 attention of the Transformer inference layers.

References: the padded SDPA attention of ``TransformerInferLayer`` with identical weights,
and, independent of it, the mask read back from FA4 itself and a step-by-step fp32
attention. Needs FlashAttention 4 (``flash_attn.cute``, which the Docker image installs)
and a CUDA GPU; otherwise the whole module is skipped, and a skip is not validation of FA4.
"""

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HSTU_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HSTU_ROOT))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FA4 needs a CUDA GPU"
)
pytest.importorskip("flash_attn.cute.interface", reason="FA4 is not installed")

from modules.transformer_infer_layer import TransformerInferLayer
from modules.transformer_infer_layer_fa4 import (
    Fa4Fp8TransformerInferLayer,
    Fa4TransformerInferLayer,
)
from test_transformer_inference import copy_kvcache_metadata
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
)


def allowed_keys(cached, new_history, candidates):
    """The recommendation mask of one user, [new tokens, all keys], absolute positions.

    The new tokens follow the cached prefix: history is causal, a candidate sees the
    history and itself only.
    """
    history = cached + new_history
    queries = cached + torch.arange(new_history + candidates, device="cuda")[:, None]
    keys = torch.arange(history + candidates, device="cuda")[None, :]
    return (keys <= queries) & ((keys < history) | (keys == queries))


def step_by_step(query, key, value, allowed):
    """fp32 attention of one user: scores, mask, softmax, weighted sum of the values."""
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float())
    scores = scores * query.shape[-1] ** -0.5
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probs, value.float())


def build_request(users, queries, keys, values, page_size=4, seed=0):
    """Pack users given as (cached, new history, candidates) for ``attention``.

    ``queries[u]`` holds user u's new tokens; ``keys[u]`` and ``values[u]`` every position
    in order: cached prefix, new history, candidates. As after the layer's append, a paged
    cache in shuffled pages holds the history; the candidates are only in this call.
    """
    history = [cached + new for cached, new, _ in users]
    pages = [-(-length // page_size) for length in history]
    order = torch.Generator().manual_seed(seed)
    page_ids = torch.randperm(sum(pages), generator=order).int().cuda()
    shape = (max(sum(pages), 1), 2, page_size, *keys[0].shape[1:])
    table = torch.zeros(shape, dtype=keys[0].dtype, device="cuda")
    for user, (length, first) in enumerate(
        zip(history, itertools.accumulate([0, *pages]))
    ):
        for pos in range(length):
            page = page_ids[first + pos // page_size]
            table[page, :, pos % page_size] = torch.stack(
                (keys[user][pos], values[user][pos])
            )
    tokens = [new + candidates for _, new, candidates in users]
    starts = [0, *itertools.accumulate(tokens)]
    cached = [user[0] for user in users]
    return SimpleNamespace(
        starts=starts,
        query=torch.cat(queries),
        key=torch.cat([k[c:] for k, c in zip(keys, cached)]),
        value=torch.cat([v[c:] for v, c in zip(values, cached)]),
        offsets=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        candidates=torch.tensor(
            [user[2] for user in users], dtype=torch.int32, device="cuda"
        ),
        cache=dict(
            cache_table=table,
            page_ids=page_ids,
            page_indptr=torch.tensor(
                [0, *itertools.accumulate(pages)], dtype=torch.int32, device="cuda"
            ),
            history_lengths=torch.tensor(history, dtype=torch.int32, device="cuda"),
        ),
    )


# Users as (cached, new history, candidates): history and candidates only, one candidate,
# no candidates, a cached prefix with or without new tokens, and users longer than one
# 128-row FA4 tile. Without a cache, all history is new.
MASK_CASES = {
    "no cache": [(0, 3, 2), (0, 0, 1), (0, 7, 0), (0, 0, 6), (0, 31, 9)],
    "cache": [(5, 2, 3), (0, 4, 2), (9, 0, 4), (12, 3, 0), (3, 0, 0)],
}
NUMERIC_CASES = {
    "no cache": [(0, 3, 2), (0, 200, 100), (0, 129, 7), (0, 0, 6)],
    "cache": [(5, 2, 3), (150, 30, 20), (9, 0, 4), (0, 64, 64)],
}


@pytest.mark.parametrize("case", MASK_CASES)
def test_score_mod_keeps_exactly_the_recommendation_mask(case):
    # Zero queries give every key the same score, so the softmax spreads evenly over the
    # keys that causal and the score_mod keep; one-hot values (key p writes dimension p)
    # make each output row show exactly which keys those were.
    users = MASK_CASES[case]
    fa4 = Fa4TransformerInferLayer(config(), 0, "cuda")
    heads, dim = fa4.num_heads, fa4.head_dim
    one_hot = torch.eye(dim, device="cuda", dtype=torch.bfloat16)[:, None]
    queries = [
        torch.zeros((new + cands, heads, dim), device="cuda", dtype=torch.bfloat16)
        for _, new, cands in users
    ]
    keys = [
        torch.randn((sum(user), heads, dim), device="cuda", dtype=torch.bfloat16)
        for user in users
    ]
    values = [one_hot[: sum(user)].expand(-1, heads, -1).contiguous() for user in users]
    request = build_request(users, queries, keys, values)
    cache = request.cache if case == "cache" else {}
    with torch.inference_mode():
        output = fa4.attention(
            request.query,
            request.key,
            request.value,
            request.offsets,
            request.candidates,
            **cache,
        )
    output = output.view(-1, heads, dim)
    for user, start, end in zip(users, request.starts, request.starts[1:]):
        allowed = allowed_keys(*user)[:, None].expand(-1, heads, -1)
        rows = output[start:end]
        assert torch.equal(rows[..., : sum(user)] > 0, allowed)
        assert not rows[..., sum(user) :].any()
        weights = allowed / allowed.sum(-1, keepdim=True)
        torch.testing.assert_close(
            rows[..., : sum(user)].float(), weights, rtol=1e-2, atol=1e-3
        )


@pytest.mark.parametrize("case", NUMERIC_CASES)
def test_attention_matches_step_by_step_reference(case):
    users = NUMERIC_CASES[case]
    fa4 = Fa4TransformerInferLayer(config(max_seq_len=512), 0, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(0)

    def random(rows):
        return torch.randn(
            (rows, fa4.num_heads, fa4.head_dim),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )

    queries = [random(new + cands) for _, new, cands in users]
    keys = [random(sum(user)) for user in users]
    values = [random(sum(user)) for user in users]
    request = build_request(users, queries, keys, values)
    cache = request.cache if case == "cache" else {}
    with torch.inference_mode():
        output = fa4.attention(
            request.query,
            request.key,
            request.value,
            request.offsets,
            request.candidates,
            **cache,
        )
    output = output.view(-1, fa4.num_heads, fa4.head_dim)
    for user, start, end, query, key, value in zip(
        users, request.starts, request.starts[1:], queries, keys, values
    ):
        expected = step_by_step(query, key, value, allowed_keys(*user))
        assert cosine(output[start:end], expected) > 0.99999
        assert (output[start:end].float() - expected).abs().max() < 0.02


def requires_fp8():
    pytest.importorskip("deep_gemm", reason="DeepGEMM is not installed")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("FP8 DeepGEMM needs an SM90+ GPU")


# Lengths and candidate counts per user: users without candidates, users made only of
# candidates, a single token, and users longer than one 128-row FA4 tile.
CASES = {
    "mixed": ([5, 1, 33, 70, 9], [2, 1, 0, 4, 9]),
    "long": ([300, 129, 513], [100, 7, 64]),
}


@pytest.mark.parametrize("case", CASES)
def test_attention_matches_reference(case):
    lengths, candidates = CASES[case]
    base, fa4 = layer_pair(Fa4TransformerInferLayer, max_seq_len=1024, max_batch_size=8)
    x, offsets, candidates = layer_inputs(lengths=lengths, candidates=candidates)
    with torch.inference_mode():
        q, k, v = base.project_qkv(x)
        expected = base.attention(q, k, v, offsets, candidates)
        actual = fa4.attention(q, k, v, offsets, candidates)
        # Plain causal attention, where candidates also see earlier candidates.
        causal = base.attention(q, k, v, offsets, torch.zeros_like(candidates))
    assert cosine(actual, expected) > 0.99999
    assert (actual.float() - expected.float()).abs().max() < 0.05
    # The candidate rule changes these outputs, so the comparison above covers it.
    assert (causal.float() - expected.float()).abs().max() > 0.1


def test_per_user_inputs_have_an_element_past_the_end(monkeypatch):
    # FA4 runs spare CTAs for the batch index past the last user, which read one element
    # past cu_seqlens and the candidate counts: every call must pass views with one more
    # element of the same allocation behind them, repeating the end (an empty sequence).
    pytest.importorskip("paged_kvcache_ops")
    from modules import transformer_infer_layer_fa4

    calls = []
    original = transformer_infer_layer_fa4.flash_attn_varlen_func

    def spy(*args, cu_seqlens_q, cu_seqlens_k, aux_tensors, **kwargs):
        calls.append((cu_seqlens_q, cu_seqlens_k, aux_tensors[0]))
        return original(
            *args,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            aux_tensors=aux_tensors,
            **kwargs,
        )

    monkeypatch.setattr(transformer_infer_layer_fa4, "flash_attn_varlen_func", spy)
    _, fa4 = layer_pair(Fa4TransformerInferLayer)
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        fa4(x, offsets, candidates)
    _, cached = layer_pair(Fa4TransformerInferLayer, **CACHE_LAYER)
    x, jd = cache_inputs()
    with torch.inference_mode():
        cached.forward_naive(2, 7, x, jd, cache_metadata([4, 3], seed=0))
    assert len(calls) == 2
    for tensors in calls:
        for values in tensors:
            behind = torch.empty(0, dtype=values.dtype, device=values.device)
            behind.set_(
                values.untyped_storage(), values.storage_offset(), (values.numel() + 1,)
            )
            assert torch.equal(behind[:-1], values) and behind[-1] == values[-1]


def test_invalid_lengths_are_rejected_before_the_kernel(monkeypatch):
    # The reference's length checks (test_invalid_lengths_are_rejected), which the FA4
    # bounds and the packed K/V rely on. A failed device assert would end the CUDA context
    # of the remaining tests, so the asserts run on the host here, and neither the K/V
    # packing nor FA4 may be reached.
    from modules import transformer_infer_layer_fa4

    class Reached(Exception):
        pass

    def reached(*args, **kwargs):
        raise Reached

    def host_assert(condition, message):
        if not condition.item():
            raise RuntimeError(message)

    monkeypatch.setattr(torch, "_assert_async", host_assert)
    monkeypatch.setattr(transformer_infer_layer_fa4, "flash_attn_varlen_func", reached)
    monkeypatch.setattr(Fa4TransformerInferLayer, "_packed_kv", reached)
    fa4 = Fa4TransformerInferLayer(config(max_seq_len=16), 0, "cuda")

    def attention(length, candidates, history=None):
        x = torch.randn(length, 4, 128, device="cuda", dtype=torch.bfloat16)
        offsets = torch.tensor([0, length], dtype=torch.int32, device="cuda")
        cache = {}
        if history is not None:
            # Only the history lengths are read before the K/V packing.
            unused = torch.empty(0, device="cuda")
            cache = dict(cache_table=unused, page_ids=unused, page_indptr=unused)
            cache["history_lengths"] = torch.tensor([history], device="cuda")
        candidates = torch.tensor([candidates], dtype=torch.int32, device="cuda")
        fa4.attention(x, x, x, offsets, candidates, **cache)

    with pytest.raises(RuntimeError, match="sequence length exceeds"):
        attention(17, 1)
    with pytest.raises(RuntimeError, match="candidate count"):
        attention(5, 6)
    with pytest.raises(RuntimeError, match="negative cached prefix"):
        attention(5, 1, history=3)
    with pytest.raises(RuntimeError, match="cached sequence exceeds"):
        attention(5, 1, history=16)
    # At the limits, the lengths are valid and reach the kernel.
    with pytest.raises(Reached):
        attention(16, 16)
    with pytest.raises(Reached):
        attention(5, 1, history=15)


def test_padded_tail_rows_are_zero_and_ignored():
    # A CUDA graph bucket packs more rows than the request has; its tail holds stale data.
    base, fa4 = layer_pair(Fa4TransformerInferLayer)
    x, offsets, candidates = layer_inputs()
    rows = x.shape[0]
    tail = torch.full((9, 512), float("nan"), device="cuda", dtype=torch.bfloat16)
    padded = torch.cat((x, tail))
    jd = SimpleNamespace(seqlen_offsets=offsets, num_candidates=candidates)
    with torch.inference_mode():
        expected = base(x, offsets, candidates)
        actual = fa4.forward_naive(len(candidates), padded.shape[0], padded, jd, None)
        attention = fa4.attention(*fa4.project_qkv(padded), offsets, candidates)
    assert torch.isfinite(actual[:rows]).all()
    assert cosine(actual[:rows], expected) > 0.99999
    assert (attention[rows:] == 0).all()


def test_layer_forward_matches_reference():
    base, fa4 = layer_pair(Fa4TransformerInferLayer)
    x, offsets, candidates = layer_inputs()
    with torch.inference_mode():
        expected = base(x, offsets, candidates)
        actual = fa4(x, offsets, candidates)
    assert cosine(actual, expected) > 0.99999


def test_paged_cache_forward_matches_reference():
    pytest.importorskip("paged_kvcache_ops")
    base, fa4 = layer_pair(Fa4TransformerInferLayer, **CACHE_LAYER)
    x, jd = cache_inputs()
    # Total history per user; the part not in this call is already cached.
    for seed, histories in enumerate(([2, 2], [3, 2], [4, 3], [9, 2])):
        expected_cache = cache_metadata(histories, seed)
        actual_cache = cache_metadata(histories, seed)
        with torch.inference_mode():
            expected = base.forward_naive(2, 7, x, jd, expected_cache)
            actual = fa4.forward_naive(2, 7, x, jd, actual_cache)
        assert cosine(actual, expected) > 0.99999


def test_cuda_graph_with_paged_cache_follows_new_requests():
    pytest.importorskip("paged_kvcache_ops")
    base, fa4 = layer_pair(Fa4TransformerInferLayer, **CACHE_LAYER)
    x, jd = cache_inputs()
    static = cache_metadata([2, 2], seed=0)
    with torch.inference_mode():
        # Warm up on a side stream so FA4 compiles outside the capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fa4.forward_input(2, 7, x, jd, static)
                fa4.forward_output(2, 7, x, jd, static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fa4.forward_input(2, 7, x, jd, static)
            output = fa4.forward_output(2, 7, x, jd, static)
        # Like a new request: same graph shapes, new cached prefix lengths, positions and
        # physical pages.
        for seed, histories in enumerate(([3, 2], [4, 3], [2, 2]), start=1):
            source = cache_metadata(histories, seed)
            static.kv_cache_table[0].copy_(source.kv_cache_table[0])
            expected = base.forward_naive(2, 7, x, jd, cache_metadata(histories, seed))
            copy_kvcache_metadata(static, source)
            graph.replay()
            assert cosine(output, expected) > 0.99999


def test_config_selects_fa4_layers():
    from modules.hstu_block_inference import HSTUBlockInference
    from utils.gin_config_args import NetworkArgs

    block = HSTUBlockInference(dense_config(transformer_attention="fa4"))
    assert all(
        type(layer) is Fa4TransformerInferLayer for layer in block._attention_layers
    )
    block = HSTUBlockInference(dense_config())
    assert all(
        type(layer) is TransformerInferLayer for layer in block._attention_layers
    )
    with pytest.raises(ValueError, match="Unknown transformer attention"):
        dense_config(transformer_attention="flash")
    with pytest.raises(ValueError, match="transformer backbone"):
        dense_config(backbone="hstu", transformer_attention="fa4")
    with pytest.raises(ValueError, match="transformer backbone"):
        NetworkArgs(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            kv_channels=64,
            transformer_attention="fa4",
        )
    with pytest.raises(ValueError, match="exported"):
        Fa4TransformerInferLayer(config(export_mode=True), 0, "cuda")


def test_config_selects_fa4_layers_with_fp8_projections():
    requires_fp8()
    from modules.hstu_block_inference import HSTUBlockInference

    cfg = dense_config(transformer_attention="fa4", transformer_fp8_linear=True)
    layers = HSTUBlockInference(cfg)._attention_layers
    assert all(type(layer) is Fa4Fp8TransformerInferLayer for layer in layers)


@pytest.mark.parametrize("fp8", [False, True])
def test_dense_module_graphs_match_reference(monkeypatch, fp8):
    # The Triton model's order: per-layer graphs captured while the module is built,
    # then the bf16 cast and the checkpoint. The reference is the eager SDPA module.
    if fp8:
        requires_fp8()
    source = dense_module(0, dense_config())
    cfg = dense_config(transformer_attention="fa4", transformer_fp8_linear=fp8)
    graphs = {"batch_size": [2], "length_per_sequence": [64]}
    served = dense_module(1, cfg, use_cudagraph=True, cudagraph_configs=graphs)
    served.load_state_dict(source.state_dict(), strict=False)
    replays = count_graph_replays(monkeypatch, served)
    batch, embeddings = dense_batch()
    expected = source(batch, embeddings)
    actual = served(batch, embeddings)
    assert len(replays) == 1
    # FP8 projections add their quantization noise to the FA4 attention.
    assert cosine(actual, expected) > (0.999 if fp8 else 0.99999)
