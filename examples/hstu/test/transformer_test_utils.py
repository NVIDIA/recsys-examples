# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the Transformer inference tests.

The tests that import this module put examples/hstu on sys.path first, and
test_transformer_inference.py also puts corelib's recsys_kvcache_manager there.
"""

import itertools
from types import SimpleNamespace

import torch
from modules.transformer_infer_layer import TransformerInferLayer
from recsys_kvcache_manager.kvcache_metadata import get_kvcache_metadata_buffer
from torch import nn
from torch.nn import functional as F


def config(hidden=512, heads=4, head_dim=128, ffn=2048, **overrides):
    args = dict(
        num_heads=heads,
        head_dim=head_dim,
        hidden_size=hidden,
        max_seq_len=128,
        max_batch_size=4,
        export_mode=False,
        residual=True,
        bf16=True,
        fp16=False,
        layernorm_epsilon=1e-5,
        transformer_ffn_dim=ffn,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def randomize_linears(layer, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                scale = module.in_features**-0.5
                for name, factor in (("weight", scale), ("bias", 0.1)):
                    param = getattr(module, name)
                    noise = torch.randn(param.shape, generator=generator, device="cuda")
                    param.copy_((noise * factor).to(param.dtype))


def layer_pair(layer_cls, **config_kwargs):
    """The reference ``TransformerInferLayer`` and a ``layer_cls`` with its weights."""
    cfg = config(**config_kwargs)
    base = TransformerInferLayer(cfg, 0, "cuda")
    randomize_linears(base)
    layer = layer_cls(cfg, 0, "cuda")
    layer.load_state_dict(base.state_dict())
    return base, layer


def layer_inputs(hidden=512, lengths=(1, 5, 33, 70), candidates=(1, 2, 0, 4), seed=1):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (sum(lengths), hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    offsets = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda"
    )
    return x, offsets, torch.tensor(candidates, dtype=torch.int32, device="cuda")


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def offsets(lengths):
    return torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32
    )


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


def manager_metadata(histories, lengths, candidates, device="cpu"):
    """Build real manager buffers for requests with two heads of width 64."""
    source = make_metadata(histories, 1, 2, 64, torch.float16)
    new_history = [n - c for n, c in zip(lengths, candidates)]
    md = get_kvcache_metadata_buffer(
        len(histories), sum(new_history), len(source.kv_indices), device=device
    )
    md.kv_cache_table = [source.kv_cache_table[0].to(device).normal_()]
    for name in ("kv_indices", "kv_indptr", "total_history_lengths"):
        getattr(md, name).copy_(getattr(source, name))
    md.kv_last_page_len.copy_(
        torch.tensor([(h - 1) % 4 + 1 if h else 0 for h in histories])
    )
    md.total_history_offsets.copy_(offsets(histories))
    md.new_history_offsets.copy_(offsets(new_history))
    md.new_history_nnz_cuda.fill_(sum(new_history))
    md.kv_seqlens.copy_(torch.tensor([h + c for h, c in zip(histories, candidates)]))
    md.kv_seqlen_offsets.copy_(offsets([h + c for h, c in zip(histories, candidates)]))
    md.batch_indices.copy_(
        torch.tensor([u for u, n in enumerate(new_history) for _ in range(n)])
    )
    md.position.copy_(
        torch.tensor(
            [p for h, n in zip(histories, new_history) for p in range(h - n, h)]
        )
    )
    return md


# manager_metadata builds two heads of width 64, so the cache tests use a layer of
# hidden size 128, which keeps every projection width FP8 eligible.
CACHE_LAYER = dict(hidden=128, heads=2, head_dim=64, ffn=256, max_seq_len=24)


def cache_inputs(seed=1):
    """Two users of 3 and 4 tokens with 1 and 2 candidates, packed."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((7, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
    jd = jagged([3, 4], [1, 2])
    jd.seqlen_offsets, jd.num_candidates = (
        jd.seqlen_offsets.cuda(),
        jd.num_candidates.cuda(),
    )
    return x, jd


def cache_metadata(histories, seed):
    """``manager_metadata`` on CUDA with a bf16 cache table; same seed, same content."""
    torch.manual_seed(seed)
    metadata = manager_metadata(histories, [3, 4], [1, 2], device="cuda")
    metadata.kv_cache_table = [t.to(torch.bfloat16) for t in metadata.kv_cache_table]
    return metadata


def dense_config(**overrides):
    """A two-layer Transformer dense config; the reference attention and bf16 by default."""
    from configs import get_inference_hstu_config

    args = dict(
        hidden_size=512,
        num_layers=2,
        num_attention_heads=4,
        head_dim=128,
        max_batch_size=2,
        max_seq_len=64,
        dtype=torch.bfloat16,
        contextual_max_seqlen=1,
        backbone="transformer",
        transformer_ffn_dim=1024,
    )
    args.update(overrides)
    return get_inference_hstu_config(**args)


def dense_module(seed, cfg, **kwargs):
    from configs import InferenceEmbeddingConfig, RankingConfig
    from modules.inference_dense_module import InferenceDenseModule

    task = RankingConfig(
        embedding_configs=[InferenceEmbeddingConfig(["item"], "item", 16, 512, False)],
        prediction_head_arch=[512, 2],
        num_tasks=2,
    )
    torch.manual_seed(seed)
    return InferenceDenseModule(cfg, None, task, **kwargs).bfloat16().eval()


def dense_batch(users=2, history=20, candidates=6, seed=2):
    """Per user: one context token, ``history`` items with actions and the candidates."""
    from commons.datasets.hstu_batch import HSTUBatch
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    generator = torch.Generator(device="cuda").manual_seed(seed)
    item_len = history + candidates
    rows = {
        "context": [1] * users,
        "item": [item_len] * users,
        "action": [item_len] * users,
    }
    embeddings = {
        name: JaggedTensor(
            values=torch.randn(
                (sum(lens), 512),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            lengths=torch.tensor(lens, device="cuda"),
        )
        for name, lens in rows.items()
    }
    lengths = sum(rows.values(), [])
    features = KeyedJaggedTensor(
        keys=list(rows),
        values=torch.arange(sum(lengths), device="cuda"),
        lengths=torch.tensor(lengths, device="cuda"),
    )
    batch = HSTUBatch(
        features=features,
        batch_size=users,
        feature_to_max_seqlen={"context": 1, "item": item_len, "action": item_len},
        contextual_feature_names=["context"],
        item_feature_name="item",
        action_feature_name="action",
        max_num_candidates=candidates,
        num_candidates=torch.full((users,), candidates, device="cuda"),
    )
    return batch, embeddings


def count_graph_replays(monkeypatch, model):
    """A list that grows by one each time the dense module replays its CUDA graphs."""
    block = model._hstu_block
    replays = []
    replay = block.predict_cudagraph
    monkeypatch.setattr(
        block, "predict_cudagraph", lambda *args: replays.append(1) or replay(*args)
    )
    return replays
