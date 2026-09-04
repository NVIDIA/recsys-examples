# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weighted pooled (SUM) training tests at the BatchedDynamicEmbeddingTablesV2
level. The DEBUG initializer fills every dim of row i with i % 100_000, so the
weighted-sum output and the SGD update have exact Python oracles."""

import pytest
import torch
from dynamicemb import (
    DynamicEmbCheckMode,
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.dynamicemb_config import DEBUG_EMB_INITIALIZER_MOD
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.types import BoundsCheckMode


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


def _debug_value(idx: torch.Tensor) -> torch.Tensor:
    return (idx % DEBUG_EMB_INITIALIZER_MOD).float()


def _make_v2(
    dims,
    pooling_mode=DynamicEmbPoolingMode.SUM,
    lr=0.5,
    num_tables=None,
    device=None,
):
    device = device or torch.cuda.current_device()
    opts = [
        DynamicEmbTableOptions(
            dim=d,
            max_capacity=4096,
            init_capacity=4096,
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=device,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.STEP,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG
            ),
        )
        for d in dims
    ]
    return BatchedDynamicEmbeddingTablesV2(
        table_options=opts,
        table_names=[f"t{i}" for i in range(len(opts))],
        feature_table_map=list(range(len(opts))),
        pooling_mode=pooling_mode,
        optimizer=EmbOptimType.SGD,
        learning_rate=lr,
        stochastic_rounding=False,
        bounds_check_mode=BoundsCheckMode.NONE,
    )


def _weighted_oracle(indices, offsets, weights, batch_size, feature_num):
    """Reference weighted-sum pooling: slot s = f * B + b. Returns [B, F]."""
    out = torch.zeros(batch_size, feature_num, device=indices.device)
    for s in range(feature_num * batch_size):
        f, b = s // batch_size, s % batch_size
        acc = 0.0
        for p in range(int(offsets[s]), int(offsets[s + 1])):
            acc += float(weights[p]) * float(indices[p] % DEBUG_EMB_INITIALIZER_MOD)
        out[b, f] = acc
    return out


def test_weighted_sum_forward(current_device):
    device = torch.cuda.current_device()
    B, F = 2, 1
    module = _make_v2([8], device=device)

    indices = torch.tensor([5, 12, 300, 7, 9], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0, 3.0], dtype=torch.float32, device=device)

    out = module(indices, offsets, pooling_weights=weights)  # [B, 8]
    ref = _weighted_oracle(indices, offsets, weights, B, F)  # [B, F]

    assert out.shape == (B, 8)
    for b in range(B):
        for c in (0, 3, 7):  # every dim of the row carries the same DEBUG value
            assert torch.allclose(
                out[b, c], ref[b, 0], rtol=1e-5, atol=1e-3
            ), f"b={b} c={c}: {out[b, c].item()} vs {ref[b, 0].item()}"


def test_weighted_sum_backward_sgd_oracle(current_device):
    device = torch.cuda.current_device()
    B, F, lr = 2, 1, 0.5
    module = _make_v2([8], lr=lr, device=device)

    indices = torch.tensor([5, 12, 300, 7, 9], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0, 3.0], dtype=torch.float32, device=device)

    out = module(indices, offsets, pooling_weights=weights)
    target = torch.full((B, 8), 2.0, dtype=torch.float32, device=device)
    (out * target).sum().backward()
    torch.cuda.synchronize()

    # grad_k = 2.0 * sum of weights over the key's positions; SGD: v -= lr * grad.
    keys_cpu = indices.tolist()
    wsum = {}
    for k, w in zip(keys_cpu, weights.tolist()):
        wsum[k] = wsum.get(k, 0.0) + w

    exp_keys, exp_vals = module.export_keys_values("t0", torch.device(device))
    got = {int(k): v for k, v in zip(exp_keys.tolist(), exp_vals)}
    for k, s in wsum.items():
        expected = float(k % DEBUG_EMB_INITIALIZER_MOD) - lr * 2.0 * s
        row = got[k]
        assert torch.allclose(row, torch.full_like(row, expected), atol=1e-3), (
            f"key {k}: got {row[0].item()}, expected {expected}"
        )


def test_weighted_sum_mixed_D_forward(current_device):
    device = torch.cuda.current_device()
    B, F = 2, 2
    module = _make_v2([8, 4], device=device)  # mixed-D: 8 + 4 = 12 columns

    # Feature-major slots: s0=f0b0, s1=f0b1, s2=f1b0, s3=f1b1
    indices = torch.tensor([5, 12, 7, 9, 100, 3, 1, 1], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64, device=device)
    weights = torch.tensor(
        [0.5, 1.5, 1.0, 3.0, 2.0, 2.0, 0.25, 0.75], dtype=torch.float32, device=device
    )

    out = module(indices, offsets, pooling_weights=weights)  # [B, 12]
    ref = _weighted_oracle(indices, offsets, weights, B, F)  # [B, F]

    assert out.shape == (B, 12)
    for b in range(B):
        for f in range(F):
            col = 0 if f == 0 else 8
            assert torch.allclose(out[b, col], ref[b, f], rtol=1e-5, atol=1e-3), (
                f"b={b} f={f}: {out[b, col].item()} vs {ref[b, f].item()}"
            )


def test_weighted_errors(current_device):
    device = torch.cuda.current_device()
    indices = torch.tensor([5, 12], dtype=torch.int64, device=device)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    weights = torch.tensor([0.5, 1.5], dtype=torch.float32, device=device)

    # weights + MEAN -> ValueError
    mean_mod = _make_v2([8], pooling_mode=DynamicEmbPoolingMode.MEAN, device=device)
    with pytest.raises(ValueError):
        mean_mod(indices, offsets, pooling_weights=weights)

    # non-fp32 weights -> ValueError
    mod = _make_v2([8], device=device)
    with pytest.raises(ValueError):
        mod(indices, offsets, pooling_weights=weights.half())

    # numel mismatch -> ValueError
    with pytest.raises(ValueError):
        mod(indices, offsets, pooling_weights=weights[:1])

    # eval mode -> ValueError
    mod_eval = _make_v2([8], device=device)
    mod_eval.eval()
    with pytest.raises(ValueError):
        mod_eval(indices, offsets, pooling_weights=weights)
