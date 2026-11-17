# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 full-row publication must retain the packed FP32 accumulator."""
import sys
from pathlib import Path

import pytest
import torch
from dynamicemb import DynamicEmbPoolingMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_batched_dynamic_embedding_tables_v2 import _rowwise_adagrad_bdeb


@pytest.mark.parametrize("hbm_bytes", [0, 4096], ids=["host", "hybrid"])
def test_fp8_overlapping_snapshots_preserve_current_weights_and_fp32_moments(hbm_bytes):
    model = _rowwise_adagrad_bdeb(
        torch.float8_e4m3fn, [16], False, hbm_bytes, DynamicEmbPoolingMode.NONE
    )
    keys = torch.arange(512, device="cuda", dtype=torch.int64)
    offsets = torch.tensor([0, 512], device="cuda", dtype=torch.int64)
    first = model.resolve_input(keys, offsets)
    later = model.resolve_input(keys, offsets)
    first_order = first.unique_keys.argsort()
    later_order = later.unique_keys.argsort()
    torch.testing.assert_close(
        first.unique_keys[first_order], later.unique_keys[later_order]
    )
    before = first.unique_values.view(torch.uint8).clone()
    (model(keys, offsets, prefetch_state=later).sum() * 1e-8).backward()
    assert not torch.equal(first.unique_values.view(torch.uint8), before)
    torch.testing.assert_close(
        first.unique_values.view(torch.uint8)[first_order],
        later.unique_values.view(torch.uint8)[later_order],
        rtol=0,
        atol=0,
    )
    (model(keys, offsets, prefetch_state=first).sum() * 1e-8).backward()
    # Width16 plus16 bytes of aligned state; only the first4 state bytes hold the FP32 accumulator.
    moments = first.unique_values[:, 16:20].contiguous().view(torch.float32)
    torch.testing.assert_close(
        moments, torch.full_like(moments, 2e-16), rtol=1e-5, atol=0
    )
    assert not model._active_rows.states
