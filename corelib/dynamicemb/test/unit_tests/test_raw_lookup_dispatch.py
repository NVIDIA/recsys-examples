# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw/FIFO compatibility and explicit resolved lookup share the same optimizer."""
import pytest
from dynamicemb import DynamicEmbPoolingMode
from test_no_grad_slot_lifetime import batch, check_values, make_model, pins


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("layout", ["hbm", "cache", "host", "hybrid"])
@pytest.mark.parametrize(
    "pooling", [DynamicEmbPoolingMode.NONE, DynamicEmbPoolingMode.SUM]
)
def test_only_explicit_resolved_state_uses_torchrec_lookup(prepared, layout, pooling):
    model, counter = make_model(pooling, layout)
    keys, offsets = batch(0)
    native = model._native_lookup
    calls = []

    def lookup(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    model._native_lookup = lookup
    state = model.resolve_input(keys, offsets) if prepared else None
    output = model(keys, offsets, prefetch_state=state)
    check_values(output, keys, pooling)
    assert len(calls) == int(prepared)
    output.sum().backward()
    if counter is not None:
        assert pins(counter) == 0
