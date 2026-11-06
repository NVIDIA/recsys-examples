# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Tests of MixedDimBuffers: one [rows, value_dim] buffer per table, tables of
# different embedding dimensions, and the row kernels addressed through them.
#
# Run (from ``corelib/dynamicemb``)::
#   python -m pytest test/unit_tests/test_mixed_dim_buffers.py -q

from __future__ import annotations

from typing import Sequence

import pytest
import torch
from dynamicemb.mixed_dim_buffers import MixedDimBuffers, BuffersView
from dynamicemb.types import CopyMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make(
    kind: str = "fixed",
    rows: Sequence[int] = (32, 48),
    emb_dims: Sequence[int] = (4, 8),
    value_dims: Sequence[int] = (12, 8),  # table 0 also has 8 floats of optimizer state
) -> MixedDimBuffers:
    return MixedDimBuffers(
        emb_dims,
        value_dims,
        rows,
        torch.float32,
        torch.device("cuda", torch.cuda.current_device()),
        kind,
    )


def idx(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.int64, device="cuda")


def test_dimensions_and_view():
    b = make()
    assert b.num_tables == 2 and b.kind == "fixed" and b.on_device
    assert b.max_emb_dim == 8 and b.max_value_dim == 12
    assert [b.emb_dim(t) for t in range(2)] == [4, 8]
    assert [b.value_dim(t) for t in range(2)] == [12, 8]
    assert [b.num_rows(t) for t in range(2)] == [32, 48]
    assert b.all_dims_vec4  # every dimension is a multiple of 4
    assert not make(emb_dims=(3, 8), value_dims=(5, 8)).all_dims_vec4

    view = b.view
    assert isinstance(view, BuffersView)
    assert view.dtype == torch.float32
    assert view.max_emb_dim == 8 and view.max_value_dim == 12
    assert view.value_dims.tolist() == [12, 8] and view.emb_dims.tolist() == [4, 8]
    assert view.ptrs.tolist() == [b.tensor(t).data_ptr() for t in range(2)]
    assert tuple(b.tensor(0).shape) == (32, 12) and tuple(b.tensor(1).shape) == (48, 8)


def test_the_constructor_checks_its_inputs():
    dev = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="one embedding dim"):
        MixedDimBuffers([4], [4, 8], [8], torch.float32, dev, "fixed")
    with pytest.raises(ValueError, match="kind"):
        MixedDimBuffers([4], [4], [8], torch.float32, dev, "nope")


def test_store_then_load_round_trips_every_table_in_its_own_width():
    # A padded row is [embedding, zero-padded to max_emb_dim][optimizer state], so the
    # widest row needs max_emb_dim + the largest optimizer width (here 8 + 16 = 24).
    # Adam-like: the optimizer state is twice the embedding.
    b = make(emb_dims=(4, 8), value_dims=(12, 24))
    assert b.max_emb_dim == 8 and b.max_value_dim == 24
    rows = idx([0, 5, 7, 1, 2])
    table_ids = idx([0, 0, 0, 1, 1])
    vals = torch.randn(5, b.max_value_dim, device="cuda")
    b.store_padded(rows, table_ids, vals)

    mx = b.max_emb_dim
    got = b.load_padded(rows, table_ids, CopyMode.VALUE)
    assert tuple(got.shape) == (5, b.max_value_dim)
    for i, t in enumerate(table_ids.tolist()):
        e, o = b.emb_dim(t), b.value_dim(t) - b.emb_dim(t)
        torch.testing.assert_close(got[i, :e], vals[i, :e])
        torch.testing.assert_close(got[i, mx : mx + o], vals[i, mx : mx + o])
        # and it is in that table's own buffer, at that row, without the padding
        row = b.tensor(t)[rows[i].item()]
        torch.testing.assert_close(row[:e], vals[i, :e])
        torch.testing.assert_close(row[e:], vals[i, mx : mx + o])

    emb = b.load_padded(rows, table_ids, CopyMode.EMBEDDING)
    assert tuple(emb.shape) == (5, b.max_emb_dim)
    for i, t in enumerate(table_ids.tolist()):
        e = b.emb_dim(t)
        torch.testing.assert_close(emb[i, :e], vals[i, :e])


def test_loading_and_storing_whole_rows_of_one_table():
    b = make()
    rows = idx([3, 4])
    vals = torch.randn(2, 8, device="cuda")  # table 1 is 8 wide
    b.store_exact(rows, 1, vals)
    torch.testing.assert_close(b.load_exact(rows, 1), vals)
    torch.testing.assert_close(b.tensor(1)[rows], vals)
    # nothing to do, nothing to fail
    assert tuple(b.load_exact(rows[:0], 0).shape) == (0, 12)
    b.store_exact(rows[:0], 0, torch.empty(0, 12, device="cuda"))
    assert tuple(b.load_padded(rows[:0], rows[:0], CopyMode.VALUE).shape) == (0, 12)


def test_a_one_dimensional_value_is_a_column():
    b = make(emb_dims=(1, 1), value_dims=(1, 1))
    rows = idx([0, 1])
    b.store_padded(rows, idx([0, 1]), torch.tensor([2.0, 3.0], device="cuda"))
    got = b.load_padded(rows, idx([0, 1]), CopyMode.VALUE)
    torch.testing.assert_close(got[:, 0], torch.tensor([2.0, 3.0], device="cuda"))


def test_a_device_buffer_grows_and_keeps_its_rows():
    b = make("device")
    assert b.kind == "device" and b.on_device
    rows = idx([0, 1])
    vals = torch.randn(2, 8, device="cuda")
    b.store_exact(rows, 1, vals)

    assert not b.grow_to(1, 10)  # it has 48 already
    assert b.grow_to(1, 100)
    assert b.num_rows(1) >= 100
    # the pointers the kernels use follow the buffers
    assert b.ptrs.tolist() == [b.tensor(t).data_ptr() for t in range(2)]
    torch.testing.assert_close(b.load_exact(rows, 1), vals)
    # and the grown part is addressable
    b.store_exact(idx([99]), 1, torch.ones(1, 8, device="cuda"))
    torch.testing.assert_close(b.load_exact(idx([99]), 1), torch.ones(1, 8, device="cuda"))


def test_a_fixed_pool_does_not_grow():
    b = make("fixed")
    assert not b.grow_to(0, 10)  # already big enough: nothing to do
    with pytest.raises(NotImplementedError):
        b.grow_to(0, 100)
    assert b.num_rows(0) == 32
