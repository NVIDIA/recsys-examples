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
from dynamicemb.state.mixed_dim_buffers import MixedDimBuffers, PaddedBuffer
from dynamicemb.types import CopyMode
from dynamicemb_extensions import PaddedBufferView

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

    view = b.flat_tables_view()
    assert view.dtype == torch.float32
    assert view.max_emb_dim == 8 and view.max_value_dim == 12 and view.all_dims_vec4
    assert view.value_dims.tolist() == [12, 8] and view.emb_dims.tolist() == [4, 8]
    assert view.ptrs.tolist() == [b.tensor(t).data_ptr() for t in range(2)]
    padded = b.padded_buffer_view()
    assert padded.max_emb_dim == 8 and padded.max_value_dim == 12
    assert padded.emb_dims.tolist() == [4, 8] and padded.all_dims_vec4
    assert tuple(b.tensor(0).shape) == (32, 12) and tuple(b.tensor(1).shape) == (48, 8)


def test_the_constructor_checks_its_inputs():
    dev = torch.device("cuda", torch.cuda.current_device())
    with pytest.raises(ValueError, match="one embedding dim"):
        MixedDimBuffers([4], [4, 8], [8], torch.float32, dev, "fixed")
    with pytest.raises(ValueError, match="kind"):
        MixedDimBuffers([4], [4], [8], torch.float32, dev, "nope")


# (emb_dims, value_dims): the first is vec4-aligned, the second is not, so the scalar
# kernels get a round trip of their own.
ALIGNED = ((4, 8), (12, 24))
UNALIGNED = ((3, 5), (9, 15))


@pytest.mark.parametrize(
    "emb_dims,value_dims", [ALIGNED, UNALIGNED], ids=["vec4", "scalar"]
)
def test_store_then_load_round_trips_every_table_in_its_own_width(emb_dims, value_dims):
    # A padded row is [embedding, zero-padded to max_emb_dim][optimizer state], so the
    # widest row needs max_emb_dim + the largest optimizer width (e.g. 8 + 16 = 24).
    # Adam-like: the optimizer state is twice the embedding.
    b = make(emb_dims=emb_dims, value_dims=value_dims)
    assert b.all_dims_vec4 == (emb_dims == ALIGNED[0])
    assert b.max_emb_dim == max(emb_dims) and b.max_value_dim == max(value_dims)
    rows = idx([0, 5, 7, 1, 2])
    table_ids = idx([0, 0, 0, 1, 1])
    vals = torch.randn(5, b.max_value_dim, device="cuda")
    b.store_from_padded(rows, table_ids, vals)

    mx = b.max_emb_dim
    got = b.load_to_padded(rows, table_ids, CopyMode.VALUE)
    assert tuple(got.shape) == (5, b.max_value_dim)
    for i, t in enumerate(table_ids.tolist()):
        e, o = b.emb_dim(t), b.value_dim(t) - b.emb_dim(t)
        torch.testing.assert_close(got[i, :e], vals[i, :e])
        torch.testing.assert_close(got[i, mx : mx + o], vals[i, mx : mx + o])
        # and it is in that table's own buffer, at that row, without the padding
        row = b.tensor(t)[rows[i].item()]
        torch.testing.assert_close(row[:e], vals[i, :e])
        torch.testing.assert_close(row[e:], vals[i, mx : mx + o])

    emb = b.load_to_padded(rows, table_ids, CopyMode.EMBEDDING)
    assert tuple(emb.shape) == (5, b.max_emb_dim)
    for i, t in enumerate(table_ids.tolist()):
        e = b.emb_dim(t)
        torch.testing.assert_close(emb[i, :e], vals[i, :e])


@pytest.mark.parametrize(
    "emb_dims,value_dims", [ALIGNED, UNALIGNED], ids=["vec4", "scalar"]
)
def test_a_padded_store_writes_only_the_columns_a_table_uses(emb_dims, value_dims):
    # Fill every buffer with a sentinel, store padded rows (as wide as the widest table)
    # and check that nothing but the stored rows of the stored tables changed: a table
    # narrower than the widest must not spill into the next row or into another table.
    b = make(emb_dims=emb_dims, value_dims=value_dims)
    for t in range(b.num_tables):
        b.tensor(t).fill_(-7.0)
    rows = idx([5, 6])
    table_ids = idx([0, 0])  # table 0 is the narrower one
    vals = torch.randn(2, b.max_value_dim, device="cuda")
    b.store_from_padded(rows, table_ids, vals)

    expected0 = torch.full_like(b.tensor(0), -7.0)
    mx = b.max_emb_dim
    e, o = b.emb_dim(0), b.value_dim(0) - b.emb_dim(0)
    for i, r in enumerate(rows.tolist()):
        expected0[r, :e] = vals[i, :e]
        expected0[r, e:] = vals[i, mx : mx + o]
    torch.testing.assert_close(b.tensor(0), expected0)
    assert (b.tensor(1) == -7.0).all()  # the other table is untouched


def test_loading_and_storing_whole_rows_of_one_table():
    b = make()
    rows = idx([3, 4])
    vals = torch.randn(2, 8, device="cuda")  # table 1 is 8 wide
    b.store(1, rows, vals)
    torch.testing.assert_close(b.load(1, rows), vals)
    torch.testing.assert_close(b.tensor(1)[rows], vals)
    # nothing to do, nothing to fail
    assert tuple(b.load(0, rows[:0]).shape) == (0, 12)
    b.store(0, rows[:0], torch.empty(0, 12, device="cuda"))
    assert tuple(b.load_to_padded(rows[:0], rows[:0], CopyMode.VALUE).shape) == (0, 12)


def test_a_one_dimensional_value_is_a_column():
    b = make(emb_dims=(1, 1), value_dims=(1, 1))
    rows = idx([0, 1])
    b.store_from_padded(rows, idx([0, 1]), torch.tensor([2.0, 3.0], device="cuda"))
    got = b.load_to_padded(rows, idx([0, 1]), CopyMode.VALUE)
    torch.testing.assert_close(got[:, 0], torch.tensor([2.0, 3.0], device="cuda"))


@pytest.mark.parametrize("kind", ["device", "host"])
def test_an_extendable_buffer_grows_and_keeps_its_rows(kind):
    b = make(kind)
    assert b.kind == kind and b.on_device == (kind == "device")
    rows = idx([0, 1])
    vals = torch.randn(2, 8, device="cuda")
    b.store(1, rows, vals)
    other_rows = idx([2, 3])  # table 0 must come through a growth of table 1 intact
    other = torch.randn(2, 12, device="cuda")
    b.store(0, other_rows, other)

    before = b.flat_tables_view()  # taken before the growth
    assert not b.grow_to(1, 10)  # it has 48 already
    assert b.grow_to(1, 100)
    assert b.num_rows(1) >= 100
    # the pointers the kernels use follow the buffers, in a view taken earlier too
    assert b.ptrs.tolist() == [b.tensor(t).data_ptr() for t in range(2)]
    assert before.ptrs.tolist() == b.ptrs.tolist()
    assert b.flat_tables_view().ptrs.tolist() == b.ptrs.tolist()
    torch.testing.assert_close(b.load(1, rows), vals)
    torch.testing.assert_close(b.load(0, other_rows), other)
    assert b.num_rows(0) == 32
    # and the grown part is addressable
    b.store(1, idx([99]), torch.ones(1, 8, device="cuda"))
    torch.testing.assert_close(b.load(1, idx([99])), torch.ones(1, 8, device="cuda"))


def test_a_fixed_pool_does_not_grow():
    b = make("fixed")
    assert not b.grow_to(0, 10)  # already big enough: nothing to do
    with pytest.raises(NotImplementedError):
        b.grow_to(0, 100)
    assert b.num_rows(0) == 32


def test_indices_on_the_host_are_accepted():
    b = make()
    rows = torch.tensor([1, 2], dtype=torch.int32)  # a key map's, on the host
    vals = torch.randn(2, 8, device="cuda")
    b.store(1, rows, vals)
    torch.testing.assert_close(b.load(1, rows), vals)
    got = b.load_to_padded(
        rows, torch.tensor([1, 1], device="cuda"), CopyMode.EMBEDDING
    )
    torch.testing.assert_close(got[:, :8], vals)


def test_an_fp8_buffer_stores_encoded_values():
    from dynamicemb.utils import FP8_STORAGE_DTYPE, decode_table_values

    b = MixedDimBuffers(
        (4,),
        (4,),
        (8,),
        FP8_STORAGE_DTYPE,
        torch.device("cuda", torch.cuda.current_device()),
        "fixed",
    )
    rows = idx([0, 3])
    vals = torch.tensor(
        [[0.5, -0.25, 0.125, 1.0]] * 2, dtype=torch.float16, device="cuda"
    )
    b.store(0, rows, vals)
    # the buffer holds codes of value * scale, not the plain cast
    torch.testing.assert_close(
        decode_table_values(b.load(0, rows), torch.float16), vals
    )
    b.store_from_padded(rows, idx([0, 0]), vals)
    torch.testing.assert_close(
        decode_table_values(b.load(0, rows), torch.float16), vals
    )


def test_gather_emb_by_entry_gathers_and_skips_negative_rows():
    b = make(emb_dims=(4, 8), value_dims=(12, 24))
    vals = torch.randn(3, b.max_value_dim, device="cuda")
    b.store_from_padded(
        idx([0, 1, 2]), idx([0, 1, 1]), vals
    )  # table 0 row 0; table 1 rows 1, 2

    # entries: 0 -> (row 0, table 0), 1 -> (row -1, missing), 2 -> (row 2, table 1)
    rows, table_ids = idx([0, -1, 2]), idx([0, 1, 1])
    # output row i takes entry entry_of_row[i]
    entry = idx([2, 0, 1])
    out = torch.full((3, b.max_emb_dim), -7.0, device="cuda")
    b.gather_emb_by_entry(rows, table_ids, entry, out)

    torch.testing.assert_close(out[0], vals[2, :8])  # row 2 of table 1, 8 wide
    torch.testing.assert_close(out[1, :4], vals[0, :4])  # row 0 of table 0, 4 wide
    assert (out[1, 4:] == -7.0).all()  # past its width: untouched
    assert (out[2] == -7.0).all()  # the missing row: left as it was


def _padded_view(emb_dims, max_value_dim, device="cuda"):
    return PaddedBufferView(
        torch.tensor(emb_dims, dtype=torch.int64, device=device),
        max(emb_dims),
        max_value_dim,
        all(d % 4 == 0 for d in emb_dims),
    )


def test_padded_buffer_finds_the_parts_of_a_row():
    view = _padded_view([4, 8], 12)
    rows = PaddedBuffer.empty(view, 3, torch.float16, torch.device("cuda"))
    assert (
        rows.has_state
        and not PaddedBuffer.empty(
            _padded_view([8], 8), 1, torch.float16, torch.device("cuda")
        ).has_state
    )
    assert rows.values.shape == (3, 12) and rows.values.dtype == torch.float16
    assert rows.max_emb_dim == 8 and rows.max_value_dim == 12

    rows.values.zero_()
    rows.emb().fill_(1)
    rows.state().fill_(2)
    # Both are views of the rows, not copies, and they split a row at max_emb_dim.
    assert rows.values[:, :8].eq(1).all() and rows.values[:, 8:].eq(2).all()
    assert rows.emb().shape == (3, 8) and rows.state().shape == (3, 4)


def test_padded_buffer_rejects_values_of_the_wrong_width():
    with pytest.raises(ValueError):
        PaddedBuffer(torch.zeros(3, 10, device="cuda"), _padded_view([4, 8], 12))
