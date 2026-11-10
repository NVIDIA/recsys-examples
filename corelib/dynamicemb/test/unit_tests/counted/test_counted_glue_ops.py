# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The glue ops of the lookup path that keeps its counts on the device: compaction,
gather / scatter / scatter_fill, count_true and expand_table_ids, each with a device
``num_valid``.

A case runs an op on capacity-N arrays whose tail past n holds garbage (set flags,
random keys, out-of-range indices of both signs), then compares the first n results
with a torch reference on the exact slice [:n]. The op is also run on that exact slice
with ``num_valid=None``. Outputs the test pre-fills with a sentinel must be untouched
past n.
"""

import pytest
import torch
from dynamicemb_extensions import (
    counted_count_true,
    counted_gather,
    counted_scatter,
    counted_scatter_fill,
    expand_table_ids_cuda,
    flagged_compact,
    flagged_compact_counted,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

N = 1000
NS = [0, 1, N // 2, N - 1, N]
DTYPES = [
    torch.bool,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.uint64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
]
COLS = [None, 1, 3, 8, 40, 1100]  # None: 1-D; else rows of that many elements
FILL = {  # a value per dtype that survives the round trip through a double
    torch.bool: True,
    torch.int8: -3,
    torch.int32: -1,
    torch.int64: -1,
    torch.uint64: 7,
    torch.float16: 2.5,
    torch.bfloat16: -1.5,
    torch.float32: 0.125,
}
INT_OF_SIZE = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
HUGE = 1 << 40


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _nv(n, view=False):
    """The count on the device; with ``view`` a view into a longer tensor."""
    if view:
        return torch.tensor([-7, n, 99], dtype=torch.int64, device="cuda")[1:2]
    return torch.tensor([n], dtype=torch.int64, device="cuda")


def _bits(t):
    """``t`` as a signed integer tensor of its element size: exact to compare, and
    indexable for the dtypes torch has few ops for (uint64)."""
    return t.contiguous().view(INT_OF_SIZE[t.element_size()])


def _same(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rand(shape, dtype):
    x = torch.randint(0, 100, shape, dtype=torch.int64, device="cuda")
    if dtype == torch.uint64:
        return x.view(torch.uint64)
    return x % 2 == 1 if dtype == torch.bool else x.to(dtype)


def _sentinel(shape, dtype):
    """An output pre-filled with an all-ones bit pattern, which no test value has."""
    out = torch.empty(shape, dtype=dtype, device="cuda")
    out.view(INT_OF_SIZE[out.element_size()]).fill_(-1)
    return out


def _const(value, dtype):
    """One element holding ``value`` as ``dtype``."""
    if dtype == torch.uint64:
        t = torch.tensor([value], dtype=torch.int64, device="cuda")
        return t.view(torch.uint64)
    return torch.tensor([value], dtype=dtype, device="cuda")


def _garbage_idx(n, rows, unique=False):
    """idx[:n]: row ids in [0, rows) (distinct if ``unique``) or -1; idx[n:]: garbage,
    out-of-range ids of both signs."""
    if unique:
        idx = torch.randperm(rows, device="cuda")[:N]
    else:
        idx = torch.randint(0, rows, (N,), device="cuda")
    idx[torch.rand(N, device="cuda") < 0.2] = -1
    junk = torch.randint(rows, HUGE, (N - n,), device="cuda")
    junk[::2] *= -1
    idx[n:] = junk
    return idx


def _tail(cols):
    return () if cols is None else (cols,)


def _selected(idx, n):
    return (idx[:n] >= 0).nonzero().flatten()


# -- flagged_compact_counted ----------------------------------------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_flagged_compact_counted(n, k):
    flags = torch.rand(N, device="cuda") < 0.4
    flags[n:] = True  # a tail that a compaction without the bound would select
    arrays = [torch.randint(-HUGE, HUGE, (N,), device="cuda") for _ in range(k)]
    if k:
        arrays[0] = arrays[0].view(torch.uint64)
    if k > 2:
        arrays[1] = None  # stays None: 0, 1, 2 and 6 real inputs
    count, positions, outs = flagged_compact_counted(flags, arrays, _nv(n, n % 2 == 1))
    ref_count, ref_positions, ref_outs = flagged_compact(
        flags[:n], [None if a is None else a[:n] for a in arrays]
    )

    c = count.item()
    assert count.dtype == torch.int64 and count.shape == (1,) and count.is_cuda
    assert c == ref_count and positions.shape == (N,)
    assert torch.equal(positions[:c], ref_positions)
    assert len(outs) == k
    for a, out, ref in zip(arrays, outs, ref_outs):
        assert (out is None) == (a is None)
        if a is not None:
            assert out.dtype == a.dtype and out.shape == (N,)
            assert _same(out[:c], ref)


def test_flagged_compact_counted_count_forms():
    flags = torch.rand(N, device="cuda") < 0.4
    keys = torch.randint(0, HUGE, (N,), device="cuda")
    ref_count, ref_positions, (ref_keys,) = flagged_compact(flags, [keys])
    for nv in (None, _nv(N + 50)):  # no count, a count past the capacity: all live
        count, positions, (out,) = flagged_compact_counted(flags, [keys], nv)
        c = count.item()
        assert c == ref_count and torch.equal(positions[:c], ref_positions)
        assert torch.equal(out[:c], ref_keys)
    assert flagged_compact_counted(flags, [keys], _nv(-3))[0].item() == 0

    empty = flags[:0]
    count, positions, outs = flagged_compact_counted(empty, [keys[:0], None], _nv(0))
    assert count.item() == 0 and positions.numel() == 0
    assert outs[0].numel() == 0 and outs[1] is None
    with pytest.raises(RuntimeError):  # at most 6 real inputs
        flagged_compact_counted(flags, [keys] * 7, None)


# -- counted_count_true ---------------------------------------------------------------


@pytest.mark.parametrize("n", NS)
def test_counted_count_true(n):
    flags = torch.rand(N, device="cuda") < 0.4
    flags[n:] = True  # a tail that an unbounded count would include
    count = counted_count_true(flags, _nv(n, n % 2 == 1))
    assert count.dtype == torch.int64 and count.shape == (1,) and count.is_cuda
    expected = flags[:n].sum().item()
    assert count.item() == expected
    assert counted_count_true(flags[:n], None).item() == expected


def test_counted_count_true_count_forms():
    flags = torch.rand(N, device="cuda") < 0.4
    total = flags.sum().item()
    assert counted_count_true(flags, None).item() == total
    assert counted_count_true(flags, _nv(N + 50)).item() == total
    assert counted_count_true(flags, _nv(-5)).item() == 0
    assert counted_count_true(flags[:0], _nv(0)).item() == 0


# -- counted_gather / counted_scatter / counted_scatter_fill ---------------------------


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cols", COLS)
def test_counted_gather(n, dtype, cols):
    rows, shape = 600, (N,) + _tail(cols)
    src = _rand((rows,) + _tail(cols), dtype)
    idx = _garbage_idx(n, rows)
    sel = _selected(idx, n)
    ref = _sentinel(shape, dtype)
    _bits(ref)[sel] = _bits(src)[idx[sel]]

    out = _sentinel(shape, dtype)
    res = counted_gather(src, idx, _nv(n, n % 2 == 1), out)
    assert res.data_ptr() == out.data_ptr()
    untouched = _sentinel(shape, dtype)
    assert _same(out[:n], ref[:n]) and _same(out[n:], untouched[n:])
    exact = counted_gather(src, idx[:n], None, _sentinel((n,) + _tail(cols), dtype))
    assert _same(exact, ref[:n])
    alloc = counted_gather(src, idx, _nv(n), None)  # allocated: only rows written
    assert alloc.shape == ref.shape and alloc.dtype == dtype
    assert torch.equal(_bits(alloc)[sel], _bits(ref)[sel])


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cols", COLS)
def test_counted_scatter(n, dtype, cols):
    rows, shape = N + 20, (N + 20,) + _tail(cols)
    src = _rand((N,) + _tail(cols), dtype)
    idx = _garbage_idx(n, rows, unique=True)
    sel = _selected(idx, n)
    ref = _sentinel(shape, dtype)
    _bits(ref)[idx[sel]] = _bits(src)[sel]

    dst = _sentinel(shape, dtype)
    counted_scatter(dst, idx, src, _nv(n, n % 2 == 1))
    assert _same(dst, ref)  # the garbage tail wrote nothing
    dst = _sentinel(shape, dtype)
    counted_scatter(dst, idx[:n], src[:n], None)
    assert _same(dst, ref)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cols", COLS)
def test_counted_scatter_fill(n, dtype, cols):
    rows, shape, value = N + 20, (N + 20,) + _tail(cols), FILL[dtype]
    idx = _garbage_idx(n, rows, unique=True)
    sel = _selected(idx, n)
    ref = _sentinel(shape, dtype)
    _bits(ref)[idx[sel]] = _bits(_const(value, dtype))

    dst = _sentinel(shape, dtype)
    counted_scatter_fill(dst, idx, value, _nv(n, n % 2 == 1))
    assert _same(dst, ref)
    dst = _sentinel(shape, dtype)
    counted_scatter_fill(dst, idx[:n], value, None)
    assert _same(dst, ref)


def test_counted_scatter_fill_rejects_values_that_do_not_fit():
    idx = torch.arange(8, device="cuda")
    small = torch.zeros(8, dtype=torch.int8, device="cuda")
    unsigned = torch.zeros(8, dtype=torch.uint64, device="cuda")
    with pytest.raises(RuntimeError):
        counted_scatter_fill(small, idx, 300, None)
    with pytest.raises(RuntimeError):
        counted_scatter_fill(unsigned, idx, -1, None)
    big = torch.zeros(8, dtype=torch.int64, device="cuda")
    counted_scatter_fill(big, idx, -(1 << 62), None)  # exact in a double
    assert big.tolist() == [-(1 << 62)] * 8


def test_rows_of_strided_views():
    # Views of the middle columns of a wider tensor: a row pitch that is no multiple
    # of 16 bytes and a base 4 bytes in, so rows move in 4-byte words; the other
    # columns must stay as they were.
    n, c = N // 2, 6
    wide = torch.randn(700, c + 4, device="cuda")
    idx = _garbage_idx(n, 700)
    sel = _selected(idx, n)
    out = counted_gather(wide[:, 1 : 1 + c], idx, _nv(n), None)
    assert torch.equal(out[sel], wide[idx[sel], 1 : 1 + c])

    idx = _garbage_idx(n, N, unique=True)
    sel = _selected(idx, n)
    rows = torch.randn(N, c, device="cuda")
    dst = torch.full((N, c + 4), -1.0, device="cuda")
    ref = dst.clone()
    ref[idx[sel], 1 : 1 + c] = rows[sel]
    counted_scatter(dst[:, 1 : 1 + c], idx, rows, _nv(n))
    assert torch.equal(dst, ref)


# -- expand_table_ids_cuda ------------------------------------------------------------


@pytest.mark.parametrize("n", NS)
def test_expand_table_ids_counted(n):
    counts = torch.tensor([0, 130, 0, 400, 1, 469, 0], dtype=torch.int64, device="cuda")
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = counts.cumsum(0)  # N = 1000 elements in total
    ref = torch.repeat_interleave(torch.arange(counts.numel(), device="cuda"), counts)

    table_ids = expand_table_ids_cuda(offsets, N, num_valid=_nv(n, n % 2 == 1))
    assert table_ids.shape == (N,) and table_ids.dtype == torch.int64
    assert torch.equal(table_ids[:n], ref[:n])
    assert torch.equal(table_ids[:n], expand_table_ids_cuda(offsets, n))  # exact slice
    assert torch.equal(expand_table_ids_cuda(offsets, N), ref)  # no count: as before
    assert torch.equal(expand_table_ids_cuda(offsets, N, None), ref)


# -- no host read ---------------------------------------------------------------------


def test_ops_do_not_read_the_count():
    flags = torch.rand(N, device="cuda") < 0.5
    keys = torch.arange(N, device="cuda")
    rows = torch.zeros(N, 4, device="cuda")
    src = torch.ones(N, 4, device="cuda")
    offsets = torch.tensor([0, N // 3, N], dtype=torch.int64, device="cuda")
    nv = _nv(N // 2)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # a .item() / .cpu() of the count raises
    try:
        flagged_compact_counted(flags, [keys, None], nv)
        flagged_compact_counted(flags, [], nv)
        counted_count_true(flags, nv)
        counted_gather(rows, keys, nv, None)
        counted_scatter(rows, keys, src, nv)
        counted_scatter_fill(rows, keys, 1.0, nv)
        expand_table_ids_cuda(offsets, N, num_valid=nv)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
