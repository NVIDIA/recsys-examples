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

"""Regression tests for the contiguous *padded buffer* optimizer kernels.

The padded buffer (produced by ``load_from_flat`` with ``CopyMode.VALUE``) lays
each row out as::

    [ emb (edim) | pad ... | m (edim) | v (edim) | pad ... ]
      ^0                     ^max_emb_dim

i.e. the embedding occupies ``[0, edim)`` while the optimizer state block always
starts at ``max_emb_dim`` regardless of the row's own ``edim``. Before the fix
for https://github.com/NVIDIA/recsys-examples/issues/419 the Adam kernel assumed
each state occupied the full ``max_emb_dim`` width (m at ``max_emb_dim``, v at
``2*max_emb_dim``). For a table with ``edim < max_emb_dim`` this made the kernel
write the velocity slot with the momentum (beta1) formula, corrupting v.

A second corruption path is covered too: the vectorized (vec4) kernel writes m
in 4-element chunks, so when a table's ``edim`` is not a multiple of 4 the final
m store would spill into the start of v with the beta1 rule. ``all_dims_vec4``
must therefore reflect *per-table* dims (as ``state.all_dims_vec4`` does), not
just ``max_emb_dim``; a misaligned table must fall back to the scalar kernel.

FTRL has the same shape of exposure: it keeps ``linear`` then ``accum``, each
the row's own ``edim`` wide, so a narrow table's ``accum`` begins at
``max_emb_dim + edim`` rather than at ``2 * max_emb_dim``.

These tests build the buffer by hand for two tables of different dims and check
that the state slots land in the right place with the right update rule. The last
ones run the ops that fill, move and update such buffers and tables -- flat load
and store, initializers, optimizer updates, ``reduce_grads``, the state reset --
under a device count (``num_valid``), as the lookup step does. They require a GPU
+ the compiled ``dynamicemb_extensions`` and are skipped otherwise.
"""

from functools import partial

import pytest
import torch
from dynamicemb import DynamicEmbInitializerArgs, DynamicEmbInitializerMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.optimizer import (
    AdaGradDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizer,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")


def _all_dims_vec4(dims):
    """Per-table vec4 alignment, mirroring create_table_state for Adam.

    Adam value dim per table is ``3 * edim`` (emb + m + v), so vec4 is safe only
    when every embedding dim (and hence value dim) is a multiple of 4.
    """
    value_dims = [3 * d for d in dims]
    return all(d % 4 == 0 for d in dims) and all(v % 4 == 0 for v in value_dims)


def _build_padded_adam_buffer(dims, max_emb_dim, m_init, v_init, dtype, device):
    """Layout one row per table: emb@0, m@max_emb_dim, v@max_emb_dim+edim.

    Returns ``(values, table_ids, table_emb_dims, value_dim)``.
    """
    # Value row stride = emb(max_emb_dim) + m(max_emb_dim) + v(max_emb_dim).
    # Each slot is reserved max_emb_dim wide, but only the row's own edim values
    # are live -- the kernel reads/writes edim elements per slot (see layout above).
    value_dim = 3 * max_emb_dim
    n = len(dims)
    values = torch.zeros(n, value_dim, dtype=dtype, device=device)
    for row, edim in enumerate(dims):
        values[row, max_emb_dim : max_emb_dim + edim] = m_init
        values[row, max_emb_dim + edim : max_emb_dim + 2 * edim] = v_init
    table_ids = torch.arange(n, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor(dims, dtype=torch.int64, device=device)
    return values, table_ids, table_emb_dims, value_dim


@cuda
@pytest.mark.parametrize(
    "dims",
    [
        [8, 4],  # all dims multiple of 4 -> vec4 path
        [8, 6],  # second table not vec4-aligned -> scalar path
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_adam_padded_buffer_mixed_dims_velocity(dims, dtype):
    """v of the smaller table must use beta2, not beta1 (issue #419).

    Parametrized over both an all-vec4-aligned mix and a mix where the smaller
    table is not 4-aligned, exercising both the vec4 and the scalar kernels.
    """
    device = torch.device("cuda")
    max_emb_dim = max(dims)

    lr = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.0
    iter_num = 1

    m_init = 2.0
    v_init = 3.0
    # grad = 0 isolates the decay term so beta1 vs beta2 is unambiguous.
    m_expected = beta1 * m_init  # 1.8
    v_expected = beta2 * v_init  # 2.997 (the bug produced beta1 * v_init = 2.7)

    values, table_ids, table_emb_dims, value_dim = _build_padded_adam_buffer(
        dims, max_emb_dim, m_init, v_init, dtype, device
    )
    grads = torch.zeros(len(dims), max_emb_dim, dtype=dtype, device=device)

    dynamicemb_extensions.adam_update_for_padded_buffer(
        grads,
        values,
        table_ids,
        table_emb_dims,
        max_emb_dim,
        value_dim,
        _all_dims_vec4(dims),
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        iter_num,
    )

    for row, edim in enumerate(dims):
        m = values[row, max_emb_dim : max_emb_dim + edim]
        v = values[row, max_emb_dim + edim : max_emb_dim + 2 * edim]
        torch.testing.assert_close(
            m,
            torch.full_like(m, m_expected),
            msg=f"m wrong for table {row} (edim={edim})",
        )
        torch.testing.assert_close(
            v,
            torch.full_like(v, v_expected),
            msg=f"v wrong for table {row} (edim={edim})",
        )


@cuda
@pytest.mark.parametrize("dims", [[8, 4], [8, 6]])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_adam_padded_buffer_matches_reference_with_grad(dims, dtype):
    """Full Adam step (nonzero grad) per row matches a torch reference."""
    device = torch.device("cuda")
    max_emb_dim = max(dims)

    lr, beta1, beta2, eps, weight_decay, iter_num = 0.1, 0.9, 0.999, 1e-8, 0.01, 3

    m_init, v_init, w_init, g_val = 0.5, 0.25, 1.0, 0.7

    value_dim = 3 * max_emb_dim  # emb + m + v, each slot max_emb_dim wide
    n = len(dims)
    values = torch.zeros(n, value_dim, dtype=dtype, device=device)
    grads = torch.zeros(n, max_emb_dim, dtype=dtype, device=device)
    for row, edim in enumerate(dims):
        values[row, :edim] = w_init
        values[row, max_emb_dim : max_emb_dim + edim] = m_init
        values[row, max_emb_dim + edim : max_emb_dim + 2 * edim] = v_init
        grads[row, :edim] = g_val
    table_ids = torch.arange(n, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor(dims, dtype=torch.int64, device=device)

    # torch reference for one row element.
    m_ref = beta1 * m_init + (1 - beta1) * g_val
    v_ref = beta2 * v_init + (1 - beta2) * g_val * g_val
    mhat = m_ref / (1 - beta1**iter_num)
    vhat = v_ref / (1 - beta2**iter_num)
    w_ref = w_init - lr * (mhat / (vhat**0.5 + eps) + weight_decay * w_init)

    dynamicemb_extensions.adam_update_for_padded_buffer(
        grads,
        values,
        table_ids,
        table_emb_dims,
        max_emb_dim,
        value_dim,
        _all_dims_vec4(dims),
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        iter_num,
    )

    for row, edim in enumerate(dims):
        w = values[row, :edim]
        m = values[row, max_emb_dim : max_emb_dim + edim]
        v = values[row, max_emb_dim + edim : max_emb_dim + 2 * edim]
        torch.testing.assert_close(w, torch.full_like(w, w_ref))
        torch.testing.assert_close(m, torch.full_like(m, m_ref))
        torch.testing.assert_close(v, torch.full_like(v, v_ref))


def _build_padded_ftrl_buffer(dims, max_emb_dim, weight, linear, accum, dtype, device):
    """Layout one row per table: emb@0, linear@max_emb_dim, accum@max_emb_dim+edim."""
    value_dim = 3 * max_emb_dim
    n = len(dims)
    values = torch.zeros(n, value_dim, dtype=dtype, device=device)
    for row, edim in enumerate(dims):
        values[row, :edim] = weight
        values[row, max_emb_dim : max_emb_dim + edim] = linear
        values[row, max_emb_dim + edim : max_emb_dim + 2 * edim] = accum
    table_ids = torch.arange(n, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor(dims, dtype=torch.int64, device=device)
    return values, table_ids, table_emb_dims, value_dim


@cuda
@pytest.mark.parametrize(
    "dims",
    [
        [8, 4],  # all dims multiple of 4 -> vec4 path
        [8, 6],  # second table not vec4-aligned -> scalar path
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_ftrl_padded_buffer_mixed_dims_accum(dims, dtype):
    """A narrow table's accum sits at max_emb_dim + edim, not at 2 * max_emb_dim.

    Had the kernel reserved max_emb_dim per slot instead, the narrow table's
    accum pointer would land in the padding past its state: it would read a
    zero, advance that, write it back out of bounds of the live row, and leave
    the real accum untouched at its seed. The assertions below catch all three
    -- accum unadvanced, linear computed from the wrong accum, and padding
    written.
    """
    device = torch.device("cuda")
    max_emb_dim = max(dims)

    lr = 0.1
    learning_rate_power = -0.5
    ftrl_beta = 0.0
    l1_reg = 0.0
    l2_reg = 0.0

    weight_init = 0.5
    linear_init = -1.0
    accum_init = 4.0
    grad = 1.0

    # accum is the only state the gradient enters quadratically, so advancing it
    # by exactly g^2 pins that the kernel found the right slot.
    accum_expected = accum_init + grad * grad
    sigma = (accum_expected**0.5 - accum_init**0.5) / lr
    linear_expected = linear_init + grad - sigma * weight_init
    weight_expected = -linear_expected / ((ftrl_beta + accum_expected**0.5) / lr)

    values, table_ids, table_emb_dims, value_dim = _build_padded_ftrl_buffer(
        dims, max_emb_dim, weight_init, linear_init, accum_init, dtype, device
    )
    before = values.clone()
    grads = torch.full((len(dims), max_emb_dim), grad, dtype=dtype, device=device)

    dynamicemb_extensions.ftrl_update_for_padded_buffer(
        grads,
        values,
        table_ids,
        table_emb_dims,
        max_emb_dim,
        value_dim,
        _all_dims_vec4(dims),
        lr,
        learning_rate_power,
        ftrl_beta,
        l1_reg,
        l2_reg,
        0.0,  # group_lasso_reg: plain FTRL
    )

    for row, edim in enumerate(dims):
        weight = values[row, :edim]
        linear = values[row, max_emb_dim : max_emb_dim + edim]
        accum = values[row, max_emb_dim + edim : max_emb_dim + 2 * edim]
        torch.testing.assert_close(
            accum,
            torch.full_like(accum, accum_expected),
            msg=f"accum wrong for table {row} (edim={edim})",
        )
        torch.testing.assert_close(
            linear,
            torch.full_like(linear, linear_expected),
            msg=f"linear wrong for table {row} (edim={edim})",
        )
        torch.testing.assert_close(
            weight,
            torch.full_like(weight, weight_expected),
            msg=f"weight wrong for table {row} (edim={edim})",
        )
        # Everything past this row's own state is reserved for wider tables and
        # must be left alone.
        tail = slice(max_emb_dim + 2 * edim, value_dim)
        torch.testing.assert_close(
            values[row, tail],
            before[row, tail],
            msg=f"padding past the state was written for table {row}",
        )


# -- Under a device count ----------------------------------------------------
# The ops of the lookup step run on capacity-CAP arrays of which only the first n
# entries are live, n being the one-element int64 CUDA tensor ``num_valid``. Past n
# the inputs are garbage, half plausible, so that a kernel that wrongly reads them
# does visible work, half far out of range, so that it also faults. The reference is
# the op on the exact slice [:n] without ``num_valid``; what lies past n must be as
# it was.

DEV = "cuda"
CAP = 37  # odd, so that CAP // 2 and CAP - 1 line up with nothing
COUNTS = [0, CAP // 2, CAP]
SENTINEL = 7.0  # what an output starts as
TABLE_ROWS = 64  # rows of a flat table
DIMS = [[8, 4], [6, 5]]  # embedding dims of two tables: the vec4 and the scalar kernels
BUF_ROWS = 41  # rows of the buffers that the initializers and the state resets write
LEFT, WIDTH = 3, 8  # columns of such a buffer left of the part written, and its width


def nv(n):
    """The count, as a view of a longer tensor."""
    return torch.tensor([-7, n, 99], device=DEV)[1:2]


def i64(values):
    return torch.tensor(values, dtype=torch.int64, device=DEV)


def rand_ids(high, size=CAP):
    return torch.randint(0, high, (size,), device=DEV)


def junk_tail(live, n, plausible):
    """``live`` with everything past its first n entries replaced by garbage."""
    wild = torch.randint(-(1 << 40), 1 << 40, live.shape, device=DEV)
    junk = torch.where(torch.rand(live.shape, device=DEV) < 0.5, plausible, wild)
    return torch.cat([live[:n], junk[n:]])


def nan_tail(x, n):
    x = x.clone()
    x[n:] = float("nan")
    return x


# -- flat table load / store ---------------------------------------------------
def flat_tables(dims, state_dim, rows=TABLE_ROWS, vec4=None):
    """One random value table per embedding dim, each row ``emb | state_dim(emb)``
    wide: their ``ptrs``, and the ``args`` that the flat kernels take after their
    per-entry arguments (``all_dims_vec4`` as the dims allow, unless ``vec4``)."""
    tables = [torch.rand(rows, d + state_dim(d), device=DEV) for d in dims]
    vec4 = _all_dims_vec4(dims) if vec4 is None else vec4
    args = (i64([t.size(1) for t in tables]), i64(dims), max(dims), vec4)
    return {"tables": tables, "ptrs": i64([t.data_ptr() for t in tables]), "args": args}


def clone_tables(t):
    tables = [x.clone() for x in t["tables"]]
    return {**t, "tables": tables, "ptrs": i64([x.data_ptr() for x in tables])}


def same_tables(a, b):
    return all(torch.equal(x, y) for x, y in zip(a["tables"], b["tables"]))


def entries(n, tables):
    """The (row, table id) of CAP entries of a flat table, the first n live. The live
    rows are distinct, so that no two stores race."""
    rows = torch.randperm(TABLE_ROWS, device=DEV)[:CAP]
    return (
        junk_tail(rows, n, rand_ids(TABLE_ROWS)),
        junk_tail(rand_ids(tables), n, rand_ids(tables)),
    )


@cuda
@pytest.mark.parametrize("kind", ["emb", "value"])
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("n", COUNTS)
def test_load_from_flat_table(kind, dims, n):
    t = flat_tables(dims, lambda d: 2 * d)
    width = max(dims) * {"emb": 1, "value": 3}[kind]
    load = getattr(dynamicemb_extensions, f"load_from_flat_table_{kind}")
    indices, tids = entries(n, len(dims))
    out = torch.full((CAP, width), SENTINEL, device=DEV)
    ref = out.clone()
    load(t["ptrs"], indices, tids, out, *t["args"], num_valid=nv(n))
    load(t["ptrs"], indices[:n], tids[:n], ref[:n], *t["args"])
    assert torch.equal(out, ref)


@cuda
@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("n", COUNTS)
def test_store_to_flat_table(dims, n):
    a = flat_tables(dims, lambda d: 2 * d)
    b = clone_tables(a)
    store = dynamicemb_extensions.store_to_flat_table_value
    indices, tids = entries(n, len(dims))
    src = torch.randn(CAP, 3 * max(dims), device=DEV)
    store(a["ptrs"], indices, tids, nan_tail(src, n), *a["args"], num_valid=nv(n))
    store(b["ptrs"], indices[:n], tids[:n], src[:n], *b["args"])
    assert same_tables(a, b)


@cuda
@pytest.mark.parametrize("dim", [8, 6])
@pytest.mark.parametrize("n", COUNTS)
def test_select_insert_failed_values(dim, n):
    k = 11
    values = torch.randn(k, dim, device=DEV)
    # >= 0: not a failed insert, skipped; < 0: -1 - (row of `values`).
    failed = torch.rand(CAP, device=DEV) < 0.5
    live = torch.where(failed, -1 - rand_ids(k), rand_ids(999))
    indices = junk_tail(live, n, -1 - rand_ids(k))
    ref_indices = indices.clone()
    select = dynamicemb_extensions.select_insert_failed_values
    out = torch.full((CAP, dim), SENTINEL, device=DEV)
    ref_out = out.clone()
    select(indices, values, out, num_valid=nv(n))
    select(ref_indices[:n], values, ref_out[:n])
    # The op marks what it served by writing -1 into `indices`.
    assert torch.equal(out, ref_out) and torch.equal(indices, ref_indices)


# -- initializers --------------------------------------------------------------
def buffer_case(n, with_indices, dtype=torch.float32):
    """A sentinel buffer, the part of it that is written, the indices (None: rows
    0..n-1; every fifth one is -1, which skips) and the rows that must change."""
    base = torch.full((BUF_ROWS, LEFT + WIDTH), SENTINEL, dtype=dtype, device=DEV)
    if not with_indices:
        return base, base[:, LEFT:], None, torch.arange(n, device=DEV)
    rows = torch.randperm(BUF_ROWS, device=DEV)[:CAP]
    rows[::5] = -1
    indices = junk_tail(rows, n, rand_ids(BUF_ROWS))
    live = rows[:n]
    return base, base[:, LEFT:], indices, live[live >= 0]


@cuda
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("tables,n", [(1, 0), (1, CAP // 2), (1, CAP), (3, CAP // 2)])
def test_const_initializer(tables, n, with_indices):
    """One table runs const_init; three, with a value each, const_init_table_params,
    whose table ids run alongside the rows of the buffer, not the indices."""
    base, buf, indices, live = buffer_case(n, with_indices)
    values = [1.0, 2.0, 3.0][:tables]
    args = [
        DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.CONSTANT, value=v)
        for v in values
    ]
    table_ids = rand_ids(tables, BUF_ROWS)
    want = base.clone()
    want[live, LEFT:] = torch.tensor(values, device=DEV)[table_ids[live]][:, None]
    init = MultiTableInitializer.create(args, torch.device(DEV))
    init(buf, None, table_ids, indices, nv(n))
    assert torch.equal(base, want)


@cuda
@pytest.mark.parametrize("with_indices", [True, False])
def test_random_initializer(with_indices):
    """Which entries it wrote, not what: the live rows of the part written, no more."""
    n = CAP // 2
    base, buf, indices, live = buffer_case(n, with_indices)
    args = DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.NORMAL)
    init = MultiTableInitializer.create([args], torch.device(DEV))
    init(buf, None, None, indices, nv(n))
    written = torch.zeros_like(base, dtype=torch.bool)
    written[live, LEFT:] = True
    assert torch.equal(base != SENTINEL, written)


@cuda
def test_initializer_without_indices_needs_num_valid():
    buf = torch.zeros(BUF_ROWS, WIDTH, device=DEV)
    with pytest.raises(Exception, match="num_valid"):
        dynamicemb_extensions.const_init(buf, None, 1.0)


# -- optimizer updates ---------------------------------------------------------
_args = partial(OptimizerArgs, learning_rate=0.1)
# name: the optimizer for a table of the given dtype, which only row-wise Adagrad uses
OPTIMIZERS = {
    "sgd": lambda dt: SGDDynamicEmbeddingOptimizer(_args()),
    "adam": lambda dt: AdamDynamicEmbeddingOptimizer(_args()),
    "adagrad": lambda dt: AdaGradDynamicEmbeddingOptimizer(_args()),
    "rowwise_adagrad": lambda dt: RowWiseAdaGradDynamicEmbeddingOptimizer(_args(), dt),
    "ftrl": lambda dt: FTRLDynamicEmbeddingOptimizer(_args(l1_reg=0.01, l2_reg=0.01)),
    # a row is a group: a block reduces its norm in shared memory
    "ftrl_group": lambda dt: FTRLDynamicEmbeddingOptimizer(
        _args(l1_reg=0.01, l2_reg=0.01, group_lasso_reg=0.05)
    ),
}
# The kernels take the optimizer as a template argument, so the count logic is one
# piece of code: Adam at every count in both kernels, each other optimizer once.
UPDATE_CASES = [
    ("adam", dims, n) for dims in DIMS for n in [0, 1, CAP // 2, CAP - 1, CAP]
]
UPDATE_CASES += [(o, DIMS[0], CAP // 2) for o in OPTIMIZERS if o != "adam"]


def make_optimizer(name, dtype=torch.float32):
    opt = OPTIMIZERS[name](dtype)
    opt.step()  # Adam's bias correction needs a first iteration
    return opt


def flat_update(opt, t, g, idx, ids, **kw):
    opt.fused_update_for_flat_table(
        g, idx, t["ptrs"], ids, *t["args"], torch.float32, **kw
    )


@cuda
@pytest.mark.parametrize("name,dims,n", UPDATE_CASES)
def test_flat_table_update(name, dims, n):
    opt = make_optimizer(name)
    a = flat_tables(dims, opt.get_state_dim)
    b = clone_tables(a)
    indices, tids = entries(n, len(dims))
    grads = torch.randn(CAP, max(dims), device=DEV)
    flat_update(opt, a, nan_tail(grads, n), indices, tids, num_valid=nv(n))
    flat_update(opt, b, grads[:n], indices[:n], tids[:n])
    assert same_tables(a, b)


@cuda
@pytest.mark.parametrize("name,dims,n", UPDATE_CASES)
def test_padded_buffer_update(name, dims, n):
    opt = make_optimizer(name)
    me = max(dims)
    vd = me + opt.get_state_dim(me)
    a = torch.rand(CAP, vd, device=DEV)
    b = a.clone()
    tids = junk_tail(rand_ids(len(dims)), n, rand_ids(len(dims)))
    grads = torch.randn(CAP, me, device=DEV)
    rest = (i64(dims), me, vd, _all_dims_vec4(dims))
    opt.update_for_padded_buffer(nan_tail(grads, n), a, tids, *rest, num_valid=nv(n))
    if n:
        opt.update_for_padded_buffer(grads[:n], b[:n], tids[:n], *rest)
    assert torch.equal(a, b)


@cuda
@pytest.mark.parametrize("name", ["rowwise_adagrad", "ftrl_group"])
@pytest.mark.parametrize("flat", [False, True], ids=["padded", "flat"])
def test_blocks_with_several_rows(name, flat):
    """Counted, the grid is capped and a block works through several rows in turn,
    reusing its shared buffer of block-wide reductions: the result of one block per
    row."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows, dim = 4 * sms * 32 * 4, 96  # more rows than the cap of any block size
    opt = make_optimizer(name)
    grads = torch.randn(rows, dim, device=DEV)
    tids = torch.zeros(rows, dtype=torch.int64, device=DEV)
    if flat:
        a = flat_tables([dim], opt.get_state_dim, rows, vec4=False)
        b = clone_tables(a)
        idx = torch.randperm(rows, device=DEV)  # distinct: no two entries race
        flat_update(opt, a, grads, idx, tids, num_valid=nv(rows))
        flat_update(opt, b, grads, idx, tids)
        assert same_tables(a, b)
        return
    vd = dim + opt.get_state_dim(dim)
    a = torch.rand(rows, vd, device=DEV)
    b = a.clone()
    rest = (tids, i64([dim]), dim, vd, False)  # False: the scalar kernel
    opt.update_for_padded_buffer(grads, a, *rest, num_valid=nv(rows))
    opt.update_for_padded_buffer(grads, b, *rest)
    assert torch.equal(a, b)


# -- reduce_grads --------------------------------------------------------------
@cuda
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("dim", [8, 6])
def test_reduce_grads_with_spare_rows(dim, pooled):
    """The step asks for as many unique rows as it has keys: the first ones are the
    answer of the exact call."""
    reduce_grads = dynamicemb_extensions.reduce_grads
    keys, batch, kw = CAP, CAP, {}
    grads = torch.randn(keys, dim, device=DEV)
    if pooled:  # two features of five bags, feature-major
        sizes = i64([3, 0, 2, 1, 4, 2, 0, 3, 1, 2])
        offsets = torch.cat([i64([0]), sizes.cumsum(0)])
        keys, batch = int(offsets[-1]), 5
        grads = torch.randn(batch, 2 * dim, device=DEV)
        mode = dynamicemb_extensions.PoolingMode.KSum
        kw = dict(offsets=offsets, pooling_mode=mode, total_D=2 * dim)
    unique = keys // 2
    # An inverse index in which every id < unique occurs.
    inv = (torch.arange(keys, device=DEV) % unique)[torch.randperm(keys, device=DEV)]
    got = reduce_grads(inv, grads, keys, batch, dim, **kw)
    assert got.shape == (keys, dim)
    want = reduce_grads(inv, grads, unique, batch, dim, **kw)
    torch.testing.assert_close(got[:unique], want, rtol=1e-5, atol=1e-5)


# -- state reset ---------------------------------------------------------------
EMB_DIMS = [4, 3, 2]  # embedding dims of three FTRL tables; 2 * max(EMB_DIMS) <= WIDTH
# (optimizer, dtype of the buffer, n): each of the three fills -- plain, FTRL's split
# of a row by its table, row-wise Adagrad's fp32 over pairs of 16-bit words -- in the
# dtypes it sees, and n = 0 once.
RESET_CASES = [
    ("adagrad", torch.float32, CAP // 2),
    ("adagrad", torch.float16, CAP // 2),
    ("adagrad", torch.bfloat16, CAP // 2),
    ("rowwise_adagrad", torch.float16, CAP // 2),
    ("rowwise_adagrad", torch.bfloat16, CAP // 2),
    ("ftrl", torch.float32, CAP // 2),
    ("ftrl", torch.float16, CAP // 2),
    ("ftrl", torch.float32, 0),
]


@cuda
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("name,dtype,n", RESET_CASES)
def test_reset_optimizer_states_counted(name, dtype, n, with_indices):
    fill = 0.25
    opt = make_optimizer(name, dtype)
    opt.set_initial_optimizer_state(fill)
    base, dst, indices, live = buffer_case(n, with_indices, dtype)
    table_ids = rand_ids(len(EMB_DIMS), BUF_ROWS)  # one per row of the buffer
    want = base.clone()
    for row, tid in zip(live.tolist(), table_ids[live].tolist()):
        e = EMB_DIMS[tid]
        if name == "ftrl":  # `linear` is 0, `accum` the value, the rest 0
            want[row, LEFT:] = 0
            want[row, LEFT + e : LEFT + 2 * e] = fill
        elif name == "rowwise_adagrad":  # an fp32 over pairs of 16-bit words
            want[row, LEFT:] = torch.full((WIDTH // 2,), fill, device=DEV).view(dtype)
        else:
            want[row, LEFT:] = fill
    opt.reset_optimizer_states(
        dst,
        indices=indices,
        num_valid=nv(n),
        table_ids=table_ids,
        table_emb_dims=i64(EMB_DIMS),
    )
    assert torch.equal(base.view(torch.uint8), want.view(torch.uint8))


@cuda
@pytest.mark.parametrize("name", ["adagrad", "rowwise_adagrad", "ftrl"])
def test_reset_optimizer_states_indices_need_num_valid(name):
    opt = make_optimizer(name)
    opt.set_initial_optimizer_state(0.25)
    buf = torch.zeros(BUF_ROWS, WIDTH, device=DEV)
    with pytest.raises(ValueError, match="num_valid"):
        opt.reset_optimizer_states(buf, indices=torch.arange(4, device=DEV))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
