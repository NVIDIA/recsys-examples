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

"""The value-side ops under a device counter (capacity N, ``num_valid = n``).

For every op family and every n in {0, 1, N // 2, N - 1, N}: run the op on
capacity-N inputs whose entries past n are garbage (half plausible, so a kernel
that wrongly uses them does visible work, half far out of range, so it also
faults) with ``num_valid = n``, and compare with the op run on the exact slice
[:n] without ``num_valid``. Outputs start as a sentinel and what lies past n
must still be the sentinel afterwards. GPU only, self-contained.
"""

import pytest
import torch
from dynamicemb import DynamicEmbInitializerArgs, DynamicEmbInitializerMode
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.optimizer import (
    AdamDynamicEmbeddingOptimizer,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA device"
)
ext = pytest.importorskip("dynamicemb_extensions")

DEV = "cuda"
N = 37  # the capacity; odd, so that N // 2 and N - 1 line up with nothing
COUNTS = [0, 1, N // 2, N - 1, N]
SENT = 7.0
ROWS = 64  # rows of a flat table
# (embedding dims of the tables, all_dims_vec4): the vec4 and the scalar kernels.
DIMS = [([8, 4], True), ([6, 5], False)]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


def nv(n):
    """The device counter."""
    return torch.tensor([n], dtype=torch.int64, device=DEV)


def randint(high, size=N):
    return torch.randint(0, high, (size,), dtype=torch.int64, device=DEV)


def junk_tail(live, n, plausible):
    """``live`` with everything past its first n entries replaced by garbage."""
    wild = torch.randint(-(1 << 40), 1 << 40, live.shape, device=DEV)
    junk = torch.where(torch.rand(live.shape, device=DEV) < 0.5, plausible, wild)
    return torch.cat([live[:n], junk[n:]])


def arange_n():
    return torch.arange(N, device=DEV)


def nan_tail(x, n):
    x = x.clone()
    x[n:] = float("nan")
    return x


def flat_tables(emb_dims, extra, dtype=torch.float32):
    """One value table per embedding dim, each row ``emb | extra * emb`` wide."""
    vdims = [e * (1 + extra) for e in emb_dims]
    tables = [torch.rand(ROWS, v, device=DEV).to(dtype) for v in vdims]
    return {
        "tables": tables,
        "table_ptrs": torch.tensor(
            [t.data_ptr() for t in tables], dtype=torch.int64, device=DEV
        ),
        "table_value_dims": torch.tensor(vdims, dtype=torch.int64, device=DEV),
        "table_emb_dims": torch.tensor(emb_dims, dtype=torch.int64, device=DEV),
        "max_emb_dim": max(emb_dims),
    }


def clone_tables(t):
    tables = [x.clone() for x in t["tables"]]
    ptrs = torch.tensor([x.data_ptr() for x in tables], dtype=torch.int64, device=DEV)
    return {**t, "tables": tables, "table_ptrs": ptrs}


def same_tables(a, b):
    return all(torch.equal(x, y) for x, y in zip(a["tables"], b["tables"]))


def tail(t, vec4):
    """The arguments after the per-row ones that the flat load, store and update share."""
    return (t["table_value_dims"], t["table_emb_dims"], t["max_emb_dim"], vec4)


# -- flat table load / store ------------------------------------------------
@pytest.mark.parametrize("dims,vec4", DIMS)
@pytest.mark.parametrize("kind", ["contiguous", "emb", "value"])
@pytest.mark.parametrize("n", COUNTS)
def test_load_from_flat_table(dims, vec4, kind, n):
    t = flat_tables(dims, extra=2)
    me = t["max_emb_dim"]
    fn = getattr(ext, f"load_from_flat_table_{kind}")
    width = {"contiguous": 3 * dims[0], "emb": me, "value": 3 * me}[kind]
    indices = junk_tail(randint(ROWS), n, randint(ROWS))
    tids = junk_tail(randint(len(dims)), n, randint(len(dims)))

    def run(idx, ids, out, **kw):
        # The contiguous form reads one table, the others one table per row.
        table = 0 if kind == "contiguous" else ids
        fn(t["table_ptrs"], idx, table, out, *tail(t, vec4), **kw)

    out = torch.full((N, width), SENT, device=DEV)
    run(indices, tids, out, num_valid=nv(n))
    ref = torch.full((n, width), SENT, device=DEV)
    run(indices[:n], tids[:n], ref)
    assert torch.equal(out[:n], ref)
    assert (out[n:] == SENT).all()


@pytest.mark.parametrize("dims,vec4", DIMS)
@pytest.mark.parametrize("kind", ["contiguous", "value"])
@pytest.mark.parametrize("n", COUNTS)
def test_store_to_flat_table(dims, vec4, kind, n):
    a = flat_tables(dims, extra=2)
    b = clone_tables(a)
    fn = getattr(ext, f"store_to_flat_table_{kind}")
    width = 3 * dims[0] if kind == "contiguous" else 3 * a["max_emb_dim"]
    rows = torch.randperm(ROWS, device=DEV)[:N]  # distinct: no two entries race
    indices = junk_tail(rows, n, randint(ROWS))
    tids = junk_tail(randint(len(dims)), n, randint(len(dims)))
    src = torch.randn(N, width, device=DEV)

    def run(t, x, idx, ids, **kw):
        table = 0 if kind == "contiguous" else ids
        fn(t["table_ptrs"], idx, table, x, *tail(t, vec4), **kw)

    run(a, nan_tail(src, n), indices, tids, num_valid=nv(n))
    run(b, src[:n], indices[:n], tids[:n])
    assert same_tables(a, b)


@pytest.mark.parametrize("dim", [8, 6])
@pytest.mark.parametrize("n", COUNTS)
def test_select_insert_failed_values(dim, n):
    k = 11
    values = torch.randn(k, dim, device=DEV)
    # >= 0: not a failed insert, skipped; < 0: -1 - (row of `values`).
    failed = torch.rand(N, device=DEV) < 0.5
    live = torch.where(failed, -1 - randint(k), randint(999))
    indices = junk_tail(live, n, -1 - randint(k))
    before = indices.clone()
    out = torch.full((N, dim), SENT, device=DEV)
    ext.select_insert_failed_values(indices, values, out, num_valid=nv(n))
    ref_idx = before[:n].clone()
    ref_out = torch.full((n, dim), SENT, device=DEV)
    ext.select_insert_failed_values(ref_idx, values, ref_out)
    assert torch.equal(out[:n], ref_out) and (out[n:] == SENT).all()
    # The op marks what it served by writing -1 back; past n nothing changes.
    assert torch.equal(indices[:n], ref_idx)
    assert torch.equal(indices[n:], before[n:])


# -- initializers -----------------------------------------------------------

M, D, PAD = 41, 12, 4  # buffer rows, embedding width, columns beyond it


def init_case(n, with_indices):
    """A sentinel buffer (a column slice of a wider one), the indices (None: rows
    0..n-1) and the rows that must change."""
    base = torch.full((M, D + PAD), SENT, device=DEV)
    rows = torch.randperm(M, device=DEV)[:N] if with_indices else arange_n()
    indices = junk_tail(rows, n, randint(M)) if with_indices else None
    return base, base[:, :D], indices, rows[:n]


@pytest.mark.parametrize("via", ["op", "python"])
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_const_init(n, with_indices, via):
    base, buf, indices, live = init_case(n, with_indices)
    want = base.clone()
    want[live, :D] = 3.0
    if via == "op":
        ext.const_init(buf, indices, 3.0, num_valid=nv(n))
    else:
        args = DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=3.0
        )
        MultiTableInitializer.create([args], torch.device(DEV))(
            buf, None, None, indices, nv(n)
        )
    assert torch.equal(base, want)


@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_per_row_initializers(n, with_indices):
    """table_ids and keys run alongside the rows of the buffer, not the indices."""
    base, buf, indices, live = init_case(n, with_indices)
    table_params = torch.tensor([[1.0], [2.0], [3.0]], device=DEV)
    table_ids = randint(3, M)
    want = base.clone()
    want[live, :D] = table_params[table_ids[live], 0][:, None]
    ext.const_init_table_params(buf, indices, table_params, table_ids, num_valid=nv(n))
    assert torch.equal(base, want)

    base, buf, indices, live = init_case(n, with_indices)
    keys = torch.arange(M, device=DEV) * 7
    want = base.clone()
    want[live, :D] = (keys[live] % 100000).float()[:, None]
    ext.debug_init(buf, indices, keys, num_valid=nv(n))
    assert torch.equal(base, want)


@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_random_initializers_touch_only_live_rows(n, with_indices):
    ctx = ext.CurandStateContext()
    base, buf, indices, live = init_case(n, with_indices)
    ext.normal_init(buf, indices, ctx, 0.0, 1.0, num_valid=nv(n))
    want = torch.zeros(M, dtype=torch.bool, device=DEV)
    want[live] = True
    assert torch.equal((base[:, :D] != SENT).all(dim=1), want)
    assert (base[:, D:] == SENT).all()


def test_initializer_without_indices_needs_num_valid():
    _, buf, _, _ = init_case(N, False)
    with pytest.raises(Exception):
        ext.const_init(buf, None, 1.0)


# -- optimizers ---------------------------------------------------------------


OPTIMIZERS = {
    "sgd": SGDDynamicEmbeddingOptimizer,
    "adam": AdamDynamicEmbeddingOptimizer,
}


def make_optimizer(name):
    opt = OPTIMIZERS[name](OptimizerArgs(learning_rate=0.1))
    opt.step()  # Adam's bias correction needs a first iteration
    return opt


@pytest.mark.parametrize("name", ["sgd", "adam"])
@pytest.mark.parametrize("dims,vec4", DIMS)
@pytest.mark.parametrize("n", COUNTS)
def test_flat_table_update(name, dims, vec4, n):
    opt = make_optimizer(name)
    a = flat_tables(dims, extra=2 if name == "adam" else 0)
    b = clone_tables(a)
    me = a["max_emb_dim"]
    rows = torch.randperm(ROWS, device=DEV)[:N]  # distinct: no two entries race
    indices = junk_tail(rows, n, randint(ROWS))
    tids = junk_tail(randint(len(dims)), n, randint(len(dims)))
    grads = torch.randn(N, me, device=DEV)

    def run(t, g, idx, ids, **kw):
        opt.fused_update_for_flat_table(
            g, idx, t["table_ptrs"], ids, *tail(t, vec4), torch.float32, **kw
        )

    run(a, nan_tail(grads, n), indices, tids, num_valid=nv(n))
    run(b, grads[:n], indices[:n], tids[:n])
    assert same_tables(a, b)


@pytest.mark.parametrize("name", ["sgd", "adam"])
@pytest.mark.parametrize("dims,vec4", DIMS)
@pytest.mark.parametrize("n", COUNTS)
def test_padded_buffer_update(name, dims, vec4, n):
    opt = make_optimizer(name)
    me = max(dims)
    vd = me * (3 if name == "adam" else 1)
    a = torch.rand(N, vd, device=DEV)
    b, before = a.clone(), a.clone()
    tids = junk_tail(randint(len(dims)), n, randint(len(dims)))
    emb_dims = torch.tensor(dims, dtype=torch.int64, device=DEV)
    grads = torch.randn(N, me, device=DEV)
    rest = (emb_dims, me, vd, vec4)
    opt.update_for_padded_buffer(nan_tail(grads, n), a, tids, *rest, num_valid=nv(n))
    if n:
        opt.update_for_padded_buffer(grads[:n], b[:n], tids[:n], *rest)
    assert torch.equal(a, b)
    assert torch.equal(a[n:], before[n:])


def test_rowwise_adagrad_blocks_with_several_rows():
    """With a counter the grid is capped, so a block works through several rows in
    turn and reuses its shared reduction buffer: same result as one block per row."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows, dim = 4 * sms * 32 * 4, 96  # more rows than the cap of any block size
    opt = RowWiseAdaGradDynamicEmbeddingOptimizer(
        OptimizerArgs(learning_rate=0.1), torch.float32
    )
    a = torch.rand(rows, dim + 4, device=DEV)
    b = a.clone()
    grads = torch.randn(rows, dim, device=DEV)
    tids = torch.zeros(rows, dtype=torch.int64, device=DEV)
    emb_dims = torch.tensor([dim], dtype=torch.int64, device=DEV)
    args = (tids, emb_dims, dim, dim + 4, False)  # the scalar kernel
    opt.update_for_padded_buffer(grads, a, *args, num_valid=nv(rows))
    opt.update_for_padded_buffer(grads, b, *args)
    assert torch.equal(a, b)


# -- reduce_grads --------------------------------------------------------------
def inverse_of(unique, keys):
    """An inverse index over `keys` keys in which every id < unique occurs."""
    rest = torch.randint(0, unique, (keys - unique,))
    inv = torch.cat([torch.arange(unique), rest]).to(DEV)
    return inv[torch.randperm(keys, device=DEV)]


@pytest.mark.parametrize("dim", [8, 6])
@pytest.mark.parametrize("unique", [1, N // 2, N - 1, N])
def test_reduce_grads_sequence(dim, unique):
    inv = inverse_of(unique, N)
    grads = torch.randn(N, dim, device=DEV)
    want = ext.reduce_grads(inv, grads, unique, N, dim)
    got = ext.reduce_grads(inv, grads, N, N, dim, num_valid=nv(unique))
    assert got.shape == (N, dim)
    torch.testing.assert_close(got[:unique], want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dim", [8, 6])
@pytest.mark.parametrize("frac", [0, 1, 2, 3])
def test_reduce_grads_pooled(dim, frac):
    features, batch = 2, 5
    sizes = torch.tensor([3, 0, 2, 1, 4, 2, 0, 3, 1, 2], device=DEV)  # feature-major
    zero = torch.zeros(1, dtype=torch.int64, device=DEV)
    offsets = torch.cat([zero, sizes.cumsum(0)])
    keys = int(offsets[-1])
    unique = [1, keys // 2, keys - 1, keys][frac]
    inv = inverse_of(unique, keys)
    grads = torch.randn(batch, features * dim, device=DEV)
    kw = dict(
        offsets=offsets, pooling_mode=ext.PoolingMode.KSum, total_D=features * dim
    )
    want = ext.reduce_grads(inv, grads, unique, batch, dim, **kw)
    got = ext.reduce_grads(inv, grads, keys, batch, dim, num_valid=nv(unique), **kw)
    assert got.shape == (keys, dim)
    torch.testing.assert_close(got[:unique], want, rtol=1e-5, atol=1e-5)


def test_reduce_grads_without_keys_zeroes_only_the_live_rows():
    """The one place the op zero-fills its result: no keys, but unique rows."""
    inv = torch.empty(0, dtype=torch.int64, device=DEV)
    grads = torch.empty(0, 8, device=DEV)
    out = ext.reduce_grads(inv, grads, 4, 1, 8, num_valid=nv(2))
    assert out.shape == (4, 8) and (out[:2] == 0).all()


# -- row fills ------------------------------------------------------------------

W, X = 8, 3  # state width, columns before it
EMB = [4, 3, 2]  # embedding dims of three FTRL tables; 2 * max(EMB) <= W


def state_case(n, with_indices, dtype):
    """A sentinel buffer, the state view of it, the indices (None: rows 0..n-1;
    some live ones are -1, which skips) and the rows that must change."""
    base = torch.full((M, X + W), SENT, dtype=dtype, device=DEV)
    rows = torch.randperm(M, device=DEV)[:N] if with_indices else arange_n()
    if with_indices:
        rows[::5] = -1
    indices = junk_tail(rows, n, randint(M)) if with_indices else None
    live = rows[:n]
    return base, base[:, X:], indices, live[live >= 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_fill_rows(n, with_indices, dtype):
    base, dst, indices, live = state_case(n, with_indices, dtype)
    want = base.clone()
    want[live, X:] = 0.25
    ext.fill_rows(dst, 0.25, num_valid=nv(n), indices=indices)
    assert torch.equal(base, want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_fill_rows_fp32_words(n, with_indices, dtype):
    """An fp32 over pairs of 16-bit words, low word first: row-wise Adagrad's accumulator."""
    base, dst, indices, live = state_case(n, with_indices, dtype)
    pattern = torch.full((W // 2,), 0.3, device=DEV).view(torch.int16)
    want = base.clone().view(torch.int16)
    want[live, X:] = pattern
    ext.fill_rows(dst, 0.3, num_valid=nv(n), indices=indices, fp32_words=True)
    assert torch.equal(base.view(torch.int16), want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_reset_ftrl_states(n, with_indices, dtype):
    base, dst, indices, live = state_case(n, with_indices, dtype)
    table_ids = randint(len(EMB), M)  # one per row of the buffer
    emb_dims = torch.tensor(EMB, dtype=torch.int64, device=DEV)
    want = base.clone()
    for r in live.tolist():
        e = EMB[int(table_ids[r])]
        want[r, X:] = 0
        want[r, X + e : X + 2 * e] = 0.5
    ext.reset_ftrl_states(
        dst, table_ids, emb_dims, 0.5, num_valid=nv(n), indices=indices
    )
    assert torch.equal(base, want)


@pytest.mark.parametrize("kind", ["ftrl", "rowwise_fp16", "adam"])
@pytest.mark.parametrize("with_indices", [True, False])
@pytest.mark.parametrize("n", COUNTS)
def test_reset_optimizer_states_counted(kind, with_indices, n):
    """The counted form against the host-sized one on exactly the live rows."""
    if kind == "ftrl":
        args = OptimizerArgs(learning_rate=0.1, initial_accumulator_value=0.5)
        opt, dtype = FTRLDynamicEmbeddingOptimizer(args), torch.float32
    elif kind == "rowwise_fp16":
        args = OptimizerArgs(initial_accumulator_value=0.25)
        opt = RowWiseAdaGradDynamicEmbeddingOptimizer(args, torch.float16)
        dtype = torch.float16
    else:
        opt, dtype = AdamDynamicEmbeddingOptimizer(OptimizerArgs()), torch.float32
    base, dst, indices, live = state_case(n, with_indices, dtype)
    ref = base.clone()
    table_ids = randint(len(EMB), M)
    emb_dims = torch.tensor(EMB, dtype=torch.int64, device=DEV)
    opt.reset_optimizer_states(
        dst, indices, num_valid=nv(n), table_ids=table_ids, table_emb_dims=emb_dims
    )
    opt.reset_optimizer_states(
        ref[:, X:], indices=live, emb_dims=emb_dims[table_ids[live]]
    )
    assert torch.equal(base.view(torch.uint8), ref.view(torch.uint8))
