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

"""FTRL, checked against Algorithm 1 of McMahan et al. 2013, and its group-sparse
variant (G-FTRL, Ni et al. 2019) against a transcription of that.

Every other optimizer here is checked against a
``SplitTableBatchedEmbeddingBagsCodegen`` built with the same hyperparameters.
FBGEMM has no FTRL, and a second dynamicemb storage is no substitute -- it takes
the very optimizer under test, so a wrong formula would come out wrong on both
sides and the comparison would pass. ``ftrl_step`` below is an independent
transcription, in float64 so it stays a reference rather than a second copy of
the kernel's rounding.

The three tests cover the three layers FTRL can break at: the fused kernels
agreeing with each other, their row-addressing contract, and a real
``BatchedDynamicEmbeddingTablesV2`` arriving where the paper says it should.
"""

from typing import Tuple

import pytest
import torch
from dynamicemb import (
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_value_table import export_keys_values_iter
from dynamicemb.optimizer import (
    DynamicEmbOptimType,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")


def ftrl_step(
    weight: torch.Tensor,
    linear: torch.Tensor,
    accum: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    learning_rate_power: float = -0.5,
    ftrl_beta: float = 0.0,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
    group_lasso_reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FTRL-Proximal step, from Algorithm 1 of McMahan et al. 2013.

    ``learning_rate``, ``ftrl_beta``, ``l1_reg`` and ``l2_reg`` are the paper's
    alpha, beta, lambda1 and lambda2. The paper fixes the accumulator's exponent
    at a square root; ``learning_rate_power`` generalizes it, and its default of
    -0.5 recovers the paper.

    Returns the new ``(weight, linear, accum)``. Elementwise, so it takes any
    shape as long as all four arguments agree.
    """
    weight = weight.to(torch.float64)
    linear = linear.to(torch.float64)
    accum = accum.to(torch.float64)
    grad = grad.to(torch.float64)

    exponent = -learning_rate_power
    new_accum = accum + grad * grad
    new_accum_pow = new_accum**exponent

    linear = (
        linear + grad - (new_accum_pow - accum**exponent) / learning_rate * weight
    )
    if group_lasso_reg > 0.0:
        # G-FTRL, Algorithm 1 and Eq. 3 of Ni et al., CIKM 2019. The last dim is
        # the group (one embedding row, width d). With s the linear term
        # soft-thresholded by l1_reg (s = linear when l1_reg = 0, the paper's
        # closed form; otherwise the sparse-group-lasso step of Yue et al.,
        # arXiv:2107.14432, Algorithm 3), the row is zeroed when
        # ||s||_2 <= lambda * sqrt(d), and otherwise shrunk by
        # 1 - lambda * sqrt(d) / ||s||_2. The shrink factor is common to the row
        # (the papers' form), so for rows whose curvature differs this is an
        # approximation of the objective's minimizer, not the minimizer itself.
        s = torch.sign(linear) * (linear.abs() - l1_reg).clamp(min=0)
        threshold = group_lasso_reg * linear.size(-1) ** 0.5
        norm = s.norm(dim=-1, keepdim=True)
        scale = torch.where(
            norm > threshold,
            1.0 - threshold / norm.clamp(min=1e-300),
            torch.zeros_like(norm),
        )
        numerator = scale * s
        denom = (ftrl_beta + new_accum_pow) / learning_rate + l2_reg
    else:
        # |linear| <= l1_reg pins the weight to exactly zero; that is the whole
        # point of the L1 term and what separates FTRL from a shrinkage update.
        numerator = torch.sign(linear) * (linear.abs() - l1_reg).clamp(min=0)
        denom = (ftrl_beta + new_accum_pow) / learning_rate + l2_reg
    # A coordinate with no gradient and no state has denom == 0 when beta and
    # l2_reg are zero; its numerator is zero too, and the answer is zero, not
    # the 0/0 an unguarded division gives.
    weight = torch.where(
        numerator == 0, torch.zeros_like(numerator), -numerator / denom
    )
    return weight, linear, new_accum


def _ftrl_optimizer(**overrides):
    """An FTRL optimizer with the paper's defaults unless overridden."""
    args = dict(
        learning_rate=0.1, learning_rate_power=-0.5, initial_accumulator_value=0.0
    )
    args.update(overrides)
    return FTRLDynamicEmbeddingOptimizer(OptimizerArgs(**args))


def _fresh_ftrl_values(
    num_rows, emb_dim, row_width, state_offset, accum_seed, device, seed=0
):
    """A value buffer with random embeddings, linear at 0 and accum seeded."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.zeros(num_rows, row_width, dtype=torch.float32)
    values[:, :emb_dim] = torch.rand(
        num_rows, emb_dim, generator=generator, dtype=torch.float32
    )
    values[:, state_offset : state_offset + emb_dim] = 0.0
    values[:, state_offset + emb_dim : state_offset + 2 * emb_dim] = accum_seed
    return values.to(device)


@cuda
@pytest.mark.parametrize("emb_dim,all_dims_vec4", [(4, True), (5, False)])
@pytest.mark.parametrize("group_lasso_reg", [0.0, 0.2], ids=["plain", "group"])
def test_ftrl_flat_table_matches_padded_buffer(emb_dim, all_dims_vec4, group_lasso_reg):
    """The two kernels differ only in how they find a row, not in the update."""
    device = torch.device("cuda")
    num_rows = 32
    row_width = 3 * emb_dim
    accum_seed = 0.1
    optimizer = _ftrl_optimizer(group_lasso_reg=group_lasso_reg)

    padded = _fresh_ftrl_values(
        num_rows, emb_dim, row_width, emb_dim, accum_seed, device, seed=3
    )
    flat = padded.clone()
    generator = torch.Generator(device="cpu").manual_seed(11)
    grads = torch.randn(num_rows, emb_dim, generator=generator, dtype=torch.float32).to(
        device
    )

    table_ids = torch.zeros(num_rows, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)
    table_value_dims = torch.tensor([row_width], dtype=torch.int64, device=device)

    optimizer.update_for_padded_buffer(
        grads, padded, table_ids, table_emb_dims, emb_dim, row_width, all_dims_vec4
    )

    optimizer.fused_update_for_flat_table(
        grads,
        torch.arange(num_rows, dtype=torch.int64, device=device),
        torch.tensor([flat.data_ptr()], dtype=torch.int64, device=device),
        table_ids,
        table_value_dims,
        table_emb_dims,
        emb_dim,
        all_dims_vec4,
        torch.float32,
    )

    torch.testing.assert_close(flat, padded, rtol=0, atol=0)


@cuda
def test_ftrl_skipped_rows_are_untouched():
    """-1 marks a key that failed to insert; its row must not be written."""
    device = torch.device("cuda")
    emb_dim = 4
    row_width = 3 * emb_dim
    optimizer = _ftrl_optimizer()

    values = _fresh_ftrl_values(8, emb_dim, row_width, emb_dim, 0.1, device, seed=9)
    before = values.clone()
    indices = torch.full((8,), -1, dtype=torch.int64, device=device)
    indices[3] = 3
    grads = torch.ones(8, emb_dim, dtype=torch.float32, device=device)

    optimizer.fused_update_for_flat_table(
        grads,
        indices,
        torch.tensor([values.data_ptr()], dtype=torch.int64, device=device),
        torch.zeros(8, dtype=torch.int64, device=device),
        torch.tensor([row_width], dtype=torch.int64, device=device),
        torch.tensor([emb_dim], dtype=torch.int64, device=device),
        emb_dim,
        True,
        torch.float32,
    )

    untouched = [r for r in range(8) if r != 3]
    torch.testing.assert_close(values[untouched], before[untouched], rtol=0, atol=0)
    assert not bool(torch.equal(values[3], before[3]))


@cuda
@pytest.mark.parametrize("emb_dim", [8, 6], ids=["vec4", "unaligned"])
@pytest.mark.parametrize(
    "opt_params",
    [
        {"learning_rate": 0.1},
        {"learning_rate": 0.1, "ftrl_beta": 1.0, "l2_reg": 0.05},
        {"learning_rate": 0.1, "l1_reg": 0.02},
        {"learning_rate": 0.1, "learning_rate_power": -0.25},
        {"learning_rate": 0.1, "initial_accumulator_value": 0.1},
        {
            "learning_rate": 0.1,
            "l1_reg": 0.005,
            "l2_reg": 0.02,
            "group_lasso_reg": 2.0,
        },
    ],
    ids=["plain", "beta_l2", "l1", "power", "seeded_accum", "group"],
)
def test_ftrl_backward_matches_reference(emb_dim, opt_params):
    """FTRL has no FBGEMM twin, so check it against a transcription of the paper.

    A second dynamicemb storage would be no use here: it shares the optimizer
    under test, so a wrong formula would come out wrong on both sides.
    ``ftrl_step`` above is independent and runs in float64.

    Sequence pooling with one unique key per bag keeps the bookkeeping honest --
    every row is touched exactly once per iteration and ``loss = embs.sum()``
    hands each of its elements a gradient of exactly 1.
    """
    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    key_type = torch.int64
    value_type = torch.float32
    num_keys = 16

    options = DynamicEmbTableOptions(
        dim=emb_dim,
        max_capacity=2048,
        index_type=key_type,
        embedding_dtype=value_type,
        device_id=device_id,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=False,
        local_hbm_for_values=1024**3,
    )
    bdeb = BatchedDynamicEmbeddingTablesV2(
        table_names=["table0"],
        table_options=[options],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=DynamicEmbOptimType.FTRL,
        **opt_params,
    )
    optimizer = bdeb.optimizer
    storage = bdeb.tables
    state_dim = optimizer.get_state_dim(emb_dim)
    max_emb_dim = storage.max_embedding_dim()
    max_value_dim = storage.max_value_dim()

    # Seed both sides from the same weights, and let the optimizer lay down its
    # own initial state so the reference starts where the table does.
    generator = torch.Generator(device="cpu").manual_seed(17)
    weights = torch.rand(num_keys, emb_dim, generator=generator).to(device)
    values = torch.zeros(num_keys, max_value_dim, dtype=value_type, device=device)
    values[:, :emb_dim] = weights
    optimizer.reset_optimizer_states(
        values[:, max_emb_dim : max_emb_dim + state_dim], emb_dims=emb_dim
    )
    keys = torch.arange(num_keys, device=device, dtype=key_type)
    storage.set_score(1)
    storage.insert(keys, torch.zeros_like(keys), values)

    ref_weight = weights.double()
    ref_linear = values[:, max_emb_dim : max_emb_dim + emb_dim].double().clone()
    ref_accum = (
        values[:, max_emb_dim + emb_dim : max_emb_dim + 2 * emb_dim].double().clone()
    )

    # initial_accumulator_value seeds the state rather than taking part in a
    # step, and ref_accum already starts from the seeded value the table holds,
    # so it is not among what ftrl_step is told.
    step_params = {
        k: v for k, v in opt_params.items() if k != "initial_accumulator_value"
    }

    offsets = torch.arange(num_keys + 1, device=device).to(key_type)
    for _ in range(4):
        embs = bdeb(keys, offsets)
        embs.sum().backward()
        torch.cuda.synchronize()
        # d(sum)/d(emb_ij) == 1 for every looked-up element.
        grad = torch.ones_like(ref_weight)
        ref_weight, ref_linear, ref_accum = ftrl_step(
            ref_weight, ref_linear, ref_accum, grad, **step_params
        )

    # Keys are 0..num_keys-1 and the reference is indexed the same way, so each
    # batch can be looked up directly -- no need to care what order the export
    # walks the table in, or how it splits the rows into batches.
    seen = 0
    for keys_b, emb_b, opt_b, _ in export_keys_values_iter(
        storage._state, device, table_id=0
    ):
        if keys_b.numel() == 0:
            continue
        assert (
            opt_b is not None
        ), "FTRL keeps per-row state, so the export must carry it"
        seen += keys_b.numel()
        torch.testing.assert_close(
            emb_b.double(), ref_weight[keys_b], rtol=1e-4, atol=1e-5, msg="weight"
        )
        torch.testing.assert_close(
            opt_b[:, :emb_dim].double(),
            ref_linear[keys_b],
            rtol=1e-4,
            atol=1e-4,
            msg="linear",
        )
        torch.testing.assert_close(
            opt_b[:, emb_dim:].double(),
            ref_accum[keys_b],
            rtol=1e-5,
            atol=1e-6,
            msg="accum",
        )
    assert seen == num_keys, f"exported {seen} of {num_keys} rows"


def _run_group_step(optimizer, values, grads, emb_dim, all_dims_vec4):
    num_rows, row_width = values.shape
    device = values.device
    optimizer.update_for_padded_buffer(
        grads,
        values,
        torch.zeros(num_rows, dtype=torch.int64, device=device),
        torch.tensor([emb_dim], dtype=torch.int64, device=device),
        emb_dim,
        row_width,
        all_dims_vec4,
    )


@cuda
@pytest.mark.parametrize("emb_dim,all_dims_vec4", [(8, True), (5, False), (70, False)])
@pytest.mark.parametrize("l2_reg", [0.0, 0.05], ids=["no_l2", "l2"])
@pytest.mark.parametrize("l1_reg", [0.0, 0.01], ids=["no_l1", "l1"])
def test_ftrl_group_sparsity_matches_reference(emb_dim, all_dims_vec4, l2_reg, l1_reg):
    """Rows of very different gradient scale, so some clear the threshold and some do not.

    Checks the weights against the float64 transcription, that the rows below
    the threshold are *exactly* zero (the point of the group term), and that
    the others are not -- otherwise a kernel that zeroed everything would pass.
    """
    device = torch.device("cuda")
    num_rows = 64
    row_width = 3 * emb_dim
    params = dict(
        learning_rate=0.1,
        l1_reg=l1_reg,
        group_lasso_reg=0.3,
        l2_reg=l2_reg,
    )
    optimizer = _ftrl_optimizer(**params)

    values = _fresh_ftrl_values(
        num_rows, emb_dim, row_width, emb_dim, 0.0, device, seed=5
    )
    generator = torch.Generator(device="cpu").manual_seed(23)
    row_scale = torch.logspace(-3, 0, num_rows, dtype=torch.float32).unsqueeze(1)
    grads = (
        torch.randn(num_rows, emb_dim, generator=generator, dtype=torch.float32)
        * row_scale
    ).to(device)

    ref_w, ref_l, ref_a = ftrl_step(
        values[:, :emb_dim].cpu(),
        values[:, emb_dim : 2 * emb_dim].cpu(),
        values[:, 2 * emb_dim :].cpu(),
        grads.cpu(),
        **params,
    )
    _run_group_step(optimizer, values, grads, emb_dim, all_dims_vec4)

    weight = values[:, :emb_dim].cpu()
    zero_rows = (ref_w == 0).all(dim=1)
    assert zero_rows.any() and not zero_rows.all(), "test needs both kinds of row"
    # Exactly zero together, not merely small.
    assert bool((weight[zero_rows] == 0).all())
    assert bool((weight[~zero_rows] != 0).any(dim=1).all())
    torch.testing.assert_close(weight.double(), ref_w, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        values[:, emb_dim : 2 * emb_dim].cpu().double(), ref_l, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        values[:, 2 * emb_dim :].cpu().double(), ref_a, rtol=1e-5, atol=1e-6
    )


# Rounding the state through the table dtype is part of what the group-sparse
# kernel does (it must, so the row norm is of what is stored), and the float64
# reference does not round, so the narrow dtypes get looser tolerances.
_NARROW_TOLERANCES = {
    torch.float32: dict(rtol=1e-4, atol=1e-5),
    torch.float16: dict(rtol=3e-2, atol=3e-3),
    torch.bfloat16: dict(rtol=6e-2, atol=1e-2),
}


def _group_case(dims, dtype, rows_per_table=16, seed=31):
    """Rows that are decisively on one side of the threshold, one set of tables.

    Even rows get tiny gradients and odd rows large ones, so after one step the
    row norm is far below the group threshold for the first and far above it for
    the second. Nothing sits near the threshold, so the narrow dtypes' rounding
    cannot flip a row between the kernel and the float64 reference. The tiny
    gradients are kept large enough that their square is not lost in fp16.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_dim = max(dims)
    num_rows = len(dims) * rows_per_table
    weights = torch.rand(num_rows, max_dim, generator=generator).to(dtype)
    grads = torch.randn(num_rows, max_dim, generator=generator)
    mags = torch.where(
        torch.arange(num_rows) % 2 == 0,
        torch.full((num_rows,), 5e-3),
        1.0 + 0.5 * torch.rand(num_rows, generator=generator),
    ).unsqueeze(1)
    grads = (grads.sign() * mags * (0.5 + torch.rand_like(grads) * 0.5)).to(dtype)
    table_of_row = torch.arange(len(dims)).repeat_interleave(rows_per_table)
    return weights, grads, table_of_row


def _check_group_rows(dims, dtype, params, weights, grads, table_of_row, out_of_row):
    """Compare every row's kernel output with the float64 transcription.

    ``out_of_row(row)`` returns that row's ``(weight, linear, accum)`` as read
    back from whichever layout the path under test uses.
    """
    tol = _NARROW_TOLERANCES[dtype]
    zero_kernel, zero_ref = [], []
    for row in range(weights.size(0)):
        edim = dims[int(table_of_row[row])]
        w, lin, acc = out_of_row(row)
        # The kernel reads the table dtype, so the reference starts from the
        # same rounded values and the same gradient it was handed.
        ref_w, ref_l, ref_a = ftrl_step(
            weights[row, :edim].double(),
            torch.zeros(edim),
            torch.zeros(edim),
            grads[row, :edim].double(),
            **params,
        )
        torch.testing.assert_close(
            w.double(), ref_w, msg=lambda m, r=row: f"weight, row {r}: {m}", **tol
        )
        torch.testing.assert_close(
            lin.double(), ref_l, msg=lambda m, r=row: f"linear, row {r}: {m}", **tol
        )
        torch.testing.assert_close(
            acc.double(), ref_a, msg=lambda m, r=row: f"accum, row {r}: {m}", **tol
        )
        zero_kernel.append(bool((w == 0).all()))
        zero_ref.append(bool((ref_w == 0).all()))
        if zero_ref[-1]:
            assert zero_kernel[-1], f"row {row} should be exactly zero"
    assert zero_kernel == zero_ref, "kernel and reference zero different rows"
    assert any(zero_ref) and not all(zero_ref), "test needs both kinds of row"


_GROUP_DIMS = [[8, 4, 12], [12, 5, 8]]
_GROUP_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_GROUP_DIMS_IDS = ["vec4_mixed", "unaligned_mixed"]
_GROUP_PARAMS = dict(
    learning_rate=0.1,
    l1_reg=0.005,
    l2_reg=0.01,
    group_lasso_reg=0.5,
)


@cuda
@pytest.mark.parametrize("dims", _GROUP_DIMS, ids=_GROUP_DIMS_IDS)
@pytest.mark.parametrize("dtype", _GROUP_DTYPES, ids=["fp32", "fp16", "bf16"])
def test_ftrl_group_padded_buffer_dtype_and_mixed_dims(dims, dtype):
    """Every table dtype, tables of different widths in one padded-buffer call.

    The threshold is ``group_lasso_reg * sqrt(d)`` with ``d`` the row's own
    width, so a kernel that took the buffer's widest row for all of them would
    zero the wrong rows. The state is rounded to the table dtype before the row
    norm is taken. Where the states sit in a padded buffer is covered by
    ``test_padded_buffer_optimizer.py``.
    """
    device = torch.device("cuda")
    max_dim = max(dims)
    weights, grads, table_of_row = _group_case(dims, dtype)
    num_rows = weights.size(0)
    optimizer = _ftrl_optimizer(**_GROUP_PARAMS)

    values = torch.zeros(num_rows, 3 * max_dim, dtype=dtype)
    for row in range(num_rows):
        edim = dims[int(table_of_row[row])]
        values[row, :edim] = weights[row, :edim]
    values = values.to(device)

    optimizer.update_for_padded_buffer(
        grads.to(device),
        values,
        table_of_row.to(device),
        torch.tensor(dims, dtype=torch.int64, device=device),
        max_dim,
        3 * max_dim,
        all(d % 4 == 0 for d in dims),
    )
    out = values.cpu()

    def out_of_row(row):
        edim = dims[int(table_of_row[row])]
        return (
            out[row, :edim],
            out[row, max_dim : max_dim + edim],
            out[row, max_dim + edim : max_dim + 2 * edim],
        )

    _check_group_rows(
        dims, dtype, _GROUP_PARAMS, weights, grads, table_of_row, out_of_row
    )


def _padded_group_step(optimizer, values, grads, emb_dim, all_dims_vec4):
    """One padded-buffer step on a single-table buffer; see _run_group_step."""
    _run_group_step(optimizer, values, grads, emb_dim, all_dims_vec4)


@cuda
@pytest.mark.parametrize("all_dims_vec4", [True, False], ids=["vec4", "scalar"])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_ftrl_group_threshold_boundary(all_dims_vec4, dtype):
    """A row exactly at the threshold is zeroed; just above it is not; just below is.

    With zero gradient the state does not move, so the row norm is the
    linear term's and the case can be built exactly: four elements of 0.5 have
    norm 1.0 and the threshold is 0.5 * sqrt(4) = 1.0.
    """
    device = torch.device("cuda")
    emb_dim = 4
    params = dict(learning_rate=0.1, group_lasso_reg=0.5)
    optimizer = _ftrl_optimizer(**params)

    linear_levels = [0.49, 0.5, 0.51]
    values = torch.zeros(len(linear_levels), 3 * emb_dim, dtype=dtype)
    values[:, :emb_dim] = 0.3
    for row, level in enumerate(linear_levels):
        values[row, emb_dim : 2 * emb_dim] = level
    values[:, 2 * emb_dim :] = 0.1
    stored_linear = values[:, emb_dim : 2 * emb_dim].double()
    # The buffer is updated in place, so keep the state as it was before the
    # step to compare against; reading it back afterwards compares it to itself.
    state_before = values[:, emb_dim:].clone()
    values = values.to(device)

    _padded_group_step(
        optimizer,
        values,
        torch.zeros(3, emb_dim, dtype=dtype, device=device),
        emb_dim,
        all_dims_vec4,
    )
    out = values.cpu()

    norms = stored_linear.norm(dim=1)
    assert bool(norms[0] < 1.0) and bool(norms[2] > 1.0)
    # The middle row's norm is exactly the threshold in every dtype.
    assert float(norms[1]) == 1.0
    assert bool((out[0, :emb_dim] == 0).all())
    assert bool((out[1, :emb_dim] == 0).all()), "norm == threshold must zero the row"
    assert bool((out[2, :emb_dim] != 0).all())
    # A zero gradient leaves the state exactly as it was: accum does not move
    # and sigma is zero, so linear does not either.
    torch.testing.assert_close(out[:, emb_dim:], state_before, rtol=0, atol=0)


@cuda
@pytest.mark.parametrize("emb_dim,all_dims_vec4", [(2048, True), (3001, False)])
def test_ftrl_group_large_dim_matches_reference(emb_dim, all_dims_vec4):
    """Rows wider than a thread block: several elements per thread, two passes."""
    device = torch.device("cuda")
    num_rows = 8
    params = dict(learning_rate=0.1, l1_reg=0.005, l2_reg=0.02, group_lasso_reg=0.3)
    values = _fresh_ftrl_values(
        num_rows, emb_dim, 3 * emb_dim, emb_dim, 0.0, device, seed=9
    )
    generator = torch.Generator(device="cpu").manual_seed(15)
    row_scale = torch.where(
        torch.arange(num_rows) % 2 == 0, torch.tensor(1e-3), torch.tensor(1.0)
    ).unsqueeze(1)
    grads = torch.randn(num_rows, emb_dim, generator=generator) * row_scale

    ref_w, ref_l, ref_a = ftrl_step(
        values[:, :emb_dim].cpu(),
        values[:, emb_dim : 2 * emb_dim].cpu(),
        values[:, 2 * emb_dim :].cpu(),
        grads,
        **params,
    )
    _padded_group_step(
        _ftrl_optimizer(**params), values, grads.to(device), emb_dim, all_dims_vec4
    )
    weight = values[:, :emb_dim].cpu()
    zero_rows = (ref_w == 0).all(dim=1)
    assert zero_rows.any() and not zero_rows.all()
    assert bool((weight[zero_rows] == 0).all())
    torch.testing.assert_close(weight.double(), ref_w, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        values[:, emb_dim : 2 * emb_dim].cpu().double(), ref_l, rtol=1e-4, atol=1e-5
    )
