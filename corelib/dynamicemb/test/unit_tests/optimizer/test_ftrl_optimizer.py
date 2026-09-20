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

"""FTRL, checked against Algorithm 1 of McMahan et al. 2013.

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
        linear
        + grad
        - (new_accum_pow - accum**exponent) / learning_rate * weight
    )
    shrunk = (l1_reg * torch.sign(linear) - linear) / (
        (ftrl_beta + new_accum_pow) / learning_rate + l2_reg
    )
    # |linear| <= l1_reg pins the weight to exactly zero; that is the whole
    # point of the L1 term and what separates FTRL from a shrinkage update.
    weight = torch.where(linear.abs() > l1_reg, shrunk, torch.zeros_like(shrunk))
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
def test_ftrl_flat_table_matches_padded_buffer(emb_dim, all_dims_vec4):
    """The two kernels differ only in how they find a row, not in the update."""
    device = torch.device("cuda")
    num_rows = 32
    row_width = 3 * emb_dim
    accum_seed = 0.1
    optimizer = _ftrl_optimizer()

    padded = _fresh_ftrl_values(
        num_rows, emb_dim, row_width, emb_dim, accum_seed, device, seed=3
    )
    flat = padded.clone()
    generator = torch.Generator(device="cpu").manual_seed(11)
    grads = torch.randn(
        num_rows, emb_dim, generator=generator, dtype=torch.float32
    ).to(device)

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
    ],
    ids=["plain", "beta_l2", "l1", "power", "seeded_accum"],
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
