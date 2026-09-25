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

"""Rowwise Adagrad keeps an fp32 accumulator in fp16 and bf16 tables.

The accumulator is a running sum of mean squared gradients. For embedding
gradients that sum routinely sits below fp16's smallest subnormal (6e-8), so
when it was stored in the table's dtype it rounded to zero on every update and
each step came out as ``lr * g / (|g| + eps)`` -- a constant-size step that never
decays. These tests pin the fix: the accumulator a 16-bit table carries must be
bit-identical to the one an fp32 table carries for the same gradients, at embedding
widths whose state slot is only 2-byte aligned, and it must survive a checkpoint.
"""

import pytest
import torch
from dynamicemb.optimizer import OptimizerArgs, RowWiseAdaGradDynamicEmbeddingOptimizer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

LEARNING_RATE = 0.01
EPS = 1e-8
NUM_ROWS = 64
NUM_STEPS = 50
# mean(g^2) per update is ~1e-10: far below fp16's smallest subnormal, so an
# accumulator stored in fp16 would read back as exactly zero.
GRAD_SCALE = 1e-5


def _optimizer(value_type: torch.dtype) -> RowWiseAdaGradDynamicEmbeddingOptimizer:
    return RowWiseAdaGradDynamicEmbeddingOptimizer(
        OptimizerArgs(
            learning_rate=LEARNING_RATE, eps=EPS, initial_accumulator_value=0.0
        ),
        value_type,
    )


def _train_padded_buffer(
    value_type: torch.dtype, emb_dim: int, all_dims_vec4: bool, seed: int = 0
):
    """Run NUM_STEPS updates on one padded buffer; return (weights, accumulator)."""
    device = torch.device("cuda")
    optimizer = _optimizer(value_type)
    state_dim = optimizer.get_state_dim(emb_dim)
    value_dim = emb_dim + state_dim

    generator = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.rand(NUM_ROWS, emb_dim, generator=generator) * 0.1
    grads = [
        (torch.randn(NUM_ROWS, emb_dim, generator=generator) * GRAD_SCALE).to(device)
        for _ in range(NUM_STEPS)
    ]

    values = torch.empty(NUM_ROWS, value_dim, dtype=value_type, device=device)
    values[:, :emb_dim] = init.to(device=device, dtype=value_type)
    optimizer.reset_optimizer_states(values[:, emb_dim:], emb_dims=emb_dim)

    table_ids = torch.zeros(NUM_ROWS, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)
    for grad in grads:
        optimizer.update_for_padded_buffer(
            grad,
            values,
            table_ids,
            table_emb_dims,
            emb_dim,
            value_dim,
            all_dims_vec4,
        )
    torch.cuda.synchronize()

    accumulator = optimizer.states_for_checkpoint(values[:, emb_dim:], emb_dim)
    return values[:, :emb_dim].float(), accumulator


@cuda
@pytest.mark.parametrize(
    "value_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "emb_dim, all_dims_vec4",
    [(16, True), (16, False), (7, False), (13, False)],
    ids=["vec4", "scalar", "odd7", "odd13"],
)
def test_accumulator_matches_fp32_table(value_type, emb_dim, all_dims_vec4):
    ref_weights, ref_accumulator = _train_padded_buffer(
        torch.float32, emb_dim, all_dims_vec4
    )
    weights, accumulator = _train_padded_buffer(value_type, emb_dim, all_dims_vec4)

    assert accumulator.dtype == torch.float32
    assert accumulator.shape == (NUM_ROWS, 1)
    assert torch.all(accumulator > 0), "accumulator underflowed to zero"
    # The accumulator depends only on the gradients, which both runs share, and
    # both kernels sum them in fp32 -- so the two must agree bit for bit.
    assert torch.equal(accumulator, ref_accumulator)
    # The weights still round to the table's dtype on every store, so they only
    # track the fp32 run to that precision.
    tolerance = 1e-2 if value_type == torch.bfloat16 else 2e-3
    torch.testing.assert_close(weights, ref_weights, rtol=tolerance, atol=tolerance)


@cuda
@pytest.mark.parametrize(
    "value_type",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
def test_checkpoint_roundtrip_is_exact(value_type):
    emb_dim = 7
    _, accumulator = _train_padded_buffer(value_type, emb_dim, all_dims_vec4=False)
    optimizer = _optimizer(value_type)

    assert optimizer.get_state_dtype(value_type) == torch.float32
    restored = optimizer.states_from_checkpoint(
        accumulator, emb_dim, value_type, accumulator.device
    )
    assert restored.dtype == value_type
    assert restored.shape == (NUM_ROWS, optimizer.get_state_dim(emb_dim))
    assert torch.equal(optimizer.states_for_checkpoint(restored, emb_dim), accumulator)


@cuda
@pytest.mark.parametrize(
    "value_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_loads_accumulator_stored_in_table_dtype(value_type):
    """Checkpoints from before the fix hold the accumulator in the table's dtype."""
    emb_dim = 7
    optimizer = _optimizer(value_type)
    legacy = torch.tensor([[0.0], [1.5e-3], [2.0]], dtype=value_type, device="cuda")

    restored = optimizer.states_from_checkpoint(
        legacy, emb_dim, value_type, legacy.device
    )
    assert torch.equal(
        optimizer.states_for_checkpoint(restored, emb_dim), legacy.float()
    )


@cuda
@pytest.mark.parametrize(
    "value_type", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_reset_writes_fp32_initial_value(value_type):
    emb_dim = 7
    initial = 3.0e-9
    optimizer = RowWiseAdaGradDynamicEmbeddingOptimizer(
        OptimizerArgs(learning_rate=0.01, initial_accumulator_value=initial),
        value_type,
    )
    rows = torch.zeros(
        4, emb_dim + optimizer.get_state_dim(emb_dim), dtype=value_type, device="cuda"
    )
    optimizer.reset_optimizer_states(
        rows[:, emb_dim:], indices=torch.tensor([1, 3], device="cuda")
    )

    accumulator = optimizer.states_for_checkpoint(rows[:, emb_dim:], emb_dim)
    expected = torch.tensor([[0.0], [initial], [0.0], [initial]], device="cuda")
    assert torch.equal(accumulator, expected.float())
