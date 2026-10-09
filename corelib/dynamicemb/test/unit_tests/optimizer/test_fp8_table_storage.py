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

"""Row-wise Adagrad on fp8 tables.

An fp8 table stores e4m3 codes of ``value * 256`` and rounds every write
stochastically. These tests pin what that buys: the fp32 accumulator is the same
one an fp32 table keeps, weights track an fp32 table to fp8 precision, the
rounding is unbiased, and optimizer steps smaller than half an fp8 step still
move the weights instead of being rounded away.
"""

import pytest
import torch
from dynamicemb.optimizer import OptimizerArgs, RowWiseAdaGradDynamicEmbeddingOptimizer
from dynamicemb.utils import FP8_STORAGE_DTYPE, decode_table_values, encode_table_values

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

EPS = 1e-8


def _optimizer(
    value_type: torch.dtype, learning_rate: float
) -> RowWiseAdaGradDynamicEmbeddingOptimizer:
    return RowWiseAdaGradDynamicEmbeddingOptimizer(
        OptimizerArgs(
            learning_rate=learning_rate, eps=EPS, initial_accumulator_value=0.0
        ),
        value_type,
    )


def _train_padded_buffer(
    value_type: torch.dtype,
    init: torch.Tensor,
    grads,
    all_dims_vec4: bool,
    learning_rate: float = 0.01,
):
    """Apply every gradient in turn to one padded buffer; return (fp32 weights, accumulator)."""
    device = torch.device("cuda")
    num_rows, emb_dim = init.shape
    optimizer = _optimizer(value_type, learning_rate)
    value_dim = emb_dim + optimizer.get_state_dim(emb_dim)

    values = torch.empty(num_rows, value_dim, dtype=value_type, device=device)
    values[:, :emb_dim] = encode_table_values(init.to(device), value_type)
    optimizer.reset_optimizer_states(values[:, emb_dim:], emb_dims=emb_dim)

    table_ids = torch.zeros(num_rows, dtype=torch.int64, device=device)
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
    weights = decode_table_values(values[:, :emb_dim], torch.float32)
    return weights, accumulator


def _on_fp8_grid(values: torch.Tensor) -> torch.Tensor:
    return decode_table_values(
        encode_table_values(values.cuda(), FP8_STORAGE_DTYPE), torch.float32
    )


def _random_case(num_rows: int, emb_dim: int, num_steps: int, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    init = _on_fp8_grid(
        (torch.rand(num_rows, emb_dim, generator=generator) - 0.5) * 0.1
    )
    grads = [
        (torch.randn(num_rows, emb_dim, generator=generator) * 1e-3).cuda()
        for _ in range(num_steps)
    ]
    return init, grads


@cuda
@pytest.mark.parametrize(
    "emb_dim, all_dims_vec4",
    [(16, True), (16, False), (7, False), (13, False)],
    ids=["vec4", "scalar", "odd7", "odd13"],
)
def test_fp8_accumulator_matches_fp32_table(emb_dim, all_dims_vec4):
    init, grads = _random_case(64, emb_dim, 50)
    _, ref_accumulator = _train_padded_buffer(torch.float32, init, grads, all_dims_vec4)
    _, accumulator = _train_padded_buffer(FP8_STORAGE_DTYPE, init, grads, all_dims_vec4)

    assert accumulator.dtype == torch.float32
    assert torch.all(accumulator > 0), "accumulator underflowed to zero"
    assert torch.equal(accumulator, ref_accumulator)


@cuda
@pytest.mark.parametrize("all_dims_vec4", [True, False], ids=["vec4", "scalar"])
def test_fp8_weights_track_fp32_table(all_dims_vec4):
    init, grads = _random_case(256, 16, 50)
    ref_weights, _ = _train_padded_buffer(torch.float32, init, grads, all_dims_vec4)
    weights, _ = _train_padded_buffer(FP8_STORAGE_DTYPE, init, grads, all_dims_vec4)

    error = weights - ref_weights
    assert error.mean().abs() < 4 * error.std() / error.numel() ** 0.5
    assert error.pow(2).mean().sqrt() < 0.5 * ref_weights.pow(2).mean().sqrt()


@cuda
def test_fp8_stochastic_rounding_is_unbiased():
    num_rows, emb_dim = 4096, 16
    init = _on_fp8_grid(torch.full((num_rows, emb_dim), 0.05))
    grads = [torch.full((num_rows, emb_dim), 1e-3, device="cuda")]
    ref_weights, _ = _train_padded_buffer(torch.float32, init, grads, True)
    weights, _ = _train_padded_buffer(FP8_STORAGE_DTYPE, init, grads, True)

    target = ref_weights[0, 0].item()
    neighbours = torch.unique(weights)
    assert neighbours.numel() == 2, f"expected the two fp8 neighbours, got {neighbours}"
    assert neighbours[0].item() < target < neighbours[1].item()
    spacing = (neighbours[1] - neighbours[0]).item()
    tolerance = 4 * spacing / (2 * weights.numel() ** 0.5)
    assert abs(weights.mean().item() - target) < tolerance


@cuda
def test_fp8_steps_below_half_an_fp8_step_still_accumulate():
    num_rows, emb_dim, num_steps = 1024, 16, 200
    init = _on_fp8_grid(torch.full((num_rows, emb_dim), 0.05))
    grads = [torch.full((num_rows, emb_dim), 1e-2, device="cuda")] * num_steps
    ref_weights, _ = _train_padded_buffer(
        torch.float32, init, grads, True, learning_rate=1e-3
    )
    weights, _ = _train_padded_buffer(
        FP8_STORAGE_DTYPE, init, grads, True, learning_rate=1e-3
    )

    ref_drift = (ref_weights - init.cuda()).mean().item()
    drift = (weights - init.cuda()).mean().item()
    assert ref_drift < -0.02
    assert abs(drift - ref_drift) < 0.05 * abs(ref_drift)


@cuda
def test_fp8_requires_an_fp32_accumulator():
    with pytest.raises(ValueError, match="fp8"):
        RowWiseAdaGradDynamicEmbeddingOptimizer(
            OptimizerArgs(
                learning_rate=0.01,
                eps=EPS,
                initial_accumulator_value=0.0,
                optimizer_state_dtype=FP8_STORAGE_DTYPE,
            ),
            FP8_STORAGE_DTYPE,
        ).get_state_dtype(FP8_STORAGE_DTYPE)
