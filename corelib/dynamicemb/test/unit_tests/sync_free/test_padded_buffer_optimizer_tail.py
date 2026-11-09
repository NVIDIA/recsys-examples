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

"""The optimizer kernels skip the padded tail of a unique batch.

The values and gradients of a batch of unique keys are allocated at the batch's
upper bound, so the rows past the number of unique keys are padding. Such a row
has table id -1 (see ``test_padded_tail.py``) or, for the flat-table kernels,
index -1, and the kernels must leave it alone: no read of the per-table dims
through the id, no write. These tests run every optimizer, with the vectorized
and the scalar kernel, on a batch whose padding holds a poison value, and check
that

* the padding is bit-for-bit what it was (and so is the memory just after it,
  where a padding row that was processed with a wrong width would write);
* the real rows come out bit-for-bit as when the same optimizer gets the real
  rows alone.

The padded-buffer kernels run with the padding at the end of the batch, as the
unique op leaves it, and with a padding row before, between and after the real
rows. The flat kernels get the same two layouts, with index -1 for the padding.

The last tests are about the offsets of a row in the gradient and value arrays,
which the kernels compute in 64 bits: past 2**32 elements a 32-bit product
wraps around to an early row. They need about 9 GiB (the padded-buffer one 18)
of GPU memory and are skipped without it.
"""

import pytest
import torch

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

from dynamicemb.optimizer import (
    AdaGradDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizer,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
)

# What the padding holds. No update of a finite, positive-state row produces
# it, and it is exact in float32 and in bfloat16 (-77824 = -152 * 512).
POISON = -77824.0
# The table-dim entry in front of the real ones: what a padding row would read
# if its table id of -1 were used to index the dims array. Wider than any real
# dim, but not so wide that such a row would write beyond the spare rows.
BAD_DIM = 64

OPTIMIZERS = [
    "sgd",
    "adam",
    "adagrad",
    "rowwise_adagrad",
    "ftrl",
    "ftrl_group",
]


def _make_optimizer(name, dtype):
    if name == "sgd":
        return SGDDynamicEmbeddingOptimizer(OptimizerArgs(learning_rate=0.1))
    if name == "adam":
        optimizer = AdamDynamicEmbeddingOptimizer(
            OptimizerArgs(learning_rate=0.1, weight_decay=0.01)
        )
        optimizer.step()  # the bias correction divides by 1 - beta ** iterations
        return optimizer
    if name == "adagrad":
        return AdaGradDynamicEmbeddingOptimizer(OptimizerArgs(learning_rate=0.1))
    if name == "rowwise_adagrad":
        return RowWiseAdaGradDynamicEmbeddingOptimizer(
            OptimizerArgs(learning_rate=0.1), dtype
        )
    if name == "ftrl":
        return FTRLDynamicEmbeddingOptimizer(
            OptimizerArgs(learning_rate=0.1, l1_reg=0.01, l2_reg=0.01)
        )
    if name == "ftrl_group":
        # The group-lasso variant reduces over a row inside the kernel: the skip
        # of a row has to be taken by all of its threads.
        return FTRLDynamicEmbeddingOptimizer(
            OptimizerArgs(learning_rate=0.1, l1_reg=0.01, group_lasso_reg=0.05)
        )
    raise AssertionError(name)


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def test_extension_announces_the_skip():
    """Callers that fall back without the skip ask for this attribute
    (``getattr(dynamicemb_extensions, NAME, False)``)."""
    assert hasattr(dynamicemb_extensions, "padded_optimizer_skips_negative_table_id")
    assert dynamicemb_extensions.padded_optimizer_skips_negative_table_id is True


def _bits(tensor):
    """The bit pattern of a float tensor: equal bits also for NaN."""
    return tensor.contiguous().view(
        torch.int32 if tensor.element_size() == 4 else torch.int16
    )


def _same(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _positive(shape, dtype, seed):
    """Random values in [0.5, 1.5): a valid state, whatever the optimizer keeps."""
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, generator=gen) + 0.5).to(device=_device(), dtype=dtype)


def _normal(shape, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device=_device(), dtype=dtype)


def _all_dims_vec4(dims, value_dims):
    return all(d % 4 == 0 for d in dims) and all(v % 4 == 0 for v in value_dims)


def _positions(layout, num_real):
    """Rows of the batch the real rows go to, and the batch length."""
    if layout == "tail":
        return torch.arange(num_real), num_real + 5
    # A padding row before, between and after the real rows.
    return torch.arange(num_real) * 2 + 1, 2 * num_real + 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["tail", "interleaved"])
@pytest.mark.parametrize("dims", [[8, 4], [8, 6]], ids=["vec4", "scalar"])
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_padded_buffer_kernels_skip_rows_without_a_table(name, dims, layout, dtype):
    optimizer = _make_optimizer(name, dtype)
    max_emb_dim = max(dims)
    value_dim = max_emb_dim + optimizer.get_state_dim(max_emb_dim)
    all_dims_vec4 = _all_dims_vec4(dims, [value_dim])
    num_real = 6
    real_ids = torch.tensor([0, 1, 0, 1, 1, 0], device=_device())
    real_values = _positive((num_real, value_dim), dtype, seed=1)
    real_grads = _normal((num_real, max_emb_dim), dtype, seed=2)
    dims_device = torch.tensor(dims, dtype=torch.int64, device=_device())

    # The run on the real rows alone.
    exact_values = real_values.clone()
    optimizer.update_for_padded_buffer(
        real_grads,
        exact_values,
        real_ids,
        dims_device,
        max_emb_dim,
        value_dim,
        all_dims_vec4,
    )
    assert not _same(exact_values, real_values), "the optimizer did not update"

    # The padded batch: real rows where `layout` puts them, poison elsewhere,
    # table id -1 for the poison rows, and spare rows after the batch for a
    # row processed with the wrong width to write into.
    real_positions, num_rows = _positions(layout, num_real)
    spare = 16
    values = torch.full(
        (num_rows + spare, value_dim), POISON, dtype=dtype, device=_device()
    )
    grads = torch.full(
        (num_rows + spare, max_emb_dim), POISON, dtype=dtype, device=_device()
    )
    values[real_positions.to(_device())] = real_values
    grads[real_positions.to(_device())] = real_grads
    table_ids = torch.full((num_rows,), -1, dtype=torch.int64, device=_device())
    table_ids[real_positions.to(_device())] = real_ids
    # Indexing the dims with -1 reads BAD_DIM instead of a table's dim.
    dims_with_bad = torch.tensor([BAD_DIM] + dims, dtype=torch.int64, device=_device())
    before_values, before_grads = values.clone(), grads.clone()

    optimizer.update_for_padded_buffer(
        grads[:num_rows],
        values[:num_rows],
        table_ids,
        dims_with_bad[1:],
        max_emb_dim,
        value_dim,
        all_dims_vec4,
    )
    torch.cuda.synchronize()

    is_real = torch.zeros(num_rows + spare, dtype=torch.bool, device=_device())
    is_real[real_positions.to(_device())] = True
    assert _same(values[is_real], exact_values), "a real row changed"
    assert _same(values[~is_real], before_values[~is_real]), "padding was written"
    assert _same(grads, before_grads), "a gradient was written"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["tail", "interleaved"])
@pytest.mark.parametrize("dims", [[8, 4], [8, 6]], ids=["vec4", "scalar"])
@pytest.mark.parametrize("name", OPTIMIZERS)
def test_flat_table_kernels_skip_index_minus_one(name, dims, layout, dtype):
    optimizer = _make_optimizer(name, dtype)
    max_emb_dim = max(dims)
    value_dims = [d + optimizer.get_state_dim(d) for d in dims]
    all_dims_vec4 = _all_dims_vec4(dims, value_dims)
    rows_per_table = 16
    # The real entries: five distinct rows of each table.
    num_real = 10
    real_ids = torch.tensor([0, 1] * 5, device=_device())
    real_rows = torch.tensor([0, 0, 1, 3, 2, 1, 3, 2, 4, 4], device=_device())
    real_grads = _normal((num_real, max_emb_dim), dtype, seed=2)
    start = [
        _positive((rows_per_table, v), dtype, seed=10 + t)
        for t, v in enumerate(value_dims)
    ]

    def run(indices, table_ids, grads):
        tables = [table.clone() for table in start]
        pointers = torch.tensor(
            [table.data_ptr() for table in tables], dtype=torch.int64, device=_device()
        )
        optimizer.fused_update_for_flat_table(
            grads,
            indices,
            pointers,
            table_ids,
            torch.tensor(value_dims, dtype=torch.int64, device=_device()),
            torch.tensor(dims, dtype=torch.int64, device=_device()),
            max_emb_dim,
            all_dims_vec4,
            dtype,
        )
        torch.cuda.synchronize()
        return tables

    exact = run(real_rows, real_ids, real_grads)
    assert not _same(exact[0], start[0]), "the optimizer did not update"

    real_positions, num_entries = _positions(layout, num_real)
    real_positions = real_positions.to(_device())
    indices = torch.full((num_entries,), -1, dtype=torch.int64, device=_device())
    # The table id of an entry without a row is garbage the kernel must not read.
    table_ids = torch.full((num_entries,), 987_654, dtype=torch.int64, device=_device())
    grads = torch.full(
        (num_entries, max_emb_dim), POISON, dtype=dtype, device=_device()
    )
    indices[real_positions] = real_rows
    table_ids[real_positions] = real_ids
    grads[real_positions] = real_grads
    padded = run(indices, table_ids, grads)

    for table, expected in zip(padded, exact):
        assert _same(table, expected)


def _free_gib():
    return torch.cuda.mem_get_info()[0] / 2**30


def _require_gib(gib):
    if _free_gib() < gib:
        pytest.skip(f"needs about {gib} GiB of free GPU memory")


@pytest.mark.parametrize("vec4", [True, False], ids=["vec4", "scalar"])
def test_flat_table_gradient_offset_past_2_pow_32_elements(vec4):
    """The gradient of entry ``i`` is at ``i * grad_stride``: 64-bit arithmetic
    keeps the entries past element 2**32 on their own gradient and not on one of
    the first rows."""
    _require_gib(11)
    dim = 1024
    # The first entry whose offset needs more than 32 bits, and 16 after it.
    num_entries = 2**32 // dim + 16
    device = _device()
    grads = torch.empty(num_entries, dim, dtype=torch.bfloat16, device=device)
    grads[:32] = 100.0  # what a 32-bit offset would read
    grads[-8:] = torch.arange(1, 9, dtype=torch.bfloat16, device=device)[:, None]
    indices = torch.full((num_entries,), -1, dtype=torch.int64, device=device)
    indices[-8:] = torch.arange(8, device=device)
    table = torch.ones(8, dim, dtype=torch.bfloat16, device=device)

    SGDDynamicEmbeddingOptimizer(
        OptimizerArgs(learning_rate=0.25)
    ).fused_update_for_flat_table(
        grads,
        indices,
        torch.tensor([table.data_ptr()], dtype=torch.int64, device=device),
        torch.zeros(num_entries, dtype=torch.int64, device=device),
        torch.tensor([dim], dtype=torch.int64, device=device),
        torch.tensor([dim], dtype=torch.int64, device=device),
        dim,
        vec4,
        torch.bfloat16,
    )
    torch.cuda.synchronize()
    expected = 1.0 - 0.25 * torch.arange(1, 9, device=device)
    assert torch.equal(table.float(), expected[:, None].expand(-1, dim))


@pytest.mark.parametrize("vec4", [True, False], ids=["vec4", "scalar"])
def test_padded_buffer_offsets_past_2_pow_32_elements(vec4):
    """Row ``r`` of the values and of the gradients is at ``r * stride``, in 64
    bits: the real rows at the end of a long, otherwise padded, batch are
    updated in place and the first rows are not."""
    _require_gib(20)
    dim = 1024
    num_rows = 2**32 // dim + 16
    device = _device()
    values = torch.empty(num_rows, dim, dtype=torch.bfloat16, device=device)
    grads = torch.empty(num_rows, dim, dtype=torch.bfloat16, device=device)
    values[:32] = POISON  # what a row wrapped around to would be written into
    grads[:32] = 100.0
    values[-8:] = 1.0
    grads[-8:] = torch.arange(1, 9, dtype=torch.bfloat16, device=device)[:, None]
    table_ids = torch.full((num_rows,), -1, dtype=torch.int64, device=device)
    table_ids[-8:] = 0
    # A kernel that did not skip the padding rows would read BAD_DIM for them
    # (and be done in a moment) instead of whatever precedes a lone dims array.
    dims = torch.tensor([BAD_DIM, dim], dtype=torch.int64, device=device)

    SGDDynamicEmbeddingOptimizer(
        OptimizerArgs(learning_rate=0.25)
    ).update_for_padded_buffer(grads, values, table_ids, dims[1:], dim, dim, vec4)
    torch.cuda.synchronize()
    expected = 1.0 - 0.25 * torch.arange(1, 9, device=device)
    assert torch.equal(values[-8:].float(), expected[:, None].expand(-1, dim))
    assert bool((values[:32] == POISON).all()), "a row wrapped around was written"
