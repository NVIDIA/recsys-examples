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

These tests build the buffer by hand for two tables of different dims and check
that m and v land in the right place with the right update rule. They require a
GPU + the compiled ``dynamicemb_extensions`` and are skipped otherwise.
"""

import pytest
import torch

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
    value_dim = max_emb_dim + 2 * max_emb_dim  # adam state = 2 * max_emb_dim
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

    value_dim = max_emb_dim + 2 * max_emb_dim
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
