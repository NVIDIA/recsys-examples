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

"""Optimizer kernels read the right gradient and write the right weights when
either one is addressed past 2**32 elements.

A row's offset is ``row * stride``, and ``row`` and ``stride`` were both 32-bit
in the optimizer kernels, so the product wrapped at 2**32 elements: a batch of
2**20 unique rows of width 4096 already reaches it. Every row from there on
read the gradient of, and in the padded buffer wrote the weights of, the row
2**20 before it, which silently trains the wrong rows.

The gradients and the weights are addressed by separate products, so each gets
a case where only it is that large, and a wrapped read or write cannot be
blamed on the other:

* gradients large, weights small: the flat-table kernels, whose table rows can
  be narrower than the gradient rows that update them;
* weights large, gradients small: the padded-buffer kernels, whose value rows
  carry optimizer state and are wider than the gradient rows;
* both large: the padded-buffer kernels with value rows as wide as gradient
  rows, the only way to reach a padded-buffer gradient address past 2**32,
  since a value row is never narrower than its gradient row.

The gradient of row ``r`` is made to differ from its wrapped neighbour's, so a
wrapped read or write cannot pass as the right one. A case holds up to 17 GB
and is skipped on a GPU without room for it.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

from dynamicemb.utils import torch_to_dyn_emb  # noqa: E402

# 2**20 rows of 4096 elements is exactly 2**32 elements; go past it.
ROWS = (1 << 20) + 16
PATTERN = 7  # row r's gradient is r % PATTERN + 1, and 2**20 % 7 != 0
LR = 1.0

# (path, gradient row width, embedding width, value row width). The gradient
# rows are ROWS * the first width long, the weights ROWS * the last.
CASES = [
    pytest.param("flat", 4096, 8, 8, id="flat-large_gradients-small_weights"),
    pytest.param("padded", 1024, 1024, 4096, id="padded-small_gradients-large_weights"),
    pytest.param("padded", 4096, 4096, 4096, id="padded-large_gradients-large_weights"),
]


def _update(path, grads, values, emb_dim, all_dims_vec4):
    device = grads.device
    table_ids = torch.zeros(ROWS, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)
    if path == "flat":
        dynamicemb_extensions.sgd_update_for_flat_table(
            grads,
            torch.arange(ROWS, dtype=torch.int64, device=device),
            torch.tensor([values.data_ptr()], dtype=torch.int64, device=device),
            table_ids,
            torch.tensor([values.size(1)], dtype=torch.int64, device=device),
            table_emb_dims,
            grads.size(1),
            all_dims_vec4,
            LR,
            torch_to_dyn_emb(values.dtype).value,
        )
    else:
        dynamicemb_extensions.sgd_update_for_padded_buffer(
            grads,
            values,
            table_ids,
            table_emb_dims,
            emb_dim,
            values.size(1),
            all_dims_vec4,
            LR,
        )


@cuda
@pytest.mark.parametrize("all_dims_vec4", [True, False], ids=["vec4", "scalar"])
@pytest.mark.parametrize("path, grad_width, emb_dim, value_dim", CASES)
def test_sgd_addresses_rows_past_2_pow_32_elements(
    path, grad_width, emb_dim, value_dim, all_dims_vec4
):
    needed_gib = ROWS * (grad_width + value_dim) * 2 / 2**30 + 2
    if torch.cuda.mem_get_info()[0] < needed_gib * 2**30:
        pytest.skip(f"needs {needed_gib:.0f} GiB of free GPU memory")
    device = torch.device("cuda")

    # What row r's gradient is, and so how far SGD moves its weights.
    step = (torch.arange(ROWS, device=device) % PATTERN + 1).to(torch.float16)
    grads = step[:, None].expand(ROWS, grad_width).contiguous()
    values = torch.zeros(ROWS, value_dim, dtype=torch.float16, device=device)

    _update(path, grads, values, emb_dim, all_dims_vec4)
    torch.cuda.synchronize()

    # Row r's first ``emb_dim`` weights moved by exactly its own step ...
    for column in (0, emb_dim // 2, emb_dim - 1):
        torch.testing.assert_close(values[:, column], -step, rtol=0, atol=0)
    # ... and the rest of its value row, which the optimizer never writes, is
    # still zero, so no write landed in another row.
    if value_dim > emb_dim:
        for column in (emb_dim, value_dim - 1):
            assert not values[:, column].any(), f"column {column} was written"
