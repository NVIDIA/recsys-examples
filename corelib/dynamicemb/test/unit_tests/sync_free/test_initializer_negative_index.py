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

"""An initializer skips an index of -1.

``-1`` in the ``indices`` an initializer gets selects no row (it is how a
padded tail, or a key that is masked out, is written down), so the row is left
alone: nothing is written to it, and nothing is read through it -- the
per-table generators read ``table_ids[index]`` and the debug one ``keys[index]``.

Every initializer mode is run in both its forms -- the plain one, whose
parameters are shared by all tables, and the ``_table_params`` one, whose
parameters differ per table -- on a buffer filled with a poison value that no
mode can produce, through an index vector with -1 in front, between the rows and
at the end. What must hold afterwards: every row that was not selected is still
the poison, as are the rows below the buffer (which is where an index of -1
would have written) and the columns beside the rows' embedding (the buffer is a
view into wider rows, as the value buffer of an optimizer-carrying table is),
and the selected rows are what the mode says.
"""

import pytest
import torch

dynamicemb_extensions = pytest.importorskip("dynamicemb_extensions")

from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.types import DynamicEmbInitializerArgs, DynamicEmbInitializerMode

MODE = DynamicEmbInitializerMode
# What every row not written must still hold. Negative, so neither a debug
# value (key % mod) nor any of the ranges below can equal it, and representable
# in float32 and bfloat16 alike (-77824 = -152 * 512).
POISON = -77824.0

NUM_ROWS = 48
EMB_DIM = 64
# The buffer rows are wider than the embedding: the initializer gets a view of
# the first EMB_DIM columns, as it does for a table that keeps optimizer state.
ROW_WIDTH = EMB_DIM + 16
GUARD_ROWS = 4  # rows below the buffer, where an index of -1 would write
NUM_TABLES = 2

# Parameters of the two tables, one set shared by both and one set per table.
# The per-table ones are far apart, so a row written with the wrong table's
# parameters is caught.
ARGS = {
    MODE.NORMAL: dict(
        shared=[dict(mean=2.0, std_dev=0.5)] * 2,
        per_table=[dict(mean=2.0, std_dev=0.5), dict(mean=-30.0, std_dev=2.0)],
    ),
    MODE.TRUNCATED_NORMAL: dict(
        shared=[dict(mean=0.0, std_dev=1.0, lower=-0.5, upper=0.75)] * 2,
        per_table=[
            dict(mean=0.0, std_dev=1.0, lower=-0.5, upper=0.75),
            dict(mean=10.0, std_dev=3.0, lower=8.0, upper=11.0),
        ],
    ),
    MODE.UNIFORM: dict(
        shared=[dict(lower=-0.25, upper=0.25)] * 2,
        per_table=[dict(lower=-0.25, upper=0.25), dict(lower=5.0, upper=6.0)],
    ),
    MODE.CONSTANT: dict(
        shared=[dict(value=1.5)] * 2,
        per_table=[dict(value=1.5), dict(value=-2.5)],
    ),
    MODE.DEBUG: dict(shared=[dict()] * 2, per_table=None),
}
CASES = [
    pytest.param(mode, form, id=f"{mode.name.lower()}-{form}")
    for mode, forms in ARGS.items()
    for form in ("shared", "per_table")
    if forms[form] is not None
]


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")


def test_extension_announces_the_skip():
    """Callers that fall back without the skip ask for this attribute
    (``getattr(dynamicemb_extensions, NAME, False)``)."""
    assert hasattr(dynamicemb_extensions, "initializer_skips_negative_index")
    assert dynamicemb_extensions.initializer_skips_negative_index is True


def _selection(device, seed):
    """An index vector: -1 first and last, -1 between the real rows, and the
    rows it selects (each once). Returns ``(indices, selected_rows)``."""
    gen = torch.Generator().manual_seed(seed)
    num_entries = 32
    rows = torch.randperm(NUM_ROWS, generator=gen)[:20]
    indices = torch.full((num_entries,), -1, dtype=torch.int64)
    slots = 1 + torch.randperm(num_entries - 2, generator=gen)[: rows.numel()]
    indices[slots] = rows
    assert indices[0] == -1 and indices[-1] == -1
    assert int((indices == -1).sum()) >= 8
    return indices.to(device), rows.to(device)


def _check_values(mode, form, written, row_table, dtype):
    """``written``: the initialized rows, ``row_table``: their table ids."""
    args = ARGS[mode][form]
    assert bool(torch.isfinite(written).all())
    assert bool((written != POISON).all()), "a selected row was not written"
    values = written.float()
    for table in range(NUM_TABLES):
        rows = values[row_table == table]
        if rows.numel() == 0:
            continue
        params = args[table]
        if mode == MODE.CONSTANT:
            assert bool((rows == params["value"]).all())
        elif mode == MODE.UNIFORM:
            assert bool((rows >= params["lower"]).all())
            assert bool((rows <= params["upper"]).all())
        elif mode == MODE.TRUNCATED_NORMAL:
            assert bool((rows >= params["lower"]).all())
            assert bool((rows <= params["upper"]).all())
        elif mode == MODE.NORMAL:
            # A draw is within 8 sigma with all but 1e-15 probability.
            assert bool(((rows - params["mean"]).abs() <= 8 * params["std_dev"]).all())
            if dtype == torch.float32:
                # And the mean of the writes is the mean asked for.
                tolerance = 6 * params["std_dev"] / rows.numel() ** 0.5
                assert abs(float(rows.mean()) - params["mean"]) <= tolerance
                assert float(rows.std()) == pytest.approx(params["std_dev"], rel=0.15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode, form", CASES)
def test_initializer_skips_minus_one(mode, form, dtype):
    device = torch.device("cuda", torch.cuda.current_device())
    args_list = [
        DynamicEmbInitializerArgs(mode=mode, **params) for params in ARGS[mode][form]
    ]
    initializer = MultiTableInitializer.create(args_list, device)

    # The table of every buffer row, and the key of every buffer row (what the
    # per-table and the debug generators read through the index).
    table_ids = (torch.arange(NUM_ROWS) % NUM_TABLES).to(device)
    keys = (100_003 + 7_919 * torch.arange(NUM_ROWS, dtype=torch.int64)).to(device)

    indices, selected = _selection(device, seed=list(MODE).index(mode))
    storage = torch.full(
        (GUARD_ROWS + NUM_ROWS, ROW_WIDTH), POISON, dtype=dtype, device=device
    )
    buffer = storage[GUARD_ROWS:, :EMB_DIM]
    assert buffer.stride() == (ROW_WIDTH, 1)

    initializer(buffer, keys, table_ids, indices)
    torch.cuda.synchronize()

    is_selected = torch.zeros(NUM_ROWS, dtype=torch.bool, device=device)
    is_selected[selected] = True

    # Nothing outside the selected rows' embedding was written: not the rows
    # below the buffer, not the columns beside the embedding, not an unselected
    # row.
    assert bool((storage[:GUARD_ROWS] == POISON).all()), "wrote below the buffer"
    assert bool((storage[GUARD_ROWS:, EMB_DIM:] == POISON).all())
    assert bool((buffer[~is_selected] == POISON).all())

    written = buffer[selected]
    _check_values(mode, form, written, table_ids[selected], dtype)
    if mode == MODE.DEBUG:
        # A debug row is its key modulo 100000, exactly.
        expected = (keys[selected] % 100_000).to(torch.float32).to(dtype)
        assert torch.equal(written, expected[:, None].expand(-1, EMB_DIM))


@pytest.mark.parametrize("mode, form", CASES)
def test_initializer_with_only_minus_one_writes_nothing(mode, form):
    device = torch.device("cuda", torch.cuda.current_device())
    args_list = [
        DynamicEmbInitializerArgs(mode=mode, **params) for params in ARGS[mode][form]
    ]
    initializer = MultiTableInitializer.create(args_list, device)
    table_ids = (torch.arange(NUM_ROWS) % NUM_TABLES).to(device)
    keys = torch.arange(1, NUM_ROWS + 1, dtype=torch.int64, device=device)

    storage = torch.full(
        (GUARD_ROWS + NUM_ROWS, EMB_DIM), POISON, dtype=torch.float32, device=device
    )
    indices = torch.full((NUM_ROWS,), -1, dtype=torch.int64, device=device)
    initializer(storage[GUARD_ROWS:], keys, table_ids, indices)
    torch.cuda.synchronize()
    assert bool((storage == POISON).all())
