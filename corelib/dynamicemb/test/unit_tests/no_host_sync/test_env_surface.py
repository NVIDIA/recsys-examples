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

"""The environment surface of the lookup paths, and the chunk plan it feeds.

No environment variable selects a blocking path. What remains are tuning knobs that never change
what a step reads from the device: ``DYNAMICEMB_MISS_PATH_GATE`` (how the miss path is launched)
and ``DYNAMICEMB_MISS_PATH_BUFFER_BYTES`` (how many rows of the miss path's buffers one chunk
allocates). This file checks the parsing of the second and the arithmetic of the chunk plan
(``sync_free.row_chunks``), which is pure host code and runs without a GPU.
"""
import warnings

import pytest

try:
    from dynamicemb import sync_free as sf
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _plain_budget(monkeypatch):
    """Every test starts from the default budget and no forced chunk size."""
    monkeypatch.setattr(sf, "BUFFER_BYTES", 4 * GIB)
    monkeypatch.setattr(sf, "CHUNK_ROWS", None)
    monkeypatch.setattr(sf, "_WARNED_SMALL_CHUNK", False)


# --------------------------------------------------------------------------
# DYNAMICEMB_MISS_PATH_BUFFER_BYTES
# --------------------------------------------------------------------------
def test_the_default_budget_is_four_gib(monkeypatch):
    monkeypatch.delenv("DYNAMICEMB_MISS_PATH_BUFFER_BYTES", raising=False)
    assert sf._buffer_bytes_from_env() == 4 * GIB
    monkeypatch.setenv("DYNAMICEMB_MISS_PATH_BUFFER_BYTES", "")
    assert sf._buffer_bytes_from_env() == 4 * GIB


@pytest.mark.parametrize(
    "text, value", [("1", 1), ("1048576", 1 << 20), ("17179869184", 16 * GIB)]
)
def test_the_budget_is_a_positive_number_of_bytes(monkeypatch, text, value):
    monkeypatch.setenv("DYNAMICEMB_MISS_PATH_BUFFER_BYTES", text)
    assert sf._buffer_bytes_from_env() == value


@pytest.mark.parametrize("text", ["0", "-5", "4G", "1.5", "lots"])
def test_a_budget_that_is_not_a_positive_number_is_refused_loudly(monkeypatch, text):
    monkeypatch.setenv("DYNAMICEMB_MISS_PATH_BUFFER_BYTES", text)
    with pytest.raises(ValueError, match="DYNAMICEMB_MISS_PATH_BUFFER_BYTES"):
        sf._buffer_bytes_from_env()


# --------------------------------------------------------------------------
# the chunk plan
# --------------------------------------------------------------------------
def _assert_covers(chunks, n):
    assert chunks[0][0] == 0 and chunks[-1][1] == n
    assert all(a < b for a, b in chunks) or (n == 0 and chunks == [(0, 0)])
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))


@pytest.mark.parametrize("n", [0, 1, 7, 64, 100, 1000])
@pytest.mark.parametrize("row_bytes", [4, 128, 1536])
@pytest.mark.parametrize("budget", [1, 4096, 1 << 20, 4 * GIB])
@pytest.mark.parametrize("buffers", [1, 2, 3])
def test_the_chunks_cover_the_rows_in_order_without_an_empty_one(
    monkeypatch, n, row_bytes, budget, buffers
):
    monkeypatch.setattr(sf, "BUFFER_BYTES", budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = sf.row_chunks(n, row_bytes, buffers)
    _assert_covers(chunks, n)
    longest = max(b - a for a, b in chunks)
    assert (
        longest == 1 or buffers * longest * row_bytes <= budget
    ), "a chunk fits the budget"
    if len(chunks) > 1:  # no chunk but the last is shorter than the others
        assert (
            len({b - a for a, b in chunks[:-1]}) == 1
            and chunks[-1][1] - chunks[-1][0] <= longest
        )


@pytest.mark.parametrize(
    "n, value_dim, itemsize",
    [(1, 384, 4), (1000, 128, 4), (4_000_000, 128, 4), (2_796_202, 384, 4)],
)
def test_a_batch_the_old_guard_let_through_is_one_chunk(n, value_dim, itemsize):
    """The memory guard of the first masked paths allowed ``2 * n * value_dim * itemsize <=
    4 GiB``; those batches are one chunk, the very kernels of an unchunked step."""
    row_bytes = value_dim * itemsize
    fits_old_guard = 2 * n * row_bytes <= 4 * GIB
    chunks = sf.row_chunks(n, row_bytes, buffers=2)
    assert (len(chunks) == 1) == fits_old_guard
    if not fits_old_guard:
        assert max(b - a for a, b in chunks) * 2 * row_bytes <= 4 * GIB


def test_a_small_budget_still_makes_progress_and_warns_once(monkeypatch):
    monkeypatch.setattr(sf, "BUFFER_BYTES", 10)  # less than one row
    with pytest.warns(UserWarning, match="DYNAMICEMB_MISS_PATH_BUFFER_BYTES=10"):
        chunks = sf.row_chunks(5, 64, buffers=2)
    assert chunks == [(i, i + 1) for i in range(5)], "at least one row per chunk"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the second time it is silent
        sf.row_chunks(5, 64, buffers=2)


def test_the_test_hook_forces_the_chunk_size_and_never_warns(monkeypatch):
    monkeypatch.setattr(sf, "CHUNK_ROWS", 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sf.row_chunks(8, 10**9) == [(0, 3), (3, 6), (6, 8)]
        assert sf.row_chunks(2, 10**9) == [(0, 2)]
    monkeypatch.setattr(
        sf, "CHUNK_ROWS", 0
    )  # 0 and None both mean "the budget decides"
    assert sf.row_chunks(8, 4) == [(0, 8)]


def test_the_plan_depends_on_host_facts_only():
    """The same arguments, the same plan: nothing but the arguments and the budget is read."""
    first = sf.row_chunks(12345, 1536, buffers=3)
    assert first == sf.row_chunks(12345, 1536, buffers=3)
    assert all(isinstance(a, int) and isinstance(b, int) for a, b in first)
