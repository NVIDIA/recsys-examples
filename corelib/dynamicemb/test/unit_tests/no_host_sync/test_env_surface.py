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
allocates). This file checks the parsing of both and the arithmetic of the chunk plan
(``sync_free.row_chunks``), which is pure host code and runs without a GPU.
"""
import os
import re
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
    monkeypatch.setattr(sf, "_WARNED_UNCHUNKED", set())


# --------------------------------------------------------------------------
# DYNAMICEMB_MISS_PATH_GATE
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, mode",
    [
        (None, "always"),
        ("", "always"),
        ("always", "always"),
        ("Always", "always"),
        ("poll", "poll"),
        ("POLL", "poll"),
    ],
)
def test_the_gate_variable_picks_one_of_two_modes(monkeypatch, text, mode):
    if text is None:
        monkeypatch.delenv("DYNAMICEMB_MISS_PATH_GATE", raising=False)
    else:
        monkeypatch.setenv("DYNAMICEMB_MISS_PATH_GATE", text)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sf._miss_gate_from_env() == mode
    assert sf._MISS_GATE_MODES == ("always", "poll"), "neither mode waits for the GPU"


def test_the_removed_read_mode_warns_and_the_default_is_used(monkeypatch):
    """A script that still asks for the blocking mode keeps running, without the wait."""
    monkeypatch.setenv("DYNAMICEMB_MISS_PATH_GATE", "read")
    with pytest.warns(UserWarning, match="DYNAMICEMB_MISS_PATH_GATE=read was removed"):
        assert sf._miss_gate_from_env() == "always"


@pytest.mark.parametrize("text", ["block", "sync", "0", "1", "never"])
def test_any_other_gate_value_is_refused_loudly(monkeypatch, text):
    monkeypatch.setenv("DYNAMICEMB_MISS_PATH_GATE", text)
    with pytest.raises(ValueError, match="DYNAMICEMB_MISS_PATH_GATE"):
        sf._miss_gate_from_env()


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


# --------------------------------------------------------------------------
# a buffer that is not split into chunks
# --------------------------------------------------------------------------
def test_an_unchunked_buffer_over_the_budget_is_named_once_per_site(monkeypatch):
    """Eval's unique rows and a DEFAULT-mode forward's value block are one row per key and cannot
    run in chunks: the budget does not bound them, so the step says so before it allocates.
    """
    monkeypatch.setattr(sf, "BUFFER_BYTES", 1000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf.warn_if_unchunked_over_budget(
            5, 100, "a pooled evaluation"
        )  # 2 x 500 <= 1000
    with pytest.warns(UserWarning, match="a pooled evaluation") as caught:
        sf.warn_if_unchunked_over_budget(6, 100, "a pooled evaluation")
    text = str(caught[0].message)
    assert (
        "6 rows of 100 bytes" in text
        and "DYNAMICEMB_MISS_PATH_BUFFER_BYTES=1000" in text
    )
    assert "not split into chunks" in text
    with warnings.catch_warnings():  # once per site and process
        warnings.simplefilter("error")
        sf.warn_if_unchunked_over_budget(60, 100, "a pooled evaluation")
    with pytest.warns(UserWarning, match="a forward"):
        sf.warn_if_unchunked_over_budget(6, 100, "a forward")


@pytest.mark.parametrize(
    "n, row_bytes, over",
    [
        (
            4 * (1 << 20),
            128 * 4,
            False,
        ),  # 4M keys of 128 fp32 columns: the old guard's edge
        (4 * (1 << 20) + 1, 128 * 4, True),
        (1_398_101, 384 * 4, False),  # Adam's value rows
        (1_398_102, 384 * 4, True),
    ],
)
def test_the_warning_starts_where_the_old_memory_guard_switched_paths(
    monkeypatch, n, row_bytes, over
):
    """The guard of the first masked paths sent a group with ``2 * n * row_bytes`` over 4 GiB down
    a path with ``O(unique keys)`` buffers; those batches now allocate ``O(n)`` and are named.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sf.warn_if_unchunked_over_budget(n, row_bytes, "a buffer")
    assert bool(caught) == over


# --------------------------------------------------------------------------
# what is not there any more
# --------------------------------------------------------------------------
def _package_sources():
    package = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "dynamicemb",
    )
    for dirpath, dirnames, filenames in os.walk(package):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                with open(path) as f:
                    yield os.path.relpath(path, package), f.read()


def test_no_variable_selects_a_blocking_path():
    """The kill switch (DYNAMICEMB_SYNC_FREE) and the old name of the budget
    (DYNAMICEMB_SYNC_FREE_MAX_BYTES) select nothing: a source search finds no use, whatever the
    file, but the table of retired variables that only warns (``sync_free._RETIRED_VARIABLES``).
    """
    forbidden = ("SYNC_FREE", "MAX_BYTES")
    found = []
    for path, source in _package_sources():
        if path == "sync_free.py":
            source = re.sub(
                r"\n_RETIRED_VARIABLES = \{.*?\n\}\n", "\n", source, flags=re.S
            )
        found += ["%s: %s" % (path, token) for token in forbidden if token in source]
    assert not found, "\n".join(found)


@pytest.mark.parametrize(
    "name", ["DYNAMICEMB_SYNC_FREE", "DYNAMICEMB_SYNC_FREE_MAX_BYTES"]
)
def test_a_retired_variable_warns_and_changes_nothing(monkeypatch, name):
    """A script that still sets one is told so, once, and everything runs as without it."""
    for other in ("DYNAMICEMB_MISS_PATH_GATE", "DYNAMICEMB_MISS_PATH_BUFFER_BYTES"):
        monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(name, "0")
    with pytest.warns(UserWarning, match="%s is not read any more" % name):
        sf._warn_retired_variables()
    assert sf._miss_gate_from_env() == "always"
    assert sf._buffer_bytes_from_env() == 4 * GIB
    monkeypatch.delenv(name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf._warn_retired_variables()


def test_the_legacy_bodies_are_not_in_the_package():
    """The host-sized lookup bodies live on only as the test oracle (``legacy_oracle``)."""
    import dynamicemb.batched_dynamicemb_function as bdf
    import dynamicemb.key_value_table as kvt

    for module, names in (
        (
            bdf,
            (
                "_prefetch_cache_path",
                "_prefetch_cache_path_legacy",
                "_prefetch_hbm_direct_path",
                "_prefetch_hbm_direct_path_legacy",
                "_generic_forward_path",
                "_generic_forward_path_legacy",
                "_apply_admission",
                "_lookup_next_tier",
                "_read_unique_offsets",
                "_masked_dispatch",
                "_SYNC_FREE",
                "_MAX_BYTES",
            ),
        ),
        (
            kvt,
            (
                "_select",
                "_eval_lookup_storage_legacy",
                "_eval_lookup_cached_legacy",
                "_masked_eval_applies",
            ),
        ),
    ):
        for name in names:
            assert not hasattr(module, name), "%s.%s is still there" % (
                module.__name__,
                name,
            )
