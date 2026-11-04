# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The calls a user makes: ``erase_by_score_async`` on a model, and the method of that
name on a module.

Which tables a strategy picks, what is refused before anything starts, and the use
the feature is for: a table that never evicts is filled, the old keys are erased by
their score, and new keys take the room.
"""

import warnings

import pytest
import torch
from dynamicemb import DynamicEmbScoreStrategy, erase_by_score_async
from dynamicemb.async_erase import EraseHandle, EraseHandleGroup
from dynamicemb.batched_dynamicemb_tables import StorageLayout
from dynamicemb.key_value_table import ERASE_CHUNK_SLOTS, ERASE_MAX_CHUNKS_IN_FLIGHT
from helpers import DIM, build_de_table, size, train_step
from torch import nn

S = DynamicEmbScoreStrategy


# ---------------------------------------------------------------------------
# Which tables a strategy picks, and what is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table_strategy, asked, expected",
    [
        (S.CUSTOMIZED, S.CUSTOMIZED, [0]),
        ((S.NO_EVICTION, S.CUSTOMIZED), S.CUSTOMIZED, [0]),
        ((S.CUSTOMIZED, S.NO_EVICTION), S.CUSTOMIZED, [0]),
        ((S.NO_EVICTION, S.PRECISE_CUSTOMIZED), S.PRECISE_CUSTOMIZED, [0]),
        (S.LFU, S.LFU, [0]),
        (S.TIMESTAMP, S.TIMESTAMP, [0]),
        # A member, not a family: the other strategy, or no score at all, is not asked for.
        (S.CUSTOMIZED, S.PRECISE_CUSTOMIZED, []),
        ((S.NO_EVICTION, S.CUSTOMIZED), S.PRECISE_CUSTOMIZED, []),
        (S.LFU, S.CUSTOMIZED, []),
        (S.NO_EVICTION, S.CUSTOMIZED, []),
    ],
)
def test_a_table_is_picked_when_its_strategy_has_the_one_asked_for_as_a_member(
    device, table_strategy, asked, expected
):
    module = build_de_table(device.index, table_strategy)
    assert module.erasable_table_ids(asked) == expected


@pytest.mark.parametrize(
    "asked, threshold, stream, error, match",
    [
        (S.NO_EVICTION, 10, "own", ValueError, "no score"),
        ("CUSTOMIZED", 10, "own", TypeError, "DynamicEmbScoreStrategy"),
        (S.CUSTOMIZED, -1, "own", ValueError, "threshold"),
        (S.CUSTOMIZED, 1 << 64, "own", ValueError, "threshold"),
        (S.CUSTOMIZED, 1.5, "own", TypeError, "threshold"),
        (S.CUSTOMIZED, True, "own", TypeError, "threshold"),
        (S.CUSTOMIZED, 10, "not a stream", TypeError, "torch.cuda.Stream"),
    ],
)
def test_what_cannot_be_asked_for_is_refused_by_both_calls_and_without_any_module(
    device, asked, threshold, stream, error, match
):
    """The arguments are checked whatever the tables are: by the call on a module, by
    the call on a model, and by the call on a model that has no module at all (which
    would otherwise answer with an empty result)."""
    if stream == "own":
        stream = torch.cuda.Stream()
    module = build_de_table(device.index, S.CUSTOMIZED)
    with pytest.raises(error, match=match):
        module.erase_by_score_async(asked, threshold, stream)
    with pytest.raises(error, match=match):
        erase_by_score_async(nn.ModuleList([module]), asked, threshold, stream)
    with pytest.raises(error, match=match):
        erase_by_score_async(nn.ModuleList(), asked, threshold, stream)


@pytest.mark.parametrize(
    "chunk_slots, max_in_flight, error, match",
    [
        (0, 2, ValueError, "chunk_slots"),
        (-4096, 2, ValueError, "chunk_slots"),
        (1.5, 2, TypeError, "chunk_slots"),
        (True, 2, TypeError, "chunk_slots"),
        (1 << 20, 0, ValueError, "max_in_flight"),
        (1 << 20, 2.0, TypeError, "max_in_flight"),
    ],
)
def test_the_chunk_and_the_limit_in_flight_are_checked_by_both_calls_at_once(
    device, chunk_slots, max_in_flight, error, match
):
    """Not left to the thread that scans, where the error would wait for the handle."""
    module = build_de_table(device.index, S.CUSTOMIZED)
    args = (S.CUSTOMIZED, 10, torch.cuda.Stream(), chunk_slots, max_in_flight)
    with pytest.raises(error, match=match):
        module.erase_by_score_async(*args)
    with pytest.raises(error, match=match):
        erase_by_score_async(nn.ModuleList([module]), *args)
    with pytest.raises(error, match=match):
        erase_by_score_async(nn.ModuleList(), *args)


def test_the_chunk_and_the_limit_in_flight_given_are_the_ones_the_scan_gets(device):
    module = build_de_table(device.index, S.CUSTOMIZED)
    seen, scan = [], module._storage.erase_by_score

    def spy(*args, **kwargs):
        seen.append((kwargs["chunk_slots"], kwargs["max_in_flight"]))
        return scan(*args, **kwargs)

    module._storage.erase_by_score = spy
    module.erase_by_score_async(
        S.CUSTOMIZED, 10, chunk_slots=1 << 20, max_in_flight=3
    ).wait()
    erase_by_score_async(
        nn.ModuleList([module]), S.CUSTOMIZED, 10, chunk_slots=1 << 19, max_in_flight=1
    ).wait()
    module.erase_by_score_async(S.CUSTOMIZED, 10).wait()  # what is not given
    assert seen == [
        (1 << 20, 3),
        (1 << 19, 1),
        (ERASE_CHUNK_SLOTS, ERASE_MAX_CHUNKS_IN_FLIGHT),
    ]


def test_a_table_of_two_score_words_cannot_be_erased_by_either(device):
    two_words = build_de_table(device.index, (S.TIMESTAMP, S.LFU))
    for member in (S.TIMESTAMP, S.LFU):
        with pytest.raises(NotImplementedError, match="reads one"):
            two_words.erasable_table_ids(member)
    assert two_words.erasable_table_ids(S.CUSTOMIZED) == []


@pytest.mark.parametrize(
    "layout, build_kwargs, stand_in",
    [
        ("HOST_ONLY", dict(local_hbm_for_values=0), False),
        ("HYBRID", dict(local_hbm_for_values=(1 << 20) * DIM * 4 // 2), False),
        ("CACHING_PS", {}, True),
        ("HOST_PS", {}, True),
    ],
)
def test_a_layout_that_cannot_be_scanned_is_refused_only_where_a_table_is_picked(
    device, layout, build_kwargs, stand_in
):
    """The first two are real layouts. The two with an external store are stand-ins -- a
    module that says it has one, since the store the tests of tables with a PS use is
    not available here -- so what is tested is that the refusal goes by the layout."""
    module = build_de_table(device.index, S.CUSTOMIZED, **build_kwargs)
    if stand_in:
        module._storage_layout = StorageLayout[layout]
    stream = torch.cuda.Stream()
    with pytest.raises(NotImplementedError, match=layout):
        module.erase_by_score_async(S.CUSTOMIZED, 10, stream)
    # No table asked for: nothing to scan, so nothing to refuse.
    handle = module.erase_by_score_async(S.LFU, 10, stream)
    assert handle.done() and handle.result() == (0, 0)


def test_a_stream_on_another_device_is_refused(device):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second device")
    module = build_de_table(device.index, S.CUSTOMIZED)
    elsewhere = torch.cuda.Stream(device=1 - device.index)
    with pytest.raises(ValueError, match="the tables are on"):
        module.erase_by_score_async(S.CUSTOMIZED, 10, elsewhere)
    with pytest.raises(ValueError, match="the tables are on"):
        erase_by_score_async(nn.ModuleList([module]), S.CUSTOMIZED, 10, elsewhere)


@pytest.mark.parametrize(
    "which",
    [
        "the current stream",
        "the default stream",
        "the prefetch stream",
        "its own",
        "none given",
    ],
)
def test_a_stream_the_training_uses_works_with_a_warning_and_one_of_its_own_without(
    device, which
):
    """On a stream the training runs on, or the prefetch stream, the erase is queued
    among the training's kernels: it is no faster, and just as right -- and the caller is
    told it will not overlap. On a stream of its own, or on none -- the module then uses
    its own -- nothing is said."""
    module = build_de_table(device.index, S.CUSTOMIZED)
    module.prefetch_stream = torch.cuda.Stream()
    stream = {
        "the current stream": torch.cuda.current_stream(),
        "the default stream": torch.cuda.default_stream(),
        "the prefetch stream": module.prefetch_stream,
        "its own": torch.cuda.Stream(),
        "none given": None,
    }[which]
    keys = list(range(1, 1001))
    train_step(module, keys, device, score=5)

    if which in ("its own", "none given"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            handle = module.erase_by_score_async(S.CUSTOMIZED, 100, stream)
    else:
        with pytest.warns(UserWarning, match="will not overlap"):
            handle = module.erase_by_score_async(S.CUSTOMIZED, 100, stream)
    handle.wait_stream()  # on the stream the erase ran on, this orders nothing new
    assert handle.result() == (1000, 0)
    assert size(module) == 0
    train_step(module, keys, device, score=200)  # and the table takes them again
    assert size(module) == 1000


def test_the_stream_a_module_makes_for_itself_is_kept_and_is_its_own(device):
    """Not the stream the training runs on, so the erase can overlap it; the same one for
    the next erase, another for another module, and none made when there is nothing to
    erase."""
    modules = [build_de_table(device.index, S.CUSTOMIZED) for _ in range(2)]
    keys = list(range(1, 1001))
    for module in modules:
        train_step(module, keys, device, score=5)

    modules[0].erase_by_score_async(S.CUSTOMIZED, 100).wait()
    own = modules[0]._eraser.own_stream(device.index)
    assert own != torch.cuda.current_stream()
    train_step(modules[0], keys, device, score=5)
    modules[0].erase_by_score_async(S.CUSTOMIZED, 100).wait()
    assert modules[0]._eraser.own_stream(device.index) == own

    assert modules[1]._eraser._streams == {}
    modules[1].erase_by_score_async(S.LFU, 100).wait()  # no table: no stream made
    assert modules[1]._eraser._streams == {}
    modules[1].erase_by_score_async(S.CUSTOMIZED, 100).wait()
    assert modules[1]._eraser.own_stream(device.index) != own


def test_nothing_is_started_when_any_module_is_refused(device):
    """The first module could be erased; the second cannot be scanned. All the checks
    come before any start, so the first one's worker was never made."""
    scannable = build_de_table(device.index, S.CUSTOMIZED)
    host_only = build_de_table(device.index, S.CUSTOMIZED, local_hbm_for_values=0)
    train_step(scannable, list(range(1, 501)), device, score=5)
    with pytest.raises(NotImplementedError, match="GPU hash table"):
        erase_by_score_async(
            nn.ModuleList([scannable, host_only]),
            S.CUSTOMIZED,
            100,
            torch.cuda.Stream(),
        )
    assert scannable._eraser._executor is None
    assert size(scannable) == 500


# ---------------------------------------------------------------------------
# On a model
# ---------------------------------------------------------------------------


def test_the_model_call_erases_from_every_module_that_has_the_strategy(device):
    """Found in the model like a dump finds them; a table with no score is left alone;
    the handle sums what the modules erased."""
    scored = [build_de_table(device.index, S.CUSTOMIZED) for _ in range(2)]
    unscored = build_de_table(device.index, S.NO_EVICTION)
    model = nn.Sequential(scored[0], nn.ModuleList([unscored, scored[1]]))
    for module, score in zip(scored, (10, 20)):
        train_step(module, list(range(1, 1001)), device, score=score)
    train_step(unscored, list(range(1, 1001)), device)

    stream = torch.cuda.Stream()
    handle = erase_by_score_async(model, S.CUSTOMIZED, 15)  # a stream of their own
    # One type for the caller; the group is what a model with several modules returns.
    assert isinstance(handle, EraseHandle) and isinstance(handle, EraseHandleGroup)
    handle.wait_stream()
    assert handle.result() == (
        1000,
        0,
    )  # the first module's thousand; the second scored 20
    assert handle.done()
    assert [size(module) for module in scored] == [0, 1000]
    assert size(unscored) == 1000

    nothing = erase_by_score_async(model, S.LFU, 15, stream)
    assert nothing.done() and nothing.result() == (0, 0)
