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

"""An external storage as the only tier (HOST_PS), through the padded generic body.

With no cache and no HBM budget for values the module's storage *is* the external storage.
The forward looks every unique key up in it, initializes the keys it lacks (admission decides
which are kept), inserts the admitted ones, and the backward writes the updated rows back with
``preserve_existing=True``. Eval reads it. A PS of the padded contract is called through
``lookup_masked`` / ``insert_masked`` and costs DynamicEmb no read; a host-sized one through its
``find`` / ``insert``, which needs the number of live rows of each call.

The reference of every test is an in-tree host-only table big enough to hold all keys, with the
same initializer (a function of the key), optimizer and updates: a module with a PS in its place has
to give the very values that table gives, in training and in eval.
"""
import dataclasses
import os
import sys

import pytest

try:
    import torch
    from dynamicemb import sync_free as sf
except ImportError as e:  # pragma: no cover
    pytest.skip("needs torch and dynamicemb: %s" % (e,), allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

UNIT_TESTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(UNIT_TESTS)
NHS = os.path.join(UNIT_TESTS, "no_host_sync")
for path in (TEST_DIR, UNIT_TESTS, NHS):
    if path not in sys.path:
        sys.path.insert(0, path)
import host_sync_probe as hsp  # noqa: E402
import ps_standins  # noqa: E402

PS_KINDS = ps_standins.PS_KINDS
ps_class = ps_standins.ps_class
KEY_RANGE = 2000
STEPS = 10
PLAN = {"max_capacity": 65536, "hbm_rows": None, "init_capacity": None}


def _train(mod, batch):
    out = mod(batch.indices, batch.offsets)
    out.backward(torch.ones_like(out))
    return out.detach().clone().cpu()


def _eval(mod, batch):
    mod.eval()
    try:
        with torch.no_grad():
            return mod(batch.indices, batch.offsets).detach().clone().cpu()
    finally:
        mod.train()


def _pair(kind, config="host_dps", reference="host_only"):
    cls = ps_standins.checked(ps_class(kind))
    _, mod, maker = ps_standins.probe_module(config, PLAN, cls)
    _, ref, ref_maker = ps_standins.probe_module(reference, PLAN)
    return mod, maker, ref, ref_maker


@pytest.mark.parametrize("kind", PS_KINDS)
def test_a_host_ps_module_gives_the_values_of_the_in_tree_host_table(kind):
    mod, maker, ref, ref_maker = _pair(kind)
    batches = ps_standins.random_batches(maker, STEPS, KEY_RANGE)
    ref_batches = ps_standins.random_batches(ref_maker, STEPS, KEY_RANGE)
    train_reads, eval_reads = [], []

    for step, (batch, ref_batch) in enumerate(zip(batches, ref_batches)):
        start = sf.READ_NOW_CALLS
        out = _train(mod, batch)
        train_reads.append(sf.READ_NOW_CALLS - start)
        assert torch.equal(out, _train(ref, ref_batch)), "step %d" % step
    # eval: keys the module has seen, and keys it has not (they take the eval initializer)
    unseen = ps_standins.random_batches(maker, 1, KEY_RANGE, seed=99)[0]
    unseen_ref = ps_standins.random_batches(ref_maker, 1, KEY_RANGE, seed=99)[0]
    for batch, ref_batch in ((batches[-1], ref_batches[-1]), (unseen, unseen_ref)):
        start = sf.READ_NOW_CALLS
        out = _eval(mod, batch)
        eval_reads.append(sf.READ_NOW_CALLS - start)
        assert torch.equal(out, _eval(ref, ref_batch)), "eval"

    ps = mod._storage
    assert any(n > 0 for name, n in ps.calls if name.startswith("find"))
    assert any(n > 0 for name, n in ps.calls if name.startswith("insert"))
    if kind == "old":
        # forward: the lookup, and the insert when something missed; backward: the update
        assert set(train_reads) <= {2, 3}, train_reads
        assert eval_reads == [1, 1], "one read of the live rows per eval lookup"
    else:
        assert train_reads == [0] * STEPS, "a padded PS costs DynamicEmb no read"
        assert eval_reads == [0, 0]


def test_a_step_of_a_host_sized_ps_that_misses_nothing_reads_the_lookup_and_the_update_only():
    """The class returned a host count of the keys it lacked (zero): the miss path is skipped
    without a read of its own; the update of the backward still needs the number of rows.
    """
    mod, maker, _, _ = _pair("old")
    batch = ps_standins.random_batches(maker, 1, KEY_RANGE)[0]
    _train(mod, batch)  # inserts every key
    _train(mod, batch)
    torch.cuda.synchronize()
    ps = mod._storage
    ps.calls.clear()
    start = sf.READ_NOW_CALLS

    _train(mod, batch)

    assert sf.READ_NOW_CALLS - start == 2
    assert [name for name, _ in ps.calls] == ["find", "insert"]


@pytest.mark.parametrize("kind", ["old", "device"])
def test_a_host_ps_with_admission_initializes_what_the_admitter_refuses(kind):
    """A frequency admitter with threshold 2: a key seen once is refused and not inserted, its
    row (the table's initializer, a function of the key) is still what the forward returns; seen
    twice it is admitted, and the PS keeps it."""
    config = dataclasses.replace(
        hsp.CONFIGS["host_dps"], name="host_dps_admit", admit=True
    )
    cls = ps_standins.checked(ps_class(kind))
    _, mod, maker = ps_standins.probe_module(config, PLAN, cls)
    _, ref, ref_maker = ps_standins.probe_module("host_only_admit", PLAN)
    # keys below the key domain of the device stand-in (the probe's own are 62-bit hashes)
    stream = ps_standins.overlap_batches(maker, 8)
    ref_stream = ps_standins.overlap_batches(ref_maker, 8)

    for step, (batch, ref_batch) in enumerate(zip(stream, ref_stream)):
        assert torch.equal(_train(mod, batch), _train(ref, ref_batch)), "step %d" % step
    torch.cuda.synchronize()
