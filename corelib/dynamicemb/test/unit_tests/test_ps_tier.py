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

"""An external storage as the tier below the table, in each of the three layouts.

The table in front -- HBM, host, or the host backing behind an HBM cache -- is
smaller than the set of keys the run touches, so it evicts, and what it evicts has
to land in the external storage with its value and come back when the key is next
used. The check is end to end: every key starts at 1.0, the loss is the sum of the
outputs, so the gradient of every occurrence of a key is 1 and plain SGD leaves
the key at ``1 - lr * occurrences``, however often it was evicted and brought
back on the way. A value lost on its way down shows as a key that is wrong.

The control runs the same scenario over a storage that drops everything it is
given, and expects the check to fail: without it, a table big enough to hold
every key would pass this test too.
"""

import os
import sys

import pytest
import torch
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
)
from dynamicemb.batched_dynamicemb_tables import (
    BatchedDynamicEmbeddingTablesV2,
    StorageLayout,
)
from dynamicemb.key_value_table import (
    DynamicEmbStorage,
    take_values_of_keys_evicted_in_the_same_insert,
)
from torchrec.modules.embedding_configs import EmbeddingConfig  # noqa: F401
from fbgemm_gpu.split_embedding_configs import EmbOptimType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_batched_dynamic_embedding_tables_v2 import PyDictStorage  # noqa: E402

DIM = 8
MAX_CAPACITY = 2048  # rows the table in front holds
KEYS = 6000  # distinct keys the run touches: three times that
BATCH = 256  # fits the HBM cache of the caching layout (1024 rows)
STEPS = 40
LR = 0.1
VALUE_BYTES = MAX_CAPACITY * DIM * 4


class DroppingStorage(PyDictStorage):
    """A lower tier that forgets whatever it is handed."""

    def insert(self, keys, table_ids, values, scores=None, preserve_existing=False):
        return None


LAYOUT = {
    "hbm": StorageLayout.HBM_ONLY,
    "host": StorageLayout.HOST_ONLY,
    "caching": StorageLayout.CACHING,
}


def _layout(name):
    """``(caching, local_hbm_for_values)`` that makes the runtime pick ``name``."""
    if name == "hbm":
        return False, 2 * VALUE_BYTES
    if name == "host":
        return False, 0
    return True, VALUE_BYTES // 2  # caching: an HBM cache of half the table


def _module(layout, ps):
    caching, local_hbm = _layout(layout)
    device_id = torch.cuda.current_device()
    options = DynamicEmbTableOptions(
        dim=DIM,
        init_capacity=MAX_CAPACITY,
        max_capacity=MAX_CAPACITY,
        index_type=torch.int64,
        embedding_dtype=torch.float32,
        device_id=device_id,
        score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
        caching=caching,
        local_hbm_for_values=local_hbm,
        external_storage=ps,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.CONSTANT, value=1.0
        ),
    )
    return BatchedDynamicEmbeddingTablesV2(
        table_names=["t"],
        table_options=[options],
        feature_table_map=[0],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        optimizer=EmbOptimType.SGD,
        learning_rate=LR,
        use_index_dedup=True,
    )


def _train(module):
    """Returns how often each key was used, per key."""
    device = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device="cpu").manual_seed(7)
    uses = torch.zeros(KEYS, dtype=torch.int64)
    for _ in range(STEPS):
        keys = torch.randint(0, KEYS, (BATCH,), generator=g, dtype=torch.int64)
        uses += torch.bincount(keys, minlength=KEYS)
        indices = keys.to(device)
        offsets = torch.arange(BATCH + 1, dtype=torch.int64, device=device)
        out = module(indices, offsets)
        out.backward(torch.ones_like(out))
    torch.cuda.synchronize()
    return uses


def _mismatched_keys(module, uses):
    """How many of the used keys do not hold ``1 - lr * uses``."""
    device = torch.device("cuda", torch.cuda.current_device())
    module.flush()
    module.eval()
    used = (uses > 0).nonzero().squeeze(1)
    indices = used.to(device)
    offsets = torch.arange(used.numel() + 1, dtype=torch.int64, device=device)
    with torch.no_grad():
        out = module(indices, offsets)
    expected = (1.0 - LR * uses[used].float()).to(device)
    bad = ~torch.isclose(out[:, 0], expected, atol=1e-4)
    return int(bad.sum().item()), used.numel()


@pytest.fixture(params=[False, True], ids=["default", "determinism"])
def determinism(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("DEMB_DETERMINISM_MODE", "ON")
    return request.param


@pytest.mark.parametrize("layout", ["hbm", "host", "caching"])
def test_what_the_table_evicts_comes_back_from_the_storage_below(layout, determinism):
    module = _module(layout, PyDictStorage)
    assert module._storage_layout == LAYOUT[layout]
    storage = module._storage
    assert isinstance(storage, DynamicEmbStorage)
    assert storage.lower is not None

    uses = _train(module)
    assert int((uses > 0).sum()) > 2 * MAX_CAPACITY, "the run must overflow the table"
    assert storage.lower.size() > 0, "nothing was evicted into the storage below"

    bad, total = _mismatched_keys(module, uses)
    assert bad == 0, f"{bad} of {total} keys lost an update on the way down or back"


@pytest.mark.parametrize("layout", ["hbm", "host", "caching"])
def test_control_a_storage_that_drops_what_it_is_given_fails_the_check(layout):
    module = _module(layout, DroppingStorage)
    assert module._storage_layout == LAYOUT[layout]
    uses = _train(module)
    bad, total = _mismatched_keys(module, uses)
    assert bad > 0, (
        "with a lower tier that forgets everything the check still passes, so it "
        "does not show that evicted values reach the storage below"
    )


def _storage_over(lower, dim=8):
    """A one-bucket table of 128 rows with CUSTOMIZED scores, over ``lower``."""
    from dynamicemb import DynamicEmbCheckMode
    from dynamicemb.optimizer import OptimizerArgs, SGDDynamicEmbeddingOptimizer

    optimizer = SGDDynamicEmbeddingOptimizer(OptimizerArgs())
    options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            device_id=torch.cuda.current_device(),
            dim=dim,
            max_capacity=128,
            init_capacity=128,
            bucket_capacity=128,
            safe_check_mode=DynamicEmbCheckMode.IGNORE,
            local_hbm_for_values=1024**3,
            score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
        )
    ]
    return DynamicEmbStorage(options, optimizer, lower=lower(options, optimizer))


def test_keys_evicted_by_the_insert_that_put_them_in_keep_their_own_values(
    determinism,
):
    """The table is full of keys that score high, and one insert brings 200 more
    that all score low: each evicts the lowest-scored key there is, which after the
    first is one of the insert's own, whose row has not been written yet. What goes
    down must be the value it came with, not what the slot held before."""
    from dynamicemb.types import CopyMode

    device = torch.device("cuda", torch.cuda.current_device())
    storage = _storage_over(PyDictStorage)
    dim = 8
    tids = lambda n: torch.zeros(n, dtype=torch.int64, device=device)

    resident = torch.arange(1, 129, dtype=torch.int64, device=device)
    resident_values = torch.full((128, dim), -1.0, device=device)
    storage.insert(
        resident,
        tids(128),
        resident_values,
        scores=torch.full((128,), 1000, dtype=torch.uint64, device=device),
    )

    new = torch.arange(1001, 1201, dtype=torch.int64, device=device)
    new_values = new.float().unsqueeze(1).repeat(1, dim)
    storage.insert(
        new,
        tids(200),
        new_values,
        scores=torch.arange(1, 201, dtype=torch.int64, device=device).to(torch.uint64),
    )

    everyone = torch.cat([resident, new])
    expected = torch.cat([resident_values, new_values])
    out = storage.find(everyone, tids(328), CopyMode.VALUE)
    founds, values = out[5], out[7]
    assert bool(founds.all()), "a key was lost between the table and the tier below"
    assert torch.equal(values[:, :dim], expected)


@pytest.mark.parametrize("order", ["colliding_first", "colliding_last"])
@pytest.mark.parametrize("key_dtype", [torch.int64, torch.uint64])
def test_victims_are_matched_by_the_whole_key_and_table_pair(order, key_dtype):
    """(0, table 0) and (K, table 1) both come to 0 when the pair is folded into one
    int64 as ``key * 1000003 + table_id`` and the product wraps, so a match that
    went by the fold alone could pair a victim with the other's value."""
    device = torch.device("cuda", torch.cuda.current_device())
    colliding = 2336937208910341525
    assert (colliding * 1000003 + 1) % 2**64 == 0

    pairs = [(0, 0), (colliding, 1)]
    if order == "colliding_last":
        pairs.reverse()
    keys = torch.tensor([k for k, _ in pairs], dtype=torch.int64, device=device)
    tids = torch.tensor([t for _, t in pairs], dtype=torch.int64, device=device)
    keys = keys.to(key_dtype) if key_dtype == torch.int64 else keys.view(torch.uint64)
    values = torch.tensor([[1.0] * 4, [2.0] * 4], device=device)

    # Both are victims, in the opposite order, with the values their rows held.
    evicted_keys, evicted_tids = keys.flip(0), tids.flip(0)
    evicted_values = torch.full((2, 4), -1.0, device=device)
    take_values_of_keys_evicted_in_the_same_insert(
        evicted_keys, evicted_tids, evicted_values, keys, tids, values
    )
    assert torch.equal(evicted_values, values.flip(0))
