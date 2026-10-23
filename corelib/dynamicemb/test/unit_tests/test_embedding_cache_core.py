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

"""The cache's placement core, driven from Python.

The kernels have their own device tests (test/embedding_cache/gpu). These cover
what only the binding does: slicing by table, the per-step call sequence, and the
events that order allocate on one stream against release on another.
"""

import pytest
import torch
from dynamicemb_extensions import EmbeddingCacheCore


@pytest.fixture
def device():
    assert torch.cuda.is_available()
    return torch.device("cuda", torch.cuda.current_device())


def _keys(xs, device):
    return torch.tensor(xs, dtype=torch.int64, device=device)


def _bind(core, keys, offsets, device):
    """lookup_pin, then allocate and publish whatever missed. Returns nodes."""
    nodes = core.lookup_pin(keys, offsets)
    miss = (nodes < 0).nonzero().squeeze(1)
    if miss.numel() == 0:
        return nodes
    miss_keys = keys[miss]
    # offsets of the misses, per table
    counts = []
    for t in range(len(offsets) - 1):
        counts.append(int(((miss >= offsets[t]) & (miss < offsets[t + 1])).sum()))
    miss_offsets = [0]
    for c in counts:
        miss_offsets.append(miss_offsets[-1] + c)
    _, _, new_nodes = core.allocate(miss_keys, miss, miss_offsets, nodes)
    core.publish(new_nodes, miss_offsets)
    return nodes


def test_tables_are_independent(device):
    core = EmbeddingCacheCore([8, 4])
    assert core.num_tables() == 2
    assert core.capacity(0) == 8 and core.capacity(1) == 4

    # key 5 is in both tables
    keys = _keys([5, 6, 7, 5, 9], device)
    offsets = [0, 3, 5]
    nodes = _bind(core, keys, offsets, device)
    assert bool((nodes >= 0).all())
    core.release(nodes, offsets)

    again = core.lookup_pin(keys, offsets)
    assert again.tolist() == nodes.tolist()
    core.release(again, offsets)

    found = core.find_nodes(
        _keys([5, 5, 99], device), torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    )
    assert found[0] >= 0 and found[1] >= 0 and found[2] == -1


def test_eviction_order_and_victims(device):
    core = EmbeddingCacheCore([4])
    first = _keys([1, 2, 3, 4], device)
    nodes = _bind(core, first, [0, 4], device)
    core.release(nodes, [0, 4])

    new = _keys([10, 11], device)
    pos = torch.arange(2, dtype=torch.int64, device=device)
    n = torch.full((2,), -1, dtype=torch.int32, device=device)
    victims, has_victim, new_nodes = core.allocate(new, pos, [0, 2], n)
    assert bool(has_victim.all())
    # the queue is first in, first out: the two oldest keys go
    assert sorted(victims.tolist()) == [1, 2]
    assert bool((new_nodes >= 0).all())
    core.publish(new_nodes, [0, 2])
    core.release(new_nodes, [0, 2])


def test_a_table_with_every_row_held_cannot_serve_another_key(device):
    core = EmbeddingCacheCore([4])
    keys = _keys([1, 2, 3, 4], device)
    held = _bind(core, keys, [0, 4], device)  # all four referenced

    extra = _keys([5], device)
    n = core.lookup_pin(extra, [0, 1])
    _, _, new_nodes = core.allocate(
        extra, torch.zeros(1, dtype=torch.int64, device=device), [0, 1], n
    )
    assert new_nodes.tolist() == [-1]

    core.release(held, [0, 4])
    n = core.lookup_pin(extra, [0, 1])
    _, _, new_nodes = core.allocate(
        extra, torch.zeros(1, dtype=torch.int64, device=device), [0, 1], n
    )
    assert new_nodes.tolist()[0] >= 0


def test_invalidate_and_resident(device):
    core = EmbeddingCacheCore([8])
    keys = _keys([1, 2, 3], device)
    nodes = _bind(core, keys, [0, 3], device)
    core.release(nodes, [0, 3])

    tids = torch.zeros(2, dtype=torch.int64, device=device)
    victims = _keys([2, 3], device)
    core.invalidate(core.find_nodes(victims, tids), tids)
    assert core.find_nodes(victims, tids).tolist() == [-1, -1]

    res_nodes, res_keys, referenced = core.resident(0)
    assert referenced == 0
    assert res_keys.tolist() == [1]
    assert core.num_resident(0) == 1

    core.reset()
    assert core.num_resident(0) == 0


def test_resident_counts_references(device):
    core = EmbeddingCacheCore([8])
    keys = _keys([1, 2, 3], device)
    nodes = _bind(core, keys, [0, 3], device)  # three references held
    _, res_keys, referenced = core.resident(0)
    assert referenced == 3 and sorted(res_keys.tolist()) == [1, 2, 3]
    core.release(nodes, [0, 3])
    assert core.resident(0)[2] == 0


@pytest.mark.parametrize("keys_per_step", [96, 40000])
@pytest.mark.parametrize("depth", [1, 3])
def test_two_streams_never_reuse_a_held_node(device, depth, keys_per_step):
    """The pipeline: allocate for step N+depth on one stream while the backward of
    step N is still to come on another.

    Every key's row is stamped with the key when it is bound. A node that is
    reclaimed while a step still holds it is rebound to another key and stamped
    with that one, so reading each step's rows back just before it releases them
    says whether any held node was ever taken. The cache is smaller than the
    working set, so it is evicting all the time. At rest, every node that says it
    is queued must be, which catches a row lost between a pop and a push.

    What this does NOT show is that the events ordering the two streams are
    needed. With them removed from the binding it still passes (checked, five runs):
    the window they close is a few instructions wide, and kernels of this size
    from a Python loop do not land in it. The ordering itself is exercised, with
    the stream order explicit, by the device tests in test/embedding_cache/gpu;
    this checks the invariants a correct ordering has to preserve.
    """
    torch.manual_seed(0)
    k = keys_per_step
    # Room for every step in flight, and no more. The large size is for the
    # kernels: they have to run long enough for the two streams to really overlap,
    # or the ordering between them is never tested.
    capacity = (depth + 2) * k
    core = EmbeddingCacheCore([capacity])
    pool = torch.zeros(capacity, dtype=torch.int64, device=device)
    bad = torch.zeros(1, dtype=torch.int64, device=device)

    prefetch = torch.cuda.Stream()
    compute = torch.cuda.Stream()
    pending = []  # (nodes, keys, event) of steps whose backward is yet to come

    def finish(entry):
        nodes, keys, ready = entry
        with torch.cuda.stream(compute):
            compute.wait_event(ready)
            nodes.record_stream(compute)
            keys.record_stream(compute)
            # What the forward and backward of the step would read and write.
            bad.add_((pool[nodes.long()] != keys).sum())
            core.release(nodes, [0, nodes.numel()])

    for step in range(60 * depth):
        # a skewed draw so keys come back and the cache both hits and evicts
        span = max(4000, 4 * k)
        keys = torch.unique((torch.rand(k * 2, device=device) ** 3 * span).long())[:k]
        if keys.numel() < k:
            continue
        with torch.cuda.stream(prefetch):
            nodes = core.lookup_pin(keys, [0, k])
            miss = (nodes < 0).nonzero().squeeze(1)
            if miss.numel() > 0:
                miss_keys = keys[miss]
                _, _, new_nodes = core.allocate(
                    miss_keys, miss, [0, miss.numel()], nodes
                )
                pool[new_nodes.long()] = miss_keys  # the row is filled...
                core.publish(new_nodes, [0, miss.numel()])  # ...then visible
            ready = torch.cuda.Event()
            ready.record(prefetch)
        pending.append((nodes, keys, ready))
        if len(pending) > depth:
            finish(pending.pop(0))

    while pending:
        finish(pending.pop(0))
    torch.cuda.synchronize()

    assert int(bad.item()) == 0, "a node a step still held was given to another key"

    # And at rest the cache is consistent: every node that says it is queued is,
    # nothing is referenced, and there is one node per key. A node queued in name
    # only is a row the cache has lost for good.
    entries, flagged = core.queue_audit(0)
    assert entries == flagged, (
        f"the queue holds {entries} entries but {flagged} nodes say they have one"
    )
    res_nodes, res_keys, referenced = core.resident(0)
    assert referenced == 0
    assert res_nodes.unique().numel() == res_nodes.numel()
    assert res_keys.unique().numel() == res_keys.numel()
    found = core.find_nodes(res_keys, torch.zeros_like(res_keys))
    assert found.tolist() == res_nodes.tolist()
