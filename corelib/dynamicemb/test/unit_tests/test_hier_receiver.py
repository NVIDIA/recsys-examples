# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from test_hbm_receiver import (  # noqa: F401
    batch,
    check_debug,
    distribute,
    distributed,
    make_model,
    output,
    tables,
)


@pytest.mark.parametrize("overflow", [False, True])
def test_hier_receiver_completion_contract(overflow):
    """Real receiver/lookup GPU code with a supplied transport result.

    This does not exercise NCCL or claim an NVLink communication test. It keeps
    the completion, status-before-mutation, ownership and output-window
    contracts executable on GPU hosts without NVLink.
    """
    from weakref import WeakKeyDictionary

    from dynamicemb.shard import hier_output_dist as h
    from dynamicemb.shard.input_dist import HierKJTAwaitable

    model = make_model()
    lookup = model._lookups[0]
    original = batch()
    count = original.values().numel()
    hier = object.__new__(h.HierDist)
    hier._arguments = dict(
        local_size=1,
        lookup_keys=count - int(overflow),
        remote_keys=count,
        lookup_groups=None,
    )
    hier._asked, hier._num_nodes = "hierarchical", 1
    hier._rows = WeakKeyDictionary()
    status = torch.zeros(
        h._LENGTH_PER_KEY + 2 * len(original.keys()), device="cuda", dtype=torch.int64
    )
    status[h._NEED_LOOKUP_KEYS] = count
    status[h._NEED_KEY_ROWS] = 1
    status[h._NUM_RECV] = count
    status[h._LENGTH_PER_KEY :] = torch.tensor(
        original.length_per_key() * 2, device="cuda"
    )
    window = torch.empty((count, 8), device="cuda")
    row_map = torch.arange(count - 1, -1, -1, dtype=torch.int32, device="cuda")
    ctx = h.DynamicEmbSequenceShardingContext()
    request = HierKJTAwaitable(
        hier,
        (
            original.values(),
            None,
            original.lengths(),
            original.offsets(),
            window,
            row_map,
            status,
            [],
            [],
        ),
        (original,),
        original.keys(),
        original,
        ctx,
        lookup.resolve_input,
    )
    if overflow:
        with patch.object(
            lookup,
            "resolve_input",
            side_effect=AssertionError("resolution before capacity check"),
        ):
            request._postprocessor = lookup.resolve_input
            with pytest.raises(RuntimeError, match="lookup_keys from"):
                request.wait()
        assert not hier._rows
        assert all(not table._active_rows.states for table in tables(model))
        return
    resolved = request.wait()
    assert isinstance(resolved, ResolvedEmbeddingInput)
    assert request.wait() is resolved
    assert hier.owns(resolved)
    assert ctx.hier_result is not None
    torch.testing.assert_close(hier.key_rows(resolved), row_map.long())
    for table in tables(model):
        table.output_alloc = hier.lookup_buffer
    result = hier.lookup(resolved, [lambda: lookup(resolved)])
    assert result.data_ptr() == window.data_ptr()
    expected = (original.values() % 100000).float()[:, None].expand_as(window)
    torch.testing.assert_close(
        result.view_as(window).index_select(0, row_map.long()), expected
    )
    result.sum().backward()
    for table in tables(model):
        owner = table._cache if table._cache is not None else table._storage
        assert torch.count_nonzero(owner._state.key_index_map._ref_counter).item() == 0


@pytest.mark.parametrize("distribution", ["torchrec", "auto"])
def test_output_dist_selection_keeps_receiver_resolution(distribution):
    model = make_model(
        dedup=True,
        table_options={
            name: {"output_dist": distribution, "output_dist_capacity": 4096}
            for name in ["t0", "t1"]
        },
    )
    features = batch()
    ctx, arrived = distribute(model, features)
    assert all(isinstance(group, ResolvedEmbeddingInput) for group in arrived)
    result = output(model, ctx, arrived)
    check_debug(result, features)
    sum(jt.values().sum() for jt in result.values()).backward()


def test_hierarchical_refuses_unlinked_topology_before_resolution():
    import dynamicemb_extensions as ext
    from dynamicemb.shard.hier_output_dist import NVLinkCheckError, node_size

    if dist.get_world_size() == 1 or not hasattr(ext, "HierOutputDist"):
        pytest.skip("Requires multiple ranks and a hierarchical-enabled build")
    try:
        node_size(dist.group.WORLD)
    except NVLinkCheckError:
        pass
    else:
        pytest.skip("This topology supports hierarchical communication")
    model = make_model(
        table_options={
            name: {"output_dist": "hierarchical", "output_dist_capacity": 4096}
            for name in ["t0", "t1"]
        }
    )
    with pytest.raises(NVLinkCheckError):
        distribute(model, batch())
    for table in tables(model):
        owner = table._cache if table._cache is not None else table._storage
        assert torch.count_nonzero(owner._state.key_index_map._ref_counter).item() == 0
