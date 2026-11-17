# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb import EmbOptimType
from dynamicemb.key_value_table import load_from_flat
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from dynamicemb.types import CopyMode
from test_hbm_receiver import (  # noqa: F401
    batch,
    check_debug,
    distribute,
    distributed,
    make_model,
    output,
    tables,
)

LAYOUTS = ["cache", "cache_ps"]


@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_input_dist_resolves_before_lookup(dedup, dtype, fused, layout):
    model = make_model(dedup=dedup, layout=layout)
    features = batch(dtype=dtype, empty_rank=dist.get_world_size() > 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ctx, arrived = distribute(model, features, fused=fused)
        assert isinstance(arrived[0], ResolvedEmbeddingInput)
        groups = arrived[0].split(model._lookups[0]._feature_splits)
        for group in groups:
            state = group.prefetch_state
            torch.testing.assert_close(
                group.values(), state.value_row_indices[state.reverse_indices]
            )
            assert (state.value_row_indices >= 0).all()
            if state.unique_keys.numel() and layout == "hbm":
                assert (state.slot_indices != state.value_row_indices).any()
        assert ctx.sharding_contexts[0].input_splits
        assert ctx.sharding_contexts[0].output_splits
    torch.cuda.current_stream().wait_stream(stream)
    arrived.record_stream(torch.cuda.current_stream())
    ctx.record_stream(torch.cuda.current_stream())
    # A prefetch pipeline must not enqueue, resolve or pin this batch twice.
    for lookup, local_features in zip(model._lookups, arrived):
        lookup.prefetch(local_features, forward_stream=stream)
    assert all(not table._prefetch_states for table in tables(model))
    with ExitStack() as checks:
        checks.enter_context(
            patch(
                "dynamicemb.batched_dynamicemb_function._find_keys",
                side_effect=AssertionError("lookup hashed a key"),
            )
        )
        native_calls = []
        for table in tables(model):
            native = table._native_lookup._dense_module(False)
            native_calls.append(
                checks.enter_context(
                    patch.object(native, "_operation", wraps=native._operation)
                )
            )
            checks.enter_context(
                patch.object(
                    table.tables,
                    "find",
                    side_effect=AssertionError("lookup read keys from a storage"),
                )
            )
        result = output(model, ctx, arrived)
        assert all(call.call_count == 1 for call in native_calls)
    check_debug(result, features)
    sum(jt.values().sum() for jt in result.values()).backward()
    for table in tables(model):
        owner = table.cache or table.tables
        if (
            hasattr(owner, "key_index_map")
            and owner.key_index_map._ref_counter is not None
        ):
            assert (owner.key_index_map._ref_counter == 0).all()
        assert (
            torch.count_nonzero(
                (table.cache or table.tables)._state.key_index_map._ref_counter
            ).item()
            == 0
        )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_prefetched_batch_reads_updates_from_previous_batch(layout):
    model = make_model(optimizer=EmbOptimType.SGD, layout=layout)
    features = batch(count=8)
    first_ctx, first_input = distribute(model, features)
    second_ctx, second_input = distribute(model, features)
    first = output(model, first_ctx, first_input)
    check_debug(first, features)
    sum(jt.values().sum() for jt in first.values()).backward()
    with torch.no_grad():
        second = output(model, second_ctx, second_input)
    # All ranks repeat the input. TorchRec averages gradients across ranks;
    # f0/f1 share a table, so each t0 key receives two gradients, each t1 key one.
    for name, multiplicity in [("f0", 2), ("f1", 2), ("f2", 1)]:
        torch.testing.assert_close(
            second[name].values(),
            first[name].values() - 0.1 * multiplicity,
        )
    assert all(
        torch.count_nonzero(
            (table.cache or table.tables)._state.key_index_map._ref_counter
        ).item()
        == 0
        for table in tables(model)
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
@pytest.mark.parametrize("reverse_backward", [False, True])
@pytest.mark.parametrize("drop_inputs", [False, True])
def test_in_flight_batches_share_current_optimizer_state(
    layout, optimizer, reverse_backward, drop_inputs
):
    model = make_model(layout=layout, optimizer=optimizer)
    features = batch(count=8)
    contexts = [distribute(model, features) for _ in range(2)]
    outputs = [output(model, ctx, arrived) for ctx, arrived in contexts]
    if drop_inputs:
        # The autograd contexts alone must keep the weakly registered rows
        # alive after a pipeline releases its distributed inputs.
        contexts.clear()
        gc.collect()
    order = [1, 0] if reverse_backward else [0, 1]
    for index in order:
        (
            (index + 1) * sum(jt.values().sum() for jt in outputs[index].values())
        ).backward()
    if layout != "cache_ps":
        model.eval()
    with torch.no_grad():
        actual = output(model, *distribute(model, features))
    for name, multiplicity in [("f0", 2), ("f1", 2), ("f2", 1)]:
        delta = torch.nn.Parameter(torch.zeros((), device="cuda"))
        ref_opt = (
            torch.optim.SGD([delta], lr=0.1)
            if optimizer == EmbOptimType.SGD
            else torch.optim.Adam([delta], lr=0.1, eps=1e-8)
        )
        for index in order:
            delta.grad = torch.full_like(delta, (index + 1) * multiplicity)
            ref_opt.step()
        torch.testing.assert_close(
            actual[name].values(), outputs[0][name].values() + delta.detach()
        )
    assert all(
        torch.count_nonzero(
            (table.cache or table.tables)._state.key_index_map._ref_counter
        ).item()
        == 0
        for table in tables(model)
    )


def test_cache_eviction_reinitializes_rows_and_adam_state():
    model = make_model(init_capacity=256, max_capacity=256, layout="cache")
    seen_rows = {}
    reused = False
    per_table = tables(model)[0].cache._state.capacity // len(
        tables(model)[0]._dynamicemb_options
    )
    count = max(32, per_table * dist.get_world_size() // 2)
    for iteration in range(6):
        features = batch(start=iteration * count, count=count)
        ctx, arrived = distribute(model, features)
        groups = arrived[0].split(model._lookups[0]._feature_splits)
        for table, group in zip(tables(model), groups):
            state = group.prefetch_state
            assert (state.value_row_indices >= 0).all()
            values = load_from_flat(
                table.cache._state,
                state.value_row_indices,
                state.unique_table_ids,
                copy_mode=CopyMode.VALUE,
            )
            # Every key is new. A reused row must have neither the old embedding
            # nor Adam's moments, regardless of which hash slot owns it now.
            assert (values[:, 8:] == 0).all()
            for tid, row in zip(
                state.unique_table_ids.tolist(), state.value_row_indices.tolist()
            ):
                key = (id(table), tid)
                rows = seen_rows.setdefault(key, set())
                reused |= row in rows
                rows.add(row)
        result = output(model, ctx, arrived)
        check_debug(result, features)
        sum(jt.values().sum() for jt in result.values()).backward()
    assert reused, "the workload did not actually reuse a value row"
