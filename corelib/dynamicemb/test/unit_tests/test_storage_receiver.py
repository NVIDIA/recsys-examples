# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import gc
import weakref
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb import DynamicEmbScoreStrategy, EmbOptimType
from dynamicemb.resolved_storage import matching_rows
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from dynamicemb.types import CopyMode
from test_hbm_receiver import (  # noqa: F401
    PipelineModel,
    batch,
    check_debug,
    distribute,
    distributed,
    make_model,
    output,
    tables,
)
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)

LAYOUTS = ["host", "hybrid", "ps"]


@pytest.mark.parametrize("layout", ["host", "hybrid", "ps"])
def test_storage_hits_skip_snapshot_matching_and_keep_optimizer_state(layout):
    table = tables(make_model(layout=layout))[0]
    features = batch()
    state = table.resolve_input(features.values(), features.offsets())
    result = table(features.values(), features.offsets(), prefetch_state=state)
    result.sum().backward(retain_graph=True)
    with patch(
        "dynamicemb.resolved_storage.matching_rows",
        side_effect=AssertionError("a current storage hit searched live snapshots"),
    ):
        following = table.resolve_input(features.values(), features.offsets())
    # Includes Adam m/v, not just the visible embedding columns.
    torch.testing.assert_close(following.unique_values, state.unique_values)
    following.release()
    result.sum().backward()
    assert not table._active_rows.states


def test_host_and_evaluation_resolve_rows():
    for layout in ["hbm", "host"]:
        model = make_model(layout=layout)
        features = batch()
        if layout == "hbm":
            # Evaluation initializes misses with its own (zero) initializer.
            # Seed real rows first so this checks evaluation of resident keys.
            ctx, arrived = distribute(model, features)
            with torch.no_grad():
                check_debug(output(model, ctx, arrived), features)
            model.eval()
        ctx, arrived = distribute(model, features)
        assert isinstance(arrived[0], ResolvedEmbeddingInput)
        for group in arrived[0].split(model._lookups[0]._feature_splits):
            torch.testing.assert_close(
                group.values(), group.prefetch_state.reverse_indices, rtol=0, atol=0
            )
        with torch.no_grad():
            check_debug(output(model, ctx, arrived), features)


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_live_row_matching_keeps_table_and_full_key_identity(dtype):
    # Casting uint64 to signed preserves its bits; narrow or floating keys
    # would conflate these values. A repeated key in another table is distinct.
    raw = [7, -(2**63) + 5, 2**63 - 2, 7]
    keys = torch.tensor(raw, dtype=torch.int64, device="cuda").to(dtype)
    tids = torch.tensor([0, 0, 1, 1], device="cuda")
    order = torch.tensor([2, 3, 0, 1], device="cuda")
    target_keys = torch.tensor([raw[i] for i in [2, 3, 0, 1]], device="cuda").to(dtype)
    source, target = matching_rows(keys, tids, target_keys, tids[order], 2)
    torch.testing.assert_close(source, order[target])
    assert source.numel() == 4


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
    assert all(not table._active_rows.states for table in tables(model))


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
    assert all(not table._active_rows.states for table in tables(model))


@pytest.mark.parametrize("layout", ["host", "hybrid"])
def test_pending_rows_survive_backing_store_eviction(layout):
    model = make_model(
        layout=layout, max_capacity=256, init_capacity=256, optimizer=EmbOptimType.SGD
    )
    features = batch(count=8)
    first_ctx, first_input = distribute(model, features)
    capacity = max(
        sum(state.key_index_map.capacity() for state in table.tables.tables)
        for table in tables(model)
    )
    for iteration in range((capacity // 32 + 8) * dist.get_world_size()):
        fresh = batch(start=1000 + iteration * 32, count=32)
        with torch.no_grad():
            output(model, *distribute(model, fresh))
    # Prove this workload evicted at least one of the original rows in the
    # finite source. The active batch still owns its initialized values.
    missing = 0
    for table, group in zip(
        tables(model), first_input[0].split(model._lookups[0]._feature_splits)
    ):
        state = group.prefetch_state
        missing += table.tables.find(
            state.unique_keys, state.unique_table_ids, copy_mode=CopyMode.EMBEDDING
        )[0]
    assert missing > 0
    second_ctx, second_input = distribute(model, features)
    first = output(model, first_ctx, first_input)
    check_debug(first, features)
    sum(jt.values().sum() for jt in first.values()).backward()
    with torch.no_grad():
        second = output(model, second_ctx, second_input)
    for name, multiplicity in [("f0", 2), ("f1", 2), ("f2", 1)]:
        torch.testing.assert_close(
            second[name].values(), first[name].values() - 0.1 * multiplicity
        )
    assert all(not table._active_rows.states for table in tables(model))


@pytest.mark.parametrize("layout", ["host", "hybrid", "ps"])
@pytest.mark.parametrize("stage", ["input_dist", "forward", "detach"])
def test_abandoned_storage_batch_is_reclaimed(layout, stage):
    model = make_model(layout=layout)
    for _ in range(3):
        ctx, arrived = distribute(model, batch())
        refs = [
            weakref.ref(group.prefetch_state)
            for local in arrived
            for group in local.split(model._lookups[0]._feature_splits)
        ]
        if stage != "input_dist":
            result = output(model, ctx, arrived)
        del ctx, arrived
        if stage != "input_dist":
            gc.collect()
            assert all(ref() is not None for ref in refs)
            if stage == "detach":
                result = {name: jt.values().detach() for name, jt in result.items()}
            del result
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert all(not table._active_rows.states for table in tables(model))


def test_cancelled_storage_rows_survive_queued_cross_stream_copy():
    table = tables(make_model(layout="host"))[0]
    features = batch()
    allocation_stream = torch.cuda.current_stream()
    source = table.resolve_input(features.values(), features.offsets())
    target = table.resolve_input(features.values(), features.offsets())
    source.unique_values.fill_(77)
    shape, dtype = source.unique_values.shape, source.unique_values.dtype
    source_ref = weakref.ref(source)
    copy_stream = torch.cuda.Stream()
    copy_stream.wait_stream(allocation_stream)

    from dynamicemb.resolved_storage import _copy_matching_rows

    def delayed_copy(*args):
        # Delay the actual copy on the consumer stream.
        # Dropping the source below must not recycle its allocation meanwhile.
        torch.cuda._sleep(100_000_000)
        return _copy_matching_rows(*args)

    with torch.cuda.stream(copy_stream), table._table_gate():
        with patch("dynamicemb.resolved_storage._copy_matching_rows", new=delayed_copy):
            table._active_rows.publish(source)
    del source
    assert source_ref() is None
    with torch.cuda.stream(allocation_stream):
        reuse = [torch.full(shape, -123, device="cuda", dtype=dtype) for _ in range(32)]
    copy_stream.synchronize()
    assert (target.unique_values == 77).all()
    assert all(tensor.shape == shape for tensor in reuse)
    table._active_rows.release(target)


@pytest.mark.parametrize(
    "layout,dedup,dtype,fused",
    [
        ("host", False, torch.int64, False),
        ("hybrid", True, torch.int32, True),
        ("ps", True, torch.int64, True),
    ],
)
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
        assert not table._active_rows.states


def test_mixed_hbm_and_host_groups():
    model = make_model(
        dedup=True,
        table_options={
            "t1": {
                "global_hbm_for_values": 0,
                "score_strategy": DynamicEmbScoreStrategy.LFU,
            }
        },
    )
    features = batch()
    ctx, arrived = distribute(model, features)
    groups = arrived[0].split(model._lookups[0]._feature_splits)
    assert sorted(isinstance(group, ResolvedEmbeddingInput) for group in groups) == [
        True,
        True,
    ]
    model.prefetch(arrived, ctx=ctx)
    with torch.no_grad():
        check_debug(output(model, ctx, arrived), features)
    assert all(not table._prefetch_states for table in tables(model))


@pytest.mark.parametrize("layout", ["host", "hybrid", "ps"])
def test_legacy_prefetch_owns_rows_until_forward(layout):
    table = tables(make_model(layout=layout))[0]
    features = batch()
    table.prefetch(features.values(), features.offsets())
    refs = [weakref.ref(state) for state in table._prefetch_states]
    gc.collect()
    assert all(ref() is not None for ref in refs)
    result = table(features.values(), features.offsets())
    assert not table._prefetch_states
    assert all(ref() is not None for ref in refs)
    del result
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert not table._active_rows.states


@pytest.mark.parametrize(
    "layout,pipeline_type,clear_inputs,scored",
    [
        ("cache", PrefetchTrainPipelineSparseDist, False, True),
        ("cache_ps", TrainPipelineSparseDist, True, False),
        ("host", TrainPipelineSparseDist, False, True),
        ("hybrid", PrefetchTrainPipelineSparseDist, False, False),
        ("ps", TrainPipelineSparseDist, False, False),
    ],
    ids=[
        "cache-prefetch-scores",
        "cache-ps-clear",
        "host-scores",
        "hybrid-prefetch",
        "ps",
    ],
)
def test_sparse_dist_pipelines_with_storage(
    pipeline_type, clear_inputs, scored, layout, monkeypatch
):
    if clear_inputs:
        import inspect

        if "clear_data_dist_inputs" not in inspect.signature(pipeline_type).parameters:
            pytest.skip("TorchRec has no input-clearing option")
        from torchrec.distributed.train_pipeline.runtime_forwards import (
            PipelinedForward,
        )

        # The public constructor updates this class option. Restore it after
        # the case so default-pipeline tests retain their original scheduling.
        monkeypatch.setattr(PipelinedForward, "_use_main_stream_for_dist_init", False)
    score = (
        DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED
        if scored
        else DynamicEmbScoreStrategy.TIMESTAMP
    )
    collection = make_model(optimizer=EmbOptimType.SGD, score=score, layout=layout)
    model = PipelineModel(collection)
    pipeline = pipeline_type(
        model,
        torch.optim.SGD([model.dense], lr=0.1),
        torch.device("cuda", torch.cuda.current_device()),
        execute_all_batches=True,
        **({"clear_data_dist_inputs": True} if clear_inputs else {}),
    )
    control = PipelineModel(
        make_model(optimizer=EmbOptimType.SGD, score=score, layout=layout)
    )
    inputs = [batch(start=iteration * 4, scored=scored) for iteration in range(4)]
    iterator = iter(inputs)
    for features in inputs:
        actual = pipeline.progress(iterator)
        loss, expected = control(features)
        loss.sum().backward()
        torch.testing.assert_close(actual, expected)
    with pytest.raises(StopIteration):
        pipeline.progress(iterator)
    assert all(not table._active_rows.states for table in tables(collection))
