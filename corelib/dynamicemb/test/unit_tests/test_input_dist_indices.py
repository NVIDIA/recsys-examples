# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration cases spanning receiver slices and backend combinations.

    python -m pytest test/unit_tests/test_input_dist_indices.py -q
    torchrun --standalone --nproc_per_node=2 -m pytest \
        test/unit_tests/test_input_dist_indices.py -q
"""

import gc
import sys
import weakref
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb import DynamicEmbScoreStrategy, EmbOptimType
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
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
from torchrec import PoolingType
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)

# The repository's functional in-memory PS implements the actual Storage API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LAYOUTS = ["hbm", "cache", "host", "hybrid", "ps", "cache_ps"]


@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("layout", ["host", "hybrid", "ps"])
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
        assert not table._active_rows.states


@pytest.mark.parametrize("pooling", [None, PoolingType.SUM])
def test_raw_grouped_prefetch_keeps_legacy_contract(pooling):
    model = make_model(pooling=pooling)
    shardings = (
        model._sharding_type_to_sharding.values()
        if pooling is None
        else model._embedding_shardings
    )
    for sharding in shardings:
        sharding._input_postprocessor = None
    features = batch()
    ctx, arrived = distribute(model, features)
    for lookup, group in zip(model._lookups, arrived):
        lookup.prefetch(group)
    result = output(model, ctx, arrived)
    if pooling is None:
        check_debug(result, features)
        loss = sum(jt.values().sum() for jt in result.values())
    else:
        expected = torch.cat(
            [jt.values().float().sum().expand(8) for jt in features.to_dict().values()]
        ).view(1, -1)
        torch.testing.assert_close(result.values(), expected)
        loss = result.values().sum()
    loss.backward()
    assert all(not table._prefetch_states for table in tables(model))
    assert all(not table._active_rows.states for table in tables(model))


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


def test_later_group_resolution_failure_releases_earlier_pins():
    model = make_model(
        table_options={"t1": {"score_strategy": DynamicEmbScoreStrategy.LFU}}
    )
    features = batch()
    grouped = model._lookups[0]
    assert len(grouped._emb_modules) == 2
    table = grouped._emb_modules[0].emb_module
    with patch.object(
        grouped._emb_modules[1],
        "resolve_input",
        side_effect=RuntimeError("receiver failure"),
    ):
        with pytest.raises(RuntimeError, match="receiver failure"):
            distribute(model, features)
    gc.collect()
    torch.cuda.synchronize()
    owner = table._cache if table._cache is not None else table._storage
    assert torch.count_nonzero(owner._state.key_index_map._ref_counter).item() == 0


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
    "pooling,dims,weighted",
    [
        (None, (8, 8), False),
        (PoolingType.SUM, (8, 8), False),
        (PoolingType.SUM, (8, 8), True),
        (PoolingType.SUM, (7, 11), False),
        (PoolingType.SUM, (7, 11), True),
    ],
)
def test_cold_lookup_after_restore_preserves_rng(pooling, dims, weighted):
    warm = tables(make_model(pooling=pooling, dims=dims, weighted=weighted))[0].eval()
    restored = tables(make_model(pooling=pooling, dims=dims, weighted=weighted))[
        0
    ].eval()
    features = batch(count=2)
    kwargs = (
        dict(pooling_weights=torch.ones(features.values().numel(), device="cuda"))
        if weighted
        else {}
    )
    with torch.no_grad():
        warm(features.values(), features.offsets(), **kwargs)
    restored.load_state_dict(warm.state_dict())
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    with torch.no_grad():
        expected = warm(features.values(), features.offsets(), **kwargs)
        expected_dropout = torch.nn.functional.dropout(expected, p=0.5, training=True)
        expected_cpu = torch.rand(16)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    with torch.no_grad():
        actual = restored(features.values(), features.offsets(), **kwargs)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        torch.testing.assert_close(actual, expected)
        actual_dropout = torch.nn.functional.dropout(actual, p=0.5, training=True)
        actual_cpu = torch.rand(16)
    torch.testing.assert_close(actual_dropout, expected_dropout)
    torch.testing.assert_close(actual_cpu, expected_cpu)


@pytest.mark.parametrize(
    "pipeline_type,clear_inputs",
    [
        (TrainPipelineSparseDist, False),
        (PrefetchTrainPipelineSparseDist, False),
        (TrainPipelineSparseDist, True),
    ],
)
@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("layout", ["cache", "host", "hybrid", "ps", "cache_ps"])
def test_native_sparse_dist_pipeline(
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
    if scored and layout in ["ps", "hybrid"]:
        pytest.skip("PRECISE_CUSTOMIZED is not supported by these storage layouts")
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
