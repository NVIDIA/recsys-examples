# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved input_dist and TorchRec lookup, on one or more real GPUs.

    python -m pytest test/unit_tests/test_hbm_receiver.py -q
    bash test/unit_tests/test_hbm_receiver_dist.sh
"""

import copy
import gc
import os
import pickle
import sys
import tempfile
import weakref
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    FrequencyAdmissionStrategy,
    KVCounter,
    ScoredKeyedJaggedTensor,
)
from dynamicemb.batched_dynamicemb_compute_kernel import BatchedDynamicEmbedding
from dynamicemb.key_value_table import load_from_flat
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from dynamicemb.types import CopyMode
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    PoolingType,
)
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.planner import Topology
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
    _override_input_dist_forwards,
)
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.types import DataType

# The repository's functional in-memory PS implements the actual Storage API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_batched_dynamic_embedding_tables_v2 import PyDictStorage

LAYOUTS = ["hbm"]
ENABLED_LAYOUTS = {"hbm", "cache", "cache_ps", "host", "hybrid", "ps"}


@pytest.fixture(scope="session", autouse=True)
def distributed():
    if dist.is_initialized():
        yield
        return
    assert torch.cuda.is_available(), "These tests require CUDA"
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    with tempfile.TemporaryDirectory() as directory:
        kwargs = (
            {}
            if "RANK" in os.environ
            else dict(init_method=f"file://{directory}/pg", rank=0, world_size=1)
        )
        dist.init_process_group("nccl", **kwargs)
        yield
        dist.destroy_process_group()


class LegacySequenceLookup:
    """The prior CUDA gather, used only as a numerical control."""

    def __call__(self, unique_embs, batch, offsets, output_dtype, pooling_weights=None):
        from dynamicemb_extensions import gather_embedding

        result = torch.empty(
            (batch.reverse_indices.numel(), unique_embs.size(1)),
            dtype=output_dtype,
            device=unique_embs.device,
        )
        gather_embedding(unique_embs, result, batch.reverse_indices)
        return result


def make_model(
    dedup=False,
    legacy=False,
    score=DynamicEmbScoreStrategy.TIMESTAMP,
    init_capacity=4096,
    max_capacity=4096,
    optimizer=EmbOptimType.ADAM,
    layout="hbm",
    table_options=None,
    static_table=False,
    pooling=None,
    weighted=False,
    dims=(8, 8),
    embedding_dtype=torch.float32,
    learning_rate=0.1,
):
    device = torch.device("cuda", torch.cuda.current_device())
    config_type = EmbeddingConfig if pooling is None else EmbeddingBagConfig
    data_type = {
        torch.float32: DataType.FP32,
        torch.float16: DataType.FP16,
        torch.bfloat16: DataType.BF16,
    }[embedding_dtype]
    configs = [
        config_type(
            name="t0",
            embedding_dim=dims[0],
            num_embeddings=max_capacity,
            feature_names=["f0", "f1"],
            data_type=data_type,
            **({"pooling": pooling} if pooling is not None else {}),
        ),
        config_type(
            name="t1",
            embedding_dim=dims[1],
            num_embeddings=max_capacity,
            feature_names=["f2"],
            data_type=data_type,
            **({"pooling": pooling} if pooling is not None else {}),
        ),
    ]
    module = (
        EmbeddingCollection(configs, device=torch.device("meta"), need_indices=True)
        if pooling is None
        else EmbeddingBagCollection(
            configs, device=torch.device("meta"), is_weighted=weighted
        )
    )
    constraints = {
        config.name: DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            use_dynamicemb=not (static_table and config.name == "t1"),
            dynamicemb_options=replace(
                DynamicEmbTableOptions(
                    init_capacity=init_capacity,
                    max_capacity=max_capacity,
                    global_hbm_for_values=(
                        1024**3
                        if layout == "hbm"
                        else (
                            max_capacity * config.embedding_dim
                            if layout in ["cache", "hybrid", "cache_ps"]
                            else 0
                        )
                    ),
                    caching=layout in ["cache", "cache_ps"],
                    external_storage=(
                        PyDictStorage if layout in ["ps", "cache_ps"] else None
                    ),
                    bucket_capacity=128,
                    score_strategy=score,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG
                    ),
                ),
                **((table_options or {}).get(config.name, {})),
            ),
        )
        for config in configs
    }
    planner = DynamicEmbeddingShardingPlanner(
        topology=Topology(dist.get_world_size(), "cuda"), constraints=constraints
    )
    sharder_type = (
        DynamicEmbeddingCollectionSharder
        if pooling is None
        else DynamicEmbeddingBagCollectionSharder
    )
    sharder = sharder_type(
        fused_params=dict(
            optimizer=optimizer,
            learning_rate=learning_rate,
            output_dtype=SparseType.FP32,
            stochastic_rounding=False,
        ),
        **({"use_index_dedup": dedup} if pooling is None else {}),
    )
    plan = planner.collective_plan(module, [sharder], dist.group.WORLD)
    model = sharder.shard(
        module, plan.plan[""], ShardingEnv.from_process_group(dist.group.WORLD), device
    )
    if legacy:
        # The old path, with the same table kernels and a lookup-owned FIFO.
        for sharding in model._sharding_type_to_sharding.values():
            sharding._input_postprocessor = None
        # Use the prior gather/pooling kernels as the numerical control.
        for table in tables(model):
            table._native_lookup = LegacySequenceLookup()
    return model


def batch(start=0, count=12, dtype=torch.int64, empty_rank=False, scored=False):
    keys = torch.arange(start, start + count, device="cuda", dtype=dtype)
    if empty_rank and dist.get_rank() == 0:
        keys = keys[:0]
    # The two features sharing t0 repeat keys, both within and across features.
    values = torch.cat([keys, keys.flip(0), keys + 10000])
    features = KeyedJaggedTensor(
        keys=["f0", "f1", "f2"],
        values=values,
        lengths=torch.tensor([keys.numel()] * 3, device="cuda", dtype=torch.int64),
    )
    return (
        ScoredKeyedJaggedTensor.from_kjt(features, values.long() * 3 + 1)
        if scored
        else features
    )


def distribute(model, features, fused=False):
    ctx = model.create_context()
    request = model.input_dist(ctx, features)
    if fused:
        # This is how both SparseDist pipelines rebuild the second awaitable.
        request = FusedKJTListSplitsAwaitable(
            [request], [ctx], dist.group.WORLD
        ).wait()[0]
    else:
        request = request.wait()
    return ctx, request.wait()


def tables(model):
    return [emb.emb_module for lookup in model._lookups for emb in lookup._emb_modules]


def output(model, ctx, distributed):
    return model.compute_and_output_dist(ctx, distributed).wait()


def check_debug(result, features):
    for name, expected in zip(features.keys(), features.split([1, 1, 1])):
        torch.testing.assert_close(
            result[name].values(),
            (expected.values().long() % 100000).float()[:, None].expand(-1, 8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result[name].weights(), expected.values(), rtol=0, atol=0, check_dtype=False
        )


def test_split_fusion_keeps_each_collections_receiver_state():
    if dist.get_world_size() < 2:
        pytest.skip("A real split collective requires multiple ranks")
    models = [make_model(), make_model()]
    # Initialize the lazy distributors, then apply the same native fusion
    # override as TrainPipelineSparseDist (including to both collections).
    for model in models:
        ctx, received = distribute(model, batch(start=1000))
        result = output(model, ctx, received)
        sum(jt.values().sum() for jt in result.values()).backward()
    _override_input_dist_forwards(models)
    features = [batch(start=0), batch(start=100)]
    contexts = [model.create_context() for model in models]
    requests = [
        model.input_dist(ctx, inputs)
        for model, ctx, inputs in zip(models, contexts, features)
    ]
    assert all(
        isinstance(awaitable, KJTSplitsAllToAllMeta)
        for request in requests
        for awaitable in request.awaitables
    )
    fused = FusedKJTListSplitsAwaitable(requests, contexts, dist.group.WORLD)
    assert fused._splits_awaitable is not None
    arrived = [request.wait() for request in fused.wait()]
    for model, ctx, inputs, received in zip(models, contexts, features, arrived):
        assert isinstance(received[0], ResolvedEmbeddingInput)
        result = output(model, ctx, received)
        check_debug(result, inputs)
        sum(jt.values().sum() for jt in result.values()).backward()
    assert all(
        torch.count_nonzero(
            (table.cache or table.tables)._state.key_index_map._ref_counter
        ).item()
        == 0
        for model in models
        for table in tables(model)
    )


@pytest.mark.parametrize(
    "dedup,dtype,fused,layout",
    [
        (False, torch.int32, False, "hbm"),
        (True, torch.int64, True, "hbm"),
        (True, torch.int32, False, "cache"),
        (False, torch.int64, True, "cache_ps"),
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
        assert (
            torch.count_nonzero(
                (table.cache or table.tables)._state.key_index_map._ref_counter
            ).item()
            == 0
        )


@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize(
    "score",
    [
        DynamicEmbScoreStrategy.LFU,
        DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
        DynamicEmbScoreStrategy.CUSTOMIZED,
    ],
)
def test_optimizer_and_score_parity_with_legacy(dedup, score):
    model = make_model(dedup=dedup, score=score)
    legacy = make_model(dedup=dedup, legacy=True, score=score)
    for iteration in range(3):
        features = batch(
            start=iteration * 4,
            scored=score == DynamicEmbScoreStrategy.PRECISE_CUSTOMIZED,
        )
        results = []
        for current in [model, legacy]:
            if score == DynamicEmbScoreStrategy.CUSTOMIZED:
                for table in tables(current):
                    table.set_score({name: iteration + 1 for name in table.table_names})
            ctx, arrived = distribute(current, features)
            result = output(current, ctx, arrived)
            if iteration == 0:
                check_debug(result, features)
            sum(jt.values().sum() for jt in result.values()).backward()
            results.append(result)
        for name in results[0]:
            torch.testing.assert_close(
                results[0][name].values(), results[1][name].values()
            )
        for table, old_table in zip(tables(model), tables(legacy)):
            for tid in range(len(table.table_names)):
                a = list(
                    table.tables.export_keys_values(torch.device("cpu"), table_id=tid)
                )
                b = list(
                    old_table.tables.export_keys_values(
                        torch.device("cpu"), table_id=tid
                    )
                )

                # GPU insertion/export order is not stable; compare by key.
                def sorted_columns(chunks):
                    columns = [
                        torch.cat(pieces) if pieces[0] is not None else None
                        for pieces in zip(*chunks)
                    ]
                    order = columns[0].argsort()
                    return [
                        column[order] if column is not None else None
                        for column in columns
                    ]

                for x, y in zip(sorted_columns(a), sorted_columns(b)):
                    torch.testing.assert_close(x, y)


def test_expansion_and_out_of_order_batches():
    model = make_model(
        init_capacity=1024,
        max_capacity=16384,
        score=DynamicEmbScoreStrategy.NO_EVICTION,
        optimizer=EmbOptimType.SGD,
    )
    first, second = batch(count=64), batch(count=1000 * dist.get_world_size())
    first_ctx, first_input = distribute(model, first)
    generations = [table.tables._state.key_map_generation for table in tables(model)]
    second_ctx, second_input = distribute(model, second)
    assert any(
        table.tables._state.key_map_generation > generation
        for table, generation in zip(tables(model), generations)
    )
    # Explicit per-batch metadata lets this batch run before the first one.
    with torch.no_grad():
        check_debug(output(model, second_ctx, second_input), second)
    first_result = output(model, first_ctx, first_input)
    check_debug(first_result, first)
    sum(jt.values().sum() for jt in first_result.values()).backward()
    assert all(
        (table.tables.key_index_map._ref_counter == 0).all() for table in tables(model)
    )


def test_request_local_receiver_types_are_collectible():
    from dynamicemb.shard.input_dist import resolve_at_receive

    features = KeyedJaggedTensor(
        keys=["f"], values=torch.tensor([7]), lengths=torch.tensor([1])
    )
    types = []
    for _ in range(32):
        source = resolve_at_receive(features, lambda received: received)
        assert isinstance(source, KeyedJaggedTensor)
        types.append(weakref.ref(type(source)))
    del source
    gc.collect()
    assert all(ref() is None for ref in types)


def test_receiver_type_pool_keeps_live_requests_independent():
    from dynamicemb.shard.input_dist import _ReceiverTypePool, resolve_at_receive

    features = KeyedJaggedTensor(
        keys=["f"], values=torch.tensor([7]), lengths=torch.tensor([1])
    )
    pool = _ReceiverTypePool()
    first = resolve_at_receive(features, lambda received: received, pool)
    second = resolve_at_receive(features, lambda received: received, pool)
    first_type = type(first)
    assert first_type is not type(second)
    second_result = _finish_receiver(second, 20)
    first_result = _finish_receiver(first, 10)
    assert _finish_receiver(first, 999) is first_result
    del first
    gc.collect()
    third = resolve_at_receive(features, lambda received: received, pool)
    assert type(third) is first_type
    third_result = _finish_receiver(third, 30)
    assert _finish_receiver(second, 999) is second_result
    for result, expected in [
        (first_result, 10),
        (second_result, 20),
        (third_result, 30),
    ]:
        torch.testing.assert_close(result.values(), torch.tensor([expected]))


def test_receiver_type_pool_bounds_idle_types_without_retaining_requests():
    from dynamicemb.shard.input_dist import _ReceiverTypePool, resolve_at_receive

    features = KeyedJaggedTensor(
        keys=["f"], values=torch.tensor([7]), lengths=torch.tensor([1])
    )
    pool = _ReceiverTypePool()
    requests = [
        resolve_at_receive(features, lambda received: received, pool)
        for _ in range(128)
    ]
    assert len({type(request) for request in requests}) == len(requests)
    types = [weakref.ref(type(request)) for request in requests]
    resolvers = [weakref.ref(request._receiver_resolver) for request in requests]
    results = [
        weakref.ref(_finish_receiver(request, i)) for i, request in enumerate(requests)
    ]
    del requests
    gc.collect()
    assert all(ref() is None for ref in resolvers)
    assert all(ref() is None for ref in results)
    retained = {ref() for ref in types if ref() is not None}
    assert 0 < len(retained) <= 8


@pytest.mark.parametrize("copy_kind", ["deepcopy", "pickle"])
def test_receiver_type_pool_copy_has_independent_request_owners(copy_kind):
    from dynamicemb.shard.input_dist import _ReceiverTypePool, resolve_at_receive

    features = KeyedJaggedTensor(
        keys=["f"], values=torch.tensor([7]), lengths=torch.tensor([1])
    )
    pool = _ReceiverTypePool()
    warm = resolve_at_receive(features, lambda received: received, pool)
    del warm
    gc.collect()
    other = (
        copy.deepcopy(pool)
        if copy_kind == "deepcopy"
        else pickle.loads(pickle.dumps(pool))
    )
    first = resolve_at_receive(features, lambda received: received, pool)
    second = resolve_at_receive(features, lambda received: received, other)
    second_result = _finish_receiver(second, 20)
    first_result = _finish_receiver(first, 10)
    torch.testing.assert_close(first_result.values(), torch.tensor([10]))
    torch.testing.assert_close(second_result.values(), torch.tensor([20]))


@pytest.mark.parametrize("clear_inputs", [False, True])
def test_native_receiver_type_stays_owned_until_awaitable_is_dropped(clear_inputs):
    from dynamicemb.shard.input_dist import _ReceiverTypePool, resolve_at_receive
    from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable

    world = dist.get_world_size()
    if world < 2:
        pytest.skip("Requires native multi-rank tensors awaitables")
    if clear_inputs and not hasattr(KJTAllToAllTensorsAwaitable, "clear_inputs"):
        pytest.skip("TorchRec does not expose clear_inputs")
    pool = _ReceiverTypePool()

    def source(offset):
        features = KeyedJaggedTensor(
            keys=[f"f{i}" for i in range(world)],
            values=torch.arange(world, device="cuda") + 100 * dist.get_rank() + offset,
            lengths=torch.ones(world, device="cuda", dtype=torch.int64),
        )
        return resolve_at_receive(features, lambda received: received, pool)

    distribution = KJTAllToAll(dist.group.WORLD, [1] * world)
    first, second = source(0), source(10000)
    first_type = type(first)
    first_request = distribution(first).wait()
    second_request = distribution(second).wait()
    if clear_inputs:
        first_request.clear_inputs()
    del first
    gc.collect()
    third = source(20000)
    assert type(third) is not first_type
    second_result = second_request.wait()
    first_result = first_request.wait()
    assert first_request.wait() is first_result
    expected = torch.arange(world, device="cuda") * 100 + dist.get_rank()
    torch.testing.assert_close(first_result.values(), expected)
    torch.testing.assert_close(second_result.values(), expected + 10000)


def test_globally_empty_input():
    model = make_model(dedup=True)
    features = batch(count=0)
    ctx, arrived = distribute(model, features)
    result = output(model, ctx, arrived)
    check_debug(result, features)
    sum(jt.values().sum() for jt in result.values()).backward()
    assert all(
        (table.tables.key_index_map._ref_counter == 0).all() for table in tables(model)
    )


def test_input_dist_validation_precedes_table_mutation():
    model = make_model(
        table_options={name: {"training": False} for name in ["t0", "t1"]},
    )
    with pytest.raises(RuntimeError, match="eval mode"):
        distribute(model, batch())
    for table in tables(model):
        assert (table.tables.key_index_map._ref_counter == 0).all()
        assert (
            sum(
                chunk[0].numel()
                for chunk in table.tables.export_keys_values(torch.device("cpu"))
            )
            == 0
        )


def test_mixed_dynamic_and_torchrec_groups():
    model = make_model(static_table=True)
    for lookup in model._lookups:
        for emb in lookup._emb_modules:
            if not isinstance(emb, BatchedDynamicEmbedding):
                for weight in emb.split_embedding_weights():
                    weight.fill_(2.0)
    features = batch()
    features = KeyedJaggedTensor(
        keys=features.keys(),
        lengths=features.lengths(),
        values=features.values() % 4096,
    )
    ctx, arrived = distribute(model, features)
    groups = arrived[0].split(model._lookups[0]._feature_splits)
    assert sorted(isinstance(group, ResolvedEmbeddingInput) for group in groups) == [
        False,
        True,
    ]
    model.prefetch(arrived, ctx=ctx)
    result = output(model, ctx, arrived)
    for name in ["f0", "f1"]:
        expected = features.to_dict()[name].values().float()[:, None].expand(-1, 8)
        torch.testing.assert_close(result[name].values(), expected)
    assert (result["f2"].values() == 2).all()
    sum(jt.values().sum() for jt in result.values()).backward()


def test_non_admitted_rows_survive_input_dist():
    strategy = FrequencyAdmissionStrategy(threshold=5, counter=KVCounter(4096))
    model = make_model(
        table_options={name: {"admit_strategy": strategy} for name in ["t0", "t1"]}
    )
    features = batch()
    for iteration in range(5):
        ctx, arrived = distribute(model, features)
        if iteration == 0:
            assert any(
                group.prefetch_state.non_admitted_positions is not None
                for group in arrived[0].split(model._lookups[0]._feature_splits)
            )
        with torch.no_grad():
            check_debug(output(model, ctx, arrived), features)
        assert all(
            (table.tables.key_index_map._ref_counter == 0).all()
            for table in tables(model)
        )


def test_eviction_reinitializes_reused_value_rows_and_adam_state():
    model = make_model(init_capacity=256, max_capacity=256)
    seen_rows = {}
    reused = False
    for iteration in range(20):
        features = batch(start=iteration * 32, count=32)
        ctx, arrived = distribute(model, features)
        groups = arrived[0].split(model._lookups[0]._feature_splits)
        for table, group in zip(tables(model), groups):
            state = group.prefetch_state
            assert (state.value_row_indices >= 0).all()
            values = load_from_flat(
                table.tables._state,
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


class PipelineModel(torch.nn.Module):
    def __init__(self, collection):
        super().__init__()
        self.collection = collection
        self.dense = torch.nn.Parameter(torch.zeros((), device="cuda"))

    def forward(self, features):
        result = self.collection(features)
        loss = self.dense * 0
        for name in ["f0", "f1", "f2"]:
            loss = loss + result[name].values().sum()
        return loss.unsqueeze(0), loss.detach()


@pytest.mark.parametrize(
    "pipeline_type,clear_inputs",
    [
        (TrainPipelineSparseDist, False),
        (PrefetchTrainPipelineSparseDist, False),
        (TrainPipelineSparseDist, True),
    ],
)
@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
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
    assert all(
        torch.count_nonzero(
            (table.cache or table.tables)._state.key_index_map._ref_counter
        ).item()
        == 0
        for table in tables(collection)
    )


def test_native_receiver_resolution_runs_once():
    model = make_model()
    ctx = model.create_context()
    request = model.input_dist(ctx, batch()).wait()
    first = request.wait()
    counts = [
        (table.cache or table.tables).key_index_map._ref_counter.clone()
        for table in tables(model)
    ]
    second = request.wait()
    assert first[0] is second[0]
    for table, expected in zip(tables(model), counts):
        torch.testing.assert_close(
            (table.cache or table.tables).key_index_map._ref_counter, expected
        )
    result = output(model, ctx, first)
    sum(jt.values().sum() for jt in result.values()).backward()


def test_optimizer_tracker_observes_original_keys():
    from torchrec.distributed.embedding_lookup import (
        GroupedEmbeddingsLookup as NativeLookup,
    )

    if not hasattr(NativeLookup, "register_optim_state_tracker_fn"):
        pytest.skip("Optimizer trackers are not available in this TorchRec version")
    model = make_model()
    ctx, arrived = distribute(model, batch())
    seen, expected = [], {}
    for lookup, features in zip(model._lookups, arrived):
        for emb, group in zip(
            lookup._emb_modules, features.split(lookup._feature_splits)
        ):
            state = group.prefetch_state
            expected[id(emb)] = state.unique_keys[state.reverse_indices]
        lookup.register_optim_state_tracker_fn(
            lambda features, values, emb: seen.append(
                (id(emb), features.values().clone())
            )
        )
    result = output(model, ctx, arrived)
    assert len(seen) == len(expected)
    for emb, keys in seen:
        torch.testing.assert_close(keys, expected[emb])
    sum(jt.values().sum() for jt in result.values()).backward()


def _finish_receiver(source, value):
    return type(source).dist_init(
        keys=["f"],
        tensors=[torch.tensor([1]), torch.tensor([value])],
        variable_stride_per_key=False,
        num_workers=1,
        recat=None,
        stride_per_rank=[1],
    )


@pytest.mark.parametrize(
    "layout",
    [
        "hbm",
        "cache",
        "cache_ps",
    ],
)
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


@pytest.mark.parametrize(
    "layout,optimizer,reverse_backward,drop_inputs",
    [
        ("hbm", EmbOptimType.SGD, False, False),
        ("hbm", EmbOptimType.ADAM, True, True),
        ("cache", EmbOptimType.ADAM, True, False),
        ("cache_ps", EmbOptimType.ADAM, True, True),
    ],
)
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


@pytest.mark.parametrize(
    "layout", [x for x in ["hbm", "cache", "host", "hybrid"] if x in ENABLED_LAYOUTS]
)
def test_fp8_receiver_decodes_rows_and_preserves_small_gradients(layout):
    initial = 12.0 / 256.0
    options = {
        name: dict(
            embedding_dtype=torch.float8_e4m3fn,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=initial
            ),
        )
        for name in ["t0", "t1"]
    }
    model = make_model(
        layout=layout,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        table_options=options,
    )
    assert all(t.embedding_dtype == torch.float8_e4m3fn for t in tables(model))
    features = batch()
    ctx, arrived = distribute(model, features)
    assert isinstance(arrived[0], ResolvedEmbeddingInput)
    with patch(
        "dynamicemb.batched_dynamicemb_function._find_keys",
        side_effect=AssertionError("resolved FP8 lookup rehashed keys"),
    ):
        result = output(model, ctx, arrived)
    values = torch.cat([jt.values() for jt in result.values()])
    torch.testing.assert_close(values, torch.full_like(values, initial), rtol=0, atol=0)
    (values.sum() * 1e-8).backward()
    with torch.no_grad():
        ctx, arrived = distribute(model, features)
        trained = output(model, ctx, arrived)
        before_eval = {k: jt.values().clone() for k, jt in trained.items()}
    # A gradient cast to fp16 or fp8 would flush these updates to zero.
    assert all((v - initial).abs().mean().item() > 1e-3 for v in before_eval.values())
    model.eval()
    with torch.no_grad():
        actual = model(features).wait()
    for key in actual:
        torch.testing.assert_close(
            actual[key].values(), before_eval[key], rtol=0, atol=0
        )


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
