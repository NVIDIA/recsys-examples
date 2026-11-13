# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolved input_dist and TorchRec lookup, on one or more real GPUs.

    python -m pytest test/unit_tests/test_input_dist_indices.py -q
    torchrun --standalone --nproc_per_node=2 -m pytest \
        test/unit_tests/test_input_dist_indices.py -q
"""

import gc
import os
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
from dynamicemb.resolved_storage import matching_rows
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from dynamicemb.shard.rw_sharding import GroupedPooledEmbeddingsLookup
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
from torchrec.distributed.embedding_sharding import FusedKJTListSplitsAwaitable
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.distributed.planner import Topology
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.types import DataType

# The repository's functional in-memory PS implements the actual Storage API.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_batched_dynamic_embedding_tables_v2 import PyDictStorage

LAYOUTS = ["hbm", "cache", "host", "hybrid", "ps", "cache_ps"]


@pytest.fixture(scope="module", autouse=True)
def distributed():
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
                        else max_capacity * config.embedding_dim
                        if layout in ["cache", "hybrid", "cache_ps"]
                        else 0
                    ),
                    caching=layout in ["cache", "cache_ps"],
                    external_storage=PyDictStorage
                    if layout in ["ps", "cache_ps"]
                    else None,
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
        request = FusedKJTListSplitsAwaitable([request], [ctx], None).wait()[0]
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
            native = table._native_lookup._module(False)
            native_calls.append(
                checks.enter_context(
                    patch.object(native, "forward", wraps=native.forward)
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


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "pooling,weighted",
    [(PoolingType.SUM, False), (PoolingType.MEAN, False), (PoolingType.SUM, True)],
)
@pytest.mark.parametrize("optimizer", [EmbOptimType.SGD, EmbOptimType.ADAM])
def test_pooled_lookup_and_update_match_unsharded_torchrec(
    layout, pooling, weighted, optimizer
):
    model = make_model(
        layout=layout,
        pooling=pooling,
        weighted=weighted,
        dims=(7, 11),
        optimizer=optimizer,
    )
    configs = [
        EmbeddingBagConfig(
            name="t0",
            embedding_dim=7,
            num_embeddings=128,
            feature_names=["f0", "f1"],
            pooling=pooling,
        ),
        EmbeddingBagConfig(
            name="t1",
            embedding_dim=11,
            num_embeddings=128,
            feature_names=["f2"],
            pooling=pooling,
        ),
    ]
    reference = EmbeddingBagCollection(
        configs, is_weighted=weighted, device=torch.device("cuda")
    )
    with torch.no_grad():
        for embedding in reference.embedding_bags.values():
            embedding.weight.copy_(
                torch.arange(128, device="cuda")[:, None].expand_as(embedding.weight)
            )
    ref_opt = (
        torch.optim.SGD(reference.parameters(), lr=0.1)
        if optimizer == EmbOptimType.SGD
        else torch.optim.Adam(reference.parameters(), lr=0.1, eps=1e-8)
    )
    features = KeyedJaggedTensor(
        keys=["f0", "f1", "f2"],
        values=torch.tensor([2, 2, 4, 3, 3, 7, 7], device="cuda"),
        lengths=torch.tensor([3, 0, 0, 2, 1, 1], device="cuda"),
        weights=torch.arange(1, 8, device="cuda", dtype=torch.float32) / 3
        if weighted
        else None,
    )
    ctx, arrived = distribute(model, features)
    assert isinstance(arrived[0], ResolvedEmbeddingInput)
    with ExitStack() as checks:
        checks.enter_context(
            patch(
                "dynamicemb.batched_dynamicemb_function._find_keys",
                side_effect=AssertionError("lookup hashed keys"),
            )
        )
        native_calls = []
        for table in tables(model):
            native = table._native_lookup._module(weighted)
            native_calls.append(
                checks.enter_context(
                    patch.object(native, "forward", wraps=native.forward)
                )
            )
            checks.enter_context(
                patch.object(
                    table.tables,
                    "find",
                    side_effect=AssertionError("lookup asked storage for keys"),
                )
            )
        actual = output(model, ctx, arrived)
        assert all(call.call_count == 1 for call in native_calls)
    expected = reference(features)
    torch.testing.assert_close(actual.values(), expected.values())
    actual.values().sum().backward()
    expected.values().sum().backward()
    ref_opt.step()
    model.eval()
    with torch.no_grad():
        actual = output(model, *distribute(model, features))
        expected = reference(features)
    torch.testing.assert_close(actual.values(), expected.values())
    assert all(not table._active_rows.states for table in tables(model))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dims", [(8, 8), (7, 11)])
@pytest.mark.parametrize(
    "pooling,weighted",
    [(PoolingType.SUM, False), (PoolingType.MEAN, False), (PoolingType.SUM, True)],
)
def test_low_precision_pooling_and_update(layout, dtype, dims, pooling, weighted):
    model = make_model(
        layout=layout,
        embedding_dtype=dtype,
        dims=dims,
        pooling=pooling,
        weighted=weighted,
        optimizer=EmbOptimType.SGD,
        learning_rate=0.25,
    )
    # Storage is low precision; pooling and its sample weights are fp32.
    # Autograd through weight.float() rounds gradients back to the low-precision
    # leaf, matching the fused SGD update without duplicating its implementation.
    weights = torch.nn.ParameterList(
        [
            torch.nn.Parameter(
                torch.arange(4096, device="cuda")
                .to(dtype)[:, None]
                .expand(-1, dim)
                .clone()
            )
            for dim in dims
        ]
    )
    optimizer = torch.optim.SGD(weights, lr=0.25)
    features = KeyedJaggedTensor(
        keys=["f0", "f1", "f2"],
        values=torch.tensor([1, 2048, 256, 1, 1, 256, 1], device="cuda"),
        lengths=torch.tensor([2, 0, 0, 2, 2, 1], device="cuda"),
        weights=torch.arange(1, 8, device="cuda", dtype=torch.float32) / 2
        if weighted
        else None,
    )

    def reference():
        return torch.cat(
            [
                torch.nn.functional.embedding_bag(
                    jt.values(),
                    weights[tid].float(),
                    jt.offsets(),
                    mode="sum" if pooling == PoolingType.SUM else "mean",
                    per_sample_weights=jt.weights() if weighted else None,
                    include_last_offset=True,
                )
                for tid, jt in zip([0, 0, 1], features.to_dict().values())
            ],
            dim=1,
        )

    actual = output(model, *distribute(model, features)).values()
    expected = reference()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        actual = output(model, *distribute(model, features)).values()
        torch.testing.assert_close(actual, reference())
    assert all(not table._active_rows.states for table in tables(model))


def test_half_pooling_keeps_finite_fp32_output():
    model = make_model(pooling=PoolingType.SUM, embedding_dtype=torch.float16)
    features = batch(start=40000, count=2)
    with torch.no_grad():
        result = output(model, *distribute(model, features))
    for name, jt in features.to_dict().items():
        expected = jt.values().to(torch.float16).float().sum().expand(1, 8)
        assert result[name].isfinite().all()
        torch.testing.assert_close(result[name], expected, rtol=0, atol=0)


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

    def delayed_match(*args):
        indices = matching_rows(*args)
        # Delay the actual copy after matching has finished its host syncs.
        # Dropping the source below must not recycle its allocation meanwhile.
        torch.cuda._sleep(100_000_000)
        return indices

    with torch.cuda.stream(copy_stream), table._table_gate():
        with patch(
            "dynamicemb.resolved_storage.matching_rows", side_effect=delayed_match
        ):
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
    "pipeline_type", [TrainPipelineSparseDist, PrefetchTrainPipelineSparseDist]
)
@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_native_sparse_dist_pipeline(pipeline_type, scored, layout):
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


class PooledPipelineModel(PipelineModel):
    def forward(self, features):
        result = self.collection(features)
        loss = result.values().sum() + self.dense * 0
        return loss.unsqueeze(0), loss.detach()


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "pipeline_type", [TrainPipelineSparseDist, PrefetchTrainPipelineSparseDist]
)
def test_native_pooled_pipeline(layout, pipeline_type):
    def make():
        return PooledPipelineModel(
            make_model(
                layout=layout, optimizer=EmbOptimType.SGD, pooling=PoolingType.SUM
            )
        )

    model, control = make(), make()
    pipeline = pipeline_type(
        model,
        torch.optim.SGD([model.dense], lr=0.1),
        torch.device("cuda", torch.cuda.current_device()),
        execute_all_batches=True,
    )
    inputs = [batch(start=iteration * 4) for iteration in range(4)]
    iterator = iter(inputs)
    for features in inputs:
        actual = pipeline.progress(iterator)
        loss, expected = control(features)
        loss.sum().backward()
        torch.testing.assert_close(actual, expected)
    with pytest.raises(StopIteration):
        pipeline.progress(iterator)
    assert all(not table._active_rows.states for table in tables(model.collection))


class ShiftKeys(BaseGroupedFeatureProcessor):
    def forward(self, features):
        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values() + 100,
            lengths=features.lengths(),
            weights=features.weights_or_none(),
        )


def processed_model(processor, weighted=False):
    model = make_model(
        pooling=PoolingType.SUM, optimizer=EmbOptimType.SGD, weighted=weighted
    )
    (old_lookup,) = model._lookups
    configs = [
        replace(
            config,
            has_feature_processor=True,
            fused_params={**config.fused_params, "output_dtype": SparseType.FP32},
        )
        for config in old_lookup.grouped_configs
    ]
    lookup = GroupedPooledEmbeddingsLookup(
        configs,
        pg=dist.group.WORLD,
        device=torch.device("cuda", torch.cuda.current_device()),
        feature_processor=processor,
        sharding_type=ShardingType.ROW_WISE,
    )
    model._lookups[0] = lookup
    (sharding,) = model._embedding_shardings
    sharding._input_postprocessor = lookup.resolve_input
    return model, lookup


def test_feature_processing_precedes_resolution_and_runs_once():
    model, lookup = processed_model(ShiftKeys())
    features = batch()
    ctx, arrived = distribute(model, features)
    for group in arrived[0].split(lookup._feature_splits):
        assert (group.prefetch_state.unique_keys >= 100).all()
    result = output(model, ctx, arrived)
    for i, feature in enumerate(features.split([1, 1, 1])):
        expected = (feature.values().float() + 100).sum().expand(1, 8)
        torch.testing.assert_close(result[f"f{i}"], expected)
    result.values().sum().backward()


class LearnedWeights(BaseGroupedFeatureProcessor):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), device="cuda"))

    def forward(self, features):
        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            lengths=features.lengths(),
            weights=features.weights() * self.scale,
        )


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_feature_processor_preserves_no_grad_contract(grad_enabled):
    model, lookup = processed_model(LearnedWeights(), weighted=True)
    raw = batch()
    features = KeyedJaggedTensor(
        keys=raw.keys(),
        values=raw.values(),
        lengths=raw.lengths(),
        weights=torch.ones(raw.values().numel(), device="cuda"),
    )
    with torch.set_grad_enabled(grad_enabled):
        if grad_enabled:
            with pytest.raises(ValueError, match="requires_grad=True"):
                distribute(model, features)
        else:
            result = output(model, *distribute(model, features))
            for i, feature in enumerate(features.split([1, 1, 1])):
                torch.testing.assert_close(
                    result[f"f{i}"], feature.values().float().sum().expand(1, 8)
                )
    assert all(not table._active_rows.states for table in tables(model))
    for table in tables(model):
        assert (table.tables.key_index_map._ref_counter == 0).all()
