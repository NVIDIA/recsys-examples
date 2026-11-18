# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from dynamicemb import EmbOptimType
from dynamicemb.shard.resolved_input import ResolvedEmbeddingInput
from dynamicemb.shard.rw_sharding import GroupedPooledEmbeddingsLookup
from fbgemm_gpu.split_embedding_configs import SparseType
from test_hbm_receiver import (  # noqa: F401
    PipelineModel,
    batch,
    distribute,
    distributed,
    make_model,
    output,
    tables,
)
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    KeyedJaggedTensor,
    PoolingType,
)
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import ShardingType

LAYOUTS = ["hbm", "cache", "host", "hybrid", "ps", "cache_ps"]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "pooling,weighted,optimizer,dims,native_dense",
    [
        (PoolingType.SUM, False, EmbOptimType.ADAM, (8, 8), True),
        (PoolingType.MEAN, False, EmbOptimType.SGD, (7, 11), False),
        (PoolingType.SUM, True, EmbOptimType.ADAM, (7, 11), False),
    ],
    ids=["dense-sum-adam", "mixed-mean-sgd", "weighted-ebc-adam"],
)
def test_pooled_lookup_and_update_match_unsharded_torchrec(
    layout, pooling, weighted, optimizer, dims, native_dense
):
    model = make_model(
        layout=layout,
        pooling=pooling,
        weighted=weighted,
        dims=dims,
        optimizer=optimizer,
    )
    configs = [
        EmbeddingBagConfig(
            name="t0",
            embedding_dim=dims[0],
            num_embeddings=128,
            feature_names=["f0", "f1"],
            pooling=pooling,
        ),
        EmbeddingBagConfig(
            name="t1",
            embedding_dim=dims[1],
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
        weights=(
            torch.arange(1, 8, device="cuda", dtype=torch.float32) / 3
            if weighted
            else None
        ),
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
            native = (
                table._native_lookup._dense_module(weighted)
                if native_dense
                else table._native_lookup._module(weighted)
            )
            method = "_operation" if native_dense else "forward"
            native_calls.append(
                checks.enter_context(
                    patch.object(native, method, wraps=getattr(native, method))
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
@pytest.mark.parametrize(
    "dtype,dims,pooling,weighted",
    [
        (torch.float16, (8, 8), PoolingType.SUM, True),
        (torch.bfloat16, (7, 11), PoolingType.MEAN, False),
        (torch.float16, (7, 11), PoolingType.SUM, False),
        (torch.bfloat16, (8, 8), PoolingType.SUM, True),
    ],
    ids=[
        "weighted-fp16-dense",
        "bf16-mixed-mean",
        "fp16-mixed-sum",
        "weighted-bf16-fallback",
    ],
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
        weights=(
            torch.arange(1, 8, device="cuda", dtype=torch.float32) / 2
            if weighted
            else None
        ),
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


@pytest.mark.parametrize(
    "pooling,dims,weighted",
    [
        (None, (7, 7), False),
        (PoolingType.SUM, (8, 8), False),
        (PoolingType.SUM, (7, 11), True),
    ],
    ids=["ec-fallback", "dense-pooled", "weighted-ebc-fallback"],
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
