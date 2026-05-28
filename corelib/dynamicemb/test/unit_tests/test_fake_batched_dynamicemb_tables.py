# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the DYNAMICEMB_FAKE_MODE dispatch on BatchedDynamicEmbeddingTablesV2.

The headline test is ``test_attribute_parity_with_real``: it constructs both
the real and the fake instances with the same arguments and asserts that
every attribute the real instance carries also exists on the fake one. This
guards against the maintenance hazard that Fake's __init__ replicates Real's
host-side bookkeeping by hand — without this check, adding a new ``self.x =``
to the real __init__ would silently break fake mode at call time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

# Pre-enable fake mode before importing dynamicemb so that, on hosts where the
# real dynamicemb_extensions C++ extension is not installed, the package's
# bootstrap installs the stub and ``import dynamicemb`` succeeds. The parity
# test toggles the env back to "0" at runtime (under @requires_cuda), and
# __new__ re-reads it on every construction, so this default does not pin the
# whole file into fake mode.
os.environ.setdefault("DYNAMICEMB_FAKE_MODE", "1")

import pytest
import torch

from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb import _fake
from dynamicemb._fake import FakeBatchedDynamicEmbeddingTablesV2
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required to construct the real instance"
)

_FAKE_ENV = "DYNAMICEMB_FAKE_MODE"

# Attributes that exist on real instances but legitimately do not on fake,
# because they are tied to GPU work that fake intentionally skips. Keep this
# list small and well-justified — every entry is a known correctness gap.
_FAKE_EXEMPT_ATTRS: set = set()


def _build_table_options(dim: int = 8, num_tables: int = 2) -> list:
    return [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            dim=dim,
            max_capacity=1024,
            local_hbm_for_values=1 << 20,
            bucket_capacity=128,
            initializer_args=DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.DEBUG,
            ),
        )
        for _ in range(num_tables)
    ]


def _build_kwargs(num_tables: int = 2) -> dict:
    return dict(
        table_options=_build_table_options(num_tables=num_tables),
        output_dtype=torch.float32,
        table_names=[f"t_{i}" for i in range(num_tables)],
        pooling_mode=DynamicEmbPoolingMode.SUM,
        optimizer=EmbOptimType.SGD,
        learning_rate=1.0,
    )


@contextmanager
def _fake_env(value: Optional[str]):
    """Temporarily set/unset DYNAMICEMB_FAKE_MODE; restore on exit.

    ``value=None`` removes the variable; otherwise sets it to ``value``.
    ``_fake.is_enabled()`` reads ``os.environ`` on every call, so no
    module reload is needed — and reloading would invalidate the cached
    Fake class identity, breaking ``isinstance`` checks.
    """
    prev = os.environ.get(_FAKE_ENV)
    if value is None:
        os.environ.pop(_FAKE_ENV, None)
    else:
        os.environ[_FAKE_ENV] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_FAKE_ENV, None)
        else:
            os.environ[_FAKE_ENV] = prev


def _construct(env_value: str = "1", **kwarg_overrides):
    """Construct via the public class with the env var set, kwargs merged."""
    kwargs = _build_kwargs()
    kwargs.update(kwarg_overrides)
    with _fake_env(env_value):
        return BatchedDynamicEmbeddingTablesV2(**kwargs)


def test_env_on_dispatches_to_fake_subclass():
    fake = _construct()
    assert isinstance(fake, FakeBatchedDynamicEmbeddingTablesV2)
    # Subclass check: external isinstance() guards in the codebase must still
    # accept fake instances.
    assert isinstance(fake, BatchedDynamicEmbeddingTablesV2)


# ---------------------------------------------------------------------------
# Attribute parity — the core drift detector.
# ---------------------------------------------------------------------------


@requires_cuda
def test_attribute_parity_with_real():
    """``dir(real) - dir(fake)`` must be empty, and every instance attribute
    on real must exist on fake. Fake may add extras (e.g. ``_fake_device``).
    """
    real = _construct(env_value="0")
    fake = _construct(env_value="1")

    # Sanity: env routing must have produced two different concrete classes,
    # otherwise the comparisons below would be self-comparisons (vacuous).
    # The actual dispatch / isinstance properties are covered by the
    # test_env_* tests in this file.
    assert type(real) is not type(fake)

    # 1. dir() parity — covers class-level methods and attributes.
    real_dir = set(dir(real))
    fake_dir = set(dir(fake))
    missing_in_dir = (real_dir - fake_dir) - _FAKE_EXEMPT_ATTRS
    assert not missing_in_dir, (
        "Fake is missing these attributes/methods present on real "
        f"(add them or extend _FAKE_EXEMPT_ATTRS with justification): "
        f"{sorted(missing_in_dir)}"
    )

    # 2. Instance attribute parity — the real drift signal. ``vars()`` shows
    #    what __init__ actually assigned, including private fields like
    #    ``_storage``, ``_optimizer`` etc. that aren't on the class.
    real_attrs = set(vars(real).keys())
    fake_attrs = set(vars(fake).keys())
    missing_in_init = (real_attrs - fake_attrs) - _FAKE_EXEMPT_ATTRS
    assert not missing_in_init, (
        "Fake.__init__ did not assign these instance attributes that "
        "Real.__init__ assigns. Update FakeBatchedDynamicEmbeddingTablesV2 "
        "to keep host-side bookkeeping in sync. "
        f"Drifted attributes: {sorted(missing_in_init)}"
    )

    # 3. nn.Module sub-collections — buffers and parameters.
    real_buffer_names = {n for n, _ in real.named_buffers()}
    fake_buffer_names = {n for n, _ in fake.named_buffers()}
    assert (
        real_buffer_names <= fake_buffer_names
    ), f"Fake is missing buffers: {sorted(real_buffer_names - fake_buffer_names)}"
    real_param_names = {n for n, _ in real.named_parameters()}
    fake_param_names = {n for n, _ in fake.named_parameters()}
    assert (
        real_param_names <= fake_param_names
    ), f"Fake is missing parameters: {sorted(real_param_names - fake_param_names)}"


# ---------------------------------------------------------------------------
# Behavioural smoke tests for the fake path — these do NOT require CUDA.
# ---------------------------------------------------------------------------


def test_env_off_does_not_dispatch_to_fake():
    """With the env var unset, the dispatch helper returns ``None`` so
    ``BatchedDynamicEmbeddingTablesV2.__new__`` falls through to the real
    path. Tests the dispatch directly (no real construction — that needs CUDA)."""
    with _fake_env(None):
        assert not _fake.is_enabled()
        assert _fake.maybe_get_fake_class() is None


def test_env_on_dispatch_helper_returns_cached_fake_class():
    """The dispatch helper returns the cached Fake class; identity must be
    stable, otherwise ``isinstance`` against the imported symbol breaks."""
    with _fake_env("1"):
        cls1 = _fake.maybe_get_fake_class()
        cls2 = _fake.maybe_get_fake_class()
        assert cls1 is cls2 is FakeBatchedDynamicEmbeddingTablesV2


def test_fake_forward_pooled_shape_and_backward():
    fake = _construct()
    # 2 tables × batch=4 → feature_batch=8 offsets entries → 9 offset values.
    indices = torch.arange(16, dtype=torch.int64)
    offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.int64)

    out = fake(indices, offsets)
    # pooling mode: [batch_size, total_D] = [4, 2*8]
    assert out.shape == (4, 16)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"

    # Backward must run without raising.
    out.sum().backward()
    assert fake._empty_tensor.grad is not None


def test_fake_forward_sequence_shape():
    fake = _construct(pooling_mode=DynamicEmbPoolingMode.NONE)

    indices = torch.arange(10, dtype=torch.int64)
    offsets = torch.tensor([0, 3, 5, 7, 8, 9, 10, 10, 10], dtype=torch.int64)
    out = fake(indices, offsets)
    # sequence mode: [indices.numel(), max_D]
    assert out.shape == (10, 8)


@contextmanager
def _single_rank_gloo():
    """Single-rank ``gloo`` process group for DMP wiring on CPU. Skips
    init if a group already exists in the same process."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="gloo")
    try:
        yield
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def test_fake_dmp_sequence_forward_backward():
    """Sequence-mode ``EmbeddingCollection`` wrapped in
    ``DistributedModelParallel`` under fake mode: forward must produce
    per-feature JaggedTensors with values of shape ``[N, embedding_dim]``,
    and ``loss.backward()`` must run without raising.

    Exercises the TorchRec compute-kernel wrapping path that the simpler
    direct-forward tests above do not cover (cf. README "Limitations").
    """
    import torch.distributed as dist
    import torchrec
    from dynamicemb.planner import (
        DynamicEmbeddingEnumerator,
        DynamicEmbeddingShardingPlanner,
        DynamicEmbParameterConstraints,
    )
    from dynamicemb.shard import DynamicEmbeddingCollectionSharder
    from fbgemm_gpu.split_embedding_configs import SparseType
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import Topology
    from torchrec.distributed.types import BoundsCheckMode, ShardingType

    embedding_dim = 8
    num_tables = 2
    num_embeddings = 1024
    batch_size = 4
    device = torch.device("cpu")

    with _single_rank_gloo():
        eb_configs = [
            torchrec.EmbeddingConfig(
                name=f"t_{i}",
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                feature_names=[f"cate_{i}"],
            )
            for i in range(num_tables)
        ]
        ebc = torchrec.EmbeddingCollection(
            device=torch.device("meta"),
            tables=eb_configs,
        )

        constraints = {
            f"t_{i}": DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                pooling_factors=[1.0],
                num_poolings=[1],
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=True,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=1 << 20,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG,
                    ),
                ),
            )
            for i in range(num_tables)
        }

        topology = Topology(
            local_world_size=1,
            world_size=1,
            compute_device=device.type,
            hbm_cap=1 << 30,
            ddr_cap=1 << 32,
            intra_host_bw=300e9,
            inter_host_bw=25e9,
        )
        enumerator = DynamicEmbeddingEnumerator(
            topology=topology,
            constraints=constraints,
        )
        planner = DynamicEmbeddingShardingPlanner(
            eb_configs=eb_configs,
            topology=topology,
            constraints=constraints,
            batch_size=batch_size,
            enumerator=enumerator,
        )
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params={
                "output_dtype": SparseType.FP32,
                "optimizer": EmbOptimType.SGD,
                "learning_rate": 1.0,
            },
            use_index_dedup=True,
        )
        plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)

        model = DistributedModelParallel(
            module=ebc,
            device=device,
            sharders=[sharder],
            plan=plan,
        )

        # One row per feature per sample → total indices per feature = batch_size.
        sparse_feature = torchrec.KeyedJaggedTensor(
            keys=[f"cate_{i}" for i in range(num_tables)],
            values=torch.arange(num_tables * batch_size, dtype=torch.int64),
            lengths=torch.ones(num_tables * batch_size, dtype=torch.int64),
        )

        ret = model(sparse_feature)

        # Sequence mode: each feature → JaggedTensor with values [N, D].
        assert set(ret.keys()) == {f"cate_{i}" for i in range(num_tables)}
        for feature_name, jt in ret.items():
            values = jt.values()
            assert (
                values.dim() == 2
            ), f"feature {feature_name}: expected 2D values, got {values.shape}"
            assert values.shape[1] == embedding_dim, (
                f"feature {feature_name}: expected dim {embedding_dim}, "
                f"got {values.shape}"
            )
            assert values.shape[0] == batch_size, (
                f"feature {feature_name}: expected {batch_size} rows, "
                f"got {values.shape}"
            )

        loss = sum(jt.values().sum() for jt in ret.values())
        loss.backward()


def test_fake_dump_load_are_noops():
    fake = _construct()
    # Should not raise even though no real backend exists; emits a warning.
    with pytest.warns(UserWarning):
        fake.dump("/nonexistent/path/should/be/ignored")
    with pytest.warns(UserWarning):
        fake.load("/nonexistent/path/should/be/ignored")


def test_fake_storage_metadata_interface():
    """``_FakeStorage`` (accessed via ``module.tables``) exposes the public
    Storage metadata methods so external readers — compute kernel
    introspection, planner queries — don't trip on missing attributes."""
    fake = _construct()
    storage = fake.tables  # property on base class returns ``self._storage``

    assert storage.embedding_dtype() == torch.float32
    assert storage.max_embedding_dim() == 8
    # value_dim = emb_dim + optimizer state dim; SGD has 0 state.
    assert storage.max_value_dim() >= 8
    for t_idx in range(2):
        assert storage.embedding_dim(t_idx) == 8
        assert storage.value_dim(t_idx) >= 8
    assert storage.init_optimizer_state() == 0.0

    # GPU primitives must raise — fake mode short-circuits the forward
    # path so these should never actually be called; a clear error beats
    # silent garbage if a future code path reaches them.
    with pytest.raises(RuntimeError, match="fake mode"):
        storage.find()
    with pytest.raises(RuntimeError, match="fake mode"):
        storage.insert()


def test_fake_get_score_monotonic_for_timestamp():
    """get_score() with TIMESTAMP strategy must return increasing values
    even without the GPU device_timestamp() call."""
    from dynamicemb.dynamicemb_config import DynamicEmbScoreStrategy

    opts = _build_table_options(num_tables=1)
    opts[0].score_strategy = DynamicEmbScoreStrategy.TIMESTAMP
    fake = _construct(table_options=opts, table_names=["t_0"])

    s1 = fake.get_score()["t_0"]
    s2 = fake.get_score()["t_0"]
    assert s2 > s1
