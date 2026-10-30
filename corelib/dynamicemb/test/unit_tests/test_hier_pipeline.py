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
"""The hierarchical dist (``output_dist="fused"``) under the train pipelines.

Each loop and pipeline trains on the same batches as the plain loop, or as
TorchRec's dists, and must give the same outputs and tables, bit for bit
(with a power-of-two world size of at most 128, every sum is exact). Its
last steps also count the host syncs and the kernels of each call. Then
calls over the capacities under the pipelines, stress runs, shardings that
take TorchRec's dists, and the teardown with a pipeline in flight. On one
GPU the sharding takes TorchRec's dists. On a build without the op (NCCL
older than 2.30.7), "fused" raises and "auto" takes TorchRec's dists.

Run it from ``corelib/dynamicemb`` with ``examples`` on ``PYTHONPATH``, one
process per GPU (on one node, or across nodes under Slurm)::

    torchrun --nproc_per_node 8 test/unit_tests/test_hier_pipeline.py --megatron
    srun --ntasks-per-node 8 python test/unit_tests/test_hier_pipeline.py --megatron

``--core`` runs the cases of the TorchRec version matrix alone, and
``--megatron`` fails when the repo's JaggedMegatron pipelines cannot be
imported.
"""

import argparse
import contextlib
import gc
import os
import random
import sys
import tempfile
import time
import warnings
import weakref
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import dynamicemb_extensions
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    FrequencyAdmissionStrategy,
    KVCounter,
    get_table_value_bytes,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.key_ownership import owning_shard
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder, quiesce
from dynamicemb.shard.embedding import ShardedDynamicEmbeddingCollection
from dynamicemb.shard.hier_output_dist import HierSharding
from dynamicemb.shard.rw_sharding import (
    RwSequenceDynamicEmbeddingSharding,
    counts_frequencies,
)
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.train_pipeline import train_pipelines as pipelines
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable

# The build has the hierarchical dist (NCCL >= 2.30.7).
GIN = hasattr(dynamicemb_extensions, "HierOutputDist")
if GIN:
    from dynamicemb_extensions import HierOutputDist

# The capacities are counted as the benchmark counts them.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
)
from benchmark_utils import hier_capacities, hier_needs, init_distributed  # noqa: E402

DEV = torch.device("cuda")
NUM_EMBEDDINGS = 1 << 20
DIM = 32
# The EC's tables and their features (t0 serves two), and the EBC's.
TABLES = (("f0", "f1"), ("f2",), ("f3",))
# The status a first wait copies to the host: status_words(num_features)
# int64 words (src/hier_output_dist/common/status.h) of the EC's features.
STATUS_WORDS = 7 + 2 * sum(len(table) for table in TABLES)
EBC_TABLES = ("e0", "e1")
DENSE = 2048
# How the DynamicEmb tables place a key on its owner (their dist_type).
DIST_TYPE = "roundrobin"
# (local batch, longest sequence) of each step: calls of several sizes under
# one set of capacities, which fit them all (an owner looks up about 98K
# keys in the large ones).
SIZES = ((64, 4), (2048, 24), (2048, 24), (1024, 16), (2048, 24), (2048, 24))
# EvalPipelineSparseDist runs between two training epochs: the first EPOCH
# batches, then the eval batches, then the other batches.
EPOCH = 3
EVAL_SIZES = ((1024, 16), (2048, 24))
# The batches of the cases with calls over the capacities: the capacities
# fit the small ones alone, so a large one (about 24 times the keys) does
# not fit.
SMALL, LARGE = (256, 8), (2048, 24)
# last steps whose syncs and kernels are counted (TrainPipelineSemiSync runs
# its input dists furthest ahead: two steps before the end)
PROFILED = 3
SYNC_CALLS = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpy",
)
# How late the late rank of the teardown cases drops its model, in seconds.
LATE = 10.0
# The ranks of a node, and whether every table sum is exact (main sets them).
LOCAL = 1
EXACT = True


def this_gpu() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def all_true(ok: bool) -> bool:
    flag = torch.tensor([int(ok)], device=DEV)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def all_max(value: int) -> int:
    flag = torch.tensor([value], dtype=torch.int64, device=DEV)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item())


def report(name: str, checks: Dict[str, bool], log: Callable, extra: str = "") -> int:
    """Each check, true on every rank or not; logs them, and returns the
    failures (0 or 1)."""
    flags = torch.tensor(
        [int(bool(ok)) for ok in checks.values()], dtype=torch.int32, device=DEV
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    flags = flags.tolist()
    good = min(flags, default=1) == 1
    log(
        f"{name}: "
        + " ".join(f"{label}={flag}" for label, flag in zip(checks, flags))
        + extra
        + f" [{'ok' if good else 'FAIL'}]"
    )
    return int(not good)


@contextlib.contextmanager
def environ(**values):
    """Sets the environment variables given meanwhile (None leaves one as it
    is): the hierarchical dist reads its knobs when the model is sharded."""
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(
        {name: str(value) for name, value in values.items() if value is not None}
    )
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ── batches and models ───────────────────────────────────────────────────────


class Batch(Pipelineable):
    """One step's input: the EC's keys and the gradient of each of its output
    rows (in the order of the KJT's values, expanded to the full batch under
    VBE); with the EBC its keys and the gradient of its pooled rows; with
    the dense layer its input."""

    def __init__(
        self,
        features: KeyedJaggedTensor,
        grads: torch.Tensor,
        ebc_features: Optional[KeyedJaggedTensor] = None,
        ebc_grads: Optional[torch.Tensor] = None,
        dense: Optional[torch.Tensor] = None,
    ) -> None:
        self.features = features
        self.grads = grads
        self.ebc_features = ebc_features
        self.ebc_grads = ebc_grads
        self.dense = dense

    def _map(self, fn: Callable) -> "Batch":
        return Batch(
            *[
                None if tensor is None else fn(tensor)
                for tensor in (
                    self.features,
                    self.grads,
                    self.ebc_features,
                    self.ebc_grads,
                    self.dense,
                )
            ]
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return self._map(
            lambda tensor: tensor.to(device=device, non_blocking=non_blocking)
        )

    def record_stream(self, stream: torch.Stream) -> None:
        self._map(lambda tensor: tensor.record_stream(stream))

    def num_loss_tokens(self) -> torch.Tensor:
        """The repo's JaggedMegatron pipelines scale the loss by the data
        parallel size over the tokens of all ranks: 1 each, a scale of 1."""
        return torch.ones((), dtype=torch.int64)


@dataclass
class Spec:
    name: str
    dedup: bool = False  # index dedup
    lfu: bool = False  # LFU scores
    admit: bool = False  # FrequencyAdmissionStrategy, threshold 2
    # t0 LFU, the other tables admission alone (threshold 2): two lookup groups
    mixed: bool = False
    keys: str = "uniform"  # or "pow-law" (with hot keys)
    plain: bool = False  # a plain TorchRec table in the EC's row-wise sharding
    ebc: bool = False  # an EBC of TorchRec tables (row-wise, TorchRec's dists)
    dense: bool = False  # a dense layer, DDP all-reduces its gradients
    vbe: bool = False  # variable batch size per feature
    # the tables' capacity per rank at first (None: their max_capacity); they
    # rehash, doubling it, as they fill
    init_capacity: Optional[int] = None
    sizes: Tuple[Tuple[int, int], ...] = SIZES
    key_range: int = 200_000
    # the steps whose batches the capacities do not count: their calls do not
    # fit, and run on TorchRec's dists (with a second EC, the second's calls
    # alone)
    over: Tuple[int, ...] = ()
    # a second EC of DynamicEmb tables on the same features: another row-wise
    # sharding, with a hierarchical dist of its own beside the first one's
    second: bool = False

    @property
    def features(self) -> List[str]:
        return [feature for table in TABLES for feature in table] + (
            ["p0"] if self.plain else []
        )


def make_batch(
    spec: Spec, step: int, rank: int, batch_size: int, max_len: int
) -> Batch:
    """Step ``step``'s batch of rank ``rank``, on the host."""
    gen = torch.Generator().manual_seed(1_000_003 * step + 7919 * rank + 17)
    feats = spec.features
    num_features = len(feats)
    strides = [batch_size] * num_features
    if spec.vbe:  # each feature its own batch size, and its map to the batch
        strides = [
            max(1, batch_size - 13 * feature * (step % 3 + 1))
            for feature in range(num_features)
        ]
    lengths = [
        torch.randint(0, max_len + 1, (stride,), generator=gen) for stride in strides
    ]
    num_keys = int(sum(int(part.sum()) for part in lengths))
    if spec.keys == "pow-law":
        probs = 1.0 / torch.arange(1, spec.key_range + 1, dtype=torch.float64) ** 1.2
        values = torch.multinomial(probs, num_keys, replacement=True, generator=gen)
    else:
        values = torch.randint(0, spec.key_range, (num_keys,), generator=gen)
    lengths_cat = torch.cat(lengths)
    kwargs = {}
    rows = num_keys
    if spec.vbe:
        inverse = torch.stack(
            [
                torch.randint(0, stride, (batch_size,), generator=gen)
                for stride in strides
            ]
        )
        kwargs = dict(
            stride_per_key_per_rank=[[stride] for stride in strides],
            inverse_indices=(feats, inverse),
        )
        rows = int(sum(int(part[inv].sum()) for part, inv in zip(lengths, inverse)))
    features = KeyedJaggedTensor(
        keys=feats, values=values, lengths=lengths_cat, **kwargs
    )
    grads = torch.randint(-8, 9, (rows, DIM), generator=gen).float()
    batch = Batch(features, grads)
    if spec.ebc:
        ebc_len = torch.randint(0, 6, (len(EBC_TABLES) * batch_size,), generator=gen)
        batch.ebc_features = KeyedJaggedTensor(
            keys=list(EBC_TABLES),
            values=torch.randint(0, 50_000, (int(ebc_len.sum()),), generator=gen),
            lengths=ebc_len,
        )
        batch.ebc_grads = (
            torch.randint(-8, 9, (batch_size, DIM * len(EBC_TABLES)), generator=gen)
            / 8.0
        )
    if spec.dense:
        batch.dense = torch.randn(batch_size, DENSE, generator=gen)
    return batch


def batches_of(spec: Spec, rank: int) -> List[Batch]:
    return [
        make_batch(spec, i, rank, batch_size, max_len)
        for i, (batch_size, max_len) in enumerate(spec.sizes)
    ]


def eval_batches_of(spec: Spec, rank: int) -> List[Batch]:
    return [
        make_batch(spec, 100 + i, rank, batch_size, max_len)
        for i, (batch_size, max_len) in enumerate(EVAL_SIZES)
    ]


def ec_rows(out, keys: List[str]) -> torch.Tensor:
    """The EC's output rows, in the order of its input's keys (opaque to the
    train pipelines' fx tracing)."""
    return torch.cat([out[feature].values() for feature in keys])


torch.fx.wrap("ec_rows")


class Model(torch.nn.Module):
    """The EC's rows times their gradients (and the second EC's times their
    negatives, so that its tables leave the first's after a step), plus the
    EBC's pooled rows times theirs and the squares of the dense layer's
    outputs: ``forward(batch)`` returns the loss and the ECs' rows, as the
    train pipelines expect."""

    def __init__(self, ec, ebc=None, dense=None, ec2=None) -> None:
        super().__init__()
        self.ec = ec
        self.ebc = ebc
        self.dense = dense
        self.ec2 = ec2

    def forward(self, batch: Batch):
        rows = ec_rows(self.ec(batch.features), batch.features.keys())
        loss = (rows.float() * batch.grads).sum()
        if self.ec2 is not None:
            rows2 = ec_rows(self.ec2(batch.features), batch.features.keys())
            loss = loss - (rows2.float() * batch.grads).sum()
            rows = torch.cat([rows, rows2])
        if self.ebc is not None:
            pooled = self.ebc(batch.ebc_features)
            loss = loss + (pooled.values() * batch.ebc_grads).sum()
        if self.dense is not None:
            loss = loss + self.dense(batch.dense).square().sum()
        return loss.view(1), rows


def build(
    spec: Spec,
    output_dist: str,
    capacities: Sequence[Optional[Dict[str, int]]] = (None,),
) -> DistributedModelParallel:
    """The model of ``spec`` with its DynamicEmb tables on ``output_dist``,
    the first EC's with ``output_dist_capacity`` ``capacities[0]`` and the
    second EC's with ``capacities[-1]``."""
    world_size = dist.get_world_size()
    torch.manual_seed(5)  # the EBC's rows and the dense layer, alike in every model
    opt = EmbOptimType.EXACT_SGD

    def config(name: str, feats) -> torchrec.EmbeddingConfig:
        return torchrec.EmbeddingConfig(
            name=name,
            embedding_dim=DIM,
            num_embeddings=NUM_EMBEDDINGS,
            feature_names=list(feats),
            data_type=DataType.FP32,
        )

    def strategies(table_idx: int) -> Dict:
        extra = {}
        if spec.lfu or (spec.mixed and table_idx == 0):
            extra["score_strategy"] = DynamicEmbScoreStrategy.LFU
        if spec.admit or (spec.mixed and table_idx > 0):
            extra["admit_strategy"] = FrequencyAdmissionStrategy(
                threshold=2, counter=KVCounter(capacity=1 << 16)
            )
        return extra

    def dynamic(
        table_idx: int, table, capacity: Optional[Dict[str, int]]
    ) -> DynamicEmbParameterConstraints:
        return DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                global_hbm_for_values=get_table_value_bytes(table, opt, world_size),
                caching=False,
                init_capacity=spec.init_capacity,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                dist_type=DIST_TYPE,
                output_dist=output_dist,
                output_dist_capacity=capacity,
                **strategies(table_idx),
            ),
        )

    tables = [config(f"t{i}", feats) for i, feats in enumerate(TABLES)]
    plain = [config("p_p0", ("p0",))] if spec.plain else []
    ec = torchrec.EmbeddingCollection(
        tables=tables + plain, device=torch.device("meta")
    )
    constraints = {
        table.name: dynamic(table_idx, table, capacities[0])
        for table_idx, table in enumerate(tables)
    }
    ec2 = None
    if spec.second:
        tables2 = [config(f"s{i}", feats) for i, feats in enumerate(TABLES)]
        ec2 = torchrec.EmbeddingCollection(tables=tables2, device=torch.device("meta"))
        constraints.update(
            {
                table.name: dynamic(table_idx, table, capacities[-1])
                for table_idx, table in enumerate(tables2)
            }
        )
    plain_tables = [table.name for table in plain]
    ebc = None
    if spec.ebc:
        ebc_tables = [
            EmbeddingBagConfig(
                name=f"b_{feature}",
                embedding_dim=DIM,
                num_embeddings=50_000,
                feature_names=[feature],
                data_type=DataType.FP32,
            )
            for feature in EBC_TABLES
        ]
        ebc = torchrec.EmbeddingBagCollection(
            tables=ebc_tables, device=torch.device("meta")
        )
        plain_tables += [table.name for table in ebc_tables]
    for name in plain_tables:
        constraints[name] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=[EmbeddingComputeKernel.FUSED.value],
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=False,
        )
    dense = torch.nn.Linear(DENSE, DENSE, device=DEV) if spec.dense else None
    model = Model(ec, ebc, dense, ec2)
    topology = Topology(
        local_world_size=min(LOCAL, world_size),
        world_size=world_size,
        compute_device="cuda",
        hbm_cap=80 * 1024**3,
        ddr_cap=1024**4 // LOCAL,
    )
    planner = DynamicEmbeddingShardingPlanner(
        topology=topology, constraints=constraints, batch_size=SIZES[1][0]
    )
    fused = {"optimizer": opt, "learning_rate": 1.0}
    sharders = [
        DynamicEmbeddingCollectionSharder(
            fused_params={**fused, "output_dtype": SparseType.FP32},
            use_index_dedup=spec.dedup,
        ),
        EmbeddingBagCollectionSharder(fused_params=fused),
    ]
    plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=model, device=this_gpu(), sharders=sharders, plan=plan
    )


def optimizer_of(model: DistributedModelParallel):
    """SGD of the dense layer (the tables update in their backward), or the
    tables' fused optimizer when there is none."""
    dense = dict(in_backward_optimizer_filter(model.named_parameters()))
    if not dense:
        return model.fused_optimizer
    return KeyedOptimizerWrapper(dense, lambda params: torch.optim.SGD(params, lr=0.25))


def hiers_of(model) -> List[Optional[HierSharding]]:
    """The hierarchical dist of each EC's row-wise sharding (None: TorchRec's
    dists), the first EC's first."""
    return [
        sharding._hier
        for ec in model.modules()
        if isinstance(ec, ShardedDynamicEmbeddingCollection)
        for sharding in ec._sharding_type_to_sharding.values()
        if isinstance(sharding, RwSequenceDynamicEmbeddingSharding)
    ]


def hier_of(model) -> Optional[HierSharding]:
    """The first EC's row-wise sharding's hierarchical dist (None:
    TorchRec's dists)."""
    hiers = hiers_of(model)
    return hiers[0] if hiers else None


def dynamic_tables_of(model) -> List[BatchedDynamicEmbeddingTablesV2]:
    """The ECs' DynamicEmb lookup groups."""
    return [
        module._emb_module
        for ec in model.modules()
        if isinstance(ec, ShardedDynamicEmbeddingCollection)
        for lookup in ec._lookups
        for module in lookup._emb_modules
        if isinstance(
            getattr(module, "_emb_module", None), BatchedDynamicEmbeddingTablesV2
        )
    ]


def tables_of(model) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Every row of the EC's DynamicEmb tables, by key."""
    out = {}
    for emb in dynamic_tables_of(model):
        for name in emb._table_names:
            keys, values = emb.export_keys_values(name, DEV)
            order = torch.argsort(keys)
            out[name] = (keys[order], values[order])
    return out


def frequencies_of(model) -> Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """Per DynamicEmb table, by key: its LFU scores, and its admission
    counter's counts."""
    out = {}
    for emb in dynamic_tables_of(model):
        lfu = counts_frequencies(emb._dynamicemb_options[0])
        for table_id, name in enumerate(emb._table_names):
            parts = {}
            if emb._admitter is not None:
                counter = emb._admission_counter.table_
                parts["admit"] = [
                    (keys, scores["counter"])
                    for keys, scores, _ in counter._batched_export_keys_scores(
                        ["counter"], DEV, table_id
                    )
                ]
            if lfu:
                parts["lfu"] = [
                    (keys, scores)
                    for keys, _, _, scores in emb._storage.export_keys_values(
                        DEV, 65536, table_id
                    )
                ]
            for kind, got in parts.items():
                empty = torch.empty(0, device=DEV)
                keys = torch.cat([part for part, _ in got]) if got else empty
                values = torch.cat([scores for _, scores in got]) if got else empty
                order = torch.argsort(keys)
                out[(kind, name)] = (keys[order], values[order])
    return out


def table_capacities(model) -> List[int]:
    """The capacity of each of the EC's DynamicEmb lookup groups: a rehash
    grows it."""
    return [emb._storage._state.capacity for emb in dynamic_tables_of(model)]


# ── capacities ───────────────────────────────────────────────────────────────


def needs_of(spec: Spec, batches: List[Batch]) -> Dict[str, int]:
    """What the EC's calls on ``batches`` (this rank's) need of each
    capacity of its hierarchical dist, the most any rank needs in any of
    them (``hier_needs``, collective)."""
    world_size = dist.get_world_size()
    tables = {feature: table for table, feats in enumerate(TABLES) for feature in feats}
    return hier_needs(
        [batch.features for batch in batches],
        tables,
        lambda keys, feature: owning_shard(keys, world_size, DIST_TYPE),
        min(LOCAL, world_size),
        index_dedup=spec.dedup,
    )


def capacities_of(spec: Spec, rank: int) -> List[Dict[str, int]]:
    """The ``output_dist_capacity`` of each EC of ``spec``: what the batches
    its runs feed need, times 1.25 and at least 4096 more
    (``hier_capacities``); those of the last EC leave out the batches of
    ``spec.over``, so that their calls do not fit, while the first of two
    ECs fits them all. Collective: every run of a spec gets the same."""
    batches = batches_of(spec, rank)
    fit = [batch for step, batch in enumerate(batches) if step not in spec.over]
    needs = needs_of(spec, batches)
    capacities = [hier_capacities(needs_of(spec, fit) if spec.over else needs)]
    if spec.second:
        capacities.insert(0, hier_capacities(needs))
    return capacities


def fallbacks_of(spec: Spec, num_shardings: int) -> List[int]:
    """The calls of each sharding a run moves to TorchRec's dists (it feeds
    every batch): the last sharding's of ``spec.over``."""
    return [0] * (num_shardings - 1) + [len(spec.over)]


# ── instrumentation: ranges ──────────────────────────────────────────────────


def _in_range(owner, name: str, label: str) -> None:
    fn = getattr(owner, name)

    def wrapped(*args, **kwargs):
        with record_function(label):
            return fn(*args, **kwargs)

    setattr(owner, name, wrapped)


def instrument() -> None:
    """Ranges around the hierarchical dist's steps, in which the profiler
    counts host syncs and launches: ``hier::part1`` (the EC's
    ``input_dist``), ``hier::start`` and ``hier::wait`` (the first and the
    second wait of a call), ``hier::forward`` and ``hier::backward`` (the
    output dist's kernels)."""
    _in_range(ShardedDynamicEmbeddingCollection, "input_dist", "hier::part1")
    if not GIN:
        return
    _in_range(HierOutputDist, "start", "hier::start")
    _in_range(HierOutputDist, "wait", "hier::wait")
    _in_range(HierOutputDist, "forward", "hier::forward")
    _in_range(HierOutputDist, "backward", "hier::backward")


# Every hierarchical dist built, to check its teardown ran.
SHARDINGS: List[weakref.ref] = []


def track(model) -> None:
    for hier in hiers_of(model):
        if hier is not None:
            SHARDINGS.append(weakref.ref(hier))


def gc_models(log: Callable) -> int:
    """Drops the models on every rank at the same point, and checks every
    hierarchical dist went."""
    torch.cuda.synchronize()
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    good = all_true(all(ref() is None for ref in SHARDINGS))
    if not good:
        log("teardown: a hierarchical dist outlived its model [FAIL]")
    return int(not good)


def ranges_in(prof, label: str):
    """(thread, time range) of each range named ``label`` the profiler saw on
    the host."""
    return [
        (event.thread, event.time_range)
        for event in prof.events()
        if event.name == label and event.device_type == DeviceType.CPU
    ]


def inside(event, ranges) -> bool:
    """Whether host event ``event`` started inside one of ``ranges``, on its
    thread (a pipeline's loader thread copies batches meanwhile)."""
    return any(
        thread == event.thread and span.start <= event.time_range.start <= span.end
        for thread, span in ranges
    )


def syncs_in(prof, label: str) -> Tuple[int, int]:
    """Ranges named ``label`` the profiler saw on the host, and the host
    syncs inside them."""
    ranges = ranges_in(prof, label)
    syncs = sum(
        inside(event, ranges) for event in prof.events() if event.name in SYNC_CALLS
    )
    return len(ranges), syncs


def launched_in(prof, label: str) -> List[str]:
    """Names of the kernels and copies launched inside the ranges named
    ``label`` (through the CUDA call that launched each)."""
    ranges = ranges_in(prof, label)
    events = prof.events()
    launches = {
        event.id
        for event in events
        if event.device_type == DeviceType.CPU
        and event.name.startswith("cu")
        and inside(event, ranges)
    }
    return [
        event.name
        for event in events
        if event.device_type == DeviceType.CUDA and event.id in launches
    ]


def d2h_shapes(prof, label: str) -> List[List[int]]:
    """The shape of each device-to-host copy launched inside the ranges named
    ``label``: the first input of the aten::copy_ whose CUDA call launched it
    (profiled with record_shapes)."""
    ranges = ranges_in(prof, label)
    events = prof.events()
    d2h = {
        event.id
        for event in events
        if event.device_type == DeviceType.CUDA and "DtoH" in event.name
    }
    return [
        event.cpu_parent.input_shapes[0]
        for event in events
        if event.device_type == DeviceType.CPU
        and event.name.startswith("cu")
        and event.id in d2h
        and inside(event, ranges)
    ]


def sync_checks(prof, part1: bool = True) -> Tuple[Dict[str, bool], str]:
    """The hierarchical dist's host syncs and kernels in a profile of whole
    steps: no sync in part 1 (but where ``part1`` is False) and the first
    wait, one D2H copy in each first wait, the status's (into pinned memory,
    non-blocking), one sync in each second wait, also in a call that does not
    fit (TorchRec's input dist runs after it); and in the calls that take the
    op, no sync in the output dist and one kernel per direction."""
    ranges_part1, syncs_part1 = syncs_in(prof, "hier::part1")
    ranges_start, syncs_start = syncs_in(prof, "hier::start")
    ranges_wait, syncs_wait = syncs_in(prof, "hier::wait")
    ranges_fwd, syncs_fwd = syncs_in(prof, "hier::forward")
    ranges_bwd, syncs_bwd = syncs_in(prof, "hier::backward")
    d2h = d2h_shapes(prof, "hier::start")
    fwd = launched_in(prof, "hier::forward")
    bwd = launched_in(prof, "hier::backward")
    checks = {
        "input dist syncs": ranges_start > 0
        and ranges_wait > 0
        and (syncs_part1 == 0 or not part1)
        and syncs_start == 0
        and syncs_wait == ranges_wait,
        "one D2H of the status": d2h == [[STATUS_WORDS]] * ranges_start,
        "output dist syncs": ranges_fwd > 0
        and ranges_bwd > 0
        and syncs_fwd == syncs_bwd == 0,
        "one kernel each": len(fwd) == ranges_fwd
        and all("fwd_kernel" in name for name in fwd)
        and len(bwd) == ranges_bwd
        and all("bwd_kernel" in name for name in bwd),
    }
    text = (
        f" (part 1 x{ranges_part1} {syncs_part1} syncs, first wait"
        f" x{ranges_start} {syncs_start} syncs D2H {d2h}, second wait"
        f" x{ranges_wait} {syncs_wait} syncs; out forward x{ranges_fwd}"
        f" {syncs_fwd} syncs {len(fwd)} kernels, backward x{ranges_bwd}"
        f" {syncs_bwd} syncs {len(bwd)} kernels)"
    )
    return checks, text


def eval_checks(prof) -> Tuple[Dict[str, bool], str]:
    """An eval epoch's: it runs on TorchRec's dists, so no step of the
    hierarchical dist at all."""
    ran = {
        name: len(ranges_in(prof, f"hier::{name}"))
        for name in ("start", "wait", "forward", "backward")
    }
    return {"eval on TorchRec's dists": not any(ran.values())}, f"; eval steps {ran}"


# ── loops ────────────────────────────────────────────────────────────────────


def plain_step(model, opt, batch: Batch) -> torch.Tensor:
    """One forward, backward and optimizer step; the EC's rows."""
    opt.zero_grad()
    loss, rows = model(batch.to(DEV, non_blocking=True))
    torch.sum(loss, dim=0).backward()
    opt.step()
    return rows.detach()


def two_forwards_step(model, opt, first: Batch, second: Batch) -> List[torch.Tensor]:
    """Two forwards, then one backward of both."""
    opt.zero_grad()
    first_loss, first_rows = model(first.to(DEV, non_blocking=True))
    second_loss, second_rows = model(second.to(DEV, non_blocking=True))
    torch.sum(first_loss + second_loss, dim=0).backward()
    opt.step()
    return [first_rows.detach(), second_rows.detach()]


def eval_step(model, batch: Batch) -> torch.Tensor:
    with torch.no_grad():
        return model(batch.to(DEV, non_blocking=True))[1]


def megatron_pipelines():
    """The repo's JaggedMegatron pipelines (Megatron's parallel state set up
    once), or the reason they cannot be imported."""
    try:
        from commons.pipeline.train_pipeline import (
            JaggedMegatronPrefetchTrainPipelineSparseDist,
            JaggedMegatronTrainPipelineSparseDist,
        )
        from megatron.core import parallel_state
    except ImportError as err:
        return None, str(err)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1)
    return {
        "megatron": JaggedMegatronTrainPipelineSparseDist,
        "megatron_prefetch": JaggedMegatronPrefetchTrainPipelineSparseDist,
    }, None


def pipeline_of(kind: str, model, opt):
    device = this_gpu()
    if kind == "base":
        return pipelines.TrainPipelineBase(model, opt, device)
    if kind == "sparse_dist":
        return pipelines.TrainPipelineSparseDist(model, opt, device)
    if kind == "prefetch":
        return pipelines.PrefetchTrainPipelineSparseDist(model, opt, device)
    if kind.startswith("fused_"):
        return pipelines.TrainPipelineFusedSparseDist(
            model, opt, device, emb_lookup_stream=kind[len("fused_") :]
        )
    if kind == "semisync":
        return pipelines.TrainPipelineSemiSync(model, opt, device, start_batch=0)
    return megatron_pipelines()[0][kind](model, opt, device)


def rows_of(kind: str, out) -> torch.Tensor:
    """The EC's rows of a pipeline's output (the repo's pipelines return the
    loss and the tokens before it)."""
    return (out[2] if kind.startswith("megatron") else out).detach()


@dataclass
class Result:
    outs: List[torch.Tensor]
    tables: Dict
    freqs: Dict
    checks: Dict[str, bool] = field(default_factory=dict)
    text: str = ""


class Stress:
    """What a stress run adds to each step of a pipeline: a sleep kernel on the
    data dist's stream before each input dist, on the embedding lookup's
    stream before each lookup (TrainPipelineFusedSparseDist), or on the
    default stream before each step (``behind``, that many cycles: the host
    runs steps ahead of it, and :attr:`lead` counts the most); a random host
    sleep per rank; GEMMs on another stream."""

    def __init__(
        self,
        sleep_kernel=False,
        lookup_delay=False,
        behind=0,
        host_sleeps=False,
        gemms=False,
    ) -> None:
        self.sleep_kernel = sleep_kernel
        self.lookup_delay = lookup_delay
        self.behind = behind
        self.host_sleeps = host_sleeps
        self.gemms = gemms
        self.rng = random.Random(dist.get_rank())
        self.background = torch.cuda.Stream()
        self.mat = torch.randn(4096, 4096, device=DEV)
        self.prod = torch.empty_like(self.mat)
        # an event after each step's sleep, and the most steps whose sleep the
        # default stream had not got past when the next one began
        self.marks: List[torch.cuda.Event] = []
        self.lead = 0

    @staticmethod
    def _delay(pipe, method: str, stream: torch.cuda.Stream) -> None:
        """A sleep kernel on ``stream`` before each call of ``pipe.method``."""
        fn = getattr(pipe, method)

        def delayed(*args, **kwargs):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(20_000_000)
            return fn(*args, **kwargs)

        setattr(pipe, method, delayed)

    def attach(self, pipe) -> None:
        if self.sleep_kernel:
            self._delay(pipe, "start_sparse_data_dist", pipe._data_dist_stream)
        if self.lookup_delay:
            self._delay(pipe, "start_embedding_lookup", pipe._emb_lookup_stream)

    def step(self) -> None:
        if self.behind:
            self.lead = max(self.lead, sum(not mark.query() for mark in self.marks))
            torch.cuda._sleep(self.behind)
            self.marks.append(torch.cuda.Event())
            self.marks[-1].record()
        if self.host_sleeps:
            time.sleep(self.rng.uniform(0, 0.02))
        if self.gemms:
            self.background.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.background):
                for _ in range(4):
                    torch.mm(self.mat, self.mat, out=self.prod)


def run(
    spec: Spec,
    output_dist: str,
    kind: str,
    rank: int,
    stress: Optional[Stress] = None,
) -> Result:
    """The batches of ``spec`` through a new model on ``output_dist``, in a
    loop of ``kind``: "plain", "two_forwards", "eval_plain" (plain training
    epochs with plain eval steps between them), "eval_epochs" (the same under
    TrainPipelineSparseDist and EvalPipelineSparseDist), or a pipeline. On
    "torchrec" its DynamicEmb tables get no capacities, else those of
    ``spec`` (:func:`capacities_of`). Its outputs, tables and frequencies,
    and its checks: whether its tables rehashed if they start small; on the
    hierarchical dist the calls of each sharding that ran on TorchRec's
    dists, the host syncs and kernels of the last steps, and that the eval
    epoch ran none of the hierarchical dist's steps."""
    capacities: List[Optional[Dict[str, int]]] = [None]
    if output_dist != "torchrec":
        capacities = capacities_of(spec, rank)
    model = build(spec, output_dist, capacities)
    track(model)
    capacities_at_start = table_capacities(model)
    opt = optimizer_of(model)
    batches = batches_of(spec, rank)
    torch.cuda.synchronize()
    num_steps = len(batches)
    outs: List[torch.Tensor] = []
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    prof, eval_prof = profile(activities=activities, record_shapes=True), None
    profiling = False

    def at_step(i: int) -> None:
        nonlocal profiling
        if i == num_steps - PROFILED:
            prof.start()
            profiling = True
        if stress is not None:
            stress.step()

    if kind == "plain":
        for i, batch in enumerate(batches):
            at_step(i)
            outs.append(plain_step(model, opt, batch))
    elif kind == "two_forwards":
        for i in range(0, num_steps - 1, 2):
            at_step(i)
            at_step(i + 1)
            outs += two_forwards_step(model, opt, batches[i], batches[i + 1])
    elif kind in ("eval_plain", "eval_epochs"):
        evals = eval_batches_of(spec, rank)
        train = None
        if kind == "eval_epochs":
            train = pipeline_of("sparse_dist", model, opt)
        for epoch_idx, epoch in enumerate((batches[:EPOCH], batches[EPOCH:])):
            model.train()
            batch_iter = iter(epoch)
            for i, batch in enumerate(epoch, start=epoch_idx * EPOCH):
                at_step(i)
                if train is None:
                    outs.append(plain_step(model, opt, batch))
                else:
                    outs.append(rows_of(kind, train.progress(batch_iter)))
            if train is not None:
                train.detach()
            if epoch_idx == 1:
                break
            model.eval()
            eval_prof = profile(activities=activities)
            eval_prof.start()
            if train is None:
                outs += [eval_step(model, batch) for batch in evals]
            else:
                eval_pipe = pipelines.EvalPipelineSparseDist(model, opt, this_gpu())
                batch_iter = iter(evals)
                with torch.no_grad():
                    outs += [
                        rows_of(kind, eval_pipe.progress(batch_iter)) for _ in evals
                    ]
                eval_pipe.detach()
                del eval_pipe
            torch.cuda.synchronize()
            eval_prof.stop()
        model.train()
    else:
        pipe = pipeline_of(kind, model, opt)
        if stress is not None:
            stress.attach(pipe)
        batch_iter = iter(batches)
        for i in range(num_steps):
            at_step(i)
            outs.append(rows_of(kind, pipe.progress(batch_iter)))
        del pipe
    torch.cuda.synchronize()
    if profiling:
        prof.stop()
    result = Result(outs, tables_of(model), frequencies_of(model))
    if spec.init_capacity is not None:
        result.checks["rehashed"] = table_capacities(model) != capacities_at_start
    hiers = [
        hier for hier in hiers_of(model) if hier is not None and hier.takes_hier_dist
    ]
    if hiers:
        fallbacks = [hier.fallbacks for hier in hiers]
        expected = fallbacks_of(spec, len(hiers))
        result.checks["fallbacks"] = fallbacks == expected
        if any(fallbacks) or any(expected):
            result.text = f" (calls on TorchRec's dists, per sharding: {fallbacks})"
        if profiling:
            # VBE adds TorchRec's own syncs to part 1 (its sequence context)
            checks, text = sync_checks(prof, part1=not spec.vbe)
            result.checks.update(checks)
            result.text += text
        if eval_prof is not None:
            checks, text = eval_checks(eval_prof)
            result.checks.update(checks)
            result.text += text
    del model, opt
    return result


def same_outs(first: List[torch.Tensor], second: List[torch.Tensor]) -> bool:
    return len(first) == len(second) and all(
        left.shape == right.shape and torch.equal(left, right)
        for left, right in zip(first, second)
    )


def same_tables(got: Dict, want: Dict) -> Tuple[bool, bool]:
    """Every table row: equal (bit for bit where every sum is exact, else
    within half of the step one occurrence makes, 1/world_size), and whether
    bit for bit."""
    atol = 0.5 / dist.get_world_size()
    ok = bitwise = got.keys() == want.keys()
    for name in got:
        if name not in want:
            continue
        (got_keys, got_values), (want_keys, want_values) = got[name], want[name]
        keys = (
            torch.equal(got_keys, want_keys) and got_values.shape == want_values.shape
        )
        equal = keys and torch.equal(got_values, want_values)
        close = keys and torch.allclose(got_values, want_values, rtol=0, atol=atol)
        ok = ok and (equal if EXACT else close)
        bitwise = bitwise and equal
    return ok, bitwise


def compare(name: str, got: Result, ref: Result, log: Callable) -> int:
    """A run against a reference run on the same batches: its outputs bit for
    bit at every step, its tables (bit for bit where every sum is exact),
    frequencies, and its own checks (calls on TorchRec's dists, syncs,
    kernels)."""
    ok_tab, bitwise = same_tables(got.tables, ref.tables)
    checks = {"fwd": same_outs(got.outs, ref.outs), "tables": ok_tab}
    if got.freqs or ref.freqs:
        checks["freq"] = got.freqs.keys() == ref.freqs.keys() and all(
            torch.equal(got.freqs[key][0], ref.freqs[key][0])
            and torch.equal(got.freqs[key][1], ref.freqs[key][1])
            for key in got.freqs
        )
    checks.update(got.checks)
    extra = "" if EXACT else f" (tables bit for bit: {all_true(bitwise)})"
    return report(name, checks, log, extra + got.text)


def run_against(
    name: str,
    ref: Result,
    spec: Spec,
    kind: str,
    *,
    rank: int,
    log: Callable,
    output_dist: str = "fused",
    **kwargs,
) -> Tuple[int, Result]:
    """A run of ``spec`` in a loop of ``kind`` against ``ref``, then its
    model dropped: the failures, and the run. A group binds ``rank`` and
    ``log`` once (``against``)."""
    got = run(spec, output_dist, kind, rank, **kwargs)
    failed = compare(f"{spec.name}: {name}", got, ref, log)
    return failed + gc_models(log), got


# ── the matrix ───────────────────────────────────────────────────────────────

UNIFORM = Spec("uniform")
# index dedup with LFU and admission: the keys carry frequency counters
COUNTED = Spec(
    "pow-law, index dedup, LFU and admission",
    dedup=True,
    lfu=True,
    admit=True,
    keys="pow-law",
)
# LFU without index dedup: across nodes the owner's KJT carries each
# node-unique key's occurrences
LFU = Spec("pow-law, LFU", lfu=True, keys="pow-law")
# and only the LFU table's lookup may see them
MIXED = Spec(
    "pow-law, an LFU table beside admission tables", mixed=True, keys="pow-law"
)
VBE = Spec("variable batch per feature", vbe=True)
EVAL = Spec("eval on TorchRec's dists")
# tables of 16384 rows per rank at first: the batches past the first fill
# them, and each rehashes a few times on the way to its max_capacity. An
# expansion only makes room for the keys it counts (DynamicEmb's rule), so
# the first must not be too far below them: keys it evicted would make the
# tables differ from tables that never rehash.
REHASH = Spec("tables that rehash as they fill", init_capacity=16384)
# small batches with two large ones the capacities do not fit, one of them
# in the profiled steps
OVER = Spec(
    "calls over the capacities",
    sizes=(SMALL, LARGE, SMALL, SMALL, LARGE, SMALL),
    over=(1, 4),
)


def fused_kinds() -> List[str]:
    """TrainPipelineFusedSparseDist's lookup streams, from TorchRec 1.3 on."""
    if hasattr(pipelines, "TrainPipelineFusedSparseDist"):
        return ["fused_data_dist", "fused_new"]
    return []


def run_matrix(rank: int, log: Callable, core: bool, need_megatron: bool) -> int:
    """Each loop and pipeline against its reference (see the module's
    docstring); with ``core`` the TorchRec version matrix's subset."""
    failures = 0
    megatron, why = megatron_pipelines()
    if megatron is None:
        log(
            f"the repo's JaggedMegatron pipelines cannot be imported ({why}): "
            + ("[FAIL]" if need_megatron else "skipped")
        )
        failures += int(need_megatron)
    megatron_kinds = list(megatron or [])
    megatron_prefetch = [kind for kind in megatron_kinds if kind.endswith("prefetch")]
    against = partial(run_against, rank=rank, log=log)

    torchrec_run = run(UNIFORM, "torchrec", "plain", rank)
    failed, plain = against(
        "plain loop vs TorchRec's dists", torchrec_run, UNIFORM, "plain"
    )
    failures += failed
    for kind in ["base", "sparse_dist", "prefetch", *fused_kinds(), *megatron_kinds]:
        failures += against(f"{kind} vs the plain loop", plain, UNIFORM, kind)[0]
    failures += against(
        "sparse_dist vs TorchRec's dists", torchrec_run, UNIFORM, "sparse_dist"
    )[0]
    if not core:
        failures += against(
            "prefetch vs the plain loop without rehashes", plain, REHASH, "prefetch"
        )[0]
        failures += against(
            "prefetch on TorchRec's dists vs the plain loop without rehashes",
            torchrec_run,
            REHASH,
            "prefetch",
            output_dist="torchrec",
        )[0]
    del torchrec_run, plain

    ref = run(EVAL, "fused", "eval_plain", rank)
    failures += against(
        "EvalPipelineSparseDist between training epochs", ref, EVAL, "eval_epochs"
    )[0]
    torchrec_run = run(COUNTED, "torchrec", "plain", rank)
    failed, plain = against(
        "plain loop vs TorchRec's dists", torchrec_run, COUNTED, "plain"
    )
    failures += failed
    failures += against(
        "sparse_dist vs TorchRec's dists", torchrec_run, COUNTED, "sparse_dist"
    )[0]
    failures += against(
        "prefetch on TorchRec's dists vs their plain loop",
        torchrec_run,
        COUNTED,
        "prefetch",
        output_dist="torchrec",
    )[0]
    for kind in ["prefetch", *megatron_prefetch]:
        failures += against(f"{kind} vs the plain loop", plain, COUNTED, kind)[0]
    del torchrec_run, plain
    failures += run_overflows(rank, log, megatron_kinds)
    failures += run_fallbacks(rank, log, core)
    if core:
        return failures

    for spec in (LFU, MIXED):
        torchrec_run = run(spec, "torchrec", "plain", rank)
        failed, plain = against(
            "plain loop vs TorchRec's dists", torchrec_run, spec, "plain"
        )
        failures += failed
        for kind in ["prefetch", *(megatron_prefetch if spec is MIXED else [])]:
            failures += against(f"{kind} vs the plain loop", plain, spec, kind)[0]
        del torchrec_run, plain
    torchrec_run = run(UNIFORM, "torchrec", "two_forwards", rank)
    for spec in (UNIFORM, REHASH):
        failures += against(
            "two forwards, one backward vs TorchRec's dists without rehashes",
            torchrec_run,
            spec,
            "two_forwards",
        )[0]
    torchrec_run = run(VBE, "torchrec", "plain", rank)
    failures += against("plain loop vs TorchRec's dists", torchrec_run, VBE, "plain")[0]
    semisync = run(UNIFORM, "fused", "semisync", rank)
    failures += against(
        "semisync vs itself without rehashes", semisync, REHASH, "semisync"
    )[0]
    del torchrec_run, semisync
    failures += gc_models(log)
    failures += run_stress(rank, log)
    return failures


def run_overflows(rank: int, log: Callable, megatron_kinds: List[str]) -> int:
    """Calls over the capacities under the pipelines. OVER's capacities fit
    its small batches alone, so that its two large calls run on TorchRec's
    dists, on every rank alike, and the sharding counts them. The plain loop
    against TorchRec's dists; each pipeline against the plain loop, but
    TrainPipelineSemiSync, whose outputs are not the plain loop's, against
    itself with capacities that fit every call. Then two row-wise shardings,
    the first with capacities that fit every call and the second with
    OVER's: only the second's calls fall back."""
    against = partial(run_against, rank=rank, log=log)
    failures = 0
    torchrec_run = run(OVER, "torchrec", "plain", rank)
    failed, plain = against(
        "plain loop vs TorchRec's dists", torchrec_run, OVER, "plain"
    )
    failures += failed
    for kind in ["base", "sparse_dist", "prefetch", *fused_kinds(), *megatron_kinds]:
        failures += against(f"{kind} vs the plain loop", plain, OVER, kind)[0]
    del torchrec_run, plain
    semisync = run(replace(OVER, over=()), "fused", "semisync", rank)
    failures += against(
        "semisync vs itself with capacities that fit every call",
        semisync,
        OVER,
        "semisync",
    )[0]
    del semisync
    spec = replace(
        OVER, name="calls over the second EC's capacities alone", second=True
    )
    torchrec_run = run(spec, "torchrec", "plain", rank)
    failed, plain = against(
        "plain loop vs TorchRec's dists", torchrec_run, spec, "plain"
    )
    failures += failed
    for kind in ["sparse_dist", "prefetch", *megatron_kinds]:
        failures += against(f"{kind} vs the plain loop", plain, spec, kind)[0]
    return failures


def run_stress(rank: int, log: Callable) -> int:
    """Pipelines under stress against the plain loop."""
    failures = 0
    spec = Spec("EC beside an EBC on TorchRec's dists", ebc=True)
    plain = run(spec, "fused", "plain", rank)
    got = run(spec, "fused", "sparse_dist", rank, Stress(sleep_kernel=True))
    failures += compare(
        f"{spec.name}: sparse_dist, the data dist's stream delayed by a sleep"
        " kernel, vs the plain loop",
        got,
        plain,
        log,
    )
    failures += gc_models(log)
    # The host runs steps ahead of the default stream (no sync waits for it):
    # the blocks a backward's prefetched state frees must not reach the next
    # prefetch before that backward has run.
    plain = run(UNIFORM, "fused", "plain", rank)
    stress = Stress(behind=200_000_000)
    got = run(UNIFORM, "fused", "prefetch", rank, stress)
    # each step the host runs ahead keeps that step's blocks reserved
    got.checks["host lead"] = all_max(stress.lead) <= 2
    failures += compare(
        f"{UNIFORM.name}: prefetch, the default stream behind the host by a sleep"
        " kernel per step, vs the plain loop",
        got,
        plain,
        log,
    )
    failures += gc_models(log)
    # The lookup and the output dist on a stream of their own: the dense
    # forward reads the rows on the default stream, which must wait for the
    # output dist's kernel; and the default stream held back.
    if "fused_new" in fused_kinds():
        for what, stress in (
            ("the lookup's stream delayed", Stress(lookup_delay=True)),
            ("the default stream behind the host", Stress(behind=20_000_000)),
        ):
            got = run(UNIFORM, "fused", "fused_new", rank, stress)
            failures += compare(
                f"{UNIFORM.name}: fused_new, {what} by a sleep kernel per step, vs"
                " the plain loop",
                got,
                plain,
                log,
            )
            failures += gc_models(log)
    spec = Spec("dense layer under DDP", dense=True)
    plain = run(spec, "fused", "plain", rank)
    with environ(DYNAMICEMB_GIN_CTAS=4):
        got = run(
            spec,
            "fused",
            "sparse_dist",
            rank,
            Stress(host_sleeps=True, gemms=True),
        )
    failures += compare(
        f"{spec.name}: sparse_dist with random host sleeps, GEMMs on another stream"
        " and 4 CTAs for the out kernels, vs the plain loop",
        got,
        plain,
        log,
    )
    failures += gc_models(log)
    # Two row-wise shardings: two hierarchical dists, whose input dists share
    # the SMs they may hold while they wait for their peers (a CTA each), the
    # next batch's beside this one's out kernels.
    spec = Spec("two ECs on the same features", second=True)
    torchrec_run = run(spec, "torchrec", "plain", rank)
    plain = run(spec, "fused", "plain", rank)
    failures += compare(
        f"{spec.name}: plain loop vs TorchRec's dists", plain, torchrec_run, log
    )
    failures += gc_models(log)
    for kind in ("sparse_dist", "prefetch"):
        got = run(spec, "fused", kind, rank)
        failures += compare(f"{spec.name}: {kind} vs the plain loop", got, plain, log)
        failures += gc_models(log)
    return failures


# ── fallbacks ────────────────────────────────────────────────────────────────


def warned(fn: Callable, text: str):
    """fn(), and whether it warned with ``text``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, any(text in str(warning.message) for warning in caught)


def run_fallbacks(rank: int, log: Callable, core: bool) -> int:
    """Shardings that take TorchRec's dists, equal to TorchRec's: a plain
    TorchRec table in the row-wise sharding (under TrainPipelineSparseDist);
    and, but in the version matrix, feature processors (the decision) and
    ``"auto"`` on this build, which takes the op with a capacity and
    TorchRec's dists, with a warning, without one."""
    failures = 0
    spec = Spec("a plain table in the row-wise sharding", plain=True)
    torchrec_run = run(spec, "torchrec", "plain", rank)
    got, fell = warned(
        lambda: run(spec, "fused", "sparse_dist", rank), "does not apply here"
    )
    got.checks["fell back"] = fell
    failures += compare(
        f"{spec.name}: sparse_dist vs TorchRec's", got, torchrec_run, log
    )
    failures += gc_models(log)
    if core:
        return failures

    # feature processors, and "auto" on this build
    capacities = capacities_of(UNIFORM, rank)
    init_distributor = RwSequenceDynamicEmbeddingSharding._init_customized_distributor

    def with_processor(self, infos):
        self._has_feature_processor = True
        init_distributor(self, infos)

    RwSequenceDynamicEmbeddingSharding._init_customized_distributor = with_processor
    try:
        model, fell = warned(
            lambda: build(UNIFORM, "fused", capacities), "has feature processors"
        )
    finally:
        RwSequenceDynamicEmbeddingSharding._init_customized_distributor = (
            init_distributor
        )
    checks = {"feature processors fall back": fell and hier_of(model) is None}
    del model
    model, auto_warned = warned(
        lambda: build(UNIFORM, "auto", capacities), 'output_dist="auto"'
    )
    track(model)
    plain_step(model, optimizer_of(model), make_batch(UNIFORM, 1, rank, *SIZES[1]))
    hier = hier_of(model)
    checks["auto takes the op"] = (
        hier is not None and hier.takes_hier_dist is True and not auto_warned
    )
    quiesce(model)
    del model, hier
    model, auto_warned = warned(lambda: build(UNIFORM, "auto"), 'output_dist="auto"')
    checks["auto without a capacity takes TorchRec's dists"] = (
        hier_of(model) is None and auto_warned
    )
    failures += report("feature processors; auto on this build", checks, log)
    del model
    return failures + gc_models(log)


def run_one_rank(rank: int, log: Callable) -> int:
    """A job of one rank (world size 1): ``"fused"`` falls back to
    TorchRec's dists, with a warning, and trains exactly as they do."""
    spec = replace(UNIFORM, name="one rank (world size 1)", sizes=SIZES[:3])
    torchrec_run = run(spec, "torchrec", "plain", rank)
    got, fell = warned(
        lambda: run(spec, "fused", "plain", rank), "the sharding has one rank"
    )
    got.checks["fell back"] = fell
    failures = compare(f"{spec.name}: plain loop vs TorchRec's", got, torchrec_run, log)
    return failures + gc_models(log)


def run_nonccl(rank: int, log: Callable) -> int:
    """A build without NCCL >= 2.30.7: "fused" raises ImportError when the
    model is sharded; "auto" with a capacity warns, once per model, and
    trains on TorchRec's dists (plain loop and TrainPipelineSparseDist)
    exactly as "torchrec" does."""
    checks = {}
    capacities = capacities_of(UNIFORM, rank)
    try:
        build(UNIFORM, "fused", capacities)
        checks["fused raised"] = False
    except ImportError as err:
        checks["fused raised"] = "NCCL >= 2.30.7" in str(err)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        auto = run(UNIFORM, "auto", "plain", rank)
        piped = run(UNIFORM, "auto", "sparse_dist", rank)
    warns = sum('output_dist="auto"' in str(warning.message) for warning in caught)
    checks["auto warned once per model"] = warns == 2
    torchrec_run = run(UNIFORM, "torchrec", "plain", rank)
    for name, got in (("auto, plain loop", auto), ("auto, sparse_dist", piped)):
        checks[f"{name}: fwd"] = same_outs(got.outs, torchrec_run.outs)
        checks[f"{name}: tables"] = same_tables(got.tables, torchrec_run.tables)[0]
    return report("a build without the hierarchical dist", checks, log) + gc_models(log)


# ── teardown ─────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def fd2_into(out: List[str]):
    """Collects what the process writes to file descriptor 2 meanwhile (the
    C++ teardown's warnings)."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)
            out.append(tmp.read().decode(errors="replace"))


def run_teardown(rank: int, world_size: int, log: Callable) -> int:
    """Every rank drops its model with TrainPipelineSparseDist in flight (the
    next batch's input dist done, its second wait not). After quiesce() the
    last rank drops it LATE seconds late and nobody waits for it; a call
    after quiesce() undoes it, and the late rank's node and rail peers wait
    for it in the exit round; without quiesce() the exit rounds free
    everything. Never a warning, and no hierarchical dist is left. A rank
    waited if its drop took more than half of LATE."""
    failures = 0
    last = rank == world_size - 1
    peer = not last and (
        rank // LOCAL == (world_size - 1) // LOCAL or rank % LOCAL == LOCAL - 1
    )
    spec = replace(UNIFORM, sizes=SIZES[:2] + SIZES[3:4] * 4)
    capacities = capacities_of(spec, rank)
    for name, quiesced, again, late in (
        (f"after quiesce(), a rank {LATE:.0f} s late", True, False, LATE),
        (f"a call after quiesce(), a rank {LATE:.0f} s late", True, True, LATE),
        ("without quiesce()", False, False, 0.0),
    ):
        model = build(spec, "fused", capacities)
        track(model)
        pipe = pipelines.TrainPipelineSparseDist(model, optimizer_of(model), this_gpu())
        batch_iter = iter(batches_of(spec, rank))
        for _ in range(3):
            pipe.progress(batch_iter)
        if quiesced:
            quiesce(model)
        if again:
            pipe.progress(batch_iter)
        torch.cuda.synchronize()
        dist.barrier()
        if last:
            time.sleep(late)
        err: List[str] = []
        with fd2_into(err):
            start = time.monotonic()
            del pipe, model, batch_iter
            gc.collect()
            took = time.monotonic() - start
        waited = took > LATE / 2
        checks = {
            "no warning": "[hier_output_dist]" not in err[0],
            "freed": all(ref() is None for ref in SHARDINGS),
            "waits": last or not late or waited == (again and peer),
        }
        took_max = torch.tensor([0.0 if last else took], device=DEV)
        dist.all_reduce(took_max, op=dist.ReduceOp.MAX)
        failures += report(
            f"teardown with the pipeline in flight, {name}",
            checks,
            log,
            f" (the other ranks dropped their models in {took_max.item():.1f} s at"
            " most)",
        )
    return failures


def main() -> None:
    global LOCAL, EXACT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core", action="store_true", help="the TorchRec version matrix's cases"
    )
    parser.add_argument(
        "--megatron",
        action="store_true",
        help="fail when the repo's JaggedMegatron pipelines cannot be imported",
    )
    args = parser.parse_args()
    # Spins that trap after two minutes instead of the process group's timeout.
    os.environ.setdefault("DYNAMICEMB_GIN_TIMEOUT_S", "120")
    rank, world_size, _, LOCAL = init_distributed()
    EXACT = (world_size & (world_size - 1)) == 0 and world_size <= 128

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    log(
        f"torchrec {torchrec.__version__}, {world_size} ranks on"
        f" {world_size // LOCAL} node(s), a build"
        f" {'with' if GIN else 'without'} the hierarchical dist"
    )
    instrument()
    if not GIN:
        failures = run_nonccl(rank, log)
    elif world_size == 1:
        failures = run_one_rank(rank, log)
    else:
        failures = run_matrix(rank, log, args.core, args.megatron)
        if not args.core:
            failures += run_teardown(rank, world_size, log)
    dist.barrier()
    log("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    dist.destroy_process_group()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
