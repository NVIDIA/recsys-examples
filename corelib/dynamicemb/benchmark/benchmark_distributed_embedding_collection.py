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
"""Multi-GPU benchmark for a row-wise sharded DynamicEmb EmbeddingCollection.

Every table lives entirely in HBM (no caching, no host memory). Each
iteration feeds a fresh batch whose sequence lengths and keys are drawn at
random, so the amount of data every rank sends and receives changes from one
iteration to the next.

Launch with torchrun or srun, one process per GPU::

    torchrun --nproc_per_node 8 benchmark/benchmark_distributed_embedding_collection.py
    srun --ntasks-per-node 8 python benchmark/benchmark_distributed_embedding_collection.py

Options and metrics: benchmark/README.md, section 3.
"""

import argparse
import contextlib
import gc
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torchrec
from benchmark_utils import (
    IB_PEAK_GBPS,
    NVLINK_PEAK_GBPS,
    hier_capacities,
    hier_needs,
    init_distributed,
)
from dataset_generator import PowerLaw
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
    EmbOptimType,
    get_table_value_bytes,
)
from dynamicemb.key_ownership import owning_shard
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder, quiesce
from dynamicemb.shard.embedding import ShardedDynamicEmbeddingCollection
from dynamicemb.shard.hier_output_dist import RwSequenceHierEmbeddingDist
from fbgemm_gpu.split_embedding_configs import SparseType
from torch.autograd import DeviceType
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding import EmbeddingCollectionAwaitable
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListAwaitable,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingDist
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingType
from torchrec.modules.embedding_configs import DataType
from torchrec.streamable import Pipelineable

_OPTIMIZERS = {
    "sgd": EmbOptimType.EXACT_SGD,
    "adam": EmbOptimType.ADAM,
    "exact_adagrad": EmbOptimType.EXACT_ADAGRAD,
    "rowwise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}
_OUTPUT_DTYPES = {
    "fp32": SparseType.FP32,
    "bf16": SparseType.BF16,
    "fp16": SparseType.FP16,
}
_TABLE_DTYPES = {"fp32": DataType.FP32, "bf16": DataType.BF16, "fp16": DataType.FP16}
_PIPELINES = {
    "sparse_dist": TrainPipelineSparseDist,
    "prefetch": PrefetchTrainPipelineSparseDist,
}


@dataclass
class BenchConfig:
    num_tables: int = 8
    features_per_table: int = 1
    num_embeddings: int = 4 * 1024 * 1024
    embedding_dim: int = 128
    local_batch_size: int = 1024
    max_seq_len: int = 32
    seq_len_dist: str = "uniform"
    key_dist: str = "uniform"
    alpha: float = 1.05
    key_range: Optional[int] = None
    dist_type: str = "roundrobin"
    emb_dtype: str = "fp32"
    output_dtype: str = "fp32"
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    use_index_dedup: bool = False
    output_dist: str = "torchrec"
    capacity: str = "measure"  # an int, or measure: from the batches
    capacity_headroom: float = 1.25
    # the output_dist_capacity the models are built with (from capacity)
    output_dist_capacity: Optional[Union[int, Dict[str, int]]] = None
    num_iterations: int = 30
    warmup: int = 10
    profile_iterations: int = 5
    pipeline: str = "none"
    check: bool = False
    trace: bool = False
    trace_out: Optional[str] = None
    kernel_times: bool = False
    host_stages: bool = False
    nic_counters: bool = False
    seed: int = 1234
    json_out: Optional[str] = None

    def label(self) -> str:
        return (
            f"T{self.num_tables}_D{self.embedding_dim}_B{self.local_batch_size}"
            f"_L{self.max_seq_len}{self.seq_len_dist[0]}_{self.key_dist}"
            f"_{self.emb_dtype}_dedup{int(self.use_index_dedup)}_{self.output_dist}"
            + (f"_F{self.features_per_table}" if self.features_per_table > 1 else "")
            + (f"_{self.pipeline}" if self.pipeline != "none" else "")
        )


# ── input generation ─────────────────────────────────────────────────────────


def table_name(i: int) -> str:
    return f"t_{i}"


def feature_name(i: int) -> str:
    return f"f_{i}"


def generate_batches(
    cfg: BenchConfig, rank: int, device: torch.device, num_batches: int
) -> List[torchrec.KeyedJaggedTensor]:
    """One KJT per iteration: ``num_tables * features_per_table`` features x
    ``local_batch_size`` samples, each sample holding a sequence of keys.

    ``seq_len_dist`` decides the sequence lengths: ``fixed`` (all equal to
    ``max_seq_len``) or ``uniform`` (each drawn from ``[1, max_seq_len]``), so
    with ``uniform`` every rank sends a different, changing number of keys.
    """
    torch.manual_seed(cfg.seed * 7919 + rank)
    key_range = cfg.key_range or cfg.num_embeddings
    num_features = cfg.num_tables * cfg.features_per_table
    batch_size = cfg.local_batch_size
    batches = []
    for _ in range(num_batches):
        if cfg.seq_len_dist == "fixed":
            lengths = torch.full(
                (num_features * batch_size,), cfg.max_seq_len, device=device
            )
        else:  # uniform
            lengths = torch.randint(
                1, cfg.max_seq_len + 1, (num_features * batch_size,), device=device
            )
        lengths = lengths.to(torch.int64)
        total = int(lengths.sum().item())
        if cfg.key_dist == "uniform":
            keys = torch.randint(0, key_range, (total,), device=device)
        else:  # pow-law
            keys = PowerLaw(1, key_range, cfg.alpha, total, device)
        batches.append(
            torchrec.KeyedJaggedTensor(
                keys=[feature_name(i) for i in range(num_features)],
                values=keys.to(torch.int64),
                lengths=lengths,
            )
        )
    return batches


# ── model ───────────────────────────────────────────────────────────────────


class Batch(Pipelineable):
    """One iteration's input, as TorchRec's train pipelines move it."""

    def __init__(self, features: torchrec.KeyedJaggedTensor) -> None:
        self.features = features

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(self.features.to(device=device, non_blocking=non_blocking))

    def record_stream(self, stream: torch.Stream) -> None:
        self.features.record_stream(stream)


class EmbeddingModel(torch.nn.Module):
    """The EC and the loss it trains on: ``forward(batch)`` returns both the
    loss and the embeddings, as TorchRec's train pipelines expect."""

    def __init__(
        self, ec: torchrec.EmbeddingCollection, weight: Optional[torch.Tensor]
    ) -> None:
        super().__init__()
        self.ec = ec
        self.weight = weight

    def forward(self, batch: Batch):
        out = self.ec(batch.features)
        return embedding_loss(out, self.weight), out


def build_model(
    cfg: BenchConfig,
    device: torch.device,
    world_size: int,
    local_world_size: int,
    output_dist: str,
    init_mode: DynamicEmbInitializerMode = DynamicEmbInitializerMode.NORMAL,
    weight: Optional[torch.Tensor] = None,
) -> DistributedModelParallel:
    """The EC of ``cfg`` on ``output_dist``, in an :class:`EmbeddingModel`
    whose loss weighs the embeddings' columns by ``weight`` (None: 1)."""
    tables = [
        torchrec.EmbeddingConfig(
            name=table_name(i),
            embedding_dim=cfg.embedding_dim,
            num_embeddings=cfg.num_embeddings,
            feature_names=[
                feature_name(i * cfg.features_per_table + j)
                for j in range(cfg.features_per_table)
            ],
            data_type=_TABLE_DTYPES[cfg.emb_dtype],
        )
        for i in range(cfg.num_tables)
    ]
    ec = torchrec.EmbeddingCollection(tables=tables, device=torch.device("meta"))

    opt = _OPTIMIZERS[cfg.optimizer]
    constraints = {}
    for table in tables:
        constraints[table.name] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            enforce_hbm=True,
            bounds_check_mode=BoundsCheckMode.NONE,
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                # HBM only: the whole table, values and optimizer states, in HBM.
                global_hbm_for_values=get_table_value_bytes(table, opt, world_size),
                caching=False,
                dist_type=cfg.dist_type,
                initializer_args=DynamicEmbInitializerArgs(mode=init_mode),
                output_dist=output_dist,
                output_dist_capacity=cfg.output_dist_capacity,
            ),
        )
    topology = Topology(
        local_world_size=local_world_size,
        world_size=world_size,
        compute_device="cuda",
        hbm_cap=80 * 1024**3,
        ddr_cap=1024**4 // local_world_size,
        intra_host_bw=NVLINK_PEAK_GBPS * 1e9,
        inter_host_bw=IB_PEAK_GBPS * 1e9,
    )
    planner = DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
        batch_size=cfg.local_batch_size,
    )
    fused_params = {
        "optimizer": opt,
        "learning_rate": cfg.learning_rate,
        "output_dtype": _OUTPUT_DTYPES[cfg.output_dtype],
    }
    sharder = DynamicEmbeddingCollectionSharder(
        fused_params=fused_params,
        use_index_dedup=cfg.use_index_dedup,
    )
    model = EmbeddingModel(ec, weight)
    plan = planner.collective_plan(model, [sharder], dist.GroupMember.WORLD)
    return DistributedModelParallel(
        module=model, device=device, sharders=[sharder], plan=plan
    )


def resolve_capacity(
    cfg: BenchConfig,
    batches: List[torchrec.KeyedJaggedTensor],
    local_world_size: int,
) -> BenchConfig:
    """``cfg`` with the output_dist_capacity its models are built with.

    An int ``--capacity`` is used as it is. Otherwise each capacity is what
    the hierarchical dist needs for ``batches`` on its busiest rank
    (``hier_needs``, collective, outside the timed iterations), times
    ``cfg.capacity_headroom``, and at least 4096 entries over the need."""
    if cfg.output_dist == "torchrec":
        return cfg
    if cfg.capacity != "measure":
        return replace(cfg, output_dist_capacity=int(cfg.capacity))
    world_size = dist.get_world_size()
    num_features = cfg.num_tables * cfg.features_per_table
    tables = {
        feature_name(feature): feature // cfg.features_per_table
        for feature in range(num_features)
    }
    # The counting runs on a process group of its own, made and destroyed
    # here (collective, in the same order on every rank): NCCL keeps buffers
    # for a group's all-to-alls until the group goes, and the models use none
    # of the default group's.
    group = dist.new_group(list(range(world_size)))
    needs = hier_needs(
        batches,
        tables,
        lambda keys, feature: owning_shard(keys, world_size, cfg.dist_type),
        local_world_size,
        index_dedup=cfg.use_index_dedup,
        group=group,
    )
    dist.destroy_process_group(group)
    return replace(
        cfg, output_dist_capacity=hier_capacities(needs, cfg.capacity_headroom)
    )


def _hier_shardings(model) -> List:
    """The model's hierarchical dists, one HierSharding per sharding."""
    return [
        output_dist._hier
        for module in model.modules()
        for output_dist in getattr(module, "_output_dists", []) or []
        if isinstance(output_dist, RwSequenceHierEmbeddingDist)
    ]


def hier_fallbacks(model) -> int:
    """Calls of the model's hierarchical dists that did not fit their
    capacities and ran on TorchRec's dists, so far."""
    return sum(hier.fallbacks for hier in _hier_shardings(model))


def _hier_op(model):
    """The model's HierOutputDist (the first sharding's), or None."""
    return next((hier.op for hier in _hier_shardings(model)), None)


def embedding_loss(out, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sum of the looked-up embeddings in ``out`` (optionally weighted per
    column). Opaque to the train pipelines' fx tracing."""
    values = [jagged.values() for jagged in out.values()]
    if weight is None:
        return sum(rows.sum() for rows in values)
    return sum((rows.float() * weight).sum() for rows in values)


torch.fx.wrap("embedding_loss")


def train_pipeline(cfg: BenchConfig, model: DistributedModelParallel):
    """The train pipeline ``--pipeline`` names, for ``model`` (its optimizer:
    the tables' fused one, which updates them in the backward)."""
    return _PIPELINES[cfg.pipeline](
        model, model.fused_optimizer, torch.device("cuda", torch.cuda.current_device())
    )


# ── stage attribution for the profiler ───────────────────────────────────────

_STAGE_PREFIX = "bench::"
# Set while the profiled iterations run: the stage ranges are recorded then
# only, so the timed iterations pay nothing for them.
_PROFILING = False
_NO_RANGE = contextlib.nullcontext()


def _range(name: str):
    """The profiler range ``name`` while profiling, else nothing."""
    return record_function(name) if _PROFILING else _NO_RANGE


# Device syncs a call's host may wait in (--host_stages): (owner, method,
# whether only a CUDA tensor's syncs).
_SYNCS = (
    (torch.Tensor, "item", True),
    (torch.Tensor, "cpu", True),
    (torch.Tensor, "tolist", True),
    (torch.Tensor, "__bool__", True),
    (torch.Tensor, "__int__", True),
    (torch.Tensor, "__float__", True),
    (torch.cuda.Stream, "synchronize", False),
    (torch.cuda.Event, "synchronize", False),
    (torch.cuda, "synchronize", False),
)


def _plan_syncs() -> Tuple:
    """The hierarchical dist's one sync per call, in C++: its operator's
    second wait (where the build has it)."""
    import dynamicemb_extensions

    op = getattr(dynamicemb_extensions, "HierOutputDist", None)
    return () if op is None else ((op, "wait", False),)


_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep


def _site() -> str:
    """Where the caller of a wrapped sync is, outside torch: its file:line and
    its own caller's."""
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith(_TORCH_DIR):
        frame = frame.f_back
    if frame is None:
        return "?"
    at = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    caller = frame.f_back
    return (
        at
        if caller is None
        else f"{at} <- {os.path.basename(caller.f_code.co_filename)}:{caller.f_lineno}"
    )


class _HostSyncs:
    """--host_stages: per call of the plain loop's timed iterations, the host
    ms the call waited in device syncs, by call site (:meth:`install` wraps
    the syncs for the timed iterations)."""

    def __init__(self, num_calls: int) -> None:
        self.syncs: List[Dict[str, float]] = [
            defaultdict(float) for _ in range(num_calls)
        ]
        self.call = -1  # the call that runs
        self._saved: List[Tuple[object, str, object]] = []

    def _sync(self, orig, cuda_only: bool):
        def wrapped(*args, **kwargs):
            if self.call < 0 or (cuda_only and not args[0].is_cuda):
                return orig(*args, **kwargs)
            start = time.perf_counter()
            out = orig(*args, **kwargs)
            self.syncs[self.call][_site()] += (time.perf_counter() - start) * 1000.0
            return out

        return wrapped

    def install(self) -> None:
        for owner, name, cuda_only in _SYNCS + _plan_syncs():
            self._saved.append((owner, name, owner.__dict__.get(name)))
            setattr(owner, name, self._sync(getattr(owner, name), cuda_only))

    def uninstall(self) -> None:
        for owner, name, orig in reversed(self._saved):
            if orig is None:  # inherited
                delattr(owner, name)
            else:
                setattr(owner, name, orig)
        self._saved = []
        self.call = -1


class _Calls:
    """What the plain loop keeps of each call of its timed iterations: CUDA
    events on the caller's stream before the input dist, after part 1, after
    the first wait and after the second wait (when the lookup may start: the
    stream then waits for everything the KJT needs), and right before and
    after the hierarchical output dist's forward and backward launches, made
    beforehand; when tracing, this GPU's globaltimer before the input dist
    and after its second wait ([2] int64 each). With --host_stages it tells
    the sync counter which call runs."""

    def __init__(
        self, num_calls: int, clocks: bool, host: Optional[_HostSyncs]
    ) -> None:
        self.events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for _ in range(num_calls)
        ]
        self.output_dist_events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            for _ in range(num_calls)
        ]
        self.output_dist_timed = [[False, False] for _ in range(num_calls)]
        self.count = 0  # calls so far
        self.clocks: Optional[List[torch.Tensor]] = [] if clocks else None
        self.host = host

    def begin(self) -> None:
        """A call starts, before its input dist."""
        if self.host is not None:
            self.host.call = self.count
        self.events[self.count][0].record()
        if self.clocks is not None:
            from dynamicemb_extensions import hier_output_dist_clock

            self.clocks.append(torch.zeros(2, dtype=torch.int64, device="cuda"))
            hier_output_dist_clock(self.clocks[-1], 0)
        self.count += 1

    def step(self, which: int) -> None:
        """Step ``which`` of the call's input dist (1: part 1, 2: the first
        wait, 3: the second wait) has returned."""
        self.events[self.count - 1][which].record()
        if which == 3 and self.clocks is not None:
            from dynamicemb_extensions import hier_output_dist_clock

            hier_output_dist_clock(self.clocks[-1], 1)

    def output_dist_launch(self, direction: int, fn, *args, **kwargs):
        """The call's output-dist launch ``direction`` (0: forward, 1:
        backward) between its two events."""
        call = self.count - 1
        events = self.output_dist_events[call]
        events[2 * direction].record()
        out = fn(*args, **kwargs)
        events[2 * direction + 1].record()
        self.output_dist_timed[call][direction] = True
        return out

    def op_calls(self) -> List[int]:
        """The calls that took the hierarchical dist (they timed both of its
        output-dist launches), or every call when none did (TorchRec's
        dists): a call over its capacities runs on TorchRec's dists, and
        the per-call figures leave it out."""
        took = [call for call in range(self.count) if all(self.output_dist_timed[call])]
        return took or list(range(self.count))

    def output_dist_ms(self) -> List[List[float]]:
        """Per call that took the hierarchical dist, the GPU ms of its output
        dist's forward and backward launches; empty when none did."""
        return [
            [events[0].elapsed_time(events[1]), events[2].elapsed_time(events[3])]
            for call, events in enumerate(self.output_dist_events[: self.count])
            if all(self.output_dist_timed[call])
        ]


# The plain loop's timed iterations' calls (None otherwise).
_CALLS: Optional[_Calls] = None
# Launches of the hierarchical output dist's kernels outside the timed
# iterations, by direction (0: forward, 1: backward): --check reads them to
# tell that the checked model ran its own dist.
_OUTPUT_DIST_COUNTS: Dict[int, int] = defaultdict(int)


def _time_output_dist_launches() -> None:
    """Wrap the hierarchical output dist's forward and backward launches (one
    kernel each), where the build has them, in the per-call events of the
    timed iterations (_Calls.output_dist_launch)."""
    import dynamicemb_extensions as ext

    if not hasattr(ext, "HierOutputDist"):
        return
    for method, direction in (("forward", 0), ("backward", 1)):
        orig = getattr(ext.HierOutputDist, method)

        def launch(self, *args, _orig=orig, _direction=direction, **kwargs):
            if _CALLS is None:
                _OUTPUT_DIST_COUNTS[_direction] += 1
                return _orig(self, *args, **kwargs)
            return _CALLS.output_dist_launch(_direction, _orig, self, *args, **kwargs)

        setattr(ext.HierOutputDist, method, launch)


def install_stage_markers() -> None:
    """Wrap the sharded module's pieces in record_function ranges (while
    profiling), so the profiler can tell which stage launched each kernel,
    and in the per-call instrumentation of the timed iterations (_CALLS).

    Forward kernels are attributed through these ranges. Backward kernels run
    on the autograd thread and are attributed through the autograd node that
    launched them (see ``_BACKWARD_STAGES``).
    """
    _time_output_dist_launches()

    def forward(self, *input, **kwargs):
        ctx = self.create_context()
        calls = _CALLS
        if calls is not None:
            calls.begin()
        with _range(_STAGE_PREFIX + "input_dist"):
            dist_input = self.input_dist(ctx, *input, **kwargs)
            if calls is not None:
                calls.step(1)
            dist_input = dist_input.wait()
            if calls is not None:
                calls.step(2)
            dist_input = dist_input.wait()
            if calls is not None:
                calls.step(3)
        return self.compute_and_output_dist(ctx, dist_input)

    ShardedDynamicEmbeddingCollection.forward = forward

    # Under a train pipeline the input dist's part 1 and its waits run from
    # the pipeline, outside the module's forward.
    def in_stage(fn):
        def wrapped(*args, **kwargs):
            with _range(_STAGE_PREFIX + "input_dist"):
                return fn(*args, **kwargs)

        return wrapped

    ShardedDynamicEmbeddingCollection.input_dist = in_stage(
        ShardedDynamicEmbeddingCollection.input_dist
    )
    for awaitable in (
        KJTListSplitsAwaitable,
        FusedKJTListSplitsAwaitable,
        KJTListAwaitable,
    ):
        awaitable._wait_impl = in_stage(awaitable._wait_impl)

    orig_compute_and_output_dist = (
        ShardedDynamicEmbeddingCollection.compute_and_output_dist
    )

    def compute_and_output_dist(self, ctx, input):
        for lookup in self._lookups:
            _wrap_once(lookup, "lookup")
        for output_dist in self._output_dists:
            _wrap_once(output_dist, "output_dist")
        return orig_compute_and_output_dist(self, ctx, input)

    ShardedDynamicEmbeddingCollection.compute_and_output_dist = compute_and_output_dist

    orig_wait = EmbeddingCollectionAwaitable._wait_impl

    def _wait_impl(self):
        with _range(_STAGE_PREFIX + "output_dist_wait"):
            return orig_wait(self)

    EmbeddingCollectionAwaitable._wait_impl = _wait_impl


def _wrap_once(module: torch.nn.Module, stage: str) -> None:
    if getattr(module, "_bench_wrapped", False):
        return
    module._bench_wrapped = True
    orig = module.forward

    def forward(*args, **kwargs):
        with _range(_STAGE_PREFIX + stage):
            return orig(*args, **kwargs)

    module.forward = forward


# Autograd node -> stage, for kernels launched during backward.
_BACKWARD_STAGES = [
    (re.compile(r"DynamicEmbeddingFunction"), "lookup"),
    (re.compile(r"All2All_Seq"), "output_dist"),
    (re.compile(r"_OutputFunction"), "output_dist"),
    (re.compile(r"IndexSelectBackward|GatherBackward"), "output_dist"),
]
_FORWARD_STAGE_RE = re.compile(re.escape(_STAGE_PREFIX) + r"(\w+)")
# Kernels of the output_dist stage (matched by name) that move embedding rows
# between GPUs: NCCL's (TorchRec's all-to-all) or the hierarchical dist's out
# kernels.
_NET_KERNEL_RE = re.compile(
    r"nccl|hier_output_dist::fwd_kernel|hier_output_dist::bwd_kernel",
    re.IGNORECASE,
)
# Per-stage values kernel_breakdown() accumulates, in the order run() gathers
# them over ranks and report() reads them back.
_STAGE_FIELDS = ("ms", "kernels", "net_ms")


# The steps of a row of the in channel's step times (op.in_trace(), on the
# host; empty without DYNAMICEMB_HIER_TRACE=1): InTraceSteps in
# src/hier_output_dist/common/trace_slots.h, each the ns from the call's
# start to the step's end, -1 for a step the call does not run. Its phases:
# each step, from the end of the step before it that ran, and the span, from
# the call's start to its last step's end.
_IN_TRACE = (
    "counts",
    "layout",
    "push",
    "done",
    "kjt",
    "dedup",
    "push_order",
    "message",
    "rail_counts",
    "rail_layout",
    "rail_keys",
    "status_exchange",
    "gathers",
    "requester",
    "owner",
    "put_list",
    "status",
)
_IN_PHASES = _IN_TRACE + ("span",)


def in_phases(model, calls: int) -> Optional[List[float]]:
    """In a build with the trace, the in channel's phases (_IN_PHASES) of
    this rank's last ``calls`` calls on the hierarchical path, us, the mean
    over the calls that ran each (nan for none)."""
    op = _hier_op(model)
    if op is None or not op.in_trace().numel():
        return None
    trace, done = op.in_trace(), op.in_calls()
    rows = [
        trace[call % trace.size(0)].tolist()
        for call in range(done - min(calls, trace.size(0), done) + 1, done + 1)
    ]
    spans: List[List[float]] = [[] for _ in _IN_PHASES]
    for row in rows:
        before = 0
        for step, ns in enumerate(row):
            if ns >= 0:
                spans[step].append((ns - before) / 1e3)
                before = ns
        if row[-1] >= 0:
            spans[-1].append(row[-1] / 1e3)
    return [sum(span) / len(span) if span else float("nan") for span in spans]


def _stage_of(event) -> Tuple[str, str]:
    """(phase, stage) of a CPU event (the call that launched a kernel): the
    stage range or autograd node it runs in."""
    while event is not None:
        name = event.name
        if name.startswith("autograd::engine::evaluate_function: "):
            node = name.split(": ", 1)[1]
            for pattern, stage in _BACKWARD_STAGES:
                if pattern.search(node):
                    return "bwd", stage
            return "bwd", "other"
        match = _FORWARD_STAGE_RE.match(name)
        if match:
            stage = match.group(1)
            return "fwd", "output_dist" if stage == "output_dist_wait" else stage
        event = event.cpu_parent
    return "fwd", "other"


def _kernels(prof):
    """(phase/stage, name, GPU ms) of every kernel and memory operation the
    profiler saw, each attributed through the CUDA runtime or driver call that
    launched it (the same correlation id). The profiler links a kernel to a
    PyTorch op only when one was running at its launch, so this also counts
    kernels a C++ extension launches directly (DynamicEmb's bucketize, the
    hierarchical input dist's first wait). Device-side copies of annotation
    ranges are left out: they have no launch (and their ids are not CUDA
    correlation ids)."""
    events = prof.events()
    launches = {
        event.id: event
        for event in events
        if event.device_type == DeviceType.CPU and event.name.startswith("cu")
    }
    annotations = {
        event.name for event in events if getattr(event, "is_user_annotation", False)
    }
    for event in events:
        if event.device_type != DeviceType.CUDA or event.name in annotations:
            continue
        launch = launches.get(event.id)
        if launch is not None:
            phase, stage = _stage_of(launch)
            ms = (event.time_range.end - event.time_range.start) / 1e3
            yield f"{phase}/{stage}", event.name, ms


def kernel_breakdown(prof, iterations: int) -> Dict[str, Dict[str, float]]:
    """Per (phase/stage): GPU kernel ms and kernel count, per iteration."""
    acc = defaultdict(lambda: dict.fromkeys(_STAGE_FIELDS, 0.0))
    for key, name, ms in _kernels(prof):
        acc[key]["ms"] += ms
        acc[key]["kernels"] += 1
        if key.endswith("/output_dist") and _NET_KERNEL_RE.search(name):
            acc[key]["net_ms"] += ms
    return {
        stage: {field: value / iterations for field, value in values.items()}
        for stage, values in sorted(acc.items())
    }


def kernel_names(
    prof, stage_filter: str, iterations: int
) -> Dict[str, Tuple[float, float]]:
    """Kernel name -> (count, GPU ms) per iteration for one ``phase/stage`` key."""
    names = defaultdict(lambda: [0, 0.0])
    for key, name, ms in _kernels(prof):
        if key == stage_filter:
            names[name][0] += 1
            names[name][1] += ms
    return {
        name: (count / iterations, ms / iterations)
        for name, (count, ms) in names.items()
    }


# ── traffic accounting ───────────────────────────────────────────────────────


class TorchRecTraffic:
    """Rows TorchRec's output dist sends, captured from the sharding context.

    In the forward each rank sends back one row per key it received, i.e.
    ``output_splits`` of the input dist (already on the host), and receives
    one per key it sent (``input_splits``): ``rows_ib`` and ``rows_ib_in``
    are those of other nodes. A call costs the host one append: the splits
    are added up afterwards.
    """

    def __init__(self, rank: int, local_world_size: int):
        self.node = rank // local_world_size
        self.rank = rank
        self.local_size = local_world_size
        self.reset()
        orig = RwSequenceEmbeddingDist.forward
        traffic = self

        def forward(self_, local_embs, sharding_ctx=None):
            if traffic.enabled and sharding_ctx is not None:
                traffic.calls.append(
                    (sharding_ctx.output_splits, sharding_ctx.input_splits)
                )
            return orig(self_, local_embs, sharding_ctx)

        RwSequenceEmbeddingDist.forward = forward

    def reset(self):
        self.enabled = False
        self.calls: List[Tuple[List[int], Optional[List[int]]]] = []

    def per_iteration(self, iterations: int) -> Dict[str, float]:
        rows_self = rows_nvlink = rows_ib = rows_ib_in = 0
        # per rail: the forward's rows to (out) and from (in) the GPUs of that
        # rail on other nodes (with NCCL's PXN a node's sends to a rail leave
        # through its NIC of that rail)
        rail_out = [0] * self.local_size
        rail_in = [0] * self.local_size
        for splits, in_splits in self.calls:
            for peer, rows in enumerate(splits):
                if peer == self.rank:
                    rows_self += rows
                elif peer // self.local_size == self.node:
                    rows_nvlink += rows
                else:
                    rows_ib += rows
                    rail_out[peer % self.local_size] += rows
            for peer, rows in enumerate(in_splits or []):
                if peer // self.local_size != self.node:
                    rows_ib_in += rows
                    rail_in[peer % self.local_size] += rows
        out = {
            "rows_nvlink": rows_nvlink / iterations,
            "rows_ib": rows_ib / iterations,
            "rows_ib_in": rows_ib_in / iterations,
            "rows_self": rows_self / iterations,
        }
        for rail in range(self.local_size):
            out[f"rail_out_{rail}"] = rail_out[rail] / iterations
            out[f"rail_in_{rail}"] = rail_in[rail] / iterations
        return out


class HierTraffic:
    """Rows the hierarchical output dist moves, from each call's plan (its
    block offsets and, across nodes, its node-unique counts, read once the
    timed iterations are over, and the rows the rank looked up). In the
    forward every rank pulls its rows from the owners of its node and from
    its node's proxies, over NVLink but for its own lookup's and its own
    proxy rows' (rows_nvlink). Other GPUs ask rows of it (rows_served; its
    max over ranks is the hottest owner's): its node peers pull theirs over
    NVLink, and other nodes' rows leave over IB (rows_ib_out). Across nodes
    each owner puts one row per node-unique key another node asked of it
    over IB (rows_ib_out), each proxy receives them (rows_ib_in) for the
    rows its node asked of other nodes' owners on its rail (rows_remote: the
    IB rows over these are the node dedup's ratio), in the rounds its proxy
    region takes them (rounds, per call); the backward moves the same rows
    the other way. A call costs the host one append: its tables are read
    afterwards.
    """

    def __init__(self, rank: int, local_world_size: int):
        self.local_rank = rank % local_world_size
        self.node = rank // local_world_size
        self.local_size = local_world_size
        self.hier = None
        self.reset()
        orig = RwSequenceHierEmbeddingDist.forward
        traffic = self

        def forward(self_, local_embs, sharding_ctx=None):
            plan = getattr(sharding_ctx, "plan", None)
            if traffic.enabled and plan is not None:
                from dynamicemb_extensions import hier_output_dist_plan_tensors

                traffic.hier = self_._hier
                tensors = hier_output_dist_plan_tensors(plan)
                traffic.calls.append(
                    (tensors["block_start"], plan.num_recv, tensors["meta"])
                )
            return orig(self_, local_embs, sharding_ctx)

        RwSequenceHierEmbeddingDist.forward = forward

    def reset(self):
        self.enabled = False
        self.calls: List[Tuple] = []

    def _ib(self, call) -> Optional[Tuple[torch.Tensor, ...]]:
        """A call's node-unique rows put to (uniq_in) and received from
        (uniq_out) each node, and its MiscSlots (kMiscRounds, kMiscChunks,
        kMiscRemoteKeys); None on one node."""
        from dynamicemb_extensions import hier_output_dist_op_meta_offsets

        hier = self.hier
        if not hier.multi_node:
            return None
        meta, at = call[2], hier_output_dist_op_meta_offsets(hier.op)
        num_nodes = hier._num_nodes
        return (
            meta[at["uniq_in"] :][:num_nodes],
            meta[at["uniq_out"] :][:num_nodes],
            meta[at["misc"] :][:3],
        )

    def per_call_ib(self) -> List[Tuple[int, int, int, int]]:
        """Per call (in order): the rows this rank puts over IB as owner
        in the forward (its node-unique keys of other nodes) and receives
        as proxy (its node's of other nodes' owners), 0 on one node; the
        keys it looked up as owner; and those of them it asked itself."""
        me = self.node * self.local_size + self.local_rank
        out = []
        for call in self.calls:
            ib = self._ib(call)
            own = int(call[0][me + 1] - call[0][me])
            uniq = (0, 0) if ib is None else (int(ib[0].sum()), int(ib[1].sum()))
            out.append((*uniq, int(call[1]), own))
        return out

    def per_iteration(self, iterations: int) -> Dict[str, float]:
        local_size, me, node = self.local_size, self.local_rank, self.node
        rows = (
            torch.stack([call[0].diff() for call in self.calls])
            .sum(0)
            .view(-1, local_size)
        )
        mine = int(rows[node, me])
        # pulled over NVLink: the node's other owners' rows, and the remote
        # rows of the node's other proxies
        remote = rows.sum(0) - rows[node]
        nvlink = int(rows[node].sum()) - mine + int(remote.sum()) - int(remote[me])
        looked_up = sum(call[1] for call in self.calls)
        out = {
            "rows_self": mine / iterations,
            "rows_nvlink": nvlink / iterations,
            "rows_served": (looked_up - mine) / iterations,
            "rows_ib": 0.0,
        }
        ibs = [self._ib(call) for call in self.calls]
        if ibs[0] is not None:
            out.update(
                rows_ib_out=sum(int(ib[0].sum()) for ib in ibs) / iterations,
                rows_ib_in=sum(int(ib[1].sum()) for ib in ibs) / iterations,
                rows_remote=sum(int(ib[2][2]) for ib in ibs) / iterations,
                rounds=sum(int(ib[2][0]) for ib in ibs) / len(ibs),
            )
            # the busier direction of this rank's NIC
            out["rows_ib"] = max(out["rows_ib_out"], out["rows_ib_in"])
        return out


# The hierarchical dist's trace rows (TraceSlots in
# src/hier_output_dist/common/trace_slots.h): globaltimer stamps (the first
# _HIER_TRACE_EARLIEST keep the earliest of their CTAs', unset at UINT64_MAX;
# the rest the latest, unset at 0), then counters, then three timelines of
# _HIER_TRACE_BINS bins (NVLink bytes, rows landed from other nodes, rows
# posted to them).
_HIER_TRACE = {
    "start": 0,
    "first_put": 1,
    "first_landed": 2,
    "enter_first": 3,
    "nv_first": 4,
    "entered": 5,
    "last_put": 6,
    "puts_done": 7,
    "landed": 8,
    "nv_last": 9,
    "arrived": 10,
    "exited": 11,
    "start_last": 12,
    "nv_bytes": 13,
    "local_bytes": 14,
    "ib_in": 15,
    "ib_out": 16,
}
_HIER_TRACE_EARLIEST, _HIER_TRACE_TIMES = 5, 13
_HIER_TRACE_SLOTS, _HIER_TRACE_BINS = 17, 512
_HIER_TRACE_WORDS = _HIER_TRACE_SLOTS + 3 * _HIER_TRACE_BINS
# The width of the timelines' bins, us: the op's.
_HIER_TRACE_BIN_US = 10.0


# One traced kernel of one rank, as trace_records() lays it out: its stamps
# 1 .. _HIER_TRACE_TIMES - 1 in us after its start (nan: unset), its
# counters, its start and its first entry in rank 0's clock (ns after the
# first calibration), the IB bytes its plan says it sends and receives, then
# the timelines.
_HIER_TRACE_FIELDS = (
    [
        name
        for name, slot in sorted(_HIER_TRACE.items(), key=lambda item: item[1])
        if 0 < slot < _HIER_TRACE_TIMES
    ]
    + ["nv_bytes", "local_bytes", "ib_in", "ib_out"]
    + ["start_abs", "enter_abs", "bytes_out", "bytes_in"]
    # plain loop: the call's input dist start and lookup start, in us after
    # the forward kernel's start (negative, the rank's own clock); the keys
    # its owner looked up, and of those the ones it asked itself (the rest
    # node peers pull through its NVLink port, on one node)
    + ["in_start", "lookup_start", "num_recv", "own"]
)
_HIER_TRACE_BINS0 = len(_HIER_TRACE_FIELDS)


def clock_offsets(device: torch.device, rounds: int = 4) -> torch.Tensor:
    """[world_size, 3] int64 on the host: per rank, the offset of its
    %globaltimer from rank 0's (ns), from ping-pongs of stamps with rank 0 (the
    round of the shortest round trip of ``rounds``), that round trip, and rank
    0's clock at its middle. Every GPU counts its own globaltimer (origins
    seconds apart, rates tens of ppm apart), so stamps of two ranks compare
    only through these; an offset is good to about half the round trip's
    asymmetry."""
    from dynamicemb_extensions import hier_output_dist_clock

    world, rank = dist.get_world_size(), dist.get_rank()
    out = torch.zeros(world, 3, dtype=torch.int64, device=device)
    buf = torch.zeros(4, dtype=torch.int64, device=device)
    torch.cuda.synchronize()
    dist.barrier()
    for peer in range(1, world):
        best = None
        for _ in range(rounds):
            if rank == 0:
                hier_output_dist_clock(buf, 0)
                dist.send(buf[0:1], dst=peer)
                dist.recv(buf[1:2], src=peer)
                hier_output_dist_clock(buf, 2)
                start, remote, end, _ = buf.tolist()
                round_trip = end - start
                if best is None or round_trip < best[1]:
                    best = (remote - (start + end) // 2, round_trip, (start + end) // 2)
            elif rank == peer:
                dist.recv(buf[0:1], src=0)
                hier_output_dist_clock(buf, 1)
                dist.send(buf[1:2], dst=0)
                torch.cuda.synchronize()
        if rank == 0:
            out[peer] = torch.tensor(best, dtype=torch.int64, device=device)
    if rank == 0:
        hier_output_dist_clock(buf, 0)
        out[0, 2] = buf[0]
    dist.broadcast(out, src=0)
    return out.cpu()


def trace_records(
    model,
    rank: int,
    calls: int,
    cal: Tuple[torch.Tensor, torch.Tensor],
    ib_calls: List[Tuple[int, int, int, int]],
    row_bytes: int,
    clocks: Optional[List[torch.Tensor]] = None,
) -> Optional[torch.Tensor]:
    """This rank's last ``calls`` traced kernels of each direction, laid out
    as _HIER_TRACE_FIELDS then the timelines: [2, calls, fields] float64. Its
    stamps go to rank 0's clock through the calibrations before and after them
    (``cal``, from clock_offsets), the offset interpolated in time."""
    op = _hier_op(model)
    if op is None or not op.trace().numel():
        return None
    trace, launches = op.trace().cpu(), list(op.trace_launches())
    ring_rows = trace.size(1)
    assert trace.size(2) == _HIER_TRACE_WORDS, "trace rows do not match TraceSlots"
    count = min(calls, ring_rows, *launches)
    (cal_before, cal_after) = cal
    offset_before, offset_after = int(cal_before[rank, 0]), int(cal_after[rank, 0])
    time_before, time_after = int(cal_before[rank, 2]), int(cal_after[rank, 2])
    ref = int(cal_before[0, 2])

    def to_rank0(stamp: int) -> float:
        # rank 0's clock (ns after the first calibration) of this rank's stamp
        shifted = stamp - offset_before
        offset = (
            offset_before
            if time_after == time_before
            else offset_before
            + (offset_after - offset_before)
            * (shifted - time_before)
            / (time_after - time_before)
        )
        return float(stamp - offset - ref)

    out = torch.full(
        (2, count, _HIER_TRACE_BINS0 + 3 * _HIER_TRACE_BINS),
        float("nan"),
        dtype=torch.float64,
    )
    ib = ib_calls[-count:] if ib_calls else [(0, 0, 0, 0)] * count
    clk = (
        [clock.tolist() for clock in clocks[-count:]]
        if clocks and len(clocks) >= count
        else None
    )
    fwd0 = [0] * count  # the forward kernels' starts
    for direction in range(2):
        for i in range(count):
            row = trace[
                direction, (launches[direction] - count + i) % ring_rows
            ].tolist()
            start = row[_HIER_TRACE["start"]]
            rec = []
            for field in _HIER_TRACE_FIELDS[: _HIER_TRACE_TIMES - 1]:
                stamp = row[_HIER_TRACE[field]]
                unset = (
                    stamp == -1
                    if _HIER_TRACE[field] < _HIER_TRACE_EARLIEST
                    else stamp == 0
                )
                rec.append(float("nan") if unset else (stamp - start) / 1e3)
            rec += [
                float(row[_HIER_TRACE[field]])
                for field in ("nv_bytes", "local_bytes", "ib_in", "ib_out")
            ]
            enter_first = row[_HIER_TRACE["enter_first"]]
            rec += [
                to_rank0(start),
                to_rank0(enter_first) if enter_first != -1 else float("nan"),
            ]
            # forward: owner puts, proxy receives
            rows_put, rows_received, num_recv, own = ib[i]
            ib_bytes = (rows_put * row_bytes, rows_received * row_bytes)
            rec += list(ib_bytes if direction == 0 else ib_bytes[::-1])
            if direction == 0:
                fwd0[i] = start
            if clk is not None:
                rec += [(clk[i][0] - fwd0[i]) / 1e3, (clk[i][1] - fwd0[i]) / 1e3]
            else:
                rec += [float("nan"), float("nan")]
            rec += [float(num_recv), float(own)]
            out[direction, i, :_HIER_TRACE_BINS0] = torch.tensor(
                rec, dtype=torch.float64
            )
            out[direction, i, _HIER_TRACE_BINS0:] = torch.tensor(
                row[_HIER_TRACE_SLOTS : _HIER_TRACE_SLOTS + 3 * _HIER_TRACE_BINS],
                dtype=torch.float64,
            )
    return out


def _quantile(values: torch.Tensor, quantile: float) -> float:
    values = values[~torch.isnan(values)]
    return float(torch.quantile(values, quantile)) if values.numel() else float("nan")


def _bin_edges(bins: torch.Tensor, bin_us: float, end_us: torch.Tensor) -> torch.Tensor:
    """[..., num_bins + 1] us after the start of timeline bins [...,
    num_bins]: bin i from i to i + 1 widths, the last one to the kernel's end
    ``end_us`` (it holds what came later too)."""
    num_bins = bins.shape[-1]
    edges = torch.arange(num_bins + 1, dtype=bins.dtype) * bin_us
    edges = edges.expand(*bins.shape[:-1], num_bins + 1).clone()
    edges[..., -1] = torch.fmax(edges[..., -1], end_us)
    return edges


def _share_in(
    bins: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    """The share of a timeline's total (bins [..., num_bins]) within [lo, hi]
    (us after the start; nan: none), each bin counted pro rata."""
    starts, ends = edges[..., :-1], edges[..., 1:]
    part = torch.minimum(ends, hi[..., None]) - torch.maximum(starts, lo[..., None])
    inside = part.clamp(min=0) / (ends - starts).clamp(min=1e-9)
    tot = bins.sum(-1)
    out = (bins * inside).sum(-1) / tot.clamp(min=1e-30)
    return torch.where(
        (tot > 0) & ~torch.isnan(lo) & ~torch.isnan(hi), out, float("nan")
    )


def _span(
    bins: torch.Tensor, edges: torch.Tensor, quantile: float
) -> Tuple[torch.Tensor, ...]:
    """When a timeline had moved the share ``quantile`` of its total and when
    1 - ``quantile`` (us after the start, linear within a bin; nan when it
    moved nothing)."""
    num_bins = bins.shape[-1]
    tot = bins.sum(-1, keepdim=True)
    cum = (bins.cumsum(-1) / tot.clamp(min=1e-30)).contiguous()
    out = []
    for target in (quantile, 1 - quantile):
        idx = torch.searchsorted(cum, torch.full_like(cum[..., :1], target)).clamp(
            max=num_bins - 1
        )
        cum_hi = cum.gather(-1, idx)
        cum_lo = torch.where(
            idx > 0, cum.gather(-1, (idx - 1).clamp(min=0)), torch.zeros_like(cum_hi)
        )
        frac = ((target - cum_lo) / (cum_hi - cum_lo).clamp(min=1e-30)).clamp(0, 1)
        edge_lo = edges.gather(-1, idx)
        at = edge_lo + frac * (edges.gather(-1, idx + 1) - edge_lo)
        out.append(torch.where(tot > 0, at, float("nan")).squeeze(-1))
    return tuple(out)


def _node_hull(values: torch.Tensor, local_size: int, lowest: bool) -> torch.Tensor:
    """[world_size, iterations]: per rank, the least (or greatest) of
    ``values`` over its node's ranks, nan where every rank's is nan."""
    world_size = values.shape[0]
    num_nodes = world_size // local_size
    fill = float("inf") if lowest else float("-inf")
    grouped = torch.where(torch.isnan(values), fill, values).view(
        num_nodes, local_size, -1
    )
    hull = (
        grouped.min(1, keepdim=True).values
        if lowest
        else grouped.max(1, keepdim=True).values
    )
    hull = hull.expand(num_nodes, local_size, hull.shape[-1]).reshape(world_size, -1)
    return torch.where(torch.isinf(hull), float("nan"), hull)


# NVLink moves, for the overlap's IB side: between the times the share
# _NV_QUANTILE of a kernel's NVLink bytes and 1 - _NV_QUANTILE of them had
# moved (a trickle at either end does not count).
_NV_QUANTILE = 0.05


def trace_summary(records: torch.Tensor, local_size: int) -> Dict:
    """The traced kernels' metrics (records: [world_size, 2, iterations, fields]
    from every rank's trace_records). Per direction and iteration, over ranks
    (min / mean / max), then the median over iterations; times in us,
    utilizations in %.
      end_us       kernel time: its last stamp (exited) after its start
      ib_win_out   bytes put / (IB_PEAK_GBPS x (puts done - first put))
      ib_win_in    bytes received / (IB_PEAK_GBPS x (all landed - the earlier
                   of its first landing and its senders' first put))
      skew_all     latest start of any rank - this rank's start (rank 0's clock)
      skew_node    first entry - start: the wait for its node peers
      ib_inkern    ib_bytes / (IB_PEAK_GBPS x (end_us - skew_all)), ib_bytes
                   = max(bytes put, bytes received) over IB: the kernel with
                   every rank started (the upstream skew taken out)
      overlap_nv   the share of its NVLink bytes moved while its IB window was
                   open (from its first put or landing to its last), from the
                   timelines' bins, counted pro rata
      overlap_ib   the share of its IB rows landed while its NVLink moved (from
                   _NV_QUANTILE to 1 - _NV_QUANTILE of its NVLink bytes)
      overlap_nv_node, overlap_ib_node
                   the same with the node's IB windows and NVLink spans
                   instead of its own (rank 0's clock: from the first of its
                   node's ranks to the last)
    """
    field_at = {key: i for i, key in enumerate(_HIER_TRACE_FIELDS)}
    world_size, _, num_iters, _ = records.shape
    res = {}
    for direction, name in ((0, "fwd"), (1, "bwd")):
        recs = records[:, direction]  # [world_size, num_iters, fields]
        field = lambda key: recs[:, :, field_at[key]]  # noqa: E731
        end_us = field("exited")
        start = field("start_abs") / 1e3
        skew_all = start.max(dim=0, keepdim=True).values - start
        skew_node = field("enter_first")
        node_spread = torch.stack(
            [
                start[node * local_size : (node + 1) * local_size].max(0).values
                - start[node * local_size : (node + 1) * local_size].min(0).values
                for node in range(world_size // local_size)
            ]
        ).mean(0)
        ib_bytes = torch.maximum(field("bytes_out"), field("bytes_in"))
        # The receive window starts no later than the first put of a rail
        # peer that sends to this rank (rows land before a late receiver's
        # kernel starts, and its first landing is seen at its first poll):
        # times in rank 0's clock, us.
        absolute = lambda key: start + field(key)  # noqa: E731
        sent0 = absolute("first_put")
        sent_min = torch.full_like(sent0, float("nan"))  # senders' earliest first put
        for rank in range(world_size):
            peers = [
                node * local_size + rank % local_size
                for node in range(world_size // local_size)
                if node != rank // local_size
            ]
            if peers:
                earliest = sent0[peers].nan_to_num(nan=float("inf")).min(0).values
                sent_min[rank] = torch.where(
                    torch.isinf(earliest),
                    torch.full_like(earliest, float("nan")),
                    earliest,
                )
        in_lo = torch.fmin(absolute("first_landed"), sent_min)
        win_in = absolute("landed") - in_lo
        # Cross-node check of the calibration: rows cannot be seen landing
        # before their sender posted its first put. Per iteration, the least
        # (first landing seen - earliest first put of its rail senders), us;
        # negative values bound the calibration's error from below.
        lag = absolute("first_landed") - sent_min
        causal = []
        for i in range(num_iters):
            col = lag[:, i]
            col = col[~torch.isnan(col)]
            causal.append(col.min().item() if col.numel() else float("nan"))
        per = {
            "end_us": end_us,
            "skew_all": skew_all,
            "skew_node": skew_node,
            "first_put": field("first_put"),
            "puts_done": field("puts_done"),
            "first_landed": field("first_landed"),
            "landed": field("landed"),
            "nv_first": field("nv_first"),
            "nv_last": field("nv_last"),
            "ib_win_out": 100
            * field("bytes_out")
            / (IB_PEAK_GBPS * 1e3 * (field("puts_done") - field("first_put"))),
            "ib_win_in": 100 * field("bytes_in") / (IB_PEAK_GBPS * 1e3 * win_in),
            "ib_inkern": 100 * ib_bytes / (IB_PEAK_GBPS * 1e3 * (end_us - skew_all)),
            "nv_GB": field("nv_bytes") / 1e9,
            "ib_rows_seen_in": field("ib_in"),
            "ib_rows_posted": field("ib_out"),
            # plain loop: the input dist, and the lookup up to this
            # forward kernel's start (the call's, on the rank's own clock)
            "in_dist": field("lookup_start") - field("in_start"),
            "lookup_to_fwd": -field("lookup_start"),
            "num_recv": field("num_recv"),
        }
        # ib_bytes = 0 (one node, or nothing to send): no IB utilization
        per["ib_inkern"][ib_bytes <= 0] = float("nan")
        ib_lo = torch.fmin(field("first_put"), in_lo - start)
        ib_hi = torch.fmax(field("puts_done"), field("landed"))
        # Byte-weighted, from the timelines (each rank's own clock from its
        # start; the node's windows through rank 0's clock).
        timeline = recs[:, :, _HIER_TRACE_BINS0:].view(
            world_size, num_iters, 3, _HIER_TRACE_BINS
        )
        nv_bins, in_bins = timeline[:, :, 0], timeline[:, :, 1]
        edges = _bin_edges(nv_bins, _HIER_TRACE_BIN_US, end_us)
        nv_from, nv_to = _span(nv_bins, edges, _NV_QUANTILE)
        per["overlap_nv"] = 100 * _share_in(nv_bins, ib_lo, ib_hi, edges)
        per["overlap_ib"] = 100 * _share_in(in_bins, nv_from, nv_to, edges)
        node_lo = _node_hull(start + ib_lo, local_size, True) - start
        node_hi = _node_hull(start + ib_hi, local_size, False) - start
        per["overlap_nv_node"] = 100 * _share_in(nv_bins, node_lo, node_hi, edges)
        nv_node0 = _node_hull(start + nv_from, local_size, True) - start
        nv_node1 = _node_hull(start + nv_to, local_size, False) - start
        per["overlap_ib_node"] = 100 * _share_in(in_bins, nv_node0, nv_node1, edges)
        stats = {}
        for key, values in per.items():
            nan = torch.isnan(values)
            lo = (
                torch.where(nan, torch.full_like(values, float("inf")), values)
                .min(0)
                .values
            )
            hi = (
                torch.where(nan, torch.full_like(values, float("-inf")), values)
                .max(0)
                .values
            )
            lo[nan.all(0)] = hi[nan.all(0)] = float("nan")
            stats[key] = {
                "min": _quantile(lo, 0.5),
                "mean": _quantile(torch.nanmean(values, dim=0), 0.5),
                "max": _quantile(hi, 0.5),
            }
        stats["start_spread_all"] = {
            "median": _quantile(skew_all.max(0).values, 0.5),
            "per_iter": skew_all.max(0).values.tolist(),
        }
        stats["start_spread_node"] = {
            "median": _quantile(node_spread, 0.5),
            "per_iter": node_spread.tolist(),
        }
        # Calibration check: node peers pass the entry barrier at one instant
        # (the last peer's flag), so their first entries in rank 0's clock
        # differ only by the calibration's error (and CTA scheduling).
        ent = field("enter_abs") / 1e3
        node_ent = (
            torch.stack(
                [
                    ent[node * local_size : (node + 1) * local_size].max(0).values
                    - ent[node * local_size : (node + 1) * local_size].min(0).values
                    for node in range(world_size // local_size)
                ]
            )
            .max(0)
            .values
        )
        stats["enter_spread_node_check"] = {
            "median": _quantile(node_ent, 0.5),
            "max": _quantile(node_ent, 1.0),
        }
        stats["causality_min_us"] = {
            "median": _quantile(torch.tensor(causal), 0.5),
            "min": _quantile(torch.tensor(causal), 0.0),
        }
        res[name] = stats
    return res


def print_trace_summary(summ: Dict) -> None:
    for name in ("fwd", "bwd"):
        stats = summ[name]
        print(
            f"  {name} kernel trace (median over iterations of min / mean / max over ranks):"
        )
        for key in (
            "end_us",
            "skew_all",
            "skew_node",
            "first_put",
            "puts_done",
            "first_landed",
            "landed",
            "nv_first",
            "nv_last",
            "ib_win_out",
            "ib_win_in",
            "ib_inkern",
            "nv_GB",
            "overlap_nv",
            "overlap_ib",
            "overlap_nv_node",
            "overlap_ib_node",
            "in_dist",
            "lookup_to_fwd",
        ):
            stat = stats[key]
            print(
                f"    {key:<14s} {stat['min']:10.3f} {stat['mean']:10.3f}"
                f" {stat['max']:10.3f}"
            )
        print(
            "    start spread all ranks"
            f" {stats['start_spread_all']['median']:.1f} us, node peers"
            f" {stats['start_spread_node']['median']:.1f} us; calibration check (node"
            f" peers' first entries, one instant): median"
            f" {stats['enter_spread_node_check']['median']:.1f} us, max"
            f" {stats['enter_spread_node_check']['max']:.1f} us; cross-node causality"
            f" (first landing seen - senders' first put) min {stats['causality_min_us']['min']:.1f} us"
        )


def device_used() -> int:
    """Device memory in use on this GPU by anything (cudaMemGetInfo): the
    CUDA context, NCCL, the caching allocator's segments, the windows, the
    tables' virtual-memory buffers."""
    free, total = torch.cuda.mem_get_info()
    return total - free


# Kernels of the hierarchical output dist, by name: the forward's and the
# backward's.
_OUTPUT_DIST_FWD_RE = re.compile(r"\bhier_output_dist::fwd_kernel\b")
_OUTPUT_DIST_BWD_RE = re.compile(r"\bhier_output_dist::bwd_kernel\b")


def output_dist_kernel_times(prof) -> List[List[float]]:
    """Per call, from a CUDA-only profile of the --kernel_times iterations:
    the GPU ms of the hierarchical output dist's forward kernel and of its
    backward kernel, each with its wait at the entry for the other ranks, in
    call order (each backward kernel with the forward kernel before it).
    Empty for TorchRec's output dist."""
    kernels = sorted(
        (event.time_range.start, event.time_range.end, event.name)
        for event in prof.events()
        if event.device_type == DeviceType.CUDA
        and (
            _OUTPUT_DIST_FWD_RE.search(event.name)
            or _OUTPUT_DIST_BWD_RE.search(event.name)
        )
    )
    calls, fwd = [], 0.0
    for begin, end, name in kernels:
        ms = (end - begin) / 1e3
        if _OUTPUT_DIST_BWD_RE.search(name):
            calls.append([fwd, ms])
        else:
            fwd = ms
    return calls


def nic_counters() -> Dict[str, Tuple[int, int]]:
    """Bytes each InfiniBand port of this node has sent and received so far
    (its port counters, which count 4-byte words), by device."""
    out = {}
    for port in sorted(glob.glob("/sys/class/infiniband/*/ports/1")):
        try:
            with open(f"{port}/link_layer") as handle:
                if handle.read().strip() != "InfiniBand":
                    continue
            words = []
            for counter in ("port_xmit_data", "port_rcv_data"):
                with open(f"{port}/counters/{counter}") as handle:
                    words.append(int(handle.read()))
        except OSError:
            continue
        out[port.split("/")[4]] = (4 * words[0], 4 * words[1])
    return out


# ── benchmark ───────────────────────────────────────────────────────────────


def time_iterations(model, batches) -> Tuple[List[float], List[float]]:
    fwd_ms, bwd_ms = [], []
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in batches
    ]
    for kjt, (start, middle, end) in zip(batches, events):
        start.record()
        loss = model(Batch(kjt))[0]
        middle.record()
        loss.backward()
        end.record()
    torch.cuda.synchronize()
    for start, middle, end in events:
        fwd_ms.append(start.elapsed_time(middle))
        bwd_ms.append(middle.elapsed_time(end))
    return fwd_ms, bwd_ms


def time_pipeline(pipeline, batches, iterations: int) -> List[float]:
    """``iterations`` calls of the pipeline's progress(), each one batch's
    forward and backward (with the next batch's input dist in between), on
    the device's timeline (the default stream)."""
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iterations)
    ]
    for start, end in events:
        start.record()
        pipeline.progress(batches)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def gather_floats(values: Union[List[float], torch.Tensor]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    out = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(out, tensor)
    return torch.stack(out).cpu()


def check_against_torchrec(
    cfg: BenchConfig, device, world_size, local_size, rank, steps: int = 4
) -> Optional[str]:
    """Same inputs through a model on TorchRec's dists and one on the
    hierarchical dist (``--output_dist``'s, or ``fused`` when that is
    ``torchrec``): None when they match, else why not, the same on every rank.

    Both models start from the DEBUG initializer (a key's row is a function of
    the key) and train with SGD at learning rate 1 against output weights of
    +-1, so every gradient is a sum of +-1/world_size. For a power-of-two
    world_size that sum is exact in fp32 in any order (the hierarchical
    backward sums a node's duplicates before the owner does), so with fp32
    rows the forward outputs must match bit for bit after every step. bf16
    rows round the node sums of hot keys, and with any other world_size the
    sums of 1/world_size depend on their order, so both get a tolerance.

    Keys come from at most an eighth of the table, so nothing is evicted: which
    rows a full table evicts is not deterministic, and two models evicting
    different rows would differ whatever their output dist.

    Under ``--pipeline`` the checked model trains under that pipeline, whose
    outputs must match the plain loop's of TorchRec's model step by step.

    The checked model must also have launched its own output dist's kernels
    at every step, in both directions, and no call may have run on TorchRec's
    dists (a sharding that fell back, or a call over its capacities): that
    would compare TorchRec's dists with themselves. Its capacities are
    ``--capacity``, or measured on the check's own batches.
    """
    key_range = min(
        cfg.key_range or cfg.num_embeddings, max(cfg.num_embeddings // 8, 1)
    )
    cfg = replace(cfg, optimizer="sgd", learning_rate=1.0, key_range=key_range)
    torch.manual_seed(7)
    weight = torch.randn(cfg.embedding_dim, device=device).sign()
    debug = DynamicEmbInitializerMode.DEBUG
    fused = cfg.output_dist if cfg.output_dist != "torchrec" else "fused"
    batches = generate_batches(cfg, rank, device, steps)
    cfg = resolve_capacity(replace(cfg, output_dist=fused), batches, local_size)
    torchrec_model = build_model(
        cfg, device, world_size, local_size, "torchrec", debug, weight
    )
    hier_model = build_model(cfg, device, world_size, local_size, fused, debug, weight)
    pipeline = train_pipeline(cfg, hier_model) if cfg.pipeline != "none" else None
    fed = iter([Batch(kjt) for kjt in batches])
    exact = (
        cfg.emb_dtype == "fp32"
        and cfg.output_dtype == "fp32"
        and (world_size & (world_size - 1)) == 0  # a power of two
    )
    ok = True
    _OUTPUT_DIST_COUNTS.clear()
    for step, kjt in enumerate(batches):
        loss_torchrec, out_torchrec = torchrec_model(Batch(kjt))
        loss_hier = None
        if pipeline is None:
            loss_hier, out_hier = hier_model(Batch(kjt))
        else:
            out_hier = pipeline.progress(fed)  # this batch's forward and backward
        for name in out_torchrec.keys():
            expected, got = out_torchrec[name].values(), out_hier[name].values()
            same = (
                torch.equal(expected, got)
                if step == 0 or exact
                else torch.allclose(expected.float(), got.float(), rtol=1e-2, atol=1e-2)
            )
            if not same:
                ok = False
                bad = (expected != got).any(dim=1).nonzero().flatten()
                row = int(bad[0]) if bad.numel() else 0
                diff = (expected.float() - got.float()).abs().max().item()
                print(
                    f"[rank {rank}] step {step} feature {name}: {bad.numel()} of"
                    f" {expected.shape[0]} rows differ, max abs diff {diff:.3e}; row {row}:"
                    f" torchrec {expected[row, :4].tolist()} fused {got[row, :4].tolist()}",
                    flush=True,
                )
        loss_torchrec.backward()
        if loss_hier is not None:
            loss_hier.backward()
    ran = all(_OUTPUT_DIST_COUNTS[direction] >= steps for direction in (0, 1))
    flags = torch.tensor(
        [int(ok), int(ran), -hier_fallbacks(hier_model)], device=device
    )
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    # The models go when this returns: the hierarchical dist then frees its
    # windows at once, without its teardown's exit round.
    quiesce(hier_model)
    ok, ran, fallbacks = flags.tolist()
    if fallbacks:
        return f"{-fallbacks} calls of the {fused} model ran on TorchRec's dists"
    if not ran:
        return f"the {fused} model fell back to TorchRec's dists"
    return None if ok else "the outputs differ"


def run(cfg: BenchConfig) -> Dict:
    global _CALLS, _PROFILING
    if cfg.trace:
        from dynamicemb_extensions import hier_output_dist_traced

        if not hier_output_dist_traced():
            sys.exit("--trace: rebuild with DYNAMICEMB_HIER_TRACE=1")
    # Device memory in use at each step (cudaMemGetInfo), first before the
    # process group exists.
    torch.cuda.set_device(
        int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    )
    used = {"start": device_used()}
    rank, world_size, local_rank, local_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    used["process_group"] = device_used()
    install_stage_markers()
    torchrec_traffic = TorchRecTraffic(rank, local_size)
    hier_traffic = HierTraffic(rank, local_size)
    row_bytes = cfg.embedding_dim * _OUTPUT_DTYPES[cfg.output_dtype].as_dtype().itemsize

    if cfg.check:
        why = check_against_torchrec(cfg, device, world_size, local_size, rank)
        if rank == 0:
            print(
                "check against torchrec output dist: "
                + ("PASSED" if why is None else f"FAILED ({why})"),
                flush=True,
            )
        if why is not None:
            # why is the same on every rank, so every rank stops here together
            dist.destroy_process_group()
            sys.exit(1)
        # free the check's models (and their windows) before timing
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    # --kernel_times: as many iterations again after the timed ones
    kernel_iters = cfg.num_iterations if cfg.kernel_times else 0
    n_batches = cfg.warmup + cfg.num_iterations + kernel_iters + cfg.profile_iterations
    batches = generate_batches(cfg, rank, device, n_batches)
    # the capacities of every batch the model will see (untimed collectives)
    cfg = resolve_capacity(cfg, batches, local_size)
    if rank == 0 and cfg.output_dist_capacity is not None:
        print(f"output_dist_capacity: {cfg.output_dist_capacity}", flush=True)
    model = build_model(cfg, device, world_size, local_size, cfg.output_dist)
    torch.cuda.synchronize()
    used["model"] = device_used()
    # what the batches take of the memory allocated before the timed loop
    input_bytes = sum(
        tensor.numel() * tensor.element_size()
        for kjt in batches
        for tensor in (kjt.values(), kjt.lengths(), kjt.offsets())
    )
    pipeline = train_pipeline(cfg, model) if cfg.pipeline != "none" else None
    fed = iter([Batch(kjt) for kjt in batches])  # what the pipeline takes from

    def train(kjts) -> None:
        for kjt in kjts:
            if pipeline is None:
                model(Batch(kjt))[0].backward()
            else:
                pipeline.progress(fed)

    def reset_traffic() -> None:
        torchrec_traffic.reset()
        torchrec_traffic.enabled = True
        hier_traffic.reset()
        hier_traffic.enabled = True

    def traffic_of(iterations: int) -> Dict[str, float]:
        """Rows per iteration: of each call that took the hierarchical dist
        when some did (a call over its capacities ran on TorchRec's), else
        of TorchRec's output dist."""
        torchrec_traffic.enabled = hier_traffic.enabled = False
        if hier_traffic.calls:
            return hier_traffic.per_iteration(len(hier_traffic.calls))
        return torchrec_traffic.per_iteration(iterations)

    torch.cuda.synchronize()
    mem0, reserved0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    train(batches[:1])
    torch.cuda.synchronize()
    used["first_call"] = device_used()
    train(batches[1 : cfg.warmup])
    torch.cuda.synchronize()
    used["warmup"] = device_used()
    dist.barrier()

    # Tracing: the clocks of all ranks against rank 0's, before and after the
    # timed iterations (their stamps then compare across ranks).
    cal_before = clock_offsets(device) if cfg.trace else None
    timed = batches[cfg.warmup : cfg.warmup + cfg.num_iterations]
    reset_traffic()
    held = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    host = _HostSyncs(len(timed)) if cfg.host_stages and pipeline is None else None
    if pipeline is None:
        _CALLS = _Calls(len(timed), cfg.trace, host)
    nics0 = nic_counters() if cfg.nic_counters and local_rank == 0 else {}
    if host is not None:
        host.install()
    fallbacks0 = hier_fallbacks(model)
    t0 = time.perf_counter()
    iter_ms = None
    if pipeline is None:
        fwd_ms, bwd_ms = time_iterations(model, timed)
    else:
        iter_ms = time_pipeline(pipeline, fed, len(timed))
    wall_ms = (time.perf_counter() - t0) * 1000.0 / len(timed)
    # calls of the timed iterations that ran on TorchRec's dists (over the
    # capacities)
    fallback_calls = hier_fallbacks(model) - fallbacks0
    if host is not None:
        host.uninstall()
    nics = None
    if cfg.nic_counters:
        after = nic_counters() if local_rank == 0 else {}
        mine = {
            dev: [now - before for before, now in zip(nics0[dev], counts)]
            for dev, counts in after.items()
        }
        nics = [None] * world_size
        dist.all_gather_object(nics, mine)
        # per node, its local rank 0's: per device, bytes sent and received
        nics = [nics[node * local_size] for node in range(world_size // local_size)]
    used["timed"] = device_used()
    calls, _CALLS = _CALLS, None
    # The input dist's segment per call (plain loop; the calls that took the
    # hierarchical dist, unless none did): from before it to after its
    # second wait, and its three steps, in ms on the caller's stream.
    in_ms = []
    if calls is not None:
        in_ms = [
            [
                events[0].elapsed_time(events[1]),
                events[1].elapsed_time(events[2]),
                events[2].elapsed_time(events[3]),
                events[0].elapsed_time(events[3]),
            ]
            for events in [calls.events[call] for call in calls.op_calls()]
        ]
    # The hierarchical output dist's forward and backward launches per call
    # that took it, GPU ms.
    output_dist_ms = calls.output_dist_ms() if calls is not None else []
    # Bytes per rank: allocated before the timed iterations (the batches and
    # the dists' persistent tensors); past what was allocated before the
    # first batch, the most allocated and reserved during them; the most
    # allocated and reserved, all told; the windows of the hierarchical dist
    # (ncclMemAlloc, outside PyTorch's allocator); the batches.
    memory = [
        held,
        torch.cuda.max_memory_allocated() - mem0,
        torch.cuda.max_memory_reserved() - reserved0,
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
        sum(
            sum(hier.op.memory()["windows"].values())
            for hier in _hier_shardings(model)
            if hier.takes_hier_dist
        ),
        input_bytes,
    ]
    ib_calls = hier_traffic.per_call_ib()
    traffic = traffic_of(len(timed))

    records = None
    cal_stats = None
    phase_us = None
    if cfg.trace:
        torch.cuda.synchronize()
        cal_after = clock_offsets(device)
        # the timed calls that took the hierarchical dist (its kernels' trace)
        records = trace_records(
            model,
            rank,
            len(ib_calls),
            (cal_before, cal_after),
            ib_calls,
            row_bytes,
            [calls.clocks[call] for call in calls.op_calls()]
            if calls is not None and calls.clocks is not None
            else None,
        )
        phase_us = in_phases(model, len(timed))
        cal_stats = {
            "round_trip_us_max": max(
                cal_before[1:, 1].max().item(), cal_after[1:, 1].max().item()
            )
            / 1e3
            if world_size > 1
            else 0.0,
            "round_trip_us_median": cal_before[1:, 1].median().item() / 1e3
            if world_size > 1
            else 0.0,
            "drift_us_max": (cal_after[:, 0] - cal_before[:, 0]).abs().max().item()
            / 1e3,
            "between_ms": (cal_after[0, 2] - cal_before[0, 2]).item() / 1e6,
        }

    # --kernel_times: as many iterations again under a CUDA-only profiler (it
    # costs the host a few us per launch, so the timed iterations run without
    # it): per call, the output dist's kernels' GPU ms (forward, backward).
    kernel_ms = None
    kernel_fallbacks = 0
    if cfg.kernel_times:
        start = cfg.warmup + cfg.num_iterations
        torch.cuda.synchronize()
        dist.barrier()
        fallbacks0 = hier_fallbacks(model)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as kprof:
            if pipeline is None:
                time_iterations(model, batches[start : start + kernel_iters])
            else:
                time_pipeline(pipeline, fed, kernel_iters)
        kernel_fallbacks = hier_fallbacks(model) - fallbacks0
        # per call that took the hierarchical dist (the same calls on every
        # rank)
        kernel_ms = output_dist_kernel_times(kprof) or None

    breakdown = {}
    names = {}
    prof_traffic = None
    prof_fallbacks = 0
    if cfg.profile_iterations > 0:
        prof_batches = batches[cfg.warmup + cfg.num_iterations + kernel_iters :]
        torch.cuda.synchronize()
        dist.barrier()
        reset_traffic()
        fallbacks0 = hier_fallbacks(model)
        _PROFILING = True
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            train(prof_batches)
            torch.cuda.synchronize()
        _PROFILING = False
        prof_fallbacks = hier_fallbacks(model) - fallbacks0
        prof_traffic = traffic_of(len(prof_batches))
        breakdown = kernel_breakdown(prof, len(prof_batches))
        names = {
            key: kernel_names(prof, key, len(prof_batches))
            for key in (
                "fwd/input_dist",
                "fwd/lookup",
                "fwd/output_dist",
                "bwd/output_dist",
            )
        }
    hier_traffic.reset()

    # Aggregate over ranks.
    if iter_ms is None:
        times_all = [
            gather_floats([sum(times) / len(times)]) for times in (fwd_ms, bwd_ms)
        ]
    else:
        times_all = [gather_floats([sum(iter_ms) / len(iter_ms)])]
    memory_all = gather_floats(memory)
    # Every rank sees the same stage keys only if they ran the same code; use
    # rank 0's keys and fill missing ones with zero.
    keys_obj = [sorted(breakdown)]
    dist.broadcast_object_list(keys_obj, src=0)
    stage_keys = keys_obj[0]
    stage_values = []
    for stage in stage_keys:
        values = breakdown.get(stage, {})
        stage_values += [values.get(field, 0.0) for field in _STAGE_FIELDS]
    stage_all = gather_floats(stage_values) if stage_values else None
    traffic_keys = sorted(traffic)
    traffic_all = gather_floats([float(traffic[name]) for name in traffic_keys])
    syncs = None
    if host is not None:
        syncs = [None] * world_size
        dist.all_gather_object(syncs, [dict(per_call) for per_call in host.syncs])
    extra = {
        "in_ms": gather_floats(in_ms) if in_ms else None,  # [world_size, calls, 4]
        # [world_size, calls, 2]
        "output_dist_ms": gather_floats(output_dist_ms) if output_dist_ms else None,
        # the hierarchical dist's IB rows per call there (put, received)
        "output_dist_ib": gather_floats([list(call[:2]) for call in ib_calls])
        if output_dist_ms and len(ib_calls) == len(output_dist_ms)
        else None,
        "prof_traffic": None
        if prof_traffic is None
        else gather_floats(
            [float(prof_traffic.get(name, 0.0)) for name in traffic_keys]
        ),
        "records": gather_floats(records) if records is not None else None,
        "cal_stats": cal_stats,
        "in_phases": gather_floats(phase_us) if phase_us is not None else None,
        "syncs": syncs,
        "kernel_ms": gather_floats(kernel_ms) if kernel_ms is not None else None,
        "used": gather_floats(list(used.values())),
        "used_keys": list(used),
        "nics": nics,
        "fallback_calls": fallback_calls,
        "fallback_calls_kernel_times": kernel_fallbacks,
        "fallback_calls_profile": prof_fallbacks,
        "fallback_calls_all": hier_fallbacks(model),
    }

    result = None
    if rank == 0:
        result = report(
            cfg,
            world_size,
            local_size,
            times_all,
            wall_ms,
            stage_keys,
            stage_all,
            traffic_keys,
            traffic_all,
            names,
            memory_all,
            extra,
        )
        if cfg.trace_out and extra["records"] is not None:
            import numpy as np

            np.savez(
                cfg.trace_out,
                records=extra["records"].numpy(),
                cal_before=cal_before.numpy(),
                cal_after=cal_after.numpy(),
                fields=np.array(_HIER_TRACE_FIELDS),
                bins=_HIER_TRACE_BINS,
                bin_us=_HIER_TRACE_BIN_US,
                label=cfg.label(),
                config=json.dumps(asdict(cfg)),
            )
        if cfg.json_out:
            with open(cfg.json_out, "a") as handle:
                handle.write(json.dumps(result) + "\n")
    quiesce(model)
    dist.barrier()
    dist.destroy_process_group()
    return result


def _ib_per_call(nic_bytes: torch.Tensor, output_dist_ms: torch.Tensor) -> Dict:
    """IB per call, in % of IB_PEAK_GBPS: each NIC's bytes (``nic_bytes``,
    [world_size, calls]) over its GPU's output-dist time (``output_dist_ms``,
    [world_size, calls], ms); the median over calls of the mean and of the
    least over NICs, and of the busiest NIC's bytes over the slowest time."""
    util = 100 * nic_bytes / (IB_PEAK_GBPS * 1e6 * output_dist_ms)
    bott = (
        100
        * nic_bytes.max(dim=0).values
        / (IB_PEAK_GBPS * 1e6 * output_dist_ms.max(dim=0).values)
    )
    return {
        "mean": _quantile(util.mean(dim=0), 0.5),
        "min": _quantile(util.min(dim=0).values, 0.5),
        "bottleneck": _quantile(bott, 0.5),
        "per_call_mean": util.mean(dim=0).tolist(),
    }


def report(
    cfg,
    world_size,
    local_size,
    times_all,
    wall_ms,
    stage_keys,
    stage_all,
    traffic_keys,
    traffic_all,
    names,
    memory_all,
    more,
) -> Dict:
    row_bytes = cfg.embedding_dim * _OUTPUT_DTYPES[cfg.output_dtype].as_dtype().itemsize
    res = {
        "config": asdict(cfg),
        "label": cfg.label(),
        "world_size": world_size,
        "nodes": world_size // local_size,
        "iter_ms_host": wall_ms,
        "capacities": cfg.output_dist_capacity,
        # calls that did not fit the capacities (TorchRec's dists): of the
        # timed iterations, of the --kernel_times and profiled ones, and all
        # calls so far
        "fallback_calls": more["fallback_calls"],
        "fallback_calls_kernel_times": more["fallback_calls_kernel_times"],
        "fallback_calls_profile": more["fallback_calls_profile"],
        "fallback_calls_all": more["fallback_calls_all"],
    }
    if more["fallback_calls_all"]:
        print(
            f"fallback: {more['fallback_calls']} timed calls,"
            f" {more['fallback_calls_kernel_times']} of --kernel_times,"
            f" {more['fallback_calls_profile']} profiled"
            f" ({more['fallback_calls_all']} in all) did not fit output_dist_capacity"
            " and ran on TorchRec's dists; the per-call figures leave them out, the"
            " times and the kernel breakdown do not",
            flush=True,
        )
    # The forward and the backward of the plain loop, or a pipeline's whole
    # iterations (which overlap the next batch's input dist with both).
    parts = ("fwd", "bwd") if cfg.pipeline == "none" else ("iter",)
    for part, times in zip(parts, times_all):
        res.update(
            {
                f"{part}_ms_mean": times.mean().item(),
                f"{part}_ms_max": times.max().item(),
            }
        )
    print(
        f"\n=== {cfg.label()}  world_size={world_size}"
        f" nodes={world_size // local_size} ==="
    )
    if cfg.pipeline == "none":
        print(
            f"fwd {res['fwd_ms_mean']:.3f} ms (max rank {res['fwd_ms_max']:.3f})   "
            f"bwd {res['bwd_ms_mean']:.3f} ms (max rank {res['bwd_ms_max']:.3f})   "
            f"host iter {wall_ms:.3f} ms"
        )
    else:
        print(
            f"{cfg.pipeline} pipeline iteration {res['iter_ms_mean']:.3f} ms (max rank"
            f" {res['iter_ms_max']:.3f})   host iter {wall_ms:.3f} ms"
        )
    # [world_size, 7]: held, peak and reserved past the start, the most
    # allocated and reserved, windows, batches
    mem = memory_all.max(dim=0).values.tolist()
    held, peak, reserved, alloc_max, reserved_max, windows, inputs = mem
    total_alloc = (memory_all[:, 3] + memory_all[:, 5]).max().item()
    total_reserved = (memory_all[:, 4] + memory_all[:, 5]).max().item()
    res.update(
        mem_held_bytes=held,
        mem_peak_bytes=peak,
        mem_reserved_bytes=reserved,
        mem_alloc_max_bytes=alloc_max,
        mem_reserved_max_bytes=reserved_max,
        mem_window_bytes=windows,
        mem_input_bytes=inputs,
        mem_total_alloc_bytes=total_alloc,
        mem_total_reserved_bytes=total_reserved,
    )
    print(
        f"memory per rank (GB, max over ranks): {held / 1e9:.3f} allocated before the"
        f" timed iterations ({inputs / 1e9:.3f} of it the batches; the tables are not"
        f" in PyTorch's allocator); during them, the most allocated {alloc_max / 1e9:.3f}"
        f" ({peak / 1e9:.3f} past the start) and reserved {reserved_max / 1e9:.3f}"
        f" ({reserved / 1e9:.3f} past the start); the hierarchical dist's windows"
        f" {windows / 1e9:.3f}; totals with the windows: allocated {total_alloc / 1e9:.3f},"
        f" reserved {total_reserved / 1e9:.3f}"
    )
    res["mem_per_rank"] = {
        name: memory_all[:, i].tolist()
        for i, name in enumerate(
            (
                "held",
                "peak",
                "reserved",
                "alloc_max",
                "reserved_max",
                "windows",
                "inputs",
            )
        )
    }
    # [world_size, checkpoints]: device memory in use (cudaMemGetInfo)
    used = more["used"]
    res["mem_device_used_bytes"] = dict(
        zip(more["used_keys"], used.max(dim=0).values.tolist())
    )
    print(
        "device memory in use per rank (GB, max over ranks; cudaMemGetInfo): "
        + ", ".join(
            f"{name} {nbytes / 1e9:.3f}"
            for name, nbytes in res["mem_device_used_bytes"].items()
        )
    )
    syncs = more.get("syncs")
    if syncs is not None:
        # per rank and call: host ms waiting in device syncs, by call site
        res["host_syncs"] = syncs
        sites = defaultdict(float)
        for per_rank in syncs:
            for call in per_rank:
                for site, ms in call.items():
                    sites[site] += ms / (len(syncs) * len(per_rank))
        print(
            "  device syncs per call (host ms, mean over ranks and calls), by call site:"
        )
        for site, ms in sorted(sites.items(), key=lambda item: -item[1])[:8]:
            print(f"    {ms:8.3f}  {site}")
    in_ms = more.get("in_ms")
    if in_ms is not None:
        # [world_size, calls, 4]: part 1, first wait, second wait, the whole
        # segment
        per_rank = in_ms.mean(dim=1)  # [world_size, 4]
        res["in_seg_ms_mean"] = per_rank[:, 3].mean().item()
        res["in_seg_ms_max_rank"] = per_rank[:, 3].max().item()
        res["in_seg_ms_call_max"] = in_ms[:, :, 3].max(dim=0).values.mean().item()
        # the last rank to get there waits least for the others' earlier steps
        res["in_seg_ms_call_min"] = in_ms[:, :, 3].min(dim=0).values.mean().item()
        res["in_seg_steps_ms"] = per_rank[:, :3].mean(dim=0).tolist()
        res["in_seg_ms_per_call"] = in_ms[:, :, 3].tolist()
        print(
            f"input dist segment (caller's stream, from before input_dist() to after its"
            f" second wait): mean {res['in_seg_ms_mean']:.3f} ms, max rank"
            f" {res['in_seg_ms_max_rank']:.3f}, mean over calls of the slowest rank"
            f" {res['in_seg_ms_call_max']:.3f} and of the least over ranks"
            f" {res['in_seg_ms_call_min']:.3f}; part 1 / first wait / second wait"
            f" {' / '.join(f'{ms:.3f}' for ms in res['in_seg_steps_ms'])} ms"
        )
    phase_us = more.get("in_phases")
    if phase_us is not None:
        # the in channel's phases (trace), mean over the ranks that ran each
        phases = {
            name: (
                float(col[~torch.isnan(col)].mean())
                if (~torch.isnan(col)).any()
                else None
            )
            for name, col in zip(_IN_PHASES, phase_us.t())
        }
        res["in_phases_us"] = phases
        print(
            "input dist phases (us, trace, mean over ranks): "
            + ", ".join(
                f"{name} {value:.1f}"
                for name, value in phases.items()
                if value is not None
            )
        )
    output_dist_ms = more.get("output_dist_ms")
    if output_dist_ms is not None:
        # [world_size, calls, 2]: the hierarchical output dist's forward and backward
        # launches (GPU ms per call, timed iterations)
        res["output_dist_event_ms"] = output_dist_ms.tolist()
        # per call, over ranks
        mean, top = output_dist_ms.mean(dim=0), output_dist_ms.max(dim=0).values
        res["output_dist_event_ms_median"] = {
            phase: _quantile(mean[:, i], 0.5) for i, phase in enumerate(("fwd", "bwd"))
        }
        res["output_dist_event_ms_max_median"] = {
            phase: _quantile(top[:, i], 0.5) for i, phase in enumerate(("fwd", "bwd"))
        }
        print(
            "output dist launches per call, timed iterations (GPU ms on the caller's"
            " stream, events right before and after them; median over calls of the mean"
            " / the max over ranks): "
            + ", ".join(
                f"{phase} {res['output_dist_event_ms_median'][phase]:.3f} /"
                f" {res['output_dist_event_ms_max_median'][phase]:.3f}"
                for phase in ("fwd", "bwd")
            )
        )
        plan_ib = more.get("output_dist_ib")
        if plan_ib is not None and plan_ib.sum() > 0:
            nic_bytes = plan_ib.max(dim=2).values * row_bytes  # [world_size, calls]
            res["ib_event_basis_calls"] = {
                phase: _ib_per_call(nic_bytes, output_dist_ms[:, :, i])
                for i, phase in enumerate(("fwd", "bwd"))
            }
            print(
                "  IB event basis per call there, over the launch's time (median over"
                " calls; mean over NICs / least NIC / busiest NIC over the slowest launch): "
                + ", ".join(
                    f"{phase} {stats['mean']:.1f} / {stats['min']:.1f} /"
                    f" {stats['bottleneck']:.1f}%"
                    for phase, stats in res["ib_event_basis_calls"].items()
                )
            )
    kernel_ms = more.get("kernel_ms")
    if kernel_ms is not None and not torch.isnan(kernel_ms).all():
        # [world_size, calls, 2]: forward, backward (GPU ms per call)
        res["output_dist_kernel_ms"] = kernel_ms.tolist()
        mean = kernel_ms.mean(dim=0)  # per call, over ranks
        res["output_dist_kernel_ms_median"] = {
            name: _quantile(mean[:, i], 0.5) for i, name in enumerate(("fwd", "bwd"))
        }
        res["output_dist_kernel_ms_mean"] = {
            name: mean[:, i].nanmean().item() for i, name in enumerate(("fwd", "bwd"))
        }
        print(
            "output dist kernels per call, CUDA-only profiled iterations (GPU ms, mean over"
            " ranks; median / mean over calls): "
            + ", ".join(
                f"{name} {res['output_dist_kernel_ms_median'][name]:.3f} /"
                f" {res['output_dist_kernel_ms_mean'][name]:.3f}"
                for name in ("fwd", "bwd")
            )
        )
    nics = more.get("nics")
    if nics is not None:
        # per node and InfiniBand device: bytes sent and received per iteration
        iters = cfg.num_iterations
        res["nic_bytes_per_iter"] = [
            {dev: [count / iters for count in counts] for dev, counts in node.items()}
            for node in nics
        ]
        for node_idx, node in enumerate(res["nic_bytes_per_iter"]):
            busy = sorted(
                (
                    (counts[0], counts[1], dev)
                    for dev, counts in node.items()
                    if max(counts) > 1e6
                ),
                reverse=True,
            )
            print(
                f"  node {node_idx}: {len(busy)} InfiniBand ports moved over 1 MB per iteration;"
                " sent / received MB: "
                + ", ".join(
                    f"{dev} {sent / 1e6:.1f} / {received / 1e6:.1f}"
                    for sent, received, dev in busy
                )
            )
    # traffic: mean per rank
    traffic_mean = {
        name: traffic_all[:, i].mean().item() for i, name in enumerate(traffic_keys)
    }
    traffic_max = {
        name: traffic_all[:, i].max().item() for i, name in enumerate(traffic_keys)
    }
    res["traffic_mean"] = traffic_mean
    res["traffic_max"] = traffic_max
    res["traffic_per_rank"] = {
        name: traffic_all[:, i].tolist() for i, name in enumerate(traffic_keys)
    }
    print("output dist traffic per rank per iteration (fwd; bwd mirrors it):")
    for name in traffic_keys:
        if name.startswith("rail_"):
            continue
        extra = ""
        if name.startswith("rows_"):
            extra = f"  = {traffic_mean[name] * row_bytes / 1e6:.2f} MB"
        print(
            f"  {name:<24s} mean {traffic_mean[name]:14.1f}"
            f"   max {traffic_max[name]:14.1f}{extra}"
        )
    if traffic_mean.get("rows_remote", 0) > 0:
        # the hierarchical dist across nodes: what node dedup kept off IB
        ib_out, remote = traffic_mean["rows_ib_out"], traffic_mean["rows_remote"]
        print(
            f"  node dedup: {ib_out:.0f} rows over IB for {remote:.0f} asked of remote"
            f" owners ({ib_out / remote:.3f})"
        )
    if stage_all is not None:
        print(
            "GPU time per iteration by stage (mean over ranks; kernel time, overlaps not removed):"
        )
        print(f"  {'stage':<26s} {'ms':>9s} {'kernels':>8s} {'net ms':>9s}")
        res["stages"] = {}
        num_fields = len(_STAGE_FIELDS)
        for i, stage in enumerate(stage_keys):
            column = stage_all[:, num_fields * i : num_fields * (i + 1)]
            values = column.mean(dim=0).tolist()
            res["stages"][stage] = dict(zip(_STAGE_FIELDS, values))
            print(f"  {stage:<26s} {values[0]:9.3f} {values[1]:8.1f} {values[2]:9.3f}")
        idx = {stage: i for i, stage in enumerate(stage_keys)}
        prof_traffic = more.get("prof_traffic")
        if prof_traffic is not None and "rail_out_0" in traffic_keys:
            # TorchRec's all-to-all, IB per NIC and direction, kernel basis
            # ("pxn"): the bytes a NIC sends and receives in the profiled
            # iterations (the more of the two) over the longest NCCL kernel of
            # its node there. NCCL's PXN sends a node's rows for the GPUs of a
            # rail through its NIC of that rail.
            profiled_traffic = {
                name: prof_traffic[:, i] for i, name in enumerate(traffic_keys)
            }
            fwd_out, fwd_in = (
                profiled_traffic["rows_ib"],
                profiled_traffic["rows_ib_in"],
            )
            num_ranks = fwd_out.numel()
            nodes = num_ranks // local_size
            rail_out = torch.stack(
                [profiled_traffic[f"rail_out_{rail}"] for rail in range(local_size)], 1
            )
            rail_in = torch.stack(
                [profiled_traffic[f"rail_in_{rail}"] for rail in range(local_size)], 1
            )
            # NIC (node, rail): its node's rows to that rail of other nodes
            pxn_fwd_out = (
                rail_out.view(nodes, local_size, local_size).sum(1).reshape(num_ranks)
            )
            pxn_bwd_out = (
                rail_in.view(nodes, local_size, local_size).sum(1).reshape(num_ranks)
            )
            pxn = {"fwd": (pxn_fwd_out, fwd_in), "bwd": (pxn_bwd_out, fwd_out)}
            res["ib_kernel_basis"] = {}
            for phase in ("fwd", "bwd"):
                key = f"{phase}/output_dist"
                if key not in idx:
                    continue
                out_rows, in_rows = pxn[phase]
                nic_bytes = torch.maximum(out_rows, in_rows) * row_bytes
                output_dist_ms = stage_all[
                    :, num_fields * idx[key] + _STAGE_FIELDS.index("net_ms")
                ]
                # a NIC carries its node's GPUs' rows: their kernels' span
                output_dist_ms = (
                    output_dist_ms.view(nodes, local_size)
                    .max(1, keepdim=True)
                    .values.expand(nodes, local_size)
                    .reshape(num_ranks)
                )
                ok = (output_dist_ms > 0) & (nic_bytes > 0)
                if not ok.any():
                    continue
                util = 100 * nic_bytes[ok] / (IB_PEAK_GBPS * 1e6 * output_dist_ms[ok])
                bott = (
                    100 * nic_bytes.max() / (IB_PEAK_GBPS * 1e6 * output_dist_ms.max())
                )
                res["ib_kernel_basis"][f"{phase}_pxn"] = {
                    "min": util.min().item(),
                    "mean": util.mean().item(),
                    "max": util.max().item(),
                    "bottleneck": bott.item(),
                }
                print(
                    f"  {phase} output dist IB (pxn), kernel basis (bytes of each NIC over"
                    f" its node's longest NCCL kernel, profiled iterations): min"
                    f" {util.min().item():5.1f}% mean {util.mean().item():5.1f}% max"
                    f" {util.max().item():5.1f}%; busiest NIC over the slowest kernel"
                    f" {bott.item():5.1f}%"
                )
        for key, kernels in names.items():
            print(f"  kernels in {key} per iteration (rank 0):")
            busiest = sorted(kernels.items(), key=lambda item: -item[1][1])[:12]
            for kernel_name, (count, ms) in busiest:
                print(f"    {count:5.1f} x  {ms:9.3f} ms  {kernel_name[:110]}")
        res["kernel_names"] = {
            key: {kernel_name: list(stats) for kernel_name, stats in kernels.items()}
            for key, kernels in names.items()
        }
    records = more.get("records")
    if records is not None:
        summ = trace_summary(records, local_size)
        res["trace"] = summ
        cal = more.get("cal_stats")
        if cal:
            res["trace_cal"] = cal
            print(
                f"  clock calibration: ping-pong round trip median"
                f" {cal['round_trip_us_median']:.1f} us, max {cal['round_trip_us_max']:.1f}"
                f" us; offsets drifted by up to"
                f" {cal['drift_us_max']:.1f} us over {cal['between_ms']:.0f} ms (interpolated)"
            )
        print_trace_summary(summ)
    return res


def parse_args(argv: List[str]) -> BenchConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    defaults = BenchConfig()
    parser.add_argument("--num_tables", type=int, default=defaults.num_tables)
    parser.add_argument(
        "--features_per_table",
        type=int,
        default=defaults.features_per_table,
        help="features sharing each table",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=defaults.num_embeddings,
        help="rows per table, all ranks together",
    )
    parser.add_argument("--embedding_dim", type=int, default=defaults.embedding_dim)
    parser.add_argument(
        "--local_batch_size", type=int, default=defaults.local_batch_size
    )
    parser.add_argument("--max_seq_len", type=int, default=defaults.max_seq_len)
    parser.add_argument(
        "--seq_len_dist", choices=["fixed", "uniform"], default=defaults.seq_len_dist
    )
    parser.add_argument(
        "--key_dist", choices=["uniform", "pow-law"], default=defaults.key_dist
    )
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument(
        "--key_range",
        type=int,
        default=defaults.key_range,
        help="keys are drawn below it; default num_embeddings",
    )
    parser.add_argument(
        "--dist_type",
        choices=["roundrobin", "hash_roundrobin"],
        default=defaults.dist_type,
    )
    parser.add_argument(
        "--emb_dtype", choices=list(_TABLE_DTYPES), default=defaults.emb_dtype
    )
    parser.add_argument(
        "--output_dtype", choices=list(_OUTPUT_DTYPES), default=defaults.output_dtype
    )
    parser.add_argument(
        "--optimizer", choices=list(_OPTIMIZERS), default=defaults.optimizer
    )
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--use_index_dedup",
        action="store_true",
        help="TorchRec's index dedup",
    )
    parser.add_argument(
        "--output_dist",
        choices=["torchrec", "fused", "auto"],
        default=defaults.output_dist,
        help="TorchRec's dists, or the hierarchical dist (fused, or auto)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["none", *_PIPELINES],
        default=defaults.pipeline,
        help="a plain loop, or one of TorchRec's train pipelines",
    )
    parser.add_argument(
        "--capacity",
        default=defaults.capacity,
        help="output_dist_capacity: an int (both capacities), or measure (from the"
        " batches)",
    )
    parser.add_argument(
        "--capacity_headroom",
        type=float,
        default=defaults.capacity_headroom,
        help="with --capacity measure: the capacities over the needs",
    )
    parser.add_argument("--num_iterations", type=int, default=defaults.num_iterations)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument(
        "--profile_iterations", type=int, default=defaults.profile_iterations
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="the hierarchical dist's kernel timelines and input dist steps (a build"
        " with DYNAMICEMB_HIER_TRACE=1; slows every call)",
    )
    parser.add_argument(
        "--trace_out",
        type=str,
        default=defaults.trace_out,
        help="with --trace: an .npz file for every rank's traced kernels",
    )
    parser.add_argument(
        "--kernel_times",
        action="store_true",
        help="the output dist's kernel times, under a CUDA-only profiler",
    )
    parser.add_argument(
        "--host_stages",
        action="store_true",
        help="plain loop: where the host waits for the GPU",
    )
    parser.add_argument(
        "--nic_counters",
        action="store_true",
        help="the bytes every InfiniBand port sent and received",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="first compare the hierarchical dist with TorchRec's dists",
    )
    parser.add_argument("--json_out", type=str, default=defaults.json_out)
    cfg = BenchConfig(**vars(parser.parse_args(argv)))
    if cfg.num_iterations < 1:
        parser.error("--num_iterations must be at least 1")
    if cfg.warmup < 1:
        parser.error("--warmup must be at least 1")
    if cfg.capacity != "measure" and not cfg.capacity.isdigit():
        parser.error("--capacity is an int or measure")
    return cfg


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
