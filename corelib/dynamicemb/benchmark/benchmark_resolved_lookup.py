# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident sequence lookup and SparseDist training pipeline A/B.

The default A/B uses resolved input_dist and DynamicEmb's optimizer. The control
swaps only the lookup adapter for the previous CUDA gather; it does not reproduce
main's old input distribution. Use --lookup default to measure the imported
checkout's unchanged lookup, including a main baseline. Results exclude data
loading and first-call setup.
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import dynamicemb
import torch
import torch.distributed as dist
import torchrec
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbPoolingMode,
    DynamicEmbTableOptions,
    EmbOptimType,
)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
)
from dynamicemb.shard import DynamicEmbeddingCollectionSharder
from dynamicemb_extensions import gather_embedding
from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.planner import Topology
from torchrec.distributed.train_pipeline import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import ShardingEnv, ShardingType


class PriorGather:
    def __call__(self, unique_embs, batch, offsets, dtype, pooling_weights=None):
        result = torch.empty(
            (batch.reverse_indices.numel(), unique_embs.size(1)),
            dtype=dtype,
            device=unique_embs.device,
        )
        gather_embedding(unique_embs, result, batch.reverse_indices)
        return result


class TrainingModel(torch.nn.Module):
    def __init__(self, collection, names, dim, device):
        super().__init__()
        self.collection = collection
        self.names = names
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(dim, 64, device=device),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1, device=device),
        )

    def forward(self, features):
        result = self.collection(features)
        rows = torch.cat([result[name].values() for name in self.names]) / 1024
        loss = (self.dense(rows) - 0.1).square().mean()
        return loss.unsqueeze(0), loss.detach()


def options(args, **kwargs):
    return DynamicEmbTableOptions(
        dim=args.dim,
        init_capacity=args.capacity,
        max_capacity=args.capacity,
        bucket_capacity=128,
        initializer_args=DynamicEmbInitializerArgs(
            mode=DynamicEmbInitializerMode.DEBUG
        ),
        caching=args.layout == "cache",
        output_dist=args.output_dist,
        output_dist_capacity=args.output_capacity,
        **kwargs,
    )


def build(args, names, device, control):
    hbm_bytes = {
        "hbm": 1024**3,
        "cache": args.capacity * args.dim,
        "host": 0,
        "hybrid": args.capacity * args.dim * 2,
    }[args.layout]
    if args.pipeline == "none":
        table = BatchedDynamicEmbeddingTablesV2(
            table_options=[
                options(
                    args,
                    device_id=device.index,
                    index_type=torch.int64,
                    embedding_dtype=torch.float32,
                    local_hbm_for_values=hbm_bytes,
                )
            ],
            table_names=["table"],
            feature_table_map=[0] * len(names),
            pooling_mode=DynamicEmbPoolingMode.NONE,
            optimizer=EmbOptimType.SGD,
            output_dtype=torch.float32,
            stochastic_rounding=False,
        )
        model = table
    else:
        raw = EmbeddingCollection(
            [
                EmbeddingConfig(
                    name="table",
                    num_embeddings=args.capacity,
                    embedding_dim=args.dim,
                    feature_names=names,
                )
            ],
            device=torch.device("meta"),
        )
        planner = DynamicEmbeddingShardingPlanner(
            topology=Topology(dist.get_world_size(), "cuda"),
            constraints={
                "table": DynamicEmbParameterConstraints(
                    sharding_types=[ShardingType.ROW_WISE.value],
                    use_dynamicemb=True,
                    dynamicemb_options=options(args, global_hbm_for_values=hbm_bytes),
                )
            },
        )
        sharder = DynamicEmbeddingCollectionSharder(
            fused_params=dict(
                optimizer=EmbOptimType.SGD,
                learning_rate=0.01,
                output_dtype=SparseType.FP32,
                stochastic_rounding=False,
            )
        )
        plan = planner.collective_plan(raw, [sharder], dist.group.WORLD)
        collection = sharder.shard(
            raw,
            plan.plan[""],
            ShardingEnv.from_process_group(dist.group.WORLD),
            device,
        )
        table = collection._lookups[0]._emb_modules[0].emb_module
        model = TrainingModel(collection, names, args.dim, device)
    if control:
        table._native_lookup = PriorGather()
    return model


def run(args, feature_num, device):
    names = [f"f{i}" for i in range(feature_num)]
    count = feature_num * args.batch_size * args.hotness
    features = KeyedJaggedTensor(
        keys=names,
        values=torch.arange(count, device=device) % min(1024, args.capacity),
        lengths=torch.full(
            (feature_num * args.batch_size,),
            args.hotness,
            dtype=torch.int64,
            device=device,
        ),
    )
    steps, pipelines, iterators = {}, {}, {}
    arms = (
        [("torchrec", False), ("prior_gather", True)]
        if args.lookup == "both"
        else [("default", False)]
    )
    for name, control in arms:
        torch.manual_seed(1234)
        model = build(args, names, device, control)
        if args.pipeline == "none":
            for _ in range(args.prefetch_depth):
                model.prefetch(features.values(), features.offsets())

            def step(model=model):
                with torch.set_grad_enabled(args.train):
                    if args.prefetch_depth:
                        model.prefetch(features.values(), features.offsets())
                    result = model(features.values(), features.offsets())
                    if args.train:
                        result.mean().backward()
                    return result.detach()

            steps[name] = step
        else:
            pipeline_type = (
                TrainPipelineSparseDist
                if args.pipeline == "sparse"
                else PrefetchTrainPipelineSparseDist
            )
            pipeline = pipeline_type(
                model,
                torch.optim.SGD(model.dense.parameters(), lr=0.01),
                device,
                execute_all_batches=True,
            )
            iterator = iter(
                [features]
                * (args.warmup + args.trials * args.iterations + args.profile_steps + 8)
            )
            pipelines[name], iterators[name] = pipeline, iterator
            steps[name] = lambda p=pipeline, it=iterator: p.progress(it)
    samples = {name: [] for name in steps}
    last = {}
    for name, step in steps.items():
        for _ in range(args.warmup):
            last[name] = step()
    for trial in range(args.trials):
        order = list(steps) if trial % 2 == 0 else list(reversed(steps))
        for name in order:
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.iterations):
                last[name] = steps[name]()
            torch.cuda.synchronize(device)
            elapsed = (time.perf_counter() - start) * 1000 / args.iterations
            if dist.is_initialized():
                value = torch.tensor(elapsed, device=device)
                dist.all_reduce(value, op=dist.ReduceOp.MAX)
                elapsed = value.item()
            samples[name].append(elapsed)
    profiles = {}
    if args.profile_steps:
        from torch.profiler import ProfilerActivity, profile

        for name, step in steps.items():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = torch.cuda.memory_allocated(device)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
            ) as prof:
                for _ in range(args.profile_steps):
                    step()
            trace = (
                Path(args.profile_dir)
                / f"{args.layout}-{args.pipeline}-{feature_num}-{name}-rank{dist.get_rank() if dist.is_initialized() else 0}.json"
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(trace))
            events = prof.events()
            sync_names = {
                "cudaStreamSynchronize",
                "cudaEventSynchronize",
                "cudaDeviceSynchronize",
                "cudaMemcpy",
            }
            profiles[name] = dict(
                steps=args.profile_steps,
                trace=str(trace),
                host_syncs=sum(e.name in sync_names for e in events),
                kernels=sum(
                    e.device_type == torch.autograd.DeviceType.CUDA
                    and not e.name.startswith("Memcpy")
                    for e in events
                ),
                copies=[
                    dict(
                        name=e.key,
                        count=e.count,
                        cpu_us=e.self_cpu_time_total,
                        device_us=e.self_device_time_total,
                    )
                    for e in prof.key_averages()
                    if "memcpy" in e.key.lower()
                ],
                operators=[
                    dict(
                        name=e.key,
                        count=e.count,
                        cpu_us=e.self_cpu_time_total,
                        device_us=e.self_device_time_total,
                        memory_bytes=e.self_device_memory_usage,
                    )
                    for e in prof.key_averages()
                    if e.key.startswith("aten::")
                ],
                peak_increment_bytes=torch.cuda.max_memory_allocated(device)
                - memory_before,
                resident_allocated_bytes=memory_before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            )
    for name, pipeline in pipelines.items():
        try:
            while True:
                last[name] = pipeline.progress(iterators[name])
        except StopIteration:
            pass
    if args.lookup == "both":
        torch.testing.assert_close(last["torchrec"], last["prior_gather"])
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            json.dumps(
                dict(
                    pipeline=args.pipeline,
                    layout=args.layout,
                    prefetch_depth=args.prefetch_depth,
                    train=args.train or args.pipeline != "none",
                    gpu_name=torch.cuda.get_device_name(device),
                    torch_version=torch.__version__,
                    torchrec_version=torchrec.__version__,
                    dynamicemb_source=dynamicemb.__file__,
                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                    features=feature_num,
                    batch_size=args.batch_size,
                    hotness=args.hotness,
                    dim=args.dim,
                    capacity=args.capacity,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    trials=args.trials,
                    samples_ms=samples,
                    median_ms={
                        name: statistics.median(values)
                        for name, values in samples.items()
                    },
                    profiles=profiles,
                )
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookup",
        choices=["both", "default"],
        default="both",
        help="Compare lookup adapters, or time the imported checkout unchanged",
    )
    parser.add_argument(
        "--pipeline", choices=["none", "sparse", "prefetch"], default="none"
    )
    parser.add_argument(
        "--train", action="store_true", help="Include backward in raw-table mode"
    )
    parser.add_argument("--features", nargs="+", type=int, default=[1, 32, 128])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hotness", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--layout", choices=["hbm", "cache", "host", "hybrid"], default="hbm"
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=0,
        help="Outstanding raw-table training batches before the next forward",
    )
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--profile-dir", default="/tmp/dynamicemb-resolved-profile")
    parser.add_argument(
        "--output-dist",
        choices=["torchrec", "auto", "hierarchical"],
        default="torchrec",
    )
    parser.add_argument("--output-capacity", type=int, default=65536)
    args = parser.parse_args()
    if args.prefetch_depth < 0 or args.profile_steps < 0:
        parser.error("prefetch depth and profile steps must be nonnegative")
    if args.prefetch_depth and (args.pipeline != "none" or not args.train):
        parser.error("--prefetch-depth requires --pipeline none --train")
    if (
        min(
            args.features
            + [
                args.batch_size,
                args.hotness,
                args.dim,
                args.capacity,
                args.warmup,
                args.iterations,
                args.trials,
            ]
        )
        <= 0
    ):
        parser.error("sizes, iterations and trials must be positive")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    with tempfile.TemporaryDirectory() as directory:
        if args.pipeline != "none":
            kwargs = (
                {}
                if "RANK" in os.environ
                else dict(
                    init_method=f"file://{directory}/pg",
                    rank=0,
                    world_size=1,
                )
            )
            dist.init_process_group("nccl", **kwargs)
        try:
            for count in args.features:
                run(args, count, device)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
