# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build, export, reload, and verify the table-fusion example workflow."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


torch.ops.load_library(
    os.path.join(os.environ["DYNAMICEMB_OPS_LIB_DIR"], "inference_emb_ops.so")
)

# Register DynamicEmb's export metadata implementations after loading the ops.
import dynamicemb.index_range_meta as _index_range_meta  # noqa: E402,F401
import dynamicemb.lookup_meta as _lookup_meta  # noqa: E402,F401
from dynamicemb import (  # noqa: E402
    DeltaDumpResult,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    EmbOptimType,
    EvictedItemMode,
)
from dynamicemb.batched_dynamicemb_tables import (  # noqa: E402
    BatchedDynamicEmbeddingTablesV2,
)
from dynamicemb.exportable_embedding import (  # noqa: E402
    BitConcatConfig,
    EmbeddingCollectionIndexerDirectory,
    EmbeddingCollectionIndexerType,
    EmbeddingCollectionUpdate,
    EmbeddingCollectionUpdateAck,
    EmbeddingCollectionUpdateSubscriber,
    InferenceEmbeddingCollectionConfig,
    export_embedding_collection_aot,
    imported_nve_generation,
    load_embedding_collection_aot,
    register_nve_export_compat,
)
from dynamicemb.exportable_tables import (  # noqa: E402
    apply_inference_embedding_collection,
)
from pynve import nve  # noqa: E402
from pynve.torch.nve_ps import NVEParameterServer  # noqa: E402
from torchrec import DataType  # noqa: E402
from torchrec.modules.embedding_configs import EmbeddingConfig  # noqa: E402
from torchrec.modules.embedding_modules import EmbeddingCollection  # noqa: E402


register_nve_export_compat()

EMBEDDING_DIM = 512
ROWS_PER_TABLE = 256
GPU_CACHE_SIZE = 4 << 20
NVHASHMAP_CAPACITY = 4 * ROWS_PER_TABLE
RANDOM_SEED = 20260903
COORDINATOR_AUTHKEY = b"exportable-embedding-example"

FUSED_IDENTITY_GPU = "to_fused_identity_gpu"
LINEAR_HASH_UVM = "to_linear_hash_uvm"
LINEAR_HASH_NVHASHMAP = "to_linear_hash_hierarchical_nvhashmap"
BIT_CONCAT_NVHASHMAP = "to_bitconcat_hierarchical_nvhashmap"
LINEAR_HASH_REDIS = "to_linear_hash_hierarchical_redis"
BIT_CONCAT_REDIS = "to_bitconcat_hierarchical_redis"

BASE_COLLECTIONS = (
    FUSED_IDENTITY_GPU,
    LINEAR_HASH_UVM,
    LINEAR_HASH_NVHASHMAP,
    BIT_CONCAT_NVHASHMAP,
)
REDIS_COLLECTIONS = (LINEAR_HASH_REDIS, BIT_CONCAT_REDIS)
DUMP_COLLECTIONS = (
    LINEAR_HASH_UVM,
    BIT_CONCAT_NVHASHMAP,
    LINEAR_HASH_REDIS,
    BIT_CONCAT_REDIS,
)
DENSE_COLLECTIONS = (FUSED_IDENTITY_GPU, LINEAR_HASH_NVHASHMAP)


def embedding_configs(collection_name: str) -> list[EmbeddingConfig]:
    return [
        EmbeddingConfig(
            name=f"{collection_name}_table_{table_id}",
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=ROWS_PER_TABLE,
            feature_names=[f"{collection_name}_feature_{table_id}"],
            data_type=DataType.FP32,
        )
        for table_id in range(2)
    ]


class TrainingSparseModel(torch.nn.Module):
    def __init__(self, collection_names: tuple[str, ...], device: torch.device):
        super().__init__()
        self.collections = torch.nn.ModuleDict(
            {
                name: EmbeddingCollection(
                    tables=embedding_configs(name), device=device
                )
                for name in collection_names
            }
        )


class ExportableSparseExample(torch.nn.Module):
    def __init__(
        self,
        converted_model: TrainingSparseModel,
        collection_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.collections = converted_model.collections
        self.collection_names = collection_names

    def forward(
        self,
        base_keys: torch.Tensor,
        base_offsets: torch.Tensor,
        redis_keys: torch.Tensor,
        redis_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = []
        for name in self.collection_names:
            if name in REDIS_COLLECTIONS:
                outputs.append(self.collections[name](redis_keys, redis_offsets))
            else:
                outputs.append(self.collections[name](base_keys, base_offsets))
        return tuple(outputs)


@torch.no_grad()
def fill_stable_random_weights(model: TrainingSparseModel) -> None:
    table_ordinal = 0
    for collection in model.collections.values():
        for table in collection.embedding_configs():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(RANDOM_SEED + table_ordinal)
            values = torch.rand(
                (ROWS_PER_TABLE, EMBEDDING_DIM),
                generator=generator,
                dtype=torch.float32,
            )
            collection.embeddings[table.name].weight.copy_(
                values.to(collection.embeddings[table.name].weight.device)
            )
            table_ordinal += 1


def checkpoint_weights(
    checkpoint_path: Path, collection_names: tuple[str, ...]
) -> dict[str, list[torch.Tensor]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return {
        name: [
            state[f"collections.{name}.embeddings.{table.name}.weight"]
            for table in embedding_configs(name)
        ]
        for name in collection_names
    }


def make_dynamicemb_copy(
    collection_name: str,
    table_weights: list[torch.Tensor],
    device: torch.device,
    *,
    retain_evictions: bool,
) -> BatchedDynamicEmbeddingTablesV2:
    names = [table.name for table in embedding_configs(collection_name)]
    options = [
        DynamicEmbTableOptions(
            embedding_dtype=torch.float32,
            dim=EMBEDDING_DIM,
            init_capacity=ROWS_PER_TABLE,
            max_capacity=ROWS_PER_TABLE,
            bucket_capacity=ROWS_PER_TABLE,
            local_hbm_for_values=4 << 20,
            device_id=device.index,
            training=False,
            caching=False,
            score_strategy=DynamicEmbScoreStrategy.CUSTOMIZED,
            evicted_item_mode=(
                EvictedItemMode.RETAIN_KEY
                if retain_evictions
                else EvictedItemMode.DISCARD
            ),
            index_type=torch.int64,
        )
        for _ in names
    ]
    tables = BatchedDynamicEmbeddingTablesV2(
        table_options=options,
        table_names=names,
        feature_table_map=[0, 1],
        pooling_mode=DynamicEmbPoolingMode.NONE,
        output_dtype=torch.float32,
        device=device,
        optimizer=EmbOptimType.NONE,
    )

    keys = torch.arange(ROWS_PER_TABLE, dtype=torch.int64, device=device).repeat(2)
    table_ids = torch.arange(2, dtype=torch.int64, device=device).repeat_interleave(
        ROWS_PER_TABLE
    )
    values = torch.cat(table_weights).to(device)
    scores = torch.ones(keys.numel(), dtype=torch.uint64, device=device)
    tables.tables.insert(keys, table_ids, values, scores=scores)
    return tables


def dump_training_artifacts(
    model: TrainingSparseModel,
    checkpoint_dir: Path,
    collection_names: tuple[str, ...],
    device: torch.device,
) -> tuple[Path, dict[str, BatchedDynamicEmbeddingTablesV2]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    weights = checkpoint_weights(checkpoint_path, collection_names)

    dump_dir = checkpoint_dir / "dynamicemb"
    dump_dir.mkdir(parents=True, exist_ok=True)
    delta_producers: dict[str, BatchedDynamicEmbeddingTablesV2] = {}
    for name in DUMP_COLLECTIONS:
        if name not in collection_names:
            continue
        tables = make_dynamicemb_copy(
            name,
            weights[name],
            device,
            retain_evictions=name in REDIS_COLLECTIONS,
        )
        tables.dump(str(dump_dir), optim=False, counter=False)
        if name in REDIS_COLLECTIONS:
            delta_producers[name] = tables
    return checkpoint_path, delta_producers


def nvhashmap_parameter_server() -> NVEParameterServer:
    return NVEParameterServer(
        num_embeddings=NVHASHMAP_CAPACITY,
        embedding_size=EMBEDDING_DIM,
        data_type=torch.float32,
        initial_size=NVHASHMAP_CAPACITY,
        ps_type=nve.NVHashMap,
    )


def redis_parameter_server(
    address: str, namespace_id: int
) -> NVEParameterServer:
    parameter_server = NVEParameterServer(
        num_embeddings=0,
        embedding_size=EMBEDDING_DIM,
        data_type=torch.float32,
        ps_type=nve.Redis,
        extra_params={
            "plugin": {"address": address, "single_node": True},
            "table": {
                "num_partitions": 0,
                "string_namespace_id": namespace_id,
            },
        },
    )
    parameter_server.clear()
    return parameter_server


def hierarchical_config(
    indexer_type: EmbeddingCollectionIndexerType,
    parameter_server: NVEParameterServer,
    **kwargs: Any,
) -> InferenceEmbeddingCollectionConfig:
    return InferenceEmbeddingCollectionConfig(
        indexer_type=indexer_type,
        nve_layer_type="hierarchical",
        gpu_cache_size=GPU_CACHE_SIZE,
        parameter_server=parameter_server,
        **kwargs,
    )


def collection_configs(
    *, enable_redis: bool, redis_address: str
) -> dict[str, InferenceEmbeddingCollectionConfig]:
    configs = {
        FUSED_IDENTITY_GPU: InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType.FUSED_IDENTITY,
            nve_layer_type="gpu",
        ),
        LINEAR_HASH_UVM: InferenceEmbeddingCollectionConfig(
            indexer_type=EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
            nve_layer_type="linear_uvm",
            bucket_capacity=ROWS_PER_TABLE,
            gpu_cache_size=GPU_CACHE_SIZE,
        ),
        LINEAR_HASH_NVHASHMAP: hierarchical_config(
            EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
            nvhashmap_parameter_server(),
            bucket_capacity=ROWS_PER_TABLE,
        ),
        BIT_CONCAT_NVHASHMAP: hierarchical_config(
            EmbeddingCollectionIndexerType.BIT_CONCAT,
            nvhashmap_parameter_server(),
            bit_concat=BitConcatConfig(table_id_bits=8, feature_id_bits=55),
        ),
    }
    if enable_redis:
        configs.update(
            {
                LINEAR_HASH_REDIS: hierarchical_config(
                    EmbeddingCollectionIndexerType.LINEAR_HASH_MAP,
                    redis_parameter_server(redis_address, 2026090301),
                    bucket_capacity=ROWS_PER_TABLE,
                ),
                BIT_CONCAT_REDIS: hierarchical_config(
                    EmbeddingCollectionIndexerType.BIT_CONCAT,
                    redis_parameter_server(redis_address, 2026090302),
                    bit_concat=BitConcatConfig(
                        table_id_bits=8, feature_id_bits=55
                    ),
                ),
            }
        )
    return configs


@torch.no_grad()
def construct_inference_model(
    training_model: TrainingSparseModel,
    configs: dict[str, InferenceEmbeddingCollectionConfig],
    checkpoint_path: Path,
    dump_dir: Path,
) -> ExportableSparseExample:
    collection_names = tuple(configs)
    by_table = {
        table.name: configs[name]
        for name in collection_names
        for table in embedding_configs(name)
    }
    apply_inference_embedding_collection(
        training_model,
        embedding_collection_configs=by_table,
        trained_emb_table_sizes={
            table.name: ROWS_PER_TABLE
            for name in collection_names
            for table in embedding_configs(name)
        },
    )

    weights = checkpoint_weights(checkpoint_path, collection_names)
    for name in collection_names:
        collection = training_model.collections[name]
        if name in DENSE_COLLECTIONS:
            collection.load_from_embedding_table(torch.cat(weights[name]))
        else:
            collection.load_from_dynamicemb_file(str(dump_dir))
    return ExportableSparseExample(training_model, collection_names).eval()


def full_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    keys = torch.arange(ROWS_PER_TABLE, dtype=torch.int64, device=device).repeat(2)
    offsets = torch.tensor(
        [0, ROWS_PER_TABLE, 2 * ROWS_PER_TABLE],
        dtype=torch.int64,
        device=device,
    )
    return keys, offsets, keys.clone(), offsets.clone()


def post_update_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([1, 1], dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2], dtype=torch.int64, device=device),
        torch.tensor(
            [1, ROWS_PER_TABLE, 1, ROWS_PER_TABLE],
            dtype=torch.int64,
            device=device,
        ),
        torch.tensor([0, 2, 4], dtype=torch.int64, device=device),
    )


def snapshot_zero_expected(
    weights: dict[str, list[torch.Tensor]], model: ExportableSparseExample
) -> list[torch.Tensor]:
    expected = []
    for name in model.collection_names:
        reference = torch.cat(weights[name]).clone()
        failed = model.collections[name].failed_build_rows
        for table_id, feature_id in failed:
            reference[table_id * ROWS_PER_TABLE + feature_id].zero_()
        expected.append(reference)
    return expected


@torch.no_grad()
def produce_delta(
    collection_name: str,
    producer: BatchedDynamicEmbeddingTablesV2,
) -> tuple[Any, dict[tuple[int, int], torch.Tensor]]:
    ordinal = REDIS_COLLECTIONS.index(collection_name)
    existing_keys = torch.tensor(
        [1, 1],
        dtype=torch.int64,
        device=torch.device("cuda", producer.device_id),
    )
    new_keys = torch.full_like(existing_keys, ROWS_PER_TABLE)
    table_ids = torch.tensor(
        [0, 1], dtype=torch.int64, device=existing_keys.device
    )
    values = torch.stack(
        [
            torch.full(
                (EMBEDDING_DIM,),
                10.0 + ordinal + row / 10.0,
                dtype=torch.float32,
                device=existing_keys.device,
            )
            for row in range(4)
        ]
    )
    # Protect the updated rows before forcing one eviction from each full table.
    producer.tables.insert(
        existing_keys,
        table_ids,
        values[::2],
        scores=torch.full(
            (2,), 3, dtype=torch.uint64, device=existing_keys.device
        ),
    )
    producer.tables.insert(
        new_keys,
        table_ids,
        values[1::2],
        scores=torch.full(
            (2,), 2, dtype=torch.uint64, device=existing_keys.device
        ),
    )
    thresholds = {name: 2 for name in producer.table_names}
    delta = producer.incremental_dump(thresholds)
    keys = torch.stack((existing_keys, new_keys), dim=1).flatten()
    expanded_table_ids = torch.stack((table_ids, table_ids), dim=1).flatten()
    expected = {
        (int(table_id), int(feature_id)): value.detach().cpu()
        for table_id, feature_id, value in zip(
            expanded_table_ids.cpu().tolist(), keys.cpu().tolist(), values
        )
    }
    return delta, expected


def expected_after_update(
    weights: dict[str, list[torch.Tensor]],
    collection_names: tuple[str, ...],
    changed: dict[str, dict[tuple[int, int], torch.Tensor]],
) -> list[torch.Tensor]:
    result = []
    for name in collection_names:
        if name not in REDIS_COLLECTIONS:
            result.append(torch.stack([weights[name][0][1], weights[name][1][1]]))
            continue
        rows = []
        for table_id in range(2):
            rows.append(changed[name][(table_id, 1)])
            rows.append(changed[name][(table_id, ROWS_PER_TABLE)])
        result.append(torch.stack(rows))
    return result


def assert_outputs(
    actual: tuple[torch.Tensor, ...] | list[torch.Tensor],
    expected: list[torch.Tensor],
) -> None:
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.detach().cpu(), reference)


def run_model(
    runtime: Any,
    directory: EmbeddingCollectionIndexerDirectory,
    inputs: tuple[torch.Tensor, ...],
) -> list[torch.Tensor] | tuple[torch.Tensor, ...]:
    with directory.inference_call():
        if hasattr(runtime, "run"):
            return runtime.run(list(inputs))
        return runtime(*inputs)


def linear_eviction_inputs(
    delta: Any, device: torch.device
) -> tuple[torch.Tensor, ...]:
    keys = []
    offsets = [0]
    for evicted in delta.evicted_keys:
        assert evicted is not None and evicted.numel() > 0
        keys.append(evicted.to(device=device, dtype=torch.int64))
        offsets.append(offsets[-1] + evicted.numel())
    return (
        torch.tensor([1, 1], dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2], dtype=torch.int64, device=device),
        torch.cat(keys),
        torch.tensor(offsets, dtype=torch.int64, device=device),
    )


def assert_linear_evictions_miss(
    runtime: Any,
    directory: EmbeddingCollectionIndexerDirectory,
    model: ExportableSparseExample,
    inputs: tuple[torch.Tensor, ...],
) -> None:
    outputs = run_model(runtime, directory, inputs)
    output = outputs[model.collection_names.index(LINEAR_HASH_REDIS)]
    torch.testing.assert_close(
        output.detach().cpu(), torch.zeros_like(output, device="cpu")
    )


def live_layer_map(
    model: ExportableSparseExample,
    directory: EmbeddingCollectionIndexerDirectory,
) -> dict[str, Any]:
    return {
        binding.nve_layer_module_path: model.get_submodule(
            binding.nve_layer_module_path
        )
        for binding in directory.bindings.values()
    }


def start_update_coordinator(
    package_dir: Path,
    update_dir: Path,
    subscriber_ids: set[str],
) -> tuple[subprocess.Popen[Any], Connection]:
    socket_path = update_dir / "coordinator.sock"
    update_dir.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("update_coordinator_main.py")),
        "--package-dir",
        str(package_dir),
        "--update-dir",
        str(update_dir),
        "--socket",
        str(socket_path),
    ]
    for subscriber_id in sorted(subscriber_ids):
        command.extend(("--subscriber", subscriber_id))
    process = subprocess.Popen(command)
    deadline = time.monotonic() + 30
    while True:
        if process.poll() is not None:
            raise RuntimeError("update coordinator exited during startup")
        try:
            connection = Client(
                str(socket_path), family="AF_UNIX", authkey=COORDINATOR_AUTHKEY
            )
            return process, connection
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                process.terminate()
                raise RuntimeError("update coordinator did not become ready")
            time.sleep(0.05)


def coordinator_delta(
    connection: Connection, collection_id: str, delta: DeltaDumpResult
) -> EmbeddingCollectionUpdate:
    buffer = io.BytesIO()
    torch.save(
        {
            "table_names": delta.table_names,
            "keys": [tensor.detach().cpu() for tensor in delta.keys],
            "values": [tensor.detach().cpu() for tensor in delta.values],
            "evicted_keys": [
                None if tensor is None else tensor.detach().cpu()
                for tensor in delta.evicted_keys
            ],
        },
        buffer,
    )
    connection.send(
        {
            "op": "delta",
            "collection_id": collection_id,
            "delta_payload": buffer.getvalue(),
        }
    )
    return EmbeddingCollectionUpdate.from_json(connection.recv())


def coordinator_acknowledge(
    connection: Connection,
    subscriber_id: str,
    acknowledgement: EmbeddingCollectionUpdateAck,
) -> None:
    connection.send(
        {
            "op": "ack",
            "subscriber_id": subscriber_id,
            "ack": acknowledgement.to_json(),
        }
    )


def run_cpp_replayer(
    executable: Path,
    package_dir: Path,
    expected: list[torch.Tensor],
) -> None:
    command = [
        str(executable),
        "--package-dir",
        str(package_dir),
        "--expected-sums",
        ",".join(str(float(value.sum())) for value in expected),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    print(completed.stdout, end="")


def start_incremental_cpp_replayer(
    executable: Path,
    package_dir: Path,
    expected: list[torch.Tensor],
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [
            str(executable),
            "--package-dir",
            str(package_dir),
            "--expected-sums",
            ",".join(str(float(value.sum())) for value in expected),
            "--wait-for-updates",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("C++ replayer stdout is unavailable")
    while True:
        line = process.stdout.readline()
        if not line:
            return_code = process.wait()
            raise RuntimeError(
                f"C++ replayer exited before its ready signal ({return_code})"
            )
        print(line, end="")
        if line.rstrip() == "READY incremental updates":
            return process


def finish_incremental_cpp_replayer(
    process: subprocess.Popen[str],
    updates: list[EmbeddingCollectionUpdate],
    expected: list[torch.Tensor],
) -> list[EmbeddingCollectionUpdateAck]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("C++ replayer pipes are unavailable")
    for update in updates:
        process.stdin.write(update.to_json() + "\n")
    process.stdin.write(
        "RUN " + ",".join(str(float(value.sum())) for value in expected) + "\n"
    )
    process.stdin.flush()

    output = process.stdout.read()
    print(output, end="")
    return_code = process.wait()
    process.stdin.close()
    process.stdout.close()
    if return_code != 0:
        raise RuntimeError(f"C++ replayer failed with exit code {return_code}")

    return [
        EmbeddingCollectionUpdateAck.from_json(line.removeprefix("ACK "))
        for line in output.splitlines()
        if line.startswith("ACK ")
    ]


def run_workflow(args: argparse.Namespace) -> None:
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=0, world_size=1)

    cpp_process: subprocess.Popen[str] | None = None
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        generation = imported_nve_generation()
        enable_redis = generation == (26, 7) and not args.disable_redis_incremental
        names = BASE_COLLECTIONS + (REDIS_COLLECTIONS if enable_redis else ())

        work_dir = Path(args.work_dir).resolve()
        checkpoint_dir = work_dir / "checkpoint"
        package_dir = work_dir / "package"
        update_dir = work_dir / "updates"

        training_model = TrainingSparseModel(names, device)
        fill_stable_random_weights(training_model)
        checkpoint_path, producers = dump_training_artifacts(
            training_model, checkpoint_dir, names, device
        )
        weights = checkpoint_weights(checkpoint_path, names)
        model = construct_inference_model(
            training_model,
            collection_configs(
                enable_redis=enable_redis, redis_address=args.redis_address
            ),
            checkpoint_path,
            checkpoint_dir / "dynamicemb",
        )

        eager_directory = EmbeddingCollectionIndexerDirectory.from_model(model)
        inputs = full_inputs(device)
        expected_zero = snapshot_zero_expected(weights, model)
        with torch.inference_mode():
            assert_outputs(run_model(model, eager_directory, inputs), expected_zero)

        key_dimension = torch.export.Dim("base_key_count", min=2, max=514)
        redis_dimension = torch.export.Dim("redis_key_count", min=2, max=514)
        export_embedding_collection_aot(
            model,
            inputs,
            package_dir,
            dynamic_shapes=(
                {0: key_dimension},
                None,
                {0: redis_dimension},
                None,
            ),
        )
        aoti_runtime, nve_layers, aoti_directory = load_embedding_collection_aot(
            package_dir, device
        )
        with torch.inference_mode():
            assert_outputs(
                run_model(aoti_runtime, aoti_directory, inputs), expected_zero
            )

        cpp_replayer = Path(args.cpp_replayer).resolve() if args.cpp_replayer else None
        if cpp_replayer is not None:
            if enable_redis:
                cpp_process = start_incremental_cpp_replayer(
                    cpp_replayer, package_dir, expected_zero
                )
            else:
                run_cpp_replayer(cpp_replayer, package_dir, expected_zero)

        if not enable_redis:
            print(f"verified {len(names)} embedding-collection combinations")
            return

        subscriber_ids = {"eager", "python_aoti"}
        if cpp_replayer is not None:
            subscriber_ids.add("cpp_aoti")
        eager_subscriber = EmbeddingCollectionUpdateSubscriber(
            indexers=eager_directory,
            nve_layers=live_layer_map(model, eager_directory),
        )
        aoti_subscriber = EmbeddingCollectionUpdateSubscriber(
            indexers=aoti_directory, nve_layers=nve_layers
        )

        coordinator_process, coordinator_connection = start_update_coordinator(
            package_dir, update_dir, subscriber_ids
        )
        try:
            updates = []
            changed = {}
            linear_eviction_query = None
            for name in REDIS_COLLECTIONS:
                delta, changed[name] = produce_delta(name, producers[name])
                if name == LINEAR_HASH_REDIS:
                    linear_eviction_query = linear_eviction_inputs(delta, device)
                updates.append(
                    coordinator_delta(
                        coordinator_connection, f"collections.{name}", delta
                    )
                )

            for update in updates:
                eager_subscriber.apply_incremental_load(update)
                aoti_subscriber.apply_incremental_load(update)

            updated_inputs = post_update_inputs(device)
            expected_one = expected_after_update(weights, names, changed)
            with torch.inference_mode():
                assert_outputs(
                    run_model(model, eager_directory, updated_inputs), expected_one
                )
                assert_outputs(
                    run_model(aoti_runtime, aoti_directory, updated_inputs),
                    expected_one,
                )
                assert linear_eviction_query is not None
                assert_linear_evictions_miss(
                    model, eager_directory, model, linear_eviction_query
                )
                assert_linear_evictions_miss(
                    aoti_runtime, aoti_directory, model, linear_eviction_query
                )

            for update in updates:
                coordinator_acknowledge(
                    coordinator_connection,
                    "eager",
                    eager_subscriber.wait_for_retirement(
                        update.collection_id, update.snapshot_id
                    ),
                )
                coordinator_acknowledge(
                    coordinator_connection,
                    "python_aoti",
                    aoti_subscriber.wait_for_retirement(
                        update.collection_id, update.snapshot_id
                    ),
                )

            if cpp_replayer is not None:
                if cpp_process is None:
                    raise RuntimeError("C++ replayer is not running")
                for acknowledgement in finish_incremental_cpp_replayer(
                    cpp_process, updates, expected_one
                ):
                    coordinator_acknowledge(
                        coordinator_connection, "cpp_aoti", acknowledgement
                    )
        finally:
            try:
                if coordinator_process.poll() is None:
                    try:
                        coordinator_connection.send({"op": "stop"})
                    except (BrokenPipeError, EOFError):
                        pass
                coordinator_connection.close()
            finally:
                coordinator_process.wait(timeout=30)

        print(
            f"verified {len(names)} embedding-collection combinations "
            "and Redis incremental load"
        )
    finally:
        if cpp_process is not None and cpp_process.poll() is None:
            cpp_process.terminate()
            cpp_process.wait(timeout=30)
        if owns_process_group:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--redis-address", default="127.0.0.1:6379")
    parser.add_argument("--disable-redis-incremental", action="store_true")
    parser.add_argument("--cpp-replayer")
    return parser.parse_args()


if __name__ == "__main__":
    run_workflow(parse_args())
