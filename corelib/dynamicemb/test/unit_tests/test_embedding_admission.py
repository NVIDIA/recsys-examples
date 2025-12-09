# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import click
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.embedding_admission import FrequencyAdmissionStrategy
from dynamicemb.key_value_table import batched_export_keys_values
from dynamicemb.types import DynamicEmbInitializerArgs

# from dynamicemb.admission_strategy import FrequencyAdmissionStrategy
from test_embedding_dump_load import (
    create_model,
    get_optimizer_kwargs,
    get_score_strategy,
    idx_to_name,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def generate_deterministic_sparse_features_with_frequency_tracking(
    num_embedding_collections: int,
    num_embeddings: List[int],
    multi_hot_sizes: List[int],
    rank: int,
    world_size: int,
    batch_size: int,
    num_iterations: int,
    seed: int = 42,
    caching: bool = False,
) -> Tuple[List[KeyedJaggedTensor], Dict[str, Dict[int, int]]]:
    """
    Generate deterministic sparse features and track frequency for each embedding table.

    Args:
        caching: If True, generate more unique keys to trigger cache eviction

    Returns:
        kjts: List of KeyedJaggedTensor for each iteration
        table_frequency_counters: Dict mapping table_name -> {key: frequency}
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size_per_rank = batch_size // world_size
    kjts = []
    table_frequency_counters = {}

    # Initialize frequency counters for each table
    for embedding_collection_id in range(num_embedding_collections):
        for embedding_id, num_embedding in enumerate(num_embeddings):
            _, embedding_name = idx_to_name(embedding_collection_id, embedding_id)
            table_frequency_counters[embedding_name] = defaultdict(int)

    for iteration in range(num_iterations):
        cur_indices = []
        cur_lengths = []
        keys = []

        for embedding_collection_id in range(num_embedding_collections):
            for embedding_id, num_embedding in enumerate(num_embeddings):
                feature_name, embedding_name = idx_to_name(
                    embedding_collection_id, embedding_id
                )

                for sample_id in range(batch_size):
                    hotness = random.randint(
                        1, multi_hot_sizes[embedding_collection_id]
                    )
                    if caching:
                        # In caching mode, use wider range to trigger eviction
                        max_key = int(num_embedding * 0.8) - 1
                    else:
                        # In storage-only mode, limit to 100 keys for more duplicates
                        max_key = min(num_embedding - 1, 100)
                    indices = [random.randint(0, max_key) for _ in range(hotness)]
                    # Track frequency for all generated indices
                    for idx in indices:
                        table_frequency_counters[embedding_name][idx] += 1

                    if sample_id // batch_size_per_rank == rank:
                        cur_indices.extend(indices)
                        cur_lengths.append(hotness)

                keys.append(feature_name)

        kjts.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(cur_indices, dtype=torch.int64).cuda(),
                lengths=torch.tensor(cur_lengths, dtype=torch.int64).cuda(),
            )
        )

    # Convert defaultdicts to regular dicts
    for table_name in table_frequency_counters:
        table_frequency_counters[table_name] = dict(
            table_frequency_counters[table_name]
        )

    return kjts, table_frequency_counters


def get_table_keys(
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Set[int]]:
    """
    Get keys from dynamic embedding tables directly (without disk I/O).

    Returns:
        Dict[table_name, Set[key]]: Keys stored in each table
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    batch_size = 65536

    all_table_keys = {}

    for _, collection_name, sharded_module in find_sharded_modules(model, ""):
        dynamic_emb_modules = get_dynamic_emb_module(sharded_module)

        for dynamic_emb_module in dynamic_emb_modules:
            dynamic_emb_module.flush()

            for table_name, table in zip(
                dynamic_emb_module.table_names, dynamic_emb_module.tables
            ):
                if table_names is not None and table_name not in set(
                    table_names[collection_name]
                ):
                    continue

                table_keys = set()

                for keys, _, _, _ in batched_export_keys_values(
                    table.table, device, batch_size
                ):
                    for key in keys:
                        table_keys.add(int(key))

                all_table_keys[table_name] = table_keys

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return all_table_keys


def validate_admission_keys(
    expected_frequencies: Dict[str, Dict[int, int]],
    actual_keys: Dict[str, Set[int]],
    threshold: int,
):
    """
    Validate that only keys with frequency >= threshold are stored in tables.

    In multi-GPU scenarios, keys are sharded across GPUs. Each rank only validates
    keys that are actually present in its local tables (i.e., keys that were sharded to it).
    This is similar to how LFU score validation works.

    Args:
        expected_frequencies: Dict mapping table_name -> {key: frequency}
        actual_keys: Dict mapping table_name -> Set[key]
        threshold: Admission threshold
    """
    for table_name in expected_frequencies:
        if table_name not in actual_keys:
            raise AssertionError(f"Table {table_name} missing from actual keys")

        expected = expected_frequencies[table_name]
        actual = actual_keys[table_name]

        for key in actual:
            if key not in expected:
                raise AssertionError(
                    f"Table {table_name}, Key {key}: "
                    f"Found in table but not in expected frequencies (unexpected key)"
                )

            if expected[key] < threshold:
                raise AssertionError(
                    f"Table {table_name}, Key {key}: "
                    f"Admitted with frequency={expected[key]} < threshold={threshold} "
                    f"(should have been rejected)"
                )

        # Count expected admitted/rejected for reporting
        expected_admitted_count = sum(
            1 for freq in expected.values() if freq >= threshold
        )
        expected_rejected_count = sum(
            1 for freq in expected.values() if freq < threshold
        )
        actual_admitted_count = len(actual)

        print(
            f"✓ Table {table_name}: "
            f"{actual_admitted_count} keys admitted on this rank "
            f"(global: {expected_admitted_count} expected admitted, "
            f"{expected_rejected_count} expected rejected), "
            f"threshold={threshold}"
        )


@click.command()
@click.option("--num-embedding-collections", type=int, default=1)
@click.option("--num-embeddings", type=str, default="1000")
@click.option("--multi-hot-sizes", type=str, default="3")
@click.option("--embedding-dim", type=int, default=16)
@click.option(
    "--optimizer-type",
    type=click.Choice(["sgd", "adam", "adagrad", "rowwise_adagrad"]),
    default="sgd",
)
@click.option("--batch-size", type=int, default=16)
@click.option("--num-iterations", type=int, default=3)
@click.option("--threshold", type=int, default=5, help="Admission frequency threshold")
@click.option("--caching", is_flag=True, help="Enable cache + storage architecture")
@click.option(
    "--cache-capacity-ratio",
    type=float,
    default=0.5,
    help="Cache capacity as ratio of storage capacity (only used when --caching is enabled)",
)
@click.option(
    "--score-strategy",
    type=click.Choice(["timestamp", "lfu", "step"]),
    default="timestamp",
    help="Score strategy",
)
def test_admission_strategy_validation(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    batch_size: int,
    num_iterations: int,
    threshold: int,
    caching: bool,
    cache_capacity_ratio: float,
    score_strategy: str,
):
    """Test admission strategy correctness by comparing with naive frequency counting.

    This test validates that only keys with frequency >= threshold are stored in tables.
    It supports two modes:
    - Storage-only (default): Tests admission in storage directly
    - Cache + Storage (--caching): Tests admission through cache to storage
    """

    num_embeddings = [int(v) for v in num_embeddings.split(",")]
    multi_hot_sizes = [int(v) for v in multi_hot_sizes.split(",")]
    use_index_dedup = True

    if not caching:
        for num_embedding, multi_hot_size in zip(num_embeddings, multi_hot_sizes):
            if batch_size * num_iterations * multi_hot_size > num_embedding:
                raise ValueError(
                    "batch_size * num_iterations * multi_hot_size > num_embedding, "
                    "this may lead to eviction of dynamicemb and cause test fail"
                )

    print(f"Configuration:")
    print(f"  - Embedding collections: {num_embedding_collections}")
    print(f"  - Num embeddings: {num_embeddings}")
    print(f"  - Multi-hot sizes: {multi_hot_sizes}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Admission threshold: {threshold}")
    print(f"  - Use index dedup: {use_index_dedup}")
    print(f"  - Score strategy: {score_strategy}")
    if caching:
        print(f"  - Caching: ENABLED ✓")
        print(f"  - Cache capacity ratio: {cache_capacity_ratio}")
    else:
        print(f"  - Caching: DISABLED")

    # Create admission strategy
    admission_strategy = FrequencyAdmissionStrategy(
        threshold=threshold,
        initializer_args=DynamicEmbInitializerArgs(
            value=0.0,
        ),
    )

    # Create model with admission strategy
    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
        score_strategy=get_score_strategy(
            score_strategy
        ),  # Use timestamp for admission
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio if caching else 0.1,
        admit_strategy=admission_strategy,  # Pass admission strategy
    )

    # Generate features with frequency tracking
    (
        kjts,
        expected_frequencies,
    ) = generate_deterministic_sparse_features_with_frequency_tracking(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        multi_hot_sizes=multi_hot_sizes,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        batch_size=batch_size,
        num_iterations=num_iterations,
        caching=caching,
    )

    # Run forward passes to trigger admission logic
    if caching:
        print(
            f"\nRunning {num_iterations} iterations with cache and admission enabled..."
        )
    else:
        print(f"\nRunning {num_iterations} iterations with admission enabled...")

    for iteration, kjt in enumerate(kjts):
        ret = model(kjt)
        torch.cuda.synchronize()
        loss = ret.sum() * dist.get_world_size()
        loss.backward()
        torch.cuda.synchronize()

    # Extract actual keys stored in tables
    actual_keys = get_table_keys(model)

    torch.cuda.synchronize()

    # Validate admission logic
    print(f"\nValidating admission with threshold={threshold}...")
    validate_admission_keys(expected_frequencies, actual_keys, threshold)

    print(f"\n✓ Admission strategy test passed!")


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    test_admission_strategy_validation()
    dist.barrier()
    dist.destroy_process_group()
