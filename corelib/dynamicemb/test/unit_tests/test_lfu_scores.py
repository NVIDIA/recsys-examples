# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import click
import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb import DynamicEmbScoreStrategy
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module
from dynamicemb.key_value_table import batched_export_keys_values
from test_embedding_dump_load import create_model, get_optimizer_kwargs, idx_to_name
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
                    )  # At least 1
                    # Generate indices with smaller range to ensure duplicates
                    if caching:
                        # In caching mode, use wider range to trigger eviction
                        # Use 80% of num_embedding to generate enough unique keys
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


def local_DynamicEmbDump(
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: Optional[bool] = False,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[int, int]]:
    """
    Load scores from dynamic embedding tables directly (without disk I/O).

    Returns:
        Dict[table_name, Dict[key, score]]: Scores organized by table name
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    # find embedding collections
    collections_list: List[Tuple[str, str, nn.Module]] = find_sharded_modules(model, "")

    rank = dist.get_rank(group=pg)
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    batch_size = 65536
    all_table_scores = {}

    for _, current_collection in enumerate(collections_list):
        (
            current_collection_path,
            current_collection_name,
            current_collection_module,
        ) = current_collection
        current_dynamic_emb_module_list = get_dynamic_emb_module(
            current_collection_module
        )

        for _, dynamic_emb_module in enumerate(current_dynamic_emb_module_list):
            dynamic_emb_module.flush()
            current_table_names = dynamic_emb_module.table_names

            # In cache mode, read from storage instead of cache
            # Check if this module has separate cache and storage
            current_tables = dynamic_emb_module.tables

            for dynamic_table_name, dynamic_table in zip(
                current_table_names, current_tables
            ):
                if table_names is not None and dynamic_table_name not in set(
                    table_names[current_collection_name]
                ):
                    continue

                table_key_scores = {}

                for keys, embeddings, opt_states, scores in batched_export_keys_values(
                    dynamic_table.table, device, batch_size
                ):
                    for key, score in zip(keys, scores):
                        table_key_scores[int(key)] = int(score)

                all_table_scores[dynamic_table_name] = table_key_scores

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return all_table_scores


def validate_lfu_scores(
    expected_frequencies: Dict[str, Dict[int, int]],
    actual_scores: Dict[str, Dict[int, int]],
    tolerance: float = 0.0,
):
    """
    Validate that actual scores match expected frequencies for LFU strategy.

    Returns:
        (is_valid, error_message)
    """
    all_errors = []

    for table_name in expected_frequencies:
        if table_name not in actual_scores:
            all_errors.append(f"Table {table_name} missing from actual scores")
            continue

        expected = expected_frequencies[table_name]
        actual = actual_scores[table_name]

        for key in actual.keys():
            exp_freq = expected[key]
            act_score = actual[key]
            assert (
                exp_freq == act_score
            ), f"Table {table_name}, Key {key}: Expected frequency: {exp_freq}, Actual score: {act_score}"


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
@click.option("--tolerance", type=float, default=0.0)
@click.option("--caching", is_flag=True, help="Enable cache + storage architecture")
@click.option(
    "--cache-capacity-ratio",
    type=float,
    default=0.5,
    help="Cache capacity as ratio of storage capacity (only used when --caching is enabled)",
)
def test_lfu_score_validation(
    num_embedding_collections: int,
    num_embeddings: str,
    multi_hot_sizes: str,
    embedding_dim: int,
    optimizer_type: str,
    batch_size: int,
    num_iterations: int,
    tolerance: float,
    caching: bool,
    cache_capacity_ratio: float,
):
    """Test LFU score correctness by comparing with naive frequency counting.

    This test supports two modes:
    - Storage-only (default): Tests LFU scores in storage directly
    - Cache + Storage (--caching): Tests LFU score propagation through cache to storage
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

    # Print test header
    print(f"\n{'='*60}")
    if caching:
        print(f"LFU SCORE VALIDATION TEST WITH CACHE")
    else:
        print(f"LFU SCORE VALIDATION TEST (STORAGE ONLY)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Embedding collections: {num_embedding_collections}")
    print(f"  - Num embeddings: {num_embeddings}")
    print(f"  - Multi-hot sizes: {multi_hot_sizes}")
    print(f"  - Embedding dim: {embedding_dim}")
    print(f"  - Optimizer: {optimizer_type}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Tolerance: {tolerance}")
    print(f"  - Use index dedup: {use_index_dedup}")
    if caching:
        print(f"  - Caching: ENABLED ✓")
        print(f"  - Cache capacity ratio: {cache_capacity_ratio}")
    else:
        print(f"  - Caching: DISABLED")

    # Create model
    optimizer_kwargs = get_optimizer_kwargs(optimizer_type)
    model = create_model(
        num_embedding_collections=num_embedding_collections,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        optimizer_kwargs=optimizer_kwargs,
        score_strategy=DynamicEmbScoreStrategy.LFU,
        use_index_dedup=use_index_dedup,
        caching=caching,
        cache_capacity_ratio=cache_capacity_ratio if caching else 0.1,
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

    # Run forward passes to populate frequency information
    if caching:
        print(f"\nRunning {num_iterations} iterations with cache enabled...")
    else:
        print(f"\nRunning {num_iterations} iterations...")

    for iteration, kjt in enumerate(kjts):
        ret = model(kjt)
        torch.cuda.synchronize()
        loss = ret.sum() * dist.get_world_size()
        loss.backward()
        torch.cuda.synchronize()

    # Extract actual scores
    actual_scores = local_DynamicEmbDump(model, optim=True)

    # debug print
    # print(f"\nExpected frequencies:")
    # for table_name, freqs in expected_frequencies.items():
    #     print(
    #         f"  {table_name}: {len(freqs)} unique keys, total frequency: {sum(freqs.values())}"
    #     )
    #     print(f"    Sample: {dict(list(freqs.items())[:20])}")

    # print(f"\nActual scores{' (from storage after flush)' if caching else ''}:")
    # for table_name, scores in actual_scores.items():
    #     print(f"  {table_name}: {len(scores)} keys")
    #     if len(scores) > 0:
    #         print(f"    Sample: {dict(list(scores.items())[:20])}")

    torch.cuda.synchronize()

    # Validate scores
    validate_lfu_scores(expected_frequencies, actual_scores, tolerance)

    # Report results
    print(f"\n{'='*60}")
    if caching:
        print(f"VALIDATION RESULT (WITH CACHE): PASS")
    else:
        print(f"VALIDATION RESULT: PASS")
    print(f"{'='*60}")


if __name__ == "__main__":
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")
    test_lfu_score_validation()
    dist.barrier()
    dist.destroy_process_group()
