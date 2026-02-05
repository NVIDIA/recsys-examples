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

import pytest
import torch
from dynamicemb_extensions import expand_table_ids_cuda, segmented_unique_cuda


@pytest.fixture
def setup_device():
    assert torch.cuda.is_available()
    device_id = 0
    return torch.device(f"cuda:{device_id}")


# ============================================================================
# Segmented Unique Tests
# ============================================================================


def test_segmented_unique_basic(setup_device):
    """Test basic segmented unique operation with large input (1M keys)."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 10
    num_keys = 1_000_000
    num_unique_per_table = 10000  # Each table has ~10K unique keys

    # Generate keys with controlled uniqueness per table
    keys = torch.randint(
        0, num_unique_per_table, (num_keys,), dtype=torch.int64, device=device
    )

    # Generate ascending table_ids (simulate sorted input)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, table_ids, num_tables, device_sm_count)
    torch.cuda.synchronize()

    # Check table offsets
    table_offsets_cpu = table_offsets.cpu()
    assert table_offsets_cpu[0].item() == 0, "First offset should be 0"

    # Verify offsets are non-decreasing
    for i in range(num_tables):
        assert (
            table_offsets_cpu[i + 1] >= table_offsets_cpu[i]
        ), "Table offsets should be non-decreasing"

    # Check that output indices correctly reconstruct keys
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    # freq_counters should be empty when not requested
    assert (
        freq_counters.numel() == 0
    ), "freq_counters should be empty when not requested"

    total_unique = num_uniques.item()
    assert (
        total_unique == table_offsets_cpu[-1].item()
    ), "num_uniques should match table_offsets[-1]"
    print(
        f"Segmented unique basic test passed: {total_unique} unique from {num_keys} keys, {num_tables} tables"
    )


def test_segmented_unique_overlapping_keys(setup_device):
    """Test segmented unique with same keys in different tables (1M keys)."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 8
    num_keys = 1_000_000
    num_unique_keys = 1000  # Small unique key space to maximize overlaps

    # Same keys appear across all tables - should be counted separately per table
    keys = torch.randint(
        0, num_unique_keys, (num_keys,), dtype=torch.int64, device=device
    )

    # Generate ascending table_ids
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    # Each table should have at most num_unique_keys unique keys
    for i in range(num_tables):
        table_count = table_offsets_cpu[i + 1].item() - table_offsets_cpu[i].item()
        assert (
            table_count <= num_unique_keys
        ), f"Table {i} has more unique keys than possible"

    # Total unique count from table_offsets[num_tables]
    total_unique = table_offsets_cpu[num_tables].item()

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    print(
        f"Segmented unique overlapping keys test passed: {total_unique} unique from {num_keys} keys"
    )


def test_segmented_unique_empty_tables(setup_device):
    """Test segmented unique with some empty tables (1M keys)."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 10
    num_keys = 1_000_000

    # Create table_ids that skip some tables (tables 2, 5, 7 will be empty)
    active_tables = [0, 1, 3, 4, 6, 8, 9]
    table_ids_list = torch.randint(
        0, len(active_tables), (num_keys,), dtype=torch.int32, device=device
    )
    # Map to actual table IDs
    active_tables_tensor = torch.tensor(active_tables, dtype=torch.int32, device=device)
    table_ids = torch.sort(active_tables_tensor[table_ids_list]).values

    keys = torch.randint(0, 10000, (num_keys,), dtype=torch.int64, device=device)

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count
    )
    torch.cuda.synchronize()

    table_offsets_cpu = table_offsets.cpu()

    # Check empty tables have 0 count
    empty_tables = [2, 5, 7]
    for t in empty_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        assert count == 0, f"Table {t} should be empty, got {count} keys"

    # Check active tables have non-zero counts
    for t in active_tables:
        count = table_offsets_cpu[t + 1].item() - table_offsets_cpu[t].item()
        # Active tables should have some keys (unless extremely unlucky with random)
        # Just verify it doesn't exceed max possible
        assert count <= 10000, f"Table {t} has more unique keys than possible"

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed"

    total_unique = num_uniques.item()
    print(
        f"Segmented unique empty tables test passed: {total_unique} unique, {len(empty_tables)} empty tables"
    )


def test_segmented_unique_empty_input(setup_device):
    """Test segmented unique with empty input."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    keys = torch.tensor([], dtype=torch.int64, device=device)
    table_ids = torch.tensor([], dtype=torch.int32, device=device)
    num_tables = 3

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(keys, table_ids, num_tables, device_sm_count)
    torch.cuda.synchronize()

    assert unique_keys.numel() == 0, "Empty input should return empty unique keys"
    assert output_indices.numel() == 0, "Empty input should return empty indices"
    assert num_uniques.item() == 0, "Empty input should have 0 unique keys"
    assert (
        table_offsets.numel() == num_tables + 1
    ), "Table offsets should have num_tables+1 elements"
    assert torch.all(table_offsets == 0), "All offsets should be 0 for empty input"
    assert freq_counters.numel() == 0, "Empty input should return empty freq_counters"

    print("Segmented unique empty input test passed")


def test_segmented_unique_random(setup_device):
    """Test segmented unique with random data (1M keys)."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 16
    num_keys = 1_000_000

    # Generate random keys with high uniqueness
    keys = torch.randint(0, 100000, (num_keys,), dtype=torch.int64, device=device)

    # Generate ascending table_ids (simulate sorted input)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count
    )
    torch.cuda.synchronize()

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed for random test"

    # Verify table offsets are non-decreasing
    table_offsets_cpu = table_offsets.cpu()
    for i in range(num_tables):
        assert (
            table_offsets_cpu[i + 1] >= table_offsets_cpu[i]
        ), "Table offsets should be non-decreasing"

    total_unique = num_uniques.item()
    print(
        f"Segmented unique random test passed: {total_unique} unique from {num_keys} keys, {num_tables} tables"
    )


def test_segmented_unique_stress(setup_device):
    """Stress test with very large input (4M keys, many tables)."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 32
    num_keys = 4_000_000

    # Generate random keys
    keys = torch.randint(0, 500000, (num_keys,), dtype=torch.int64, device=device)

    # Generate ascending table_ids
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    # Warmup
    torch.cuda.synchronize()

    import time

    start = time.perf_counter()

    num_uniques, unique_keys, output_indices, table_offsets, _ = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count
    )
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Verify reconstruction
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(reconstructed, keys_cpu), "Reconstruction failed for stress test"

    total_unique = table_offsets.cpu()[-1].item()
    throughput = num_keys / elapsed / 1e6
    print(
        f"Segmented unique stress test: {total_unique} unique from {num_keys} keys in {elapsed*1000:.2f}ms ({throughput:.2f}M keys/s)"
    )


def test_segmented_unique_with_frequency_counters(setup_device):
    """Test segmented unique with frequency counting enabled."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 4
    num_keys = 100000

    # Generate keys with known frequencies
    keys = torch.randint(0, 1000, (num_keys,), dtype=torch.int64, device=device)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    # Enable frequency counting by passing an empty tensor (numel==0)
    # This enables counting with each key occurrence counted as 1
    empty_freq_tensor = torch.empty(0, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count, empty_freq_tensor
    )
    torch.cuda.synchronize()

    # freq_counters should have values
    total_unique = num_uniques.item()
    assert (
        freq_counters.numel() == num_keys
    ), "freq_counters should have num_keys elements"

    # Sum of frequencies should equal num_keys (each input counted once)
    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == num_keys
    ), f"Sum of frequencies should be {num_keys}, got {freq_sum}"

    # Verify reconstruction still works
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(
        reconstructed, keys_cpu
    ), "Reconstruction failed with freq counters"

    print(
        f"Segmented unique with frequency counters test passed: {total_unique} unique, freq_sum={freq_sum}"
    )


def test_segmented_unique_with_custom_frequencies(setup_device):
    """Test segmented unique with custom input frequencies."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    num_tables = 2
    num_keys = 1000

    # Generate keys with duplicates
    keys = torch.randint(0, 100, (num_keys,), dtype=torch.int64, device=device)
    table_ids = torch.sort(
        torch.randint(0, num_tables, (num_keys,), dtype=torch.int32, device=device)
    ).values

    # Custom frequencies: each key occurrence has frequency 2
    input_frequencies = torch.full((num_keys,), 2, dtype=torch.int64, device=device)

    (
        num_uniques,
        unique_keys,
        output_indices,
        table_offsets,
        freq_counters,
    ) = segmented_unique_cuda(
        keys, table_ids, num_tables, device_sm_count, input_frequencies
    )
    torch.cuda.synchronize()

    total_unique = num_uniques.item()

    # Sum of frequencies should equal 2 * num_keys (each input counted as 2)
    freq_sum = freq_counters[:total_unique].sum().item()
    assert (
        freq_sum == 2 * num_keys
    ), f"Sum of frequencies should be {2 * num_keys}, got {freq_sum}"

    # Verify reconstruction still works
    unique_keys_cpu = unique_keys.cpu()
    output_indices_cpu = output_indices.cpu()
    keys_cpu = keys.cpu()

    reconstructed = unique_keys_cpu[output_indices_cpu]
    assert torch.equal(
        reconstructed, keys_cpu
    ), "Reconstruction failed with custom freq"

    print(f"Segmented unique with custom frequencies test passed: freq_sum={freq_sum}")


def test_expand_table_ids(setup_device):
    """Test expand_table_ids_cuda helper function."""
    device = setup_device
    device_sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    # Simulate a jagged tensor with 2 tables, 2 features per table, batch_size=3
    # Table 0: features 0, 1
    # Table 1: features 2, 3
    # Offsets structure: offsets[feature_idx * batch_size + batch_idx]
    num_tables = 2
    local_batch_size = 3
    features_per_table = 2
    num_features = num_tables * features_per_table  # 4 features

    # Create lengths for each (feature, batch) pair
    # Feature 0: batch lengths [2, 1, 3] = 6 elements
    # Feature 1: batch lengths [1, 2, 1] = 4 elements
    # Feature 2: batch lengths [3, 2, 2] = 7 elements
    # Feature 3: batch lengths [1, 1, 2] = 4 elements
    # Total: 21 elements
    lengths = torch.tensor(
        [
            2,
            1,
            3,  # Feature 0, batches 0-2
            1,
            2,
            1,  # Feature 1, batches 0-2
            3,
            2,
            2,  # Feature 2, batches 0-2
            1,
            1,
            2,  # Feature 3, batches 0-2
        ],
        dtype=torch.int64,
        device=device,
    )

    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    # Table offsets in features: table 0 starts at feature 0, table 1 at feature 2
    table_offsets_in_feature = torch.tensor([0, 2, 4], dtype=torch.int64, device=device)

    num_elements = offsets[-1].item()

    table_ids = expand_table_ids_cuda(
        offsets,
        table_offsets_in_feature,
        num_tables,
        local_batch_size,
        num_elements,
        device_sm_count,
    )
    torch.cuda.synchronize()

    assert (
        table_ids.numel() == num_elements
    ), f"Expected {num_elements} table_ids, got {table_ids.numel()}"
    assert table_ids.dtype == torch.int32, "table_ids should be int32"

    # Verify table_ids are correct
    # Table 0 has features 0 and 1: elements 0-9 (6 + 4 = 10 elements)
    # Table 1 has features 2 and 3: elements 10-20 (7 + 4 = 11 elements)
    table_ids_cpu = table_ids.cpu()

    # Table 0 ends at offset of feature 2 (which is table_offsets_in_feature[1] * batch_size)
    table0_end_offset_idx = 2 * local_batch_size  # feature 2 * batch_size
    table0_end = offsets[table0_end_offset_idx].item()  # = 10

    assert torch.all(
        table_ids_cpu[:table0_end] == 0
    ), f"First table elements should have table_id=0, got {table_ids_cpu[:table0_end]}"
    assert torch.all(
        table_ids_cpu[table0_end:] == 1
    ), f"Second table elements should have table_id=1, got {table_ids_cpu[table0_end:]}"

    print(f"expand_table_ids test passed: {num_elements} elements, {num_tables} tables")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
