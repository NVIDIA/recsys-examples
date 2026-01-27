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
from typing import List

import commons.utils.initialize as init
import pytest
import torch
from commons.datasets.gpt_sid_batch import FeatureConfig, GPTSIDBatch
from megatron.core import parallel_state
from utils.sid_batch_balancer import SIDGRBalancedBatchShuffler


def generate_random_sid_batches(
    batch_size: int,
    num_hierarchies: int,
    max_history_length: int,
    codebook_sizes: List[int],
    device: torch.device,
    num_batches: int = 10,
) -> List[GPTSIDBatch]:
    """
    Generate random SID-GR batches for testing.

    Args:
        batch_size: Batch size
        num_hierarchies: Number of SID hierarchies
        max_history_length: Maximum history sequence length
        codebook_sizes: Size of each codebook
        device: Device to create batches on
        num_batches: Number of batches to generate

    Returns:
        List of GPTSIDBatch
    """
    assert (
        len(codebook_sizes) == num_hierarchies
    ), "codebook_sizes length should match num_hierarchies"

    batches = []

    # Define feature names
    raw_hist_sid_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_sid_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]

    # Create feature config for history (jagged)
    feature_config_hist = FeatureConfig(
        feature_names=raw_hist_sid_names,
        max_item_ids=codebook_sizes,
        max_sequence_length=max_history_length,
        is_jagged=True,  # Variable length history
        min_item_ids=[0] * num_hierarchies,
    )

    # Create feature config for candidate (fixed length = num_hierarchies)
    feature_config_cand = FeatureConfig(
        feature_names=raw_cand_sid_names,
        max_item_ids=codebook_sizes,
        max_sequence_length=1,  # Each hierarchy has 1 candidate
        is_jagged=False,  # Fixed length
        min_item_ids=[0] * num_hierarchies,
    )

    for _ in range(num_batches):
        batch = GPTSIDBatch.random(
            batch_size=batch_size,
            feature_configs=[feature_config_hist, feature_config_cand],
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            contextual_feature_names=[],
            combined_history_feature_name="history_sequence",
            combined_candidate_feature_name="candidate_sequence",
            device=device,
        )
        batches.append(batch)

    return batches


@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sid_gr_batch_balancer(batch_size, num_heads, head_dim):
    """
    Test SID-GR batch balancer functionality.

    This test verifies:
    1. Batch shuffling preserves total data
    2. Workloads are correctly calculated
    3. Lengths are preserved across shuffle
    4. Labels are correctly shuffled
    """
    init.initialize_distributed()
    init.set_random_seed(1234)
    init.initialize_model_parallel(1)
    device = torch.cuda.current_device()

    # SID-GR specific parameters
    num_hierarchies = 4
    max_history_length = 100
    codebook_sizes = [256, 256, 256, 256]
    num_batches = 10

    # Generate random batches
    batches = generate_random_sid_batches(
        batch_size=batch_size,
        num_hierarchies=num_hierarchies,
        max_history_length=max_history_length,
        codebook_sizes=codebook_sizes,
        device=device,
        num_batches=num_batches,
    )

    # Create batch shuffler
    sid_gr_batch_balancer = SIDGRBalancedBatchShuffler(
        num_heads=num_heads, head_dim=head_dim
    )

    # Test each batch
    for batch in batches:
        # Shuffle the batch
        shuffled_batch, indices, workloads = sid_gr_batch_balancer.shuffle(
            batch,
            pg_group=parallel_state.get_data_parallel_group(),
            return_indices=True,
            return_workloads=True,
        )

        # Test 1: Verify workloads match
        assert torch.equal(
            workloads, sid_gr_batch_balancer.get_workloads(shuffled_batch)
        ), "Shuffled batch workloads should match"

        # Test 2: Verify feature lengths are preserved
        for key in shuffled_batch.features.keys():
            feature = shuffled_batch.features[key]

            lengths_before_shuffle_this_rank = batch.features[key].lengths().sum()
            lengths_after_shuffle_this_rank = feature.lengths().sum()

            # Allreduce to verify global preservation
            reduced_lengths_before_shuffle = lengths_before_shuffle_this_rank.clone()
            reduced_lengths_after_shuffle = lengths_after_shuffle_this_rank.clone()

            torch.distributed.all_reduce(
                reduced_lengths_before_shuffle,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )
            torch.distributed.all_reduce(
                reduced_lengths_after_shuffle,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

            assert torch.equal(
                reduced_lengths_before_shuffle, reduced_lengths_after_shuffle
            ), f"Total lengths for feature '{key}' should be preserved after shuffle"

        # Test 3: Verify history sequence lengths (used for workload calculation)
        history_lengths = shuffled_batch.features[batch.history_feature_name].lengths()

        # History lengths should be reasonable (not all zeros, within bounds)
        assert history_lengths.min() >= 0, "History lengths should be non-negative"
        assert (
            history_lengths.max() <= max_history_length * num_hierarchies
        ), "History lengths should not exceed maximum"

        # Test 4: Verify labels are preserved
        assert isinstance(
            shuffled_batch.labels, torch.Tensor
        ), "Labels should be a tensor"
        if batch.labels is not None and shuffled_batch.labels is not None:
            labels_before_numel = batch.labels.values().numel()
            labels_after_numel = shuffled_batch.labels.values().numel()

            # Allreduce to verify global preservation
            labels_before_numel_tensor = torch.tensor(
                labels_before_numel, device=device
            )
            labels_after_numel_tensor = torch.tensor(labels_after_numel, device=device)

            torch.distributed.all_reduce(
                labels_before_numel_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )
            torch.distributed.all_reduce(
                labels_after_numel_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

            assert torch.equal(
                labels_before_numel_tensor, labels_after_numel_tensor
            ), "Total number of labels should be preserved after shuffle"

        # Test 5: Verify batch_size and actual_batch_size
        assert (
            shuffled_batch.batch_size == batch.batch_size
        ), "Batch size should remain the same"
        assert (
            shuffled_batch.actual_batch_size == batch.actual_batch_size
        ), "Actual batch size should remain the same"

    init.destroy_global_state()


if __name__ == "__main__":
    # Run basic test
    test_sid_gr_batch_balancer(batch_size=64, num_heads=4, head_dim=64)
