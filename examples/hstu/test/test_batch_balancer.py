from typing import Dict

import commons.utils.initialize as init
import pytest
import torch
from commons.datasets.hstu_batch import FeatureConfig
from hstu.utils.hstu_batch_balancer import HASTUBalancedBatchShuffler
from megatron.core import parallel_state
from test_utils import generate_random_batches
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_hstu_batch_balancer(batch_size, num_heads, head_dim):
    init.initialize_distributed()
    init.set_random_seed(1234)
    init.initialize_model_parallel(1)
    device = torch.cuda.current_device()

    item_feature_name = "Item"
    action_feature_name = "Action"
    contextual_feature_names = ["C"]
    max_num_candidates = 0
    num_batches = 10
    replicate_batches = False
    feature_configs = [
        FeatureConfig(
            feature_names=["Item", "Action"],
            max_item_ids=[
                1,
                10000,
            ],  # halve the max ids to `minimize` eviction
            max_sequence_length=100,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=["C"],
            max_item_ids=[1000],
            max_sequence_length=2,
            is_jagged=False,
        ),
    ]
    # the random always generate KJT labels
    history_batches = generate_random_batches(
        task_type="ranking",
        num_tasks=1,
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        device=device,
        num_batches=num_batches,
        replicate_batches=replicate_batches,
    )
    hstu_batch_balancer = HASTUBalancedBatchShuffler(num_heads, head_dim)
    lengths_item_action: Dict[str, torch.Tensor] = {}
    for batch in history_batches:
        shuffled_batch, indices, workloads = hstu_batch_balancer.shuffle(
            batch,
            pg_group=parallel_state.get_data_parallel_group(),
            return_indices=True,
            return_workloads=True,
        )
        assert torch.equal(
            workloads, hstu_batch_balancer.get_workloads(shuffled_batch)
        ), "shuffled_batch workloads should match"
        for key in shuffled_batch.features.keys():
            feature = shuffled_batch.features[key]
            lengths_item_action[key] = feature.lengths()
            lengths_before_shuffle_this_rank = batch.features[key].lengths().sum()
            lengths_after_shuffle_this_rank = feature.lengths().sum()
            # we do allreduce on lengths before and after shuffle
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
            ), "numel should match"
        lengths_to_compare = torch.stack(
            [
                lengths_item_action[item_feature_name],
                lengths_item_action[action_feature_name],
            ],
            dim=0,
        )
        assert torch.all(
            torch.all(lengths_to_compare == lengths_to_compare[0], dim=0)
        ), "lengths should match"
        labels = shuffled_batch.labels

        if isinstance(labels, KeyedJaggedTensor):
            lengths_before_shuffle_this_rank = batch.labels.lengths().sum()
            lengths_after_shuffle_this_rank = labels.lengths().sum()

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
            ), "numel should match"
    init.destroy_global_state()
