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
from typing import List, Optional

from commons.datasets.data_loader import get_data_loader
from commons.datasets.gpt_sid_batch import FeatureConfig
from commons.datasets.sid_random_dataset import SIDRandomDataset
from commons.datasets.sid_sequence_dataset import SIDSequenceDataset
from configs.sid_gin_config_args import DatasetArgs, TrainerArgs
from torch.distributed import get_rank, get_world_size


def get_sid_random_dataset(
    max_history_length: int,
    max_candidate_length: int,
    num_hierarchies: int,
    codebook_sizes: List[int],
    train_batch_size: int,
    eval_batch_size: int,
    max_train_iters: int,
    max_eval_iters: int,
    is_train_dataset: bool,
):
    """
    Get SID random dataset for generating random batches of data.

    Args:
        max_history_length: Maximum history length.
        max_candidate_length: Maximum candidate length.
        num_hierarchies: Number of hierarchies.
        codebook_sizes: List of codebook sizes.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        max_train_iters: Maximum training iterations.
        max_eval_iters: Maximum evaluation iterations.
        is_train_dataset: Whether this is a training dataset.

    Returns:
        SIDRandomDataset instance.
    """
    assert (
        len(codebook_sizes) == num_hierarchies
    ), "codebook_sizes should have the same length as num_hierarchies"
    # we need to use feature configs to generate random data
    feature_configs = []
    raw_hist_sid_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_sid_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    # history sid features
    feature_configs.append(
        FeatureConfig(
            feature_names=raw_hist_sid_names,
            max_item_ids=[codebook_sizes[i] for i in range(num_hierarchies)],
            max_sequence_length=max_history_length,
            is_jagged=True,
        )
    )
    # candidate sid features
    feature_configs.append(
        FeatureConfig(
            feature_names=raw_cand_sid_names,
            max_item_ids=[codebook_sizes[i] for i in range(num_hierarchies)],
            max_sequence_length=max_candidate_length,
            is_jagged=True,
        )
    )
    # no contextual
    return SIDRandomDataset.get_dataset(
        batch_size=train_batch_size if is_train_dataset else eval_batch_size,
        feature_configs=feature_configs,
        raw_hist_sid_names=raw_hist_sid_names,
        raw_cand_sid_names=raw_cand_sid_names,
        combined_history_feature_name="hist_sids",
        combined_candidate_feature_name="cand_sids",
        contextual_feature_names=[],
        num_generated_batches=1,
        num_batches=max_train_iters if is_train_dataset else max_eval_iters,
    )


def get_sid_sequence_dataset(
    max_history_length: int,
    max_candidate_length: int,
    num_hierarchies: int,
    codebook_sizes: List[int],
    train_batch_size: int,
    eval_batch_size: int,
    is_train_dataset: bool,
    sequence_features_training_data_path: Optional[str],
    sequence_features_testing_data_path: Optional[str],
    item_to_sid_mapping_path: Optional[str],
    shuffle: bool = False,
    deduplicate_label_across_hierarchy: bool = False,
    history_sid_feature_name: str = "hist_sids",
    candidate_sid_feature_name: str = "cand_sids",
    rank: int = 0,
    world_size: int = 1,
    random_seed: int = 1234,
):
    """
    Get SID sequence dataset for loading sequence data from disk.

    Args:
        max_history_length: Maximum history length.
        max_candidate_length: Maximum candidate length.
        num_hierarchies: Number of hierarchies.
        codebook_sizes: List of codebook sizes.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        is_train_dataset: Whether this is a training dataset.
        sequence_features_training_data_path: Path to training data.
        sequence_features_testing_data_path: Path to testing data.
        item_to_sid_mapping_path: Path to item to SID mapping.
        shuffle: Whether to shuffle data.
        deduplicate_label_across_hierarchy: Whether to deduplicate labels across hierarchy.
        history_sid_feature_name: History SID feature name.
        candidate_sid_feature_name: Candidate SID feature name.
        rank: Process rank.
        world_size: Total number of processes.
        random_seed: Random seed.

    Returns:
        SIDSequenceDataset instance.
    """
    assert (
        len(codebook_sizes) == num_hierarchies
    ), "codebook_sizes should have the same length as num_hierarchies"
    return SIDSequenceDataset.get_dataset(
        raw_sequence_data_path=sequence_features_training_data_path
        if is_train_dataset
        else sequence_features_testing_data_path,
        item_id_to_sid_mapping_tensor_path=item_to_sid_mapping_path,
        batch_size=train_batch_size if is_train_dataset else eval_batch_size,
        max_history_length=max_history_length,  # +1 for the candidate
        max_candidate_length=max_candidate_length
        if is_train_dataset
        else 1,  # only 1 candidate item for eval.
        raw_sequence_feature_name="sequence_data",  # TODO: make it configurable!!!
        num_hierarchies=num_hierarchies,
        codebook_sizes=codebook_sizes,
        output_history_sid_feature_name=history_sid_feature_name,
        output_candidate_sid_feature_name=candidate_sid_feature_name,
        rank=rank,
        world_size=world_size,
        shuffle=shuffle,
        random_seed=random_seed,
        is_train_dataset=is_train_dataset,
        deduplicate_data_across_hierarchy=True,  # deduplicate data because we are using single embedding tables
        deduplicate_label_across_hierarchy=deduplicate_label_across_hierarchy,
    )


def get_sid_dataset(
    dataset_type_str: str,
    max_history_length: int,
    max_candidate_length: int,
    num_hierarchies: int,
    codebook_sizes: List[int],
    train_batch_size: int,
    eval_batch_size: int,
    max_train_iters: int,
    max_eval_iters: int,
    is_train_dataset: bool,
    sequence_features_training_data_path: Optional[str] = None,
    sequence_features_testing_data_path: Optional[str] = None,
    item_to_sid_mapping_path: Optional[str] = None,
    shuffle: bool = False,
    deduplicate_label_across_hierarchy: bool = False,
    history_sid_feature_name: str = "hist_sids",
    candidate_sid_feature_name: str = "cand_sids",
    rank: int = 0,
    world_size: int = 1,
    random_seed: int = 1234,
):
    """
    Get SID dataset based on dataset type string.

    This is a wrapper function that calls either get_sid_random_dataset or
    get_sid_sequence_dataset based on the dataset_type_str parameter.

    Args:
        dataset_type_str: Dataset type string ("sid_random_dataset" or "sid_sequence_dataset").
        All other arguments are passed to the respective dataset creation function.

    Returns:
        Dataset instance (either SIDRandomDataset or SIDSequenceDataset).
    """
    if dataset_type_str == "sid_random_dataset":
        return get_sid_random_dataset(
            max_history_length=max_history_length,
            max_candidate_length=max_candidate_length,
            num_hierarchies=num_hierarchies,
            codebook_sizes=codebook_sizes,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            max_train_iters=max_train_iters,
            max_eval_iters=max_eval_iters,
            is_train_dataset=is_train_dataset,
        )
    elif dataset_type_str == "sid_sequence_dataset":
        return get_sid_sequence_dataset(
            max_history_length=max_history_length,
            max_candidate_length=max_candidate_length,
            num_hierarchies=num_hierarchies,
            codebook_sizes=codebook_sizes,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            is_train_dataset=is_train_dataset,
            sequence_features_training_data_path=sequence_features_training_data_path,
            sequence_features_testing_data_path=sequence_features_testing_data_path,
            item_to_sid_mapping_path=item_to_sid_mapping_path,
            shuffle=shuffle,
            deduplicate_label_across_hierarchy=deduplicate_label_across_hierarchy,
            history_sid_feature_name=history_sid_feature_name,
            candidate_sid_feature_name=candidate_sid_feature_name,
            rank=rank,
            world_size=world_size,
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type_str}")


def get_train_and_test_data_loader(
    dataset_args: DatasetArgs,
    trainer_args: TrainerArgs,
):
    train_dataset = get_sid_dataset(
        dataset_type_str=dataset_args.dataset_type_str,
        max_history_length=dataset_args.max_history_length,
        max_candidate_length=dataset_args.max_candidate_length,
        num_hierarchies=dataset_args.num_hierarchies,
        codebook_sizes=dataset_args.codebook_sizes,
        train_batch_size=trainer_args.train_batch_size,
        eval_batch_size=trainer_args.eval_batch_size,
        max_train_iters=trainer_args.max_train_iters,
        max_eval_iters=trainer_args.max_eval_iters,
        is_train_dataset=True,
        sequence_features_training_data_path=dataset_args.sequence_features_training_data_path,
        sequence_features_testing_data_path=dataset_args.sequence_features_testing_data_path,
        item_to_sid_mapping_path=dataset_args.item_to_sid_mapping_path,
        shuffle=dataset_args.shuffle,
        deduplicate_label_across_hierarchy=dataset_args.deduplicate_label_across_hierarchy,
        history_sid_feature_name=dataset_args._history_sid_feature_name,
        candidate_sid_feature_name=dataset_args._candidate_sid_feature_name,
        rank=get_rank(),
        world_size=get_world_size(),
        random_seed=trainer_args.seed,
    )
    eval_dataset = get_sid_dataset(
        dataset_type_str=dataset_args.dataset_type_str,
        max_history_length=dataset_args.max_history_length,
        max_candidate_length=dataset_args.max_candidate_length,
        num_hierarchies=dataset_args.num_hierarchies,
        codebook_sizes=dataset_args.codebook_sizes,
        train_batch_size=trainer_args.train_batch_size,
        eval_batch_size=trainer_args.eval_batch_size,
        max_train_iters=trainer_args.max_train_iters,
        max_eval_iters=trainer_args.max_eval_iters,
        is_train_dataset=False,
        sequence_features_training_data_path=dataset_args.sequence_features_training_data_path,
        sequence_features_testing_data_path=dataset_args.sequence_features_testing_data_path,
        item_to_sid_mapping_path=dataset_args.item_to_sid_mapping_path,
        shuffle=dataset_args.shuffle,
        deduplicate_label_across_hierarchy=dataset_args.deduplicate_label_across_hierarchy,
        history_sid_feature_name=dataset_args._history_sid_feature_name,
        candidate_sid_feature_name=dataset_args._candidate_sid_feature_name,
        rank=get_rank(),
        world_size=get_world_size(),
        random_seed=trainer_args.seed,
    )

    train_loader = get_data_loader(train_dataset)
    eval_loader = get_data_loader(eval_dataset)

    return train_loader, eval_loader
