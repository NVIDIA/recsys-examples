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
from configs.sid_gin_config_args import DatasetArgs, DatasetType, TrainerArgs

from .disk_sequence_dataset import DiskSequenceDataset
from .gpt_sid_batch import FeatureConfig
from .in_memory_random_dataset import InMemoryRandomDataset


def get_dataset(
    dataset_args: DatasetArgs,
    trainer_args: TrainerArgs,
    is_train_dataset: bool,
    rank: int = 0,
    world_size: int = 1,
    random_seed: int = 1234,
):
    max_history_length = dataset_args.max_history_length
    max_candidate_length = dataset_args.max_candidate_length
    num_hierarchies = dataset_args.num_hierarchies
    codebook_sizes = dataset_args.codebook_sizes
    assert (
        len(codebook_sizes) == num_hierarchies
    ), "codebook_sizes should have the same length as num_hierarchies"
    if dataset_args.dataset_type == DatasetType.InMemoryRandomDataset:
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
        return InMemoryRandomDataset.get_dataset(
            batch_size=trainer_args.train_batch_size
            if is_train_dataset
            else trainer_args.eval_batch_size,
            feature_configs=feature_configs,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            combined_history_feature_name="hist_sids",
            combined_candidate_feature_name="cand_sids",
            contextual_feature_names=[],
            num_generated_batches=1,
            num_batches=trainer_args.max_train_iters
            if is_train_dataset
            else trainer_args.max_eval_iters,
        )
    elif dataset_args.dataset_type == DatasetType.DiskSequenceDataset:
        dataset_args.dataset_name
        return DiskSequenceDataset.get_dataset(
            raw_sequence_data_path=dataset_args.sequence_features_training_data_path
            if is_train_dataset
            else dataset_args.sequence_features_testing_data_path,
            item_id_to_sid_mapping_tensor_path=dataset_args.item_to_sid_mapping_path,
            batch_size=trainer_args.train_batch_size
            if is_train_dataset
            else trainer_args.eval_batch_size,
            max_history_length=max_history_length,  # +1 for the candidate
            max_candidate_length=max_candidate_length
            if is_train_dataset
            else 1,  # only 1 candidate item for eval.
            raw_sequence_feature_name="sequence_data",  # TODO: make it configurable!!!
            num_hierarchies=num_hierarchies,
            codebook_sizes=codebook_sizes,
            output_history_sid_feature_name=dataset_args._history_sid_feature_name,
            output_candidate_sid_feature_name=dataset_args._candidate_sid_feature_name,
            rank=rank,
            world_size=world_size,
            shuffle=dataset_args.shuffle,
            random_seed=random_seed,
            is_train_dataset=is_train_dataset,
            deduplicate_data_across_hierarchy=True,  # deduplicate data because we are using single embedding tables
            deduplicate_label_across_hierarchy=dataset_args.deduplicate_label_across_hierarchy,
        )
    else:
        raise ValueError(f"Invalid dataset type: {dataset_args.dataset_type}")
