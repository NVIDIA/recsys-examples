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
import torch
from configs.sid_gin_config_args import DatasetArgs, TrainerArgs
from torch.distributed import get_rank, get_world_size
from torch.utils.data import DataLoader

from .dataset import get_dataset


def _get_train_and_test_data_loader_from_dataset(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )


def get_train_and_test_data_loader(
    dataset_args: DatasetArgs,
    trainer_args: TrainerArgs,
):
    train_dataset = get_dataset(
        dataset_args,
        trainer_args,
        is_train_dataset=True,
        rank=get_rank(),
        world_size=get_world_size(),
        random_seed=trainer_args.seed,
    )
    eval_dataset = get_dataset(
        dataset_args,
        trainer_args,
        is_train_dataset=False,
        rank=get_rank(),
        world_size=get_world_size(),
        random_seed=trainer_args.seed,
    )

    train_loader = _get_train_and_test_data_loader_from_dataset(train_dataset)
    eval_loader = _get_train_and_test_data_loader_from_dataset(eval_dataset)

    return train_loader, eval_loader
