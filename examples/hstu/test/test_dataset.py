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
from typing import Optional, Union

import commons.utils.initialize as init
import fbgemm_gpu  # to load permute_2D_sparse_data
import pytest
import torch
from dataset import get_data_loader
from dataset.dummy_dataset import DummySequenceDataset
from dataset.sequence_dataset import get_dataset
from dataset.utils import FeatureConfig, RankingBatch, RetrievalBatch, is_batch_valid
from torch import distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def batch_slice(
    batch: Union[RankingBatch, RetrievalBatch],
    batch_size: int,
    rank: int,
    world_size: int,
) -> Union[RankingBatch, RetrievalBatch]:
    """
    Slice the batch.
    """
    split_size = [batch_size for _ in range(world_size)]
    keys = batch.features.keys()
    values = []
    lengths = []
    for key in keys:
        feature = batch.features[key]
        sliced_lengths = torch.split(feature.lengths(), split_size)[rank]
        segment_start = feature.offsets()[rank * batch_size]
        segment_end = feature.offsets()[(rank + 1) * batch_size]
        # in case of zero-sized segment
        sliced_values = feature.values()[segment_start:segment_end].to(
            feature.values().dtype
        )
        values.extend(sliced_values)
        lengths.extend(sliced_lengths)
    sliced_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=torch.tensor(values, device=batch.features.device()).long(),
        lengths=torch.tensor(lengths, device=batch.features.device()),
    )

    if batch.num_candidates is not None:
        num_candidates = batch.num_candidates[
            rank * batch_size : (rank + 1) * batch_size
        ]
    else:
        num_candidates = None
    batch_kwargs = dict(
        features=sliced_feature,
        feature_to_max_seqlen=batch.feature_to_max_seqlen,
        batch_size=batch_size,
        contextual_feature_names=batch.contextual_feature_names,
        item_feature_name=batch.item_feature_name,
        action_feature_name=batch.action_feature_name,
        max_num_candidates=batch.max_num_candidates,
        num_candidates=num_candidates,
    )
    if isinstance(batch, RankingBatch):
        if batch.num_candidates is not None:
            num_candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                batch.num_candidates
            )
            segment_start = num_candidates_offsets[batch_size * rank].cpu().item()
            segment_end = num_candidates_offsets[batch_size * (rank + 1)].cpu().item()
        else:
            item_seqlen_offsets = batch.features[batch.item_feature_name].offsets()
            segment_start = item_seqlen_offsets[batch_size * rank]
            segment_end = item_seqlen_offsets[batch_size * (rank + 1)]
        return RankingBatch(
            labels=batch.labels[segment_start:segment_end], **batch_kwargs
        )
    else:
        return RetrievalBatch(**batch_kwargs)


def assert_optional_tensor_equal(a: Optional[torch.Tensor], b: Optional[torch.Tensor]):
    if a is not None or b is not None:
        assert torch.allclose(a, b), f"a:{a}, b:{b}"


@pytest.mark.parametrize("batch_size", [128])
@pytest.mark.parametrize(
    "max_seqlen,max_num_candidates", [(100, 10), (100, 100), (100, 0)]
)
@pytest.mark.parametrize(
    "contextual_feature_names", [[], ["user_feature0", "user_feature1"]]
)
@pytest.mark.parametrize("action_feature_name", ["action", None])
@pytest.mark.parametrize("num_tasks", [2, 1])
def test_dummy_dataset(
    batch_size,
    max_seqlen,
    contextual_feature_names,
    action_feature_name,
    max_num_candidates,
    num_tasks,
):
    init.initialize_distributed()
    init.initialize_model_parallel()

    device = torch.cuda.current_device()

    item_feature_name = "item"
    item_and_action_feature_names = (
        [item_feature_name]
        if action_feature_name is None
        else [item_feature_name, action_feature_name]
    )
    feature_configs = [
        FeatureConfig(
            feature_names=item_and_action_feature_names,
            max_item_ids=[1000 for _ in item_and_action_feature_names],
            max_sequence_length=max_seqlen,
            is_jagged=True,
        )
    ]
    for n in contextual_feature_names:
        feature_configs.append(
            FeatureConfig(
                feature_names=[n],
                max_item_ids=[1000],
                max_sequence_length=max_seqlen,
                is_jagged=True,
            )
        )
    dataset = DummySequenceDataset(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        num_generated_batches=10,
        num_tasks=num_tasks,
        num_batches=1000,
    )
    print("start generating")

    dataloader = get_data_loader(dataset=dataset)

    for batch in dataloader:
        batch.to(device)
        is_batch_valid(batch)

    init.destroy_global_state()


@pytest.mark.parametrize(
    "dataset_name",
    ["kuairand-pure", "kuairand-1k", "ml-1m", "ml-20m"],
)
@pytest.mark.parametrize(
    "batch_size_per_rank",
    [128],
)
@pytest.mark.parametrize(
    "max_seqlen,max_num_candidates",
    [
        (1024, 128),
        (1024, 0),
    ],
)
@pytest.mark.parametrize(
    "shuffle",
    [True, False],
)
@pytest.mark.parametrize("random_seed", [0])
@pytest.mark.parametrize(
    "num_tasks",
    [1, 0],
)
def test_sequence_dataset(
    dataset_name,
    batch_size_per_rank,
    max_seqlen,
    max_num_candidates,
    num_tasks,
    shuffle: bool,
    random_seed,
):
    init.initialize_distributed()
    init.initialize_model_parallel()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dataset, _ = get_dataset(
        dataset_name,
        None,
        max_sequence_length=max_seqlen,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        batch_size=batch_size_per_rank,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        shuffle=shuffle,
        random_seed=random_seed,
        nrows=1000,
    )
    reference_dataset, _ = get_dataset(
        dataset_name,
        None,
        max_sequence_length=max_seqlen,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        batch_size=batch_size_per_rank * world_size,
        rank=0,
        world_size=1,
        shuffle=shuffle,
        random_seed=random_seed,
        nrows=1000,
    )
    batch_size_per_rank * world_size
    dataloader = get_data_loader(dataset=dataset)
    dataloader_iter = iter(dataloader)
    ref_dataloader = get_data_loader(dataset=reference_dataset)

    for ref_batch in ref_dataloader:
        is_batch_valid(ref_batch)

        ref_batch = batch_slice(
            ref_batch, batch_size=batch_size_per_rank, rank=rank, world_size=world_size
        )
        batch = next(dataloader_iter)
        is_batch_valid(batch)
        ref_batch_features = ref_batch.features.to_dict()
        batch_features = batch.features.to_dict()
        assert batch_features.keys() == ref_batch_features.keys()
        for key in batch_features.keys():
            assert torch.allclose(
                batch_features[key].values(), ref_batch_features[key].values()
            )
            assert torch.allclose(
                batch_features[key].lengths(), ref_batch_features[key].lengths()
            )
            assert torch.allclose(
                batch_features[key].offsets(), ref_batch_features[key].offsets()
            )
        if isinstance(batch, RankingBatch):
            assert torch.allclose(
                ref_batch.labels, batch.labels
            ), f"labels result: {ref_batch.labels}, {batch.labels}"

    logging_txt = []
    logging_txt.append(f"batch_size_per_rank:{batch_size_per_rank}")
    logging_txt.append(f"max_seqlen:{max_seqlen}")
    logging_txt.append(f"num_tasks:{num_tasks}")
    print(",".join(logging_txt))
    init.destroy_global_state()
