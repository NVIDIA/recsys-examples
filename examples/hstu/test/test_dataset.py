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
from typing import Optional

import commons.utils.initialize as init
import fbgemm_gpu  # to load permute_2D_sparse_data
import pytest
import torch
from commons.datasets import get_data_loader
from commons.datasets.hstu_batch import (
    DistType,
    FeatureConfig,
    RandomDistribution,
    is_batch_valid,
)
from commons.datasets.hstu_random_dataset import HSTURandomDataset
from commons.datasets.hstu_sequence_dataset import get_dataset
from test_utils import batch_slice
from torch import distributed as dist


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
def test_hstu_random_dataset(
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
    dataset = HSTURandomDataset(
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


@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("max_seqlen,max_num_candidates", [(200, 10)])
@pytest.mark.parametrize("action_feature_name", ["action"])
@pytest.mark.parametrize("num_tasks", [2])
@pytest.mark.parametrize(
    "seqlen_dist_type", [DistType.UNIFORM, DistType.NORMAL, DistType.ZIPF]
)
@pytest.mark.parametrize(
    "value_dist_type", [DistType.UNIFORM, DistType.NORMAL, DistType.ZIPF]
)
def test_hstu_random_dataset_with_distributions(
    batch_size,
    max_seqlen,
    action_feature_name,
    max_num_candidates,
    num_tasks,
    seqlen_dist_type,
    value_dist_type,
):
    """Test random dataset generation with configurable RandomDistribution for seqlen and values."""
    init.initialize_distributed()
    init.initialize_model_parallel()

    device = torch.cuda.current_device()

    item_feature_name = "item"
    max_item_id = 5000
    seqlen_low = 1
    seqlen_high = max_seqlen
    value_low = 0
    value_high = max_item_id

    # --- build seqlen_dist ---
    if seqlen_dist_type == DistType.UNIFORM:
        seqlen_dist = RandomDistribution(
            dist_type=DistType.UNIFORM, low=seqlen_low, high=seqlen_high
        )
    elif seqlen_dist_type == DistType.NORMAL:
        seqlen_dist = RandomDistribution(
            dist_type=DistType.NORMAL,
            low=seqlen_low,
            high=seqlen_high,
            mean=float(max_seqlen) / 2,
            std=float(max_seqlen) / 4,
        )
    else:  # ZIPF
        seqlen_dist = RandomDistribution(
            dist_type=DistType.ZIPF,
            low=seqlen_low,
            high=seqlen_high,
            alpha=1.5,
        )

    # --- build value_dists (one per feature) ---
    def _make_value_dist(dist_type):
        if dist_type == DistType.UNIFORM:
            return RandomDistribution(
                dist_type=DistType.UNIFORM, low=value_low, high=value_high
            )
        elif dist_type == DistType.NORMAL:
            return RandomDistribution(
                dist_type=DistType.NORMAL,
                low=value_low,
                high=value_high,
                mean=float(value_high) / 2,
                std=float(value_high) / 4,
            )
        else:  # ZIPF
            return RandomDistribution(
                dist_type=DistType.ZIPF,
                low=value_low,
                high=value_high,
                alpha=1.2,
            )

    item_and_action_feature_names = (
        [item_feature_name]
        if action_feature_name is None
        else [item_feature_name, action_feature_name]
    )
    value_dists = {
        name: _make_value_dist(value_dist_type)
        for name in item_and_action_feature_names
    }

    feature_configs = [
        FeatureConfig(
            feature_names=item_and_action_feature_names,
            max_item_ids=[max_item_id for _ in item_and_action_feature_names],
            max_sequence_length=max_seqlen,
            is_jagged=True,
            seqlen_dist=seqlen_dist,
            value_dists=value_dists,
        )
    ]

    dataset = HSTURandomDataset(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=[],
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        num_generated_batches=3,
        num_tasks=num_tasks,
        num_batches=10,
    )

    dataloader = get_data_loader(dataset=dataset)

    for batch in dataloader:
        batch.to(device)
        is_batch_valid(batch)

        # Verify value bounds for every feature in the KJT
        kjt = batch.features
        for key in kjt.keys():
            vals = kjt[key].values()
            if seqlen_dist_type == DistType.ZIPF:
                import pdb

                pdb.set_trace()
            if vals.numel() == 0:
                continue
            assert (
                vals.min().item() >= value_low
            ), f"feature '{key}': min value {vals.min().item()} < low {value_low}"
            # high is exclusive for uniform, inclusive-clamp for normal/zipf,
            # but sample() always clamps to max=high-1, so max value < high.
            assert (
                vals.max().item() < value_high
            ), f"feature '{key}': max value {vals.max().item()} >= high {value_high}"

        # Verify seqlen bounds (lengths per feature should be within [seqlen_low, seqlen_high))
        for key in kjt.keys():
            lengths = kjt[key].lengths()
            non_zero_lengths = lengths[lengths > 0]
            if non_zero_lengths.numel() == 0:
                continue
            assert (
                non_zero_lengths.min().item() >= seqlen_low
            ), f"feature '{key}': min seqlen {non_zero_lengths.min().item()} < low {seqlen_low}"
            assert (
                non_zero_lengths.max().item() < seqlen_high
            ), f"feature '{key}': max seqlen {non_zero_lengths.max().item()} >= high {seqlen_high}"

    init.destroy_global_state()


def _rank_seed(base_seed: int) -> int:
    """Derive the per-rank effective seed, mirroring ``set_random_seed``.

    ::

        seed = base_seed + 100 * pp_rank + 10 * dp_rank
    """
    from megatron.core import parallel_state

    return (
        base_seed
        + 100 * parallel_state.get_pipeline_model_parallel_rank()
        + 10 * parallel_state.get_data_parallel_rank()
    )


def _reset_cpu_seeds(base_seed: int):
    """Reset CPU-side RNG states (python random, numpy, torch CPU).

    Computes the per-rank effective seed from ``base_seed`` using the real
    ``parallel_state`` rank information and resets all CPU RNG backends.
    This mirrors what ``set_random_seed`` does for CPU RNG without touching
    the CUDA RNG tracker (which would raise on duplicate ``add`` calls).
    """
    import random

    import numpy as np

    effective_seed = _rank_seed(base_seed)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)


def _collect_batches(
    base_seed: int,
    feature_configs,
    item_feature_name: str,
    action_feature_name,
    max_num_candidates: int,
    num_tasks: int,
    batch_size: int,
    num_generated_batches: int,
):
    """Reset RNG with the per-rank seed derived from *base_seed*, then create
    ``HSTURandomDataset`` and return its cached batches."""
    _reset_cpu_seeds(base_seed)
    dataset = HSTURandomDataset(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=[],
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        num_generated_batches=num_generated_batches,
        num_tasks=num_tasks,
        num_batches=num_generated_batches,
    )
    return list(iter(dataset))


def _assert_batches_equal(batches_a, batches_b):
    """Assert that two lists of ``HSTUBatch`` are identical element-by-element."""
    assert len(batches_a) == len(
        batches_b
    ), f"batch count mismatch: {len(batches_a)} vs {len(batches_b)}"
    for i, (a, b) in enumerate(zip(batches_a, batches_b)):
        # --- features (KJT) ---
        assert a.features.keys() == b.features.keys(), f"batch[{i}] feature keys differ"
        for key in a.features.keys():
            assert torch.equal(
                a.features[key].values(), b.features[key].values()
            ), f"batch[{i}] feature '{key}' values differ"
            assert torch.equal(
                a.features[key].lengths(), b.features[key].lengths()
            ), f"batch[{i}] feature '{key}' lengths differ"
        # --- labels ---
        if a.labels is not None:
            assert b.labels is not None, f"batch[{i}] labels: one is None"
            assert torch.equal(
                a.labels.values(), b.labels.values()
            ), f"batch[{i}] label values differ"
        else:
            assert b.labels is None, f"batch[{i}] labels: one is not None"
        # --- num_candidates ---
        if a.num_candidates is not None:
            assert b.num_candidates is not None
            assert torch.equal(
                a.num_candidates, b.num_candidates
            ), f"batch[{i}] num_candidates differ"


def _batches_fingerprint(batches) -> torch.Tensor:
    """Compute a deterministic fingerprint (1-D int64 tensor) for a list of batches.

    The fingerprint is built by hashing feature values of every batch so that
    it can be all-gathered across ranks for cross-rank divergence checks.
    """
    parts = []
    for b in batches:
        for key in b.features.keys():
            vals = b.features[key].values()
            if vals.numel() > 0:
                parts.append(vals.to(torch.int64).sum())
                parts.append(torch.tensor(vals.numel(), dtype=torch.int64))
    # Return a single scalar tensor
    return (
        torch.stack(parts).sum().unsqueeze(0)
        if parts
        else torch.zeros(1, dtype=torch.int64)
    )


@pytest.mark.parametrize("seed", [42, 1234])
@pytest.mark.parametrize("batch_size", [64])
@pytest.mark.parametrize("max_seqlen", [200])
@pytest.mark.parametrize("action_feature_name", ["action", None])
@pytest.mark.parametrize("num_tasks", [2])
@pytest.mark.parametrize(
    "seqlen_dist_type", [DistType.UNIFORM, DistType.NORMAL, DistType.ZIPF]
)
@pytest.mark.parametrize("value_dist_type", [DistType.UNIFORM, DistType.ZIPF])
def test_random_dataset_reproducibility(
    seed,
    batch_size,
    max_seqlen,
    action_feature_name,
    num_tasks,
    seqlen_dist_type,
    value_dist_type,
):
    """Verify that HSTURandomDataset is reproducible and rank-differentiated.

    Launch with ``torchrun --nproc_per_node N`` to test with N ranks.
    Each rank derives its own effective seed via::

        effective_seed = base_seed + 100 * pp_rank + 10 * dp_rank

    Checks (all executed on every rank):
      1. **Same rank, same seed → identical batches** (reproducibility).
      2. **Same rank, different seed → different batches** (seed effectiveness).
      3. **Different ranks, same seed → different batches** (rank divergence,
         verified via all-gather of per-rank fingerprints).
    """
    init.initialize_distributed()
    init.initialize_model_parallel()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    item_feature_name = "item"
    max_item_id = 5000
    max_num_candidates = 10
    num_generated_batches = 5

    # --- build seqlen_dist ---
    if seqlen_dist_type == DistType.UNIFORM:
        seqlen_dist = RandomDistribution(
            dist_type=DistType.UNIFORM, low=1, high=max_seqlen
        )
    elif seqlen_dist_type == DistType.NORMAL:
        seqlen_dist = RandomDistribution(
            dist_type=DistType.NORMAL,
            low=1,
            high=max_seqlen,
            mean=float(max_seqlen) / 2,
            std=float(max_seqlen) / 4,
        )
    else:
        seqlen_dist = RandomDistribution(
            dist_type=DistType.ZIPF,
            low=1,
            high=max_seqlen,
            alpha=1.5,
        )

    # --- build value_dists ---
    item_and_action_feature_names = (
        [item_feature_name]
        if action_feature_name is None
        else [item_feature_name, action_feature_name]
    )
    if value_dist_type == DistType.UNIFORM:
        vd = RandomDistribution(dist_type=DistType.UNIFORM, low=0, high=max_item_id)
    else:
        vd = RandomDistribution(
            dist_type=DistType.ZIPF, low=0, high=max_item_id, alpha=1.2
        )
    value_dists = {name: vd for name in item_and_action_feature_names}

    feature_configs = [
        FeatureConfig(
            feature_names=item_and_action_feature_names,
            max_item_ids=[max_item_id for _ in item_and_action_feature_names],
            max_sequence_length=max_seqlen,
            is_jagged=True,
            seqlen_dist=seqlen_dist,
            value_dists=value_dists,
        )
    ]

    common_kwargs = dict(
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        num_tasks=num_tasks,
        batch_size=batch_size,
        num_generated_batches=num_generated_batches,
    )

    # ---- 1. Same rank, same seed → identical batches (reproducibility) ----
    batches_run1 = _collect_batches(base_seed=seed, **common_kwargs)
    batches_run2 = _collect_batches(base_seed=seed, **common_kwargs)
    _assert_batches_equal(batches_run1, batches_run2)

    # ---- 2. Same rank, different seed → different batches ----
    batches_alt = _collect_batches(base_seed=seed + 9999, **common_kwargs)
    any_diff = False
    for a, b in zip(batches_run1, batches_alt):
        for key in a.features.keys():
            if not torch.equal(a.features[key].values(), b.features[key].values()):
                any_diff = True
                break
        if any_diff:
            break
    assert (
        any_diff
    ), f"rank {rank}: different seeds produced identical data — RNG not effective"

    # ---- 3. Cross-rank divergence (only meaningful when world_size > 1) ----
    if world_size > 1:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        local_fp = _batches_fingerprint(batches_run1).to(device)  # shape [1], on CUDA
        gathered = [torch.zeros_like(local_fp) for _ in range(world_size)]
        dist.all_gather(gathered, local_fp)
        fingerprints = torch.cat(gathered).cpu()  # shape [world_size]
        # Every rank should have a unique fingerprint
        assert fingerprints.unique().numel() == world_size, (
            f"Expected {world_size} unique fingerprints across ranks, "
            f"got {fingerprints.unique().numel()}: {fingerprints.tolist()}"
        )

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
        max_history_seqlen=max_seqlen,
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
        max_history_seqlen=max_seqlen,
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
        if batch.labels is not None:
            assert ref_batch.labels is not None, "ref labels should not be None"
            assert torch.allclose(
                ref_batch.labels.values(), batch.labels.values()
            ), f"labels result: {ref_batch.labels.values()}, {batch.labels.values()}"

    logging_txt = []
    logging_txt.append(f"batch_size_per_rank:{batch_size_per_rank}")
    logging_txt.append(f"max_seqlen:{max_seqlen}")
    logging_txt.append(f"num_tasks:{num_tasks}")
    print(",".join(logging_txt))
    init.destroy_global_state()
