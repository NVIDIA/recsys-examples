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
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import gin
import numpy as np
import torch
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class DistType(str, Enum):
    """Supported random distribution types.

    Only used for benchmark / random data generation (see :class:`RandomDistribution`).
    """

    UNIFORM = "uniform"
    NORMAL = "normal"
    ZIPF = "zipf"
    LOGNORMAL = "lognormal"


@gin.configurable
@dataclass
class RandomDistribution:
    """
    A configurable random distribution for generating non-negative integer (natural number) samples.

    .. note::
        **Benchmark only** — This class is designed for synthetic / random data generation
        in benchmarking and testing scenarios.  It is **not** used when training with real
        datasets (e.g. MovieLens, KuaiRand).

    All samples are natural numbers (>= 0). The default lower bound is 0, and the default
    upper bound is unbounded (None means no upper clamp).

    Supports four distribution types:
      - **uniform**: Samples uniformly from [low, high). ``high`` is required for uniform.
      - **normal**: Samples from N(mean, std), then rounds and clamps to [low, high].
      - **zipf**: Samples from Zipf(alpha) with P(k) ∝ k^{-alpha} (k >= 1),
            shifted to start from ``low``, then clamps to [low, high].
      - **lognormal**: Samples from LogNormal(mean, std), then rounds and clamps to
            [low, high]. Useful for modeling sequence lengths which are typically
            right-skewed (many short sequences, few long ones). For lognormal, the [M/r,M⋅r]
            interval ratio only depends on r, and is independent of M.

    Args:
        dist_type: One of ``DistType.UNIFORM``, ``DistType.NORMAL``, ``DistType.ZIPF``,
                   ``DistType.LOGNORMAL``.
        low: Inclusive lower bound. Default: 0.
        high: Optional exclusive upper bound for uniform, or inclusive upper clamp for
              normal/zipf. Default: None (no upper bound, except uniform which requires it).
        mean: Mean parameter. For normal this is the distribution mean; for lognormal
              this is the **actual (real-space) mean** E[X], *not* the underlying μ.
              Default: None (auto-inferred).
        std: Std parameter. For normal this is the distribution std; for lognormal
             this is the **actual (real-space) standard deviation** SD[X], *not* the
             underlying σ.  The underlying lognormal parameters are derived as:
             ``σ² = ln(1 + (std/mean)²)``, ``μ = ln(mean) - σ²/2``.
             Default: None. For lognormal, defaults to ``mean / 2`` (CV = 0.5,
             ~80% of samples within [0.49M, 1.64M]).
        alpha: Shape parameter for Zipf distribution (must be > 1.0). Default: 1.5.

    Example:
        >>> # Zipf with no upper limit
        >>> dist = RandomDistribution(DistType.ZIPF, alpha=1.2)
        >>> samples = dist.sample(size=128, device=torch.device("cpu"))
        >>> # Normal clamped to [10, 500]
        >>> dist = RandomDistribution(DistType.NORMAL, low=10, high=500, mean=100, std=50)
        >>> # Lognormal: actual mean=512, actual std=256 (80% within ~[256, 1024])
        >>> dist = RandomDistribution(DistType.LOGNORMAL, mean=512, std=256, high=2048)
    """

    dist_type: DistType
    low: int = 0
    high: Optional[int] = None
    # normal distribution parameters
    mean: Optional[float] = None
    std: Optional[float] = None
    # zipf distribution parameter
    alpha: Optional[float] = None

    def sample(
        self,
        size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate ``size`` non-negative integer samples from the configured distribution.

        Args:
            size: Number of samples to generate.
            device: Target device for the returned tensor.

        Returns:
            A 1-D ``torch.Tensor`` of shape ``(size,)`` with dtype ``torch.long``.
        """
        lo = self.low
        hi = self.high  # None means no upper bound

        if self.dist_type == DistType.UNIFORM:
            assert hi is not None, "uniform distribution requires `high` to be set"
            assert hi > lo, f"uniform requires high > low, got [{lo}, {hi})"
            return torch.randint(lo, hi, (size,), device=device)

        elif self.dist_type == DistType.NORMAL:
            assert (
                self.mean is not None and self.std is not None
            ), "normal distribution requires `mean` and `std` to be set"
            assert self.std > 0, f"normal requires std > 0, got {self.std}"
            samples = torch.normal(self.mean, self.std, (size,))
            samples = samples.clamp(min=lo)
            if hi is not None:
                samples = samples.clamp(max=hi - 1)
            return samples.round().long().to(device)

        elif self.dist_type == DistType.LOGNORMAL:
            assert (
                self.mean is not None
            ), "lognormal distribution requires `mean` to be set"
            assert self.mean > 0, f"lognormal requires mean > 0, got {self.mean}"
            # Default CV = 0.5 (std = mean / 2) when std is not specified
            actual_std = self.std if self.std is not None else self.mean / 2.0
            assert actual_std > 0, f"lognormal requires std > 0, got {actual_std}"
            # User specifies actual (real-space) mean M and std S.
            # Convert to underlying normal parameters:
            #   σ² = ln(1 + (S/M)²)
            #   μ  = ln(M) - σ²/2
            cv_sq = (actual_std / self.mean) ** 2
            sigma_sq = math.log(1.0 + cv_sq)
            mu = math.log(self.mean) - sigma_sq / 2.0
            sigma = math.sqrt(sigma_sq)
            samples = torch.distributions.LogNormal(mu, sigma).sample((size,))
            samples = samples.clamp(min=lo)
            if hi is not None:
                samples = samples.clamp(max=hi - 1)
            return samples.round().long().to(device)

        elif self.dist_type == DistType.ZIPF:
            alpha = self.alpha if self.alpha is not None else 1.5
            assert alpha > 1.0, f"zipf requires alpha > 1.0, got {alpha}"
            # numpy zipf generates integers >= 1, shift by (lo - 1) so minimum is lo
            raw = np.random.zipf(alpha, size=size)
            samples = torch.from_numpy(raw).long() + (lo - 1)
            samples = samples.clamp(min=lo)
            if hi is not None:
                samples = samples.clamp(max=hi - 1)
            return samples.to(device)

        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


@dataclass
class FeatureConfig:
    """
    Configuration for features in a dataset.

    .. note::
        **Benchmark / test only** — ``FeatureConfig`` is consumed by
        :meth:`HSTUBatch.random` and :class:`HSTURandomDataset` to generate synthetic
        data.  It is **not** used when training with real datasets (e.g. MovieLens,
        KuaiRand).  In the gin configuration layer the corresponding entry point is
        :class:`FeatureArgs` (inside ``BenchmarkDatasetArgs``).

    A ``FeatureConfig`` groups features that share the same sequence length (and the
    same ``max_sequence_length``).  For example, in HSTU-based models an item feature
    is always paired with a timestamp token — both share one ``FeatureConfig``.

    Attributes:
      feature_names (List[str]): List of names for the features.
      max_item_ids (List[int]): List of maximum item IDs for each feature.
      max_sequence_length (int): The maximum length of sequences in the dataset.
      is_jagged (bool): Whether the sequences are jagged (i.e., have varying lengths).
      seqlen_dist (Optional[RandomDistribution]): Distribution for generating sequence lengths.
          Only used when ``is_jagged=True``. If None, defaults to uniform [0, max_sequence_length).
      value_dists (Optional[Dict[str, RandomDistribution]]): Per-feature distributions for
          generating values, keyed by feature name. Features not present in the dict fall back
          to uniform [0, max_item_id). If None, all features use the default uniform distribution.
    """

    feature_names: List[str]
    max_item_ids: List[int]
    max_sequence_length: int
    is_jagged: bool
    seqlen_dist: Optional[RandomDistribution] = None
    value_dists: Optional[Dict[str, RandomDistribution]] = None


@dataclass
class HSTUBatch(BaseBatch):
    """
    HSTU Batch class for ranking and retrieval tasks.

    Inherits from BaseBatch which provides:
      - features (KeyedJaggedTensor)
      - batch_size (int)
      - feature_to_max_seqlen (Dict[str, int])
      - contextual_feature_names (List[str])
      - labels (Optional[torch.Tensor])
      - to(), pin_memory(), record_stream() methods

    Additional HSTU-specific attributes:
      item_feature_name (str): The name of the item feature.
      action_feature_name (Optional[str]): The name of the action feature, if applicable.
      max_num_candidates (int): The maximum number of candidate items.
      num_candidates (Optional[torch.Tensor]): A tensor containing the number of candidates for each batch element.
    """

    # HSTU-specific fields (BaseBatch fields are inherited)
    item_feature_name: str = "item_id"
    action_feature_name: Optional[str] = None
    max_num_candidates: int = 0
    num_candidates: Optional[torch.Tensor] = None

    def __post_init__(self):
        # Call parent __post_init__ first
        super().__post_init__()

        # HSTU-specific validations
        assert isinstance(
            self.item_feature_name, str
        ), "item_feature_name must be a string"
        assert self.action_feature_name is None or isinstance(
            self.action_feature_name, str
        ), "action_feature_name must be None or a string"
        assert isinstance(
            self.max_num_candidates, int
        ), "max_num_candidates must be an int"

    # to(), pin_memory(), record_stream() are inherited from BaseBatch
    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[FeatureConfig],
        item_feature_name: str,
        contextual_feature_names: List[str] = [],
        action_feature_name: Optional[str] = None,
        max_num_candidates: int = 0,
        num_tasks: Optional[int] = None,  # used for ranking task
        actual_batch_size: Optional[int] = None,  # for incomplete batch
        *,
        device: torch.device,
    ) -> "HSTUBatch":
        """
        Generate a random Batch.

        Args:
            batch_size (int): The target batch size (for padding).
            feature_configs (List[FeatureConfig]): List of configurations for each feature.
            item_feature_name (str): The name of the item feature.
            contextual_feature_names (List[str], optional): List of names for the contextual features. Defaults to [].
            action_feature_name (Optional[str], optional): The name of the action feature. Defaults to None.
            max_num_candidates (int, optional): The maximum number of candidate items. Defaults to 0.
            num_tasks (Optional[int], optional): Number of tasks for ranking. Defaults to None.
            actual_batch_size (Optional[int], optional): Actual number of samples (< batch_size for incomplete batch).
                If None, equals to batch_size. Defaults to None.
            device (torch.device): The device on which the batch will be generated.

        Returns:
            HSTUBatch: The generated random Batch.
        """
        # Use actual_batch_size for data generation, batch_size for padding
        if actual_batch_size is None:
            actual_batch_size = batch_size

        assert (
            actual_batch_size <= batch_size
        ), f"actual_batch_size ({actual_batch_size}) must be <= batch_size ({batch_size})"

        keys = []
        values = []
        lengths = []
        num_candidates = None
        feature_to_max_seqlen = {}
        labels_numel = 0
        history_seqlen = 0

        for fc in feature_configs:
            # Generate data for actual_batch_size samples
            if fc.is_jagged:
                if fc.seqlen_dist is not None:
                    seqlen = fc.seqlen_dist.sample(
                        size=actual_batch_size,
                        device=device,
                    )
                else:
                    seqlen = torch.randint(
                        fc.max_sequence_length, (actual_batch_size,), device=device
                    )
            else:
                seqlen = torch.full(
                    (actual_batch_size,), fc.max_sequence_length, device=device
                )

            if actual_batch_size < batch_size:
                padded_size = batch_size - actual_batch_size
                seqlen = torch.cat(
                    [
                        seqlen,
                        torch.zeros(padded_size, dtype=seqlen.dtype, device=device),
                    ]
                )

            cur_seqlen_sum = torch.sum(seqlen).item()

            for feature_name, max_item_id in zip(fc.feature_names, fc.max_item_ids):
                if feature_name in contextual_feature_names and fc.is_jagged:
                    warnings.warn(f"contextual feature {feature_name} is jagged")
                if fc.value_dists is not None and feature_name in fc.value_dists:
                    value = fc.value_dists[feature_name].sample(
                        size=cur_seqlen_sum,
                        device=device,
                    )
                else:
                    value = torch.randint(max_item_id, (cur_seqlen_sum,), device=device)
                keys.append(feature_name)
                values.append(value)
                lengths.append(seqlen)
                if feature_name == item_feature_name:
                    labels_numel = cur_seqlen_sum
                    history_seqlen = seqlen
                if max_num_candidates > 0 and feature_name == item_feature_name:
                    non_candidates_seqlen = torch.clamp(
                        seqlen - max_num_candidates, min=0
                    )
                    num_candidates = seqlen - non_candidates_seqlen
                    labels_numel = num_candidates.sum()
                feature_to_max_seqlen[feature_name] = fc.max_sequence_length

        if num_tasks is not None:
            label_values = torch.randint(1 << num_tasks, (labels_numel,), device=device)
            # when no candidates, we use the history seqlen as the label length.
            label_lengths = history_seqlen if num_candidates is None else num_candidates
            labels = KeyedJaggedTensor.from_lengths_sync(
                keys=["label"],
                values=label_values,
                lengths=label_lengths,
            )
        else:
            labels = None
        return HSTUBatch(
            features=KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.concat(values).to(device),
                lengths=torch.concat(lengths).to(device).long(),
            ),
            batch_size=batch_size,
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=contextual_feature_names,
            item_feature_name=item_feature_name,
            action_feature_name=action_feature_name,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates.to(device)
            if num_candidates is not None
            else None,
            labels=labels,
            actual_batch_size=actual_batch_size,
        )


def is_batch_valid(
    batch: HSTUBatch,
):
    """
    Validates a batch of data to ensure it meets the necessary criteria.

    Args:
        batch (HSTUBatch): The batch to validate.

    Raises:
        AssertionError: If any of the validation checks fail.
    """
    assert isinstance(batch, HSTUBatch), "batch type should be HSTUBatch"

    assert (
        batch.item_feature_name in batch.features.keys()
    ), "batch must have item_feature_name in features"
    assert (
        batch.features[batch.item_feature_name].lengths().numel() == batch.batch_size
    ), "item_feature shape is not equal to batch_size"

    if batch.action_feature_name is not None:
        assert (
            batch.action_feature_name in batch.features.keys()
        ), "action_feature_name is configured, but not in features"
        assert (
            batch.features[batch.action_feature_name].lengths().numel()
            == batch.batch_size
        ), "action_feature shape is not equal to batch_size"
        assert torch.allclose(
            batch.features[batch.item_feature_name].lengths(),
            batch.features[batch.action_feature_name].lengths(),
        ), "item_feature and action_feature shape should equal"

    if batch.num_candidates is not None:
        assert (
            batch.max_num_candidates > 0
        ), "max_num_candidates should > 0 when num_candidates configured"
        assert torch.all(
            batch.features[batch.item_feature_name].lengths() - batch.num_candidates
            >= 0
        ), "num_candidates is larger than item_feature seqlen"
        expected_label_size = torch.sum(batch.num_candidates).cpu().item()
    else:
        expected_label_size = (
            torch.sum(batch.features[batch.item_feature_name].lengths()).cpu().item()
        )

    # Validate labels if present
    if batch.labels is not None:
        assert isinstance(
            batch.labels, KeyedJaggedTensor
        ), "labels should be a KeyedJaggedTensor"
        batchsize = batch.labels.lengths().numel()
        assert (
            batchsize == batch.batch_size
        ), "label batchsize should be equal to batch_size"

    for contextual_feature_name in batch.contextual_feature_names:
        assert (
            contextual_feature_name in batch.features.keys()
        ), f"contextual_feature {contextual_feature_name} is configured, but not in features"
        assert (
            batch.features[contextual_feature_name].lengths().numel()
            == batch.batch_size
        ), f"contextual_feature {contextual_feature_name} shape is not equal to batch_size"
