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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from commons.sequence_batch.batch import BaseBatch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

try:
    from megatron.core.packed_seq_params import PackedSeqParams
except ImportError:
    # Define fallback PackedSeqParams if megatron is not available
    @dataclass
    class PackedSeqParams:  # type: ignore[no-redef]
        """
        parameters to TEDotProductAttention and fused rope kernels for the
        `thd` (packed) sequence format
        """

        qkv_format: Optional[str] = None
        cu_seqlens_q: Optional[torch.Tensor] = None
        cu_seqlens_kv: Optional[torch.Tensor] = None
        cu_seqlens_q_padded: Optional[torch.Tensor] = None
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None
        max_seqlen_q: Optional[int] = None
        max_seqlen_kv: Optional[int] = None
        local_cp_size: Optional[int] = None
        cp_group: Optional[torch.distributed.ProcessGroup] = None


def to_packed_seq_params(
    cu_seqlens_q,
    max_seqlen_q,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    max_seqlen_kv: Optional[int] = None,
) -> PackedSeqParams:
    cu_seqlens_kv = cu_seqlens_kv or cu_seqlens_q
    max_seqlen_kv = max_seqlen_kv or max_seqlen_q
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        cu_seqlens_kv=cu_seqlens_kv.to(torch.int32),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )


@dataclass
class FeatureConfig:
    """

    A FeatureConfig is a collection of features that share the same seqlen (also the same max_seqlence_length).
    For example, an item id feature is mapped to [sid_0, sid_1, sid_2, sid_3] for 4 hierarchies. Those 4 features share one FeatureConfig.
    Note that FeatureConfig is only used to generate random data.

    Attributes:
      max_item_ids (List[int]): List of maximum item IDs for each feature.
      max_history_length (int): The maximum length of sequences in the dataset.
      is_jagged (bool): Whether the sequences are jagged (i.e., have varying lengths).
      min_item_ids (List[int]): List of minimum item IDs for each feature.
      feature_names (List[str]): List of feature names.
    """

    max_item_ids: List[int]  # From dataset args
    max_sequence_length: int
    is_jagged: bool
    feature_names: List[str]

    min_item_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if len(self.min_item_ids) == 0:
            self.min_item_ids = [0] * len(self.max_item_ids)
        else:
            assert len(self.min_item_ids) == len(
                self.max_item_ids
            ), "min_item_ids should have the same length as max_item_ids"
        assert len(self.feature_names) == len(
            self.max_item_ids
        ), "feature_names should have the same length as max_item_ids"


@dataclass
class GPTSIDBatch(BaseBatch):
    """
    SID-GR batch data structure, inheriting from BaseBatch for batch shuffler support.

    This batch contains:
    - History SID features (user behavior sequence)
    - Candidate SID features (items to predict)
    - Labels (for training)
    """

    # SID-GR specific fields
    raw_hist_sid_names: List[str] = field(
        default_factory=lambda: []
    )  # all those features compose history_feature_name, this is used for random generation
    raw_cand_sid_names: List[str] = field(
        default_factory=lambda: []
    )  # all those features compose candidate_feature_name, this is used for random generation

    history_feature_name: str = (
        "history_sequence"  # raw sid features are combined into this feature.
    )
    candidate_feature_name: str = (
        "candidate_sequence"  # raw sid features are combined into this feature.
    )
    _num_hierarchies: int = 4
    user_id: Optional[torch.Tensor] = None

    def retain_candidate_hierarchies(
        self,
        remained_hierarchies: int,
    ) -> "GPTSIDBatch":
        candidate_jt = self.features[self.candidate_feature_name]
        original_hierarchies = self._num_hierarchies
        assert (
            original_hierarchies >= remained_hierarchies
        ), "remained_hierarchies should be less than or equal to original_hierarchies"
        candidate_lengths = candidate_jt.lengths() - (
            original_hierarchies - remained_hierarchies
        )
        candidate_features = (
            candidate_jt.values()
            .view(-1, original_hierarchies)[:, :remained_hierarchies]
            .reshape(-1)
        )
        labels = None
        if self.labels is not None:
            # labels is a KeyedJaggedTensor, take only the first remained_hierarchies for candidate labels
            cand_labels = self.labels[self.candidate_feature_name]
            cand_labels_values = (
                cand_labels.values()
                .view(-1, original_hierarchies)[:, :remained_hierarchies]
                .reshape(-1)
            )
            cand_labels_lengths = cand_labels.lengths() - (
                original_hierarchies - remained_hierarchies
            )
            # Build labels as KeyedJaggedTensor for candidate only.
            labels = KeyedJaggedTensor.from_lengths_sync(
                keys=[self.candidate_feature_name],
                values=cand_labels_values,
                lengths=cand_labels_lengths,
            )
        history_jt = self.features[self.history_feature_name]
        history_lengths = history_jt.lengths()
        history_features = history_jt.values()

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                self.history_feature_name,
                self.candidate_feature_name,
            ],
            values=torch.cat([history_features, candidate_features]),
            lengths=torch.cat([history_lengths, candidate_lengths]),
        )
        new_batch = GPTSIDBatch(
            features=kjt,
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            raw_hist_sid_names=self.raw_hist_sid_names,
            raw_cand_sid_names=self.raw_cand_sid_names,
            _num_hierarchies=remained_hierarchies,
            history_feature_name=self.history_feature_name,
            candidate_feature_name=self.candidate_feature_name,
            user_id=self.user_id,
            labels=labels,
        )
        return new_batch

    @staticmethod
    def random(
        batch_size: int,
        feature_configs: List[
            FeatureConfig
        ],  # hist and cand share the same feature config.
        raw_hist_sid_names: List[str],
        raw_cand_sid_names: List[str],
        contextual_feature_names: List[str],
        *,
        combined_history_feature_name: str = "history_sequence",
        combined_candidate_feature_name: str = "candidate_sequence",
        device: torch.device,
    ) -> "GPTSIDBatch":
        feature_name_kvl: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        keys = []
        values = []
        lengths = []
        feature_to_max_seqlen = {}
        sid_min_ids = []
        for feature_config in feature_configs:
            if feature_config.is_jagged:
                seqlen = torch.randint(
                    feature_config.max_sequence_length, (batch_size,), device=device
                )
                # the random guarantee the sequence length is at least 1.
                # when candidate
                seqlen = seqlen.clamp(min=1)
            else:
                seqlen = torch.full(
                    (batch_size,), feature_config.max_sequence_length, device=device
                )
            total_seqlen = torch.sum(seqlen).item()
            feature_names = feature_config.feature_names
            max_item_ids = feature_config.max_item_ids
            min_item_ids = feature_config.min_item_ids
            assert (
                len(feature_names) == len(max_item_ids) == len(min_item_ids)
            ), "feature_names, max_item_ids, and min_item_ids should have the same length"
            for i in range(len(feature_names)):
                key = feature_names[i]
                value = torch.randint(
                    min_item_ids[i],
                    max_item_ids[i],
                    (total_seqlen,),
                    device=device,
                )
                feature_name_kvl[key] = (
                    value,
                    seqlen,
                    feature_config.max_sequence_length,
                )
                if key in raw_cand_sid_names:
                    sid_min_ids.append(min_item_ids[i])

        history_sid_kvl = {key: feature_name_kvl.pop(key) for key in raw_hist_sid_names}
        candidate_sid_kvl = {
            key: feature_name_kvl.pop(key) for key in raw_cand_sid_names
        }
        feature_name_kvl.update(
            {
                combined_history_feature_name: (
                    torch.stack([v[0] for v in history_sid_kvl.values()], dim=1).view(
                        -1
                    ),
                    torch.sum(
                        torch.stack([v[1] for v in history_sid_kvl.values()], dim=1),
                        dim=1,
                    ).view(-1),
                    sum(v[2] for v in history_sid_kvl.values()),
                ),
                combined_candidate_feature_name: (
                    torch.stack([v[0] for v in candidate_sid_kvl.values()], dim=1).view(
                        -1
                    ),
                    torch.sum(
                        torch.stack([v[1] for v in candidate_sid_kvl.values()], dim=1),
                        dim=1,
                    ).view(-1),
                    sum(v[2] for v in candidate_sid_kvl.values()),
                ),
            }
        )
        num_hierarchies = len(raw_hist_sid_names)
        assert num_hierarchies == len(
            raw_cand_sid_names
        ), "number of hierarchies should be the same as the number of candidate sid feature names"
        keys = list(feature_name_kvl.keys())
        values = [feature_name_kvl[key][0] for key in keys]
        lengths = [feature_name_kvl[key][1] for key in keys]
        feature_to_max_seqlen = {key: feature_name_kvl[key][2] for key in keys}
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=torch.cat(values).to(device),
            lengths=torch.cat(lengths).to(device).long(),
        )

        sid_min_ids = torch.tensor(sid_min_ids, device=device).unsqueeze(0)
        # labels are the candidate sids but starting from 0 -> we build a KeyedJaggedTensor for labels
        # with keys ["label_0", "label_1", ...] for each hierarchy
        cand_values = (
            features[combined_candidate_feature_name].values().view(-1, num_hierarchies)
            - sid_min_ids
        )
        labels = KeyedJaggedTensor.from_lengths_sync(
            keys=["label"],
            values=cand_values.reshape(-1),
            lengths=features[combined_candidate_feature_name].lengths(),
        )
        return GPTSIDBatch(
            features=features,
            labels=labels,
            batch_size=batch_size,
            actual_batch_size=batch_size,
            feature_to_max_seqlen=feature_to_max_seqlen,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            history_feature_name=combined_history_feature_name,
            candidate_feature_name=combined_candidate_feature_name,
            contextual_feature_names=contextual_feature_names,
            _num_hierarchies=num_hierarchies,
            user_id=None,
        )
