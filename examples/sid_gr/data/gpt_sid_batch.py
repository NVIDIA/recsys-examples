from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


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

    max_item_ids: List[int]  # From embedding args
    max_history_length: int
    is_jagged: bool

    min_item_ids: Optional[List[int]] = None
    feature_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.min_item_ids is None:
            self.min_item_ids = [0] * len(self.max_item_ids)
        else:
            assert len(self.min_item_ids) == len(
                self.max_item_ids
            ), "min_item_ids should have the same length as max_item_ids"
        assert len(self.feature_names) == len(
            self.max_item_ids
        ), "feature_names should have the same length as max_item_ids"


@dataclass
class GPTSIDBatch(Pipelineable):
    features: KeyedJaggedTensor  # contextual features, user history features, candidate features
    batch_size: int
    feature_to_max_seqlen: Dict[str, int]
    # currently we do not have contextual features.
    contextual_feature_names: List[str] = field(default_factory=lambda: [])
    raw_hist_sid_names: List[str] = field(
        default_factory=lambda: []
    )  # all those features compose history_feature_name
    raw_cand_sid_names: List[str] = field(
        default_factory=lambda: []
    )  # all those features compose history_feature_name

    history_feature_name: str = (
        "history_sequence"  # raw sid features are combined into this feature.
    )
    candidate_feature_name: str = (
        "candidate_sequence"  # raw sid features are combined into this feature.
    )
    _num_hierarchies: int = 4
    user_id: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None  # For retrieval, candidates are labels!

    def to(self, device: torch.device, non_blocking: bool = True) -> "GPTSIDBatch":  # type: ignore
        return GPTSIDBatch(
            features=self.features.to(device=device, non_blocking=non_blocking),
            batch_size=self.batch_size,
            feature_to_max_seqlen=self.feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            raw_hist_sid_names=self.raw_hist_sid_names,
            raw_cand_sid_names=self.raw_cand_sid_names,
            labels=self.labels.to(device=device, non_blocking=non_blocking)
            if self.labels is not None
            else None,
            history_feature_name=self.history_feature_name,
            candidate_feature_name=self.candidate_feature_name,
            _num_hierarchies=self._num_hierarchies,
            user_id=self.user_id.to(device=device, non_blocking=non_blocking)
            if self.user_id is not None
            else None,
        )

    def record_stream(self, stream: torch.cuda.Stream):
        self.features.record_stream(stream)
        if self.labels is not None:
            self.labels.record_stream(stream)

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
        for feature_config in feature_configs:
            if feature_config.is_jagged:
                seqlen = torch.randint(
                    feature_config.max_history_length, (batch_size,), device=device
                )
                # the random guarantee the sequence length is at least 1.
                seqlen = seqlen.clamp(min=1)
            else:
                seqlen = torch.full(
                    (batch_size,), feature_config.max_history_length, device=device
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
                    feature_config.max_history_length,
                )

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

        min_item_ids = torch.tensor(min_item_ids, device=device).unsqueeze(0)
        # labels are the candidate sids but starting from 0.
        labels = (
            features[combined_candidate_feature_name].values().view(-1, num_hierarchies)
            - min_item_ids
        )
        return GPTSIDBatch(
            features=features,
            labels=labels,
            batch_size=batch_size,
            feature_to_max_seqlen=feature_to_max_seqlen,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            history_feature_name=combined_history_feature_name,
            candidate_feature_name=combined_candidate_feature_name,
            contextual_feature_names=contextual_feature_names,
            _num_hierarchies=num_hierarchies,
            user_id=None,
        )
