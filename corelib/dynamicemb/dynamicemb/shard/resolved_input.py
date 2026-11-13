# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local value rows and the batch metadata that keeps those rows usable.

This is an internal input_dist -> grouped lookup boundary. Keys used for output
indices remain in SequenceShardingContext.features_before_input_dist. The
original local keys, hash slots and row-lifetime metadata remain in PrefetchState;
only the KJT consumed by lookup carries value rows instead of keys.
"""

from typing import List, Optional

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..batched_dynamicemb_function import PrefetchState
from ..scored_kjt import _shallow_copy_as


class ResolvedEmbeddingInput(KeyedJaggedTensor):
    """A grouped lookup's KJT, with resolved groups retained across its split.

    The only split after input_dist is TorchRec's split by kernel group. Keeping
    those pieces also keeps their explicit PrefetchStates, including for groups
    that share a feature across tables. Unresolved groups are ordinary KJTs.
    """

    @classmethod
    def from_rows(
        cls, features: KeyedJaggedTensor, state: PrefetchState
    ) -> "ResolvedEmbeddingInput":
        assert state.value_row_indices is not None
        rows = state.value_row_indices[state.reverse_indices]
        return _shallow_copy_as(
            cls,
            features,
            _values=rows,
            # A cached JaggedTensor dictionary still refers to the old keys.
            _jt_dict=None,
            prefetch_state=state,
            _feature_splits=[len(features.keys())],
            _groups=None,
        )

    @classmethod
    def from_groups(
        cls,
        features: KeyedJaggedTensor,
        groups: List[KeyedJaggedTensor],
        feature_splits: List[int],
        recompute_lengths: bool = False,
    ) -> "ResolvedEmbeddingInput":
        fields = {}
        if recompute_lengths:
            fields = dict(
                _lengths=torch.cat([group.lengths() for group in groups]),
                _offsets=None,
                _length_per_key=None,
                _offset_per_key=None,
            )
        return _shallow_copy_as(
            cls,
            features,
            _values=torch.cat([group.values() for group in groups]),
            _jt_dict=None,
            prefetch_state=None,
            _feature_splits=feature_splits,
            _groups=groups,
            **fields,
        )

    prefetch_state: Optional[PrefetchState]
    _feature_splits: List[int]
    _groups: Optional[List[KeyedJaggedTensor]]

    def split(self, segments: List[int]) -> List[KeyedJaggedTensor]:
        if segments != self._feature_splits:
            raise ValueError(
                "Resolved embedding input must be split by its kernel groups"
            )
        return self._groups if self._groups is not None else [self]

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        super().record_stream(stream)
        if self.prefetch_state is not None:
            self.prefetch_state.record_stream(stream)
        if self._groups is not None:
            for group in self._groups:
                group.record_stream(stream)
