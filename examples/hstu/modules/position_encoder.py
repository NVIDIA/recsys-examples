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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

from math import sqrt
from typing import Optional

import torch
from commons.utils.nvtx_op import output_nvtx_hook
from ops.triton_ops.triton_position import (  # type: ignore[attr-defined]
    triton_add_position_embeddings,
    triton_add_timestamp_positional_embeddings,
)
from torch.fx._symbolic_trace import is_fx_tracing


@torch.fx.wrap
def _get_high_inds(
    high_inds: torch.Tensor,
    position_embeddings_weight: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    max_pos_ind = position_embeddings_weight.size(0)
    if num_targets is not None:
        if interleave_targets:
            high_inds = high_inds - num_targets * 2
        else:
            high_inds = high_inds - num_targets
    high_inds = torch.clamp(high_inds, max=max_pos_ind - 1)
    return high_inds


class HSTUPositionalEncoder(torch.nn.Module):
    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        training_dtype: torch.dtype,
        is_inference: bool = True,
        use_time_encoding: bool = True,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._training_dtype = training_dtype
        self._use_time_encoding: bool = use_time_encoding
        self._embedding_dim: int = embedding_dim
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        if self._use_time_encoding:
            self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                    -sqrt(1.0 / num_time_buckets),
                    sqrt(1.0 / num_time_buckets),
                ),
            )

    @output_nvtx_hook(nvtx_tag="HSTUPositionalEncoder")
    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        seq_timestamps: Optional[torch.Tensor] = None,
        seq_start_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = self._embedding_dim**0.5
        if self._use_time_encoding:
            seq_embeddings = seq_embeddings * alpha
            seq_embeddings = triton_add_timestamp_positional_embeddings(
                seq_embeddings=seq_embeddings,
                seq_offsets=seq_offsets,
                pos_embeddings=self._position_embeddings_weight,
                ts_embeddings=self._timestamp_embeddings_weight,
                timestamps=seq_timestamps,
                max_seq_len=max_seq_len,
                max_contextual_seq_len=0,
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                interleave_targets=False,
                time_bucket_fn="sqrt",
            )
        elif not self._is_inference or seq_start_position is None:
            high_inds = _get_high_inds(
                seq_lengths, self._position_embeddings_weight, num_targets, False
            )
            if not is_fx_tracing():
                _, D = seq_embeddings.shape
                torch._assert(
                    seq_offsets.size(0) - 1 == high_inds.size(0),
                    "wrong jagged_offsets shape[0]",
                )
                _, D2 = self._position_embeddings_weight.shape
                torch._assert(D2 == D, "wrong dense shape[1]")

            seq_embeddings = triton_add_position_embeddings(
                jagged=seq_embeddings,
                jagged_offsets=seq_offsets,
                high_inds=high_inds,
                max_seq_len=max_seq_len,
                dense=self._position_embeddings_weight,
                scale=alpha,
            )
        else:  # use position embeddings and inference
            ind_offsets = seq_start_position
            high_inds = _get_high_inds(
                seq_lengths + seq_start_position,
                self._position_embeddings_weight,
                num_targets,
                False,
            )
            if not is_fx_tracing():
                _, D = seq_embeddings.shape
                torch._assert(
                    seq_offsets.size(0) - 1 == high_inds.size(0),
                    "wrong jagged_offsets shape[0]",
                )
                _, D2 = self._position_embeddings_weight.shape
                torch._assert(D2 == D, "wrong dense shape[1]")
            seq_embeddings = triton_add_position_embeddings(
                jagged=seq_embeddings,
                jagged_offsets=seq_offsets,
                high_inds=high_inds,
                max_seq_len=max_seq_len,
                dense=self._position_embeddings_weight,
                scale=alpha,
                ind_offsets=ind_offsets,
            )
        return seq_embeddings
