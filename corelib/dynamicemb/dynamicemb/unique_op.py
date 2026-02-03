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

"""
Simplified unique operation for deduplicating GPU tensors.

This module provides a stateless unique function that allocates all intermediate
buffers internally, making it simple to use without managing hash table state.
"""

from typing import Optional, Tuple

import torch
from dynamicemb_extensions import unique_cuda as _unique_cuda  # pyre-ignore


def unique(
    keys: torch.Tensor,
    *,
    frequency_counters: Optional[torch.Tensor] = None,
    input_frequencies: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deduplicate keys using a GPU hash table.

    This function allocates internal hash table buffers automatically based on
    input size. The buffers are temporary and freed after the operation.
    Uses the current CUDA stream.

    Args:
        keys: Input keys tensor (int64 or uint64) on CUDA device
        frequency_counters: Optional output tensor to accumulate frequency counts
            per unique key
        input_frequencies: Optional input tensor with frequency weight per input key

    Returns:
        Tuple of (unique_keys, output_indices, num_unique):
            - unique_keys: Deduplicated keys tensor
            - output_indices: Index mapping from input to unique keys (0-based)
            - num_unique: Scalar tensor with count of unique keys

    Example:
        >>> keys = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.int64, device='cuda')
        >>> unique_keys, indices, count = unique(keys)
        >>> print(unique_keys[:count.item()])
        tensor([1, 2, 3], device='cuda:0')
        >>> print(indices)
        tensor([0, 1, 0, 2, 1, 0], device='cuda:0')
    """
    if not keys.is_cuda:
        raise ValueError("keys must be on CUDA device")

    return _unique_cuda(
        keys,
        frequency_counters,
        input_frequencies,
    )
