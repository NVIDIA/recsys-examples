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

from typing import Dict, List, Optional, Tuple

import torch
from dynamicemb_extensions import block_bucketize_sparse_features  # pyre-ignore
from torch import distributed as dist
from torchrec.distributed.dist_data import KJTAllToAll
from torchrec.distributed.embedding_sharding import BaseSparseFeaturesDist
from torchrec.distributed.types import Awaitable
from torchrec.fx.utils import assert_fx_safe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

torch.fx.wrap("len")

CACHE_LOAD_FACTOR_STR: str = "cache_load_factor"


# torch.Tensor.to can not be fx symbolic traced as it does not go through __torch_dispatch__ => fx.wrap it
@torch.fx.wrap
def _fx_wrap_tensor_to_device_dtype(
    t: torch.Tensor, tensor_device_dtype: torch.Tensor
) -> torch.Tensor:
    return t.to(device=tensor_device_dtype.device, dtype=tensor_device_dtype.dtype)


@torch.fx.wrap
def _fx_wrap_batch_size_per_feature(kjt: KeyedJaggedTensor) -> Optional[torch.Tensor]:
    return (
        torch.tensor(
            kjt.stride_per_key(), device=kjt.device(), dtype=kjt.lengths().dtype
        )
        if kjt.variable_stride_per_key()
        else None
    )


@torch.fx.wrap
def _fx_wrap_max_B(kjt: KeyedJaggedTensor) -> int:
    return max(kjt.stride_per_key()) if kjt.variable_stride_per_key() else -1


@torch.fx.wrap
def _fx_wrap_stride(kjt: KeyedJaggedTensor) -> Optional[int]:
    return None if kjt.variable_stride_per_key() else kjt.stride()


@torch.fx.wrap
def _fx_wrap_stride_per_key_per_rank(
    kjt: KeyedJaggedTensor, num_buckets: int
) -> Optional[List[List[int]]]:
    return (
        kjt.stride_per_key_per_rank() * num_buckets
        if kjt.variable_stride_per_key()
        else None
    )


@torch.fx.wrap
def _fx_wrap_gen_list_n_times(ls: List[str], n: int) -> List[str]:
    # Syntax for dynamo (instead of generator kjt.keys() * num_buckets)
    ret: List[str] = []
    for _ in range(n):
        ret.extend(ls)
    return ret


def bucketize_kjt_before_all2all(
    kjt: KeyedJaggedTensor,
    num_buckets: int,
    block_sizes: torch.Tensor,
    output_permute: bool = False,
    bucketize_pos: bool = False,
    block_bucketize_row_pos: Optional[List[torch.Tensor]] = None,
    dist_type_per_feature: Optional[Dict[str, str]] = None,
) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
    """
    Bucketizes the `values` in KeyedJaggedTensor into `num_buckets` buckets,
    `lengths` are readjusted based on the bucketization results.

    Note: This function should be used only for row-wise sharding before calling
    `KJTAllToAll`.

    Args:
        num_buckets (int): number of buckets to bucketize the values into.
        block_sizes: (torch.Tensor): bucket sizes for the keyed dimension.
        output_permute (bool): output the memory location mapping from the unbucketized
            values to bucketized values or not.
        bucketize_pos (bool): output the changed position of the bucketized values or
            not.
        block_bucketize_row_pos (Optional[List[torch.Tensor]]): The offsets of shard size for each feature.

    Returns:
        Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]: the bucketized `KeyedJaggedTensor` and the optional permute mapping from the unbucketized values to bucketized value.
    """

    num_features = len(kjt.keys())
    assert_fx_safe(
        block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received.",
    )
    block_sizes_new_type = _fx_wrap_tensor_to_device_dtype(block_sizes, kjt.values())

    dist_type_list = []
    for key in kjt.keys():
        assert key in dist_type_per_feature
        dist_type_str = dist_type_per_feature[key]
        if dist_type_str == "continuous":
            dist_type = 0
        elif dist_type_str == "roundrobin":
            dist_type = 1
        else:
            raise ValueError("Not support dist type of ", dist_type_str)
        dist_type_list.append(dist_type)
    dist_type_t = torch.tensor(
        dist_type_list, dtype=torch.int32, device=kjt.values().device
    )

    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
    ) = block_bucketize_sparse_features(
        kjt.lengths().view(-1),
        kjt.values(),
        bucketize_pos=bucketize_pos,
        sequence=output_permute,
        block_sizes=block_sizes_new_type,
        my_size=num_buckets,
        weights=kjt.weights_or_none(),
        batch_size_per_feature=_fx_wrap_batch_size_per_feature(kjt),
        max_B=_fx_wrap_max_B(kjt),
        block_bucketize_pos=block_bucketize_row_pos,  # each tensor should have the same dtype as kjt.lengths()
        dist_type_per_feature=dist_type_t,
    )

    return (
        KeyedJaggedTensor(
            # duplicate keys will be resolved by AllToAll
            keys=_fx_wrap_gen_list_n_times(kjt.keys(), num_buckets),
            values=bucketized_indices,
            weights=_determine_output_weights(
                kjt, pos, bucketize_pos, bucketized_weights
            ),
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=_fx_wrap_stride(kjt),
            stride_per_key_per_rank=_fx_wrap_stride_per_key_per_rank(kjt, num_buckets),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


def _determine_output_weights(kjt, pos, bucketize_pos, bucketized_weights):
    """
    Determine which weights to return: pos or bucketized_weights.

    If the input weights appear to be frequency counters (float values that were
    converted from uint64), preserve them instead of overriding with pos.
    """

    if not bucketize_pos:
        return bucketized_weights

    if kjt.weights_or_none() is not None and bucketized_weights is not None:
        # weights_as_int = bucketized_weights.long()
        # weights_back_to_float = weights_as_int.float()

        # if torch.allclose(
        #     bucketized_weights, weights_back_to_float, atol=1e-6
        # ) and torch.all(bucketized_weights >= 0):
        return bucketized_weights

    result = pos if bucketize_pos else bucketized_weights

    return result


class RwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
            communication.
        num_features (int): total number of features.
        feature_hash_sizes (List[int]): hash sizes of features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).

    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        dist_type_per_feature: Dict[str, str] = None,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._num_features = num_features
        feature_block_sizes = [
            (hash_size + self._world_size - 1) // self._world_size
            for hash_size in feature_hash_sizes
        ]
        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int64,
            ),
        )
        self._dist = KJTAllToAll(
            pg=pg,
            splits=[self._num_features] * self._world_size,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None
        self._dist_type_per_feature = dist_type_per_feature

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Bucketizes sparse feature values into world size number of buckets and then
        performs AlltoAll operation.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=self._is_sequence,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            dist_type_per_feature=self._dist_type_per_feature,
        )

        return self._dist(bucketized_features)
