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

from typing import List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DIST_TYPE_CODES
from dynamicemb_extensions import block_bucketize_sparse_features  # pyre-ignore
from torch import distributed as dist
from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable
from torchrec.distributed.embedding_sharding import (
    KJTSplitsAllToAllMeta,
    _fx_wrap_batch_size_per_feature,
    _fx_wrap_gen_list_n_times,
    _fx_wrap_max_B,
    _fx_wrap_stride,
    _fx_wrap_stride_per_key_per_rank,
    _fx_wrap_tensor_to_device_dtype,
)
from torchrec.distributed.sharding.rw_sharding import (
    RwSparseFeaturesDist as _RwSparseFeaturesDist,
)
from torchrec.distributed.types import Awaitable
from torchrec.fx.utils import assert_fx_safe
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from ..scored_kjt import ScoredKeyedJaggedTensor

torch.fx.wrap("len")


def dist_type_codes(
    dist_types: List[str], device: Optional[torch.device] = None
) -> torch.Tensor:
    """The per-feature ``dist_type`` as the int32 codes the kernel switches on.

    Built once, when the distributor is, rather than per forward: the features
    a distributor serves are fixed at construction, and so is their order.

    There is no default: the caller holds the sharding's feature order, and a
    rule guessed here would be applied to every key without a word.
    """
    if dist_types is None:
        raise ValueError(
            "dist_types is required, one entry per feature in the same order as "
            "feature_hash_sizes."
        )
    try:
        codes = [DIST_TYPE_CODES[dist_type] for dist_type in dist_types]
    except KeyError as missing:
        raise ValueError(
            f"Not support dist type of {missing.args[0]!r}, "
            f"expected one of {sorted(DIST_TYPE_CODES)}"
        ) from None
    return torch.tensor(codes, dtype=torch.int32, device=device)


def bucketize_kjt_before_all2all(
    kjt: KeyedJaggedTensor,
    num_buckets: int,
    block_sizes: torch.Tensor,
    output_permute: bool = False,
    bucketize_pos: bool = False,
    block_bucketize_row_pos: Optional[List[torch.Tensor]] = None,
    dist_type_per_feature: Optional[torch.Tensor] = None,
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
        dist_type_per_feature (Optional[torch.Tensor]): int32 key -> rank rule per
            feature, from :func:`dist_type_codes`, in the same feature order as
            ``block_sizes`` -- the kernel indexes both with the same ``t``.

    Returns:
        Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]: the bucketized `KeyedJaggedTensor` and the optional permute mapping from the unbucketized values to bucketized value.
    """

    num_features = len(kjt.keys())
    assert_fx_safe(
        block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received.",
    )
    assert_fx_safe(
        dist_type_per_feature.numel() == num_features,
        f"Expecting dist types for {num_features} features, but {dist_type_per_feature.numel()} received.",
    )
    block_sizes_new_type = _fx_wrap_tensor_to_device_dtype(block_sizes, kjt.values())
    dist_type_t = dist_type_per_feature.to(device=kjt.values().device)

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


def permute_scores_like_keys(
    scores: torch.Tensor, unbucketize_permute: torch.Tensor
) -> torch.Tensor:
    """``scores`` in the order :func:`bucketize_kjt_before_all2all` put its keys.

    ``unbucketize_permute[i]`` is where the bucketize moved the ``i``-th input
    key, so the score of that key goes there too. It is a permutation, so the
    scatter writes each place once and gives the same result every time.
    """
    return torch.empty_like(scores).index_copy_(0, unbucketize_permute, scores)


def cut_to_length_per_key(
    kjt: KeyedJaggedTensor,
    unbucketize_permute: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Cut the arrays that are longer than the keys the lengths of ``kjt`` describe.

    The index dedup of an embedding collection keeps its unique keys at the length of its
    input -- the host does not know how many there are -- with the real ones first, and
    the bucketize allocates its outputs at that length and fills a prefix: what follows is
    not data, and is not even written. Everything that goes into the all2all has to be
    exactly as long as the lengths say (c10d wants the split sizes to add up to the
    tensor), and the permute of the sequence output has to hold only places that exist.

    The number of keys is the sum of ``kjt.length_per_key()``, the read the all2all
    makes next anyway; it stays cached on ``kjt``, so asking for it here adds none. The
    values and weights of ``kjt`` are cut in place (``kjt`` is the caller's own, fresh
    from the bucketize); ``unbucketize_permute`` and ``scores`` are returned cut. An array
    that is the right length is returned as it is, so a distribution of exact arrays
    (no dedup) is untouched.
    """
    total = sum(kjt.length_per_key())
    if kjt._values.numel() != total:
        kjt._values = kjt._values[:total]
    if kjt._weights is not None and kjt._weights.numel() != total:
        kjt._weights = kjt._weights[:total]
    if unbucketize_permute is not None and unbucketize_permute.numel() != total:
        unbucketize_permute = unbucketize_permute[:total]
    if scores is not None and scores.numel() != total:
        scores = scores[:total]
    return unbucketize_permute, scores


class _ScoredTensorsAwaitable(KJTAllToAllTensorsAwaitable):
    """The keys' tensors all2all, waiting for the scores' too.

    It is a ``KJTAllToAllTensorsAwaitable`` on purpose: TorchRec reads the
    ``input_splits``, ``output_splits`` and recat the output distribution needs
    off the instances of that class, and skips anything else. It takes over the
    state of the keys' awaitable instead of issuing the all2all again.
    """

    def __init__(
        self,
        keys_awaitable: KJTAllToAllTensorsAwaitable,
        scores_awaitable: KJTAllToAllTensorsAwaitable,
    ) -> None:
        self.__dict__.update(keys_awaitable.__dict__)
        self._scores_awaitable = scores_awaitable

    def clear_inputs(self) -> int:
        return super().clear_inputs() + self._scores_awaitable.clear_inputs()

    def _wait_impl(self) -> ScoredKeyedJaggedTensor:
        keys = super()._wait_impl()
        scores = self._scores_awaitable.wait()
        # Same lengths, same recat: the scores come out in the keys' order.
        return ScoredKeyedJaggedTensor.from_kjt(keys, scores.values())


class _ScoredSplitsAwaitable(Awaitable[KJTAllToAllTensorsAwaitable]):
    """The first level of the keys' all2all and of the scores'.

    The keys' tensors are waited for before the scores', on every rank, so the
    two all2alls are issued in the same order everywhere.
    """

    def __init__(self, keys_awaitable, scores_awaitable) -> None:
        super().__init__()
        self._keys_awaitable = keys_awaitable
        self._scores_awaitable = scores_awaitable

    def _wait_impl(self) -> KJTAllToAllTensorsAwaitable:
        keys = self._keys_awaitable.wait()
        return _ScoredTensorsAwaitable(keys, self._scores_awaitable.wait())


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


class RwSparseFeaturesDist(_RwSparseFeaturesDist):
    """TorchRec's row-wise distributor, bucketizing by DynamicEmb's rule.

    The distribution itself is entirely TorchRec's: the per-feature block
    sizes, the KJTAllToAll and its splits, the sequence permute, all come from
    ``super().__init__``. One call differs. TorchRec places a key by the block
    of rows it falls in and rewrites it to its offset inside that block;
    DynamicEmb has three rules (``dyn_emb::DistType``), two of which leave the
    key global, because a hash table has no row index to rewrite it into.

    Args:
        dist_types: the rule per feature, in the same order as
            ``feature_hash_sizes`` -- the kernel indexes both with the same
            ``t``, and building them together here is what keeps them aligned.
        per_key_score: the sparse features carry a score per key
            (:class:`~dynamicemb.scored_kjt.ScoredKeyedJaggedTensor`), sent by
            an all2all of their own beside the keys', and handed on with the
            received keys.
    """

    # What the index dedup of the embedding collection asks before it hands over its
    # unique keys at the length of its input, the real ones first, instead of cutting
    # them to length with a read of their count: ``forward`` cuts them (see
    # :func:`cut_to_length_per_key`). A distributor of TorchRec's gets exact arrays.
    accepts_padded_dedup: bool = True

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        dist_types: Optional[List[str]] = None,
        per_key_score: bool = False,
    ) -> None:
        super().__init__(
            pg=pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device,
            is_sequence=is_sequence,
            has_feature_processor=has_feature_processor,
            need_pos=need_pos,
        )
        self.register_buffer(
            "_dist_type_tensor",
            dist_type_codes(dist_types, device=device),
            persistent=False,
        )
        self._score_dist: Optional[KJTAllToAll] = (
            KJTAllToAll(pg=pg, splits=[num_features] * pg.size())
            if per_key_score
            else None
        )

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """Bucketize by ``dist_type``, then TorchRec's AlltoAll unchanged."""
        # A collection that takes a score per key refuses a plain input while it
        # trains (see ``require_scores``) and, in evaluation, drops the scores of
        # an input that has them (``without_scores``), so what arrives here is
        # scored exactly when training: the same on every rank, which the
        # all2all of the scores, a collective, needs.
        takes_scores = self._score_dist is not None and isinstance(
            sparse_features, ScoredKeyedJaggedTensor
        )
        bucketized_features, unbucketize_permute = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            # The scores are moved by it too, though only a sequence embedding
            # needs it afterwards.
            output_permute=self._is_sequence or takes_scores,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            dist_type_per_feature=self._dist_type_tensor,
        )
        scores = sparse_features.scores() if takes_scores else None
        unbucketize_permute, scores = cut_to_length_per_key(
            bucketized_features, unbucketize_permute, scores
        )
        self.unbucketize_permute_tensor = (
            unbucketize_permute if self._is_sequence else None
        )

        keys_dist = self._dist(bucketized_features)
        if not takes_scores:
            return keys_dist

        # TrainPipelineSparseDist swaps ``_dist.forward`` for one that returns
        # the metadata of a splits all2all it fuses across modules, not an
        # awaitable. The fused path rebuilds the keys' tensors awaitable itself
        # and has nowhere to carry the scores, so it is refused for now.
        if isinstance(keys_dist, KJTSplitsAllToAllMeta):
            raise NotImplementedError(
                "A table with score_strategy PRECISE_CUSTOMIZED cannot be "
                "trained with a TorchRec train pipeline "
                "(TrainPipelineSparseDist and its variants) yet: the pipeline "
                "fuses the splits all2all of the keys and has no channel for "
                "per-key scores. Run the model without the pipeline."
            )

        scores = permute_scores_like_keys(scores, unbucketize_permute)
        # The scores are sent as the values of a KJT of the keys' shape. Its
        # lengths are a copy: ``clear_inputs`` frees what an all2all was given
        # once that all2all is done, and must not take the other's lengths away.
        scores_kjt = KeyedJaggedTensor(
            keys=bucketized_features.keys(),
            values=scores,
            lengths=bucketized_features.lengths().clone(),
            stride=bucketized_features.stride(),
        )
        return _ScoredSplitsAwaitable(keys_dist, self._score_dist(scores_kjt))
