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

from itertools import accumulate
from typing import Callable, List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DIST_TYPE_CODES
from dynamicemb_extensions import block_bucketize_sparse_features  # pyre-ignore
from torch import distributed as dist
from torchrec.distributed.dist_data import KJTAllToAll
from torchrec.distributed.embedding_sharding import (
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

from .hier_output_dist import DynamicEmbSequenceShardingContext, HierSharding

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

    Apart from :meth:`hier_forward`, the distribution is TorchRec's: the
    per-feature block sizes, the KJTAllToAll and its splits, the sequence
    permute, all come from ``super().__init__``. One call differs. TorchRec
    places a key by the block of rows it falls in and rewrites it to its
    offset inside that block; DynamicEmb has three rules
    (``dyn_emb::DistType``), two of which leave the key global, because a hash
    table has no row index to rewrite it into.

    Args:
        dist_types: the rule per feature, in the same order as
            ``feature_hash_sizes`` -- the kernel indexes both with the same
            ``t``, and building them together here is what keeps them aligned.
        hier: the sharding's hierarchical dist (``shard/hier_output_dist.py``),
            whose input dist :meth:`hier_forward` runs, or None. ``forward``
            stays TorchRec's distribution, for the calls that do not take it.
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
        dist_types: Optional[List[str]] = None,
        hier: Optional[HierSharding] = None,
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
        self.hier = hier

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """Bucketize by ``dist_type``, then TorchRec's AlltoAll unchanged."""
        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = self._bucketize(sparse_features)
        return self._dist(bucketized_features)

    def _bucketize(
        self, sparse_features: KeyedJaggedTensor
    ) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
        return bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=self._is_sequence,
            bucketize_pos=(
                self._has_feature_processor
                if sparse_features.weights_or_none() is None
                else self._need_pos
            ),
            dist_type_per_feature=self._dist_type_tensor,
        )

    def hier_setup(self, weights: bool) -> bool:
        """Whether the sharding takes its hierarchical dist; the first call
        builds it, collectively (``HierSharding.setup``), with this input
        dist's rule of each feature. ``weights``: the keys carry weights."""
        return self.hier.setup(
            self._feature_block_sizes_tensor, self._dist_type_tensor, weights
        )

    def hier_forward(
        self,
        features: KeyedJaggedTensor,
        input_features: KeyedJaggedTensor,
        reverse_indices: Optional[torch.Tensor],
        sharding_ctx: DynamicEmbSequenceShardingContext,
        on_ok: Optional[Callable[[], None]] = None,
    ) -> "HierPlanAwaitable":
        """Part 1 of the hierarchical input dist, local: the sharding's
        operator sorts the keys of ``features`` by owner (``dist_type``), on
        the device (``op.plan``). Nothing moves until the first wait. A call
        that does not fit the capacities moves by :meth:`fallback` at the
        second wait. See docs/hier_output_dist.md, sections 7.3 and 10.1.

        Args:
            features: the keys to send, ``int32`` or ``int64``. Under index
                dedup only the first keys are valid, and only the device
                knows how many.
            input_features: the KJT ``construct_jagged_tensors`` reads (the
                input before index dedup).
            reverse_indices: index dedup's map from the input's rows to the
                rows of ``features``.
            sharding_ctx: the call's sharding context, which gets the plan.
            on_ok: what to run once the call takes this path.
        """
        if features.stride() == 0:
            raise ValueError(
                "RwSparseFeaturesDist: an empty local batch (stride 0) is not "
                "supported: DynamicEmb's bucketize and TorchRec's "
                "construct_jagged_tensors take no such batch"
            )
        offsets = features.offsets_or_none()
        weights = features.weights_or_none()
        plan = self.hier.op.plan(
            features.values().contiguous(),
            offsets,
            features.lengths() if offsets is None else None,
            features.stride(),
            None if weights is None else weights.contiguous(),
            None if input_features is features else input_features.offsets(),
            input_features.stride(),
            reverse_indices,
        )
        sharding_ctx.plan = plan
        return HierPlanAwaitable(
            self.hier.op,
            plan,
            HierKJTAwaitable(self, plan, features, input_features, sharding_ctx, on_ok),
        )

    def fallback(
        self,
        features: KeyedJaggedTensor,
        sent: int,
        sharding_ctx: DynamicEmbSequenceShardingContext,
    ) -> KeyedJaggedTensor:
        """TorchRec's input dist of a call that does not fit the capacities.
        It bucketizes the first ``sent`` keys of ``features`` by
        ``dist_type`` (under index dedup only those are valid), so each key
        goes to the owner part 1 gave it. TorchRec's all-to-alls then move
        them; their split sizes cost a host sync. ``sharding_ctx`` gets what
        TorchRec's output dist reads. The pipelines replace ``self._dist``'s
        ``forward``, so it is called on the class. This module's own
        ``unbucketize_permute_tensor`` stays as it was."""
        weights = features.weights_or_none()
        kjt = KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values()[:sent],
            weights=None if weights is None else weights[:sent],
            lengths=features.lengths(),
            stride=features.stride(),
        )
        bucketized, unbucketize_permute = self._bucketize(kjt)
        tensors = KJTAllToAll.forward(self._dist, bucketized).wait()
        sharding_ctx.input_splits = tensors._input_splits["values"]
        sharding_ctx.output_splits = tensors._output_splits["values"]
        sharding_ctx.sparse_features_recat = tensors._recat
        if tensors._stride_per_rank is not None:
            sharding_ctx.batch_size_per_rank = tensors._stride_per_rank
        sharding_ctx.unbucketize_permute_tensor = unbucketize_permute
        return tensors.wait()


class HierPlanAwaitable(Awaitable[Awaitable[KeyedJaggedTensor]]):
    """Part 1 of a call on the hierarchical dist. Waiting on it is the first
    wait: ``op.start(plan)`` queues the rest of the call (its comm kernels and
    planner steps) on the op's side stream. Dropping it changes nothing (the
    prefetch pipeline runs part 1 of its first batch twice). A second wait
    returns what the first did (TorchRec's prefetch pipeline up to 1.6 and the
    JaggedMegatron one wait twice on their last batch)."""

    def __init__(self, op, plan, second: "HierKJTAwaitable") -> None:
        super().__init__()
        self._op = op
        self._plan = plan
        self._second = second
        self._started = False

    def _wait_impl(self) -> "HierKJTAwaitable":
        if not self._started:
            self._op.start(self._plan)
            self._started = True
        return self._second


class HierKJTAwaitable(Awaitable[KeyedJaggedTensor]):
    """The second wait: ``op.wait(plan)``, the call's one host sync, then the
    owner's KJT (the same KJT on a later wait). A call that does not fit the
    capacities runs on TorchRec's dists, on every rank alike:
    :meth:`RwSparseFeaturesDist.fallback` moves its keys and fills the
    sharding context for TorchRec's output dist."""

    def __init__(
        self,
        input_dist: RwSparseFeaturesDist,
        plan,
        features: KeyedJaggedTensor,
        input_features: KeyedJaggedTensor,
        sharding_ctx: DynamicEmbSequenceShardingContext,
        on_ok: Optional[Callable[[], None]],
    ) -> None:
        super().__init__()
        self._input_dist = input_dist
        self._plan = plan
        self._features = features
        self._input_features = input_features
        self._sharding_ctx = sharding_ctx
        self._on_ok = on_ok
        self._kjt: Optional[KeyedJaggedTensor] = None

    def _wait_impl(self) -> KeyedJaggedTensor:
        if self._kjt is not None:
            return self._kjt
        hier, plan = self._input_dist.hier, self._plan
        fits = hier.op.wait(plan)
        # The totals construct_jagged_tensors reads, without a host sync.
        input_length_per_key = plan.input_length_per_key
        self._input_features._length_per_key = input_length_per_key
        self._input_features._offset_per_key = [0] + list(
            accumulate(input_length_per_key)
        )
        if fits:
            if self._on_ok is not None:
                self._on_ok()
            self._kjt = hier.owner_kjt(plan, self._features.keys())
        else:
            hier.fell_back(plan)
            self._sharding_ctx.plan = None
            self._kjt = self._input_dist.fallback(
                self._features, plan.sent, self._sharding_ctx
            )
        # What only this wait needed goes: a pipeline keeps this awaitable
        # until after the backward.
        self._features = self._input_features = self._sharding_ctx = self._on_ok = None
        return self._kjt
