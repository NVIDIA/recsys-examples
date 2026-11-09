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

"""Which rank owns a key.

A dynamic embedding table stores global keys in a hash table. There is no row
index and no shard metadata that means anything, so *which rank holds a key* is
not recorded anywhere -- it is a **rule**, recomputed from the key wherever
ownership matters: when the input distributor buckets a batch, when a
checkpoint is loaded, when a delta is replayed.

This module is that rule, and it is the only place it is written down in Python.
Its device twins are ``_block_bucketize_sparse_features_cuda_kernel1/2``
(``hashed_shard``) in ``src/sparse_block_bucketize_features.cu`` and, for the
bucket inside the rank, ``LinearBucketTable::home_bucket`` in
``src/table_operation/types.cuh``; the three must agree key for key, or a key
is looked up on one rank and stored on another with nothing raising.

Three rules exist, and they are not alike:

* ``roundrobin``: the rank is ``key % num_shards`` (unsigned). The bucket a key
  lands in on that rank is then decided by the table from the per-rank
  capacity, so it changes whenever the number of shards does.
* ``hash_roundrobin``: the rank is ``murmur3_fmix64(key) % num_shards``
  (unsigned); the bucket is again the table's per-rank one. Robust to
  structured key spaces, but, like ``roundrobin``, re-dealt by a reshard.
* ``hash_bucket``: the rank *and* the bucket are read off one global bucket
  index. With ``h = murmur3_fmix64(key) & INT64_MAX`` (the table's own hash),
  ``C_g`` the capacity of the whole table across every shard
  (``DynamicEmbTableOptions.global_capacity``, a multiple of the bucket
  capacity ``bc``) and ``G = C_g / bc`` global buckets::

      g     = (h % C_g) // bc                 # global bucket, in [0, G)
      P     = ceil(G / num_shards)            # buckets per shard
      shard = g // P  ==  (h % C_g) // (P * bc)

  Shard ``r`` holds global buckets ``[r*P, (r+1)*P)`` and bucket ``g`` lives at
  local bucket ``g - r*P``. Nothing in ``g`` depends on ``num_shards``, so the
  keys that share a bucket at one fan-out share a bucket at every other -- a
  checkpoint written by 8 shards fills the same buckets when 1 shard loads it,
  and a bucket that fit then fits now. ``P * bc`` is what the planner stores as
  the shard's ``max_capacity``.

Three things the rule is easy to get wrong, and why they live here:

* **The modulo is unsigned.** The kernel narrows in
  ``make_unsigned_t<index_t>``, so a negative key wraps rather than staying
  negative. ``torch``'s ``%`` follows Python and returns a non-negative
  remainder for a positive divisor, which is *not* the same number.
* **``dist_type`` selects the rule**, and the three rules disagree for almost
  every key. Code that hardcodes one of them keeps roughly
  ``1/num_shards`` of the keys when handed the other, and reports no error.
* **``hash_bucket`` has parameters.** Without the table's
  ``global_capacity`` and ``bucket_capacity`` there is no rule, and this module
  raises rather than fall back to some other modulus the kernel does not use.

``continuous`` has no inverse: it maps a key to a rank by range, and the
ranges depend on a per-feature block size the consumers here do not have.
Callers that meet it raise.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

# MurmurHash3's 64-bit finalizer constants, as the int64 bit patterns torch
# works in. torch has no uint64, and its int64 multiply wraps exactly as the
# unsigned one does, so the products agree bit for bit.
_FMIX64_C1: int = 0xFF51AFD7ED558CCD - (1 << 64)
_FMIX64_C2: int = 0xC4CEB9FE1A85EC53 - (1 << 64)
# 64 - 33, the width left after the shift, hence the mask that turns torch's
# arithmetic (sign-propagating) right shift into the logical one C does.
_FMIX64_SHIFT: int = 33
_FMIX64_MASK: int = (1 << (64 - _FMIX64_SHIFT)) - 1
# The table hashes with the top bit cleared (``LinearBucket::hash``), so the
# hash_bucket rule does too (hash_roundrobin keeps the full unsigned word).
INT64_MAX: int = (1 << 63) - 1


def murmur3_fmix64(keys):
    """MurmurHash3's 64-bit finalizer -- the host twin of ``murmur3_fmix64`` in
    ``src/murmur_hash.cuh``.

    Takes a Python int or anything ``np.asarray`` accepts, and returns the same
    shape as ``uint64``. Only the avalanche step is here: callers narrow the
    result themselves -- masked to a non-negative int64 and reduced by the
    global capacity to pick a hash bucket -- exactly as the device callers do.

    A key is a bit pattern here, not a magnitude, so a negative one is
    reinterpreted rather than rejected -- ``numpy`` refuses to build a ``uint64``
    from a negative Python int, while ``astype`` on an array wraps the way C
    would. Wrapping is likewise the algorithm and not an error for the
    multiplies, which numpy is silent about for arrays but warns about for
    scalars; the warning is turned off rather than left to depend on the input's
    shape.

    :func:`owning_shard` is the torch twin of this plus the shard arithmetic,
    and is what anything holding a tensor should use; this stays for scalars
    and for callers already in numpy.
    """
    if isinstance(keys, (int, np.integer)):
        k = np.uint64(int(keys) & 0xFFFFFFFFFFFFFFFF)
    else:
        k = np.asarray(keys).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        k = k ^ (k >> np.uint64(33))
        k = k * np.uint64(0xFF51AFD7ED558CCD)
        k = k ^ (k >> np.uint64(33))
        k = k * np.uint64(0xC4CEB9FE1A85EC53)
        k = k ^ (k >> np.uint64(33))
    return k


def murmur3_hash_64bits(key: int) -> int:
    """Scalar :func:`murmur3_fmix64`, for constants computed once at import."""
    return int(murmur3_fmix64(key))


def _fmix64_torch(keys: Tensor) -> Tensor:
    """:func:`murmur3_fmix64` on an int64 tensor, on the tensor's own device.

    The result is the same 64 bits, read as int64 rather than uint64 -- which is
    all the caller needs: ``hash_roundrobin`` reads it back as unsigned through
    :func:`_unsigned_mod`, and ``hash_bucket`` masks the sign bit away.
    """
    k = keys.to(torch.int64)
    for constant in (_FMIX64_C1, _FMIX64_C2):
        k = k ^ ((k >> _FMIX64_SHIFT) & _FMIX64_MASK)
        k = k * constant
    return k ^ ((k >> _FMIX64_SHIFT) & _FMIX64_MASK)


def _unsigned_mod(values: Tensor, divisor: int) -> Tensor:
    """``values % divisor`` with ``values`` read as unsigned 64-bit.

    A power-of-two divisor is the common case (8 ranks a node) and needs no
    correction: the low bits of a two's-complement pattern are already the
    unsigned remainder. Otherwise a negative value stands for ``v + 2**64``, so
    its unsigned remainder is ``(v mod d + 2**64 mod d) mod d`` -- and torch's
    ``%`` gives the mathematical ``v mod d`` for a positive divisor, which is
    the piece that needs correcting rather than replacing.
    """
    if divisor & (divisor - 1) == 0:
        return values & (divisor - 1)
    remainder = values % divisor
    wrap = (1 << 64) % divisor
    return torch.where(values < 0, (remainder + wrap) % divisor, remainder)


def hashed_shard_capacity(
    global_capacity: int, bucket_capacity: int, num_shards: int
) -> int:
    """Rows one shard of a ``hash_bucket`` table holds: ``ceil(G / S) * bc``.

    ``G = global_capacity / bucket_capacity`` is the whole table's bucket count,
    ``S`` the number of shards. The planner stores the same number as the
    shard's ``max_capacity`` (``_sharded_table_bucket_layout``); keeping the
    derivation here lets a tier whose own capacity was rescaled (a hybrid host
    tier) still state the routing rule from the global numbers alone.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if not global_capacity or global_capacity <= 0:
        raise ValueError(
            "hash_bucket ownership needs the table's global_capacity "
            "(DynamicEmbTableOptions.global_capacity, set by the planner from "
            "num_embeddings and bucket_capacity); got "
            f"{global_capacity!r}"
        )
    if not bucket_capacity or bucket_capacity <= 0:
        raise ValueError(
            f"hash_bucket ownership needs bucket_capacity, got {bucket_capacity!r}"
        )
    if global_capacity % bucket_capacity != 0:
        raise ValueError(
            f"global_capacity {global_capacity} is not a multiple of "
            f"bucket_capacity {bucket_capacity}"
        )
    num_global_buckets = global_capacity // bucket_capacity
    return math.ceil(num_global_buckets / num_shards) * bucket_capacity


def owning_shard(
    keys: Tensor,
    num_shards: int,
    dist_type: str,
    *,
    global_capacity: Optional[int] = None,
    bucket_capacity: Optional[int] = None,
) -> Tensor:
    """The shard index each key belongs to, as an int64 tensor on ``keys``'s device.

    ``num_shards`` is the fan-out of the rule, not the size of the cluster: under
    row-wise sharding it is the world size, and a table sharded over a subset of
    the ranks would pass the size of that subset.

    ``global_capacity`` and ``bucket_capacity`` are the parameters of the
    ``hash_bucket`` rule (see the module docstring) and are required by it;
    the other rules ignore them.

    Raises:
        NotImplementedError: ``dist_type`` is ``continuous``.
        ValueError: ``dist_type`` is not a known one, or ``hash_bucket`` is
            missing its parameters.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if dist_type == "hash_bucket":
        shard_capacity = hashed_shard_capacity(
            global_capacity, bucket_capacity, num_shards
        )
        # Non-negative int64 on both sides, so torch's % is the C one here.
        h = _fmix64_torch(keys) & INT64_MAX
        return (h % global_capacity) // shard_capacity
    if dist_type == "hash_roundrobin":
        return _unsigned_mod(_fmix64_torch(keys), num_shards)
    if dist_type == "roundrobin":
        return _unsigned_mod(keys.to(torch.int64), num_shards)
    if dist_type == "continuous":
        raise NotImplementedError(
            "dist_type 'continuous' maps a key to a rank by range, which cannot "
            "be reconstructed from a key alone. Use 'roundrobin', "
            "'hash_roundrobin' or 'hash_bucket'."
        )
    raise ValueError(f"Unknown dist_type: {dist_type!r}")


def owned_key_mask(
    keys: Tensor,
    rank: int,
    world_size: int,
    dist_type: str,
    *,
    global_capacity: Optional[int] = None,
    bucket_capacity: Optional[int] = None,
) -> Optional[Tensor]:
    """Boolean mask selecting the keys *rank* owns under row-wise sharding.

    Returns ``None`` when no filtering is needed (single rank), so callers can
    skip the mask entirely rather than build an all-true one.

    Ownership is recomputed from the key rather than read off whatever produced
    the keys, so a globally gathered set can be handed to every rank unchanged.
    It does not reshard: a shard-local position (a checkpoint file's order, a
    delta's slot index) names a place inside one rank's table and carries no
    rank, so two source ranks folded onto one target rank would collide. Callers
    that could be asked to reshard check the fan-out themselves.
    """
    if world_size <= 1:
        return None
    return (
        owning_shard(
            keys,
            world_size,
            dist_type,
            global_capacity=global_capacity,
            bucket_capacity=bucket_capacity,
        )
        == rank
    )


def owned_keys(
    keys: Optional[Tensor],
    rank: int,
    world_size: int,
    dist_type: str,
    *,
    global_capacity: Optional[int] = None,
    bucket_capacity: Optional[int] = None,
) -> Optional[Tensor]:
    """:func:`owned_key_mask` applied, tolerating a list that is absent or empty.

    Removal lists are optional and often empty, so the caller would otherwise
    repeat that guard at every use.
    """
    if keys is None or keys.numel() == 0:
        return keys
    mask = owned_key_mask(
        keys,
        rank,
        world_size,
        dist_type,
        global_capacity=global_capacity,
        bucket_capacity=bucket_capacity,
    )
    return keys if mask is None else keys[mask]


@dataclass(frozen=True)
class OwnershipRule:
    """One table's key -> shard rule, with the parameters it needs.

    Built from the table's options with :meth:`of`; ``num_shards`` is the
    table's fan-out (world size for row-wise, one node for table-row-wise).
    ``global_capacity`` / ``bucket_capacity`` only matter for
    ``hash_bucket``.
    """

    dist_type: str
    num_shards: int
    global_capacity: Optional[int] = None
    bucket_capacity: Optional[int] = None

    @classmethod
    def of(cls, option: Any, num_shards: int) -> "OwnershipRule":
        return cls(
            dist_type=option.dist_type,
            num_shards=num_shards,
            global_capacity=getattr(option, "global_capacity", None),
            bucket_capacity=option.bucket_capacity,
        )

    @property
    def shard_capacity(self) -> Optional[int]:
        """Rows per shard under ``hash_bucket``; ``None`` for other rules."""
        if self.dist_type != "hash_bucket":
            return None
        return hashed_shard_capacity(
            self.global_capacity, self.bucket_capacity, self.num_shards
        )

    def owner(self, keys: Tensor) -> Tensor:
        return owning_shard(
            keys,
            self.num_shards,
            self.dist_type,
            global_capacity=self.global_capacity,
            bucket_capacity=self.bucket_capacity,
        )

    def mask(self, keys: Tensor, rank: int) -> Optional[Tensor]:
        return owned_key_mask(
            keys,
            rank,
            self.num_shards,
            self.dist_type,
            global_capacity=self.global_capacity,
            bucket_capacity=self.bucket_capacity,
        )
