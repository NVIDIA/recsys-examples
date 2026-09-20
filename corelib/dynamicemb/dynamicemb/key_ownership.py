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
Its device twin is ``_block_bucketize_sparse_features_cuda_kernel1/2`` in
``src/sparse_block_bucketize_features.cu``; the two must agree key for key, or
a key is looked up on one rank and stored on another with nothing raising.

Two things the rule is easy to get wrong, and why they live here:

* **The modulo is unsigned.** The kernel narrows in
  ``make_unsigned_t<index_t>``, so a negative key wraps rather than staying
  negative. ``torch``'s ``%`` follows Python and returns a non-negative
  remainder for a positive divisor, which is *not* the same number.
* **``dist_type`` selects the rule**, and ``roundrobin`` and ``hash_roundrobin``
  disagree for almost every key. Code that hardcodes one of them keeps roughly
  ``1/num_shards`` of the keys when handed the other, and reports no error.

``continuous`` has no inverse: it maps a key to a rank by range, and the
ranges depend on a per-feature block size the consumers here do not have.
Callers that meet it raise.
"""

from typing import Optional

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


def murmur3_fmix64(keys):
    """MurmurHash3's 64-bit finalizer -- the host twin of ``murmur3_fmix64`` in
    ``src/murmur_hash.cuh``.

    Takes a Python int or anything ``np.asarray`` accepts, and returns the same
    shape as ``uint64``. Only the avalanche step is here: callers narrow the
    result themselves -- modulo the world size to pick an owning rank, masked to
    a non-negative int64 to pick a hash bucket -- exactly as the two device
    callers do.

    A key is a bit pattern here, not a magnitude, so a negative one is
    reinterpreted rather than rejected -- ``numpy`` refuses to build a ``uint64``
    from a negative Python int, while ``astype`` on an array wraps the way C
    would. Wrapping is likewise the algorithm and not an error for the
    multiplies, which numpy is silent about for arrays but warns about for
    scalars; the warning is turned off rather than left to depend on the input's
    shape.

    :func:`owning_shard` is the torch twin of this plus the modulo, and is what
    anything holding a tensor should use; this stays for scalars and for callers
    already in numpy.
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
    all the caller needs, because :func:`_unsigned_mod` reads it back as
    unsigned.
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


def owning_shard(keys: Tensor, num_shards: int, dist_type: str) -> Tensor:
    """The shard index each key belongs to, as an int64 tensor on ``keys``'s device.

    ``num_shards`` is the fan-out of the rule, not the size of the cluster: under
    row-wise sharding it is the world size, and a table sharded over a subset of
    the ranks would pass the size of that subset.

    Raises:
        NotImplementedError: ``dist_type`` is ``continuous``.
        ValueError: ``dist_type`` is not a known one.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if dist_type == "hash_roundrobin":
        return _unsigned_mod(_fmix64_torch(keys), num_shards)
    if dist_type == "roundrobin":
        return _unsigned_mod(keys.to(torch.int64), num_shards)
    if dist_type == "continuous":
        raise NotImplementedError(
            "dist_type 'continuous' maps a key to a rank by range, which cannot "
            "be reconstructed from a key alone. Use 'roundrobin' or "
            "'hash_roundrobin'."
        )
    raise ValueError(f"Unknown dist_type: {dist_type!r}")


def owned_key_mask(
    keys: Tensor,
    rank: int,
    world_size: int,
    dist_type: str,
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
    return owning_shard(keys, world_size, dist_type) == rank


def owned_keys(
    keys: Optional[Tensor], rank: int, world_size: int, dist_type: str
) -> Optional[Tensor]:
    """:func:`owned_key_mask` applied, tolerating a list that is absent or empty.

    Removal lists are optional and often empty, so the caller would otherwise
    repeat that guard at every use.
    """
    if keys is None or keys.numel() == 0:
        return keys
    mask = owned_key_mask(keys, rank, world_size, dist_type)
    return keys if mask is None else keys[mask]
