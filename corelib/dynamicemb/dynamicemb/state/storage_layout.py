# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Where a module keeps its values: the tiers, and how a cache relates to them."""

from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import Optional


class CachePolicy(Enum):
    """How a caching table's cache relates to what is below it.

    INCLUSIVE: the cache is a subset of a ``DynamicEmbStorage``, which owns every
    key's score (the cache ranks by timestamp only) and may have an external
    store below it that receives what it evicts. Chosen whenever our storage is
    used: no ``external_storage``, or one with a ``global_ddr_for_values`` budget.

    NINE (non-inclusive, non-exclusive): the cache sits directly on the external
    store, which is opaque -- a key may be in both, the store cannot be asked
    to keep a subset -- and the cache owns the scores. Chosen for a caching
    table with an ``external_storage`` and no ``global_ddr_for_values``.
    """

    INCLUSIVE = auto()
    NINE = auto()


class StorageTier(Flag):
    """A tier a table's values live in. A module has some of them (a ``StorageLayout``)."""

    HBM = auto()  # GPU memory
    DDR = auto()  # host memory, our own storage (DynamicEmbStorage)
    EXTERNAL = auto()  # an external_storage (a PS): opaque to us


@dataclass(frozen=True)
class StorageLayout:
    """Where a module keeps its values: which tiers, and whether HBM is a cache.

    ``tiers`` holds the tiers actually in use (an ``external_storage`` a layout
    ignores is not in it). ``cache_policy`` is None when nothing is cached; else HBM
    is a cache, and the path says how it relates to what is below it (see
    ``CachePolicy``). The tiers alone do not tell a cache from a split of the rows:
    HBM | DDR is both ``HybridStorage`` (no cache) and a cache over host storage.

    =============================  =========================  ==========
    layout                         tiers                      cache_policy
    =============================  =========================  ==========
    HBM only                       HBM                        None
    host only                      DDR                        None
    HybridStorage                  HBM | DDR                  None
    external storage only          EXTERNAL                   None
    cache over host storage        HBM | DDR                  INCLUSIVE
    ... with an external storage   HBM | DDR | EXTERNAL       INCLUSIVE
    cache over external storage    HBM | EXTERNAL             NINE
    =============================  =========================  ==========
    """

    tiers: StorageTier
    cache_policy: Optional[CachePolicy] = None

    @property
    def caching(self) -> bool:
        return self.cache_policy is not None

    @property
    def erasable(self) -> bool:
        """Whether an erase by score can scan this layout's keys.

        Only a single-tier GPU storage and the backing storage of a cache keep a
        key map the erase can scan; the others keep their keys where it cannot
        (host values read by the generic path, two tiers, an external store).
        """
        if self.cache_policy is None:
            return self.tiers == StorageTier.HBM
        return (
            self.cache_policy == CachePolicy.INCLUSIVE
            and StorageTier.EXTERNAL not in self.tiers
        )

    @property
    def hbm_direct(self) -> bool:
        """Whether the values are read and written in place in GPU memory: one
        tier, HBM, no cache."""
        return self.cache_policy is None and self.tiers == StorageTier.HBM

    def __str__(self) -> str:
        names = "+".join(tier.name for tier in StorageTier if tier in self.tiers)
        if self.cache_policy is None:
            return names
        return f"{names} (HBM caches, {self.cache_policy.name})"
