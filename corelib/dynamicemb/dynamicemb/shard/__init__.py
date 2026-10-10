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

"""Everything that runs a sharded DynamicEmb table, laid out like TorchRec's.

TorchRec splits this across ``distributed/`` -- the sharded modules and their
sharders at the top, the per-sharding-type machinery under
``distributed/sharding/``. This package holds both, flat, because DynamicEmb
will only ever have a handful of the latter:

    embedding.py / embeddingbag.py   the sharded EC / EBC and their sharders
                                     (TorchRec: distributed/embedding{,bag}.py)
    rw_sharding.py                   the row-wise EmbeddingSharding and lookups
                                     (TorchRec: distributed/sharding/rw_sharding.py)
    input_dist.py                    the bucketizer feeding them
                                     (TorchRec keeps this inside rw_sharding.py)

Nothing here belongs under ``planner/``: planning decides how a table is laid
out, this runs the layout. A new sharding type goes beside rw_sharding.py.
"""

from .embedding import (
    DynamicEmbeddingCollectionSharder,
    ShardedDynamicEmbeddingCollection,
)
from .embeddingbag import (
    DynamicEmbeddingBagCollectionSharder,
    ShardedDynamicEmbeddingBagCollection,
)

__all__ = [
    "ShardedDynamicEmbeddingCollection",
    "DynamicEmbeddingCollectionSharder",
    "ShardedDynamicEmbeddingBagCollection",
    "DynamicEmbeddingBagCollectionSharder",
]
