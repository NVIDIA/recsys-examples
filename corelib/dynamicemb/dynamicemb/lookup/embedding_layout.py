# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The input and output layout of a batched embedding lookup.

This is the single place the layout is written down. Everything else -- the
Python lookup path, the CUDA kernels, and the comments in both -- should refer
here rather than restate it. The device side mirrors the pooled half of this in
``src/pooled_layout.cuh``; keep the two in step.

The input and the output number the same bags differently
---------------------------------------------------------
A lookup covers ``F`` features over a batch of ``B`` samples, so it has ``F*B``
bags, each holding the keys of one (feature, sample) pair. The input and the
output disagree about how to number them, and keeping the two apart is what
this module exists for:

**input order** is feature-major, ``s = f*B + b``. This is how a
``KeyedJaggedTensor``, and therefore ``offsets``, lays bags out: all of feature
0's samples, then all of feature 1's. ``offsets`` has ``F*B + 1`` entries and
bag ``s`` owns ``values[offsets[s]:offsets[s+1]]``.

**output order** is batch-major, ``r = b*F + f``. This is how the pooled output
lays them out: one row per sample, features side by side within the row. Under
uniform dims the pooled output ``[B, total_D]`` is also a free ``[B*F, D]``
view, and ``r`` is the row index into that view.

The two are transposes of each other. Converting between them is what the
backward path does when it works out which gradient row each key reads, and it
is the single most error-prone step in the lookup -- go through the helpers below rather than
writing ``s / B`` or ``b * F`` by hand.

Worked example: ``F=2``, ``B=3``, both tables dim 2, so ``total_D = 4``::

    lengths, in input order:  [2, 0, 1,   1, 3, 2]
                               \\__f0__/  \\__f1__/
    offsets:                  [0, 2, 2, 3, 4, 7, 9]      (F*B + 1 = 7 entries)

    input index s      0       1       2       3       4       5
    (feature, sample) (0,0)   (0,1)   (0,2)   (1,0)   (1,1)   (1,2)
    its keys          v[0:2]  v[2:2]  v[2:3]  v[3:4]  v[4:7]  v[7:9]
                              ^ an empty bag is normal

    output, [B=3, total_D=4] -- one row per sample:

                       cols 0:2   cols 2:4
                b=0  [    f0    |    f1    ]
                b=1  [    f0    |    f1    ]
                b=2  [    f0    |    f1    ]

    output index r     0       1       2       3       4       5
    (feature, sample) (0,0)   (1,0)   (0,1)   (1,1)   (0,2)   (1,2)

    Lining the two up by bag, every bag carries both numbers at once::

    (feature, sample) (0,0)   (0,1)   (0,2)   (1,0)   (1,1)   (1,2)
    input index s      0       1       2       3       4       5
    output index r     0       2       4       1       3       5

    Read a column, not a row: bag (0,1) -- feature 0 of sample 1 -- is entry 1
    of ``offsets`` but row 2 of the ``[B*F, D]`` view of the output. Only the
    first and last bag happen to get the same number in both orders.

Shapes
------
pooled (``SUM`` / ``MEAN``)   output ``[B, total_D]``; feature ``f`` occupies
                              columns ``[col_begin(f), col_begin(f) + col_width(f))``
sequence (``NONE``)           output ``[num_keys, D]``, one row per key, so
                              there is no pooling and no bag arithmetic at all

Dimensions
----------
``dims`` is per *table*; ``total_D`` sums over *features*, so a table shared by
several features contributes once per feature. ``max_D`` is the widest table.
``mixed_D`` means the tables do not all share a dim; only then is ``D_offsets``
materialized, and only then do the kernels need it to find a feature's columns.

The deduplicated embedding buffer the kernels gather from is always
``max(dims)`` wide, mixed dims or not -- that is the storage's row width, which
``Storage.max_embedding_dim()`` reports. A narrower table simply leaves
``[dim_t, max(dims))`` unused in its rows, and the kernels never read into the
hole because ``D_offsets`` tells them each feature's real width. Note that the
storage row is wider still: it is embedding ++ optimizer state, ``max_value_dim``
in total, which is why the source row stride is passed separately from the
copy width.

Invariants
----------
* ``offsets.numel() - 1 == F*B`` and ``offsets[0] == 0``. The backward path
  indexes a ``num_keys``-sized array with a raw ``offsets[s]``, so a non-zero
  base would write out of bounds.
* ``reverse_indices``, ``values`` and ``pooling_weights`` are aligned position
  by position: entry ``j`` of each describes the same key occurrence.
* Sequence mode requires a uniform dim, because its output is ``[N, D]`` with
  no per-feature column map to place a ragged row in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DynamicEmbPoolingMode


@dataclass(frozen=True)
class EmbeddingLayout:
    """Everything the lookup needs to know about shapes, for one call.

    Covers both output shapes, but they need very different amounts of it.
    A sequence lookup emits one row per key and has no bags to number, so it
    only reads ``pooling_mode`` and ``batch_size``; ``total_D``, ``max_D``,
    ``D_offsets`` and every helper below the shape section describe the pooled
    output and mean nothing without it.

    Built per forward because ``batch_size`` comes from the input. The rest is
    fixed at construction time and cached on the module.
    """

    pooling_mode: DynamicEmbPoolingMode
    batch_size: int  # B
    feature_num: int  # F
    num_keys: int  # keys in this batch, i.e. offsets[-1]
    total_D: int  # sum over features; pooled only
    max_D: int  # widest table; also the storage's row width
    # The dim every table shares, or None when they differ.
    common_D: Optional[int]
    # [F+1] int32 on device, materialized exactly when common_D is None.
    D_offsets: Optional[torch.Tensor]
    # Every feature's dim is a multiple of 4, so every D_offsets entry is too.
    # The kernels need this to know whether a feature's columns start on a
    # 16-byte boundary; they cannot work it out themselves because D_offsets is
    # on the device. Only consulted under mixed dims -- with a uniform dim a
    # feature starts at f*max_D, which max_D % 4 already settles.
    feature_dims_vec4: bool

    def __post_init__(self) -> None:
        # The two mixed-dim signals have to agree; the kernels pick their code
        # path off D_offsets while everything here reads common_D.
        if (self.common_D is None) != (self.D_offsets is not None):
            raise ValueError(
                f"common_D={self.common_D} and D_offsets="
                f"{'set' if self.D_offsets is not None else 'None'} disagree "
                "about whether the tables have mixed dims"
            )
        if not self.is_pooling and self.common_D is None:
            raise ValueError(
                "sequence lookup needs a uniform dim: its output is [N, D] "
                "with no per-feature column map to place a ragged row in"
            )

    @property
    def is_pooling(self) -> bool:
        return self.pooling_mode != DynamicEmbPoolingMode.NONE

    @property
    def mixed_D(self) -> bool:
        """True when the tables do not all share an embedding dim."""
        return self.common_D is None

    @property
    def num_bags(self) -> int:
        """``F*B``. The same count in either order; only the numbering differs."""
        return self.feature_num * self.batch_size

    def pooled_shape(self) -> Tuple[int, int]:
        return (self.batch_size, self.total_D)

    def sequence_shape(self) -> Tuple[int, int]:
        """One row per key -- no pooling, so nothing here depends on the bags.

        Only reachable under a uniform dim, which ``__post_init__`` enforces.
        """
        return (self.num_keys, self.common_D)

    # -- numbering a bag ---------------------------------------------------

    def input_index(self, f: int, b: int) -> int:
        """Where ``offsets`` keeps this bag: feature-major ``s = f*B + b``."""
        return f * self.batch_size + b

    def output_index(self, f: int, b: int) -> int:
        """Where the pooled output keeps it: batch-major ``r = b*F + f``."""
        return b * self.feature_num + f

    def feature_sample_of_input(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.batch_size)

    def feature_sample_of_output(self, r: int) -> Tuple[int, int]:
        b, f = divmod(r, self.feature_num)
        return f, b

    def output_index_of_input(self, s: int) -> int:
        """Renumber a bag from the input's order into the output's.

        This is the transpose; the backward path applies it to every key.
        """
        f, b = self.feature_sample_of_input(s)
        return self.output_index(f, b)

    # -- placing a feature in the output ------------------------------------

    def col_begin(self, f: int) -> int:
        """First output column of feature ``f``.

        Mirrors what the kernels do: look ``D_offsets`` up under mixed dims,
        stride by the uniform dim otherwise.
        """
        if self.D_offsets is not None:
            return int(self.D_offsets[f])
        return f * self.max_D

    def col_width(self, f: int) -> int:
        if self.D_offsets is not None:
            return int(self.D_offsets[f + 1]) - int(self.D_offsets[f])
        return self.max_D
