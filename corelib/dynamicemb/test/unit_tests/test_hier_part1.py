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
"""The planner's part 1 and node layout of the hierarchical input dist on
one GPU, on plain tensors:

* the partition (``hier_output_dist_partition``) against the bucketize of
  TorchRec's dists in DynamicEmb (``block_bucketize_sparse_features``) on the
  same KJT, bit for bit: the keys by owner, their rows in the KJT (``inv``),
  their weights, their cells, where each owner's and each (owner, feature)'s
  keys start, and the input's per-feature totals, for many world sizes (most
  not powers of two). A second run gives the same bits;
* index dedup's fan-out (``hier_output_dist_fanout``) against a torch
  reference: the original rows of each row, its heavy rows of more than 64,
  their slots and consecutive sub-tasks, and where part 1 put each of them;
* the node layout (``hier_output_dist_node_layout``): the test plays every
  rank of one node (part 1 on a batch of its own, the node's counts stacked
  from their count vectors), pushes their keys where the layout says, and
  checks that each target's own part holds its node's keys in its KJT order
  and its remote part requester-major, then owner node, then feature-major,
  the needs and the skip; on one node also the owner's KJT and each
  requester's map into it.

Every call of the planner runs under ``torch.cuda.set_sync_debug_mode
("error")``, so an op that waits for the device fails the test.

Run it on one GPU, from ``corelib/dynamicemb``::

    python test/unit_tests/test_hier_part1.py
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DIST_TYPE_CODES
from dynamicemb_extensions import (
    block_bucketize_sparse_features,
    hier_output_dist_fanout,
    hier_output_dist_node_layout,
    hier_output_dist_partition,
)

# Owners of a case: the bucketize's 32-bit rules give other owners than the
# 64-bit ones where world_size is not a power of two.
WORLD_SIZES = (1, 2, 3, 5, 6, 7, 8, 12, 24, 32, 100, 255, 256)
CONTINUOUS, ROUNDROBIN, HASH_ROUNDROBIN = (
    DIST_TYPE_CODES[name] for name in ("continuous", "roundrobin", "hash_roundrobin")
)
HEAVY = 64  # kHeavy: a row of more original rows is heavy


def planner(binding, *args, **kwargs):
    """A planner binding under the sync debug mode "error": an op in it that
    waits for the device raises."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        return binding(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@dataclass
class Case:
    name: str
    batch_size: int = 64
    max_len: int = 12  # keys of a sample, at most
    # the table of each feature: features 0 and 1 share table 0
    tables: Tuple[int, ...] = (0, 0, 1, 2)
    # dist_type of each table
    rules: Tuple[int, ...] = (ROUNDROBIN, HASH_ROUNDROBIN, CONTINUOUS)
    hash_sizes: Tuple[int, ...] = (5000, 300, 1 << 20)  # rows of each table
    key_range: Tuple[int, int] = (0, 6000)
    values: torch.dtype = torch.int64
    lengths: torch.dtype = torch.int64
    weights: bool = False
    empty: Tuple[int, ...] = ()  # features without keys
    hot: float = 0.0  # share of places that hold key 7
    edges: float = 0.0  # share of places that hold keys at the edges (edge_keys)
    bound: int = 0  # index dedup: values past the valid keys
    world_sizes: Tuple[int, ...] = WORLD_SIZES


CASES = [
    Case("roundrobin", rules=(ROUNDROBIN, ROUNDROBIN, ROUNDROBIN)),
    Case("hash_roundrobin", rules=(HASH_ROUNDROBIN, HASH_ROUNDROBIN, HASH_ROUNDROBIN)),
    Case(
        "continuous: keys past the blocks", rules=(CONTINUOUS, CONTINUOUS, CONTINUOUS)
    ),
    Case(
        "roundrobin and hash_roundrobin",
        rules=(ROUNDROBIN, HASH_ROUNDROBIN, HASH_ROUNDROBIN),
    ),
    Case(
        "a rule per table, int32 keys and lengths",
        values=torch.int32,
        lengths=torch.int32,
    ),
    Case("int32 keys, int64 lengths", values=torch.int32),
    Case("int64 keys, int32 lengths", lengths=torch.int32),
    Case("weights", weights=True),
    Case("empty features", empty=(0, 3), weights=True),
    Case("no keys", empty=(0, 1, 2, 3)),
    Case("hot key", hot=0.7, max_len=40, weights=True),
    Case(
        "negative keys, continuous and roundrobin",
        rules=(CONTINUOUS, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 40), 1 << 40),
    ),
    Case(
        "negative int32 keys",
        rules=(CONTINUOUS, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 31), 1 << 31),
        values=torch.int32,
    ),
    Case(
        "int32 keys at the edges",
        rules=(CONTINUOUS, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 31), 1 << 31),
        values=torch.int32,
        edges=0.5,
    ),
    Case(
        "int64 keys at the edges",
        rules=(CONTINUOUS, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 40), 1 << 40),
        edges=0.5,
    ),
    Case(
        "int32 keys, continuous blocks whose int32 product wraps",
        rules=(CONTINUOUS, CONTINUOUS, CONTINUOUS),
        hash_sizes=(1 << 32, 3 << 31, (1 << 34) + 5),
        key_range=(-(1 << 31), 1 << 31),
        values=torch.int32,
        lengths=torch.int32,
        edges=0.5,
    ),
    Case(
        "int64 keys, continuous blocks of 2^62 rows and more",
        rules=(CONTINUOUS, CONTINUOUS, CONTINUOUS),
        hash_sizes=(1 << 62, (1 << 63) - 1, 1 << 40),
        key_range=(-(1 << 62), 1 << 62),
        edges=0.5,
    ),
    Case("index dedup's bound", bound=777, weights=True),
    Case(
        "long sequences", batch_size=3, max_len=5000, hot=0.2, world_sizes=(8, 32, 256)
    ),
    Case(
        "large",
        batch_size=4096,
        max_len=40,
        hot=0.1,
        weights=True,
        world_sizes=(1, 8, 32, 256),
    ),
    Case(
        "64 features",
        batch_size=256,
        max_len=8,
        tables=tuple(feature // 2 for feature in range(64)),
        rules=tuple(
            (ROUNDROBIN, HASH_ROUNDROBIN, CONTINUOUS)[table % 3] for table in range(32)
        ),
        hash_sizes=tuple(1000 + 37 * table for table in range(32)),
        world_sizes=(8, 129, 256),
    ),
]


@dataclass
class Batch:
    lengths: torch.Tensor  # [num_features * batch_size]
    offsets: torch.Tensor  # [num_features * batch_size + 1], of the lengths' dtype
    values: torch.Tensor  # [valid + bound]
    weights: Optional[torch.Tensor]
    in_offsets: torch.Tensor  # the input KJT's (index dedup: another one)
    in_totals: List[int]
    valid: int


def offsets_of(lengths: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros(1, dtype=lengths.dtype, device=lengths.device)
    return torch.cat([zero, torch.cumsum(lengths, 0).to(lengths.dtype)])


def block_size(case: Case, feature: int, world_size: int) -> int:
    """A feature's block size, TorchRec's: its table's rows over
    ``world_size``, rounded up."""
    return -(-case.hash_sizes[case.tables[feature]] // world_size)


def edge_keys(dtype: torch.dtype, block: int, world_size: int) -> List[int]:
    """Keys near the edges of a feature's rules, as ``dtype`` holds them: near
    0, ``2^31``, ``2^32``, ``2^63`` and ``world_size``, and near the ends of
    the first and the last ``continuous`` block in ``dtype``'s arithmetic."""
    bits = 32 if dtype == torch.int32 else 64
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    blocks_end = ((block & mask) * world_size) & mask
    near = (0, 1 << 31, 1 << 32, 1 << 63, world_size, block, blocks_end)
    return [
        (((point + step) & mask) ^ sign) - sign
        for point in near
        for step in (-2, -1, 0, 1, 2)
    ]


def make_batch(case: Case, seed: int, world_size: int = 1) -> Batch:
    """The case's KJT; its keys at the edges (``case.edges``) are those of
    ``world_size`` owners."""
    generator = torch.Generator().manual_seed(seed)
    num_features, batch_size = len(case.tables), case.batch_size
    lengths = torch.randint(
        0, case.max_len + 1, (num_features, batch_size), generator=generator
    )
    for feature in case.empty:
        lengths[feature] = 0
    lengths = lengths.flatten()
    valid = int(lengths.sum())
    lo, hi = case.key_range
    values = torch.randint(
        lo, hi, (valid + case.bound,), generator=generator, dtype=torch.int64
    )
    values[:valid][torch.rand(valid, generator=generator) < case.hot] = 7
    if case.edges:
        # each feature's edge keys, at a share of its places
        feature_of = torch.repeat_interleave(
            torch.arange(num_features), lengths.view(num_features, batch_size).sum(1)
        )
        chosen = torch.rand(valid, generator=generator) < case.edges
        for feature in range(num_features):
            keys = torch.tensor(
                edge_keys(
                    case.values, block_size(case, feature, world_size), world_size
                )
            )
            places = torch.nonzero(chosen & (feature_of == feature)).flatten()
            picks = torch.randint(
                0, keys.numel(), (places.numel(),), generator=generator
            )
            values[places] = keys[picks]
    weights = None
    if case.weights:
        weights = torch.rand(
            valid + case.bound, generator=generator, dtype=torch.float32
        )
    if case.bound:
        # index dedup: the input KJT is the original, which holds more keys
        in_lengths = lengths + torch.randint(0, 3, lengths.shape, generator=generator)
    else:
        in_lengths = lengths
    lengths = lengths.to(case.lengths).cuda()
    in_lengths = in_lengths.to(case.lengths).cuda()
    return Batch(
        lengths=lengths,
        offsets=offsets_of(lengths),
        values=values.to(case.values).cuda(),
        weights=None if weights is None else weights.cuda(),
        in_offsets=offsets_of(in_lengths),
        in_totals=in_lengths.view(num_features, batch_size).sum(1).tolist(),
        valid=valid,
    )


def rules_of(case: Case, world_size: int) -> Tuple[List[int], torch.Tensor]:
    """Each feature's dist_type, and its block size (:func:`block_size`,
    int64, on the GPU)."""
    dist_types = [case.rules[table] for table in case.tables]
    block_sizes = torch.tensor(
        [block_size(case, feature, world_size) for feature in range(len(case.tables))],
        dtype=torch.int64,
    )
    return dist_types, block_sizes.cuda()


def reference(
    case: Case, batch: Batch, world_size: int, dist_types, block_sizes
) -> Dict[str, torch.Tensor]:
    """The bucketize on the KJT as it is, as TorchRec's dists run it: the
    block sizes converted to the keys' dtype, and the keys it writes widened
    to int64 with their sign."""
    num_features, batch_size, valid = len(case.tables), case.batch_size, batch.valid
    new_lengths, new_values, new_weights, _, perm = block_bucketize_sparse_features(
        batch.lengths,
        batch.values,
        bucketize_pos=False,
        sequence=True,
        dist_type_per_feature=torch.tensor(dist_types, dtype=torch.int32).cuda(),
        block_sizes=block_sizes.to(batch.values.dtype),
        my_size=world_size,
        weights=batch.weights,
        batch_size_per_feature=None,
        max_B=-1,
    )
    cells = new_lengths.cpu().long().view(world_size, num_features, batch_size).sum(2)
    block_start = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(cells.sum(1), 0)]
    )
    feature_start = torch.cat(
        [torch.zeros(world_size, 1, dtype=torch.int64), torch.cumsum(cells, 1)], 1
    )
    inv = torch.empty(valid, dtype=torch.int64)
    inv[perm.cpu().long()[:valid]] = torch.arange(valid)
    return {
        "keys": new_values.cpu().long()[:valid],
        "inv": inv,
        "weights": new_weights.cpu()[:valid]
        if new_weights is not None
        else torch.empty(0),
        "block_start": block_start,
        "feature_start": feature_start.flatten(),
        "cell_start": torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.cumsum(cells.flatten(), 0)]
        ),
        # each key's cell, and world_size * num_features past the valid keys
        "row_cell": torch.cat(
            [
                torch.repeat_interleave(
                    torch.arange(world_size * num_features), cells.flatten()
                ),
                torch.full((case.bound,), world_size * num_features),
            ]
        ),
    }


def partition(
    case: Case,
    batch: Batch,
    world_size: int,
    dist_types,
    block_sizes,
    lengths: bool = False,
) -> Dict[str, torch.Tensor]:
    """The partition of the batch, from its offsets, or (``lengths``) from its
    lengths alone, the input's totals then from this KJT when it is the input
    (no index dedup's bound)."""
    return planner(
        hier_output_dist_partition,
        batch.values,
        None if lengths else batch.offsets,
        batch.lengths if lengths else None,
        case.batch_size,
        batch.weights,
        block_sizes,
        dist_types,
        world_size,
        None if lengths and not case.bound else batch.in_offsets,
        case.batch_size,
    )


def run_case(case: Case, seed: int, world_size: int) -> Dict[str, bool]:
    batch = make_batch(case, seed, world_size)
    dist_types, block_sizes = rules_of(case, world_size)
    ref = reference(case, batch, world_size, dist_types, block_sizes)
    out = partition(case, batch, world_size, dist_types, block_sizes)
    valid, num_keys = batch.valid, batch.values.numel()
    checks = {
        "keys": out["keys"].dtype == torch.int64
        and out["keys"].numel() == num_keys
        and torch.equal(out["keys"].cpu()[:valid], ref["keys"]),
        "inv": out["inv"].dtype == torch.int32
        and torch.equal(out["inv"].cpu().long()[:valid], ref["inv"])
        and torch.equal(out["inv"].cpu().long()[valid:], torch.arange(valid, num_keys)),
        "weights": ("weights" in out) == (batch.weights is not None)
        and (
            batch.weights is None
            or torch.equal(out["weights"].cpu()[:valid], ref["weights"])
        ),
        "block_start": torch.equal(out["block_start"].cpu(), ref["block_start"]),
        "feature_start": torch.equal(out["feature_start"].cpu(), ref["feature_start"]),
        "cells": torch.equal(out["cell_start"].cpu(), ref["cell_start"])
        and out["row_cell"].dtype == torch.int32
        and torch.equal(out["row_cell"].cpu().long(), ref["row_cell"]),
        "totals": out["input_length_per_key"].tolist() == batch.in_totals,
    }

    def same(other: Dict[str, torch.Tensor]) -> bool:
        return other.keys() == out.keys() and all(
            torch.equal(other[name][:valid], tensor[:valid])
            if name in ("keys", "weights")
            else torch.equal(other[name], tensor)
            for name, tensor in out.items()
        )

    checks["again"] = same(partition(case, batch, world_size, dist_types, block_sizes))
    checks["lengths"] = same(
        partition(case, batch, world_size, dist_types, block_sizes, lengths=True)
    )
    return checks


# ── index dedup's fan-out ────────────────────────────────────────────────────


@dataclass
class FanoutCase:
    name: str
    rows: int  # the deduplicated KJT's keys (a bound)
    valid: int  # its valid rows: the reverse indices point at [0, valid)
    originals: int  # original rows
    # rows that hold this many original rows each; the others share the rest
    fixed: Tuple[Tuple[int, int], ...] = ()
    hot: float = 0.0  # share of the rest that goes to one row


FANOUT_CASES = [
    FanoutCase("no heavy rows", rows=1000, valid=1000, originals=3000),
    FanoutCase(
        "rows at the heavy edge",
        rows=500,
        valid=500,
        originals=4000,
        fixed=((3, 64), (7, 65), (11, 128), (12, 129), (499, 640)),
    ),
    FanoutCase("one hot row", rows=2000, valid=2000, originals=50000, hot=0.9),
    FanoutCase("every row heavy", rows=300, valid=300, originals=90000),
    FanoutCase(
        "index dedup's bound",
        rows=4000,
        valid=1500,
        originals=6000,
        fixed=((0, 70),),
        hot=0.2,
    ),
    FanoutCase("one row", rows=1, valid=1, originals=1000),
    FanoutCase("no rows", rows=0, valid=0, originals=0),
    FanoutCase("no original rows", rows=50, valid=50, originals=0),
]


def reverse_indices_of(case: FanoutCase, generator) -> torch.Tensor:
    """Each original row's deduplicated row: the fixed rows' counts exactly,
    the rest spread over the other valid rows (a share ``hot`` on one)."""
    fixed_rows = [row for row, _ in case.fixed]
    others = torch.tensor(
        [row for row in range(case.valid) if row not in fixed_rows], dtype=torch.int64
    )
    parts = [torch.full((count,), row) for row, count in case.fixed]
    rest = case.originals - sum(count for _, count in case.fixed)
    if rest:
        picks = others[torch.randint(0, others.numel(), (rest,), generator=generator)]
        picks[torch.rand(rest, generator=generator) < case.hot] = others[0]
        parts.append(picks)
    reverse_indices = torch.cat(parts + [torch.zeros(0, dtype=torch.int64)])
    return reverse_indices[torch.randperm(case.originals, generator=generator)]


def fanout_reference(
    reverse_indices: torch.Tensor, inv: torch.Tensor, rows: int
) -> Dict[str, torch.Tensor]:
    """On the host: each row's original rows in increasing order; the heavy
    rows (more than HEAVY) in row order, each with its slot and its
    sub-tasks of HEAVY original rows, consecutive and in entry order; and
    each heavy row's place among part 1's keys (inv's inverse)."""
    counts = torch.bincount(reverse_indices, minlength=rows)
    exp_off = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)])
    heavy_rows = torch.nonzero(counts > HEAVY).flatten()
    subs = (counts[heavy_rows] + HEAVY - 1) // HEAVY
    slot = torch.full((rows,), -1, dtype=torch.int64)
    slot[heavy_rows] = torch.arange(heavy_rows.numel())
    sub = [
        (row, int(exp_off[row]) + part * HEAVY)
        for row, row_subs in zip(heavy_rows.tolist(), subs.tolist())
        for part in range(row_subs)
    ]
    place = torch.empty(rows, dtype=torch.int64)
    place[inv] = torch.arange(rows)
    return {
        "exp_off": exp_off,
        "exp_idx": torch.sort(reverse_indices, stable=True).indices,
        "slot": slot,
        "sub": torch.tensor(sub, dtype=torch.int64).flatten(),
        "count": torch.tensor([heavy_rows.numel(), int(subs.sum())]),
        "pos": place[heavy_rows],
        "first": torch.cumsum(subs, 0) - subs,
    }


def run_fanout_case(case: FanoutCase, seed: int) -> Dict[str, bool]:
    generator = torch.Generator().manual_seed(seed)
    reverse_indices = reverse_indices_of(case, generator)
    inv = torch.randperm(case.rows, generator=generator).to(torch.int32)
    out = planner(hier_output_dist_fanout, reverse_indices.cuda(), inv.cuda())
    ref = fanout_reference(reverse_indices, inv.long(), case.rows)
    max_slots = case.originals // (HEAVY + 1) + 1
    max_subs = 2 * case.originals // HEAVY + 2
    slots, subs = ref["count"].tolist()
    # dtype and size of each table, and its entries that hold something
    tables = {
        "exp_off": (torch.int32, case.rows + 1, None),
        "exp_idx": (torch.int32, case.originals, None),
        "slot": (torch.int32, case.rows, None),
        "sub": (torch.int64, 2 * max_subs, 2 * subs),
        "count": (torch.int64, 2, None),
        "pos": (torch.int32, max_slots, slots),
        "first": (torch.int32, max_slots, slots),
    }
    checks = {
        "shapes": all(
            (out[name].dtype, out[name].numel()) == (dtype, size)
            for name, (dtype, size, _) in tables.items()
        )
    }
    for name, (_, _, valid) in tables.items():
        checks[name] = torch.equal(out[name].cpu().long()[:valid], ref[name])
    again = planner(hier_output_dist_fanout, reverse_indices.cuda(), inv.cuda())
    checks["again"] = all(
        torch.equal(again[name][:valid], out[name][:valid])
        for name, (_, _, valid) in tables.items()
    )
    return checks


# ── the node layout ──────────────────────────────────────────────────────────


@dataclass
class NodeCase:
    name: str
    num_nodes: int
    local_size: int
    num_features: int = 3
    node: int = 0  # the node the test plays
    batch_size: int = 16
    max_len: int = 6
    bound: int = 0  # index dedup: keys past the valid ones
    weights: bool = False
    empty: Tuple[int, ...] = ()
    # the capacities: the node's largest part plus this (below 0: the call
    # does not fit, and the node skips it)
    lookup_slack: int = 5
    remote_slack: int = 5


NODE_CASES = [
    NodeCase("one GPU", num_nodes=1, local_size=1, num_features=1),
    NodeCase("one node of 2", num_nodes=1, local_size=2),
    NodeCase("one node of 8, weights, bound", 1, 8, weights=True, bound=50),
    NodeCase("one node of 4, empty features", 1, 4, num_features=5, empty=(0, 3)),
    NodeCase("one node, exact fit", 1, 4, lookup_slack=0),
    NodeCase("one node, lookup_keys overflow", 1, 4, lookup_slack=-1),
    NodeCase("2 nodes of 2", num_nodes=2, local_size=2, num_features=2),
    NodeCase("2 nodes of 8, node 1, weights", 2, 8, node=1, weights=True),
    NodeCase("3 nodes of 4, node 2, bound", 3, 4, num_features=2, node=2, bound=40),
    NodeCase("4 nodes of 3, node 1", 4, 3, num_features=1, node=1),
    NodeCase("2 nodes of 1", num_nodes=2, local_size=1),
    NodeCase("4 nodes of 8", 4, 8, num_features=4, node=3),
    NodeCase("2 nodes, exact fit", 2, 4, node=1, lookup_slack=0, remote_slack=0),
    NodeCase("2 nodes, remote_keys overflow", 2, 4, remote_slack=-1),
    NodeCase("2 nodes, lookup_keys overflow", 2, 4, node=1, lookup_slack=-1),
]


def count_vector(row_cell: torch.Tensor, valid: int, case: NodeCase) -> torch.Tensor:
    """A requester's counts for its node, from its rows' cells: its keys for
    each target's remote part, then for each owner of the node and
    feature."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    world_size = num_nodes * local_size
    cells = torch.bincount(
        row_cell[:valid].long(), minlength=world_size * num_features
    ).view(num_nodes, local_size, num_features)
    same = cells[case.node]
    return torch.cat([cells.sum((0, 2)) - same.sum(1), same.flatten()])


def run_node_case(case: NodeCase, seed: int) -> Dict[str, bool]:
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    world_size, node = num_nodes * local_size, case.node
    kjt = Case(
        case.name,
        batch_size=case.batch_size,
        max_len=case.max_len,
        tables=tuple(range(num_features)),
        rules=(HASH_ROUNDROBIN,) * num_features,
        hash_sizes=(1 << 20,) * num_features,
        key_range=(-(1 << 40), 1 << 40),
        weights=case.weights,
        empty=case.empty,
        bound=case.bound,
    )
    dist_types, block_sizes = rules_of(kjt, world_size)
    batches = [
        make_batch(kjt, seed * 100 + local_rank) for local_rank in range(local_size)
    ]
    parts = [
        partition(kjt, batch, world_size, dist_types, block_sizes) for batch in batches
    ]
    valid = [batch.valid for batch in batches]
    cell_start = [part["cell_start"].tolist() for part in parts]
    # what node_count_exchange gives every rank of the node
    node_counts = torch.stack(
        [
            count_vector(part["row_cell"].cpu(), valid[local_rank], case)
            for local_rank, part in enumerate(parts)
        ]
    )
    remote_counts = node_counts[:, :local_size]  # [requester][target]
    same_counts = node_counts[:, local_size:].view(
        local_size, local_size, num_features
    )  # [requester][owner][feature]
    own_cells = same_counts.permute(1, 2, 0).reshape(local_size, -1)
    own_total, remote_total = own_cells.sum(1), remote_counts.sum(0)
    need_lookup, need_remote = int(own_total.max()), int(remote_total.max())
    lookup_keys = need_lookup + case.lookup_slack
    remote_keys = need_remote + case.remote_slack if num_nodes > 1 else 0
    skip = need_lookup > lookup_keys or need_remote > remote_keys

    def layout_of(local_rank: int, **owner) -> Dict[str, torch.Tensor]:
        part = parts[local_rank]
        out = planner(
            hier_output_dist_node_layout,
            node_counts.cuda(),
            part["cell_start"],
            part["block_start"],
            part["row_cell"],
            num_nodes,
            local_size,
            node,
            local_rank,
            num_features,
            lookup_keys,
            remote_keys,
            **owner,
        )
        return {name: tensor.cpu() for name, tensor in out.items()}

    layouts = [layout_of(local_rank) for local_rank in range(local_size)]
    checks = {
        "counts": all(
            torch.equal(layout["count_vector"], node_counts[local_rank])
            for local_rank, layout in enumerate(layouts)
        ),
        "own_cells": all(
            torch.equal(layout["own_cells"], own_cells)
            and torch.equal(
                layout["own_starts"], torch.cumsum(own_cells, 1) - own_cells
            )
            and torch.equal(layout["own_total"], own_total)
            for layout in layouts
        ),
        "needs": all(
            int(layout["need_lookup"]) == need_lookup
            and bool(layout["skip"]) == skip
            and int(layout["rows_to_push"]) == (0 if skip else valid[local_rank])
            for local_rank, layout in enumerate(layouts)
        ),
    }
    if num_nodes > 1:  # one node has no remote part
        checks["requester_start"] = all(
            torch.equal(
                layout["requester_start"],
                torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int64),
                        remote_counts[:, local_rank].cumsum(0),
                    ]
                ),
            )
            for local_rank, layout in enumerate(layouts)
        )
        checks["need_remote"] = all(
            int(layout["need_remote"]) == need_remote for layout in layouts
        )
    if skip:
        return checks

    def rows_of(source: int, owner: int, feature: int) -> range:
        cell = owner * num_features + feature
        return range(cell_start[source][cell], cell_start[source][cell + 1])

    # Every requester pushes its valid rows where its layout says: rows of
    # (target, destination, requester, row).
    pushes = []
    for local_rank in range(local_size):
        cell = parts[local_rank]["row_cell"].cpu()[: valid[local_rank]].long()
        rows = torch.arange(valid[local_rank])
        pushes.append(
            torch.stack(
                [
                    cell // num_features % local_size,
                    layouts[local_rank]["destination_base"][cell] + rows,
                    torch.full_like(rows, local_rank),
                    rows,
                ],
                1,
            )
        )
    pushes = torch.cat(pushes)
    places = pushes[:, 0] * (lookup_keys + remote_keys) + pushes[:, 1]
    checks["one row a place"] = places.unique().numel() == places.numel()
    own_order, own_good, remote_good = [], True, True
    for target in range(local_size):
        owner = node * local_size + target
        mine = pushes[pushes[:, 0] == target]
        own = mine[mine[:, 1] < lookup_keys]
        own = own[own[:, 1].argsort()]
        # the owner's KJT order: [feature][requester][requester's order]
        order = [
            (source, row)
            for feature in range(num_features)
            for source in range(local_size)
            for row in rows_of(source, owner, feature)
        ]
        own_order.append(order)
        own_good &= torch.equal(
            own[:, 1], torch.arange(int(own_total[target]))
        ) and own[:, 2:].tolist() == [list(entry) for entry in order]
        remote = mine[mine[:, 1] >= lookup_keys]
        remote = remote[remote[:, 1].argsort()]
        # requester-major, then owner node, then [feature][position]
        order = [
            [source, row]
            for source in range(local_size)
            for owner_node in range(num_nodes)
            if owner_node != node
            for feature in range(num_features)
            for row in rows_of(source, owner_node * local_size + target, feature)
        ]
        remote_good &= (
            torch.equal(
                remote[:, 1] - lookup_keys, torch.arange(int(remote_total[target]))
            )
            and remote[:, 2:].tolist() == order
        )
    checks["own parts"] = own_good
    checks["remote parts"] = remote_good
    if num_nodes > 1:
        return checks

    # One node: each owner's KJT from its node_keys after the pushes, and each
    # requester's map into its owners' KJTs.
    keys = [part["keys"].cpu() for part in parts]
    weights = [part["weights"].cpu() for part in parts] if case.weights else None
    place = [
        torch.full((valid[local_rank],), -1, dtype=torch.int64)
        for local_rank in range(local_size)
    ]
    owner_good = True
    for target in range(local_size):
        mine = pushes[pushes[:, 0] == target]
        node_keys = torch.zeros(lookup_keys, dtype=torch.int64)
        node_weights = torch.zeros(lookup_keys)
        for source in range(local_size):
            sent = mine[mine[:, 2] == source]
            node_keys[sent[:, 1]] = keys[source][sent[:, 3]]
            if weights:
                node_weights[sent[:, 1]] = weights[source][sent[:, 3]]
        out = layout_of(
            target,
            node_keys=node_keys.cuda(),
            node_weights=node_weights.cuda() if weights else None,
        )
        order = own_order[target]
        num_recv = len(order)
        lengths = own_cells[target]
        for position, (source, row) in enumerate(order):
            place[source][row] = position
        owner_good &= (
            int(out["num_recv"]) == num_recv
            and torch.equal(out["kjt_lengths"], lengths)
            and torch.equal(
                out["kjt_offsets"],
                torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]),
            )
            and torch.equal(
                out["length_per_key"], lengths.view(num_features, local_size).sum(1)
            )
            and out["kjt_values"].tolist()[:num_recv]
            == [int(keys[source][row]) for source, row in order]
            and (
                ("kjt_weights" in out) == bool(weights)
                and (
                    not weights
                    or out["kjt_weights"].tolist()[:num_recv]
                    == [float(weights[source][row]) for source, row in order]
                )
            )
        )
        layouts[target]["src"] = out["src"]
    checks["owner KJTs"] = owner_good
    checks["src"] = all(
        layouts[local_rank]["src"].dtype == torch.int32
        and torch.equal(
            layouts[local_rank]["src"][: valid[local_rank]].long(), place[local_rank]
        )
        for local_rank in range(local_size)
    )
    return checks


def report(label: str, checks: Dict[str, bool]) -> bool:
    good = all(checks.values())
    print(
        f"{label}: "
        + " ".join(f"{check}={int(passed)}" for check, passed in checks.items())
        + f" [{'ok' if good else 'FAIL'}]",
        flush=True,
    )
    return good


def main() -> None:
    torch.cuda.set_device(0)
    failures = 0
    for case_idx, case in enumerate(CASES):
        for world_size in case.world_sizes:
            checks = run_case(case, 100 + case_idx, world_size)
            label = f"partition '{case.name}' world_size={world_size}"
            failures += not report(label, checks)
    for case_idx, case in enumerate(FANOUT_CASES):
        checks = run_fanout_case(case, 200 + case_idx)
        failures += not report(f"fanout '{case.name}'", checks)
    for case_idx, case in enumerate(NODE_CASES):
        checks = run_node_case(case, 300 + case_idx)
        failures += not report(f"node layout '{case.name}'", checks)
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
