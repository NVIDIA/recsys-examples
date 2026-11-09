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
"""The planner's bucketize and node layout of the hierarchical input dist,
and the push of its node's keys, on one GPU, on plain tensors:

* the op's owner rules: its configuration check (``hier_output_dist_invalid``)
  refuses every ``dist_type`` but ``roundrobin`` and ``hash_roundrobin``, and
  a table whose features have different rules, and the bucketize's hook
  raises for those codes;
* the bucketize (``hier_output_dist_bucketize``) against the bucketize of
  TorchRec's dists in DynamicEmb (``block_bucketize_sparse_features``) on the
  same KJT, bit for bit: the keys by owner, their rows in the KJT (``inv``),
  their weights, their bucket features, where each owner's and each (owner,
  feature)'s keys start, and the input's per-feature totals, for many world
  sizes (most not powers of two). The keys by owner, those past the valid
  ones too, are the KJT's keys at ``inv``, widened to int64 with their sign,
  as the push gathers them. A second run gives the same bits;
* index dedup's fan-out (``hier_output_dist_fanout``) against a torch
  reference: the original rows of each row, its heavy rows of more than 64,
  their heavy slots and consecutive sub-tasks, and where the bucketize put
  each of them. The kernel numbers the heavy slots in the order its heavy
  rows reserve them, which can change from run to run, so the heavy tables
  are compared per row, and the numbering is checked: the heavy rows' slots
  a permutation of [0, heavy rows), each slot's sub-tasks a range of its own
  row's, the ranges in slot order from 0. Two more calls with the words the
  in channel keeps across calls give the same tables per row, and leave the
  words at 0;
* the node layout and the push (``hier_output_dist_node_layout``,
  ``hier_output_dist_node_push``): the test plays every rank of one node (the
  bucketize of a batch of its own; the node's offsets, their
  ``offset_per_bucket_feature`` stacked). It checks each rank's layout
  against counts of the requesters' rows: the needs and the skip, and where
  it puts each key in its target's parts. Then every requester pushes its
  KJT's keys (int64, or int32 widened with their sign) into one set of
  stand-ins for the node's targets, which hold garbage. Each target's own
  part holds its node's keys in its KJT order, and its remote part
  requester-major, then owner node, then feature-major. The entries past
  them, and under skip every entry, keep their garbage. On one node it also
  checks each requester's map into its owners' KJTs (``src``) and each
  owner's KJT. The layout's items, laid out on one CTA and on a grid
  (node_exchange's two paths), give the same tables; cases of 127 to 300
  features take the grid by default.

Every call of the planner runs under ``torch.cuda.set_sync_debug_mode
("error")``, so an op that waits for the device fails the test.

Run it on one GPU, from ``corelib/dynamicemb``::

    python test/unit_tests/test_hier_bucketize.py

Under memcheck, turn off the caching allocator. Then each tensor is an
allocation of its own, and memcheck sees an access past a tensor's end::

    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    compute-sanitizer --tool memcheck python test/unit_tests/test_hier_bucketize.py
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from dynamicemb.dynamicemb_config import DIST_TYPE_CODES
from dynamicemb_extensions import (
    block_bucketize_sparse_features,
    hier_output_dist_bucketize,
    hier_output_dist_fanout,
    hier_output_dist_invalid,
    hier_output_dist_node_layout,
    hier_output_dist_node_push,
)

# Owners of a case: the bucketize's 32-bit rules give other owners than the
# 64-bit ones where world_size is not a power of two.
WORLD_SIZES = (1, 2, 3, 5, 6, 7, 8, 12, 24, 32, 100, 255, 256)
ROUNDROBIN, HASH_ROUNDROBIN = (
    DIST_TYPE_CODES[name] for name in ("roundrobin", "hash_roundrobin")
)
# the op refuses it (run_rule_refusals)
CONTINUOUS = DIST_TYPE_CODES["continuous"]
HEAVY = 64  # kHeavy: a row of more original rows is heavy


def planner(binding, *args, **kwargs):
    """A planner binding under the sync debug mode "error": an op in it that
    waits for the device raises."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        return binding(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def garbage(shape: Tuple[int, ...], dtype: torch.dtype, generator) -> torch.Tensor:
    """Random bits of ``dtype`` on the GPU, as memory holds before a call."""
    bits = torch.randint(-(1 << 31), (1 << 31) - 1, shape, generator=generator)
    if dtype == torch.int64:
        bits = (bits << 32) ^ torch.randint(0, 1 << 32, shape, generator=generator)
    return bits.to(torch.int32 if dtype == torch.float32 else dtype).view(dtype).cuda()


def same_bits(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Whether two tensors hold the same bits (garbage floats can be NaNs)."""
    if got.dtype != want.dtype or got.shape != want.shape:
        return False
    if got.dtype == torch.float32:
        got, want = got.view(torch.int32), want.view(torch.int32)
    return torch.equal(got, want)


@dataclass
class Case:
    name: str
    batch_size: int = 64
    max_len: int = 12  # keys of a sample, at most
    # the table of each feature: features 0 and 1 share table 0
    tables: Tuple[int, ...] = (0, 0, 1, 2)
    # dist_type of each table
    rules: Tuple[int, ...] = (ROUNDROBIN, HASH_ROUNDROBIN, ROUNDROBIN)
    key_range: Tuple[int, int] = (0, 6000)
    values: torch.dtype = torch.int64
    lengths: torch.dtype = torch.int64
    weighted: bool = False
    empty: Tuple[int, ...] = ()  # features without keys
    hot: float = 0.0  # share of places that hold key 7
    edges: float = 0.0  # share of places that hold keys at the edges (edge_keys)
    bound: int = 0  # index dedup: values past the valid keys
    world_sizes: Tuple[int, ...] = WORLD_SIZES


CASES = [
    Case("roundrobin", rules=(ROUNDROBIN, ROUNDROBIN, ROUNDROBIN)),
    Case("hash_roundrobin", rules=(HASH_ROUNDROBIN, HASH_ROUNDROBIN, HASH_ROUNDROBIN)),
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
    Case("weights", weighted=True),
    Case("empty features", empty=(0, 3), weighted=True),
    Case("no keys", empty=(0, 1, 2, 3)),
    Case("hot key", hot=0.7, max_len=40, weighted=True),
    Case(
        "negative keys, both rules",
        rules=(HASH_ROUNDROBIN, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 40), 1 << 40),
    ),
    Case(
        "negative int32 keys",
        rules=(HASH_ROUNDROBIN, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 31), 1 << 31),
        values=torch.int32,
    ),
    Case(
        "int32 keys at the edges",
        rules=(HASH_ROUNDROBIN, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 31), 1 << 31),
        values=torch.int32,
        edges=0.5,
    ),
    Case(
        "int64 keys at the edges",
        rules=(HASH_ROUNDROBIN, ROUNDROBIN, HASH_ROUNDROBIN),
        key_range=(-(1 << 40), 1 << 40),
        edges=0.5,
    ),
    Case("index dedup's bound", bound=777, weighted=True),
    Case(
        "long sequences", batch_size=3, max_len=5000, hot=0.2, world_sizes=(8, 32, 256)
    ),
    Case(
        "large",
        batch_size=4096,
        max_len=40,
        hot=0.1,
        weighted=True,
        world_sizes=(1, 8, 32, 256),
    ),
    Case(
        "300 features",
        batch_size=64,
        max_len=8,
        tables=tuple(feature // 2 for feature in range(300)),
        rules=tuple((ROUNDROBIN, HASH_ROUNDROBIN)[table % 2] for table in range(150)),
        world_sizes=(8, 129, 1024),
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


def edge_keys(dtype: torch.dtype, world_size: int) -> List[int]:
    """Keys near the edges of the rules, as ``dtype`` holds them: near 0,
    ``2^31``, ``2^32``, ``2^63`` and ``world_size``."""
    bits = 32 if dtype == torch.int32 else 64
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    near = (0, 1 << 31, 1 << 32, 1 << 63, world_size)
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
        # the edge keys, at a share of the places
        keys = torch.tensor(edge_keys(case.values, world_size))
        places = torch.nonzero(
            torch.rand(valid, generator=generator) < case.edges
        ).flatten()
        picks = torch.randint(0, keys.numel(), (places.numel(),), generator=generator)
        values[places] = keys[picks]
    weights = None
    if case.weighted:
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


def rules_of(case: Case) -> List[int]:
    """Each feature's dist_type, its table's."""
    return [case.rules[table] for table in case.tables]


def reference(
    case: Case, batch: Batch, world_size: int, dist_types
) -> Dict[str, torch.Tensor]:
    """The bucketize on the KJT as it is, as TorchRec's dists run it, and the
    keys it writes widened to int64 with their sign. Its block sizes, one
    per feature in the keys' dtype, are ones: the cyclic rules do not read
    them."""
    num_features, batch_size, valid = len(case.tables), case.batch_size, batch.valid
    new_lengths, new_values, new_weights, _, perm = block_bucketize_sparse_features(
        batch.lengths,
        batch.values,
        bucketize_pos=False,
        sequence=True,
        dist_type_per_feature=torch.tensor(dist_types, dtype=torch.int32).cuda(),
        block_sizes=torch.ones(num_features, dtype=batch.values.dtype).cuda(),
        my_size=world_size,
        weights=batch.weights,
        batch_size_per_feature=None,
        max_B=-1,
    )
    length_per_bucket_feature = (
        new_lengths.cpu().long().view(world_size, num_features, batch_size).sum(2)
    )
    offset_per_bucket = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(length_per_bucket_feature.sum(1), 0),
        ]
    )
    feature_offsets = torch.cat(
        [
            torch.zeros(world_size, 1, dtype=torch.int64),
            torch.cumsum(length_per_bucket_feature, 1),
        ],
        1,
    )
    inv = torch.empty(valid, dtype=torch.int64)
    inv[perm.cpu().long()[:valid]] = torch.arange(valid)
    return {
        "keys": new_values.cpu().long()[:valid],
        "inv": inv,
        "weights": new_weights.cpu()[:valid]
        if new_weights is not None
        else torch.empty(0),
        "offset_per_bucket": offset_per_bucket,
        "feature_offsets": feature_offsets.flatten(),
        "offset_per_bucket_feature": torch.cat(
            [
                torch.zeros(1, dtype=torch.int64),
                torch.cumsum(length_per_bucket_feature.flatten(), 0),
            ]
        ),
        # each key's bucket feature, and world_size * num_features past the
        # valid keys
        "bucket_feature_of_row": torch.cat(
            [
                torch.repeat_interleave(
                    torch.arange(world_size * num_features),
                    length_per_bucket_feature.flatten(),
                ),
                torch.full((case.bound,), world_size * num_features),
            ]
        ),
    }


def bucketize(
    case: Case,
    batch: Batch,
    world_size: int,
    dist_types,
    lengths: bool = False,
) -> Dict[str, torch.Tensor]:
    """The bucketize of the batch, from its offsets, or (``lengths``) from its
    lengths alone; the input's totals from the input KJT's offsets."""
    return planner(
        hier_output_dist_bucketize,
        batch.values,
        None if lengths else batch.offsets,
        batch.lengths if lengths else None,
        case.batch_size,
        batch.weights,
        dist_types,
        world_size,
        batch.in_offsets,
    )


def run_rule_refusals() -> Dict[str, bool]:
    """The op's configuration check (``hier_output_dist_invalid``, which
    builds nothing and waits for nothing) refuses every dist_type but
    roundrobin and hash_roundrobin, and a table whose features have
    different rules; the bucketize's hook raises for those codes."""

    def why(feature_tables: List[int], codes: List[int]) -> str:
        return planner(
            hier_output_dist_invalid,
            rank=0,
            world_size=8,
            local_size=8,
            feature_tables=feature_tables,
            dist_types=codes,
            dim=32,
            dtype=torch.float32,
            weighted=False,
            node_dedup=True,
            lookup_keys=4096,
            remote_keys=0,
            proxy_bytes=16 << 20,
            timeout_s=60.0,
            out_ctas=0,
            reduce_ctas=8,
            payload_kb=512,
        )

    def bucketize_error(rules: Tuple[int, ...]) -> str:
        case = Case("a refused rule", rules=rules)
        batch = make_batch(case, 99)
        try:
            bucketize(case, batch, 8, rules_of(case))
        except RuntimeError as error:
            return str(error)
        return ""

    return {
        "cyclic rules": why([0, 0, 1], [ROUNDROBIN, ROUNDROBIN, HASH_ROUNDROBIN]) == "",
        "continuous": '"continuous"'
        in why([0, 0, 1], [ROUNDROBIN, ROUNDROBIN, CONTINUOUS]),
        "an unknown code": "code 7" in why([0, 1, 2], [ROUNDROBIN, HASH_ROUNDROBIN, 7]),
        "two rules in a table": "share a table"
        in why([0, 0, 1], [ROUNDROBIN, HASH_ROUNDROBIN, ROUNDROBIN]),
        "a rule per feature": "per feature" in why([0, 0, 1], [ROUNDROBIN, ROUNDROBIN]),
        "bucketize": f"got {CONTINUOUS}"
        in bucketize_error((ROUNDROBIN, HASH_ROUNDROBIN, CONTINUOUS))
        and bucketize_error((ROUNDROBIN, HASH_ROUNDROBIN, ROUNDROBIN)) == "",
    }


def run_case(case: Case, seed: int, world_size: int) -> Dict[str, bool]:
    batch = make_batch(case, seed, world_size)
    dist_types = rules_of(case)
    ref = reference(case, batch, world_size, dist_types)
    out = bucketize(case, batch, world_size, dist_types)
    valid, num_keys = batch.valid, batch.values.numel()
    checks = {
        "keys": out["keys"].dtype == torch.int64
        and out["keys"].numel() == num_keys
        and torch.equal(out["keys"].cpu()[:valid], ref["keys"]),
        "inv": out["inv"].dtype == torch.int32
        and torch.equal(out["inv"].cpu().long()[:valid], ref["inv"])
        and torch.equal(out["inv"].cpu().long()[valid:], torch.arange(valid, num_keys)),
        # every row, the keys past the valid ones too, is the KJT's key at inv,
        # widened to int64 with its sign
        "gather": torch.equal(out["keys"], batch.values.long()[out["inv"].long()]),
        "weights": ("weights" in out) == (batch.weights is not None)
        and (
            batch.weights is None
            or torch.equal(out["weights"].cpu()[:valid], ref["weights"])
        ),
        "offset_per_bucket": torch.equal(
            out["offset_per_bucket"].cpu(), ref["offset_per_bucket"]
        ),
        "feature_offsets": torch.equal(
            out["feature_offsets"].cpu(), ref["feature_offsets"]
        ),
        "offset_per_bucket_feature": torch.equal(
            out["offset_per_bucket_feature"].cpu(), ref["offset_per_bucket_feature"]
        )
        and out["bucket_feature_of_row"].dtype == torch.int32
        and torch.equal(
            out["bucket_feature_of_row"].cpu().long(), ref["bucket_feature_of_row"]
        ),
        "totals": out["input_length_per_key"].tolist() == batch.in_totals,
    }

    def same(other: Dict[str, torch.Tensor]) -> bool:
        return other.keys() == out.keys() and all(
            torch.equal(other[name][:valid], tensor[:valid])
            if name in ("keys", "weights")
            else torch.equal(other[name], tensor)
            for name, tensor in out.items()
        )

    checks["again"] = same(bucketize(case, batch, world_size, dist_types))
    checks["lengths"] = same(
        bucketize(case, batch, world_size, dist_types, lengths=True)
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
    # about 15,000 sub-tasks of one row, and many rows past the valid ones
    FanoutCase("a very hot row", rows=200000, valid=1000, originals=1000000, hot=0.98),
    # rows 0-2 hold no original rows, so the first run is row 3's (heavy)
    FanoutCase(
        "rows before the first run",
        rows=500,
        valid=500,
        originals=4000,
        fixed=((0, 0), (1, 0), (2, 0), (3, 70)),
    ),
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
    rows (more than HEAVY) in row order, each with its heavy slot and its
    sub-tasks of HEAVY original rows, consecutive and in entry order; and
    each heavy row's place among the bucketized keys (inv's inverse). The
    heavy slots go in row order, as fanout_by_row renumbers the kernel's."""
    counts = torch.bincount(reverse_indices, minlength=rows)
    exp_off = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(counts, 0)])
    heavy_rows = torch.nonzero(counts > HEAVY).flatten()
    subs = (counts[heavy_rows] + HEAVY - 1) // HEAVY
    heavy_slot = torch.full((rows,), -1, dtype=torch.int64)
    heavy_slot[heavy_rows] = torch.arange(heavy_rows.numel())
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
        "heavy_slot": heavy_slot,
        "sub": torch.tensor(sub, dtype=torch.int64).flatten(),
        "count": torch.tensor([heavy_rows.numel(), int(subs.sum())]),
        "pos": place[heavy_rows],
        "first": torch.cumsum(subs, 0) - subs,
    }


def fanout_by_row(
    out: Dict[str, torch.Tensor],
) -> Tuple[Optional[Dict[str, torch.Tensor]], bool]:
    """The heavy tables of the kernel's fan-out with its heavy slots renumbered
    in row order, as fanout_reference numbers them (None when they cannot be),
    and whether its numbering holds: the heavy rows' slots a permutation of
    [0, heavy rows), and slot s's sub-tasks [first[s], first[s + 1]) (the
    last slot's up to count[1]) from 0 on, each slot's of its own row."""
    heavy_slot = out["heavy_slot"].cpu().long()
    slots, subs = out["count"].cpu().tolist()
    heavy_rows = torch.nonzero(heavy_slot >= 0).flatten()
    heavy_slots = heavy_slot[heavy_rows]  # of each heavy row, in row order
    if heavy_rows.numel() != slots or not torch.equal(
        torch.sort(heavy_slots).values, torch.arange(slots)
    ):
        return None, False
    first = out["first"].cpu().long()[:slots]
    ends = torch.cat([first[1:], torch.tensor([subs])])
    sub = out["sub"].cpu().long()[: 2 * subs].view(-1, 2)
    row_of_slot = torch.empty(slots, dtype=torch.int64)
    row_of_slot[heavy_slots] = heavy_rows
    sound = (
        (slots > 0 or subs == 0)
        and (slots == 0 or int(first[0]) == 0)
        and bool((ends > first).all())
        and torch.equal(sub[:, 0], torch.repeat_interleave(row_of_slot, ends - first))
    )
    if not sound:
        return None, False
    renumbered = heavy_slot.clone()
    renumbered[heavy_rows] = torch.arange(slots)
    row_subs = (ends - first)[heavy_slots]
    return {
        "heavy_slot": renumbered,
        # each slot's sub-tasks are its row's, in order: a stable sort by row
        "sub": sub[torch.sort(sub[:, 0], stable=True).indices].flatten(),
        "pos": out["pos"].cpu().long()[heavy_slots],
        "first": torch.cumsum(row_subs, 0) - row_subs,
    }, True


def run_fanout_case(case: FanoutCase, seed: int) -> Dict[str, bool]:
    generator = torch.Generator().manual_seed(seed)
    reverse_indices = reverse_indices_of(case, generator)
    inv = torch.randperm(case.rows, generator=generator).to(torch.int32)
    out = planner(hier_output_dist_fanout, reverse_indices.cuda(), inv.cuda())
    ref = fanout_reference(reverse_indices, inv.long(), case.rows)
    max_heavy_slots = case.originals // (HEAVY + 1) + 1
    max_subs = 2 * case.originals // HEAVY + 2
    # dtype and size of each table
    tables = {
        "exp_off": (torch.int32, case.rows + 1),
        "exp_idx": (torch.int32, case.originals),
        "heavy_slot": (torch.int32, case.rows),
        "sub": (torch.int64, 2 * max_subs),
        "count": (torch.int64, 2),
        "pos": (torch.int32, max_heavy_slots),
        "first": (torch.int32, max_heavy_slots),
    }
    checks = {
        "shapes": all(
            (out[name].dtype, out[name].numel()) == (dtype, size)
            for name, (dtype, size) in tables.items()
        )
    }
    exact = ("exp_off", "exp_idx", "count")
    for name in exact:
        checks[name] = torch.equal(out[name].cpu().long(), ref[name])
    # the heavy tables per row: the -1 heavy slots exactly, and each heavy
    # row's heavy slot (renumbered), pos, first sub-task and sub-tasks
    by_row, checks["heavy_slots"] = fanout_by_row(out)
    for name in ("heavy_slot", "sub", "pos", "first"):
        checks[name] = by_row is not None and torch.equal(by_row[name], ref[name])
    # again, twice, with the words the in channel keeps across calls (int64
    # [2], zeroed once): the same tables per row, and the words back at 0
    words = torch.zeros(2, dtype=torch.int64, device="cuda")
    checks["again"] = checks["words"] = True
    for _ in range(2):
        again = planner(
            hier_output_dist_fanout, reverse_indices.cuda(), inv.cuda(), words
        )
        again_by_row, sound = fanout_by_row(again)
        checks["again"] = (
            checks["again"]
            and sound
            and by_row is not None
            and all(torch.equal(again_by_row[name], by_row[name]) for name in by_row)
            and all(torch.equal(again[name], out[name]) for name in exact)
        )
        checks["words"] = checks["words"] and not bool(words.any())
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
    weighted: bool = False
    values: torch.dtype = torch.int64  # the KJT's keys
    empty: Tuple[int, ...] = ()
    hot: float = 0.0  # share of places that hold key 7, which one owner holds
    # the capacities: the node's largest part plus this (below 0: the call
    # does not fit, and the node skips it)
    lookup_slack: int = 5
    remote_slack: int = 5


NODE_CASES = [
    NodeCase("one GPU", num_nodes=1, local_size=1, num_features=1),
    NodeCase("one node of 2", num_nodes=1, local_size=2),
    NodeCase("one node of 8, weights, bound", 1, 8, weighted=True, bound=50),
    NodeCase("one node of 4, empty features", 1, 4, num_features=5, empty=(0, 3)),
    NodeCase("one node, exact fit", 1, 4, lookup_slack=0),
    NodeCase("one node, lookup_keys overflow", 1, 4, lookup_slack=-1),
    # The hot key makes one part the largest by far, so that part alone
    # overflows, by one key. Of 2 nodes of 4, rank 5 (on node 1) owns key 7:
    # node 1 holds its keys in an own part, and node 0 in a remote part.
    NodeCase(
        "one node, hot key: one own part overflows by one, bound",
        1,
        4,
        hot=0.5,
        bound=20,
        lookup_slack=-1,
    ),
    NodeCase("2 nodes of 2", num_nodes=2, local_size=2, num_features=2),
    NodeCase("2 nodes of 8, node 1, weights", 2, 8, node=1, weighted=True),
    NodeCase("3 nodes of 4, node 2, bound", 3, 4, num_features=2, node=2, bound=40),
    NodeCase("4 nodes of 3, node 1", 4, 3, num_features=1, node=1),
    NodeCase("2 nodes of 1", num_nodes=2, local_size=1),
    NodeCase("4 nodes of 8", 4, 8, num_features=4, node=3),
    # many features: the rows past one CTA's 64 KiB, the items past its share
    # (node_exchange's grids); odd local_size * num_features
    NodeCase("one node of 3, 127 features", 1, 3, num_features=127, max_len=2),
    NodeCase(
        "2 nodes of 8, 300 features, node 1", 2, 8, num_features=300, node=1, max_len=2
    ),
    NodeCase(
        "3 nodes of 3, 129 features, node 2", 3, 3, num_features=129, node=2, max_len=2
    ),
    NodeCase("2 nodes, exact fit", 2, 4, node=1, lookup_slack=0, remote_slack=0),
    NodeCase("2 nodes, remote_keys overflow", 2, 4, remote_slack=-1),
    NodeCase("2 nodes, lookup_keys overflow", 2, 4, node=1, lookup_slack=-1),
    NodeCase(
        "2 nodes, node 1, hot key: one own part overflows by one",
        2,
        4,
        node=1,
        hot=0.5,
        lookup_slack=-1,
    ),
    NodeCase(
        "2 nodes, hot key: one remote part overflows by one, weights",
        2,
        4,
        hot=0.5,
        weighted=True,
        remote_slack=-1,
    ),
    NodeCase("one node of 40, 70 features", 1, 40, num_features=70, max_len=3),
    NodeCase("2 nodes of 36, 70 features", 2, 36, num_features=70, node=1, max_len=3),
    NodeCase("40 nodes of 2", 40, 2, num_features=3, node=33),
    # the key range wraps to int32 keys, negative ones among them: the push
    # widens each with its sign, as the bucketize's gather does
    NodeCase("2 nodes, int32 keys", 2, 4, values=torch.int32),
]


def gathered(values: List[torch.Tensor], rows: List[Tuple[int, ...]]) -> torch.Tensor:
    """``values[requester][row]`` of each ``(requester, row, ...)`` of
    ``rows``, in order."""
    offsets = [0]
    for tensor in values:
        offsets.append(offsets[-1] + tensor.numel())
    index = [offsets[source] + row for source, row, *_ in rows]
    return torch.cat(values)[torch.tensor(index, dtype=torch.int64)]


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
        key_range=(-(1 << 40), 1 << 40),
        values=case.values,
        weighted=case.weighted,
        empty=case.empty,
        hot=case.hot,
        bound=case.bound,
    )
    dist_types = rules_of(kjt)
    batches = [
        make_batch(kjt, seed * 100 + local_rank) for local_rank in range(local_size)
    ]
    ranks = [bucketize(kjt, batch, world_size, dist_types) for batch in batches]
    valid = [batch.valid for batch in batches]
    offset_per_bucket_feature = [
        bucketized["offset_per_bucket_feature"].tolist() for bucketized in ranks
    ]
    # The node's offsets, as node_exchange gathers them: row r is requester
    # r's offset_per_bucket_feature.
    node_offsets = torch.stack(
        [bucketized["offset_per_bucket_feature"] for bucketized in ranks]
    )
    # The reference: each requester's keys of each owner and feature, counted
    # from its rows. [requester][owner node][target][feature].
    counts = torch.stack(
        [
            torch.bincount(
                bucketized["bucket_feature_of_row"].cpu()[: valid[local_rank]].long(),
                minlength=world_size * num_features,
            )
            for local_rank, bucketized in enumerate(ranks)
        ]
    ).view(local_size, num_nodes, local_size, num_features)
    same_counts = counts[:, node]  # [requester][target][feature]
    # [requester][target]: its keys for the owners on the target's rail on the
    # other nodes, which go to the target's remote part
    remote_counts = counts.sum((1, 3)) - same_counts.sum(2)
    own_lengths = same_counts.permute(1, 2, 0).reshape(local_size, -1)
    own_total, remote_total = own_lengths.sum(1), remote_counts.sum(0)
    need_lookup, need_remote = int(own_total.max()), int(remote_total.max())
    lookup_keys = need_lookup + case.lookup_slack
    remote_keys = need_remote + case.remote_slack if num_nodes > 1 else 0
    skip = need_lookup > lookup_keys or need_remote > remote_keys
    generator = torch.Generator().manual_seed(seed)

    def layout_of(local_rank: int, **owner) -> Dict[str, torch.Tensor]:
        """A rank's layout from the node's offsets, on the GPU. On one node,
        given its own part and its src, also its KJT."""
        return planner(
            hier_output_dist_node_layout,
            node_offsets,
            num_nodes,
            local_size,
            node,
            local_rank,
            num_features,
            lookup_keys,
            remote_keys,
            **owner,
        )

    # Each rank's layout: on the GPU for its push, and on the host.
    on_gpu = [layout_of(local_rank) for local_rank in range(local_size)]
    layouts = [
        {name: tensor.cpu() for name, tensor in layout.items()} for layout in on_gpu
    ]
    # node_exchange lays the items out on its one CTA or on a grid after it:
    # both give the same tables
    both_paths = all(
        all(
            torch.equal(tensor.cpu(), layouts[local_rank][name])
            for name, tensor in layout_of(local_rank, items_in_cta=in_cta).items()
        )
        for local_rank in range(local_size)
        for in_cta in (True, False)
    )
    checks = {
        "items on a CTA or a grid": both_paths,
        "own_lengths": all(
            torch.equal(layout["own_lengths"], own_lengths[local_rank])
            and torch.equal(
                layout["own_offsets"],
                torch.cumsum(own_lengths[local_rank], 0) - own_lengths[local_rank],
            )
            and torch.equal(layout["own_total"], own_total)
            for local_rank, layout in enumerate(layouts)
        ),
        "needs": all(
            int(layout["need_lookup"]) == need_lookup and bool(layout["skip"]) == skip
            for layout in layouts
        ),
    }
    if num_nodes > 1:  # one node has no remote part
        checks["remote_offsets"] = all(
            torch.equal(
                layout["remote_offsets"],
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
    else:  # one node: a rank's own part is its KJT as owner
        checks["kjt_offsets"] = all(
            torch.equal(
                layout["kjt_offsets"],
                torch.cat(
                    [
                        torch.zeros(1, dtype=torch.int64),
                        own_lengths[local_rank].cumsum(0),
                    ]
                ),
            )
            for local_rank, layout in enumerate(layouts)
        )
        checks["length_per_key"] = all(
            torch.equal(
                layout["length_per_key"],
                own_lengths[local_rank].view(num_features, local_size).sum(1),
            )
            for local_rank, layout in enumerate(layouts)
        )

    def rows_of(source: int, owner: int, feature: int) -> range:
        bucket_feature = owner * num_features + feature
        return range(*offset_per_bucket_feature[source][bucket_feature:][:2])

    def own_order(target: int) -> List[Tuple[int, int]]:
        """(requester, row) of each key of a target's own part, in the
        owner's KJT order: [feature][requester][requester's order]."""
        owner = node * local_size + target
        return [
            (source, row)
            for feature in range(num_features)
            for source in range(local_size)
            for row in rows_of(source, owner, feature)
        ]

    def remote_order(target: int) -> List[Tuple[int, int, int]]:
        """(requester, row, bucket feature) of each key of a target's remote
        part: requester-major, then owner node, then [feature][position]. The
        bucket feature is owner node * num_features + feature."""
        return [
            (source, row, owner_node * num_features + feature)
            for source in range(local_size)
            for owner_node in range(num_nodes)
            if owner_node != node
            for feature in range(num_features)
            for row in rows_of(source, owner_node * local_size + target, feature)
        ]

    if not skip:
        # Where each requester's valid rows are in its targets' parts, as its
        # layout says (row + row_shift): rows of (target, place, requester,
        # row).
        pushes = []
        for local_rank in range(local_size):
            bucket_feature = (
                ranks[local_rank]["bucket_feature_of_row"]
                .cpu()[: valid[local_rank]]
                .long()
            )
            rows = torch.arange(valid[local_rank])
            pushes.append(
                torch.stack(
                    [
                        bucket_feature // num_features % local_size,
                        layouts[local_rank]["row_shift"][bucket_feature] + rows,
                        torch.full_like(rows, local_rank),
                        rows,
                    ],
                    1,
                )
            )
        pushes = torch.cat(pushes)
        places = pushes[:, 0] * (lookup_keys + remote_keys) + pushes[:, 1]
        checks["one row a place"] = places.unique().numel() == places.numel()
        own_good, remote_good = True, True
        for target in range(local_size):
            mine = pushes[pushes[:, 0] == target]
            own = mine[mine[:, 1] < lookup_keys]
            own = own[own[:, 1].argsort()]
            own_good &= torch.equal(
                own[:, 1], torch.arange(int(own_total[target]))
            ) and own[:, 2:].tolist() == [list(entry) for entry in own_order(target)]
            remote = mine[mine[:, 1] >= lookup_keys]
            remote = remote[remote[:, 1].argsort()]
            remote_good &= torch.equal(
                remote[:, 1] - lookup_keys, torch.arange(int(remote_total[target]))
            ) and remote[:, 2:].tolist() == [
                [source, row] for source, row, _ in remote_order(target)
            ]
        checks["own parts"] = own_good
        checks["remote parts"] = remote_good

    # Every requester pushes its valid rows into one set of stand-ins for the
    # node's targets (row t: target t's node_keys, node_key_weights and
    # node_bucket_feature_of_row), which hold garbage. Each stand-in is the
    # first local_size rows of a tensor with one more row, a guard that keeps
    # its garbage: a push past the end of the last target's row stores there.
    # On one node each requester also writes its map into its owners' KJTs
    # (src) over garbage.
    entries = lookup_keys + remote_keys  # of a target's node_keys
    widths = dict(node_keys=(entries, torch.int64))
    if case.weighted:
        widths["node_key_weights"] = (entries, torch.float32)
    if num_nodes > 1:
        widths["node_bucket_feature_of_row"] = (remote_keys, torch.int32)
    guarded = {
        name: garbage((local_size + 1, width), dtype, generator)
        for name, (width, dtype) in widths.items()
    }
    held = {name: tensor.cpu() for name, tensor in guarded.items()}
    srcs = [
        garbage((bucketized["inv"].numel(),), torch.int32, generator)
        if num_nodes == 1
        else None
        for bucketized in ranks
    ]
    for local_rank, bucketized in enumerate(ranks):
        planner(
            hier_output_dist_node_push,
            batches[local_rank].values,
            bucketized["inv"],
            batches[local_rank].weights,
            bucketized["bucket_feature_of_row"],
            bucketized["offset_per_bucket"],
            on_gpu[local_rank]["row_shift"],
            on_gpu[local_rank]["skip"],
            num_nodes,
            local_size,
            node,
            local_rank,
            num_features,
            lookup_keys,
            remote_keys,
            src=srcs[local_rank],
            **{name: tensor[:local_size] for name, tensor in guarded.items()},
        )

    # Each target's row once every requester pushed. Its own part holds its
    # node's keys in its KJT order. Across nodes its remote part holds its
    # node's keys for its rail requester-major, then owner node, then
    # [feature][position], with their bucket features. Every other entry and
    # the guard row keep their garbage. Under skip no requester pushes, and
    # every entry keeps its garbage.
    got = {name: tensor.cpu() for name, tensor in guarded.items()}
    keys = [bucketized["keys"].cpu() for bucketized in ranks]
    weights = (
        [bucketized["weights"].cpu() for bucketized in ranks] if case.weighted else None
    )
    own_good, remote_good, kept = True, True, True
    for target in range(local_size):
        own = [] if skip else own_order(target)
        remote = [] if skip else remote_order(target)
        remote_end = lookup_keys + len(remote)
        for name, values in (("node_keys", keys), ("node_key_weights", weights)):
            if name not in got:
                continue
            row, held_row = got[name][target], held[name][target]
            own_good &= same_bits(row[: len(own)], gathered(values, own))
            remote_good &= same_bits(
                row[lookup_keys:remote_end], gathered(values, remote)
            )
            kept &= same_bits(
                row[len(own) : lookup_keys], held_row[len(own) : lookup_keys]
            ) and same_bits(row[remote_end:], held_row[remote_end:])
        if num_nodes > 1:
            row = got["node_bucket_feature_of_row"][target]
            held_row = held["node_bucket_feature_of_row"][target]
            bucket_features = [bucket_feature for _, _, bucket_feature in remote]
            remote_good &= same_bits(
                row[: len(remote)], torch.tensor(bucket_features, dtype=torch.int32)
            )
            kept &= same_bits(row[len(remote) :], held_row[len(remote) :])
    kept &= all(same_bits(got[name][-1], held[name][-1]) for name in got)
    if not skip:
        checks["pushed own parts"] = own_good
        if num_nodes > 1:
            checks["pushed remote parts"] = remote_good
    checks["parts untouched" if skip else "untouched tails"] = kept
    if num_nodes > 1:
        return checks

    # One node: each requester's map into its owners' KJTs, which its push
    # writes under skip too: a valid row's place in its owner's KJT, and a
    # row past them its own index.
    place = [
        torch.full((valid[local_rank],), -1, dtype=torch.int64)
        for local_rank in range(local_size)
    ]
    for target in range(local_size):
        for position, (source, row) in enumerate(own_order(target)):
            place[source][row] = position
    checks["src"] = all(
        src.dtype == torch.int32
        and torch.equal(
            src.cpu().long(),
            torch.cat(
                [place[local_rank], torch.arange(valid[local_rank], src.numel())]
            ),
        )
        for local_rank, src in enumerate(srcs)
    )
    # Each owner's KJT from its own part once every requester pushed: its
    # lengths and offsets, and a copy of its keys (none under skip). The copy
    # is a tensor of its own, because the node's next call rewrites node_keys.
    owner_good = True
    for target in range(local_size):
        own_part = {
            name: tensor[target, :lookup_keys] for name, tensor in guarded.items()
        }
        out = layout_of(target, src=srcs[target], **own_part)
        copied = (
            out["kjt_values"].untyped_storage().data_ptr()
            != guarded["node_keys"].untyped_storage().data_ptr()
        )
        out = {name: tensor.cpu() for name, tensor in out.items()}
        own, lengths = own_order(target), own_lengths[target]
        owner_good &= (
            copied
            and out["kjt_values"].shape == (lookup_keys,)
            and int(out["num_recv"]) == len(own)
            and torch.equal(out["kjt_lengths"], lengths)
            and torch.equal(
                out["kjt_offsets"],
                torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]),
            )
            and torch.equal(
                out["length_per_key"], lengths.view(num_features, local_size).sum(1)
            )
            and torch.equal(out["src"], srcs[target].cpu())
            and ("kjt_weights" in out) == bool(weights)
            and (
                skip  # nothing copied
                or same_bits(out["kjt_values"][: len(own)], gathered(keys, own))
                and (
                    not weights
                    or same_bits(out["kjt_weights"][: len(own)], gathered(weights, own))
                )
            )
        )
    checks["owner KJTs"] = owner_good
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
    failures = int(not report("rule refusals", run_rule_refusals()))
    for case_idx, case in enumerate(CASES):
        for world_size in case.world_sizes:
            checks = run_case(case, 100 + case_idx, world_size)
            label = f"bucketize '{case.name}' world_size={world_size}"
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
