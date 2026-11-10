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
  as the push gathers them. A second run gives the same bits, and so does
  the radix sort by bucket feature (``sorted_path``), which the counting sort
  by owner replaces up to 2047 owners. The in channel's bucketize on each
  node of a sharding (``num_nodes``, ``local_size``, ``node``) gives the same
  tables, its offsets per bucket feature at each owner's first one and at
  every one of its node's owners (``node_entries``), the entries
  ``node_exchange`` reads. The hook says which sort ran (``counted``): the
  counting sort wherever its shared memory fits a CTA of the GPU
  (``counting_runs``). The test scrubs each call's tables once it compared
  them (``scrub``): the next call of the same sizes gets the same blocks
  back from the caching allocator, and an entry it does not write then
  holds -1, not the right value of the call before;
* the counting sort against the radix sort on a grid of features (1 to
  4,096), owners (1 to 2,048) and keys (0 to about a million), bit for bit,
  whole and on every node. Forced (``sorted_path`` False), the counting sort
  runs exactly where the bucketize chooses it, and elsewhere raises with the
  reason: more than 2,047 owners (2,048 take the radix sort), or more shared
  memory than a CTA of the GPU takes. Then the shapes it is for
  (``PRODUCTION``: 4,096 features on 2, 4 and 64 nodes of 8, at up to 12.6M
  keys, so its CTAs take many sub-tiles each), on the first, a middle and
  the last node; KJTs whose features start on the bounds of the counting
  sort's chunks and beside them, with 1 to 4,096 features: its CTAs find
  those features with a warp's 32 probes a round; and with a second GPU, a
  grid point there after the first GPU's (the shared memory attribute is
  per GPU);
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
  ``offset_per_bucket_feature`` stacked, each as its in channel writes it,
  at ``node_entries``, and garbage elsewhere: a read of another entry
  shows). It checks each rank's layout
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

The counting sort shares shared memory between its threads and warps; check
it with racecheck and synccheck too. With the caching allocator off,
initcheck reports an entry of a node's offsets per bucket feature that the
test compares and no kernel wrote, whatever the memory held. Under these
tools the whole file can take hours: ``sanitize`` runs the bucketize's cases
on fewer and smaller calls (each case at its first and last world size, the
grid's points of at most 70,000 keys, the chunk-bound cases, one KJT whose
CTAs take two sub-tiles each, and the first, a middle and the last node of
each sharding)::

    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    compute-sanitizer --tool racecheck python test/unit_tests/test_hier_bucketize.py sanitize

and so with ``--tool synccheck`` and ``--tool initcheck``; memcheck takes
it too, for a quicker run than the whole file's.
"""

import itertools
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
    Case("all keys in one feature", empty=(0, 1, 3), max_len=40, bound=50),
    # key 0 has owner 0 under both rules: one owner takes every key
    Case("one owner takes every key", key_range=(0, 1), bound=3, weighted=True),
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
    node: Optional[Tuple[int, int, int]] = None,
    sorted_path: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """The bucketize of the batch, from its offsets, or (``lengths``) from its
    lengths alone; the input's totals from the input KJT's offsets. node:
    (num_nodes, local_size, node), the bucketize of the in channel of a rank
    of that node (its offsets per bucket feature only at node_entries); none,
    the whole table. sorted_path: the radix sort by bucket feature (True),
    the counting sort (False), or as the sizes choose (None). The hook gets
    only the arguments given (a stand-in of the hook may take no others)."""
    given = {}
    if node is not None:
        given.update(zip(("num_nodes", "local_size", "node"), node))
    if sorted_path is not None:
        given["sorted_path"] = sorted_path
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
        **given,
    )


def node_entries(
    world_size: int, num_features: int, local_size: int, node: int
) -> torch.Tensor:
    """The entries of offset_per_bucket_feature that the in channel of a rank
    of ``node`` writes, those node_exchange reads: each owner's first one, and
    every one of its node's owners (on the GPU)."""
    first = node * local_size * num_features
    return (
        torch.cat(
            [
                torch.arange(world_size + 1) * num_features,
                torch.arange(first, first + local_size * num_features + 1),
            ]
        )
        .unique()
        .cuda()
    )


def nodes_of(world_size: int) -> Tuple[int, int]:
    """(num_nodes, local_size) of a test's sharding of world_size ranks: the
    GPUs per node nearest 8, of at most 128 nodes."""
    local_size = min(
        (
            gpus
            for gpus in range(1, world_size + 1)
            if world_size % gpus == 0 and world_size // gpus <= 128
        ),
        key=lambda gpus: (abs(gpus - 8), gpus),
    )
    return world_size // local_size, local_size


def same_tables(
    got: Dict[str, torch.Tensor],
    want: Dict[str, torch.Tensor],
    valid: int,
    entries: Optional[torch.Tensor] = None,
) -> bool:
    """Whether two bucketizes of one batch give the same tables, bit for bit
    (whichever sort made them, ``counted``): the keys and weights up to the
    valid ones, the others whole. entries: a node's bucketize (no
    feature_offsets), its offset_per_bucket_feature compared there only."""
    names = set(want) - {"counted"}
    if entries is not None:
        names -= {"feature_offsets"}
    if set(got) - {"counted"} != names:
        return False
    for name in names:
        mine, theirs = got[name], want[name]
        if name in ("keys", "weights"):
            mine, theirs = mine[:valid], theirs[:valid]
        elif name == "offset_per_bucket_feature" and entries is not None:
            mine, theirs = mine[entries], theirs[entries]
        if not torch.equal(mine, theirs):
            return False
    return True


def scrub(tables: Dict[str, torch.Tensor]) -> None:
    """-1 into every table of a bucketize the test is done with (all but the
    host's counted), before the caching allocator takes its blocks back. The
    next call of the same sizes gets those blocks, so an entry it does not
    write holds -1, which no offset, row or bucket feature is, and not the
    right value this call wrote. Never a table the test still compares
    against."""
    for name, tensor in tables.items():
        if name != "counted":
            tensor.fill_(-1)


# The most owners of the counting sort (bucketize.cu, kMaxCountingBuckets - 1).
MAX_COUNTING_OWNERS = 2047
# The keys of a sub-tile of the counting sort (kThreads * kSortItems).
SUB_TILE = 4096
# The most static shared memory of a CTA of the counting sort's second kernel
# (owner_scatter_kernel: 4,448 bytes on sm_90), which takes its dynamic
# shared memory besides.
SCATTER_STATIC_SMEM = 8 << 10


def scatter_smem(world_size: int, owners: int) -> int:
    """The dynamic shared memory of a CTA of the counting sort's second kernel
    (bucketize.cu, scatter_smem), which writes the offsets per bucket feature
    of ``owners`` owners."""
    buckets = (world_size + 1 + 3) // 4 * 4
    return 4 * (2 * SUB_TILE + 3 * buckets + 1 + owners) + 2 * (SUB_TILE + 16 * buckets)


def counting_runs(world_size: int, owners: int) -> Optional[bool]:
    """Whether the bucketize takes the counting sort, as the sizes choose:
    up to MAX_COUNTING_OWNERS, where its shared memory fits a CTA of this GPU
    (``shared_memory_per_block_optin``). None where the test cannot tell:
    within SCATTER_STATIC_SMEM of the GPU's most, or a PyTorch that does not
    give it."""
    if world_size > MAX_COUNTING_OWNERS:
        return False
    most = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        "shared_memory_per_block_optin",
        None,
    )
    if most is None:
        return None
    need = scatter_smem(world_size, owners)
    if need + SCATTER_STATIC_SMEM <= most:
        return True
    return False if need > most else None


def took(tables: Dict[str, torch.Tensor], counting: Optional[bool]) -> bool:
    """Whether a bucketize took the sort ``counting`` names: the counting
    sort (True), the radix sort (False), or either (None)."""
    return counting is None or bool(tables["counted"]) == counting


def run_nodes(
    case: Case,
    batch: Batch,
    world_size: int,
    dist_types,
    whole: Dict[str, torch.Tensor],
    every_node: bool = True,
) -> bool:
    """The in channel's bucketize on each node of a sharding of world_size
    ranks (nodes_of) against the whole one: the same tables, and at the
    node's entries the same offsets per bucket feature; the counting sort
    where it runs for the node's owners. Past MAX_COUNTING_OWNERS, where the
    radix sort writes every entry, and not every_node: the first, a middle
    and the last node. Each node's tables are scrubbed once compared."""
    num_nodes, local_size = nodes_of(world_size)
    num_features = len(dist_types)
    counting = counting_runs(world_size, local_size)
    nodes = range(num_nodes)
    if world_size > MAX_COUNTING_OWNERS or not every_node:
        nodes = sorted({0, num_nodes // 2, num_nodes - 1})
    good = True
    for node in nodes:
        got = bucketize(
            case, batch, world_size, dist_types, node=(num_nodes, local_size, node)
        )
        entries = node_entries(world_size, num_features, local_size, node)
        good = (
            good
            and took(got, counting)
            and same_tables(got, whole, batch.valid, entries)
        )
        scrub(got)
    return good


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


def run_case(
    case: Case, seed: int, world_size: int, every_node: bool = True
) -> Dict[str, bool]:
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
        "counted": took(out, counting_runs(world_size, world_size)),
    }
    # Again, from the lengths, and by the radix sort: the same tables, the
    # first two by the same sort; each scrubbed once compared.
    for check, given, counting in (
        ("again", {}, bool(out["counted"])),
        ("lengths", {"lengths": True}, bool(out["counted"])),
        ("sorted", {"sorted_path": True}, False),
    ):
        got = bucketize(case, batch, world_size, dist_types, **given)
        checks[check] = took(got, counting) and same_tables(got, out, valid)
        scrub(got)
    checks["nodes"] = run_nodes(case, batch, world_size, dist_types, out, every_node)
    return checks


# ── the counting sort against the radix sort ─────────────────────────────────

# Features, owners and keys: each pair of features and owners, with the keys
# in turn. 2,048 owners take the radix sort.
GRID_FEATURES = (1, 8, 300, 1024, 4096)
GRID_WORLD_SIZES = (1, 8, 16, 129, 512, 1024, 2047, 2048)
GRID_KEYS = (0, 31, 4095, 4096, 4097, 70_000, 1_000_000)
GRID = [
    (num_features, world_size, GRID_KEYS[index % len(GRID_KEYS)])
    for index, (num_features, world_size) in enumerate(
        itertools.product(GRID_FEATURES, GRID_WORLD_SIZES)
    )
]
# The shapes the counting sort is for, past the grid's: 4,096 features on 2,
# 4 and 64 nodes of 8, at about the keys of a GPU (4.2M, and 12.6M), so its
# CTAs take many sub-tiles each (4, 4 and 24 at 132 SMs). Each with the seed
# of its KJT (run_grid_case): the first, uniform features with int32 keys,
# weights and keys past the valid ones; the others, no keys in the first and
# the last feature, with int64 keys, and the last with keys past the valid
# ones.
PRODUCTION = (
    (4096, 16, 4_200_000, 603),
    (4096, 32, 3_300_000, 602),
    (4096, 512, 12_600_000, 605),
)


def feature_batch(
    starts: List[int],
    num_keys: int,
    batch_size: int,
    seed: int,
    values: torch.dtype = torch.int64,
    weighted: bool = False,
) -> Tuple[Case, Batch]:
    """A KJT of len(starts) - 1 features whose keys start at ``starts`` (the
    last entry: the valid keys), num_keys keys in all (index dedup's keys
    past the valid ones after them), each feature's keys over random samples;
    a rule per feature, roundrobin or hash_roundrobin at random, some keys at
    the edges of the rules."""
    generator = torch.Generator().manual_seed(seed)
    num_features, valid = len(starts) - 1, starts[-1]
    feature = torch.repeat_interleave(
        torch.arange(num_features), torch.tensor(starts).diff()
    )
    sample = torch.randint(0, batch_size, (valid,), generator=generator)
    lengths = torch.bincount(
        feature * batch_size + sample, minlength=num_features * batch_size
    )
    keys = torch.randint(
        -(1 << 40), 1 << 40, (num_keys,), generator=generator, dtype=torch.int64
    )
    keys[torch.rand(num_keys, generator=generator) < 0.2] = 7
    edges = torch.tensor(edge_keys(values, 2048))
    places = torch.nonzero(torch.rand(num_keys, generator=generator) < 0.1).flatten()
    keys[places] = edges[
        torch.randint(0, edges.numel(), (places.numel(),), generator=generator)
    ]
    rules = tuple(
        (ROUNDROBIN, HASH_ROUNDROBIN)[rule]
        for rule in torch.randint(0, 2, (num_features,), generator=generator).tolist()
    )
    in_lengths = lengths + torch.randint(0, 3, lengths.shape, generator=generator)
    weights = torch.rand(num_keys, generator=generator).cuda() if weighted else None
    case = Case(
        f"{num_features} features",
        batch_size=batch_size,
        tables=tuple(range(num_features)),
        rules=rules,
        values=values,
        weighted=weighted,
    )
    batch = Batch(
        lengths=lengths.cuda(),
        offsets=offsets_of(lengths.cuda()),
        values=keys.to(values).cuda(),
        weights=weights,
        in_offsets=offsets_of(in_lengths.cuda()),
        in_totals=in_lengths.view(num_features, batch_size).sum(1).tolist(),
        valid=valid,
    )
    return case, batch


def run_sorts(
    case: Case, batch: Batch, world_size: int, every_node: bool = True
) -> Dict[str, bool]:
    """The counting sort against the radix sort, bit for bit: whole, from the
    offsets and from the lengths, and on every node (run_nodes); as the sizes
    choose, the counting sort where counting_runs says it runs. Forced, the
    counting sort runs exactly where the sizes choose it, and elsewhere
    raises with the reason: more than MAX_COUNTING_OWNERS owners, or more
    shared memory than a CTA of this GPU takes."""
    dist_types = rules_of(case)
    valid = batch.valid
    radix = bucketize(case, batch, world_size, dist_types, sorted_path=True)
    chosen = bucketize(case, batch, world_size, dist_types)
    counted = bool(chosen["counted"])
    checks = {
        "chosen": took(radix, False)
        and took(chosen, counting_runs(world_size, world_size))
        and same_tables(chosen, radix, valid),
        "totals": chosen["input_length_per_key"].tolist() == batch.in_totals,
    }
    scrub(chosen)
    lengths = bucketize(case, batch, world_size, dist_types, lengths=True)
    checks["lengths"] = took(lengths, counted) and same_tables(lengths, radix, valid)
    scrub(lengths)
    checks["nodes"] = run_nodes(case, batch, world_size, dist_types, radix, every_node)
    try:
        forced = bucketize(case, batch, world_size, dist_types, sorted_path=False)
        checks["counting"] = (
            counted and took(forced, True) and same_tables(forced, radix, valid)
        )
        scrub(forced)
    except RuntimeError as error:
        reason = (
            f"at most {MAX_COUNTING_OWNERS} owners"
            if world_size > MAX_COUNTING_OWNERS
            else "shared memory"
        )
        checks["counting"] = not counted and reason in str(error)
    scrub(radix)
    return checks


def run_grid_case(
    num_features: int,
    world_size: int,
    num_keys: int,
    seed: int,
    every_node: bool = True,
) -> Dict[str, bool]:
    """A KJT of the grid: uniform features, all keys in one feature, or empty
    features at both ends, in turn; index dedup's keys past the valid ones in
    every other case; int32 keys and weights in every third."""
    generator = torch.Generator().manual_seed(seed)
    valid = num_keys - (num_keys // 7 if seed % 2 else 0)
    pattern = seed % 3
    if pattern == 0 or num_features < 3:  # uniform
        feature = torch.randint(0, num_features, (valid,), generator=generator)
    elif pattern == 1:  # one feature
        feature = torch.full((valid,), num_features // 2)
    else:  # none in the first and last feature
        feature = torch.randint(1, num_features - 1, (valid,), generator=generator)
    starts = [0] + torch.bincount(feature, minlength=num_features).cumsum(0).tolist()
    case, batch = feature_batch(
        starts,
        num_keys,
        batch_size=4 if num_features > 512 else 64,
        seed=seed,
        values=torch.int32 if seed % 3 == 0 else torch.int64,
        weighted=seed % 3 == 0,
    )
    return run_sorts(case, batch, world_size, every_node)


# Features of the KJTs whose features start on the bounds of the counting
# sort's chunks: its CTAs find the features of their bounds by a warp's 32
# probes a round.
SEARCH_FEATURES = (1, 31, 32, 33, 1023, 1024, 4096)
SEARCH_KEYS = 70_000


def counting_chunk(num_keys: int, world_size: int) -> int:
    """The keys of a CTA of the counting sort (bucketize.cu, counting_grid)."""
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    buckets = (world_size + 1 + 3) // 4 * 4
    ctas = min(max(-(-num_keys // SUB_TILE), 1), (2 if buckets <= 128 else 1) * sms)
    chunk = -(-max(num_keys, 1) // ctas)
    return -(-chunk // 32) * 32


def run_search_case(
    num_features: int, world_size: int, seed: int, every_node: bool = True
) -> Dict[str, bool]:
    """A KJT of SEARCH_KEYS keys whose features start at the bounds of the
    counting sort's chunks, one key before and one after; the others at
    random places, some features empty. Without keys past the valid ones (the
    last feature ends at the last key), or (odd seeds) with them from a
    chunk's bound on."""
    generator = torch.Generator().manual_seed(seed)
    chunk = counting_chunk(SEARCH_KEYS, world_size)
    bounds = list(range(chunk, SEARCH_KEYS, chunk))
    valid = bounds[len(bounds) // 2] if seed % 2 else SEARCH_KEYS
    near = sorted(
        {
            bound + step
            for bound in bounds
            for step in (-1, 0, 1)
            if 0 < bound + step < valid
        }
    )
    inner = num_features - 1
    if inner <= len(near):
        picks = torch.linspace(0, len(near) - 1, inner).round().long().tolist()
        starts = [near[pick] for pick in picks]
    else:
        starts = (
            near
            + torch.randint(
                0, valid + 1, (inner - len(near),), generator=generator
            ).tolist()
        )
    case, batch = feature_batch(
        [0] + sorted(starts) + [valid], SEARCH_KEYS, batch_size=2, seed=seed
    )
    return run_sorts(case, batch, world_size, every_node)


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
    # Each requester's whole bucketize, which the reference reads (its
    # offsets per bucket feature at every entry) and which it pushes.
    ranks = [bucketize(kjt, batch, world_size, dist_types) for batch in batches]
    valid = [batch.valid for batch in batches]
    offset_per_bucket_feature = [
        bucketized["offset_per_bucket_feature"].tolist() for bucketized in ranks
    ]
    generator = torch.Generator().manual_seed(seed)
    # The node's offsets, as node_exchange gathers them: row r is requester
    # r's offset_per_bucket_feature as its in channel writes it, at the
    # entries the node layout reads (node_entries), and garbage at the
    # others, so a read of another entry shows. Only those entries are
    # copied: the bucketize does not write the others.
    contract = node_entries(world_size, num_features, local_size, node)
    node_offsets = garbage(
        (local_size, world_size * num_features + 1), torch.int64, generator
    )
    for local_rank, batch in enumerate(batches):
        in_channel = bucketize(
            kjt, batch, world_size, dist_types, node=(num_nodes, local_size, node)
        )
        node_offsets[local_rank, contract] = in_channel["offset_per_bucket_feature"][
            contract
        ]
        scrub(in_channel)
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
    # sanitize: the bucketize's cases on fewer and smaller calls, for
    # compute-sanitizer (the module's docstring)
    sanitize = sys.argv[1:] == ["sanitize"]
    every_node = not sanitize
    torch.cuda.set_device(0)
    failures = 0
    if not sanitize:
        failures += not report("rule refusals", run_rule_refusals())
    for case_idx, case in enumerate(CASES):
        world_sizes = case.world_sizes
        if sanitize:
            world_sizes = sorted({world_sizes[0], world_sizes[-1]})
        for world_size in world_sizes:
            checks = run_case(case, 100 + case_idx, world_size, every_node)
            label = f"bucketize '{case.name}' world_size={world_size}"
            failures += not report(label, checks)
    grid = [
        (num_features, world_size, num_keys, 400 + case_idx)
        for case_idx, (num_features, world_size, num_keys) in enumerate(GRID)
        if not sanitize or num_keys <= SEARCH_KEYS
    ]
    if sanitize:
        # CTAs of two sub-tiles each, the second of 32 keys (counting_grid):
        # the fewest keys whose CTAs carry a sub-tile's state to the next
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        grid.append((300, 129, sms * (SUB_TILE + 32), 453))
    else:
        grid += PRODUCTION
    for num_features, world_size, num_keys, seed in grid:
        checks = run_grid_case(num_features, world_size, num_keys, seed, every_node)
        label = (
            f"sorts {num_features} features, world_size={world_size}, {num_keys} keys"
        )
        failures += not report(label, checks)
    for case_idx, num_features in enumerate(SEARCH_FEATURES):
        # with keys past the valid ones (an odd seed) at 129 owners only
        for tail, world_size in enumerate((16, 129)):
            checks = run_search_case(
                num_features, world_size, 500 + 2 * case_idx + tail, every_node
            )
            label = f"chunk bounds {num_features} features, world_size={world_size}"
            failures += not report(label, checks)
    if sanitize:
        print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
        sys.exit(1 if failures else 0)
    if torch.cuda.device_count() > 1:
        # The shared memory attribute is per GPU (bucketize.cu, takes_smem): a
        # grid point past 48 KiB on GPU 1, after GPU 0's.
        with torch.cuda.device(1):
            checks = run_grid_case(300, 129, SEARCH_KEYS, 701)
        failures += not report(
            f"sorts on GPU 1, 300 features, world_size=129, {SEARCH_KEYS} keys", checks
        )
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
