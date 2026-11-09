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
"""The planner's steps across nodes (``src/hier_output_dist/input_dist/planner.h``) on
one GPU, on plain tensors: the proxy's node dedup, the chunks of its stream
through its ring, slots and push order (``hier_output_dist_proxy_plan``), the
rail's layout, status and owner tables (``hier_output_dist_rail_tables``),
and the requester's tables (``hier_output_dist_requester_tables``).

The test plays every rank of an N-node job: each rank's bucketized rows (by
owner and feature, and past the valid ones index dedup's bound; a key's owner
follows from the key, as under roundrobin), index-deduplicated (a rank has
each (table, key) once, so a node-unique key has at most local_size
occurrences at its proxy), whose keys and weights are in the order of its KJT
(a random permutation, inv), its node layout
(``hier_output_dist_node_layout``), and between the steps what the comm
kernels do: the node's offsets (each requester's offset_per_bucket_feature),
the pushes into node_keys (``hier_output_dist_node_push``), the count
messages of each rail (each rank's message to each rail peer: a header the
same to all, so every rank of a rail lays the rail's rows out from the same
headers, then the peer's own part), the key records each proxy puts into its
owners' rail_keys, the needs and group shifts each node exchanges (the
requester binding gathers the proxies' words itself). The windows hold
garbage before the call. The checks:

* each proxy against a plain reference of the planner's rules: its
  node-unique keys (identities (table, key), each in the bucket feature
  (owner node, feature) of its representative, its occurrence of the least
  bucket feature and of those the first; inside a bucket feature the
  requesters in turn: each one's first representative, requester after
  requester, then each one's second, and so on; without node dedup each
  occurrence, requester after requester), their records, the chunks of its
  stream (each owner node's keys cut by their counted rows: a key's rows,
  but at least rows_per_payload; round after round, packed, in laps when
  they pass the ring), their bounds, and where each lies, each occurrence's
  row and slot (the first occurrence's row, then duplicate slots), push
  positions, push tables and the rows before each key; its records of each
  bucket feature against a dedup by merge ranks (each key's bucket feature
  that of its occurrence of least merge rank), as equal sets;
* each requester's maps (its rows of its node's owners: their rows in the
  owners' lookup_rows; its keys' rows and slots at its proxies), push order,
  push tables and remote task offsets;
  and per node a backward with integer gradients through the requesters'
  push tasks (each counting its rows in its chunk's pushed_rows, which must
  reach the chunk's rows once every row is in) and the proxies' reducers,
  each key's rows its key row and then its duplicates' slots;
* the rail: each proxy's slices start where its owners' offsets say; each
  owner's KJT, the chunks it got, and its lookup_rows' rows against a plain
  reference (each lookup group's keys in a slice of their own, its own part
  first, then each other node's in node order): the row map, where each
  group's slice starts, the group shifts, and the group runs of each source
  node's records, equal to that node's proxy's runs of its keys of the owner
  (the rows its backward puts land in); the needs, the same on every rank.

Calls that do not fit, and calls whose keys are not index-deduplicated (a
key past local_size occurrences: the need key_rows), check the needs alone
(and that no access was out of range). Every planner call runs under
``torch.cuda.set_sync_debug_mode("error")``.

Run it on one GPU, from ``corelib/dynamicemb``::

    python test/unit_tests/test_hier_dedup.py
"""

import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, Dict, List, Optional, Tuple

import torch
from dynamicemb_extensions import (
    hier_output_dist_node_layout,
    hier_output_dist_node_push,
    hier_output_dist_proxy_plan,
    hier_output_dist_rail_tables,
    hier_output_dist_requester_tables,
)

TASK_ROWS = 32  # kTaskRows: rows of a push task


def planner(binding, *args, **kwargs):
    """A planner binding under the sync debug mode "error": an op in it that
    waits for the device raises."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        return binding(*args, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def ceil32(count: int) -> int:
    return -(-count // TASK_ROWS)


def max_key_rows(case: "Case") -> int:
    """The most rows of one key, a row per occurrence (Sizes::max_key_rows)."""
    return case.local_size


def rows_per_payload(case: "Case") -> int:
    """The ring rows a chunk may spend per row it sends over IB, the least
    rows a key counts in a chunk (Sizes::rows_per_payload, as
    rows_per_payload_for makes it): the chunk's rows in the ring per row of
    its payload, at least 1."""
    return max(case.max_chunk_rows // case.chunk_payload_rows, 1)


def counted_rows(case: "Case", rows: int) -> int:
    """A key's counted rows in a chunk: its rows, but at least
    rows_per_payload."""
    return max(rows, rows_per_payload(case))


def chunk_span(case: "Case") -> int:
    """The counted rows one chunk spans (Sizes::chunk_span): a key counts at
    most max(max_key_rows, rows_per_payload) (Sizes::max_counted_rows)."""
    return case.max_chunk_rows - counted_rows(case, max_key_rows(case)) + 1


def max_chunks(case: "Case", remote_keys: int) -> int:
    """The most chunks of a stream (Sizes::max_chunks)."""
    return rows_per_payload(case) * remote_keys // chunk_span(case) + case.num_nodes


def max_owner_chunks(case: "Case", lookup_keys: int, remote_keys: int) -> int:
    """The most chunks an owner gets (Sizes::max_owner_chunks): a key counts at
    most rows_per_payload times its rows, and at most max_counted_rows."""
    others = case.num_nodes - 1
    return others + min(
        rows_per_payload(case) * others * remote_keys,
        counted_rows(case, max_key_rows(case)) * lookup_keys,
    ) // chunk_span(case)


@dataclass
class Case:
    name: str
    num_nodes: int = 4
    local_size: int = 4
    tables: Tuple[int, ...] = (0, 0, 1, 2)  # features 0 and 1 share table 0
    max_block: int = 6  # a requester's keys of one (owner, feature), at most
    key_range: int = 40
    hot: float = 0.0  # share of keys that are key 7
    weighted: bool = False  # the keys carry weights (whole numbers: frequency counts)
    # index dedup: a rank keeps each (table, key) once (False: the call breaks
    # the bound of local_size occurrences per key, a need)
    dedup: bool = True
    node_dedup: bool = True
    num_proxy_rows: int = 1 << 20  # small: the ring wraps
    # the most rows of a chunk (max_chunk_rows); 0: the least, 2 * max_key_rows
    max_chunk_rows_arg: int = 0
    # the rows of a chunk's payload (plan_sizes' payload_rows); 0: max_chunk_rows,
    # rows_per_payload 1 (the counted rows are the rows)
    payload_rows: int = 0
    bound: int = 0  # index dedup: rows past the valid ones
    # the capacities: the most a rank needs plus this (below 0: the call
    # does not fit)
    lookup_slack: int = 5
    remote_slack: int = 5
    job: Optional[Callable] = None  # the ranks' rows, when not random
    seeds: int = 2
    # the features of each lookup group (Config::lookup_groups); empty: one group
    groups: Tuple[int, ...] = ()

    @property
    def num_features(self) -> int:
        return len(self.tables)

    @property
    def group_ranges(self) -> List[Tuple[int, int]]:
        """Each lookup group's features [first, last)."""
        starts = list(accumulate([0] + list(self.groups or (self.num_features,))))
        return list(zip(starts, starts[1:]))

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.local_size

    @property
    def record_bytes(self) -> int:
        return 16 if self.weighted else 8

    @property
    def max_chunk_rows(self) -> int:
        """Sizes::max_chunk_rows."""
        return self.max_chunk_rows_arg or 2 * max_key_rows(self)

    @property
    def chunk_payload_rows(self) -> int:
        """The rows of a chunk's payload (plan_sizes' payload_rows)."""
        return self.payload_rows or self.max_chunk_rows


def one_owner_job(case: Case, keys_of_rank: List[List[int]]) -> List[Tuple]:
    """Each rank's rows: its keys (keys_of_rank, the ranks past them none) for
    owner (node 1, 0), feature 0."""
    num_bucket_features = case.world_size * case.num_features
    job = []
    for rank in range(case.world_size):
        keys = keys_of_rank[rank] if rank < len(keys_of_rank) else []
        counts = torch.zeros(num_bucket_features, dtype=torch.int64)
        counts[case.local_size * case.num_features] = len(keys)  # owner (node 1, 0)
        job.append(
            (counts, torch.tensor(keys, dtype=torch.int64), torch.ones(len(keys)))
        )
    return job


def second_lap_key(case: Case) -> int:
    """The first key of the second lap's first chunk in same_place_job's stream
    (its keys 0, 1, ... in key order; key 0 of two occurrences, the keys up to
    this one of one each)."""
    span, ring = chunk_span(case), case.num_proxy_rows
    lap_span = ring - case.max_chunk_rows + 1
    keys = ring + 2 * span
    first, other = counted_rows(case, 2), counted_rows(case, 1)
    key_round = [0] + [
        (first + (key - 1) * other) // span for key in range(1, keys)
    ]  # each key's round: its chunk
    rows_before = 0  # the stream's rows before each chunk
    for chunk in range(key_round[-1] + 1):
        if rows_before >= lap_span:
            return key_round.index(chunk)
        rows_before += key_round.count(chunk) + (chunk == 0)
    return 0


def same_place_job(case: Case, generator) -> List[Tuple]:
    """Requester 0 asks owner (node 1, 0) for enough keys, once each and in
    key order, for the stream to pass the ring; then the first key of the
    second lap's first chunk (second_lap_key) lies where key 0 does. Requester
    1 asks those two keys too. Requester 0's occurrences come first in the
    proxy's remote part, so they are the keys' representatives and hold their
    key rows. Requester 0 holds every representative and requester 1 none, so
    with the requesters in turn the k-th key is requester 0's k-th: the keys
    are numbered in key order, and requester 0's rows of the two keys are two
    rows of one slot in two chunks."""
    keys = case.num_proxy_rows + 2 * chunk_span(case)
    return one_owner_job(case, [list(range(keys)), [0, second_lap_key(case)]])


def wide_node_job(case: Case, generator) -> List[Tuple]:
    """More GPUs per node than a task's rows: every requester of node 0 asks
    owner (node 1, 0) for key 7 once and for key 8 once: two keys of
    local_size occurrences, the most rows of a key."""
    return one_owner_job(case, [[7, 8] for _ in range(case.local_size)])


def past_bound_job(case: Case, generator) -> List[Tuple]:
    """As wide_node_job, but requester 0 asks key 8 twice (keys not
    index-deduplicated): key 8 has one occurrence more than its rows."""
    return one_owner_job(
        case, [[7, 8, 8] if rank == 0 else [7, 8] for rank in range(case.local_size)]
    )


CASES = [
    Case("uniform"),
    # lookup groups: each group's rows in a slice of lookup_rows of their own,
    # so a chunk's rows cross in a put per group they span
    Case("uniform, 2 lookup groups", groups=(1, 3)),
    Case("uniform, 3 lookup groups", groups=(2, 1, 1), max_block=20),
    Case("a lookup group per feature", groups=(1, 1, 1, 1), max_block=20),
    Case("hot keys", max_block=40, key_range=30, hot=0.5),
    # chunks by counted rows: a key counts its rows, but at least
    # rows_per_payload (max_chunk_rows // payload_rows), so the counted rows differ
    # from the rows
    Case(
        "hot keys, 2 rows per payload",
        max_block=40,
        key_range=30,
        hot=0.5,
        max_chunk_rows_arg=40,
        payload_rows=20,
    ),
    Case(
        "weights, 3 nodes",
        num_nodes=3,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=300,
    ),
    Case(
        "weights, 3 nodes of 2, 6 rows per payload, above a key's rows",
        num_nodes=3,
        local_size=2,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=60,
        payload_rows=10,
    ),
    Case(
        "weights, 3 nodes of 2, 2 lookup groups",
        num_nodes=3,
        local_size=2,
        tables=(0, 0, 1),
        groups=(2, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=60,
        payload_rows=10,
    ),
    Case(
        "2 nodes, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=1280,
    ),
    Case(
        "2 nodes, a wrapping ring, weights",
        num_nodes=2,
        max_block=200,
        key_range=400,
        hot=0.08,
        weighted=True,
        num_proxy_rows=1280,
    ),
    Case(
        "2 nodes, a wrapping ring, 4 rows per payload",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=640,
        max_chunk_rows_arg=160,
        payload_rows=40,
    ),
    Case(
        "node dedup off",
        max_block=10,
        key_range=60,
        hot=0.3,
        weighted=True,
        node_dedup=False,
    ),
    Case(
        "node dedup off, 3 rows per payload",
        max_block=10,
        key_range=60,
        hot=0.3,
        weighted=True,
        node_dedup=False,
        max_chunk_rows_arg=30,
        payload_rows=10,
        seeds=1,
    ),
    Case(
        "2 nodes of 8, hot keys",
        num_nodes=2,
        local_size=8,
        tables=(0, 1, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
    ),
    Case(
        "2 nodes of 8, hot keys, 3 rows per payload",
        num_nodes=2,
        local_size=8,
        tables=(0, 1, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        max_chunk_rows_arg=60,
        payload_rows=20,
    ),
    Case(
        "large, a wrapping ring",
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=3840,
        seeds=1,
    ),
    Case(
        "large, weights, a wrapping ring",
        max_block=200,
        key_range=400,
        hot=0.08,
        weighted=True,
        num_proxy_rows=3840,
        seeds=1,
    ),
    Case(
        "large, weights, a wrapping ring, 8 rows per payload",
        max_block=200,
        key_range=400,
        hot=0.08,
        weighted=True,
        num_proxy_rows=960,
        max_chunk_rows_arg=400,
        payload_rows=50,
        seeds=1,
    ),
    Case(
        "one table, weights, a wrapping ring",
        num_nodes=3,
        tables=(0, 0, 0),
        max_block=250,
        key_range=3000,
        hot=0.1,
        weighted=True,
        num_proxy_rows=2560,
        seeds=1,
    ),
    Case(
        "node dedup off, 4 nodes of 2, a wrapping ring",
        local_size=2,
        tables=(0, 1),
        max_block=700,
        key_range=5000,
        node_dedup=False,
        num_proxy_rows=3840,
        seeds=1,
    ),
    Case("2 nodes of 1", num_nodes=2, local_size=1, max_block=60, hot=0.3),
    Case(
        "4 nodes of 1, hot", local_size=1, max_block=100, hot=0.4, num_proxy_rows=3840
    ),
    Case(
        "4 nodes of 1, hot, 5 rows per payload",
        local_size=1,
        max_block=100,
        hot=0.4,
        num_proxy_rows=3840,
        max_chunk_rows_arg=50,
        payload_rows=10,
    ),
    Case("4 nodes of 8", local_size=8, max_block=3, key_range=20, hot=0.1, seeds=1),
    Case("index dedup's bound", bound=50, weighted=True, hot=0.2),
    Case("an empty call", max_block=0),
    Case("an empty call, index dedup's bound", max_block=0, bound=20),
    Case(
        "2 nodes, a wrapping ring, one hot key",
        num_nodes=2,
        max_block=300,
        key_range=5000,
        hot=0.3,
        num_proxy_rows=1280,
    ),
    Case(
        "one slot in two laps of the ring",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        num_proxy_rows=1280,
        job=same_place_job,
        seeds=1,
    ),
    Case(
        "one slot in two laps of the ring, 2 rows per payload",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        num_proxy_rows=1280,
        max_chunk_rows_arg=8,
        payload_rows=4,
        job=same_place_job,
        seeds=1,
    ),
    Case("remote_keys overflow", max_block=20, hot=0.3, remote_slack=-1),
    Case(
        "remote_keys overflow, 2 rows per payload",
        max_block=20,
        hot=0.3,
        remote_slack=-1,
        max_chunk_rows_arg=16,
        payload_rows=8,
    ),
    Case("lookup_keys overflow", max_block=20, hot=0.3, lookup_slack=-1),
    Case(
        "remote_keys overflow, weights, bound",
        max_block=20,
        weighted=True,
        bound=30,
        remote_slack=-1,
    ),
    Case(
        "lookup_keys far above the call",
        num_nodes=2,
        local_size=2,
        weighted=True,
        lookup_slack=1 << 22,
        seeds=1,
    ),
    Case(
        "70 features, 3 nodes of 4, a wrapping ring",
        num_nodes=3,
        local_size=4,
        tables=tuple(feature // 3 for feature in range(70)),
        max_block=12,
        key_range=4000,
        hot=0.05,
        num_proxy_rows=2560,
        max_chunk_rows_arg=200,
        seeds=1,
    ),
    Case(
        "36 nodes of 1",
        num_nodes=36,
        local_size=1,
        tables=(0, 1),
        max_block=6,
        key_range=100,
        hot=0.2,
        num_proxy_rows=1280,
        seeds=1,
    ),
    Case(
        "2 nodes of 34",
        num_nodes=2,
        local_size=34,
        tables=(0, 0),
        max_block=2,
        key_range=30,
        hot=0.3,
        seeds=1,
    ),
    Case(
        "2 nodes of 72, keys of local_size occurrences",
        num_nodes=2,
        local_size=72,
        tables=(0,),
        remote_slack=0,
        job=wide_node_job,
        seeds=1,
    ),
    # keys not index-deduplicated: the need key_rows
    Case("keys not deduplicated", max_block=40, key_range=30, hot=0.5, dedup=False),
    Case(
        "keys not deduplicated, 2 rows per payload",
        max_block=40,
        key_range=30,
        hot=0.5,
        dedup=False,
        max_chunk_rows_arg=16,
        payload_rows=8,
    ),
    Case(
        "keys not deduplicated, weights, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=400,
        hot=0.08,
        weighted=True,
        dedup=False,
        num_proxy_rows=1280,
    ),
    Case(
        "2 nodes of 72, a key past local_size occurrences",
        num_nodes=2,
        local_size=72,
        tables=(0,),
        remote_slack=0,
        dedup=False,
        job=past_bound_job,
        seeds=1,
    ),
]


# ── the job ──────────────────────────────────────────────────────────────────


def random_job(case: Case, generator) -> List[Tuple]:
    """Each rank's rows: its keys for each bucket feature (owner, feature),
    and their weights. The keys are placed as under roundrobin: a drawn x of
    owner o is the key x * world_size + o. Index dedup (case.dedup) keeps a
    rank's first row of each (table, key)."""
    num_bucket_features = case.world_size * case.num_features
    job = []
    for _ in range(case.world_size):
        counts = torch.randint(
            0, case.max_block + 1, (num_bucket_features,), generator=generator
        )
        valid = int(counts.sum())
        keys = torch.randint(0, case.key_range, (valid,), generator=generator)
        keys[torch.rand(valid, generator=generator) < case.hot] = 7
        weights = torch.randint(1, 4, (valid,), generator=generator).float()
        bucket_feature = torch.repeat_interleave(
            torch.arange(num_bucket_features), counts
        )
        # a key's owner follows from it (roundrobin: key mod world_size), so
        # one (table, key) has one owner, as the op's rules give it
        keys = keys * case.world_size + bucket_feature // case.num_features
        if case.dedup:
            seen, keep = set(), []
            for row, key in enumerate(keys.tolist()):
                identity = (
                    case.tables[int(bucket_feature[row]) % case.num_features],
                    key,
                )
                keep.append(identity not in seen)
                seen.add(identity)
            keep = torch.tensor(keep, dtype=torch.bool)
            counts = torch.bincount(bucket_feature[keep], minlength=num_bucket_features)
            keys, weights = keys[keep], weights[keep]
        job.append((counts, keys, weights))
    return job


def bucketized_rows(case: Case, counts, keys, weights, generator) -> Dict:
    """A rank's bucketized rows as the planner takes them: by bucket feature
    (owner * num_features + feature), then index dedup's bound (garbage
    keys). Their keys and weights stay in the order of its KJT, as the
    bucketize leaves them (values, kjt_weights: the KJT's): bucketized row
    ``row`` is KJT row inv[row], a random permutation."""
    num_bucket_features = counts.numel()
    offset_per_bucket_feature = torch.cat(
        [torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]
    )
    row_keys = torch.cat(
        [keys, torch.randint(-(1 << 62), 1 << 62, (case.bound,), generator=generator)]
    )
    row_weights = torch.cat([weights, torch.full((case.bound,), 1e9)])
    inv = torch.randperm(row_keys.numel(), generator=generator)
    values, kjt_weights = torch.empty_like(row_keys), torch.empty_like(row_weights)
    values[inv], kjt_weights[inv] = row_keys, row_weights
    return dict(
        offset_per_bucket_feature_list=offset_per_bucket_feature.tolist(),
        keys_list=keys.tolist(),
        weights_list=weights.tolist(),
        valid=int(counts.sum()),
        counts=counts,
        values=values.cuda(),
        inv=inv.int().cuda(),
        kjt_weights=kjt_weights.cuda(),
        bucket_feature_of_row=torch.cat(
            [
                torch.repeat_interleave(torch.arange(num_bucket_features), counts),
                torch.full((case.bound,), num_bucket_features),
            ]
        )
        .int()
        .cuda(),
        offset_per_bucket_feature=offset_per_bucket_feature.cuda(),
        offset_per_bucket=offset_per_bucket_feature[:: case.num_features]
        .contiguous()
        .cuda(),
    )


def garbage(count: int, dtype: torch.dtype, generator) -> torch.Tensor:
    """Random bits of ``dtype``, as a window holds before a call."""
    bits = torch.randint(-(1 << 31), (1 << 31) - 1, (count,), generator=generator)
    if dtype == torch.int64:
        bits = (bits << 32) ^ torch.randint(0, 1 << 32, (count,), generator=generator)
    return bits.to(torch.int32 if dtype == torch.float32 else dtype).view(dtype).cuda()


def make_windows(
    case: Case, lookup_keys: int, remote_keys: int, generator
) -> Tuple[Dict, List[Dict]]:
    """A node's window buffers as the planner across nodes reads and writes
    them, garbage in each. node_keys, node_key_weights (with weights) and
    node_bucket_feature_of_row are one tensor each for the node (row r: rank
    r's), as the push takes its targets'. Returns those, and each rank's
    window: its row of each, and its other buffers."""
    local_size = case.local_size
    chunks = max_chunks(case, remote_keys)
    entries = local_size * (lookup_keys + remote_keys)
    buffers = dict(
        node_keys=garbage(entries, torch.int64, generator).view(local_size, -1),
        node_bucket_feature_of_row=garbage(
            local_size * remote_keys, torch.int32, generator
        ).view(local_size, remote_keys),
    )
    if case.weighted:
        buffers["node_key_weights"] = garbage(entries, torch.float32, generator).view(
            local_size, -1
        )
    windows = []
    for local_rank in range(local_size):
        window = {name: rows[local_rank] for name, rows in buffers.items()}
        window.update(
            remote_map=garbage(3 * remote_keys, torch.int32, generator),
            key_stage=garbage(
                chunks * 2 + remote_keys * case.record_bytes // 8,
                torch.int64,
                generator,
            ),
            proxy_header=garbage(
                2 + chunks + local_size * (chunks + 1),
                torch.int32,
                generator,
            ),
            rail_keys=garbage(
                (
                    lookup_keys * case.record_bytes
                    + max_owner_chunks(case, lookup_keys, remote_keys) * 16
                )
                // 8,
                torch.int64,
                generator,
            ),
        )
        windows.append(window)
    return buffers, windows


def proxy_occurrences(case: Case, ranks: List[Dict], node: int, proxy: int) -> List:
    """A proxy's remote part: its occurrences (requester, row, owner node,
    feature, key, weight), requester-major, then owner node, feature, row."""
    occurrences = []
    for requester in range(case.local_size):
        bucketized = ranks[node * case.local_size + requester]
        offsets = bucketized["offset_per_bucket_feature_list"]
        for owner_node in range(case.num_nodes):
            if owner_node == node:
                continue
            for feature in range(case.num_features):
                bucket_feature = (
                    owner_node * case.local_size + proxy
                ) * case.num_features + feature
                for row in range(offsets[bucket_feature], offsets[bucket_feature + 1]):
                    occurrences.append(
                        (
                            requester,
                            row,
                            owner_node,
                            feature,
                            bucketized["keys_list"][row],
                            bucketized["weights_list"][row],
                        )
                    )
    return occurrences


# ── references ───────────────────────────────────────────────────────────────


def proxy_reference(case: Case, occurrences: List, node: int, remote_keys: int) -> Dict:
    """A proxy's tables by the planner's rules, in plain Python, from its
    occurrences."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    ring, span = case.num_proxy_rows, chunk_span(case)
    capacity = max_chunks(case, remote_keys)
    # node-unique keys: (table, key), or each occurrence. A key's
    # representative is its occurrence of the least bucket feature (owner
    # node, feature), and of those the first. The keys are numbered by their
    # representative's bucket feature, then, under node dedup, with the
    # requesters in turn: by (k, requester), the representative the k-th
    # (from 0, in occurrence order) of its requester's representatives of
    # that bucket feature; without node dedup requester after requester, in
    # occurrence order. A (table, key) has one owner node: its owner follows
    # from the key (random_job).
    members: Dict = {}
    for occ, (_, _, _, feature, value, _) in enumerate(occurrences):
        identity = (case.tables[feature], value) if case.node_dedup else occ
        members.setdefault(identity, []).append(occ)
    assert all(
        len({occurrences[occ][2] for occ in group}) == 1 for group in members.values()
    ), "a (table, key) of two owner nodes"
    keys = []  # each key's (owner node, feature, representative, occurrences)
    for group in members.values():
        owner_node, feature, rep = min(
            (occurrences[occ][2], occurrences[occ][3], occ) for occ in group
        )
        keys.append((owner_node, feature, rep, group))
    keys.sort()
    if case.node_dedup:
        # each representative's k: its requester's representatives of its
        # bucket feature before it
        nth: Dict = {}
        ks: Dict = {}
        for owner_node, feature, rep, _ in keys:
            block = (occurrences[rep][0], owner_node, feature)
            ks[rep] = nth.get(block, 0)
            nth[block] = ks[rep] + 1
        keys.sort(key=lambda key: (key[0], key[1], ks[key[2]], occurrences[key[2]][0]))
    node_num_uniques = len(keys)
    key_of = [0] * len(occurrences)
    for key, (*_, group) in enumerate(keys):
        for occ in group:
            key_of[occ] = key
    node_of = [owner_node for owner_node, *_ in keys]
    node_length_per_bucket_feature = [0] * (num_nodes * num_features)
    for owner_node, feature, *_ in keys:
        node_length_per_bucket_feature[owner_node * num_features + feature] += 1
    node_offset_per_bucket = [
        sum(node_length_per_bucket_feature[: owner_node * num_features])
        for owner_node in range(num_nodes + 1)
    ]
    # a row per occurrence, max_key_rows at most (a key of more: the call
    # raises)
    count = [len(group) for *_, group in keys]
    rows = [min(key_count, max_key_rows(case)) for key_count in count]
    row_scan = [0] + list(accumulate(rows))
    # a key's counted rows: its rows, but at least rows_per_payload
    counted = [counted_rows(case, row_count) for row_count in rows]
    counted_off = [0] + list(accumulate(counted))
    occurrence_scan = [0] + list(accumulate(count))

    # the rounds: owner node o's chunk j, in the stream's round j, takes its
    # keys whose counted rows before them, among the node's, are in
    # [j * span, (j + 1) * span)
    def first_key(owner_node: int, j: int) -> int:
        lo, hi = (
            node_offset_per_bucket[owner_node],
            node_offset_per_bucket[owner_node + 1],
        )
        return bisect_left(counted_off, counted_off[lo] + j * span, lo, hi) - lo

    chunk_length_per_bucket = []
    for owner_node in range(num_nodes):
        lo, hi = (
            node_offset_per_bucket[owner_node],
            node_offset_per_bucket[owner_node + 1],
        )
        chunk_length_per_bucket.append(
            0
            if owner_node == node or hi == lo
            else (counted_off[hi - 1] - counted_off[lo]) // span + 1
        )
    # the stream: round after round, owner node after owner node in a round,
    # in turn from the node after the proxy's own
    stream = [
        (owner_node, j)
        for j in range(max(chunk_length_per_bucket, default=0))
        for owner_node in ((node + turn) % num_nodes for turn in range(1, num_nodes))
        if chunk_length_per_bucket[owner_node] > j
    ]
    num_chunks = len(stream)
    chunk_rows, chunk_keys, chunk_counted = [], [], []
    for owner_node, j in stream:
        lo, hi = first_key(owner_node, j), first_key(owner_node, j + 1)
        first = node_offset_per_bucket[owner_node]
        chunk_keys.append((first + lo, hi - lo))
        chunk_rows.append(row_scan[first + hi] - row_scan[first + lo])
        chunk_counted.append(counted_off[first + hi] - counted_off[first + lo])
    stream_offsets = list(accumulate([0] + chunk_rows))
    total, lap_span = stream_offsets[-1], ring - case.max_chunk_rows + 1
    offsets = []
    for chunk in range(num_chunks):
        if total <= ring:
            offsets.append(stream_offsets[chunk])
        else:
            lap = stream_offsets[chunk] // lap_span
            first = bisect_left(stream_offsets, lap * lap_span, 0, num_chunks)
            offsets.append(lap * ring + stream_offsets[chunk] - stream_offsets[first])
    stream_end = offsets[-1] + chunk_rows[-1] if num_chunks else 0
    table = [[0] * 8 for _ in range(capacity)]
    chunk_offset_per_bucket = list(accumulate([0] + chunk_length_per_bucket))
    chunk_inv = [0] * chunk_offset_per_bucket[-1]
    chunk_of, key_row, first_duplicate_slot = ([0] * node_num_uniques for _ in range(3))
    for chunk, (owner_node, j) in enumerate(stream):
        key_lo, num = chunk_keys[chunk]
        start, end = offsets[chunk], offsets[chunk] + chunk_rows[chunk]
        pulls = occurrence_scan[key_lo + num] - occurrence_scan[key_lo]
        table[chunk] = [owner_node, key_lo, num, pulls, start, end, j, 0]
        chunk_inv[chunk_offset_per_bucket[owner_node] + j] = chunk
        # the chunk's rows: its keys' rows, then each key's duplicates' slots
        at = start % ring
        slot = at + num
        for key in range(key_lo, key_lo + num):
            chunk_of[key], key_row[key] = chunk, at + key - key_lo
            first_duplicate_slot[key], slot = slot, slot + rows[key] - 1
    # each occurrence's key row and slot (one past its key's rows: the key's
    # row; each of a key of more occurrences, in a call that raises: the
    # key's row), and its place in its requester's push order
    word0, slots = [0] * len(occurrences), [0] * len(occurrences)
    for key, (*_, group) in enumerate(keys):
        for nth, occ in enumerate(group):
            word0[occ] = key_row[key]
            slots[occ] = (
                first_duplicate_slot[key] + nth - 1
                if 0 < nth < rows[key] and len(group) <= max_key_rows(case)
                else key_row[key]
            )

    position = [0] * len(occurrences)
    pushes = [[0] * (capacity + 1) for _ in range(local_size)]
    for requester in range(local_size):
        mine = [
            occ
            for occ, occurrence in enumerate(occurrences)
            if occurrence[0] == requester
        ]
        for pos, occ in enumerate(sorted(mine, key=lambda occ: chunk_of[key_of[occ]])):
            position[occ] = pos
            pushes[requester][chunk_of[key_of[occ]]] += 1
    # push tables: per requester and chunk where its positions start
    push_tables = [
        list(accumulate([0] + pushes[requester][:capacity]))
        for requester in range(local_size)
    ]
    # the rows before each key: its key row, then its duplicates' slots
    node_exp_off = list(accumulate([0] + rows))
    # records: the key and, in 16 bytes, its weights' sum
    records = []
    for *_, rep, group in keys:
        weight = 0.0
        if case.weighted:
            weight = float(sum(occurrences[occ][5] for occ in group))
        records.append((occurrences[rep][4], weight))
    slice_bytes = [
        0
        if owner_node == node
        else chunk_length_per_bucket[owner_node] * 16
        + (node_offset_per_bucket[owner_node + 1] - node_offset_per_bucket[owner_node])
        * case.record_bytes
        for owner_node in range(num_nodes)
    ]
    return dict(
        occurrences=occurrences,
        occurrence_of={
            (occurrence[0], occurrence[1]): occ
            for occ, occurrence in enumerate(occurrences)
        },
        node_num_uniques=node_num_uniques,
        need_key_rows=max(count, default=0),
        key_of=key_of,
        node_of=node_of,
        node_length_per_bucket_feature=node_length_per_bucket_feature,
        node_offset_per_bucket=node_offset_per_bucket,
        num_chunks=num_chunks,
        chunks=table,
        chunk_counted_rows=chunk_counted,
        chunk_length_per_bucket=chunk_length_per_bucket,
        chunk_offset_per_bucket=chunk_offset_per_bucket,
        chunk_inv=chunk_inv,
        stream_end=stream_end,
        key_row=key_row,
        word0=word0,
        slot=slots,
        position=position,
        push_tables=push_tables,
        node_exp_off=node_exp_off,
        records=records,
        reduce_task_offsets=list(
            accumulate([0] + [ceil32(table[chunk][2]) for chunk in range(capacity)])
        ),
        stage_offset_per_bucket=[
            sum(slice_bytes[:owner_node]) for owner_node in range(num_nodes)
        ],
    )


def merge_rank(requester: int, nth: int, counts: List[int]) -> int:
    result = nth
    for other, other_count in enumerate(counts):
        if other == requester or other_count == 0:
            continue
        result += (
            min(other_count, nth * other_count // counts[requester] + 1)
            if other < requester
            else min(other_count, -(-nth * other_count // counts[requester]))
        )
    return result


def dedup_by_merge_rank(case: Case, occurrences: List) -> Dict:
    """A dedup by merge ranks: an occurrence's merge rank orders it among the
    node's (bucket features (owner node, feature) in order; merge ranks inside
    a bucket feature), a node-unique key's representative is its occurrence of
    least merge rank, and its bucket feature is the representative's. Each
    bucket feature's records (key, weight), sorted."""
    occs_of: Dict = {}  # bucket feature -> each requester's occurrences
    for occ, (requester, _, owner_node, feature, _, _) in enumerate(occurrences):
        per_requester = occs_of.setdefault(
            (owner_node, feature), [[] for _ in range(case.local_size)]
        )
        per_requester[requester].append(occ)
    rank_of, base = {}, 0
    for bucket_feature in sorted(occs_of):
        counts = [len(occs) for occs in occs_of[bucket_feature]]
        for requester, occs in enumerate(occs_of[bucket_feature]):
            for nth, occ in enumerate(occs):
                rank_of[occ] = base + merge_rank(requester, nth, counts)
        base += sum(counts)
    members: Dict = {}
    for occ in sorted(rank_of, key=rank_of.get):
        feature, key = occurrences[occ][3:5]
        identity = (case.tables[feature], key) if case.node_dedup else occ
        members.setdefault(identity, []).append(occ)
    records_of: Dict = {}
    for group in members.values():
        owner_node, feature, key = occurrences[group[0]][2:5]
        weight = 0.0
        if case.weighted:
            weight = float(sum(occurrences[occ][5] for occ in group))
        records_of.setdefault((owner_node, feature), []).append((key, weight))
    return {
        bucket_feature: sorted(records)
        for bucket_feature, records in records_of.items()
    }


def lookup_rows_reference(
    case: Case, lengths: List[int]
) -> Tuple[List[int], List[int]]:
    """An owner's lookup_rows (lengths: its KJT's lengths, [feature][column]):
    the row of each of its KJT's keys, and where each lookup group's slice
    starts (num_groups + 1, then the keys in all). Each group's keys take a
    slice of their own, in group order; in it the group's own part (the
    requesters' columns, feature by feature), then each other node's column
    (in node order), feature by feature."""
    local_size = case.local_size
    kjt_stride = local_size + case.num_nodes - 1
    offsets = list(accumulate([0] + list(lengths)))
    rows, starts, row = [0] * offsets[-1], [], 0
    for first, last in case.group_ranges:
        starts.append(row)
        own = [
            (feature, column)
            for feature in range(first, last)
            for column in range(local_size)
        ]
        other = [
            (feature, column)
            for column in range(local_size, kjt_stride)
            for feature in range(first, last)
        ]
        for feature, column in own + other:
            index = feature * kjt_stride + column
            for position in range(offsets[index], offsets[index + 1]):
                rows[position] = row
                row += 1
    starts.append(row)
    return rows, starts


# ── the call ─────────────────────────────────────────────────────────────────


def run_call(
    case: Case, ranks: List[Dict], lookup_keys: int, remote_keys: int, generator
):
    """Every rank's planner steps of one call, with what the comm kernels do
    between them."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    world_size = case.world_size
    tables = torch.tensor(case.tables, dtype=torch.int32, device="cuda")

    def sizes(rank: int) -> Dict[str, int]:
        return dict(
            num_nodes=num_nodes,
            local_size=local_size,
            node=rank // local_size,
            local_rank=rank % local_size,
            num_features=num_features,
            lookup_keys=lookup_keys,
            remote_keys=remote_keys,
            num_proxy_rows=case.num_proxy_rows,
            max_chunk_rows=case.max_chunk_rows,
            payload_rows=case.chunk_payload_rows,
            record_bytes=case.record_bytes,
        )

    layouts, windows, node_buffers = [], [], []
    for node in range(num_nodes):
        # node_exchange: the node's offsets (row r: requester r's
        # offset_per_bucket_feature), the same in each window of the node, and
        # each rank's node layout on them
        node_offsets = torch.stack(
            [
                ranks[node * local_size + requester]["offset_per_bucket_feature"]
                for requester in range(local_size)
            ]
        )
        for local_rank in range(local_size):
            layouts.append(
                planner(
                    hier_output_dist_node_layout,
                    node_offsets,
                    num_nodes,
                    local_size,
                    node,
                    local_rank,
                    num_features,
                    lookup_keys,
                    remote_keys,
                )
            )
        buffers, node_windows = make_windows(case, lookup_keys, remote_keys, generator)
        node_buffers.append(buffers)
        windows += node_windows
    # node_key_push: each requester's valid rows into its targets' node_keys
    # (row t of its node's buffers: target t's) where its layout says, with
    # their weights, and the bucket feature (owner node, feature) of each
    # remote occurrence
    for rank, bucketized in enumerate(ranks):
        node, local_rank = divmod(rank, local_size)
        planner(
            hier_output_dist_node_push,
            bucketized["values"],
            bucketized["inv"],
            bucketized["kjt_weights"] if case.weighted else None,
            bucketized["bucket_feature_of_row"],
            bucketized["offset_per_bucket"],
            layouts[rank]["row_shift"],
            layouts[rank]["skip"],
            num_nodes,
            local_size,
            node,
            local_rank,
            num_features,
            lookup_keys,
            remote_keys,
            **node_buffers[node],
        )
    proxies = []
    for rank, window in enumerate(windows):
        layout = layouts[rank]
        proxies.append(
            planner(
                hier_output_dist_proxy_plan,
                window["node_keys"][lookup_keys:],
                window["node_key_weights"][lookup_keys:] if case.weighted else None,
                window["node_bucket_feature_of_row"],
                layout["remote_offsets"],
                layout["skip"],
                tables,
                layout["need_remote"],
                layout["own_lengths"],
                window["remote_map"],
                window["key_stage"],
                window["proxy_header"],
                case.node_dedup,
                sizes(rank),
                list(case.groups),
            )
        )

    # rail_exchange: each rank sends each rail peer its count message (row n
    # of its messages to the peer on node n), so every rank of a rail gets
    # the rail's headers, node after node, and its own part of each
    def messages_of(rank: int) -> torch.Tensor:
        node, rail_rank = divmod(rank, local_size)
        return torch.stack(
            [
                proxies[source * local_size + rail_rank]["message"][node]
                for source in range(num_nodes)
            ]
        )

    def rail_tables(rank: int, **later) -> Dict:
        return planner(
            hier_output_dist_rail_tables,
            messages_of(rank),
            layouts[rank]["own_lengths"],
            layouts[rank]["own_offsets"],
            sizes=sizes(rank),
            lookup_groups=list(case.groups),
            **later,
        )

    rails = [rail_tables(rank) for rank in range(world_size)]
    # rail_key_push: each proxy's slice for each owner of its rail
    problems = []
    for rank, rail in enumerate(rails):
        node, rail_rank = divmod(rank, local_size)
        put_src = proxies[rank]["stage_offset_per_bucket"].tolist()
        for owner_node, (skip, dst, size) in enumerate(
            zip(
                rail["put_skip"].tolist(),
                rail["put_dst"].tolist(),
                rail["put_bytes"].tolist(),
            )
        ):
            if owner_node == node or skip:
                continue
            owner = owner_node * local_size + rail_rank
            rail_keys = windows[owner]["rail_keys"]
            if dst + size > 8 * rail_keys.numel():
                problems.append("a put past rail_keys")
                continue
            source = windows[rank]["key_stage"][put_src[owner_node] // 8 :]
            rail_keys[dst // 8 : (dst + size) // 8] = source[: size // 8]
    # node_status_exchange: each rank's needs and group shifts to its node;
    # then the owners' tables and the requesters' tables
    needs = torch.stack([rail["needs"] for rail in rails]).view(
        num_nodes, local_size, -1
    )
    owners, requesters = [], []
    for rank, window in enumerate(windows):
        node, local_rank = divmod(rank, local_size)
        owners.append(
            rail_tables(
                rank,
                rail_keys=window["rail_keys"],
                node_keys=window["node_keys"][:lookup_keys],
                node_key_weights=window["node_key_weights"][:lookup_keys]
                if case.weighted
                else None,
            )
        )
        node_ranks = range(node * local_size, (node + 1) * local_size)
        requesters.append(
            planner(
                hier_output_dist_requester_tables,
                torch.stack([windows[peer]["remote_map"] for peer in node_ranks]),
                torch.stack([windows[peer]["proxy_header"] for peer in node_ranks]),
                # [owner][group]: each owner's of the node
                torch.stack([rails[peer]["group_shift"] for peer in node_ranks]),
                ranks[rank]["bucket_feature_of_row"],
                ranks[rank]["offset_per_bucket"],
                layouts[rank]["row_shift"],
                sizes(rank),
                list(case.groups),
            )
        )
    torch.cuda.synchronize()  # an access out of range would raise here
    return dict(
        windows=windows,
        proxies=proxies,
        rails=rails,
        owners=owners,
        requesters=requesters,
        needs=needs,
        problems=problems,
    )


# ── checks ───────────────────────────────────────────────────────────────────


def u32(values: List[int]) -> List[int]:
    """int32 words as the uint32 they hold."""
    return [value & 0xFFFFFFFF for value in values]


def check_proxy(
    case: Case, node: int, ref: Dict, out: Dict, window: Dict, remote_keys: int
):
    """A proxy's outputs and windows against its reference."""
    num_nodes, local_size = case.num_nodes, case.local_size
    capacity = max_chunks(case, remote_keys)
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    num_occurrences, node_num_uniques = len(ref["occurrences"]), ref["node_num_uniques"]
    num_chunks = ref["num_chunks"]
    expect("occurrences", int(out["occurrence_count"]), num_occurrences)
    expect("need_key_rows", int(out["need_key_rows"]), ref["need_key_rows"])
    expect(
        "keys", out["node_reverse_indices"][:num_occurrences].tolist(), ref["key_of"]
    )
    expect(
        "node_length_per_bucket_feature",
        out["node_length_per_bucket_feature"].tolist(),
        ref["node_length_per_bucket_feature"],
    )
    expect(
        "node_offset_per_bucket",
        out["node_offset_per_bucket"].tolist(),
        ref["node_offset_per_bucket"],
    )
    expect("chunks", int(out["num_chunks"]), num_chunks)
    # the chunks' bounds: none empty, its rows at most its counted rows, which
    # are at most max_chunk_rows, and none past the ring's end
    expect(
        "chunk bounds",
        all(
            row[2] >= 1
            and row[5] - row[4] <= counted <= case.max_chunk_rows
            and row[4] % case.num_proxy_rows + row[5] - row[4] <= case.num_proxy_rows
            for row, counted in zip(ref["chunks"], ref["chunk_counted_rows"])
        ),
        True,
    )
    expect(
        "chunk table",
        [
            row[:4] + u32(row[4:6]) + row[6:]
            for row in out["chunks"].view(-1, 8).tolist()
        ],
        ref["chunks"],
    )
    expect(
        "chunk_length_per_bucket",
        out["chunk_length_per_bucket"].tolist(),
        ref["chunk_length_per_bucket"],
    )
    expect(
        "chunk_offset_per_bucket",
        out["chunk_offset_per_bucket"].tolist(),
        ref["chunk_offset_per_bucket"],
    )
    expect(
        "chunk_inv",
        out["chunk_inv"][: len(ref["chunk_inv"])].tolist(),
        ref["chunk_inv"],
    )
    expect(
        "reduce_task_offsets",
        out["reduce_task_offsets"].tolist(),
        ref["reduce_task_offsets"],
    )
    words = window["remote_map"].view(3, -1)[:, :num_occurrences].long().cpu()
    expect("key rows", words[0].tolist(), ref["word0"])
    expect("slots", words[1].tolist(), ref["slot"])
    expect("push positions", words[2].tolist(), ref["position"])
    header = window["proxy_header"].cpu()
    expect("header chunks", int(header[0]), num_chunks)
    expect("header end", int(header[1]) & 0xFFFFFFFF, ref["stream_end"])
    expect(
        "header ends",
        u32(header[2 : 2 + capacity].tolist()),
        [row[5] for row in ref["chunks"]],
    )
    expect(
        "push tables",
        header[2 + capacity :].view(local_size, capacity + 1).tolist(),
        ref["push_tables"],
    )
    # key_stage: per owner node its chunks, then its keys' records
    stage = window["key_stage"].cpu()
    stage_offset_per_bucket = out["stage_offset_per_bucket"].tolist()
    expect("stage slices", stage_offset_per_bucket, ref["stage_offset_per_bucket"])
    words32 = stage.view(torch.int32)
    got, want = [], []
    for owner_node in range(num_nodes):
        if owner_node == node:
            continue
        for j in range(ref["chunk_length_per_bucket"][owner_node]):
            word = (stage_offset_per_bucket[owner_node] + 16 * j) // 4
            got.append(
                words32[word : word + 2].tolist()
                + u32(words32[word + 2 : word + 4].tolist())
            )
            chunk = ref["chunks"][
                ref["chunk_inv"][ref["chunk_offset_per_bucket"][owner_node] + j]
            ]
            record_lo = chunk[1] - ref["node_offset_per_bucket"][owner_node]
            want.append([record_lo, record_lo + chunk[2], chunk[4], chunk[5]])
    expect("stage chunks", got, want)
    word = torch.tensor(
        [
            (
                stage_offset_per_bucket[owner_node]
                + 16 * ref["chunk_length_per_bucket"][owner_node]
                + (key - ref["node_offset_per_bucket"][owner_node]) * case.record_bytes
            )
            // 8
            for key, owner_node in enumerate(ref["node_of"])
        ],
        dtype=torch.int64,
    )
    weights = (
        stage.view(torch.float32)[2 * word + 2] if case.record_bytes == 16 else None
    )
    records = [
        (value, float(weights[key]) if weights is not None else 0.0)
        for key, value in enumerate(stage[word].tolist())
    ]
    expect("records", records, ref["records"])
    # the records of each bucket feature against a dedup by merge ranks, as
    # sets
    bucket_feature_ends = list(
        accumulate(out["node_length_per_bucket_feature"].tolist())
    )
    records_of: Dict = {}
    for key, record in enumerate(records):
        records_of.setdefault(
            divmod(bisect_right(bucket_feature_ends, key), case.num_features), []
        ).append(record)
    expect(
        "records by merge ranks",
        {bucket_feature: sorted(group) for bucket_feature, group in records_of.items()},
        dedup_by_merge_rank(case, ref["occurrences"]),
    )
    # the rows before each key
    expect(
        "node_exp_off",
        out["node_exp_off"][: node_num_uniques + 1].tolist(),
        ref["node_exp_off"],
    )
    return problems


def check_requester(
    case: Case,
    rank: int,
    ranks: List[Dict],
    refs: Dict,
    out: Dict,
    kjt_offsets,
    lookup_rows,
    remote_keys: int,
):
    """A requester's tables against its proxies' references and its node's
    owners' KJTs and lookup_rows (lookup_rows[owner]: the row of each of its
    KJT's keys)."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    kjt_stride = local_size + num_nodes - 1
    node, local_rank = divmod(rank, local_size)
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    bucketized = ranks[rank]
    valid, offsets = bucketized["valid"], bucketized["offset_per_bucket_feature_list"]
    # A row of its node's owner is in the owner's KJT, at index feature *
    # kjt_stride + local_rank, and in the owner's lookup_rows at that key's
    # row; another node's row reads its key's row at its proxy, and pushes to
    # its occurrence's slot.
    src, dst = [0] * valid, [0] * valid
    at_proxy: List[List] = [[] for _ in range(local_size)]  # (push position, row)
    for bucket_feature in range(case.world_size * num_features):
        owner, feature = divmod(bucket_feature, num_features)
        owner_node, proxy = divmod(owner, local_size)
        for row in range(offsets[bucket_feature], offsets[bucket_feature + 1]):
            if owner_node == node:
                src[row] = dst[row] = lookup_rows[owner][
                    kjt_offsets[owner][feature * kjt_stride + local_rank]
                    + row
                    - offsets[bucket_feature]
                ]
            else:
                ref = refs[(node, proxy)]
                occ = ref["occurrence_of"][(local_rank, row)]
                src[row], dst[row] = (
                    ref["key_row"][ref["key_of"][occ]],
                    ref["slot"][occ],
                )
                at_proxy[proxy].append((ref["position"][occ], row))
    expect("src", out["src"][:valid].tolist(), src)
    expect("dst", out["dst"][:valid].tolist(), dst)
    push_order_off = list(accumulate([0] + [len(rows) for rows in at_proxy]))
    expect("push_order_off", out["push_order_off"].tolist(), push_order_off)
    expect(
        "push order",
        out["push_order"][: push_order_off[-1]].tolist(),
        [row for rows in at_proxy for _, row in sorted(rows)],
    )
    # Its push tables at each proxy made absolute.
    capacity = max_chunks(case, remote_keys)
    tables = []
    for proxy in range(local_size):
        start = push_order_off[proxy]
        tables.append(
            [entry + start for entry in refs[(node, proxy)]["push_tables"][local_rank]]
        )
    expect("push tables", out["proxy_push_tables"].tolist(), tables)
    expect(
        "proxy chunks",
        out["proxy_num_chunks"].tolist(),
        [refs[(node, proxy)]["num_chunks"] for proxy in range(local_size)],
    )
    expect(
        "proxy ends",
        out["proxy_ends"].tolist(),
        [
            [row[5] for row in refs[(node, proxy)]["chunks"]]
            + [refs[(node, proxy)]["stream_end"]]
            for proxy in range(local_size)
        ],
    )
    # remote task offsets: chunk after chunk, the proxies in turn from its own
    # one, 32 positions a task
    remote_task_offsets = [0]
    for chunk in range(capacity):
        for turn in range(local_size):
            proxy = (local_rank + turn) % local_size
            tasks = 0
            if chunk < refs[(node, proxy)]["num_chunks"]:
                tasks = ceil32(tables[proxy][chunk + 1] - tables[proxy][chunk])
            remote_task_offsets.append(remote_task_offsets[-1] + tasks)
    expect(
        "remote_task_offsets", out["remote_task_offsets"].tolist(), remote_task_offsets
    )
    return problems


def check_backward(
    case: Case,
    node: int,
    ranks: List[Dict],
    refs: Dict,
    proxies,
    requesters,
    generator,
):
    """A backward of the node with integer gradients: each requester pushes
    its rows of a chunk at a proxy as its push tasks do, each task counting
    its rows in the chunk's pushed_rows; each proxy's reducers wait until a
    chunk's count reaches its rows, then sum their keys' rows (a key's row,
    then its duplicates' slots after the chunk's keys' rows, from the rows
    before each key). The pushes of a chunk fill each of its rows once, so
    its count reaches its rows exactly when every row is in, and every key
    gets the sum of its occurrences' gradients."""
    local_size, ring = case.local_size, case.num_proxy_rows
    problems = []
    node_ranks = range(node * local_size, (node + 1) * local_size)
    grads = [
        torch.randint(
            -1000, 1000, (ranks[rank]["valid"],), generator=generator
        ).tolist()
        for rank in node_ranks
    ]
    tables = [
        {
            name: requesters[rank][name].tolist()
            for name in ("proxy_push_tables", "push_order", "dst")
        }
        for rank in node_ranks
    ]
    for proxy in range(local_size):
        ref = refs[(node, proxy)]
        node_exp_off = proxies[node * local_size + proxy]["node_exp_off"].tolist()
        occurrences_of = [[] for _ in range(ref["node_num_uniques"])]
        for occ, key in enumerate(ref["key_of"]):
            occurrences_of[key].append(ref["occurrences"][occ])
        for chunk in range(ref["num_chunks"]):
            store: Dict = {}  # slot -> the gradients pushed there
            pushed = 0  # the chunk's pushed_rows: each push task's rows
            for requester, table in enumerate(tables):
                lo = table["proxy_push_tables"][proxy][chunk]
                hi = table["proxy_push_tables"][proxy][chunk + 1]
                for first in range(lo, hi, TASK_ROWS):
                    pushed += min(TASK_ROWS, hi - first)
                for position in range(lo, hi):
                    row = table["push_order"][position]
                    store.setdefault(table["dst"][row], []).append(
                        grads[requester][row]
                    )
            if any(len(pushes) > 1 for pushes in store.values()):
                problems.append("two pushes to one slot")
            _, key_lo, num, _, start, end, *_ = ref["chunks"][chunk]
            at, chunk_rows = start % ring, end - start
            if pushed != chunk_rows:
                problems.append("a chunk's pushed rows != its rows")
            if sorted(store) != list(range(at, at + chunk_rows)):
                problems.append("a chunk's pushes miss or pass its rows")
            for key in range(key_lo, key_lo + num):
                begin, end = node_exp_off[key], node_exp_off[key + 1]
                first_duplicate_slot = (
                    at + num + begin - node_exp_off[key_lo] - (key - key_lo)
                )
                total = 0
                for row in range(begin, end):
                    slot = (
                        at + key - key_lo
                        if row == begin
                        else first_duplicate_slot + row - begin - 1
                    )
                    pushes = store.get(slot)
                    if not pushes:
                        problems.append("a key's row without a push")
                        continue
                    total += pushes[0]
                want = sum(grads[occ[0]][occ[1]] for occ in occurrences_of[key])
                if total != want:
                    problems.append("a key's gradient sum")
    return problems


def group_runs_reference(
    case: Case, lengths: List[int], source_column: int
) -> List[int]:
    """The group runs (GroupRuns, src/hier_output_dist/common/common.h) of a
    source node's records at an owner (lengths: the owner's KJT lengths;
    source_column: the node's column there): where each lookup group's
    records start among the node's records (num_groups + 1), then each
    group's shift: record k of group g lies at row k + shift(g) of the
    owner's lookup_rows (its slice's start, its own part, the columns of the
    other nodes before this one)."""
    local_size = case.local_size
    kjt_stride = local_size + case.num_nodes - 1

    def keys(first: int, last: int, columns) -> int:
        return sum(
            lengths[feature * kjt_stride + column]
            for feature in range(first, last)
            for column in columns
        )

    first, shift, start = [0], [], 0
    for lo, hi in case.group_ranges:
        before = keys(lo, hi, range(local_size)) + keys(
            lo, hi, range(local_size, source_column)
        )
        shift.append(start + before - first[-1])
        first.append(first[-1] + keys(lo, hi, [source_column]))
        start += keys(lo, hi, range(kjt_stride))
    return first + shift


def check_rail(
    case: Case,
    rank: int,
    refs: Dict,
    rails: List[Dict],
    recv: List[int],
    lookup_keys: int,
):
    """What a rank as proxy computed of its rail's owners against what they
    computed: where its slice starts there and its puts; the group runs of
    its keys at each owner (its bwd_groups, the owner's fwd_groups row for its
    node), against the owner's KJT; and as owner where its groups' slices
    start and its group shifts."""
    num_nodes, local_size = case.num_nodes, case.local_size
    kjt_stride = local_size + num_nodes - 1
    node, rail = divmod(rank, local_size)
    ref = refs[(node, rail)]
    mine = {name: tensor.tolist() for name, tensor in rails[rank].items()}
    runs = 2 * len(case.group_ranges) + 1
    problems = []
    for owner_node in range(num_nodes):
        owner = rails[owner_node * local_size + rail]
        skip = owner_node == node or recv[owner_node * local_size + rail] > lookup_keys
        if mine["put_skip"][owner_node] != skip:
            problems.append("put_skip")
        if owner_node == node:
            if mine["bwd_groups"][node] != [0] * runs:
                problems.append("bwd_groups of its own node")
            if mine["fwd_groups"][node] != [0] * runs:
                problems.append("fwd_groups of its own node")
            continue
        slice_in = owner["slice_in"].tolist()
        column = local_size + (node if node < owner_node else node - 1)
        want = group_runs_reference(case, owner["kjt_lengths"].tolist(), column)
        if owner["fwd_groups"].tolist()[node] != want:
            problems.append("the owner's fwd_groups")
        if mine["bwd_groups"][owner_node] != owner["fwd_groups"].tolist()[node]:
            problems.append("bwd_groups != the owner's fwd_groups")
        if mine["put_dst"][owner_node] != slice_in[node][0]:
            problems.append("put_dst != the owner's slice_in")
        uniq = (
            ref["node_offset_per_bucket"][owner_node + 1]
            - ref["node_offset_per_bucket"][owner_node]
        )
        chunks = ref["chunk_length_per_bucket"][owner_node]
        if mine["put_bytes"][owner_node] != 16 * chunks + uniq * case.record_bytes:
            problems.append("put_bytes")
        if slice_in[node][1] != chunks:
            problems.append("slice_in chunks")
        if owner["node_length_per_source"].tolist()[node] != uniq:
            problems.append(
                "node_length_per_source != the proxy's node_length_per_bucket"
            )
    # As owner: where each group's slice starts, and each group's own part's
    # shift (its slice's start less the own part's keys before it).
    lengths = mine["kjt_lengths"]
    _, starts = lookup_rows_reference(case, lengths)
    if mine["group_offsets"] != starts:
        problems.append("group_offsets")
    own_before, shifts = 0, []
    for (lo, hi), start in zip(case.group_ranges, starts):
        shifts.append(start - own_before)
        own_before += sum(
            lengths[feature * kjt_stride + column]
            for feature in range(lo, hi)
            for column in range(local_size)
        )
    if mine["group_shift"] != shifts:
        problems.append("group_shift")
    return problems


def check_owner(case: Case, rank: int, ranks: List[Dict], refs: Dict, out: Dict):
    """An owner's KJT: per feature its node's keys, each requester's in its
    order, then each other node's node-unique keys of its rail's proxy there,
    in that proxy's key order (u)."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    kjt_stride = local_size + num_nodes - 1
    node, rail = divmod(rank, local_size)
    others = [other_node for other_node in range(num_nodes) if other_node != node]
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    lengths, keys, weights = [], [], []
    # where the feature's keys start in each other node's keys for this node
    feature_offsets = {other_node: 0 for other_node in others}
    for feature in range(num_features):
        bucket_feature = rank * num_features + feature
        for requester in range(local_size):
            bucketized = ranks[node * local_size + requester]
            rows = range(
                *bucketized["offset_per_bucket_feature_list"][bucket_feature:][:2]
            )
            lengths.append(len(rows))
            keys += [bucketized["keys_list"][row] for row in rows]
            weights += [
                bucketized["weights_list"][row] if case.weighted else 1.0
                for row in rows
            ]
        for other_node in others:
            ref = refs[(other_node, rail)]
            count = ref["node_length_per_bucket_feature"][node * num_features + feature]
            first = ref["node_offset_per_bucket"][node] + feature_offsets[other_node]
            feature_offsets[other_node] += count
            lengths.append(count)
            keys += [key for key, _ in ref["records"][first : first + count]]
            weights += [weight for _, weight in ref["records"][first : first + count]]
    offsets = list(accumulate([0] + lengths))
    num_recv = offsets[-1]
    expect("kjt_lengths", out["kjt_lengths"].tolist(), lengths)
    expect("kjt_offsets", out["kjt_offsets"].tolist(), offsets)
    expect("num_recv", int(out["num_recv"]), num_recv)
    expect(
        "length_per_key",
        out["length_per_key"].tolist(),
        [
            sum(lengths[feature * kjt_stride : (feature + 1) * kjt_stride])
            for feature in range(num_features)
        ],
    )
    expect("kjt_values", out["kjt_values"][:num_recv].tolist(), keys)
    if case.record_bytes == 16:
        expect("kjt_weights", out["kjt_weights"][:num_recv].tolist(), weights)
    # each key's row in its group's slice of lookup_rows: none when every
    # group has one feature (a key's row is then its place in the group)
    rows, starts = lookup_rows_reference(case, lengths)
    group_of = [
        group
        for group, (lo, hi) in enumerate(case.group_ranges)
        for feature in range(lo, hi)
        for _ in range(kjt_stride)
    ]
    within = [
        row - starts[group_of[index]]
        for index in range(len(lengths))
        for row in rows[offsets[index] : offsets[index + 1]]
    ]
    if all(hi - lo == 1 for lo, hi in case.group_ranges):
        expect("no row map", "row_map" in out, False)
        expect(
            "one feature a group",
            within,
            [
                position - starts[group_of[index]]
                for index in range(len(lengths))
                for position in range(offsets[index], offsets[index + 1])
            ],
        )
    else:
        expect("row map", out["row_map"][:num_recv].tolist(), within)
    return problems


def check_owner_chunks(case: Case, rank: int, refs: Dict, out: Dict):
    """The chunks a rank got as owner (fwd_chunks, what its forward puts):
    each source node's chunks of its node, in order, against the proxies'
    references."""
    num_nodes, local_size = case.num_nodes, case.local_size
    node, rail = divmod(rank, local_size)
    fwd_chunks, fwd_chunk_offsets = [], [0]
    for source in range(num_nodes):
        if source != node:
            source_ref = refs[(source, rail)]
            first = source_ref["chunk_offset_per_bucket"][node]
            for j in range(source_ref["chunk_length_per_bucket"][node]):
                chunk = source_ref["chunks"][source_ref["chunk_inv"][first + j]]
                record_lo = chunk[1] - source_ref["node_offset_per_bucket"][node]
                fwd_chunks.append([record_lo, record_lo + chunk[2], chunk[4], chunk[5]])
        fwd_chunk_offsets.append(len(fwd_chunks))
    problems = []
    got_chunks = out["fwd_chunks"][: len(fwd_chunks)].tolist()
    if [row[:2] + u32(row[2:]) for row in got_chunks] != fwd_chunks:
        problems.append("fwd_chunks")
    if out["fwd_chunk_offsets"].tolist() != fwd_chunk_offsets:
        problems.append("fwd_chunk_offsets")
    return problems


def run_case(case: Case, seed: int) -> Tuple[Dict[str, List[str]], str]:
    generator = torch.Generator().manual_seed(seed)
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    world_size = case.world_size
    job = (case.job or random_job)(case, generator)
    ranks = [bucketized_rows(case, *rows, generator) for rows in job]
    occurrences = {
        (node, proxy): proxy_occurrences(case, ranks, node, proxy)
        for node in range(num_nodes)
        for proxy in range(local_size)
    }
    need_remote = max(len(occs) for occs in occurrences.values())
    remote_keys = max(need_remote + case.remote_slack, 1)
    refs = {
        where: proxy_reference(case, occs, where[0], remote_keys)
        for where, occs in occurrences.items()
    }
    # what each owner looks up: its node's keys for it, and the node-unique
    # keys of the other nodes' proxies on its rail
    recv = []
    for rank in range(world_size):
        node, rail = divmod(rank, local_size)
        own = sum(
            int(
                ranks[node * local_size + requester]["counts"][rank * num_features :][
                    :num_features
                ].sum()
            )
            for requester in range(local_size)
        )
        recv.append(
            own
            + sum(
                refs[(other_node, rail)]["node_offset_per_bucket"][node + 1]
                - refs[(other_node, rail)]["node_offset_per_bucket"][node]
                for other_node in range(num_nodes)
                if other_node != node
            )
        )
    need_lookup = max(recv)
    lookup_keys = max(need_lookup + case.lookup_slack, 1)
    need_key_rows = max(ref["need_key_rows"] for ref in refs.values())
    laps = max(-(-ref["stream_end"] // case.num_proxy_rows) for ref in refs.values())
    call = run_call(case, ranks, lookup_keys, remote_keys, generator)
    # each node's agreed needs: the most of what its ranks publish
    agreed = call["needs"].amax(1).tolist()
    checks = {"puts": call["problems"]}
    stats = (
        f"chunks {max(ref['num_chunks'] for ref in refs.values())}, laps {laps}, "
        f"most occurrences of a key {need_key_rows}"
    )
    if lookup_keys < need_lookup or remote_keys < need_remote:
        # The call does not fit: every rank sees it in the same needs (an
        # overflowing node pushed nothing, so the lookups it caused are
        # unknown; its remote need is exact).
        fits = agreed[0][0] <= lookup_keys and agreed[0][1] <= remote_keys
        exact = remote_keys >= need_remote or agreed[0][1] == need_remote
        checks["needs"] = (
            []
            if all(row == agreed[0] for row in agreed) and not fits and exact
            else ["needs"]
        )
        return checks, stats
    checks["needs"] = (
        []
        if all(row == [need_lookup, need_remote, need_key_rows] for row in agreed)
        else ["needs"]
    )
    if need_key_rows > max_key_rows(case):
        # Keys not index-deduplicated: the call raises on its need key_rows,
        # every rank alike, and its tables stayed in bounds.
        if case.dedup:
            checks["needs"].append("a deduplicated job past the bound")
        return checks, stats
    proxies, rails, owners = call["proxies"], call["rails"], call["owners"]
    requesters = call["requesters"]
    kjt_offsets = [rail["kjt_offsets"].tolist() for rail in rails]
    lookup_rows = [
        lookup_rows_reference(case, rail["kjt_lengths"].tolist())[0] for rail in rails
    ]
    checks["proxies"] = sum(
        (
            check_proxy(
                case,
                rank // local_size,
                refs[divmod(rank, local_size)],
                proxies[rank],
                call["windows"][rank],
                remote_keys,
            )
            for rank in range(world_size)
        ),
        [],
    )
    checks["requesters"] = sum(
        (
            check_requester(
                case,
                rank,
                ranks,
                refs,
                requesters[rank],
                kjt_offsets,
                lookup_rows,
                remote_keys,
            )
            for rank in range(world_size)
        ),
        [],
    )
    checks["backward"] = sum(
        (
            check_backward(case, node, ranks, refs, proxies, requesters, generator)
            for node in range(num_nodes)
        ),
        [],
    )
    checks["rail"] = sum(
        (
            check_rail(case, rank, refs, rails, recv, lookup_keys)
            for rank in range(world_size)
        ),
        [],
    )
    checks["owners"] = sum(
        (
            check_owner(case, rank, ranks, refs, owners[rank])
            for rank in range(world_size)
        ),
        [],
    )
    checks["owner chunks"] = sum(
        (
            check_owner_chunks(case, rank, refs, owners[rank])
            for rank in range(world_size)
        ),
        [],
    )
    # the owner tables' call computed the rail's tables again
    checks["again"] = [
        name
        for rank in range(world_size)
        for name, tensor in rails[rank].items()
        if not torch.equal(owners[rank][name], tensor)
    ]
    if case.job is same_place_job:
        # requester 0's rows of key 0 and of the second lap's first key at
        # proxy 0: chunks in two laps, one slot
        dst = requesters[0]["dst"].tolist()
        second = second_lap_key(case)
        checks["shape"] = (
            [] if laps >= 2 and dst[0] == dst[second] else ["the case's shape"]
        )
    return checks, stats


def main() -> None:
    torch.cuda.set_device(0)
    failures = 0
    for case_idx, case in enumerate(CASES):
        for seed in range(case.seeds):
            checks, stats = run_case(case, 1000 + 17 * case_idx + seed)
            good = not any(checks.values())
            failures += not good
            print(
                f"dedup '{case.name}' seed {seed} ({stats}): "
                + " ".join(
                    f"{area}={int(not problems)}" for area, problems in checks.items()
                )
                + f" [{'ok' if good else 'FAIL'}]",
                flush=True,
            )
            for area, problems in checks.items():
                if problems:
                    print(f"    {area}: {sorted(set(problems))[:8]}", flush=True)
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
