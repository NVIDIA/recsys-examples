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
rail's layout, status, owner tables and the put lists
(``hier_output_dist_rail_tables``), and the requester's tables
(``hier_output_dist_requester_tables``).

The test plays every rank of an N-node job: each rank's part 1 (its rows by
owner and feature, and past the valid ones index dedup's bound), its node
layout (``hier_output_dist_node_layout``), and between the steps what the
comm kernels do: the node's summed counts, the pushes into node_keys, the
count messages of each rail, the key records each proxy puts into its owners'
rail_keys and the cell starts each owner puts into its proxies' rail_cells,
the needs and cell starts each node exchanges (the requester binding gathers
the proxies' words itself). The windows hold garbage before the call. The
checks:

* each proxy against a plain reference of the planner's rules: its
  node-unique keys (identities (owner node, table, key), cells by smallest
  feature, key order inside a cell), their records, the chunks of its stream
  (levels of each owner node's rows, packed, in laps when they pass the ring)
  and where each lies, each occurrence's row and slot (the first
  occurrence's row, duplicate slots, heavy partials), push positions, push
  tables, flags and reducer lists; its cells and records against a dedup by
  merge ranks (each key's cell that of its occurrence of least merge rank),
  as equal sets;
* each requester's maps (its node's owners' KJT places, its keys' rows and
  slots at its proxies), push order, push tables, heavy groups, sub-tasks and
  task starts; and per node a backward with integer gradients through the
  requesters' push tasks and the proxies' reducer lists;
* the rail: each proxy's slices start where its owners' offsets say, and
  each owner sends each proxy its KJT's offsets of that node's cells; each
  owner's KJT and the chunks it got; each rank's put lists (as proxy and as
  owner) against a plain loop over the chunks and cells; the needs, the same
  on every rank.

Calls that do not fit check the needs alone (and that no access was out of
range). Every planner call runs under ``torch.cuda.set_sync_debug_mode
("error")``.

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
    hier_output_dist_proxy_plan,
    hier_output_dist_rail_tables,
    hier_output_dist_requester_tables,
)

HEAVY = 64  # kHeavy: a node-unique key of more occurrences is heavy
TASK_ROWS = 32  # kTaskRows: rows (or heavy groups) of a push task
# reducer entries (common.h): the flag of no push, a heavy key's own row,
# and the bit of a partial slot
ENTRY_NO_PUSH, ENTRY_OWN_ROW, ENTRY_PARTIAL = -1, -2, 1 << 31


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


def heavy_at(case: "Case") -> int:
    """A node-unique key of more occurrences is heavy (Sizes::heavy_at)."""
    return max(HEAVY, case.local_size * case.partial_rows)


def max_key_rows(case: "Case") -> int:
    """The most rows of one key (Sizes::max_key_rows)."""
    return max(heavy_at(case), 1 + case.local_size * case.partial_rows)


def chunk_room(case: "Case") -> int:
    """Rows of a level (Sizes::chunk_room)."""
    return case.slot_rows - max_key_rows(case) + 1


def max_chunks(case: "Case", remote_keys: int) -> int:
    """The most chunks of a stream (Sizes::max_chunks)."""
    return remote_keys // chunk_room(case) + case.num_nodes


def owner_chunks(case: "Case", lookup_keys: int, remote_keys: int) -> int:
    """The most chunks an owner gets (Sizes::owner_chunks)."""
    others = case.num_nodes - 1
    return others + min(
        others * remote_keys, max_key_rows(case) * lookup_keys
    ) // chunk_room(case)


def flag_words(case: "Case", remote_keys: int) -> int:
    """The push-task flags of a call (Sizes::flag_words)."""
    return (
        remote_keys // TASK_ROWS
        + 2 * case.local_size * max_chunks(case, remote_keys)
        + 1
    )


@dataclass
class Case:
    name: str
    num_nodes: int = 4
    local_size: int = 4
    tables: Tuple[int, ...] = (0, 0, 1, 2)  # features 0 and 1 share table 0
    max_block: int = 6  # keys of a (requester, owner, feature) cell, at most
    key_range: int = 40
    hot: float = 0.0  # share of keys that are key 7
    weights: bool = False  # the keys carry weights (whole numbers: frequency counts)
    counts: bool = False  # the owner's KJT carries weights: without weights, counts
    node_dedup: bool = True
    num_proxy_rows: int = 1 << 20  # small: the ring wraps
    chunk_rows: int = 0  # the most rows of a chunk; 0: the least, 2 * max_key_rows
    itemsize: int = 4  # of an embedding row; 2: a partial takes 2 rows
    bound: int = 0  # index dedup: rows past the valid ones
    # the capacities: the most a rank needs plus this (below 0: the call
    # does not fit)
    lookup_slack: int = 5
    remote_slack: int = 5
    job: Optional[Callable] = None  # the ranks' rows, when not random
    seeds: int = 2

    @property
    def num_features(self) -> int:
        return len(self.tables)

    @property
    def num_tables(self) -> int:
        return max(self.tables) + 1

    @property
    def world_size(self) -> int:
        return self.num_nodes * self.local_size

    @property
    def partial_rows(self) -> int:
        return -(-4 // self.itemsize)

    @property
    def record_bytes(self) -> int:
        return 16 if self.weights or self.counts else 8

    @property
    def slot_rows(self) -> int:
        """Sizes::chunk_rows."""
        return self.chunk_rows or 2 * max_key_rows(self)


def same_place_job(case: Case, generator) -> List[Tuple]:
    """Requester 1 makes every key of owner (node 1, 0) heavy (65
    occurrences each), enough of them for the stream to pass the ring; then
    the first key of the second lap's first chunk lies where key 0 does, and
    so do their partial slots. Requester 0 asks those two keys once each: two
    rows of one slot in two chunks."""
    heavy_rows = 1 + case.local_size * case.partial_rows
    room, ring = chunk_room(case), case.num_proxy_rows
    lap_room = ring - case.slot_rows + 1
    keys = ring // heavy_rows + 2 * (room // heavy_rows + 2)
    level = [key * heavy_rows // room for key in range(keys)]  # each key's chunk
    rows_before, second_lap = 0, 0  # the stream's rows before each chunk
    for chunk in range(level[-1] + 1):
        if rows_before >= lap_room:
            second_lap = level.index(chunk)
            break
        rows_before += level.count(chunk) * heavy_rows
    cells = case.world_size * case.num_features
    rows = [
        [0, second_lap],
        [key for key in range(keys) for _ in range(HEAVY + 1)],
    ]
    job = []
    for keys_of in rows + [[]] * (case.world_size - 2):
        counts = torch.zeros(cells, dtype=torch.int64)
        counts[case.local_size * case.num_features] = len(keys_of)  # owner (node 1, 0)
        job.append(
            (
                counts,
                torch.tensor(keys_of, dtype=torch.int64),
                torch.ones(len(keys_of)),
            )
        )
    return job


def wide_node_job(case: Case, generator) -> List[Tuple]:
    """More GPUs per node than kHeavy: every requester of node 0 asks owner
    (node 1, 0) for key 7 once and for key 8 once, requester 0 twice: key 7's
    local_size occurrences are not more than heavy_at, key 8's one more are,
    and take its row and a partial for each requester, its occurrences."""
    cells = case.world_size * case.num_features
    job = []
    for rank in range(case.world_size):
        counts = torch.zeros(cells, dtype=torch.int64)
        keys = []
        if rank < case.local_size:  # node 0's requesters
            keys = [7, 8, 8] if rank == 0 else [7, 8]
            counts[case.local_size * case.num_features] = len(keys)  # owner (node 1, 0)
        job.append(
            (counts, torch.tensor(keys, dtype=torch.int64), torch.ones(len(keys)))
        )
    return job


CASES = [
    Case("uniform"),
    Case("hot keys: heavy", max_block=40, key_range=30, hot=0.5),
    Case(
        "weights, 3 nodes",
        num_nodes=3,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weights=True,
        chunk_rows=300,
    ),
    Case(
        "occurrence counts as weights",
        num_nodes=3,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        counts=True,
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
        "2 nodes, a wrapping ring, weights, bf16 partials",
        num_nodes=2,
        max_block=200,
        key_range=400,
        hot=0.08,
        weights=True,
        num_proxy_rows=1280,
        itemsize=2,
    ),
    Case(
        "node dedup off",
        max_block=10,
        key_range=60,
        hot=0.3,
        weights=True,
        node_dedup=False,
    ),
    Case(
        "2 nodes of 8, heavy",
        num_nodes=2,
        local_size=8,
        tables=(0, 1, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
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
        "large, weights, bf16 partials, a wrapping ring",
        max_block=200,
        key_range=400,
        hot=0.08,
        weights=True,
        num_proxy_rows=3840,
        itemsize=2,
        seeds=1,
    ),
    Case(
        "one table, counts, a wrapping ring",
        num_nodes=3,
        tables=(0, 0, 0),
        max_block=250,
        key_range=3000,
        hot=0.1,
        counts=True,
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
    Case("4 nodes of 8", local_size=8, max_block=3, key_range=20, hot=0.1, seeds=1),
    Case("index dedup's bound", bound=50, weights=True, hot=0.2),
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
    Case("remote_keys overflow", max_block=20, hot=0.3, remote_slack=-1),
    Case("lookup_keys overflow", max_block=20, hot=0.3, lookup_slack=-1),
    Case(
        "remote_keys overflow, weights, bound",
        max_block=20,
        weights=True,
        bound=30,
        remote_slack=-1,
    ),
    # the owner gather's second slice of 1 << 22 positions (past num_recv)
    Case(
        "lookup_keys past one gather slice",
        num_nodes=2,
        local_size=2,
        weights=True,
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
        chunk_rows=200,
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
        "2 nodes of 72, keys at heavy_at and past it",
        num_nodes=2,
        local_size=72,
        tables=(0,),
        remote_slack=0,
        job=wide_node_job,
        seeds=1,
    ),
]


# ── the job ──────────────────────────────────────────────────────────────────


def random_job(case: Case, generator) -> List[Tuple]:
    """Each rank's rows: its keys for each (owner, feature) cell, and their
    weights."""
    cells = case.world_size * case.num_features
    job = []
    for _ in range(case.world_size):
        counts = torch.randint(0, case.max_block + 1, (cells,), generator=generator)
        valid = int(counts.sum())
        keys = torch.randint(0, case.key_range, (valid,), generator=generator)
        keys[torch.rand(valid, generator=generator) < case.hot] = 7
        job.append(
            (counts, keys, torch.randint(1, 4, (valid,), generator=generator).float())
        )
    return job


def part_one(case: Case, counts, keys, weights, generator) -> Dict:
    """A rank's part 1 as the planner takes it: its rows by cell (owner *
    num_features + feature), then index dedup's bound (garbage keys)."""
    cells = counts.numel()
    cell_start = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    return dict(
        cell_start_list=cell_start.tolist(),
        keys_list=keys.tolist(),
        weights_list=weights.tolist(),
        valid=int(counts.sum()),
        counts=counts,
        keys=torch.cat(
            [
                keys,
                torch.randint(-(1 << 62), 1 << 62, (case.bound,), generator=generator),
            ]
        ).cuda(),
        weights=torch.cat([weights, torch.full((case.bound,), 1e9)]).cuda(),
        row_cell=torch.cat(
            [
                torch.repeat_interleave(torch.arange(cells), counts),
                torch.full((case.bound,), cells),
            ]
        )
        .int()
        .cuda(),
        cell_start=cell_start.cuda(),
        block_start=cell_start[:: case.num_features].contiguous().cuda(),
    )


def garbage(count: int, dtype: torch.dtype, generator) -> torch.Tensor:
    """Random bits of ``dtype``, as a window holds before a call."""
    bits = torch.randint(-(1 << 31), (1 << 31) - 1, (count,), generator=generator)
    if dtype == torch.int64:
        bits = (bits << 32) ^ torch.randint(0, 1 << 32, (count,), generator=generator)
    return bits.to(torch.int32 if dtype == torch.float32 else dtype).view(dtype).cuda()


def make_windows(case: Case, lookup_keys: int, remote_keys: int, generator) -> Dict:
    """A rank's window buffers as the planner across nodes reads and writes
    them, garbage in each."""
    num_nodes, local_size = case.num_nodes, case.local_size
    chunks = max_chunks(case, remote_keys)
    windows = dict(
        node_keys=garbage(lookup_keys + remote_keys, torch.int64, generator),
        remote_cells=garbage(remote_keys, torch.int32, generator),
        remote_map=garbage(3 * remote_keys, torch.int32, generator),
        key_stage=garbage(
            chunks * 2 + remote_keys * case.record_bytes // 8 + 2,
            torch.int64,
            generator,
        ),
        proxy_header=garbage(
            4 + chunks + 4 * local_size * (chunks + 1),
            torch.int32,
            generator,
        ),
        rail_keys=garbage(
            (
                lookup_keys * case.record_bytes
                + owner_chunks(case, lookup_keys, remote_keys) * 16
            )
            // 8,
            torch.int64,
            generator,
        ),
        rail_cells=garbage(num_nodes * case.num_features, torch.int64, generator),
    )
    if case.weights:
        windows["node_weights"] = garbage(
            lookup_keys + remote_keys, torch.float32, generator
        )
    return windows


def proxy_occurrences(case: Case, ranks: List[Dict], node: int, proxy: int) -> List:
    """A proxy's remote part: its occurrences (requester, row, owner node,
    feature, key, weight), requester-major, then owner node, feature, row."""
    occurrences = []
    for requester in range(case.local_size):
        part = ranks[node * case.local_size + requester]
        cell_start = part["cell_start_list"]
        for owner_node in range(case.num_nodes):
            if owner_node == node:
                continue
            for feature in range(case.num_features):
                cell = (
                    owner_node * case.local_size + proxy
                ) * case.num_features + feature
                for row in range(cell_start[cell], cell_start[cell + 1]):
                    occurrences.append(
                        (
                            requester,
                            row,
                            owner_node,
                            feature,
                            part["keys_list"][row],
                            part["weights_list"][row],
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
    partial_rows, ring, room = case.partial_rows, case.num_proxy_rows, chunk_room(case)
    capacity = max_chunks(case, remote_keys)
    # node-unique keys: (owner node, table, key), or each occurrence; numbered
    # by (owner node, smallest feature), then key order (then occurrence)
    members: Dict = {}
    for occ, (_, _, owner_node, feature, value, _) in enumerate(occurrences):
        identity = (owner_node, case.tables[feature], value) if case.node_dedup else occ
        members.setdefault(identity, []).append(occ)
    keys = sorted(
        (
            occurrences[group[0]][2],
            min(occurrences[occ][3] for occ in group),
            occurrences[group[0]][4],
            group[0],
            group,
        )
        for group in members.values()
    )
    num_keys = len(keys)
    key_of = [0] * len(occurrences)
    for key, (*_, group) in enumerate(keys):
        for occ in group:
            key_of[occ] = key
    node_of = [owner_node for owner_node, *_ in keys]
    cell_uniq = [0] * (num_nodes * num_features)
    for owner_node, feature, *_ in keys:
        cell_uniq[owner_node * num_features + feature] += 1
    uniq_off = [
        sum(cell_uniq[: owner_node * num_features])
        for owner_node in range(num_nodes + 1)
    ]
    count = [len(group) for *_, group in keys]
    heavy = [case.node_dedup and key_count > heavy_at(case) for key_count in count]
    rows = [
        1 + local_size * partial_rows if key_heavy else key_count
        for key_heavy, key_count in zip(heavy, count)
    ]
    row_scan = [0] + list(accumulate(rows))
    occurrence_scan = [0] + list(accumulate(count))

    # the levels: owner node o's chunk j takes its keys whose rows before them,
    # among the node's, are in [j * room, (j + 1) * room)
    def first_key(owner_node: int, level: int) -> int:
        lo, hi = uniq_off[owner_node], uniq_off[owner_node + 1]
        return bisect_left(row_scan, row_scan[lo] + level * room, lo, hi) - lo

    chunks_out = []
    for owner_node in range(num_nodes):
        lo, hi = uniq_off[owner_node], uniq_off[owner_node + 1]
        chunks_out.append(
            0
            if owner_node == node or hi == lo
            else (row_scan[hi - 1] - row_scan[lo]) // room + 1
        )
    # the stream: level after level, owner node after owner node in a level
    stream = [
        (owner_node, level)
        for level in range(max(chunks_out, default=0))
        for owner_node in range(num_nodes)
        if chunks_out[owner_node] > level
    ]
    num_chunks = len(stream)
    chunk_rows, chunk_keys = [], []
    for owner_node, level in stream:
        lo, hi = first_key(owner_node, level), first_key(owner_node, level + 1)
        first = uniq_off[owner_node]
        chunk_keys.append((first + lo, hi - lo))
        chunk_rows.append(row_scan[first + hi] - row_scan[first + lo])
    ends = list(accumulate([0] + chunk_rows))
    total, lap_room = ends[-1], ring - case.slot_rows + 1
    starts = []
    for chunk in range(num_chunks):
        if total <= ring:
            starts.append(ends[chunk])
        else:
            lap = ends[chunk] // lap_room
            first = bisect_left(ends, lap * lap_room, 0, num_chunks)
            starts.append(lap * ring + ends[chunk] - ends[first])
    stream_end = starts[-1] + chunk_rows[-1] if num_chunks else 0
    table = [[0] * 8 for _ in range(capacity)]
    node_chunk_start = list(accumulate([0] + chunks_out))
    node_chunks = [0] * node_chunk_start[-1]
    chunk_of, key_row, duplicate_base, partial_base = ([0] * num_keys for _ in range(4))
    for chunk, (owner_node, level) in enumerate(stream):
        key_lo, num = chunk_keys[chunk]
        start, end = starts[chunk], starts[chunk] + chunk_rows[chunk]
        pulls = occurrence_scan[key_lo + num] - occurrence_scan[key_lo]
        table[chunk] = [owner_node, key_lo, num, pulls, start, end, level, 0]
        node_chunks[node_chunk_start[owner_node] + level] = chunk
        # the chunk's rows: its keys' rows, the duplicates' slots, the partials
        at = start % ring
        slot = at + num
        for key in range(key_lo, key_lo + num):
            chunk_of[key], key_row[key] = chunk, at + key - key_lo
            if not heavy[key]:
                duplicate_base[key], slot = slot, slot + count[key] - 1
        for key in range(key_lo, key_lo + num):
            if heavy[key]:
                partial_base[key], slot = slot, slot + local_size * partial_rows
    # each occurrence's row word, slot, and place in its requester's push order
    word0, slots = [0] * len(occurrences), [0] * len(occurrences)
    for key, (*_, group) in enumerate(keys):
        for nth, occ in enumerate(group):
            word0[occ] = key_row[key] | (heavy[key] << 31)
            if heavy[key]:
                slots[occ] = partial_base[key] + occurrences[occ][0] * partial_rows
            else:
                slots[occ] = key_row[key] if nth == 0 else duplicate_base[key] + nth - 1

    def push_key(occ: int) -> Tuple:
        key = key_of[occ]
        return (chunk_of[key], heavy[key], key if heavy[key] else 0, occ)

    position = [0] * len(occurrences)
    asks: Dict = {}  # (requester, heavy key) -> its occurrences of the key
    plain = [[0] * (capacity + 1) for _ in range(local_size)]
    heavy_rows = [[0] * (capacity + 1) for _ in range(local_size)]
    groups: List[List[List[int]]] = [
        [[] for _ in range(capacity + 1)] for _ in range(local_size)
    ]
    for requester in range(local_size):
        mine = [
            occ
            for occ, occurrence in enumerate(occurrences)
            if occurrence[0] == requester
        ]
        for pos, occ in enumerate(sorted(mine, key=push_key)):
            position[occ] = pos
            key = key_of[occ]
            if not heavy[key]:
                plain[requester][chunk_of[key]] += 1
            else:
                heavy_rows[requester][chunk_of[key]] += 1
                if (requester, key) not in asks:
                    groups[requester][chunk_of[key]].append(key)
                asks[(requester, key)] = asks.get((requester, key), 0) + 1
    # push tables: per requester and chunk its start, heavy start, flags,
    # groups; the flags chunk after chunk, requester after requester in one
    tasks = [
        [
            ceil32(plain[requester][chunk]) + ceil32(len(groups[requester][chunk]))
            for chunk in range(capacity + 1)
        ]
        for requester in range(local_size)
    ]
    flag_start = list(
        accumulate(
            [0]
            + [
                sum(tasks[requester][chunk] for requester in range(local_size))
                for chunk in range(capacity)
            ]
        )
    )
    push_tables = []
    for requester in range(local_size):
        entries, start, group_start = [], 0, 0
        for chunk in range(capacity + 1):
            flags = flag_start[chunk] + sum(
                tasks[lower][chunk] for lower in range(requester)
            )
            entries.append([start, start + plain[requester][chunk], flags, group_start])
            start += plain[requester][chunk] + heavy_rows[requester][chunk]
            group_start += len(groups[requester][chunk])
        push_tables.append(entries)
    # reducer lists: a key's entries in occurrence order; a heavy key's own
    # row, then each requester's partial; each the flag of the push task that
    # fills its slot
    entries, reducer_start = [], [0]
    for key, (*_, group) in enumerate(keys):
        chunk = chunk_of[key]
        if heavy[key]:
            entries.append((key_row[key], ENTRY_OWN_ROW))
            for requester in range(local_size):
                flag = ENTRY_NO_PUSH
                if key in groups[requester][chunk]:
                    entry = push_tables[requester][chunk]
                    flag = (
                        entry[2]
                        + ceil32(entry[1] - entry[0])
                        + groups[requester][chunk].index(key) // TASK_ROWS
                    )
                entries.append(
                    (
                        (partial_base[key] + requester * partial_rows) | ENTRY_PARTIAL,
                        flag,
                    )
                )
        else:
            for occ in group:
                entry = push_tables[occurrences[occ][0]][chunk]
                entries.append(
                    (slots[occ], entry[2] + (position[occ] - entry[0]) // TASK_ROWS)
                )
        reducer_start.append(len(entries))
    # records: the key and, in 16 bytes, its weights' sum or its count
    records = []
    for *_, value, _, group in keys:
        weight = 0.0
        if case.weights:
            weight = float(sum(occurrences[occ][5] for occ in group))
        elif case.counts:
            weight = float(len(group))
        records.append((value, weight))
    slice_bytes = [
        0
        if owner_node == node
        else chunks_out[owner_node] * 16
        + (uniq_off[owner_node + 1] - uniq_off[owner_node]) * case.record_bytes
        for owner_node in range(num_nodes)
    ]
    return dict(
        occurrences=occurrences,
        occurrence_of={
            (occurrence[0], occurrence[1]): occ
            for occ, occurrence in enumerate(occurrences)
        },
        num_keys=num_keys,
        key_of=key_of,
        node_of=node_of,
        heavy=heavy,
        cell_uniq=cell_uniq,
        uniq_off=uniq_off,
        num_chunks=num_chunks,
        chunks=table,
        chunks_out=chunks_out,
        node_chunk_start=node_chunk_start,
        node_chunks=node_chunks,
        stream_end=stream_end,
        chunk_of=chunk_of,
        key_row=key_row,
        word0=word0,
        slot=slots,
        position=position,
        push_tables=push_tables,
        groups=groups,
        asks=asks,
        reducer_start=reducer_start,
        entries=entries,
        records=records,
        reduce_task_starts=list(
            accumulate([0] + [ceil32(table[chunk][2]) for chunk in range(capacity)])
        ),
        stage_slice=[sum(slice_bytes[:owner_node]) for owner_node in range(num_nodes)],
    )


def cell_merge_rank(requester: int, nth: int, counts: List[int]) -> int:
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


def merge_rank_cells(case: Case, occurrences: List) -> Dict:
    """A dedup by merge ranks: an occurrence's merge rank orders it among the
    node's (cells in owner node, then feature order; merge ranks inside a
    cell), a node-unique key's representative is its occurrence of least
    merge rank, and its cell is the representative's. Each cell's records
    (key, weight), sorted."""
    by_cell: Dict = {}
    for occ, (requester, _, owner_node, feature, _, _) in enumerate(occurrences):
        cell = by_cell.setdefault(
            (owner_node, feature), [[] for _ in range(case.local_size)]
        )
        cell[requester].append(occ)
    rank_of, base = {}, 0
    for cell in sorted(by_cell):
        counts = [len(occs) for occs in by_cell[cell]]
        for requester, occs in enumerate(by_cell[cell]):
            for nth, occ in enumerate(occs):
                rank_of[occ] = base + cell_merge_rank(requester, nth, counts)
        base += sum(counts)
    members: Dict = {}
    for occ in sorted(rank_of, key=rank_of.get):
        owner_node, feature, key = occurrences[occ][2:5]
        identity = (owner_node, case.tables[feature], key) if case.node_dedup else occ
        members.setdefault(identity, []).append(occ)
    cells: Dict = {}
    for group in members.values():
        owner_node, feature, key = occurrences[group[0]][2:5]
        weight = 0.0
        if case.weights:
            weight = float(sum(occurrences[occ][5] for occ in group))
        elif case.counts:
            weight = float(len(group))
        cells.setdefault((owner_node, feature), []).append((key, weight))
    return {cell: sorted(records) for cell, records in cells.items()}


def segments_reference(case: Case, chunks, cells, cell_rows):
    """The put lists by a plain loop, over 2 * num_nodes nodes: the backward
    list's owner nodes, then the forward list's proxy nodes. Each node's keys,
    chunk by chunk, each chunk cut at its feature cells. chunks[node]: its
    chunks in order, each (lo, hi, row of lo in proxy_rows, its number in its
    list); cells[node][feature]: its keys of each feature; cell_rows[node]
    [feature]: the owner's KJT row of each cell's first key. Returns the
    segments [src, dst, rows, chunk] and where each node's start, [2 *
    num_nodes + 1]."""
    segments, starts = [], []
    for node in range(2 * case.num_nodes):
        starts.append(len(segments))
        cell_ends = list(accumulate([0] + list(cells[node])))
        for lo, hi, row, number in chunks[node]:
            cuts = sorted(set([lo, hi] + [end for end in cell_ends if lo < end < hi]))
            for first, last in zip(cuts, cuts[1:]):
                feature = bisect_right(cell_ends, first) - 1
                chunk_row = row + first - lo
                cell_row = cell_rows[node][feature] + first - cell_ends[feature]
                segments.append(
                    [cell_row, chunk_row, last - first, number]
                    if node >= case.num_nodes
                    else [chunk_row, cell_row, last - first, number]
                )
    starts.append(len(segments))
    return segments, starts


# ── the call ─────────────────────────────────────────────────────────────────


def count_vector(case: Case, part: Dict, node: int) -> torch.Tensor:
    """A requester's counts for its node: its keys for each target's remote
    part, then for each owner of its node and feature."""
    cells = part["counts"].view(case.num_nodes, case.local_size, case.num_features)
    same = cells[node]
    return torch.cat([cells.sum((0, 2)) - same.sum(1), same.flatten()])


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
    common = 2 + 2 * num_nodes  # a count message's common part (CountMessage::common)

    def sizes(rank: int) -> Dict[str, int]:
        return dict(
            num_nodes=num_nodes,
            local_size=local_size,
            node=rank // local_size,
            local_rank=rank % local_size,
            num_features=num_features,
            num_tables=case.num_tables,
            lookup_keys=lookup_keys,
            remote_keys=remote_keys,
            num_proxy_rows=case.num_proxy_rows,
            partial_rows=case.partial_rows,
            chunk_rows=case.slot_rows,
            record_bytes=case.record_bytes,
        )

    layouts, windows = [], []
    for rank, part in enumerate(ranks):
        node, local_rank = divmod(rank, local_size)
        # node_count_exchange: the count vectors' sum over the node and over
        # the ranks before this one, and each requester's counts for it
        vectors = torch.stack(
            [
                count_vector(case, ranks[node * local_size + requester], node)
                for requester in range(local_size)
            ]
        )
        own_columns = vectors[:, local_size + local_rank * num_features :][
            :, :num_features
        ]
        mine = torch.cat([vectors[:, local_rank : local_rank + 1], own_columns], 1)
        layouts.append(
            planner(
                hier_output_dist_node_layout,
                vectors.sum(0).cuda(),
                vectors[:local_rank].sum(0).cuda(),
                mine.contiguous().view(-1).cuda(),
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
            )
        )
        windows.append(make_windows(case, lookup_keys, remote_keys, generator))
    # node_key_push: each requester's valid rows into its targets' node_keys
    # (weights, and the cell of each remote occurrence) where its layout says
    for rank, part in enumerate(ranks):
        node = rank // local_size
        rows = torch.arange(int(layouts[rank]["rows_to_push"]), device="cuda")
        cell = part["row_cell"][rows].long()
        owner = cell // num_features
        destination = layouts[rank]["destination_base"][cell] + rows
        for target in range(local_size):
            mine = owner % local_size == target
            window = windows[node * local_size + target]
            window["node_keys"][destination[mine]] = part["keys"][rows[mine]]
            if case.weights:
                window["node_weights"][destination[mine]] = part["weights"][rows[mine]]
            remote = mine & (destination >= lookup_keys)
            window["remote_cells"][destination[remote] - lookup_keys] = (
                (owner[remote] // local_size) * num_features
                + cell[remote] % num_features
            ).int()
    proxies = []
    for rank, window in enumerate(windows):
        layout = layouts[rank]
        proxies.append(
            planner(
                hier_output_dist_proxy_plan,
                window["node_keys"][lookup_keys:],
                window["node_weights"][lookup_keys:] if case.weights else None,
                window["remote_cells"],
                layout["requester_start"],
                layout["skip"],
                tables,
                layout["need_remote"],
                layout["own_total"][rank % local_size : rank % local_size + 1],
                window["remote_map"],
                window["key_stage"],
                window["proxy_header"],
                case.node_dedup,
                sizes(rank),
            )
        )

    # rail_count_exchange: each rank gets from the rail peer on each node its
    # common part and its part for this rank's node
    def messages_of(rank: int) -> torch.Tensor:
        node, rail_rank = divmod(rank, local_size)
        rows = []
        for source in range(num_nodes):
            message = proxies[source * local_size + rail_rank]["message"]
            first = common + node * num_features
            rows.append(
                torch.cat([message[:common], message[first : first + num_features]])
            )
        return torch.stack(rows)

    def rail_tables(rank: int, **later) -> Dict:
        return planner(
            hier_output_dist_rail_tables,
            messages_of(rank),
            layouts[rank]["own_cells"],
            layouts[rank]["own_starts"],
            proxies[rank]["chunks"],
            proxies[rank]["node_chunk_start"],
            proxies[rank]["node_chunks"],
            proxies[rank]["uniq_off"],
            proxies[rank]["cell_uniq"],
            sizes=sizes(rank),
            **later,
        )

    rails = [rail_tables(rank) for rank in range(world_size)]
    # rail_key_exchange: each proxy's slice for each owner of its rail, and
    # each owner's cell starts for each proxy of its rail
    problems = []
    for rank, rail in enumerate(rails):
        node, rail_rank = divmod(rank, local_size)
        put_src = proxies[rank]["stage_slice"].tolist()
        for owner_node, (skip, dst, size) in enumerate(
            zip(
                rail["put_skip"].tolist(),
                rail["put_dst"].tolist(),
                rail["put_bytes"].tolist(),
            )
        ):
            if owner_node == node:
                continue
            owner = owner_node * local_size + rail_rank
            windows[rank]["rail_cells"].view(num_nodes, num_features)[
                owner_node
            ] = rails[owner]["cell_stage"][node]
            if skip:
                continue
            rail_keys = windows[owner]["rail_keys"]
            if dst + size > 8 * rail_keys.numel():
                problems.append("a put past rail_keys")
                continue
            source = windows[rank]["key_stage"][put_src[owner_node] // 8 :]
            rail_keys[dst // 8 : (dst + size) // 8] = source[: size // 8]
    # node_status_exchange, then the owners' tables, the proxies' put lists
    # and the requesters' tables
    publish = torch.stack([rail["publish"] for rail in rails]).view(
        num_nodes, local_size, -1
    )
    owners, requesters = [], []
    for rank, window in enumerate(windows):
        node, local_rank = divmod(rank, local_size)
        owners.append(
            rail_tables(
                rank,
                rail_cells=window["rail_cells"],
                rail_keys=window["rail_keys"],
                node_keys=window["node_keys"][:lookup_keys],
                node_weights=window["node_weights"][:lookup_keys]
                if case.weights
                else None,
            )
        )
        cell_starts = publish[node, :, 2:].view(local_size, local_size, num_features)
        node_ranks = range(node * local_size, (node + 1) * local_size)
        requesters.append(
            planner(
                hier_output_dist_requester_tables,
                torch.stack([windows[peer]["remote_map"] for peer in node_ranks]),
                torch.stack([windows[peer]["proxy_header"] for peer in node_ranks]),
                cell_starts[:, local_rank].contiguous(),
                ranks[rank]["row_cell"],
                ranks[rank]["cell_start"],
                ranks[rank]["block_start"],
                layouts[rank]["destination_base"],
                sizes(rank),
            )
        )
    torch.cuda.synchronize()  # an access out of range would raise here
    return dict(
        layouts=layouts,
        windows=windows,
        proxies=proxies,
        rails=rails,
        owners=owners,
        requesters=requesters,
        publish=publish,
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

    num_occurrences, num_keys = len(ref["occurrences"]), ref["num_keys"]
    num_chunks = ref["num_chunks"]
    expect("occurrences", int(out["occurrence_count"]), num_occurrences)
    expect("keys", out["key_of_occurrence"][:num_occurrences].tolist(), ref["key_of"])
    expect("cells", out["cell_uniq"].tolist(), ref["cell_uniq"])
    expect("uniq_off", out["uniq_off"].tolist(), ref["uniq_off"])
    expect("chunks", int(out["num_chunks"]), num_chunks)
    expect(
        "chunk table",
        [
            row[:4] + u32(row[4:6]) + row[6:]
            for row in out["chunks"].view(-1, 8).tolist()
        ],
        ref["chunks"],
    )
    expect("chunks_out", out["chunks_out"].tolist(), ref["chunks_out"])
    expect(
        "node_chunk_start", out["node_chunk_start"].tolist(), ref["node_chunk_start"]
    )
    expect(
        "node_chunks",
        out["node_chunks"][: len(ref["node_chunks"])].tolist(),
        ref["node_chunks"],
    )
    expect(
        "reduce_task_starts",
        out["reduce_task_starts"].tolist(),
        ref["reduce_task_starts"],
    )
    words = window["remote_map"].view(3, -1)[:, :num_occurrences].long().cpu()
    expect("key rows", (words[0] & 0xFFFFFFFF).tolist(), ref["word0"])
    expect("slots", words[1].tolist(), ref["slot"])
    expect("push positions", words[2].tolist(), ref["position"])
    header = window["proxy_header"].cpu()
    expect("header chunks", int(header[0]), num_chunks)
    expect("header end", int(header[1]) & 0xFFFFFFFF, ref["stream_end"])
    expect(
        "header ends",
        u32(header[4 : 4 + capacity].tolist()),
        [row[5] for row in ref["chunks"]],
    )
    expect(
        "push tables",
        header[4 + capacity :].view(local_size, capacity + 1, 4).tolist(),
        ref["push_tables"],
    )
    # key_stage: per owner node its chunks, then its keys' records
    stage = window["key_stage"].cpu()
    stage_slice = out["stage_slice"].tolist()
    expect("stage slices", stage_slice, ref["stage_slice"])
    words32 = stage.view(torch.int32)
    got, want = [], []
    for owner_node in range(num_nodes):
        if owner_node == node:
            continue
        for level in range(ref["chunks_out"][owner_node]):
            word = (stage_slice[owner_node] + 16 * level) // 4
            got.append(
                words32[word : word + 2].tolist()
                + u32(words32[word + 2 : word + 4].tolist())
            )
            chunk = ref["chunks"][
                ref["node_chunks"][ref["node_chunk_start"][owner_node] + level]
            ]
            uniq_lo = chunk[1] - ref["uniq_off"][owner_node]
            want.append([uniq_lo, uniq_lo + chunk[2], chunk[4], chunk[5]])
    expect("stage chunks", got, want)
    word = torch.tensor(
        [
            (
                stage_slice[owner_node]
                + 16 * ref["chunks_out"][owner_node]
                + (key - ref["uniq_off"][owner_node]) * case.record_bytes
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
    # the cells against a dedup by merge ranks, as sets of records
    cell_ends = list(accumulate(out["cell_uniq"].tolist()))
    cells: Dict = {}
    for key, record in enumerate(records):
        cells.setdefault(
            divmod(bisect_right(cell_ends, key), case.num_features), []
        ).append(record)
    expect(
        "cells by merge ranks",
        {cell: sorted(cell_records) for cell, cell_records in cells.items()},
        merge_rank_cells(case, ref["occurrences"]),
    )
    # the reducer lists
    reducer_start = out["reducer_start"][: num_keys + 1].tolist()
    expect("reducer starts", reducer_start, ref["reducer_start"])
    entries = out["reducer_entries"][: reducer_start[-1]].tolist()
    expect(
        "reducer entries",
        [(entry & 0xFFFFFFFF, entry >> 32) for entry in entries],
        ref["entries"],
    )
    return problems


def check_requester(
    case: Case,
    rank: int,
    ranks: List[Dict],
    refs: Dict,
    out: Dict,
    owner_offsets,
    remote_keys: int,
):
    """A requester's tables against its proxies' references and its node's
    owners' KJTs."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    num_sources = local_size + num_nodes - 1
    node, local_rank = divmod(rank, local_size)
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    part = ranks[rank]
    valid, cell_start = part["valid"], part["cell_start_list"]
    # A row of its node's owner is in the owner's KJT, cell (feature,
    # local_rank); another node's row reads its key's row at its proxy, and
    # pushes to its occurrence's slot.
    src, dst = [0] * valid, [0] * valid
    at_proxy: List[List] = [[] for _ in range(local_size)]  # (push position, row)
    for cell in range(case.world_size * num_features):
        owner, feature = divmod(cell, num_features)
        owner_node, proxy = divmod(owner, local_size)
        for row in range(cell_start[cell], cell_start[cell + 1]):
            if owner_node == node:
                src[row] = dst[row] = (
                    owner_offsets[owner][feature * num_sources + local_rank]
                    + row
                    - cell_start[cell]
                )
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
    # Its push tables at each proxy made absolute, and its heavy groups:
    # per proxy and chunk the positions of each heavy key it asks.
    capacity = max_chunks(case, remote_keys)
    tables, heavy_groups, group_at = [], [], 0
    for proxy in range(local_size):
        ref = refs[(node, proxy)]
        start = push_order_off[proxy]
        entries = [
            [entry[0] + start, entry[1] + start, entry[2], entry[3] + group_at]
            for entry in ref["push_tables"][local_rank]
        ]
        tables.append(entries)
        for chunk in range(ref["num_chunks"]):
            position = entries[chunk][1]
            for key in ref["groups"][local_rank][chunk]:
                asks = ref["asks"][(local_rank, key)]
                heavy_groups.append([position, position + asks])
                position += asks
        group_at += ref["push_tables"][local_rank][capacity][3]
    expect("push tables", out["push_tables"].tolist(), tables)
    expect(
        "proxy chunks",
        out["proxy_chunks"].tolist(),
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
    groups = len(heavy_groups)
    expect("heavy groups", out["heavy_groups"][:groups].tolist(), heavy_groups)
    subs = [-(-(end - start) // HEAVY) for start, end in heavy_groups]
    first_sub = list(accumulate([0] + subs))
    expect("group_first_sub", out["group_first_sub"][:groups].tolist(), first_sub[:-1])
    expect("group_subs", int(out["group_subs"]), first_sub[-1])
    expect(
        "group_of_sub",
        out["group_of_sub"][: first_sub[-1]].tolist(),
        [group for group, count in enumerate(subs) for _ in range(count)],
    )
    # task starts: chunk after chunk, the proxies in turn from its own one;
    # the backward's each chunk's heavy groups of every proxy, then its rows
    fwd, bwd = [0], [0]

    def tasks(proxy: int, chunk: int):
        if chunk >= refs[(node, proxy)]["num_chunks"]:
            return 0, 0, 0
        entry, next_entry = tables[proxy][chunk], tables[proxy][chunk + 1]
        return (
            ceil32(entry[1] - entry[0]),
            ceil32(next_entry[0] - entry[1]),
            ceil32(next_entry[3] - entry[3]),
        )

    for chunk in range(capacity):
        turn = [(local_rank + k) % local_size for k in range(local_size)]
        for proxy in turn:
            rows, heavy_rows, _ = tasks(proxy, chunk)
            fwd.append(fwd[-1] + rows + heavy_rows)
        for proxy in turn:
            bwd.append(bwd[-1] + tasks(proxy, chunk)[2])
        for proxy in turn:
            bwd.append(bwd[-1] + tasks(proxy, chunk)[0])
    expect("fwd_task_starts", out["fwd_task_starts"].tolist(), fwd)
    expect("bwd_task_starts", out["bwd_task_starts"].tolist(), bwd)
    return problems


def check_backward(
    case: Case,
    node: int,
    ranks: List[Dict],
    refs: Dict,
    proxies,
    requesters,
    remote_keys: int,
    generator,
):
    """A backward of the node with integer gradients: each requester pushes
    its rows of a chunk at a proxy as its push tasks do (rows, then each
    heavy group's sum into its partial slot), each with its task's flag; each
    proxy's reducers sum their keys' entries, which must name those flags.
    Every key gets the sum of its occurrences' gradients, and every flag of
    the call is one task's."""
    local_size = case.local_size
    flags = flag_words(case, remote_keys)
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
            for name in ("push_tables", "push_order", "dst", "heavy_groups")
        }
        for rank in node_ranks
    ]
    for proxy in range(local_size):
        ref = refs[(node, proxy)]
        reducer_start = proxies[node * local_size + proxy]["reducer_start"].tolist()
        entries = proxies[node * local_size + proxy]["reducer_entries"].tolist()
        occurrences_of = [[] for _ in range(ref["num_keys"])]
        for occ, key in enumerate(ref["key_of"]):
            occurrences_of[key].append(ref["occurrences"][occ])
        flags_used = [0] * flags
        for chunk in range(ref["num_chunks"]):
            store: Dict = {}  # slot -> pushes of (gradient, flag)
            for requester, table in enumerate(tables):
                entry = table["push_tables"][proxy][chunk]
                next_entry = table["push_tables"][proxy][chunk + 1]
                row_tasks = ceil32(entry[1] - entry[0])
                last_flag = entry[2] + row_tasks + ceil32(next_entry[3] - entry[3])
                if last_flag > flags:
                    problems.append("flags past flag_words")
                for flag in range(entry[2], min(last_flag, flags)):
                    flags_used[flag] += 1
                for position in range(entry[0], entry[1]):
                    row = table["push_order"][position]
                    store.setdefault(table["dst"][row], []).append(
                        (
                            grads[requester][row],
                            entry[2] + (position - entry[0]) // TASK_ROWS,
                        )
                    )
                for group in range(entry[3], next_entry[3]):
                    start, end = table["heavy_groups"][group]
                    if not entry[1] <= start < end <= next_entry[0]:
                        problems.append("a heavy group outside its chunk's heavy range")
                    rows = table["push_order"][start:end]
                    slots = {table["dst"][row] for row in rows}
                    if len(slots) != 1:
                        problems.append("a heavy group without one slot")
                        continue
                    flag = entry[2] + row_tasks + (group - entry[3]) // TASK_ROWS
                    store.setdefault(slots.pop(), []).append(
                        (sum(grads[requester][row] for row in rows), flag)
                    )
            if any(len(pushes) > 1 for pushes in store.values()):
                problems.append("two pushes to one slot")
            for key in range(ref["num_keys"]):
                if ref["chunk_of"][key] != chunk:
                    continue
                total = 0
                for entry in entries[reducer_start[key] : reducer_start[key + 1]]:
                    flag = entry >> 32
                    if flag < 0:  # no push: a heavy key's own row, or no partial
                        continue
                    pushes = store.get(entry & 0x7FFFFFFF)
                    if not pushes:
                        problems.append("a reducer entry without a push")
                        continue
                    if pushes[0][1] != flag:
                        problems.append("a reducer entry names another task's flag")
                    total += pushes[0][0]
                want = sum(grads[occ[0]][occ[1]] for occ in occurrences_of[key])
                if total != want:
                    problems.append("a key's gradient sum")
        if max(flags_used, default=0) > 1:
            problems.append("flag ranges of tasks overlap")
    return problems


def check_rail(
    case: Case,
    rank: int,
    refs: Dict,
    rails: List[Dict],
    recv: List[int],
    lookup_keys: int,
):
    """What a rank as proxy computed of its rail's owners against what they
    computed: where its slice starts there and its puts; what each owner
    sends it of where its cells start; and its own cell starts as owner."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    num_sources = local_size + num_nodes - 1
    node, rail = divmod(rank, local_size)
    ref = refs[(node, rail)]
    mine = {name: tensor.tolist() for name, tensor in rails[rank].items()}
    problems = []
    for owner_node in range(num_nodes):
        owner = rails[owner_node * local_size + rail]
        skip = owner_node == node or recv[owner_node * local_size + rail] > lookup_keys
        if mine["put_skip"][owner_node] != skip:
            problems.append("put_skip")
        if owner_node == node:
            continue
        column = node if node < owner_node else node - 1
        offsets, slice_in = owner["kjt_offsets"].tolist(), owner["slice_in"].tolist()
        if owner["cell_stage"].tolist()[node] != [
            offsets[feature * num_sources + local_size + column]
            for feature in range(num_features)
        ]:
            problems.append("cell_stage != the owner's KJT offsets")
        if mine["put_dst"][owner_node] != slice_in[node][0]:
            problems.append("put_dst != the owner's slice_in")
        uniq = ref["uniq_off"][owner_node + 1] - ref["uniq_off"][owner_node]
        chunks = ref["chunks_out"][owner_node]
        if mine["put_bytes"][owner_node] != 16 * chunks + uniq * case.record_bytes:
            problems.append("put_bytes")
        if slice_in[node][1] != chunks:
            problems.append("slice_in chunks")
        if owner["uniq_in"].tolist()[node] != uniq:
            problems.append("uniq_in != the proxy's uniq_out")
    offsets = mine["kjt_offsets"]
    if mine["my_cell_starts"] != [
        [offsets[feature * num_sources + source] for feature in range(num_features)]
        for source in range(local_size)
    ]:
        problems.append("my_cell_starts")
    if mine["cell_stage"][node] != [0] * num_features:
        problems.append("cell_stage of its own node")
    return problems


def check_owner(case: Case, rank: int, ranks: List[Dict], refs: Dict, out: Dict):
    """An owner's KJT: per feature its node's keys, each requester's in its
    order, then each other node's node-unique keys of its rail's proxy there,
    in key order."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    num_sources = local_size + num_nodes - 1
    node, rail = divmod(rank, local_size)
    others = [other_node for other_node in range(num_nodes) if other_node != node]
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    lengths, keys, weights = [], [], []
    cell_first = {other_node: 0 for other_node in others}
    for feature in range(num_features):
        cell = rank * num_features + feature
        for requester in range(local_size):
            part = ranks[node * local_size + requester]
            rows = range(
                part["cell_start_list"][cell], part["cell_start_list"][cell + 1]
            )
            lengths.append(len(rows))
            keys += [part["keys_list"][row] for row in rows]
            weights += [
                part["weights_list"][row] if case.weights else 1.0 for row in rows
            ]
        for other_node in others:
            ref = refs[(other_node, rail)]
            count = ref["cell_uniq"][node * num_features + feature]
            first = ref["uniq_off"][node] + cell_first[other_node]
            cell_first[other_node] += count
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
            sum(lengths[feature * num_sources : (feature + 1) * num_sources])
            for feature in range(num_features)
        ],
    )
    expect("kjt_values", out["kjt_values"][:num_recv].tolist(), keys)
    if case.record_bytes == 16:
        expect("kjt_weights", out["kjt_weights"][:num_recv].tolist(), weights)
    return problems


def check_put_lists(case: Case, rank: int, refs: Dict, out: Dict, rail_cells):
    """A rank's put lists, as proxy the backward's and as owner the
    forward's, and the chunks it got as owner, against a plain loop."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    num_sources = local_size + num_nodes - 1
    ring = case.num_proxy_rows
    node, rail = divmod(rank, local_size)
    ref = refs[(node, rail)]
    offsets = out["kjt_offsets"].tolist()

    def node_chunks(proxy_ref: Dict, owner_node: int):
        """A proxy's chunks of an owner node, in order: (uniq_lo, uniq_hi,
        start, end, its place in the stream)."""
        out_chunks = []
        first = proxy_ref["node_chunk_start"][owner_node]
        for level in range(proxy_ref["chunks_out"][owner_node]):
            place = proxy_ref["node_chunks"][first + level]
            chunk = proxy_ref["chunks"][place]
            uniq_lo = chunk[1] - proxy_ref["uniq_off"][owner_node]
            out_chunks.append((uniq_lo, uniq_lo + chunk[2], chunk[4], chunk[5], place))
        return out_chunks

    # the backward's owner nodes: this proxy's chunks and cells, at the rows
    # its owners sent; then the forward's proxy nodes: each source node's
    # chunks for this owner, numbered in the order it got them, at its cells in
    # this owner's KJT
    chunks = [
        [
            (lo, hi, start % ring, place)
            for lo, hi, start, _, place in node_chunks(ref, owner)
        ]
        for owner in range(num_nodes)
    ]
    cells = [
        ref["cell_uniq"][owner_node * num_features :][:num_features]
        for owner_node in range(num_nodes)
    ]
    cell_rows = list(rail_cells)
    fwd_chunks, fwd_chunk_start = [], [0]
    for source in range(num_nodes):
        if source == node:
            chunks.append([])
            cells.append([0] * num_features)
            cell_rows.append([0] * num_features)
            fwd_chunk_start.append(len(fwd_chunks))
            continue
        source_ref = refs[(source, rail)]
        column = source if source < node else source - 1
        got = node_chunks(source_ref, node)
        chunks.append(
            [
                (lo, hi, start % ring, len(fwd_chunks) + nth)
                for nth, (lo, hi, start, _, _) in enumerate(got)
            ]
        )
        fwd_chunks += [[lo, hi, start, end] for lo, hi, start, end, _ in got]
        fwd_chunk_start.append(len(fwd_chunks))
        cells.append(source_ref["cell_uniq"][node * num_features :][:num_features])
        cell_rows.append(
            [
                offsets[feature * num_sources + local_size + column]
                for feature in range(num_features)
            ]
        )
    segments, starts = segments_reference(case, chunks, cells, cell_rows)
    problems = []
    got_chunks = out["fwd_chunks"][: len(fwd_chunks)].tolist()
    if [row[:2] + u32(row[2:]) for row in got_chunks] != fwd_chunks:
        problems.append("fwd_chunks")
    if out["fwd_chunk_start"].tolist() != fwd_chunk_start:
        problems.append("fwd_chunk_start")
    if out["put_list"][: len(segments)].tolist() != segments:
        problems.append("put list")
    if out["put_starts"].tolist() != starts:
        problems.append("put starts")
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
    ranks = [part_one(case, *rows, generator) for rows in job]
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
                refs[(other_node, rail)]["uniq_off"][node + 1]
                - refs[(other_node, rail)]["uniq_off"][node]
                for other_node in range(num_nodes)
                if other_node != node
            )
        )
    need_lookup = max(recv)
    lookup_keys = max(need_lookup + case.lookup_slack, 1)
    laps = max(-(-ref["stream_end"] // case.num_proxy_rows) for ref in refs.values())
    call = run_call(case, ranks, lookup_keys, remote_keys, generator)
    # each node's agreed needs: the most of what its ranks publish
    agreed = call["publish"][:, :, :2].amax(1).tolist()
    checks = {"puts": call["problems"]}
    stats = (
        f"chunks {max(ref['num_chunks'] for ref in refs.values())}, laps {laps}, "
        f"heavy keys {sum(sum(ref['heavy']) for ref in refs.values())}, "
        f"heavy groups {sum(len(ref['asks']) for ref in refs.values())}"
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
        [] if all(row == [need_lookup, need_remote] for row in agreed) else ["needs"]
    )
    proxies, rails, owners = call["proxies"], call["rails"], call["owners"]
    requesters = call["requesters"]
    owner_offsets = [rail["kjt_offsets"].tolist() for rail in rails]
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
                case, rank, ranks, refs, requesters[rank], owner_offsets, remote_keys
            )
            for rank in range(world_size)
        ),
        [],
    )
    checks["backward"] = sum(
        (
            check_backward(
                case, node, ranks, refs, proxies, requesters, remote_keys, generator
            )
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
    checks["put lists"] = sum(
        (
            check_put_lists(
                case,
                rank,
                refs,
                owners[rank],
                call["windows"][rank]["rail_cells"]
                .view(num_nodes, num_features)
                .tolist(),
            )
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
        # requester 0's two rows at proxy 0: chunks in two laps, one partial slot
        dst = requesters[0]["dst"].tolist()
        checks["shape"] = [] if laps >= 2 and dst[0] == dst[1] else ["the case's shape"]
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
