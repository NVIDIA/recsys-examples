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
requester binding reads the proxies' words itself, as its kernel reads its
node peers' windows). The windows hold garbage before the call. The checks:

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
* each rank's count messages, word for word, against sums over the
  features: its own part's keys of each lookup group and its proxy's
  node-unique keys of each (owner node, group), which the op takes as
  differences of two offsets (its node layout's own_offsets and own_total,
  its proxy's node_offset_per_bucket_feature), and each row's part;
* the rail: each proxy's slices start where its owners' offsets say; each
  owner's KJT, the chunks it got, and its lookup_rows' rows against a plain
  reference (each lookup group's keys in a slice of their own, its own part
  first, then each other node's in node order): the row map, where each
  group's slice starts, the group shifts, and the group runs of each source
  node's records, equal to that node's proxy's runs of its keys of the owner
  (the rows its backward puts land in); the needs, the same on every rank.
  The owner's records come from the requester kernel's owner CTAs, on
  proxies of no chunks, and the owner's tables complete that kernel's remote
  task offsets: all 0.

Calls that do not fit, and calls whose keys are not index-deduplicated (a
key past local_size occurrences: the need key_rows), check the needs alone
(and that no access was out of range). Every planner call runs under
``torch.cuda.set_sync_debug_mode("error")``.

Most cases size remote_keys at the largest remote part plus 5. Others size
it 20 times as large (remote_factor) or exactly at it, or give node 0 no
keys across nodes (its proxies get no occurrence in a call that fits): the
proxy's hash table and the loops over its chunks take the valid
occurrences, keys and chunks alone, counts only the device knows, with
max_chunks far above the chunks. Its scans (of the representatives, and of
its keys' rows) are CUB's, over remote_keys + 1 items in tiles (on an H100
3,072 int32 values, or 2,688 int64 values or words), and read and write no
item past the count but the one at it: calls of a few hundred occurrences a
proxy take a tile or two, and calls of 2,000 to 24,000 up to 9 tiles; each
with rows_per_payload 1 (a scan of the rows alone) and above 1 (of the rows
and the counted rows together, in one word, 32 bits each). A few cases
scan the rows and the counted rows as a pair of int64 sums
(``hier_output_dist_row_counts64``), which the sizes choose only when
rows_per_payload x remote_keys reaches 2^32 (tiles of 384 pairs, up to 57).
In seven cases proxy (0, 0) alone gets occurrences (one_proxy_job): 2,688
to 3,073 of 2,687 to 3,072 node-unique keys, so the count and the keys fall
at, before and after the edge of a tile, with remote_keys at the count (the
scans' last item is the count's; of 2,688 keys, by words and by pairs, it
is alone in the last tile) or whole tiles past it. Past its node-unique
keys each word of the proxy's rows before each key keeps the garbage the
hook filled in: the scan writes none of them.

The proxy lays its chunks out on a grid, a warp per chunk; CUB's scan over
its max_chunks + 1 places (tiles of 384) then gives the rows and the reducer
tasks before each chunk, and a thread per place places each chunk in its
ring. As max_chunks chooses, a warp per place of max_chunks, so the calls of
16 places or fewer (as in "weights, 3 nodes" and "2 nodes of 72") take one
CTA. A few cases take a given number of CTAs
(``hier_output_dist_chunk_ctas``): 1 for up to 4,800 chunks (about 300 a
warp; the scan in 13 tiles), 3 in rings of 2 laps and of 3 to 6 laps (the
latter with 3,500 to 7,000 chunks), 2 for 36 owner nodes (more turns in a
round than a warp's lanes), 600 for a few dozen chunks (most CTAs lay out
none), and 7 where node 0's proxies have no chunk. Two give the proxies a
max_chunks below their chunks (``max_chunks_arg``, which no call that fits
remote_keys needs): each proxy lays out the first max_chunks chunks of its
stream, in a ring of many laps, one case at a max_chunks of 12 (one CTA, the
scan in one tile); the checks take the proxies alone, and of a proxy whose
chunks pass max_chunks its chunk layout alone. A proxy still writes the
records of all its keys in key_stage, after the chunks of all its owner
nodes, and the hooks size key_stage by max_chunks: remote_keys 600 past the
need keeps them in it (a case whose records would pass it fails on its shape
before it runs). One case sizes the ring by proxy (0, 0)'s stream
(``ring_from_stream``), which then exactly fills it: a stream of the ring's
rows fits it, so its chunks lie as they are, the last one past lap_span.

The count messages' keys of each lookup group come with one group, 2 and 3
groups and a group per feature, also at 1,100 features (groups of 1, 1,098
and 1 features, and a group per feature): a group's features can hold no
keys (of an own part, or of an owner node at a proxy), and a proxy holds no
keys of its own node.

The proxy's push tables come from a binary search of each requester's
sorted push keys for each chunk below its stream's chunks, and its
positions' end for the chunks past them; the checks compare all
local_size x (max_chunks + 1) entries of each proxy's push tables and of
each requester's, and most are past the chunks where max_chunks is far
above them. A few cases sort the push order by int64 push keys
(``hier_output_dist_push_keys64``), which the sizes choose only when
local_size x (max_chunks + 1) + 1 reaches 2^31: with remote_keys 20 times
the need, without node dedup in a ring that wraps, and at a count of 0.
Each proxy gives the bytes of its push keys, which the checks compare, so
the hook's int64 keys and the sizes' int32 ones show. Two cases sort in
three passes of 8 bits (local_size x (max_chunks + 1) of 2^16 or more), by
int32 and by int64 keys, with remote_keys 85,000 past the need: far past
CUB's single tile, so its onesweep passes run, as at the sizes that choose
int64 keys. In a few, requester 1 of each node asks nothing across nodes,
so at each proxy it has no push position, between requesters 0 and 2's.

The proxy's second step fills the rows before each key, its chunk table,
reduce_task_offsets and its push keys' indices with garbage before its
kernels write them, and makes each push key part 0
(``hier_output_dist_proxy_garbage``), as the windows hold garbage before
the call. So the checks of the chunk table, reduce_task_offsets and the
header's chunk ends, over all max_chunks entries, see a word the proxy
leaves out past its chunks, and the checks of the push tables and
positions a push key it leaves out, also past the valid occurrences (with
every bit set that key would sort last, as the key there does), whatever
the caching allocator gave. With the caching allocator off (under
compute-sanitizer, below) the proxy leaves them as they come, so initcheck
sees a read of a word that nothing wrote.

The requester's kernel also runs alone (``REQUESTER_CASES``): one
requester's calls through its binding, on stand-ins for its node's proxies'
headers and remote_map as the proxy steps leave them (each proxy's chunks;
each requester's occurrences there in its push order by chunk; chunk ends 0
past the chunks; push table entries from the chunks on where its rows
end), against a plain reference of its rules, every table entry, every
row's src and dst and every push position. The kernel reads a proxy's
header up to its chunks and writes the entries past them from the entry at
its chunks; the cases take calls of no rows, of rows past the valid ones
alone, node skips (stale remote_map words), no keys across nodes, proxies
of no chunks and of max_chunks, chunk ends past 2^31, an odd max_chunks
(each push table starts at an odd word, so the kernel's 16-byte loads peel
the words before a 16-byte boundary), lookup groups, 72 GPUs per node and
40 nodes (more than a warp's lanes), max_chunks past 8,192 (a table CTA's
loads in two passes, the scan in many tiles), the rows on 1 and 3 CTAs
(``hier_output_dist_requester_ctas``: a thread's rows in several steps),
and the calls of the design's reference ports (ref_requester and
e2e_requester_check: local_size 1 to 8, 2 to 4 nodes, ends past 2^31, the
proxies' occurrences of each requester from none to 100, node skips).

Run it on one GPU, from ``corelib/dynamicemb``::

    python test/unit_tests/test_hier_dedup.py

Under compute-sanitizer, turn off the caching allocator. Then each tensor is
an allocation of its own: memcheck sees an access past a tensor's end, and
initcheck a read of a word that nothing wrote, whatever the memory held (the
proxy keeps garbage in its hash table past the buckets the call takes and in
its tables per occurrence and per key past the valid ones, and a requester
in its push order past its rows at the proxies; no kernel or check may read
it)::

    export PYTORCH_NO_CUDA_MEMORY_CACHING=1
    compute-sanitizer --tool initcheck python test/unit_tests/test_hier_dedup.py sanitize

and so with ``--tool memcheck``, ``--tool racecheck`` and ``--tool synccheck``.
Under initcheck the whole file takes hours (about 2 minutes a case):
``sanitize`` runs the cases of ``SANITIZE_CASES`` on their first seed, the
plain and weighted ones, a group per feature (the count messages read every
offset of the own part's features and of the proxy's bucket features), a
wrapping ring, node dedup off, an overflow, and those of the proxy's live
counts (20 times the need, at the need, a count of 0, hot keys, several rows
per payload, the scans at the edges of CUB's tiles, the keys' rows by int64
pairs), its chunk layout on 3 CTAs in a wrapping ring and on 7 CTAs with no
chunks, its int64 push keys and a requester with nothing to send; and the
requester cases of ``SANITIZE_REQUESTER_CASES``.
``sanitize past`` runs the cases of ``SANITIZE_PAST_CASES``, whose proxies'
chunks pass max_chunks, under memcheck, racecheck and synccheck alone: the
keys of the chunks past max_chunks read words of chunk_inv that no chunk
wrote, which initcheck reports::

    compute-sanitizer --tool memcheck python test/unit_tests/test_hier_dedup.py sanitize past
"""

import os
import random
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Callable, Dict, List, Optional, Tuple

import torch
from dynamicemb_extensions import (
    hier_output_dist_chunk_ctas,
    hier_output_dist_node_layout,
    hier_output_dist_node_push,
    hier_output_dist_proxy_garbage,
    hier_output_dist_proxy_plan,
    hier_output_dist_push_keys64,
    hier_output_dist_rail_tables,
    hier_output_dist_requester_ctas,
    hier_output_dist_requester_tables,
    hier_output_dist_row_counts64,
)

TASK_ROWS = 32  # kTaskRows: rows of a push task
# the proxy fills tables of its second step with garbage first, unless the
# caching allocator is off (then compute-sanitizer's initcheck sees a word
# that nothing wrote)
PROXY_GARBAGE = "PYTORCH_NO_CUDA_MEMORY_CACHING" not in os.environ


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
    """The most chunks of a stream (Sizes::max_chunks), or the case's smaller
    max_chunks_arg."""
    most = rows_per_payload(case) * remote_keys // chunk_span(case) + case.num_nodes
    return min(case.max_chunks_arg, most) if case.max_chunks_arg else most


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
    # remote_keys: this many times the most a rank needs, plus remote_slack
    # (far above the call: the proxy's counts far below its capacity)
    remote_factor: int = 1
    job: Optional[Callable] = None  # the ranks' rows, when not random
    seeds: int = 2
    # the features of each lookup group (Config::lookup_groups); empty: one group
    groups: Tuple[int, ...] = ()
    # the CTAs of the proxy's chunk layout (hier_output_dist_chunk_ctas); 0: as
    # max_chunks chooses (a warp per place, so one CTA up to 16 places)
    chunk_ctas: int = 0
    # max_chunks (the hooks' sizes); 0: as remote_keys sets it. A smaller one
    # lets a proxy's chunks pass it, which no call that fits remote_keys does:
    # the proxy lays out the first max_chunks of them, and the test checks the
    # proxies alone
    max_chunks_arg: int = 0
    # the ring: the rows of proxy (0, 0)'s stream (they do not depend on the
    # ring), which its stream then exactly fills; False: num_proxy_rows
    ring_from_stream: bool = False
    # the push order's sort by int64 push keys (hier_output_dist_push_keys64);
    # False: by int32 ones, as the sizes choose
    push_keys64: bool = False
    # with rows_per_payload above 1, the proxy's scan of its keys' rows by a
    # pair of int64 sums (hier_output_dist_row_counts64); False: by one word,
    # the two sums in 32 bits each, where they fit, as the sizes choose
    row_counts64: bool = False

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


def one_proxy_job(keys: int, again: int) -> Callable:
    """A job as one_owner_job's: requester 0 of node 0 asks owner (node 1, 0)
    for keys 0, ..., keys - 1, and requester 1 for the first `again` of them.
    So proxy (0, 0) gets keys + again occurrences of `keys` node-unique keys
    (with node dedup), and every other proxy none."""

    def job(case: Case, generator) -> List[Tuple]:
        return one_owner_job(case, [list(range(keys)), list(range(again))])

    return job


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


def local_node_job(case: Case, generator) -> List[Tuple]:
    """As random_job, but node 0's requesters ask only the owners of their
    node: node 0's proxies get no occurrence (a count of 0) in a call that
    fits, so they write their tables of no keys and no chunks."""
    job = random_job(case, generator)
    node_bucket_features = case.local_size * case.num_features  # node 0's owners
    for rank in range(case.local_size):
        counts, keys, weights = job[rank]
        bucket_feature = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        keep = bucket_feature < node_bucket_features
        counts = counts.clone()
        counts[node_bucket_features:] = 0
        job[rank] = (counts, keys[keep], weights[keep])
    return job


def quiet_requester_job(case: Case, generator) -> List[Tuple]:
    """As random_job, but local rank 1 of each node asks only the owners of
    its node: at each proxy requester 1 has no occurrence, between
    requesters that have some (local_size above 2), so its push order there
    is empty and each entry of its push tables is 0."""
    job = random_job(case, generator)
    node_bucket_features = case.local_size * case.num_features
    for node in range(case.num_nodes):
        rank = node * case.local_size + 1
        counts, keys, weights = job[rank]
        bucket_feature = torch.repeat_interleave(torch.arange(counts.numel()), counts)
        first = node * node_bucket_features  # its node's owners' bucket features
        keep = (bucket_feature >= first) & (
            bucket_feature < first + node_bucket_features
        )
        counts = counts.clone()
        counts[:first] = 0
        counts[first + node_bucket_features :] = 0
        job[rank] = (counts, keys[keep], weights[keep])
    return job


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
    # remote_keys 20 times the need: the proxy's hash table and its scan of the
    # representatives take the valid occurrences alone, far fewer than the
    # capacity, and max_chunks is far above the chunks
    Case("remote_keys 20 x the need", remote_factor=20),
    Case(
        "remote_keys 20 x the need, node dedup off, weights",
        max_block=10,
        key_range=60,
        hot=0.3,
        weighted=True,
        node_dedup=False,
        remote_factor=20,
    ),
    Case(
        "remote_keys 20 x the need, 2 rows per payload",
        max_block=40,
        key_range=30,
        hot=0.5,
        max_chunk_rows_arg=40,
        payload_rows=20,
        remote_factor=20,
    ),
    Case(
        "remote_keys 20 x the need, weights, 3 nodes of 2, 6 rows per payload",
        num_nodes=3,
        local_size=2,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=60,
        payload_rows=10,
        remote_factor=20,
    ),
    Case(
        "remote_keys 20 x the need, 2 nodes, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=1280,
        remote_factor=20,
        seeds=1,
    ),
    # remote_keys at the need: the largest remote part fills it
    Case("remote_keys at the need", max_block=20, hot=0.3, remote_slack=0),
    Case(
        "remote_keys at the need, node dedup off",
        max_block=20,
        hot=0.3,
        node_dedup=False,
        remote_slack=0,
    ),
    # node 0's proxies get no occurrence in a call that fits
    Case("no keys across nodes from node 0", job=local_node_job),
    Case(
        "no keys across nodes from node 0, node dedup off, 20 x the need",
        node_dedup=False,
        remote_factor=20,
        job=local_node_job,
    ),
    # many sparse features: 2,400 and 4,400 bucket features at a proxy, most of
    # their blocks empty
    Case(
        "600 features on 4 nodes of 2",
        local_size=2,
        tables=tuple(feature // 5 for feature in range(600)),
        max_block=1,
        key_range=100,
        hot=0.05,
        seeds=1,
    ),
    Case(
        "1,100 features on 4 nodes of 2, weights, 20 x the need",
        local_size=2,
        tables=tuple(feature // 11 for feature in range(1100)),
        max_block=1,
        key_range=100,
        weighted=True,
        remote_factor=20,
        seeds=1,
    ),
    # 1,100 features in lookup groups of 1, 1,098 and 1 features, and in a group
    # per feature: a count message's keys of a group, a difference of two
    # offsets, span one feature (often of no keys) or most of them
    Case(
        "1,100 features on 4 nodes of 2, 3 lookup groups",
        local_size=2,
        tables=tuple(feature // 11 for feature in range(1100)),
        max_block=1,
        key_range=100,
        groups=(1, 1098, 1),
        seeds=1,
    ),
    Case(
        "1,100 features on 2 nodes of 2, a lookup group per feature",
        num_nodes=2,
        local_size=2,
        tables=tuple(feature // 11 for feature in range(1100)),
        max_block=1,
        key_range=100,
        groups=(1,) * 1100,
        seeds=1,
    ),
    # the proxy's scans (CUB's, over remote_keys + 1 items) from one tile to
    # many: calls of a few hundred occurrences a proxy (a tile or two), 20
    # times the need (items past the count), at the need, with a count of 0,
    # and of 2,000 to 24,000 occurrences a proxy (several tiles; on an H100 a
    # tile is 3,072 int32 values, or 2,688 int64 values or words); each with
    # rows_per_payload 1 (a scan of the rows alone) and above 1 (of the rows
    # and the counted rows together)
    Case(
        "thousands of occurrences a proxy, 2 nodes of 2",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        seeds=1,
    ),
    Case("hot keys, another job", max_block=40, key_range=30, hot=0.5),
    Case("hot keys, weights", max_block=40, key_range=30, hot=0.5, weighted=True),
    Case("uniform, another job"),
    Case("remote_keys 20 x the need, another job", remote_factor=20, seeds=1),
    Case(
        "thousands of occurrences a proxy, 2 nodes of 2, another job",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        seeds=1,
    ),
    Case(
        "thousands of occurrences a proxy, 2 nodes of 2, a third job",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        seeds=1,
    ),
    Case(
        "remote_keys at the need, another job",
        max_block=20,
        hot=0.3,
        remote_slack=0,
    ),
    Case("no keys across nodes from node 0, another job", job=local_node_job),
    # rows_per_payload above 1: the scan of the keys' rows takes each key's
    # rows and counted rows together, in one word, or as a pair of int64 sums
    # (hier_output_dist_row_counts64, which the sizes choose only when
    # rows_per_payload * remote_keys reaches 2^32: tiles of 384 pairs on an
    # H100) in calls of thousands of occurrences a proxy (57 tiles), of hot
    # keys and of 20 times the need; the chunks' loops and the places past
    # them, with max_chunks far above the chunks, in a ring that wraps and
    # without node dedup
    Case(
        "thousands of occurrences a proxy, 2 nodes of 2, 4 rows per payload",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        max_chunk_rows_arg=64,
        payload_rows=16,
        seeds=1,
    ),
    Case(
        "a few thousand occurrences a proxy, 2 nodes of 2, 4 rows per payload",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=3000,
        key_range=10000,
        hot=0.05,
        max_chunk_rows_arg=64,
        payload_rows=16,
        seeds=1,
    ),
    Case(
        "the keys' rows by int64 pairs, thousands of occurrences a proxy, "
        "2 nodes of 2, 4 rows per payload",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        max_chunk_rows_arg=64,
        payload_rows=16,
        row_counts64=True,
        seeds=1,
    ),
    Case(
        "the keys' rows by int64 pairs, hot keys, 2 rows per payload",
        max_block=40,
        key_range=30,
        hot=0.5,
        max_chunk_rows_arg=40,
        payload_rows=20,
        row_counts64=True,
    ),
    Case(
        "hot keys, weights, 2 rows per payload",
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=40,
        payload_rows=20,
    ),
    Case(
        "the keys' rows by int64 pairs, 20 x the need, weights, 3 nodes of 2, "
        "6 rows per payload",
        num_nodes=3,
        local_size=2,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=60,
        payload_rows=10,
        remote_factor=20,
        row_counts64=True,
    ),
    Case(
        "no keys across nodes from node 0, 2 rows per payload, 20 x the need",
        max_chunk_rows_arg=16,
        payload_rows=8,
        remote_factor=20,
        job=local_node_job,
    ),
    Case(
        "remote_keys 20 x the need, 2 nodes, a wrapping ring, 4 rows per payload",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=640,
        max_chunk_rows_arg=160,
        payload_rows=40,
        remote_factor=20,
        seeds=1,
    ),
    Case(
        "remote_keys 20 x the need, node dedup off, 3 rows per payload",
        max_block=10,
        key_range=60,
        hot=0.3,
        weighted=True,
        node_dedup=False,
        max_chunk_rows_arg=30,
        payload_rows=10,
        remote_factor=20,
    ),
    # the chunk layout on a given number of CTAs (hier_output_dist_chunk_ctas),
    # a warp per chunk, then CUB's scan of the chunks and a thread per place:
    # on 1 CTA, about 300 chunks a warp and the scan in 13 tiles; on 3 CTAs in
    # a ring of several laps (a lap's first chunk in one tile of the scan, its
    # others in later ones); on 2 CTAs, 36 owner nodes (two ballots of the turn
    # of a round); on 600 CTAs, far more warps than chunks; on 7 CTAs, no
    # chunks at node 0's proxies (the scan writes the totals of none)
    Case(
        "the chunk layout on 1 CTA, 2 nodes of 2",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        chunk_ctas=1,
        seeds=1,
    ),
    Case(
        "the chunk layout on 3 CTAs, 2 nodes of 2, a wrapping ring",
        num_nodes=2,
        local_size=2,
        tables=(0, 1),
        max_block=9000,
        key_range=30000,
        hot=0.05,
        num_proxy_rows=3840,
        chunk_ctas=3,
        seeds=1,
    ),
    Case(
        "2 nodes, a wrapping ring, the chunk layout on 3 CTAs",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=1280,
        chunk_ctas=3,
    ),
    Case(
        "36 nodes of 1, the chunk layout on 2 CTAs",
        num_nodes=36,
        local_size=1,
        tables=(0, 1),
        max_block=6,
        key_range=100,
        hot=0.2,
        num_proxy_rows=1280,
        chunk_ctas=2,
        seeds=1,
    ),
    Case(
        "the chunk layout on 600 CTAs, 20 x the need",
        remote_factor=20,
        chunk_ctas=600,
        seeds=1,
    ),
    Case(
        "no keys across nodes from node 0, the chunk layout on 7 CTAs",
        job=local_node_job,
        chunk_ctas=7,
        seeds=1,
    ),
    # max_chunks below the proxies' chunks (max_chunks_arg), which no call that
    # fits remote_keys needs: a proxy lays out the first max_chunks chunks of its
    # stream, in a ring of many laps, on the grid max_chunks chooses (7 CTAs at
    # 100, one at 12). It still writes the records of all its keys in key_stage,
    # after the chunks of all its owner nodes, those past max_chunks too, and the
    # hooks size key_stage by max_chunks: remote_keys 600 past the need keeps the
    # records in it (run_case refuses a case whose proxy would write past it)
    Case(
        "chunks past max_chunks of 100, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=320,
        remote_slack=600,
        max_chunks_arg=100,
        seeds=1,
    ),
    Case(
        "chunks past max_chunks of 12, one CTA, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=40,
        remote_slack=600,
        max_chunks_arg=12,
        seeds=1,
    ),
    # a ring that proxy (0, 0)'s stream exactly fills (ring_from_stream): a
    # stream of the ring's rows fits it, so its chunks lie as they are, the last
    # one from past lap_span to the ring's end; the other proxies' streams fit
    # or wrap
    Case(
        "a ring that proxy (0, 0)'s stream exactly fills",
        max_block=40,
        key_range=200,
        hot=0.1,
        ring_from_stream=True,
    ),
    # the push order's sort by int64 push keys (hier_output_dist_push_keys64),
    # which the sizes choose only when local_size * (max_chunks + 1) + 1
    # reaches 2^31: the same push order, push tables and positions as by
    # int32 keys, with max_chunks far above the chunks, without node dedup
    # in a ring that wraps, and at a count of 0
    Case("the push order by int64 push keys", push_keys64=True),
    Case(
        "the push order by int64 push keys, 20 x the need, weights, 3 nodes of 2, "
        "6 rows per payload",
        num_nodes=3,
        local_size=2,
        tables=(0, 0, 1),
        max_block=40,
        key_range=30,
        hot=0.5,
        weighted=True,
        max_chunk_rows_arg=60,
        payload_rows=10,
        remote_factor=20,
        push_keys64=True,
    ),
    Case(
        "the push order by int64 push keys, node dedup off, 2 nodes, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        node_dedup=False,
        num_proxy_rows=1280,
        push_keys64=True,
        seeds=1,
    ),
    Case(
        "no keys across nodes from node 0, the push order by int64 push keys",
        job=local_node_job,
        push_keys64=True,
        seeds=1,
    ),
    # the push order's sort in three passes of 8 bits (local_size *
    # (max_chunks + 1) of 2^16 or more) by int32 and by int64 push keys, with
    # remote_keys 85,000 past the need, whatever the job: far past CUB's
    # single tile, so its onesweep passes run, as at the sizes that choose
    # int64 keys
    Case(
        "the push order in three passes, remote_keys 85,000 past the need, 2 nodes, "
        "a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=1280,
        remote_slack=85000,
        seeds=1,
    ),
    Case(
        "the push order in three passes by int64 push keys, remote_keys 85,000 past "
        "the need, 2 nodes, a wrapping ring",
        num_nodes=2,
        max_block=200,
        key_range=600,
        hot=0.15,
        num_proxy_rows=1280,
        remote_slack=85000,
        push_keys64=True,
        seeds=1,
    ),
    # requester 1 of each node asks nothing across nodes: at each proxy no
    # push position of it, between requesters 0 and 2's, and with max_chunks
    # far above the chunks
    Case(
        "requester 1 asks nothing across nodes, 20 x the need",
        remote_factor=20,
        job=quiet_requester_job,
    ),
    Case(
        "requester 1 asks nothing across nodes, node dedup off, int64 push keys",
        node_dedup=False,
        job=quiet_requester_job,
        push_keys64=True,
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
    # the proxy's scans at the edges of CUB's tiles on an H100 (3,072 int32
    # values for the representatives, 2,688 int64 values or words for the
    # keys' rows, and 384 pairs under the hook): proxy (0, 0) alone gets
    # occurrences (one_proxy_job), the count and its node-unique keys at a
    # tile's edge, one before it and one after it, and remote_keys at the
    # count (the scans' last item is the count's) or past it (whole tiles past
    # the count). In the last two cases the node-unique keys are remote_keys,
    # 2,688 (7 tiles of pairs): the item of the rows' scan at their count, the
    # total, is alone in the last tile, of words and of pairs
    Case(
        "3,072 occurrences of 2,688 keys at one proxy, remote_keys at them",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        max_chunk_rows_arg=64,
        remote_slack=0,
        job=one_proxy_job(2688, 384),
        seeds=1,
    ),
    Case(
        "3,071 occurrences of 2,687 keys at one proxy, remote_keys at them",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        max_chunk_rows_arg=64,
        remote_slack=0,
        job=one_proxy_job(2687, 384),
        seeds=1,
    ),
    Case(
        "3,072 occurrences of 2,688 keys at one proxy, 4 rows per payload",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        max_chunk_rows_arg=64,
        payload_rows=16,
        remote_slack=0,
        job=one_proxy_job(2688, 384),
        seeds=1,
    ),
    Case(
        "3,073 occurrences of 2,689 keys at one proxy, 4 rows per payload, "
        "remote_keys 6,143",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        max_chunk_rows_arg=64,
        payload_rows=16,
        remote_slack=3070,
        job=one_proxy_job(2689, 384),
        seeds=1,
    ),
    Case(
        "3,072 occurrences at one proxy, node dedup off, remote_keys at them",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        node_dedup=False,
        max_chunk_rows_arg=64,
        remote_slack=0,
        job=one_proxy_job(3072, 0),
        seeds=1,
    ),
    Case(
        "2,688 occurrences at one proxy, node dedup off, 4 rows per payload, "
        "remote_keys at them",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        node_dedup=False,
        max_chunk_rows_arg=64,
        payload_rows=16,
        remote_slack=0,
        job=one_proxy_job(2688, 0),
        seeds=1,
    ),
    Case(
        "the keys' rows by int64 pairs, 2,688 occurrences at one proxy, "
        "node dedup off, 4 rows per payload, remote_keys at them",
        num_nodes=2,
        local_size=2,
        tables=(0,),
        node_dedup=False,
        max_chunk_rows_arg=64,
        payload_rows=16,
        remote_slack=0,
        job=one_proxy_job(2688, 0),
        row_counts64=True,
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
    # the chunks the proxy lays out: the stream's first max_chunks (no call that
    # fits remote_keys has more; a case's max_chunks_arg may be fewer)
    num_chunks = min(len(stream), capacity)
    chunk_rows, chunk_keys, chunk_counted = [], [], []
    for owner_node, j in stream[:num_chunks]:
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
    for chunk, (owner_node, j) in enumerate(stream[:num_chunks]):
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
        stream_chunks=len(stream),
        stream_rows=total,
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
    case: Case,
    ranks: List[Dict],
    lookup_keys: int,
    remote_keys: int,
    generator,
    proxies_only: bool = False,
):
    """Every rank's planner steps of one call, with what the comm kernels do
    between them; with proxies_only, up to the proxies' steps."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    world_size = case.world_size
    tables = torch.tensor(case.tables, dtype=torch.int32, device="cuda")

    def sizes(rank: int) -> Dict[str, int]:
        named = dict(
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
        if case.max_chunks_arg:
            named["max_chunks"] = max_chunks(case, remote_keys)
        return named

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
                layout["own_offsets"],
                layout["own_total"],
                window["remote_map"],
                window["key_stage"],
                window["proxy_header"],
                case.node_dedup,
                sizes(rank),
                list(case.groups),
            )
        )
    if proxies_only:
        torch.cuda.synchronize()  # an access out of range would raise here
        return dict(windows=windows, proxies=proxies)

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
    expect("node_num_uniques", int(out["node_num_uniques"]), node_num_uniques)
    expect("need_key_rows", int(out["need_key_rows"]), ref["need_key_rows"])
    # the push keys' width: int64 under the hook, or when local_size *
    # (max_chunks + 1) + 1 reaches 2^31; else int32
    expect(
        "push key bytes",
        int(out["push_key_bytes"]),
        8 if case.push_keys64 or local_size * (capacity + 1) + 1 >= 1 << 31 else 4,
    )
    expect(
        "keys", out["node_reverse_indices"][:num_occurrences].tolist(), ref["key_of"]
    )
    expect(
        "node_length_per_bucket_feature",
        out["node_length_per_bucket_feature"].tolist(),
        ref["node_length_per_bucket_feature"],
    )
    # where the keys of each bucket feature start, then all of them
    expect(
        "node_offset_per_bucket_feature",
        out["node_offset_per_bucket_feature"].tolist(),
        list(accumulate([0] + ref["node_length_per_bucket_feature"])),
    )
    expect(
        "node_offset_per_bucket",
        out["node_offset_per_bucket"].tolist(),
        ref["node_offset_per_bucket"],
    )
    node_offset_per_bucket = ref["node_offset_per_bucket"]
    expect(
        "node_length_per_bucket",
        out["node_length_per_bucket"].tolist(),
        [
            node_offset_per_bucket[owner_node + 1] - node_offset_per_bucket[owner_node]
            for owner_node in range(num_nodes)
        ],
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
    # the rows before each key; past them the scan writes nothing, so each word
    # keeps the garbage the hook filled in (every bit set)
    expect(
        "node_exp_off",
        out["node_exp_off"][: node_num_uniques + 1].tolist(),
        ref["node_exp_off"],
    )
    if PROXY_GARBAGE:
        expect(
            "node_exp_off past the keys",
            bool((out["node_exp_off"][node_num_uniques + 1 :] == -1).all()),
            True,
        )
    return problems


def check_chunk_layout(
    case: Case, ref: Dict, out: Dict, window: Dict, remote_keys: int
) -> List[str]:
    """The chunk layout of a proxy whose chunks pass max_chunks (a case's
    max_chunks_arg) against its reference: it lays out the first max_chunks
    chunks of its stream, as a stream of no more, and keeps each owner node's
    chunks whole (their counts, offsets and slices of key_stage). Its keys of
    the chunks past max_chunks take a chunk from words of chunk_inv that no
    chunk wrote (a call that does not fit may read them), so their rows,
    slots, push order and pulls are not compared."""
    capacity = max_chunks(case, remote_keys)
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    num_chunks = ref["num_chunks"]
    expect("chunks", int(out["num_chunks"]), num_chunks)
    for name in (
        "chunk_length_per_bucket",
        "chunk_offset_per_bucket",
        "stage_offset_per_bucket",
        "reduce_task_offsets",
    ):
        expect(name, out[name].tolist(), ref[name])
    # the chunk table, but each chunk's pulls
    expect(
        "chunk table",
        [
            row[:3] + u32(row[4:6]) + row[6:]
            for row in out["chunks"].view(-1, 8).tolist()
        ],
        [row[:3] + row[4:] for row in ref["chunks"]],
    )
    header = window["proxy_header"].cpu()
    expect("header chunks", int(header[0]), num_chunks)
    expect("header end", int(header[1]) & 0xFFFFFFFF, ref["stream_end"])
    expect(
        "header ends",
        u32(header[2 : 2 + capacity].tolist()),
        [row[5] for row in ref["chunks"]],
    )
    # each chunk laid out at its round in its owner node's slice of key_stage
    words32 = window["key_stage"].cpu().view(torch.int32)
    got, want = [], []
    for owner_node, key_lo, keys, _, start, end, j, _ in ref["chunks"][:num_chunks]:
        word = (ref["stage_offset_per_bucket"][owner_node] + 16 * j) // 4
        got.append(
            words32[word : word + 2].tolist()
            + u32(words32[word + 2 : word + 4].tolist())
        )
        record_lo = key_lo - ref["node_offset_per_bucket"][owner_node]
        want.append([record_lo, record_lo + keys, start, end])
    expect("stage chunks", got, want)
    return problems


def check_message(case: Case, rank: int, ranks: List[Dict], refs: Dict, out: Dict):
    """The count messages a rank sends its rail (out["message"], row n to the
    peer on node n), word for word, against sums over the features. Each row:
    the header (its node's largest remote part; as proxy the most occurrences
    of one of its keys; the keys of each lookup group of its own part, over
    its node's requesters; as proxy its chunks per owner node and its
    node-unique keys of each (owner node, group)), then row n's part: its
    node-unique keys of each feature of node n. The op takes each key count
    of the header as a difference of two offsets."""
    num_nodes, local_size, num_features = (
        case.num_nodes,
        case.local_size,
        case.num_features,
    )
    node, rail = divmod(rank, local_size)
    ref = refs[(node, rail)]
    groups = case.group_ranges
    num_groups = len(groups)
    header = 2 + num_groups + num_nodes * (num_groups + 1)
    lengths = ref["node_length_per_bucket_feature"]
    problems = []

    def expect(label: str, got, want) -> None:
        if got != want:
            problems.append(label)

    # its own part's keys of each feature: those of its node's requesters for it
    own = [0] * num_features
    for requester in range(local_size):
        counts = ranks[node * local_size + requester]["counts"]
        for feature, keys in enumerate(
            counts[rank * num_features : (rank + 1) * num_features].tolist()
        ):
            own[feature] += keys
    # the largest remote part of its node: the most occurrences of a proxy
    need_remote = max(
        len(refs[(node, proxy)]["occurrences"]) for proxy in range(local_size)
    )
    rows = out["message"].tolist()
    expect("message rows", len(rows), num_nodes)
    for owner_node, row in enumerate(rows):
        expect("message length", len(row), header + num_features)
        expect("message needs", row[:2], [need_remote, ref["need_key_rows"]])
        expect(
            "message own keys of each group",
            row[2 : 2 + num_groups],
            [sum(own[lo:hi]) for lo, hi in groups],
        )
        expect(
            "message chunks",
            row[2 + num_groups : 2 + num_groups + num_nodes],
            ref["chunk_length_per_bucket"],
        )
        expect(
            "message keys of each (owner node, group)",
            row[2 + num_groups + num_nodes : header],
            [
                sum(lengths[other * num_features + lo : other * num_features + hi])
                for other in range(num_nodes)
                for lo, hi in groups
            ],
        )
        expect(
            "message part",
            row[header:],
            lengths[owner_node * num_features : (owner_node + 1) * num_features],
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
    # each requester's tables, of its push order the positions it writes (its
    # rows at the proxies)
    tables = [
        {
            "proxy_push_tables": requesters[rank]["proxy_push_tables"].tolist(),
            "push_order": requesters[rank]["push_order"][
                : int(requesters[rank]["push_order_off"][-1])
            ].tolist(),
            "dst": requesters[rank]["dst"].tolist(),
        }
        for rank in node_ranks
    ]
    for proxy in range(local_size):
        ref = refs[(node, proxy)]
        # the rows before each key, as far as the proxy writes them
        node_exp_off = proxies[node * local_size + proxy]["node_exp_off"][
            : ref["node_num_uniques"] + 1
        ].tolist()
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
    remote_keys = max(case.remote_factor * need_remote + case.remote_slack, 1)
    refs = {
        where: proxy_reference(case, occs, where[0], remote_keys)
        for where, occs in occurrences.items()
    }
    # The shapes that a case needs, refused before anything runs. Under a
    # max_chunks below a proxy's chunks, the records of all its keys still
    # land in key_stage, which the hooks size by max_chunks, after the chunks
    # of all its owner nodes.
    if case.max_chunks_arg:
        stage_bytes = (
            max_chunks(case, remote_keys) * 16 + remote_keys * case.record_bytes
        )
        if any(
            ref["stream_chunks"] * 16 + ref["node_num_uniques"] * case.record_bytes
            > stage_bytes
            for ref in refs.values()
        ):
            return {"shape": ["records past key_stage"]}, "not run"
    if case.ring_from_stream:
        # A ring of proxy (0, 0)'s stream rows (they do not depend on the
        # ring), which the stream then exactly fills. The ring holds two
        # chunks of the most rows (plan_sizes), and the stream's last chunk
        # has fewer rows, so it starts past lap_span: were the stream not to
        # fit, that chunk would start its next lap.
        ref = refs[(0, 0)]
        last = ref["chunks"][max(ref["num_chunks"] - 1, 0)]
        if (
            ref["stream_rows"] < 2 * case.max_chunk_rows
            or last[5] - last[4] >= case.max_chunk_rows
        ):
            return {"shape": ["a ring the stream exactly fills"]}, "not run"
        case = replace(case, num_proxy_rows=ref["stream_rows"])
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
    hier_output_dist_chunk_ctas(case.chunk_ctas)
    hier_output_dist_proxy_garbage(PROXY_GARBAGE)
    hier_output_dist_push_keys64(case.push_keys64)
    hier_output_dist_row_counts64(case.row_counts64)
    try:
        call = run_call(
            case,
            ranks,
            lookup_keys,
            remote_keys,
            generator,
            proxies_only=bool(case.max_chunks_arg),
        )
    finally:
        hier_output_dist_chunk_ctas(0)
        hier_output_dist_proxy_garbage(False)
        hier_output_dist_push_keys64(False)
        hier_output_dist_row_counts64(False)
    if case.max_chunks_arg:
        # max_chunks below the chunks of some proxies: the proxies alone, and of
        # a proxy whose chunks pass max_chunks its chunk layout alone
        capacity = max_chunks(case, remote_keys)
        checks = {"proxies": [], "shape": []}
        for rank in range(world_size):
            ref = refs[divmod(rank, local_size)]
            proxy, window = call["proxies"][rank], call["windows"][rank]
            if ref["stream_chunks"] > capacity:
                checks["proxies"] += check_chunk_layout(
                    case, ref, proxy, window, remote_keys
                )
            else:
                checks["proxies"] += check_proxy(
                    case, rank // local_size, ref, proxy, window, remote_keys
                )
        past = max(ref["stream_chunks"] for ref in refs.values())
        if past <= capacity or laps < 2:
            checks["shape"].append("the case's shape")
        stats = (
            f"chunks {past} past max_chunks {capacity}, laps {laps}, "
            f"largest remote part {need_remote} of {remote_keys}"
        )
        return checks, stats
    # each node's agreed needs: the most of what its ranks publish
    agreed = call["needs"].amax(1).tolist()
    checks = {"puts": call["problems"]}
    stats = (
        f"chunks {max(ref['num_chunks'] for ref in refs.values())} "
        f"of {max_chunks(case, remote_keys)}, laps {laps}, "
        f"most occurrences of a key {need_key_rows}, "
        f"largest remote part {need_remote} of {remote_keys}"
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
    checks["messages"] = sum(
        (
            check_message(case, rank, ranks, refs, proxies[rank])
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
    # the owner tables complete the remote task offsets of the requester
    # kernel whose owner CTAs made their records, here on proxies of no chunks
    checks["remote task offsets"] = [
        f"rank {rank}"
        for rank in range(world_size)
        if any(owners[rank]["remote_task_offsets"].tolist())
    ]
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


# ── the requester's kernel on stand-ins ──────────────────────────────────────
#
# Calls of one requester through its hook alone, on stand-ins for its node's
# proxies' headers and remote_map (as the proxy steps write them), against a
# plain reference of its rules. Each case's fields that are tuples are drawn
# per call.


@dataclass
class RequesterCase:
    name: str
    num_nodes: Tuple[int, ...] = (3,)
    local_size: Tuple[int, ...] = (4,)
    num_features: Tuple[int, ...] = (3,)
    rows: Tuple[int, ...] = (0, 1, 3)  # a requester's rows of one bucket feature
    bound: Tuple[int, ...] = (0, 5)  # its rows past the valid ones
    # the other requesters' occurrences at a proxy
    others: Tuple[int, ...] = (0, 0, 3, 17)
    # remote_keys at least (raised to the occurrences of a proxy), and with
    # odd_chunks raised until max_chunks is odd
    remote_keys: Tuple[int, ...] = (8, 40)
    odd_chunks: bool = False
    max_chunk_rows_arg: int = 0  # 0: the least, 2 * local_size
    payload_rows: int = 0  # 0: max_chunk_rows (rows_per_payload 1)
    # each proxy's chunks: "random" (none, max_chunks or any), "full"
    # (max_chunks), "zero_or_full" (alternately), "zero" (no keys across nodes)
    chunks: str = "random"
    skip: float = 0.0  # the share of calls whose node skips
    high_ends: float = 0.0  # the share of proxies whose chunk ends pass 2^31
    groups: bool = False  # random lookup groups
    row_ctas: int = 0  # the requester kernel's row CTAs (0: as its sizes choose)
    calls: int = 4


@dataclass
class RequesterSizes:
    """What the sizing helpers above read of a Case, for one requester call."""

    num_nodes: int
    local_size: int
    max_chunk_rows: int
    chunk_payload_rows: int
    max_chunks_arg: int = 0  # max_chunks as the sizes give it


def requester_call(case: RequesterCase, rng: random.Random) -> Dict:
    """One requester's call, with each field of `case` drawn: its node's proxies
    as the proxy steps leave them, and its bucketized rows. A proxy has C
    chunks; each requester's occurrences there take chunks below C, and its
    push order there sorts them by chunk, stably. The proxy's header holds C,
    its stream's end, each chunk's end (0 past C) and per requester where its
    rows of each chunk start in its push order, the end of its rows from C
    on; its remote_map each occurrence's key row and slot (anything) and its
    push position. Under skip no occurrence is valid: C is 0, each push table
    entry the requester's -remote_offsets, and remote_map holds stale words
    (push positions all different, so no two rows take one). The requester's
    rows of the owners (n, p) of the other nodes are its occurrences at
    proxy p, in order, where row_shift places them."""
    draw = {
        name: rng.choice(getattr(case, name))
        for name in (
            "num_nodes",
            "local_size",
            "num_features",
            "rows",
            "bound",
            "remote_keys",
        )
    }
    num_nodes, local_size, num_features = (
        draw["num_nodes"],
        draw["local_size"],
        draw["num_features"],
    )
    world_size = num_nodes * local_size
    node, me = rng.randrange(num_nodes), rng.randrange(local_size)
    skip = rng.random() < case.skip
    # the sizes as plan_sizes makes them (max_chunks from remote_keys)
    max_chunk_rows = case.max_chunk_rows_arg or 2 * local_size
    sized = RequesterSizes(
        num_nodes, local_size, max_chunk_rows, case.payload_rows or max_chunk_rows
    )
    # each proxy's chunks, as a share of max_chunks (0: none, 1: all of them);
    # "random": none, all, or any, as the design's reference port draws them
    if case.chunks == "zero":
        share = [0.0] * local_size
    elif case.chunks == "full":
        share = [1.0] * local_size
    elif case.chunks == "zero_or_full":
        share = [float(proxy % 2) for proxy in range(local_size)]
    else:
        share = [
            rng.choice([0.0, 1.0, rng.random(), rng.random()])
            for _ in range(local_size)
        ]
    # the requester's rows of each bucket feature: in a call that fits none
    # of a proxy without chunks
    counts = []
    for bucket_feature in range(world_size * num_features):
        owner = bucket_feature // num_features
        across = owner // local_size != node
        none = across and share[owner % local_size] == 0.0 and not skip
        counts.append(0 if none else rng.randrange(draw["rows"] + 1))
    offsets = list(accumulate([0] + counts))
    valid = offsets[-1]
    num_rows = valid + draw["bound"]
    # its occurrences at each proxy, in its order there
    mine = [0] * local_size
    for bucket_feature, count in enumerate(counts):
        owner = bucket_feature // num_features
        if owner // local_size != node:
            mine[owner % local_size] += count
    occurrences = [
        [
            mine[proxy]
            if requester == me
            else (0 if share[proxy] == 0.0 and not skip else rng.choice(case.others))
            for requester in range(local_size)
        ]
        for proxy in range(local_size)
    ]
    remote_keys = max([draw["remote_keys"]] + [sum(o) for o in occurrences])
    while case.odd_chunks and max_chunks(sized, remote_keys) % 2 == 0:
        remote_keys += 1
    chunks_max = max_chunks(sized, remote_keys)
    lookup_keys = rng.choice([1, 7, 1000])
    words = 2 + chunks_max + local_size * (chunks_max + 1)
    # under skip, where the requester's rows land in its push order: all
    # different, some past its rows
    push_off = list(accumulate([0] + mine))
    stale_targets = rng.sample(range(-num_rows - 5, 3 * num_rows + 6), sum(mine))
    headers, remote_maps, before_me = [], [], []
    for proxy in range(local_size):
        if skip:
            chunks = 0
        else:
            chunks = min(chunks_max, int(share[proxy] * chunks_max))
            if 0.0 < share[proxy] < 1.0 and chunks == 0:
                chunks = 1
        header = [0] * words
        header[0] = chunks
        at = (
            rng.choice([(1 << 31) - 7, (1 << 32) - 3000])
            if rng.random() < case.high_ends
            else rng.randrange(100)
        )
        for chunk in range(chunks):
            at += rng.randrange(1, 4000)
            header[2 + chunk] = at & 0xFFFFFFFF
        header[1] = header[1 + chunks] if chunks else 0
        remote_offsets = list(accumulate([0] + occurrences[proxy]))
        positions = []
        for requester in range(local_size):
            n = occurrences[proxy][requester]
            chunk_of = [rng.randrange(chunks) if chunks else 0 for _ in range(n)]
            by_chunk = sorted(range(n), key=lambda o: (chunk_of[o], o))
            position = [0] * n
            for place, occurrence in enumerate(by_chunk):
                position[occurrence] = place
            if skip and requester == me:
                first = push_off[proxy]
                position = [
                    stale_targets[first + i] - push_off[proxy] for i in range(n)
                ]
            elif skip:
                position = [rng.randrange(-(1 << 20), 1 << 20) for _ in range(n)]
            positions += position
            ordered = sorted(chunk_of)
            table = 2 + chunks_max + requester * (chunks_max + 1)
            for chunk in range(chunks_max + 1):
                if skip:
                    header[table + chunk] = -remote_offsets[requester]
                elif chunk < chunks:
                    header[table + chunk] = bisect_left(ordered, chunk)
                else:
                    header[table + chunk] = n
        before_me.append(remote_offsets[me])
        past = [
            rng.randrange(-(1 << 20), 1 << 20)
            for _ in range(remote_keys - len(positions))
        ]
        remote_maps.append(
            [rng.randrange(1 << 20) for _ in range(2 * remote_keys)] + positions + past
        )
        headers.append(header)
    # row_shift: a row of owner (n, p) of another node goes to its occurrence
    # at p, lookup_keys + its place in p's remote part
    row_shift = [
        rng.randrange(-(1 << 20), 1 << 20) for _ in range(world_size * num_features + 1)
    ]
    taken = [0] * local_size
    for bucket_feature, count in enumerate(counts):
        owner = bucket_feature // num_features
        if owner // local_size == node or count == 0:
            continue
        proxy = owner % local_size
        row_shift[bucket_feature] = (
            lookup_keys + before_me[proxy] + taken[proxy] - offsets[bucket_feature]
        )
        taken[proxy] += count
    groups = []
    if case.groups:
        cuts = sorted(rng.sample(range(1, num_features), rng.randrange(num_features)))
        groups = [b - a for a, b in zip([0] + cuts, cuts + [num_features])]
    num_groups = len(groups) or 1
    return dict(
        num_nodes=num_nodes,
        local_size=local_size,
        node=node,
        me=me,
        num_features=num_features,
        lookup_keys=lookup_keys,
        remote_keys=remote_keys,
        max_chunks=chunks_max,
        max_chunk_rows=sized.max_chunk_rows,
        payload_rows=sized.chunk_payload_rows,
        groups=groups,
        num_groups=num_groups,
        skip=skip,
        headers=headers,
        remote_maps=remote_maps,
        counts=counts,
        offsets=offsets,
        num_rows=num_rows,
        row_shift=row_shift,
        owner_group_shift=[
            rng.randrange(-1000, 1000) for _ in range(local_size * num_groups)
        ],
    )


def requester_reference(call: Dict) -> Dict:
    """The requester's tables by its rules, in plain Python: where its rows at
    each proxy start in its push order; each proxy's chunks (its header's,
    clamped into [0, max_chunks]), chunk ends and stream end (uint32 words),
    and its push table there made absolute; the remote task offsets, chunk
    after chunk, the proxies in turn from its own, 32 positions a task, none
    past a proxy's chunks; and per row its src and dst (a row of its node's
    owner: its place in the owner's own part plus its group's shift; a valid
    row of another node's owner: its occurrence's key row and slot in its
    proxy's remote_map; past the valid rows: -1) and its push position."""
    local_size, num_nodes, num_features = (
        call["local_size"],
        call["num_nodes"],
        call["num_features"],
    )
    node, me, chunks_max = call["node"], call["me"], call["max_chunks"]
    world_size = num_nodes * local_size
    offsets, row_shift = call["offsets"], call["row_shift"]
    opb = offsets[::num_features]
    groups = call["groups"] or [num_features]
    group_of_feature = [g for g, size in enumerate(groups) for _ in range(size)]
    push_order_off = list(
        accumulate(
            [0]
            + [
                sum(
                    opb[n * local_size + p + 1] - opb[n * local_size + p]
                    for n in range(num_nodes)
                    if n != node
                )
                for p in range(local_size)
            ]
        )
    )
    table_word = 2 + chunks_max + me * (chunks_max + 1)
    headers = call["headers"]
    num_chunks = [min(max(h[0], 0), chunks_max) for h in headers]
    ends = [
        [h[2 + c] & 0xFFFFFFFF for c in range(chunks_max)] + [h[1] & 0xFFFFFFFF]
        for h in headers
    ]
    tables = [
        [i32(h[table_word + c] + push_order_off[p]) for c in range(chunks_max + 1)]
        for p, h in enumerate(headers)
    ]
    remote_task_offsets = [0]
    for index in range(chunks_max * local_size):
        chunk, proxy = index // local_size, (me + index % local_size) % local_size
        tasks = 0
        if chunk < num_chunks[proxy]:
            table = headers[proxy][table_word:]
            tasks = ceil32(table[chunk + 1] - table[chunk])
        remote_task_offsets.append(remote_task_offsets[-1] + tasks)
    remote_keys, lookup_keys = call["remote_keys"], call["lookup_keys"]
    valid = offsets[-1]
    bucket_feature_of_row = [
        bf for bf, count in enumerate(call["counts"]) for _ in range(count)
    ] + [world_size * num_features] * (call["num_rows"] - valid)
    src, dst, push_order = [], [], {}
    for row, bucket_feature in enumerate(bucket_feature_of_row):
        owner = min(bucket_feature // num_features, world_size - 1)
        proxy = owner % local_size
        if owner // local_size == node:
            group = group_of_feature[bucket_feature % num_features]
            place = i32(
                row
                + row_shift[bucket_feature]
                + call["owner_group_shift"][proxy * call["num_groups"] + group]
            )
            src.append(place)
            dst.append(place)
            continue
        if row >= valid:
            src.append(-1)
            dst.append(-1)
            continue
        occurrence = min(
            max(row_shift[bucket_feature] + row - lookup_keys, 0), remote_keys - 1
        )
        remote_map = call["remote_maps"][proxy]
        src.append(remote_map[occurrence])
        dst.append(remote_map[remote_keys + occurrence])
        position = push_order_off[proxy] + remote_map[2 * remote_keys + occurrence]
        if 0 <= position < call["num_rows"]:
            push_order[position] = row
    return dict(
        bucket_feature_of_row=bucket_feature_of_row,
        push_order_off=push_order_off,
        proxy_num_chunks=num_chunks,
        proxy_ends=ends,
        proxy_push_tables=tables,
        remote_task_offsets=remote_task_offsets,
        src=src,
        dst=dst,
        push_order=push_order,
    )


def i32(value: int) -> int:
    """An int64 value cast to int32 (two's complement)."""
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def run_requester_case(
    case: RequesterCase, seed: int
) -> Tuple[Dict[str, List[str]], str]:
    """Every call of the case through the hook, against the reference."""
    rng = random.Random(seed)
    checks = {"tables": [], "maps": [], "push order": []}
    stats = dict(calls=0, rows=0, chunks0=0, chunks_full=0, odd_max_chunks=0, skips=0)
    stats.update(high_ends=0, most_chunks=0, most_proxies=0, most_nodes=0)
    hier_output_dist_requester_ctas(case.row_ctas)
    try:
        for _ in range(case.calls):
            call = requester_call(case, rng)
            ref = requester_reference(call)
            local_size = call["local_size"]
            out = planner(
                hier_output_dist_requester_tables,
                torch.tensor(call["remote_maps"], dtype=torch.int32).cuda(),
                torch.tensor(call["headers"], dtype=torch.int64)
                .to(torch.int32)
                .cuda()
                .view(-1),
                torch.tensor(call["owner_group_shift"], dtype=torch.int64).cuda(),
                torch.tensor(ref["bucket_feature_of_row"], dtype=torch.int32).cuda(),
                torch.tensor(call["offsets"][:: call["num_features"]]).cuda(),
                torch.tensor(call["row_shift"]).cuda(),
                dict(
                    num_nodes=call["num_nodes"],
                    local_size=local_size,
                    node=call["node"],
                    local_rank=call["me"],
                    num_features=call["num_features"],
                    lookup_keys=call["lookup_keys"],
                    remote_keys=call["remote_keys"],
                    num_proxy_rows=1 << 20,
                    max_chunk_rows=call["max_chunk_rows"],
                    payload_rows=call["payload_rows"],
                    record_bytes=8,
                ),
                call["groups"],
            )
            for name in (
                "push_order_off",
                "proxy_num_chunks",
                "proxy_ends",
                "proxy_push_tables",
                "remote_task_offsets",
            ):
                if out[name].tolist() != ref[name]:
                    checks["tables"].append(name)
            if out["src"].tolist() != ref["src"] or out["dst"].tolist() != ref["dst"]:
                checks["maps"].append("src/dst")
            # the positions the requester pushes, and only those (initcheck)
            written = sorted(ref["push_order"])
            got = out["push_order"][torch.tensor(written, dtype=torch.int64).cuda()]
            if got.tolist() != [ref["push_order"][p] for p in written]:
                checks["push order"].append("push order")
            if not call["skip"] and len(written) != ref["push_order_off"][-1]:
                checks["push order"].append("a call that fits pushes every row once")
            chunks = ref["proxy_num_chunks"]
            stats["calls"] += 1
            stats["rows"] += call["num_rows"]
            stats["chunks0"] += chunks.count(0)
            stats["chunks_full"] += chunks.count(call["max_chunks"])
            stats["odd_max_chunks"] += call["max_chunks"] % 2
            stats["skips"] += call["skip"]
            stats["high_ends"] += sum(
                1 for row in ref["proxy_ends"] if any(end >= 1 << 31 for end in row)
            )
            stats["most_chunks"] = max(stats["most_chunks"], call["max_chunks"])
            stats["most_proxies"] = max(stats["most_proxies"], local_size)
            stats["most_nodes"] = max(stats["most_nodes"], call["num_nodes"])
        torch.cuda.synchronize()  # an access out of range would raise here
    finally:
        hier_output_dist_requester_ctas(0)
    return checks, " ".join(f"{k} {v}" for k, v in stats.items())


REQUESTER_CASES = [
    RequesterCase("uniform"),
    RequesterCase("no rows", rows=(0,), bound=(0,)),
    RequesterCase("rows past the valid ones alone", rows=(0,), bound=(1, 40)),
    RequesterCase("skip", skip=1.0, rows=(2, 5), bound=(0, 3)),
    RequesterCase("no keys across nodes", chunks="zero"),
    RequesterCase("proxies of no chunks and of max_chunks", chunks="zero_or_full"),
    RequesterCase("every proxy at max_chunks", chunks="full"),
    RequesterCase("chunk ends past 2^31", high_ends=1.0, chunks="full"),
    # an odd max_chunks starts every push table at an odd word: the 16-byte
    # loads peel the words before a 16-byte boundary
    RequesterCase("odd max_chunks", odd_chunks=True, local_size=(1, 2, 3, 4, 8)),
    RequesterCase("lookup groups", groups=True, num_features=(5,), calls=3),
    # more proxies than a warp's lanes, more nodes than a warp's lanes
    RequesterCase("72 GPUs per node", num_nodes=(2,), local_size=(72,), calls=2),
    RequesterCase(
        "40 nodes of 2", num_nodes=(40,), local_size=(2,), num_features=(1, 2)
    ),
    # a table CTA's loads in more than one pass (8,192 words a pass), and the
    # scan in tiles of 2,048 entries
    RequesterCase(
        "max_chunks past 8,192",
        num_nodes=(2,),
        local_size=(2, 3),
        num_features=(2,),
        remote_keys=(11000,),
        max_chunk_rows_arg=8,
        payload_rows=2,
        chunks="full",
        calls=2,
    ),
    # the rows on 1 and 3 CTAs: a thread's rows in several steps of 4
    RequesterCase(
        "rows on one CTA",
        num_nodes=(2, 3),
        local_size=(4,),
        rows=(1000, 3000),
        bound=(0, 77),
        row_ctas=1,
        calls=2,
    ),
    RequesterCase("rows on 3 CTAs", rows=(20, 60), row_ctas=3, calls=2),
    # the design's reference ports, through the hook: ref_requester's calls
    # (local_size 1 to 8, 2 to 4 nodes, max_chunks 1 to 16, ends past 2^31)
    # and e2e_requester_check's (the proxies' occurrences of each requester
    # from none to 100, node skips)
    RequesterCase(
        "the design's requester calls",
        num_nodes=(2, 3, 4),
        local_size=(1, 2, 3, 4, 8),
        num_features=(1, 2, 3, 5),
        rows=(0, 1, 2, 9),
        bound=(0, 0, 3),
        remote_keys=(1, 5, 40),
        high_ends=0.3,
        groups=True,
        calls=120,
    ),
    RequesterCase(
        "the design's end-to-end calls",
        num_nodes=(2, 3, 4),
        local_size=(1, 2, 3, 4, 8),
        num_features=(1, 3),
        rows=(0, 2, 5),
        bound=(0, 3),
        others=(0, 0, 1, 3, 17, 40, 100),
        skip=0.1,
        calls=120,
    ),
]


# python test_hier_dedup.py sanitize: these cases, on their first seed, for
# compute-sanitizer (the module's docstring)
SANITIZE_CASES = {
    "uniform",
    "a lookup group per feature",
    "weights, 3 nodes of 2, 6 rows per payload, above a key's rows",
    "2 nodes, a wrapping ring",
    "node dedup off",
    "remote_keys overflow",
    "remote_keys 20 x the need",
    "remote_keys at the need",
    "no keys across nodes from node 0",
    "hot keys, another job",
    "a few thousand occurrences a proxy, 2 nodes of 2, 4 rows per payload",
    "3,072 occurrences of 2,688 keys at one proxy, remote_keys at them",
    "3,073 occurrences of 2,689 keys at one proxy, 4 rows per payload, "
    "remote_keys 6,143",
    "the keys' rows by int64 pairs, 2,688 occurrences at one proxy, "
    "node dedup off, 4 rows per payload, remote_keys at them",
    "2 nodes, a wrapping ring, the chunk layout on 3 CTAs",
    "no keys across nodes from node 0, the chunk layout on 7 CTAs",
    "the push order by int64 push keys",
    "requester 1 asks nothing across nodes, 20 x the need",
}
# python test_hier_dedup.py sanitize past: these cases, on their first seed,
# for memcheck, racecheck and synccheck alone (their proxies' chunks pass
# max_chunks, and the keys of the chunks past it read words of chunk_inv that
# no chunk wrote, which initcheck reports)
SANITIZE_PAST_CASES = {
    "chunks past max_chunks of 100, a wrapping ring",
    "chunks past max_chunks of 12, one CTA, a wrapping ring",
}
# and these requester cases, with the cases of SANITIZE_CASES
SANITIZE_REQUESTER_CASES = {
    "uniform",
    "no rows",
    "skip",
    "proxies of no chunks and of max_chunks",
    "chunk ends past 2^31",
    "odd max_chunks",
    "72 GPUs per node",
    "40 nodes of 2",
    "rows on one CTA",
}


def main() -> None:
    torch.cuda.set_device(0)
    # "sanitize" or "sanitize past": the cases of one set, for compute-sanitizer
    past = sys.argv[1:] == ["sanitize", "past"]
    sanitize = past or sys.argv[1:] == ["sanitize"]
    sanitize_cases = SANITIZE_PAST_CASES if past else SANITIZE_CASES
    sanitize_requester_cases = set() if past else SANITIZE_REQUESTER_CASES
    failures = 0
    for case_idx, case in enumerate(CASES):
        if sanitize and case.name not in sanitize_cases:
            continue
        for seed in range(1 if sanitize else case.seeds):
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
    for case_idx, case in enumerate(REQUESTER_CASES):
        if sanitize and case.name not in sanitize_requester_cases:
            continue
        checks, stats = run_requester_case(case, 2000 + 31 * case_idx)
        good = not any(checks.values())
        failures += not good
        print(
            f"requester '{case.name}' ({stats}): "
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
