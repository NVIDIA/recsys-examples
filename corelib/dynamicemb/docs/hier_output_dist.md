# Hierarchical input and output dist (NVLink inside a node, NCCL GIN between nodes)

Turn it on per table with `DynamicEmbTableOptions(output_dist="hierarchical")`
and the capacities of its buffers (`output_dist_capacity`), or with `"auto"`
(section 4).

**Contents**

1. [Summary](#1-summary)
2. [Words used here](#2-words-used-here)
3. [Why](#3-why)
4. [Using it](#4-using-it)
5. [Capacities](#5-capacities)
6. [When a sharding takes TorchRec's dists](#6-when-a-sharding-takes-torchrecs-dists)
7. [The operator](#7-the-operator)
8. [The path and rail alignment](#8-the-path-and-rail-alignment)
9. [Node dedup](#9-node-dedup)
10. [The input dist's steps](#10-the-input-dists-steps)
11. [The output dist's kernels](#11-the-output-dists-kernels)
12. [The ring and credits](#12-the-ring-and-credits)
13. [Memory order, and why nothing hangs](#13-memory-order-and-why-nothing-hangs)
14. [The CTA budget](#14-the-cta-budget)
15. [Memory](#15-memory)
16. [Pipelines](#16-pipelines)
17. [TorchRec versions](#17-torchrec-versions)
18. [Correctness and determinism](#18-correctness-and-determinism)
19. [Limitations](#19-limitations)
20. [Where the code is](#20-where-the-code-is)

---

## 1. Summary

(Section 2 explains the terms this document uses.)

A row-wise sharded `EmbeddingCollection` (sequence embeddings) must send each
key to the GPU that holds its table row (the *owner*). The owner looks the row
up and sends it back to the GPU that asked for it. The backward sends each
row's gradient back the same way. TorchRec does this in two steps: the
*input dist* moves the keys and the *output dist* moves the rows. For a
sharding of DynamicEmb tables the hierarchical dist replaces both. One
operator per sharding, `HierOutputDist`, does the work (section 7):

* **A static operator.** It is built once, at the sharding's first call, and
  makes its window then (the buffers its peers read and write), at the
  capacities the tables give (`output_dist_capacity`, section 5); its size
  never changes. A call's own tensors come from PyTorch's caching
  allocator. The ranks first check that they all build the op from the same
  options. A call that does not fit the capacities raises an error at its
  second wait, on every rank alike, which names the capacity to raise.
* **Only an op, on the caller's stream.** Its interface takes and gives
  tensors and plain values only: the constructor, `input_dist()`,
  `output_dist_fwd()` and `output_dist_bwd()` (section 7.1). Each queues
  its work on the current stream and returns. The caller orders the calls
  on its streams and records the call's tensors on the streams that use
  them, as TorchRec's training loop and train pipelines do (section 16).
* **The input dist does all the planning, with one host sync.** Keys move
  over NVLink inside a node and over InfiniBand between nodes, sent by the
  GPU itself (NCCL GIN). Eight custom kernels move them and wait for peers
  (the *comm kernels*); the steps between them are kernels of its own and
  CUB's sorts and scans (the *planner*). `input_dist()` queues all of it at
  once. At the end every owner has a ready KeyedJaggedTensor (KJT) of the
  keys it looks up, and every GPU has the per-row maps of the output dist.
  The CPU waits for the GPU once per call, at the call's second wait, in
  one blocking copy of a small status (`8 × (2 × num_features + 7)` bytes)
  to the host.
* **The lookup only looks up.** DynamicEmb's lookup writes its rows straight
  into a buffer that the output dist reads (zero copy).
* **The output dist only moves rows.** The forward uses one kernel. The
  backward uses one kernel. When the call has heavy rows, two short kernels
  add up their gradients first (section 11.4). The output dist computes no
  index.
* **Rail-aligned path.** A key that a GPU asks of an owner on another node
  first goes over NVLink to the *proxy*: the GPU of its own node on the
  owner's rail. The proxy sends it over that rail's InfiniBand link to the
  owner. No traffic crosses rails.
* **Node dedup.** The proxy sends each (table, key) of its node once. So
  the owner looks the key up once per node, its row crosses InfiniBand once
  per node, and the node adds up its gradients before they cross back.
* **No static limit on the nodes, GPUs per node or features.** The kernels
  keep no table sized by them, and a GIN warp's lanes take any number of
  rail peers in turn.
* **Memory that grows little with the number of GPUs.** Apart from
  `node_offsets` (a row of `world_size × num_features + 1` offsets per GPU
  of the node: 1 MB at 32 nodes, 8 GPUs per node and 64 features), the
  rail's count messages (about `16 × num_nodes² × num_features` bytes:
  1.0 MB at 32 nodes and 64 features) and small tables that grow linearly
  with the nodes and with the features (section 15.4), no buffer is sized
  by the number of GPUs. The proxy's buffer for
  rows (`proxy_rows`) is a ring with a fixed cap: a call's rows stream
  through it, chunk after chunk, and a chunk's rows are taken again once
  every reader is done with them. By default a chunk takes at most 2 MiB
  of the ring and moves about 512 KiB of rows over IB (section 12.1).

[benchmark/README.md](../benchmark/README.md), section 3, says how to compare
this dist with TorchRec's dists.

## 2. Words used here

* **Rank**: one GPU and the process that drives it; `world_size` ranks in
  all. **Node**: one machine; its `local_size` GPUs are joined by
  **NVLink**. There are `num_nodes` nodes, and ranks are numbered node by
  node: `rank = node × local_size + local_rank`.
* **InfiniBand (IB)**: the network between nodes. Each GPU has its own
  network card (**NIC**). A **rail** is the set of GPUs with the same
  `local_rank` on every node. On a rail-optimized network the NICs of one
  rail share one switch, so traffic that stays on its rail takes the
  shortest path. A GPU's **node peers** are the other GPUs of its node, and
  its **rail peers** the other GPUs of its rail.
* **NCCL**: NVIDIA's communication library. This dist uses its device API,
  which kernels call. With **LSA** (load/store access) a kernel reads and
  writes the memory of the other GPUs of its node over NVLink. With **GIN**
  (GPU-initiated networking) a kernel sends data over IB itself, without the
  CPU. GIN sends *puts* (copy bytes into a peer's memory) and *signals* (add
  a number to a counter at the peer).
* **Window**: the memory an op shares with its peers. Each rank allocates
  its copy with `ncclMemAlloc` and registers it with NCCL. Every rank's copy
  has the same size and layout, so a rank finds its data in a peer's copy at
  the same place. The window has three *sections*: the control words, the
  keys and the rows. Each is cut into named *buffers*, one per kind of data
  (section 7.5). Its size is a whole number of *allocation granules*, the
  unit in which the GPU driver maps memory.
* **Op**: the operator `HierOutputDist`, one per sharding (section 7). It
  holds its window and everything else that lives as long as the sharding,
  and runs every step of a call on the caller's current stream. Each step
  returns plain data, which the next step takes. **Plans**: what the input
  dist of a call gives the output dist, the forward's (`FwdPlan`) and the
  backward's (`BwdPlan`): the requester's maps and the call's tables.
* **Comm kernel**: one of the input dist's eight custom kernels, each of
  which moves data between GPUs or waits for peers (section 10.2). The
  first, `node_exchange`, also lays out the node's keys (section 10.3), and
  `rail_exchange` the rail's (section 10.4).
  **Planner**: the input dist's other steps, which lay a call out: kernels
  of its own, and the radix sorts and scans of CUB (NVIDIA's library of GPU
  algorithms), called from C++ (section 10).
* **Capacity**: how many keys a part of the op's buffers holds, fixed when
  the op is built: `lookup_keys` and `remote_keys` (section 5). **Need**:
  how many a call needs, the most of any rank.
* **KJT**: TorchRec's `KeyedJaggedTensor`, the keys of a batch per feature
  and sample. **VBE**: a KJT whose features have different batch sizes
  (TorchRec's variable batch size per feature). **Awaitable**: TorchRec's
  handle to work whose result comes later; its `wait()` returns the result.
* **Owner**: the rank that holds a key's row. **Requester**: a rank whose
  batch has the key (every rank is both). **Proxy**: for an owner on another
  node, the GPU of the requester's node on the owner's rail.
* **Occurrence**: one place where a key appears in a requester's batch.
  **Node-unique key**: one (table, key) as the proxy sends it on, however
  many occurrences its node has.
* **Lookup output**: the owner's rows for the keys it looks up, in its
  buffer `lookup_rows`. **Proxy rows** (`proxy_rows`): the proxy's buffer
  for the rows that pass between its node and the other nodes, in both
  directions: a **ring**, through which a call's rows pass as a **stream**
  of **chunks**; a stream that does not fit the ring takes it again, lap
  after lap (section 12.1). A chunk holds the rows of some keys of one
  owner node. In the ring it takes a row per occurrence of its keys: at
  most 2 MiB of rows (or `DYNAMICEMB_HIER_PAYLOAD_KB`, if that is more). Over
  IB it moves one row per key, its **payload**. When its keys have few
  occurrences, that is about `DYNAMICEMB_HIER_PAYLOAD_KB` (512 KiB by
  default) or more, up to about twice that (section 4.4).
* **Dedup** (deduplication): removing repeated keys. **Index dedup**:
  TorchRec's `use_index_dedup`. Each rank removes its own repeated keys
  before the input dist, and the rows are copied back to every original
  position after the output dist. The op takes index-deduplicated keys
  only (section 6). **Node dedup**: the proxy removes the repeated keys of
  its whole node (section 9).
* **Skewed keys**: keys drawn so that a few of them are very common (a *hot*
  key appears in many samples).
* **Fall back**: take TorchRec's dists instead, for a whole sharding
  (section 6), or for an eval forward.

## 3. Why

**TorchRec's dists.** The input dist sorts each rank's keys into one block
per owner (the bucketize). It sends the block sizes with an NCCL all-to-all,
reads them on the CPU (a host sync), then sends the lengths and keys with
more all-to-alls. The bucketize keeps lengths and offsets per (owner,
feature, sample): `16 × world_size × num_features × batch_size` bytes, which
grow with the number of GPUs. The output dist reorders the rows, sends them
with an all-to-all and reorders them again: about a dozen kernels in each
direction. Every occurrence of a key crosses IB. With skewed keys, say a hot
key makes up a share `hot_share` of every rank's keys, and each rank sends
`num_keys` keys. Then the hot key's owner receives about
`(1 + (world_size − 1) × hot_share) × num_keys` rows. So its memory and its
NIC's load grow with the number of GPUs.

## 4. Using it

### 4.1 Turning it on

```python
from dynamicemb import DynamicEmbTableOptions
from dynamicemb.shard import DynamicEmbeddingCollectionSharder

DynamicEmbTableOptions(
    ...,
    output_dist="hierarchical",       # "torchrec" (default), "hierarchical" or "auto"
    output_dist_capacity=capacities,  # an int, or a dict by name (section 5)
    node_dedup=True,                  # across nodes: one key per (table, key)
)
DynamicEmbeddingCollectionSharder(..., use_index_dedup=True)  # the op takes no other keys
```

It serves `ROW_WISE` sequence shardings (`EmbeddingCollection`) of
`"roundrobin"` and `"hash_roundrobin"` tables (`dist_type`), under index
dedup. Without `use_index_dedup=True` on the sharder, or with a
`"continuous"` table, `"hierarchical"` raises `ValueError` at the first
`input_dist()` and `"auto"` takes TorchRec's dists (section 6). The op
carries no score per key: with tables that take one (`score_strategy`
`PRECISE_CUSTOMIZED`), `"hierarchical"` raises `ValueError` when the model is
sharded and `"auto"` takes TorchRec's dists. Pooled
shardings (`EmbeddingBagCollection`) ignore `output_dist`. All DynamicEmb
tables of one sharding must agree on `output_dist`, `output_dist_capacity`
and `node_dedup`, whatever their `output_dist`; otherwise building the
sharding raises `ValueError`. Tables without DynamicEmb options (plain
TorchRec tables) do not vote. Every rank must give the same options and
environment variables: the ranks compare them before they build the op, and
raise the same `ValueError` when they differ (section 7.2). Training calls
take the op; eval forwards take TorchRec's dists, while the op's window
stays allocated. It runs in a plain training loop and under TorchRec's train
pipelines; the loop orders its steps on its streams (sections 4.2 and 16).
Section 5.2 shows how to choose the capacities. Call `quiesce(model)` on
every rank before you drop a model (section 4.6).

### 4.2 Requirements

* DynamicEmb built against **NCCL 2.30.7 or newer**: `setup.py` builds
  `src/hier_output_dist/` only then, and else prints a notice and builds
  DynamicEmb without it. It reads the version from
  `$NCCL_HOME/include/nccl.h`, else from the first `nccl.h` in
  `$CUDA_HOME/include`, `/usr/local/include` or `/usr/include`. At run time
  the process must load an NCCL at least that new (`libnccl.so.2` is one
  file name for every version: with `NCCL_HOME`, put its `lib` directory on
  `LD_LIBRARY_PATH`).
* NCCL's device API on the GPUs. Across nodes: GIN on the rail connection,
  with at least 4 contexts, all of one backend (proxy, GDAKI or GPI;
  section 11.1).
* Ranks numbered node by node (the code assumes it; torchrun and Slurm's
  default task layout do this), and forming whole nodes.
* Fewer than 2^31 − 1024 (rank, feature) pairs: `world_size × num_features`
  (the planner numbers them with int32).
* Rows (embedding dim × bytes of the output dtype) a multiple of 16 bytes,
  in fp32, bf16 or fp16.
* Index dedup: the sharder's `use_index_dedup=True`. Each rank then sends
  each (table, key) of a call once, so a key has at most `local_size`
  occurrences at its proxy (section 9).
* Tables of `dist_type` `"roundrobin"` (the default) or `"hash_roundrobin"`.
  The tables of a sharding can have different rules. The op refuses
  `"continuous"` (sections 6 and 10.1).
* Keys of one dtype, `int32` or `int64`, on all GPUs of a node in a call
  (section 9). TorchRec's dists need one key dtype on every rank too.
* Capacities (`output_dist_capacity`) that fit the calls (section 5).
* Every rank makes the same calls in the same order, as with NCCL
  collectives (operations that all ranks run together).
* The caller orders the op's steps on its streams, the same way on every
  rank; the op orders nothing across streams itself. The caller runs each
  step of a call on the stream of the step before it, or on a stream it
  ordered after that one. It runs the input dists of all calls on one
  stream, and the output dists' kernels (forward and backward) on one
  stream. It runs each lookup after the lookup's backward has read the last
  `lookup_grad` (section 11.2). And it records the call's tensors on the
  other streams that use them. TorchRec's training loop and train pipelines
  do all of this (section 16).
* Local batch sizes may differ between ranks, and VBE works. A local batch
  of no samples raises `ValueError` (TorchRec's dists do not take one
  either).

### 4.3 Table options

| option | default | meaning |
|---|---|---|
| `output_dist` | `"torchrec"` | `"torchrec"`: TorchRec's dists. `"hierarchical"`: the hierarchical dist. It needs `output_dist_capacity` (without it, making the options raises `ValueError`), index dedup and tables of `dist_type` `"roundrobin"` or `"hash_roundrobin"` (without them, the first `input_dist()` raises `ValueError`), and a build without NCCL ≥ 2.30.7 raises `ImportError` when the model is sharded. It carries no score per key: with tables that take one (`PRECISE_CUSTOMIZED`), it raises `ValueError` when the model is sharded. `"auto"`: `"hierarchical"` where the build has it, `output_dist_capacity` is set, the keys are index-deduplicated and no table is `"continuous"` or takes a score per key, else TorchRec's dists, with a warning. Any other value raises `ValueError`. |
| `output_dist_capacity` | `None` | The capacities of the op's buffers, in keys (section 5). An int sets `lookup_keys` and `remote_keys` to that many. A dict sets them by name: `lookup_keys`, and across nodes also `remote_keys` (one node uses `lookup_keys` alone). Each capacity is a positive int below 2^31 − 1024; other values or names raise `ValueError` when the options are made. A call that needs more of one raises `RuntimeError` at its second wait, on every rank (section 5.4). TorchRec's dists ignore it. |
| `node_dedup` | `True` | Across nodes, the proxy sends one key per (table, key), and the node adds up its gradient rows before they cross back. `False` sends each GPU's occurrence on its own (more IB traffic) and keeps TorchRec's gradient rows (section 18). Gradient clipping turns it off by itself. |

### 4.4 Environment variables

None is needed. All of them are read when the model is sharded (when the
sharding is built). A value the op refuses (a timeout of 0 or less, a
negative CTA count, a `DYNAMICEMB_HIER_REDUCE_CTAS` below 1, or a
`DYNAMICEMB_HIER_PAYLOAD_KB` below 64 or not a multiple of 16) fails on every
rank alike (section 6).

| variable | default | meaning |
|---|---|---|
| `DYNAMICEMB_HIER_PROXY_MB` | `512` | Cap of `proxy_rows`, in MiB (section 15.3). |
| `DYNAMICEMB_HIER_TIMEOUT_S` | the process group's NCCL timeout | How long a kernel waits for a peer before it prints what it waited for and traps (stops, which loses the CUDA context). |
| `DYNAMICEMB_HIER_OUT_CTAS` | `0` | CTAs of the output-dist kernels. `0`: 32 for the forward and 64 for the backward on one node, 32 for both across nodes (section 14). |
| `DYNAMICEMB_HIER_REDUCE_CTAS` | `8` | Across nodes: the backward's CTAs that add up the proxy's rows (1 or more; the kernel uses at most its CTAs minus one). |
| `DYNAMICEMB_HIER_PAYLOAD_KB` | `512` | Across nodes: the payload of a chunk of keys of few occurrences, the rows it moves over IB, in KiB: about this value or more (below). At least 64, and a multiple of 16. |
| `LOCAL_WORLD_SIZE` | TorchRec's `get_local_size` | GPUs per node (`local_size`); torchrun sets it. |

Across nodes the dist moves rows over IB in chunks (section 12.1). A chunk
takes a row of the ring per occurrence of its keys: at most 2 MiB of rows
(or `DYNAMICEMB_HIER_PAYLOAD_KB`, if that is more). Over IB it moves one row
per key, its payload. When the proxy cuts its chunks, it counts each key
as its rows, but at least `rows_per_payload` (its counted rows, section
12.1). Section 15.3 gives the exact rule; what follows holds for rows that
are small beside a chunk. When the ring holds two chunks of 2 MiB and the
value is at most 2048, `rows_per_payload` is about 2048 /
`DYNAMICEMB_HIER_PAYLOAD_KB`, rounded down. So a chunk of keys of at most
`rows_per_payload` occurrences each moves about 2 MiB / `rows_per_payload`
of rows over IB: about the value when it divides 2048, else up to about
twice the value. A chunk of keys of more occurrences moves less. A value
from 1040 to 2032 gives `rows_per_payload` 1: the chunks of 2048. From 2048
on, when half the ring holds that much, a chunk of keys of one occurrence
each moves about the value.

**The trace** is a build option: `DYNAMICEMB_HIER_TRACE=1` in the
environment when DynamicEmb is built. `setup.py` then compiles the dist with
`-DDEMB_HIER_TRACE=1`, and says so in its output. Such a build keeps:

* a timeline of every output-dist kernel, in a ring of 64 per direction
  (the op's `trace()`; `trace_launches()` counts the kernels). It costs two
  memsets per kernel and counters in shared memory, and across nodes the
  backward gets a warp that watches the rows land (section 11.4);
* `hier_output_dist_clock`, which reads the GPU's clock for the benchmark.

Without it the trace compiles away: the out kernels have no timeline code,
`trace()` is empty, `trace_launches()` is `[0, 0]`, and
`hier_output_dist_clock` raises.
`dynamicemb_extensions.hier_output_dist_traced()` tells which build it is. A
traced build's times do not compare with an untraced build's.

### 4.5 NCCL settings

* **Set `NCCL_CROSS_NIC=0` for multi-node jobs**, before the process makes
  its first NCCL communicator. With it, NCCL connects GIN on the rails only,
  so each GPU keeps GIN queues to its `num_nodes − 1` rail peers. With the
  default (2), NCCL connects GIN to every rank, and GIN's queues then take
  memory that grows with the number of GPUs; the op warns about it when it
  is built. The setting applies to the whole process: the job's other NCCL
  communicators (TorchRec's, PyTorch DDP's) also keep each ring and tree on
  one NIC index, which suits a rail-optimized network. IB use looked
  steadier with it, but it has not been compared with the default on the
  same nodes, for this dist or for the other communicators.
* Where the NCCL network plugin cannot run GIN (on Eos with its default
  container), set `NCCL_NET_PLUGIN=none NCCL_GIN_PLUGIN=none`. Under Slurm,
  export them before `srun`: `--export=ALL,VAR=x` does not override a
  variable that is already exported.

### 4.6 Teardown and `quiesce()`

```python
from dynamicemb.shard import quiesce

quiesce(model)  # on every rank, after the model's last call, before dropping it
del model
```

Teardown is local: ranks may drop a model at different times. After a rank's
kernels end, its node peers may still read its window, and its rail peers may
still send it credits (section 12.2). So dropping a sharding first waits for
the rank's kernels, then until every node peer and rail peer has dropped its
own (the teardown handshake, `teardown_kernel`), for at most
`min(timeout, 30 s)`. If a peer keeps its model, or the teardown meets a CUDA
error (a kernel that trapped, say), the rank prints a message and keeps its
window until the process exits. `quiesce(model)` calls each op's `quiesce()`,
in the same order on every rank: the op syncs its device and meets the other
ranks over its own communicator. Dropping the model then lets go of each op's
window at once. A call of the model after `quiesce` undoes it. The views of
`lookup_rows` share the window's memory with the op: a view that outlives the
sharding (a lookup output kept by the caller) keeps that memory until the view
is dropped.

## 5. Capacities

The op makes its window once, when it is built (section 7.2), at the
capacities of `output_dist_capacity`, and keeps its size for its life.
Each capacity counts keys. On one node only `lookup_keys` is used.

### 5.1 What each capacity bounds

The words used in this section and in section 15:

* `num_keys`: the number of keys a rank sends in one call (after index
  dedup);
* `uniq_share`: the number of node-unique keys per occurrence at a proxy.
  It is about 0.935 with the benchmark's uniform keys and 0.27 with its
  power-law keys, and 1 when no two GPUs of a node share keys.

| capacity | what it bounds | what it sizes | with uniform keys, about |
|---|---|---|---|
| `lookup_keys` | the keys an owner looks up: those of its own node and, across nodes, one per node-unique key of each other node | `lookup_rows` (the lookup output), the own part of `node_keys` and, across nodes, `rail_keys`; per call, the owner's KJT | `num_keys × (1 + (num_nodes − 1) × uniq_share) / num_nodes`, at most `num_keys` |
| `remote_keys` | across nodes: the occurrences a proxy deduplicates, the keys its node sends the owners of its rail on the other nodes | the remote part of `node_keys`, `node_bucket_feature_of_row`, `remote_map` and `key_stage`; `proxy_rows`, up to its cap; per call, the proxy's lists | `num_keys × (num_nodes − 1) / num_nodes` |

Two more sizes follow from these (section 15.3): `proxy_rows` holds
`remote_keys` rows, up to its cap, and the proxy's tables hold the most
chunks that `remote_keys` rows make (`max_chunks`). A call whose rows do not
fit `proxy_rows` takes it again as its rows stream through (section 12.1).

However hot a key is, index dedup sends it at most once per rank and
table: it adds at most `local_size` keys to its owner's `lookup_keys` from
its own node, and at most `local_size` occurrences to `remote_keys` at its
proxy on every other node, which after node dedup add one key each to its
owner's `lookup_keys`. That holds whatever the number of nodes.

### 5.2 How to choose them

Count the needs on batches like the ones the model will see, then add
headroom. Two helpers in `benchmark/benchmark_utils.py` do this (it is not
part of the installed package: put `benchmark/` on `sys.path`, as the tests
do):

* `hier_needs(batches, tables, owner_of, local_world_size, index_dedup=True,
  node_dedup=True, group=None)` counts what the op needs of each capacity
  for some calls, as the op counts it: the most any rank needs in any call.
  It is collective: every rank of the sharding's process group calls it with
  its own batches. `tables` maps each feature of the sharding to its table,
  and `owner_of(keys, feature)` gives the rank that owns each key.
* `hier_capacities(needs, factor=1.25, min_extra=4096)` makes the dict for
  `output_dist_capacity`: each capacity `factor` times its need, and at
  least `min_extra` entries more (small calls).

```python
import sys

sys.path.insert(0, "corelib/dynamicemb/benchmark")
from benchmark_utils import hier_capacities, hier_needs
from dynamicemb.key_ownership import owning_shard

needs = hier_needs(
    sample_batches,                  # this rank's KJTs of a few calls
    {"item": 0, "user": 1},          # the table of each of the sharding's features
    # each key's owner, for int64 keys; for int32 keys give the bucketize's
    # 32-bit rule (section 10.1)
    lambda keys, feature: owning_shard(keys, world_size, "roundrobin"),
    local_world_size,
)
capacities = hier_capacities(needs, factor=1.25)
```

* **An int** sets both capacities alike. It is the simplest choice: it must
  hold the larger need. Across nodes it then gives the other capacity more
  room than it needs, which costs memory (section 5.3).
* **Skewed keys and changing batches.** The needs grow with the hot keys'
  share and with the longest batches. Count them on enough batches, or add
  more headroom. A call that does not fit raises, and its error gives what
  it needed (section 5.4).
* **The benchmark** counts the needs of its own batches before it builds
  the model (`--capacity measure`, the default) and takes
  `--capacity_factor` (1.25) times them, and at least 4096 entries more
  (`benchmark/README.md`, section 3). The tests size their capacities the
  same way.

### 5.3 What they cost

Bytes per entry of each capacity, per rank. `row_bytes` is the embedding dim
× the bytes of the output dtype; `record_bytes`, the size of a key record,
is 8, or 16 when the owner's KJT carries weights. Section 15.2 gives every
formula.

| capacity | in the window | elsewhere |
|---|---|---|
| `lookup_keys` | `row_bytes` (`lookup_rows`) + 8 (`node_keys`; 12 when the keys carry weights) + across nodes `record_bytes` (`rail_keys`) | per call, 8 (12 with weights) for the owner's KJT |
| `remote_keys` | 8 (`node_keys`; 12 with weights) + 4 (`node_bucket_feature_of_row`) + 12 (`remote_map`) + `record_bytes` (`key_stage`), and `row_bytes` in `proxy_rows`, up to its cap | per call, 8 (the proxy's rows before each key, `node_exp_off`) |

The input dist's working memory comes on top, in the caching allocator's
pool of the stream it runs on: that of its largest step (section 15.2).

For the benchmark's configuration (540,672 keys per rank, rows of 512
bytes), capacities of 1.25 times the needs take a window of 0.35 GB per rank
on one node and 0.53–0.62 GB on 2 to 4 nodes (section 15.4).

### 5.4 A call that does not fit

The call's status holds its needs: the most any rank needs of each
capacity, the same on every rank (section 10.5). When the call needs more of
some capacity than the op has:

* **Its second wait raises RuntimeError, on every rank alike.** The error
  names each capacity the call overflowed, how large it is, and what the
  call needed of it. For example: `output_dist="hierarchical": a call needs
  more than the capacities hold; raise output_dist_capacity's lookup_keys from
  541000 to at least 541801`. The call makes no lookup and no output dist.
* Its input dist still ran every step, and its comm kernels sent their
  messages, so the sequence numbers and running totals stay valid: the op
  takes the next call that fits. Nothing is written past a capacity. When a
  GPU would get more of its node's keys than a capacity holds (its own part
  above `lookup_keys`, or across nodes its remote part above
  `remote_keys`), no GPU of that node pushes its keys. And an owner whose
  keys pass `lookup_keys` gets no key records. Telling that a call does not
  fit costs nothing more: the needs come with the call's one host sync.
* When a node pushes no keys, its proxies deduplicate none, so the call's
  `lookup_keys` need leaves out the keys that node sends to other nodes. A
  later call may then name `lookup_keys` too.

A call whose keys are not index-deduplicated, so that a proxy gets one key
more than `local_size` times, also raises at its second wait, on every rank
alike: `output_dist="hierarchical": a proxy got one key 9 times from the 8
GPUs of its node; the keys must be index-deduplicated, each table's at most
once per GPU`. Its status carries that need too (`key_rows`, against
`local_size`). The proxy gives such a key `local_size` rows (section 9), so
the input dist stays in bounds, and the call makes no lookup and no output
dist.

## 6. When a sharding takes TorchRec's dists

A sharding that cannot take the hierarchical dist uses TorchRec's
`KJTAllToAll` and `RwSequenceEmbeddingDist`. The choice is the same on every
rank: it is made from the sharding alone, collectively, or per call from
state that every rank shares.

| condition | decided | result |
|---|---|---|
| `output_dist="torchrec"` (the default) | when sharding | TorchRec's dists |
| an unknown `output_dist`; an `output_dist_capacity` that is not a positive int below 2^31 − 1024, or a dict that sets `lookup_keys` (and maybe `remote_keys`) to such ints; `"hierarchical"` without `output_dist_capacity` | when the options are made | `ValueError` |
| tables of one sharding that disagree on `output_dist`, `output_dist_capacity` or `node_dedup` | when sharding | `ValueError` |
| a lookup group whose output dtype is not the sharding's | when sharding | `ValueError` |
| a build without NCCL ≥ 2.30.7 | when sharding | `"hierarchical"`: `ImportError`; `"auto"`: a warning, TorchRec's dists |
| `"auto"` without `output_dist_capacity` | when sharding | a warning, TorchRec's dists |
| one rank in the sharding; a table of the sharding that is not a DynamicEmb table; feature processors | when sharding | a warning, TorchRec's dists |
| tables that take a score per key (`score_strategy` `PRECISE_CUSTOMIZED`): a key crosses with one float32 weight, which a proxy adds up, and no score | when sharding | `"hierarchical"`: `ValueError`; `"auto"`: a warning, TorchRec's dists |
| ranks that would build the op from different options, capacities or environment values | first `input_dist()`, before the op is built | `ValueError` on every rank, naming the first rank and value that differ |
| ranks that are not whole nodes; `world_size × num_features` of 2^31 − 1024 or more; rows not a multiple of 16 bytes | first `input_dist()` | a warning, TorchRec's dists |
| keys that are not index-deduplicated (a sharder without `use_index_dedup=True`) | first `input_dist()`, before the op is built | `"hierarchical"`: `ValueError` on every rank; `"auto"`: a warning, TorchRec's dists |
| a DynamicEmb table of the sharding with `dist_type` `"continuous"` (below) | first `input_dist()`, before the op is built | `"hierarchical"`: `ValueError` on every rank; `"auto"`: a warning, TorchRec's dists |
| a configuration the op cannot take: across nodes, capacities without `remote_keys`; a proxy cap below the least rows of its ring (section 15.3); a `remote_keys` whose stream could pass position 2^32 (section 12.1); rows that are not fp32, bf16 or fp16; an environment value it refuses (section 4.4) | first `input_dist()`, before the op is built | `"hierarchical"`: `ValueError` on every rank; `"auto"`: a warning, TorchRec's dists |
| On some rank: the NCCL LSA team is not the whole node. Or, across nodes, GIN does not connect the rails, has fewer than 4 contexts, or its signals do not start at 0. Or the window cannot be allocated. | first `input_dist()`, collectively | a warning that names each rank and why, TorchRec's dists on every rank |
| an eval forward | per call | TorchRec's dists for that call |
| a call that does not fit the capacities, or a proxy that gets one key more than `local_size` times | per call, at its second wait | `RuntimeError` on every rank, naming the capacity to raise or the key's occurrences (section 5.4) |

The op places keys by `roundrobin` and `hash_roundrobin` only
(section 10.1). One op serves all tables of a sharding, so one
`continuous` table refuses the whole sharding. Under `"hierarchical"` the
error names the first such feature, in the sharding's feature order
(`feature_names()`):
`output_dist="hierarchical": it places keys by dist_type "roundrobin" or
"hash_roundrobin" only, and the sharding's feature 2 has "continuous" (set
its table's dist_type to one of those, or output_dist to "torchrec")`.
Under `"auto"` the warning starts with `output_dist="auto":`, has the same
words, and ends with `; using TorchRec's dists`.

## 7. The operator

### 7.1 Its interface

The op is a C++ class with a small interface, which takes and gives tensors
and plain values only. Python reaches it through pybind11, as
`dynamicemb_extensions.HierOutputDist`, beside four functions of the
module: `hier_output_dist_unique_id()` (the id of the op's communicator,
made on one rank), `hier_output_dist_unsupported()` and
`hier_output_dist_invalid()` (why the op does not apply to a sharding, or
refuses a configuration, or ""; they build nothing, section 7.2), and
`hier_output_dist_traced()`. Each call of the op queues its work on the
caller's current stream (section 7.3). What a call gives back is data that
the caller hands to the next call: nothing else keeps a call's state.

```cpp
namespace dyn_emb::hier_output_dist {

class HierOutputDist {  // one per sharding
 public:
  // Collective (section 7.2). A configuration it does not take throws
  // std::invalid_argument (Python: ValueError), on every rank alike.
  HierOutputDist(const std::string &unique_id, int64_t rank, int64_t world_size,
                 int64_t local_size, const std::vector<int64_t> &feature_tables,
                 const std::vector<int64_t> &dist_types, int64_t dim, at::ScalarType dtype,
                 bool weighted, bool node_dedup, int64_t lookup_keys, int64_t remote_keys,
                 int64_t proxy_bytes, double timeout_s, int64_t out_ctas, int64_t reduce_ctas,
                 int64_t payload_kb);
  ~HierOutputDist();  // local (section 4.6)

  // The whole input dist of a call, with no host sync: (values, weights,
  // lengths, offsets, lookup_rows, status, fwd_plan, bwd_plan).
  InDistTensors input_dist(const at::Tensor &values, const c10::optional<at::Tensor> &offsets,
                           const c10::optional<at::Tensor> &lengths, int64_t batch_size,
                           const c10::optional<at::Tensor> &weights,
                           const at::Tensor &input_offsets, const at::Tensor &reverse_indices);
  // The output dist: one kernel for the forward, one to three for the backward.
  at::Tensor output_dist_fwd(const std::vector<c10::optional<at::Tensor>> &fwd_plan,
                             const at::Tensor &lookup_out);
  at::Tensor output_dist_bwd(const std::vector<c10::optional<at::Tensor>> &bwd_plan,
                             const std::vector<int64_t> &counts, const at::Tensor &grad,
                             bool divide);
  void quiesce();  // collective (section 4.6)
  // diagnostics: memory(), trace(), trace_launches()
};

}  // namespace dyn_emb::hier_output_dist
```

The constructor takes the configuration (`Config`'s fields): the ranks, the
table and the bucketize's rule of each feature, the rows and their weights,
`node_dedup`, the capacities (with the proxy cap in bytes) and the knobs of
section 4.4. A rule is a `DistType` code (`src/utils.h`) on the host: 1
(`roundrobin`) or 2 (`hash_roundrobin`), one for all features of a table.
The constructor and `hier_output_dist_invalid()` refuse any other code, a
table whose features have different codes, and a list without a code per
feature (`Config::validate`). Their messages:

* `it places keys by dist_type "roundrobin" or "hash_roundrobin" only, and
  the sharding's feature 2 has "continuous" (set its table's dist_type to
  one of those, or output_dist to "torchrec")`; for an unknown code, `...
  has dist_type code 7`;
* `the sharding's features 0 and 1 share a table but not its dist_type (1
  and 2)`;
* `it takes a dist_type per feature, got 2 for 3 features`.

Python gives each table's `dist_type` to all its features, and
`DynamicEmbTableOptions` takes no `dist_type` but `"continuous"`,
`"roundrobin"` and `"hash_roundrobin"`. So from Python only the
`"continuous"` refusal can come (section 6).

`input_dist()` takes the requester's KJT, index-deduplicated
(its values; offsets, or lengths when it has none; its batch size; its
weights or none), the offsets of the input KJT before index dedup, which
`construct_jagged_tensors` reads, and index dedup's reverse indices. It
returns:

* the owner's KJT: its values and weights (or none) hold `lookup_keys`
  entries, the first `num_recv` its own; its lengths and offsets have an
  entry per (feature, column), `num_features × kjt_stride` of them;
* `lookup_rows`, `[lookup_keys, dim]`, a view of the window: the call's
  lookup writes its first `num_recv` rows;
* the call's status, `int64 [2 × num_features + 6]` on the device (section
  10.5): the needs, `num_recv`, the backward's count, the remote
  occurrences this rank deduplicated as proxy, and the totals per feature;
* the output dist's plans, as lists of their tensors (`None` for those the
  call has not).

One training step in Python, as the sharding runs it (section 7.4):

```python
values, weights, lengths, offsets, rows, status, fwd_plan, bwd_plan = op.input_dist(
    values, offsets, lengths, batch_size, weights,
    input_offsets, reverse_indices)  # input_dist()
words = status.cpu().tolist()  # the second wait, the call's one host sync
# a need above its capacity: raise RuntimeError (section 5.4)
num_recv, heavy_subs = words[3:5]
kjt = owner_kjt(values[:num_recv], lengths, offsets, words)  # the lookup's input
...                            # the lookup writes its rows into rows[:num_recv]
out = op.output_dist_fwd(fwd_plan, rows[:num_recv])
...                            # the dense forward and backward
lookup_grad = op.output_dist_bwd(bwd_plan, [num_recv, heavy_subs], grad_out, divide)
```

The forward's plan (`FwdPlan`) and the backward's (`BwdPlan`) hold only what
the input dist made: the requester's maps and the call's tables (section
11.1). Each table both kernels read is the same tensor in both plans. The
op gives each as the list of its tensors (`tensors()`), and makes it again
from such a list (`of()`); to Python the lists are opaque. The output dist
takes the lookup's output with the forward's plan, and the gradient with
the backward's plan and its counts, which only the status gives.

Of the diagnostics, `memory()` gives the bytes of the op's window, in two
maps: `memory()["windows"]["window"]` is this rank's copy, in whole
allocation granules, and `memory()["buffers"]` gives what each buffer takes
in it, by name (`ctrl` for the ctrl section; sections 7.5 and 15.2).

Python reads the options and the environment when the sharding is built,
and hands them over when it builds the op. The benchmark reads a call's
tables from its plans (`HierCall`, `shard/hier_output_dist.py`). The tests
run the planner's steps, and the node layout, the push and the rail layout
of the comm kernels, on plain tensors through functions of `testing.cu`. It also has a
hook that makes the next construction fail, a hook that makes a rank read
its node's offsets and keys late (section 18), and the GPU clock by which
the benchmark aligns its timelines.

### 7.2 Construction

The op is built at the sharding's first `input_dist()`, in any mode, at the
same point on every rank:

1. **The ranks agree on what they build it from.** Each rank asks
   `hier_output_dist_invalid()` whether the op takes its configuration. Then one
   all-gather over the sharding's process group gives every rank what every
   rank would build the op from (all but its rank), and a unique id for the
   op's own NCCL communicator, made on rank 0. If some rank's configuration
   differs from rank 0's, every rank raises the same `ValueError`, which
   names that rank and the first value that differs. Else every rank has
   the same configuration, so every rank decides alike. If the op does not
   apply (`hier_output_dist_unsupported()`, section 6), every rank warns and
   takes TorchRec's dists. If the op refuses the configuration (a
   `continuous` table, say), or the keys are not index-deduplicated,
   `"hierarchical"` raises `ValueError` on every rank and `"auto"` warns and
   takes TorchRec's dists.
2. Else every rank builds the op with that id. The constructor checks the
   configuration again (it throws before any collective if it refuses it),
   then makes the communicator and its device communicator: LSA inside the
   node and, across nodes, GIN on the rails with 4 contexts.
3. Each rank checks that it can run: its LSA team is the whole node, and
   across nodes GIN connects the rails with at least 4 contexts whose
   signals start at 0. Then it allocates its copy of the window (section
   7.5), once.
4. The ranks agree: an all-gather of one readiness code per rank, on the
   op's own communicator, before anything collective uses the window. If
   some rank cannot run, every rank ends the communicator together, frees
   what it allocated, and raises the same `RuntimeError`, which names each
   rank and why. Python warns and takes TorchRec's dists for the sharding.
5. Else every rank registers its window (one collective registration) and
   makes its two channels (section 7.5). An error inside the collective
   registration cannot be agreed: that rank raises alone, and the others
   wait in the registration, as they would in NCCL's own initialization.

Building the op costs a few host syncs: the all-gather, the communicator,
the window's zeroing and the agreement. After it, a call costs one.

### 7.3 One call

As in TorchRec, `input_dist()` returns an awaitable, whose wait (the *first
wait*) returns a second awaitable. Here `input_dist()` itself queues the
whole input dist of the call (`op.input_dist()`), so the first wait returns
the second awaitable at once. Waiting on that one (the *second wait*) is
the call's one host sync: it copies the status to the host, raises if the
call does not fit (section 5.4), and returns the owner's KJT. Every call of
the op queues its work on the current stream and returns:

```
the caller's stream
-------------------
input_dist() (op.input_dist()): the bucketize (the keys sorted by owner),
               then the steps below (comm kernels and planner kernels), the
               last of which writes the call's status
first wait:    nothing
second wait:   a blocking copy of the status to the host, the call's one
               host sync; then the owner's KJT
lookup:        writes its rows into lookup_rows (zero copy)
forward kernel (op.output_dist_fwd(fwd_plan, rows)): the rows
... dense forward and backward ...
backward kernel (op.output_dist_bwd(bwd_plan, counts, grad, divide)): gradient
               rows into the owners' lookup_rows, handed to the lookup's
               backward as lookup_grad
```

The op orders nothing across streams. The caller runs each step on the
stream of the step before it, or on a stream it ordered after that one. In
a plain loop every step runs on the default stream. Under TorchRec's train
pipelines the input dist (`input_dist()` and both waits) runs on the
pipeline's data_dist stream (section 16).

After the bucketize, `input_dist()` queues these steps, in this order, on
every rank. A *comm* step is a comm kernel (section 10.2); the others are
the planner's kernels, with CUB's sorts and scans.

| step | kind | what it does | section |
|---|---|---|---|
| `node_exchange` | comm | this rank's offsets into `node_offsets` at each GPU of its node, the node barrier, then the node layout from the node's offsets: where each rank of the node pushes its keys; the node's needs | 10.2, 10.3 |
| `node_key_push` | comm | the bucketize's gather: this rank's keys into their places in its targets' `node_keys` over NVLink; on one node also the requester's map | 10.2 |
| `node_done` | comm | tells its targets; waits until every requester of the node told it | 10.2 |
| owner's KJT (one node) | kernel | a copy of the own part of `node_keys` | 10.3 |
| node dedup | kernels, CUB | the proxy's node-unique keys, rows, the chunks of its stream and their slots | 9, 10.4 |
| push order | kernels, CUB | its requesters' push orders and push tables | 10.4 |
| `rail_exchange` | comm | its count message to every rail peer, and theirs; then the rail layout from the rail's messages: where each proxy's key records land, this owner's KJT layout, where each other node's keys of each feature start in it (`source_offsets`) and where its node's start in each rail owner's KJT (`owner_offsets`); the rail's needs | 10.2, 10.4 |
| `rail_key_push` | comm | its key records to its rail owners; it does not wait | 10.2 |
| `node_status_exchange` | comm | the needs and `requester_offsets` of the node; the call's needs | 10.2 |
| `peer_gather`, twice | comm (and a kernel) | the node's proxy headers, then the requester's gather list and its rows' `remote_map` words | 10.2, 10.4 |
| requester tables | kernels, CUB | its maps, push order, push tables and remote task offsets | 10.4 |
| `rail_done` | comm | waits until every rail proxy's key records are in; flushes this rank's key puts | 10.2, 13 |
| owner tables | kernels | its KJT's keys, and the chunks its forward put list sends | 10.4 |
| put lists | kernels, CUB | the owner's forward and the proxy's backward put lists | 10.4 |
| status | kernel | the call's status, which the second wait copies to the host | 10.5 |

One node runs the first three steps, the owner's KJT and the status; across
nodes every step runs but the owner's KJT. A call with no rows launches no
`node_key_push`.

The second wait queues nothing after its sync: the owner's KJT is a view of
the first `num_recv` of its values.

`input_dist()` checks the call's inputs first. The call takes the in
channel's next sequence number when it launches its first comm kernel,
`node_exchange`. Every rank runs the same calls in the same order, so a
call has the same number on every rank. A call that raises before
`node_exchange`, on an input check or a failed allocation, takes no
sequence number and launches no comm kernel. On one node the call
allocates the requester's map (`src`) before `node_exchange`. So nothing
allocates between `node_exchange` and `node_done`, where a raise would
leave the node peers waiting in `node_done`.
A call whose awaitables are dropped still ran its input dist
(TorchRec's prefetch pipeline runs the input dist of its first batch
twice): every rank does the same, so nothing goes out of step. The second
awaitable holds the call's inputs until its wait, since the input dist
reads them on its stream. The sharding context keeps what the output dist
takes until the output dist's forward takes the plans: the forward's plan
goes once its kernel is queued, and autograd keeps the backward's
(sections 7.4 and 11.1).

### 7.4 Around the operator: the Python sharding

* **When the sharding is built** (no collective): Python reads the options,
  the capacities and the environment, and sets each DynamicEmb table's
  `output_alloc` to the sharding. In each call on the op, the sharding then
  hands each lookup group the next rows of the call's rows of `lookup_rows`
  (section 11.2).
* **The input dist.** On a call that takes the op, `input_dist()` queues
  the whole input dist (`op.input_dist()`) and returns an awaitable, whose
  wait (the first wait) returns the second awaitable at once; waiting on it
  again returns the same. The second wait copies the status to the host
  (`HierDist.finish`), raises `RuntimeError` on every rank if the call
  does not fit (section 5.4), and returns the owner's KJT, built from views
  of the input dist's tensors with no copy. It also sets the input KJT's
  per-feature totals from the status, so TorchRec's
  `construct_jagged_tensors` needs no host sync. The sharding context
  carries what the output dist takes (`hier_result`, a `HierCall`: the
  plans and the backward's counts). Its `record_stream`, which TorchRec's
  pipelines call, also records the plans' tensors.
* **Index dedup.** The op takes index-deduplicated keys only: a sharding
  whose sharder has no `use_index_dedup=True` takes TorchRec's dists under
  `"auto"`, and raises `ValueError` under `"hierarchical"`, at its first input
  dist. For a call that takes the op, the output dist itself writes each
  deduplicated row to all its original positions, and sums their
  gradients. So the sharding skips TorchRec's own expansion for that call
  (it clears the context's reverse indices when the call starts). A call on
  TorchRec's dists (an eval forward) keeps it.
* **Owner rules.** The op takes `roundrobin` and `hash_roundrobin` tables
  only (`Config::validate`): a sharding with a `continuous` table takes
  TorchRec's dists under `"auto"`, and raises `ValueError` under
  `"hierarchical"`, at its first input dist (section 6).
* **The lookup.** For the owner's KJT of a call on the op, the sharding
  takes the call's rows, the first `num_recv` rows of the `lookup_rows`
  that `op.input_dist()` gave, and each DynamicEmb lookup group writes its
  rows straight into the next rows of them. The lookup returns those rows,
  which cover all groups, with no copy. Any other KJT (an eval forward, on
  TorchRec's dists) takes TorchRec's lookup path.
* **The output dist.** It takes the call's plans from the sharding context.
  An autograd function runs the forward kernel with the forward's plan
  (`op.output_dist_fwd()`), and keeps the backward's plan as saved tensors
  (the list, and its counts beside them), so they live as long as autograd
  keeps the graph. A second backward (`retain_graph`) works, and they go
  after the last backward. Its backward runs the backward kernel with them
  (`op.output_dist_bwd()`), which returns `lookup_grad`, the gradient of
  the lookup output, a view of `lookup_rows` valid until the next lookup. A
  call without a plan (an eval forward) takes TorchRec's
  `RwSequenceEmbeddingDist`.
* **Streams.** Every step runs on the current stream. The output dist's
  awaitable, like those of TorchRec's own dists, makes the stream that
  waits on it wait for the forward kernel (an event), and records the rows'
  use there. So a dense forward on another stream reads the rows after the
  kernel. The lookup records the owner's KJT, and the output dist's forward
  the plans, on the stream they run on, which may be one of their own
  (TorchRec's fused pipeline). Section 16 covers the pipelines.

### 7.5 The channels and the window

* The op owns an NCCL communicator, one window and, across nodes, 4 GIN
  contexts. A call's tensors are in what its calls return (section 7.1).
* The **in channel** is the input dist's steps (section 7.3), on the current
  stream. It keeps views of this rank's own buffers: tensors over its copy
  of the window, without an allocator. It also keeps a few constant tensors,
  made when the op is built: the table of each feature, the other nodes, and
  the words of the proxy headers that this requester reads. The **out
  channel** is the forward and backward kernels, cooperative kernels on the
  current stream. The caller runs them on one stream, in the same order on
  every rank (section 13). The in channel starts each call, and the out
  channel each kernel, with a node barrier tagged with its sequence number
  (`in_seq`, `out_seq`). The two channels share no buffer, control word,
  signal or GIN context: each has its own words in the ctrl section.
* **Rule:** the window holds only data in transit and control words.
  Anything a later step reads is first copied into the call's own tensors.

The window has three sections, in this order (the out kernels' sections
first, so the places of their buffers take the least to compute):

| section | size | buffers |
|---|---|---|
| rows | `lookup_keys` rows; across nodes also `proxy_rows` | `lookup_rows` (the lookup output, then its gradient). Across nodes also `proxy_rows`. |
| ctrl | its control words (section 15.2: 1.1 MB at 32 nodes, 8 GPUs per node and 64 features), most of them the count messages | one buffer, `ctrl`: the control words of both channels (sequence numbers, running totals, the credits sent, the count messages, each GIN warp's place with each rail peer) |
| keys | from `lookup_keys`, across nodes `remote_keys`, and from `local_size × world_size × num_features` (`node_offsets`) | `node_offsets`, `node_keys` and, with weights, `node_key_weights`. Across nodes also `node_bucket_feature_of_row`, `remote_map`, `proxy_header`, `key_stage` and `rail_keys`. |

`node_offsets` holds a row per GPU of the node, of `world_size ×
num_features + 1` int64 words: row `r` is the bucketize's
`offset_per_bucket_feature` of the node's requester `r` in the call, which
that requester stores there (`node_exchange`, section 10.2). `node_keys`
holds the keys a GPU's node pushes to it: its *own part*, which
it looks up as owner, and across nodes its *remote part*, which it
deduplicates as proxy (section 10.3). The buffers that only an op across
nodes has:

* `node_bucket_feature_of_row`: the bucket feature of each occurrence in the
  remote part, owner node × `num_features` + feature, which the push writes
  beside its key;
* `remote_map` (`RemoteMap`): three int32 arrays with an entry for each
  such occurrence: its key's row in `proxy_rows`, its own slot in
  `proxy_rows`, and its place in its requester's push order;
* `proxy_header` (`ProxyHeader`): the proxy's header for its node's
  requesters, in int32 words: the chunks of its stream and the stream's
  end, each chunk's end (section 12.1), and each requester's push tables;
* `key_stage`: per other owner node the slice one key put sends: that
  node's chunks (`OwnerChunk`), then its node-unique keys' records;
* `rail_keys`: the slices the rail proxies put into this owner, one per
  source node, in node order.

The window is made when the op is built, from `ncclMemAlloc` and
`ncclCommWindowRegister`. It has the same size on every rank, is rounded up
to whole allocation granules, and keeps its size for the op's life. Each
buffer, the ctrl section too, starts at a 4 KiB boundary (`kBufferAlign`).
Before the registration each rank zeroes its ctrl section (sequence numbers
start at 0); every other buffer is written before it is read. Each view of `lookup_rows` holds this rank's copy:
its memory goes once the op has let go of it and every view is gone (section
4.6).

**The layouts.** The host and the kernels lay the window out with the same
code (`window.h`). `WindowLayout` builds the three sections,
`RowsLayout`, `CtrlLayout` and `KeysLayout`, from the op's sizes (`Sizes`:
its ranks, features and capacities, and what follows from them) over a
base pointer, each from the end of the one before. Each has an accessor per
buffer or control word, which returns its address in the copy at that base
(`keys.node_keys()`, `rows.proxy_row(slot)`,
`ctrl.in_written()`), so no other code computes where a buffer
or a word is. The host lays out its copy to size the window and to view its
buffers as tensors. Each kernel of the op takes the window, the sizes and,
when it uses GIN, the device communicator, each as a parameter of its own;
a kernel that waits for peers also takes its sequence number. The kernel
lays out this rank's copy as the node's GPUs map it (`NodeWindow`). The
sizes, the window and the communicator stay in the op's C++ classes:
Python never sees them. A place has the same offset in every rank's copy:
`NodeWindow::peer(pointer, peer)` gives a node peer's copy of the place a
pointer names in this rank's copy, and a GIN put to a rail peer takes
pointers into this rank's copy for both of its ends (`RowPut`,
`RailMessage`).

## 8. The path and rail alignment

### 8.1 Roles

There are `world_size = num_nodes × local_size` ranks. In a call every rank
is:

* a **requester**: it sorts its keys by owner and, in the end, gets their
  rows back in its batch's order;
* an **owner**: it looks up the keys it owns;
* across nodes, a **proxy**: each GPU relays between the requesters of its
  node and the owners of its rail on the other nodes.

The picture shows rail 3. GPU 3 of the left node is its node's proxy for the
owners on rail 3, such as GPU 3 of the right node.

```
           a node (requesters)                       another node (owners)
  +-------------------------------------+      +--------------------------------+
  |  GPU 0   GPU 1   ...   GPU 7        |      |  GPU 0  ...  GPU 3  ...        |
  |     \      |            /           |      |               ^                |
  |      \     |  NVLink   /            |rail 3|               | lookup         |
  |       +--> GPU 3 <----+  (proxy) <==+======+==>  lookup_rows of GPU 3       |
  |  same-node owners: NVLink only      |  IB  |                                |
  +-------------------------------------+      +--------------------------------+
```

On one node there is no proxy, no GIN and no node dedup: the owner's KJT
holds every occurrence its node asked for.

### 8.2 Rail alignment

Every GIN put goes to a rail peer (NCCL's rail team,
`NCCL_GIN_CONNECTION_RAIL`). So no IB traffic crosses rails, and each NIC
carries only its own GPU's traffic. A remote owner's rows reach a node
through one NIC and spread over NVLink. NVLink moves about 7 times as much
per GPU as one NIC. The proxy hands rows on as they land, while its NIC is
still busy.

NCCL's all-to-all, which TorchRec's dists use, also sends between nodes rail
by rail: NCCL's PXN routing sends through the NIC of the destination's rail.
But every occurrence of a key crosses IB, and with skewed keys the NICs of
the hot owner's rail carry the most.

Each GIN context sends to a peer through its own queue pair (QP: a NIC's
send queue, which runs in order). So traffic of one kind never waits behind
another kind. Each signal belongs to one context and has one sender. It
starts at 0 (checked once, when the op is built) and is compared with
running totals, so it is never reset.

| context | carries | signals and counters |
|---|---|---|
| 0 | forward rows, owner → proxy | `fwd_rows[owner node]` at the proxy |
| 1 | backward rows, proxy → owner | `bwd_rows[proxy node]` at the owner; when the proxy's stream wraps its ring, its put counters (chunk `c` counts on counter `c % counters`, `counters` = `min(max_chunks, 64)`) |
| 2 | the input dist: count messages and key puts | `count[node]`, `keys[node]` |
| 3 | credits and teardown | `grant[node]`, `bwd_ready[owner node]`, `closing[node]` |

Why `NCCL_CROSS_NIC=0`: section 4.5.

## 9. Node dedup

A proxy holds, in the remote part of its `node_keys`, every occurrence that
its node's requesters ask of the owners of its rail on the other nodes, and
in `node_bucket_feature_of_row` the (owner node, feature) of each. Its planner
finds their identities with a hash table and numbers them with a scan
(section 10.4):

* **Identity: (table, key).** Equal keys of a table are one node-unique
  key, whichever requesters and features asked for them. Features of one
  table share keys. A hash table of the call finds each occurrence's
  identity.
* **Why (table, key) is enough.** The op places a key by its table's rule,
  `roundrobin` or `hash_roundrobin` (section 10.1). Python gives each
  table's rule to all its features, and the op refuses a table whose
  features have different rules (section 7.1). Both rules keep the key and
  compute the owner from the key alone. So equal keys of a table, of one
  dtype, have one owner, and so one owner node: the owner node adds nothing
  to the identity. (`continuous` rewrites a key to its offset in its
  owner's block, so equal stored keys of two owners would be different
  keys. The op refuses it, section 6.)
* **One key dtype.** Both rules work in the width of the key: `roundrobin`
  takes a negative `int32` key as `2^32 + key`, and `hash_roundrobin` takes
  the low 32 bits of the hash of an `int32` key. So a key can have one
  owner as `int32` and another as `int64`. The GPUs of a node must give the
  op keys of one dtype in a call (TorchRec's dists need one dtype on every
  rank). Else a proxy can merge an `int32` key and the same `int64` key of
  two owner nodes, and serve both from the smaller owner node, its
  representative's (below).
* **Representative.** A node-unique key's representative is its occurrence
  of the least (owner node, feature), and of those the first in the remote
  part. So the key's feature is the table's smallest feature under which the
  node asked for the key. The owner looks the key up under it.
* **Numbering.** The node-unique keys are numbered (`uniq`) per owner node,
  feature by feature (the representative's), and inside each (owner node,
  feature), under node dedup, with the requesters in turn: each requester's
  first representative there, requester after requester, then each one's
  second, and so on (a requester with no more is passed over), each
  requester's in its bucketized order. So a chunk of the stream mostly
  holds keys of all the node's requesters, and in the backward its pushes
  mostly come from all of them. Requester by requester, with uniform keys
  at 2 nodes of 8 GPUs, a chunk's pushes came from 3 requesters (the
  median), about half of them from one; in turn they come from all 8, the
  largest share 0.13 (the median, a CPU study). That order follows the
  requesters' index dedup, not the key's value. Without node dedup the
  keys go requester after requester (below).
* **Rows.** In `proxy_rows` a key takes one row per occurrence. Index
  dedup leaves each requester each (table, key) once, so a key has at most
  `local_size` occurrences, however hot it is (`max_key_rows`, section
  15.3). Its first occurrence holds the key's row: the owner's put fills it
  in the forward, and in the backward that occurrence's gradient row and
  then the key's sum go there. Each other occurrence holds a duplicate's
  slot for its gradient row. A key of more occurrences (keys that are not
  index-deduplicated) takes `local_size` rows, all its occurrences take the
  key's row, and the call raises at its second wait (section 5.4).
* **Weights** ride along when the keys carry them: index dedup's frequency
  counters, where a table of the sharding has LFU scores (which count how
  often each key is seen) or an admission strategy. The proxy adds them up
  per node-unique key, as whole numbers.
* **`node_dedup=False`** (or gradient clipping): every occurrence is a
  node-unique key of its own. Inside each (owner node, feature) the keys
  go requester after requester, each requester's in its bucketized order.
  So the rows of a (table, key) that several requesters ask reach the
  owner's lookup in requester order every run. With the requesters in
  turn their order would follow the requesters' index dedup, which can
  change between runs, and so could the bits of the lookup's fp32 sum of
  them. A chunk's pushes come from a few requesters, as above.

**The owner's KJT** has `kjt_stride = local_size + num_nodes − 1` columns
per feature (its stride): its lengths go by (feature, column), at index
feature × `kjt_stride` + column. Entry (feature, column) with `column <
local_size` holds the keys of that feature that same-node requester
`column` sent, in its order. Entry (feature, `local_size + nth`) holds the
node-unique keys of the `nth` other node whose representative feature is
that feature, in `uniq` order.
The other nodes are counted from 0, in node order, with this node left out.
The KJT comes with its lengths, offsets and per-feature totals. So the
TorchRec code that splits it per lookup group needs no host sync.

**Example** (2 nodes × 2 GPUs; features `f0`, `f1` of one table; owner =
key mod 4). GPUs 0 and 1 are node 0; GPUs 2 and 3 are node 1. Node 1 asks
owner GPU 0 (node 0, rail 0) for two keys. GPU 2 asks for key 4 under `f1`,
and GPU 3 under `f0`. GPUs 2 and 3 both ask for key 8 under `f1`. That is
four occurrences and two node-unique keys. The proxy is GPU 2 (node 1,
rail 0). It puts both keys to GPU 0 at once. GPU 0 looks up key 4 under
`f0` and key 8 under `f1`. Two rows cross IB where TorchRec's dists move
four. In the backward GPU 3 pushes its gradient rows into GPU 2's
`proxy_rows`, and GPU 2 puts the two sums into GPU 0's `lookup_rows`.

## 10. The input dist's steps

`input_dist()` queues the bucketize and the steps after it on the caller's
stream, in the order of section 7.3. Eight of them are the comm kernels (section 10.2). The planner
does the rest (`src/hier_output_dist/input_dist/planner.h`): kernels of its
own and CUB's radix sorts and scans, on the current stream. Each step is one
or a few ordinary kernels. A kernel over items (keys, rows, occurrences)
runs a thread per item; a kernel over a step's small tables (sized by the
nodes, ranks, features and chunks) runs on one CTA, which loops over them.
No kernel waits for another CTA: a step that needs every item done goes on
in its next kernel, or in the kernel's last CTA out. That is the fan-out
table's counts (section 10.1): each CTA takes a ticket once its items are
done, and the CTA that takes the last ticket goes on. It goes on because
every other CTA has already arrived, not because it waited for one. The
tensors' sizes are what the host knows from the call's keys, the
capacities and the topology. The counts that only the
device knows (valid keys, node-unique keys, chunks) stay on the device, and
no tensor's size depends on them; a kernel reads such a count to bound its
own loop. So the host reads no device value before the second wait, and no
step syncs. Entries past the valid ones hold values that no one reads, and
an index taken from device data is clamped first: a call that does not fit
runs every step on what arrived and stays in bounds.

With the benchmark's configuration (8 features, one owner rule), the
bucketize queues 29 kernels and copies per call (DynamicEmb's index dedup
and the fan-out table included: 38 when the fan-out took 10 launches after
its sort, where it now takes one), and the steps after it 5 on one node and
about 50 across nodes, CUB's included. The 50 is an estimate from the code,
not a measurement: about 88 were measured at 2 nodes while the proxy's node
dedup took 44 launches (three radix sorts), and it now takes 8. The node's
part of a call is three comm kernels (`node_exchange`, `node_key_push`,
`node_done`); on one node a fourth kernel copies the own part. Across
nodes the rail's part is three more (`rail_exchange`, `rail_key_push`,
`rail_done`).

### 10.1 The bucketize (requester, caller's stream, local)

The requester's own steps, local: DynamicEmb's index dedup, then the
bucketize (`bucketize_rows`, `bucketize.cu`), named after TorchRec's
`bucketize_kjt_before_all2all`, whose key order it gives:

1. **Index dedup**: DynamicEmb's `segmented_unique`, sized on the
   GPU. The deduplicated KJT keeps its full length, and its lengths say how
   many keys are valid. So `num_uniques.item()` (a host sync) is not needed.
   `segmented_unique` takes `int64` keys, so `int32` keys are deduplicated
   as `int64`, and the deduplicated keys go back to `int32`: the bucketize
   places each key by the rules of its own width (see below). TorchRec's
   path in DynamicEmb (an eval forward) does the same.
2. **The keys by owner.** Each key's feature comes from the KJT's offsets, and
   its owner from its feature's rule (`dist_type`: `roundrobin` or
   `hash_roundrobin`), in the arithmetic of the keys' own width (see
   below). The kernel reads the rule of each key's feature at run time, so
   the tables of a sharding can have different rules. A warp's keys are
   consecutive, so all but a few warps have the keys of one feature and
   take one branch. Then one stable radix sort by bucket feature (an
   int32, owner × `num_features` + feature) puts the keys' rows in
   `[owner][feature][sample][position]` order: `inv` holds each one's row
   in the KJT; the keys past the valid ones (index dedup) go last. Both
   rules keep the key, so the keys stay in the KJT: bucketized row `row` is
   the KJT's key at `inv[row]`, widened to `int64` with its sign, and its
   weight is the KJT's at `inv[row]`. The
   bucketize does not gather the keys into that order. The push does, as it
   stores them (`node_key_push`, section 10.2). Searches of the sorted
   bucket features give where each bucket feature's and each owner's keys start
   (`offset_per_bucket_feature`, `offset_per_bucket`). It also gives the
   input KJT's per-feature totals. Its outputs take 8 bytes per key, with
   or without weights, and nothing of size `world_size × samples`.
3. **The fan-out table**: for each deduplicated row, its
   original rows in increasing order, from a stable radix sort of the
   reverse indices. A row with more than 64 original rows is a *heavy row*:
   it gets a heavy slot and consecutive sub-tasks of 64 original rows. Two
   kernels make the table, one before the sort (`original_rows_kernel`)
   and one after it (`fanout_runs_kernel`):
   * The kernel before the sort writes each original row's sort key and
     index, and where the bucketize put each row.
   * The kernel after the sort runs a thread per sorted position. The
     first position of each row's run writes where the row's original rows
     start, and so do the rows without original rows around it, with
     grid-wide loops for those before the first run and after the last
     (index dedup leaves many rows past the valid ones).
   * A run of more than 64 positions is a heavy row, and its warp takes
     it. The warp finds where the run ends, 32 probes at a time: one round
     for a run of up to 96 positions, seven for a million. The run's first
     thread then reserves the row's heavy slot and its sub-tasks with one
     64-bit atomic add: the heavy slot is the old word's high half, the first
     sub-task its low half. So consecutive heavy slots have consecutive
     ranges of sub-tasks. The warp writes the sub-tasks, 32 at a time.
   * The kernel's last CTA out writes the counts (heavy rows, sub-tasks)
     and sets the counter and the ticket back to 0. The in channel keeps
     them from call to call.

   The heavy slots go in the order the heavy rows reserve them, which can
   differ between two calls on the same keys. No sum depends on that
   order: a heavy row's
   sub-tasks stay consecutive and in original-row order, so the heavy
   rows' sums and the gradient keep their bits (section 18).

**The owner rules by key width.** On TorchRec's path DynamicEmb bucketizes
the KJT as the caller gave it, and the bucketize reads each key as an
unsigned number of the keys' width (a negative `int32` key as
`2^32 + key`). `bucketize_rows` computes the same way, in the keys' own
type. So on both paths a key goes to the same owner:

| `dist_type` | `int64` keys | `int32` keys |
|---|---|---|
| `roundrobin` | key mod `world_size` | the same, in 32 bits |
| `hash_roundrobin` | the key's 64-bit hash mod `world_size` | the low 32 bits of the hash of the key (zero-extended), mod `world_size` |

Both rules keep the key. The owner gets an `int32` key widened with its
sign to `int64`: the 64-bit key DynamicEmb's lookup makes of an `int32` key
on TorchRec's path. So a model may take either path call by call (eval
forwards, calls that do not fit) and finds its rows. Checkpoint loading and
delta replay (`dynamicemb/key_ownership.py`) follow another rule for some
`int32` keys (section 19).

The op does not take `continuous`, TorchRec's block rule. That rule
rewrites a key to its offset in its owner's block, so equal stored keys of
two owners would be different keys, and node dedup's identity (table, key)
would merge them (section 9). With a `continuous` table, `"hierarchical"`
raises `ValueError` and `"auto"` takes TorchRec's dists (section 6).

### 10.2 The comm kernels

They are in `input_dist/comm_kernels.cu`. A comm kernel that waits runs on
one CTA (or one warp) and takes the call's sequence number `in_seq`, the
same on every rank. It waits only for the same kernel (the same `in_seq`) of
its node or rail peers. The one exception is `rail_done`: it waits for its
rail peers' `rail_key_push` of the call, a kernel that does not wait. Its
threads take the peers in turn (thread `t` the peers `t`, `t + 32`, ... in a
warp), so it serves any number of them. Section 13 gives their memory order
and why no wait hangs.

* **`node_exchange`** (1 CTA): stores this rank's offsets, its bucketize's
  `offset_per_bucket_feature` (`world_size × num_features + 1` int64
  words), into its row of `node_offsets` at every GPU of its node, its own
  too, over NVLink. Then thread 0 makes a system fence and sets its
  `in_started` word to `in_seq`, and the CTA waits until every node peer
  has (the node barrier of the call). Every peer's offsets are then in this
  rank's `node_offsets`. From them the CTA computes the node layout
  (section 10.3) into the call's tensors. Passing the barrier also means
  that every peer finished its previous call's in channel, whose steps read
  what this call writes into their copies of the window. The offsets go out
  before the barrier: section 13 says why that is safe.
* **`node_key_push`** (threads stride over the bucketized rows, at most two
  CTAs per SM): the bucketize's gather. For each of this rank's valid rows
  it reads the KJT's key at `inv[row]` (an `int32` key widened with its
  sign) and its weight, and stores them into the target's `node_keys` (and
  `node_key_weights`) over NVLink, at the place the node layout gives.
  Across nodes a key for an owner on another node also writes its bucket
  feature (owner node × `num_features` + feature) into the target's
  `node_bucket_feature_of_row`. Consecutive keys
  of a bucket feature go to consecutive places, so the stores coalesce. On
  one node it also writes each row's place in its owner's KJT into `src`,
  the requester's map (a row past the valid ones gets its own index). Under
  skip it stores no key. It does not wait: `node_exchange` passed the node
  barrier before it. Each CTA ends with a system fence.
* **`node_done`** (1 warp): makes a system fence, tells each node peer that
  this rank's keys are there (its flag in the peer's `in_written` words),
  then waits until every peer told it. The tell also says that this rank's
  `node_exchange` read `node_offsets`, so the peers may store their rows
  there again in the next call. It runs in every call, also with no rows
  and under skip.
* **`rail_exchange`** (1 CTA, across nodes): writes this rank's count
  message (section 10.4), the same for every rail peer, into its
  `count_msg_out` words and into its `count_msg` slot of its own node: the
  slots go by `in_seq` parity and source node. Then the lanes of one warp,
  which take the rail peers in turn, put each peer the message into the
  same slot in the peer's copy: one put, which carries a GIN signal
  (context 2). The lanes of another warp wait until the message of every
  rail peer has landed. From the `num_nodes` messages of the call, read in
  their slots, the CTA lays out the rail (section 10.4) into the call's
  tensors: every rank of the rail computes it from the same messages.
  Before it ends, the putting warp flushes its puts.
* **`rail_key_push`** (1 warp, across nodes): its lanes, which take the
  rail owners in turn, put this proxy's `key_stage` slice for each owner
  into the owner's `rail_keys`, where the rail layout says, with a GIN
  signal. When it puts no keys (the owner's keys would pass `lookup_keys`,
  or the slice is empty), it sends the signal alone. A proxy puts only once
  it has the owner's count message of the call, which the owner sends after
  its previous call read `rail_keys`. It does not wait.
* **`node_status_exchange`** (1 CTA, across nodes): writes this rank's rail
  needs and where each requester of its node starts in its KJT (the rail
  layout's `needs` and `requester_offsets`) into its `needs` and
  `requester_offsets` words, sets its `in_status` word to
  `in_seq`, and waits until every node peer has. Then it reads every peer's
  needs, and where each peer's KJT holds this requester's keys.
* **`peer_gather`** (threads stride over the words, at most two CTAs per SM;
  across nodes, twice per call): reads words of node peers' proxy headers
  (`ProxyHeader`: one word per item, by its word number) or `remote_map`
  (`RemoteMap`: per item, an occurrence's entry in each of the three
  arrays). The peers wrote them before the `in_status` that
  `node_status_exchange` waited for.
* **`rail_done`** (1 warp, across nodes): its lanes, which take the rail
  proxies in turn, wait until the key signal of every rail proxy of the
  call has landed. Then the warp flushes this rank's puts on context 2, so
  the NIC has read its `key_stage` slices. It runs just before the owner's
  tables, the first step that reads `rail_keys`. So the key puts overlap
  `node_status_exchange`, the `peer_gather`s and the requester's tables
  (section 13).

### 10.3 The node layout, and the owner's KJT on one node

`node_exchange` lays out the node's keys on its CTA, from the node's
offsets (`node_layout_of`, `input_dist/node.cuh`). Every rank of the node
reads the same offsets, so all compute the same needs and skip. Row `r` of
`node_offsets`, `O[r]`, is requester `r`'s `offset_per_bucket_feature`. So
requester `r` has:

* `c(r, t, f) = O[r][b + 1] − O[r][b]` keys of feature `f` for owner `t` of
  this node, whose bucket feature is
  `b = (node × local_size + t) × num_features + f`;
* `R(r, t)` keys for the owners `(n, t)` of the other nodes, which go to the
  remote part of target `t`: the sum over the other nodes `n` of its block,
  `O[r][(n × local_size + t + 1) × num_features] − O[r][(n × local_size + t) × num_features]`.

The layout, with `me` this rank's local rank:

* the *own part* of an owner's `node_keys`, at `[0, lookup_keys)`, holds the
  keys of its node for it in its KJT's order: feature after feature, then
  requester after requester, each in the requester's order. This rank's
  lengths are `own_lengths[f × local_size + r] = c(r, me, f)`, and
  `own_offsets` gives where each starts. Owner `t`'s own part holds
  `own_total[t]` keys, the sum of `c(r, t, f)` over `r` and `f`;
* across nodes the *remote part* of a proxy's `node_keys`, at
  `[lookup_keys, lookup_keys + remote_keys)`, holds the keys of its node for
  the owners of its rail on the other nodes. They are requester after
  requester, then owner node, then feature, then position. In this rank's
  remote part, requester `r`'s keys start at `remote_offsets[r]`, the sum
  of `R(r', me)` over `r' < r`;
* the node's needs: its largest own part (the most `own_total[t]`) and its
  largest remote part (the most, over `t`, of the sum of `R(r, t)` over
  `r`). When one is above its capacity, no rank of the node pushes (`skip`,
  section 5.4).

Bucketized row `row` of bucket feature `b` goes to place
`row + row_shift[b]` in its target's `node_keys`. `row_shift[b]` is where
this rank's rows of `b` start there, less `O[me][b]`, where they start among
its bucketized rows. So one add places a row, wherever it is inside `b`:

* for an owner `t` of this node, this rank's rows of `b` start after the
  node's keys for `t` of the features before `f` (the sum of `c(r, t, f')`
  over every `r` and `f' < f`) and those of the requesters before this rank
  in `f` (the sum of `c(r, t, f)` over `r < me`);
* for an owner `(n, t)` of another node, they start at `lookup_keys`, after
  the keys of the requesters before this rank in `t`'s remote part (the sum
  of `R(r, t)` over `r < me`) and this rank's blocks for the owners
  `(n', t)` of the other nodes before `n`. That block holds every feature of
  the owner in the bucketize's order, so `row_shift[b]` is this start less
  `O[me][(n × local_size + t) × num_features]` for every feature of the
  owner;
* the rows past the valid ones (bucket feature `world_size × num_features`)
  are never pushed: their `row_shift` is 0, so each is at its own index.

On one node the own part is the owner's KJT. Its lengths are `own_lengths`,
and `node_exchange` also gives its offsets (`kjt_offsets`, `num_features ×
local_size + 1` of them, whose first ones are `own_offsets`) and its keys of
each feature (`length_per_key[f]`, the sum of `c(r, me, f)` over `r`). After
`node_done` the owner copies the first `min(own_total[me], lookup_keys)`
keys of its own part into the call's KJT values, of `lookup_keys` entries
(`one_node_owner`; the second wait narrows them to the keys that arrived).
With weights it also copies their weights into the KJT's weights. Under
skip it copies none. The copy is needed: the next call's peers
rewrite `node_keys` after its barrier, which does not wait for this call's
lookup. A requester's map to the owners' rows (`src`) is where the push put
each key.

### 10.4 Across nodes: the proxy, the rail, the requester and the owner

**The proxy** numbers its node-unique keys (section 9), cuts them into the
chunks of its stream by their counted rows and places each chunk in its ring
(section 12.1), and gives each occurrence its slot. It writes the three
buffers of its window that its node's requesters and its rail's owners
read:

* `remote_map` (`RemoteMap`): for each occurrence, its entry in three int32
  arrays: its key's row, its slot, and its place in its requester's push
  order;
* `key_stage`: per other owner node a slice: that node's chunks
  (`OwnerChunk`, 16 bytes each: its keys' records among the node's, and its
  start and end in the stream), then its node-unique keys' records in `uniq`
  order. A record is the key (`int64`) and, with weights, the weight's float
  bits and 4 zero bytes;
* `proxy_header` (`ProxyHeader`, int32 words): the chunks of its stream at
  word 0 and the stream's end at word 1; from word 4 each chunk's end; then
  its push tables, per requester `max_chunks + 1` positions: where its rows
  of each chunk start (the last: where they end), relative to that
  requester's push order.

**Its node dedup** takes eight launches and no sort (`proxy_keys`,
`input_dist/proxy.cu`); with `node_dedup=False` it takes three, without the
hash table, the pass and the scan:

* one kernel clears a hash table of the call, from the caching allocator:
  a power of two of at least `2 × remote_keys` buckets, each the int32
  head of a chain, and an int32 chain link per occurrence. Its threads also
  find each (requester, bucket feature) block of the remote part, by a
  binary search: a requester's occurrences go by owner node, then feature;
* a thread per occurrence finds its identity (table, key) with index
  dedup's chained insert (`chain_insert`, `src/dedup_chain.cuh`, which
  `segmented_unique` runs too). A chain links the occurrences that claimed
  an identity, one per identity. The occurrence walks its bucket's chain
  and ends on its identity's occurrence, or appends itself at the end with
  an `atomicCAS` and claims a new identity. On the identity it then lowers
  the representative with an `atomicMin` of (bucket feature, occurrence),
  counts the occurrence, puts it in a list of the identity's occurrences,
  and adds its weight as an `int64`;
* a pass marks the representatives, and a CUB scan (three launches) counts
  them before each occurrence;
* one CTA counts the representatives of each block: the keys of each
  bucket feature, and where they start;
* a thread per occurrence takes its key `uniq` from its representative,
  and its rank among its key's occurrences from the list. The
  representative writes its key's tables. The representative is the `k`-th
  representative (from 0) of its block (`r`, `b`), and `reps(r', b)` is
  the representatives of block (`r'`, `b`). With node dedup its key is
  where the keys of `b` start, plus the sum over `r'` of
  `min(k, reps(r', b))`, plus the requesters `r' < r` with
  `reps(r', b) > k`: before it come every representative of `b` of place
  below `k`, of every requester, and those of place `k` of the requesters
  before `r`. So the keys of `b` go by (`k`, `r`), the requesters in turn
  (section 9), one key per representative. That takes about four loads per
  requester and occurrence, the same words for every occurrence of `b`.
  Without node dedup the keys of `b` go requester after requester: the key
  is where they start, plus the occurrences of `b` at the requesters
  before `r`, plus `k`.

The bucket is the high bits of `murmur3_fmix64(key ^ seed)`, mixed with
the table. An owner rule fixes the low bits of the key (`roundrobin`) or of
`murmur3_fmix64(key)` (`hash_roundrobin`) for all keys of one owner, so
neither gives the bucket. On a proxy's keys under either rule, a CPU study
of up to 4 million occurrences finds what random buckets give: chains of a
few occurrences, and fewer than one comparison per occurrence on average.

**Names.** A table that holds data from another step is named after that
data where it came from, with a prefix for where it is now. The requesters
push their keys and weights in bucketized order, which the push gathers
from the KJT's keys and weights, and their bucket features
(`bucket_feature_of_row`) into `node_keys`, `node_key_weights` and
`node_bucket_feature_of_row`. Node
dedup gives `node_unique_keys`, `node_num_uniques`, `node_reverse_indices`
and `node_exp_off`, as index dedup gives `unique_keys`, `num_uniques`,
`reverse_indices` and `exp_off`, and its grouping by owner node gives
`node_offset_per_bucket(_feature)` and `node_length_per_bucket(_feature)`,
as the bucketize gives `offset_per_bucket(_feature)`. An owner keeps
the proxies' counts as `node_length_per_source(_feature)`, and a requester
the proxies' tables as `proxy_num_chunks`, `proxy_ends` and
`proxy_push_tables`.

A requester's push order at the proxy comes from one stable radix sort of
the push key, the occurrence's part: requester × (`max_chunks` + 1) + its
key's chunk. So a requester pushes its rows chunk after chunk, in each chunk
in its own order. Its push tables give, per chunk, its first position; a
pass over the sorted push keys writes where each part starts, so the tables
take no search. A key's rows are its key row, then its duplicates' slots
in occurrence order; `node_exp_off` holds the rows before each key, and
where the rows lie follows from the chunk (section 12.1).

**The rail.** The rail all-gathers its counts, as the node gathers its
offsets. Each rank sends every rail peer the same count message
(`CountMessage`, int64 words): its node's largest remote part; as owner,
the keys of its own node that it looks up, in total and per feature; as
proxy, the most occurrences of one of its keys, its chunks per owner node,
and its node-unique keys per (owner node, feature). That is
`3 + num_nodes + (num_nodes + 1) × num_features` words. From the
`num_nodes` messages of the call, its own too, every rank of the rail
computes the layout of the whole rail (`rail_layout_of`,
`input_dist/rail.cuh`):

* as proxy, where its slice lands in each owner's `rail_keys` (a slice per
  source node, in node order), and where its node's keys of each feature
  start in each owner's KJT (`owner_offsets`);
* as owner, its KJT lengths and offsets, and where each other node's keys of
  each feature start in its KJT (`source_offsets`);
* the rail's needs: every owner's keys are in the messages.

One function (`source_offset`) gives both offsets from the same messages.
So a proxy's `owner_offsets` row for an owner is that owner's
`source_offsets` row for the proxy's node, bit for bit, and no owner sends
its offsets back: `rail_key_push` puts only the keys.
`node_status_exchange` then gives every rank its node's rail needs. Their
maximum is the call's needs, the same on every rank: a rank's rail covers
every node, and its node holds one rank of every rail.

**The all-gather's scale.** A count message has about
`num_nodes × num_features` words. A rank sends it to its `num_nodes − 1`
rail peers and gets as many. So per call a rank sends about
`num_nodes² × num_features` words, and gets as many: 13 KiB at 8 nodes and
26 features, 201 KiB at 16 nodes and 100 features, 865 KiB at 64 nodes and
26 features (about the size of `node_offsets` at 8 GPUs per node), and
about 500 MiB at 256 nodes and 1,000 features. The ctrl section keeps the
messages in a slot per `in_seq` parity and node: about
`16 × num_nodes² × num_features` bytes (section 15.2). And `rail_exchange`
lays out the rail on one CTA, which reads the `num_nodes² × num_features`
counts a few times. All three grow with the square of the nodes (section
19).

**The requester.** Two `peer_gather`s read what the node's proxies wrote. One
reads the words of their headers it needs: the chunks, the stream's end, each
chunk's end and its own push table, by word lists made when the op is built. The other reads, for
each of its keys for an owner on another node, its occurrence's entry in the
three `remote_map` arrays at its proxy. From them, and from where its node's
owners hold its keys (`node_status_exchange`), it makes its maps. `src` is
where the forward reads each key's row: a row of its owner's `lookup_rows`
when the owner is on its node, else the key's row in its proxy's `proxy_rows`.
`dst` is where the backward writes the row's gradient: the same row, or its
own slot in `proxy_rows`. It also makes its rows in push order at each proxy,
its push tables with absolute positions, and the remote task offsets of each
(chunk, proxy), which the forward's pulls and the backward's pushes share.

**The owner.** Its KJT has `kjt_stride = local_size + num_nodes − 1`
columns per feature (section 9). Its keys come in slices of 2^22 positions:
its own node's columns from the own part of `node_keys`, the other nodes'
from their records in `rail_keys`. Its forward put list sends each source
node the rows of the chunks of this owner's node in that node's stream,
which come first in its slice (`fwd_chunks`).

**The put lists.** One step (`put_lists`) makes both, in one tensor: the
owner's forward list, which cuts the chunks of each source node's stream at
the features of its KJT they fill (`source_offsets`), and the proxy's
backward list, which cuts its own chunks at where its keys of each feature
start in each owner's KJT (`owner_offsets`). A node's node-unique keys are
cut at its chunks and where its keys of each feature end; each part between
two cuts is one segment, with its rows in `proxy_rows` and in the owner's
KJT, and its chunk (the proxy's, or the owner's `fwd_chunks` entry). A
list's segments are sorted by (node, key): node after node, so a node's
chunks come in stream order and the segments of a chunk follow each other;
a table gives the first segment of each node. The cuts are the chunks and
the ends of the (node, feature) keys, so the lists hold at most
`max_chunks + max_owner_chunks + 2 × num_nodes × num_features` segments. The
proxy also gets its reducers' tasks of each chunk.

### 10.5 The status and the one host sync

The last step writes the call's status, `8 × (2 × num_features + 6)` bytes.
The call's tables need no copy: each is a tensor of its own, which the plans
keep for the out kernels (section 11.1). The status holds the needs: the
most any rank needs of `lookup_keys` and `remote_keys`, and the most
occurrences of one key at a proxy (`key_rows`, against `local_size`). It
also holds the keys this owner looks up, the sub-tasks of its heavy rows
(the backward's count), the remote occurrences this rank
deduplicated as proxy (for the benchmark), and the per-feature totals of
the owner's and of the input's KJT. `input_dist()` returns it on the
device; the second wait copies it to the host on the current stream: a
blocking copy, which waits for every step before it, and the call's only
host sync. The host then compares the needs with the capacities (section
5.4), and builds the owner's KJT from views, with no copy.
TorchRec's code around the dist keeps its own host syncs where they apply:
`split()` when a call spans several shardings or has input features that no
table serves, and VBE. So does DynamicEmb's lookup, the same on every dist.

## 11. The output dist's kernels

The output dist runs one cooperative kernel per direction on the caller's
stream: the forward kernel `fwd_kernel` and the backward kernel
`bwd_kernel`. When a call has heavy rows, the backward first runs
`bwd_heavy_partial_kernel` and `bwd_heavy_sum_kernel`. These ordinary kernels
add up the gradients of the heavy rows (section 11.4).

### 11.1 What both kernels share

* **Entry.** CTA 0 publishes `out_seq` in its `out_started` word (after a
  system fence) and every CTA waits until all node peers have published
  theirs. A peer past this point has finished its lookup (which wrote its
  `lookup_rows`) and every earlier out kernel. Before it publishes, the
  backward's CTA 0 zeroes the words it counts from 0 (across nodes, each
  chunk's `ready` and `pushed_rows`), so no peer counts on one before it is
  zeroed.
* **Tasks.** Warps take 32-row tasks from two queues (`task` words):
  * The *local* queue holds the tasks of the rows of the node's owners.
  * Across nodes, the *remote* queue holds the tasks of the rows that cross
    nodes, chunk after chunk of the proxies' streams.

  A local task does not wait. A remote task can wait for its chunk at its
  proxy. While local tasks are left, a warp takes the next remote task only
  if that task does not wait. Otherwise, the warp takes a local task (ready
  first). Each queue keeps its order. Owners (and proxies) take turns,
  starting from each rank's own, so the ranks of a node start on different
  peers.
* **Exit.** Every CTA counts its arrival in a running total (its `arrive`
  word, so nothing is reset); the last CTA ends the rank's part of the
  kernel.
* Across nodes, the last warps of CTA 0 take the GIN roles and skip the
  entry barrier. A GIN warp's lane `l` serves the rail peers on nodes `l`,
  `l + 32`, ... It polls them in turn and waits for no single one, so no
  order in which peers move can stall it; what it keeps of a peer between
  polls is in its words for that peer in the ctrl section (`put_at`,
  `grant_sent`, ...). Every kernel ends by sending the credits for the next
  out kernel (GIN context 3) and flushing them. So no credit is in flight
  once a kernel has ended.
* **Arguments.** Each out kernel takes the op's window, its GIN device
  communicator and its sizes, its `out_seq` and the `arrive` word before
  it, and its call (`FwdArgs`, `BwdArgs`): the call's plan and tensors.
* **One build for each GIN backend.** Each out kernel is built once for each
  GIN backend (proxy, GDAKI and GPI), and once for one node, where it uses
  no GIN. The communicator checks that all its GIN contexts have the same
  backend, and the out channel launches the kernels of that backend. NCCL
  compiles each GIN call for all its backends and selects one at run time.
  With the code of all the backends, the out kernels needed more registers
  than they have (128 for each thread, at 512 threads for each CTA), so the
  compiler moved values to local memory (spills). With the code of one
  backend, and with 4 vectors in flight per lane in the pulls of the forward
  across nodes (section 11.3), there are no spills: `ptxas -v` shows 0 bytes
  of spill stores and loads for each out kernel, for sm_75 to sm_120
  (CUDA 13.2).
* **QP sharing and polls.** Only the issuer warp of CTA 0 uses the QPs of
  its rows context, and one lane posts on each QP (lane `l` on those of the
  rail peers on nodes `l`, `l + 32`, ...). So the issuers take the sharing
  mode that fits (`issuer_sharing`). On GDAKI it is THREAD: a QP has one
  posting thread, so a put takes its slots and rings the doorbell with
  plain loads and stores, without atomics, a lock or CTA fences. GPI has
  one channel per context for all peers, on which every lane of a warp's
  flush posts: CTA there, as DeepEP does for a QP that one SM uses. The
  proxy backend does not use the mode. In the default GPU mode, each put
  takes its slots in the QP with GPU-scope atomics and a GPU-scope acquire
  fence. An acquire at GPU or system scope drops the L1 cache of the SM
  (`CCTL.IVALL`), and the other warps of CTA 0 then miss in L1. For the
  same reason, the relay and the backward's issuer poll with relaxed loads
  (`Credits::read_relaxed`, `ld_relaxed_sys`). The warp makes one system
  fence per pass when a poll found news, for each 32 nodes (section 13).
  The forward's issuer reads its grants relaxed, with no fence after them.
* **Tables.** The kernels read the call's tables from global memory,
  through typed pointers in the call: the forward's plan as it reads it
  (`FwdPlanView`, in `FwdArgs`) and the backward's (`BwdPlanView`, in
  `BwdArgs`). Each table is a tensor of a plan. Both read the requester's
  `offset_per_bucket` and maps, and across nodes its push order and its tables of
  the node's proxies (`proxy_num_chunks`, `proxy_ends`, `proxy_push_tables`,
  `remote_task_offsets`), the proxy's chunks (`num_chunks`, `chunks`, `node_offset_per_bucket`)
  and the put lists (`put_list`, `put_offsets`). The forward also reads the
  requester's `src` and `push_order_off`, the owner's `fwd_chunks` and the
  proxy's `chunk_offset_per_bucket`, `chunk_inv` and `node_length_per_bucket`. The backward
  reads its `dst`, the owner's `node_length_per_source`, and the proxy's
  `reduce_task_offsets` and `node_exp_off`.
  Their strides are `max_chunks` and `num_nodes`. The call also holds what the op gives
  the kernel for it: the forward's output, the backward's gradient and its
  scratch (the heavy rows' sums and counts), across nodes the counts the
  kernel zeroes (the forward's `pulled_from_proxy`, the backward's `ready`),
  and the kernel's row of the trace. The forward's plan goes once its kernel
  is queued; autograd keeps the backward's (section 7.4).
* **The window.** A kernel lays out this rank's copy of the window from the
  sizes (section 7.5), and names each buffer and control word by its
  accessor: in this rank's copy, or in a node peer's through
  `NodeWindow::peer`. The one per-peer array in shared memory is the
  backward's `free_seen`, sized at launch by `local_size`.

### 11.2 The lookup

The lookup takes the call's rows, the first `num_recv` rows of the
`lookup_rows` that `op.input_dist()` gave: a view, valid until the next out
kernel. Each
DynamicEmb lookup group writes its rows straight into the next rows of the
call's rows: the tables' `output_alloc`. The lookup returns the call's rows,
which cover all groups, with no copy into a new tensor; the forward checks
that it gets them. The lookup writes where the last backward left
`lookup_grad` (section 11.4). So the caller runs it after the lookup's
backward that reads that gradient: on the same stream, or on one it ordered
after it. TorchRec's loops do: autograd runs a backward on the stream of its
forward, so the lookup and its backward share a stream.

### 11.3 The forward kernel

* **Owner** (across nodes): one warp of CTA 0, whose lanes take the rail
  proxies in turn, each over its own QP. First one system fence. Then, chunk
  by chunk as each proxy grants the rows of its stream they take, it puts
  straight from `lookup_rows` into that proxy's `proxy_rows`, one put per
  segment of its forward put list (a part of one chunk's payload). In a
  pass it reads each proxy's grant once, relaxed, and puts every segment
  that the grant covers. Each put adds its row count to the proxy's
  `fwd_rows` signal. Then a flush.
* **Proxy relay** (across nodes): the last warp of CTA 0, whose lanes take
  the owner nodes in turn. In a pass it polls with relaxed loads, then
  makes one system fence if a poll found news. It counts the rows of each
  owner node that have landed (`landed_rows`), and gives each chunk all of
  whose rows have landed this `out_seq` in its `landed` word, where its
  node's requesters read it. When its stream wraps its ring, it marks the
  chunks done with in order, once every requester of the node has pulled
  all their rows of them (their `pulled_rows` words count the pulls): lane
  `i` looks at the `i`-th chunk from the first not done with. Then it
  grants each owner node, once for the pass (once for each 32 chunks it
  grants), the rows that this frees: up to the end of the node's last
  chunk that ends within them (sections 12.1 and 12.2).
* **Requester** (all other warps): pulls each of its rows from the
  same-node owner's `lookup_rows`, or, once its chunk has landed, from its
  proxy's `proxy_rows`, and writes it to every original row of its
  deduplicated row in its output. A warp copies the rows of a task in
  16-byte vectors through registers. Each lane has 8 vectors in flight on
  one node, and 4 across nodes: across nodes, the task loop also needs the
  registers of its multi-node paths, and with 8 vectors it had spills.
  NVLink moves the rows as fast with 4 vectors in flight as with 8 (32 CTAs
  of 16 warps).
* **Task order.**
  * Local queue: the row tasks of the node's owners. Index dedup's
    sub-tasks of the heavy rows of these owners are spread among them.
  * Remote queue (across nodes): first the tasks of the proxies' streams,
    chunk after chunk. In a chunk, the tasks of each proxy come in turn,
    each 32 rows of the push order there. Then come the sub-tasks of the
    heavy rows that come from other nodes. Such a sub-task reads the row in
    its proxy's `proxy_rows` when its chunk has landed. At a proxy whose
    stream wraps its ring, other rows can take that place while the kernel
    runs. So there, the sub-task copies from the first original row of the
    row. The task of the row wrote it, and the sub-task waits until all rows
    of this rank at that proxy are in.

  A warp takes the next remote task when its chunk has landed. Otherwise,
  it takes a local task. The late sub-tasks wait for the local tasks. Thus
  the pulls empty the rings as the rows land, and the local rows fill the
  time between. (When the local tasks came first, the rows of a wrapping
  ring stayed in the ring, and the owners waited for their grants.) No task
  waits for a task that comes later in its queue.
* **Exit.** The last CTA tells the node that this rank has pulled
  everything, and waits until every node peer has too. After that nobody
  reads this rank's `lookup_rows` or `proxy_rows`. Across nodes it then
  sends the credits for the next kernel: `bwd_ready` to each rail proxy and
  a grant of the ring's rows (the stream's positions below `num_proxy_rows`)
  to each rail owner.

On one node, the forward kernel and the backward kernel are built without
the code for several nodes (section 11.1).

### 11.4 The backward kernel

* **Heavy rows** (`bwd_heavy_partial_kernel`, then `bwd_heavy_sum_kernel`).
  These two kernels run before the backward kernel, and only when the call
  has heavy rows. They wait for no peer and can use all SMs.
  1. `bwd_heavy_partial_kernel` adds the gradients of each sub-task's 64
     original rows in fp32, in order, into the sub-task's own fp32 row of
     `partial`. A group of lanes does a sub-task: one lane per 16-byte vector
     of a row, at most 32. Thus a warp does 2 sub-tasks of 256-byte rows at a
     time.
  2. `bwd_heavy_sum_kernel` adds the partial rows of each heavy row in
     sub-task order. It writes the sum into the first partial row of the
     row. A CTA does one row. A thread does one value of the row and keeps
     16 partial rows in flight, so a very hot row does not hold one warp
     for a long time.

  The kernel boundary orders the two steps, so no counter is needed.
  `partial` holds one fp32 row per sub-task, as before. The split adds no
  memory, and each sum keeps its order.
* **Requester**: it first adds up each deduplicated row's original rows
  in fp32, in a fixed order (increasing original row). For a heavy row, it
  reads the sum that `bwd_heavy_sum_kernel` wrote. Under TorchRec's
  gradient division it then multiplies by `1 / world_size` in fp32 and
  rounds once to the row dtype, as TorchRec's `div_(world_size)` does. It
  pushes each row, in push order, into a same-node owner's `lookup_rows`,
  or into its proxy's `proxy_rows` at the row's slot. Each 32-row task
  then adds its rows to its chunk's `pushed_rows` word at the proxy (a
  release add). A push into a chunk past the ring's first lap waits until
  the proxy has freed the chunk's rows.
* **Proxy reducers** (the last CTAs, 8 by default). A reducer task (32
  keys of a chunk) waits until the chunk's `pushed_rows` reaches its rows
  (`end − start`): the chunk's pushes fill each of its rows once, so every
  row is then in. They add up each key of several rows in fp32, in a fixed
  order: its key row, then its duplicates' slots, its occurrences by
  requester and position. They round the sum once into the key's row, and
  count the keys of the chunk as ready. A key of one row is not touched:
  the NIC reads it where the requester's NVLink stores left it.
* **Proxy issuer**: the last warp of CTA 0, whose lanes take the owner
  nodes in turn. Once an owner's `bwd_ready` is in, it puts the chunks of
  that owner node into the owner's `lookup_rows`, each once the reducers
  counted all its keys ready. In a pass each lane polls the look-ahead of its
  owner node's put list, with relaxed loads: at most 8 chunks
  (`kIssueLookahead`) from the first chunk not yet put. Then the warp makes
  one system fence if a lane found a ready chunk, and each lane puts the
  ready chunks of its look-ahead in list order, all the segments of each. So
  an owner node's chunks go out of order only within the look-ahead. Each put
  adds to the owner's `bwd_rows` signal. With more than 32 owner nodes, the
  lanes take 32 at a time, with a fence for each 32. When its stream wraps
  its ring, its puts count on the put counters. Then, after the puts of
  each pass, the warp marks the chunks the NIC has read, in order (lane `i`
  looks at the `i`-th from the first not yet read), and lane 0 frees their
  rows with one store (`bwd_free_end`, section 12.1). At the end it grants
  each rail owner the ring's rows of the next kernel.
* **Landing watcher** (a traced build only, across nodes: section 4.4):
  the warp before the issuer bins the gradient rows the rail proxies put
  into this rank's `lookup_rows` as they land, for the timeline.
* **Owner**: waits until every same-node requester has pushed and every
  rail proxy's rows have landed. Its `lookup_grad`, the gradient of its
  lookup output, is a view of `lookup_rows`. Every row of `lookup_grad` is
  written exactly once, so nothing is zeroed or scattered. The owner hands
  it to the lookup's backward. It is valid until the next lookup, which
  writes `lookup_rows` (section 11.2).
* **Task order.**
  * Remote queue (across nodes): the push tasks of the proxies' streams,
    chunk after chunk, the proxies in turn. Thus the chunks fill in stream
    order.
  * Local queue: the row tasks of the node's owners.

  A warp takes the next remote task when the rows of its chunk in the ring
  are free. Otherwise, it takes a local task. Thus the pushes feed the
  proxies as fast as the rings free rows, and the local rows fill the time
  between. (When the remote tasks came first, the local rows waited behind
  pushes that waited for the ring.) The heavy rows' sums are ready before
  the kernel starts, so no task waits for another task.

## 12. The ring and credits

### 12.1 The ring

A proxy's `proxy_rows` is a ring of `num_proxy_rows` rows, and its rows of
a call are a *stream* of chunks through it. A chunk is consecutive
node-unique keys of one owner node. Its rows are its keys' rows (each
key's first occurrence's), then their duplicates' slots (for the backward),
key after key in `uniq` order. Its *payload* is what it moves over IB: one
row per key. Under node dedup a key has up to `max_key_rows` rows in the
ring, but one row of payload. So the chunks are cut by counted rows. A
key's *counted rows* are its rows, but at least `rows_per_payload`: the
ring rows a chunk may spend per row it sends over IB, its most rows in the
ring over the payload rows that `DYNAMICEMB_HIER_PAYLOAD_KB` sets
(`chunk_ring_rows / payload_rows`, section 15.3; 4 at 512 KiB). A chunk
takes the keys whose counted rows before them, among the owner node's, fall
in one span of `chunk_span` counted rows: the owner node's chunk `j`, which
goes in the stream's round `j`. A key counts at most `max_counted_rows`, and
`chunk_span = max_chunk_rows − max_counted_rows + 1`. So:

* a chunk's counted rows are at most `max_chunk_rows`, and so are its rows,
  as a key's rows are at most its counted rows. A chunk takes at most
  `max_chunk_rows` rows of the ring: 2 MiB of rows, or
  `DYNAMICEMB_HIER_PAYLOAD_KB` if that is more, and at most half the ring
  (section 15.3);
* a chunk of keys of at most `rows_per_payload` rows each holds about
  `max_chunk_rows / rows_per_payload` keys: by default about 512 KiB of
  payload (section 4.4). A chunk of keys of more rows stops at
  `max_chunk_rows` rows, so its payload is smaller, down to about
  `max_chunk_rows / max_counted_rows` rows. Each owner node's last chunk
  takes the keys that are left;
* a key counts fewer rows than `chunk_span`, so no chunk is empty.

On Eos, on 2 nodes with power-law keys, chunks of 512 KiB of rows in the
ring made the forward and backward kernels 6.1% and 11.1% slower than
chunks of 2 MiB. Chunks cut by counted rows, with 512 KiB of payload, made
each 0.3% slower.

Only the cut of the chunks uses counted rows. A chunk's rows and its place
in the ring, a key's rows and slots, and the requesters' push orders and
tables count rows. With `rows_per_payload` 1 the counted rows are the rows.
A chunk is the unit of every step of the rows across the rail: one IB put
list per chunk and owner, grants and frees of whole chunks, and the
reducers' tasks of a chunk.

```
the stream   [ round 0: owner node a's chunk 0 | b's chunk 0 | ... ][ round 1: a's chunk 1 | ... ] ...
a chunk      [ rows of its keys | duplicates' slots, key by key ]
the ring     its chunks one after another; when the stream passes the ring, laps from the ring's start
```

* The stream takes the chunks round after round (in round `j` each owner
  node's chunk `j`), owner node after owner node in a round, each right
  after the one before. In a round, the owner nodes come in turn from the
  node after the proxy's own (`node + 1`, `node + 2`, ... mod `num_nodes`).
  All agents take the stream in order, on every node at about the same
  pace. So at a place of a round, the proxies of a rail serve different
  owner nodes, and each NIC of the rail sends and receives a share of the
  puts. In node order, the proxies served the same owner node at the same
  time: in the backward its NIC took the puts of several proxies, and in
  the forward of a stream that wraps the ring it sent to several proxies,
  while the other NICs were idle (incast). DeepEP turns its RDMA peers the
  same way (each rank starts from a different peer).
  `benchmark/gin_rail_congestion.cu` measures the effect on the rails
  (`benchmark/README.md`, *Link tools*). With the benchmark's configuration
  (`--use_index_dedup`, uniform and power-law keys), the backward kernel
  takes 0.4 to 1.6% less time in this order on 4 nodes, and 6.0 to 7.7%
  less on 8 nodes. The forward kernel takes the same time. With a large
  batch (16384 samples of 1536 keys of one feature), the time of neither
  kernel changes.
* Its positions are unwrapped: a stream that fits the ring lies as it is,
  which is the case whenever `remote_keys` rows fit under the cap (section
  15.3).
* A stream that does not fit takes laps. Lap `k` holds the chunks whose rows
  before them are in `[k × lap_span, (k + 1) × lap_span)`, with `lap_span =
  num_proxy_rows − max_chunk_rows + 1` (the stream rows in which a lap's
  chunks start), packed from position `k × num_proxy_rows`: no chunk crosses
  the ring's end. A chunk lies at its start `mod num_proxy_rows`.
* Every row of the ring is free when an out kernel starts. A chunk is done
  with in the forward once every requester of the node pulled it (the
  relay counts their pulls), in the backward once the NIC read its sums
  (the issuer counts the reads on the put counters). Chunks are done with
  in order. The rows of chunk `c` may be written once the first chunk not
  done with starts at `end(c) − num_proxy_rows` or later: in the forward
  the relay grants its owner those rows, in the backward the issuer frees
  them for the node's requesters (`bwd_free_end`). Each does so once per pass
  of its warp (the relay once per owner node for each 32 chunks it
  grants), for all the chunks done with in the pass (section 12.2).
* Every agent takes the chunks in stream order: an owner puts its chunks at
  a proxy in order, a requester takes its tasks chunk after chunk at every
  proxy, the reducers take theirs chunk after chunk. The backward's issuer
  puts an owner node's chunks out of order only within its look-ahead, which
  starts at the node's first chunk not yet put (section 11.4). So no task
  waits behind one that needs a later chunk, and the first chunk not done
  with can always be done with: its rows were free before it.
* The ring changes no result: every sum keeps its order (section 18). The
  tests across nodes also run with a 1 MiB proxy cap, where the larger
  calls' streams wrap the ring, against TorchRec's dists.
* When the stream wraps the ring, the backward's puts of chunk `c` count on
  GIN counter `c mod counters` (`counters = min(max_chunks, 64)`), and the
  issuer puts a chunk only within `counters` chunks of the first one not yet
  read, so two chunks in flight never share a counter.

### 12.2 Credits and sequence numbers

`in_seq` counts input dists and `out_seq` counts out kernels, the same on
every rank. The credits for the next out kernel (`out_seq + 1`) go out at the
end of kernel `out_seq`. At that time no one knows yet whether the next one
is a forward or a backward. A credit names the kernel it lets in, and other
kernels ignore it:

* `grant[proxy node] = (out_seq << 32) | position`, proxy → owner: "put the
  rows of my stream of kernel `out_seq` below `position` into my
  `proxy_rows`";
* `bwd_ready[owner node] = out_seq`, owner → proxy: "put the rows of
  backward `out_seq` into my `lookup_rows`".

For a forward with `out_seq` = 7, then its backward (8), then the next
forward (9):

```
end of forward 7    proxy -> owner: grant (8<<32)|ring      owner -> proxy: bwd_ready 8
backward 8          the proxy puts each chunk once it is ready, within its
                    issuer's look-ahead (section 11.4)
end of backward 8   proxy -> owner: grant (9<<32)|ring
                    (no bwd_ready: the owner's lookup_grad is still in use)
forward 9           the owner puts the ring's first lap at once; when the
                    stream wraps, the relay grants the rows that the pulled
                    chunks free, once per pass (below)
```

(`ring` is `num_proxy_rows`.)

The owner sends no `bwd_ready` at the end of a backward, because its
`lookup_grad` stays in use until the lookup's backward reads it. A backward
right after a backward therefore waits one round trip for `bwd_ready` (the
issuer, as owner, sends it when the kernel starts). Every other kernel can
put its first IB rows as soon as they are ready, without asking first.
Credits are signals with no data, each adding the difference to the value
sent before, on context 3.

Inside a kernel, the relay and the backward's issuer send their grants and
flags once per pass of their warp (section 13 gives their memory order).
In a pass the relay raises one grant per owner node for each 32 chunks it
grants: up to the end of that node's last chunk among them that the freed
rows hold. It also sets the `landed` word
of each chunk whose rows landed in the pass. The issuer stores `bwd_free_end`
once, for all the chunks the NIC read in the pass.

A call that does not fit raises and runs no out kernel, so it sends and
takes no credit. The credits of the last out kernel wait for the next call that takes
the op.

The in channel sends no credit of its own. On context 2, `count[node]` and
`keys[node]` count the count messages and the key messages of the rail peer
on that node, one of each per call. So `rail_exchange` waits until each
peer's `count` reaches `in_seq`, and `rail_done` until each one's `keys`
does. A rank's count message of a call also lets its rail proxies put into
its `rail_keys` again: it sends it after `rail_done` and the owner's tables
of the call before (section 13).

## 13. Memory order, and why nothing hangs

* **NVLink.** A writer stores rows with 16-byte stores, waits at a warp or
  CTA barrier, then does one *release* store (system scope) of a flag
  tagged with the sequence number, or one release add of its rows to a
  count (the backward's pushes into a chunk, the forward's pulls of one).
  The reader does an *acquire* load of the flag, or of the count until it
  reaches the rows it waits for, then reads the rows past the L1 cache.
  (Release and acquire are ordering rules: every write before the release
  is seen by a reader after its acquire.)
* **GIN.** The sender makes one system-scope fence (`fence.acq_rel.sys`,
  `RowPut::fence`), then puts. The forward fences once per kernel, since
  the lookup wrote `lookup_rows` before the kernel. The backward's issuer
  fences once per pass that found a ready chunk (for each 32 owner nodes).
  Each put adds to a signal at the receiver, which reads the signal before
  the rows. In the backward, a node peer writes rows over NVLink and the
  NIC then reads them. A chain orders the two: the push task's count
  (release), the reducer (acquire), the chunk's ready counter (release),
  the issuer's relaxed poll of it, and its fence before the put, which is
  the acquire of that poll.
* **The GIN warps' polls.** The relay and the backward's issuer poll with
  relaxed loads (`ld_relaxed_sys`, `Credits::read_relaxed`). After a pass's
  polls, the warp makes one `fence.acq_rel.sys` if they found news. That
  fence is the acquire of what the polls saw, and the release of the
  stores and puts after it. So the relay stores the chunks' `landed` words
  with relaxed stores after it (`st_relaxed_sys`); `landed_rows` keeps its
  release store. The issuer frees rows with one release store of
  `bwd_free_end` per pass. The forward's issuer reads its grants relaxed and
  makes no fence after them, as it reads nothing that a grant guards. A
  grant only lets its puts write rows of the ring that the proxy's
  requesters were done with before the relay granted them (the relay's
  fence, then the grant). And each put depends on the grant value that the
  lane read.
* **The in channel.** What peers wrote is read only after an acquire of
  their flag or signal. The comm kernel that acquired it reads past the L1
  cache (`node_exchange` reads `node_offsets`, and `rail_exchange` the count
  messages, with `__ldcg`), and so does
  `peer_gather` later on the stream; the planner's kernels read with plain
  loads. What this rank writes for its peers, with its own stores or with
  the planner's kernels earlier on the stream, comes before the flag it
  releases or the put it makes: a system fence orders it. In
  `node_exchange` thread 0 makes the fence after the CTA's barrier, so it
  covers the offsets that every thread stored, then releases `in_started`.
  Each CTA of the push ends with one, and `node_done`,
  `node_status_exchange`, the warp of `rail_exchange` that puts and
  `rail_key_push` make one before their flag or puts. `rail_exchange`
  flushes its puts before it ends, and `rail_done` flushes those of
  `rail_key_push` (below). What this rank stores into a peer's buffer comes
  after an acquire, earlier on the stream, of the peer's flag that says the
  peer read that buffer (the table below).
* **The late key wait.** `rail_key_push` only posts its puts. Each slice
  carries a strong signal, so an owner that sees the signal also sees the
  slice. `rail_done` acquires every rail proxy's key signal (system scope),
  then flushes this rank's puts. A flush waits until the NIC has done every
  put in the context's queue to each peer, from any kernel. Between the
  push and `rail_done` run `node_status_exchange`, the `peer_gather`s, the
  requester's `remote_map_gather` and its tables: none of them reads
  `rail_keys` or writes `key_stage`, and the planner reads the rail's
  offsets from the layout's tensors, not from the window. The first reader
  of `rail_keys`, the owner's tables, runs after `rail_done` on the stream.
  The next call writes `key_stage` in its proxy steps, after this
  `rail_done` on the stream, so after the NIC read the slices. Its rail
  proxies put into `rail_keys` only once they have this rank's count message
  of that call, which its `rail_exchange` sends after this call's
  `rail_done` and owner's tables.
* **No hang.** A kernel that waits for peers waits only for the same kernel
  on its peers (`rail_done` for its rail peers' `rail_key_push`, which
  never waits), for credits from earlier out kernels, and for its own
  rank's earlier work. The planner's kernels, the pushes and the gathers
  never wait. The op orders nothing across streams: the caller orders the kernels
  the same way on every rank (section 4.2). It makes the same calls in the
  same order, runs the input dists on one stream and the out kernels on one
  stream, with the same waits between its streams. So every rank issues the
  same kernels in the same order. The CTA budget (section 14) lets the
  kernels that may wait at the same time run together: an out kernel, a
  comm kernel on another stream, and NCCL's kernels. So every kernel ends:
  the first one can always end, and each later one can end once the ones
  before it have. A call that does not fit runs its input dist like any
  other call, then raises on every rank, and runs no out kernel. A wait that lasts longer than
  the timeout prints what it waited for and traps.

A buffer or control word of the window is written in call `in_seq + 1` only
once no peer still reads what call `in_seq` wrote there. The words marked
(ctrl) are in the ctrl section, named by their accessors (`CtrlLayout`):

| buffer or word | written by | last read by | why that is safe |
|---|---|---|---|
| `node_offsets` | node peers' `node_exchange`, and this rank its own row | this rank's `node_exchange` | a peer stores its row of the next call before that call's node barrier, but only after its `node_done` of this call got this rank's tell, which this rank sends after its `node_exchange` read the rows; `node_done` runs in every call, so the rows need no slot per `in_seq` parity |
| `node_keys`, `node_key_weights`, `node_bucket_feature_of_row` | node peers' push | this rank's planner | the peers push only after the next call's node barrier, which this rank publishes only once its in channel finished the call |
| `remote_map`, `proxy_header` | this rank's planner | node peers' gathers | this rank writes them after the next call's node barrier, which every peer publishes only once its in channel finished the call |
| `key_stage` | this rank's planner (the proxy) | the NIC | `rail_done` flushes the key puts before the owner's tables, so before the next call's proxy steps |
| `count_msg_out` (ctrl) | this rank's `rail_exchange` | the NIC | `rail_exchange` flushes its puts before it ends |
| `rail_keys` | rail proxies' key puts | this rank's owner tables | a rail proxy puts only after it has this rank's count message of the next call, which this rank's `rail_exchange` sends after this call's `rail_done` and owner tables |
| `count_msg` (ctrl) | rail peers' count puts, and this rank its own | this rank's `rail_exchange` | a slot per `in_seq` parity (and source node): a peer's message of call `in_seq + 2` needs this rank's of call `in_seq + 1`, which this rank sends after its `rail_exchange` of call `in_seq` read the slots |
| `requester_offsets`, `needs` and `in_status` (ctrl) | this rank | node peers' `node_status_exchange`, which copies them out | this rank writes them after the next call's node barrier |

## 14. The CTA budget

A kernel that waits for peers must be running at the same time as every kernel
it may wait on, or the GPU could hang. Three kinds of kernels wait. The caller
runs the out kernels on one stream, so one of them waits at a time. A comm
kernel that waits runs on one CTA, and an input dist may run at the same time
on another stream (a pipeline's data_dist stream). NCCL's own kernels may wait
too. So an out kernel takes `min(want, SMs − 34)` CTAs: it leaves one SM to
such a comm kernel, 32 to NCCL's kernels, and one to spare. It takes at least
1 CTA, and the backward across nodes at least 2 (its issuer's CTA and a
reducing one). `want` is `DYNAMICEMB_HIER_OUT_CTAS` when set, else 32, or 64
for the backward on one node (section 4.4). The input dist's other kernels
never wait (section 13), so they need no SM of their own. The kernels of the
heavy rows do not wait for peers either. They can use all SMs. **Example**, H100 (132
SMs): an out kernel may take up to 98 CTAs, so the out kernels keep their 32,
and the backward on one node its 64.

## 15. Memory

### 15.1 What is counted

Per rank and sharding, the window (outside PyTorch's caching allocator) is
*persistent*: made when the op is built, and kept through the dense part of
every step. A call's tensors live *per call*, from its bucketize to its
backward (one call in a plain loop, two or three under the train
pipelines). The input dist's working memory, the bucketize's and the
planner's, is *transient*: a step frees its temporaries as soon as it is done with them.
The caching allocator keeps their blocks, though, in its pool of the stream
they were made on: the caller's stream that the input dist runs on. So that
pool holds the largest step's working memory.

### 15.2 Formulas

The capacities are the ones in place (section 5), and `row_bytes` and
`record_bytes` those of section 5.3. Section 15.3 defines `num_proxy_rows`,
`max_chunk_rows`, `max_chunks` and `max_owner_chunks`.

| buffer | bytes | kind |
|---|---|---|
| `ctrl` (the ctrl section) | one node: `16 × local_size + 72`; across nodes about `8 × (num_features × (local_size + (2 × num_nodes + 1) × (num_nodes + 1)) + 2 × num_nodes² + 15 × num_nodes + 2 × local_size + 3 × max_chunks + 64)`, most of it the count messages (this rank's, and a slot per `in_seq` parity and node): 1.1 MB at 32 nodes, 8 GPUs per node and 64 features, 0.98 GB at 256 nodes and 1,000 features, and 24 bytes per chunk of `max_chunks` | persistent |
| `node_offsets` | `8 × local_size × (world_size × num_features + 1)`: 16 KB at 4 nodes of 8 GPUs and 8 features | persistent |
| `node_keys` | `8 × (lookup_keys + remote_keys)` (one node: `lookup_keys` alone) | persistent |
| `node_key_weights` | with weights, `4 × (lookup_keys + remote_keys)` | persistent |
| `node_bucket_feature_of_row` | `4 × remote_keys` | persistent |
| `remote_map` | `12 × remote_keys`: three int32 arrays | persistent |
| `proxy_header` | `4 × (4 + max_chunks + local_size × (max_chunks + 1))` | persistent |
| `key_stage` | `record_bytes × remote_keys + 16 × max_chunks + 16` | persistent |
| `rail_keys` | `record_bytes × lookup_keys + 16 × max_owner_chunks` | persistent |
| `lookup_rows` | `row_bytes × lookup_keys` | persistent |
| `proxy_rows` | `row_bytes × num_proxy_rows` (section 15.3) | persistent |
| the requester's maps | 8 bytes per key on one node (`inv`, `src`), 16 across nodes (also `dst`, `push_order`) | per call |
| the call's small tables across nodes: the requester's of the node's proxies and its remote task offsets, the owner's and the proxy's (section 11.1) | about `24 × local_size × max_chunks + 48 × max_chunks + 16 × max_owner_chunks`, plus a few KB | per call |
| the owner's KJT values | `8 × lookup_keys` (12 with weights), from the input dist until the forward | per call |
| the proxy's rows before each key (`node_exp_off`) | `8 × remote_keys` | per call |
| the put lists and the ready counters | `24 × max_chunks + 16 × max_owner_chunks + 32 × num_nodes × num_features` | per call |
| index dedup's fan-out table | about 13 bytes per original key (on this path the deduplicated KJT keeps the original length) | per call |
| the bucketize, in the caller's pool | at its peak 32 bytes per key, its outputs (8) included, for either owner rule; about 60 with index dedup, measured while the fan-out table was made by scans, when the fan-out's own tensors reached about 53 bytes per key; they now reach about 28 at most, during its sort (from the code) | transient |
| the planner, in the caller's pool | its largest step: across nodes the proxy's about 64 bytes per entry of `remote_keys` (60 without weights), the requester's about 73 per key, or the owner's gather about 65 per entry of `lookup_keys`, at most 2^22 entries at a time; on one node about 20 bytes per key | transient |

The transient figures are upper bounds from the peaks measured with the
benchmark's batches (bytes in live tensors; the caching allocator's rounding
and the cached blocks it hands out add up to about 13%), but the one-node
planner's and the proxy's, which come from the code. They were measured
before the chunks were cut by counted rows, and while the bucketize also
kept each key as `int64` (8 bytes per key, until the input dist returned).
It no longer does: the push reads the KJT's keys (section 10.1).

The proxy's figure counts its steps' live tensors. Its node dedup holds
about 52 bytes per entry, 64 with weights: during the insert its hash table
(the bucket heads, 8 to 16 bytes, and the chain links, 4) and each
identity's state (24, or 32 with weights), then that state, the
representatives before each occurrence and its tables per occurrence and
per key. By the same count its sorts held about 79 bytes per entry, and 99
with weights. The 99 measured earlier with the benchmark's batches, which
carry no weights, was of an older proxy. With `rows_per_payload` above 1, as
by default, its next step holds each key's counted rows (int32) and the
counted rows before each key (int64) while it cuts its chunks and gives the
slots: about 60 bytes per entry, 64 with weights, now its largest.

The constructors of the layouts size the window's buffers as above
(`CtrlLayout`, `KeysLayout` and `RowsLayout`, in `window.h`). The buffers
from `node_bucket_feature_of_row` to `rail_keys`, and `proxy_rows`, exist
across nodes only, and `node_key_weights` with weights
only: else they take 0 bytes. Each buffer, the ctrl section too, takes a whole number of 4 KiB
units (`kBufferAlign`). So the window's bytes are the sum of its buffers,
each rounded up to 4 KiB (`WindowLayout::bytes`), and its copy on each rank
is a whole number of allocation granules. `memory()` gives both (section
7.1).

With uniform keys across nodes, and capacities 1.25 times the needs
(section 5.2), the keys section takes about
`1.25 × num_keys × (16 × (1 + (num_nodes − 1) × uniq_share) + 32 × (num_nodes − 1)) / num_nodes`
bytes. That is about 39 bytes per key at 2 nodes, and 58 at 32 nodes. The
per-call tensors take about 31 to 36 bytes per key across nodes: 16 for the
requester's maps, about 10 for the owner's KJT until the forward, and
`8 × 1.25 × (num_nodes − 1) / num_nodes` for the proxy's rows before each
key.

### 15.3 `proxy_rows` and its ring

```
num_proxy_rows   = min(max(remote_keys, 4 × max_key_rows), proxy_bytes / row_bytes)   (rounded down)
max_key_rows     = local_size
payload_rows     = max(payload_kb KiB / row_bytes, 32)
cap_rows         = max(max(payload_kb KiB, 2 MiB) / row_bytes, 32)
chunk_ring_rows  = min(cap_rows, num_proxy_rows / 2)
max_chunk_rows   = max(chunk_ring_rows, 2 × max_key_rows)
rows_per_payload = max(chunk_ring_rows / payload_rows, 1)                             (rounded down)
max_counted_rows = max(max_key_rows, rows_per_payload)
chunk_span       = max_chunk_rows − max_counted_rows + 1
max_chunks       = rows_per_payload × remote_keys / chunk_span + num_nodes
max_owner_chunks = (num_nodes − 1) + min(rows_per_payload × (num_nodes − 1) × remote_keys, max_counted_rows × lookup_keys) / chunk_span
proxy_bytes      = DYNAMICEMB_HIER_PROXY_MB MiB (512 by default)
payload_kb       = DYNAMICEMB_HIER_PAYLOAD_KB (512 by default)
```

`payload_rows` is a chunk's payload in rows, and `cap_rows` the most rows
it takes in the ring: 2 MiB of rows, or the payload if that is more (each
at least one task of 32 rows). `chunk_ring_rows` keeps a chunk within half
the ring, and `max_chunk_rows` gives it room for two keys of the most rows
at least. `rows_per_payload` is `chunk_ring_rows / payload_rows`, rounded
down and at least 1: the ring rows a chunk may spend per row it sends over
IB. A key of fewer rows counts that many, so a chunk of such keys holds
about `max_chunk_rows / rows_per_payload` keys. `rows_per_payload` is 1 when
half the ring holds less than the payload, and when `payload_kb` is 1040 or
more (but with very large rows). The counted rows are then the rows.

So the ring holds a call's rows in one lap when `remote_keys` rows fit under
the cap: its stream then lies in it as it is. A cap below `4 × max_key_rows
× row_bytes` (two chunks of keys of the most rows) is refused when the op is
built (section 6), and so is a `remote_keys` whose stream could pass
position 2^32 (`2 × remote_keys + num_proxy_rows` of 2^32 or more).

The **rows a proxy needs** count, per node-unique key, one row per
occurrence (the key's row and its duplicates' slots): at most `local_size`,
as index dedup leaves each GPU each (table, key) once (a key of more, in a
call that raises, counts `local_size`). So a call never needs more rows
than it has occurrences: a call within `remote_keys` fits in
`num_proxy_rows` rows when they hold `remote_keys` rows. It makes at most
`max_chunks` chunks. An owner node's chunk `j` starts where the node's
counted rows before it reach `j × chunk_span` or more. So a node of `r`
counted rows has at most `r / chunk_span + 1` chunks (rounded down). A key's
counted rows are at most `rows_per_payload` times its rows, so a proxy's
counted rows are at most `rows_per_payload × remote_keys`. An owner gets at
most `max_owner_chunks` chunks: a proxy's stream holds at most
`r / chunk_span + 1` chunks of the owner's node, `r` its counted rows of
that node, and the keys the owner looks up, at most `lookup_keys`, count at
most `max_counted_rows` each. The laps of a stream follow from its rows
(section 12.1).

### 15.4 How memory scales

**With the number of GPUs and features.** Only one buffer of the window is
sized by `world_size`: `node_offsets`, a row of `world_size × num_features + 1`
int64 words per GPU of the node. It takes `8 × local_size × (world_size ×
num_features + 1)` bytes: 1 MB at 32 nodes, 8 GPUs per node and 64
features, and 125 MB at 256 nodes and 1,000 features. Per call, the
bucketize's `offset_per_bucket_feature` and the node layout's places are
sized by `world_size` too (`8 × world_size × num_features` bytes each),
and so are the proxy's block tables while its node dedup runs (about
`8 × (world_size + num_nodes) × num_features` bytes). Apart from these,
`num_nodes` enters only through:

* the share `(num_nodes − 1) / num_nodes`;
* the control words, most of them the count messages: this rank's, and a
  slot per `in_seq` parity and node, about
  `8 × (2 × num_nodes + 1) × (num_nodes + 1) × num_features` bytes, which
  grow with the square of the nodes. With the other control words that is
  1.1 MB at 32 nodes, 8 GPUs per node and 64 features, and 0.98 GB at 256
  nodes and 1,000 features (section 10.4 gives what a call sends). Per
  call, the rail layout's KJT offsets (`source_offsets`, `owner_offsets`)
  are linear in `num_nodes × num_features`;
* GIN's queues to `num_nodes − 1` rail peers per context (with
  `NCCL_CROSS_NIC=0`);
* the chunk tables, which grow with `max_chunks` (linear in
  `rows_per_payload × remote_keys`, plus `num_nodes`): `proxy_header`, the
  chunks of `key_stage`, and per call the requester's tables of the node's
  proxies (which also grow with `local_size`) and the proxy's; and the
  chunks an owner gets (`max_owner_chunks`: `rail_keys` and the owner's
  tables), which grow with `num_nodes` until
  `max_counted_rows × lookup_keys / chunk_span`;
* the put lists, a segment per chunk and per (node, feature).

`local_size` enters through `node_offsets` (a row per GPU of the node),
and through the tables kept per GPU of the node: the proxy header's push
tables, and per call the requester's tables of each proxy and the proxy's
`part_offsets` of each requester. Two tables hold a count per (sender,
receiver, feature). The count messages hold one per (node, node, feature):
each rail peer's node-unique keys per owner node and feature. `node_offsets`
holds one per (GPU of the node, GPU, feature): the bucketize's
`offset_per_bucket_feature` of each GPU of the node.

Everything else is linear in the capacities, except `proxy_rows` (capped: a
larger call's stream takes more laps of the ring).

**With batch × sequence length.** The table below is computed from the
sizing rules in the code, with capacities 1.25 times the needs and at least
4096 entries more (`hier_capacities`), and from the transient figures of
section 15.2. It assumes uniform keys, mean loads, no weights, one owner
rule and 8 GPUs per node; it is not measured. It counts chunks of 2 MiB of
rows (`rows_per_payload` 1). The default chunks make the chunk tables, a
small part of each figure, up to `rows_per_payload` times as large, and add
to the proxy's step (section 15.2), which stays below the largest. GB per
rank.
"Persistent" is the window: `lookup_rows` + `proxy_rows` + its other
buffers (the ctrl and keys sections); "per call" is one call's tensors;
"planner" is the planner's largest step's working memory, and "bucketize"
the bucketize's, both in the caller's pool.

| case | nodes (GPUs) | `lookup_rows` | `proxy_rows` | other buffers | persistent | per call | planner | bucketize | laps |
|---|---|---|---|---|---|---|---|---|---|
| the benchmark: 540,672 keys per rank, fp32 dim 128, `uniq_share` 0.935 | 1 (8) | 0.35 | – | 0.005 | **0.35** | 0.010 | 0.011 | 0.014 | – |
| | 2 (16) | 0.34 | 0.17 | 0.021 | **0.53** | 0.018 | 0.043 | 0.014 | 1 |
| | 4 (32) | 0.33 | 0.26 | 0.027 | **0.62** | 0.019 | 0.042 | 0.014 | 1 |
| | 8 (64) | 0.33 | 0.30 | 0.029 | **0.66** | 0.020 | 0.041 | 0.014 | 1 |
| | 32 (256) | 0.32 | 0.34 | 0.031 | **0.69** | 0.020 | 0.041 | 0.014 | 1 |
| 16,384 samples × 1,536 keys per sample, 1 feature: 25.2M keys per rank, bf16 dim 128, `uniq_share` 1 | 2 (16) | 8.05 | 0.54 | 1.01 | **9.60** | 0.82 | 1.84 | 0.63 | 7 |
| | 4 (32) | 8.05 | 0.54 | 1.26 | **9.85** | 0.90 | 1.84 | 0.63 | 10 |
| | 8 (64) | 8.05 | 0.54 | 1.38 | **9.97** | 0.93 | 1.84 | 0.63 | 11 |
| | 32 (256) | 8.05 | 0.54 | 1.48 | **10.1** | 0.97 | 1.84 | 0.63 | 12 |
| the same with 2 features: 50.3M keys per rank, fp32 dim 128 | 2 (16) | 32.2 | 0.54 | 2.01 | **34.8** | 1.62 | 3.67 | 1.26 | 25 |
| | 8 (64) | 32.2 | 0.54 | 2.78 | **35.5** | 1.87 | 3.67 | 1.26 | 43 |
| | 32 (256) | 32.2 | 0.54 | 2.96 | **35.7** | 1.94 | 3.67 | 1.26 | 47 |

The planner's working memory adds 3% to the persistent memory on one node and
6–8% across nodes at the benchmark's size, and 10–19% at the large batches
above. Its largest step is now the owner's gather or the requester's: the
proxy's step was the largest from 4 nodes on while its node dedup sorted,
by that earlier measurement (section 15.2; 8–9% and 11–30%). Take 4 nodes,
fp32 dim 128, one feature, and every key distinct. For 1, 4, 16 and 64 Mi
keys per rank, the persistent memory is 1.2, 3.4, 12.1 and 47 GB, in 1, 4,
13 and 49 laps of the ring.

Most of the persistent memory at large batches is `lookup_rows`. It is one
copy of the owner's rows after node dedup, kept through the dense part
because the lookup writes it in place. TorchRec's dists keep no such buffer.
But their output dist holds several copies of the rows at its peak: the
reordered rows, the all-to-all's output, and the output in batch order.
Under skewed keys its hot owner's buffers grow with
`world_size × hot_share × num_keys`.

## 16. Pipelines

| loop | status |
|---|---|
| plain loop (`.wait().wait()`), `TrainPipelineBase` | supported, tested |
| `TrainPipelineSparseDist` | supported, tested: the next batch's in channel runs during this batch's forward and backward |
| `PrefetchTrainPipelineSparseDist` | supported, tested; frequency weights reach the prefetch through DynamicEmb's `GroupedEmbeddingsLookup.prefetch` |
| `TrainPipelineFusedSparseDist`, with `emb_lookup_stream` `"data_dist"` and `"new"` | supported, tested (TorchRec 1.3 and up) |
| `EvalPipelineSparseDist` | supported, tested: eval forwards take TorchRec's dists |
| the repo's `JaggedMegatronTrainPipelineSparseDist` and its prefetch variant | supported, tested |
| `TrainPipelineSemiSync` | runs; tested against itself (its outputs are not the plain loop's), without rehashes |
| several forwards, then their backwards; `retain_graph`; forwards without grad | supported, tested |
| a call that does not fit the capacities, under the plain loop and each train pipeline above | tested: it raises `RuntimeError` on every rank (section 5.4) |
| a different call order on different ranks | not supported: it hangs until the timeout |

* **Streams.** The op queues each step on the current stream and orders
  nothing across streams; TorchRec's loops order its steps (section 4.2).
  In a plain loop every step runs on the default stream, so the input dist
  and its host sync run in line: the second wait's copy waits for
  everything queued before it, and the CPU cannot run ahead of the GPU
  across it. Under the train pipelines the input dist runs on the
  pipeline's data_dist stream, so it overlaps with the previous batch's
  compute, and the second wait waits for that stream alone. So the op does
  not bound how far the CPU runs ahead of the default stream; the pipeline
  does. Each step the CPU runs ahead keeps its memory blocks reserved in
  the caching allocator. The out kernels of a model run on one stream, in
  the same order on every rank: autograd runs each backward on the stream
  of its forward.
* **`record_stream`.** The pipelines call the sharding context's
  `record_stream` for the streams they hand a call to, and it records the
  plans' tensors there too (`HierCall.record_stream`). With `emb_lookup_stream="new"`, TorchRec's fused pipeline runs the
  lookup and the output dist on that stream but records the context on the
  default stream only. So the lookup records the owner's KJT, and the
  output dist's forward the plans, on the stream they run on: the forward's
  plan goes as soon as the forward kernel is queued, and without the record
  the input dist of the next call, on the data_dist stream, could get its
  memory while the kernel still reads it. The op records no stream itself.
* A call's tensors live from its bucketize to its backward (section 15.1), so
  under `TrainPipelineSparseDist` the next batch's are alive during this
  batch's forward and backward.
* With `emb_lookup_stream="new"`, TorchRec's fused pipeline runs the lookup
  and the forward kernel on a stream of its own. The output dist's
  awaitable makes the dense forward's stream wait for the kernel (section
  7.4).

## 17. TorchRec versions

TorchRec 1.2 to 1.8 are supported. The dist uses only these parts of TorchRec:
its `Awaitable` classes, its sequence sharding context, the KJT constructor,
and `RwSequenceEmbeddingDist` and `KJTAllToAll` for calls on TorchRec's dists
(eval forwards). The input dist keeps a real `KJTAllToAll`, because
TorchRec's pipelines check its type; an eval forward runs it. Two version checks handle the
rest: `_create_embedding_kernel` takes an `env` argument from 1.5 on, and the
model delta tracker hook exists from 1.5 on. `TrainPipelineFusedSparseDist`
exists from 1.3 on.

## 18. Correctness and determinism

**Forward: bit for bit equal to TorchRec's dists (under index dedup).**
Each output row is a byte copy of the owner's row. For each table the owner
looks up the same set of keys as TorchRec's owner: every same-node
occurrence, and one entry per (other node, table, key). It sees the same
frequency counts, so inserts, admission and LFU scores match. As with any
change of key order, random initializers and eviction under capacity
pressure may differ. TorchRec's own `unique` order is not fixed either. So
the bit-exact tests use the DEBUG initializer and no eviction.

**Backward: per (table, key), the same terms.** Every row of `lookup_grad`
is written once, and every occurrence adds to exactly one row, so the
lookup's backward adds up the same terms as TorchRec's:

* fp32 rows: equal up to the order of fp32 additions. The `1 / world_size`
  scaling is TorchRec's bit for bit (a multiplication by `1 / world_size`
  in fp32, rounded once).
* bf16 or fp16 rows with node dedup: a node's sum of several occurrences is
  rounded to the row dtype once more before it crosses IB. The error is at
  most `2^-8` of that sum for bf16, and `2^-11` for fp16. Index dedup's sums
  are made in fp32, where TorchRec's `index_add` adds in the row dtype.
  `node_dedup=False` sends each GPU's row across IB on its own, at the cost
  of more IB traffic.
* Gradient clipping clamps each row before the per-key sum, so it turns node
  dedup off and keeps TorchRec's meaning.

**Run to run.** Every sum the dist makes has a fixed order: a key's rows at
its proxy by requester and position, and index dedup's sums by original
row. The planner numbers keys, slots and tasks with stable sorts,
cumulative sums and integer atomics whose result does not depend on their
order, which give the same result every run (all but index dedup's heavy
slots and their sub-tasks, below), and adds weights as whole numbers. The
proxy's node dedup takes each key's representative with an atomic minimum,
counts its occurrences and adds its weights with atomic adds, and keeps a
list of its occurrences that it reads only as a set. No step reads which
bucket of its hash table a key took, and no result depends on which of its
occurrences claimed it: that occurrence only holds the key's state. None of
the planner's steps adds up
floating-point values with atomics. One step hands out heavy slots with an
atomic whose order shows: index dedup's heavy rows take their heavy slots and
sub-tasks in the order the fan-out's kernel reserves them, which can change
from run to run (section 10.1). No sum depends on that order: a heavy row's
sub-tasks stay consecutive and in original-row order, so its sums add the
same rows in the same order. So
the output rows and the gradient handed to the lookup are the same bits every
run. That holds on one node and across nodes, whether a stream wraps its
ring or not. The keys in the owner's KJT columns follow DynamicEmb's
`unique` numbering: the same-node requesters' columns directly, and the
other nodes' columns through their representatives' places in their
requesters' orders, with node dedup the requesters in turn (section 9). So
do the rows of `lookup_grad`, and across nodes the chunk cuts and the put
lists. That numbering may change between runs; the value per key does not.
Without node dedup such a column can hold a key once per requester that
asked it, and those rows come in requester order every run.
DynamicEmb's
lookup backward adds a hot key's partial sums with 16-bit atomic additions on
every dist, so tables of 16-bit rows may still differ from run to run.

**Tests** (`test/unit_tests/`): `test_hier_bucketize.py`, `test_hier_dedup.py`,
`test_hier_dist.py`, `test_hier_overflow.py` and `test_hier_pipeline.py`;
each module's docstring says what it checks and how to run it. On one node
`test_hier_dist.sh`, `test_hier_overflow.sh` and `test_hier_pipeline.sh`
run them, and `test/unit_test.sh` runs those.

The node's part of a call has hooks of its own in `testing.cu`:

* `hier_output_dist_node_layout(node_offsets, num_nodes, local_size, node,
  local_rank, num_features, lookup_keys, remote_keys, ...)` gives one rank's
  node layout on its node's offsets (row `r` holds requester `r`'s
  `offset_per_bucket_feature`, as `node_exchange` gathers them). On one
  node, given the own part of its `node_keys` after every requester pushed
  and the `src` of its own push, it also gives its KJT;
* `hier_output_dist_node_push` runs one requester's push (`push_rows`, as
  `node_key_push` runs it) into stand-ins for its targets' `node_keys`;
* `hier_output_dist_delay_node_reads(ns)` makes each later input dist of the
  process wait `ns` on the GPU twice: in `node_exchange` between the node
  barrier and the read of `node_offsets`, and between `node_done` and the
  first read of `node_keys`. The skew case of `test_hier_pipeline.py` uses
  it on one node: under `TrainPipelineSparseDist` rank 1 reads its node's
  offsets and keys 20 ms late, while its node peers start the next call.
  Its outputs must equal the plain loop's bit for bit, and so must its
  tables where every sum is exact (a power-of-two world size of at most
  128).

The rail's part has two: `hier_output_dist_proxy_plan` gives a proxy's plan
and its count message (`"message"`), and `hier_output_dist_rail_tables` one
rank's rail layout from its rail's messages (row `n` the message of the rail
peer on node `n`, as `rail_exchange` reads them) and, given its `rail_keys`
and the own part of its `node_keys`, its owner tables and put lists. Both
run the device code of `rail_exchange` (`input_dist/rail.cuh`).

`test_hier_bucketize.py` plays every rank of one node with the first two
hooks, and `test_hier_dedup.py` every rank of an N-node job. Among its
checks: each proxy's `owner_offsets` equal its owners' `source_offsets` and
KJT offsets.

## 19. Limitations

* **Scope**: row-wise sequence shardings of DynamicEmb tables, within the
  limits of section 4.2.
* **Owner rules**: `roundrobin` and `hash_roundrobin` only. A sharding with
  a `continuous` table takes TorchRec's dists under `"auto"`, and raises
  `ValueError` under `"hierarchical"` (section 6).
* **No score per key**: tables of `score_strategy` `PRECISE_CUSTOMIZED` take
  TorchRec's dists under `"auto"`, and raise `ValueError` under
  `"hierarchical"` (section 6).
* **One key dtype**: the GPUs of a node give the op keys of one dtype
  (`int32` or `int64`) in a call. The op does not check it (section 9).
* **Fixed capacities**: a call that needs more raises, on every rank
  (section 5.4); the capacities must hold the largest call.
* **Errors no agreement covers**: an NCCL error while the op makes its
  communicator, or inside the collective window registration, on some ranks
  only leaves the other ranks waiting (section 7.2). An error inside an
  input dist, such as an allocation that fails, reaches Python on its rank. The
  other ranks then trap at the timeout in the call's next comm kernel that
  waits, as when a rank dies.
* **Memory held through the dense part**: `lookup_rows` and `proxy_rows`
  stay allocated, each sharding has its own window, the caching allocator's
  pool of the input dist's stream keeps the largest step's working memory,
  and the hottest owner's need sizes `lookup_keys` on every rank (section
  15).
* **Host time of the input dist**: across nodes an input dist launches
  about 80 kernels, CUB's included (section 10; an estimate from the code,
  as 117 were measured before the proxy's node dedup dropped its sorts),
  and the training thread spends their host time, also under the train
  pipelines.
* **Size of one call**: a GPU may receive, deduplicate or look up fewer than
  about 2.1 billion keys of each kind in one call (section 4.3).
* **Many nodes with many features**: the rail all-gathers count messages of
  about `num_nodes × num_features` words. So per call a rank sends and gets
  about `num_nodes² × num_features` words, its ctrl section keeps about
  `16 × num_nodes² × num_features` bytes, and one CTA lays the rail out
  from them. That is 865 KiB each way per call and 1.7 MB of ctrl at 64
  nodes and 26 features, and about 500 MiB each way per call and 0.98 GB of
  ctrl at 256 nodes and 1,000 features (sections 10.4 and 15.4). The
  proxy's node dedup also counts the keys of its `local_size × num_nodes ×
  num_features` blocks on one CTA, as many as the node layout's
  `row_shift` has entries: 16K at 32 nodes of 8 GPUs and 64
  features, and 2M at 256 nodes and 1,000 features.
* **Views**: a lookup output is a view of `lookup_rows`, valid until the
  next out kernel, and `lookup_grad` is one, valid until the next lookup
  (sections 11.2 and 11.4).
* **Call order**: every rank must make the same calls in the same order. A
  different call order on different ranks hangs until the timeout, which
  traps (section 13).
* **Streams**: the op orders nothing across streams. The caller orders the
  steps of each call on its streams, runs the input dists on one stream and
  the out kernels on one stream, and records the call's tensors on the
  streams that use them, as TorchRec's loops do (sections 4.2 and 16).
* **The owner's KJT**: code that reads it (trackers, for example) sees
  `kjt_stride` columns per feature and one entry per (other node, table,
  key) (section 9).
* **16-bit rows with node dedup** get one more rounding per node sum
  (section 18).
* **Skewed keys**: lower IB use in the backward, and a slow rank's lookup
  delays the other ranks' output-dist kernels. A chunk of hot keys, which
  have many rows in the ring, moves less than `DYNAMICEMB_HIER_PAYLOAD_KB`
  over IB: down to `max_chunk_rows / max_counted_rows` rows (section 12.1).
* **`int32` keys**: checkpoint loading and delta replay
  (`dynamicemb/key_ownership.py`) place every `roundrobin` and
  `hash_roundrobin` key by the 64-bit rule (`continuous` keys raise there).
  For negative keys, and under `hash_roundrobin` on a world size that is not
  a power of two, that rule gives other ranks than the bucketize's (section
  10.1). Such keys load on other ranks than the ones that look them up: a gap
  of DynamicEmb's, not of this dist.
* **Freeing tables**: a model dropped right after its last step, without a
  device sync, can crash later with an illegal memory access. DynamicEmb
  frees its table memory without waiting for the kernels that still read it
  (seen with TorchRec 1.2.0); `quiesce()` syncs the device first (section
  4.6).
* **Teardown** without `quiesce()` may wait up to `min(timeout, 30 s)`, or
  keep the window until the process exits (section 4.6).
* **A timeout traps**, which loses the CUDA context (section 13).

## 20. Where the code is

Paths from `corelib/dynamicemb`.

* **Python** (`dynamicemb/shard/`): `hier_output_dist.py` holds
  `HierDist` (the op, what a sharding's calls share, and a call's
  second wait, `finish`), what a call leaves its output dist (`HierCall`),
  the output dist and `quiesce`; `input_dist.py` the hierarchical input
  dist (`hier_forward` and its two awaitables); `rw_sharding.py` the choice
  of dist and the lookup; `embedding.py` the choice per call (eval
  forwards take TorchRec's dists). The
  table options are in `dynamicemb/dynamicemb_config.py`.
* **C++** (`src/hier_output_dist/`, namespace `dyn_emb::hier_output_dist`):
  the op's own files, and a folder for what both dists share and one for
  each dist.
  * `hier_output_dist.h`, `hier_output_dist.cu`: the op, `HierOutputDist`,
    and what it owns: its communicator (`Communicator`), its window
    (`SymmetricWindow`) and its two channels; its construction, teardown and
    calls, and the kernels of its setup and teardown;
  * `bind.cu`: the Python bindings; `testing.cu`: bindings for the tests and
    the benchmark only.
  * `common/`, what both dists share:
    * `common.h`: what host and device code share: the constants, the ranks
      of a sharding (`Topo`) and the op's sizes (`Sizes`), the formats of the
      call's tables, the call's status, the GIN contexts and signals, and the
      trace's build flag and row format;
    * `config.h`, `config.cu`: error checks and small host helpers, the op's
      configuration (`Config`) and its checks, the payload of a chunk and its
      cap in the ring (`kDefaultPayloadKb`, `kMinPayloadKb`,
      `kChunkRowCapBytes`), and the op's sizes (`sizes_of`,
      `rows_per_payload_for`, `max_chunks_for`);
    * `window.h`: the window's layout (`WindowLayout`) over its sections:
      `CtrlLayout` (with the format of the count message), `KeysLayout`
      (with `ProxyHeader`, `RemoteMap` and what `peer_gather` reads) and
      `RowsLayout`;
    * `device.cuh`: the kernels' building blocks: loads, stores and fences
      with a memory order, bounded waits, warp and CTA helpers, a search, row
      vectors, the window as the node's GPUs map it (`NodeWindow`), the syncs
      on its control words (`NodeBarrier`, `PeerFlags`), and GIN on the rail
      (`gin_context`, `issuer_sharing`, `RowPut`, `RailMessage`,
      `Credits`);
  * `input_dist/`, the input dist:
    * `input_dist.h`, `input_dist.cu`: what the input dist of a call gives
      (`InDist`) and the in channel (`InChannel`), whose `input_dist`
      queues a call's whole input dist;
    * `plan.h`, `plan.cu`: the output dist's plans (`FwdPlan`, `BwdPlan`),
      and the same as the out kernels read them (`FwdPlanView`,
      `BwdPlanView`);
    * `comm_kernels.cu`: the comm kernels, and the in channel's functions
      that launch them;
    * `node.cuh`: the device code of the node's part of a call, which the
      comm kernels, the planner's kernels and the bucketize share: the node
      layout from the node's offsets (`node_layout_of`), the bucketize's
      gather (`gather_rows`) and the push (`push_rows`);
    * `rail.cuh`: the device code of the rail's part of a call, which
      `rail_exchange` and the hooks share: a rank's count message
      (`count_message_of`) and the rail's layout from the rail's messages
      (`rail_layout_of`), with the one function that gives both
      `source_offsets` and `owner_offsets` (`source_offset`);
    * `planner.h`: the planner: the declarations of every step, and of the
      CUB calls in `cub.cu` (radix sorts and scans); `planner.cuh`: what the
      planner's kernels share (how they are launched, searches in sorted
      tables, the offsets of sorted runs, a CTA's scans). The bucketize is in
      `bucketize.cu` and the proxy in `proxy.cu` (with its node dedup's
      hash table, `Identities`, which takes index dedup's chained insert
      from `src/dedup_chain.cuh`, and the rounds of its chunks, cut by
      counted rows, `Rounds`); the node layout, the push, the count
      message and the rail layout as the hooks run them (`node_layout`,
      `push_to_node_keys`, `count_message`, `rail_layout`), the owner's KJT
      on one node (`one_node_owner`), the owner's tables, the requester and
      the put lists are in `planner.cu`;
  * `output_dist/`, the output dist:
    * `output_dist.h`, `output_dist.cu`: the out channel (`OutChannel`),
      which launches the out kernels, and their trace (`Trace`).
      `output_dist.cu` is the one file that compiles the out kernels, once
      for each GIN backend;
    * `args.cuh`: the call each out kernel takes, its plan and its tensors
      (`FwdArgs`, `BwdArgs`);
    * `out_kernels.cuh`: what both out kernels share (their timeline, the
      call's tables as a requester's warps read them, their task schedule
      and their protocol); `forward.cuh` and `backward.cuh`: the forward and
      backward kernels, with their GIN warps (the forward's relay and its
      grants, `grant_warp`; the backward's issuer and its look-ahead,
      `poll_lookahead` and `post_lookahead`).
* `setup.py` builds `src/hier_output_dist/` only against NCCL ≥ 2.30.7, and
  else defines `DEMB_NO_HIER_OUTPUT_DIST`. With `DYNAMICEMB_HIER_TRACE=1` it
  defines `DEMB_HIER_TRACE=1` (section 4.4).
* **Tests**: section 18. **Benchmark**:
  `benchmark/benchmark_distributed_embedding_collection.py`
  (`benchmark/README.md`, section 3), `benchmark/benchmark_utils.py`
  (`hier_needs`, `hier_capacities`) and `benchmark/nvlink_ceiling.cu`.
