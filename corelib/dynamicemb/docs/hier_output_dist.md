# Hierarchical input and output dist (NVLink inside a node, NCCL GIN between nodes)

Turn it on per table with `DynamicEmbTableOptions(output_dist="fused")` and
the capacities of its buffers (`output_dist_capacity`), or with `"auto"`
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
12. [Rounds and credits](#12-rounds-and-credits)
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
  options. A call that does not fit the capacities takes TorchRec's dists
  instead: that call alone, on every rank alike, with a warning that names
  the capacity to raise.
* **Only an op, on the caller's stream.** Each of its steps queues its work
  on the current stream and returns. The caller orders the steps on its
  streams and records the call's tensors on the streams that use them, as
  TorchRec's training loop and train pipelines do (section 16).
* **The input dist does all the planning, with one host sync.** Keys move
  over NVLink inside a node and over InfiniBand between nodes, sent by the
  GPU itself (NCCL GIN). Seven custom kernels move them and wait for peers
  (the *comm kernels*); the steps between them are PyTorch's tensor ops
  called from C++ and CUB sorts (the *planner*). At the end every owner has
  a ready KeyedJaggedTensor (KJT) of the keys it looks up, and every GPU has
  the per-row maps of the output dist. The CPU waits for the GPU once per
  call, in one blocking copy of a small status (`8 × (2 × num_features + 7)`
  bytes) to the host.
* **The lookup only looks up.** DynamicEmb's lookup writes its rows straight
  into a buffer that the output dist reads (zero copy).
* **The output dist only moves rows**, in one kernel per direction. It
  computes no index.
* **Rail-aligned path.** A key that a GPU asks of an owner on another node
  first goes over NVLink to the *proxy*: the GPU of its own node on the
  owner's rail. The proxy sends it over that rail's InfiniBand link to the
  owner. No traffic crosses rails.
* **Node dedup.** The proxy sends each (owner node, table, key) of its node
  once. So the owner looks the key up once per node, its row crosses
  InfiniBand once per node, and the node adds up its gradients before they
  cross back.
* **No static limit on the nodes, GPUs per node or features.** The kernels
  keep no table sized by them, and a GIN warp's lanes take any number of
  rail peers in turn.
* **Memory that does not grow with the number of GPUs.** Apart from small
  tables that grow linearly with the nodes and with the features (section
  15.4), no buffer is sized by the number of GPUs. The proxy's buffer for
  rows (`proxy_rows`) has a fixed cap. A call whose rows do not fit moves
  them in rounds.

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
* **Comm kernel**: one of the input dist's seven custom kernels, each of
  which moves data between GPUs or waits for peers (section 10.2).
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
  **Node-unique key**: one (owner node, table, key) as the proxy sends it
  on, however many occurrences its node has.
* **Lookup output**: the owner's rows for the keys it looks up, in its
  buffer `lookup_rows`. **Proxy rows** (`proxy_rows`): the proxy's buffer
  for the rows that pass between its node and the other nodes, in both
  directions.
* **Dedup** (deduplication): removing repeated keys. **Index dedup**:
  TorchRec's `use_index_dedup`. Each rank removes its own repeated keys
  before the input dist, and the rows are copied back to every original
  position after the output dist. **Node dedup**: the proxy removes the
  repeated keys of its whole node (section 9).
* **Skewed keys**: keys drawn so that a few of them are very common (a *hot*
  key appears in many samples).
* **Fall back**: take TorchRec's dists instead, for a whole sharding
  (section 6) or for one call (section 5.4).

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

DynamicEmbTableOptions(
    ...,
    output_dist="fused",              # "torchrec" (default), "fused" or "auto"
    output_dist_capacity=capacities,  # an int, or a dict by name (section 5)
    node_dedup=True,                  # across nodes: one key per (owner node, table, key)
)
```

It serves `ROW_WISE` sequence shardings (`EmbeddingCollection`). Pooled
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
  with at least 4 contexts.
* Ranks numbered node by node (the code assumes it; torchrun and Slurm's
  default task layout do this), and forming whole nodes.
* Fewer than 2^31 − 1024 (rank, feature) pairs: `world_size × num_features`
  (the planner numbers them with int32).
* Rows (embedding dim × bytes of the output dtype) a multiple of 16 bytes,
  in fp32, bf16 or fp16.
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
| `output_dist` | `"torchrec"` | `"torchrec"`: TorchRec's dists. `"fused"`: the hierarchical dist. It needs `output_dist_capacity` (without it, making the options raises `ValueError`), and a build without NCCL ≥ 2.30.7 raises `ImportError` when the model is sharded. `"auto"`: `"fused"` where the build has it and `output_dist_capacity` is set, else TorchRec's dists, with a warning. Any other value raises `ValueError`. |
| `output_dist_capacity` | `None` | The capacities of the op's buffers, in keys (section 5). An int sets `lookup_keys` and `remote_keys` to that many. A dict sets them by name: `lookup_keys`, and across nodes also `remote_keys` (one node uses `lookup_keys` alone). Each capacity is a positive int below 2^31 − 1024; other values or names raise `ValueError` when the options are made. A call that needs more of one runs on TorchRec's dists (section 5.4). TorchRec's dists ignore it. |
| `node_dedup` | `True` | Across nodes, the proxy sends one key per (owner node, table, key), and the node adds up its gradient rows before they cross back. `False` sends every occurrence (more IB traffic) and keeps TorchRec's gradient row per occurrence (section 18). Gradient clipping turns it off by itself. |

### 4.4 Environment variables

None is needed. All of them are read when the model is sharded (when the
sharding is built). A value the op refuses (a timeout of 0 or less, a
negative CTA count, or a `DYNAMICEMB_GIN_REDUCE_CTAS` below 1) fails on
every rank alike (section 6).

| variable | default | meaning |
|---|---|---|
| `DYNAMICEMB_GIN_PROXY_MB` | `512` | Cap of `proxy_rows`, in MiB (section 15.3). |
| `DYNAMICEMB_GIN_TIMEOUT_S` | the process group's NCCL timeout | How long a kernel waits for a peer before it prints what it waited for and traps (stops, which loses the CUDA context). |
| `DYNAMICEMB_GIN_CTAS` | `0` | CTAs of the output-dist kernels. `0`: 32 for the forward and 64 for the backward on one node, 32 for both across nodes (section 14). |
| `DYNAMICEMB_GIN_REDUCE_CTAS` | `8` | Across nodes: the backward's CTAs that add up the proxy's rows (1 or more; the kernel uses at most its CTAs minus one). |
| `LOCAL_WORLD_SIZE` | TorchRec's `get_local_size` | GPUs per node (`local_size`); torchrun sets it. |

The hierarchical dist always puts rows over IB in pieces of at most 2 MiB.

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
own (the teardown's `exit_kernel`), for at most `min(timeout, 30 s)`. If a
peer keeps its model, or the teardown meets a CUDA error (a kernel that
trapped, say), the rank prints a message and keeps its window until the
process exits. `quiesce(model)` calls each op's `quiesce()`, in the same order
on every rank: the op syncs its device and meets the other ranks over its own
communicator. Dropping the model then lets go of each op's window at once. A
call of the model after `quiesce` undoes it. The views of `lookup_rows` share
the window's memory with the op: a view that outlives the sharding (a lookup
output kept by the caller) keeps that memory until the view is dropped.

## 5. Capacities

The op makes its window once, when it is built (section 7.2), at the
capacities of `output_dist_capacity`, and keeps its size for its life.
Each capacity counts keys. On one node only `lookup_keys` is used.

### 5.1 What each capacity bounds

The words used in this section and in section 15:

* `num_keys`: the number of keys a rank sends in one call (after index dedup
  when it is on);
* `uniq_share`: the number of node-unique keys per occurrence at a proxy.
  It is about 0.935 with the benchmark's uniform keys and 0.27 with its
  power-law keys, and 1 when no two GPUs of a node share keys.

| capacity | what it bounds | what it sizes | with uniform keys, about |
|---|---|---|---|
| `lookup_keys` | the keys an owner looks up: those of its own node and, across nodes, one per node-unique key of each other node | `lookup_rows` (the lookup output), the own part of `node_keys` and, across nodes, `rail_keys`; per call, the owner's KJT | `num_keys × (1 + (num_nodes − 1) × uniq_share) / num_nodes`, at most `num_keys` |
| `remote_keys` | across nodes: the occurrences a proxy deduplicates, the keys its node sends the owners of its rail on the other nodes | the remote part of `node_keys`, `remote_cells`, `remote_map` and `key_stage`; `proxy_rows`, up to its cap; per call, the proxy's lists | `num_keys × (num_nodes − 1) / num_nodes` |

Two more sizes follow from these (section 15.3): `proxy_rows` holds
`remote_keys` rows, up to its cap, and the round tables hold the rounds
that `remote_keys` rows take in it (`rounds_cap`). A call whose rows do not
fit `proxy_rows` moves them in rounds (section 12.1). The op's
`capacities()` gives them all.

Say a hot key makes up a share `hot_share` of every rank's keys. It adds
about `local_size × hot_share × num_keys` occurrences on its owner's rail:
to its owner's `lookup_keys` (its own node's occurrences), and to
`remote_keys` at its proxy on every other node. After node dedup each other
node adds one key to its owner's `lookup_keys`. That holds whatever the
number of nodes. TorchRec's hot owner receives about
`(1 + (world_size − 1) × hot_share) × num_keys` rows.

### 5.2 How to choose them

Count the needs on batches like the ones the model will see, then add
headroom. Two helpers in `benchmark/benchmark_utils.py` do this (it is not
part of the installed package: put `benchmark/` on `sys.path`, as the tests
do):

* `hier_needs(batches, tables, owner_of, local_world_size, index_dedup=False,
  node_dedup=True, group=None)` counts what the op needs of each capacity
  for some calls, as the op counts it: the most any rank needs in any call.
  It is collective: every rank of the sharding's process group calls it with
  its own batches. `tables` maps each feature of the sharding to its table,
  and `owner_of(keys, feature)` gives the rank that owns each key.
* `hier_capacities(needs, headroom=1.25, room=4096)` makes the dict for
  `output_dist_capacity`: each capacity `headroom` times its need, and at
  least `room` entries more (small calls).

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
    index_dedup=use_index_dedup,
)
capacities = hier_capacities(needs, headroom=1.25)
```

* **An int** sets both capacities alike. It is the simplest choice: it must
  hold the larger need. Across nodes it then gives the other capacity more
  room than it needs, which costs memory (section 5.3).
* **Skewed keys and changing batches.** The needs grow with the hot keys'
  share and with the longest batches. Count them on enough batches, or add
  more headroom. A call that does not fit still runs, on TorchRec's dists,
  and its warning gives what it needed (section 5.4).
* **The benchmark** counts the needs of its own batches before it builds
  the model (`--capacity measure`, the default) and takes
  `--capacity_headroom` (1.25) times them, and at least 4096 entries more
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
| `remote_keys` | 8 (`node_keys`; 12 with weights) + 4 (`remote_cells`) + 12 (`remote_map`) + `record_bytes` (`key_stage`), and `row_bytes` in `proxy_rows`, up to its cap | per call, 12 (the proxy's reducer list) and about 2 (the requesters' heavy groups) |

The input dist's working memory comes on top, in the caching allocator's
pool of the stream it runs on: that of its largest step (section 15.2).

For the benchmark's configuration (540,672 keys per rank, rows of 512
bytes), capacities of 1.25 times the needs take a window of 0.35 GB per rank
on one node and 0.53–0.62 GB on 2 to 4 nodes (section 15.4).

### 5.4 A call that does not fit

The call's status holds its needs: the most any rank needs of each
capacity, the same on every rank (section 10.5). When the call needs more of
some capacity than the op has:

* **That call alone takes TorchRec's dists, on every rank.** Its input dist
  runs TorchRec's all-to-alls on the keys that part 1 took (DynamicEmb's
  bucketize first, with the host syncs TorchRec's input dist makes). Its
  lookup and output dist are TorchRec's too, and the lookup does not write
  `lookup_rows`. Under index dedup, TorchRec copies the rows back to their
  original positions as it does on its own path. The calls before and after
  take the op.
* **A warning names each capacity the call overflowed**, how large it is,
  and what the call needed of it. For example:
  `output_dist="fused": a call does not fit the capacities, so it runs on
  TorchRec's dists (1 calls so far); raise output_dist_capacity's
  lookup_keys from 541000 to at least 541801`. It comes at the first such
  call, and then at the 2nd, 4th, 8th and every further power of two.
* **A counter.** Each sharding counts such calls (`HierSharding.fallbacks`;
  the benchmark sums them over a model and reports them).
* The first wait still runs every step of such a call, and its comm kernels
  send their messages, so the sequence numbers and running totals stay
  valid for the next call. Nothing is written past a capacity. When a GPU
  would get more of its node's keys than a capacity holds (its own part
  above `lookup_keys`, or across nodes its remote part above `remote_keys`),
  no GPU of that node pushes its keys. And an owner whose keys pass
  `lookup_keys` gets no key records. Telling that a call does not fit costs
  nothing more: the needs come with the call's one host sync.
* When a node pushes no keys, its proxies deduplicate none, so the call's
  `lookup_keys` need leaves out the keys that node sends to other nodes. A
  later call may then name `lookup_keys` too.
* Such a call costs what TorchRec's dists cost, and the op's window stays
  allocated meanwhile. Calls that do not fit are meant to be rare: raise the
  capacity the warning names.

## 6. When a sharding takes TorchRec's dists

A sharding that cannot take the hierarchical dist uses TorchRec's
`KJTAllToAll` and `RwSequenceEmbeddingDist`. The choice is the same on every
rank: it is made from the sharding alone, collectively, or per call from
state that every rank shares.

| condition | decided | result |
|---|---|---|
| `output_dist="torchrec"` (the default) | when sharding | TorchRec's dists |
| an unknown `output_dist`; an `output_dist_capacity` that is not a positive int below 2^31 − 1024, or a dict that sets `lookup_keys` (and maybe `remote_keys`) to such ints; `"fused"` without `output_dist_capacity` | when the options are made | `ValueError` |
| tables of one sharding that disagree on `output_dist`, `output_dist_capacity` or `node_dedup` | when sharding | `ValueError` |
| a lookup group whose output dtype is not the sharding's | when sharding | `ValueError` |
| a build without NCCL ≥ 2.30.7 | when sharding | `"fused"`: `ImportError`; `"auto"`: a warning, TorchRec's dists |
| `"auto"` without `output_dist_capacity` | when sharding | a warning, TorchRec's dists |
| one rank in the sharding; a table of the sharding that is not a DynamicEmb table; feature processors | when sharding | a warning, TorchRec's dists |
| ranks that would build the op from different options, capacities or environment values | first `input_dist()`, before the op is built | `ValueError` on every rank, naming the first rank and value that differ |
| ranks that are not whole nodes; `world_size × num_features` of 2^31 − 1024 or more; rows not a multiple of 16 bytes | first `input_dist()` | a warning, TorchRec's dists |
| a configuration the op cannot take: across nodes, capacities without `remote_keys`; a proxy cap below the rows a round needs (section 15.3); a `remote_keys` that would take more than 65536 rounds of `proxy_rows`, or whose push key would pass 63 bits (section 10.4); rows that are not fp32, bf16 or fp16; an environment value it refuses (section 4.4) | first `input_dist()`, before the op is built | `"fused"`: `ValueError` on every rank; `"auto"`: a warning, TorchRec's dists |
| On some rank: the NCCL LSA team is not the whole node. Or, across nodes, GIN does not connect the rails, has fewer than 4 contexts, or its signals do not start at 0. Or the window cannot be allocated. | first `input_dist()`, collectively | a warning that names each rank and why, TorchRec's dists on every rank |
| an eval forward | per call | TorchRec's dists for that call |
| a call that does not fit the capacities | per call, at its second wait | TorchRec's dists for that call, a warning (section 5.4) |

## 7. The operator

### 7.1 Its interface

The op is a C++ class with a small interface. Python reaches it through
pybind11, as `dynamicemb_extensions.HierOutputDist`, what it is built from
(`HierOutputDistConfig`), the data the steps of a call hand each other
(`HierPartOne`, `HierPending`, `HierInDistResult` with its plans
`HierFwdPlan` and `HierBwdPlan`), `hier_output_dist_unique_id()` and
`hier_output_dist_traced()`. The op runs every step of a call, and each
step queues its work on the caller's current stream (section 7.3). Each step
returns plain data, which the caller hands to the next step: nothing else
keeps a call's state.

```cpp
namespace dyn_emb::hier_output_dist {

class HierOutputDist {  // one per sharding
 public:
  static std::string unique_id();  // the communicator's id, made on one rank
  // Why the op does not apply to such a sharding, or "" (section 6).
  static std::string unsupported(int world_size, int local_size, int num_features,
                                 int64_t row_bytes);
  // Why the op refuses a configuration, or "". It builds nothing (section 7.2).
  static std::string invalid(const Config &config);
  HierOutputDist(const std::string &unique_id, Config config);  // collective (section 7.2)
  ~HierOutputDist();  // local (section 4.6)

  PartOne part_one(PlanInput input);          // part 1 of a call
  Pending start(const PartOne &part);         // the first wait
  InDistResult wait(const Pending &pending);  // the second wait, the call's one host sync
  at::Tensor lookup_rows(int64_t num_recv);   // the rows the lookup writes, [num_recv, dim]
  at::Tensor forward(const FwdPlan &plan, const at::Tensor &lookup_out);  // one kernel
  at::Tensor backward(const BwdPlan &plan, const at::Tensor &grad,        // one kernel
                      bool divide);
  void quiesce();  // collective (section 4.6)
  // diagnostics: capacities(), memory(), trace(), trace_launches(), gin_railed()
};

struct InDistResult {  // what the input dist of a call gives, after its host sync
  bool fits;                             // false: the call does not fit (section 5.4)
  std::map<std::string, int64_t> needs;  // also overflows, sent, input_length_per_key
  // when it fits: the owner's KJT (values, weights, lengths, offsets, num_sources,
  // length_per_key, num_recv), and the output dist's plans
  FwdPlan fwd_plan;  // what the forward kernel reads of the call
  BwdPlan bwd_plan;  // what the backward kernel reads of it
};

}  // namespace dyn_emb::hier_output_dist
```

One training step in Python, as the sharding runs it (section 7.4):

```python
part = op.part_one(values, offsets, lengths, batch_size, weights,
                   input_offsets, input_batch_size, reverse_indices)  # input_dist()
pending = op.start(part)                 # the first wait
result = op.wait(pending)                # the second wait; not result.fits: TorchRec's dists
kjt = owner_kjt(result)                  # result.values, result.lengths, ...
rows = op.lookup_rows(result.num_recv)   # the lookup writes its rows here
out = op.forward(result.fwd_plan, rows)
...                                      # the dense forward and backward
lookup_grad = op.backward(result.bwd_plan, grad_out, divide)
```

The forward's plan (`FwdPlan`) and the backward's (`BwdPlan`) hold only what
the input dist made: the requester's maps and the call's tables (section
11.1). Each table both kernels read is the same tensor in both plans. The
output dist takes the lookup's output with the forward's plan, and the
gradient with the backward's.

`Config` holds everything the op needs, read once: the ranks, the table and
the bucketize's rule of each feature, the rows and their weights,
`node_dedup`, the capacities (with the proxy cap in bytes) and the knobs of
section 4.4.

Of the diagnostics, `capacities()` gives the capacities in place (section
5.1), and `memory()` the bytes of the op's window, in two maps:
`memory()["windows"]["window"]` is this rank's copy, in whole allocation
granules, and `memory()["buffers"]` gives what each buffer takes in it, by
name (`ctrl` for the ctrl section; sections 7.5 and 15.2).

Python reads the options and the environment when the sharding is built,
and hands them over when it builds the op. The benchmark reads a call's
tables from its plans. The tests run the planner's steps on plain tensors
through functions of `testing.cu`, which also has a hook that makes the
next construction fail, and the GPU clock by which the benchmark aligns its
timelines.

### 7.2 Construction

The op is built at the sharding's first `input_dist()`, in any mode, at the
same point on every rank:

1. **The ranks agree on what they build it from.** Each rank makes its
   `Config` and asks `invalid()` whether the op takes it. Then one
   all-gather over the sharding's process group gives every rank what every
   rank would build the op from (all but its rank), and a unique id for the
   op's own NCCL communicator, made on rank 0. If some rank's configuration
   differs from rank 0's, every rank raises the same `ValueError`, which
   names that rank and the first value that differs. Else every rank has
   the same configuration, so every rank decides alike. If the op does not
   apply (`unsupported`, section 6), every rank warns and takes TorchRec's
   dists. If the op refuses the configuration, `"fused"` raises `ValueError`
   on every rank and `"auto"` warns and takes TorchRec's dists.
2. Else every rank builds the op with that id. The constructor makes the
   communicator and its device communicator: LSA inside the node and,
   across nodes, GIN on the rails with 4 contexts.
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

As in TorchRec, `input_dist()` returns an awaitable. Waiting on it (the
*first wait*) queues the rest of the input dist and returns a second
awaitable. Waiting on that one (the *second wait*) is the call's one host
sync, and returns the owner's KJT. Every step queues its work on the
current stream and returns:

```
the caller's stream
-------------------
input_dist(): part 1 (op.part_one()): the keys sorted by owner
first wait (op.start(part)): the steps below (comm kernels and planner kernels),
                             the last of which writes the call's status
second wait (op.wait(pending)): a blocking copy of the status to the host, the
                                call's one host sync; then the owner's KJT and the
                                plans
lookup: op.lookup_rows(num_recv), then writes its rows there (zero copy)
forward kernel (op.forward(fwd_plan, rows)): the rows
... dense forward and backward ...
backward kernel (op.backward(bwd_plan, grad, divide)): gradient rows into the owners'
                 lookup_rows, handed to the lookup's backward as lookup_grad
```

The op orders nothing across streams. The caller runs each step on the
stream of the step before it, or on a stream it ordered after that one. In
a plain loop every step runs on the default stream. Under TorchRec's train
pipelines the input dist (part 1 and both waits) runs on the pipeline's
data_dist stream (section 16).

The first wait queues these steps, in this order, on every rank. A *comm*
step is a comm kernel (section 10.2); the others are the planner's kernels,
with CUB's sorts and scans.

| step | kind | what it does | section |
|---|---|---|---|
| count vector | kernel | this rank's keys for each GPU of its node | 10.3 |
| `node_count_exchange` | comm | the node barrier, and the node's count vectors summed as this rank needs them | 10.2 |
| node layout | kernel | where each rank of the node pushes its keys; the node's needs | 10.3 |
| `node_key_push` | comm | this rank's keys into its targets' `node_keys` over NVLink | 10.2 |
| `node_done` | comm | tells its targets; waits until every requester of the node told it | 10.2 |
| owner's KJT (one node) | kernels | a copy of the own part of `node_keys`, and the requester's map | 10.3 |
| node dedup | kernels, CUB | the proxy's node-unique keys, rows, rounds and slots | 9, 10.4 |
| push order | kernels, CUB | its requesters' push orders and push tables, and its reducer list | 10.4 |
| count message | kernel | what this rank's rail peers must know of it | 10.4 |
| `rail_count_exchange` | comm | its count message to each rail peer, and theirs | 10.2 |
| rail layout | kernel | where each proxy's key records land, this owner's KJT layout, and where each proxy's cells start in it | 10.4 |
| `rail_key_exchange` | comm | its key records to its rail owners and its cell starts to its rail proxies, and theirs | 10.2 |
| `node_status_exchange` | comm | the needs and cell starts of the node; the call's needs | 10.2 |
| `peer_gather`, twice | comm (and a kernel) | the node's proxy headers, then the requester's gather list and its rows' `remote_map` words | 10.2, 10.4 |
| requester tables | kernels, CUB | its maps, push order, heavy groups and task starts | 10.4 |
| owner tables | kernels | its KJT's keys, and the round pieces its forward put list sends | 10.4 |
| put lists | kernels, CUB | the owner's forward and the proxy's backward put lists, and the IB chunks | 10.4 |
| status | kernel | the call's status, which the second wait copies to the host | 10.5 |

One node runs the first five steps, the owner's KJT and the status; across
nodes every step runs but the owner's KJT.

The second wait queues nothing after its sync: the owner's KJT is a view of
the first `num_recv` of its values.

Part 1 is local: no sync, no collective, no sequence number. So its result
may be dropped before the first wait (TorchRec's prefetch pipeline does
this): nothing has moved yet. Part 1 also does every check of the call's
inputs. Each step's data lives as long as the caller keeps it: the first
wait drops part 1's once the rest of the call is queued, and the second wait
the pending call's once it has the result. The sharding context keeps the
result until the output dist's forward takes the plans: the forward's plan
goes once its kernel is queued, and autograd keeps the backward's (sections
7.4 and 11.1).

### 7.4 Around the operator: the Python sharding

* **When the sharding is built** (no collective): Python reads the options,
  the capacities and the environment, and sets each DynamicEmb table's
  `output_alloc` to the sharding. In each call on the op, the sharding then
  hands each lookup group the next rows of the call's rows of `lookup_rows`
  (section 11.2).
* **The input dist.** On a call that takes the op, `input_dist()` runs part
  1 and returns an awaitable. Its wait is the first wait; waiting on it
  again returns the same second awaitable, and dropping it changes nothing
  (TorchRec's prefetch pipeline does both). The second wait returns the
  owner's KJT, built from views of the result's tensors with no copy. It
  also sets the input KJT's per-feature totals from the status, so
  TorchRec's `construct_jagged_tensors` needs no host sync. The sharding
  context carries the result (`hier_result`). Its `record_stream`, which
  TorchRec's pipelines call, also records the result's tensors (through the
  result's own `record_stream`).
* **A call that does not fit** gets TorchRec's input dist at its second
  wait (section 5.4): DynamicEmb's bucketize of the first `sent` keys, in
  their own dtype (part 1 placed them by the same rules), then TorchRec's
  `KJTAllToAll`. That fills the sharding context for TorchRec's output
  dist, and the context carries no result.
* **Index dedup.** For a call that takes the op, the output dist itself
  writes each deduplicated row to all its original positions, and sums
  their gradients. So TorchRec's own expansion is skipped for that call
  only. A call on TorchRec's dists keeps it.
* **The lookup.** For the owner's KJT of a call on the op, the sharding
  takes the call's rows from `op.lookup_rows(num_recv)`, and each DynamicEmb
  lookup group writes its rows straight into the next rows of them. The
  lookup returns those rows, which cover all groups, with no copy. Any other
  KJT (a call on TorchRec's dists) takes TorchRec's lookup path.
* **The output dist.** It takes the call's plans from the sharding context.
  An autograd function runs the forward kernel with the forward's plan
  (`op.forward`), and keeps the backward's plan as saved tensors (its
  tensors, and its counts beside them), so they live as long as autograd
  keeps the graph. A second backward (`retain_graph`) works, and they go
  after the last backward. Its backward makes the plan again from them and
  runs the backward kernel (`op.backward`), which returns `lookup_grad`,
  the gradient of the lookup output, a view of `lookup_rows` valid until
  the next lookup. A call without a result takes TorchRec's
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
  contexts. A call's tensors are in the data its steps return (section 7.1).
* The **in channel** is the first wait's steps (section 7.3), on the current
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
| rows | `lookup_keys` rows; across nodes also `proxy_rows` | `lookup_rows` (the lookup output, then its gradient). Across nodes also `proxy_rows` and `push_flags`. |
| ctrl | its control words (section 15.2: 80 KB at 32 nodes, 8 GPUs per node and 64 features), most of them the count messages | one buffer, `ctrl`: the control words of both channels (sequence numbers, running totals, the credits sent, the count vectors and count messages, each GIN warp's place with each rail peer) |
| keys | from `lookup_keys`, and across nodes `remote_keys` | `node_keys` and, with weights, `node_weights`. Across nodes also `remote_cells`, `remote_map`, `proxy_header`, `key_stage`, `rail_keys`, `cell_stage` and `rail_cells`. |

`node_keys` holds the keys a GPU's node pushes to it: its *own part*, which
it looks up as owner, and across nodes its *remote part*, which it
deduplicates as proxy (section 10.3). The buffers that only an op across
nodes has:

* `remote_cells`: the cell (owner node, feature) of each occurrence in the
  remote part, which the push writes beside its key;
* `remote_map` (`RemoteMap`): three int32 arrays with an entry for each
  such occurrence: its key's row in `proxy_rows` (with a bit for a heavy
  key), its own slot in `proxy_rows`, and its place in its requester's push
  order;
* `proxy_header` (`ProxyHeader`): the proxy's header for its node's
  requesters, in int32 words: its rounds, its round table (section 12.1)
  and each requester's push tables;
* `key_stage`: per other owner node the slice one key put sends: its round
  pieces, then its node-unique keys' records;
* `rail_keys`: the slices the rail proxies put into this owner, one per
  source node, in node order;
* `cell_stage`: as owner, where the cells of each rail proxy's node start in
  its KJT, per feature, which it puts to that proxy;
* `rail_cells`: as proxy, where its node's cells start in each rail owner's
  KJT, as the owners put them;
* `push_flags`: the flags of the rows the requesters push into `proxy_rows`,
  in two halves, one per round parity (section 12.1).

The window is made when the op is built, from `ncclMemAlloc` and
`ncclCommWindowRegister`. It has the same size on every rank, is rounded up
to whole allocation granules, and keeps its size for the op's life. Each
buffer, the ctrl section too, starts at a 4 KiB boundary (`kBufferAlign`).
Before the registration each rank zeroes its ctrl section (sequence numbers
start at 0) and `push_flags` (their tags start at 0); every other buffer is
written before it is read. Each view of `lookup_rows` holds this rank's copy:
its memory goes once the op has let go of it and every view is gone (section
4.6).

**The layouts.** The host and the kernels lay the window out with the same
code (`window.h`). `WindowLayout` builds the three sections,
`RowsLayout`, `CtrlLayout` and `KeysLayout`, from the op's sizes (`Sizes`:
its ranks, features and capacities, and what follows from them) over a
base pointer, each from the end of the one before. Each has an accessor per
buffer or control word, which returns its address in the copy at that base
(`keys.node_keys()`, `rows.proxy_row(slot)`,
`ctrl.done(requester, parity)`), so no other code computes where a buffer
or a word is. The host lays out its copy to size the window and to view its
buffers as tensors. A kernel's argument carries the sizes, the window and
the device communicator (`ChannelStatic`), and the kernel lays out this
rank's copy as the node's GPUs map it (`NodeWindow`). A place has the same
offset in every rank's copy: `NodeWindow::peer(pointer, peer)` gives a node
peer's copy of the place a pointer names in this rank's copy, and a GIN put
to a rail peer takes pointers into this rank's copy for both of its ends
(`rail_put`, `RowPut`, `RailMessage`).

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
| 1 | backward rows, proxy → owner | `bwd_rows[proxy node]` at the owner; with rounds, the counters `sent[half]` at the proxy |
| 2 | the input dist: count messages, key puts and cell starts | `count[node]`, `keys[node]` |
| 3 | credits and teardown | `grant[node]`, `bwd_ready[owner node]`, `closing[node]` |

Why `NCCL_CROSS_NIC=0`: section 4.5.

## 9. Node dedup

A proxy holds, in the remote part of its `node_keys`, every occurrence that
its node's requesters ask of the owners of its rail on the other nodes, and
in `remote_cells` the (owner node, feature) of each. Its planner numbers
them with stable radix sorts (section 10.4):

* **Identity: (owner node, table, key).** The owner node is part of it
  because the `continuous` rule (`dist_type`) rewrites a key for its owner,
  so equal stored keys of two owners are different keys. Features of one
  table share keys. A sort by key, then one by (owner node, table), brings
  each identity's occurrences together, in the order of the remote part.
* **Representative feature.** A node-unique key's feature is the table's
  smallest feature under which the node asked for the key; a sort of the
  (identity, feature) pairs finds it. The owner looks the key up under it.
* **Numbering.** The node-unique keys are numbered (`uniq`) per owner node,
  feature by feature (the representative's), and in key order (signed
  `int64`) inside each (owner node, feature) cell.
* **Rows.** In `proxy_rows` a key of at most 64 occurrences takes one row
  per occurrence. Its first occurrence holds the key's row: the owner's put
  fills it in the forward, and in the backward that occurrence's gradient
  row and then the key's sum go there. Each other occurrence holds a
  duplicate's slot for its gradient row.
* **Heavy keys.** A key with more than 64 occurrences on the node is
  *heavy*. In `proxy_rows` it takes one row, plus one fp32 partial row per
  requester (each partial takes `ceil(4 × dim / row_bytes)` rows of
  `proxy_rows`). In the backward each requester first adds up, in fp32, the
  gradients of its own occurrences of the key (its *heavy group*). It pushes
  them as one partial. The proxy then adds the at most `local_size` partials
  in requester order. In the forward, the occurrences of one heavy key in a
  32-row task read its row once.
* **Weights** ride along only when TorchRec's owner would see frequency
  counts. The keys' own weights (index dedup with LFU scores, which count
  how often each key is seen, or with an admission strategy) are added up
  per node-unique key, as whole numbers. With LFU tables across nodes and no
  such weights, the proxy sends each node-unique key's number of
  occurrences. The owner hands them to the lookup groups of LFU tables only.
* **`node_dedup=False`** (or gradient clipping): every occurrence is a
  node-unique key of its own.

**The owner's KJT** has `num_sources = local_size + num_nodes − 1` columns
per feature. Cell (feature, source) with `source < local_size` holds the
keys of that feature that same-node requester `source` sent, in its order.
Cell (feature, `local_size + nth`) holds the node-unique keys of the `nth`
other node whose representative feature is that feature, in `uniq` order.
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

Part 1 and the first wait's steps run on the caller's stream, in the order of
section 7.3. Seven of them are the comm kernels (section 10.2). The planner
does the rest (`src/hier_output_dist/input_dist/planner.h`): kernels of its
own and CUB's radix sorts and scans, on the current stream. Each step is one
or a few ordinary kernels. A kernel over items (keys, rows, occurrences)
runs a thread per item; a kernel over a step's small tables (sized by the
nodes, ranks, features and rounds) runs on one CTA, which loops over them.
No kernel waits for another CTA: a step that needs every item done goes on
in its next kernel. The tensors' sizes are what the host knows from the
call's keys, the capacities and the topology. The counts that only the
device knows (valid keys, node-unique keys, rounds) stay on the device, and
no tensor's size depends on them; a kernel reads such a count to bound its
own loop. So the host reads no device value before the second wait, and no
step syncs. Entries past the valid ones hold values that no one reads, and
an index taken from device data is clamped first: a call that does not fit
runs every step on what arrived and stays in bounds.

With the benchmark's configuration (8 features, one owner rule), part 1
queues 11 kernels and copies per call (39 with index dedup, DynamicEmb's own
dedup included), and the first wait 9 on one node and about 120 across
nodes, CUB's included.

### 10.1 Part 1 (requester, caller's stream, local)

1. **Index dedup** (when on): DynamicEmb's `segmented_unique`, sized on the
   GPU. The deduplicated KJT keeps its full length, and its lengths say how
   many keys are valid. So `num_uniques.item()` (a host sync) is not needed.
   `segmented_unique` takes `int64` keys, so `int32` keys are deduplicated
   as `int64`, and the deduplicated keys go back to `int32`: part 1 places
   each key by the rules of its own width (see below). (TorchRec's path in
   DynamicEmb takes no `int32` keys under index dedup: `segmented_unique`
   raises.)
2. **The partition.** Each key's feature comes from the KJT's offsets, and
   its owner and the key its owner stores from the bucketize's rules
   (`dist_type`, and `continuous`'s key rewrite), in the arithmetic of the
   keys' own width (see below). Only the rules of some feature run. Then one
   stable radix sort by cell (owner, feature) puts the keys in
   `[owner][feature][sample][position]` order, as `int64`, with `inv` (each
   key's row in the KJT); the keys past the valid ones (index dedup) go
   last. Searches of the sorted cells give where each cell and each owner's
   block start (`cell_start`, `block_start`). Part 1 also gives the input
   KJT's per-feature totals. Its outputs take 16 bytes per key (20 with
   weights), and nothing of size `world_size × samples`.
3. **The fan-out table** (index dedup): for each deduplicated row, its
   original rows in increasing order, from a stable radix sort of the
   reverse indices. A row with more than 64 original rows is a *heavy row*:
   it gets a slot and consecutive sub-tasks of 64 original rows, numbered
   by cumulative sums in row order.

**The owner rules by key width.** On TorchRec's path DynamicEmb bucketizes
the KJT as the caller gave it. `bucketize_kjt_before_all2all` converts the
block sizes to the keys' dtype, and the bucketize reads each key as an
unsigned number of the keys' width (a negative `int32` key as
`2^32 + key`). Part 1 computes the same way, on the keys' bits in `int64`
tensors, whose products wrap as the bucketize's do. So on both paths a key
goes to the same owner:

| `dist_type` | `int64` keys | `int32` keys |
|---|---|---|
| `roundrobin` | key mod `world_size` | the same, in 32 bits |
| `hash_roundrobin` | the key's 64-bit hash mod `world_size` | the low 32 bits of the hash of the key (zero-extended), mod `world_size` |
| `continuous` | key / block size below block size × `world_size`, else key mod `world_size` | the same in 32 bits: the block size's low 32 bits, and a product with `world_size` that wraps at 32 bits, as the bucketize's `int` product does |

`continuous` also rewrites the key: to its offset in its block, or past the
blocks to key / `world_size`. The other rules keep it. The owner stores an
`int32` key as the `int32` the bucketize writes, widened with its sign to
`int64`: the 64-bit key DynamicEmb's lookup makes of an `int32` key on
TorchRec's path. So a model may take either path call by call (eval
forwards, calls that do not fit) and finds its rows. Checkpoint loading and
delta replay (`dynamicemb/key_ownership.py`) follow another rule for some
`int32` keys (section 19).

### 10.2 The comm kernels

They are in `input_dist/comm_kernels.cu`. A comm kernel that waits runs on
one CTA and takes the call's sequence number `in_seq`, the same on every
rank. It waits only for the same kernel (the same `in_seq`) of its node or
rail peers. Its threads take the peers in turn (thread `t` the peers `t`,
`t + 32`, ... in a warp), so it serves any number of them. Section 13 gives
their memory order and why no wait hangs.

* **`node_count_exchange`** (1 CTA): copies this rank's count vector
  (section 10.3) into its `count_vector` words of the call, in the ctrl
  section (a slot per `in_seq` parity). It sets its `in_started` word to
  `in_seq`, and waits until every node peer has (the node barrier of the
  call). Then it reads the peers' count vectors and keeps only what this
  rank needs (section 10.3): their sum, the sum of those of the ranks before
  it, and each requester's counts for this rank. Passing the barrier also
  means that every peer finished its previous call's in channel, whose
  steps read what this call writes into their copies of the window.
* **`node_key_push`** (threads stride over the keys, at most two CTAs per
  SM): stores each of this rank's valid keys (and its weight) into its
  target's `node_keys` over NVLink, at the place the node layout gives.
  Across nodes a key for an owner on another node also writes its cell
  (owner node × `num_features` + feature) into the target's
  `remote_cells`. Consecutive keys of a cell go to consecutive places, so
  the stores coalesce.
* **`node_done`** (1 warp): tells each node peer that this rank's keys are
  there (its flag in the peer's `in_written` words), then waits until every
  peer told it.
* **`rail_count_exchange`** (1 CTA, across nodes): copies this rank's
  count messages (section 10.4) into its `count_msg_out` words: the common
  part, then the part for each node. Its own message goes into its
  `count_msg` slot of its own node: the slots go by `in_seq` parity and
  source node. Then it puts each rail peer its message into the same slot
  in the peer's copy, in two puts: the common part, then the part for the
  peer's node, which carries a GIN signal (context 2). A signal is seen only
  after the earlier puts of its context to that peer, so it covers both.
  Then it waits until the message of every rail peer has landed, and reads
  them from its `count_msg` slots of the call.
* **`rail_key_exchange`** (1 CTA, across nodes): for each rail peer, puts
  this proxy's `key_stage` slice for the owner there into the owner's
  `rail_keys` (none when the owner's keys would pass `lookup_keys`), then,
  as owner, the peer's node's row of `cell_stage` (where that node's cells
  start in this owner's KJT) into this node's row of the peer's
  `rail_cells`, with a GIN signal. Then it waits for the key stream of
  every other node. A proxy puts only once it has the owner's count message
  of the call, which the owner sends after its previous call ended.
* **`node_status_exchange`** (1 CTA, across nodes): writes this rank's rail
  needs and where each requester of its node starts in its KJT into its
  `needs` and `cell_start` words, sets its `in_done` word to `in_seq`, and
  waits until every node peer has. Then it reads every peer's needs, and
  where each peer's KJT holds this requester's keys.
* **`peer_gather`** (threads stride over the words, at most two CTAs per SM;
  across nodes, twice per call): reads words of node peers' proxy headers
  (`ProxyHeader`: one word per item, by its word number) or `remote_map`
  (`RemoteMap`: per item, an occurrence's entry in each of the three
  arrays). The peers wrote them before the `in_done` that
  `node_status_exchange` waited for.

### 10.3 The node layout, and the owner's KJT on one node

Each rank counts its keys for its node (its count vector, `CountVector`:
`local_size × (num_features + 1)` int64 words): first per GPU of the node,
those for its remote part, then per owner of the node and feature, those
for its own part. `node_count_exchange` gives each rank three sums of
the node's vectors: over the node, over the ranks before it, and each
requester's counts for it. That is all its layout needs, so no rank keeps
the whole node's counts. The layout:

* the *own part* of an owner's `node_keys`, at `[0, lookup_keys)`, holds the
  keys of its node for it in its KJT's order: feature after feature, then
  requester after requester, each in the requester's order;
* across nodes the *remote part* of a proxy's `node_keys`, at
  `[lookup_keys, lookup_keys + remote_keys)`, holds the keys of its node for
  the owners of its rail on the other nodes. They are requester after
  requester, then owner node, then feature, then position;
* the node's needs: its largest own part and largest remote part. When one
  is above its capacity, no rank of the node pushes (section 5.4).

A requester's cell (feature, requester) in an owner's own part starts after
the node's keys of the earlier features for that owner, and after those of
the earlier requesters in that feature: both sums are in what the exchange
gives.

On one node the own part is the owner's KJT, so the owner takes one copy of
`lookup_keys` entries of it (the second wait narrows it to the keys that
arrived). The copy is needed: the next call's peers rewrite `node_keys`
after its barrier, which does not wait for this call's lookup. Its lengths
and offsets come from the counts, and a requester's map to the owners' rows
is where it pushed each key.

### 10.4 Across nodes: the proxy, the rail, the requester and the owner

**The proxy** numbers its node-unique keys (section 9), cuts its rows into
rounds (section 12.1) and gives each occurrence its slot. It writes the
three buffers of its window that its node's requesters and its rail's
owners read:

* `remote_map` (`RemoteMap`): for each occurrence, its entry in three int32
  arrays: its key's row (bit 31: a heavy key), its slot, and its place in
  its requester's push order;
* `key_stage`: per other owner node a slice: its round pieces (16 bytes
  each), then its node-unique keys' records in `uniq` order. A record is the
  key (`int64`) and, with weights, the weight's float bits and 4 zero bytes;
* `proxy_header` (`ProxyHeader`, int32 words): its rounds at word 0; from
  word 4 its round table, a `RoundPiece` per round and owner node; then its
  push tables, per requester `rounds_cap + 1` entries (`ReqEntry`), with
  positions relative to that requester's push order and groups relative to
  its first group.

A requester's push order at the proxy comes from one stable radix sort of
the push key. Its fields, from the highest, are the requester, the round and
whether the key is heavy, then the IB chunk and owner node of a key with a
row each (chunk × `num_nodes` + owner node), or the index of a heavy key.
Each field is as wide as its largest value needs, so the key's width follows
from `local_size`, `num_nodes`, `remote_keys` and the rounds; the op refuses
a configuration whose key would pass 63 bits. So in each round a requester
first pushes the rows of keys with a row each, by IB chunk, then its heavy
keys' rows, key by key. Its push
tables give, per round, its first position, its first heavy position, its
first push-task flag and its first heavy group. A round's flags number the
tasks of 32 rows (and then of 32 heavy groups) of each requester after the
lower requesters' tasks. The reducer list gives each key's entries, in
occurrence order (for a heavy key its own row, then each requester's partial),
each with the flag of the push task that fills it.

**The rail.** Each rank sends each rail peer a count message
(`CountMessage`, int64 words). Its common part, the same for every peer:
its node's largest remote part; as proxy its rounds and its node-unique keys
per owner node; as owner the keys of its own node it looks up. Then the part
for the peer's node alone: its node-unique keys there per feature. From the
`num_nodes` messages it gets, a rank computes: as proxy, where its slice
lands in each owner's `rail_keys` (a slice per source node, in node order);
as owner, its KJT lengths and offsets, and where each other node's cells
start in its KJT (`cell_stage`, which `rail_key_exchange` puts to that
node's proxy, into its `rail_cells`); and the rail's needs, as every
owner's keys are in the common parts.
`node_status_exchange` then gives every rank its node's rail needs. Their
maximum is the call's needs, the same on every rank: a rank's rail covers
every node, and its node holds one rank of every rail.

**The requester.** Two `peer_gather`s read what the node's proxies wrote. One
reads the words of their headers it needs: the rounds, the round table and its
own push table, by word lists made when the op is built. The other reads, for
each of its keys for an owner on another node, its occurrence's entry in the
three `remote_map` arrays at its proxy. From them, and from where its node's
owners hold its keys (`node_status_exchange`), it makes its maps. `src` is
where the forward reads each key's row: a row of its owner's `lookup_rows`
when the owner is on its node, else the key's row in its proxy's `proxy_rows`.
`dst` is where the backward writes the row's gradient: the same row, or its
own slot in `proxy_rows`. It also makes its rows in push order at each proxy,
its push tables with absolute positions and groups, and the task starts of
each round. Its heavy groups are the runs of its push order of one partial
slot inside one (proxy, round) heavy range. Groups and their sub-tasks of 64
rows are numbered by cumulative sums.

**The owner.** Its KJT has `num_sources = local_size + num_nodes − 1` cells
per feature (section 9). Its keys come in slices of 2^22 positions: its own
node's cells from the own part of `node_keys`, the other nodes' from their
records in `rail_keys`. Its forward put list sends each source node the rows
of its round pieces, which come first in that node's slice.

**The put lists.** One step (`put_lists`) makes both, in one tensor: the
owner's forward list, which cuts each source node's round pieces at the
cells of its KJT they fill, and the proxy's backward list, which cuts its own
pieces at where its cells start in each owner's KJT (`rail_cells`). A
node's node-unique keys are cut at its IB chunks (each round piece cut every
`chunk_rows` keys) and at its feature cells; each part between two cuts is
one segment, with its rows in `proxy_rows` and in the owner's KJT. A list's
segments are sorted by (node, key): node after node, round by round, and the
segments of a chunk follow each other. A table per list gives the first
segment of each (node, round). IB chunks are numbered piece by piece, round
after round, the backward's pieces before the forward's in each round. The
cuts are a node's chunks and its cells, so the lists hold at most `chunks +
2 × num_nodes × num_features` segments. The proxy also gets its reducers'
tasks of each round.

### 10.5 The status and the one host sync

The last step writes the call's status, `8 × (2 × num_features + 7)` bytes.
The call's tables need no copy: each is a tensor of its own, which the plans
keep for the out kernels (section 11.1). The status holds the needs: the
most any rank needs of `lookup_keys` and `remote_keys`, and the most rounds.
It also holds the keys this owner looks up, the sub-tasks of its heavy rows
and heavy groups, the rows sent, and the per-feature totals of the owner's
and of the input's KJT. The second wait copies it to the host on the current
stream: a blocking copy, which waits for every step before it, and the
call's only host sync. The host then compares the needs with the capacities
(section 5.4), and builds the owner's KJT from views, with no copy.
TorchRec's code around the dist keeps its own host syncs where they apply:
`split()` when a call spans several shardings or has input features that no
table serves, and VBE. So does DynamicEmb's lookup, the same on every dist.

## 11. The output dist's kernels

The output dist runs one cooperative kernel per direction on the caller's
stream: the forward kernel `fwd_kernel` and the backward kernel
`bwd_kernel`.

### 11.1 What both kernels share

* **Entry.** CTA 0 publishes `out_seq` in its `out_started` word (after a
  system fence) and every CTA waits until all node peers have published
  theirs. A peer past this point has finished its lookup (which wrote its
  `lookup_rows`) and every earlier out kernel.
* **Tasks.** Warps take 32-row tasks from one queue (a `task` word). Owners
  (and proxies) take turns, starting from each rank's own, so the ranks of a
  node start on different peers.
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
* **Tables.** The kernels read the call's tables from global memory,
  through typed pointers in their argument: the forward's plan as it reads
  it (`FwdPlanView`, in `FwdArgs`) and the backward's (`BwdPlanView`, in
  `BwdArgs`). Each table is a tensor of a plan. Both read the requester's
  block starts and maps, and across nodes its push order and its tables of
  the node's proxies (`proxy_rounds`, `push_tables`), the proxy's rounds and
  the put lists. The forward also reads the requester's `src`, its
  `proxy_round_tables`, `push_order_off` and `fwd_task_starts`, the owner's
  `fwd_put_start` and the proxy's `uniq_out` and `push_start`. The backward
  reads its `dst`, its heavy groups and `bwd_task_starts`, the owner's
  `uniq_in`, and the proxy's `num_chunks`, `uniq_off`, `round_table`,
  `bwd_put_start`, `reduce_task_starts` and its reducer list. Their strides
  are `rounds_cap` and `num_nodes`. Each kernel's argument also holds what
  the op gives it for the call: the forward's output, the backward's
  gradient and its scratch (the heavy rows' sums and counts), and across
  nodes the counts the kernel zeroes (`ready`). The forward's plan goes once
  its kernel is queued; autograd keeps the backward's (section 7.4).
* **The window.** A kernel lays out this rank's copy of the window from the
  sizes in its argument (section 7.5), and names each buffer and control
  word by its accessor: in this rank's copy, or in a node peer's through
  `NodeWindow::peer`. The one per-peer array in shared memory is the
  backward's `half_seen`, sized at launch by `local_size`.

### 11.2 The lookup

The lookup takes the call's rows, the first `num_recv` rows of `lookup_rows`,
from `op.lookup_rows(num_recv)`: a view, valid until the next out kernel. Each
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
  proxies in turn, each over its own QP. First one system fence. Then, round
  by round as each proxy grants it, it puts straight from `lookup_rows` into
  that proxy's `proxy_rows`, one put per segment of its forward put list (at
  most 2 MiB). Each put adds its row count to the proxy's `fwd_rows` signal.
  Then a flush.
* **Proxy relay** (across nodes): the last warp of CTA 0, whose lanes take
  the owner nodes and the requesters of the node in turn. It publishes to
  its node, in its `landed_rows` words, how many rows of each owner node
  have landed. With rounds, it grants round `rnd + 2` into round `rnd`'s
  half of `proxy_rows` once every requester of the node has pulled round
  `rnd` (its `done` words count their pulls).
* **Requester** (all other warps): pulls each of its rows from the
  same-node owner's `lookup_rows`, or, once it has landed, from its proxy's
  `proxy_rows`, into its output. Under index dedup it writes every original
  row.
* **Task order.** The row tasks of the node's owners, with index dedup's
  heavy-row sub-tasks of those rows spread among them. Across nodes, round
  after round: a slice of those local tasks, then the round's tasks at every
  proxy, proxies in turn. At a proxy they follow its push order: first the
  rows of keys that have a row each, then the rows of heavy keys. Last, the
  sub-tasks of heavy rows that come from other nodes. So requesters pull a
  round soon after it lands, and local pulls fill the wait for each round.
  No task waits for a task that comes later in the queue.
* **Exit.** The last CTA tells the node that this rank has pulled
  everything, and waits until every node peer has too. After that nobody
  reads this rank's `lookup_rows` or `proxy_rows`. Across nodes it then
  sends the credits for the next kernel: `bwd_ready` to each rail proxy and
  a grant of rounds 0 and 1 to each rail owner.

On one node the forward kernel is built without the code for several nodes.

### 11.4 The backward kernel

* **Requester**: under index dedup it first adds up each deduplicated row's
  original rows in fp32, in a fixed order (increasing original row). Under
  TorchRec's gradient division it then multiplies by `1 / world_size` in
  fp32 and rounds once to the row dtype, as TorchRec's `div_(world_size)`
  does. It pushes each row, in push order, into a same-node owner's
  `lookup_rows`, or into its proxy's `proxy_rows` at the row's slot. It sets
  one flag per 32-row task, tagged with `out_seq` and the round. A heavy key
  goes as one fp32 partial per requester, its heavy group added up in
  sub-tasks of at most 64 rows. A push into round `rnd ≥ 2` waits until the
  proxy has freed that half.
* **Proxy reducers** (the last CTAs, 8 by default). They wait for the flags
  of each key's entries. They add up each key of several rows in fp32, in a
  fixed order: its occurrences by requester and position, or a heavy key's
  partials by requester. They round the sum once into the key's row, and
  count the keys of the IB chunk as ready. A key of one row is not touched:
  the NIC reads it where the requester's NVLink stores left it.
* **Proxy issuer**: the last warp of CTA 0, whose lanes take the owner
  nodes in turn. Once an owner's `bwd_ready` is in, it puts each IB chunk
  (2 MiB of rows) of that owner node into the owner's `lookup_rows` as soon
  as the chunk's keys are ready, in whatever order they complete. It makes
  one system fence per chunk, and each put adds to the owner's `bwd_rows`
  signal. With rounds it counts the NIC's reads per half, and frees a half
  once its sums have been read. At the end it grants each rail owner rounds
  0 and 1 of the next kernel.
* **Landing watcher** (a traced build only, across nodes: section 4.4):
  the warp before the issuer bins the gradient rows the rail proxies put
  into this rank's `lookup_rows` as they land, for the timeline.
* **Owner**: waits until every same-node requester has pushed and every
  rail proxy's rows have landed. Its `lookup_grad`, the gradient of its
  lookup output, is a view of `lookup_rows`. Every row of `lookup_grad` is
  written exactly once, so nothing is zeroed or scattered. The owner hands
  it to the lookup's backward. It is valid until the next lookup, which
  writes `lookup_rows` (section 11.2).
* **Task order.** Across nodes, first the sub-tasks of heavy index-dedup
  rows that go to other nodes, and the heavy groups' sub-tasks. Then, round
  after round, each proxy's heavy-group push tasks, then its row push tasks,
  proxies in turn. So the IB chunks fill in push order. Last, the row tasks
  of the node's owners, with the sub-tasks of heavy rows that stay on the
  node spread among them. The last sub-task of such a row to end adds them
  up and writes the row. A task that needs sub-tasks waits until they end.
  They come earlier in the queue, so a running warp has them, and they wait
  for nothing.

## 12. Rounds and credits

### 12.1 Rounds

The proxy lays its rows out in `proxy_rows` by rounds. A round holds the
owner nodes one after another. An owner node's piece of a round holds its
node-unique keys `[uniq_lo, uniq_hi)`. It has their rows, then their
duplicates' slots (for the backward), then the fp32 partial slots of its
heavy keys. The proxy's *round table* lists every round's pieces.

```
one round        [ 1st owner node's piece | 2nd owner node's piece | ... |    free    ]
three or more    [ half 0: rounds 0, 2, 4, ...          | half 1: rounds 1, 3, 5, ... ]
a piece          [ rows of keys uniq_lo..uniq_hi | duplicates' slots | heavy partials ]
```

* One round when the proxy's rows fit in `proxy_rows`; the round may then
  use all of it.
* Else `num_rounds = ceil(rows / floor(0.9 × floor(num_proxy_rows / 2)))`,
  which is at least 3: a round fills at most 90% of a half, and round `rnd`
  uses half `rnd mod 2`. The other 10% takes what does not split evenly,
  since a key is never split: up to 64 rows for each owner node. The least
  size of `proxy_rows` leaves room for that (section 15.3). Two rounds never
  happen.
* Each owner node's keys are cut into `num_rounds` pieces of equal rows.
* Rounds 0 and 1 are free when an out kernel starts. In the forward, round
  `rnd + 2` refills round `rnd`'s half once every requester of the node has
  pulled round `rnd` (the relay's grants). In the backward it does so once
  the NIC has read round `rnd`'s sums (the issuer frees the half). A
  requester takes every task of a round, at every proxy, before any task of
  the next round, so no task waits behind one that needs a later round.
* Rounds change no result: every sum keeps its order (section 18). The
  tests across nodes also run with a 1 MiB proxy cap, where the larger calls
  take several rounds, against TorchRec's dists.
* A call's tables hold at most 65536 rounds: a `remote_keys` capacity whose
  rows would take more is refused when the op is built (section 15.3).

### 12.2 Credits and sequence numbers

`in_seq` counts first waits and `out_seq` counts out kernels, the same on
every rank. The credits for the next out kernel (`out_seq + 1`) go out at the
end of kernel `out_seq`. At that time no one knows yet whether the next one
is a forward or a backward. A credit names the kernel it lets in, and other
kernels ignore it:

* `grant[proxy node] = (out_seq << 20) | rounds`, proxy → owner: "put the
  rows of rounds below `rounds` of kernel `out_seq` into my `proxy_rows`";
* `bwd_ready[owner node] = out_seq`, owner → proxy: "put the rows of
  backward `out_seq` into my `lookup_rows`".

For a forward with `out_seq` = 7, then its backward (8), then the next
forward (9):

```
end of forward 7    proxy -> owner: grant (8<<20)|2      owner -> proxy: bwd_ready 8
backward 8          the proxy puts its summed rows as soon as they are ready
end of backward 8   proxy -> owner: grant (9<<20)|2
                    (no bwd_ready: the owner's lookup_grad is still in use)
forward 9           the owner puts rounds 0 and 1 at once; the relay grants
                    round rnd+2 once round rnd is pulled
```

The owner sends no `bwd_ready` at the end of a backward, because its
`lookup_grad` stays in use until the lookup's backward reads it. A backward
right after a backward therefore waits one round trip for `bwd_ready` (the
issuer, as owner, sends it when the kernel starts). Every other kernel can
put its first IB rows as soon as they are ready, without asking first.
Credits are signals with no data, each adding the difference to the value
sent before, on context 3.

A call that does not fit runs no out kernel, so it sends and takes no
credit. The credits of the last out kernel wait for the next call that takes
the op.

## 13. Memory order, and why nothing hangs

* **NVLink.** A writer stores rows with 16-byte stores, waits at a warp or
  CTA barrier, then does one *release* store (system scope) of a flag
  tagged with the sequence number. The reader does an *acquire* load of the
  flag, then reads the rows past the L1 cache. (Release and acquire are
  ordering rules: every write before the release is seen by a reader after
  its acquire.)
* **GIN.** The sender makes one system-scope fence, then puts. The forward
  fences once per kernel, since the lookup wrote `lookup_rows` before the
  kernel. The backward fences once per IB chunk. Each put adds to a signal
  at the receiver, which reads the signal before the rows. In the backward,
  a node peer writes rows over NVLink and the NIC then reads them. A chain
  orders the two: the task flag (release), the reducer (acquire), the
  chunk's ready counter (release), the issuer (acquire), and its fence
  before the put.
* **The in channel.** What peers wrote is read only after an acquire of
  their flag or signal. The comm kernel that acquired it reads past the L1
  cache, and so do the gathers later on the stream; the planner's kernels
  read with plain loads. What this rank writes for its peers, with its
  own stores or with the planner's kernels earlier on the stream, comes
  before the flag it releases or the put it makes. In
  `node_count_exchange` the release of `in_started` after the CTA's barrier
  orders it. Everywhere else a system fence does: each CTA of the push ends
  with one, and `node_done`, `node_status_exchange` and each lane that puts
  in the rail exchanges make one before their flag or put. The rail
  exchanges flush their puts before they end.
* **No hang.** A kernel that waits for peers waits only for the same kernel
  on its peers, for credits from earlier out kernels, and for its own
  rank's earlier work. The planner's kernels, the push and the gathers never
  wait. The op orders nothing across streams: the caller orders the kernels
  the same way on every rank (section 4.2). It makes the same calls in the
  same order, runs the input dists on one stream and the out kernels on one
  stream, with the same waits between its streams. So every rank issues the
  same kernels in the same order. The CTA budget (section 14) lets the
  kernels that may wait at the same time run together: an out kernel, a
  comm kernel on another stream, and NCCL's kernels. So every kernel ends:
  the first one can always end, and each later one can end once the ones
  before it have. A call that does not fit runs its input dist like any
  other call, and no out kernel on any rank. A wait that lasts longer than
  the timeout prints what it waited for and traps.

A buffer or control word of the window is written in call `in_seq + 1` only
once no peer still reads what call `in_seq` wrote there. The words marked
(ctrl) are in the ctrl section, named by their accessors (`CtrlLayout`):

| buffer or word | written by | last read by | why that is safe |
|---|---|---|---|
| `node_keys`, `node_weights`, `remote_cells` | node peers' push | this rank's planner | the peers push after the next call's node barrier, which this rank passes only after its in channel finished the call |
| `remote_map`, `proxy_header` | this rank's planner | node peers' gathers | this rank writes them after the next call's node barrier, which every peer passes only after its in channel finished the call |
| `key_stage`, `cell_stage`, `count_msg_out` (ctrl) | this rank | the NIC | the rail exchanges flush their puts before they end |
| `rail_keys`, `rail_cells` | rail peers' key and cell puts | this rank's planner | a rail peer puts only after it has this rank's count message of the next call, which this rank sends after the call ended here |
| `count_msg` (ctrl) | rail peers' count puts, and this rank its own | this rank's `rail_count_exchange` | a slot per `in_seq` parity (and source node): a peer's message of call `in_seq + 2` needs this rank's of call `in_seq + 1` |
| `count_vector` (ctrl) | this rank's `node_count_exchange` | node peers' `node_count_exchange` | a slot per `in_seq` parity |
| `cell_start`, `needs` and `in_done` (ctrl) | this rank | node peers' `node_status_exchange`, which copies them out | this rank writes them after the next call's node barrier |

## 14. The CTA budget

A kernel that waits for peers must be running at the same time as every kernel
it may wait on, or the GPU could hang. Three kinds of kernels wait. The caller
runs the out kernels on one stream, so one of them waits at a time. A comm
kernel that waits runs on one CTA, and an input dist may run at the same time
on another stream (a pipeline's data_dist stream). NCCL's own kernels may wait
too. So an out kernel takes `min(want, SMs − 34)` CTAs: it leaves one SM to
such a comm kernel, 32 to NCCL's kernels, and one to spare. It takes at least
1 CTA, and the backward across nodes at least 2 (its issuer's CTA and a
reducing one). `want` is `DYNAMICEMB_GIN_CTAS` when set, else 32, or 64 for
the backward on one node (section 4.4). The input dist's other kernels never
wait (section 13), so they need no SM of their own. **Example**, H100 (132
SMs): an out kernel may take up to 98 CTAs, so the out kernels keep their 32,
and the backward on one node its 64.

## 15. Memory

### 15.1 What is counted

Per rank and sharding, the window (outside PyTorch's caching allocator) is
*persistent*: made when the op is built, and kept through the dense part of
every step. A call's tensors live *per call*, from its part 1 to its
backward (one call in a plain loop, two or three under the train
pipelines). The input dist's working memory, part 1's and the planner's, is
*transient*: a step frees its temporaries as soon as it is done with them.
The caching allocator keeps their blocks, though, in its pool of the stream
they were made on: the caller's stream that the input dist runs on. So that
pool holds the largest step's working memory.

### 15.2 Formulas

The capacities are the ones in place (section 5), and `row_bytes` and
`record_bytes` those of section 5.3. Section 15.3 defines `num_proxy_rows`
and `rounds_cap`.

| buffer | bytes | kind |
|---|---|---|
| `ctrl` (the ctrl section) | one node: `16 × local_size × (num_features + 2) + 64`; across nodes about `8 × (3 × num_features × (local_size + num_nodes) + 2 × num_nodes² + 16 × num_nodes + 8 × local_size)`, most of it the count messages: 80 KB at 32 nodes, 8 GPUs per node and 64 features, 7.1 MB at 256 nodes and 1,000 features | persistent |
| `node_keys` | `8 × (lookup_keys + remote_keys)` (one node: `lookup_keys` alone) | persistent |
| `node_weights` | with weights, `4 × (lookup_keys + remote_keys)` | persistent |
| `remote_cells` | `4 × remote_keys` | persistent |
| `remote_map` | `12 × remote_keys`: three int32 arrays | persistent |
| `proxy_header` | `16 × (1 + rounds_cap × num_nodes + local_size × (rounds_cap + 1))` | persistent |
| `key_stage` | `record_bytes × remote_keys + 16 × (num_nodes − 1) × rounds_cap + 16` | persistent |
| `rail_keys` | `record_bytes × lookup_keys + 16 × (num_nodes − 1) × rounds_cap` | persistent |
| `cell_stage`, `rail_cells` | `8 × num_nodes × num_features` each | persistent |
| `lookup_rows` | `row_bytes × lookup_keys` | persistent |
| `proxy_rows` | `row_bytes × num_proxy_rows` (section 15.3) | persistent |
| `push_flags` | `16 × (num_proxy_rows / 32 + 2 × local_size)` | persistent |
| the requester's maps | 8 bytes per key on one node (`inv`, `src`), 16 across nodes (also `dst`, `push_order`) | per call |
| the requester's heavy groups and their sub-tasks (across nodes) | `12 × heavy_groups + 4 × (num_keys / 64 + heavy_groups + 1)`, with `heavy_groups = min(num_keys, local_size × (remote_keys / 65 + 1))`: about 2 bytes per entry of `remote_keys` | per call |
| the call's small tables across nodes: the requester's of the node's proxies and its task starts, the owner's and the proxy's (section 11.1) | about `16 × rounds_cap × num_nodes × (local_size + 2)`, plus a few KB | per call |
| the owner's KJT values | `8 × lookup_keys` (12 with weights), from the first wait until the forward | per call |
| the proxy's reducer list | `12 × remote_keys` | per call |
| the put lists and the ready counters | `24 × chunks + 32 × num_nodes × num_features`, with `chunks = (remote_keys + lookup_keys) / chunk_rows + 2 × rounds_cap × num_nodes + 2` (`chunk_rows`: rows in 2 MiB) | per call |
| index dedup's fan-out table | about 13 bytes per original key (on this path the deduplicated KJT keeps the original length) | per call |
| part 1, in the caller's pool | at its peak 25 bytes per key with one owner rule (67 with three), and about 60 with index dedup while the fan-out table is made | transient |
| the planner, in the caller's pool | its largest step: across nodes the proxy's about 99 bytes per entry of `remote_keys`, the requester's about 73 per key, or the owner's gather about 65 per entry of `lookup_keys`, at most 2^22 entries at a time; on one node about 20 bytes per key | transient |

The transient figures are the peaks measured with the benchmark's batches
(bytes in live tensors; the caching allocator's rounding and the cached
blocks it hands out add up to about 13%), but the one-node planner's, which
comes from the code.

The constructors of the layouts size the window's buffers as above
(`CtrlLayout`, `KeysLayout` and `RowsLayout`, in `window.h`). The buffers
from `remote_cells` to `rail_cells`, and `proxy_rows` and `push_flags`, exist
across nodes only, and `node_weights` with weights only: else they take 0
bytes. Each buffer, the ctrl section too, takes a whole number of 4 KiB
units (`kBufferAlign`). So the window's bytes are the sum of its buffers,
each rounded up to 4 KiB (`WindowLayout::bytes`), and its copy on each rank
is a whole number of allocation granules. `memory()` gives both (section
7.1).

With uniform keys across nodes, and capacities 1.25 times the needs
(section 5.2), the keys section takes about
`1.25 × num_keys × (16 × (1 + (num_nodes − 1) × uniq_share) + 32 × (num_nodes − 1)) / num_nodes`
bytes. That is about 39 bytes per key at 2 nodes, and 58 at 32 nodes. The
per-call tensors take about 35 to 43 bytes per key across nodes: 16 for the
requester's maps, 1 to 2.5 for its heavy groups, about 10 for the owner's
KJT until the forward, and `12 × 1.25 × (num_nodes − 1) / num_nodes` for the
reducer list.

### 15.3 `proxy_rows` and rounds

```
num_proxy_rows = min(max(remote_keys, min_proxy_rows), proxy_bytes / row_bytes)   (rounded down)
min_proxy_rows = 20 × 64 × (num_nodes − 1)
proxy_bytes    = DYNAMICEMB_GIN_PROXY_MB MiB (512 by default)
rounds_cap     = the rounds that remote_keys rows take in num_proxy_rows rows (section 12.1)
```

So `proxy_rows` holds a call's rows in one round when `remote_keys` rows
fit under the cap. A cap below `min_proxy_rows × row_bytes` is refused when
the op is built (section 6), and so is a `remote_keys` whose rows would take
more than 65536 rounds.

The **rows a proxy needs** count, per node-unique key, one row per
occurrence (the key's row and its duplicates' slots), up to 64 occurrences.
A heavy key, with more, counts `1 + local_size × ceil(4 × dim / row_bytes)`
rows: its row and its partials. So a call never needs more rows than it has
occurrences, and a call within `remote_keys` fits in `rounds_cap` rounds.
The number of rounds follows from these rows (section 12.1).

### 15.4 How memory scales

**With the number of GPUs and features.** No buffer is sized by
`world_size`, but for part 1's cell offsets and the node layout's places
(`8 × world_size × num_features` bytes each per call). `num_nodes` enters
only through:

* the share `(num_nodes − 1) / num_nodes`;
* the control words and the cell starts (`cell_stage`, `rail_cells`):
  linear in `num_nodes × num_features`, plus `16 × num_nodes²` bytes of the
  count messages' common parts. That is 0.11 MB at 32 nodes, 8 GPUs per node
  and 64 features, and 11 MB at 256 nodes and 1,000 features;
* GIN's queues to `num_nodes − 1` rail peers per context (with
  `NCCL_CROSS_NIC=0`);
* small round tables that grow with `num_nodes × rounds_cap`:
  `proxy_header`, the round pieces of `key_stage` and `rail_keys`, and per
  call the round tables of the requester and of the proxy (the requester's
  also grow with `local_size`) and the starts of the put lists;
* the put lists, a segment per IB chunk and per (node, feature) cell.

`local_size` enters through the node's counts (`8 × local_size ×
(num_features + 1)` bytes for each of the three sums, and the
`count_vector` words), and through the tables kept per GPU of the node:
the proxy header's push tables, and per call the requester's tables of
each proxy and the proxy's push starts of each requester. No table holds a
count per (node, node, feature) or per (GPU, GPU, feature).

Everything else is linear in the capacities, except `proxy_rows` (capped: a
larger call takes more rounds).

**With batch × sequence length.** The table below is computed from the
sizing rules in the code, with capacities 1.25 times the needs and at least
4096 entries more (`hier_capacities`), and from the transient figures of
section 15.2. It assumes uniform keys, mean loads, no weights, one owner
rule and 8 GPUs per node; it is not measured. GB per rank. "Persistent" is
the window: `lookup_rows` + `proxy_rows` + its other buffers (the ctrl and
keys sections and `push_flags`); "per call" is one call's tensors;
"planner" is the planner's largest step's working memory, and "part 1"
part 1's, both in the caller's pool.

| case | nodes (GPUs) | `lookup_rows` | `proxy_rows` | other buffers | persistent | per call | planner | part 1 | rounds |
|---|---|---|---|---|---|---|---|---|---|
| the benchmark: 540,672 keys per rank, fp32 dim 128, `uniq_share` 0.935 | 1 (8) | 0.35 | – | 0.005 | **0.35** | 0.010 | 0.011 | 0.014 | – |
| | 2 (16) | 0.34 | 0.17 | 0.021 | **0.53** | 0.019 | 0.043 | 0.014 | 1 |
| | 4 (32) | 0.33 | 0.26 | 0.027 | **0.62** | 0.021 | 0.050 | 0.014 | 1 |
| | 8 (64) | 0.33 | 0.30 | 0.029 | **0.66** | 0.022 | 0.059 | 0.014 | 1 |
| | 32 (256) | 0.32 | 0.34 | 0.031 | **0.69** | 0.023 | 0.065 | 0.014 | 1 |
| 16,384 samples × 1,536 keys per sample, 1 feature: 25.2M keys per rank, bf16 dim 128, `uniq_share` 1 | 2 (16) | 8.05 | 0.54 | 1.01 | **9.60** | 0.88 | 1.84 | 0.63 | 14 |
| | 4 (32) | 8.05 | 0.54 | 1.26 | **9.85** | 0.99 | 2.34 | 0.63 | 21 |
| | 8 (64) | 8.05 | 0.54 | 1.39 | **9.98** | 1.04 | 2.73 | 0.63 | 24 |
| | 32 (256) | 8.05 | 0.54 | 1.48 | **10.1** | 1.08 | 3.02 | 0.63 | 26 |
| the same with 2 features: 50.3M keys per rank, fp32 dim 128 | 2 (16) | 32.2 | 0.54 | 2.01 | **34.8** | 1.75 | 3.67 | 1.26 | 54 |
| | 8 (64) | 32.2 | 0.54 | 2.77 | **35.5** | 2.08 | 5.45 | 1.26 | 94 |
| | 32 (256) | 32.2 | 0.54 | 2.96 | **35.7** | 2.16 | 6.03 | 1.26 | 104 |

The planner's working memory adds 3% to the persistent memory on one node and
8–9% across nodes at the benchmark's size, and 11–30% at the large batches
above. Take 4 nodes, fp32 dim 128, one feature, and every key distinct. For 1,
4, 16 and 64 Mi keys per rank, the persistent memory is 1.2, 3.4, 12.1 and 47
GB, in 1, 7, 27 and 107 rounds.

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
| `TrainPipelineSemiSync` | runs; tested against itself (its outputs are not the plain loop's): without rehashes, and with capacities that fit every call |
| several forwards, then their backwards; `retain_graph`; forwards without grad | supported, tested |
| calls that do not fit the capacities, under the plain loop and the train pipelines above | supported, tested: they take TorchRec's dists, and the calls around them take the op |
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
  `record_stream` for the streams they hand a call to, and it calls the
  result's (`HierInDistResult.record_stream`, which records its plans
  too). With `emb_lookup_stream="new"`, TorchRec's fused pipeline runs the
  lookup and the output dist on that stream but records the context on the
  default stream only. So the lookup records the owner's KJT, and the
  output dist's forward the plans, on the stream they run on: the forward's
  plan goes as soon as the forward kernel is queued, and without the record
  the input dist of the next call, on the data_dist stream, could get its
  memory while the kernel still reads it. The op records no stream itself.
* A call's tensors live from its part 1 to its backward (section 15.1), so
  under `TrainPipelineSparseDist` the next batch's are alive during this
  batch's forward and backward.
* With `emb_lookup_stream="new"`, TorchRec's fused pipeline runs the lookup
  and the forward kernel on a stream of its own. The output dist's
  awaitable makes the dense forward's stream wait for the kernel (section
  7.4).

## 17. TorchRec versions

TorchRec 1.2 to 1.8 are supported. The dist uses only these parts of TorchRec:
its `Awaitable` classes, its sequence sharding context, the KJT constructor,
and `RwSequenceEmbeddingDist` and `KJTAllToAll` for calls on TorchRec's dists.
The input dist keeps a real `KJTAllToAll`, because TorchRec's pipelines check
its type; a call that does not fit runs it. Two version checks handle the
rest: `_create_embedding_kernel` takes an `env` argument from 1.5 on, and the
model delta tracker hook exists from 1.5 on. `TrainPipelineFusedSparseDist`
exists from 1.3 on.

## 18. Correctness and determinism

**Forward: bit for bit equal to TorchRec's dists, index dedup included.**
Each output row is a byte copy of the owner's row. For each table the owner
looks up the same set of keys as TorchRec's owner: every same-node
occurrence, and one entry per (other node, table, key). It sees the same
frequency counts, so inserts, admission and LFU scores match. As with any
change of key order, random initializers and eviction under capacity
pressure may differ. TorchRec's own `unique` order is not fixed either. So
the bit-exact tests use the DEBUG initializer and no eviction. A call that
does not fit runs on TorchRec's dists, so it gives TorchRec's results.

**Backward: per (table, key), the same terms.** Every row of `lookup_grad`
is written once, and every occurrence adds to exactly one row, so the
lookup's backward adds up the same terms as TorchRec's:

* fp32 rows: equal up to the order of fp32 additions. The `1 / world_size`
  scaling is TorchRec's bit for bit (a multiplication by `1 / world_size`
  in fp32, rounded once). On one node without index dedup every gradient
  row equals TorchRec's bit for bit.
* bf16 or fp16 rows with node dedup: a node's sum of several occurrences is
  rounded to the row dtype once more before it crosses IB. The error is at
  most `2^-8` of that sum for bf16, and `2^-11` for fp16. Index dedup's sums
  are made in fp32, where TorchRec's `index_add` adds in the row dtype; a
  heavy key's occurrences are added in fp32 without rounding each one.
  `node_dedup=False` gives TorchRec's rows back, at the cost of more IB
  traffic.
* Gradient clipping clamps each row before the per-key sum, so it turns node
  dedup off and keeps TorchRec's meaning.

**Run to run.** Every sum the dist makes has a fixed order: duplicates by
requester and position, heavy partials by requester, and index dedup's sums by
original row. The planner numbers keys, slots and tasks with stable sorts and
cumulative sums, which give the same result every run, and adds weights as
whole numbers. None of its steps adds up or hands out slots with atomics. So
the output rows and the gradient handed to the lookup are the same bits every
run. That holds across nodes, with or without index dedup, in one round or
many. Under index dedup, the keys in the owner's KJT cells of same-node
requesters follow DynamicEmb's `unique` numbering, and so do those rows of
`lookup_grad` (the other nodes' cells are in key order, section 9). That
numbering may change between runs; the value per key does not. DynamicEmb's
lookup backward adds a hot key's partial sums with 16-bit atomic additions on
every dist, so tables of 16-bit rows may still differ from run to run.

**Tests** (`test/unit_tests/`): `test_hier_part1.py`, `test_hier_dedup.py`,
`test_hier_dist.py`, `test_hier_fallback.py` and `test_hier_pipeline.py`;
each module's docstring says what it checks and how to run it. On one node
`test_hier_dist.sh`, `test_hier_fallback.sh` and `test_hier_pipeline.sh`
run them, and `test/unit_test.sh` runs those.

## 19. Limitations

* **Scope**: row-wise sequence shardings of DynamicEmb tables, within the
  limits of section 4.2.
* **Fixed capacities**: a call that needs more runs on TorchRec's dists, at
  their cost, while the op's window stays allocated (section 5.4).
* **Errors no agreement covers**: an NCCL error while the op makes its
  communicator, or inside the collective window registration, on some ranks
  only leaves the other ranks waiting (section 7.2). An error inside a first
  wait, such as an allocation that fails, reaches Python on its rank. The
  other ranks then trap at the timeout in the call's next comm kernel that
  waits, as when a rank dies.
* **Memory held through the dense part**: `lookup_rows` and `proxy_rows`
  stay allocated, each sharding has its own window, the caching allocator's
  pool of the input dist's stream keeps the largest step's working memory,
  and the hottest owner's need sizes `lookup_keys` on every rank (section
  15).
* **Host time of the input dist**: across nodes a first wait launches
  about 120 kernels, CUB's included (section 10), and the training thread
  spends their host time, also under the train pipelines.
* **Size of one call**: a GPU may receive, deduplicate or look up fewer than
  about 2.1 billion keys of each kind in one call (section 4.3).
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
  `num_sources` columns per feature and one entry per (other node, table,
  key) (section 9).
* **16-bit rows with node dedup** get one more rounding per node sum
  (section 18).
* **Skewed keys**: lower IB use in the backward, and a slow rank's lookup
  delays the other ranks' output-dist kernels.
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
  `HierSharding` (the op and what a sharding's calls share), the output
  dist and `quiesce`; `input_dist.py` the hierarchical input dist
  (`hier_forward`, its two awaitables and `fallback`); `rw_sharding.py` the
  choice of dist and the lookup; `embedding.py` the choice per call. The
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
      configuration (`Config`) and its checks, and the op's sizes
      (`sizes_of`);
    * `window.h`: the window's layout (`WindowLayout`) over its sections:
      `CtrlLayout` (with the formats of the count vector and the count
      message, and a channel's sequence number), `KeysLayout` (with
      `ProxyHeader`, `RemoteMap` and what `peer_gather` reads) and
      `RowsLayout`;
    * `device.cuh`: the kernels' building blocks: loads, stores and fences
      with a memory order, bounded waits, warp and CTA helpers, a search, row
      vectors, the window as the node's GPUs map it (`NodeWindow`), what the
      kernels of both channels take (`ChannelStatic`), the syncs on its
      control words (`NodeBarrier`, `PeerFlags`), and GIN on the rail
      (`rail_put`, `RowPut`, `RailMessage`, `Credits`);
  * `input_dist/`, the input dist:
    * `input_dist.h`, `input_dist.cu`: the data of a call's steps
      (`PartOne`, `Pending`, `InDistResult`) and the in channel
      (`InChannel`), whose `start` queues the first wait's steps and whose
      `wait` makes the result;
    * `plan.h`, `plan.cu`: the output dist's plans (`FwdPlan`, `BwdPlan`),
      and the same as the out kernels read them (`FwdPlanView`,
      `BwdPlanView`);
    * `comm_kernels.cu`: the comm kernels, and the functions that launch
      them;
    * `planner.h`: the planner: the declarations of every step, and of the
      CUB calls in `cub.cu` (radix sorts and scans); `planner.cuh`: what the
      planner's kernels share (how they are launched, searches in sorted
      tables, the starts of sorted runs, a CTA's scans). Part 1 is in
      `part_one.cu` and the proxy in `proxy.cu`; the node layout, the rail
      and the owner, the requester and the put lists are in `planner.cu`;
  * `output_dist/`, the output dist:
    * `output_dist.h`, `output_dist.cu`: the out channel (`OutChannel`),
      which launches the out kernels, and their trace (`Trace`).
      `output_dist.cu` is the one file that compiles the out kernels;
    * `args.cuh`: the out kernels' arguments: what both take (`OutArgs`), and
      each one's own with its plan (`FwdArgs`, `BwdArgs`);
    * `out_kernels.cuh`: what both out kernels share (their timeline, the
      call's tables as a requester's warps read them, their task schedule
      and their protocol); `forward.cuh` and `backward.cuh`: the forward and
      backward kernels.
* `setup.py` builds `src/hier_output_dist/` only against NCCL ≥ 2.30.7, and
  else defines `DEMB_NO_HIER_OUTPUT_DIST`. With `DYNAMICEMB_HIER_TRACE=1` it
  defines `DEMB_HIER_TRACE=1` (section 4.4).
* **Tests**: section 18. **Benchmark**:
  `benchmark/benchmark_distributed_embedding_collection.py`
  (`benchmark/README.md`, section 3), `benchmark/benchmark_utils.py`
  (`hier_needs`, `hier_capacities`) and `benchmark/nvlink_ceiling.cu`.
