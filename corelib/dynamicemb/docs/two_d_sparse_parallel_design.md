# Design: DynamicEmb under TorchRec 2D sparse parallelism (`DMPCollection`)

Status: **implemented**. Scope: `corelib/dynamicemb`.

## 1. Goal

TorchRec 2D sparse parallelism (`DMPCollection`) shards each table over a
*sharding group* of S ranks and replicates that group R = world / S times. Each
replica trains on its own data, and every few steps `DMPCollection.sync()`
averages the replicas. The embedding all-to-alls stay inside a group, so a job
can grow its world without growing its all-to-all.

This lets DynamicEmb tables run in a `DMPCollection`: built, trained, kept in
agreement across replicas, and checkpointed.

## 2. Why `DMPCollection.sync` does not cover DynamicEmb

`DMPCollection.sync` all-reduces each fused TBE weight tensor (and its
optimizer state) element by element. That is right for a TBE table, where row r
holds the same id in every replica. A DynamicEmb table is a hash table: each
replica places keys in its own slots, and each holds its own key set, since a
key only one replica saw was inserted only there. An element-wise average would
mix unrelated rows, so `DMPCollection` leaves DynamicEmb kernels out of its sync,
and the replicas are merged **by key** instead.

## 3. Usage

```python
from dynamicemb import DynamicEmbDump, DynamicEmbLoad, DynamicEmbReplicaSync

# Plan for one sharding group: Topology(world_size=S, local_world_size=...).
plan = planner.collective_plan(module, sharders, dist.GroupMember.WORLD)
model = DMPCollection(module=module, plan=plan, sharding_group_size=S, ...)
replica_sync = DynamicEmbReplicaSync(model)

for step, batch in enumerate(loader):
    train_step(model, batch)
    if (step + 1) % sync_interval == 0:
        model.sync(include_optimizer_state=True)  # TBE tables
        replica_sync.sync()                        # DynamicEmb tables

# A checkpoint, right after a sync: the first replica writes it.
DynamicEmbDump(path, model, optim=True)

# Restoring: every replica loads its own shard, so the replicas agree.
DynamicEmbLoad(path, model, optim=True)
replica_sync.mark_synced()
```

## 4. The sync

For each table, in chunks of keys that bound the staging buffers:

1. **Touched keys.** Scan the timestamp column on the GPU and retain keys
   touched since the last sync.
2. **Union.** Gather the keys over the replica group and take their sorted
   union. A const lookup maps the union to local value rows and hash slots.
   After a dense sync, ranks can reuse this mapping when the live key count
   and every cached key, slot, row and timestamp pass validation on all replicas.
3. **Pack.** A CUDA kernel reads the held rows directly from the table and
   packs their embedding bytes, live optimizer state and optional LFU counters.
   FP8 values retain their stored encoding and packed Adagrad accumulators
   retain their fp32 precision.
4. **Exchange.** The general path all-gathers packed buffers over the replica
   group. Each chunk is padded to the largest local held count. Sparse chunks
   include a separate held mask. For dense chunks the next all-gather can
   overlap the previous merge. Two local peers can also use symmetric memory
   to exchange aligned embedding-only rows.
5. **Merge.** Sums are taken in fp32. With a `CONSTANT` initializer, a replica
   without the key counts as a fresh row (the initial value at table precision
   and a fresh optimizer state), and the sum is divided by R. That is the average
   a single, unreplicated table would reach, and the rule TorchRec 2D applies to
   a TBE row only some replicas updated. With a random initializer there is no
   one fresh row, so only the replicas that hold the key are averaged.
6. **Write.** Each replica inserts the keys it lacked (expanding first), stores
   the merged rows, and writes the scores: timestamp = now, LFU = sum // R (at
   least 1, as for a fresh insert), and any other word sum // R.
7. **Mark.** The threshold becomes the device time after the sync.

The timestamp scan visits the hash table. Row traffic depends on touched keys,
row widths, the number of holders and padding. Each replica group uses the
smallest `max_staging_bytes` among its ranks for payload staging. For two local
GPUs with mutual P2P atomic access and both devices visible to each process,
an aligned, fully held table with a single timestamp score can use one
symmetric-memory inbox. A compact table that fits can DMA directly into it.
Other layouts reuse two slabs in the same workspace. When either peer cannot
allocate the inbox the pair tries the slab path, then packed NCCL. Falling
back to NCCL releases the peer workspace. The budget excludes cached key and
row coordinates. NCCL processes at least one row per slab.

## 5. Checkpoints

A DynamicEmb kernel names its checkpoint files by its own shard index and shard
count, read off the placements TorchRec planned. Under 2D, shard s of every
replica therefore writes the same file names. `DynamicEmbDump` given a
`DMPCollection` has the first replica dump, over its sharding group, while the
others wait at `pg`. Each replica loads its own shard's files. Checkpoints
with a compound score store the timestamp column as an age and preserve the
LFU column. Loading reconstructs the timestamp using the reader GPU's clock.
For older compound checkpoints that store raw timestamps, the loader clamps
future timestamps to the local time. A shard's meta records `sharding_type: table_row_wise`,
because the table is on fewer ranks than the world. That field is informational,
and a checkpoint written under 2D loads into a row-wise world of the same shard
count as any other.

## 6. Constraints, checked when `DynamicEmbReplicaSync` is built

- **Values in HBM:** a `DynamicEmbStorage`, with no cache and no host tier. The
  CUDA kernels read and write rows through the flat value buffer.
- **A timestamp score:** `TIMESTAMP` or `(TIMESTAMP, LFU)`. It is how the sync
  finds the touched keys.
- **Row-wise tables of the default context:** a table must be sharded over this
  rank's sharding group. Table-row-wise tables inside 2D, and tables in a
  `submodule_configs` context with its own groups, are rejected.

## 7. Not covered

- **`incremental_dump` / `replay_increment` under 2D.** One replica would
  publish the delta and the others would only drain their retained evictions.
  This is not wired here.
- **`erase_by_score_async` / TTL.** Replicas age keys on their own device
  clocks, so they would erase different keys.
- **Eviction during the sync's insert.** A replica whose bucket is full can
  evict a key the others keep, so in a capacity-bound table the key sets can
  drift apart. Rows the replicas share still agree.
- **TBE optimizer-state parity.** `DMPCollection.sync(include_optimizer_state=False)`
  has no counterpart here: the optimizer state shares the value row and is always
  merged.

## 8. Relation to the table-row-wise design

`table_row_wise_sharding_design.md` (F11) recommended rejecting `DMPCollection`
until the key-to-rank fan-out came from one source. All three of its sources
have since been closed, and `get_state_dict` now finds its shard by global rank,
so a `DMPCollection` builds. This design adds what remains: keeping the replicas
in agreement, and writing one checkpoint.

## 9. Tests

`test/unit_tests/test_two_d_sparse.sh` runs `test_two_d_sparse.py` and is
registered in both the `fwd_bwd` and the `load_dump` groups:

- **`sync`:** each replica trains on its own keys, some of them shared. After a
  sync, every replica holds the union, each row the average described above
  (embedding, optimizer state and LFU count), and the replicas agree exactly. A
  second sync covers no key. It runs on two GPUs in groups of one for fp32,
  fp16 and fp8 tables, `TIMESTAMP` scores and SGD, and on four GPUs in groups of
  two, both contiguous and strided.
- **`checkpoint`:** replicas that disagree dump. A fresh model's every replica
  then loads the first replica's rows, so one replica wrote, and wrote alone.
- **`reject`:** a table without a timestamp score is refused.
