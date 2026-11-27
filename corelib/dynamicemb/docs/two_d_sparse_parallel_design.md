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

Each call to `DynamicEmbReplicaSync.sync()` passes every local DynamicEmb table
to one Torch operator, `torch.ops.dyn_emb.replica_sync_batch_`. Tables can
belong to different kernels and have different widths, dtypes, initializers and
score layouts. The returned statistics follow the original table order.

1. **Discover keys.** Each table scans its timestamp column for keys touched
   since the last sync. The replicas exchange one vector of table counts and
   one concatenated key buffer, then build a sorted union for each table.
   A const lookup maps those keys to local value rows and hash slots. Dense
   tables can reuse their routes after one batch-wide vote checks the live
   counts, keys, slots, rows and timestamps.
2. **Exchange rows.** CUDA kernels pack the embedding, live optimizer state
   and optional LFU counters for each held row. FP8 values keep their stored
   encoding and packed Adagrad accumulators keep their fp32 precision. A
   packed NCCL slab can span several tables. All replica groups use this
   collective, and NCCL selects the transport for the available topology.
   A shared mask identifies which replicas contributed sparse rows. The next
   slab's all-gather can overlap writeback for the preceding slab.
3. **Merge and place.** The operator averages in fp32. For a constant
   initializer, an absent replica contributes the initial embedding at table
   precision and fresh optimizer state. For a random initializer, only the
   holders contribute. Each replica writes its held rows in place, sets the
   timestamp from its GPU clock, and stores LFU as the sum divided by the
   replica count, rounded down with a minimum of one. For locally absent keys
   the operator returns the averaged rows and scores. Python grows and inserts
   once per affected storage, after all communication has finished. This lets
   a rehash change the slot map without invalidating an in-flight merge.
4. **Advance the threshold.** After writeback, each replica records the device
   time. The next sync uses it to select touched keys.

The smallest `max_staging_bytes` requested by the replicas sets the payload
staging budget. NCCL processes at least one row per slab. The routing metadata
and the output rows awaiting local insertion are separate from the payload
budget.

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
- **`sync --mixed-tables`:** one native dispatch covers fp32 and fp16 tables
  with different widths and score layouts. The check exercises growth, a
  table touched on only one replica, and a subsequent empty sync.
- **`checkpoint`:** replicas that disagree dump. A fresh model's every replica
  then loads the first replica's rows, so one replica wrote, and wrote alone.
- **`reject`:** a table without a timestamp score is refused.
