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

1. **Touched keys.** `incremental_dump` over the table's timestamp score finds
   the keys touched since the last sync, whose device time is the threshold.
2. **Union.** The keys are all-gathered over the replica group and made unique.
   `torch.unique` sorts them, so every replica merges them in the same order.
3. **Read.** A const lookup (scores unchanged) finds which keys this replica
   holds. Their embeddings are read at the precision a checkpoint holds them at,
   which is lossless (fp16 for an fp8 table). The optimizer state is read at
   checkpoint width (row-wise Adagrad's fp32 accumulator), along with all score
   words.
4. **Exchange.** Each replica sends a held mask per key (one byte) and only the
   rows it holds. The sync is bound by bandwidth, and a replica that lacks a key
   sends no row for it.
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

A sync costs in proportion to the keys touched since the last one, not to the
size of the table.

## 5. Checkpoints

A DynamicEmb kernel names its checkpoint files by its own shard index and shard
count, read off the placements TorchRec planned. Under 2D, shard s of every
replica therefore writes the same file names. `DynamicEmbDump` given a
`DMPCollection` has the first replica dump, over its sharding group, while the
others wait at `pg`. `DynamicEmbLoad` needs no change: every replica reads its
own shard's files. A shard's meta records `sharding_type: table_row_wise`,
because the table is on fewer ranks than the world. That field is informational,
and a checkpoint written under 2D loads into a row-wise world of the same shard
count as any other.

## 6. Constraints, checked when `DynamicEmbReplicaSync` is built

- **Values in HBM:** a `DynamicEmbStorage`, with no cache and no host tier. The
  sync reads and writes rows through the flat-table operations.
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
