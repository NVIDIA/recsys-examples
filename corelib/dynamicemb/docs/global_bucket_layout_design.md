# Fan-out-invariant global bucket layout: the `hash_bucket` dist_type

## 1. The problem

A `DynamicEmb` table is a bucketed hash table (`LinearBucketTable`,
`src/table_operation/types.cuh`). A key is probed only inside its home bucket;
when that bucket is full, `insert` evicts the resident with the lowest score, or
returns `Busy` and the key is dropped. The home bucket used to be

```
bucket = bkt_begin[t] + ((murmur3(key) & INT64_MAX) % table_cap) / bc
```

with `table_cap` the capacity of **this rank's shard** of logical table `t`.
The rank itself was `key % W` (`roundrobin`) or `murmur3(key) % W`
(`hash_roundrobin`). Both numbers depend on the fan-out `W`, so the set of keys
that share a bucket at one world size is re-dealt at another. Neither rule is
changed by this design; a third one, `hash_bucket`, is added beside them.

That is where a restore loses rows. A table trained at 8 ranks to 97 % fill
holds, in every bucket, at most `bc` keys -- the overflow was evicted during
training. Loaded at 1 rank, those keys land in different buckets: some now hold
more than `bc` keys, and the load path (`insert` with `ASSIGN`) evicts the
surplus. Measured on a real 8-rank checkpoint folded onto one GPU
(`TorchEasyRec`, `model.ckpt-412`, 89 tables, STEP/FTRL): 1.3-2.0 % of the rows
of every in-memory table were lost; synthetic CACHING tables at 97 % fill lost
2.3 % with `bucket_capacity=128` and 0.3 % with 1024; HYBRID tables lost a
further ~1.2 % on every same-topology reload. Only the external (RocksDB)
storage, which has no buckets, restored exactly.

## 2. The invariant

For a `hash_bucket` table with `N = num_embeddings`, effective bucket
capacity `bc` (not `MAX_BUCKET_CAPACITY`) and fan-out `S` (the world for a
row-wise table, one node for a table-row-wise one):

```
G         = ceil(N / bc)                 # global buckets        -- invariant to S
C_g       = G * bc                       # global capacity, rows -- invariant to S
P         = ceil(G / S)                  # buckets per shard
shard_cap = P * bc                       # == max_capacity of the shard
h         = murmur3_fmix64(key) & INT64_MAX
g         = (h % C_g) / bc               # global bucket, in [0, G)
shard     = g / P   ==  (h % C_g) / shard_cap
local     = g % P   ==  g - shard * P    # bucket inside the shard's table
iter      = h % bc                       # start slot, unchanged
```

Nothing in `g` depends on `S`. The keys that share global bucket `g` share it at
every fan-out, and shard `r` holds the contiguous range `[r·P, (r+1)·P)`. A
bucket that fit when it was written fits when it is read back: a checkpoint
written by 8 shards is loaded by 1, 2 or 3 with **no bucket overflowing**.

Two identities make this a layout change and not a sizing change:

* `ceil(ceil(N/bc)/S) == ceil(ceil(N/S)/bc)`, so `P` is exactly the per-rank
  bucket count the planner computed before, and `max_capacity`, memory, and
  `get_sharded_table_capacity` are unchanged. Only the *sum* over shards
  changes: `S·P ≥ G`, the last shard owns `G - (S-1)·P` buckets (fewer than
  `P` by at most `S-1`), and when `G < S` the shards `r ≥ G` own none. Those
  surplus buckets exist but no key maps to them.
* When `C_g == table_cap` (a single shard, or any table without a global
  capacity) `g < P` and `g % P == g`, so the formula is bit-identical to the old
  one. `roundrobin`, `hash_roundrobin` and `continuous` tables, caches and
  hybrid tiers hash exactly as before; a changed partition would cost every
  near-full table a one-time loss, so they must not move.

`iter = h % bc` reuses the low part of `h % C_g` (as it reused the low part of
`h % table_cap` before, since `bc` divides both), so no new correlation between
the shard and the slot was introduced. The shard is a quotient, not a modulus,
so `gcd(S, bc)` plays no role.

## 3. Where the rule lives

The rule is written in three places, which must agree key for key:

| Twin | Where | Formula |
|---|---|---|
| Table kernels | `LinearBucketTable::home_bucket`, `src/table_operation/types.cuh`; used by lookup, insert, insert_and_evict, erase, `scatter_keys_at_slots` (`kernels.cuh`) and `BucketizeFunctor` (`bucketize.cu`) | `bkt_begin + ((h % global_cap[t]) / bc) % nb_t`, `global_cap[t] <= 0` meaning the table's own capacity |
| Routing kernel | `hashed_shard`, `src/sparse_block_bucketize_features.cu` (`DistType::kHashedBucket`) | `(h % global_capacity[t]) / shard_capacity[t]` |
| Host | `dynamicemb.key_ownership.owning_shard` (and `OwnershipRule`) | the same, in torch; raises without the parameters |

The table carries `global_capacities_` (`LinearBucketTable.__init__`,
`scored_hashtable.py`), one entry per logical table, 0 for "own capacity"; the
LruLfu eviction cubin receives it through a field **appended** to
`EvictParams` (`src/jit/evict_abi.cuh`) -- the struct is the launch contract,
so the extension and both fatbins must be rebuilt together.

The input distributors (`RwSparseFeaturesDist`, `TwRwSparseFeaturesDist`) carry
two int64 buffers next to the dist-type codes, `(global_capacity,
shard_capacity)` per feature (`routing_param_tensors`, `shard/input_dist.py`),
built once from the options the planner put into `fused_params`
(`routing_params_per_feature`, `shard/rw_sharding.py`) and validated on the host
so the per-batch kernel call has no checks and no sync. A `hash_bucket`
feature without them is refused, not routed by some other modulus.

## 4. Options and planner

`DynamicEmbTableOptions.global_capacity` (rows, `None` by default) is set by
`_prepare_dynemb_table_options` (`planner/planners.py`):

| `dist_type` | `bucket_capacity` | `global_capacity` |
|---|---|---|
| `roundrobin`, `hash_roundrobin`, `continuous` | any | `None` -- hashes within the shard, as before |
| `hash_bucket` | a multiple of 16 | `ceil(N / bc) * bc` (`get_global_table_capacity`) |
| `hash_bucket` | `MAX_BUCKET_CAPACITY` | `fanout * max_capacity`: one bucket per shard has no whole-table form, so the modulus is the shards laid end to end -- routed uniformly, **not** fan-out invariant |

A `hash_bucket` table with a global capacity is forced to
`init_capacity == max_capacity` (with a warning if the user set less): the
partition is only realised with exactly `P` buckets, and a rehash would
re-partition. `_expand_tables_impl` refuses to grow such a table.

`global_capacity` is **not** part of `get_grouped_key`: it is per table, and
fused tables keep their own (`G` differs with `N`). A fused group hashes every
table with `options[0].bucket_capacity`; a `global_capacity` derived with a
different one is refused in `create_table_state`.

## 5. Which storages are lossless

`create_table_state(..., global_bucket_layout=...)` decides whether a state
hashes into the global bucket space. It can only hold for a state that *is*
the whole shard, with exactly `P` buckets:

| Layout | Uses the global layout | Lossless across fan-outs |
|---|---|---|
| `DynamicEmbStorage`, HBM-only or host-only | yes | **yes** |
| CACHING: storage tier (host, full table) | yes | **yes** -- the cache is a subset and never the source of truth |
| CACHING: cache tier (`DynamicEmbCache`) | no (rescaled, `bc=1024`) | n/a |
| HYBRID: HBM and host tiers | no (each a rescaled part of the shard) | **no** -- see §8 |
| `NO_EVICTION` tables | no (key map is `2P` buckets at load factor 0.5) | n/a: they never evict |
| `MAX_BUCKET_CAPACITY` | no whole-table form | **no** |
| `roundrobin`, `hash_roundrobin`, `continuous` | no | **no** (unchanged) |
| External storage (RocksDB) | no buckets | yes, as before |

Lossless also requires the same `num_embeddings` and `bucket_capacity` on both
sides, and that the checkpoint was written under this layout (§6). Row-wise
and table-row-wise are interchangeable: `S` is the table's fan-out either way.

## 6. Checkpoints

`dump` records the layout in the table's meta (`_write_shard_meta`,
`batched_dynamicemb_tables.py`): `dist_type`, `global_capacity` (JSON `null`
when none) and `bucket_capacity`, from the **module-level** option, since a
tier's copy may have rescaled them.

`load` (`BatchedDynamicEmbeddingTablesV2.load`) compares them with the table's
(`_checkpoint_layout_matches`):

* **Match, same shard count:** each rank reads only its own shard file -- every
  key in it is its own. Fast, and exact.
* **Match, different shard count:** every rank reads every file and keeps its
  share. Exact: no bucket overflows.
* **Different `dist_type` or different layout:** every rank reads every file
  (`get_loading_files(..., read_all=True)`) and re-partitions the keys by the
  table's rule. Shard 0 warns once per table: rows may be lost where a bucket
  overflows. This is the migration path from a `roundrobin` or
  `hash_roundrobin` checkpoint to a `hash_bucket` table -- `_validate_load_meta`
  used to refuse a `dist_type` mismatch; it now warns, because ownership is
  recomputed from the key by the *table's* rule and the old check only ever
  protected the own-file shortcut. Since the rule is named by `dist_type`, no
  guessing is needed about which rule wrote a checkpoint.

The first restore of an old checkpoint into the invariant layout is therefore a
one-time lossy fold (the same loss as before); every save/restore after it is
lossless at any fan-out.

`replay_increment` writes by slot, so its layout check
(`_replay_compatibility`) also compares `global_capacity`; a delta from an older
version (no field) matches a target without one.

## 7. Load balance

Shard `r < S-1` owns `P` buckets, the last shard `G - (S-1)·P`, so the relative
imbalance is at most `(S-1)/P` -- for a 1 M-row table at `bc=128` and `S=8`,
shard 7 owns 974 of 7813 buckets against 977 for the others. For a table
smaller than `S` buckets (`G < S`, i.e. fewer than `S·bc` rows) the shards
`r ≥ G` receive no keys at all; such a table carries negligible traffic. An
interleaved assignment (`shard = g % S`) would balance to within one bucket at
the price of a non-contiguous bucket range per shard; the contiguous range was
chosen so that a shard's checkpoint file is one range of the global bucket
space.

## 8. Phase 2: lossless HYBRID (designed, not implemented)

A HYBRID key lives in one of two bucketed tiers (HBM, `bc=1024`, capacity
`max_cap·scale`; host, `bc`, capacity `max_cap·(1-scale)`), and which tier is
dynamic: a full HBM bucket spills its lowest-score key to the host tier.
`HybridStorage.load` re-inserts every key into HBM with `insert_and_evict`, so
the spilled subset differs from the original and host buckets overflow even if
the host partition were invariant -- the ~1.2 % loss per same-topology reload
measured above.

Making it lossless needs three pieces:

1. **A nested invariant HBM partition.** HBM bucket = a fixed group of
   `m = bc_hbm / bc` consecutive global host buckets (`g // m`), its width
   derived from `global_hbm_for_values` (not today's per-rank
   `int(max_cap·scale)`), and the per-shard `P` aligned up to a multiple of `m`
   so both buckets of a key sit on one rank.
2. **Tier membership in the checkpoint.** Per shard file, how many leading rows
   came from the HBM tier (dump already writes the tiers in order).
3. **A tier-respecting load.** Insert each row into its recorded tier with a
   plain `insert`, no cross-tier eviction; files without tier counts fall back
   to today's path.

Phase 1 leaves this open: `home_bucket` reads a per-table capacity pointer (an
HBM tier can carry its own), the meta is a dict, and `OwnershipRule` is the
single routing object.

## 9. Tests

* `test/unit_tests/test_global_bucket_layout.py` -- the table kernels place
  keys in their global bucket (and keep the old buckets when no global
  capacity is set); routing kernel, host rule and an independent oracle agree
  key for key; the planner derives the layout and keeps the per-shard size;
  **8→1, 1→2, 2→1 and 8→3 reloads of a 97 %-full table lose nothing** (HBM-only
  and CACHING), while the `roundrobin` control loses rows; a `roundrobin` or
  `hash_roundrobin` checkpoint, or one of another layout, is read whole,
  re-filtered and warned about.
* `test/unit_tests/test_hash_roundrobin_kuairand.py` -- the bucketize kernel
  against a second CPU implementation of all three hash-related rules.
* `test/unit_tests/test_embedding_dump_load.sh` -- multi-GPU dump/load of
  `hash_bucket` tables, including 2→1 and 1→2 (the `hash_roundrobin` cases
  are unchanged).
* `test/unit_tests/incremental_dump/test_replay_increment.py` -- replay refuses
  a `global_capacity` mismatch.
