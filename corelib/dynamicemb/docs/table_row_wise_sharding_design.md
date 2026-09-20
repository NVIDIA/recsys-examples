# Table-Row-Wise sharding for DynamicEmb · Design & Change Document

> Branch: `feat/dynamicemb-table-row-wise` (on top of `main` @ `97062d9`).
> Goal: let a DynamicEmb table be sharded `TABLE_ROW_WISE` (TRW) instead of
> `ROW_WISE` (RW), for `EmbeddingBagCollection` (pooled) only.
> Status: **design only** — nothing in this document has been implemented.

---

## 1. What TRW buys, and what it costs

RW spreads a table across all `world_size` ranks; the pooled output must then be
reduce-scattered across every rank, which means the reduction crosses the
network. TRW places the whole table on **one node** and row-shards it only
*within* that node: the reduction happens over NVLink, and only the already
pooled `dim_sum_per_node` crosses the network.

| | RW | TRW |
|---|---|---|
| Ranks holding a table | `world_size` | `local_size`, all on one node |
| Rows per rank | `rows / world_size` | `rows / local_size` (**N× more**, N = node count) |
| Cross-node output traffic | `O(B · D · world_size)` | `O(B · D)` |
| Input dist | bucketize(`world_size`) → KJT a2a | bucketize(`local_size`) → **staggered shuffle** → KJT a2a |
| Output dist | one global reduce-scatter | intra-node reduce-scatter + cross-node a2a |
| Process groups | global `pg` | `pg` + `intra_pg` + `cross_pg` |

The trade is explicit: **TRW trades N× per-rank capacity for 1/local_size of the
cross-node traffic.** For DynamicEmb this is sharper than for a static table,
because a DynamicEmb table is a *bounded* hash table with eviction — N× capacity
pressure per rank does not merely cost memory, it changes the eviction rate and
therefore accuracy. TRW is only worth doing when the table fits comfortably in
one node's budget and the cross-node all-to-all is a measured bottleneck.

---

## 2. Where RW lives today

```
plan       planner/planner.py:322-345   _dyn_emb_plan, sharding_type hardcoded ROW_WISE (:341)
           planner/planner.py:174-182   per-rank capacity = f(world_size)
sharding   shard/embeddingbag.py:62     RW -> RwPooledDynamicEmbeddingSharding
           planner/rw_sharding.py:183   subclass of torchrec RwPooledEmbeddingSharding
input      input_dist.py:199            bucketize(num_buckets=world_size) -> KJTAllToAll
kernel     src/sparse_block_bucketize_features.cu:215 / :292
lookup     batched_dynamicemb_compute_kernel.py -> BatchedDynamicEmbeddingBag
output     (inherited) torchrec RwPooledEmbeddingDist -- reduce-scatter
```

Two facts drive most of this document:

**(a) DynamicEmb tables bypass the TorchRec planner's placement decision.**
`DynamicEmbeddingShardingPlanner.__init__` builds `_dyn_emb_plan` directly
(`planner.py:322-345`) and `collective_plan` overwrites whatever the TorchRec
planner produced for those tables (`planner.py:376-384`). `sharding_type` is a
literal at `planner.py:341`; `ranks` is `range(world_size)` at `:343`. The
enumerator's `_filter_sharding_types` returning `[ROW_WISE]`
(`planner/enumerators.py:366`) is *not* what makes a table RW.

Consequence: **there is no "which node" decision anywhere in the system.** RW
never needed one. TRW cannot work without one. This is the single largest new
piece of work, and it is a planning problem, not a sharding problem.

**(b) Key ownership is recomputed from the key, in five separate places.**
A DynamicEmb table stores global keys in a hash table; there is no row index and
no shard metadata that means anything. Which rank owns a key is a *rule*, applied
independently wherever ownership matters:

| Site | Rule as written | dist_type aware? |
|---|---|---|
| `src/sparse_block_bucketize_features.cu:250/329` | `idx % my_size`, `murmur(idx) % my_size`, or block | yes |
| `batched_dynamicemb_tables.py:140` `owned_key_mask` | same three, in numpy | yes |
| `key_value_table.py:1792` (checkpoint load) | `keys % world_size == rank` | **no** |
| `scored_hashtable.py:1012` (counter load) | `keys % world_size == rank` | **no** |
| `construct_twin_module.py:532` | `indices % self._world_size == self._rank` | **no** |

Under RW the rule has one parameter (`world_size`). Under TRW it has three
(`dist_type`, `local_size`, and the table's node). Five independent copies of a
three-parameter rule, where a mistake is silent — keys land on the wrong rank,
nothing raises, accuracy drifts — is the main functional risk of this project.

---

## 3. How TorchRec's TWRW works end to end

Reference section for §4: everything below is what TRW does and RW does not.
Read from `~/projects/torchrec` @ `c5e147ff`.

Notation: `W` = world size, `L` = local size, `N = W / L` = node count. Rank `r`
is written `(n, l)` with `n = r // L`, `l = r % L`.

### 3.1 Construction

**Shard count is `L`, not `W`.** `ShardMetadata` describes how the *tensor* is
cut, not how many ranks exist; the shards must tile `[rows, dim]` exactly once.

| | shards | each | tiles `rows`? |
|---|---|---|---|
| TW | 1 | whole table | yes |
| RW | `W` | `rows/W` | yes |
| TRW | **`L`** | `rows/L` | yes |

`W` entries would describe an RW layout, not a TRW one. Three places agree:

```python
# enumerators.py:194 -- divisor is local_world_size, so L entries
return _calculate_rw_shard_sizes_and_offsets(rows, local_world_size, columns)

# sharding_plan.py:815
assert len(size_and_offsets) <= local_size

# twrw_sharding.py:203 -- indexes shards with rank_idx in [0, L)
rank_idx = rank - (table_node * local_size)
local_rows = shards[rank_idx].shard_sizes[0]
```

The node identity is **not** in the shard count. It is in each
`ShardMetadata.placement` and in `param_sharding.ranks`, from which `_shard`
recovers `table_node = ranks[0] // L`. Under RW the two questions ("how many
pieces" and "on which ranks") happen to have the same answer, which is why the
shard count is easy to mistake for `world_size`.

Direct consequence for us: a TRW table's `ShardedTensor` state_dict has `L`
shards, so its checkpoint has `L` files — and `find_files` today requires
`len(files) == world_size` (§4.5, §8.1).

**Two extra process groups.** `intra_and_cross_node_pg` (`comm.py:164`):

```
intra,  node n:    [n*L, n*L+1, ..., n*L+L-1]
cross,  index l:   [l, l+L, l+2L, ..., l+(N-1)*L]
```

`_INTRA_PG` / `_CROSS_PG` are module-level singletons, reused by every TWRW
sharding after the first. Creating them loops `dist.new_group` `group_count`
times and ends with `dist.barrier()` (`comm.py:209-219`) — **every rank must
execute the whole loop**, including ranks on nodes holding none of the tables.

**`_shard` writes one table into `L` ranks** (`twrw_sharding.py:194-210`), each
taking `shards[rank_idx]`. RW's equivalent loops over all `W` ranks taking
`shards[rank]`; the row-cutting logic is identical, only the rank set shrinks.

**Two views of the grouped configs.** `_grouped_embedding_configs_per_rank` for
lookup (`create_lookup` takes `[self._rank]`), and
`_grouped_embedding_configs_per_node` — one head rank per node — for
`embedding_dims` / `embedding_names` / `feature_names` / `_dim_sum_per_node` /
`_emb_dim_per_node_per_feature`. The `L` ranks of a node hold the *same tables*
with the same dims and features and differ only in rows, so the output side's
dimensions must be computed per node, never per rank. RW has no such split.

**What the three submodules get:**

| | |
|---|---|
| `create_input_dist` | `TwRwSparseFeaturesDist(pg=global, local_size=L, features_per_rank, feature_hash_sizes)` |
| `create_lookup` | `GroupedPooledEmbeddingsLookup(per_rank[rank], sharding_type=TABLE_ROW_WISE)` |
| `create_output_dist` | `TwRwPooledEmbeddingDist(rank, cross_pg, intra_pg, dim_sum_per_node, emb_dim_per_node_per_feature)` |

`sharding_type` reaching the lookup has **exactly one effect**:
`pooling_type_to_pooling_mode` (`embedding_configs.py:113-119`) downgrades `MEAN`
to `SUM` for `ROW_WISE` and `TABLE_ROW_WISE`, because the cross-rank sum is not
finished yet; the real mean is an EBC output callback
(`embeddingbag.py:1961`, `_apply_mean_pooling`). `TABLE_ROW_WISE` is already in
that list, so passing it preserves today's behaviour.

The RS / A2A modules inside `TwRwPooledEmbeddingDist` are built **lazily on the
first forward** (`_create_output_dist_modules`), because choosing between the
fixed-batch and variable-batch variants needs `sharding_ctx`.

### 3.2 Forward

#### (a) bucketize into `L` buckets, not `W`

```python
feature_block_sizes = [ceil(hash_size / local_size) for ...]
bucketize_kjt_before_all2all(features, num_buckets=local_size, ...)
```

Output KJT keys are the input keys repeated `L` times, laid out
`[bucket][all features]`, index `bucket * num_features + feature`.

#### (b) staggered shuffle — the step RW does not have

```python
# twrw_sharding.py:439
return [bucket * num_features + feature
        for node in range(nodes)
        for bucket in range(L)
        for feature in range(node_offsets[node], node_offsets[node + 1])]
```

`KJTAllToAll` requires the send buffer to be contiguous per destination rank.
RW's splits are `[num_features] * W` — uniform, and the bucketized layout is
already ordered by destination. TRW's splits are `features_per_rank` —
non-uniform, and destination `(n, l)` wants *node n's features × bucket l*, so
the layout must be permuted to `[node][bucket][that node's features]`.

Worked example: 3 features, 2 nodes x 2 GPUs (`W=4, L=2, N=2`), `f0`/`f1` on
node 0, `f2` on node 1, `features_per_rank = [2, 2, 1, 1]`:

```
after bucketize:   f0 f1 f2 | f0 f1 f2          (bucket 0 | bucket 1)
permute:           [0, 1, 3, 4, 2, 5]
after permute:     f0 f1 | f0 f1 | f2 | f2
destination:       rank0   rank1   rank2  rank3
                   (n0,l0) (n0,l1) (n1,l0)(n1,l1)
splits:            [2,      2,      1,     1]     -- matches
```

#### (c) `KJTAllToAll(splits=features_per_rank, stagger=N)`

Two-stage awaitable: one a2a for output splits and `batch_size_per_rank`, then
one for the tensors. The whole forward runs under `torch.no_grad()`
(`dist_data.py:1240`).

`stagger` feeds `_get_recat(local_split, num_splits=W, stagger=N)`
(`dist_data.py:219`), which reorders the **received** blocks:

```python
feature_order = [x + (W // stagger) * y for x in range(W // stagger) for y in range(stagger)]
#              = [0, L, 2L, ..., 1, 1+L, ...]
recat = [i + j * local_split for i in range(local_split) for j in feature_order]
```

For the example above, `_get_recat(2, 4, 2) == [0, 4, 2, 6, 1, 5, 3, 7]`, which
reorders sources `src0, src1, src2, src3` into `src0, src2, src1, src3` — that
is, **grouped by the source's local index `l_s` first, then by its node**, not
by global rank.

This ordering is the sole precondition that makes the two-stage output work.

#### (d) lookup

Rank `(n, l)` now holds: all of node `n`'s features, bucket `l` only, over a
batch dimension of the **whole global batch** `B_g` in `(l_s, n_s)` order. TBE
reads its `1/L` of the rows and produces `[B_g, dim_sum_per_node[n]]`, with
`MEAN` tables computed as `SUM`.

#### (e) two-stage output

**Intra-node reduce-scatter** on `intra_pg`. The ids of one `(feature, sample)`
are spread over the node's `L` ranks, each producing a partial pool; RS sums
those `L` partials and splits the batch into `L` chunks:

```
[B_g, dim_sum_per_node[n]]  --RS-->  [B_g / L, dim_sum_per_node[n]]
```

Rank `l` receives the chunk whose sources had local index `l` — exactly what the
recat arranged. Unequal batches go through `reduce_scatter_v_pooled`.

**Cross-node all-to-all** on `cross_pg = {(g, l) : g in nodes}`, with
`dim_sum_per_rank = dim_sum_per_node`:

```
[B_g / L, dim_sum_per_node[n]]  --A2A-->  [B_local(n,l), sum_g dim_sum_per_node[g]]
```

Consistency check: rank 0's `cross_pg` is `{0, 2} = {(n0,l0), (n1,l0)}`, and
after the RS it holds exactly the batches of `src0` and `src2` — its own cross
group's batches.

Then the EBC mean-pooling callback, if any table pools by mean.

#### (f) variable batch

`sharding_ctx.batch_size_per_rank` comes from the KJT a2a's `_stride_per_rank`
(`embedding_sharding.py:768`) in **global rank order**, so
`TwRwPooledEmbeddingDist` must re-sort it to match the recat:

```python
for local_rank in range(L):
    for node in range(N):
        batch_size_per_rank[local_rank + node * L]
```

yielding `batch_size_sum_by_cross_group` (the RS `input_splits`) and
`batch_size_per_rank_by_cross_group[l]` (the cross a2a's batch sizes). None of
this machinery (`twrw_sharding.py:590-648`) is exercised by the RW path.

### 3.3 Backward

| Stage | Forward | Backward |
|---|---|---|
| input dist | bucketize + permute + KJT a2a | **none** — runs under `no_grad`, indices carry no gradient |
| cross A2A | `alltoall_pooled` on `cross_pg` | recat grad, a2a back, then `div_(N)` (`comm_ops.py:1657`, `:1565`) |
| intra RS | `reduce_scatter_*_pooled` on `intra_pg` | `all_gather_into_tensor` (or `all_gather` when uneven), then `div_(L)` (`comm_ops.py:2518`, `:2475`) |
| lookup | TBE forward | TBE fused backward, in-place weight update, no dense grad |

**The gradient-division accounting must balance.** `GRADIENT_DIVISION` defaults
to `True` (`comm_ops.py:83`), commented "Make it equivalent to running on a
single rank":

| | backward collective(s) | total divisor |
|---|---|---|
| RW | one all-gather | `÷ W` |
| TW | one a2a | `÷ W` |
| **TRW** | all-gather(intra) + a2a(cross) | `÷ L × ÷ N` = **`÷ W`** |

All three agree. **If either TRW collective is ever replaced with a custom
implementation, the factor must be preserved** — getting it wrong scales every
sparse gradient by `L` or `N` and raises nothing; it looks like a mis-set
learning rate.

Every collective is split into a `Req` / `Wait` autograd `Function` pair around
an `async_op=True` NCCL call, in both directions, so the backward all-gather and
a2a overlap with the TBE backward.

### 3.4 What this leaves DynamicEmb to do

**Reusable as-is:** the staggered shuffle, `KJTAllToAll`'s stagger/recat, both
output collectives, the entire backward, and the gradient-division accounting.
All of it lives in TorchRec and is not DynamicEmb-aware.

**Must be replaced (two methods):** `create_input_dist`, to use DynamicEmb's
bucketizer with `num_buckets = L`; and `create_lookup`, to reach
`BatchedDynamicEmbeddingBag`. The CUDA bucketizer itself needs no change (§4.3).

**Must be written:** the staggered shuffle. DynamicEmb's `RwSparseFeaturesDist`
has no equivalent, and it is not optional — without it the a2a splits do not
line up (§4.4).

**Must be passed correctly:** `sharding_type=TABLE_ROW_WISE`, for the MEAN→SUM
downgrade in §3.1.

**Must be verified:** the variable-batch path in §3.2(f). The recat order,
`_preprocess_batch_size_per_rank`, and DynamicEmb's
`stride_per_key_per_rank * num_buckets` (`input_dist.py:62-69`, where
`num_buckets` becomes `L`) must all agree. Each can be individually right and
the combination wrong, and the symptom is mismatched samples, not an error (F4).

---

## 4. Change list

### 4.1 Placement: a node decision that does not exist yet

`planner.py:322-345` must produce, for a TRW table:

- `sharding_type = TABLE_ROW_WISE`
- `ranks = [node * local_size + i for i in range(local_size)]`
- `local_size` shard metadata entries, not `world_size`

and something must choose `node`. Two options:

1. **User-specified** (`DynamicEmbTableOptions.host_index`), mirroring TorchRec's
   `table_row_wise(host_index=...)`. Simple, predictable, testable.
2. **Automatic** — greedy bin-packing of tables over nodes by `max_capacity`.
   Needs a cost notion DynamicEmb tables currently do not participate in.

**Recommendation: ship (1) first.** (2) can follow once TRW is proven; it is a
separate problem and mixing it in makes the first version hard to debug.

### 4.2 Capacity: the divisor becomes per-table, and it is computed too early

`_prepare_dynemb_table_options` (`planner.py:124`, per-table loop at `:162-210`) computes, for every
DynamicEmb table:

```python
num_buckets, eff = _sharded_table_bucket_layout(cfg, world_size, opts.bucket_capacity)
opts.max_capacity = num_buckets * eff                       # rows per rank
opts.local_hbm_for_values = ceil(opts.global_hbm_for_values / world_size)
```

with `_sharded_table_bucket_layout` doing `shard_rows = ceil(num_embeddings /
world_size)` (`dynamicemb_config.py:867`).

For a TRW table both divisors must be `local_size`. Two problems:

- The divisor becomes **per table** (RW tables keep `world_size`), so the helper
  needs the table's sharding type, not a global world size.
- `_prepare_dynemb_table_options` runs at `planner.py:271`, **before** any
  placement is known. Under option (1) above this is fine (`host_index` is user
  input, available up front). Under option (2) the ordering inverts: placement
  must run first, because capacity depends on it. Another reason to ship (1)
  first.

Also audit `get_sharded_table_capacity` (`dynamicemb_config.py:889`) and every
caller in tests/examples that divides by world size.

### 4.3 Sharding class

New `TwRwPooledDynamicEmbeddingSharding(TwRwPooledEmbeddingSharding)` in
`planner/tw_rw_sharding.py`, overriding exactly two methods, mirroring what
`RwPooledDynamicEmbeddingSharding` does today:

- `create_input_dist` → DynamicEmb's bucketizer with `num_buckets = local_size`
- `create_lookup` → `GroupedPooledEmbeddingsLookup` with
  `sharding_type=ShardingType.TABLE_ROW_WISE`, so `CUSTOMIZED_KERNEL` still maps
  to `BatchedDynamicEmbeddingBag`

`create_output_dist` is **not** overridden — TorchRec's `TwRwPooledEmbeddingDist`
(reduce-scatter + all-to-all) is used as-is, exactly as RW uses
`RwPooledEmbeddingDist` as-is.

Dispatch in `shard/embeddingbag.py:62` gains a `TABLE_ROW_WISE` branch.

### 4.4 Input dist: a new full-data permute

DynamicEmb's `RwSparseFeaturesDist` (`input_dist.py:199`) has no equivalent of
TorchRec's `_staggered_shuffle` (`twrw_sharding.py:439`), because RW's a2a splits
are uniform (`[num_features] * world_size`) and the bucketized layout is already
ordered by destination rank.

TRW's splits are `features_per_rank` — non-uniform, and a rank only receives the
features of tables on its own node. The bucketized layout `[bucket][all features]`
must be permuted into `[node][bucket][that node's features]` before the a2a, and
the a2a itself takes `stagger=world_size // local_size`.

This permute is **new work RW does not do**: a gather over the whole value
tensor, every step. See §6.

### 4.5 Checkpoint layout

Today (`batched_dynamicemb_tables.py:84-98`) a table's files are named
`{table}_emb_{item}.rank_{r}.world_size_{ws}`, and `find_files` (`:102`) parses
`ws` out of the filename and **requires `len(files) == ws`** (`:124-128`).

A TRW table has `local_size` shards, not `world_size`. Whatever is written today
either breaks that check or lies about it. Proposal:

- Name shards by **shard index** `0..num_shards-1`, not global rank.
- Record `sharding_type`, `num_shards`, `local_world_size` and `dist_type` in the
  per-table meta json (`{table}_opt_args.json`, `encode_meta_json_file_path`).
- `find_files` reads the count from meta instead of parsing a filename, and
  refuses a checkpoint whose meta it does not understand.

This is versioned-format work. It is also the natural place to fix the fact that
**the ownership rule is currently implicit in the checkpoint** — see §8.1.

### 4.6 Load / replay / twin-module ownership

All five sites in §2(b) must move to **one shared function**:

```python
def owning_rank(keys, *, dist_type, num_shards, shard_base) -> Tensor
# RW:  num_shards = world_size, shard_base = 0
# TRW: num_shards = local_size,  shard_base = node * local_size
```

`batched_dynamicemb_tables.py:140` `owned_key_mask` is already the closest thing
to this and already handles `dist_type`; it should become the single
implementation and the other four sites should call it.

### 4.7 Incremental dump / replay

`meta["world_size"]` (`incremental_dump.py:110`) means "the source's row-wise
fan-out, the modulo base for key → rank". Under TRW that is `local_size`, and the
key→rank mapping additionally needs the node. `_replay_compatibility`
(`incremental_dump.py:511`) compares it for equality, which is correct in spirit
— replay does not reshard — but the field needs to carry enough to describe a TRW
layout. Recommend replacing the single `world_size` field with the same triple
the ownership function takes, and rejecting any delta whose triple differs.

### 4.8 Explicitly out of scope: EmbeddingCollection (sequence)

TorchRec has no `twrw_sequence_sharding.py`, and
`EmbeddingCollectionSharder.sharding_types` (`torchrec/distributed/embedding.py:1858`)
does not list `TABLE_ROW_WISE`. TRW for sequence embeddings would mean writing a
TwRw sequence sharding from scratch (the output side is an all-gather, not a
reduce-scatter). Not in this project. The planner must therefore **reject**
`TABLE_ROW_WISE` for any DynamicEmb table in an `EmbeddingCollection`, with a
clear error rather than a fallback.

---

## 5. Files to change

| File | Change | Size |
|---|---|---|
| `dynamicemb_config.py` | `host_index` (or equivalent) in `DynamicEmbTableOptions`; `_sharded_table_bucket_layout` takes a per-table divisor | S |
| `planner/planner.py` | TRW branch in `_dyn_emb_plan`; per-table capacity divisor; ordering of `_prepare_dynemb_table_options` | M |
| `planner/enumerators.py` | `_filter_sharding_types` must allow TRW; the `[[1,1]]*world_size` fake shard becomes `local_size` entries | S |
| `planner/tw_rw_sharding.py` | **new** — `TwRwPooledDynamicEmbeddingSharding` | M |
| `shard/embeddingbag.py` | dispatch TRW | S |
| `input_dist.py` | `TwRwSparseFeaturesDist` with staggered shuffle; hoist the dist_type tensor into `__init__` | M |
| `batched_dynamicemb_tables.py` | shard-index file naming; `find_files` reads meta; `owned_key_mask` generalized | M |
| `key_value_table.py` | load mask → shared ownership fn (also fixes §8.1) | S |
| `scored_hashtable.py` | counter load mask → shared ownership fn | S |
| `construct_twin_module.py` | twin mask → shared ownership fn | S |
| `incremental_dump.py` | meta carries the ownership triple | S |
| `src/sparse_block_bucketize_features.cu` | none required | — |

The CUDA bucketizer needs **no change**: it is already parameterized by
`my_size`, and `roundrobin` / `hash_roundrobin` already preserve the global key
(`new_idx = idx`). Passing `local_size` is enough.

---

## 6. Performance risks

**R1 — N× per-rank capacity (inherent, and the reason to measure first).**
`max_capacity` goes from `rows/world_size` to `rows/local_size`. On 8 nodes ×
8 GPUs that is 8× more rows per rank. For a hash table with eviction this shows
up as a higher eviction rate, i.e. **accuracy**, not an OOM. It may also push a
table out of HBM into the host tier and change lookup latency. Quantify before
building: the cross-node saving has to beat this.

**R2 — the staggered shuffle is new per-step work.** §4.4. RW has no equivalent.
It is a full gather over the bucketized values every step, on the critical path,
*before* the a2a it is meant to enable. Needs to be measured against the traffic
it saves, not assumed free.

**R3 — two collectives replace one.** RW output = one reduce-scatter. TRW = a
reduce-scatter plus an all-to-all, each with its own launch and sync cost. At
small batch sizes the fixed cost of the second collective can eat the bandwidth
win outright.

**R4 — intra-node skew is worse than global skew.** Keys spread over
`local_size` (typically 8) instead of `world_size` (typically 64). Modulo 8 is
far more exposed to structured IDs than modulo 64. `hash_roundrobin` becomes
close to mandatory for TRW rather than merely advisable.

**R5 — node-level hotspots.** A table lives entirely on one node, so an uneven
table→node assignment makes one node the bottleneck for every step. RW has no
such failure mode. With user-specified `host_index` (§4.1) this is the user's
problem, but the planner should at least warn on a visibly lopsided assignment.

**R6 — extra NCCL communicators.** `intra_and_cross_node_pg` creates two more
process groups, on top of whatever DynamicEmb already holds.

**R7 — bucketize atomics.** `kernel1` atomically increments
`new_lengths_data[p * lengths_size + b_t]`; shrinking `p`'s range from
`world_size` to `local_size` concentrates those atomics. Likely small next to R2,
but it moves the wrong way.

---

## 7. Functional risks

**F1 — the ownership rule is replicated five times and fails silently.** §2(b).
If one site is not updated, keys go to the wrong rank, nothing raises, and the
symptom is a slow accuracy regression. **Mitigation: collapse the rule into one
function *before* starting TRW**, so the TRW change touches one place.

**F2 — checkpoint compatibility.** Every existing checkpoint is RW. `find_files`
will report "Checkpoints is corrupted" for a TRW checkpoint under today's rule.
The meta format needs a version and a refusal path, not a silent reinterpretation.

**F3 — resharding.** RW→TRW, or a change of `local_size`, is a reshard.
`replay_increment` already rejects reshards. The dump/load reshard path (file
count ≠ world size, `batched_dynamicemb_tables.py:265-273`) only implements the
roundrobin rule. **Recommend: reject explicitly with a clear message in v1**
rather than implementing TRW reshard.

**F4 — variable batch (VBE).** `TwRwPooledEmbeddingDist` carries a whole
`_preprocess_batch_size_per_rank` / `_preprocess_batch_size_per_rank_per_feature`
machinery (`twrw_sharding.py:590-648`) that the RW path never exercises. Mean-
while DynamicEmb's bucketizer multiplies `stride_per_key_per_rank` by
`num_buckets` (`input_dist.py:62-69`), which under TRW is `local_size`, not
`world_size`. These two must agree. This is the easiest place to be subtly wrong
and the hardest to cover with a test.

**F5 — mixed sharding types in one collection.** An EBC may hold RW tables, TRW
tables and plain TorchRec tables. `create_embedding_bag_sharding` dispatches on
`sharding_infos[0]` (`shard/embeddingbag.py:60`) — already a per-group decision,
but worth re-checking that grouping never mixes RW and TRW into one sharding.

**F6 — EC must be rejected, not silently downgraded.** §4.8.

**F7 — the fake shard metadata gains a third variant.** `planner.py:332-335`
claims contiguous row ranges; `get_state_dict`
(`batched_dynamicemb_compute_kernel.py:97-105`) rewrites them to `[1,1] @ [i,0]`;
TRW adds a node dimension. Recommend collapsing the first two into one generator
before adding the third.

**F8 — the gradient-division factor must stay at `÷ W`.** §3.3. TRW reaches it
as `÷ L` (intra all-gather) times `÷ N` (cross a2a), where RW gets there in one
step. Reusing TorchRec's collectives unchanged keeps this correct for free; any
custom replacement of either stage must preserve its factor. Getting it wrong
scales every sparse gradient by `L` or `N`, raises nothing, and presents as a
mis-set learning rate.

**F9 — process-group creation is collective.** `intra_and_cross_node_pg` loops
`dist.new_group` once per node group and ends in `dist.barrier()`
(`comm.py:209-219`). Every rank must run the whole loop, including ranks on
nodes that hold none of the TRW tables. A code path that builds the sharding
only where a table lives will hang.

---

## 8. Pre-existing defects found while writing this

### 8.1 Checkpoint load drops keys for `hash_roundrobin` tables

`_iter_batches_from_files` (`key_value_table.py:1715`) applies, unconditionally
when `world_size > 1`:

```python
masks = keys % world_size == rank
```

This is the **roundrobin** rule. For a `hash_roundrobin` table, rank `r`'s
checkpoint file holds keys satisfying `murmur(k) % ws == r`, and the mask keeps
only those that *also* satisfy `k % ws == r` — roughly `1/ws` of them. The rest
are silently discarded on load.

Reachable in the ordinary matched-file path: `get_loading_files`
(`batched_dynamicemb_tables.py:237`) hands rank `r` its own file when
`world_size == num_key_files`, and the mask is still applied.

Not covered by tests: the `hash_roundrobin` dump/load smoke runs with
`--nproc_per_node 1` (`test/unit_tests/test_embedding_dump_load.sh:42,58`), where
the mask is skipped entirely; the multi-GPU dump/load tests use the default
`roundrobin`, where the mask is a no-op.

`scored_hashtable.py:1012` has the same line with the same problem for the
admission counter.

> Derived by reading, not by running — confirm with a 2-GPU
> dump/load of a `hash_roundrobin` table before fixing.

Fixing this is a prerequisite for TRW anyway (§4.6), and it is worth its own
commit and its own test regardless of whether TRW proceeds.

### 8.2 `continuous` is the silent fallback for a missing `dist_type`

`planner/rw_sharding.py:126` and `:227` default a feature with no `dist_type` in
its `fused_params` to `"continuous"`. Correct for plain TorchRec tables; for a
DynamicEmb table it would silently select a mode that rewrites indices
(`new_idx = idx % blk_size`) and that `incremental_dump` refuses
(`DynamicEmb_APIs.md:808`). Should raise for DynamicEmb tables.

### 8.3 The "same dist_type for all tables" assertion is unnecessary

`planner/rw_sharding.py:119-124` / `:221-226` forbid two DynamicEmb tables in one
sharding from using different `dist_type`s. The kernel is per-feature
(`dist_type_per_feature[t]`); the restriction has no basis.

### 8.4 The `dist_type` tensor is rebuilt every forward

`input_dist.py:117-132` rebuilds a per-feature int32 tensor on every step from a
Python loop over `kjt.keys()`, while the sibling `_feature_block_sizes_tensor` is
a constructor-time buffer. Both are per-feature constants in the same order. It
should be a buffer; doing so also removes an unguarded ordering assumption
between the two arrays, which the kernel indexes with the same `t`.

---

## 9. Suggested staging

| Milestone | Content | Gate |
|---|---|---|
| **M0** | §8.1–§8.4. One ownership function (§4.6), all five call sites on it. 2-GPU `hash_roundrobin` dump/load test. | No TRW code yet; ships on its own merit |
| **M1** | Measure R1: per-rank capacity and eviction rate at `local_size` vs `world_size` divisor, on a real table | **Go/no-go for the whole project** |
| **M2** | Placement + capacity (§4.1, §4.2) with user-specified `host_index`. Plan is TRW-shaped; nothing consumes it yet | Plan inspection test |
| **M3** | `TwRwPooledDynamicEmbeddingSharding` + staggered-shuffle input dist (§4.3, §4.4). **TRW tables reject dump/load/incremental-dump with a clear error** | Numerical parity vs RW on a small model |
| **M4** | Checkpoint (§4.5), ownership under TRW (§4.6), incremental dump (§4.7) | Dump→load round-trip across TRW |
| **M5** | Perf validation: R2, R3 measured against the cross-node saving | Beat RW on the target topology, or stop |

M1 before M2 is deliberate: R1 is inherent to TRW and cannot be engineered away.
If N× capacity pressure is unacceptable for the target tables, nothing after it
matters.

---

## 10. Open questions

1. **Target topology and table sizes?** TRW only pays off when cross-node a2a is
   a measured bottleneck *and* the table fits one node. Both need numbers.
2. **Must TRW tables checkpoint in v1?** If not, M4 can be deferred and the
   project is roughly half the size.
3. **`host_index` by hand, or automatic packing?** This document assumes by hand
   (§4.1). Automatic packing inverts the ordering in §4.2 and needs a cost model
   DynamicEmb tables do not currently participate in.
4. **Is `hash_roundrobin` acceptable as the TRW default?** R4 argues it is close
   to required. It changes the key→rank mapping, so it is a reshard for any
   existing table.
