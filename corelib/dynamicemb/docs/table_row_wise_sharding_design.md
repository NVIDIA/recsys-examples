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
settle     planner/planners.py:305      _settle -- options filled, _dyn_emb_plan built
           planner/planners.py:357      sharding_type hardcoded ROW_WISE
           planner/planners.py:182-190  per-rank capacity and HBM budget = f(world_size)
budget     planner/plan.py:119,176      per_rank_storage -> topology_minus
sharding   shard/embeddingbag.py:63     RW -> RwPooledDynamicEmbeddingSharding
           shard/rw_sharding.py:224     subclass of torchrec RwPooledEmbeddingSharding
input      shard/input_dist.py:227      bucketize(num_buckets=world_size) -> KJTAllToAll
kernel     src/sparse_block_bucketize_features.cu:217 / :299
lookup     batched_dynamicemb_compute_kernel.py -> BatchedDynamicEmbeddingBag
output     (inherited) torchrec RwPooledEmbeddingDist -- reduce-scatter
```

### 2.1 How the planner works now

`DynamicEmbeddingShardingPlanner` subclasses TorchRec's
`EmbeddingShardingPlanner`, and plans in four steps:

1. **Plan the DynamicEmb tables**, producing a `ParameterSharding` for each.
2. **Modify the planner's Topology** to deduct what those tables cost.
3. **Plan the remaining tables** -- TorchRec's own planning, over what is left.
4. **Assemble both halves** into one `ShardingPlan`.

```python
def plan(self, module, sharders):                  # one rank
    self._plan_dynamicemb(module, sharders)                       # 1
    return self._attach_options(self._plan_torchrec(module, sharders))   # 2-4

def collective_plan(self, module, sharders, pg):   # all of them
    self._plan_dynamicemb(module, sharders)                       # 1, every rank
    plan = invoke_on_rank_and_broadcast_result(
        pg, 0, self._plan_torchrec, module, sharders               # 2-4, rank 0
    )
    return self._attach_options(plan)                              # every rank
```

`__init__` splits the constraints and does nothing else: the base is handed only
the TorchRec half. There is no module and no sharders yet, so nothing that
depends on the optimizer can be computed there.

**Step 1 -- `_plan_dynamicemb`.** Reads the table configs off the module
(`_table_configs`, which is what the `eb_configs` argument used to carry), fills
each DynamicEmb table's options in place (`_prepare_dynemb_table_options`:
initializer bounds, `max_capacity` via `_sharded_table_bucket_layout`,
`local_hbm_for_values`, `dim`, `index_type`, `embedding_dtype`), and writes one
`DynamicEmbParameterSharding` per table -- `ROW_WISE`, `world_size` shards of
`[max_capacity, dim]`, placement by rank. Nothing is searched: capacity comes
from the options and placement from the key -> rank rule.

It runs once per planner, and on every rank rather than inside the broadcast,
because the options it fills are the half of a DynamicEmb table that does not
travel -- `external_storage` is a live handle, `score_function` a callable --
and step 4's reattachment needs them everywhere.

**Steps 2 to 4 -- `_plan_torchrec`,** on rank 0 alone under `collective_plan`:

```python
spent = per_rank_storage(self._dyn_emb_plan, optimizer_types(...), n)   # 2
try:
    self._topology = topology_minus(whole_topology, spent)              # 2
    reduced_module = module_without_tables(module, table_names, sharders)
    torchrec_plan = super().plan(reduced_module, sharders)              # 3
finally:
    self._topology = whole_topology
self._insert_dynamicemb_plan(torchrec_plan, module, sharders)           # 4
```

- **2.** Optimizer types come from the sharders' `fused_params` or the
  parameters' `_optimizer_classes`, and each shard's cost is charged to the rank
  it sits on. Row-wise makes those equal; TRW will not (§4.2). `self._topology`
  is assigned rather than passed, because the enumerator and the estimators were
  built against it in `__init__`; it is restored afterwards, so a second `plan`
  call starts from what the machine has rather than from what the first one left.
- **3.** TorchRec plans the rest of the model, on a copy of the module with the
  DynamicEmb tables taken out. Only the modules on the path to a pruned
  collection are copied, and the collection keeps its type and its shape, so the
  plan's paths are the caller's paths -- which is what `ShardingPlan` is keyed
  by. The caller's `storage_reservation`, `enumerator`, `proposer`, `partitioner`
  and `stats` all apply here, unchanged.
- **4.** The walk is over the *caller's* module: the reduced one no longer admits
  to having these tables, so it cannot say what path they live under. What goes
  on the plan is a copy with the options off and the shard metadata deep-copied
  (F10). `_attach_options` then puts this rank's options back -- the decisions
  are broadcast, the configuration is not.

Three consequences worth stating plainly, because the rest of this document
assumed the older arrangement in places:

- A DynamicEmb table never enters TorchRec's search space, so nothing estimates
  or partitions it. There is no placeholder to keep honest.
- What those tables cost is out of the Topology before TorchRec sees it, per
  rank. Under TRW that is what lets `_cohost_partition` avoid a node DynamicEmb
  has already filled, without the two halves having to negotiate.
- Ranks cannot disagree about placement: rank 0 decides and the rest are told.

---

Two facts drive most of this document:

**(a) DynamicEmb tables bypass the TorchRec planner's placement decision.**
`_settle` builds `_dyn_emb_plan` directly (`planner/planners.py:305-362`) and
`_decide` takes those tables out of the module before TorchRec sees it, then
puts the entries back afterwards. `sharding_type` is a literal at
`planner/planners.py:357`; `ranks` is `range(world_size)` at `:358`.

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
# torchrec planner/enumerators.py:194 -- divisor is local_world_size, so L entries
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
`stride_per_key_per_rank * num_buckets` (`shard/input_dist.py:141`, where
`num_buckets` becomes `L`) must all agree. Each can be individually right and
the combination wrong, and the symptom is mismatched samples, not an error (F4).

---

## 4. Change list

### 4.1 Placement: a node decision that does not exist yet. Done (M2).

`_settle` (`planner/planners.py:342-363`) must produce, for a TRW table:

- `sharding_type = TABLE_ROW_WISE`
- `ranks = [node * local_size + i for i in range(local_size)]`
- `local_size` shard metadata entries, not `world_size`

and something must choose `node`. **Shipped: a component of its own**, so the
rule can be replaced without touching how a plan is assembled.

```python
class HostPlacer(abc.ABC):
    def place(self, tables, topology, committed) -> Dict[str, int]: ...
```

`planner/placement.py`. The planner takes one as `host_placer=` and defaults to
`BalancedHostPlacer`.

This is a bin-packing problem and not a search, which is why it can be its own
component and needs nothing from TorchRec's cost model: a DynamicEmb table's
size is *declared* rather than estimated, so what a node will owe is known
before anything is placed on it. An earlier draft of this section called that a
reason to defer automatic placement -- it is the opposite.

**`BalancedHostPlacer`** is first-fit-decreasing with two choices worth stating:

- *Fit against the tightest rank of a node, not the node's total.* A TRW table
  charges every rank of its node the same, so it is limited by the worst of
  them.
- *Emptiest node rather than first that fits.* TorchRec plans its own tables
  afterwards, into what this leaves, and has a node decision of its own to make
  for its `TABLE_ROW_WISE` and `GRID_SHARD` tables. Filling nodes in order would
  hand it a machine with nothing left on the low-numbered ones.

Its inputs are prepared by `_choose_hosts`: what each table costs one rank
(`get_local_value_bytes_by_tier`, so optimizer state is in it), and what the
**row-wise** DynamicEmb tables have already taken from every rank. Those shift
no choice between nodes -- they are on all of them -- but they decide whether a
TRW table still fits, so the placer is fitting into the room that will actually
be there.

It must be deterministic: it runs on every rank, and two ranks that disagreed
would shard one table onto two nodes. Ties break on the table name, then the
lower node index.

**`host_index` survives as a pin.** Set it and the placer honours it; leave it
unset and the placer chooses. Pinned tables are charged first, so unpinned ones
fit around them. This is how a caller who knows their layout for some tables
gets it without having to name every node.

`table_layout` (`planner/plan.py`) turns a sharding type and a node into the
ranks a table sits on, and `_plan_dynamicemb` writes one `ShardMetadata` per
rank in that list -- so a TRW table gets `local_size` entries and `ranks` is
that node's range. It rejects, rather than ignores, a `host_index` on a
row-wise table: such a table is on every rank, so naming a node for it means the
caller expected something the plan will not do.

The sharding type is read from `ParameterConstraints.sharding_types`, the field
TorchRec already has, rather than a DynamicEmb-only one; it must name exactly
one, since a DynamicEmb table is placed rather than searched for.

### 4.2 Capacity: the divisor becomes per-table, and it is computed too early. Done (M2).

`_prepare_dynemb_table_options` (`planner/planners.py:130`, per-table loop at
`:170-218`) computes, for every DynamicEmb table:

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
- `_prepare_dynemb_table_options` runs at the top of `_plan_dynamicemb`,
  **before** any placement is known.

**Both are done, and the second was not a problem.** The divisor is
`table_fanout(sharding_type, world_size, local_size)`: it depends on how big a
node is, not on *which* node, so capacity is settled before placement and the
ordering does not invert. That is what lets the placer size a table before
deciding where to put it. `table_layout` is the same arithmetic carried one step
further, so the number of pieces and the list of ranks cannot disagree.
`table_fanout` feeds `_sharded_table_bucket_layout` and the
`local_hbm_for_values` division alike.

`_prepare_dynemb_table_options` now takes `world_size` and `local_size` as
arguments instead of reading `dist`, and `_plan_dynamicemb` passes the
Topology's. That removes the planner's copy of the F11 defect: there is one
statement of how many ranks and how many nodes, and it is the Topology the
caller handed in.

Also audit `get_sharded_table_capacity` (`dynamicemb_config.py:889`) and every
caller in tests/examples that divides by world size.

**The deduction from the budget stops being uniform. Done.**
It used to be a `StorageReservation` subtracting `sum(local_hbm_for_values)`
from every device, which is right only because row-wise puts a slice of every
table on every rank. Under TRW a table's HBM lands on the `local_size` ranks of
one node and is zero everywhere else, so subtracting the same figure from every
device charges ranks that hold nothing -- the planner would then refuse plans
that fit. It fails loudly, but for a reason that is hard to guess from the
error.

`per_rank_storage` and `topology_minus` (`planner/plan.py`) now do it per rank,
off the plan's own shard metadata:

    for device in reduced.devices:
        device.storage -= per_rank[device.rank]

`Topology.devices` is a list of `DeviceHardware`, each with its own `rank` and
`storage`, so TorchRec can express this; nothing has needed it yet because
every reservation upstream subtracts a uniform quantity. What `_hbm_on_rank`
needs is which ranks hold which table, and that is exactly what the node
placement decision of §4.1 produces -- so this lands with §4.1, not after it.

### 4.3 Sharding class. Done (M3).

New `TwRwPooledDynamicEmbeddingSharding(TwRwPooledEmbeddingSharding)` in
`shard/twrw_sharding.py`, overriding exactly two methods, mirroring what
`RwPooledDynamicEmbeddingSharding` does today:

- `create_input_dist` → DynamicEmb's bucketizer with `num_buckets = local_size`
- `create_lookup` → `GroupedPooledEmbeddingsLookup` with
  `sharding_type=ShardingType.TABLE_ROW_WISE`, so `CUSTOMIZED_KERNEL` still maps
  to `BatchedDynamicEmbeddingBag`

`create_output_dist` is **not** overridden — TorchRec's `TwRwPooledEmbeddingDist`
(reduce-scatter + all-to-all) is used as-is, exactly as RW uses
`RwPooledEmbeddingDist` as-is.

Dispatch in `shard/embeddingbag.py:62` gains a `TABLE_ROW_WISE` branch.

**Shipped as written.** `shard/twrw_sharding.py` overrides exactly those two
methods, and the dispatch branch is one `elif`. `GroupedPooledEmbeddingsLookup`
is reused unchanged from the row-wise module -- it only overrides
`_create_embedding_kernel`, which is sharding-type agnostic.

`create_lookup` passes `sharding_type=TABLE_ROW_WISE`, which is not decoration:
it is what downgrades MEAN to SUM in the kernel (§3.1). Passing `ROW_WISE` would
be arithmetically identical today and wrong the moment TorchRec separates them.

`shard/embedding.py` **refuses** `TABLE_ROW_WISE` rather than letting it fall
through to `super()` (§4.8): a sequence output has nothing to reduce inside a
node, so the arrangement has no meaning there, and TorchRec's own sequence side
has no TwRw sharding to fall through to.

### 4.4 Input dist: a full-data permute, inherited. Done (M3).

DynamicEmb's `RwSparseFeaturesDist` (`shard/input_dist.py:175`) has no equivalent of
TorchRec's `_staggered_shuffle` (`twrw_sharding.py:439`), because RW's a2a splits
are uniform (`[num_features] * world_size`) and the bucketized layout is already
ordered by destination rank.

TRW's splits are `features_per_rank` — non-uniform, and a rank only receives the
features of tables on its own node. The bucketized layout `[bucket][all features]`
must be permuted into `[node][bucket][that node's features]` before the a2a, and
the a2a itself takes `stagger=world_size // local_size`.

This permute is **new work RW does not do**: a gather over the whole value
tensor, every step. See §6.

**It did not have to be written.** TorchRec's `TwRwSparseFeaturesDist` already
computes `_sf_staggered_shuffle` in its constructor and applies it between the
bucketize and the a2a, so DynamicEmb's subclass overrides `forward` for the one
call that differs -- the bucketizer -- exactly as the row-wise one does, and
inherits the shuffle.

That is not a lucky accident: the shuffle is arithmetic over *feature counts*,
not over keys, so it is correct whatever the bucketizer did with the values.
§3.4 listed it under "must be written"; that was wrong. What is real is the
cost, which §6 R2 already records: we pay for the gather, we just do not own it.

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

### 4.9 The inheritance boundary

Everything in §4 either extends TorchRec by **subclassing** or by **calling its
API**. Which one is not a matter of taste: it follows from the direction of the
call.

**Rule — subclass only where TorchRec calls us; use composition where we call
TorchRec.**

#### Seams where TorchRec calls us (subclassing is the only entry)

These are template-method hooks: a TorchRec base class invokes them from inside
its own `__init__` / construction flow. There is no registration table to put an
implementation into, so overriding is the only way in.

| Hook | Declared by | Our override |
|---|---|---|
| `ModuleSharder.shard()` | `distributed/types.py:1462` | `shard/embeddingbag.py:85`, `shard/embedding.py:355` |
| `create_embedding_bag_sharding()` (classmethod) | `distributed/embeddingbag.py:979` | `shard/embeddingbag.py:46` |
| `create_embedding_sharding()` (classmethod) | `distributed/embedding.py` | `shard/embedding.py:103` |
| `EmbeddingSharding.create_lookup()` | `distributed/embedding_sharding.py:1232` | `shard/rw_sharding.py:179,267` |
| `BaseEmbeddingLookup._create_embedding_kernel()` | `distributed/embedding_lookup.py:254,693` | `shard/rw_sharding.py:100,193` |
| `get_additional_fused_params()` (duck-typed) | `torchrec/distributed/utils.py:518-526` | `planner/planners.py:99` |

Plus one enum value, `EmbeddingComputeKernel.CUSTOMIZED_KERNEL`
(`embedding_types.py:98`).

**`CUSTOMIZED_KERNEL` is a skip-pass, not a dispatch table.** It appears in
exactly five places upstream -- `embedding.py:937,967`, `embeddingbag.py:1212,1240`,
`utils.py:523` -- and every one of them *skips* something: state_dict handling,
fused_params validation. The upstream comment says so directly: *"Skip state_dict
handling for CUSTOMIZED_KERNEL, this should be implemented in child class."*
The branch that actually selects our kernel is ours (`shard/rw_sharding.py:107`):

```python
if config.compute_kernel is not EmbeddingComputeKernel.CUSTOMIZED_KERNEL:
    return super()._create_embedding_kernel(...)      # hand back to TorchRec
return BatchedDynamicEmbeddingBag(config=config, pg=pg, device=device)
```

So the chain that ends at `BatchedDynamicEmbeddingTablesV2` is ours at every
step that matters; TorchRec supplies the scaffolding and the skip-pass, nothing
more.

This is a real cost, not a free win. `shard/rw_sharding.py:111-117` carries a
`Version(torchrec.__version__) < Version("1.5.0")` branch because
`_create_embedding_kernel` gained an `env` parameter. Six seams means six
signatures to track. The mitigation is to keep the count at six and to keep each
override thin -- a dispatch and a `super()` call, no copied logic -- not to try
to eliminate them.

#### Seams where we call TorchRec (composition)

Planning is the other direction: we drive, TorchRec answers. Nothing upstream
calls back into a planner subclass, so subclassing here buys nothing and costs
the same maintenance. These become wrapped, not inherited:

| Component | Today | Target |
|---|---|---|
| `EmbeddingEnumerator` | subclass, `enumerate()` post-filters | held as a field; call `enumerate()` and filter the result |
| `HeuristicalStorageReservation` | subclass | held as a field; call `reserve()` with a pre-reduced budget |
| `EmbeddingShardingPlanner` | subclass | held as a field; call `collective_plan()`, then merge |

`ParameterSharding` / `ParameterConstraints` stay subclassed even though they sit
on this side of the line: they are plain dataclasses carrying data across the
boundary, not behaviour, and `get_additional_fused_params` (above) is the
upstream-sanctioned way to move that data. Subclassing a dataclass to add
defaulted fields is upgrade-safe; `get_additional_fused_params` computes its
payload as a **field-set difference** (`planners.py:100-107`), so fields added
upstream are excluded automatically.

#### Why this is exactly the C2 split

Approach C2 -- DynamicEmb builds its own plan, reports the per-rank memory it
consumed, hands the reduced budget and the remaining tables to TorchRec's
planner, then merges the two plans -- moves work only across the **second**
table. Nothing in the first table changes. That is the argument for C2 beyond
the placement problem in §4.1: it is the half of the surface where the
dependency *can* be reduced to API level.

---

## 5. Files to change

Paths are as of the cleanup this branch sits on: the sharding runtime moved
into `shard/`, `planner/planner.py` became `planner/planners.py`, and the
key -> rank rule now lives in `key_ownership.py`.

| File | Change | Size |
|---|---|---|
| `dynamicemb_config.py` | `host_index` (or equivalent) in `DynamicEmbTableOptions`; `_sharded_table_bucket_layout` takes a per-table divisor | S |
| `planner/planners.py` | TRW branch in `_settle`; per-table capacity divisor; ordering of `_prepare_dynemb_table_options` | M |
| `planner/plan.py` | none required -- `per_rank_storage` is already per rank, and reads the shard metadata `_settle` writes | -- |
| `shard/twrw_sharding.py` | **new** -- `TwRwPooledDynamicEmbeddingSharding` | M |
| `shard/embeddingbag.py` | dispatch TRW | S |
| `shard/input_dist.py` | `TwRwSparseFeaturesDist` with the staggered shuffle | M |
| `key_ownership.py` | the rule takes a shard base, not only a fan-out | S |
| `batched_dynamicemb_tables.py` | shard-index file naming; `find_files` reads meta | M |
| `incremental_dump.py` | meta carries the ownership triple | S |
| `src/sparse_block_bucketize_features.cu` | none required | -- |

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

**R1b — the same N×, applied to a budget that already does not account for it.**
The planner is told the whole device (`hbm_cap =
torch.cuda.get_device_properties(0).total_memory` in `planner/get_planner.py`)
and gives back five percent (`HeuristicalStorageReservation(percentage=0.05)`).
Nothing subtracts what the DynamicEmb tables will take: `local_hbm_for_values`
is computed in `_prepare_dynemb_table_options` and read only when the tables are
built, never by the planner. The DynamicEmb tables are simultaneously *free* in
the search space -- one 1x1 shard per rank, because the planner does not decide
their placement -- and *absent* from the budget, so TorchRec tables are planned
into HBM that is already spoken for. `DynamicEmbeddingShardingPlanner`'s
docstring says so outright: "The memory budget does not include the consumption
of dynamicemb."

This predates TRW and is not caused by it. What TRW does is multiply it: with
`local_hbm_for_values` becoming `global / local_size` instead of
`global / world_size`, the unaccounted HBM per rank grows by the same N as the
capacity. A gap that costs a model nothing today, because the tables happen to
fit, becomes N times wider on the exact configurations TRW is for.

Unlike R1 this one is fixable, and TorchRec has the seam for it: a
`StorageReservation` is precisely the place to declare memory that is spent but
not searched over. Doing it there rather than by shrinking `hbm_cap` keeps the
reason visible. Three things have to be settled first: a `caching=True` table's
`global_hbm_for_values` is a cache size and not the table, a HybridStorage
table only spends part of it on HBM, and the figure is finalised in
`_prepare_dynemb_table_options` -- which runs after `get_planner` has already
built the Topology.

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
`num_buckets` (`shard/input_dist.py:141`), which under TRW is `local_size`, not
`world_size`. These two must agree. This is the easiest place to be subtly wrong
and the hardest to cover with a test.

**F5 — mixed sharding types in one collection.** An EBC may hold RW tables, TRW
tables and plain TorchRec tables. `create_embedding_bag_sharding` dispatches on
`sharding_infos[0]` (`shard/embeddingbag.py:60`) — already a per-group decision,
but worth re-checking that grouping never mixes RW and TRW into one sharding.

**F6 — EC must be rejected, not silently downgraded.** §4.8.

**F7 — the fake shard metadata gains a third variant.** `planner/planners.py:348-353`
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

**F10 — the plan is shared by reference, and TorchRec writes to it.** Nothing
between `collective_plan` and the kernel copies a `ParameterSharding`: DMP hands
`self._plan.get_plan_for_module(path)` straight to `sharder.shard()`
(`model_parallel.py:538-546`), and `create_sharding_infos_by_sharding` stores the
same object on the `EmbeddingShardingInfo` (`embeddingbag.py:333`,
`embedding.py:284,725`) -- in the same call that `copy.deepcopy`s
`feature_names`, so the asymmetry is deliberate on TorchRec's side, not an
oversight.

TorchRec then writes to it in two places. `replace_placement_with_meta_device`
rewrites every `ShardMetadata.placement` in place when DMP's device is `meta`
(`embeddingbag.py:995-996`, `177-188`), which is the normal way to build a plan
on `cuda` and a model on `meta`. `DMPCollection._remap_sharding_plan` assigns
`param_sharding.ranks` and rewrites the shards (`model_parallel.py:1782-1820`).

For us this reaches further than it does for TorchRec, because
`_insert_dynamicemb_plan` inserts the entries held on the planner instance
(`self._dyn_emb_plan`) by reference. A second `collective_plan` from the same
planner would then hand out metadata already rewritten by the first DMP. No
current caller does this, and the failure would be a wrong placement rather than
an error. **Recommend: deep-copy each `ParameterSharding` on insertion**, so a
plan the caller passes on cannot reach back into the planner.

**F11 — the key -> rank fan-out has three sources, and a sub-group splits
them.** Routing takes it from the sharding process group -- `RwSparseFeaturesDist`
inherits `self._world_size = pg.size()` (`torchrec .../sharding/rw_sharding.py:395`)
and bucketizes into that many buckets (`shard/input_dist.py:227`). Ownership takes
it from the default group: `self._shard_world_size = dist.get_world_size()`
(`batched_dynamicemb_tables.py:566`), which is what `owned_key_mask` is given on
the dump/load path (`batched_dynamicemb_tables.py:2056`). Planning takes it from
the default group too, and ignores the `pg` it was handed:
`_prepare_dynemb_table_options` and `_dyn_emb_plan` both open with
`world_size = dist.get_world_size()` (`planner/planners.py:152,333`), so the row
count per shard, the `local_hbm_for_values` divisor (`:196`) and the number of
`ShardMetadata` entries are all sized for the whole job while TorchRec sizes its
half of the same plan for `pg.size()`.

Under `DistributedModelParallel` on the default group all three are equal and
nothing shows. They come apart whenever the sharding group is not the whole job
-- `DMPCollection` (2D parallel), where the embedding is sharded over
`sharding_group_size` ranks and replicated across groups, or simply a `pg`
argument to `collective_plan` that is not `GroupMember.WORLD`. Then
`pg.size() < dist.get_world_size()`, and the three halves of DynamicEmb disagree
with each other and with TorchRec: a rank is planned for one partition, fed the
keys of a second, and dumps the keys of a third. Nothing raises.

This is independent of TRW -- it is true on the branch today -- but TRW makes it
worse, since TRW introduces a second legitimate reason for the fan-out to differ
from the world size (`local_size`), and §4.6 already has to thread a shard base
through the rule. **Recommend: give the rule one source, taken from the sharding
environment rather than from `dist`, and reject `DMPCollection` explicitly until
that is done.**

---

## 8. Pre-existing defects found while writing this, and fixed since

All four were found while reading for this document and are fixed on the branch
this one sits on. Kept here because each says something about how the code got
that way, and because the shape of §8.1 is the one TRW is most likely to
reproduce.

### 8.1 Checkpoint load dropped keys for `hash_roundrobin` tables

`_iter_batches_from_files` filtered with a bare `keys % world_size == rank` --
the roundrobin rule spelled out by hand. A `hash_roundrobin` table's rank `r`
file holds the keys with `murmur(k) % ws == r`; that filter kept only the ones
that also satisfied `k % ws == r`, roughly `1/ws` of them, and discarded the
rest in silence. It was signed where the device kernel is unsigned, so a
negative key disagreed with the kernel even under roundrobin.

Its test could not have caught it: the checks decided what each rank should
hold with the same hand-written rule, and the `hash_roundrobin` coverage ran at
one rank, where the only rank owns everything and the oracle cannot be wrong.
Raised to two ranks with the oracle fixed, the case separates the trees exactly
as it should -- `main` fails, the fix passes.

**The rule was written out by hand in six places, not five.** The sixth was in
the test. Anything that recomputes ownership under TRW has to be found the same
way, and `test/` is part of where to look.

### 8.2 `continuous` was the silent fallback for a missing `dist_type`

A feature whose `fused_params` carried no `dist_type` fell back to
`continuous`, which is right for a plain TorchRec table and wrong for a
DynamicEmb one: it rewrites indices on the way in, so the table stores
something that is not a global key, and incremental dump, replay and checkpoint
load all refuse it. A missing setting became a table that could not be dumped
incrementally, and said nothing at the point the setting was missing. It now
raises.

### 8.3 The "same dist_type for all tables" assertion was unnecessary

It forbade two DynamicEmb tables in one sharding from using different rules.
The kernel reads the rule per feature, so nothing needed them to agree -- and
the assertion had a branch that bypassed it, so the invariant it claimed was
already routinely violated by the common case of a DynamicEmb table beside a
plain one.

### 8.4 The `dist_type` tensor was rebuilt every forward

A Python loop over `kjt.keys()` and a `torch.tensor(..., device=cuda)` per step,
while its sibling `_feature_block_sizes_tensor` was a constructor-time buffer.
Both describe the same fixed features. Building them together also removed an
unguarded assumption: the kernel indexes both with the same `t`, and nothing
checked that the orders agreed.

## 9. Suggested staging

| Milestone | Content | Gate |
|---|---|---|
| ~~**M0**~~ | **Done.** §8.1-§8.4, one ownership function (§4.6) with all six call sites on it, a 2-GPU `hash_roundrobin` dump/load test, and the planner cleanup that came with it: the copied `enumerate` and filters handed back to TorchRec, and the budget in §6 R1b. | Shipped on its own merit, as intended |
| ~~**M1**~~ | ~~Measure R1 as a go/no-go~~ **Cancelled.** Customer demand for TWRW is strong enough that it ships regardless of the R1 outcome. The measurement still has value as *sizing* input for §4.2 and for the `host_index` guidance in §10.3 -- it is folded into M5, not a gate. |  |
| ~~**M2**~~ | **Done.** Placement + capacity (§4.1, §4.2). Nodes are chosen by a `HostPlacer` component, with `host_index` surviving as a pin. Plan is TRW-shaped; nothing consumes it yet -- `shard/embeddingbag.py` still dispatches only ROW_WISE, so a TRW table reaches TorchRec's own TwRw sharding and will not work. M3 is what makes it run. | `table_fanout` / `table_layout` / `BalancedHostPlacer` unit tests; plan inspection still owed |
| ~~**M3**~~ | **Written, unrun.** `TwRwPooledDynamicEmbeddingSharding` + the input dist (§4.3, §4.4); the staggered shuffle turned out to be inherited. TRW tables reject dump / load / incremental-dump with a clear error, and `EmbeddingCollection` refuses TRW outright. | Numerical parity vs RW on a small model -- **still owed, and the gate M3 does not pass without** |
| **M4** | Checkpoint (§4.5), ownership under TRW (§4.6), incremental dump (§4.7) | Dump→load round-trip across TRW |
| **M5** | Perf validation: R2, R3 measured against the cross-node saving | Beat RW on the target topology, or stop |

With M1 cancelled, M2 is the first thing to do. R1 is still inherent to TRW and
cannot be engineered away -- dropping the gate does not make the N× capacity
pressure go away, it only means the answer no longer decides whether to build.
It decides which tables a user should put on TRW, so it belongs in the
documentation rather than in the schedule.

The seams in §4.9 partition this schedule: M2 is entirely on the composition
side of the line, M3 entirely on the inheritance side.

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
