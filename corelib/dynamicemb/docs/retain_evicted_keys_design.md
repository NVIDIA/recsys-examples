# Retain Evicted Keys — Design & Change Document

> Branch: `feat/dynamicemb-retain-evicted-keys` (built on PR #437's LruLfu / JIT
> infrastructure).
> Goal: let the **last-tier storage** optionally retain the keys it evicts, and
> expose a `pop_evicted_keys(...)` API to read back, per table, the deduplicated
> keys evicted over a recent window of training.

---

## 1. Background & Motivation

DynamicEmb's key→slot hash table evicts a "victim key" by score when a bucket is
full. There are two kinds of eviction, with completely different behavior:

- **`table_insert_and_evict`** (used by the GPU main table / cache / HBM tier):
  the evicted key is written to the `evicted_keys/scores/indices/table_ids`
  output buffers, then the Python layer **moves it to the next tier**
  (host/backing). Such keys are not truly lost.
- **`table_insert`** (used by the **last-tier** host / backing table): the evict
  output pointer is passed `nullptr` (`insert_body` in `kernels.cuh` currently
  passes `nullptr` for `evict_key_out`), so the evicted key is **silently
  overwritten and dropped** — this is the one place a key truly disappears from
  the system, and there was no buffer retaining it.

This feature adds an **optional collection hook** to that "last-tier `insert`"
path.

---

## 2. Confirmed Design Decisions

| Decision | Choice |
|---|---|
| Trigger tier | **Last tier only** (the tier that truly drops a key). Intermediate-tier eviction is a spill and is not recorded |
| Collection vehicle | **A compile-time template sink on `insert_body`** (not numba-JIT). The numba path stays reserved for the `user_score_fn` numeric function |
| What is collected | Only **key + table_id** (no score / value / index) — minimal memory, matches the need |
| Coverage | **All score policies**: both insert paths — AoT (`DefaultEvictor`) and the LruLfu cubin (`RankedEvictor`) |
| Dedup & memory | **Append buffer + `torch.unique` on read** (no compaction while appending; dedup only on `pop`) |
| Time window | **`pop` reads-and-clears / incremental**: returns the unique keys evicted since the previous call |
| Distributed | `pg=None` returns each rank's local part (row-wise sharding → naturally disjoint, zero comm); with a `pg`, `all_gather` the group-wide union. **Clearing always clears only this rank's buffer** |
| Config | `DynamicEmbTableOptions.evicted_item_mode: EvictedItemMode = DISCARD` (DISCARD by default, zero overhead) |
| API name | `pop_evicted_keys` |

The confirmed decisions are recorded in §9.

---

## 3. End-to-end Data Flow

**Collection trigger points (two "last-tier insert" paths)** — retain collection
must hook every "last-tier, truly-drops-the-victim" insert. A code walk found
**two** such places (the second was missed initially and added after the
module-level test caught it):

- **(A) the `storage.insert` path** → `key_value_table._insert_key_values`. Used
  by: a direct `storage.insert` (`DynamicEmbStorage` / the host tier of
  `HybridStorage`), the caching-mode cache-eviction write-back to backing
  (`_prefetch_cache_path`), and `_generic_forward_path`.
- **(B) the forward HBM-direct path** →
  `batched_dynamicemb_function._prefetch_hbm_direct_path`. Used by **non-caching
  training forward** (single-tier HBM `DynamicEmbStorage`), which calls
  `key_index_map.insert()` **directly** (bypassing `_insert_key_values`).

Both share the same collection primitive:

```
state.key_index_map.insert(..., collect_evicted=True)          # LinearBucketTable
  └─ C++ table_insert_collect_evicted(...)                     # new host function
       ├─ AoT:    table_insert_collect_kernel                  # non-LruLfu
       └─ LruLfu: dyn_emb_insert_collect_entry (cubin)         # via jit_link
            └─ insert_body<Table, Evictor, CollectEvicted=true>
                 · grabs the victim key at the eviction point
                 · compaction writes evicted_keys[out]=victim,
                                     evicted_table_ids[out]=tid
  → (indices, num_evicted, evicted_keys, evicted_table_ids)
_append_evicted(state, ...): clone the first num_evicted entries and append to the
  state's evicted_key_chunks / evicted_tid_chunks (pure append, no dedup)
```

**Ref-counter & training semantics**: in forward, prefetched keys are
`counter++` (in use), so they cannot be evicted within the same step (this shows
up as `Busy`, which is not `Evict` and is not collected); only after a
cross-step backward decrements the counter back to 0 can a later step actually
`Evict` and collect them. So during training, retain produces evictions at step
boundaries — a bare forward (no backward) collects nothing.

```
User call (any time)
  dynamicemb.pop_evicted_keys(model, table_names=None, pg=None)
    └─ walk collections → module.pop_evicted_keys → storage.pop_evicted_keys(tid)
         local keys = torch.unique(cat(chunks)[tid==table_id])   # dedup only on pop
         with a pg → all_gather + concat (each rank disjoint)
         clear this rank's chunks for that table
    → {collection_path: {table_name: 1-D int64 keys}}  (host tensors)
```

---

## 4. C++ Changes (by file)

### 4.1 `src/table_operation/kernels.cuh`

**(a) `insert_body` gains a template sink + output params** (template param
appended at the end for backward compatibility; output params default to
`nullptr`)

```cpp
template <typename Table, typename KernelTraits,
          typename Evictor = DefaultEvictor, bool CollectEvicted = false>
__device__ __forceinline__ void
insert_body(..., int32_t *__restrict__ counter, uint64_t cur_ts = 0,
            typename Table::KeyType *__restrict__ evicted_keys = nullptr,
            int64_t *__restrict__ evicted_table_ids = nullptr,
            CounterType *evicted_counter = nullptr) {
```

- Existing instantiations `insert_body<Table, KernelTraits, DefaultEvictor>(...,
  counter, 0)` and `insert_body<..., RankedEvictor<C>>(..., counter, cur_ts)` are
  **unaffected** (`CollectEvicted` defaults to false, new params default to
  nullptr).

**(b) grab the victim key at the eviction point** (currently passes `nullptr` for
`evict_key_out`)

```cpp
    KeyType evict_key = KeyType();
    insert<ReductionGroupSize, BufferDim, Policy, Evictor>(
        bucket, key, score, ..., &iter, &result,
        CollectEvicted ? &evict_key : static_cast<KeyType *>(nullptr),  // was nullptr
        static_cast<ScoreType *>(nullptr), counter, counter_offset, cur_ts);
```
`CollectEvicted` is a compile-time constant; the false branch folds back to
`nullptr`, so the non-collecting path's code is unchanged.

**(c) a compaction block at the loop tail (`if constexpr`-guarded, writes only key
+ table_id)**

```cpp
    if constexpr (CollectEvicted) {
      // CompactTileSize == 1 for every insert instantiation
      // (InsertKernelTraits<...,1,...> in insert.cu / evict_lrulfu.cu), so this
      // size-1 tile ballot needs no cross-thread convergence; the `continue` on
      // table_cap==0 above is harmless (those threads evicted nothing).
      auto g = cg::tiled_partition<KernelTraits::CompactTileSize>(
          cg::this_thread_block());
      bool evicted = (result == InsertResult::Evict);
      uint32_t vote = g.ballot(evicted);
      int group_cnt = __popc(vote);
      CounterType group_offset = 0;
      if (g.thread_rank() == 0)
        group_offset =
            atomicAdd(evicted_counter, static_cast<CounterType>(group_cnt));
      group_offset = g.shfl(group_offset, 0);
      int previous_cnt = group_cnt - __popc(vote >> g.thread_rank());
      int64_t out_id = group_offset + previous_cnt;
      if (evicted) {
        evicted_keys[out_id] = evict_key;
        evicted_table_ids[out_id] = table_ids[i];
      }
    }
```
> **Only `InsertResult::Evict` is collected** (an existing key pushed out). `Busy`
> (a new key that failed to insert due to contention / a full bucket) is **not**
> collected — that is an insert failure, in the `safe_check_mode` domain, and the
> key reappears in a later batch. [Decision 1]

**(d) a new AoT collecting `__global__`** (next to `table_insert_kernel`)

```cpp
template <typename Table, typename KernelTraits>
__global__ void table_insert_collect_kernel(
    /* same params as table_insert_kernel */ ..., int32_t *counter,
    typename Table::KeyType *evicted_keys, int64_t *evicted_table_ids,
    CounterType *evicted_counter) {
  insert_body<Table, KernelTraits, DefaultEvictor, /*CollectEvicted=*/true>(
      ..., counter, /*cur_ts=*/0, evicted_keys, evicted_table_ids,
      evicted_counter);
}
```

### 4.2 `src/jit/evict_lrulfu.cu`

Add a collecting entry for the LruLfu cubin (reusing the existing `EvictParams`
fields `evicted_keys / evicted_table_ids / evicted_counter`):

```cpp
__device__ __forceinline__ void run_insert_collect(const EvictParams &p,
                                                   uint64_t cur_ts) {
  using KernelTraits =
      InsertKernelTraits<256, 1, 1, /*CompactTileSize=*/1, 8,
                         ScorePolicyType::LruLfu, /*OutputScore=*/true>;
  EvictTable table(p.table_storage, p.num_buckets, p.bucket_capacity,
                   p.num_scores);
  insert_body<EvictTable, KernelTraits, RankedEvictor<EvictComparator>,
              /*CollectEvicted=*/true>(
      table, p.table_bucket_offsets, p.bucket_sizes, p.batch, p.input_keys,
      p.table_ids, reinterpret_cast<InsertResult *>(p.insert_results),
      p.indices, p.score_input, p.score_output, p.table_key_slots, p.counter,
      cur_ts, reinterpret_cast<KeyType *>(p.evicted_keys), p.evicted_table_ids,
      p.evicted_counter);
}
extern "C" __global__ void dyn_emb_insert_collect_entry(EvictParams p) {
  run_insert_collect(p, read_globaltimer());
}
```
> The `EvictParams` struct is **unchanged** (fields already present). This entry
> goes into both `evict_lrulfu_lex.fatbin` (AoT, default Lex evictor) and the
> custom fatbin, compiled alongside the existing 3 entries.

### 4.3 `src/jit/jit_link.h` / `jit_link.cpp`

- Add `CUfunction insert_collect;` to `EvictModule`.
- On load, `cuModuleGetFunction(&m.insert_collect, m.module,
  "dyn_emb_insert_collect_entry")` (inside the existing "unload if any entry is
  missing" guard).
- New getter:
  ```cpp
  CUfunction demb_get_insert_collect_fn(int64_t key);  // same key space / module
  ```
- Launch reuses the existing `demb_launch_evict(fn, EvictParams, batch, stream)`
  — no new launcher.

### 4.4 `src/table_operation/insert.cu`

- New `launch_table_insert_collect_kernel<Table, PolicyTypeV, OutputScoreV>(...)`,
  isomorphic to `launch_table_insert_kernel` but:
  - LruLfu branch: fill `EvictParams`'s
    `evicted_keys/evicted_table_ids/evicted_counter` and call
    `demb_get_insert_collect_fn(score_fn_key)`;
  - AoT branch: call `table_insert_collect_kernel`.
  - followed by the same `table_unlock_kernel`.
- New host entry (**without changing the existing `table_insert` signature**):
  ```cpp
  // returns (indices, num_evicted, evicted_keys, evicted_table_ids)
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
  table_insert_collect_evicted(/* same inputs as table_insert */ ...);
  ```
  It allocates `evicted_keys[num_total]`, `evicted_table_ids[num_total]`
  (`int64`), `evicted_counter[1]=0`, dispatches to
  `launch_table_insert_collect_kernel`, and returns the buffers + count (the
  Python side slices by `num_evicted`). The buffer upper bound is the batch key
  count (each input key evicts at most one victim).

### 4.5 `src/table_operation/table.cu`

Bind `table_insert_collect_evicted` in `bind_table_operation` (signature/return
as in 4.4).

---

## 5. Python Changes (by file)

### 5.1 `dynamicemb/dynamicemb_config.py`
`DynamicEmbTableOptions` gains a field:
```python
evicted_item_mode: EvictedItemMode = EvictedItemMode.DISCARD
```
Docstring: effective on the last-tier storage only; when enabled, evicted keys
can be read back with `pop_evicted_keys`.

### 5.2 `dynamicemb/scored_hashtable.py`
`LinearBucketTable.insert` gains an optional param (off by default, return type
unchanged in the default case):
```python
def insert(self, keys, table_ids, score, insert_results=None, score_out=None,
           collect_evicted: bool = False):
    ...
    if collect_evicted:
        return table_insert_collect_evicted(
            ..., score_fn_key=self.score_fn_key_)  # (indices, num_evicted, ev_keys, ev_tids)
    indices = table_insert(...)   # as before
    return indices
```
> The `_deterministic_insert` path (`DEMB_DETERMINISM_MODE`) supports collection
> too: it inserts one wave at a time with `table_insert_collect_evicted` and
> concatenates each wave's victims in wave order, so the list is deterministic.
> [Decision 3, revised: the cache's eager flow needs the victim list under
> determinism]

### 5.3 `dynamicemb/key_value_table.py` (core)

**(a) `DynamicEmbTableState` gains retain state** (initialized in
`create_table_state` from `evicted_item_mode`, only for a last-tier state):
- `evicted_item_mode: EvictedItemMode`
- `evicted_key_chunks: List[torch.Tensor]` (per-insert 1-D int64 key chunks)
- `evicted_tid_chunks: List[torch.Tensor]` (per-insert 1-D int64 table_id chunks)

> No compaction/dedup while appending ([Decision 2]); dedup happens only on `pop`
> via `torch.unique`.

**(b) `_insert_key_values` (the last-tier common entry)** becomes:
```python
if state.evicted_item_mode == EvictedItemMode.RETAIN_KEY:
    indices, num_evicted, ev_keys, ev_tids = state.key_index_map.insert(
        unique_keys, table_ids, score_arg, score_out=score_out_flat,
        collect_evicted=True)
    _append_evicted(state, ev_keys, ev_tids, num_evicted)
else:
    indices = state.key_index_map.insert(...)   # as before
...
store_to_flat(state, flat_indices, table_ids, unique_values)   # unchanged
```
`_append_evicted` clones the first `num_evicted` entries and appends (key, tid)
to the two chunk lists (pure append, no compaction).

**(c) `pop` helpers** (storage layer)
- `DynamicEmbStorage.pop_evicted_keys(table_id) -> Tensor`: from its state chunks,
  take `key[tid==table_id]` → `torch.unique` → clear **this rank's** entries for
  that table (rebuild chunks keeping the other tables' rows).
- `HybridStorage.pop_evicted_keys(table_id)`: reads `self._host`'s chunks only
  (the last tier).
- `DynamicEmbCache`: not involved (not a last tier).

Module-/model-level `pop_evicted_keys` return **host** tensors (dedup + NCCL
gather run on device, then `.cpu()`).

**(d) Last-tier determination** (`evicted_item_mode` hooks only the state that
truly drops keys)
- non-caching single-tier `DynamicEmbStorage` → its state is a last tier ✔
- `HybridStorage` → `self._host` state is a last tier ✔; `self._hbm` ✘ (spills via
  insert_and_evict)
- caching → the backing `self._storage` (`DynamicEmbStorage` or an external PS) is
  a last tier ✔; `self._cache` ✘

### 5.4 `dynamicemb/batched_dynamicemb_tables.py`
- At construction, pass each table's `evicted_item_mode` to the **last-tier**
  storage's `create_table_state` (dispatched per the rules above in
  `_create_cache_storage`).
- New method:
  ```python
  def pop_evicted_keys(self, table_names=None) -> Dict[str, torch.Tensor]:
      # only retain-enabled tables; table_name -> table_id via
      # self._table_names.index(name); calls self._storage.pop_evicted_keys(table_id)
      # and returns host tensors. Cross-rank aggregation is the model-level API's job.
  ```
  (For caching: read from the backing `self._storage`.)

### 5.5 `dynamicemb/incremental_dump.py`
Model-level function, styled after `incremental_dump`:
```python
def pop_evicted_keys(
    model: torch.nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,  # None = all retain-enabled tables
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # {collection_path: {table_name: 1-D int64 unique evicted keys (host)}}
```
Walks via `find_sharded_modules` / `get_dynamic_emb_module`, calling
`pop_evicted_keys` on each `BatchedDynamicEmbeddingTablesV2`; with a `pg` it
`all_gather`s each table's keys within the group (variable-length gather).

### 5.6 `dynamicemb/__init__.py`
`from .incremental_dump import pop_evicted_keys` and add it to `__all__`.

---

## 6. `pop_evicted_keys` API Spec

```python
def pop_evicted_keys(model, table_names=None, pg=None)
    -> Dict[str, Dict[str, torch.Tensor]]
```
- **Returns**: `{collection_path: {table_name: keys}}`, `keys` a 1-D `int64` host
  tensor, unique within a table.
- **`pg=None`**: each rank returns the keys evicted on its local shard (zero comm).
- **With a `pg`**: `all_gather` within the group and concat (ranks are disjoint, so
  the concatenation is the global unique set).
- **Incremental semantics**: returns keys evicted "since the previous call", and
  clears **this rank's buffer** afterward (only its own, regardless of aggregation).
- Tables without `evicted_item_mode=RETAIN_KEY`: **omitted** from the result (no empty
  tensor). [Decision 4]

---

## 7. Memory & Semantics Details

- **Collection upper bound**: the evicted count of one insert ≤ the input key
  count; the C++ side preallocates `num_total`, `evicted_counter` gives the real
  count, and the Python side slices.
- **Buffer growth**: chunks are appended with no dedup; memory ≈ O(total keys
  evicted between two pops, including duplicates). Dedup is done only on `pop`
  ([Decision 2]).
- **Clear granularity**: a per-table `pop` should clear only that table's entries.
  Chunks mix tables, so pop concatenates, selects `tid==table_id`, and rebuilds
  the chunks keeping `tid != table_id`.
- **Disjoint guarantee**: under row-wise sharding a key lands on exactly one rank,
  so cross-rank aggregation is just a concat.
- **Orthogonal to `incremental_dump`**: independent; retain records keys that have
  **left** the table, while incremental_dump exports keys **still in** the table
  whose score crosses a threshold.

---

## 8. Tests (compiled and passing on EOS / H100)

All retain tests live under `test/unit_tests/retain_evicted_keys/`.

- **Table level** (`test_insert_collect_evicted.py`, 5 cases): a whole-set
  eviction oracle checks the (evicted_keys, evicted_table_ids) collected by
  `insert(collect_evicted=True)` — **one each for the LruLfu cubin and the AoT
  path**, table_id routing across logical tables sharing storage, empty on no
  eviction, and collection under `DEMB_DETERMINISM_MODE` (same victims as the
  ordinary insert).
- **Storage level** (same file, 3 cases): `DynamicEmbStorage` end-to-end (collect
  → pop unique → evicted keys are gone from the table → second pop empty /
  incremental), `retain=False` always empty, and a buffer-logic unit test
  (cross-batch dedup + table_id filter + clear isolation).
- **Module level** (`test_pop_evicted_keys.py`, 3 cases): **the non-caching
  training forward (HBM-direct path) retains its cross-step evictions** — this
  test found and drove the §3(B) `_prefetch_hbm_direct_path` fix; disabled tables
  omitted; `table_names` filter + table_id isolation. All use full
  forward+backward steps (see §3's ref-counter semantics: bare forward evicts
  nothing).
- **Distributed / model level** (`test_distributed_pop_evicted_keys.py`,
  self-contained, launched with torchrun): row-wise sharded — `pg=None` disjoint
  per rank, `pg` gives the group-wide union identical on every rank, clearing is
  rank-local, plus the model-level `dynamicemb.pop_evicted_keys(model, ...)`
  walking a sharded collection.
- **Regression**: with `evicted_item_mode=DISCARD` (default), the existing
  insert/evict paths are unchanged by construction (`if constexpr` / Python
  branches).

---

## 9. Decision Record (confirmed)

1. **Collection scope**: collect only `InsertResult::Evict`; **not `Busy`**.
2. **No mid-flight compaction**: no compaction while appending; dedup only on
   `pop` via `torch.unique`.
3. **Determinism mode**: collection is supported; the waves' victims are
   concatenated in wave order.
4. **Disabled tables**: omitted from the `pop` result (no empty tensor).
5. **API location**: `pop_evicted_keys` lives in `dynamicemb/incremental_dump.py`.
6. **No `clear=False`**: keep only the read-and-clear semantics; no read-only
   variant.

---

## 10. Changed Files

| File | Change |
|---|---|
| `src/table_operation/kernels.cuh` | `insert_body` sink template + compaction; new `table_insert_collect_kernel` |
| `src/jit/evict_lrulfu.cu` | new `dyn_emb_insert_collect_entry` |
| `src/jit/jit_link.h` / `.cpp` | cache the new entry + `demb_get_insert_collect_fn` |
| `src/table_operation/insert.cu` | `launch_table_insert_collect_kernel` + host `table_insert_collect_evicted` |
| `src/table_operation/table.cu` | bind `table_insert_collect_evicted` |
| `dynamicemb/dynamicemb_config.py` | `evicted_item_mode` option |
| `dynamicemb/scored_hashtable.py` | `LinearBucketTable.insert(collect_evicted=...)` |
| `dynamicemb/key_value_table.py` | state retain chunks + `_insert_key_values` collection + `_append_evicted` / `_pop_state_evicted_keys` + storage `pop_evicted_keys` |
| `dynamicemb/batched_dynamicemb_function.py` | **retain collection on the forward HBM-direct path (`_prefetch_hbm_direct_path`) — §3(B) gap fix** |
| `dynamicemb/batched_dynamicemb_tables.py` | thread the config through + module-level `pop_evicted_keys` |
| `dynamicemb/incremental_dump.py` | model-level `pop_evicted_keys` + variable-length all_gather |
| `dynamicemb/__init__.py` | export `pop_evicted_keys` |
| `test/unit_tests/retain_evicted_keys/test_insert_collect_evicted.py` | table + storage level (8 cases) |
| `test/unit_tests/retain_evicted_keys/test_pop_evicted_keys.py` | module level (3 cases) |
| `test/unit_tests/retain_evicted_keys/test_distributed_pop_evicted_keys.py` | distributed + model level |
