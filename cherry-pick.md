# Cherry-Pick Dependency Analysis: Embedding Admission Strategy

Goal: get "Embedding admission strategy" (#236) working on top of base commit
`e48294d3` (Add LFU evict strategy, #52).

All 5 cherry-picks are required with the current local chain.

## Cherry-Pick Chain

| # | Local Commit | Upstream | Description | Why needed |
|---|-------------|----------|-------------|------------|
| 1 | `3c79809` | `6f7281a` | Refactor dynamicemb with Cache&Storage (#128) | Introduces `key_value_table.py`, `batched_dynamicemb_tables.py` (V2), `batched_dynamicemb_function.py` — the entire architecture that #236 builds on. These files do not exist at `e48294d3`. |
| 2 | `b818725` | `44525fb` | Support eval mode, move insert to forward (#136) | Introduces `find_and_initialize`, `lookup_forward_dense_eval`, and the `EventQueue` pattern used by `key_value_table.py`. Required to make #128 functional. |
| 3 | `befb8c9` | `d497241` | Fix LFU mode frequency count bug (#176) | **Introduces `types.py`** as a new file in the local cherry-pick. Commit #229 then modifies `types.py` — without this step, #229 will not apply. |
| 4 | `78c8ebc` | `c6adf64` | Counter table interface and ScoredHashTable (#229) | Introduces `scored_hashtable.py`. `embedding_admission.py` (#236) directly imports `ScoreArg`, `ScorePolicy`, `ScoreSpec`, and `get_scored_table` from it. |
| 5 | `68266e9` | `f5b608e` | Embedding admission strategy (#236) | The target feature. |

## Could Any Be Skipped?

In the **upstream** repo, `types.py` is first introduced by commit #229 itself
(`c6adf64`, confirmed via `--diff-filter=A`), not by #176. So if cherry-picking
raw upstream commits, the chain might reduce to 3 picks:
`6f7281a` (#128) → `c6adf64` (#229) → `f5b608e` (#236).

However, the local versions of these commits diverged during conflict resolution
(#136 and #176 were applied with `-X theirs`, modifying the same files). As a
result, the local #229 (`78c8ebc`) modifies `types.py` rather than creating it,
making the local #176 (`befb8c9`) a required intermediate step.

**Conclusion:** With the current local cherry-picked commits, all 5 are necessary.

---

## Bugs Found in Commit #5 (`68266e9` — Embedding Admission Strategy)

The local cherry-pick (`68266e9`) was sourced from the jiashuy fork (`c2babbe`)
rather than NVIDIA upstream (`f5b608e`). The two diverge significantly in
`batched_dynamicemb_function.py` and `key_value_table.py`, leaving the admission
strategy non-functional.

### Bug 1 — `DynamicEmbeddingFunctionV2.forward`: missing `admit_strategy`, `evict_strategy`, `admission_counter` parameters

`batched_dynamicemb_tables.py` calls `DynamicEmbeddingFunctionV2.apply()` with
these positional arguments:

| Position | Passed value | Received as (local) | Should be received as |
|----------|-------------|---------------------|-----------------------|
| 13 | `self._admit_strategy` | `frequency_counters` | `admit_strategy` |
| 14 | `self._evict_strategy` | `*args` (silently ignored) | `evict_strategy` |
| 15 | `per_sample_weights` | `*args` (silently ignored) | `frequency_counters` |
| 16 | `self._admission_counter` | `*args` (silently ignored) | `admission_counter` |

When admission is enabled (`_admit_strategy is not None`), the function
immediately crashes on `frequency_counters.long()` because it received an
`AdmissionStrategy` object instead of a tensor.

**Fix:** Add `admit_strategy=None`, `evict_strategy=None`, `admission_counter=None`
to the signature (before `frequency_counters`), matching the upstream layout.

### Bug 2 — `DynamicEmbeddingFunctionV2.forward`: `segmented_unique` receives a `bool` instead of `Optional[EvictStrategy]`

```python
# Local (wrong):
segmented_unique(indices, indices_table_range, unique_op,
                 is_lfu_enabled,          # bool: False→kLru, True→kLfu by accident
                 frequency_counts_int64)

# Upstream (correct):
segmented_unique(indices, indices_table_range, unique_op,
                 EvictStrategy(evict_strategy.value) if evict_strategy else None,
                 frequency_counts_int64)
```

The C++ binding expects `c10::optional<EvictStrategy>`. Passing `False` maps to
`kLru` (value 0) instead of `nullopt`, causing incorrect eviction-strategy
selection for non-LFU tables.

**Fix:** Use `evict_strategy` parameter and convert with
`EvictStrategy(evict_strategy.value) if evict_strategy else None`.

### Bug 3 — `DynamicEmbeddingFunctionV2.forward`: lookup calls missing new parameters

Both lookup calls omit `evict_strategy`, `admit_strategy`, and `admission_counter`:

```python
# KeyValueTableCachingFunction.lookup (local, wrong):
KeyValueTableCachingFunction.lookup(
    caches[i], storages[i], unique_indices_per_table, unique_embs_per_table,
    initializers[i], enable_prefetch, training,
    lfu_accumulated_frequency_per_table,   # lands in evict_strategy slot → type error
)

# KeyValueTableFunction.lookup (local, wrong):
KeyValueTableFunction.lookup(
    storages[i], unique_indices_per_table, unique_embs_per_table,
    initializers[i], training,
    lfu_accumulated_frequency_per_table,   # lands in evict_strategy slot → type error
)
```

Both functions now require `evict_strategy: EvictStrategy` as a positional
argument (added by the local #236), so passing `lfu_accumulated_frequency_per_table`
(a tensor or None) in that slot is a type error at runtime.

**Fix:** Insert `EvictStrategy(evict_strategy.value) if evict_strategy else None`
before `lfu_accumulated_frequency_per_table`, and append `admit_strategy` and
`admission_counter[i] if admission_counter else None` at the end of each call.

### Bug 4 — `DynamicEmbeddingFunctionV2.backward`: wrong `None` return count

```python
# Local:
return (None,) * 14   # reflects old parameter count

# Should be:
return (None,) * 17   # 3 new forward params (admit_strategy, evict_strategy,
                       # admission_counter) require 3 more None gradients
```

PyTorch autograd requires `backward` to return one gradient per `forward` input.
With 3 extra params the count must increase to 17; returning 14 will cause a
shape mismatch assertion in autograd.

**Fix:** Change `return (None,) * 14` → `return (None,) * 17`.

### Bug 5 — `DynamicEmbeddingFunctionV2.backward`: update routed to wrong function

```python
# Local (wrong): always calls KeyValueTableFunction.update with (cache, storage, ...)
KeyValueTableFunction.update(
    caches[i],             # lands in storage slot → type error
    storages[i],           # lands in unique_keys slot → type error
    unique_indices_per_table,
    unique_embs_per_table,
    optimizer,
    enable_prefetch,       # KeyValueTableFunction.update has no such param
)

# Upstream (correct): branch on caching flag
if caching:
    KeyValueTableCachingFunction.update(caches[i], storages[i], ...)
else:
    KeyValueTableFunction.update(storages[i], ...)
```

`KeyValueTableFunction.update` takes `(storage, unique_keys, ...)` — no `cache`
parameter. The local backward unconditionally called it as
`KeyValueTableFunction.update(caches[i], storages[i], ...)`, so `caches[i]`
landed in `storage` and `storages[i]` in `unique_keys`, causing a type error on
the first iteration.

**Fix:** Add `caching = caches[0] is not None` at the top of backward, then
dispatch to `KeyValueTableCachingFunction.update` or `KeyValueTableFunction.update`
accordingly.

### Bug 6 — `KeyValueTableFunction.lookup`: extra unused `cache` parameter shifts all args

```python
# Local (wrong):
def lookup(
    cache: Optional[Cache],   # ← never used in body; shifts everything
    storage: Storage,
    unique_keys, unique_embs, initializer, training,
    evict_strategy, accumulated_frequency=None,
    admit_strategy=None, admission_counter=None,
)

# Upstream (correct):
def lookup(
    storage: Storage,
    unique_keys, unique_embs, initializer, training,
    evict_strategy, accumulated_frequency=None,
    admit_strategy=None, admission_counter=None,
)
```

The call site (`DynamicEmbeddingFunctionV2.forward`) passes `storages[i]` as the
first argument, which lands in `cache` instead of `storage`, causing a type error
when the body calls `storage.embedding_dim()` on a `KeyValueTable` object that
ended up in `cache`.

**Fix:** Remove the `cache: Optional[Cache]` parameter from `KeyValueTableFunction.lookup`.

### Known Issue — `setup.py`: missing `import sys`

`sys.executable` is referenced at lines 29 and 65 but `import sys` is absent.
This causes `NameError: name 'sys' is not defined` at build time.
This is not a cherry-pick bug (the upstream also lacks it); must be patched manually
before each build. Already documented in `merge.md`.

---

## Fixes Applied (commit 268cf1d)

| Bug | File | Change |
|-----|------|--------|
| 1 — `forward` missing params | `batched_dynamicemb_function.py` | Added `admit_strategy`, `evict_strategy`, `admission_counter` at positions 13–16; shifted `frequency_counters` to 16 |
| 2 — `segmented_unique` bool arg | `batched_dynamicemb_function.py` | `EvictStrategy(evict_strategy.value) if evict_strategy else None` |
| 3 — lookup calls missing params | `batched_dynamicemb_function.py` | Added `evict_strategy`, `admit_strategy`, `admission_counter[i]` to both caching and non-caching lookup calls |
| 4 — `backward` wrong return count | `batched_dynamicemb_function.py` | `(None,) * 14` → `(None,) * 17` |
| 5 — `backward` wrong update function | `batched_dynamicemb_function.py` | Added `caching` flag; dispatch to `KeyValueTableCachingFunction.update` or `KeyValueTableFunction.update` accordingly |
| 6 — extra `cache` param in `lookup` | `key_value_table.py` | Removed `cache: Optional[Cache]` from `KeyValueTableFunction.lookup` |
| Known — `import sys` missing | `setup.py` | Added `import sys` |
