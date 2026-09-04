# `replay_increment` · Design & Change Document

> Branch: `feat/dynamicemb-replay-increment` (on top of the merged
> `DeltaDumpResult` refactor, `docs/delta_dump_result_design.md`).
> Goal: write a `DeltaDumpResult` produced by `incremental_dump` back into a
> dynamic embedding model — the "replay" half of the delta pipeline.
> `docs/delta_dump_result_design.md` §8 left this to "the next round"; this is
> that round.

---

## 1. Background

`incremental_dump` already returns, per embedding collection, a
`DeltaDumpResult` whose per-table `meta` was designed for this feature:

| `meta[i]` field | why replay needs it |
|---|---|
| `slot_index` | the exact key slot / value row each dumped key occupied |
| `current_capacity` | a source slot is only transferable when the target's capacity matches |
| `world_size` | the source's row-wise fan-out (key → rank modulo base) |
| `table_options` | `bucket_capacity` / `score_strategy` / `dim` / `dist_type` compatibility |
| `current_score` | the table score after the dump |

Everything replay needs is already there; §3 records what deliberately is *not*.

---

## 2. Confirmed Design Decisions

| Decision | Conclusion |
|---|---|
| Write-back | **By slot.** Every key is written at the slot and value row it held in the source table, leaving the target layout-identical to it |
| Layout mismatch | **Raise**, before writing anything. Checked at table level (delta `meta` vs target) and enforced again per key in the kernel |
| Per-key scores | **Dumped, not yet replayed** — `DeltaDumpResult.scores`; a restored key is scored as if freshly inserted into the target (§3) |
| Optimizer state | **Dumped, not yet replayed** — `DeltaDumpResult.optimizer_states`, split out of the same value row as `values` (§3) |
| `evicted_keys` | **Never replayed.** Overwriting a slot already reproduces an eviction, so this list exists only for other consumers of the dump |
| `erased_keys` | A separate buffer and a separate list, applied whenever non-empty, **erase before upsert** — nothing takes over an erased key's slot, so no write reproduces it |
| What replay writes | `ReplayContent` flag: embedding / optimizer state / score, any combination, default all three |
| Optimizer state | Never carried in the delta. A slot whose previous occupant was the *same* key keeps its optimizer state; any other slot is re-initialised |
| `dist_type` | `roundrobin` / `hash_roundrobin` only — same restriction the dump side already enforces; `continuous` raises |

---

## 3. What a delta carries

`DeltaDumpResult` holds five column-aligned per-table lists: `keys`, `values`,
`optimizer_states`, `scores`, plus the two removal lists `evicted_keys` and
`erased_keys`.

**`values` / `optimizer_states` are the two halves of one stored row.**
`load_from_flat_single_table` returns the table's own compact `[N, value_dim]`
row, so the optimizer state is the trailing `value_dim - dim` columns.

That runtime width is **not** what gets dumped. Rowwise Adagrad reserves a fixed
16 bytes per row in the fused FBGEMM layout but only ever fills one accumulator
scalar, so the block is narrowed with the same
`truncate_optimizer_states_for_checkpoint` the file dump uses, and replay expands
it back with `pad_optimizer_states_from_checkpoint`. A delta and a checkpoint
then describe a row identically, and the padding is not shipped on every dump.
A table whose optimizer keeps no per-row state (plain SGD) reports `None`, not a
zero-width tensor — matching `export_keys_values_iter`.

**`scores` carries every score word**, in the user's configured (logical) column
order. Timestamp columns are converted to an **age** (`cur_ts - score`) because
`%globaltimer` is per device and resets across boots, so a raw timestamp means
nothing anywhere but the rank that produced it; the baseline is
`meta["current_score"]`, and a consumer rebases against its own clock. Every
other column (LFU frequency, STEP, CUSTOMIZED, NO_EVICTION row) is verbatim.
This is the same convention the file `dump`/`load` already uses for single-column
LRU tables.

**Replay does not consume either yet.** A restored key is scored as if it had
just been inserted into the target (`_fresh_score_block`), and keeps its
optimizer state only when it already occupies the target row. The consequence is
worth stating plainly, because it is the one place a replica is *not* a copy of
its source: **the two can evict in different orders.** What they never disagree
on is the embedding of a key they both hold — which is what a delta exists to
synchronise. Wiring the two fields into `replay_increment` closes that gap and is
the natural next step.

**NO_EVICTION is the exception, and it is exact.** Its score word is not a score
at all but the value row (§12), so replay writes it verbatim from `slot_index`.
There is nothing to approximate: reproducing the source's rows is the point.

`cur_ts` is sampled **once, before** the per-table dump loop and reused as both
the age baseline and `meta["current_score"]`. Sampling before (rather than after)
the dump makes the next window at-least-once instead of at-most-once: a key
touched *during* the dump is re-dumped next time rather than dropped.

Storage-level `incremental_dump` returns
`(keys, values, slot_index, optimizer_states, scores)`, and
`_all_gather_dumped_keys_values` is generalised to gather an arbitrary list of
aligned tensors — the optimizer column is present or absent as a whole, and
every rank agrees, so the collective stays symmetric.

---

## 4. Write-back by slot

### 4.1 Why the target must have the same layout

A key can only ever live in its own home bucket:

```
bucket_id = bkt_begin + (hash(key) % table_cap) / bucket_capacity
```

So writing key `K` into slot `S` is only *findable* if `S` lies inside `K`'s home
bucket — which requires the target's `table_cap` and `bucket_capacity` to equal
the source's. §4.2 checks that up front and §4.3 re-checks it per key.

### 4.2 Table-level compatibility check (Python)

A table is replayed only when **all** of these match; otherwise `replay_increment`
raises `ValueError` naming the first mismatch, before writing anything:

- `meta["current_capacity"]` vs target `key_index_map.capacity(table_id)`
- `meta["table_options"].bucket_capacity` vs the target table's
- `meta["table_options"].score_strategy` vs the target's, compared by
  **physical** word order — `(TIMESTAMP, LFU)` and `(LFU, TIMESTAMP)` are the
  same on-device layout, and with no scores in flight the configured tuple order
  no longer matters
- `meta["table_options"].dim` vs the target's embedding dim
- `meta["world_size"]` vs the target's `_shard_world_size`
- storage kind (single-tier vs hybrid) and `num_scores`

### 4.3 Key-level enforcement (kernel)

New primitive `table_scatter_keys_at_slots`, one thread per key:

1. `slot < 0`, invalid key, or `slot >= table_cap` → `status[i] = -1`.
2. `home_bucket(key) != bucket_of(slot)` → `status[i] = -1` (the layout check
    can still fail per key if the target rehashed).
3. Probe the home bucket for `key`; if it already lives at a *different* slot,
    reclaim that slot (this is the only way a duplicate could appear).
4. Lock the target slot, write digest + all score words, unlock with the key.
    `bucket_sizes` is incremented only when the slot did not previously hold a
    valid key.
5. `status[i] = slot`, and `same_key[i]` records whether the slot's previous
    occupant was this very key (used to decide optimizer-state handling).

Any `status[i] == -1` raises: the table-level check already established that the
layouts match, so a slot outside its key's home bucket means they have diverged
anyway, and continuing would silently drop that key.

### 4.4 Precondition: the target must be a replica

Writing at the source's slot **overwrites whatever occupies it** in the target.
That is the convergent behaviour a replica needs: if the source evicted key `B` to
make room for `A` in slot `S`, a same-capacity replica must do the same, so
refusing to overwrite would leave the replica permanently diverged.

The corollary is a precondition: a replayed table must be built *only* by
loading/replaying from its source. A table that also takes independent writes can
lose a key whose slot a delta key claims.

### 4.5 `slot_index` unpacking

Mirrors `_encode_slot_index` (`docs/delta_dump_result_design.md` §4):

- single-tier, normal policy: `slot_index` **is** both key slot and value row
- single-tier, NO_EVICTION: high 32 bits = key slot, low 32 bits = value row;
  the target's `no_eviction_next_index` counter is bumped past `max(row) + 1`
- `HybridStorage`: bit 63 selects the tier (0 = HBM, 1 = host), bits 0..62 are
  the key slot; each tier is replayed into its own state

---

## 5. Value writes and optimizer state

The delta carries embeddings only (`values[:, :emb_dim]`), never optimizer
state. `store_to_flat_table_contiguous` copies `min(value_dim, input_dim)`
columns from the row base, so writing an `[N, emb_dim]` tensor touches the
embedding region and leaves the optimizer region untouched. Replay therefore
splits every batch in two:

| group | write |
|---|---|
| slot previously held the **same key** (`same_key` / `founds`) | embedding columns only — the key's own optimizer state survives |
| new key, or the slot's previous occupant was a different key | embedding columns **plus** `initial_optim_state` for the optimizer region |

Without the second group a new key would silently inherit the evicted key's
optimizer moments.

---

## 6. Sharding and removals

**Rank filtering.** Replay keeps only the keys the **target** rank owns,
recomputing ownership from the key with the target's own fan-out:

```
roundrobin       : key % world_size == rank
hash_roundrobin  : fmix64(key) % world_size == rank
continuous       : NotImplementedError (range-based; not reconstructible per key)
```

The delta's scope depends on how it was dumped: `incremental_dump(..., pg)`
all-gathers, so every rank holds the *global* delta and the filter hands each
rank its share; `pg=None` leaves each rank holding only its own keys, so that
delta belongs on the rank that produced it (elsewhere the filter drops all of it,
reported as `skipped` rather than as an error).

Because ownership is recomputed with the *target's* `world_size`, a globally
gathered delta reshards for free — an 8-rank dump can be replayed into a 4-rank
replica. `world_size == 1` skips filtering entirely. `meta["world_size"]` is
compared only to decide whether the table may be replayed at all, never to route
keys.

**Removals.** A key leaves a table two ways, and the two go to **separate
retained buffers**:

| | buffer | retained when | replayed? |
|---|---|---|---|
| evicted to make room | `evicted_keys` | `RETAIN_KEY` | never |
| removed by an explicit erase | `erased_keys` | the `erase` call's own `EvictedItemMode` | whenever non-empty |

Splitting them is what lets replay be correct *and* cheap. An eviction needs no
action on the replica: the key that took the evicted one's slot is in the very
same delta and overwrites it. An erase does, because nothing takes over that
slot. With both in one buffer a replay could not tell them apart and would have
to erase keys that were merely evicted -- dropping live data whose new owner had
not yet been dumped.

Splitting them also removed a piece of machinery. An earlier design carried a
`ReplayMode` enum in `meta` to say whether the removals were worth applying,
precisely because one buffer could not distinguish the two cases. With separate
buffers there is nothing left to declare: a non-empty `erased_keys` *is* the
signal, so the enum and the "an erase happened" flag it depended on are both
gone.

The erase runs **before** the upsert, so a key erased and re-added inside the
same window survives -- it appears in both `keys` and `erased_keys`.

Retention is configured at different times for the two, and for a reason.
Retaining evictions swaps in a collecting insert kernel and costs an extra kernel
output plus a device sync on *every* evicting insert, so it has to be a
table-level decision made up front (`DynamicEmbTableOptions.evicted_item_mode`).
An erase already holds its keys, so recording them costs a copy and nothing has
to be prepared -- which is why `erase` takes its own `EvictedItemMode` per call.

That also settles a `HybridStorage` question cleanly: only the host tier is a
*last* tier, so only it retains evictions (the HBM tier spills into it rather
than really evicting). Erases are not tier-specific -- `erase_keys` runs against
both tiers, passing its mode down -- so a key erased out of HBM is recorded just
the same, and `pop_erased_keys` drains both tiers.

---

## 7. Cache interaction

For a module with a `DynamicEmbCache` in front of `DynamicEmbStorage`:

1. `flush_cache(cache, storage)` — push dirty cache entries down, so the storage
   copy is authoritative before it is overwritten.
2. Replay into the storage.
3. Erase the replayed keys from the cache, so the next lookup refetches. (A full
   `cache.reset()` would also be correct but throws away unrelated entries.)

---

## 8. API surface (three layers, symmetric with the dump side)

```python
# model level -- dynamicemb/incremental_dump.py
def replay_increment(
    model: torch.nn.Module,
    deltas: Dict[str, DeltaDumpResult],
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[str, ReplayStats]]: ...

# module level -- BatchedDynamicEmbeddingTables
def replay_increment(self, delta, pg=None) -> Dict[str, ReplayStats]: ...

# storage level -- DynamicEmbStorage / HybridStorage
def replay_increment(self, table_id, keys, values, slot_index, insert_score,
                     content=ReplayContent.ALL, timestamp=None
                     ) -> ReplayStats: ...
def erase_keys(self, table_id, keys) -> int: ...
```

`ReplayStats` (in `dynamicemb/types.py`, alongside the other shared dataclasses)
is a plain work report -- replay either succeeds or raises:

```python
@dataclass
class ReplayStats:
    upserted: int   # keys written back at their source slot
    erased: int     # keys actually removed (REMOVE_THEN_OVERWRITE only)
    skipped: int    # keys filtered out as not owned by this rank
```

---

## 9. Changed Files

| File | Change |
|---|---|
| `src/table_operation/kernels.cuh` | new `scatter_keys_at_slots_kernel` |
| `src/table_operation/insert.cu` | `table_scatter_keys_at_slots` implementation |
| `src/table_operation/table.cuh` / `table.cu` | declaration + pybind registration |
| `dynamicemb/scored_hashtable.py` | `LinearBucketTable.scatter_keys_at_slots` |
| `dynamicemb/key_value_table.py` | dump-side `slot_index` / `_split_value_row` / `_dump_score_block` (+ age transform); storage-level `replay_increment` / `erase_keys`; `_fresh_score_block`; generalised all-gather |
| `dynamicemb/batched_dynamicemb_tables.py` | module-level `replay_increment`; `meta` layout fields; single `cur_ts` |
| `dynamicemb/incremental_dump.py` | model-level `replay_increment`; `DeltaDumpResult` doc |
| `dynamicemb/types.py` | `ReplayStats` |
| `dynamicemb/dynamicemb_config.py` | `ReplayContent`, `EvictedItemMode` as a flag |
| `dynamicemb/__init__.py` | export `replay_increment`, `ReplayStats`, `pop_erased_keys`, `ReplayContent` |
| `dynamicemb/batched_dynamicemb_function.py` | split slot vs value-row indices in the HBM-direct prefetch (§12) |
| `DynamicEmb_APIs.md` | document both |
| `test/unit_tests/incremental_dump/test_replay_increment.py` | new |
| `test/unit_tests/test_no_eviction_row_indexing.py` | new (§12 regression) |

---

## 10. Tests

`test/unit_tests/incremental_dump/test_replay_increment.py` (single process, one
`BatchedDynamicEmbeddingTablesV2` as source and another as target):

| Test | Covers |
|---|---|
| `test_replay_round_trip` | keys, embeddings and **slot_index** all round-trip; parametrised over TIMESTAMP / STEP / LFU / compound `(TIMESTAMP, LFU)` / NO_EVICTION |
| `test_dump_splits_the_value_row` | `values` + `optimizer_states` widths add up to the table's value row; a table with no per-row state reports `None`; parametrised over SGD / ROWWISE_ADAGRAD |
| `test_dump_scores_are_column_aligned` | `scores` is `[N, num_scores]` in logical order, and the LFU column is verbatim (keys touched six times outrank those touched once) |
| `test_replay_does_not_carry_scores` | the deliberate gap: a threshold that splits the source's table splits nothing on the replica, because every restored key carries the same fresh score |
| `test_replay_is_idempotent` | replaying the same delta twice is a no-op |
| `test_replay_rejects_layout_mismatch` | capacity mismatch raises **and leaves the target untouched** (no partial write) |
| `test_replay_rejects_score_strategy_mismatch` | a genuinely different score layout raises |
| `test_replay_accepts_swapped_score_order` | `(TIMESTAMP, LFU)` vs `(LFU, TIMESTAMP)` is the *same* physical layout, so it must replay rather than be rejected |
| `test_replay_rejects_missing_table_options` | a delta from an older version raises instead of writing blind |
| `test_replay_applies_erasures_and_ignores_evictions` | `erased_keys` is applied, erase before upsert, an erased-then-readmitted key survives; a non-empty `evicted_keys` changes nothing |
| `test_replay_content_restores_optimizer_state` | with `OPTIMIZER_STATE` the source's state lands even on a row the key is taking over, where the default would have initialised it |
| `test_replay_content_restores_scores` | with `SCORE` the replica reproduces the source's ranking -- the mirror of the test above it |
| `test_replay_without_embedding_needs_aligned_rows` | omitting `EMBEDDING` raises when a key does not already own its target row, and works once the replica is aligned |
| `test_erased_and_evicted_use_separate_buffers` | an erase and an eviction land in different buffers, neither leaks into the other, both are read-and-clear |
| `test_erase_retention_is_per_call_not_per_table` | the same table records one erase and not the next; a table that discards evictions still reports its erases |
| `test_replay_erased_counts_only_keys_actually_present` | `erased` reports removals that really happened, not removals asked for |
| `test_replay_tolerates_an_empty_removal_list` | nothing to remove is normal, not an error |
| `test_replay_optimizer_state` | a key already at its target slot keeps its optimizer state, a fresh row is initialised |
| `test_replay_with_caching` | a lookup after replay sees the replayed value, not a stale cached one |

| Test | Covers |
|---|---|
| `test_replay_increment_distributed` | the sharding path: each rank keeps only its own keys (`upserted + skipped == total`), the ranks' shares sum to the whole delta, and the replica's dump matches the source's keys/values |

Not covered yet: replaying into a **differently sized** world (it needs two world
sizes in one job).

---

## 11. Fixed along the way: NO_EVICTION slot-vs-row in the forward path

Adding the `NO_EVICTION` parametrisation to `test_replay_round_trip` surfaced a
pre-existing fault that had nothing to do with replay: a plain **second forward
pass** over already-resident keys crashed with `an illegal memory access` inside
`load_from_flat` (`dynamic_emb_op.cu:561`). It reproduced with every dump and
replay call removed, so it was not introduced here -- but it blocked the
`NO_EVICTION` coverage, so it is fixed in this change.

**Cause.** NO_EVICTION is the one strategy where the value row is not the hash
slot: rows come from a per-table auto-increment counter (`key_value_table.py`
`_get_no_eviction_insert_scores`), and its key map is sized
`ceil(init_capacity / 0.5)` -- **twice** the value buffer
(`key_value_table.py:357-367` vs `:403-411`). `_prefetch_hbm_direct_path`
translated slot to row on the **insert** branch only:

```python
new_indices = (score_arg.value.to(torch.int64)          # the row
               if state.no_eviction_next_index is not None else new_indices)
...
if h_num_missing == 0:
    return indices, indices.clone(), None               # the raw hash slot
```

so a batch whose keys all hit returned slots, the forward fed them to
`load_from_flat`, and roughly half of them (slots are uniform over a space twice
the buffer) addressed rows past the end of it. The helper for exactly this,
`_flat_row_indices_for_value_load`, was never called on that path.

**Fix.** Carry the two index spaces separately instead of hoping they coincide.
`PrefetchState` gains `value_row_indices` / `update_value_row_indices`, `None`
meaning "same as the slot tensors" so nothing extra is allocated for the other
strategies. Consumers pick by what they address:

| Consumer | Index space |
|---|---|
| `load_from_flat` (forward), `fused_update_for_flat_table` (backward) | **row** |
| `increment_counter` / `decrement_counter` (ref counter beside the hash table) | **slot** |

That also corrects a second, quieter defect on the same line: the old code
overwrote `new_indices` with the row *before* `increment_counter`, so newly
inserted NO_EVICTION keys pinned the wrong counter entry while looked-up keys
pinned the right one.

Regression coverage: `test/unit_tests/test_no_eviction_row_indexing.py` drives
resident keys through the forward and checks each row against its own key's DEBUG
pattern (`key % 100000`), so a wrong-row read fails on values, not merely on a
crash.
