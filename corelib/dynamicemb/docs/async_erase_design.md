# Async erase by score · Design

> Branch: `feat/dynamicemb-async-erase`, on top of PR #530 (per-key score) and the
> merged #515 (value row per key). Status: **implemented and tested**; §8 says what the
> tests cover and §9 what is still open. User-facing description: `erase_by_score_async`
> in `DynamicEmb_APIs.md`.
>
> Where a number is *measured* it says so; where it is *inferred* it says that. A number
> from an earlier draft that a later measurement overturned is kept and marked, in §4.8.

---

## 1. Requirement

- A **module-level** API, shaped like `DynamicEmbDump`: it takes a model, finds every
  `BatchedDynamicEmbeddingTablesV2` in it, and erases from each. The same call is a
  method of the module.
- Erase rule: **a key whose score is strictly below the given threshold is erased.**
- It runs on a CUDA stream and a thread of its own, **concurrently** with prefetch,
  forward and backward, so the training loop does not stall on it.
- The name carries `async`; the caller gets a handle to wait for.

## 2. Scope

| Storage mode | Supported | Why |
| --- | --- | --- |
| `HBM_DIRECT` (`DynamicEmbStorage`, values in HBM) | yes | in-flight keys are pinned by the storage's own ref counter |
| `CACHE` (`DynamicEmbCache` over a `DynamicEmbStorage`) | yes | only the backing storage is erased, the cache is left alone; see §5 |
| `DEFAULT` (host values, no `PrefetchState`) | **no** | no ref counter: a key can be erased between forward and the key-based write-back in backward |
| `HybridStorage` | **no** | two tiers, no ref counter in use |
| `CACHE` with an `external_storage` (`CACHING_PS`), or the PS alone (`HOST_PS`) | **no** | erase drops data that is no longer used, so it has nothing to do with a PS; the table is rejected rather than left with a stale copy there (§5, consequence 2) |

An unsupported layout is rejected at call time (an error, not a silent skip), but only
when a table of the module would be scanned: a module with no table that qualifies
answers with an already-finished handle, whatever its layout.

## 3. What the code was before this work

1. **Streams.** `prefetch` runs on whatever stream is current and remembers it as
   `prefetch_stream`. There was no background-maintenance stream, and no cross-stream
   event or CPU lock on the table path.
2. **A prefetched batch borrows slots until backward.** Prefetch resolves keys to
   `slot_indices` (kept in `PrefetchState`) and `increment_counter`s them; backward
   `decrement_counter`s them (`_prefetch_cache_path`, `_prefetch_hbm_direct_path`,
   backward).
3. **The ref counter is the existing pin.** The evict kernel re-reads `counter[slot]`
   after `try_lock` and skips a slot with `counter > 0`. `_ref_counter` used to be
   allocated for every scored key map though only `DynamicEmbCache` and HBM
   `DynamicEmbStorage` use it; since `7efc056` only `HBM_DIRECT` storage allocates it
   (`use_ref_counter=False` for the storage under a cache, for host-value storage and
   for both `HybridStorage` tiers), and the kernels skip the check when the pointer is
   null.
4. **`table_erase_kernel` erases by key** and ignores the counter: it `try_lock`s the
   slot, clears the score block and digest, marks the slot reclaimed (its value row
   stays with the slot) and decrements `bucket_sizes`. There was no erase-by-score
   kernel; `export_batch` / `incremental_dump` already scanned with a
   `score >= threshold` predicate.
5. **Table expansion replaces the key map.** `_expand_tables_impl` builds a
   `new_key_index_map` and swaps `state.key_index_map`. It runs inside `prefetch`
   (`storage.expand_if_need`).
6. **The pybind bindings did not release the GIL** (no `gil_scoped_release` /
   `call_guard` in `src/`).

## 4. Concurrency

### 4.1 Why a background thread

If erase ran on the caller's thread with only a different CUDA stream, the caller could
not return until every kernel was *launched*, including any host sync on the way. That
blocks the training thread and defeats the overlap. So erase is launched from a
**background Python thread** (`AsyncEraser`: one `ThreadPoolExecutor(1)` per module, so
erases of a module run one at a time, in the order they were asked for) and the call
returns a handle at once.

`torch.cuda.current_stream()` and the current device are per thread, so the worker sets
both itself. Two Python threads now touch the same storage, so CUDA ordering alone is
not enough: a **CPU lock** is needed as well (§4.4).

### 4.2 Consistency target: "best effort, spare what is in use"

A key a batch is using should not have a score below the threshold; if one does, it is
**spared**, not erased. So erase honours the ref counter but need not wait for in-flight
batches. Forward and backward are covered by the counters once prefetch has incremented
them, so erase only has to be mutually exclusive with **prefetch**, not with
forward/backward. (Checked: for `HBM_DIRECT` and `CACHE`, forward and backward only
write values and `decrement_counter`; the paths that touch the key map in forward or
backward are `DEFAULT`'s, which is rejected.)

### 4.3 The windows that need mutual exclusion

- **`HBM_DIRECT` prefetch's find → `increment_counter`.** Between the two, a found slot
  is not yet pinned; an erase on another stream could reclaim it, and the batch would
  then train a row that no longer belongs to its key. The cache path has no such window
  for erase, since it does not consult the cache (§5).
- **Table expansion.** The swap of `state.key_index_map` and the rehash must not
  interleave with an erase that holds the old key map.
- Host-side `estimated_table_sizes` / `collect_table_sizes` read a size that erase
  changes. A stale read could make a table that never evicts grow once more than it
  needs; the sliding-window test (§8) did not see that after the table had grown to its
  window, at the sizes it uses.

### 4.4 The gate

`TableAccessGate` (`async_erase.py`), one per `BatchedDynamicEmbeddingTablesV2`
(`module._table_gate`). A section is one `with gate():`.

```python
with self._lock:                                   # host: one section at a time
    stream = torch.cuda.current_stream()
    if self._last_access is not None:
        stream.wait_event(self._last_access)       # device: after the section before
    try:
        yield
    finally:                                       # also after a failure
        event = torch.cuda.Event(); event.record(stream)
        self._last_access = event
```

A lock orders *launches*, not GPU execution: two threads that launch on two streams in
lock order may still run out of that order on the device. So the holder, on leaving,
records an event on the stream it launched on, and the next holder's stream waits for it
before anything it launches. The kernels of the sections are then one chain on the
device, in the order of the lock. Forward and backward never enter, so their streams are
not part of the chain. The lock is held while *launching*, never while waiting for the
device. Waiting on an event recorded on the same stream is a no-op, so the gate also
works when both sides use one stream (without overlap). A copy or pickle of a gate or a
worker is a new one (a lock and an event belong to one process and device).

**Who takes it.**

- **Prefetch** (`dynamicemb_prefetch(..., table_gate=...)`) takes a section from
  `expand_if_need` through the find and the `increment_counter`, for both storage
  modes. Two things were moved to before the section, because they do not read the
  tables: `expand_table_ids_cuda` and the choice of `storage_mode`. The rest of
  `dynamicemb_prefetch` is unchanged apart from indentation.
- **Erase** takes a section per chunk of its scan (§4.8, `_erase_state_by_score`). Inside
  it the key map and the table's bucket range are read **again** each time: between two
  chunks a prefetch may have expanded the table and freed the old key map, so a plan made
  earlier would address a buffer that no longer exists.

`key_map_generation` (a counter on `DynamicEmbTableState`, bumped by
`_expand_tables_impl` whenever it replaces `state.key_index_map`) tells the scan the key
map was replaced: a rehash moves every key to a new slot, so "where the scan had got to"
means nothing in the new one, and **that table's scan starts again from its first
bucket**. Erasing by score is idempotent, so this is safe. The counter is per storage,
not per table, so an expansion of another table of the same storage also restarts this
scan (§9). (Holding a reference to the old key map and comparing identity is not used: it
would keep a replaced, possibly very large, key map alive for as long as the erase runs.)

Alternative that was weighed and left: fuse the find and the increment into one kernel so
the first window disappears. It does not help expansion, so the lock is needed anyway.

### 4.5 The lock is fair

**What was found.** The first version used `threading.Lock`. A stress test of three scans
in one-bucket chunks beside training passed in some runs and failed in others (`the erases
never finished`). An investigation on an H100 node found:

- Measured: the worker's step is ~18.7 chunks per training step; three scans need ~25,000
  chunks, so they would end around step 1,500. The runs that passed were carried by a few
  steps (128 of 41,000, 0.3%) in which the training thread waited **6 to 136 ms** for the
  lock, and the worker ran a whole scan meanwhile (28% of all chunks).
- Measured: those waits are lock barging. A thread that lets go of a `threading.Lock` and
  asks again at once takes it again: the worker's gap between its release and its next
  acquire was a median 1.22 µs, while a woken waiter needs 2 to 8 µs (a pure-Python toy on
  the node: 2.1 to 4.8 µs, by placement of the threads) to be scheduled. Pinning the two threads on SMT siblings
  removed the stalls, across sockets raised them.
- Measured: the pass probability of the same code moved between 15% and 83% with changes
  of a few tenths of a microsecond in the worker's loop. A failure of the original test
  was therefore one draw from that distribution (the test's pass rate was 53%, 61 of 116),
  not a hang. Not established: which acquisition decided the one failing run.
- Measured: with a FIFO hand-off lock, the training thread's wait for the gate had a
  p99.9 of 0.03 ms and 0 stalls in 60 rounds.

**What is done.** `FairLock` (`async_erase.py`) hands the lock to the first waiter
*itself* on `release`, and `_held` stays true: the lock is never free while someone
waits, so a thread that lets go and asks again queues behind. A waiter interrupted while
it waits leaves the queue, or, if the lock had already been handed to it, passes it on.
With two parties (one training thread, one erase worker) each waits for at most the
section the other is in. Cost, measured: the training thread's wait for the gate p50
20 µs against 15 µs.

**What it does and does not buy.** Measured with chunks of 2^24 slots on a big table
(§4.8), the training step with the old lock and with the fair one are the same (4.68 and
4.57 ms, chunks unbounded): at that chunk size the step is bound by what is on the
device, not by the lock. The fair lock is what keeps the wait bounded when chunks are small, and what
makes the stress test a test of the code and not of the scheduler (120 repetitions in
three processes, all passed).

### 4.6 Bounding what is on the device

Launching a chunk takes microseconds and running it does not, so a scan left to itself
queues every chunk of the table at once. The prefetch that takes the gate next has its
stream wait for the scan's *last* event, so it waits on the device for all of them.

`_erase_state_by_score` therefore keeps at most `max_in_flight` chunks launched and not
yet finished: before launching a chunk it waits -- **outside** the gate, holding nothing,
and without the GIL (`Event.synchronize`) -- for the chunk `max_in_flight` before it.
What the next prefetch can wait for on the device is then bounded by that many chunks.
Default 2: one chunk running, the next already queued.

Measured (§4.8): without the limit the training step went from 0.72 to 4.57 ms while
scans ran; with 1, 2 and 4 chunks in flight it was 0.91, 1.47 and 1.86 ms, and the scans
finished in the same window went from 120 to about 40, 110 and 120. The wait in the
unbounded case was as long as one whole scan (3.4 ms on that table), which is what the
mechanism predicts.

### 4.7 The GIL

None of the older `dynamicemb_extensions` bindings releases the GIL. A binding that only
**launches** holds it for microseconds. A binding that **waits for the GPU inside C++**
(`flagged_compact`, `index_calculation.cu`: `num_selected.cpu().item<int64_t>()`) would
hold it for as long as the stream's queued work takes, and the training thread could not
launch a kernel meanwhile. PyTorch's own ops (`.item()`, `.cpu()`, `Event.synchronize()`)
release it themselves.

Rules, kept by this work: every **new** binding the erase uses (`table_erase_by_score`)
releases the GIL (`py::call_guard<py::gil_scoped_release>()`; they take only tensors and
integers), and the worker calls only launch-only bindings and PyTorch ops. Existing
bindings are not changed; the scan needs none of the blocking ones (counts come from
device-side atomics, the worker waits with `Event.synchronize()`). A later change that
makes the worker call a blocking binding (e.g. `flagged_compact` for `RETAIN_KEY`) must
give it the same guard first.

The worker's Python between two chunks still needs the GIL. Measured per chunk with
nothing else running: 14.6 µs of worker CPU (two `.item()` of a CPU tensor, 1.0 µs each;
`torch.cuda.current_stream()`, 3.1 µs; an event, 0.5 µs, its record, 0.6 µs, a
`wait_event`, 0.7 µs; the wrapper of the binding, 5.7 µs of which the binding 2.7 µs); and
30 to 32 µs under training, with the worker blocked about 70% of the time (inferred: the
training thread's GIL-free time per step decides how many chunks the worker gets, 19 per
step at the 1.6 ms step measured there, 89 when the training thread slept 1 ms per step).
This is why a small chunk is dominated by overhead.

### 4.8 Scan cost, chunk size and what the scan costs the training (measured)

**Method.** One H100 80GB HBM3. One table, 2^29 slots, bucket 128, `CUSTOMIZED`. A scan
that reads every slot and changes nothing (threshold 0). Training steps of 4096 keys,
forward + backward + synchronize, timed on the host (0.66 to 0.72 ms without a scan).
Scans submitted back to back, 120 at a time, and the steps taken while they run counted;
each condition three times, interleaved. Two tables: nearly empty (300,000 keys) and half
full (268 million keys, a quarter scored 100 and a quarter 300).

**One scan alone** (ms; nearly empty / half full):

| chunk (slots) | chunks | nearly empty | half full |
| --- | --- | --- | --- |
| 2^18 | 2048 | 47 | 46 |
| 2^20 | 512 | 12.2 | 11.6 |
| 2^22 (default) | 128 | 3.0 | 3.7 |
| 2^24 | 32 | 2.5 | 3.4 |
| 2^26 | 8 | 2.4 | 3.3 |
| whole table | 1 | 2.0 | 3.0 |

The kernel reads the key of each slot and the score only of a live one: ~2 TB/s
(derived: 2^29 slots, 8 bytes each, plus 8 more for the half that is live, in 3.0 ms).
A small chunk is bound by the cost of launching it (about 23 µs a chunk at 2^18 to 2^20);
chunking costs +23% at 2^22 and +13% at 2^24 over the whole table on the half-full one.
**A scan that erases** is slower: removing 134 million keys took 21.4 ms (chunks of 2^24),
against 3.4 ms for one that finds nothing.

**The training step while scans run** (chunk 2^24, mean ms; nearly empty / half full):

| condition | nearly empty | half full | scans done in 300 steps (half full) |
| --- | --- | --- | --- |
| no scan | 0.66 | 0.72 | |
| chunks unbounded, fair lock | 2.18 | 4.57 | 120 |
| chunks unbounded, `threading.Lock` (as it was) | 2.06 | 4.68 | 120 |
| 1 in flight | 0.81 | 0.91 | ~40 |
| 2 in flight | 0.93 | 1.47 | ~110 |
| 4 in flight | 1.11 | 1.86 | 120 |

**By chunk size** (2 in flight, mean ms; half full): 2^20 0.78 and 3 scans in 300 steps;
2^22 0.80 and 13 to 15; 2^24 1.51 and 115; 2^26 6.12 and 120. (Nearly empty: 0.79, 0.79,
0.92, 4.50.)

**What follows.**

- The cost to the training of a chunk is the length of its kernel: the training's own
  small kernels wait behind it. A chunk of 2^22 slots slows a step by about 10%, 2^24 by
  110%, 2^26 by 8×, and a smaller chunk makes the scan take longer under training
  (inferred from the counts: ~17 ms a scan at 2^22 against ~4 ms at 2^24). Derived, not
  measured directly: the total extra delay per scan, summed over steps, is about the same
  for 2^22 to 2^26 (roughly 2 to 3 ms) and larger for 2^20 (~6 ms): the size chooses
  between a short, high disturbance and a long, low one.
- **Chunk size: 2^22 slots (4,194,304)**, a parameter (`chunk_slots`), default
  `ERASE_CHUNK_SLOTS`. **In flight: 2**, a parameter (`max_in_flight`), default
  `ERASE_MAX_CHUNKS_IN_FLIGHT`. These favour the training's latency for a scan that is
  allowed to take some tens of milliseconds; a user who wants the scan over sooner can
  raise either.

**What the numbers leave out.**

- The scans read only; erasing is ~6× slower per scan and its effect on the training
  *while it erases* was **not measured**.
- One GPU, one table, a step of 0.7 ms. A real step is longer, so the same disturbance
  should cost it proportionally less (inferred), and the best chunk may move with the
  step's length.
- Training kernels were light; with heavier ones sharing the memory bandwidth the scan
  takes longer, and so does the wait.

**An earlier draft's numbers, overturned.** The draft took a stand-in kernel
(`table_count_matched`) for the scan: ~500 GB/s, 32 ms for a 1 G-slot table, a 2^24 chunk
of ~0.5 ms, "chunking costs 0.4%", and a stall "of tens of milliseconds" without a bound.
The erase kernel is several times faster than the stand-in, so none of those holds: a
2^24 chunk is ~0.1 ms here, chunking costs +13%, and the unbounded stall is one scan, 3 to
6 ms on the tables measured (tens of milliseconds would take billions of slots).

**Bounds tied to the table.** The chunk size is a target, not the chunk itself
(`next_erase_chunk_end`). A chunk is always a whole number of buckets, always inside one
table, and always of the key map that is current:

1. *Whole buckets.* `chunk_buckets = chunk_slots // bucket_capacity`, a chunk is
   `chunk_buckets * bucket_capacity` slots.
2. *Clamped to the table.* `chunk_buckets = max(1, min(num_buckets_of_table,
   chunk_slots // bucket_capacity))`: a table smaller than a chunk is one chunk, and a
   chunk is never less than one bucket.
3. *Never past the table.* The tables of a storage share one `table_storage`; table `t`
   owns buckets `[table_bucket_offsets[t], table_bucket_offsets[t+1])`. A chunk starts and
   ends inside that range and never reaches into the next table.
4. *The current key map, read again for every chunk* (§4.4), and a table's scan restarts
   if its key map was replaced.

## 5. The cache case: erase only the backing storage

With a cache, the keys to erase are the **storage's** keys, and the erase **does not
touch or consult the cache.** A cached copy of an erased key is left alone; it goes away
the way a cached copy of any key storage has evicted does.

Why that is safe (checked, not assumed):

- **Scores live only in storage.** The cache ranks by timestamp, and prefetch
  `_find_keys` every key in storage, hits included (step 2 of `_prefetch_cache_path`).
  "Score below the threshold" is decided on storage.
- **"In storage but not in cache" and "in cache but not in storage" are both already
  legal states.** `_write_back_evicted` looks each key the cache evicts up in storage: one
  still there only has its value updated (`preserve_existing=True`); one storage has since
  evicted goes to `external_storage` if there is one, otherwise it is dropped, and
  recorded as an evicted key when the storage retains evictions. A storage eviction
  already produces that state without asking the cache; an erase is one more way to reach
  it.
- **The storage under a cache has no ref counter.** A cache-mode prefetch pins *cache*
  slots; the storage's counter was never touched, so it is no longer allocated. The erase
  kernel therefore takes the counter as an optional pointer: null (cache mode) skips the
  check, non-null (`HBM_DIRECT`) re-reads `counter[slot]` after `try_lock` and spares a
  pinned slot (the evict-kernel pattern). In cache mode `spared` is always 0.

Accepted consequences (cache mode):

1. Until the cache evicts it, an erased key can still be hit, served, and trained; its
   updates are then **lost** at write-back (the key is no longer in storage), or sent to
   the PS / recorded as an evicted key if those exist.
2. Tables with an `external_storage` are **not supported** (§2), so there is no PS for a
   cached copy to be written back to and nothing to go stale there. For the record, the
   argument that settled it: a lookup asks the cache first and the PS only for keys both
   the cache and the storage lack (`_lookup_next_tier`), and the cache's eviction writes
   the key back over whatever older copy the PS held (`_write_back_evicted`), so a cached
   key never reads a stale PS value. What it did not cover is a key that is not cached
   when it is erased but has an older copy in the PS (nothing deletes from the PS when a
   key is promoted from it); that key would reappear with the older value. Rejecting the
   layout avoids both.
3. With `RETAIN_KEY` (not part of this work, §5.1), such a key would show up in
   `pop_evicted_keys`, not in `pop_erased_keys`.

What still needs exclusion with prefetch:

- `HBM_DIRECT`: the find → `increment_counter` window (§4.3).
- Both modes: table **expansion**, which swaps `state.key_index_map` of the storage (a
  cache never rehashes).
- Checked: the **find does take the slot lock** when it updates a score
  (`table_lookup_kernel`, the non-`Const` policies: `bucket.try_lock(iter, key)` around
  `ScorePolicy::update`). The lock is a CAS on the slot's key from `key` to a locked
  marker, so it only succeeds while the slot still holds that key. Against
  `table_erase_by_score_kernel`, which takes the same lock before clearing the score
  block and marking the slot reclaimed:
  - erase first: the find's `try_lock` fails (the slot is locked, or holds
    `reclaimed_key`), the find reports the key **not found** and writes nothing. No stale
    score reaches a reclaimed slot.
  - find first: erase's `try_lock` fails and that key is **not erased this round**. A
    later erase retries it.
  Residual, both benign: (a) `Const` finds (`const_lookup`, eval) read the score without a
  lock, so they can report a key that is being erased as found; `_write_back_evicted` uses
  one, then `storage.insert(preserve_existing=True)`, so a key erased in that window can
  come back with a default score. (b) The `HBM_DIRECT` find → `increment_counter` window
  of §4.3 is still real, because the lock is released when the find returns, before the
  slot is pinned; the gate closes it. Not examined: the insert / insert-and-evict kernels'
  handling of a key that is erased under them.

### 5.1 Retain option and the erased-key buffer — **not implemented**

`RETAIN_KEY` is not part of this work, and nothing here blocks it. What a later change
would need, kept from the earlier draft: a `mode: EvictedItemMode` argument like
`erase_keys` (`DISCARD` by default); with `RETAIN_KEY` the keys actually removed go to the
state's erased buffer, read by `pop_erased_keys` (and, for a table that stores rows, the
`[M, 2]` (slot, row) of every reclaimed slot for `pop_erased_slot_index`); without it a
table that stores rows is marked `erase_untracked`, as `erase_keys` does. Today the
erased keys are **not recorded**: `pop_erased_keys` does not see what an erase by score
removed, and a table it removed a key from is marked `erase_untracked` (it reads back the
count of each table when the table is done), so `replay_increment` refuses that table's
deltas instead of leaving the replica with the deleted keys.

That buffer is not safe to touch from two threads (four parallel lists on
`DynamicEmbTableState` appended with separate calls; `_pop_chunked_keys` reads and
**reassigns** them so an append in between is lost; `erase_untracked_tables` is a plain
`set`; the chunks are tensors made on the erase stream). A change would need a lock on the
state guarding all four lists, the worker appending a chunk's keys, table ids, slots and
slot table ids in one critical section after the chunk has finished (event synchronised on
the worker thread), `record_stream` in the pops, and `erase_untracked` set under the same
lock. It would also have to give any binding it calls from the worker the GIL guard of
§4.7. The contract of §6 (nothing exports while an erase is outstanding) is what keeps a
reader from seeing a half-filled buffer.

## 6. Decided

- Module-level entry `erase_by_score_async(model, score_strategy, threshold, stream=None,
  chunk_slots=2**22, max_in_flight=2)`; finds `BatchedDynamicEmbeddingTablesV2` in the
  model the way `DynamicEmbDump` does (`get_dynamic_emb_module`), and starts one erase per
  module that has a table to scan. The same call is a method of the module. The function
  lives in `async_erase.py` (importing `dump_load` inside the function, because `dump_load`
  imports the tables and they import `async_erase`) and is exported from `dynamicemb`.
- Threshold rule: score **strictly below** the value → erase; an int in `[0, 2**64)`.
- **Stream: optional.** By default each module makes one stream of its own, when it first
  has a table to erase, and keeps it (`AsyncEraser.own_stream`); the scan then overlaps
  the training. A stream passed in must be a `torch.cuda.Stream` of the tables' device.
  The stream the training runs on, the default stream or the prefetch stream **are
  accepted** and the scan is queued among the training's kernels without overlap, which a
  `UserWarning` says. (An earlier draft required a dedicated stream and rejected the
  others; the requirement was for overlap, which a warning conveys without refusing a
  correct use.)
- Only `HBM_DIRECT` and `CACHE`; `DEFAULT`, `HybridStorage` and any layout with a PS are
  rejected at call time, when a table of the module would be scanned.
- Consistency: best effort, in-flight keys are spared and counted.
- A background thread per module, serial; the call returns an `EraseHandle`.
- Async erase overlaps **training only** (forward, backward, prefetch); concurrent dump /
  load / incremental dump / replay / `pop_*` / evaluation are not supported (the caller
  `wait()`s first). Documented, not enforced.
- **Handle.** `EraseHandle` wraps a `Future` (the launch is finished; carries the
  worker's exception) and a `torch.cuda.Event` recorded on the erase stream after the last
  chunk. `done()` (finished on the device, does not block), `wait()` (blocks until it has),
  `wait_stream(stream=None)` (blocks until everything is launched -- an event not recorded
  yet orders nothing -- then orders `stream` after it), `result()` → `(erased, spared)`.
  Any of them raises what the worker raised. A call with nothing to erase returns
  `EraseHandle.completed()`. A model with several modules gets an `EraseHandleGroup`, a
  subclass, so a caller holds one type; its `result()` adds up. The handle needs no stream
  of its own: the event is its position on the erase stream.
- **Arguments are checked before anything starts**, by both entries
  (`check_erase_arguments`): the strategy is a `DynamicEmbScoreStrategy` and not
  `NO_EVICTION`; the threshold an int (not a bool) in range; the stream a
  `torch.cuda.Stream` or `None`; `chunk_slots` and `max_in_flight` ints (not bool) of at
  least 1. Then, per module, the tables (`erasable_table_ids`) and the layout; a stream on
  another device than the tables is refused. The model function does all of it for every
  module before it starts the first, so an error leaves nothing erased.
- **Score column**: single-score tables only; a table that keeps two score words
  (`(TIMESTAMP, LFU)`) is an error when one of its members is named.
- **The call names the kind of table it erases from** (`DynamicEmbScoreStrategy`), because
  one threshold means different things for different strategies. **Strategy matching** is
  by member, on the part of the strategy that scores (`get_score_part`): `CUSTOMIZED`
  picks `CUSTOMIZED` and `(NO_EVICTION, CUSTOMIZED)`; `PRECISE_CUSTOMIZED` picks
  `PRECISE_CUSTOMIZED` and `(NO_EVICTION, PRECISE_CUSTOMIZED)`; `NO_EVICTION` alone keeps
  no score and is skipped; naming `NO_EVICTION` is an error. **No matching table** is not
  an error: the handle is already complete, count 0.
- **Threshold** is supplied by the caller, in the unit of the table's strategy. For
  `TIMESTAMP` the caller reads the device timer with `dynamicemb_extensions.device_timestamp()`.
- **Ranks**: each rank erases what it holds with the same threshold; no collective, and
  `result()` is the rank's own.
- **Prefetch participation: option A.** Prefetch takes the gate around the part that
  touches the tables (§4.4); erase takes it per chunk. Python only on the prefetch side. A
  fused find + increment kernel (option B) is a possible follow-up.
- **The gate's lock is fair** (§4.5) and **the scan keeps at most `max_in_flight` chunks on
  the device** (§4.6).
- **Chunk size: 2^22 slots**, in whole buckets, clamped to the table's own bucket range
  (§4.8), a parameter. **The key map is read again for every chunk**, and a table's scan
  restarts if its key map was replaced (`key_map_generation`).
- **`external_storage`: not supported.** A table whose layout has a PS (`CACHING_PS`,
  `HOST_PS`) is rejected at call time, like `DEFAULT` and `HybridStorage`.
- **Return value**: the number erased and the number spared (a key whose score is below
  the threshold but that was pinned, so it stays).
- **`NO_EVICTION` tables can be erased**, as `(NO_EVICTION, CUSTOMIZED)` /
  `(NO_EVICTION, PRECISE_CUSTOMIZED)`. This is the main use: start from an
  `init_capacity`, grow as keys arrive, then drop old data and let new keys use the room.
  What it relies on is in `731f58b`: the key map grows with the keys held, the value
  buffer with the rows handed out (never past the key map's slots), and a full bucket
  gives up an erased slot before a key in use. Scores of such tables must be at least 1
  (an erased slot is cleared to 0, so a key scored 0 ties with one).
- **GIL (§4.7)**: every new binding the erase uses releases it; the worker calls only
  launch-only bindings and PyTorch ops.
- **`RETAIN_KEY` is not part of this work** (§5.1). The erase kernel's per-slot result
  stays available so the keys and the slot index can be compacted out of it later.

## 7. Where it is

| What | Where |
| --- | --- |
| Kernel, one slot at a time: read the key; if live and `score < threshold`, `try_lock`, re-read the score under the lock, spare a pinned slot (`slot_is_pinned`, a volatile read of the counter), else clear the score block, write the reclaimed digest, unlock with `reclaimed_key`, `atomicSub(bucket_sizes)`; counts by ballot; grid-stride, 64-bit | `src/table_operation/kernels.cuh` (`table_erase_by_score_kernel`), `erase_by_score.cu`; binding releases the GIL |
| A reclaimed slot has a digest of its own (`reclaimed_digest`, apart from the empty one), written by every place that makes one: erase, erase by score, rehash placement, replay restore | `types.cuh`, `table_rows.cu`, `kernels.cuh` (`4ec7d48`) |
| Python wrapper of the kernel over a slot range `[begin, end)` of one table | `scored_hashtable.py`: `LinearBucketTable.erase_by_score` |
| Chunks, `key_map_generation`, the in-flight limit | `key_value_table.py`: `next_erase_chunk_end`, `_erase_state_by_score`, `DynamicEmbStorage.erase_by_score`, `ERASE_CHUNK_SLOTS`, `ERASE_MAX_CHUNKS_IN_FLIGHT` |
| `FairLock`, `TableAccessGate`, `AsyncEraser`, `EraseHandle`, `EraseHandleGroup`, `check_erase_arguments`, `erase_by_score_async` | `async_erase.py` |
| Prefetch's section | `batched_dynamicemb_function.py`: `dynamicemb_prefetch(table_gate=...)` |
| The module's gate and worker, `erasable_table_ids`, the method | `batched_dynamicemb_tables.py` |
| No ref counter where nothing pins; `NO_EVICTION` reuse and the `FLAG` policy | `7efc056`, `731f58b` |

## 8. Tests and what they do not cover

`test/unit_tests/erase_async/` (`test_erase_async.sh` runs it, and a `torchrun` of two
ranks). 127 pytest items and one distributed script, passing on an H100 node.

| File | What it covers |
| --- | --- |
| `test_erase_by_score.py` | the kernel through the table wrapper: exactly the keys below the threshold (thresholds, with and without a counter), chunked = whole, only the named table, a pinned slot is spared (tables 0 and 1), an erased slot takes a new key and the sizes stay right, argument checks, the digest of a reclaimed slot, the binding's tensor checks |
| `test_storage_erase_by_score.py` | the chunk rule (a pure function) and its tiling at all sizes, the scan of a storage in chunks, with one, two or eight in flight, erasing the same keys each way (guard once per chunk, untouched tables untouched), a key map replaced between two chunks restarts that table, a table the scan removed keys from is marked `erase_untracked` and no other, a table that is not there is refused first, the in-flight limit (1, 2, 4, with a control that shows the chunks pile up without it) |
| `test_async_erase.py` | the lock (order, never free while one waits, a waiter interrupted before and after the hand-over, a hot loop does not keep another thread out of the gate), the gate (sections ordered on two streams, with a control; a failed section lets go; copies), the worker and the handle (thread, stream, counts; serial; the three moments; every way of asking raises; `wait_stream`), a real prefetch held back by the chunk the scan is in, and the stress test (three scans in chunks of eight buckets while training grows the table; no step limit) |
| `test_erase_api.py` | strategy matching, argument checks (also for a model with no module), two-score-word tables, layouts refused (two real, two stand-ins) only where a table is picked, streams (type, device, the ones the training uses warn, the default one, kept), the new parameters (checked, passed through), nothing started when one module is refused, a model with several modules |
| `test_erase_end_to_end.py` | the sliding window of a table that never evicts (does not grow), training beside the call, by time with `device_timestamp()`, a key a prefetched batch uses is spared until its backward, the same key in two tables with opposite ages, a cache (only the storage is erased, the cache is unchanged) |
| `test_erase_distributed.py` (`torchrun`, 2 ranks) | `erase_by_score_async` on a `DistributedModelParallel` of row-wise `EmbeddingCollection`: per table, per key and per key in tables that never evict, with and without `use_index_dedup`, and beside training; the ranks erase every old key once between them; a host-only layout is refused |
| elsewhere | `test_table_rows.py` (digest of a reclaimed slot: rehash, restore), `incremental_dump/test_replay_increment.py` (a delta dumped after an erase by score is refused by `replay_increment`; one after a call that removed nothing is replayed), `table_operation/` (`test_table_without_ref_counter`), `test_no_eviction_row_indexing.py` (the `FLAG` policy, reuse) |

**Not covered, chosen:**

- Nothing proves the gate closes the find → `increment_counter` window. The sparing is
  tested at the table and at the module (a key held between forward and backward), but
  no test fails when the gate is removed. A test would hold the prefetch between find and
  increment and show the erase cannot get in.
- Only `CUSTOMIZED`, `PRECISE_CUSTOMIZED` and `TIMESTAMP` tables are actually erased
  (`LFU` and `STEP` are only named in the "no table" paths).
- No randomized test against a reference model with inserts, lookups, erases and
  expansions interleaved.
- The `external_storage` layouts are rejected by layout; a module with a real PS is not
  built.
- Distributed: one node, two ranks, row-wise only; no cache, no table-wise or
  table-row-wise sharding.
- No test of the default chunk on a table big enough for several chunks (the tables of
  the module tests are below 2^22 slots); the multi-chunk walk is tested at the storage
  level with small chunks, and the real thing only in the measurements of §4.8.
- The stress test reaches an expansion during a scan in about half of its runs (inferred
  from the chunk counts of 120 runs: ~190 chunks without a restart, ~310 with one); the
  restart is tested for certain in the test that holds the worker inside a chunk.
- Performance is measured by scratch scripts, not by a test.
- No line or branch coverage was measured.

## 9. Known limitations and open points

- **The restart is per storage, not per table.** An expansion of table B restarts the
  scan of table A of the same storage (its slots did not move; the restart is wasted
  work, not a wrong result). A counter per table would fix it.
- **`spared` can count a pinned slot twice** if the table is scanned again after an
  expansion. `erased` is not affected.
- **The cost of erasing, not only scanning, on the training was not measured** (§4.8). The
  defaults are chosen on read-only scans.
- Two parameters are exposed so a user can move the trade: `chunk_slots` and
  `max_in_flight`.
- **Not enforced:** that nothing exports while an erase is outstanding.
- Keys erased by score are **not recorded** (`pop_erased_keys`, `replay_increment`) (§5.1). A table the scan removed a
  key from is therefore marked `erase_untracked` (sticky), like one erased by `erase_keys` without `RETAIN_KEY`, and
  `replay_increment` refuses its deltas: without the mark a delta looked complete and a replica kept the deleted
  keys and their rows. A scan that removes nothing does not mark.
- A **cache** may write an erased key back to the storage when it evicts it, with the
  value it holds (§5); not tested.
- **Multi-table modules and the evict kernels:** the way the evict kernels indexed the ref counter
  (§10) was wrong for tables after the first; it is fixed upstream in #514, which this branch is
  rebased onto. This work did not depend on it: the erase kernel always used the global slot.
- The `FairLock` costs ~5 µs more per section than `threading.Lock`; at the default chunk
  that is below the noise of §4.8.

## 10. Found on the way: the ref counter indexing of the evict kernels (fixed upstream in #514)

While checking how the ref counter is indexed for the erase kernel, the evict kernels
turned out to read it with a table-relative offset (`kernels.cuh`, `counter_offset =
(bucket_id - bkt_begin) * capacity`) while `increment_counter` writes it by global slot.
For table 0 they agree; for table 1 and later of a multi-table storage the evictor read
table 0's counters, so a pinned slot was evicted as if it were free. Measured: two tables
of 512 slots, every resident key of table 1 pinned, 2000 new keys inserted -- all 512
pinned keys lost; the same on table 0 or on a single table loses none. `erase_by_score`
uses the global index and was not affected.

It was kept out of this work, and then fixed upstream by #514 (`0f5c0a0`): `counter_offset =
bucket_id * bucket.capacity()` at both sites (`insert_body`, `insert_and_evict_body`), with
a regression test, `table_operation/test_multitable_ref_counter.py`. This branch is rebased
onto it, so the evict kernels and the erase kernel now index the counter the same way, by the
slot's position in all the tables.
