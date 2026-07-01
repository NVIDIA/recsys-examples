# Fused Sparse Cache–Storage Prefetch

## Summary

Replace `_prefetch_cache_path` completely with:

1. Cache `find_or_insert`.
2. Storage metadata processing and one bidirectional value-exchange kernel.
3. Admission over full tensors using `founds` as the sparse mask.
4. Direct initialization of admitted rows and reclamation of rejected provisional slots.

There will be no legacy-prefetch fallback for deterministic mode or external storage.

## API Changes

- Add `Cache.find_or_insert(...)`, returning full-length slots/cache-hit flags plus input-aligned eviction tensors selected by a device-resident boolean mask.
- Add mandatory `Storage.exchange(...)` for every backend configured with caching.
  - Its request carries the shared, concretely typed `DynamicEmbTableState` descriptor from `types.py`; it does not use an untyped `Any` cache-state field.
  - `DynamicEmbStorage` uses mapped buffers and fused kernels.
  - External storage implementations own their exchange implementation; the existing reference `PyDictStorage` is updated accordingly.
  - Configuration fails immediately when a caching backend does not implement `exchange`.
- Modify the existing admission APIs instead of adding parallel sparse methods:
  - `Counter.add(keys, table_ids, frequencies, founds)` skips positions already found and returns full-length frequencies.
  - `Counter.erase(keys, table_ids, mask)` erases admitted positions directly from full tensors.
  - `AdmissionStrategy.admit(keys, frequencies, founds)` returns a full-length admission mask and ignores found positions.
  - `initialize_non_admitted_embeddings` and initializer selectors accept boolean masks directly.
- Update all built-in and reference implementations and every existing call site to the new signatures.

## Implementation

- Implement fused cache `find_or_insert` with lookup, score update, insertion/eviction, overflow, reference-counter increment, and input-aligned eviction metadata generation in one native operation.
  - The hot-path CUDA kernel is cooperatively launched: phase one finds and protects every hit, a grid barrier preserves lookup-before-eviction ordering, and phase two provisions only misses.
  - Query the exact per-policy/output/overflow cooperative resident-grid limit during Python table initialization and reuse it on every launch; the steady-state path performs no device-property or occupancy queries.
  - Keep the table device current on the calling thread so the hot path does not need a per-call CUDA device query or guard.
  - Preserve LFU, timestamp, customized/step, mixed-table, and overflow semantics.
  - Use input-aligned sparse eviction tensors without a separate eviction counter, so the hot path does not slice or compact them; derive the count from `evicted_mask` only when cache metrics are enabled.
  - Increment the reference counter before publishing the cache key.
- Deterministic mode uses an all-input hit-acquire pass followed by stable one-key-per-table `find_or_insert` waves. This remains the specialized implementation of the current fused API and avoids both main-bucket and shared-overflow ordering races; it does not restore the legacy Python prefetch pipeline.
- Implement `DynamicEmbStorage.exchange` with:
  - One masked storage lookup kernel.
  - Storage lookup hits acquire their physical rows before insertion; the value-exchange kernel releases each row only after its inbound read completes, preventing source/destination aliasing across blocks.
  - One masked storage insert/row-assignment kernel for input-aligned evicted keys. Each successful destination is pinned before release-publication, and the value-exchange kernel releases that temporary pin after the outbound write is system-visible; the ordinary table-insert API retains its existing deferred-publication behavior.
  - One vectorized value kernel performing cache→storage and storage→cache transfers together.
    - Eight warps cooperatively preload eight input-aligned metadata tensors into a shared-memory SoA with coalesced global transactions.
    - Each warp owns eight input rows and uses a four-slot ring of fixed 512-byte stages per direction, so arbitrary physical row widths are handled without row-sized shared-memory allocation.
    - For each tile, prime `depth - 1` groups, with each group loading both the cache and storage sides of one row. The unrolled steady-state loop waits for the head, issues the next row into the free tail slot, then writes the completed head to cache and storage. Tail storage reads overlap cache→storage writes. Do not use TMA or `cp.async.bulk`.
    - Compute the exchange launch target from device properties during Python state construction and reuse it on the hot path.
  - Per-input pairing so an evicted cache row is loaded before its slot is overwritten.
  - Direct physical-row copies, including optimizer state, without padded `storage_values` or `evicted_values`.
  - Correct LFU score propagation and stable `NO_EVICTION` logical rows.
  - Cache or storage slot-provisioning failure is fatal in the exchange kernel; there is no silent value-staging or legacy fallback.
- Rewrite `_prefetch_cache_path` without `flagged_compact`, dense miss tensors, or separate flat load/store calls:
  - `founds` is updated by exchange to mean “previously present in cache or storage.”
  - `new_mask = ~founds`; admission operates directly on this sparse mask.
  - Admitted new rows are initialized directly in their cache buffers, including optimizer state.
  - Rejected provisional entries are released by known slot rather than hash lookup: clear score/digest/counter, mark the key as a reclaimable tombstone, decrement occupancy, and set the batch’s slot index to `-1`.
  - The displaced old key has already been persisted by exchange; the rejected new key is never inserted into storage.
  - The rejection mask is retained so forward initializes those output rows. Using `-1` is required because the cache slot is already reusable and must not be read or updated by forward/backward.

## Validation and Benchmarking

- Do not add new tests. Run the existing dynamic-embedding, admission, table-operation, deterministic, external-storage, cache/flush, mixed-dimension, and optimizer test suites.
- After code review approval, use the existing benchmark without adding new benchmark modes or metrics (do not run it during implementation review):
  - Run `benchmark_batched_dynamicemb_tables.sh -k TestCaching`.
  - Use the first reporting iteration as the cold-cache measurement and `dyn_forward_ms` as the warmed/churn measurement.
  - Compare medians across five identical runs.
  - Require at least 15% lower forward latency for Adam with `cfr=0.8`, including the cold-cache result.
  - Allow at most 5% regression for SGD and `cfr=1.0`.
  - Use the existing profiler modes to confirm the steady-state non-deterministic built-in path contains fused cache insertion, at most two storage metadata kernels, and one value-exchange kernel, with no prefetch-side `flagged_compact` or temporary embedding buffers. Deterministic mode intentionally uses stable serialized metadata waves.

## Implementation Status

- Implemented the fused cache lookup/insertion, input-aligned sparse storage exchange, sparse admission, direct flat initialization, and rejected-slot reclamation described above.
- The current input-aligned, block-pipelined exchange revision builds successfully for SM75, SM80, SM90, and SM100 in the requested `devel_latest` container. SM75 uses the intrinsic's synchronous fallback; hardware `cp.async` overlap starts at SM80.
- Existing validation completed successfully before the current exchange redesign for the full batched-table test file, direct and external cache backends, deterministic mode, odd and mixed embedding dimensions, SGD/Adam optimizer state exchange, LFU score propagation, `NO_EVICTION` backing, and admission rejection/reclamation plus cache churn. The current revision has not yet been tested.
- Benchmarking remains intentionally pending code review approval.

## Assumptions

- The performance target applies to built-in mapped-host `DynamicEmbStorage`; external backends must implement the same semantics but may use backend-specific transfers.
- Cache `find_or_insert`, storage metadata operations, value exchange, admission, and initialization execute on one ordered CUDA stream. Publishing a provisional hash entry is not a readiness signal for unsynchronized streams.
- Reclaimable tombstones are used instead of literal empty keys so hash-probe chains remain valid while the slot is immediately reusable.
- Existing non-cache paths remain operational after their admission call sites are migrated to the new current APIs.
