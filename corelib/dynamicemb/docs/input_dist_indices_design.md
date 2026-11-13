# RFC: Resolve DynamicEmb keys at receiver completion

Status: **proposed, Draft implementation; not accepted as completion of #459**. This document replaces the earlier prototype description. The implementation is [PR #551](https://github.com/NVIDIA/recsys-examples/pull/551); design discussion is [RFC #552](https://github.com/NVIDIA/recsys-examples/issues/552). Neither this RFC nor the PR closes the feature request.

## 1. Requirement and reviewed baseline

[Issue #459](https://github.com/NVIDIA/recsys-examples/issues/459) asks to move key-to-index resolution from lookup into input distribution so DynamicEmb can reuse TorchRec lookup. [shijieliu's clarification](https://github.com/NVIDIA/recsys-examples/issues/459#issuecomment-6189054416) recommends starting with HBM-backed row-wise sequence embeddings, but explicitly expects the cache/storage cases too. It calls out eviction, replacement-row initialization and the cache/PS interface. Optimizer state must follow the identity of the replacement key, rather than the previous occupant of a slot.

The original PR head was `7ef51667b75d9e498776ee9c4a29426828282d7c`, based on `476cdc01d1625337070d71659c9bb304c1593d8f`. Review included that entire diff and the actual code of [#528](https://github.com/NVIDIA/recsys-examples/pull/528), merged as `d53b9c8ca504f239d64078062d020d8e27f52bcf`. During review, main advanced to `8accb875333f82a9446877ec929f53ed42a283b7`: #550 separates hierarchical CI groups, and #553 refuses LSA teams that lack NVLink connectivity. Both updates are incorporated. The #528 C++ implementation is reused; this PR does not change its transport, plans, capacity protocol or topology checks.

This is a data-flow change with a performance objective, not an established throughput improvement. #459 sets no numeric speedup requirement. The current measurements show remaining overhead relative to main, and supported NVLink/multi-node hierarchical execution still needs validation. The executable acceptance matrix and evidence are in [input_dist_acceptance.md](input_dist_acceptance.md); performance samples and profiler summaries are in [input_dist_review_results.json](../benchmark/input_dist_review_results.json). Historical results against `476cdc01` do not validate current main.

## 2. Before and after: where work actually runs

### Before: current main

For native row-wise sequence distribution, the caller's KJT holds keys. `ShardedDynamicEmbeddingCollection.input_dist` optionally index-deduplicates and retains the inverse/output-index context, then `RwSparseFeaturesDist` bucketizes and issues native `KJTAllToAll`. The first wait resolves splits; the second returns the receiver's key KJT. `GroupedEmbeddingsLookup.forward` splits kernel groups and calls `BatchedDynamicEmbedding.forward`. Its table forward consumes a legacy prefetch state or performs `dynamicemb_prefetch`: segmented uniqueness, frequency/score reduction, admission, backend lookup, eviction/replacement initialization and pins. `DynamicEmbeddingFunction.forward` loads/gathers or pools rows, and its backward reduces gradients, runs the DynamicEmb optimizer and releases pins. In DEFAULT storage mode some backend work occurs in compute. The hash table and embedding access are consequently coupled to lookup.

With `output_dist="hierarchical"`, #528 performs index dedup even when the sharder did not, then `HierStartAwaitable` returns `HierKJTAwaitable`. Its second wait invokes `HierDist.finish`, checks the GPU-produced status on the host, constructs the receiver KJT, registers that object's output-window ownership in a weak-key map and attaches `HierCall` plans to the sharding context. `HierGroupedEmbeddingsLookup` calls `HierDist.lookup`; each table's `output_alloc` supplies a window slice and optional key-to-window row map. The existing native gather writes into that window. Output distribution moves rows using the plans; backward applies the same row map before embedding-gradient reduction. Eval calls use native TorchRec distribution. Unsupported configurations follow the existing explicit refusal/fallback rules.

### After: receiver preparation, then row lookup

```text
raw-key KJT
  -> existing index dedup / frequency and original-index context
  -> native bucketize + KJTAllToAll, or hierarchical input transport
  -> first wait: transport/splits only
  -> second wait: finish transport and validate its status
  -> grouped resolve_input(raw local keys), including feature processing once
       -> existing segmented unique / admission / find / insert / row initialization
       -> HBM/cache: physical value rows + independently owned pin lease
       -> Host/Hybrid/PS: compact values including optimizer state + live-row registry
  -> ResolvedEmbeddingInput, split into the same kernel groups
  -> BatchedDynamicEmbedding.forward(prefetch_state=...)
  -> TorchRec EmbeddingCollection / EmbeddingBagCollection
       -> single HBM/cache sequence table without rejected rows: persistent matrix + physical occurrence rows
       -> otherwise: compact embedding workspace + unique reverse indices
  -> native output distribution, or mapped copy into hierarchical output window
  -> existing DynamicEmb gradient reduction and fused optimizer
  -> backing-store write-through / publication to live snapshots if needed
  -> release on final backward, no-grad completion or abandoned ownership
```

The receiver runs on the CUDA stream executing the second wait. Native tensors-awaitable identity and its split/recat metadata are preserved: `_PostprocessedSplitsAwaitable` attaches one `_ResolveOnce` callback to the original tensors awaitable. Successful repeated waits return the same resolved KJT and do not take more pins. Score tensors complete before resolution. Hierarchical completion first calls the existing `finish`; only a successful status permits resolution, and `rebind_input(original, resolved)` transfers output-window ownership to the replacement KJT. Its successfully resolved result is cached too. A callback exception is not cached as a successful raw-key result.

Feature processing operates on keys before resolution, exactly once for dynamic groups. Static TorchRec groups retain their ordinary KJTs and compute. The grouped input retains explicit child KJTs and their states, so splitting it later cannot detach the rows from their owners. Sequence output indices still come from the original-key sharding context. Optimizer/model trackers explicitly receive `original_features()`; observers must not mistake value rows for keys.

## 3. Boundary fields and invariants

These are internal fields; callers continue to use the existing public collection/table interfaces. The new optional table `prefetch_state` argument and distributor postprocessor preserve old call forms. No Storage method signature is made mandatory for existing implementations.

| Field / index space | Meaning | Required invariant |
| --- | --- | --- |
| `ResolvedEmbeddingInput.values()` | Per-occurrence value row, not a key | Same feature/occurrence order as the receiver KJT; existing weights, lengths and output context keep their meaning |
| `unique_keys`, `unique_table_ids` | Unique identities | Identity is the full `(table, key)` pair; no lossy packing of 64-bit keys |
| `reverse_indices` | Occurrence -> compact unique position | Repeated occurrences contribute to one unique gradient; this is not a persistent hash slot |
| `slot_indices` | Hash slot used by ref counters | Refresh after key-map generation changes; never index an HBM storage's value matrix with it |
| `value_row_indices` | Unique identity -> physical value row, or snapshot row | HBM storage rows survive rehash; cache rows equal cache slots and the cache does not rehash |
| `occurrence_rows` | Cached physical rows of the resolved KJT | Alias the already-created per-occurrence row tensor; only the eligible persistent-matrix adapter consumes it directly |
| `unique_values` | DEFAULT snapshot of embedding plus optimizer state | Padding and per-table optimizer layout follow Storage/optimizer contracts; lookup only sees the embedding view |
| `persisted_unique_indices` | Ascending unique positions actually found/admitted | Rejected scratch rows never enter persistent optimizer writes; a full selection is identity and can avoid gathers |
| `unique_scores`, `input_frequencies` | Score reduction and original frequency input | Preserve per-key maximum scores and existing admission/LFU occurrence semantics; mode/reuse re-resolution retains the inputs |
| `key_map_generation` | Generation of the pin's hash-slot lookup | A later rehash requires read-only slot refresh before decrement; value-row identity does not change |
| `pin_lease`, `used`, `released`, `is_eval` | Ownership and lifecycle | One reservation is released once; separate forwards do not share a consumed reservation |
| `HierDist._rows[kjt]` | Output-window slice and row map for one completed call | Replacement KJT inherits the same entry; plans and output context are not regenerated from row indices |
| `output_alloc(rows, dim, dtype)` | `None`, or `(window, row_map)` | Inactive on native/fallback calls; row map is applied to forward placement and to gradient reduction |

A hash-free first compute is an observable condition, not a documentation claim: `test_input_dist_resolves_before_lookup` spies on backend access and checks initialized state before compute. Rehash refresh during finalization/backward, legacy raw-table preparation and explicit reuse/mode transitions are separate paths that may resolve keys again.

## 4. State lifetime and ordering

### HBM/cache pins

`dynamicemb_prefetch` uses the existing admission, row allocation, replacement initialization and optimizer-reset routines. New/moved identities initialize both their embedding and optimizer values before use. The state retains hash slots separately from value rows. Eviction and async erase spare pinned identities; a later rehash may move hash slots, so release looks them up read-only when the generation changed. No alternate eviction algorithm or optimizer is introduced.

A `_PinnedRowLease` owns the table, pin metadata, CPU cache-reservation counter and last-use CUDA event independently of `PrefetchState`; it has no reference back to that state. The input KJT/FIFO owns preparation; the autograd context owns a training use. `finish_forward` returns the CPU prefetch reservation once, while the device pin remains until the last required use. A final backward releases it once; `retain_graph=True` retains it, and every backward records its last GPU use. A no-grad call releases after its forward. If an input or graph is discarded, lease finalization waits on that last-use event on the cleanup stream and records all metadata used by its queued kernels before decrementing pins.

```text
prepared -> forward-used -> retained backward(s) -> final backward -> released
    |            |                  |
    +------------+------------------+-> owner discarded -> event-ordered release
prepared -> no-grad/eval consumption -> released
```

GC can execute while the current thread holds `TableAccessGate`, including inside its non-reentrant `FairLock`. A finalizer must not recursively acquire it. A thread-local cleanup scope queues those releases and drains them only after the outermost gate/host-lock section unlocks, on the same thread. `host_section` serializes CPU reservation-counter changes without inserting a device fence. The original fair-lock and async-erase ordering are retained. The finalizer skips Python interpreter shutdown, where CUDA teardown cannot be guaranteed; process exit reclaims those resources.

An eval-prepared input reused for training is prepared again. A training-prepared input consumed after a mode flip to eval uses its prepared rows without backward and releases its lease. Recomputing an already-used resolved input prepares a fresh independent state from saved keys/frequencies/scores. These compatibility cases deliberately pay resolution again instead of sharing a released pin. They preserve the pipeline's existing ability to change model mode between input distribution and compute.

### Opaque storage and optimizer consistency

Host, Hybrid and external Storage/PS implementations expose `find`/`insert`, not stable CUDA row addresses. Preparation reads embedding and optimizer values together into a compact snapshot. `ActiveStorageRows` weakly indexes outstanding states; the distributed input, raw prefetch FIFO and autograd contexts own them. If no states are active, find immediately returns the backend result without matching/filtering kernels.

Each DEFAULT optimizer update is written through with `preserve_existing=True`, then published to other live snapshots by full `(table, key)` matching. The next prefetched forward/backward therefore sees the latest embedding and moments even when backward order differs from input order. If a finite store evicts a still-live key, resolution overlays its live value and restores it instead of reinitializing it. Only persisted rows participate. A snapshot must include optimizer state; publishing embeddings alone would still allow stale Adam moments to overwrite a later update.

```text
input stream: prepare A ---- prepare B (same key) ---- prepare C
compute:                    forward A -> backward A -> forward B -> backward B
gate/event chain: backend read B ------ write A + publish B ------ read/use B
```

`TableAccessGate` orders DEFAULT backend operations and snapshot publication between streams and with async workers. Publish/find hold strong local references while enqueuing copies and record both snapshots' CUDA tensors on the consuming stream. Cancellation may remove the Python owner immediately; queued copies still own their allocations through stream recording. Reverse-order backward is serializable in the order backward calls execute; it does not promise equivalence to a different optimizer-step order.

GPU stream concurrency supported by the existing pipelines is distinct from arbitrary concurrent Python calls into one grouped lookup. The native TorchRec module temporarily rebinds frozen weight parameters, and `HierDist.lookup` has a current output-window cursor. Those objects are not reentrant across independent host threads. This PR does not add a claim that simultaneous host-thread forwards on one module are supported. The gate is an ordering mechanism for backend/erase operations, not a global model lock.

## 5. TorchRec adapter selection and rationale

The adapter calls TorchRec's public unsharded `EmbeddingCollection` or `EmbeddingBagCollection`. DynamicEmb continues to own admission, storage, gradient reduction, clipping and fused optimizer updates. Replacing the DynamicEmb optimizer with a TorchRec fused optimizer would change state layouts, step ordering and checkpoint semantics, and is outside this change.

For one HBM/cache sequence table, no rejected rows and native TorchRec adapter selection, the table's VMM-backed matrix already has the required addressable row layout. Compute obtains a fresh view of its current shape, slices away optimizer columns, and uses the resolved physical occurrence rows. Pins keep identities alive; growth preserves the VMM address. The view is not cached across growth. This avoids copying unique embedding rows into a compact workspace. Raw/FIFO callers that do not already carry occurrence rows compute that mapping once. SGD/Adam and FP32/FP16/BF16 regressions check the direct path without allowing `load_from_flat` to run.

Multiple persistent tables have separate matrices, and rejected keys have no persistent row. The general adapter therefore loads their resolved unique embedding rows into a compact matrix, or uses the DEFAULT snapshot's embedding view. It addresses that matrix with `reverse_indices`. Thus the implementation consumes resolved rows, but it is not a universal zero-copy binding of every backend to a fused TorchRec TBE. This distinction is necessary when judging #459: mapping moved to input distribution; workspace adaptation remains in compute.

Uniform-D groups use one synthetic feature over the matrix and reshape pooled output back to the original feature/batch layout. Mixed-D pooling uses per-table strided views. FP16/BF16 pooling promotes the temporary matrix to FP32, preserving DynamicEmb accumulation and FP32 sample-weight semantics, and casts the requested output at the end. Learned pooling weights remain explicitly rejected during training; no-grad behavior is preserved. Modules are initialized on meta, materialized and zeroed without advancing caller RNG. Frozen weights are rebound for the call and restored in `finally`; placeholders live outside the model tree, checkpoints and external optimizers.

The retained-graph release decision currently uses PyTorch's `_get_current_graph_task_keep_graph` binding, exercised on PyTorch 2.7 and 2.13. It is a private binding and needs compatibility testing when the supported PyTorch matrix changes. Unconditionally releasing after every backward is incorrect; retaining until graph GC alone delays release even after a completed non-retained backward.

## 6. Native/hierarchical compatibility and failure behavior

Native TorchRec split fusion recognizes `_dist` and can reconstruct a tensors awaitable without this postprocessor. DynamicEmb uses `_key_dist` for shardings that resolve dynamic groups; the opt-out is uniform across ranks, including empty table-row-wise ranks. This preserves native communication and awaitable metadata, but sacrifices that split-fusion optimization. Keeping the callback while allowing fusion requires an explicit TorchRec integration; restoring the `_dist` name without that integration loses preparation and scores.

For hierarchical calls, the existing transport remains authoritative for dedup, row plans, frequency counts, overflow and topology. After status validation, `rebind_input` preserves object-based ownership. `output_alloc` is honored in both training and evaluation/mode-transition compute. TorchRec produces key-ordered output, so the adapter copies/index-copies it into the supplied window and passes the map to `reduce_grads`. The writes follow the existing native raw-window contract and do not bump the shared window's autograd view version; otherwise a later lookup group invalidates a previously returned custom-Function view. `_LookupOutput` and DynamicEmb backward explicitly route gradients. This is an extra output copy compared with main's direct native gather into the window, not a zero-copy output optimization.

| Path / capability | Decision / existing restriction | Local evidence |
| --- | --- | --- |
| RW sequence; native distribution | All HBM/Cache/Host/Hybrid/PS-interface layouts; dedup on/off, int32/int64 keys | One/two GPU matrix |
| RW / TW-RW pooled | Existing SUM/MEAN, weighted SUM, mixed-D and static/dynamic grouping | Focused GPU matrix; prior four-rank TW-RW results are historical until rerun on this base |
| Eval | Native distribution, read-through cache/storage/PS, initialize misses without insertion | Full table and focused tests; no remote PS deployment claim |
| Precise per-key scores | Native score channel survives resolution/pipelines | Native score tests; Host/Hybrid and PS-only score restrictions still apply |
| Hierarchical RW sequence | Same #528 supported shapes, dedup/frequency semantics and capacity errors | Supplied-result completion/window tests; d53 transport experiment is not current-main NVLink certification |
| Hierarchical precise scores / clipping without index dedup / continuous placement / static groups | Keep upstream explicit refusal or native fallback | No silent support extension; actual native fallback tested |
| `output_dist="auto"` on unlinked PCIe GPUs | #553 warns and chooses native | Real target two-rank fallback/refusal tests |
| `output_dist="hierarchical"` on unlinked PCIe GPUs | #553 raises before row preparation | Real topology test; no override of NVLink checks |
| Build without hierarchical extension | `auto` falls back; explicit hierarchical mode raises | Compatible old-stack build and pipeline fallback tests |
| Physical multi-node GIN / remote network PS | Requires corresponding deployment validation | Not run locally |

Validation of arguments, score requirements, unsupported weight gradients and mode precedes row mutation where the existing public API promises it. If a later grouped resolver fails, already-prepared earlier groups release their reservations. A rejected cache-capacity reservation leaves the CPU outstanding count unchanged, so releasing earlier batches permits a valid retry. If compute raises, its prepared state is released. Capacity failure is checked before invoking row resolution. Native and hierarchical successful completion are idempotent, but a failed backend operation is not a transaction: insertion, admission counts or external writes already completed are not rolled back. Callers must handle distributed backend failure as a failed step/job; retrying one rank alone or assuming a partially failed batch is score-neutral is unsupported. No new global collective is added to recover an arbitrary backend exception.

## 7. Costs, measurements and alternatives

Let N be received occurrences, U unique `(table, key)` rows, D maximum embedding width, S optimizer columns and B active DEFAULT batches. Pinned HBM/cache states retain O(U) metadata plus O(N) occurrence mapping; the general compact embedding workspace costs O(U*D). DEFAULT retains O(U*(D+S)) per outstanding batch. Low-precision pooling adds an FP32 O(U*D) workspace. Hierarchical output adaptation writes O(N*D) bytes into its window; output placement can add an int64 row-map conversion. Weak ownership prevents abandoned states from being permanently retained, but legitimate pipeline depth still multiplies live storage.

`ActiveStorageRows` currently matches per table with filtering, sorting and searchsorted for each active pair. Preparation/publish can cost O(B*U log U), plus row copies, and dynamic `nonzero`/boolean selections may synchronize with the host. The empty-registry fast path avoids this work when no other states exist. Matching is not a free cache, and its cost at larger overlap must be profiled. A device hash index or persistent row-identity cache could reduce it, but would introduce another concurrent ownership structure; that optimization is not justified solely by a style preference.

The benchmark compares **complete imported main versus complete imported PR**, separately from an adapter-only old-gather control. It includes raw forward, training with outstanding raw prefetch batches and actual SparseDist training with a dense MLP/SGD. Inputs reside on GPU; timings exclude data loading, first-call setup and profiler overhead. Wall time includes a synchronization at each trial boundary and max rank time for distributed trials. CPU/CUDA traces count synchronization, memcpy, kernels and peak allocator increment; profiler exit contributes one terminal device sync. Shared GPUs/host scheduling and limited trial samples constrain conclusions. VMM and NCCL allocations are not all visible to the PyTorch allocator peak.

On the target stack, raw Host training at zero overlap measures 2.745 ms versus main 2.002 ms; with two outstanding batches it measures 8.782 ms versus 2.058 ms. The five-call Host profiles grow from 175 kernels/11 sync calls to 925/121 with overlap. The two-rank SparseDist plus dense MLP case measures 17.871 ms versus main 15.537 ms, about 15% slower. A separate warmed Host allocator reading grows from 4,498,944 bytes resident at depth 0 to 5,580,288 at depth 2; these figures omit VMM and some communication memory. Commands, all samples, source manifests and limits are recorded in the acceptance report and JSON. They do not establish a throughput improvement. In particular, the measured two-rank SparseDist case is slower than main; the split-fusion opt-out and adapter CPU work remain acceptance concerns. A reduction in host sync count or compact workspace is not itself proof of end-to-end speedup.

Alternatives considered:

1. **Keep resolution in lookup.** Preserves existing compute locality and split fusion, but does not meet the requested input_dist boundary or let native lookup consume a prepared input.
2. **Universal persistent TorchRec TBE binding.** Could eliminate workspace/output copies, but requires a multi-table stable weight-layout contract, admission scratch rows, optimizer/checkpoint integration and a representation for opaque stores. The restricted single-matrix public-module path is verifiable without imposing those contracts on every backend.
3. **Use only compact snapshots for all backends.** Simplifies adapter indexing, but makes HBM/cache prefetch snapshots stale unless updated after every optimizer step and adds avoidable embedding copies. Pinned persistent rows avoid that extra consistency layer.
4. **Refresh opaque storage only during compute.** Avoids a live-row registry, but restores backend key resolution to lookup and can load stale optimizer state after early reads or finite-store eviction. The selected snapshot publication makes the new boundary explicit at a measurable cost.
5. **Re-enable native split fusion immediately.** Requires TorchRec to preserve the callback, scored channel and per-request ownership when fusing/splitting. A dedicated adapter/upstream hook with collective-order tests is preferable to depending on the current fuser's private reconstruction details.
6. **Release pins on every backward, or hold them forever until GC.** The first corrupts counters with retained graphs; the second unnecessarily pins completed steps. The selected release state machine is backed by negative controls and cross-stream tests.

## 8. Review findings and acceptance decision

The original #551 had reproducible leaks for abandoned HBM/cache inputs/graphs, repeated-wait double preparation, mode-transition failure and incorrect tracker key identities. Retained backward also decremented counters twice, reproduced on unmodified d53 main as an inherited defect. Integrating #528 additionally required preserving receiver-object ownership and output-window row mapping; a multi-group GPU test demonstrated the autograd shared-view failure of ordinary in-place copies. These defects are repaired with executable regressions, not only conflict resolution. Further review of the new cleanup found and covered gate-local GC and retained-backward cancellation on another stream.

The unresolved items are distinguished from those fixed defects: supported NVLink/multi-node hierarchical transport/overflow/pipelines have not been validated on the current base; a real network PS has not been exercised; the private PyTorch retain-graph binding requires version-matrix maintenance; and current measurements show material performance overhead. The pre-#553 PCIe experiment reached matching outputs, per-key gradients, tables and frequencies, but its standalone timeout-teardown hung on unmodified main too. #553 now correctly refuses that topology, so neither that experiment nor supplied-result tests are advertised as NVLink acceptance.

**Decision: Needs Fixes / validation before merge; #459 is not yet accepted as complete.** The design and repaired native/storage paths are reviewable in Draft. Promote the PR only after supported-hardware hierarchical CI passes, representative performance is improved or explicitly accepted, and the remaining backend/deployment claims are bounded by corresponding evidence. This review does not change PR Draft state, close issues or post discussion comments.
