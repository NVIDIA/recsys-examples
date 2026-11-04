# CPU Host updates: atomic access and the stock-NVE experiment

**Decision:** commit 6 pins official NVE 26.10 (`803e278db2419261369a3d68c69a27858f7ca6d0`)
and removes the companion Host atomic-access patch. Commit 5 already supplies
tier selection locally. Host update/lookup scheduling and the existing demos
are unchanged; continuous Identity/FusedIdentity updates remain experimental.

## Why the patch existed

Stock HostLayer uses ordinary copies on the same CPU storage:

```cpp
// Two CPU threads may overlap on the same row.
lookup: memcpy(output, table[row], bytes);
update: memcpy(table[row], new_values, bytes);
```

That is a C++ data race with undefined behavior. Accepting stale or mixed
embeddings does not define the behavior of those memory accesses. The patch
changed both sides to relaxed, lock-free atomic word loads/stores; CUDA execution
was not the reason for it. See the [C++ data-race rules](https://eel.is/c++draft/intro.races)
and [GCC relaxed atomics](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html).

Example: one aligned FP32 row, accessed as two 64-bit words:

```text
Before: [1, 2 | 3, 4]        Target: [10, 20 | 30, 40]
         word0  word1

Updater CPU thread             Lookup CPU thread
store_relaxed(word0, [10,20])
                               load_relaxed(word0) -> [10,20]
                               load_relaxed(word1) -> [3,4]
store_relaxed(word1, [30,40])

Lookup result: [10,20,3,4] -- allowed by the weak-consistency contract.
```

## Follow-up: is it per-row atomicity, and can we drop it?

It is **per-word atomicity**, using 1/2/4/8-byte words according to alignment and
row size. A row can contain mixed generations. Relaxed accesses do not introduce
a reader/writer lock, waiting, a whole-row transaction, or CUDA synchronization.

Accepting arbitrary lookup values during updates still does not make concurrent
non-atomic copies valid C++. Removing the patch also removes the basis for the
"parallel lookup/update must not break" guarantee. Safe removal requires avoiding
access overlap or replacing it with another valid concurrency mechanism. With
atomic accesses, observing update completion before a later lookup supplies the
ordering for that lookup to see the new values.

The requested experiment deliberately removes this protection without pausing
fixed-mapping readers. Repeated tests can show observed progress, finite results,
and final-value checks on this build; they cannot establish data-race freedom.
Paused LinearHashMap still drains readers before row/mapping changes.

## Validation

Tested on AMD EPYC 7313P / L40S, stock NVE 26.10, Torch 2.13.0a0+9186a08b2c.nv26.07
and CUDA 13.3. The inference update libraries and Host C++ example were rebuilt
against stock NVE. Host gather/update source hashes match the official pin.

| Check | Result |
| --- | --- |
| Existing Host/indexer regression suite | 116 passed, including the 8 concurrent cases. |
| Concurrent CPU tests: 5 GPU-visible + 5 GPU-hidden rounds | 80/80 passed. |
| Python/C++ Host AOT demos: 3 rounds x 3 indexers x 2 dtype/pooling modes | 18 Python + 18 C++ runs passed, all with GPU visibility disabled. |
| Original Python/C++ incremental demo | All 4 combinations passed, including paused and concurrent Redis rounds. |

Host modes: single-table Identity, FusedIdentity, and paused LinearHashMap;
FP32/plain and FP16/mean; sidecar state and embedded fixed mappings. Concurrent
unit tests also cover all FP16/FP32 x plain/mean fixed-mapping combinations.
The existing assertions were unchanged. No Host crashes, timeouts, non-finite
results, or final-value mismatches were observed. Fixed-mapping readers progressed
during updates; hash readers stayed paused. Across 32-update AOT runs, the examples
reported 85-111 Python and 15-27 C++ lookups for fixed mappings (not a throughput
benchmark). The distribution of stale/mixed values was not measured.

**Separate unresolved finding:** the non-Host tier suite had 16 passes and one
failure in `test_hierarchical_updates_only_selected_cache[0]`: the final cold
lookup through an independent binding did not match the backing-store reference.
That unchanged case passed 2/5 isolated reruns and failed 3/5. The assertions
checking selected-tier updates before that lookup passed. No timing workaround
was added and the root cause is not established.

**Conclusion:** the requested Host workloads passed repeatedly on this build.
Removing the patch still leaves concurrent non-atomic copies outside C++'s
data-race-free guarantees; these runs do not establish safe concurrent execution.
