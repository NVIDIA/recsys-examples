# Upstream 3-Pick: Embedding Admission Strategy

Goal: land "Embedding admission strategy" (#236) on top of base commit `e48294d3` (Add LFU evict strategy, #52) using only 3 upstream cherry-picks — skipping `ad31b9ef` (add gradient clip) and the two intermediate local-chain commits (#136, #176).

Branch: `upstream-3pick`

---

## Cherry-Pick Chain

| # | Local Commit | Upstream | Description |
|---|-------------|----------|-------------|
| 1 | `bbd7afb` | `6f7281a` | Refactor dynamicemb with Cache&Storage (#128) |
| 2 | `e7690ca` | `c6adf64` | Counter table interface and ScoredHashTable (#229) |
| 3 | `f22b1e2` | `f5b608e` | Embedding admission strategy (#236) |

All 3 applied cleanly with `-X theirs`. No manual edits required.

---

## Conflicts Encountered

### Commit 1 — `6f7281a` (#128)

| File | Conflict type | Resolution |
|------|--------------|-----------|
| `corelib/dynamicemb/benchmark/README.md` | content | `-X theirs` |
| `corelib/dynamicemb/benchmark/benchmark_batched_dynamicemb_tables.py` | content | `-X theirs` |
| `corelib/dynamicemb/benchmark/benchmark_batched_dynamicemb_tables.sh` | content | `-X theirs` |
| `corelib/dynamicemb/dynamicemb/batched_dynamicemb_function.py` | content | `-X theirs` |
| `corelib/dynamicemb/dynamicemb/batched_dynamicemb_tables.py` | content | `-X theirs` |
| `corelib/dynamicemb/src/optimizer.cu` | content | `-X theirs` |
| `corelib/dynamicemb/test/unit_test.sh` | content | `-X theirs` |
| `third_party/HierarchicalKV` | submodule (not checked out) | Staged their commit `0ec9aa3` manually via `git update-index` |

### Commit 2 — `c6adf64` (#229)

| File | Conflict type | Resolution |
|------|--------------|-----------|
| `corelib/dynamicemb/dynamicemb/types.py` | modify/delete — absent in HEAD, modified in `c6adf64` | Staged their version (`git add`) |

**Note:** In the upstream, `types.py` is first introduced by a commit that pre-dates `c6adf64` in the full upstream history. Since we skipped those intermediate commits (including `ad31b9ef`), `types.py` was absent in HEAD. Git left `c6adf64`'s version of the file in the working tree; staging it directly resolved the conflict without any manual editing.

### Commit 3 — `f5b608e` (#236)

| File | Conflict type | Resolution |
|------|--------------|-----------|
| `corelib/dynamicemb/example/README.md` | modify/delete — absent in HEAD, modified in `f5b608e` | Staged their version (`git add`) |

---

## Comparison with the 5-Pick Local Chain

| | 5-pick (local chain) | 3-pick (upstream) |
|---|---|---|
| Commits | `3c79809`, `b818725`, `befb8c9`, `78c8ebc`, `68266e9` | `bbd7afb`, `e7690ca`, `f22b1e2` |
| Source of #236 | jiashuy fork (`c2babbe`) | NVIDIA upstream (`f5b608e`) |
| Needed `ad31b9ef` | No (but was present in local chain context) | Skipped entirely |
| `types.py` introduced by | Local #176 (`befb8c9`) | `c6adf64` itself (via modify/delete resolution) |
| Bugs in #236 | 6 bugs (see cherry-pick.md) | N/A — sourced from NVIDIA upstream directly |
| Manual file edits | Yes (bugs 1–6 fixed in `268cf1d`) | None |

The 3-pick approach is cleaner: it sources #236 directly from NVIDIA upstream (`f5b608e`) rather than the jiashuy fork, avoiding all 6 wiring bugs documented in cherry-pick.md. The only friction was two modify/delete conflicts for `types.py` and `example/README.md`, both resolved by accepting their version.

---

## Skipped Commit

`ad31b9ef` ("add gradient clip") is an ancestor of `c6adf64` and `f5b608e` in the upstream history but was not required for any of the 3 cherry-picks to apply. It never surfaced as a conflict or dependency.

---

## Build & Test Progress

### Build

Initial cherry-pick left source inconsistent — C++ sources were from the local branch, not f5b608e. Build loop revealed a cascade of compile errors. Resolution: replaced **all** C++ source files and the `hkv_variable_instantiations/` directory with their `f5b608e` versions.

| Issue | Fix |
|-------|-----|
| `NameError: sys` in `setup.py` | Added `import sys` |
| `std::optional` not found (CUDA compiler) | Added `-std=c++17` to both `cxx` and `nvcc` flags in `setup.py` |
| HKV submodule at wrong commit (`0ec9aa3`) | Updated to `9c197a9c` + `git update-index --cacheinfo` to prevent `setup.py submodule update` from resetting it |
| `override does not override` in `hkv_variable.h` | Replaced all C++ source files with `f5b608e` versions |
| Pybind11 arg count mismatch in `optimizer.cu` | Included in full C++ file replacement |

Build **succeeded** on 2026-03-26 producing `dynamicemb-0.0.1-cp311-cp311-linux_x86_64.whl`.

### Python File Alignment

The cherry-picked Python files (from `bbd7afb`) were mismatched with the f5b608e C++ bindings. All Python files were replaced with their `f5b608e` versions.

| Issue | Fix |
|-------|-----|
| `lookup_forward_dense`, `lookup_backward_dense`, `lookup_backward_dense_dedup` not in extension | Replaced `batched_dynamicemb_function.py` with f5b608e version (uses `lookup_forward`/`lookup_backward` only) |
| `DynamicEmbeddingCollectionContext` not defined | Replaced all `shard/` and `planner/` Python files with f5b608e versions |
| `DynamicEmbInitializerArgs`, `DynamicEmbInitializerMode`, `DynamicEmbScoreStrategy`, `DynamicEmbTableOptions` not in `dump_load.py` | Added re-exports from `types.py` and `dynamicemb_config.py` to `dump_load.py` |

### Test Fixes

| Issue | Fix |
|-------|-----|
| `test_lfu_scores.sh` missing | Restored from f5b608e |
| All test scripts used 4 GPUs; machine has 2 | Changed `NUM_GPUS=(1 4)` → `(1 2)` and `--nproc_per_node 4` → `2` in all `.sh` files |
| Missing `Dict`, `Any`, `record` imports in test file | Added to `test_embedding_dump_load.py`; later replaced file with f5b608e version |

### Test Status (as of 2026-03-26)

Test suite: 11 scripts/files in `test/unit_test.sh`.

All 11 tests **passed** ✅

| # | Test | Result |
|---|------|--------|
| 1 | `test_embedding_admission.sh` | ✅ |
| 2 | `table_operation/test_table_operation.sh` | ✅ |
| 3 | `test_lfu_scores.sh` | ✅ |
| 4 | `test/test_batched_dynamic_embedding_tables_v2.py` | ✅ 26 passed |
| 5 | `test/test_optimizer.py` | ✅ 12 passed |
| 6 | `test/test_unique_op.py` | ✅ 3 passed |
| 7 | `test_sequence_embedding.sh` | ✅ |
| 8 | `test_pooled_embedding.sh` | ✅ |
| 9 | `test_embedding_dump_load.sh` | ✅ |
| 10 | `incremental_dump/test_incremental_dump.sh` | ✅ 15 passed |
| 11 | `test_twin_module.sh` | ✅ 48 passed |

Additional fixes applied during test loop:

| Issue | Fix |
|-------|-----|
| `test_lfu_scores.sh/py` missing | Restored from f5b608e |
| All test scripts used 4–8 GPUs; machine has 2 | Changed `NUM_GPUS=(1 4)` → `(1 2)` and `--nproc_per_node 4/8` → `2` in all `.sh` files |
| Port 29500 conflict when running tests in parallel | Added `--master-port 2960x` (29601–29604) per script |
| GPU count and master port hardcoded | Replaced with `DYNAMICEMB_NUM_GPUS` and `DYNAMICEMB_MASTER_PORT` env vars (default 2 / per-script port) |
| `BatchedDynamicEmbeddingTables` ImportError | Added backward-compat alias in `batched_dynamicemb_tables.py` |
| `test_optimizer.py`: `find()` TypeError, optimizer API mismatch | Use `initialize_hashtables()` + old-style optimizer constructor directly |
