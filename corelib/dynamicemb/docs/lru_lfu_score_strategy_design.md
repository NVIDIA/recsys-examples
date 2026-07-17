# Design: `LRU_LFU` score strategy with configurable / JIT-compiled eviction

Status: **proposal / analysis** — no code written yet.
Scope: `corelib/dynamicemb`.

## 1. Goal

Add a first-class score strategy `DynamicEmbScoreStrategy.LRU_LFU` that stores
**both** an access frequency and a last-access timestamp per key, and evicts by
a policy that combines the two:

- **Default eviction:** rank by LFU (frequency). Break ties by the timestamp —
  when two keys have equal frequency, evict the one with the **older**
  (smaller) last-access timestamp.
- **Custom eviction:** the user may pass a Python `score_function` (numba-style)
  that computes a `float64` "decay" score per key from
  `(scores, cur_timestamp, gamma)`; eviction then removes the key(s) with the
  **lowest** computed score. `gamma` comes from a new `lfu_decay_gamma` option.

Target usage:

```python
import math
from numba.types import float32, int64, float64

def compute_lfu_decay(scores, cur_timestamp: int64, gamma: float32) -> float64:
    # scores[0] = last-access timestamp, scores[1] = frequency (raw uint64 AoS
    # words, no remapping). Use (cur_timestamp - scores[0]) -- the elapsed time
    # since last access -- NOT (scores[0] - cur_timestamp): the latter underflows
    # in uint64 (last < now) AND has the wrong sign. With age >= 0 and gamma in
    # (0, 1) (so log(gamma) < 0), age*log(gamma) grows more negative with
    # staleness, so stale keys score lower and are evicted first.
    return float64(math.log(max(scores[1], 1)) + (cur_timestamp - scores[0]) * math.log(gamma))

options = DynamicEmbTableOptions(
    dim=64,
    max_capacity=10_700_000,
    score_strategy=DynamicEmbScoreStrategy.LRU_LFU,
    score_function=compute_lfu_decay,   # optional; omit for the default policy
    lfu_decay_gamma=0.9,                # only consumed by score_function
)
```

## 2. What already exists (reuse, don't rebuild)

The heavy lifting is already in the tree from the `need_incremental_dump` work:

- **`ScorePolicyType::LruLfu = 4`** (`src/table_operation/score.cuh`): a compound
  policy occupying **two contiguous AoS score words per key**:
  - word 0 = last-access timestamp (device `%globaltimer`)
  - word 1 = accumulated frequency

  `update()` stamps word 0 with the timer and accumulates word 1 on every access.
- **AoS score region + accessors** (`types.cuh`): `scores(iter, k) = base + iter*num_scores_ + k`;
  `num_scores()`, `reduction_score(iter) = scores(iter, num_scores_-1)` (the last
  word, == frequency for LruLfu).
- **2-score `reduce()` path** (`types.cuh:467`): scans the score region, treats
  each `uint4` as one key's `[ts, freq]` pair, and ranks by **word 1 (frequency)
  only** — `if (freq < dst_score) ...`. **No timestamp tiebreak today.**
- **num_scores plumbing** end-to-end: bucket/table carry `num_scores_`;
  gather/scatter/copy score-block kernels; dump/load and rehash preserve both
  words; `incremental_dump` thresholds on word 0 (timestamp).
- **Python**: `score_policy_num_scores(policy)` decouples physical score words
  from spec count; `get_score_policy()` builds the `LRU_LFU` spec; batched layer
  returns `device_timestamp()` for such tables.

So `LRU_LFU` as a *strategy* is largely "expose the existing `LruLfu` *policy* as
its own `DynamicEmbScoreStrategy`, plus (a) a timestamp tiebreak in eviction and
(b) an optional JIT decay function." It is a **superset** of today's
`LFU + need_incremental_dump=True`, which produces the same policy/layout.

### Current strategy/policy/evict enums

- `DynamicEmbScoreStrategy` (Python `IntEnum`): `TIMESTAMP=0, STEP=1,
  CUSTOMIZED=2, LFU=3` (`dynamicemb_config.py:141`). **No `LRU_LFU` yet.**
- `DynamicEmbEvictStrategy`: `LRU, LFU, EPOCH_LRU, EPOCH_LFU, CUSTOMIZED`.
- `get_score_policy(score_strategy, need_incremental_dump)` maps strategy →
  `ScoreSpec(policy=...)`.

## 3. Design decisions (confirmed)

### 3.1 Score-word ordering — `scores[0] = timestamp, scores[1] = frequency` ✅

The `score_function` contract is pinned to **`scores[0] = last-access timestamp,
scores[1] = frequency`**, matching both the example's formula and the internal
AoS layout (word 0 = ts, word 1 = freq). Consequences:

- The device call passes the raw AoS words to `user_fn` with **zero remapping**.
- The example's formula `log(scores[1]) + (cur_ts - scores[0]) * log(gamma)`
  decays `log(frequency)` by the elapsed time `age = cur_ts - scores[0] >= 0`.
  With `gamma` in (0, 1) (so `log(gamma) < 0`), `age*log(gamma)` grows more
  negative with staleness, so stale keys score lower and are evicted first.
  NOTE: it must be `cur_ts - scores[0]`, not `scores[0] - cur_ts` -- the latter
  underflows in uint64 (last < now) and has the wrong sign.
- The internal layout is **unchanged** — word 0 stays the timestamp, so
  `incremental_dump` (thresholds on word 0) and `reduction_score` (the *last*
  word == frequency) keep working. The example's original comment
  (`scores[0]=frequency`) was inconsistent with its formula and is corrected.

### 3.2 Timestamp tiebreak applies to all `LruLfu` ✅

The "equal frequency → evict older timestamp" tiebreak applies to the **whole
2-score `LruLfu` `reduce()` path**, not just the new strategy. `reduce()` keys
off `num_scores_ == 2` (it does not see the score strategy), which is also
simplest. This makes today's `LFU + need_incremental_dump` eviction deterministic
(currently first-seen wins on a tie) — a strict improvement, accepted as an
intentional behavior change to the existing path.

## 4. Public API changes

`DynamicEmbTableOptions` (`dynamicemb_config.py`):

- `score_strategy` gains `LRU_LFU`. Add `DynamicEmbScoreStrategy.LRU_LFU = 4`
  (extend the `IntEnum`; keep existing values stable).
- New `score_function: Optional[Callable] = None` — a numba-compilable Python
  function `(scores, cur_timestamp: int64, gamma: float32) -> float64`. Only
  valid with `score_strategy == LRU_LFU`. When `None`, use the default
  freq+timestamp policy (no JIT).
- New `lfu_decay_gamma: float = <default>` — passed to `score_function`; ignored
  otherwise.
- `get_grouped_key()` must include `score_function` identity (e.g. its
  `__qualname__`/source hash) and `lfu_decay_gamma` so tables with different
  eviction functions are not merged into one physical table.

`get_score_policy()`:

- `LRU_LFU` → `ScoreSpec(name="lru_lfu", policy=ScorePolicy.LRU_LFU, dtype=uint64,
  is_reduction=True)` (same spec today's `LFU + need_incremental_dump` builds).
- `evict_strategy` for such tables defaults to `LFU` (frequency-ranked reduce),
  with the tiebreak/decay layered on top.
- `LRU_LFU` tables implicitly support `incremental_dump` (word 0 is a timestamp),
  so `need_incremental_dump` becomes redundant/implied for this strategy.

## 5. Eviction: one comparator-templated reduce for ALL LruLfu tables

**Scope decision (Q):** every table using the `LruLfu` policy (`num_scores == 2`)
— both the new `LRU_LFU` strategy **and** the existing
`LFU + need_incremental_dump` — routes insert-and-evict through a
**driver-launched cubin**. The AoT 2-score `reduce()` path retires. This gives
`need_incremental_dump` the deterministic freq→ts tiebreak too (honors R1) at the
cost of changing that shipped feature's launch path (must re-run its 18+7
regression).

### 5.1 One scan skeleton, a `Comparator` template

`reduce<Comparator>()` keeps the tuned **async-prefetch** scan (global→shared
pipelined loads, `__pipeline_memcpy_async`, double-buffered) and passes the
**shared-memory pointer** to the key's prefetched `[ts, freq]` pair to the
comparator. Only the innermost compare varies:

- **`LexFreqTsComparator` (default):** exact lexicographic (frequency asc, then
  timestamp asc) on the shared pair. Pure C++, no `double`, **no precision loss**,
  no numba.
- **`UserFnComparator` (custom):** calls
  `extern "C" __device__ double user_score_fn(const uint64_t* scores, uint64_t cur_ts, double gamma)`
  on the shared pointer and ranks by **min `double`**. `cur_ts` read once at kernel
  entry (`%globaltimer`); `gamma` is a kernel parameter.

Because the comparator receives a **shared-memory** pointer into the prefetched
buffer, the async-prefetch bulk reduce is **PRESERVED for both** (this is the R4
fix): the memory-bound global→shared pipeline is untouched; only the
per-candidate ALU differs (a cheap integer compare for Lex; the user's `double`
math for custom). Each key's two words are contiguous in the shared buffer (one
`uint4`), so `&sm_block[key]` is a valid `const uint64_t*` with `scores[0]=ts,
scores[1]=freq`.

⚠️ Verify in the smoke test that a numba-compiled `user_score_fn` dereferences a
**shared-memory** pointer correctly (numba emits generic loads for `CPointer`;
generic addressing resolves shared on sm_90, but confirm).

### 5.2 Two cubins, one launch path

- **Default cubin** — `reduce<LexFreqTsComparator>`, **prebuilt at build** (`nvcc`),
  shipped as package_data. **No numba at runtime.** Used by any LruLfu table
  without a `score_function` (incl. existing `LFU + need_incremental_dump`).
- **Custom cubin** — `reduce<UserFnComparator>`: the LTO-IR fatbin (built with the
  `user_score_fn` undefined) + numba's user LTO-IR, linked via `nvJitLink` at first
  use, cached per (arch, fn identity). Used by `LRU_LFU` tables with a
  `score_function`.
- Both are loaded via `cuModuleLoadData` and launched via driver `cuLaunchKernel`
  on PyTorch's stream (ext_jit `binding.cpp` pattern) — **one launch code path**.
- **Routing:** `num_scores == 2` → driver-launch a cubin (custom if
  `score_function` else the prebuilt Lex cubin); else → the existing AoT extension
  launch. Single-score strategies are untouched.

This resolves R3 (one scan source via the template; one launch path) and R4 (async
prefetch preserved). Only the `Comparator` (a template parameter, in one header)
and the cubin source (prebuilt vs numba-linked) vary.

### 5.3 JIT pipeline (mirrors `jiashuy/CUDA-JIT/ext_jit`)

Toolchain **confirmed present** in the devel container (EOS probe): Python 3.12,
torch 2.12 / CUDA 13.2, **numba 0.64.0** (`numba.cuda.compile` supports
`output=...`), **`cuda.bindings.nvjitlink`** importable, `libnvJitLink.so.13`,
`libnvidia-nvvm.so.4`, `libnvrtc.so.13`, and `nvcc 13.2` (supports `-dlto`).

- **Build time (`setup.py`):** compile the custom-evict TU (§5.2) with
  `nvcc -dlto` (or `-gencode ... -dlto`) to an **LTO-IR fatbin**, embedded in the
  extension. It declares `extern "C" __device__ double user_score_fn(...)`
  (undefined).
- **Runtime `customize(score_function)`:** compile the Python function to LTO-IR:
  ```python
  from numba import cuda, types
  ltoir, _ = cuda.compile(
      score_function,
      sig=(types.CPointer(types.uint64), types.uint64, types.float64),  # -> float64
      device=True, output='ltoir', abi='c',
      abi_info={'abi_name': 'user_score_fn'},
  )
  ```
  (`CPointer(uint64)` maps to `const uint64_t*` so `scores[i]` indexes the raw
  AoS words; pin the exact kwargs against numba 0.64 during impl.)
- **Link (Python `cuda.bindings.nvjitlink` or C++ `libnvJitLink`):**
  `nvJitLinkCreate(arch=sm_90, "-lto")` → add kernel fatbin (`INPUT_FATBIN`) +
  user `ltoir` (`INPUT_LTOIR`) → `Complete` → `GetLinkedCubin` →
  `cuModuleLoadData` → `cuModuleGetFunction`. Cache the module keyed by (arch,
  score_function identity, gamma dtype). Launch on the current stream.
- **New deps:** `numba-cuda`, `cuda-python` (both present); native
  `libnvJitLink`, `libnvidia-nvvm`. The linked artifact is bound to (cpython,
  torch ABI, CUDA major).
- ✅ **LTO-IR version compat — CONFIRMED** by the smoke test: numba 0.64 LTO-IR
  (3824 B) + `nvcc -arch=lto_90 -dc` object (9448 B) linked via `nvJitLink` into a
  cubin (6544 B) on CUDA 13.2 / H100. (The Python-side smoke hit
  `CUDA_ERROR_INVALID_CONTEXT` at `cuModuleLoadData` — a driver-context artifact
  of doing load/launch from Python; the C++ path below avoids it entirely by
  using PyTorch's ambient context.)

### 5.4 Reference implementation to model on — `jiashuy/CUDA-JIT/ext_jit`

Do link + load + launch in **C++** (PyTorch's context), numba-compile in Python.
Verbatim-shaped recipe from the reference:

- **`setup.py`** — build the LTO fatbin, ship it as package_data, link the ext:
  ```python
  nvcc --fatbin -gencode arch=compute_XX,code=lto_XX  # per arch (80;90;100)
       -O3 -lineinfo --use_fast_math -std=c++17  kernel.cu -o kernel.fatbin
  CUDAExtension(..., extra_link_args=["-Wl,--no-as-needed","-lnvJitLink","-lcuda"])
  package_data={"pkg": ["kernel.fatbin"]}
  ```
- **Python glue** (numba → LTO-IR, hand bytes to C++):
  ```python
  cc = torch.cuda.get_device_capability()
  ltoir, _ = nbcuda.compile(user_op, signature, device=True, abi="c",
                            abi_info={"abi_name": abi_name}, cc=cc, output="ltoir")
  _C.link(kernel_fatbin_bytes, bytes(ltoir), cc[0], cc[1])   # C++ links + loads
  ```
- **`binding.cpp`** (`link` + `apply`):
  ```cpp
  const char* opts[] = {archbuf /*"-arch=sm_90"*/, "-lto"};
  nvJitLinkCreate(&h, 2, opts);
  nvJitLinkAddData(h, NVJITLINK_INPUT_FATBIN, fatbin.data(), fatbin.size(), "kernel");
  nvJitLinkAddData(h, NVJITLINK_INPUT_LTOIR,  ltoir.data(),  ltoir.size(),  "user");
  nvJitLinkComplete(h);
  nvJitLinkGetLinkedCubinSize(h, &sz); nvJitLinkGetLinkedCubin(h, cubin.data());
  nvJitLinkDestroy(&h);
  cuModuleLoadData(&g_module, cubin.data());
  cuModuleGetFunction(&g_func, g_module, "apply");   // -> our insert_and_evict entry
  // apply(): pack void* args, cuLaunchKernel(..., getCurrentCUDAStream().stream())
  ```

For dynamicemb the `apply` entry becomes our custom insert-and-evict kernel
(§5.2); everything else (link/cache/launch) follows ext_jit as-is.

## 6. Components to touch

| Layer | File(s) | Change |
|---|---|---|
| Score policy | `score.cuh` | none for layout; (optional) document `LRU_LFU`-strategy reuse |
| Eviction scan | `types.cuh` | make `reduce()` → `reduce<Comparator>()` (one skeleton, comparator sees the shared-mem `[ts,freq]` pair); retire the AoT 2-score reduce |
| Comparators | new header | `LexFreqTsComparator` (exact freq→ts); `UserFnComparator` (calls `user_score_fn`, min double) |
| Evict TU | new `src/.../evict_lrulfu.cu` + `setup.py` | instantiate insert_and_evict for both comparators; `user_score_fn` extern-undefined; build **prebuilt Lex cubin** + **custom LTO-IR fatbin** (`nvcc --fatbin code=lto_XX`), ship as package_data |
| JIT link/launch | new `src/.../jit_link.{h,cpp}` (C++, `-lnvJitLink -lcuda`) | link user LTO-IR into fatbin → cubin; `cuModuleLoadData`/`cuModuleGetFunction`; per-(arch,fn) module cache; `cuLaunchKernel` on torch stream (both cubins) |
| Bindings | `table.cu`/bindings | expose `link(fatbin, ltoir, cc)` + the driver-launched insert-and-evict entry |
| Config | `dynamicemb_config.py` | `LRU_LFU` enum, `score_function`, `lfu_decay_gamma`, `get_grouped_key` |
| Policy select | `key_value_table.py` `get_score_policy` | map `LRU_LFU`; route `num_scores==2` tables to the cubin (custom vs prebuilt Lex); `evict_strategy=LFU` |
| Python JIT glue | new `dynamicemb/score_jit.py` | `torch.cuda.get_device_capability()`; numba `cuda.compile(...,output='ltoir')`; hand bytes to C++ `link`; module cache |
| Batched | `batched_dynamicemb_tables.py` | `_create_score` LRU_LFU → evict=LFU/score=1; `get_score` returns `device_timestamp()` for LRU_LFU |

## 7. Interactions / invariants preserved

- **Layout unchanged** (word 0 = ts, word 1 = freq) ⇒ dump/load, rehash,
  gather/scatter/copy, overflow score maintenance, and `incremental_dump` all
  keep working with no change.
- **`num_scores == 1` fast path** is byte-identical; single-score strategies use
  the existing AoT extension launch, untouched.
- **Overflow eviction** stays counter-based (unchanged) — so a `score_function`
  does **not** affect eviction of keys living in the overflow table (R7). Overflow
  score words are still maintained for `LruLfu`.
- **`LFU + need_incremental_dump` behavior change (from decision Q):** now routes
  through the driver-launched Lex cubin, so its eviction gains the freq→ts
  tiebreak and its insert-and-evict launch path changes. Re-run its 18+7
  regression to confirm equivalence otherwise.

## 8. Testing plan

- **JIT link smoke test FIRST** (de-risks §5.3 ⚠️): numba-compile a trivial
  `user_score_fn` → LTO-IR; nvJitLink-link it with a minimal hand-written kernel
  fatbin; load + launch; assert the result. Confirms numba 0.64 LTO-IR ⇄ CUDA
  13.2 nvJitLink compatibility before building the real evict TU.
- **Default policy (no JIT):** extend `test_lru_lfu.py` — equal-frequency keys,
  assert the older-timestamp key is evicted and the newer survives; existing
  7 + 18 LruLfu/incremental tests stay green.
- **Custom decay (JIT):**
  - Unit: compile the example `compute_lfu_decay`; over random `[ts, freq]`,
    assert the device `user_score_fn` matches a CPU/numpy reference.
  - Eviction: construct keys where decay order ≠ frequency order; assert the
    lowest-decay key is evicted (not the lowest-frequency one).
  - Guardrails: `score_function` without `LRU_LFU` → error; missing numba-cuda →
    actionable error; module cache hit across repeated calls / distinct fns kept
    separate.
  - End-to-end on EOS (H100) in the devel container.

## 9. Risks / open questions

- ✅ **Score-word ordering (§3.1)** — resolved: `scores[0]=timestamp,
  scores[1]=frequency` (formula- and layout-consistent, zero remapping).
- ✅ **Tiebreak scope (§3.2)** — resolved: applies to all `LruLfu`.

- ✅ **JIT approach** — resolved: unified comparator-templated reduce; both
  default (Lex) and custom (UserFn) run as driver-launched cubins.
- ✅ **JIT scope** — resolved (Q): all `LruLfu` (`num_scores==2`) tables route
  through the cubin, incl. existing `LFU + need_incremental_dump`.
- ✅ **Default path dependency** — resolved (option ii): default Lex cubin is
  prebuilt at build; **numba only needed when a `score_function` is set**.
- ✅ **Async-prefetch preserved** — resolved: comparator reads the shared-mem
  prefetched pair, so the bulk reduce pipeline is retained (R4 mitigated).
- ✅ **Toolchain + LTO-IR compat — CONFIRMED** by the smoke test (numba 0.64
  LTO-IR ⇄ CUDA 13.2 nvJitLink → cubin; §5.3).
- ✅ **Link/launch location** — resolved: C++ (`binding.cpp`-style, torch's
  context), numba-compile in Python (avoids the `CUDA_ERROR_INVALID_CONTEXT` the
  Python-side smoke hit).

Remaining open / to-verify during impl:

1. ⚠️ **Shared-pointer generic load** — confirm the numba `user_score_fn`
   dereferences a shared-memory `scores` pointer correctly (extend the smoke test).
2. ✅ **uint64 timestamp arithmetic (R5)** — resolved: the example computes the
   age as `cur_ts - scores[0]` (a non-negative uint64), which avoids underflow and
   has the correct sign; `scores[0] - cur_ts` would both underflow and invert the
   decay direction.
3. **Numba API surface** — pin exact `cuda.compile(..., cc=<device cc>,
   output='ltoir', abi='c', abi_info={'abi_name':'user_score_fn'})` (note: `cc`
   is required — default `sm_50` is rejected by CUDA 13.2 NVVM).
4. **`get_grouped_key` identity** — `(module, qualname, getsource-hash)`; fragile
   for lambdas/REPL-defined fns (fall back to `id`).
5. **Regression from Q** — `LFU + need_incremental_dump` now uses the cubin path;
   re-run its 18+7 tests.
6. **Multi-GPU / determinism** — `%globaltimer` per-GPU; decay comparable only
   within a rank (fine, eviction is per-table-shard). The evict-module cache is
   device-keyed (one CUmodule per (device, key), since a CUmodule is bound to its
   context), so multiple GPUs in one process are supported.
7. ⚠️ **Heterogeneous multi-arch + custom score_function** — a registered custom
   function links with the compute capability recorded at first registration (the
   numba LTO-IR is arch-specific). If one process drives GPUs of *different* arch,
   loading the module on a mismatched-arch device fails. Homogeneous GPUs (the
   norm) are fine; heterogeneous would need per-arch numba compiles.

## 10. Implementation order (single delivery, not phased)

Landing it all at once, but in a dependency-safe sequence so each step is
verifiable:

1. **Smoke test ✅ (done)** — numba 0.64 LTO-IR ⇄ CUDA 13.2 nvJitLink link
   confirmed. Still TODO: extend it to (a) load+launch via C++/torch context and
   (b) pass a **shared-memory** `scores` pointer to the numba fn.
2. **`reduce<Comparator>()`** — refactor the scan to be comparator-templated,
   comparator reads the shared-mem `[ts,freq]` pair; add `LexFreqTsComparator` and
   `UserFnComparator`.
3. **Evict TU + build** — `evict_lrulfu.cu` instantiating insert-and-evict for
   both comparators; build the **prebuilt Lex cubin** and the **custom LTO-IR
   fatbin** (`user_score_fn` undefined), ship as package_data.
4. **`jit_link.{h,cpp}` + bindings** — C++ nvJitLink link + module cache +
   `cuLaunchKernel`; expose `link(...)` and the driver-launched insert-and-evict.
5. **Config + wiring** — `LRU_LFU` enum, `score_function`, `lfu_decay_gamma`,
   `get_grouped_key`; `get_score_policy` mapping; `_create_score` / `get_score`;
   route `num_scores==2` tables to the cubin (custom vs prebuilt Lex).
6. **`score_jit.py`** — numba compile + hand LTO-IR to C++ `link`; per-fn cache.
7. **Tests** (§8) + EOS H100 end-to-end, incl. the `need_incremental_dump` 18+7
   regression (decision Q changed its launch path).

## 11. Detailed design to neutralize the cubin/JIT risks

The §9 risks are addressed up front by two decisions: **(A) one `extern "C"` entry
per cubin taking a single POD params struct**, and **(B) fixed compile-time traits
for LruLfu so there is exactly one kernel variant per comparator.** Together these
collapse the "instantiation matrix × unchecked void\* ABI" surface to a single,
compiler-checked struct.

### 11.1 Single stable entry + POD ABI (kills R1, most of R2)

One shared header `evict_abi.cuh` defines the ABI, `#include`d by **both** the
cubin source and the C++ launcher — single source of truth, compiler-checked on
both sides:

```cpp
struct EvictParams {              // POD; layout is the ABI contract
  uint8_t*  table_storage;
  int64_t*  table_bucket_offsets;
  int32_t*  bucket_sizes;
  int64_t   batch;
  const int64_t* keys;           // KeyType fixed = int64 (index_type)
  const int64_t* table_ids;
  const uint64_t* in_scores;      // frequency deltas (ones); null => none
  uint8_t*  insert_results;
  int64_t*  evict_keys_out;       // nullable
  uint64_t* evict_scores_out;     // nullable  (replaces the OutputScore<bool> template)
  int32_t*  counter;
  int64_t   counter_offset;
  int64_t   bucket_capacity;
  int64_t   num_scores;           // == 2 for LruLfu
  double    gamma;                // consumed only by UserFnComparator
};
extern "C" __global__ void dyn_emb_evict_entry(EvictParams p);
```

- The entry is launched with **one argument** (`&p`), so `cuLaunchKernel`'s
  `kernelParams` is a single `void*` — no long hand-packed list to get wrong.
- `OutputScore<true/false>` becomes a **runtime null check** on
  `evict_scores_out`/`evict_keys_out` → removes that template dimension.
- `dyn_emb_evict_entry` unpacks `p`, builds the `Table`, reads `cur_ts` once
  (`%globaltimer`), and calls the shared body
  `insert_and_evict_body<Comparator>(p, cur_ts)`.

### 11.2 One variant per cubin (kills the rest of R2)

Fix all remaining compile-time knobs for LruLfu so each cubin has exactly **one**
entry named `dyn_emb_evict_entry` (no C++ mangling to resolve, no matrix):

- **KeyType = int64** (LruLfu tables use `index_type=int64`). If another key type
  is ever needed, it is a *new named entry*, enumerated explicitly — never
  implicit.
- **BufferDim / traits = the single tuned value for LruLfu** (pick the value the
  current dispatch uses for the 2-score path; document it). No `{32,1}` fork.
- **Comparator** is the only cross-cubin difference: Lex cubin compiles the body
  with `LexFreqTsComparator` (complete cubin, no undefined symbols); custom cubin
  compiles it with `UserFnComparator` (leaves `user_score_fn` undefined → LTO-IR
  fatbin for nvJitLink). **Same `.cu`, selected by a `-D` macro.**

`cuModuleGetFunction(mod, "dyn_emb_evict_entry")` — one stable name for both.

### 11.3 Surgical launch swap, op stays multi-kernel (kills R3)

Keep the existing `table_insert_and_evict` C++ launcher and ALL its orchestration
(the `update_counter_with_layout_kernel` at `insert_and_evict.cu:424`, dispatch,
overflow handling) **AoT and unchanged**. Only the single
`table_insert_and_evict_kernel<<<>>>` launch is replaced, for `num_scores==2`
tables, by a `cuLaunchKernel` of `dyn_emb_evict_entry`. One `if` at one call site;
the helper kernels never move into the cubin (they don't call the comparator).

### 11.4 No module-local globals (kills R4-globals)

Audit the evict device chain for `__constant__` / `__device__` global variables:
a separate CUDA module gets its **own uninitialized copy** of any such global.
Reserved keys (`EmptyKey`, `LockedKey`, …) are `static constexpr` (baked into
code, safe). Any real global found must be **passed through `EvictParams`**
instead. Add a grep gate to CI/build.

### 11.5 Multi-arch fatbins (kills R5-arch)

`setup.py` builds both artifacts for all target archs (env `DEMB_LRULFU_ARCHS`,
default `80;90;100`), mirroring ext_jit:
- **Lex (default):** `nvcc --fatbin -gencode arch=compute_XX,code=sm_XX`
  (+ `code=compute_XX` PTX fallback) → complete multi-arch fatbin.
- **Custom (LTO):** `nvcc --fatbin -gencode arch=compute_XX,code=lto_XX` → LTO-IR
  fatbin; nvJitLink picks the current device arch at link time.

### 11.6 One build, one flag list (kills R6)

`evict_abi.cuh` + the evict `.cu` are compiled in the **same `setup.py` run** as
the `.so`, from the same headers, with a **shared nvcc flag list** (a Python
constant reused for `.so` and both fatbins). No stale cubin (prebuilt each build);
flag parity guaranteed. A trivial build-time check loads each fatbin and asserts
`dyn_emb_evict_entry` resolves, so a missing/misnamed entry fails the **build**,
not runtime.

### 11.7 Residual, accepted

- **R5 (uint64 arithmetic in `user_score_fn`)** — contract requires
  `float64(scores[i])` before subtraction; enforced in docs + the shipped example;
  a unit test compares device vs numpy over random `[ts,freq]`.
- **First-use link latency (~100ms) + per-fn module cache** — acceptable; cache
  keyed by `(arch, score_function group key)`.
- **numba dependency** — only when `score_function` is set; default LruLfu
  (incl. `need_incremental_dump`) loads the prebuilt Lex fatbin, no numba.
