# HSTU CP v0 — Implementation Plan

**Drives**: `docs/cp/SPEC.md` (the contract)
**Companion**: `docs/cp/hstu_cp_design.md` (research / rationale)
**Updated**: 2026-04-27

This plan turns the SPEC into ordered, vertically-sliced tasks. Each task is
**one complete path from input to verified output** — not a horizontal "data
structure" / "algorithm" / "test" layer. Each task ends with a runnable check
that either passes or doesn't; a slice isn't done until its check passes.

## Global build-phase rules (apply to every task in every phase)

1. **Reference-driven**: every correctness check compares against the
   single-GPU baseline harness from Phase 0 (T0.1). We never compare
   against the PT reference or Triton inside production tests; the
   CUTLASS kernel itself is the reference.
2. **No functional regression**: every task ends with
   `bash examples/hstu/cp/run_regression.sh` (built in T0.3) exiting 0.
   That command runs all prior unit tests + the reference smoke test.
   The script must **fail-on-skip** for tests that the current
   environment is supposed to be able to run (i.e. multi-GPU tests
   are not silently skipped if `WORLD_SIZE` is unset on a multi-GPU box).
3. **No performance regression** — tiered:
   - **Phase 0 only**: `bench/baseline.py` numbers stay self-stable
     (re-running on the same commit shifts < 5 %).
   - **Slice 3 onward** (after T3.1 builds the CP wrapper): the wrapper's
     `cp_size=1` passthrough vs the unwrapped baseline must be within
     **+10 %** for shapes `< 1 ms` and **+5 %** for shapes `≥ 1 ms`.
     Smaller looser because Python-side autograd dispatch + guard checks
     are a fixed-cost overhead that swamps very small kernels. PRs that
     regress beyond these thresholds must either fix or re-baseline
     `bench_baseline.json` with explicit owner sign-off (never silent).
4. **Boundary coverage**: every new test must cover at least:
   minimum legal shape (single chunk per rank, i.e. `seqlens=[2*cp_size]`),
   one varlen mix where padding is large fraction of total tokens (e.g.
   `[2*cp_size, 2*cp_size, 2*cp_size, 32*cp_size]`), and at least one
   cell at `head_dim=256` (the full kernel-supported range; the
   `{32, 64, 128}` set is the test-matrix focus). Sliding-causal cases
   are **not** in v0 (see SPEC §2).
5. **Per-task tests are additive**: a slice never deletes tests from a
   prior slice. `tests/cp/` only grows.
6. **Runtime authority**: the *installed* `hstu` package's
   `hstu_attn_varlen_func` signature is the source of truth for kernel
   calls (it is what runs in production). The in-tree
   `corelib/hstu/hstu_attn/hstu_attn_interface.py` source can lag the
   installed package signature; T0.1 records the installed signature
   and tests pin to it.

---

## Dependency graph

```
        ┌──────────  Phase 0  ──────────┐
        │  Reference test + benchmark    │
        │  infrastructure                │
        └────────────────┬───────────────┘
                         ▼
                ┌── Checkpoint 0 ──┐
                │ infra ready;     │
                │ baseline locked  │
                └────────┬─────────┘
                         ▼
                    Slice 1 (DONE)
                    PoC equal-len cp=2 fwd
                            │
                            ▼
        ┌──────────  Slice 2  ──────────┐
        │  PoC general cp_size + varlen │
        │     (single-rank simulation)   │
        └────────────────┬───────────────┘
                         ▼
                ┌── Checkpoint A ──┐
                │ math validated   │
                │ across (cp, len) │
                └────────┬─────────┘
                         ▼
        ┌──────────  Slice 3  ──────────┐
        │  Multi-GPU forward,            │
        │  sequential ring P2P           │
        └────────────────┬───────────────┘
                         ▼
                ┌── Checkpoint B ──┐
                │ real fwd correct │
                └────────┬─────────┘
                         ▼
        ┌──────────  Slice 4  ──────────┐
        │  Multi-GPU backward            │
        │  (reverse ring)                │
        └────────────────┬───────────────┘
                         ▼
                ┌── Checkpoint C ──┐
                │  v0 done? merge? │
                └────────┬─────────┘
                         ▼
        ┌──────────  Slice 5  ──────────┐  (v0+ / v0.5)
        │  Two-stream + comm/compute     │
        │  overlap + perf gate           │
        └────────────────┬───────────────┘
                         ▼
                ┌── Checkpoint D ──┐
                │ perf gate met?   │
                └────────┬─────────┘
                         ▼
        ┌──────────  Slice 6  ──────────┐  (post-v0; optional)
        │  HSTUConfig + module +         │
        │  dataloader integration        │
        └────────────────────────────────┘
```

**Hard dependencies**: each slice depends on the prior one's checkpoint passing.
**Parallelisable**: nothing — by design we go top-down, slice by slice.
**Re-plan triggers**: see "Risk & re-planning" at the bottom.

---

## Phase 0 — Reference test + benchmark infrastructure

**Goal**: lock in a single-GPU reference test harness and a single-GPU
reference benchmark **before** we add any CP machinery, so every subsequent
task can verify "no functional regression" and "no performance regression"
against a stable, committed baseline.

**Total estimate**: 0.5–1 day.

### T0.1 — Reference test harness

**Why first**: every CP correctness test compares against a single-GPU
baseline. We need a stable, reusable harness that produces this baseline
deterministically, plus a battery of boundary-case shapes.

**Deliverable**:
- `examples/hstu/test/cp/conftest.py` (new) — pytest fixtures:
  - `single_gpu_baseline_fwd(q, k, v, cu_seqlens, max_seqlen, *, alpha,
    window_size, scaling_seqlen)` → calls `hstu_attn_varlen_func`,
    returns the output.
  - `single_gpu_baseline_fwd_bwd(...)` → as above + grads via autograd
    on `dout`.
  - `random_varlen_batch(seqlens, num_heads, head_dim, dtype, seed)` →
    deterministic input generator (same seed → same `q,k,v`).
  - `assert_cp_close(actual, ref, fwd: bool)` → wraps the existing
    `examples/commons/utils/hstu_assert_close.py` with bf16 tolerance.
- `examples/hstu/test/cp/test_reference.py` (new) — runs the baseline on
  the SPEC §3 Slice 2 matrix + the boundary-case extension below; asserts
  output shape, finiteness, and self-consistency (running twice with the
  same seed gives identical bytes).
- **Installed-signature pin**: `test_reference.py::test_hstu_signature_pinned`
  introspects `inspect.signature(hstu.hstu_attn_varlen_func)` and asserts
  the parameter names + defaults match the canonical list captured in a
  test fixture. If a future package upgrade changes the signature, this
  test fails first — explicit re-pin required (per Global rule 6).
- **Finiteness guard**: every matrix entry asserts `torch.isfinite(out).all()`
  *and* `torch.isfinite(grad).all()` for the largest tested shape. This
  guards against the lower-triangle K-zero-padding NaN/Inf risk at scale.

**Boundary cases (mandatory in every task's tests, set up here)**:
- `cp_size = 1` (degenerate; passthrough path).
- `seqlens = [2 * cp_size]` (smallest legal — exactly one chunk per rank).
- All-equal-length and one-very-long-one-very-short (e.g.
  `[16, 16, 16, 4096]` — must be divisible by `2*cp_size`, so for
  `cp_size=4` use `[16, 16, 16, 4096]` directly; for `cp_size=8` use
  `[16, 16, 16, 4096]` with all values multiples of 16).
- Mask is pure causal `window_size=(-1, 0)` only (sliding-causal is
  out of v0 scope — see SPEC §2).
- `head_dim ∈ {32, 64, 128}` (the values existing HSTU tests already
  cover); plus at least one cell at `head_dim=256` to exercise the
  full kernel-supported range.

**Acceptance**:
- `pytest examples/hstu/test/cp/test_reference.py -v` PASS on a single GPU.
- The reference matrix has ≥ 12 distinct `(cp_size, seqlens, mask, H, D)`
  tuples explicitly enumerated.

**Verification**:
- `pytest examples/hstu/test/cp/test_reference.py -v` PASS.

**Estimate**: 0.5 day.

### T0.2 — Reference benchmark harness

**Why first**: any change after this point that makes the `cp_size=1`
passthrough slower than the baseline is a perf regression we want to
catch *at PR time*, not at v0.5 once damage has accumulated.

**Deliverable**:
- `examples/hstu/cp/bench/baseline.py` (new) — runs `hstu_attn_varlen_func`
  over a fixed shape grid (committed in the file itself), N timed
  iterations after warmup. Outputs JSON to a path (default
  `tasks/bench_baseline.json`):
  ```json
  {
    "commit": "<git sha>",
    "device": "<device name>",
    "shapes": [
      {"label": "h2_d32_b4_s64",    "median_ms": ..., "p95_ms": ..., "tokens_per_s": ...},
      ...
    ]
  }
  ```
- `examples/hstu/cp/bench/compare.py` (new) — reads two JSONs (baseline +
  candidate), prints a delta table, exits non-zero if any shape regresses
  by > 5 % (median_ms threshold). Tolerates new shapes added; flags
  removed shapes as warnings.
- `tasks/bench_baseline.json` (committed) — the canonical numbers from
  the current commit, generated on the user's reference GPU (A100 80GB
  PCIe per the existing container).

**Acceptance**:
- Run `bench/baseline.py` twice on the same commit; `compare.py` exits 0.
- The committed `bench_baseline.json` has ≥ 6 shape entries spanning
  small (h=2 d=32 seqlen=64) to medium (h=8 d=128 seqlen=8192).

**Verification**:
- `python examples/hstu/cp/bench/baseline.py --output /tmp/cur.json &&
  python examples/hstu/cp/bench/compare.py tasks/bench_baseline.json
  /tmp/cur.json` exits 0.

**Estimate**: 0.5 day.

### T0.3 — One-shot regression command

**Deliverable**:
- `examples/hstu/cp/run_regression.sh` (new) — single command every PR
  runs locally before claiming green:
  - All `examples/hstu/test/cp/*.py` pytest files (single GPU, no
    torchrun required for the reference subset).
  - `bench/compare.py tasks/bench_baseline.json /tmp/cur.json` after
    re-running `bench/baseline.py`.
- Prints a summary at the end: `OK: 12 tests passed, perf within 5%` or
  the failing items.

**Acceptance**:
- `bash examples/hstu/cp/run_regression.sh` exits 0 on a clean checkout
  immediately after Phase 0 completes.

**Verification**:
- Same.

**Estimate**: 0.2 day.

---

### ✅ Checkpoint 0 — reference infra ready; safe to start adding CP

**Stop, review, decide**:
1. Is the `bench_baseline.json` representative of the workloads we
   actually care about? If not, expand the shape grid before any CP work.
2. Are the boundary cases in T0.1 the ones we'll regret missing? Add now,
   not later.
3. Does the regression command run fast enough (< 60s) that contributors
   will actually run it before each PR? If not, split into "fast" and
   "full" tiers.

**Owner sign-off required to proceed.**

---

## Phase 1 — PoC generalisation (Slice 2)

**Goal**: prove the per-tile recipe (DualChunkSwap + 3-region tiling +
plain-sum reduction) holds for arbitrary `cp_size` and varlen inputs, all
on a single GPU. No comm.

**Total estimate**: 2–3 days.

### T2.1 — PoC supports cp_size > 2 (still equal-len)

**Why first**: cp_size=2 has only one off-diagonal tile per rank. cp_size=4
has THREE. Generalising the (rank, step) classification grid is the first
new piece of math; varlen is a separate concern layered on top.

**Deliverable**:
- Generalise `cp_simulate(...)` in `examples/hstu/cp/poc_dualrank_sim.py` to
  accept `CP_SIZE ∈ {2, 4, 8}`. The simulator must walk the full
  `cp_size × cp_size` (rank, step) grid and classify each tile as
  `diagonal | lower-triangle | upper-triangle`.
- Maintain the cp_size=2 special case as a regression check.

**Files**:
- `examples/hstu/cp/poc_dualrank_sim.py` — extend `cp_simulate`,
  `build_local_shard`, `select_second_half_per_sample`,
  `zero_second_half_per_sample`. Keep all of these single-rank-only.

**Acceptance**:
- `cp_size=4` equal-len matches single-GPU baseline at bf16 tolerance.
- `cp_size=2` regression still passes with same tolerance.

**Verification**:
- `python examples/hstu/cp/poc_dualrank_sim.py --cp-size 2` PASS
- `python examples/hstu/cp/poc_dualrank_sim.py --cp-size 4` PASS

**Estimate**: 0.5 day.

### T2.2 — PoC supports varlen (cp_size=2 first)

**Why next**: varlen is orthogonal to `cp_size > 2`. Re-introduce at
cp_size=2 to keep the change small.

**Deliverable**:
- `make_batch(seqlens=[…])` accepts a list of per-sample lengths, each
  divisible by `2 * cp_size`.
- `build_local_shard` already varlen-aware from previous (interrupted)
  refactor — verify it still works after the cp_size>2 generalisation in T2.1.

**Files**:
- `examples/hstu/cp/poc_dualrank_sim.py`.

**Acceptance**:
- Varlen at `cp_size=2`, seqlens `[16, 32, 48, 64]` matches single-GPU baseline.

**Verification**:
- `python examples/hstu/cp/poc_dualrank_sim.py --cp-size 2 --varlen` PASS.

**Estimate**: 0.3 day.

### T2.3 — Matrix sweep (varlen × cp_size, pure causal only)

**Deliverable**:
- A small driver in the same file that runs the SPEC §3 Slice 2 matrix:

  | cp_size | seqlens                                | H | D   | mask   |
  | --      | --                                     | - | --  | --     |
  | 2       | `[64,64,64,64]`                        | 2 | 32  | causal |
  | 2       | `[16, 32, 48, 64]`                     | 2 | 32  | causal |
  | 4       | `[8, 8, 8, 256]`  *(padding-heavy)*    | 2 | 32  | causal |
  | 4       | `[128, 256, 384, 512]`                 | 4 | 64  | causal |
  | 8       | `[16]*7 + [1024]`  *(padding-heavy)*   | 4 | 128 | causal |
  | 8       | `[512, 1024, 1024, 2048]`              | 4 | 128 | causal |

**Files**:
- `examples/hstu/cp/poc_dualrank_sim.py` (`main()` reads a small config list).

**Acceptance**:
- All 6 cells in the matrix PASS.
- Output prints a clean table: `cp_size / seqlens / max|diff| / PASS|FAIL`.

**Verification**:
- `python examples/hstu/cp/poc_dualrank_sim.py --matrix` PASS on all cells.

**Estimate**: 0.5 day.

---

## ✅ Checkpoint A — math validated; ready for production code

**Stop, review, decide**:
1. Did all matrix cells pass within bf16 tolerance? If any failed,
   investigate before moving on.
2. **Padding-cost measurement (SPEC §9.1)** — required before Phase 2.
   Run T2.3's matrix on a representative recsys seqlen distribution
   (or sample from a real training set) and report:
   `padding_token_count / total_token_count` per cp_size. If
   overhead > 30 % at the target cp_size, **escalate to Track B
   (MagiAttention chunk-dispatch)** — Phase 2 is blocked until owner
   re-plans.
3. Is the PoC code clean enough to use as the numerical oracle for
   Slices 3 and 4, or does it need a refactor first? (Default: it's an
   oracle, leave it; clarity beats speed.)

**Owner sign-off required to proceed.**

---

## Phase 2 — Multi-GPU forward (Slice 3)

**Goal**: a real `hstu_attn_varlen_cp_func` that runs on `cp_size` GPUs and
produces forward output numerically equal to the single-GPU baseline.

**Total estimate**: 4–8 days (T3.1 0.5d + T3.2 0.5d + T3.3 2-4d + T3.4 1-2d).

### T3.1 — Public API skeleton + hard guards (single-GPU passthrough)

**Why first**: get the API surface right and reviewed before implementing
any comm. If `cp_size == 1`, the function short-circuits to
`hstu_attn_varlen_func` *before* any wrapping — that's the trivial smoke
test, and it keeps the cp=1 perf overhead near zero.

**Deliverable**:
- New file `corelib/hstu/hstu_attn/hstu_attn_cp.py`.
- Public `hstu_attn_varlen_cp_func` whose signature **mirrors the
  installed `hstu_attn_varlen_func` exactly**, plus four CP arguments at
  the end. The mirror includes every argument the user might supply,
  whether or not v0 supports it — that way the wrapper's body is what
  rejects v0+ modes, not the function signature.

  ```python
  def hstu_attn_varlen_cp_func(
      q, k, v,
      cu_seqlens_q, cu_seqlens_k,
      seqused_q, seqused_k,         # mirrors installed kernel sig
      max_seqlen_q, max_seqlen_k,
      scaling_seqlen,
      num_contexts, num_targets,    # guarded inside body
      target_group_size=1,
      window_size=(-1, -1),
      alpha=1.0,
      rab=None,                     # guarded inside body
      has_drab=False,               # guarded inside body
      kv_cache=None, page_offsets=None, page_ids=None,
      last_page_lens=None, func=None, quant_mode=-1,
      *,
      cp_group=None,                # CP arg
      cp_global_ranks=None,         # CP arg
      cp_stream=None,               # CP arg
      cp_comm_type="p2p",           # CP arg
  )
  ```

  T0.1 records the canonical installed signature; this stays in sync.
- Body order:
  1. `cp_group is None or cp_size == 1` short-circuit: direct
     `hstu_attn_varlen_func(...)` call with all the original args.
     (No autograd wrap, no guard cycle — keeps overhead near zero.)
  2. Hard guards: reject each of the following with a `ValueError`
     referencing SPEC §2 — `rab is not None`, `has_drab=True`,
     `num_contexts is not None`, `num_targets is not None`,
     `target_group_size > 1`, `window_size != (-1, 0)`,
     `kv_cache is not None`, `page_offsets is not None`,
     `page_ids is not None`, `last_page_lens is not None`,
     `func is not None`, `quant_mode != -1`.
  3. `head_dim` guard: assert `q.shape[-1] in {32, 64, 128, 256}`.
  4. Divisibility guard: per-sample seqlens divisible by `2 * cp_size`.
  5. Multi-GPU path → autograd-Function (forward raises
     `NotImplementedError("v0 forward arrives in T3.3")` in this task,
     replaced in T3.3). Backward raises `NotImplementedError("v0
     backward arrives in T4.2")`.
- `__init__.py` exports the new symbol.

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (new, ~150 lines).
- `corelib/hstu/hstu_attn/__init__.py` (one new export).

**Acceptance**:
- `cp_size == 1` (or `cp_group is None`): bit-exact match with
  `hstu_attn_varlen_func` on a small random input.
- Hard guards: each rejected input produces the documented `ValueError`.
- `cp_size > 1` path raises `NotImplementedError` with the right message.
- A monkeypatch test asserts cp=1 path does **not** call any
  `torch.distributed` collective and does **not** call into the chunking
  helper (proves it's a true short-circuit).

**Verification**:
- `pytest examples/hstu/test/cp/test_cp_api_smoke.py` PASS on a single GPU.
- `bash examples/hstu/cp/run_regression.sh` PASS (perf at cp=1 within
  tier-3 thresholds from §Global rule 3).

**Estimate**: 0.5 day.

### T3.2 — Pure-permutation `get_batch_on_this_cp_rank_for_hstu` helper

**Why next**: pure, no comm, easy to test independently. Used by Slices
3, 4, and 6.

**Deliverable**:
- Function that, given a packed jagged batch `(q, k, v, cu_seqlens)` and
  `(cp_size, cp_rank)`, returns the local shard plus a `local_to_global`
  index tensor. Pure permutation; no comm.
- An inverse function `gather_global_from_cp_rank` that runs on all ranks
  to materialise the global tensor (used only for testing).
- Each per-sample seqlen must be divisible by `2*cp_size`; otherwise
  raises `ValueError`.

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (extend, +~80 lines).

**Acceptance**:
- Round-trip: `gather(scatter(global)) == global` for any varlen batch
  divisible by `2*cp_size`.
- Per-rank shard sizes balanced (each rank holds exactly
  `total_tokens / cp_size`).

**Verification**:
- `pytest examples/hstu/test/cp/test_cp_dispatch.py` (new file, single
  GPU): round-trip and balance asserts at `cp_size ∈ {2, 4, 8}`.

**Estimate**: 0.5 day.

### T3.3 — Multi-GPU forward (sequential ring, cp_size=2)

**Why incremental**: get one ring step right before scaling.

**Deliverable**:
- Inside `hstu_attn_varlen_cp_func`, when `cp_size > 1`:
  - Use the helper from T3.2 to assert local shard.
  - Run the ring loop: `cp_size` attention steps, each preceded by a
    blocking `dist.batch_isend_irecv` of the next KV.
  - Per step, classify the tile (`diagonal | lower-tri | upper-tri`).
  - Slice Q / K / V per the SPEC §2 recipe (referencing `poc_dualrank_sim.py`
    as oracle).
  - Plain-sum partial outputs; divide by `scaling_seqlen` once via the
    per-call kernel argument.
  - Allgather the per-rank output if `cp_comm_type` requires it (default:
    return local shard so the caller decides — match TE pattern).

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (extend).
- `examples/hstu/test/cp/test_cp_forward.py` (new, torchrun-based pytest).

**Acceptance**:
- `torchrun --nproc-per-node=2` cp=2 equal-len fwd matches single-GPU
  baseline at bf16 tolerance.

**Verification**:
- `torchrun --standalone --nproc-per-node=2 -m pytest
  examples/hstu/test/cp/test_cp_forward.py -k cp2` PASS.

**Estimate**: 2–4 days. (Highest-risk task in Phase 2 — first real comm.
Budget includes likely NCCL-ordering / stream-event-sync debugging time.)

### T3.4 — Scale to cp_size ∈ {4, 8} + varlen (causal only)

**Deliverable**:
- Run the same matrix from T2.3 (causal-only, no sliding) but on real
  GPUs at cp_size=4 and cp_size=8.
- New helper `examples/hstu/cp/run_cp_tests.sh` (explicit deliverable):
  loops `torchrun --standalone --nproc-per-node=N` for `N ∈ {2,4,8}`
  and runs the multi-GPU CP test files. Accepts an optional `--bwd`
  switch that gets used by Slice 4 (no-op here).
- No new CP-internal code paths expected; if the PoC matrix passed and
  T3.3 passed, this should "just work". If it doesn't, we discovered
  bugs the single-rank simulator missed — backport that case to the
  simulator first, then fix.

**Files**:
- `examples/hstu/test/cp/test_cp_forward.py` (extend with parametrised
  matrix).
- `examples/hstu/cp/run_cp_tests.sh` (new).

**Acceptance**:
- Full SPEC §3 Slice 3 matrix PASS via torchrun N=2,4,8.

**Verification**:
- `bash examples/hstu/cp/run_cp_tests.sh` PASS.

**Estimate**: 1–2 days.

---

## ✅ Checkpoint B — real multi-GPU forward correct; backward next

**Stop, review, decide**:
1. Did anything in T3.3/T3.4 reveal a bug the single-rank simulator missed?
   If so, **add a regression test in the simulator** before moving on.
2. Was 50% of effort spent debugging NCCL plumbing? If yes, before Slice 4
   make sure the comm/event/stream conventions are documented in
   `hstu_cp_design.md` so Slice 4 reuses them.
3. Has the public API surface stabilised? Slice 4 will not change it.

**Owner sign-off required to proceed.**

---

## Phase 3 — Multi-GPU backward (Slice 4)

**Goal**: backward pass produces gradients numerically equal to single-GPU
baseline at the same tolerance.

**Total estimate**: 2.5–4 days.

### T4.1 — PoC backward (single-rank simulator)

**Why first**: an oracle for Slice 4's multi-GPU backward, just as the
forward simulator was the oracle for Slice 3.

**Deliverable**:
- Add an `autograd.Function` to the PoC: forward calls `cp_simulate`,
  backward simulates the reverse-ring backward (dQ accumulates locally;
  dKV ride a reverse ring, copy on first visit / add after).
- Oracle: a single call to `hstu_attn_varlen_func` with `requires_grad=True`,
  same `dout`. Compare `q.grad / k.grad / v.grad`.

**Files**:
- `examples/hstu/cp/poc_dualrank_sim.py` (extend, +~150 lines).

**Acceptance**:
- Gradients match single-GPU baseline at bf16 tolerance over the SPEC §3
  Slice 2 matrix.

**Verification**:
- `python examples/hstu/cp/poc_dualrank_sim.py --matrix --bwd` PASS.

**Estimate**: 0.5–1 day.

### T4.2 — Multi-GPU backward at cp_size=2

**Deliverable**:
- Implement `autograd.Function.backward` in `hstu_attn_varlen_cp_func`.
- Reverse-direction ring (send to `(rank-1)`, recv from `(rank+1)`).
- dQ stays local; dKV ride the reverse ring with `copy_` on first visit
  and `add_` after.
- Same per-tile classification as forward; call `hstu_attn_cuda.varlen_bwd`
  per tile.
- For lower-triangle tiles with K-zero-padding: dK/dV partials at the
  zero-padded positions are dropped at scatter time.

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (extend).
- `examples/hstu/test/cp/test_cp_backward.py` (new).

**Acceptance**:
- `torchrun --nproc-per-node=2` cp=2 fwd+bwd grads match single-GPU
  at bf16 tolerance.

**Verification**:
- `torchrun --standalone --nproc-per-node=2 -m pytest
  examples/hstu/test/cp/test_cp_backward.py -k cp2` PASS.

**Estimate**: 1–2 days. (Highest-risk in Phase 3.)

### T4.3 — Scale backward to cp_size ∈ {4, 8} + varlen (causal only)

**Deliverable**:
- Run the same matrix from T2.3 / T3.4, now with backward.
- Extend `run_cp_tests.sh` to honour `--bwd` (now actually does
  something — runs the bwd pytest files).
- Add regression of the forward matrix (so a backward bug doesn't break
  forward).

**Files**:
- `examples/hstu/test/cp/test_cp_backward.py` (extend).
- `examples/hstu/cp/run_cp_tests.sh` (extend with `--bwd`).

**Acceptance**:
- Full SPEC §3 Slice 4 matrix PASS via torchrun N=2,4,8.

**Verification**:
- `bash examples/hstu/cp/run_cp_tests.sh --bwd` PASS at N=2/4/8.

**Estimate**: 0.5–1 day.

---

## ✅ Checkpoint C — v0 correctness done; merge decision

**Stop, review, decide**:
1. Are Slices 1–4 all green? → v0 correctness done. Cut a PR or split
   into multiple PRs (one per slice — recommended).
2. **v0 ships here.** SPEC §8 locks v0 = Slices 1–4 (correctness only);
   Slice 5 (overlap + perf) is v0+/v0.5 and ships separately. Do not
   block v0 on Slice 5 perf numbers.
3. Set up release notes / migration guide for users who'll be the early
   adopters.

**Owner sign-off required.** This is the v0 merge gate.

---

## Phase 4 — Comm/compute overlap + perf (Slice 5; v0+ / v0.5)

**Goal**: bring multi-GPU step time down to within `1.5× single-GPU
per-token throughput` at cp_size=4.

**Total estimate**: 2–4 days.

### T5.1 — Two-stream skeleton + double buffering

**Deliverable**:
- Take `cp_stream` (currently unused) seriously. Step `i` runs on
  `streams[i % 2]`; KV ring slot double-buffered.
- Sync via `torch.cuda.Event`s between corrections.

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (refactor ring loop).

**Acceptance**:
- Correctness regression: full Slice 4 matrix still PASS.

**Verification**:
- All Slice 4 tests still PASS unchanged.

**Estimate**: 1–2 days.

### T5.2 — Perf harness + NSys profile

**Deliverable**:
- `examples/hstu/cp/bench_cp.py` — non-pytest perf script. Single shape
  matrix, prints throughput per cp_size; emits NSys profile.

**Files**:
- `examples/hstu/cp/bench_cp.py` (new).

**Acceptance**:
- Profile shows next-step P2P running concurrently with current-step
  attention kernel (visual check on NSys).

**Verification**:
- `nsys profile --output cp_profile python examples/hstu/cp/bench_cp.py
  --cp-size 4 --shape h8d128b8s8k` and inspect.

**Estimate**: 0.5 day.

### T5.3 — Hit the perf gate

**Deliverable**:
- Tune until `cp_size=4` HSTU forward+backward step time ≤
  `1.5 × single-GPU baseline-per-token-throughput` on a single H100/A100
  node. (See SPEC §3 Slice 5.)

**Files**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (incremental tuning).

**Acceptance**:
- `bench_cp.py` numbers meet the gate.

**Estimate**: variable (0.5–2 days depending on what the bottleneck turns
out to be).

---

## ✅ Checkpoint D — perf gate decision

**Stop, review, decide**:
1. Did we hit the gate? If not, what's the next bottleneck and is it worth
   another iteration vs. accepting and shipping?
2. If perf is good enough, decide whether Slice 6 (training integration)
   begins now or waits for the next quarter.

**Owner sign-off.**

---

## Phase 5 — HSTU module / training integration (Slice 6; post-v0, optional)

Not detailed here. Re-plan when (if) Slice 6 starts. High-level tasks:

- T6.1 — `HSTUConfig.cp_size`, Megatron parallel-state CP group creation.
- T6.2 — `HSTUAttention` module accepts `cp_group`, routes to
  `hstu_attn_varlen_cp_func`.
- T6.3 — Dataloader DualChunkSwap shuffle + per-rank `cu_seqlens`
  precompute.
- T6.4 — E2E training smoke test.

---

## Risk & re-planning

| Trigger | Re-plan action |
| -- | -- |
| Slice 2 matrix fails on a particular case | Deep-dive that case before continuing. Possible cause: my mask-construction understanding wrong → SPEC §1 may need correction. |
| Slice 3 multi-GPU fails despite Slice 2 passing | Single-rank simulator is missing something (most likely: NCCL ordering or stream sync). Add the missing test to the simulator before fixing on real GPUs. |
| Padding overhead > 30% on a real recsys workload | Escalate to Track B (MagiAttention-style chunk-dispatch). Phase 2 is blocked at Checkpoint A until owner re-plans (matches §Checkpoint A bullet 2). |
| Slice 4 backward gradients drift outside tolerance | Step through per-tile bwd output vs single-GPU. Likely cause: K-zero-padding tile produces wrong dK at the padded positions and we don't drop it correctly at scatter. |
| Slice 5 can't hit the perf gate at cp_size=4 | Decide between (a) ship without overlap and reduce gate, (b) explore Ulysses (`a2a`) mode — but this is a major scope expansion. |
| **Perf regression** detected by `bench/compare.py` at any task | Block the PR. Three options: (a) fix the regression, (b) prove the slowdown is acceptable and re-baseline `bench_baseline.json` with explicit owner sign-off, (c) revert. Default: (a). Never silently re-baseline. |
| **Functional regression** in any prior slice's tests | Block the PR. The new task is wrong (most likely) or it surfaced a latent bug — investigate before merging. The "tests are additive" rule means we caught it; don't disable the failing test. |
| Reference benchmark numbers shift between runs on the same commit > 5 % | The bench harness is too noisy; bump warmup, increase iteration count, or pin the GPU/clocks before adding more shapes. |
| Multi-GPU ring hangs / deadlocks during T3.3 / T3.4 / T4.x | Required observability: every ring loop logs `rank:step:event` to stdout under `HSTU_CP_DEBUG=1`; tests run with `NCCL_ASYNC_ERROR_HANDLING=1` and a 60s watchdog timeout. On hang, dump the last `(rank, step, event)` per rank. Fix root cause; do not "just retry". |
| Lower-triangle K-zero-padding produces NaN/Inf at large seqlens / head_dim | The PoC was small; SiLU(0)·V=0 is bounded, but kernel-internal numerics may differ at scale. T0.1 includes a finiteness assertion at the largest tested shape; T2.3's padding-heavy cases (`[8, 8, 8, 256]` at cp_size=4 and `[16]*7 + [1024]` at cp_size=8) maximise the padded region inside the lower-triangle tile. |
| Installed `hstu` package signature changes between revisions | T0.1 records the canonical signature in a test-time assertion. If a future package upgrade breaks the signature, the assertion fails before any CP test runs — explicit re-pin required. |

---

## Day-1 starting point

After this plan is signed off and Codex review LGTM, the first concrete
action is **Phase 0**:

1. **T0.1**: Build the reference test harness (`conftest.py` +
   `test_reference.py`) and prove it runs green on the current commit.
2. **T0.2**: Build the reference benchmark harness, generate
   `tasks/bench_baseline.json` on the reference GPU, commit it.
3. **T0.3**: Wire the one-shot regression command and prove it runs in
   under a minute green.
4. **Stop at Checkpoint 0**, owner sign-off.
5. *Then* address the in-progress varlen refactor of `poc_dualrank_sim.py`
   (T2.0): verify it still passes equal-len, or revert to the verified
   version. Begin Slice 2 from a clean reference-checked base.

Phase 0 is the prerequisite. Slices only start after Checkpoint 0.
