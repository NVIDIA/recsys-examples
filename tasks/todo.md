# HSTU CP v0 — TODO

Flat checklist. Drives `tasks/plan.md`. Tick as we go; checkpoints are
**owner sign-off** moments — do not start the next task until the prior
checkpoint is signed.

**Global rules** (apply to every task):
- Reference-driven: compare against single-GPU baseline from T0.1.
- No functional regression: `bash examples/hstu/cp/run_regression.sh` exits 0
  (fail-on-skip for required multi-GPU tests).
- No perf regression — tiered:
  - Phase 0: same-commit `bench/baseline.py` re-runs within 5%.
  - Slice 3+: cp=1 passthrough vs unwrapped baseline within +10% (shapes
    < 1ms) / +5% (shapes ≥ 1ms).
- Boundary coverage required: smallest legal shape `[2*cp_size]`, padding-
  heavy varlen, at least one cell at `head_dim=256` (full kernel-supported
  range). **No sliding-causal in v0** (dropped from scope; see SPEC §2).
- Tests are additive: never delete a prior test.
- Runtime authority: installed `hstu` package signature is source of truth.

---

## Phase 0 — Reference infrastructure

- [ ] **T0.1**: Reference test harness
  - File: `examples/hstu/test/cp/conftest.py` + `test_reference.py` (new)
  - Acceptance: `pytest examples/hstu/test/cp/test_reference.py -v` PASS; ≥ 12 boundary tuples
- [ ] **T0.2**: Reference benchmark harness + commit `tasks/bench_baseline.json`
  - File: `examples/hstu/cp/bench/baseline.py` + `bench/compare.py` (new)
  - Acceptance: same-commit double-run via `compare.py` exits 0; ≥ 6 shapes
- [ ] **T0.3**: One-shot regression command
  - File: `examples/hstu/cp/run_regression.sh` (new)
  - Acceptance: exits 0 in < 60s on clean checkout

### ✅ Checkpoint 0 — reference infra ready

- [ ] Owner sign-off

---

## Phase 1 — PoC generalisation (Slice 2)

- [x] **S1**: PoC equal-len cp=2 fwd PASS (max |diff|=1.95e-3, bf16) — done before SPEC
- [ ] **T2.0**: Verify or revert in-progress varlen refactor of `poc_dualrank_sim.py` (start from clean base)
- [ ] **T2.1**: Generalise PoC to `cp_size > 2` (3-region tile grid for cp_size=4) — equal-len only
  - Acceptance: `--cp-size 4` equal-len PASS; `--cp-size 2` regression PASS
- [ ] **T2.2**: PoC supports varlen at cp_size=2
  - Acceptance: varlen `[16,32,48,64]` cp=2 PASS
- [ ] **T2.3**: Matrix sweep (cp ∈ {2,4,8}, varlen, causal only; padding-heavy boundaries)
  - Acceptance: all 6 SPEC §3 Slice 2 matrix cells PASS

### ✅ Checkpoint A — math validated; ready for production code

- [ ] Padding-cost measurement on representative recsys seqlens
      (per plan §Checkpoint A bullet 2): `padding/total ≤ 30 %` at
      target cp_size, else escalate to Track B and block Phase 2.
- [ ] Owner sign-off

---

## Phase 2 — Multi-GPU forward (Slice 3)

- [ ] **T3.1**: Public API skeleton + hard guards + cp=1 passthrough
  - File: `corelib/hstu/hstu_attn/hstu_attn_cp.py` (new)
  - Acceptance: cp=1 path bit-exact match; each rejected input → documented `ValueError`
- [ ] **T3.2**: `get_batch_on_this_cp_rank_for_hstu` helper (pure permutation)
  - Acceptance: round-trip identity; per-rank shard size balanced at cp ∈ {2,4,8}
- [ ] **T3.3**: Multi-GPU forward, sequential ring P2P, cp_size=2
  - File: `examples/hstu/test/cp/test_cp_forward.py` (new, torchrun)
  - Acceptance: torchrun N=2 cp=2 fwd matches single-GPU (bf16)
- [ ] **T3.4**: Scale to cp_size ∈ {4, 8} + varlen (causal only)
  - File: `examples/hstu/cp/run_cp_tests.sh` (new helper)
  - Acceptance: SPEC §3 Slice 3 matrix PASS via torchrun N=2,4,8

### ✅ Checkpoint B — multi-GPU forward correct

- [ ] Owner sign-off

---

## Phase 3 — Multi-GPU backward (Slice 4)

- [ ] **T4.1**: PoC backward (autograd in single-rank simulator)
  - Acceptance: SPEC §3 Slice 2 matrix PASS for grads (single-rank oracle)
- [ ] **T4.2**: Multi-GPU backward at cp_size=2 (reverse ring)
  - File: `examples/hstu/test/cp/test_cp_backward.py` (new, torchrun)
  - Acceptance: torchrun N=2 cp=2 fwd+bwd grads match single-GPU
- [ ] **T4.3**: Scale backward to cp_size ∈ {4, 8} + varlen (causal only)
  - Acceptance: SPEC §3 Slice 4 matrix PASS via torchrun N=2,4,8 with bwd

### ✅ Checkpoint C — v0 correctness done; merge decision

- [ ] Owner sign-off
- [ ] Cut PR(s) — recommended: one per slice (S2/S3/S4)
- [ ] **v0 ships here.** SPEC §8 locks v0 = Slices 1–4. Slice 5 is v0+
      and ships separately; do NOT block v0 merge on Slice 5 perf.

---

## Phase 4 — Overlap + perf (Slice 5; v0+ / v0.5)

- [ ] **T5.1**: Two-stream + double-buffered KV ring
  - Acceptance: Slice 4 matrix still PASS unchanged
- [ ] **T5.2**: `bench_cp.py` perf harness + NSys profile
  - Acceptance: visual check on NSys shows P2P/compute concurrency
- [ ] **T5.3**: Hit perf gate (cp=4 step time ≤ 1.5× single-GPU per-token)
  - Acceptance: `bench_cp.py` numbers meet gate

### ✅ Checkpoint D — perf decision

- [ ] Owner sign-off

---

## Phase 5 — Module / training integration (Slice 6; post-v0, optional)

Re-plan when this phase starts. Sketch:

- [ ] **T6.1**: `HSTUConfig.cp_size` + Megatron parallel-state CP group
- [ ] **T6.2**: `HSTUAttention` module accepts `cp_group`, routes to CP fn
- [ ] **T6.3**: Dataloader DualChunkSwap shuffle
- [ ] **T6.4**: E2E training smoke test

---

## Day-1 next action

Phase 0 in order: T0.1 (reference test harness) → T0.2 (reference
benchmark + commit `bench_baseline.json`) → T0.3 (one-shot regression
command). All three must pass before Checkpoint 0. Owner sign-off at
Checkpoint 0, then T2.0 (PoC cleanup) → Phase 1 → Checkpoint A.
