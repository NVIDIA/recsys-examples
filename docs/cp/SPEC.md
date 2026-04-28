# HSTU Context Parallelism — v0 SPEC

**Branch**: `junzhang/hstu_cp` (worktree `recsys-hstu_cp/`)
**Companion**: `docs/cp/hstu_cp_design.md` (research / rationale)
**Status**: Draft — pending owner sign-off
**Author**: junzhang
**Date**: 2026-04-27

This SPEC is the **executable contract** for v0. The companion design doc
captures research and tradeoffs; this file captures *what we are committing to
build, in what order, and how we know each slice is done*.

Guiding principle (owner-stated): **start small, scale fast; modules independent,
controllable, testable; top-down; kernel optimisation last.**

---

## 1. Objective

### The problem
Today, one HSTU attention call runs entirely on one GPU. As recsys user-history
sequences get longer (32K, 64K, 100K+ tokens), that single GPU runs out of
HBM or attention compute time. We already have other parallel knobs in the
repo:

- **DP** (data parallel): replicate the model, split *batches* across GPUs.
  Doesn't help — each GPU still holds the full sequence.
- **TP** (tensor parallel): split the *attention heads* across GPUs.
  Doesn't help — each GPU still holds the full sequence per head.
- **SP** (sequence parallel for non-attention layers like LayerNorm/MLP).
  Doesn't help inside attention — by design SP gathers the full sequence
  before attention runs.

None of these split the sequence dimension *inside* the attention call.
That's what **Context Parallelism (CP)** does.

### What v0 ships
A new Python function — call it `hstu_attn_varlen_cp_func` — that the user can
swap in for the existing `hstu_attn_varlen_func`. Same input tensors
(`q, k, v, cu_seqlens_*`, etc.), plus four new arguments describing the CP
process group. Each GPU in that process group holds only `1/cp_size` of the
tokens; the function internally coordinates them so the **output of the
multi-GPU call is numerically equal (within bf16 tolerance) to the
single-GPU call.**

### What v0 does NOT ship
- Performance tuning. v0 chooses correctness > speed: a single CUDA stream,
  no comm/compute overlap. The multi-GPU run will likely be **slower per
  token** than single-GPU at the same shape — that's intentional, perf is
  Slice 5.
- Training-loop integration. v0 does not change the HSTU module, the
  config, or the data loader. The user wires CP into their own training
  script for now. Full integration is Slice 6 (post-v0).
- HSTU's "fancy" mask features: relative attention bias (`rab`),
  context-prefix tokens, target groups, and sliding-causal masks
  (`window_size != (-1, 0)`). v0 supports **pure causal only**; the
  wrapper rejects the rest with a clear error. Sliding-causal lands in
  v0.5 once we have per-tile window remapping designed.

### Who this is for
HSTU users whose sequence length already saturates one GPU (HBM OOM, or
training-step time dominated by attention) and who can afford to spend
extra GPUs to extend sequence length further. Not for users whose
bottleneck is batch size — DP still wins for that.

---

## 2. v0 scope (locked)

### What v0 supports

| Feature | What this means |
| -- | -- |
| **Pure causal attention** | Each Q at position `i` attends to K/V at positions `0..i` within its own sample. Kernel arg `window_size=(-1, 0)`. **Only this mask flavour.** |
| **Variable-length packed input** | Same input format HSTU already uses: `q, k, v` are flat tensors of shape `(total_tokens, num_heads, head_dim)`; per-sample boundaries given by `cu_seqlens_q/k`. |
| **CUTLASS kernel only** | We call the existing fused HSTU CUDA kernel (under `corelib/hstu/csrc/hstu_attn/`) as-is. No new C++/CUDA code. The PyTorch reference and the Triton kernel stay where they are; we don't touch them. |
| **`head_dim ∈ {32, 64, 128, 256}`** | The full set the CUTLASS kernel supports today (`hstu_api.cpp:391`). The CP wrapper guards on this set; `head_dim ∈ {32, 64, 128}` is the test-matrix focus, with at least one cell at `head_dim=256` to exercise the upper bound. |
| **Forward + backward correctness** | At `cp_size ∈ {2, 4, 8}`, output matches single-GPU baseline at `rtol=atol=2e-2` (bf16 fwd tolerance). Gradients (`q.grad / k.grad / v.grad`) match at `rtol=atol=5e-2` — looser to match the existing in-tree `assert_hstu_close` convention (multiplier 5 for bwd vs 2 for fwd; see `examples/commons/utils/hstu_assert_close.py`). |
| **Reduction in fp32** | Partial outputs across ring steps accumulate in fp32, then cast back to the input dtype on return. This matches the validated PoC and avoids bf16 add-error pile-up. |

### What v0 does NOT support

| Excluded | Why excluded |
| -- | -- |
| **`rab` / `has_drab=True`** (relative attention bias term added inside `QK^T`) | The bias values would need to be sliced and shipped across ranks alongside K/V — its own chunk of design work. Defer. |
| **HSTU "heterogeneous mask"**: `num_contexts != None`, `num_targets != None`, `target_group_size > 1` | The CUTLASS kernel supports these, but they break the balanced load sharding we use (TE-style DualChunkSwap, see §3). Defer to a follow-on that uses MagiAttention-style chunk dispatch. v0 wrapper raises `ValueError` if the user passes any of these. |
| **Sliding-causal `window_size=(w, 0)` with `w > 0`** | After DualChunkSwap shuffles tokens, "local" Q/K positions are no longer contiguous in global time. A naive per-tile `window_size` argument applied to local positions does **not** reproduce the global window. Correct sliding-causal under DualChunkSwap requires either per-tile mask remapping or per-tile in-window/out-of-window classification — non-trivial design work. v0 wrapper rejects `window_size != (-1, 0)`. Sliding-causal lands in v0.5 alongside whatever sliding-window perf optimisations need it. |
| **FP8 communication** | We ship K/V as bf16/fp16 over the ring. FP8 ring transport is a perf win; not v0. |
| **Other CP communication patterns**: Ulysses (head-dim all-to-all), hierarchical (intra/inter-node split), MagiAttention's GroupCast | We use one pattern only: ring P2P. The other patterns are reasonable add-ons later. |
| **Inference-time `delta_q`** (KV cache: only recompute the last few Q positions) | A separate code path even single-GPU. Punt entirely. |
| **Comm / compute overlap** (running the next P2P send on a separate CUDA stream while the current attention kernel runs) | Slice 5 only. v0 ring loop is fully sequential: wait for KV → run kernel → send KV → wait again. |
| **Module / training integration** (changing `examples/hstu/modules/hstu_attention.py`, `HSTUConfig`, dataloader) | Slice 6 (post-v0). |
| **Performance gate** | v0 ships once correctness passes. No "must be ≥ X TFLOPS" requirement until Slice 5. |

### Quick glossary (terms used elsewhere in this SPEC)

- **DualChunkSwap**: a load-balancing trick from NVIDIA TransformerEngine.
  Each sequence is split into `2*cp_size` equal-size pieces; rank `r` is
  assigned pieces `{r, 2*cp_size-1-r}`. This makes every rank do the same
  amount of causal-attention work. We adopt this verbatim.
- **Ring P2P**: each rank sends its local K/V to its right neighbour and
  receives K/V from its left neighbour, via NCCL `isend/irecv`. After
  `cp_size` such hops, every rank has seen every K/V chunk.
- **Three-region tile classification**: at each ring step, the local Q
  meets a particular K/V chunk. Depending on whether that chunk is from
  the past, the future, or self, the kernel call slices Q or K differently
  and uses either causal or no mask. Detail in §3 of `hstu_cp_design.md`.
- **`cu_seqlens`**: cumulative-sum offsets that mark per-sample boundaries
  inside the packed `(total_tokens, ...)` tensor. Standard HSTU input.
- **`scaling_seqlen`**: a scalar HSTU divides the attention output by.
  Always the *global* (unsharded) max sequence length in CP — never a
  per-rank value.

---

## 3. Slice plan

Each slice is independent, self-validating, and lands as a separate PR. Later
slices may *build on* earlier ones but must not require unmerged changes.

### Slice 1 — Single-rank PoC, equal-len, cp_size=2 forward [DONE]

**Status**: Numerically validated. `examples/hstu/cp/poc_dualrank_sim.py` —
max |diff|=1.95e-3 against single-call baseline at bf16, batch=4 seqlen=64
H=2 D=32.

**Why it matters**: proves the math (mask understanding + DualChunkSwap +
three-region tile classification + plain-sum reduction + zero-padding of K
for the lower-triangle case) on the smallest non-trivial input.

### Slice 2 — Single-rank PoC, varlen + cp_size ∈ {2, 4, 8} forward

**Goal**: prove the design generalises beyond cp_size=2 and beyond equal-len.

**Deliverables**:
- Extend `poc_dualrank_sim.py` (or split into a sibling `poc_simrank_sim.py`)
  to:
  - Accept arbitrary cp_size, generating the full `(rank, step)`
    classification grid (diagonal / lower-triangle / upper-triangle) — not
    just the cp_size=2 special case.
  - Accept varlen batches (per-sample seqlens, each padded to `2*cp_size`).
  - Run multiple shape configurations as a matrix.

**Files touched**:
- `examples/hstu/cp/poc_dualrank_sim.py` (rename to `poc_simrank_sim.py`?
  decide during impl).

**Acceptance**:
- All cases in matrix below PASS at bf16 tolerance `(rtol=2e-2, atol=2e-2)`.
  All v0 mask is pure causal (`window_size=(-1, 0)`).
  | cp_size | seqlens                                | H | D   |
  | --      | --                                     | - | --  |
  | 2       | `[64,64,64,64]`                        | 2 | 32  |
  | 2       | `[16, 32, 48, 64]`                     | 2 | 32  |
  | 4       | `[8, 8, 8, 256]` *(padding-heavy)*     | 2 | 32  |
  | 4       | `[128, 256, 384, 512]`                 | 4 | 64  |
  | 8       | `[16]*7 + [1024]` *(padding-heavy)*    | 4 | 128 |
  | 8       | `[512, 1024, 1024, 2048]`              | 4 | 128 |

**Estimated effort**: 0.5 day.

### Slice 3 — Multi-GPU public API, sequential ring, forward only

**Goal**: ship a real multi-GPU CP forward callable users can adopt.

**Deliverables**:
- New module `corelib/hstu/hstu_attn/hstu_attn_cp.py` exporting
  `hstu_attn_varlen_cp_func(...)` — same kernel-side signature as
  `hstu_attn_varlen_func` plus `(cp_group, cp_global_ranks, cp_stream=None,
  cp_comm_type="p2p")`. (`cp_stream` is unused in this slice but reserved.)
- Inside the function: `autograd.Function` with **forward only** in this
  slice (backward raises `NotImplementedError`).
- Sequential ring loop: `cp_size` attention steps, each preceded by a blocking
  `dist.batch_isend_irecv` of the next KV. Single CUDA stream.
- Hard guards: each of the following raises `ValueError` —
  `rab is not None`, `has_drab=True`, `num_contexts is not None`,
  `num_targets is not None`, `target_group_size > 1`,
  `window_size != (-1, 0)`, `kv_cache is not None`,
  `page_offsets is not None`, `page_ids is not None`,
  `last_page_lens is not None`, `func is not None`, `quant_mode != -1`,
  `head_dim not in {32, 64, 128, 256}`. (Full list mirrored from
  plan.md T3.1.)
- Helper `get_batch_on_this_cp_rank_for_hstu(...)` in same file: given a
  global jagged batch (q/k/v/cu_seqlens) and a CP group, returns the local
  shard + per-tile metadata. Pure permutation; no comm.

**Files touched**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (new, ~400 lines).
- `corelib/hstu/hstu_attn/__init__.py` (export the new symbol).
- `examples/hstu/test/cp/test_cp_forward.py` (new pytest, multi-GPU).

**Acceptance**:
- Single-rank PoC (Slice 2) and `hstu_attn_varlen_cp_func` produce numerically
  matching output on all matrix cases.
- `torchrun --nproc-per-node=N python -m pytest examples/hstu/test/cp/`
  passes for `N ∈ {2, 4, 8}` against the single-GPU baseline at bf16
  tolerance.
- Hard-guard rejections produce clear `ValueError` messages.

**Estimated effort**: 2-3 days.

### Slice 4 — Backward

**Goal**: forward + backward both correct on multi-GPU.

**Deliverables**:
- Implement `autograd.Function.backward` in `hstu_attn_varlen_cp_func`.
- Reverse-direction P2P delivery: dQ accumulates locally; dK/dV produced
  where KV currently is must reach the owning rank by traversing the
  ring **backward**. v0 implements this as `cp_size - 1` independent
  `batch_isend_irecv` exchanges per backward call: at step `i`, rank
  sends partial dK/dV to `(rank-i) % cp_size` (the owner whose KV was
  used at forward step `i`) and receives the matching partials from
  `(rank+i) % cp_size`. This is functionally equivalent to a one-hop-
  per-step reverse ring with copy/add semantics; we pick direct
  point-to-point because it's simpler and lets every rank `add_` into a
  single fp32 accumulator with no `copy_`-vs-`add_` first-visit branch.
  (Slice 5 may revisit if perf data favours one-hop ring overlap.)
- Same per-tile (rank, step) classification as forward; same kernel calls
  but with the bwd entry (`hstu_attn_cuda.varlen_bwd`).
- For lower-triangle tiles: same K/V zero-padding trick as forward — bwd
  produces dK/dV for the padded positions (which are 0 because dSiLU(0)=0
  contribution path); these get scattered into the correct K-half slots and
  the zero-padded slots' grads are dropped at scatter time.

**Files touched**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (extend).
- `examples/hstu/test/cp/test_cp_backward.py` (new pytest).
- `examples/hstu/cp/poc_dualrank_sim.py` (extend with autograd backward path
  to keep the single-rank oracle parity).

**Acceptance**:
- `q.grad / k.grad / v.grad` from `hstu_attn_varlen_cp_func` numerically
  match the single-GPU baseline grads at bf16 tolerance, on the full Slice 2
  matrix.
- `torchrun --nproc-per-node=N python -m pytest examples/hstu/test/cp/test_cp_backward.py`
  passes for `N ∈ {2, 4, 8}`.

**Estimated effort**: 3-4 days.

### Slice 5 — Comm/compute overlap + perf

**Goal**: hide ring P2P latency under attention compute; meet perf gate.

**Deliverables**:
- Two CUDA streams: `current_stream` and `cp_stream`. Step `i` runs on
  `streams[i % 2]`.
- Double-buffered KV ring slot.
- `cp_stream` exposed in API; user can pass it in.
- Issue next P2P at top of step `i`, run FA on current KV, sync via
  `torch.cuda.Event` between corrections.
- (Optional) `NVTE_BATCH_MHA_P2P_COMM`-equivalent env knob to switch between
  bare isend/irecv and `batch_isend_irecv`.

**Files touched**:
- `corelib/hstu/hstu_attn/hstu_attn_cp.py` (refactor ring loop).
- `examples/hstu/cp/bench_cp.py` (new perf harness; not pytest).

**Acceptance**:
- Correctness still passes the Slice 4 matrix.
- On a single H100/A100 node, `cp_size=4` HSTU forward+backward step time
  ≤ 1.5 × the matched single-GPU baseline-per-token-throughput (i.e. CP
  overhead < 50 % at cp_size=4). Concrete shapes: H=8, D=128, batch=8,
  seqlen=8K (single-GPU baseline) vs 32K with cp_size=4.
- NSys profile shows the next-step P2P running concurrently with current-
  step kernel.

**Estimated effort**: 3-5 days.

### Slice 6 — Module / training integration (post-v0, optional)

**Goal**: drop-in CP for HSTU training without users hand-wiring CP.

**Deliverables** (only if/when needed):
- `examples/hstu/modules/hstu_attention.py` accepts `cp_group` and routes to
  `hstu_attn_varlen_cp_func` when set.
- `HSTUConfig.cp_size` (Megatron parallel-state-style).
- `examples/hstu/training/...` dataloader emits the DualChunkSwap-shuffled
  batch + `local_cu_seqlens` per rank.
- One e2e training smoke test under `examples/hstu/test/cp/`.

This slice is **explicitly post-v0** and may ship in a separate cycle.

---

## 4. Project structure

```
recsys-hstu_cp/
├── docs/cp/
│   ├── hstu_cp_design.md       # research / rationale
│   └── SPEC.md                 # this file
├── corelib/hstu/hstu_attn/
│   ├── hstu_attn_interface.py  # untouched (existing single-GPU wrapper)
│   ├── hstu_attn_cp.py         # NEW (Slice 3): public CP wrapper + helpers
│   └── __init__.py             # add export of CP symbol
├── examples/hstu/cp/
│   ├── poc_dualrank_sim.py     # Slice 1 + 2 (numerical oracle)
│   └── bench_cp.py             # Slice 5 (perf harness; non-pytest)
└── examples/hstu/test/cp/
    ├── test_cp_forward.py      # Slice 3 (torchrun pytest)
    └── test_cp_backward.py     # Slice 4 (torchrun pytest)
```

The PoC and the production wrapper live in **separate trees** (`examples/`
vs. `corelib/`) so the PoC can never be accidentally taken as a runtime
dependency. The PoC's role is *oracle*, not library.

---

## 5. Code style

Inherit from the rest of `recsys-examples`. Specifically:

- License headers as per existing files (`SPDX-License-Identifier: Apache-2.0`,
  copyright NVIDIA + original-author lines where applicable).
- Type hints on all public functions; `from __future__ import annotations`
  in new files.
- No new dependencies beyond `torch`, `torch.distributed`, the existing
  `hstu_attn_2_cuda` binding, and `pytest` for tests.
- Docstrings on public functions; one-line `# why` comments only when the
  *why* is non-obvious. No narrative block comments.
- File length budget: `hstu_attn_cp.py` ≤ 600 lines. If it grows past that,
  factor helpers (e.g. `_dual_chunk_swap.py`, `_ring_p2p.py`) — consciously,
  with reviewer agreement, not reflexively.

---

## 6. Testing strategy

### Per-slice gate

| Slice | Single-rank PoC | Multi-GPU pytest | Perf  | Owner sign-off |
| --    | --              | --               | --    | --             |
| 1     | ✓ DONE          | -                | -     | ✓              |
| 2     | ✓ matrix        | -                | -     | required       |
| 3     | (regression)    | ✓ N=2,4,8        | -     | required       |
| 4     | (regression)    | ✓ N=2,4,8        | -     | required       |
| 5     | (regression)    | ✓ N=2,4,8        | ✓ gate| required       |
| 6     | (regression)    | ✓ + e2e          | -     | required       |

### Numerical oracle hierarchy

For every kernel-level test, the oracle is a single-GPU call to
`hstu_attn_varlen_func` on the **un-shuffled** global batch with the same
mask spec. We never compare against the PT reference (`pytorch_hstu_mha`)
directly because of accumulated bf16 noise — the CUTLASS kernel itself is
the reference.

Tolerance: `rtol=2e-2, atol=2e-2` for bf16 (matches existing
`hstu_assert_close` conventions; see `examples/commons/utils/hstu_assert_close.py`).

### Multi-GPU test harness

`torchrun --standalone --nproc-per-node=N python -m pytest
examples/hstu/test/cp/test_cp_*.py`. Tests gate-skip if `WORLD_SIZE` doesn't
match the parametrised `cp_size`.

---

## 7. Boundaries

### Always do
- **Verify before claiming.** Every slice has a runnable test that must pass
  on a real GPU before declaring the slice done.
- **Hard-guard out-of-scope inputs.** v0 wrapper raises `ValueError`
  with a clear message for any of the 13 unsupported modes (full list
  in §3 Slice 3 / plan T3.1: `rab`, `has_drab`, `num_contexts`,
  `num_targets`, `target_group_size>1`, `window_size != (-1, 0)`,
  `kv_cache`, `page_offsets`, `page_ids`, `last_page_lens`, `func`,
  `quant_mode != -1`, `head_dim not in {32, 64, 128, 256}`).
- **Use the global `scaling_seqlen`** in every per-tile kernel call. The
  per-tile call's `max_seqlen_q` / `max_seqlen_k` reflect tile geometry but
  `scaling_seqlen` is always the unsharded value, otherwise the partial
  sums are normalised inconsistently.
- **DualChunkSwap padding**: per-sample seqlen must be divisible by
  `2*cp_size` before entering the CP path. The wrapper checks this and
  errors out — padding is the caller's job.

### Ask first
- Any deviation from the slice plan (re-ordering, merging, splitting).
- Adding a new dependency (DeepEP, NCCL custom build, NVTE binding, etc.).
- Changing the public API surface after Slice 3 lands.
- Touching files under `examples/hstu/training/` or `corelib/hstu/csrc/`
  (this means kernel C++ code) before Slice 5.
- Making `examples/hstu/cp/poc_dualrank_sim.py` a runtime dependency of
  anything (it's an oracle, not a library).

### Never do
- Materialise an `[B,N,N]` mask tensor anywhere on the CP path.
  (We use `window_size + cu_seqlens` exclusively.)
- Use the PT reference (`pytorch_hstu_mha`) inside the production wrapper.
  (Oracle-only.)
- Skip kernel correctness tests "because it looks right".
- Hand-edit the CUTLASS kernel sources to add CP knobs. v0 wraps the kernel
  as-is. Kernel changes are out of scope until at least Slice 5, and require
  an explicit owner decision.
- Commit a slice without re-running the prior slices' tests as regressions.

---

## 8. Acceptance criteria for v0

v0 is complete when **Slices 1–4 are merged** and:

1. Forward + backward of `hstu_attn_varlen_cp_func` numerically match the
   single-GPU baseline at bf16 tolerance for `cp_size ∈ {2, 4, 8}` across the
   shape matrix in Slice 2.
2. The hard-guard error messages are tested.
3. The `docs/cp/SPEC.md` and `docs/cp/hstu_cp_design.md` are updated with
   any decisions / deviations made during implementation.

Slice 5 (overlap + perf) and Slice 6 (training integration) are NOT v0
gates — they ship as v0.5 / v1.

---

## 9. Open items requiring owner decision

These are deferred from §6 of `hstu_cp_design.md` and are not blockers for
Slice 1-2 but **must be resolved before Slice 3**:

1. **Padding cost on real recsys batches.** Slice 3 will need to measure
   how much DualChunkSwap padding inflates token count on a representative
   workload. Decision point: if > 30 % overhead, escalate to MagiAttention-
   style chunk-dispatch (Track B in design doc) earlier than planned.
2. **`scaling_seqlen` global vs local.** Locked: always pass the unsharded
   global `max_seqlen` (see §7).
3. *(Stale — moved to v0.5: sliding-causal `window_size != (-1, 0)` is
   rejected by the v0 wrapper. Revisit when v0.5 designs per-tile
   in-window/out-of-window classification under DualChunkSwap.)*

---

## 10. References

- `docs/cp/hstu_cp_design.md` — research, TE/MagiAttention deep-dives,
  HSTU/DSPA computational diff, per-tile recipe rationale.
- `examples/hstu/cp/poc_dualrank_sim.py` — Slice 1 numerical oracle.
- TransformerEngine: `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`.
- MagiAttention: `magi_attention/api/magi_attn_interface.py` (Track B).
- HSTU: `corelib/hstu/hstu_attn/hstu_attn_interface.py`,
  `examples/hstu/ops/pt_ops/pt_hstu_attention.py`,
  `corelib/hstu/csrc/hstu_attn/hstu_api.cpp`.
