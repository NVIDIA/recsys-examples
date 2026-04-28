# Jagged HSTU Context Parallelism — Design Note

**Author**: junzhang
**Date**: 2026-04-27
**Branch**: `junzhang/hstu_cp` (worktree at `recsys-hstu_cp/`)
**Status**: Design + Slice 1 (single-rank PoC) done; production wrapper
not yet started. See `tasks/plan.md` and `tasks/todo.md` for ordered
slices and `examples/hstu/cp/poc_dualrank_sim.py` for the validated
numerical oracle.

## 0. Goal & v0 scope

Add Context Parallelism (CP) to jagged (THD / packed varlen) HSTU attention so that
ultra-long user-history sequences can be sharded along the sequence dimension across
GPUs, in addition to the DP/TP/SP that already work today.

The exercise is two-step:

1. Read **NVIDIA TransformerEngine (TE)** and **SandAI MagiAttention** to harvest
   prior art on (a) process-group layout, (b) work partitioning & load balancing,
   (c) communication scheme, (d) compute/comm overlap, (e) backward.
2. Diff that against HSTU's actual math + jagged input layout, and decide what we
   reuse, what we change, and what's net-new for HSTU.

### v0 scope (locked)

* **Kernel**: the **CUTLASS** fused HSTU kernel under
  `corelib/hstu/csrc/hstu_attn/`, called via
  `corelib/hstu/hstu_attn/hstu_attn_interface.py::hstu_attn_varlen_func`.
  We do **not** use the PT reference (`pt_hstu_attention.py`) or Triton
  paths in the production CP path or in oracles — the **single-GPU
  CUTLASS call** itself is the numerical oracle (per SPEC §6 / plan
  Phase 0). PT and Triton stay as exploratory debug aids only.
* **No mask tensor.** The CUTLASS kernel constructs the mask on-the-fly from
  `(window_size_left, window_size_right, cu_seqlens_q, cu_seqlens_k)`. We do not
  materialise a `[B,N,N]` `valid_attn_mask`, do not slice / replicate one across
  CP ranks, and do not need any host-side mask logic.
* **Mask family**: pure causal only (`window_size=(-1, 0)`). Sliding-
  causal is out of v0 scope — under DualChunkSwap the local-position
  window does not map back to the global window without per-tile
  remapping; deferred to v0.5. See SPEC §2.
* **Out of v0 scope**, deferred to a follow-up:
  * `num_contexts != None` / `num_targets != None` / `target_group_size > 1`
    (heterogeneous per-row masks; chunks could split a target group, breaking
    DualChunkSwap balance).
  * `rab` / `has_drab=True` (relative attention bias would need its own
    chunk-wise slice + ring; orthogonal complication).
  * FP8 ring transport, Ulysses (`a2a`), hierarchical (`a2a+p2p`).

---

## 1. Reference design A — TransformerEngine CP (Megatron-style ring)

### 1.1 Where it lives

* `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`
  (~4960 lines). Single function entry: `attn_forward_func_with_cp(...)` (L4535)
  routes to one of four CP variants, selected by `cp_comm_type`:

  | `cp_comm_type` | Class                              | Communication                     |
  | --             | --                                 | --                                |
  | `"p2p"`        | `AttnFuncWithCPAndKVP2P` (L1363)   | Ring P2P send/recv of KV          |
  | `"all_gather"` | `AttnFuncWithCPAndKVAllGather`     | Llama-3 style: AG of KV           |
  | `"a2a"`        | `AttnFuncWithCPAndQKVOA2A`         | DeepSpeed-Ulysses (heads ↔ seq A2A)|
  | `"a2a+p2p"`    | hierarchical                        | A2A intra-node + P2P inter-node   |

  The `"p2p"` path is the most general and the most relevant to us; bullets below
  describe that.

### 1.2 Process group / sharding model

* CP gets its **own ProcessGroup** orthogonal to TP/DP/PP/SP. Caller passes:
  `cp_group`, `cp_global_ranks` (absolute world ranks; needed for NCCL P2P),
  `cp_stream` (a dedicated CUDA stream for comm/compute ping-pong),
  and `cp_comm_type`.
* CP shards the **sequence dim only** (never batch). For `bshd`/`sbhd` the seq
  axis is divided; for packed `thd` the `t` axis is divided.
* Inside the kernel each rank starts holding `s/cp_size` tokens of Q, K, V locally
  (`max_seqlen_q //= cp_size`, L1456).
* Heads stay untouched in the `p2p` path → CP is composable with TP-on-heads.
  The hierarchical `"a2a+p2p"` mode takes `cp_group=[a2a_group, p2p_group]` so
  two-level CP maps cleanly onto NVLink-island + IB topologies.

### 1.3 Load balancing — "DualChunkSwap" (a.k.a. zigzag / striped)

The killer trick. Naïve contiguous chunking is terrible under causal: rank N-1
holds the last chunk and attends to all earlier KV; rank 0 attends only to its
own short prefix. TE fixes this:

* Each sequence is split into `2*cp_size` equal chunks. Rank `r` is given **two**
  non-contiguous chunks: chunk `r` and chunk `2*cp_size - 1 - r`.
* For `cp_size=4`: rank 0={0,7}, rank 1={1,6}, rank 2={2,5}, rank 3={3,4}.
* Every rank covers the same total causal-triangle area. Perfect balance.

```python
# context_parallel.py:213-227
def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
    for rank in range(cp_size):
        chunk_ids[rank]            = 2 * rank
        chunk_ids[rank + cp_size]  = 2 * cp_size - 2*rank - 1
```

The dataloader pre-shuffles tokens into this order via Megatron's
`get_batch_on_this_cp_rank` so each rank's flat input already equals
`[first_segment, second_segment]`. Each sequence must be padded to be divisible
by `2*cp_size` (`cu_seqlens_padded`).

The local Q view inside the kernel is therefore `[b, 2, s/(2*cp_size), h, d]` —
explicitly two halves so the causal kernel can slice them.

### 1.4 Three-region tile classification

For each (rank, step) pair, the local Q meets a KV chunk that originated `i` hops
upstream. With DualChunkSwap + causal, the `(rank, step)` grid splits into three
regimes the kernel handles differently:

* `i == 0`  — **diagonal**: full causal sub-mask (Q attends own KV).
* `i ≤ rank` — **lower-triangle**: full Q × half KV, Q is fully later than KV; KV's
  *second half* is dropped, no mask needed in the sub-tile.
* `i > rank` — **upper-triangle**: half Q × full KV, only Q's *second half* is
  later than KV; Q's first half skipped.

```python
# context_parallel.py:780-826 (bshd causal)
elif section == "lower-triangle":
    q_part = q_part.view(q_part.shape[0], -1, *q_part.shape[-2:])  # full sq
    k_part = k_part[:, 0, ...]                                     # half sk
    v_part = v_part[:, 0, ...]
elif section == "upper-triangle":
    q_part = q_part[:, 1, ...]                                     # half sq
    k_part = k_part.view(k_part.shape[0], -1, *k_part.shape[-2:])  # full sk
    v_part = v_part.view(v_part.shape[0], -1, *v_part.shape[-2:])
```

After slicing, each tile becomes a **non-causal** rectangular attention with
halved dims — the union of all tiles exactly reconstructs the global causal
triangle.

### 1.5 Communication — ring P2P + online softmax

* **Ring direction (forward)**: send to `(rank+1)%cp_size`, recv from
  `(rank-1)%cp_size` (L1425).
* **Steps**: exactly `cp_size` attention iterations + 1 trailing correction
  (`for i in range(cp_size+1)` at L1704).
* **Primitive**: `flash_attn_p2p_communicate` wraps either bare
  `dist.isend/irecv` with even/odd parity ordering (deadlock-free) or
  `dist.batch_isend_irecv` (env `NVTE_BATCH_MHA_P2P_COMM=1`).
* **KV ring buffer**: a single flat tensor sized to one rank's KV. After step 0,
  `p2p_comm_buffers[i]` holds the KV that originated `i` hops upstream.
  Effectively double-buffered (`buffers[i % 2]`).
* **Online softmax merge**: each step's local FA returns
  `(out_per_step, lse_per_step, rng_state)`; the running LSE is corrected with the
  log-sum-exp merge (L161-170) and the running output rescaled (L129-141).
  A "second-half-only" merge variant (L144, L173) handles the upper-triangle case
  where only Q's second half got contributions.

### 1.6 Compute/comm overlap

* Two CUDA streams: `[current_stream, cp_stream]`. Step `i` runs on `streams[i%2]`
  — ping-pong.
* At top of step `i`, the rank issues the next P2P (`i → i+1`) on the current
  stream, then runs FA on the *current* KV; step `i+1` lands on the alternate
  stream and waits on `send_recv_reqs[(i+1)%2]`.
* `fwd_results_correction_done` event lets stream A's correction finish before
  stream B's correction begins.
* Comment in TE explains the second purpose: a single FA kernel doesn't fill the
  GPU → alternating two streams overlaps two kernels and recovers utilisation.

### 1.7 Backward

* Reverse the ring: send to `(rank-1)`, recv from `(rank+1)` (L2220-2221).
* `rng_state` consumed in reverse → dropout reproduced exactly.
* `dQ` accumulates locally (Q owner). `dK/dV` are produced on whichever rank
  currently holds KV → ride the reverse ring back; receiving rank does `copy_`
  on first visit, `add_` after.
* Same (rank, step) → tile classification used; per-seq half-aware adds for THD.

### 1.8 Variable-length / packed (THD) support

THD is first-class. Three THD-specific CUDA kernels in
`transformer_engine/common/fused_attn/context_parallel.cu`:

* `thd_read_half_tensor` — per-seq first/second half slicing of Q or KV.
* `thd_out_correction` / `thd_second_half_lse_correction` — per-seq online
  softmax merge.
* `thd_grad_correction` — per-seq dK/dV add/copy.

Plus dataloader helpers: `pad_thd_sequences_for_cp`, `generate_positional_ids_for_cp`,
`get_batch_on_this_cp_rank`. Sliding-window mask is **not supported** under
`p2p`, only under `a2a` / `all_gather`.

### 1.9 Public API

```python
DotProductAttention(
    ...,
    cp_group: ProcessGroup | List[ProcessGroup] = None,
    cp_global_ranks: List[int] = None,
    cp_stream: torch.cuda.Stream = None,
    cp_comm_type: str = "p2p",   # | "all_gather" | "a2a" | "a2a+p2p"
)
# functional core also exposed: attn_forward_func_with_cp(...)
```

### 1.10 What we'd inherit from TE

1. The four-knob public API shape `(cp_group, cp_global_ranks, cp_stream, cp_comm_type)`.
2. DualChunkSwap dataloader contract (pad to `2*cp_size`, give each rank chunks
   `{r, 2*cp_size-1-r}`).
3. Three-region tile classification under causal.
4. Ring P2P with even/odd parity and two-stream double-buffering.
5. Reverse-direction backward + RNG replay structure.
6. Per-seq THD-aware kernels for slicing / accumulation / grad-merge as a
   *concept* — the actual reduction math is different for HSTU (see §3).

---

## 2. Reference design B — MagiAttention CP (general dispatch + GroupCast)

### 2.1 Positioning

MagiAttention is built for ultra-long-context, **heterogeneous-mask** training
(video diffusion / Magi-1). Its README claims four novelties over ring-attention
/ TE-CP / Megatron-CP:

1. **Flexible mask is a first-class object.** A global mask is a list of
   `AttnSlice = (q_range, k_range, mask_type)` items with closed-form
   `area()`. Causal, full, sliding-window, document, block-sparse, "bicausal"
   are all just different `AttnSlice` flavours.
2. **Computation load balancing via a dispatch solver.** A global LPT-style
   solver assigns chunks to ranks to equalise *attention area*, not token count.
3. **Zero-redundant communication** via two new primitives **GroupCast** and
   **GroupReduce** (built on `all_to_all_v` + optional DeepEP), replacing ring
   P2P. A KV chunk is sent only to the ranks that actually need it under the
   current mask.
4. **Adaptive multi-stage overlap** — `overlap_degree` independent async
   GroupCasts run while a persistent FFA kernel (with reserved `sm_margin`)
   computes.

### 2.2 Process group / sharding

* `cp_group_or_mesh` is either a single `ProcessGroup` or a 2-D `DeviceMesh`
  `[inter_node, intra_node]`. With `MAGI_ATTN_HIERARCHICAL_COMM` on, GroupCast
  lowers to intra-node a2av + inter-node a2av.
* No Ulysses heads-A2A in the data path — A2A appears only inside GroupCast's
  *implementation*.
* In self-attn, Q and KV share the same dispatch meta (`partitions`,
  `chunk_size`, perm idxs).

### 2.3 Dispatch — chunk tokens, not split sequences

```
global tokens
  → split into `num_chunks` chunks of size `chunk_size`
  → compute area(chunk) under the mask (sum of slice areas)
  → LPT solver: bin-pack chunks into `cp_size` buckets by area, with a
    ceil(num_chunks / cp_size) cap
  → partitions[rank] = [chunk_idx, …]
  → each rank gathers its chunks → local shard = concat(chunks)
```

Dispatch is **static, computed once per `DistAttnRuntimeKey`** and cached. Ten
solvers ship (greedy LPT default, DP, BS, BTP, Topp-heap, …); some account for
*affinity* between chunks (chunks whose k-ranges intersect → same rank to reduce
later GroupCast volume).

```python
# meta/container/slice.py:36-66 — the four area() formulas the solver consumes
if mask_type == FULL:        area = q_seqlen * k_seqlen
elif mask_type in (CAUSAL, INVCAUSAL):
    area = triangle_or_trapezoid(q_seqlen, k_seqlen)
elif mask_type == BICAUSAL:  area = parallelogram(q_seqlen, k_seqlen)
```

### 2.4 Varlen / packed input

`magi_attn_varlen_key(cu_seqlens_q, cu_seqlens_k, ..., causal, window_size)` and
`magi_attn_flex_key(q_ranges, k_ranges, attn_mask_type, ...)` are the two entry
points. The varlen API just rewrites cu_seqlens to `AttnRanges`. THD is preserved
end-to-end; padding is handled internally.

### 2.5 Communication — GroupCast / GroupReduce

* **GroupCast**: many-to-many scatter. Each input split goes to listed
  `dst_indices`; each output split is sourced from `src_index[i]`.
* **GroupReduce**: dual; sums inbound contributions into the destination
  buffer. Used for partial-out/lse merge (fwd) and partial-d{Q,K,V} merge (bwd).
* Three backends: `all_to_all_v` based, hierarchical 2D-mesh, native DeepEP.

Why this matters: ring sends every KV shard to every rank. GroupCast sends a
chunk only to the ranks that need it (computed once from partition + mask).
For sparse cross-rank dependencies (block-sparse, SWA, multi-target HSTU
groups) this strictly reduces volume.

### 2.6 Overlap

`overlap_degree` independent async GroupCasts in flight, gated by
`KernelBarrier` events keeping DeepEP comm strictly behind the corresponding
compute kernel. With `prefetch_stage_by_stage=False` (FFA default), all stages'
fetches are issued up front and the persistent FFA kernel (reserved
`sm_margin`) overlaps with all of them. The overlap *schedule* is solved
offline by `OverlapSolver`, cached alongside dispatch.

### 2.7 Backward

Mirror of forward: GroupCast remote KV (and remote q/o/do/lse if `enable_qo_comm`),
FFA backward → partial dQ/dK/dV → GroupReduce into local accumulators.
Same dispatch meta reused.

### 2.8 Public API (5-step pattern)

```python
key = magi_attn_flex_key(q_ranges, k_ranges, attn_mask_type=…,
                        total_seqlen_q=…, total_seqlen_k=…,
                        num_heads_q=…, num_heads_kv=…, head_dim=…,
                        cp_group_or_mesh=…, dist_attn_config=…)
local_x      = dispatch(total_x, key)            # 1. shard
local_q,k,v  = q_proj(local_x), k_proj(local_x), v_proj(local_x)
local_out, _ = calc_attn(local_q, local_k, local_v, key)   # 2. dist attn
total_out    = undispatch(local_out, key)        # 3. (optional) gather
```

### 2.9 What we'd inherit from MagiAttention

1. **Make the mask a first-class object** — `AttnSlice = (q_range, k_range,
   mask_type)`. For HSTU we extend with `hstu_mask_type ∈ {causal,
   context_then_causal, target_group, sliding}` and define `area()` per kind.
2. **Chunk first, dispatch second.** Decouples chunk granularity from cp_size
   and from sequence boundaries — exactly what packed THD HSTU needs.
3. **LPT on per-chunk area.** Cheap, deterministic, mask-agnostic.
4. **Cache the dispatch.** Per-step training pays only GPU work; CPU planning
   amortises across iterations with the same shape signature.
5. **Replace ring with GroupCast/GroupReduce** when per-rank dependency is
   sparse (HSTU target-groups produce exactly this pattern).
6. **Two kernel-barrier persistent-kernel overlap** — short to port; pairs
   well with our existing fused HSTU kernel if we expose an `sm_margin`.

---

## 3. HSTU vs DSPA — what's actually different

### 3.1 Math

**HSTU** (`examples/hstu/ops/pt_ops/pt_hstu_attention.py:176-191`):

```python
qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha           # alpha = 1/sqrt(d)
qk_attn = F.silu(qk_attn) / scaling_seqlen                         # SiLU + fixed-N normalize
qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)                   # multiplicative 0/1 mask
attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)
```

So:

```
HSTU:  O = ( SiLU(α · Q K^T) ⊙ M ) · V / N
DSPA:  O =   softmax(α · Q K^T + mask_inf)         · V
```

The single most consequential difference for CP: **HSTU has no softmax → no
log-sum-exp → no online-softmax correction across ring steps.** The reduction
over the K dimension is a plain sum-of-products. Partial outputs from any
partition can be **simply added** to form the final output:

```
O_final = (1/N) · Σ_p [ ( SiLU(α · Q K_p^T) ⊙ M_p ) · V_p ]
```

This is a strict simplification over softmax-CP. No LSE, no rescale.

### 3.2 Mask structure

HSTU mask types (from `corelib/hstu/README.md:47-63`):

* **Causal** (`window_size=(-1, 0)`): `j ≤ i` within own sequence.
* **Context + causal**: contextual prefix can be attended by all later
  positions; remainder is causal.
* **Target groups**: within `target_group_size`-sized sub-groups of the tail,
  causal; across groups, no attention.
* **Delta-q**: only the last few Q positions are computed (KV-cache inference).

Mask is a `[B,N,N]` (or `[1,N,N]`) boolean tensor in the PT reference; in the
fused kernel it's expressed as `(window_size, num_contexts, num_targets,
target_group_size)` per sample. **Multiplicative**, not additive — sets invalid
entries to 0 *after* SiLU. No `-inf` games.

### 3.3 Layout

Jagged THD throughout. Public CUDA-kernel signature
(`corelib/hstu/hstu_attn/hstu_attn_interface.py:209-233`):

```python
hstu_attn_varlen_func(
    q, k, v,                   # (total_q, nheads, headdim) packed
    cu_seqlens_q, cu_seqlens_k,# (B+1,)
    max_seqlen_q, max_seqlen_k,
    num_contexts=None,         # (B,)
    num_targets=None,          # (B,)
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0, rab=None, has_drab=False, scaling_seqlen=-1,
)
```

The Python module wrapper (`examples/hstu/modules/hstu_attention.py:84-138`)
takes packed `tq, tk, tv ∈ (T, d)` plus `offsets ∈ (B+1,)` and the same
metadata.

### 3.4 Existing parallelism in repo

* **TP** is wired (`native_hstu_layer.py`): heads sharded via Megatron-core
  `TEColumnParallelLinear` / `TERowParallelLinear`.
* **SP** is partially tracked (`hstu_block.py:40-43` exposes a
  `sequence_parallel` flag on the post-processor).
* **No production CP wrapper exists yet** — only `examples/hstu/cp/poc_dualrank_sim.py`
  (Slice 1 single-rank PoC, numerical oracle). Generic distributed
  helpers in `examples/commons/.../collective_ops.py`
  (`split/gather_along_first_dim`, `gatherv_along_first_dim`) are
  TP/DP/SP utilities; no ring/all-to-all-v scheduling. **The CP wrapper
  in `corelib/hstu/hstu_attn/hstu_attn_cp.py` is what we add at Slice 3.**

### 3.5 DSPA sibling

The repo's SID-GR model uses Megatron-core `DotProductAttention` /
`TEDotProductAttention` (softmax). It is a **separate model family** — there is
no in-tree HSTU↔DSPA comparison harness. Practical implication: we cannot
borrow CP code from the SID-GR side because TE's softmax-CP path is structurally
about LSE accumulation, which HSTU doesn't have. The *control plane* (process
group, dataloader pre-shuffle, kernel dispatch) ports cleanly; the *data plane*
(reduction math) does not.

---

## 4. Implications for HSTU CP design

What HSTU's math/layout makes **easier** than softmax-CP:

* **No LSE merge.** Forward partial-output reduction is a plain sum. Backward
  is similarly free of softmax-grad rescales — `dQ, dK, dV` partials simply
  accumulate.
* **Multiplicative mask.** Tile-level "drop second half of K" is a slice op,
  not a mask edit; no `-inf` to manage.
* **Fixed normalizer (`1/scaling_seqlen`).** Hoist once at the end; never
  participates in cross-rank ops.

What it makes **harder**:

* **Jagged input is the *only* layout.** All slicing / accumulation / grad-merge
  ops must be cu_seqlens-aware. This is exactly the THD work TE has to do too,
  but for us it's the *default* not a special path.
* **Heterogeneous mask families.** `target_group_size > 1` and "context +
  causal" produce non-uniform per-row workloads. DualChunkSwap balances pure
  causal but is suboptimal for these. → MagiAttention's `AttnSlice + LPT solver`
  is a better fit when target groups are present.
* **`num_contexts` / `num_targets` are per-sample.** Any chunk striping must
  preserve the ability of any local Q position to know its own (context, target)
  membership. The dispatch metadata has to travel with the tokens — and the
  dataloader shuffle has to either rewrite per-sample offsets or carry an
  index-mapping tensor.
* **Delta-q inference.** Different ranks may hold zero Q to compute — we'll
  need a "no-op tile" path or skip mask in the per-rank schedule.

### 4.1 Proposed approach (two-track)

**Track A — "fast path" for pure causal HSTU (v0):**

* **Dataloader shuffle.** Reuse TE's DualChunkSwap contract verbatim. Pad each
  packed sequence to `2*cp_size`; rank-`r` gets chunks `{r, 2*cp_size-1-r}` of
  every sequence. Rebuild local `cu_seqlens_q` / `cu_seqlens_k` from the
  shuffled token layout (each local sequence has length
  `original_padded_len / cp_size`, two halves concatenated).
* **Per-rank Q view.** `[t_local, h, d]` where `t_local =
  total_tokens_padded / cp_size`, but conceptually two halves: the kernel
  invocation supplies the right cu_seqlens for whichever half is active per
  tile.
* **Per-tile kernel-call recipe.** For each ring step `i ∈ [0, cp_size)` and
  the local rank `r`, classify the tile and invoke the CUTLASS kernel
  (`hstu_attn_varlen_func` or its raw `varlen_fwd` C++ entry) with:

  | Region                | Q slice              | K/V slice            | `window_size`  | Why                                                           |
  | --                    | --                   | --                   | --             | --                                                            |
  | `i == 0` diagonal     | full local Q (both halves) | full local KV (both halves) | `(-1, 0)`     | Mini-causal on own tokens. Kernel applies causal natively.    |
  | `i ≤ r` lower-tri.    | full local Q (both halves) | first-half-only KV   | `(-1, -1)`    | All Q tokens are strictly later than all KV → no mask needed. |
  | `i > r` upper-tri.    | second-half-only Q   | full remote KV (both halves) | `(-1, -1)`    | Only Q's second half is later than KV → first half skipped.   |

  All three calls share `alpha`, `scaling_seqlen=GLOBAL_seqlen`,
  `num_contexts=None`, `num_targets=None`, `target_group_size=1`, `rab=None`,
  `has_drab=False`. Sliding-causal is **out of v0 scope** (SPEC §2):
  changing the diagonal-tile call to `(w, 0)` only handles the diagonal
  region, while the lower/upper-triangle tiles still need per-tile
  in-window/out-of-window classification under DualChunkSwap. Deferred
  to v0.5.
* **Output reduction = plain add.** No LSE. Each tile produces `O_partial`
  shaped like the relevant Q slice; the rank's running `O_local` is just
  `O_local[Q_slice] += O_partial`. After all `cp_size` steps,
  `O_local /= scaling_seqlen` is *implicit* — the kernel already divides by
  `scaling_seqlen` per call. So the per-call `scaling_seqlen` argument must
  remain the **global** `max_seqlen_q` (not the post-shard local one),
  otherwise the normalisation factor changes meaning across ranks.

  Wait, double-check: with the kernel doing the divide per call, partial sums
  add cleanly only if the divisor is identical across all calls. ✅ It is, as
  long as we always pass the global `scaling_seqlen`.
* **Reuse ring P2P + reverse-direction-backward skeleton from TE**.
  v0 uses a **single CUDA stream** with a sequential ring (per SPEC §2);
  the two-stream comm/compute overlap from TE is Slice 5 (post-v0 perf
  work). The reduction simplification (no LSE) collapses all of TE's
  `flash_attn_fwd_*_correction` / `flash_attn_fwd_second_half_*`
  functions to plain `add_` — which is the entire numerical-side
  simplification.
* **New code** (against TE's `AttnFuncWithCPAndKVP2P` template) — **v0
  uses Python/torch only; no new C++/CUDA per SPEC §2**:
  1. `thd_read_half_tensor`-equivalent: per-sequence first/second half
     slicer for packed THD, written in **pure torch** as a `torch.cat` of
     per-sample slices. Slower than a fused CUDA op but correct; perf
     optimisation (folding into a CUDA op) is a Slice 5 follow-up if
     profiling shows it matters.
  2. The three-call dispatcher (the table above), wrapping
     `hstu_attn_cuda.varlen_fwd` / `varlen_bwd`.
  3. Plain accumulation of `O_local` across steps (jagged tensor `add_`,
     accumulated in fp32 then cast back).
  4. `get_batch_on_this_cp_rank_for_hstu` *helper function* (pure
     permutation, no comm) — this is v0 (per plan T3.2). Wiring it INTO
     the training-loop dataloader is Slice 6 (post-v0); v0 callers
     invoke the helper themselves.
  5. Backward: reverse-ring of dKV, dQ stays local, kernel called with the
     same per-tile classification.

**Track B — "general path" for context + target_group HSTU (post-v0):**

The pieces are already in place on both sides; Track B is wiring, not new
algorithms.

* **HSTU side already handles arbitrary mask.** The CUTLASS kernel takes
  `(window_size_left, window_size_right, num_contexts, num_targets,
  target_group_size)` directly and constructs the per-row mask on-chip. So the
  kernel call surface for arbitrary HSTU masks is *unchanged* — we just need
  to ship the right metadata to each rank's local kernel.
* **MagiAttention side already has the abstraction.** `AttnSlice = (q_range,
  k_range, mask_type)` with closed-form `area()` per type is exactly what we
  need to (a) describe an HSTU sample's mask as a list of slices, (b) give the
  LPT dispatch solver a balanced-area objective, (c) compute `comm_meta` for
  GroupCast routing.

The mapping per HSTU sample becomes:

  * `[0, num_contexts)` × `[0, num_contexts)` `FULL`  — context-prefix block
    is fully attended by itself,
  * `[num_contexts, body_end)` × `[0, body_end)` `CAUSAL` — main body is
    causal over context+main,
  * `num_target_groups` × `(q_range_group_g, k_range_group_g, CAUSAL)` for the
    tail target groups (intra-group causal; inter-group is implicitly
    excluded).

Then: dispatch via LPT solver, cache per-shape, GroupCast/GroupReduce on top of
`all_to_all_v` for the comm primitive. Same plain-sum reduction as Track A
(still no LSE — HSTU math doesn't change).

We start with Track A (smaller diff, ships faster, covers pure causal
only — sliding-window is v0.5 per SPEC §2). Track B is opened only when
target-group benchmarks justify it; the design is shaped so Track B
reuses Track A's per-rank kernel-invocation recipe verbatim and only
swaps out the dataloader-shuffle + comm-primitive layers.

### 4.2 Python / Megatron-side plumbing (in addition to Track A's per-tile work)

The kernel-side new code is enumerated under Track A. The Python/wrapper side
needs:

1. **HSTU-CP public wrapper** in a new module
   `corelib/hstu/hstu_attn/hstu_attn_cp.py` (per SPEC §4 / plan T3.1; this
   keeps the existing `hstu_attn_interface.py` untouched and lets users
   import `hstu_attn_varlen_cp_func` from the package alongside the
   single-GPU symbol). Signature mirrors `hstu_attn_varlen_func` plus
   `(cp_group, cp_global_ranks, cp_stream, cp_comm_type)`. Internally an
   `autograd.Function` that drives the ring loop. **The full hard-guard
   list (13 items) is the canonical one in SPEC §3 Slice 3 / plan T3.1**;
   do not duplicate it here.
2. **Module-level wiring** *(Slice 6, post-v0 only — SPEC §1)*.
3. **Process-group plumbing** *(Slice 6, post-v0 only — SPEC §1)*.
4. **Dataloader-loop shuffle integration** *(Slice 6, post-v0 only — SPEC §1)*. v0 ships the pure helper function (T3.2 above); v0 callers wire it themselves.
5. **CUTLASS-kernel numerical oracle**: a single-GPU call to
   `hstu_attn_varlen_func` (the production CUTLASS kernel) on the
   un-shuffled global batch is the oracle for every CP correctness test.
   The PT and Triton references are not used inside production tests
   (only as exploratory debug aids for the kernel itself).

### 4.3 What we explicitly skip on the first cut

* **`rab` / `has_drab=True`.** Pass `rab=None` only. Bias would need its own
  per-tile slice + ring; orthogonal complication, not v0.
* **Heterogeneous mask (`num_contexts`, `num_targets`, `target_group_size>1`).**
  v0 enforces these are `None / None / 1` in the CP wrapper. The HSTU kernel
  already accepts these knobs, so when Track B opens we don't change the
  kernel — we change the dataloader shuffle and per-rank metadata plumbing
  (and the chunking strategy from DualChunkSwap → MagiAttention `AttnSlice`-
  driven dispatch).
* **Ulysses (`a2a`) mode.** HSTU's per-row work is small enough that the
  Ulysses overhead is not justified yet.
* **Hierarchical `a2a+p2p`.** Until we hit cross-island bandwidth issues,
  flat ring is fine.
* **FP8 ring transport.** Possible win (HSTU kernel already supports FP8) but
  separable from CP correctness.

---

## 5. Interaction with rest of recsys-examples

* **DynamicEmb** is upstream of attention and per-rank already; CP changes
  *after* the embedding lookup. Unaffected.
* **Pipeline / multi-stream pipeline** (work-in-progress on
  `junzhang/rework-mtms`): the CP attention call must compose with whatever
  stream and event handles the pipeline emits. Plan: use the pipeline's
  current stream as the "main" stream and create `cp_stream` as a sibling
  in the same priority class. Verify no event collision.
* **Watchdog**: we add a long-running NCCL P2P loop. Keep an eye on the
  symptoms tracked under `docs(perf)` series — if a rank stalls inside the ring
  loop, the watchdog must dump the ring step index, not just the kernel name.

---

## 6. Open questions

1. **Padding cost.** TE pads each sequence to `2*cp_size`. For HSTU traffic
   with very heavy-tailed sequence-length distributions, this can be wasteful.
   Quantify on a real recsys batch before committing.
2. **`rab` (relative attention bias) under CP.** `rab[i,j]` lives in the global
   (i,j) coordinate. After dispatch the local kernel tile sees only a subset.
   Decision: either (a) shard `rab` along the same chunking as Q/K and slice
   per tile (cheap if `rab` is dense, painful if `rab` is structured / Toeplitz),
   or (b) recompute `rab` from a positional encoding per tile. Need to read
   `rab` plumbing in the kernel before deciding.
3. **`scaling_seqlen` semantics.** Currently fed as `max_seqlen_q`. After CP
   the per-rank `max_seqlen_q` is `global / cp_size`. We must keep using the
   *global* value (otherwise the normalizer changes meaning). Easy — just
   plumb the unsharded value through the wrapper.
4. **Backward of `num_contexts` / `num_targets`.** These are integer tensors,
   no grad. But they index into the mask-construction path, which has to be
   re-entered identically in backward. Confirm no recompute-side issue.
5. **Track B scheduler choice.** If/when we go to MagiAttention-style dispatch,
   start with `MinHeapDispatchAlg` (LPT). DP/BS solvers are O(n^2)+ — only used
   for offline tuning.

---

## 7. Next steps

1. Confirm pure-causal-only as v0 scope (Track A) — locked, see SPEC §2.
2. Use the **single-GPU CUTLASS** `hstu_attn_varlen_func` call on the
   un-shuffled global batch as the numerical oracle for forward + backward
   correctness across the SPEC §3 Slice 2 matrix (per plan T0.1).
3. Implement the jagged half-slicer in **pure torch** (`torch.cat` of
   per-sample slices); CUDA fusion is a Slice 5 follow-up.
4. Wire sequential ring P2P (Slice 3 single-stream) → multi-GPU forward
   → backward → only then add two-stream + comm/compute overlap (Slice 5).
5. End-to-end correctness test on 2/4/8 GPUs with synthetic varlen batches.
6. Benchmark on realistic recsys distributions; revisit padding cost and
   target-group support to decide if Track B is justified.

---

## Appendix — file pointers

**TransformerEngine (read order):**

* `transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`
  — main orchestration, all four variants. Especially L213, L780, L1363, L1456,
  L1684, L1704, L4535, L4574, L4758, L4871.
* `transformer_engine/common/fused_attn/context_parallel.cu` — THD-aware CUDA
  ops (`thd_read_half_tensor`, `thd_*_correction`, `thd_grad_correction`).
* `tests/pytorch/attention/run_attention_with_cp.py` — cleanest e2e usage.
* `transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py`
  L351-401, L564-603 — public API plumbing.

**MagiAttention (read order):**

* `magi_attention/api/magi_attn_interface.py` — public API surface.
* `magi_attention/api/functools.py` — varlen → ranges + padding helpers.
* `magi_attention/meta/container/{slice,chunk,bucket}.py` — `AttnSlice`,
  `AttnChunk`, `AttnBucket` and `area()`.
* `magi_attention/meta/_make_dispatch_meta.py` — global bucket → solver →
  partitions.
* `magi_attention/meta/solver/dispatch_solver.py` — all 10 solvers.
* `magi_attention/functional/dispatch.py` — chunk-permute autograd.
* `magi_attention/functional/dist_attn.py` — multi-stage fwd/bwd pipeline
  (L141, L244, L1462, L2992, L3061-3157).
* `magi_attention/comm/primitive/grpcoll/_group_collective.py` and
  `_group_collective_hier.py` — GroupCast / GroupReduce.

**HSTU (this repo):**

* `corelib/hstu/README.md` — math + mask types reference.
* `corelib/hstu/hstu_attn/hstu_attn_interface.py:185-279` — kernel public API.
* `examples/hstu/ops/pt_ops/pt_hstu_attention.py` — PT reference math (L176-191).
* `examples/hstu/modules/hstu_attention.py` — module wrapper (L84-138, 253-314).
* `examples/hstu/ops/triton_ops/triton_hstu_attention.py` — triton kernel
  (3103 lines; only the entry signatures are relevant for CP wrapping).
* `examples/commons/.../collective_ops.py` — generic dist helpers (no CP today).
