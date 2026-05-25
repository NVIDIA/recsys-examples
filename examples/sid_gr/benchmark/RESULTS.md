# Benchmark: `generate()` vs `generate_beam_decode()`

**Scope: end-to-end SID-GR generation latency.** Both timed paths run
the full transformer stack (8 layers × { attention + MLP + LayerNorm })
plus the LM head and beam-search bookkeeping — not just the attention
kernel. The `beam_decode_attn` kernel itself (correctness sweep,
per-call kernel latency) lives in the upstream repo
`gitlab-master.nvidia.com:cjerry/gr-decode_atten` (`tests/test_fwd.py`,
`tests/benchmark.py`); we vendor a snapshot at
`corelib/gr_decode_atten/`. The numbers below are the wallclock a real
inference caller sees, including all Python orchestration, embedding
lookup, and per-step KJT overhead.

Hardware: **NVIDIA H100 NVL** (94 GB). Container: recsys-examples
Docker (bf16). Branch: post-merge of PR #379.

## What the two paths do

- **`generate()`** (baseline): at every hierarchy step, re-runs the
  full transformer over `[history + already-generated SIDs]`. The
  effective batch grows to `B × beam_width` after the first step;
  attention re-attends over the full sequence each step (O(seqlen²)
  per layer).
- **`generate_beam_decode()`** (optimized): one prefill over
  `[history + BOS]` populates a per-layer context KV cache, then
  `num_hierarchies - 1` decode steps each process only `beam_width`
  new tokens through the transformer, using the `beam_decode_attn`
  kernel to reuse the context KV cache and track per-beam ancestry via
  `topk_indices`.

The savings come from (a) MLP / projections only run on `beam_width`
new tokens per step instead of on the full prefix × effective batch,
and (b) attention complexity drops from O(seqlen²) to O(seqlen × W).

## Speedup

Fixed across the grid: `hidden=1024`, `num_heads=8`, `kv_channels=128`
(head_dim), `num_layers=8`, `num_hierarchies=4`, `codebook_size=256`,
`beam_width=200`, `bf16`. Median of 20 iterations after 5 warmup;
`cuda.synchronize()` before/after each iteration. All configs PASS
top-K beam set overlap ≥ 70% between the two paths.

`generate_beam_decode()` / `generate()` speedup:

| Batch ↓  hist → | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|
|  1 | 1.08× | 1.07× | 1.04× |  1.41× |
|  4 | 1.25× | 1.59× | 1.82× |  7.68× |
|  8 | 2.37× | 3.14× | 4.82× | 10.55× |
| 16 | 5.31× | 8.69× | 13.45× | **28.30×** |

Absolute `generate()` latency (ms) for context — `generate_beam_decode`
runs in 20-270 ms across the same grid:

| Batch ↓  hist → | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|
|  1 |  22.22 |  21.49 |   21.54 |    45.42 |
|  4 |  26.39 |  42.08 |   66.72 |   477.74 |
|  8 |  55.63 |  95.23 |  234.54 |  1183.50 |
| 16 | 154.58 | 388.56 | 1256.90 | **7579.60** |

## Summary

Speedup grows monotonically along both dimensions. The bottom-right
corner (`B=16, hist=2048, beam_w=200`) is the realistic offline
candidate-generation regime: `generate()` takes **7.58 s**,
`generate_beam_decode()` takes **268 ms** — a 28× e2e wallclock cut
that turns "unusable for online retrieval" into "usable as part of a
serving pipeline."

The top-left corner (`B=1, hist≤1024, beam_w=200`) is essentially
flat (~1.05×). At single-user scale the per-step Python orchestration
(KJT construction, embedding lookup, layer-stack launch overhead)
dominates wallclock, so saving the prefix recomputation work has
nowhere to land. The optimization is targeted at batched offline /
warm-pool inference, not single-request online serving.

## How to reproduce

```bash
cd examples/sid_gr
torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --sweep \
  --batch_size 16 --num_hierarchies 4 --num_layers 8 \
  --hidden_size 1024 --num_heads 8 --kv_channels 128 \
  --sweep_hist 256,512,1024,2048 \
  --sweep_beam 200 --sweep_dtype bf16
```

Vary `--batch_size` to fill in the other rows of the grid. The
Dockerfile adds `corelib/gr_decode_atten/` to `PYTHONPATH`, so no
extra setup is needed inside the container.

## Jagged-native context K/V (`use_jagged_kv=True`)

`generate_beam_decode` exposes a `use_jagged_kv` flag that feeds the
prefill K/V as a flattened `[total_tokens, H, D]` stream + `cu_seqlens_k`
instead of dense `[B, Sk_max, H, D]` + `seqused_k`. Measured at the
same config (`B=16, beam_w=200, bf16`):

| hist | dense (B) | jagged (C) | dense / jagged |
|---:|---:|---:|---:|
|  256 |  29.4 ms |  34.5 ms | **0.85×** |
| 1024 |  87.5 ms | 268.8 ms | **0.33×** |
| 2048 | 262.5 ms |  1.60 s  | **0.16×** |

Dense is decisively faster — by 17% at short history and **6× at hist=2048**.
The prefill `arbitrary=True` path in FA pays an O(N²) block-sparsity
setup cost that scales much worse than the `causal=True` fast path,
and that cost dominates as history grows. `use_jagged_kv=False` (the
default) is the right choice on every measured shape.

## Correctness verification

Three layers, weakest to strongest:

1. **Kernel reference oracle** (upstream `cjerry/gr-decode_atten`,
   `tests/test_fwd.py` — 14 quick cases via `make tt`, 1200
   parametrized cases via `make vt`): per-call kernel output compared
   against a fp32 PyTorch reference. This is the mathematical
   equivalence check for the attention kernel itself.
2. **Mask isolation unit tests** (`TestBeamIsolationMask` in
   `tests/test_beam_decode_generate.py`): direct geometry check on
   `padded_target_aware_causal_mask`.
3. **End-to-end regression guard** (this benchmark + the
   `test_generate_vs_generate_beam_decode_regression_guard` unit test):
   asserts top-K beam SID set overlap ≥ 70% between the two paths.
   bf16 noise plus beam-search topk tie-breaking make bit-exact
   equivalence impossible; the overlap metric stays bounded in [0, 1]
   regardless of scale, so the threshold remains meaningful as
   workloads grow.

## Known issues

- **Split-KV + `seqused_k`** hangs the K1 context-attention launch on
  SM90 (observed once during kernel development). The vendored kernel
  forces `num_splits=1` when `seqused_k` is set; the workaround costs
  a few percent on small-batch shapes but avoids the hang.
- **`use_jagged_kv=True`** is kept as a developer-facing flag for
  experiments but is not recommended for production use given the
  numbers above.
