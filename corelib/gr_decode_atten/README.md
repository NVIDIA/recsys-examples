# Beam Search Decode Attention

High-performance beam search decode attention kernels implemented in CuTe DSL, supporting SM8x (A100/L40/L20), SM90 (H100/H800/H20), SM100 (B200/B300), and SM120 (RTX PRO6000).

## Provenance

This directory is a vendored snapshot of the internal NVIDIA repository
`gitlab-master.nvidia.com:cjerry/gr-decode_atten`. That repository remains
the canonical source and the location for ongoing kernel development.

- **Source repo**: `ssh://git@gitlab-master.nvidia.com:12051/cjerry/gr-decode_atten.git`
- **Pinned commit**: `1c540f6` (upstream `master`, includes `seqused_k` and `cu_seqlens_k` support)
- **Initial import date**: 2026-05-13

### Sync policy

Updates land on an **on-demand** basis — pulled in only when this repository
needs a new feature or a bug fix that the upstream kernel has already
landed. There is no scheduled cadence.

To resync, copy the upstream tree at the target commit over this directory
(preserving the NVIDIA SPDX headers and this `Provenance` section), update
the **Pinned commit** line above, and use a single commit with this message
shape:

```
sync(gr_decode_atten): bump to upstream <sha>

Source: cjerry/gr-decode_atten@<sha>
Range:  <prev_sha>..<sha>
Reason: <feature or fix that triggered the sync>
```

Do not make in-place edits to this directory that are not also upstreamed
to `cjerry/gr-decode_atten`; otherwise the two will drift.

## Overview

During beam search decoding, KV cache splits into two parts with fundamentally different access patterns:

| | Context KV | Beam KV |
|---|---|---|
| Source | Prefill stage | Decode stage |
| Sharing | All beams share | Per-beam independent |
| Access | Dense, sequential | TopK gather, irregular |
| Length | Thousands of tokens | `decode_nums` tokens |

A single kernel cannot efficiently handle both. This project provides two execution paths:

### 3-Kernel Pipeline

Three independent kernels, each using the optimal compute strategy:

```
K1: Context Attention   (Tensor Core MMA, split-KV for SM occupancy)
K2: Beam Sparse Attention (CUDA Core FMA, topK gather)
K3: Combine             (log-sum-exp merge of K1 + K2 partials)
```

With split-KV enabled on K1, the pipeline produces `ns + 1` partial results (`ns` from K1 splits, 1 from K2), all merged by K3.

### Fused Kernel (SM8x/SM90)

Merges K1 and K2 into a single kernel launch. The beam sparse phase runs as CUDA core FMA directly on the MMA accumulator after the context attention mainloop, sharing the same softmax state. Combined with split-KV, only K3 (combine) remains as a separate launch.

```
Fused (ns=1):  1 kernel launch  → context + beam fused → bf16 output
Fused (ns>1):  2 kernel launches → fused (ns fp32 partials) + K3 combine → bf16 output
3-kernel:      3 kernel launches → K1 (ns fp32 partials) + K2 (1 fp32 partial) + K3 combine (ns+1 partials) → bf16 output
```

## Architecture Support

| GPU | Arch | Default Path | K1 split-KV | Fused Beam |
|-----|------|--------------|-------------|------------|
| A100 / L40 / L20 | SM8x | Fused + split-KV | Yes | Yes (256 threads, MMA acc FMA) |
| H100 / H800 / H20 | SM90 | Fused + split-KV | Yes | Yes (HGMMA, 4-thread parallel QK) |
| B200 | SM100 | 3-kernel + split-KV | Yes | No |
| RTX PRO6000 | SM120 | Fused + split-KV | Yes | Yes (inherits SM80) |

### Path Selection

The optimal path depends on your workload (`decode_nums`, `beam_width`, `seqlen_context`, etc.). Use the benchmark tool to find the best configuration:

```bash
PYTHONPATH=. python tests/benchmark.py --mode benchmark --decode_nums 1 4 8 16
```

Override the default path via the `backend` parameter:

```python
# Force fused kernel
out, lse = beam_decode_attn(..., backend="dsl")

# Force 3-kernel pipeline
out, lse = beam_decode_attn(..., backend="3kernel")
```

## Tensor Shapes

```
Q:            [batch, seqlen_q, beam_width, head_q, dim]     bf16/fp16
K/V context:  [batch, seqlen_context, head_kv, dim]          bf16/fp16
K/V beam:     [batch, decode_nums * beam_width, head_kv, dim] bf16/fp16
topK indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width] int32
Output:       [batch, seqlen_q, beam_width, head_q, dim]     same as Q
LSE:          [batch, seqlen_q, beam_width, head_q]           fp32
```

## Quick Start

```python
from interface import beam_decode_attn

# Auto-selects fused or 3-kernel based on GPU architecture
out, lse = beam_decode_attn(
    q, k_context, v_context, k_beam, v_beam,
    topk_indices, decode_nums,
    return_lse=True,
)

# Force 3-kernel pipeline
out, lse = beam_decode_attn(
    q, k_context, v_context, k_beam, v_beam,
    topk_indices, decode_nums,
    return_lse=True,
    backend="3kernel",
)
```

## Project Structure

```
corelib/gr_decode_atten/
├── interface.py                 # Public API: BeamDecodeAttn + beam_decode_attn
├── src/
│   ├── common/                  # Shared: config, softmax, mask, block_info, tile_scheduler, ...
│   ├── sm80/flash_fwd.py        # SM80 context attention + fused beam phase
│   ├── sm90/flash_fwd.py        # SM90 context attention + fused beam phase
│   ├── sm100/flash_fwd.py       # SM100 context attention (3-kernel only)
│   ├── sm120/flash_fwd.py       # SM120 context attention + fused beam phase (inherits SM80)
│   ├── decode/flash_fwd.py      # K2: CUDA core beam sparse attention
│   └── flash_fwd_combine.py     # K3: split-KV combine kernel
└── tests/
    ├── reference.py             # PyTorch golden reference + test data generation
    ├── test_fwd.py              # End-to-end tests (672 cases: 288 3-kernel + 384 fused)
    ├── test_context.py          # K1 unit tests (144 cases)
    ├── test_beam.py             # K2 unit tests (384 cases)
    └── benchmark.py             # Performance benchmark + ncu profile
```

## Testing

```bash
# Quick regression (14 configs, ~30s)
PYTHONPATH=. python tests/test_fwd.py

# Full test suite (672 cases)
pytest tests/test_fwd.py -v

# By backend
pytest tests/test_fwd.py -v -k "3kernel"   # 288 cases
pytest tests/test_fwd.py -v -k "fused"     # 384 cases

# Individual kernel tests
pytest tests/test_context.py -v             # K1: 144 cases
pytest tests/test_beam.py -v                # K2: 384 cases
```

### Tolerance

- **Output**: `kernel_diff <= 2 * pt_diff + fwd_atol` (FA-style, bf16 precision baseline)
- **LSE**: `abs_diff <= 1e-3`

## Benchmark

```bash
# End-to-end: 3-kernel vs fused, dn=1..16
PYTHONPATH=. python tests/benchmark.py --mode benchmark

# Profile for ncu/nsys
PYTHONPATH=. python tests/benchmark.py --mode profile --decode_nums 1
```
