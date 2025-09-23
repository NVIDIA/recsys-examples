# HSTU

HSTU is a high-performance attention implementation for different NVIDIA GPU architectures:
- **HSTU-2**: Runs on Ampere and Ada GPUs (A100, L20...)
- **HSTU-3**: Optimized for Hopper GPUs (H100, H20...)

## Installation

### HSTU-2
```sh
make install
```

### HSTU-3
```sh
cd hopper/
make install
```

## Features

### HSTU-2
- **Source**: `csrc/hstu_attn/src`
- **Interface**: `hstu_attn/hstu_attn_interface.py`
- **Supported GPUs**: Ampere, Ada, Hopper (without Hopper-specific features)
- **Data types**: FP16, BF16
- **Head dimensions**: 32, 64, 128, 256
- **Attention masks**:
  * No mask
  * Local mask (0 <= window_size_left < max_seqlen_k or 0 <= window_size_right < max_seqlen_k)
  * Causal mask (window_size_left = -1 and window_size_right = 0)
  * Context mask + causal mask
  * Target mask + causal mask
  * Context mask + causal mask + target mask

    ![Context+causal+target mask](context_causal_target.png)
  * Delta_q ('seqlen_q <= seqlen_k')
  * Delta_q + local mask

    ![Delta_q+local mask](deltaq_local.png)
  * Delta_q + causal mask

    ![Delta_q+causal mask](deltaq_causal.png)

  * Arbitrary mask

### HSTU-3
- **Source**: `hopper/`
- **Interface**: `hopper/hstu_attn_interface.py`
- **Supported GPUs**: Hopper only (H100, H20)
- **Data types**: FP16, BF16, and FP8
- **Head dimensions**: 32, 64, 128, 256 for FP16/BF16; 64, 128, 256 for FP8
- **Quantization for FP8**: Support 1xDIM and 128x1 quantization, per-block quantization, per-head quantization, per-batch quantization and per-tensor quantization.
- **Attention masks**: Same as HSTU-2.
- **Note**: Only undeterministic backward implementation

## Unified Interface

Both HSTU-2 and HSTU-3 share the same function signature:

```python
def hstu_attn_varlen_func(
    q,                      # (total_q, nheads, headdim)
    k,                      # (total_k, nheads_k, headdim), nheads should be equal to nhead_k
    v,                      # (total_k, nheads_k, headdim)
    cu_seqlens_q,          # (batch_size + 1,), cumulative sequence lengths for q
    cu_seqlens_k,          # (batch_size + 1,), cumulative sequence lengths for k/v
    max_seqlen_q,           # Maximum query sequence length
    max_seqlen_k,           # Maximum k/v sequence length
    num_contexts=None,      # (batch_size,), context tokens per batch
    num_targets=None,       # (batch_size,), target tokens per batch
    target_group_size=1,    # Number of target tokens per group
    window_size=(-1, -1),   # (left, right) for sliding window, -1 means infinite window size
    alpha=1.0,              # Scaling factor between add rab and silu
    rab=None,               # (batch_size, nhead_rab, max_seqlen_k, max_seqlen_k), relative attention bias
                            # nheads should be divisible by nhead_rab
    has_drab=False,         # Whether to apply drab
    is_delta_q=False,       # Whether to apply delta_q
    quantization_mode=-1,   # -1: no quantization, 0: cast to fp8, 1: 1xDIM&128x1 quantization, 2: per-block quantization, 3: per-head quantization, 4: per-batch quantization, 5: per-tensor quantization.
)
```

See Makefiles for customization compiling options (dimensions, data types, functions).