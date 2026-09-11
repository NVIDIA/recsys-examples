# FA2 + `gr_decode_atten` benchmark

SID-GR now has one production generation path:

1. standard FA2 varlen causal prefill over packed `[history + BOS]` tokens;
2. a per-layer context K/V cache shared by all beams;
3. incremental beam decode through the vendored `gr_decode_atten` CuTe kernel.

`SIDGRModel.generate()` selects this path by default and is equivalent to
calling `generate_beam_decode()` with its default arguments. The old
arbitrary-mask/full-prefix jagged comparison is no longer applicable, so its
historical numbers have been removed.

The benchmark compares two context-cache layouts within the production
backend:

- packed K/V (default): FA2 varlen prefill plus `cu_seqlens_k`;
- padded K/V: dense FA2 prefill plus `seqused_k`, retained as a kernel-layout
  comparison and for callers that already provide padded input.

Run inside the project container:

```bash
cd examples/sid_gr
torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --sweep --batch_size 16 \
  --num_hierarchies 4 --num_layers 8 \
  --hidden_size 1024 --num_heads 8 --kv_channels 128 \
  --sweep_hist 256,512,1024,2048 \
  --sweep_beam 200 --sweep_dtype bf16
```

The SID-GR model resolves the vendored
[`corelib/gr_decode_atten/`](../../../corelib/gr_decode_atten/) path from the
repository checkout. The benchmark fails early if it resolves to the PyTorch
correctness fallback instead of the real CuTe kernel.

For the Megatron reference backend, set
`NetworkArgs.use_jagged_flash_attn = False`. It intentionally retains dense
full-prefix generation for behavioral/reference coverage and is not mixed into
this production-backend microbenchmark because it owns a different parameter
layout.
