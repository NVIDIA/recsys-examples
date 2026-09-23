# Transformer recommendation inference

Set `NetworkArgs.backbone = 'transformer'` to select a Transformer dense
backbone in the existing ranking inference workflow. The default remains
`'hstu'`. Sparse embeddings, item/action interleaving, context tokens, position
embeddings, candidate postprocessing, prediction heads and KV-cache management
use the existing recommendation components.

## Architecture and input contract

Each Transformer layer applies Pre-LayerNorm, biased Q/K/V projections,
scaled softmax multi-head self-attention, a biased output projection, and a
Pre-LayerNorm GELU feed-forward network. Both sublayers have residual additions
when `residual=True`. `transformer_ffn_dim` defaults to four times `hidden_size`.
The attention projection width is `num_heads * head_dim`; it need not equal
`hidden_size`. All heads have K/V, with matching K/V dimensions. There is no
dropout in inference.

Position information comes from the existing preprocessing position embeddings.
This is not a Llama/Hugging Face checkpoint loader: RoPE, GQA/MQA, cross-attention
and model-specific weight conversion are not provided. Context tokens use the
causal inference mask, like history tokens; they do not attend to future history.
The existing training entry points remain HSTU-only and reject a Transformer
selection rather than silently training the wrong architecture.

Packed inputs must contain each user's context/history followed by candidates.
History queries attend causally. A candidate sees the history and itself, but
not other candidates (`target_group_size=1`). A plain causal mask would introduce
cross-candidate dependencies and is not equivalent. `max_seq_len` bounds the
complete sequence, including candidates, even when the current query only
contains the uncached suffix.

## Running

In the repository's CUDA development environment, from `examples/hstu`:

```bash
export PYTHONPATH="$(realpath ..):$PWD:$PYTHONPATH"
python inference/inference_gr_ranking.py \
  --gin_config_file inference/configs/kuairand_1k_transformer_ranking.gin \
  --checkpoint_dir /path/to/transformer-checkpoint --mode eval
```

The `get_inference_hstu_config` API also accepts `backbone="transformer"` and
`transformer_ffn_dim=...`; its historical name is retained for compatibility.
The Triton Python dense model reads the same NetworkArgs configuration.

## Checkpoints

The sparse checkpoint layout is unchanged. The dense state lives in the existing
`torch_module/model.0.pth` file under `model_state_dict`. Transformer layers use
these names beneath `_hstu_block._attention_layers.<layer_index>.`:

- `input_norm.weight`, `input_norm.bias`
- `qkv.weight`, `qkv.bias` (Q then K then V, each of width `num_heads * head_dim`)
- `proj.weight`, `proj.bias`
- `ffn_norm.weight`, `ffn_norm.bias`
- `ffn.0.weight`, `ffn.0.bias`, `ffn.2.weight`, `ffn.2.bias`

The shared processor and MLP retain their existing names. To obtain the complete
dense schema, construct `get_inference_ranking_gr(...)` with a Transformer config
and inspect `model.dense_module.state_dict()`. Nonpersistent capture buffers are
excluded. An external trainer/converter must provide weights for this exact
architecture and the shared processor/head, plus the matching sparse weights.
Missing/unexpected keys fail loading, including when the outer workflow uses
`strict=False` to filter embedding keys. HSTU UVQK transposition and cached weight
refresh are applied only to HSTU checkpoints.

## Cache, graph and export paths

The layer uses the existing NHD page layout
`[pages, 2, page_size, num_heads, head_dim]`. New history K/V are appended through
`paged_kvcache_ops.append_kvcache` on CUDA; candidates are never persisted.
Reads materialize padded K/V and use PyTorch SDPA with an explicit recommendation
mask and cached query offset. A tensor implementation of the same cache writes
is available on CPU for numerical tests. It does not emulate asynchronous GPU
transfers or the cache manager.

The layer implements `forward_naive`, `forward_input`, `forward_output` and
`output_buffer_` for the existing per-layer CUDA graph capture/replay orchestration.
Capture passes a nonzero token-count upper bound to the append operator so it
does not read a GPU scalar back to the host. Native cache onload synchronization
and offload are owned by the existing inference orchestration.

Both existing exporter entry points propagate the backbone selection:

```bash
python inference_aoti/export_inference_gr_ranking.py \
  --gin_config_file inference/configs/kuairand_1k_transformer_ranking.gin \
  --checkpoint_dir /path/to/transformer-checkpoint --max_bs 2 \
  --export_dir /path/to/empty-transformer-export \
  --dump_dir /path/to/empty-transformer-replay

python inference_aoti/export_inference_gr_ranking_kvcache.py \
  --gin_config_file inference/configs/kuairand_1k_transformer_ranking.gin \
  --checkpoint_dir /path/to/transformer-checkpoint --max_bs 2 \
  --kvcache_config_file inference_aoti/kvcache_cpp_runtime.yaml \
  --export_dir /path/to/empty-transformer-kv-export \
  --dump_dir /path/to/empty-transformer-kv-replay
```

Adjust the KV runtime YAML dimensions, dtype and maximum lengths to match the
model. Follow the [AOTI workflow](../inference_aoti/README.md) for dependency
versions, C++ replay and Triton deployment. The exporters use the existing
training shell to obtain sparse/processor/head schemas; they construct fresh
Transformer layers and load the supplied Transformer checkpoint. They do not
convert HSTU dense weights into Transformer weights.

## Validation and limitations

```bash
python -m pytest test/test_transformer_inference.py \
  test/test_hstu_block_inference.py test/test_nve_aoti_compat.py -q -ra
```

CPU tests compare the actual layer against independent, unpadded softmax
arithmetic; cover multi-layer paged-prefix reuse, variable-length users,
noncontiguous physical pages, candidate independence, page-tail preservation,
empty histories, half/bfloat16, HSTU/Transformer checkpoint handling, and
torch.export save/reload with dynamic shapes and cache mutation. CUDA-only tests
exercise the native append operator, graph capture/replay and the shared
recommendation pre/postprocessor. A skip is not GPU validation.

This backend prioritizes functional integration. Eager attention bounds query
padding by the smaller of the packed token count and `max_seq_len`; export uses
the fixed `max_seq_len` bound to avoid specializing dynamic token dimensions.
Cached K/V always pad to `max_seq_len`. Attention creates a dense boolean mask
per user, so large configured limits can consume substantial memory. This is not an
optimized paged Transformer attention kernel and makes no latency/throughput
claim. Profile realistic workloads before production use. This implementation
was locally tested on CPU; full NVE + GPU cache manager + AOTI/C++ + Triton
integration and recommendation-quality benchmarks still require a compatible
CUDA environment and a trained Transformer checkpoint.
