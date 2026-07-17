# HSTU AOTI Inference Benchmarks

This benchmark note summarizes the checked-in HSTU ranking inference paths that use PyTorch AOTInductor (AOTI), native C++ replay, Triton Server, and the KV-cache runtime.

Covered paths:

- [PyTorch export, no cache](../export_inference_gr_ranking.py)
- [PyTorch export, with KV cache](../export_inference_gr_ranking_kvcache.py)
- C++ Torch replay, no cache: `../cpp_inference/build/inference_hstu_gr_ranking_exported_model`
- C++ Torch replay, with KV cache: `../cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
- [Triton deployment config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt)
- [Triton request replay client](../test_tritonserver_aoti_hstu_model.py)

## Benchmark results

The tables below report single-GPU KuaiRand-1K ranking measurements from the checked-in export and replay flows.

HSTU setup: KuaiRand-1K ranking.

Unless otherwise specified, all results use the same model and dataset configuration.

### 1. No-cache HSTU model with Torch C++ runtime

These results are emitted by the no-cache [export script](../export_inference_gr_ranking.py), which exports the model, loads the packaged AOTI artifact, and compares Python runtime time against the native C++ replay executable on the KuaiRand-1K evaluation data.

**Performance results:**

| Hardware   | Python Runtime E2E Time (s) | C++ Runtime E2E Time (s) | Speedup    |
| ---------- | --------------------------- | ------------------------ | ---------- |
| L20        |   1.755                     | 1.079                    | **1.63x**  |
| L40        |   1.416                     | 0.850                    | **1.66x**  |
| L40S       |   1.340                     | 0.704                    | **1.90x**  |
| RTX PRO 6000 Blackwell<br>Workstation Edition |   0.688                   |  0.498               | **1.38x**  |

### 2. KV-cache HSTU model on Triton Server

These results use the model exported by [export_inference_gr_ranking_kvcache.py](../export_inference_gr_ranking_kvcache.py) and deployed with the Triton [AOTI model config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt).

The Triton benchmark is driven by [test_tritonserver_aoti_hstu_model.py](../test_tritonserver_aoti_hstu_model.py). The client loads the dumped `export_test_dump/batch_*.pt` replay tensors, sends one warmup request, then runs the same measured cases twice. The second measured run represents the GPU KV-cache hit path.

**Performance results with Triton Server PyTorch AOTI backend:**

| Hardware   | No KVCache Time (s) | GPU KVCache Hit E2E Time (s) | Speedup    |
| ---------- | --------------------------- | ------------------------ | ---------- |
| RTX PRO 6000 Blackwell<br>Workstation Edition |   1.065                 |  0.804              | **1.33x**  |

Note: the KV-cache number assumes the measured requests hit GPU KV cache after the warmup/measured replay sequence.