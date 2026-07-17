# HSTU inference benchmark

We provide tests for difference scenarios of HSTU inference with Pytorch AOTI.

- [Pytorch Export (no cache)] inference_aoti/export_inference_gr_ranking.py
- [Pytorch Export (with kvcache)] inference_aoti/export_inference_gr_ranking_kvcache.py
- [C++ Torch (no cache)] inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_exported_model
- [C++ Torch (with kvcache)] inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model
- [Triton Deploy Config] inference_aoti/triton_aoti
- [Triton Client] inference_aoti/test_tritonserver_aoti_hstu_model.py


## Benchmark results

Here we present the benchmark results of the HSTU layers with KV cache on a single L20 GPU.

HSTU Setup: Kuairand-1k-ranking

Notice: This config is for all following benchmarks (unless otherwise specified).

### 1. HSTU model (training version, no cache) performance with Torch C++ Runtime

Benchmark results are measured in the model [exporting script](../export_inference_gr_ranking.py).
Here we present the benchmark results on the evaluation dataset **Kuairand-1k**.

* **Performance Results with Torch C++ Runtime:**

| Hardware   | Python Runtime E2E Time (s) | C++ Runtime E2E Time (s) | Speedup    |
| ---------- | --------------------------- | ------------------------ | ---------- |
| L20        |   1.755                     | 1.079                    | **1.63x**  |
| L40        |   1.416                     | 0.850                    | **1.66x**  |
| L40S       |   1.340                     | 0.704                    | **1.90x**  |
| RTX PRO 6000 Blackwell<br>Workstation Edition |   0.688                   |  0.498               | **1.38x**  |

### 2. HSTU model (with kvcache) performance on Triton Server

Benchmark results are measured in the model exported by [exporting script](../export_inference_gr_ranking_kvcache.py),
and deployed by Triton Server via [config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt).

Test with [test_tritonserver_aoti_hstu_model.py](../test_tritonserver_aoti_hstu_model.py). Here we present the benchmark results on the evaluation dataset **Kuairand-1k**.

* **Performance Results with Triton Server Pytorch AOTI Backend:**

| Hardware   | No KVCache Time (s) | GPU KVCache Hit E2E Time (s) | Speedup    |
| ---------- | --------------------------- | ------------------------ | ---------- |
| RTX PRO 6000 Blackwell<br>Workstation Edition |   1.065                 |  0.804              | **1.33x**  |

Note: In the scenario with KV cache, we benchmark for requests all hit on GPU KV cache.