# HSTU Inference Benchmarks

This benchmark note summarizes the checked-in HSTU ranking inference paths that
use the Triton Python backend, PyTorch AOTInductor (AOTI), native C++ replay,
Triton Server, and the KV-cache runtime.

Covered paths:

- [Triton Python-backend client](../../inference/triton/hstu_model/client.py)
- [PyTorch export, no cache](../export_inference_gr_ranking.py)
- [PyTorch export, with KV cache](../export_inference_gr_ranking_kvcache.py)
- C++ Torch replay, no cache: `../cpp_inference/build/inference_hstu_gr_ranking_exported_model`
- C++ Torch replay, with KV cache: `../cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
- [Triton AOTI deployment config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt)
- [Triton AOTI request replay client](../test_tritonserver_aoti_hstu_model.py)

## Benchmark results

The tables below report single-GPU KuaiRand-1K ranking measurements. Unless
otherwise specified, all results use the same model and dataset configuration.

### 1. No-cache HSTU model with Torch C++ runtime

These results are emitted by the no-cache
[export script](../export_inference_gr_ranking.py), which exports the model,
loads the packaged AOTI artifact, and compares Python runtime time against the
native C++ replay executable on the KuaiRand-1K evaluation data.

| Hardware | Python runtime E2E (s) | C++ runtime E2E (s) | C++ speedup |
| --- | ---: | ---: | ---: |
| L20 | 1.755 | 1.079 | **1.63x** |
| L40 | 1.416 | 0.850 | **1.66x** |
| L40S | 1.340 | 0.704 | **1.90x** |
| RTX PRO 6000 Blackwell Workstation Edition | 0.688 | 0.498 | **1.38x** |

### 2. Triton Server backend comparison

These results compare the no-cache
[Triton Python backend](../../inference/triton/hstu_model/client.py) with the
PyTorch AOTI backend deployed using the Triton
[AOTI model config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt).

Hardware: **RTX PRO 6000 Blackwell Workstation Edition**  
Batch size: **2**

| Triton serving path | Cache state | Total E2E time (s) | Speedup vs. Python backend | Time reduction |
| --- | --- | ---: | ---: | ---: |
| Python backend | No cache | 1.311 | 1.00x | Baseline |
| PyTorch AOTI backend | No cache | 1.065 | **1.23x** | **18.8%** |
| PyTorch AOTI backend | GPU KV-cache hit | 0.804 | **1.63x** | **38.7%** |

The AOTI GPU KV-cache hit path is **1.32x** faster than the AOTI no-cache path.
Speedups and time reductions in the table use the no-cache Triton Python
backend as the baseline.

#### Triton benchmark protocol

- Both Triton clients use KuaiRand-1K evaluation data with batch size 2.
- One request is sent as warmup and excluded from all measured runs.
- The Python-backend client performs the same measured pass three times and
  reports each total plus their average. Each run is annotated as
  `no cache, Python backend`.
- The AOTI replay client loads the dumped `export_test_dump/batch_*.pt` tensors
  and runs the same measured cases twice. The first pass is the no-cache path;
  the second pass measures GPU KV-cache hits.
- Total E2E time is measured by the synchronous HTTP clients around request
  preparation and inference. Dataset loading, warmup sleep, and metric
  aggregation are excluded.

The KV-cache number assumes that the second AOTI replay pass hits GPU KV cache
after the warmup and first measured pass.
