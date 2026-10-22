# HSTU inference benchmark with FlexKV cache

We provide benchmarks for HSTU inference with a FlexKV-backed KV cache, covering GPU-hit, CPU(Host)-hit, (local)SSD-hit, and a no-cache recompute baseline on the same request shape.

## How to run the benchmark

```bash
~$ cd recsys-examples/examples/hstu
~$ export PYTHONPATH=${PYTHONPATH}:$(realpath ../)

# FlexKV cache-tier hits (default: gpu_hit,cpu_hit,ssd_hit). CUDA Graph is on.
~$ python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --history-len 1024 --append-history-len 0 --only-onboard \
    --warmup-iters 100 --timed-iters 50

# GPU-hit and CPU-hit only (CPU-only FlexKV, no SSD config)
~$ python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --scenarios gpu_hit,cpu_hit \
    --history-len 1024 --append-history-len 0 --only-onboard \
    --warmup-iters 100 --timed-iters 50

# No-cache baseline (recompute; does not start FlexKV)
~$ python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --scenarios no_cache \
    --history-len 1024 --append-history-len 0 --only-onboard \
    --warmup-iters 100 --timed-iters 50
```

- `--scenarios`: comma-separated paths. Supported values are `gpu_hit`, `cpu_hit`, `ssd_hit`, and `no_cache`. Default is `gpu_hit,cpu_hit,ssd_hit`. `no_cache` must be run by itself.
- `--layerwise`: enable FlexKV layerwise onboard (`wait_layer` per original layer).
- `--layer-granularity N`: SSD DISK2H span size while `--layerwise` is on. Unset defaults to 1. Ignored without `--layerwise`.
- `--history-len`: cached or recomputed prefix length per feature (action and item each). Default is 1024.
- `--append-history-len`: extra tokens appended on the timed request after the warmed prefix. Default is 1024. These new tokens are written into the GPU KV cache and then offloaded to Host/SSD, so this flag is used to measure the follow-up offload path. Set it to 0 (or pass `--only-onboard`) when you only want to time prefix reuse.
- `--only-onboard`: time the warmed prefix only (`history-len`). Without this flag, the timed sequence length is `history-len + append-history-len`.
- `--flexkv-config-path`: FlexKV tier config file.

For SSD-hit, set `ssd_cache_dir` in `[flexkv_ssd.yml](../../../corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml)` to a writable local SSD path, then:

```bash
~$ python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --scenarios ssd_hit \
    --history-len 1024 --append-history-len 0 --only-onboard \
    --warmup-iters 100 --timed-iters 50 \
    --flexkv-config-path ../../corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml
```



## Benchmark results

Here we present the benchmark results of HSTU inference with FlexKV cache on a single RTX PRO 6000 Blackwell GPU.

HSTU Setup for benchmark:


| Parameter                   | Value |
| --------------------------- | ----- |
| Number of HSTU layers       | 8     |
| Hidden Dim Size             | 1024  |
| Number of Heads             | 4     |
| Head Dim Size               | 256   |
| Max batch size              | 16    |
| Max Per Sequence Length     | 16384 |
| Per Sequence Targets Number | 256   |
| Data type                   | BF16  |
| CUDA Graph                  | on    |




### 1. End-to-end inference performance

GPU-hit is the latency floor once the prefix is long enough. Versus the no-cache baseline we see **1.4x ~ 2.5x** on GPU-hit as sequence_length grows from 8192 to 16384. CPU-hit is **1.2x** at 16384. SSD-hit is the same naive DISK2H + H2D path as section 2, with the measured DISK2H time retimed to 7 GB/s so it matches the naive column below. No-cache, GPU-hit, and CPU-hit are the original measurements. Plotted `sequence_length` is `2 * history-len` (action + item).

* **Performance Results on RTX PRO 6000 (batch size = 1):**

![Local Image](flexkv_cache_e2e.png)

Note:

1. The baseline (`No-cache`) is `forward_nokvcache` (`--scenarios no_cache`), without FlexKV and without KV-cache reuse.
2. The end-to-end latency includes the embedding part, which utilizes both native `EmbeddingCollection` from TorchRec and `DynamicEmbedding`.
3. `--history-len` is the per-feature prefix reused by `gpu_hit` / `cpu_hit` / `ssd_hit`, or fully recomputed by the no-cache baseline. The x-axis `sequence_length` is `2 * history-len` (action + item).
4. In the test cases with KV cache enabled, onboarding of the cached prefix is within time measurement.
5. At short history, no-cache can beat GPU-hit because the timed path includes KV lookup/onboard. At long history, GPU-hit is clearly faster.
6. SSD-hit E2E is `measured_e2e - disk_ms + disk_ms * measured_gbps / 7`. H2D and HSTU stay as measured. Naive measured DISK2H was 6.62–6.96 GB/s.

### 2. SSD-hit layerwise vs naive

Layerwise onboard splits SSD DISK2H into spans of `N` original layers and overlaps H2D with HSTU. Naive is sequential DISK2H + H2D with no overlap. Results below are 100% local-SSD restore on RTX PRO 6000 Blackwell, batch size 1, LayerFirst layout, `--only-onboard --no-prefetch`. The x-axis `sequence_length` is `2 * history-len` (action + item), the same scale as section 1. `--history-len 1024` is `sequence_length` 2048. E2E below retimes every measured DISK2H to 7 GB/s (`e2e - disk_ms + disk_ms * measured_gbps / 7`); H2D and HSTU are unchanged. The bandwidth figure is the measurement before that retiming.

```bash
# naive: sequential DISK2H + H2D
python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --scenarios ssd_hit --history-len 1024 --append-history-len 0 --only-onboard --no-prefetch \
    --warmup-iters 100 --timed-iters 50 \
    --flexkv-config-path ../../corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml

# layerwise, N = 1 2 3 4 8
python3 ./inference/benchmark/inference_benchmark_flexkv.py \
    --scenarios ssd_hit --layerwise --layer-granularity N \
    --history-len 1024 --append-history-len 0 --only-onboard --no-prefetch \
    --warmup-iters 100 --timed-iters 50 \
    --flexkv-config-path ../../corelib/recsys_kvcache_manager/configs/flexkv_ssd.yml
```

- naive: sequential DISK2H + H2D.
- `N = 1 2 3 4 8`: DISK2H span size with H2D + HSTU overlap (`wait_layer` per original layer).
- `N = 8`: overlap is only H2D and HSTU (one full-KV disk read), matching FlexKV's original layerwise scheme.

* **End-to-end latency (ms) at 7 GB/s DISK2H, RTX PRO 6000 (batch size = 1):**

![Local Image](layerfirst_bs1_e2e.png)

| sequence_length | naive (ms) | N=1 | N=2 | N=3 | N=4 | N=8 |
| --------------- | ---------- | --- | --- | --- | --- | --- |
| 2048            | 15.96      | 13.36 | 13.19 | 12.77 | 13.17 | 14.24 |
| 4096            | 28.14      | 23.42 | 23.81 | 22.95 | 23.88 | 25.50 |
| 8192            | 52.28      | 43.69 | 44.70 | 43.32 | 45.37 | 47.59 |
| 16384           | 98.97      | 85.50 | 86.37 | 84.48 | 87.48 | 91.82 |

With DISK2H held at 7 GB/s, layerwise is faster than naive at every sequence length. N=8 stays the closest to naive because both read the full KV in one DISK2H; its remaining gain is H2D overlapped with HSTU. Finer N is faster still once the small-transfer bandwidth gap in the measured figure is removed.

* **Measured SSD read bandwidth (GB/s), before the 7 GB/s retiming:**

![Local Image](layerfirst_bs1_ssd_bw.png)

| sequence_length | naive (GB/s) | N=1 | N=2 | N=3 | N=4 | N=8 |
| --------------- | ------------ | --- | --- | --- | --- | --- |
| 2048            | 6.62         | 4.86 | 5.77 | 5.88 | 6.26 | 6.62 |
| 4096            | 6.74         | 5.60 | 6.17 | 6.37 | 6.57 | 6.77 |
| 8192            | 6.73         | 5.84 | 6.24 | 6.34 | 6.54 | 6.69 |
| 16384           | 6.96         | 6.24 | 6.64 | 6.69 | 6.82 | 7.02 |

SSD read bandwidth rises with transfer granularity: N=1 is lowest (~4.9–6.2 GB/s) because each DISK2H is a single layer, and N=8 matches naive (~6.6–7.0 GB/s) because both issue one full-KV disk read.
