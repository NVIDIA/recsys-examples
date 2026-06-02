<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SID-GR Inference

## 1. Background and Motivation

### Algorithm Background

Semantic ID based generative recommender modeling is one of the main directions
for recommendation, search, and advertising systems. The core workflow is:

```text
offline clustering
-> map each real item ID to a multi-level cluster ID tuple
-> autoregressively generate a short semantic ID sequence at inference time
-> map the generated semantic ID tuple back to real item IDs
```

This pattern has direct implications for inference systems:

- The cluster depth is usually small, for example 3 or 5, so autoregressive
  decode is short.
- User history can be long, so prefill/context computation can dominate cost.
- Recommendation and search systems often need result diversity. Larger and
  sometimes dynamic beam widths are a practical way to improve diversity.

### Inference Problem

The resulting inference workload is:

```text
long context + short decode + large beam width
```

This is different from the common chat LLM serving workload. vLLM, SGLang, and
TensorRT-LLM are optimized primarily for:

```text
many user requests
dynamic batching
paged KV
long decode
OpenAI/chat APIs
```

Those capabilities matter, but they are not the full bottleneck for SID-GR
inference. In recommendation and search workloads, a typical request has a long
user or candidate context, a large beam width such as 128 or 256, and only a few
decode steps. Many beams share the same request-level context.

Current general-purpose LLM inference frameworks do not fully match this
pattern:

- **vLLM:** does not provide a stable production beam-search serving path. Users
  often implement beam search by repeatedly calling vLLM from business logic.
- **TensorRT-LLM:** does not natively expose logprobs for this path, and large
  beam widths can create memory pressure. Its decode attention and postprocess
  kernels are also not specialized for short SID-GR decode.
- **SGLang:** is currently the most usable open-source baseline for this
  workload, but large-beam support is still in a feature PR and is not merged to
  upstream main.

General LLM serving frameworks are also large and complex. Adapting that full
stack for a specialized recommender inference workload can work for one business
case, but it is not the best long-term path for maintainability or for reaching
the speed-of-light target of this workload.

## 2. Goals and Approach

### Goals

- Optimize Qwen-family models toward speed-of-light performance for the SID-GR
  workload: long context, short decode, and large beam width.
- Provide a compact framework that supports practical SID-GR serving needs,
  including both feature requirements and performance requirements.

### Approach

The implementation keeps SID-GR specific runtime contracts while selectively
reusing mature ideas from open-source serving systems:

- **Keep SID-GR native abstractions.** `ContextKV`, `BeamKV`, `BeamPath`, dynamic
  beam policies, item-constrained decode, and request x active-beam batching are
  represented directly by the runtime.
- **Reuse proven serving ideas selectively.** The project borrows concepts from
  vLLM and SGLang for continuous batching, paged KV, HTTP serving, benchmark
  tooling, and APIs. It also borrows from TensorRT-LLM for kernels, CUDA graph,
  operator fusion, and model-layer optimization. These pieces support the
  SID-GR runtime rather than defining it.
- **Drive changes with benchmarks.** Correctness, performance, and Nsight
  breakdowns are used to decide which optimizations should enter the hot path.
- **Keep the framework small and specialized.** General-purpose LLM serving is
  used as a source of ideas, while KV ownership, beam state, decode attention,
  batching units, and business constraints remain SID-GR specific.

### Current Status

The repository has a single-node alpha path with real model weights:

```text
Qwen3-1.7B real weights
+ SID-GR native ContextKV / BeamKV / BeamPath
+ real gr-decode_atten backend
+ continuous batching
+ BeamKV / ContextKV dense pools
+ direct pool-view decode CUDA graph
+ HTTP /generate
+ SGLang-equivalent beam_results output
+ offline and online SID-GR vs SGLang benchmarks
```

This path validates the main value proposition: for the tested long-context,
short-decode, large-beam matrix, SID-GR offline performance is consistently
faster than the SGLang beam-search PR branch. Online serving also runs through
the same HTTP client benchmark. CUDA graph capture has been stabilized as a
startup warmup path, with replay-only execution during the measured serving
window.

## 3. Design Highlights

| Dimension | General LLM serving path | SID-GR Inference path |
| --- | --- | --- |
| KV abstraction | Sequence, token block, and paged KV centric | Explicit request-level `ContextKV` plus short `BeamKV` |
| Large-beam decode | Flattens `batch * beam` into many decode rows | Processes shared context by request and beam tile |
| Attention kernel | General paged decode attention | `gr-decode_atten` receives `ContextKV + BeamKV + BeamPath` directly |
| Batch CUDA graph | General batch buckets and paged KV constraints | Fixed SID-GR shapes and stable pool slices for replay |
| Output | General API output and beam management | Fast path returns `beam_results`; debug paths can enable `beam_details` |

Core implementation points:

- **SID-GR native KV layout.** A request's long context is stored once in
  `ContextKV`. Short decode history is stored in `BeamKV`. `BeamPath` records
  logical parent-child relations.
- **Dense ContextKV hot path.** The current `ContextKV` layout is dense and
  contiguous. This supports kernel-friendly decode attention and stable pool
  slices for CUDA graph replay.
- **BeamKV and ContextKV pools.** Continuous serving uses dense pools with
  leases, capacity tracking, high-water marks, utilization metrics, and leak
  checks.
- **Specialized decode attention.** `gr-decode_atten` understands request-level
  shared context and short beam history, instead of treating every beam as an
  independent generic decode row.
- **SID-GR continuous batching.** The scheduler groups by decode step, beam
  width, and context shape. The batching unit is request x active beams.
- **Direct pool-view decode CUDA graph.** Fixed-shape graphs bind stable
  ContextKV and BeamKV pool slices. Captured graphs replay on the serving path;
  dynamic non-contiguous KV slices fall back to eager execution.
- **Last-token logits only.** Serving prefill computes only the last-position
  logits needed for the next token.
- **Dynamic beam policy support.** The runtime supports fixed, scheduled, and
  score-margin beam policies, including request-level HTTP configuration.
- **Item-constrained generation support.** Runtime and HTTP paths include item
  tries, masks, constrained topK, catalog reload/rollback, and item metadata.
- **Correctness and performance alignment.** The fast benchmark path returns
  SGLang-equivalent `beam_results`; debug-rich `beam_details` is enabled only
  for correctness and debugging.

## 4. Default Production Path

- Continuous serving uses the real `gr-decode_atten` backend when
  `--decode-backend real` is selected.
- Eligible continuous decode batches use decode CUDA graph by default. Set
  `GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH=1` to fall back to eager decode.
- `ContextKV` and `BeamKV` use pool slices directly. Graph replay updates only
  small inputs such as beam token IDs and topK indices.
- Decode graph cache uses an entry limit, LRU eviction, and pointer guards. A
  graph is not reused if pool slice addresses do not match.
- Online serving warms up the common batch, pool-window, and `/generate
  ignore_eos` shapes at startup, then freezes new graph capture by default.
- QK norm and RoPE prefer the fastest available SGLang-style in-place kernels,
  with FlashInfer/Torch fallback. Experimental branches are not part of the
  serving hot path.
- Prefill uses SGLang-style piecewise CUDA graph by default. The current
  Qwen3-1.7B bs4/ctx1000 shape captures six graph pieces: embed, four layer
  chunks, and output.
- `/generate` returns SGLang-equivalent `beam_results` by default. When
  `ignore_eos=true`, tokenizer special tokens are suppressed by default to match
  SGLang fixed-length generation semantics.

Fallback to eager decode:

```bash
GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH=1
```

Disable SID-GR experimental JIT kernels:

```bash
GR_INFERENCE_GR_TRTLLM_KERNELS_JIT=0
```

Decode graph cache size can be configured with:

```bash
GR_INFERENCE_DECODE_CUDA_GRAPH_MAX_ENTRIES=32
```

## 5. Baseline

The headline numbers compare against the SGLang beam-search PR branch:

```text
repo:   https://github.com/cswuyg/sglang.git
branch: feature/beam_search
PR:     https://github.com/sgl-project/sglang/pull/15645
```

Test environment and workload:

```text
GPU: NVIDIA H100 80GB HBM3
model: Qwen3-1.7B
context_len: 1000 / 5000
beam_width: 256
effective output length: 3 tokens
```

## 6. Offline Performance

Performance measurement uses the default SID-GR output mode, which returns
SGLang-equivalent `beam_results` and does not construct debug-rich
`beam_details`. SID-GR enables prefill CUDA graph and direct pool-view decode
CUDA graph. Prefix/prefill cache is disabled, and Qwen special tokens are
suppressed when `ignore_eos` fixes output length. SGLang timing wraps the
beam-search PR branch `Engine.generate` call with radix cache disabled.

Radix cache off, which matches the production no-prefix-reuse setting:

| ctx | batch | SID-GR ms | SGLang ms | SGLang/SID-GR | winner | SID-GR prefill | SID-GR decode |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1000 | 1 | 17.611 | 33.515 | 1.903x | SID-GR | 7.027 | 9.952 |
| 1000 | 2 | 27.768 | 57.926 | 2.086x | SID-GR | 12.633 | 14.170 |
| 1000 | 4 | 47.736 | 102.318 | 2.143x | SID-GR | 23.442 | 22.566 |
| 1000 | 8 | 93.230 | 199.280 | 2.138x | SID-GR | 46.521 | 43.707 |
| 5000 | 1 | 42.255 | 94.554 | 2.238x | SID-GR | 31.029 | 10.701 |
| 5000 | 2 | 80.904 | 179.369 | 2.217x | SID-GR | 63.763 | 16.216 |
| 5000 | 4 | 154.224 | 349.857 | 2.269x | SID-GR | 126.087 | 26.772 |
| 5000 | 8 | 307.917 | 685.354 | 2.226x | SID-GR | 253.345 | 51.448 |

Offline conclusion: SID-GR is faster than SGLang in all tested production
no-prefix-reuse cases. For `ctx=5000, batch=8`, SID-GR is `2.23x` faster than
SGLang.

## 7. Online In-flight Serving

Online serving uses SGLang `bench_serving` as the shared HTTP client with
`request_rate=inf`, `max_concurrency=4`, and `requests=64`. SID-GR serves the
compatible `/generate` endpoint with default `beam_results`, prefill CUDA graph,
direct pool-view decode CUDA graph, and SGLang-aligned special-token suppression
for `ignore_eos`. SGLang uses the beam-search PR branch.

The measured command uses `warmup_requests=0`; external server startup warmup
and priming are not counted.

SID-GR stable reproduction mode:

- Warm up online batch sizes and KV pool slot windows during server startup.
- Run warmup requests through the same `ignore_eos=true` and special-token
  suppression path as real `/generate` requests.
- Freeze prefill and decode CUDA graph capture after warmup.
- Skip CUDA graph for dynamic non-contiguous KV composition and use eager
  execution instead.

Fixed case:

```text
ctx=5000, beam=256, output=3, requests=64, max_concurrency=4
```

Latest three reruns:

| server / output mode | round | req/s | median ms | p90 ms | p99 ms | input tok/s | output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SID-GR `/generate`, `beam_results`, frozen graph capture | 1 | 19.41 | 199.32 | 240.16 | 323.75 | 97045 | 14906 |
| SID-GR `/generate`, `beam_results`, frozen graph capture | 2 | 20.10 | 195.62 | 223.38 | 252.40 | 100505 | 15438 |
| SID-GR `/generate`, `beam_results`, frozen graph capture | 3 | 19.66 | 199.92 | 235.61 | 304.19 | 98294 | 15098 |
| SGLang `/generate`, beam results, primed steady | 1 | 10.69 | 369.69 | 374.14 | 379.08 | 53450 | 8208 |
| SGLang `/generate`, beam results, primed steady | 2 | 10.65 | 370.59 | 373.94 | 378.19 | 53250 | 8177 |
| SGLang `/generate`, beam results, primed steady | 3 | 10.68 | 370.36 | 373.68 | 375.22 | 53400 | 8201 |

SID-GR graph stability gate:

| checkpoint | prefill captures | decode captures | captures enabled | dynamic graph skips |
| --- | ---: | ---: | ---: | ---: |
| startup | 10 | 30 | 0 | 0 |
| round1 | 10 | 30 | 0 | 8 |
| round2 | 10 | 30 | 0 | 11 |
| round3 | 10 | 30 | 0 | 15 |

Online conclusion: with the stable reproduction mode, SID-GR reaches
`19.41 / 20.10 / 19.66` req/s over three rounds. Decode CUDA graph captures stay
fixed at `30` and no longer grow during online scheduling. Compared with the
three primed-steady SGLang runs, SID-GR average throughput is about `1.85x`
higher and median latency is about `46%` lower. p99 can still fluctuate due to
the HTTP client, Python scheduling, request arrival timing, and batch fill.

Reproduce the SID-GR online stable mode:

```bash
BASE_OUT=benchmark_artifacts/sglang_compare/gr_online_repro
mkdir -p "${BASE_OUT}/gr"

env GR_MODEL_DIR=/workspace/models/Qwen3-1.7B \
  GR_CONTEXT_LEN=5000 \
  GR_DECODE_STEPS=3 \
  GR_BEAM_WIDTH=256 \
  GR_MAX_BATCH_SIZE=4 \
  GR_BEAM_KV_POOL_CAPACITY=4 \
  GR_CONTEXT_KV_POOL_CAPACITY=4 \
  GR_HTTP_HOST=0.0.0.0 \
  GR_HTTP_PORT=8000 \
  GR_DECODE_BACKEND=real \
  GR_DEVICE=cuda \
  GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS=1,2,4,8 \
  GR_WARMUP_ONLINE_SHAPES=1 \
  GR_WARMUP_ONLINE_POOL_WINDOWS=1 \
  GR_WARMUP_ONLINE_MAX_CASES=64 \
  GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=1 \
  GR_ENABLE_PREFILL_CACHE=0 \
  scripts/serve_qwen3_gr_http.sh \
  > "${BASE_OUT}/gr_server.log" 2>&1 &
SERVER_PID=$!

until curl -fsS http://127.0.0.1:8000/ready >/dev/null; do sleep 2; done
curl -fsS http://127.0.0.1:8000/metrics \
  -o "${BASE_OUT}/metrics_after_startup.json"

for round in 1 2 3; do
  OUT_DIR="${BASE_OUT}/gr/round${round}" \
  REQUESTS=64 CONTEXT_LEN=5000 DECODE_STEPS=3 BEAM_WIDTH=256 \
  REQUEST_RATE=inf MAX_CONCURRENCY=4 WARMUP_REQUESTS=0 \
  scripts/run_gr_sglang_bench_serving_beam_benchmark.sh \
  2>&1 | tee "${BASE_OUT}/gr_round${round}.log"
  curl -fsS http://127.0.0.1:8000/metrics \
    -o "${BASE_OUT}/metrics_after_round${round}.json"
done

kill "${SERVER_PID}"
```

Using `scripts/serve_qwen3_gr_http.sh` plus
`scripts/run_gr_sglang_bench_serving_beam_benchmark.sh` follows this stable
reproduction path by default. Set `GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=0` only
when debugging graph coverage.

Online `/generate` top1 correctness smoke:

| ctx | beam | requests | max concurrency | top1 exact | topK overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5000 | 256 | 64 | 4 | 58/64 | min 0.918 / mean 0.956 |

This uses the real HTTP `/generate` path and default `beam_results` output.
Requests whose top1 differs still have high TopK overlap.

## 8. Correctness

Correctness compares SID-GR default `beam_results` output against SGLang
`beam_results`. SID-GR enables CUDA graph and suppresses Qwen special tokens to
match SGLang `ignore_eos=true` fixed-length generation semantics.

| ctx | batch | top1 exact | topK overlap | note |
| ---: | ---: | ---: | ---: | --- |
| 1000 | 1 | 1.00 | 0.949 | top1 exact |
| 1000 | 2 | 1.00 | 0.953 | top1 exact |
| 1000 | 4 | 1.00 | 0.956 | top1 exact |
| 1000 | 8 | 1.00 | 0.958 | top1 exact |
| 5000 | 1 | 1.00 | 0.969 | top1 exact |
| 5000 | 2 | 1.00 | 0.961 | top1 exact |
| 5000 | 4 | 1.00 | 0.955 | top1 exact |
| 5000 | 8 | 1.00 | 0.959 | top1 exact |

Offline correctness conclusion: in the production no-prefix-cache mode, all
eight fixed beam=256 cases have exact Top1 agreement and high TopK overlap. For
the full 24-case matrix, top1 min/mean is `1.000 / 1.000`, and TopK overlap
min/mean is `0.945 / 0.960`.

## 9. Performance Breakdown

Fixed case:

```text
ctx=1000, beam=256, batch=4, output=3
```

Nsight profile summary:

| metric | SID-GR | SGLang |
| --- | ---: | ---: |
| active CUDA window | 46.856 ms | 99.944 ms |
| kernel total | 43.168 ms | 78.859 ms |
| CUDA runtime API total | 42.886 ms | 42.975 ms |
| CPU runtime gaps >50us | 2.186 ms | 28.305 ms |
| CUDA graph launches, active window total | 8 | 29 |
| kernel launch count | 1261 | 1620 |

Prefill stage:

| metric | SID-GR | SGLang |
| --- | ---: | ---: |
| stage total | 24.404 ms | 27.549 ms |
| attention kernels | 1.367 ms | 1.389 ms |
| non-attention kernels | 20.825 ms | 19.102 ms |
| CPU overhead | 2.213 ms | 7.057 ms |
| CUDA graph launches | 6 | 29 |

Decode stage:

| metric | SID-GR | SGLang |
| --- | ---: | ---: |
| stage total | 23.318 ms | 55.663 ms |
| attention kernels | 1.593 ms | 35.235 ms |
| non-attention kernels | 19.384 ms | 13.228 ms |
| CPU overhead | 2.341 ms | 7.200 ms |
| CUDA graph launches | 2 | 0 |

Additional kernel buckets:

| metric | SID-GR | SGLang |
| --- | ---: | ---: |
| topK / beam selection | 4.027 ms | 4.716 ms |
| attention bucket total | 2.960 ms | 40.951 ms |

The main latency tables should be used for end-to-end latency. Nsight is used
here to explain the active CUDA window, stage split, and kernel breakdown. The
raw Nsight output is in:

```text
benchmark_artifacts/sglang_compare/prod_breakdown_ctx1000_beam256_b4_20260525_121548/
```

For batch 4, the main gap is not topK sorting. It comes from the large-beam
decode attention path and general serving overhead.

```text
decode CUDA graph shortens the fixed-shape path;
SID-GR decode attention reduces core compute.
```

In this case, decode attention kernels differ by about `35.235 - 1.593 = 33.6
ms`, and the active CUDA window differs by about `99.944 - 46.856 = 53.1 ms`.
The largest gain still comes from SID-GR specific decode attention and KV
structure.

SGLang decode attention sees:

```text
batch * beam = 4 * 256 = 1024 decode rows
```

Each row follows a general paged decode attention path.

SID-GR decode attention preserves the workload structure:

```text
4 requests
256 beams per request
one shared ContextKV per request
short BeamKV history per beam
```

The context part is processed by request and beam tile. The short BeamKV part
attends only a few decode tokens. Decode CUDA graph further reduces launch and
CPU scheduling overhead for fixed-shape decode steps.

In short, SGLang is a general beam serving path; SID-GR is specialized for long
context, large beam, and short decode.

## 10. TODO and Roadmap

The single-node alpha core path is complete. The remaining work is
productionization, broader validation, and framework maintainability.

| Area | Existing foundation | Follow-up work |
| --- | --- | --- |
| Online serving hardening | HTTP `/generate`, background worker, continuous batching, pool metrics, online correctness/perf benchmarks, frozen CUDA graph capture after warmup | Improve admission, batch fill, and tail latency; add request-rate, max-concurrency, arrival-pattern, and long soak regressions |
| ContextKV memory strategy | Dense ContextKV pool on offline/online hot path with stable pool slices for CUDA graph | Add multiple context buckets based on real context length distribution; integrate page-backed ContextKV storage; eventually support native page tables in decode attention |
| CUDA graph productionization | Direct pool-view decode graph enabled by default; startup warmup covers batch, pool window, and `/generate ignore_eos` shapes | Expand warmup shapes; add graph coverage, fallback, eviction, and metric regressions |
| Beam selection graph | Decode forward is already in graph; beam selection remains outside graph | Move `log_softmax + topK + beam selection` into graph once pool ownership, item masks, special-token suppression, and output trimming are safe |
| SID-GR vs SGLang benchmark | Final offline mode covers `ctx=1000/5000, beam=256, batch=1/2/4/8`; online HTTP benchmark has a stable recipe | Extend context lengths, beam widths, dtypes, model sizes, and GPUs; provide one-command final offline/online summary scripts |
| Beam result output | `/generate` returns SGLang-equivalent `beam_results`; debug-rich `beam_details` is opt-in | Further optimize Python construction and JSON serialization; define score normalization, length penalty, and tie-break policy |
| Dynamic beam policies | Fixed, scheduled, and score-margin policies exist and are configurable over HTTP | Use real quality metrics to set default policies, score margins, shrinking rules, and quality regression gates |
| Item-constrained generation | Item trie, legal-token mask, constrained topK, catalog reload/rollback, and item metadata exist in runtime/HTTP | Test with real large catalogs; add item-level correctness, illegal-token checks, constrained topK optimization, and serving semantics regressions |
| Memory admission and reclamation | KV budget, dense pool metrics, high-water marks, leak checks, and cancel/timeout lifecycle exist | Combine page/offload support with finer reclamation; improve high-concurrency admission policy; connect memory estimates to serving decisions |
| Model and backend matrix | Main path validates Qwen3-0.6B / H100 / BF16 | Expand Qwen-family sizes, dtypes, quantization, head configs, checkpoint compatibility, and backend fallback tables |
| Multi-GPU and scale-out | Current main path is single-node / single-GPU | Design TP/PP, multi-replica scheduling, cross-GPU KV and beam ownership, load balancing, and deployment orchestration |
| Tests and documentation | Smoke tests, offline/online benchmarks, Nsight breakdown, and memory estimator exist | Consolidate stable entrypoints; add one-command final offline/online runs; add CI/nightly correctness and performance regressions |

## 11. Repository Structure

```text
src/gr_inference/gr_models/      Qwen-family model integration
src/gr_inference/gr_kv/          ContextKV, BeamKV, BeamPath
src/gr_inference/gr_kernels/     kernel wrappers and backend selection
src/gr_inference/gr_runtime/     beam search runtime and logits processing
src/gr_inference/gr_serving/     continuous batching, memory pools, HTTP serving
tools/                           benchmarks, comparison, profiling utilities
scripts/                         reproducible benchmark and serving entrypoints
tests/                           runtime, serving, model, kernel selection tests
```

## 12. Quickstart

### Default Environment and Model

Default Docker image: `lmsysorg/sglang:dev-cu13`

Default model: `Qwen/Qwen3-1.7B`

From a `recsys-examples` checkout:

```bash
cd examples/sid-gr-inference
```

Or clone this branch directly:

```bash
git clone --recurse-submodules -b merge_gr_inference_to_main git@github.com:cb521/recsys-examples.git
cd recsys-examples/examples/sid-gr-inference
```

Enter the container:

```bash
scripts/run_container.sh
```

### Select a Model

Hugging Face model:

```bash
MODEL=Qwen/Qwen3-0.6B scripts/run_container.sh scripts/quickstart_offline.sh
```

Existing local model:

```bash
MODEL_ROOT=/path/to/models MODEL_DIR=/workspace/models/Qwen3-1.7B scripts/run_container.sh scripts/quickstart_offline.sh
```

Pinned Hugging Face revision:

```bash
MODEL=Qwen/Qwen3-1.7B MODEL_REVISION=main scripts/run_container.sh scripts/quickstart_offline.sh
```

### Common Commands

Quick performance and accuracy check:

```bash
RUN_ACCURACY=1 scripts/run_container.sh scripts/quickstart_offline.sh
```

Full offline performance and accuracy matrix:

```bash
CONTEXT_LENS="1000 5000" \
BATCH_SIZES="1 2 4 8" \
REPEAT=3 \
RUN_ACCURACY=1 \
scripts/run_container.sh scripts/quickstart_offline.sh
```

Results are written to:

```text
benchmark_artifacts/sglang_compare/offline_perf_YYYYmmdd_HHMMSS/summary.md
benchmark_artifacts/sglang_compare/offline_accuracy_YYYYmmdd_HHMMSS/summary.md
```

### Other Reproduction Entrypoints

In a container where dependencies are already installed, run:

```bash
# Full offline performance comparison.
scripts/run_offline_perf_benchmark.sh

# Full offline correctness alignment.
scripts/run_offline_accuracy_benchmark.sh
```

Run a fair evaluation with radix on/off, performance, and correctness:

```bash
OUT_DIR=benchmark_artifacts/sglang_compare/fair_eval_correctness_quick \
CONTEXT_LENS="1000" \
BEAM_WIDTHS="256" \
BATCH_SIZES="1 4" \
PERF_REPEAT=1 \
CORRECTNESS_REPEAT=1 \
scripts/run_gr_sglang_fair_eval.sh
```

Run Nsight breakdown for a fixed case:

```bash
CONTEXT_LEN=5000 \
BEAM_WIDTH=256 \
REQUESTS=4 \
MAX_BATCH_SIZE=4 \
scripts/run_short_context_nsys_compare.sh
```
