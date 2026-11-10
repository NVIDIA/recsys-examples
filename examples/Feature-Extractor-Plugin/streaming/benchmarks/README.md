# Streaming Feature Computation Benchmarks: Flink CPU vs GPU (Mega-Flink)

Six performance cases (c1-c6) compare Flink on CPU with Flink plus Mega-Flink GPU operators on a replayed recommendation log; the GPU wins two of them (window aggregation, dimension-table lookup) and the CPU wins the other four once both sides are tuned. This document holds the shared setup and cross-case sections; each case has its own page with plans, SQL and results. The entry point for readers is "Part 5" of the parent README.

> Status: the numbers come from the latest rewrite of an internal measurement report. That rewrite, made after supplementary runs, has **not been independently re-reviewed**. An earlier version was independently reviewed and every comment was addressed. "Not verified" means no direct evidence was found.

## Contents

- [1. Overview](#overview)
- [2. Environment and dataset](#env)
- [3. Methodology](#method)
- [4. Cases grouped by result](#cases): [Group 1: GPU wins](group1-gpu-wins.md) (c2, c6), [Group 2: CPU wins](group2-cpu-wins.md) (c1, c3, c4, c5)
- [5. Higher parallelism and a larger dimension table](#parallelism)
- [6. Where the time goes (breakdown and nsys)](#breakdown)
- [7. Known limitations](#limits)
- [8. Reproduction](#repro)

<a id="overview"></a>
## 1. Overview

Saturated-input throughput (records/s), single GPU, parallelism 1, exclusive node, with `pipeline.object-reuse` enabled on both sides ("tuned"):

| Case | CPU | GPU | Faster |
| --- | --- | --- | --- |
| c1 filter / cleaning | 2,472,421 | 2,059,561 | CPU +20% |
| c2 HOP two-phase | 1,475,619 | 1,861,833 | GPU +26% |
| c2 TUMBLE | 1,613,516 | 2,020,259 | GPU +25% |
| c3 TopN | 1,363,629 | 1,231,075 (batch) | CPU +11% |
| c4 dedup | 1,033,538 | 850,292 (batch) | CPU +22% |
| c5 join | 291,797 | 253,233 (fused) | CPU +15% |
| c5 join-filter | 307,921 | 262,297 (fused) | CPU +17% |
| c5 join-agg | 293,385 | 236,056 (fused) | CPU +24% (GPU p50 latency about 23% lower) |
| c6 Lookup (INNER) | 1,299,187 | 2,284,322 | GPU +76% |

Percentage = faster throughput / slower throughput - 1. Every number is a median of 3 rounds.

**The configuration matters.** Under default settings (`pipeline.object-reuse=false`) the GPU is faster on c1 (1.03M vs 0.83M, GPU +23%); with object-reuse enabled on both sides the result flips to CPU faster. Object-reuse helps the CPU more than the GPU (c1: roughly x2.95 vs x2.0). Tune the CPU side before choosing. Each case page gives both default and tuned results.

<a id="env"></a>
## 2. Environment and dataset

**Hardware and software**

- GPU: a single NVIDIA L20 (driver for CUDA 12.4, native code built for sm_89).
- Flink 2.2.1; Mega-Flink is the matching build from the same source drop. JVM heap 16 GB, G1, default heap state backend, watermark delay 20 ms, everything else default (including the default buffer-timeout).
- Mostly parallelism 1; input read from local disk. Host CPU model and core count: not verified, to be added before publishing.
- GPU operator settings: batch of 4,096 rows, 5 ms maximum wait; the c6 device hash table is capped at 4,194,304 rows; for c6, batches smaller than 1,024 rows fall back to a JVM lookup.

**Dataset**

- [KuaiRand](https://kuairand.com/) (a public recommendation dataset with randomly exposed videos). We use the KuaiRand-27K user behavior log: 92,919,789 rows, sorted by `time_ms`, converted to int64 column files and replayed in-process in time order. Please follow the dataset's public page for citation and license terms (re-check the link before publishing).
- The video table has 32,038,725 rows. For c6, only the videos that appear in the replayed slice are loaded as the dimension table (1,039,412 rows for mid/saturated tiers, 413,436 rows for the low tier; two BIGINT columns, video_id and author_id). The CPU HashMap and the GPU device hash table use the same data.

<a id="method"></a>
## 3. Methodology

**Three load tiers**

| Tier | Input rate | Records |
| --- | --- | --- |
| Low | 20,000 rec/s | 600 thousand |
| Mid | 100,000 rec/s | 3 million |
| Saturated | unthrottled | 3 million |

Each tier is warmed up with 300 thousand records at the same rate before timing. Low and mid tiers are rate-limited, so both sides reach the target throughput and the tiers compare latency; the saturated tier compares throughput.

**Rounds and statistics**: each case x tier x mode runs 3 rounds with the mode order rotated; throughput is the median of 3; p50 / p95 / p99 are the per-round percentiles, then the median across rounds (not the values from the median-throughput round).

**Latency definition**: from the record's scheduled send time to its arrival at the sink. Rate-limited tiers use the scheduled time to avoid coordinated omission. In the saturated tier the source is back-pressured and latency reflects a backlog, so it is only useful for relative comparison. "<0.01" or "0.00" means below 0.01 ms (for c1 and c6, source to sink sits in one operator chain). A few c2 window outputs do not trigger a reference record (24,354 rows at mid/saturated, 19,205 at low) and are excluded from latency.

**Correctness check**: for each case, the SHA-256 of the final table of the CPU run and each GPU mode is compared per case x tier. All published results are MATCH (zero mismatches). The c5 join pair count was also checked against an independent Java program (1,284,869 pairs in the formal run). Note that the comparison is on the **final table**; the intermediate changelog row counts differ between CPU and GPU (GPU in-batch merging emits fewer or more intermediate updates), for example c3 saturated: CPU 965,196 vs GPU batch 876,508 rows; c4: 3,361,901 vs 3,302,645; c5-join-agg (default config): CPU 765,548 vs fused 825,672. Latency sample counts differ slightly as a result.

**Exclusive runs**: each job owns the whole node (one job at a time) with 1 GPU. A separate "concurrent screening" pass (4 jobs on one node at once) was used for screening only and is not used in the tables below. Same-model nodes differ little (mostly within +/-5%); concurrent-screening throughput was mostly 3-6% higher than exclusive, with an outlier for c3 GPU batch, so only exclusive results are shown.

**CPU baseline**: Flink's own best configuration, with identical parallelism and mini-batch settings on both sides. For c2 we measured both the default two-phase (Local+Global) and one-phase (`1p`) plans; for c4 we also measured CPU with mini-batch (`cpu-mb`); each side's best is the reference. "Tuned" means `pipeline.object-reuse=true` on both sides (plus `table.exec.rank.topn-cache-size=100000` for c3, because the saturated tier has about 200 thousand distinct users). No further per-case tuning was done for either side.

**Breakdown data has two kinds, with different standing**:

- JFR sampling (CPU-side share per stage) is a **sampled estimate**: grouped by thread stack, with sampling overhead; use it for the rough distribution only.
- The CUDA-event timers built into Mega-Flink operators (row-to-column, H2D, kernel, D2H, column-to-row) are **measured**. H2D/D2H include queuing and synchronization and read 1.5-2.5x higher than the copy time nsys reports.
- nsys was used as an independent check on a few cases: total kernel time agrees with the Mega-Flink kernel timer to within about 10%.
- Breakdown jobs ran 1 round per case with profiling on, so their throughput is slightly lower and is not used to compare speed.

<a id="cases"></a>
## 4. Cases grouped by result

The six cases are grouped by the tuned, saturated-throughput result in section 1. The case pages contain, for each case: the scenario, the full SQL, the real execution plans (CPU and GPU), what the plugin replaces, the plans per mode, evidence that the GPU ran, the result tables (default and tuned), the time composition and the takeaway. A case that was faster on the GPU only under default settings (c1) is listed in group 2, because the tuned result is the reference.

**Group 1: GPU wins** ([group1-gpu-wins.md](group1-gpu-wins.md))

| Case | Result in one sentence | Details |
| --- | --- | --- |
| c2 window aggregation (HOP and TUMBLE) | GPU is better on throughput and latency: after tuning HOP two-phase GPU +26% and TUMBLE +25%. | [c2](group1-gpu-wins.md#c2-window) |
| c6 dimension-table Lookup Join (INNER) | GPU +76% at saturation after tuning (+39% by default); at low/mid rates the throughput is equal and most batches use a JVM lookup. | [c6](group1-gpu-wins.md#c6-lookup) |

**Group 2: CPU wins** ([group2-cpu-wins.md](group2-cpu-wins.md))

| Case | Result in one sentence | Details |
| --- | --- | --- |
| c1 filter / cleaning | GPU +23% under default settings, CPU +20% after tuning. | [c1](group2-cpu-wins.md#c1-filter) |
| c3 TopN | CPU +11% over GPU batch after tuning. | [c3](group2-cpu-wins.md#c3-topn) |
| c4 keep-last dedup | CPU +22% after tuning. | [c4](group2-cpu-wins.md#c4-dedup) |
| c5 Interval Join (join, join-filter, join-agg) | CPU +15%, +17% and +24% after tuning (against GPU fused); with a following aggregation GPU p50 latency is about 23% lower. | [c5](group2-cpu-wins.md#c5-join) |

**How the plans differ between CPU and GPU.** For every exported plan pair, the Abstract Syntax Tree and the Optimized Physical Plan are the same on CPU and GPU (for c2 one-phase, compare with the CPU one-phase plan). The Optimized Execution Plan is where they differ: in GPU mode some nodes carry Mega-Flink names (for example `GpuBoundedLookupJoin`, `GpuNativeStreamExecLocalWindowAggregate`), and nodes that stay on the CPU are shown as `FlinkCpuStreamExecCalc(...)`. See the case pages for the real plan lines.

<a id="parallelism"></a>
## 5. Higher parallelism and a larger dimension table

Method: the same replay, with one lightweight Long sequence source distributing by `k mod p` to p replay maps (each subtask replays in time order); downstream operators run at parallelism p on both CPU and GPU, and the p subtasks share one L20. Only c2, c3 and c6 were run. Under default settings, 18 groups (case x tier x parallelism) were compared and all MATCH; the check status for the tuned parallelism 2/4 runs: not verified.

**Tuned, saturated throughput (records/s)** (parallelism 2/4 figures are taken from the internal report; this draft did not re-check them against the raw results one by one)

| Case | Parallelism 2 (CPU / GPU) | Parallelism 4 (CPU / GPU) | Faster |
| --- | --- | --- | --- |
| c2 HOP two-phase | 2,015k / 2,532k | 2,907k / 3,365k | GPU +26% / +16% |
| c2 HOP one-phase | 1,846k / 2,017k | 3,028k / 3,340k | GPU +9% / +10% |
| c3 TopN (GPU batch) | 2,517k / 1,824k | 3,827k / 3,154k | CPU +38% / +21% |
| c6 Lookup | 1,592k / 2,991k | 3,168k / 3,809k | GPU +88% / +20% |

c3 GPU adaptive: 980k at parallelism 2 and 1,545k at parallelism 4.

**c6 larger dimension table** (tuned, parallelism 1, table padded to 4 million rows; GPU lookup cap 4,194,304 rows): saturated CPU 1,292k, GPU 2,360k (+83%), gpu-min1 2,370k; at mid rate both sides keep up, GPU p50 3.1 ms and CPU <0.01 ms.

**Latency caveat**: with parallelism above 1 there is an extra network exchange between replay and operators (Flink's default buffer-timeout is 100 ms, so at low rates a buffer waits for the timeout before being sent). Low/mid p50 values are therefore much higher than at parallelism 1 (for example c3 low tier 83-126 ms vs 14.7 ms at parallelism 1). This is introduced by the test driver and is common to CPU and GPU; use low/mid latency at parallelism above 1 only for CPU-vs-GPU comparison.

<a id="breakdown"></a>
## 6. Where the time goes (breakdown and nsys)

- At parallelism 1, most of the time goes to Flink's own serialization / network exchange, source replay, state access and DataStream-Table row conversion; GPU kernels are a small share. For example, c3 GPU batch at saturation: the kernel takes about 0.12 s while the whole job (3.3M rows) takes about 3.2 s (formal median).
- The new cost on the GPU side is row-to-column conversion, H2D / D2H, column-to-row conversion and JVM-side batch packing/unpacking. State access, timers, changelog merging and Global aggregation stay on the JVM and are not replaced by the GPU.
- See section 3 for how sampled estimates differ from measurements. JFR samples only task threads whose name contains "(1/1)"; if the persistent form has a separate resident thread it is not counted, so its GPU share is indicative only.
- The CPU and GPU Abstract Syntax Trees and Optimized Physical Plans (Calc / Rank / WindowAggregate / IntervalJoin / LookupJoin structure) are identical; EXPLAIN's Optimized Execution Plan is where the GPU node names appear (for example `GpuBoundedLookupJoin`, `GpuNativeStreamExecLocalWindowAggregate`, with CPU Calc shown as `FlinkCpuStreamExecCalc`). The replacement happens when ExecNodes are translated into Transformations. Earlier versions of this document said EXPLAIN showed no difference; that was only true of the physical plan.
- Per-case time composition is in each case page.

<a id="limits"></a>
## 7. Known limitations

1. **Single GPU, mostly parallelism 1.** Parallelism 2 and 4 were run only for c2, c3 and c6; c4 and c5 at parallelism 2/4 were not measured; multi-GPU and higher parallelism were not measured.
2. **c6 with the full dimension table was not measured.** The full 32.04 million rows exceed the GPU lookup cap (4,194,304 rows); only the 1,039,412-row slice and a table padded to 4 million rows were tested.
3. **The rewritten report has not been independently re-reviewed.** The version rewritten after the supplementary runs (object-reuse tuning, parallelism, TUMBLE, nsys) has not been through independent review.
4. **LEFT lookup cannot run on the GPU in this build**, so there is no LEFT performance comparison.
5. **The c6 JFR share is not used because of a misclassification.** The "GPU-related" share needs re-checking for any case that uses the same package.
6. **JFR is a sampled estimate,** not an exact timing; the Mega-Flink timers and nsys are the measurements, and H2D / D2H include queuing.
7. **Tuned results do not cover every tier.** The tuned low tier (20k/s) was not measured; c4, c5 and c6 tuning covers only mid and saturated. Tuning is limited to `pipeline.object-reuse` (and the c3 topn-cache); no further per-case tuning was done, and other CPU-side options were not explored (apart from c4 `cpu-mb` and the c5-join-agg mini-batch).
8. **Default configuration is not the best configuration.** Conclusions under defaults (for example the GPU winning c1) change after tuning; state the basis when quoting.
9. **The check compares final tables,** and intermediate changelogs differ between CPU and GPU (see section 3).
10. **GPU persistent and adaptive:** no support was found for "choose persistent"; adaptive sends 0 rows to the GPU at low/mid rates. For c5 persistent, command counts were recorded only in the breakdown round and were 0 in the formal rounds; the reason is not verified.
11. **Not done:** other GPU models (for example H20), multi-GPU, larger dimension tables, other datasets. Only one dataset and one GPU model were used.
12. **c3 GPU batch shows two clusters of results at saturation** (see c3 in [group2-cpu-wins.md](group2-cpu-wins.md#c3-topn)); there is no finer evidence for the mechanism and none is claimed.

<a id="repro"></a>
## 8. Reproduction

The converter, the test driver and the run scripts are in this directory (`prepare_stream_data.py`, `driver/`, `scripts/`) and take their paths from environment variables; [REPRODUCE.md](REPRODUCE.md) gives the step-by-step procedure. The c2 and c6 GPU side can also be run with the library through the example jobs in `../examples`; the c1, c3, c4 and c5 GPU side still needs a Mega-Flink build.

What you need:

- One NVIDIA L20 (or another sm_89 GPU) with a driver supporting CUDA 12.4, on an exclusive node, with enough host memory for a 16 GB JVM heap plus the data files.
- Flink 2.2.1 and the matching Mega-Flink build (GPU operators and shim). How to obtain and build it: not verified, to be added.
- The KuaiRand-27K dataset (from its public page), and a converter that turns the log into int64 column files sorted by `time_ms` (`prepare_stream_data.py`; `driver/` also has a Java reference converter).
- A test driver that does the in-process replay, defines the SQL, collects the sink changelog, computes latency and does the SHA-256 comparison (`driver/`).
- Run scripts that rotate "case x mode x tier x round", summarize medians and run the comparison (`scripts/`).
- Optional: JFR and nsys for breakdown and kernel cross-checks.

Mode names (internal naming, given only to map to the result tables): `cpu`, `cpu-mb`, `cpu-1p`, `gpu`, `gpu-1p`, `gpu-batch`, `gpu-persistent`, `gpu-adaptive`, `*-fuse` / `*-nofuse`, `gpu-min1`; tuned modes add a `-tuned` suffix.

Notes:

- Do not edit a shell script while it is being read by a running job; internally this once caused a few rounds to be mislabeled as failed (the data were valid and were counted).
- Latency at rate-limited tiers is computed from the scheduled send time; saturated-tier latency is for relative comparison only.
- Concurrent runs change throughput (in screening, concurrent numbers were mostly 3-6% higher, with outliers for some cases); compare on an exclusive node.
