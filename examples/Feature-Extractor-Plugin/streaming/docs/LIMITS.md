<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Known limits (streaming)

English | [中文](LIMITS.zh-CN.md)

Each item names its source: **README** = the streaming README, **REPORT** = the benchmark report ([benchmarks/README.md](../benchmarks/README.md)), **DESIGN** = [DESIGN.md](DESIGN.md), or a file in this directory. This list is not complete.

Two kinds of evidence meet here. The throughput numbers in the README and REPORT come from the Mega-Flink build, not from the library in `library/`. For the library we checked only that its results equal the CPU results (next section). Limits in the sections after that describe the Mega-Flink build we measured unless a source says "library".

## What was verified for the library

| Item | Source |
|---|---|
| Java unit tests: 63 pass without a GPU (`mvn -q test` in `library/`). They build job graphs and serialize the operator factories; tests that call the native kernels are not included. | `library/src/test` |
| Native self-test (`library/native/selftest.sh`, no JVM): `SELFTEST_PASS`, 13 checks, on one L20 with a build for sm_89 only and with a build for "89 90a". | `library/native/selftest/` |
| End to end through a JVM with the example jobs: one NVIDIA L20 (sm_89), CUDA 12.4, parallelism 1, embedded local mini cluster, KuaiRand-27K data, 300,000 warm-up rows and 3,000,000 measured rows. The SHA-256 of the sorted final table was equal between the CPU run (Flink SQL) and the GPU run (the library) in all four cases: c2 HOP one-phase and two-phase (718,449 rows, `60fccd33`), c2 TUMBLE 5 min (283,544 rows, `6dcf4c11`), c6 INNER lookup of `author_id` (3,300,000 rows, `fe6b958a`). Each pair ran in separate JVMs. These digests also equal the final-table digests of the earlier Mega-Flink runs of c2-hop, c2-tumble and c6. | `examples/README.md` |
| Throughput of the example jobs is not reported here: the two engines feed different row objects (`Row` for SQL, `RowData` for the library) and each number is one run. | `examples/README.md` |

Not verified for the library: any GPU other than the L20, multi-GPU, parallelism above 1, a cluster, Kafka or other real sources, other datasets, Flink versions other than 2.2.1, RocksDB state, checkpoint and restore, and the `gpu` external-resource lookup (the examples select device 0 with `GpuResourceMode.LEGACY_DEVICE`).

## Operator shape

| Limit | Source |
|---|---|
| Window aggregation needs at least one key column. Mega-Flink's own check does not require it; the keyless path was not tested, so the builder rejects it. | `GpuWindows` Javadoc |
| DISTINCT aggregates are rejected: the one-phase operator has no distinct state and the two-phase local precheck rejects them. Only COUNT, SUM, AVG, MIN and MAX are offered. | `GpuWindows`, `Agg` |
| Only event-time TUMBLE and HOP. CUMULATE and SESSION windows are not offered. | `GpuWindows` |
| Input must be append-only. This cannot be proven when the job graph is built; the operators throw on the first retraction or update row, and on a NULL time column. | `GpuWindows`, `GpuJoins` Javadoc |
| The output column types and nullability are derived from Flink SQL rules (SUM, AVG, MIN, MAX keep the input column type, COUNT is BIGINT NOT NULL, `window_*` are TIMESTAMP(p) NOT NULL). They were not compared with the planner's output. | `GpuWindows` |
| Two-phase window aggregation is an experimental feature in Mega-Flink (it needs the experimental-operators switch there). The library exposes it as `.twoPhase()` because it was measured. | `GpuWindows.Builder.twoPhase` |
| Lookup join is INNER only, one BIGINT probe key, and a dimension of two BIGINT columns with unique keys. Other key types and wider rows are not supported. LEFT lookup join is not offered. | `GpuJoins`, `LookupDimension` |
| The dimension is bounded: at most 1,048,576 rows by default (`maxDimensionRows`); more rows fail the job. The GPU lookup table limit in the Mega-Flink build was 4,194,304 rows. We loaded 1,039,412 rows in the main runs and padded to 4M rows in one run. The full 32M-row table was not tested. | `GpuJoins`, README 2.4, REPORT |
| The `gpuMinBatchRows` setting decides whether a lookup batch runs the GPU kernel: batches smaller than it are answered by a JVM lookup. The library default is 1,024 rows (the value of the benchmark runs; `batchRows` defaults to 1,048,576); the example job `LookupAuthorJob` also sets `--batch-rows 4096 --gpu-min-batch-rows 1024`. Measured with the operator counters on one L20 (parallelism 1, 3,300,000 rows, saturated replay): with 1,024, 809 batches ran the GPU kernel and 2 were answered by the JVM; with 1 (every batch on the GPU), 812 batches ran the kernel and none by the JVM; nsys counted 809 and 812 `lookup_kernel` launches. The final-table SHA-256 equals the CPU run in both settings. The saturated replay fills batches of 4,096 rows, so this says nothing for low or mid rates, where batches are flushed by the 5 ms wait with fewer than 1,024 rows and mostly use the JVM (see the c6 section of the benchmark report). With the library defaults, the operator code sends every batch below 1,048,576 rows to the JVM, so this replay would use it almost always (inferred from the code, not measured). | `GpuJoins`, REPORT |
| Probe rows are not repartitioned: every parallel subtask loads the whole dimension. | `GpuJoins` Javadoc |
| LEFT/SEMI/ANTI join, standalone Calc, COUNT DISTINCT, SESSION window and cross join are not in the library. Dedup, TopN, filter and interval join stay in Mega-Flink. | README 3.1 |

## Native layer

| Limit | Source |
|---|---|
| Device admission was widened. Mega-Flink admitted only a Hopper device whose name contains "H20". `build.sh` now defines `FEP_ADMITTED_DEVICE_MAJORS_MASK` from the architectures it compiles, so the library admits any GPU whose compute-capability major was compiled in (major 8 for `--archs 89`, major 9 for `90a`), not only the card names we tested. | `library/native/gpu_arch_compat.h`, `build.sh` |
| The 90a (Hopper) code was compiled but never run on a Hopper GPU. | `library/native/build.sh`, selftest |
| The `.cu` files are whole copies, so the libraries export 58 JNI functions and the Java classes declare 30 of them. The other 28 are dead code: 18 belong to Java classes that are not in the library (`GpuDenseLongAggregateKernel`, `PersistentAggregateKernel`, `GpuDeviceBatch`, `LimitKernel`) and 10 are Calc and device-batch entries that `DenseGroupbyKernel` no longer declares. | `library/native/*.cu`, `DenseGroupbyKernel` |
| JNI argument and return types are checked by name matching and by the self-test, which covers only some methods (device binding, lookup join, grouped aggregation with COUNT, SUM, MIN, MAX). Other paths, such as STRING lanes and the window buffers of two-phase HOP, are covered only by whatever the end-to-end runs above touch; no type-level check was made for them. | `library/native/selftest/` |
| The self-test must be run by hand; it needs a GPU and a JDK's `jni.h`. | `library/native/selftest.sh` |

## Versions, hardware, data

| Limit | Source |
|---|---|
| Only Flink 2.2.1 was verified. Whether 2.0 and 2.1 compile is unverified. Flink 1.x is not a target. Some Flink classes the library uses are `@Internal` and may change. | README, DESIGN 4.4 |
| One NVIDIA L20, single GPU. Other GPUs and multi-GPU are not tested. | README |
| Parallelism was mainly 1. Parallelism 2 and 4 were measured for some Mega-Flink cases only (window aggregate, TopN, lookup join). Above 4 is not measured. Dedup and interval join at parallelism 2/4 are not measured. The library was run at parallelism 1 only. | README 2.4, REPORT |
| Single machine only. No cross-machine test, no cluster. The library was run in a local mini cluster. | README |
| No Kafka test. Input is a replay of the KuaiRand log. Real source formats are not measured. | README 1, REPORT |
| Only the KuaiRand dataset. Other datasets and heavier expressions are not measured. | README 2.4 |
| Heap state only. RocksDB state is not measured. | README |
| Saturated input, 3M records after 300k warm-up, median of 3 rounds. Low and medium rate runs are rate-limited, so throughput there only shows that both sides keep up. | README, REPORT |

## Behaviour

| Limit | Source |
|---|---|
| In the Mega-Flink build the default mode is `cpu-row` (original Flink nodes). The GPU is used only when `gpu-native` is set explicitly. The library has no such switch: a job that calls `GpuWindows` or `GpuJoins` uses the GPU operator. | related-work.md |
| Results compared are the final tables (SHA-256, all matched). The intermediate changelog rows of CPU and GPU differ in count, because the GPU merges updates inside a batch. | REPORT |
| Persistent (resident kernel) mode showed no gain in our data. On TopN at saturation it reached 0.43M rows/s against 1.36M on the CPU. On interval join it was worse at every rate. The library has only the batch mode. | README 2.5 |
| Adaptive mode showed no gain. At low and medium rates 0 rows went to the GPU. We found no case where it beat the better of CPU and batch. The library does not carry it. | README 2.5 |
| Fusion gave no throughput gain for interval join followed by filter or aggregate (join +0.8%, join+filter +1.1%, join+aggregate -7%). It lowered median latency only for join+aggregate. Other operator pairs are unverified. | README 2.6 |
| The GPU gain depends on a tuned CPU baseline. With `pipeline.object-reuse` off the filter result flipped. Other CPU options were not explored. | README 2.3 |
| Time breakdowns come from JFR sampling in an earlier run with Flink defaults. They are estimates. | README 1 |
| The GPU side of c1, c3, c4 and c5 in `benchmarks/` needs a Mega-Flink build. | benchmarks/REPRODUCE.md |

## Tools

| Limit | Source |
|---|---|
| Qualification and profiling match operators by SQL physical node name. A pure DataStream job is not identified. | README 3.1, DESIGN 7 |
| The tools are gated to Flink 2.2.1. Other 2.x versions are unverified. | DESIGN 7 |
| Tested only on synthetic fixtures; not run on a real production job or a real History Server archive. The review-priority thresholds (60% and 25%) are heuristics and were not validated against measured GPU gains. | tools/README.md |

## State and results

| Limit | Source |
|---|---|
| Operator state names differ from Flink's own operators. A savepoint of a SQL job cannot be restored into a job that uses the library. | README 3.2, DESIGN 8 |
| The benchmark report was rewritten after review. The rewrite has not been independently re-reviewed. | README |
| Table bridge conversion cost is unverified. | DESIGN 8 |
