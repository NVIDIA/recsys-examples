<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Streaming library design

English | [中文](DESIGN.zh-CN.md)

Status legend: **In this directory** = code is here (`library/`, `library/native/`, `tools/`, `examples/`, `benchmarks/`). **Mega-Flink only** = exists in Mega-Flink, not extracted. **Not started**. How far the code was verified: [LIMITS.md](LIMITS.md). Items marked "unverified" were not tested.

## 1. Goals and non-goals

Goals:

- A small GPU operator library for Flink that a job calls explicitly, only for steps where a GPU pays off.
- Results checked against Flink's own CPU output.
- Operators extracted from the research plugin [Mega-Flink](https://github.com/cb521/Mega-Flink).

Non-goals:

- No override of Flink planner classes. No automatic rewrite of SQL plans.
- No automatic proof that a SQL fragment is safe to move. The caller picks the step.
- No operators that did not beat the CPU in Mega-Flink's own tests (see 4.3).

## 2. First version scope

**Window aggregation + bounded INNER lookup join** (decision 3, section 9). Other operators join only if a re-measurement shows a win. This matches the status table in the streaming README, section 3.

| Component | Status |
|---|---|
| Window aggregation (HOP, TUMBLE; one or two phase) | **In this directory**: `library/`, class `GpuWindows` |
| INNER lookup join, bounded dimension table | **In this directory**: `library/`, class `GpuJoins` |
| Native layer (CUDA, JNI) for the two operators | **In this directory**: `library/native/` |
| Builder API and shape validation | **In this directory**: checks run at `build()` |
| Dedup, TopN, interval join, INNER join with fused filter/aggregate, filter | **Mega-Flink only.** Optional later, only if a re-measurement shows a win |
| Table/SQL bridge | **Not started.** Not in the first version (decided); DataStream API only |
| Tools (qualification, profiling) | **In this directory**: `tools/` |
| Example jobs, benchmark driver and scripts | **In this directory**: `examples/`, `benchmarks/` (the GPU side of c1, c3, c4, c5 needs a Mega-Flink build) |

## 3. Public API

The DataStream API plus builders. A job passes a `DataStream<RowData>` and the Flink `RowType` that describes its rows to a builder, and gets a `DataStream<RowData>` back. Package `com.nvidia.fep.flink`.

```java
// window aggregation: key columns, aggregate columns, then window_start and window_end
DataStream<RowData> stats =
    GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
        .timeField("ts")
        .keyBy("user_id")
        .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_ms_sum"))
        .twoPhase()            // optional; also .offset(...), .windowProperties(...)
        .build();

// INNER lookup join of a BIGINT probe column against a bounded dimension snapshot
LookupDimension videos = LookupDimension.ofLongPairs("videos-v7", videoIds, authorIds);
DataStream<RowData> enriched =
    GpuJoins.lookupInner(events, eventType, "video_id", videos)
        .dimensionColumns("dim_video_id", "author_id")
        .build();
```

`GpuWindows.tumble(stream, rowType, size)` is the tumbling variant. Other builder options: `batchRows`, `maxWaitMs`, `gpuResource`, and for the join `gpuMinBatchRows` and `maxDimensionRows`. The class Javadoc of `GpuWindows` and `GpuJoins` lists every check.

- A builder checks the declared shape at `build()`: column and aggregate types, window parameters, at least one key column, one supported join type, batch settings. A failed check throws `GpuUnsupportedShapeException` (or `IllegalArgumentException` for an invalid argument such as an unknown column). There is no silent fallback in the middle of a run.
- **Append-only input cannot be proven at build time** from a `DataStream`. The operators throw when they see a non-INSERT row. Rows with a NULL time column are rejected the same way.
- Mega-Flink's planner proves these conditions today. In the library the builder does it, using the planner's own admission checks split out of its `tryCompile` code, so a missing check can produce wrong results (risk 1).
- Operators are ordinary Flink one-input stream operators. Batching and the barrier and watermark flush were carried over from Mega-Flink. Only the code that constructs the operators changed.
- Two-phase aggregation (stateless GPU combiner, then keyed merge) is one builder switch, `.twoPhase()`. It is marked experimental in Mega-Flink; the library exposes it because our measurements covered it.
- **Table bridge** (`GpuTables`): not started and not in the first version (decided). Conversion cost is unverified (risk 3).

Alternatives considered, not chosen:

- Table API process table functions (Flink 2.1+). They are called per row inside a partition and cannot reach operator hooks such as batching before a checkpoint barrier. Unverified on 2.2.1.
- SQL hints with a custom planner extension. Flink has no public hook to replace physical operators, so this still uses planner internals and ties the library to one Flink version. It could be an optional module later.

## 4. What is extracted from Mega-Flink

### 4.1 Java

- 51 classes and about 15,000 lines in `library/src/main` (43 classes moved from Mega-Flink, trimmed where needed, plus 8 new facade classes), and 12 test classes with about 2,100 lines. Mega-Flink's runtime package has a few hundred classes and about a hundred thousand lines.
- Kept:
  - the operators for the first version (one-phase window, two-phase local and global window, bounded lookup join);
  - packing between rows and columns, device memory, resource lifecycle;
  - Java wrappers of the native kernels (`DenseGroupbyKernel`, `LookupJoinKernel`, `CudaDeviceBinding`);
  - serializable plan descriptions (data and validation only);
  - metrics helpers.
- Rewritten:
  - plan classes lost their Calcite and planner entry points. `AggregateCallSpec` replaces Calcite's `AggregateCall`;
  - the planner-generated key selector is replaced by `FieldRowDataKeySelector`, and the planner's snapshot-provider contract by `LookupDimension`;
  - the Calc and Expand fusion paths were cut from the operators and from `DenseGroupbyKernel`.
- Not rewritten: JVM system-property reads (for example `cudf.flink.window.partial-cache.entries`, `cudf.flink.lookup-join.library`) are still system properties, not builder parameters. GPU device acquisition is a builder option (`gpuResource`, default: Flink external resource `gpu`).

### 4.2 Native (CUDA, JNI)

- Four `.cu` files and two headers, 7,815 lines: device binding, grouped aggregation (including the device-batch runtime it links against) and the lookup join. They are copied whole, not trimmed. `build.sh` builds four shared libraries from them: `libcudf_flink_device.so`, `libcudf_flink_device_batch.so`, `libcudf_flink_groupby.so`, `libcudf_flink_lookup_join.so`.
- Not taken: the generic Calc interpreter, expand, fused filter/project, hash join, nested-loop join, TopN, dedup, over, row number, temporal sort and the benchmark executables.
- Only the CUDA Toolkit is needed. CUB is not used by these files.
- JNI symbol names contain the Java package name. They were renamed by prefix replacement (`Java_io_github_cb521_cudfflink_` to `Java_com_nvidia_fep_flink_`).
- One change in the code: device admission in `gpu_arch_compat.h` (see [LIMITS.md](LIMITS.md)).
- Because the files are whole, the libraries export JNI symbols that no Java class declares (dead code, see LIMITS).

### 4.3 Cut

- The Flink-internal override layer, the Flink 2.2.1 installer and bridges.
- Benchmark code and reference implementations (the benchmark driver here is new).
- Operator islands (experimental, no performance admission in Mega-Flink's own records).
- Operators that did not beat the CPU in Mega-Flink's tests: standalone Calc, LEFT/SEMI/ANTI join, cross join, COUNT DISTINCT, SESSION window, over aggregate (in the first version).
- The Calc interpreter on the Java side. Mega-Flink's own result is that it is off by default and did not win end to end.
- The adaptive controller and the persistent paths. The library runs in batch mode only.
- In the Java code, CUMULATE, SESSION, DISTINCT, FILTER, SUM0 and LEFT lookup join exist internally but are not offered by the builders.

### 4.4 Flink dependencies

- `flink-streaming-java` and `flink-table-runtime`, both `provided`. No planner, no Calcite. Tests also use `flink-runtime` and JUnit.
- Flink 2.2.1 is the version built and tested. The Java release level is 11. 2.0 and 2.1 are unverified (decision 4).

## 5. Directory layout

```text
streaming/
  README.md, README.zh-CN.md   method and results
  docs/                        this file, LIMITS.md, how-it-works.md, bottlenecks, related work, figures
  library/                     Maven module fep-flink-gpu (package com.nvidia.fep.flink)
    pom.xml                    Flink dependencies are provided
    src/main/java/             facade (GpuWindows, GpuJoins, ...) and runtime classes, same package
    src/test/java/             CPU unit tests (no GPU needed)
    NOTICE
    native/                    CUDA and JNI sources, build.sh, selftest.sh, selftest/
  examples/                    two KuaiRand example jobs (WindowFeaturesJob, LookupAuthorJob)
  tools/                       qualification and profiling (fep-flink-tools)
  benchmarks/                  results and method; driver/, scripts/, cases/, prepare_stream_data.py, REPRODUCE.md
```

The extracted classes use package-private access heavily, so they stay in one package and the public API is a thin facade over it.

## 6. License and NOTICE

- Apache-2.0. Every new or extracted source file, Java and CUDA, gets an SPDX header: `SPDX-License-Identifier: Apache-2.0`.
- Mega-Flink is Apache-2.0. Its NOTICE names "Mega-Flink contributors" as copyright holder. The copyright line here is NVIDIA (decided).
- NOTICE text (in `library/NOTICE`, `library/native/NOTICE` and `tools/NOTICE`): "Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0)", plus the Apache Software Foundation attribution line, kept as a precaution.
- Mega-Flink marks only its Flink override layer as modified Apache Flink code. That layer is not extracted. A keyword search of the kept sources for "adapted from", "copied from" and "based on flink" found nothing. We did not compare line by line, so "no Flink-derived code" is unverified.
- The CUDA Toolkit is a build requirement and is not bundled. The tools do not use XGBoost.

## 7. Tools

`fep-flink-tools`: Python 3, standard library only, no GPU and no Flink jar. Read-only: they read the Flink REST API, a history archive or a saved snapshot. Version 0.1.0.

- `qualification`: maps the job's SQL physical nodes to the library's operators and reports how much of the compute time sits in them.
- `profiling`: finds hot spots, back pressure and skew.
- The operator-family file lists four library-provided node names (window aggregate and its local and global split, lookup join) and maps them to `GpuWindows.tumble` / `GpuWindows.hop` and `GpuJoins.lookupInner`. Everything else is `not-provided` or `cpu-boundary`.
- The Flink version gate is 2.2.1; other 2.x are unverified. The self-test (`tests/self_test.py`) runs on synthetic fixtures and passes.
- **Limit**: they match operators by SQL physical node name. A pure DataStream job, where the user names the operators, is not identified. They answer "which part of an existing SQL job could move to the library", not "which DataStream step". They have not been run on a real production job.
- `compare`, `calibrate`, `train` (XGBoost, need paired measurements) were not taken over (decided).

## 8. Risks

1. **Semantic proof moves from the planner to the builder.** A missed check can give wrong results. Mitigation: the builder reuses Mega-Flink's admission checks, and each supported shape is compared against Flink CPU output. Append-only input is enforced at run time, not at build time.
2. **`@Internal` Flink classes** (key selector, slice assigners, `InternalTypeInfo`, clock service) can change between minor versions. A compile matrix over 2.x versions is needed if more than 2.2.x is supported.
3. **Table bridge conversion cost is unverified.** If it is not free, it eats part of the gain.
4. **Dead code.** The native files are whole, so some exported JNI functions have no Java caller. Trimming them is a possible second step.
5. **State is not compatible.** Operator state names follow Mega-Flink. A savepoint of a SQL job cannot be restored into a job that uses the library. A job moved to the library starts without old state.
6. **Defaults need a re-check.** Our measurements did not show a gain from persistent or adaptive modes (see LIMITS.md). The lookup join's `gpuMinBatchRows` default is 1,024, the value of the benchmark runs; its `batchRows` default (1,048,576, carried over from Mega-Flink) is not.

## 9. Decisions

| # | Decision | Outcome | Status |
|---|---|---|---|
| 1 | Java package name | renamed to `com.nvidia.fep.flink`; a prefix replacement, so JNI symbols follow the same rule | **Decided** |
| 2 | Copyright line and NOTICE wording | NVIDIA; conservative NOTICE as in section 6 | **Decided** |
| 3 | Scope of the first version | window aggregation and bounded INNER lookup join; everything else optional, only if a re-measurement shows a win | **Decided** |
| 4 | Flink versions | 2.2.x only | **Decided** |
| 5 | Table bridge in the first version | no; DataStream API only | **Decided** |
| 6 | XGBoost subcommands in Tools | dropped | **Decided** |
| 7 | Calc interpreter in native code | cut in the first version | **Decided** |
