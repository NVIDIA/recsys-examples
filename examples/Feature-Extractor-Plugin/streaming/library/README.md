<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flink GPU operators (DataStream library)

- A small Java library with two GPU operators that a Flink DataStream job calls explicitly: event-time window aggregation (TUMBLE and HOP, one phase or two phase) and INNER lookup join against a bounded dimension snapshot.
- Entry points are builders, `GpuWindows` and `GpuJoins`. `build()` checks the declared shape (types, window parameters, key and aggregate combinations, batch settings) and throws `GpuUnsupportedShapeException` with the reason. There is no automatic fallback to a CPU operator.
- Input must be append-only (`RowKind.INSERT`); this is a property of the data, so the operators throw when they see a retraction or update row.
- Requires Flink 2.2.x and JDK 11 or newer. `flink-streaming-java` and `flink-table-runtime` are `provided`; there is no planner or Calcite dependency, and no Table/SQL bridge in this version.
- The operators call native CUDA libraries through JNI. Their sources are in [native/](native) (`native/build.sh` builds them); the Java side and its CPU tests build and run without a GPU (`mvn -q test`). Verified range: one NVIDIA L20 (sm_89), CUDA 12.4, parallelism 1, local mini cluster, KuaiRand data; the final tables of c2 HOP (one and two phase), c2 TUMBLE and c6 INNER lookup have the same SHA-256 as the Flink SQL CPU runs (see the [examples](../examples/README.md)). Not verified: other GPUs (the sm_90a build was only compiled), multi-GPU, parallelism above 1, a cluster, Kafka, checkpoint restore. Full list: [LIMITS](../docs/LIMITS.md). Derived from [Mega-Flink](https://github.com/cb521/Mega-Flink); see [NOTICE](NOTICE).

## Minimal example

```java
RowType eventType = RowType.of(
        new LogicalType[] {new BigIntType(false), new BigIntType(), new TimestampType(3)},
        new String[] {"user_id", "play_ms", "ts"});

// events: DataStream<RowData> whose rows follow eventType, with watermarks on column "ts"
DataStream<RowData> stats =
        GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
                .timeField("ts")
                .keyBy("user_id")
                .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_ms_sum"))
                .build();   // user_id, cnt, play_ms_sum, window_start, window_end
```

`.twoPhase()` switches to a stateless GPU combiner followed by a keyed merge. `GpuJoins.lookupInner(probe, probeType, "video_id", LookupDimension.ofLongPairs(identity, keys, payloads))` adds the lookup join. The class Javadoc of `GpuWindows` and `GpuJoins` lists every check.
