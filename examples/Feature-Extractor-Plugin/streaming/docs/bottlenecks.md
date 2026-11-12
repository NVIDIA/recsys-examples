<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Where Flink jobs usually spend their time

English | [中文](bottlenecks.zh-CN.md)

Details behind section 1 of the [README](../README.md).
Each bottleneck class lists where time goes, public evidence, and our measurement. Each public number comes from one company and one scenario. We quote it as published; it is not a general percentage ("vendor" means the vendor measured its own product). Our numbers come from this repo's benchmarks (see benchmarks; conditions at the top). "Not measured" means we claim nothing.

## Serialization and copying between operators

- **Time goes to**: Kryo/POJO serialization, deep copy between operators, encode/decode around exchanges. Usual fixes: POJO/Tuple/Row instead of Kryo, object reuse, chaining, column pruning.
- **Public**: [Flink blog](https://flink.apache.org/2020/04/15/flink-serialization-tuning-vol.-1-choosing-your-serializer-if-you-can/) (micro-benchmark): a POJO that falls back to Kryo costs 75%. [Iron Vector](https://docs.google.com/document/d/1ledkVUcWSlx5l4yLaK9nmdIoOPMpUd_1GbjBezbNMPs) (vendor, one Kafka-Avro-to-SQL-to-blackhole pipeline): object reuse alone raised native Flink throughput by 35% to 40%.
- **Ours** (c1 simple filter): turning object-reuse on raised CPU from 0.84M to 2.47M rows/s and GPU from 1.03M to 2.06M rows/s. Serialization/exchange is 29% of sampled time on both sides. The direction matches, but our gain (about 3x on a trivial filter) is much larger than the public figure. Do not extrapolate either.

## State access

- **Time goes to**: with RocksDB, per-access serialization, seek, compaction and disk I/O; with heap state, hash lookups and object access. Usual fixes: cache hot keys, key-value separation, other state backends, rewriting a join as a lookup, TTL, mini-batch.
- **Public** (almost all RocksDB, large state): [Zalando](https://engineering.zalando.com/posts/2026/07/migrating-ad-event-processing-to-flink.html): Interval Join spent 15% to 30% of CPU on seek. [Taobao](https://www.alibabacloud.com/blog/the-delta-join-in-apache-flink-architectural-decoupling-for-hyper-scale-stream-processing_602645): Delta Join removed 50 TB of join state and saved 80%+ of CPU and memory.
- **Ours** (heap state, JFR estimate): state is 52% (CPU) and 43% (GPU) of sampled time in c5 Interval Join; 32% / 40% in join + aggregate; 14% (CPU) in c2 window aggregate; 14% / 17% in c4 keep-last dedup. GPU state stays in the JVM, so these cases show little or no GPU gain. The direction matches; the cost structure differs (public: RocksDB; ours: heap).

## Source, sink and format parsing

- **Time goes to**: JSON/Avro/PB deserialization, connector threads, external writes. Usual fixes: binary formats, pruning unused columns or fields, connector parallelism, batched sink writes, consumer fetch tuning.
- **Public**: [ByteDance/Douyin](https://developer.aliyun.com/article/796471): pruning the PB schema of Kafka sources saved about 30% CPU in most jobs (a CPU saving, not a share of run time). [Iron Vector](https://docs.google.com/document/d/1ledkVUcWSlx5l4yLaK9nmdIoOPMpUd_1GbjBezbNMPs) (vendor, Avro source decoding): 274 to 528 MiB/s (+93%), but only after tuning Kafka fetch settings; the first run was 23% slower.
- **Ours**: local replay of int64 columns, no text parsing. The source thread takes 15% (CPU) and 18% (GPU) of sampled time in c1, and 10% (GPU side) in c3. Kafka and real formats not measured. Partly consistent: our source is far lighter than Kafka plus JSON/PB decoding, so it understates this class.

## Row/column conversion (and the extra H2D/D2H on a GPU)

- **Time goes to**: converting RowData to the columnar format a vectorized or GPU engine needs, and back. On a GPU it also means H2D and D2H copies. Usual fixes: stay columnar end to end, enter and leave the device rarely, convert only used columns, batch.
- **Public** (no public Flink-on-GPU figure found): [Ant Flex](https://segmentfault.com/a/1190000046642882) (CPU vectorized): converting all fields cost nearly 11%; converting only the fields the expression uses cost 6.68%. [Gluten-Flink issue](https://github.com/apache/gluten/issues/12426): "two row-column conversions per batch", no number.
- **Ours** (c1 filter, Flink-defaults run, GPU-side timers): five timed GPU steps total 499 ms: row-to-column 254, H2D 20, kernel 25, D2H 14, column-to-row 186. Conversion is 88% of the five; kernel is 5%. Not counted: JVM-side processing and collect, 1,725 and 1,096 ms. nsys agrees on kernel and copy time within about 10%. On c5 Interval Join (GPU), column-to-row takes 843 ms of 1,107 ms timed. Ant's 11% is a CPU engine and we have not verified why it is lower, so it does not support our figure. This cost belongs to GPU offload, not to Flink in general.

## Data skew

- **Time goes to**: a few subtasks saturate while others idle, and upstream backs up. Usual fixes: two-phase aggregation, key salting, load-aware routing.
- **Public**: [Rippling](https://www.rippling.com/blog/eliminating-data-skew-in-flink) (CDC pipeline): 60% of load sat on 2 subtasks; fixing it raised throughput by 38% and cut run time from 17 h to 12.4 h. **Ours**: not measured (parallelism 1 cannot show skew).

## Checkpoint and recovery

- **Time goes to**: state upload, barrier alignment, metadata serialization, restoring large state. Usual fixes: incremental checkpoints, changelog, unaligned checkpoints with buffer debloating, disaggregated state (ForSt).
- **Public**: [Netflix](https://www.slideshare.net/StevenWu5/scaling-flink-in-cloud-94274600) (200-node test cluster): a 21 TB savepoint took 27 minutes; fixing S3 HEAD requests cut checkpoint duration 10x. [Alibaba VLDB'25](https://www.vldb.org/pvldb/vol18/p4846-mei.pdf) (vs Flink 1.20): up to 94% shorter checkpoints and up to 49x faster recovery. The evidence is mostly about stability, tail latency and recovery, less about steady throughput. **Ours**: not measured.

## Network and backpressure

- **Time goes to**: cross-machine transfer, serialization, buffer queueing. Usual fixes: chaining or multiple-input to drop shuffles, task placement, credit-based flow control, buffer debloating.
- **Public**: [Pinterest](https://www.slideshare.net/slideshow/tuning-flink-clusters-for-stability-and-efficiency/266280602): better task placement cut cross-host traffic by 70%+ and job cost by about 75% (not all of the cost came from the network). **Ours**: single machine, parallelism 1 (2 and 4 for some cases). Exchange is counted inside "serialization/exchange" (12% to 34% of sampled time across cases). No cross-machine test. Partly consistent.

## Garbage collection

- **Time goes to**: full GC with large heap state; pauses cause heartbeat timeouts and backpressure. Usual fixes: RocksDB or off-heap state, more memory or parallelism, another collector.
- **Public**: only thresholds and qualitative statements. For example, [Tencent Cloud](https://www.tencentcloud.com/document/product/1000/56810) treats full GC above 30% of a 5-minute window as severe (an alert threshold, not a measurement). No credible quantified case found. **Ours**: not measured separately (16 GB heap, G1).

## JVM pipeline ceiling (per-record path)

- **Time goes to**: even with no computation, one source thread, exchange and one sink thread cap single-parallelism throughput. Usual fixes: higher parallelism, chaining, less per-record work.
- **Public**: no direct figure. Vendor native engines report large Nexmark gains (Alibaba [Flash](https://developer.aliyun.com/article/1634363) 5x+; [VERA-X](https://ververica.com/blog/vera-x-introducing-the-first-native-vectorized-apache-flink-engine) 5x to 10x vs Flink 1.19), but these include many factors beyond the per-record path.
- **Ours** (c0-pass probe: source + exchange + sink only, 3.3M rows, Flink defaults): CPU 0.856M rows/s, GPU 0.851M rows/s. For Top-N on GPU (c3), kernels take about 3.6% of job time and the GPU is idle over 95%; the JVM side bounds throughput. This is our own finding.

## Time breakdown of our test cases

Share of sampled time (%), JFR estimates from the object-reuse-off run. "GPU-related" means GPU operator code and copies. c6 (dimension lookup) is omitted because the GPU-share classifier misattributed some CPU frames.

| Case | CPU side | GPU side |
|---|---|---|
| c1 filter | operator 29, serialization/exchange 29, row conversion 20, source 15 | serialization/exchange 29, operator 21, source 18, GPU-related 13, row conversion 9 |
| c2 window aggregate (GPU: single-stage) | operator 35, serialization/exchange 28, state 14 | GPU-related 38, serialization/exchange 29, operator 12 |
| c3 per-user Top-N | operator 45, serialization 34, state 3 | serialization 30, operator 24, GPU-related 23, source 10 |
| c4 keep-last dedup | serialization 34, operator 22, state 14, sink 10 | serialization 34, operator 20, state 17, GPU-related 7 |
| c5 Interval Join | state 52, operator 27, serialization 12 | state 43, operator 26, GPU-related 15 |
| c5 join + aggregate | state 32, operator 31, serialization 28 | (fused) state 40, operator 24, GPU-related 21 |
