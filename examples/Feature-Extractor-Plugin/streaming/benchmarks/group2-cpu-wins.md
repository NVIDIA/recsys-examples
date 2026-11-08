# Group 2: CPU wins (c1, c3, c4, c5)

[中文](group2-cpu-wins.zh-CN.md) | English

This group has four cases: c1 filter / cleaning, c3 per-user TopN, c4 keep-last dedup, and c5 Interval Join (with the fused variants join-filter and join-agg). In all of them the CPU has the higher throughput under the tuned configuration (`pipeline.object-reuse=true` on both sides). The cost is not in the computation. It sits in serialization, JVM state access and row/column conversion, and the GPU does not replace those. For environment, dataset, load tiers, statistics and the correctness check, see the methodology section of [benchmarks/README.md](README.md#method).

> Status: the numbers are copied from the earlier combined benchmark document. They come from a rewrite of an internal measurement report that has **not been independently re-reviewed**. "Not verified" means no direct evidence was found.

How to read this page:

- Throughput is in records/s, latency in ms. "Default" is Flink's default configuration; "tuned" has object-reuse on both sides (c3 adds `table.exec.rank.topn-cache-size=100000`). Every throughput is a median of 3 rounds. Percentage = faster throughput / slower throughput - 1.
- **Plans.** Each case lists the real plan of every measured mode. EXPLAIN prints three stages. The **Optimized Physical Plan is the same for CPU and GPU in every case here** (we compared the saved plan texts; the only exception is c4 `cpu-mb`, which adds a mini-batch node). The **Optimized Execution Plan is where the replacement happens**: the CPU node name is replaced by a `Gpu*` node name. Earlier text said "EXPLAIN shows no difference between the CPU and GPU plans". That is only true of the physical plan. Plan lines below are copied from the saved plan files; only the long `fields=[...]` lists are cut and marked with `…`.
- **Gpu\* nodes.** The node names and the bracketed parameters are defined by Mega-Flink. The Mega-Flink operator coverage document is the source for what we say about them; where it gives no basis we write "defined by Mega-Flink". In every GPU plan, `mode=gpu-native` is a label defined by Mega-Flink, and `engine=batch|persistent|adaptive` matches the per-operator engine settings of the coverage document (for example `join.engine`, `topn.engine`, `deduplicate.engine`). `batchRows=4096, maxWaitMs=5` are the batch size and the maximum wait; they equal the GPU operator settings of the test environment (a batch of 4,096 rows, 5 ms maximum wait), and the coverage document describes `batch.rows` / `batch.max-wait-ms` as the settings whose waiting the user must accept. `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` is a Calc that stays on Flink's CPU implementation. The coverage document lists `calc.engine=auto|cpu|persistent|batch` and says `auto` uses Flink CPU under the latency-safe policy; which policy these runs used: not verified.
- **Proof that the GPU ran.** The Mega-Flink operators expose runtime counters (batches, H2D time, compute time, kernel count, persistent command count). We quote them per case. They are runtime evidence, not EXPLAIN labels.

Contents:

- [c1 Filter / cleaning](#c1-filter)
- [c3 Per-user top 10 by watch time (TopN)](#c3-topn)
- [c4 Keep-last dedup](#c4-dedup)
- [c5 Interval Join and its fusion](#c5-join): [join](#c5-join-only), [join-filter](#c5-join-filter), [join-agg](#c5-join-agg)
- [Summary of the group](#summary)

<a id="c1-filter"></a>
## c1 Filter / cleaning

### What it does

Sample cleaning and basic features. Drop invalid or randomly exposed rows, keep rows with play time over 3 s or a click, and compute watch ratio, duration bucket and interaction count. Modes measured: `cpu`, `gpu-batch`.

### SQL

```sql
SELECT seq, user_id, video_id, time_ms,
       CAST(play_time_ms AS DOUBLE) / CAST(duration_ms + 1 AS DOUBLE) AS watch_ratio,
       CASE WHEN duration_ms < 15000 THEN 0 WHEN duration_ms < 60000 THEN 1 ELSE 2 END AS duration_bucket,
       is_like + is_follow + is_comment + is_forward AS interact_cnt,
       is_click
FROM events
WHERE duration_ms > 0 AND play_time_ms >= 0 AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
```

### Real execution plan

CPU, Optimized Execution Plan:

```text
Calc(select=[seq, user_id, video_id, time_ms, (CAST(play_time_ms AS DOUBLE) / CAST((duration_ms + 1) AS DOUBLE)) AS watch_ratio, CASE((duration_ms < 15000), 0, (duration_ms < 60000), 1, 2) AS duration_bucket, (((is_like + is_follow) + is_comment) + is_forward) AS interact_cnt, is_click], where=[((duration_ms > 0) AND (play_time_ms >= 0) AND (is_rand = 0) AND ((play_time_ms > 3000) OR (is_click = 1)))])
+- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch, Optimized Execution Plan:

```text
GpuNativeStreamExecCalc(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5) Calc(select=[seq, user_id, video_id, time_ms, (CAST(play_time_ms AS DOUBLE) / CAST((duration_ms + 1) AS DOUBLE)) AS watch_ratio, CASE((duration_ms < 15000), 0, (duration_ms < 60000), 1, 2) AS duration_bucket, (((is_like + is_follow) + is_comment) + is_forward) AS interact_cnt, is_click], where=[((duration_ms > 0) AND (play_time_ms >= 0) AND (is_rand = 0) AND ((play_time_ms > 3000) OR (is_click = 1)))])
+- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

The Optimized Physical Plan is the same on both sides (a single `Calc` over `TableSourceScan`). The replacement happens in the Optimized Execution Plan: `Calc` becomes `GpuNativeStreamExecCalc`, with the original Calc expression kept after the parameters.

- `GpuNativeStreamExecCalc`: the filter-and-project node run by Mega-Flink. The node name is defined by Mega-Flink. `engine=batch` is the batch form; `batchRows` and `maxWaitMs` are explained above. The coverage document lists a fixed filter-project Calc as an experimental GPU operator that falls back to CPU by default.
- The source scan stays as it is.

### Plan differences between modes

Only two modes were measured. The GPU job graph has the same vertex chain as the CPU job graph (source, Calc, constraint enforcer, sink in one chain); only the Calc node name differs. There is no extra exchange.

### Evidence that the GPU ran

In the batch mode, the batch count, H2D time and compute time in the runtime counters are non-zero in every formal round. The measured timers of the default saturated run (927 batches) are in the time breakdown below, and nsys cross-checks them.

### Results

**Default**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | 20,000 | <0.01 / <0.01 | 19,999 | 2.74 / 6.16 | MATCH |
| Mid | 100,000 | <0.01 / 0.21 | 99,989 | 2.96 / 7.60 | MATCH |
| Saturated | 831,997 | <0.01 / <0.01 | 1,025,360 | 1.86 / 3.38 | MATCH |

**Tuned (object-reuse on both sides)**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | not measured | not measured | not measured | not measured | not measured |
| Mid | 100,000 | <0.01 / 0.01 | 99,978 | 2.80 / 6.23 | MATCH |
| Saturated | 2,472,421 | <0.01 / <0.01 | 2,059,561 | 0.92 / 1.77 | MATCH |

### Where the time goes

GPU-side measured timers (default, saturated, 927 batches; Mega-Flink CUDA-event timers, measured): row-to-column 254 ms, H2D 20 ms, kernel 25 ms, D2H 14 ms, column-to-row 186 ms. These five sum to 499 ms; the kernel is about 5% and row/column conversion about 88%. nsys cross-check: kernel 13.2 ms, H2D 15.9 ms, D2H 8.1 ms.

### Conclusion

GPU +23% under default settings, CPU +20% after tuning. **The configuration matters.** Under default settings (`pipeline.object-reuse=false`) the GPU is faster on c1 (1.03M vs 0.83M, GPU +23%); with object-reuse enabled on both sides the result flips to CPU faster. Object-reuse helps the CPU more than the GPU (c1: roughly x2.95 vs x2.0). The GPU kernel is about 5% of the measured GPU-side time; row/column conversion is about 88%.

<a id="c3-topn"></a>
## c3 Per-user top 10 by watch time (TopN)

### What it does

Top-K features over user behavior, such as "the 10 longest-watched videos". An append-only `ROW_NUMBER` TopN. The GPU replaces `Rank` with a GPU TopN in three forms: batch, persistent and adaptive. Modes measured: `cpu`, `gpu-batch`, `gpu-persistent`, `gpu-adaptive`.

### SQL

```sql
SELECT seq, user_id, video_id, play_time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn
  FROM events) WHERE rn <= 10
```

### Real execution plan

CPU, Optimized Execution Plan:

```text
Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
+- Exchange(distribution=[hash[user_id]])
   +- Calc(select=[seq, user_id, video_id, play_time_ms])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch, Optimized Execution Plan:

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=batch, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
+- Exchange(distribution=[hash[user_id]])
   +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, play_time_ms])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

The Optimized Physical Plan is the same on all four modes (`Rank(AppendFastStrategy)` over `Exchange(hash[user_id])` over `Calc`). The replacement happens in the Optimized Execution Plan.

- `GpuNativeStreamExecRank`: replaces `Rank`. `rankType=ROW_NUMBER` and `topN=10` repeat the rank type and range of the Flink Rank that follows it. The coverage document lists Rank as an experimental hybrid operator; `topn.engine=adaptive` picks CPU or batch by pressure, and persistent applies to ROW_NUMBER only.
- `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)`: the Calc below the exchange stays on CPU (see the Gpu\* note at the top).
- `Exchange(hash[user_id])` and the source scan are unchanged.

### Plan differences between modes

The three GPU plans have the same shape. Only the `engine` value in the Rank line changes. The Rank lines of the other two modes:

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=persistent, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
```

```text
GpuNativeStreamExecRank(mode=gpu-native, engine=adaptive, rankType=ROW_NUMBER, topN=10, batchRows=4096, maxWaitMs=5) Rank(strategy=[AppendFastStrategy], rankType=[ROW_NUMBER], rankRange=[rankStart=1, rankEnd=10], partitionBy=[user_id], orderBy=[play_time_ms DESC, seq ASC], select=[seq, user_id, video_id, play_time_ms])
```

- `engine=batch`: batched GPU Top-N.
- `engine=persistent`: the resident form. Per the coverage document, persistent applies to ROW_NUMBER only.
- `engine=adaptive`: the operator keeps both a CPU path and a GPU path and picks by pressure. In the measured runs it sent 0 rows to the GPU at low/mid rates (its latency equals CPU and is not a GPU result); at saturation about 36% of rows take the CPU path.

The job graph is the same for all four modes: one chain with the source and Calc, and one chain with Rank, constraint enforcer and sink.

### Evidence that the GPU ran

In batch mode, the batch count, H2D time and compute time are non-zero in every formal round. For persistent, the persistent command count equals the whole input. For adaptive, the counters are 0 in all low/mid runs and non-zero in most saturated runs, consistent with the 0 rows sent to the GPU at low/mid rates. nsys, default config, exclusive: 910 `select_top_n_by_group` kernels total 108 ms, plus 5 ms H2D and 2 ms D2H.

### Results

**Default**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 19,995 | 14.70 / 33.03 | MATCH |
| Low | GPU batch | 19,995 | 16.81 / 33.23 | MATCH |
| Low | GPU persistent | 19,995 | 14.73 / 33.14 | MATCH |
| Low | GPU adaptive | 19,995 | 14.99 / 33.04 | MATCH |
| Mid | CPU | 99,977 | 3.58 / 6.69 | MATCH |
| Mid | GPU batch | 99,976 | 8.59 / 16.11 | MATCH |
| Mid | GPU persistent | 99,976 | 4.14 / 8.92 | MATCH |
| Mid | GPU adaptive | 99,979 | 4.15 / 7.40 | MATCH |
| Saturated | CPU | 1,024,400 | 4.69 / 20.25 | MATCH |
| Saturated | GPU batch | 930,760 | 5.61 / 25.01 | MATCH |
| Saturated | GPU persistent | 489,815 | 11.20 / 33.67 | MATCH |
| Saturated | GPU adaptive | 690,440 | 10.28 / 32.89 | MATCH |

GPU batch varies a lot between rounds at saturation: an 8-round rerun gave 0.90-1.16M with two clusters (5 rounds near 0.93M, 3 rounds at 1.13-1.16M); CPU gave 1.02-1.12M over 8 rounds (median 1.06M). It is therefore not correct to say the GPU is consistently ahead or behind.

**Tuned (object-reuse on both sides, plus topn-cache-size=100000)**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 99,978 | 3.53 / 6.70 | MATCH |
| Mid | GPU batch | 99,972 | 8.64 / 15.12 | MATCH |
| Mid | GPU persistent | 99,977 | 4.16 / 13.48 | MATCH |
| Mid | GPU adaptive | 99,978 | 4.05 / 10.77 | MATCH |
| Saturated | CPU | 1,363,629 | 3.59 / 22.79 | MATCH |
| Saturated | GPU batch | 1,231,075 | 5.22 / 29.97 | MATCH |
| Saturated | GPU persistent | 430,154 | 12.87 / 38.05 | MATCH |
| Saturated | GPU adaptive | 718,364 | 9.48 / 31.29 | MATCH |

Tuned low tier: not measured.

### Where the time goes

Default config, exclusive, nsys and parameter sweeps: 910 `select_top_n_by_group` kernels total 108 ms, plus 5 ms H2D and 2 ms D2H, about 3.6% of the roughly 3.2 s job, so the GPU is more than 95% idle. At parallelism 1, the kernel takes about 0.12 s while the whole job (3.3M rows) takes about 3.2 s (formal median).

### Conclusion

The CPU is not behind: CPU +11% over GPU batch after tuning, and +38% / +21% at parallelism 2/4 (section 5 of [benchmarks/README.md](README.md)). Persistent has lower latency than batch only at low/mid rates and is not better than CPU, with low saturated throughput. Adaptive sends 0 rows to the GPU at low/mid rates; at saturation about 36% of rows take the CPU path.

**Why the GPU does not win** (default config, exclusive, nsys and parameter sweeps): sweeping batch size (1,024 to 65,536) and max wait (1/5/20 ms) leaves saturated throughput within 0.87-1.11M with no clear trend. A ceiling probe with only source + exchange + sink and no computation reaches 0.856M on CPU and 0.851M on GPU, which indicates the JVM pipeline is the bottleneck. The mechanism of the two result clusters of GPU batch at saturation is not claimed.

<a id="c4-dedup"></a>
## c4 Keep-last dedup

### What it does

Sample dedup and latest-state extraction. Keep the newest row per (user_id, video_id). The GPU replaces the dedup node with a GPU Deduplicate (batch / adaptive). CPU with mini-batch (`cpu-mb`) was also measured. Modes measured: `cpu`, `cpu-mb`, `gpu-batch`, `gpu-adaptive`.

### SQL

```sql
SELECT seq, user_id, video_id, play_time_ms, time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms, time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY ts DESC) AS rn
  FROM events) WHERE rn = 1
```

### Real execution plan

CPU, Optimized Execution Plan:

```text
Calc(select=[seq, user_id, video_id, play_time_ms, time_ms])
+- Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
   +- Exchange(distribution=[hash[user_id, video_id]])
      +- Calc(select=[seq, user_id, video_id, time_ms, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU batch, Optimized Execution Plan:

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, play_time_ms, time_ms])
+- GpuNativeStreamExecDeduplicate(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5) Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
   +- Exchange(distribution=[hash[user_id, video_id]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, user_id, video_id, time_ms, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Note on an earlier description: the c4 plan was described as `Rank(AppendFast, keep-last)`. The saved plan shows that Flink turns this SQL into a `Deduplicate(keep=[LastRow], ...)` node, not a Rank.

The Optimized Physical Plan is the same for `cpu`, `gpu-batch` and `gpu-adaptive`. The replacement happens in the Optimized Execution Plan.

- `GpuNativeStreamExecDeduplicate`: replaces `Deduplicate`. The coverage document lists keep-last Deduplicate as an experimental hybrid operator, enabled with `deduplicate.engine=adaptive` or an explicit batch setting.
- The Calc above and below the dedup node is shown as `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` (see the Gpu\* note at the top).

### Plan differences between modes

`gpu-adaptive` differs from `gpu-batch` only in the engine value:

```text
+- GpuNativeStreamExecDeduplicate(mode=gpu-native, engine=adaptive, batchRows=4096, maxWaitMs=5) Deduplicate(keep=[LastRow], key=[user_id, video_id], order=[ROWTIME], outputInsertOnly=[false])
```

`cpu-mb` (CPU with mini-batch) is a different CPU plan: both its Optimized Physical Plan and Optimized Execution Plan add a `MiniBatchAssigner` node. Lines that differ from `cpu`:

```text
         +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Job graph: in the CPU job graph, the vertex with `Deduplicate` is chained with a Calc after it. In both GPU job graphs, the Deduplicate vertex goes straight to the constraint enforcer and no separate Calc vertex is listed, although the execution plan above still shows the Calc. Why: not verified.

### Evidence that the GPU ran

In batch mode, the batch count, H2D time and compute time are non-zero in every formal round. Measured GPU timers (default, saturated): kernel 13 ms, H2D 8 ms, D2H 8 ms over 983 batches. For adaptive, the counters are 0 in all low/mid runs and non-zero in the saturated runs, consistent with 0 rows to the GPU at low/mid rates.

### Results

**Default**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 19,995 | 11.49 / 24.98 | MATCH |
| Low | CPU mini-batch | 19,995 | 22.50 / 43.65 | MATCH |
| Low | GPU batch | 19,995 | 14.54 / 25.23 | MATCH |
| Low | GPU adaptive | 19,995 | 11.52 / 24.95 | MATCH |
| Mid | CPU | 99,962 | 2.88 / 5.06 | MATCH |
| Mid | CPU mini-batch | 99,950 | 14.93 / 30.94 | MATCH |
| Mid | GPU batch | 99,967 | 5.99 / 16.08 | MATCH |
| Mid | GPU adaptive | 99,967 | 2.86 / 5.06 | MATCH |
| Saturated | CPU | 685,882 | 5.23 / 19.63 | MATCH |
| Saturated | CPU mini-batch | 580,305 | 7.74 / 26.90 | MATCH |
| Saturated | GPU batch | 660,054 | 7.22 / 27.85 | MATCH |
| Saturated | GPU adaptive | 675,786 | 6.98 / 28.34 | MATCH |

**Tuned**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 99,970 | 2.74 / 5.06 | MATCH |
| Mid | GPU batch | 99,964 | 5.93 / 13.95 | MATCH |
| Mid | GPU adaptive | 99,970 | 2.86 / 7.19 | MATCH |
| Saturated | CPU | 1,033,538 | 3.30 / 17.82 | MATCH |
| Saturated | GPU batch | 850,292 | 5.96 / 21.96 | MATCH |
| Saturated | GPU adaptive | 852,773 | 6.03 / 22.28 | MATCH |

Tuned low tier: not measured.

### Where the time goes

The main costs of dedup are serialization and JVM state access, which the GPU does not replace. JFR estimate (sampled, not exact): state about 14% on CPU and 17% in the GPU mode. GPU measured kernel 13 ms, H2D 8 ms, D2H 8 ms over 983 batches.

### Conclusion

CPU is slightly ahead under default settings (0.686M vs 0.660M) and +22% after tuning. Adaptive sends 0 rows to the GPU at low/mid rates; at saturation about 4.5% of rows take the CPU path.

<a id="c5-join"></a>
## c5 Impression x click Interval Join, and fusion with a following Filter / aggregation

Impression-click attribution and training-sample assembly. The impression stream `imp` and click stream `clk` (`is_click=1`) come from the same log; the join key is `user_id*10^8+video_id` (a single BIGINT) and the click must fall within 30 minutes after the impression. Three variants:

- `c5-join`: the join only.
- `c5-join-filter`: adds `c.play_time_ms >= i.duration_ms` after the join. Flink pushes it down as a residual join condition; in GPU fused mode the residual is evaluated on the device, so "fused vs not fused" = fusion + residual push-down.
- `c5-join-agg`: adds a per-user_id aggregation (mini-batch 5 ms / 4096 on both sides).

Shared setup (the two views used by all three variants):

```sql
CREATE TEMPORARY VIEW imp AS
  SELECT seq, user_id * 100000000 + video_id AS uv, user_id, duration_ms, ts FROM events;
CREATE TEMPORARY VIEW clk AS
  SELECT seq, user_id * 100000000 + video_id AS uv, play_time_ms, ts FROM events WHERE is_click = 1;
```

In GPU fused mode, the Calc / LocalAggregate after the join runs in the same GPU stage as the join, saving one column-to-row and one row-to-column conversion. For `c5-join`, GPU batch is fused (Join -> projection) by default.

**What fusion looks like in the plans.** For `join-filter` and `join-agg`, the Optimized Execution Plan text of the fused and not-fused runs is **identical**. The difference shows only in the job graph: in the fused run, the operators that follow the join (the Calc, and for join-agg also the LocalGroupAggregate) are no longer listed as separate vertices after `IntervalJoin`; in the not-fused run they are chained behind it. The sections below show both.

**Gpu\* nodes in c5.**

- `GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=..., bounds=[-1800000,0], ...)`: replaces `IntervalJoin`. The coverage document lists Interval Join as an experimental GPU operator with batch and persistent engines, for append-only event-time INNER joins with a single BIGINT equality key and one bounded time condition. `bounds` carries the same two values as `leftLowerBound` / `leftUpperBound` in the IntervalJoin line (30 minutes in milliseconds); other parameters as in the Gpu\* note at the top. Joins outside that shape fall back to CPU, per the document.
- `FlinkCpuStreamExecCalc(...)`: Calc nodes stay on Flink CPU.
- The coverage document describes deep fusion of Interval Join with a directly following Calc (numeric condition and direct-field or safe-cast projection) and with a following two-phase Local Aggregate.

<a id="c5-join-only"></a>
### c5-join

#### What it does

The join only: pair each impression with the clicks of the same user and video within 30 minutes. Modes measured: `cpu`, `gpu-batch` (fused by default), `gpu-persistent` (fused by default). A "GPU not fused" column also appears in the tables; no plan file was saved for that run (not verified).

#### SQL

```sql
-- views imp and clk as defined above
SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE;
```

#### Real execution plan

CPU, Optimized Execution Plan:

```text
Calc(select=[seq, seq0, uv, play_time_ms, duration_ms])
+- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

GPU batch, Optimized Execution Plan:

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, seq0, uv, play_time_ms, duration_ms])
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

The Optimized Physical Plan is the same for `cpu`, `gpu-batch` and `gpu-persistent`. The replacement happens in the Optimized Execution Plan: `IntervalJoin` becomes `GpuNativeStreamExecIntervalJoin`, and the Calc nodes become `FlinkCpuStreamExecCalc`.

#### Plan differences between modes

`gpu-persistent` differs from `gpu-batch` only in the engine value:

```text
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=persistent, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
```

Job graph: in the CPU job graph, `IntervalJoin` is chained with a Calc after it. In the `gpu-batch` and `gpu-persistent` job graphs, the vertex holds only `IntervalJoin` and goes straight to the constraint enforcer: the projection is fused into the join (this is the default fused form).

CPU job graph (source vertex omitted):

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

GPU batch job graph (source vertex omitted):

```text
- IntervalJoin[13] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

#### Evidence that the GPU ran

In batch mode, the join counters (kernels, H2D time, compute time) are non-zero in every formal round. For persistent, command counts were recorded only in the breakdown round and were 0 in the formal rounds; the reason is not verified.

#### Results

**Default** (explicit fuse / nofuse in the same job, median of 3; throughput / p50 / p99)

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join | Low | 19,994 / 29.91 / 79.26 | 19,993 / 29.88 / 80.09 | 19,993 / 29.98 / 79.91 | MATCH |
| join | Mid | 99,928 / 10.57 / 24.84 | 99,921 / 14.08 / 27.56 | 99,933 / 14.23 / 26.58 | MATCH |
| join | Saturated | 259,430 / 20.62 / 55.11 | 232,091 / 24.97 / 56.24 | 221,821 / 26.10 / 56.35 | MATCH |

GPU persistent (fused by default): join 166,494 saturated (p99 199.94 ms), join-filter 171,651 saturated (p99 197.54 ms); tail latency is about 200 ms across tiers, clearly worse than both CPU and batch.

Tuned (object-reuse on both sides; low tier: not measured):

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join | Mid | 99,928 / 10.38 / 37.56 | 99,931 / 13.89 / 26.79 | 99,938 / 13.84 / 25.97 | MATCH |
| join | Saturated | 291,797 / 19.63 / 60.01 | 253,233 / 24.07 / 54.08 | 251,270 / 24.24 / 52.71 | MATCH |

#### Where the time goes

The largest cost on both sides is JVM state access (the two join buffers; JFR estimate, sampled: about 52% on CPU join and 43% on GPU join), which the GPU does not replace. The join fan-out on this data is about 1, so there is little data movement for fusion to save.

#### Conclusion

CPU is faster on throughput: after tuning join +15% (against GPU fused). Fusion gives no throughput gain after tuning (join +0.8%, within noise); join latency is unchanged. Persistent is clearly worse than both CPU and batch on tail latency.

<a id="c5-join-filter"></a>
### c5-join-filter

#### What it does

The join plus a filter on the joined pair, `c.play_time_ms >= i.duration_ms`. Modes measured: `cpu`, `gpu-batch-fuse`, `gpu-batch-nofuse`; persistent (fused by default) appears in the notes.

#### SQL

```sql
-- views imp and clk as defined above
SELECT i.seq, c.seq, i.uv, c.play_time_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE
WHERE c.play_time_ms >= i.duration_ms;
```

#### Real execution plan

CPU, Optimized Execution Plan:

```text
Calc(select=[seq, seq0, uv, play_time_ms])
+- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)) AND (play_time_ms >= duration_ms))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

GPU batch, fused (`gpu-batch-fuse`), Optimized Execution Plan:

```text
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, seq0, uv, play_time_ms])
+- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=3, rightTimeIndex=3], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)) AND (play_time_ms >= duration_ms))], select=[seq, uv, duration_ms, ts, seq0, uv0, play_time_ms, ts0])
   :- Exchange(distribution=[hash[uv]])
   :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, duration_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
   :     +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])(reuse_id=[1])
   +- Exchange(distribution=[hash[uv]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, ((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
         +- Reused(reference_id=[1])
```

The Optimized Physical Plan is the same for `cpu`, `gpu-batch-fuse` and `gpu-batch-nofuse`. The filter has already been pushed into the join: the `where=[...]` of the IntervalJoin line contains `play_time_ms >= duration_ms`, so the Calc after the join is a pure projection. The replacement happens in the Optimized Execution Plan, as in c5-join.

#### Plan differences between modes (fuse / nofuse)

The Optimized Execution Plan of `gpu-batch-nofuse` is identical to `gpu-batch-fuse` (we compared the two texts). The difference is in the job graph (source vertex omitted):

Fused:

```text
- IntervalJoin[13] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

Not fused:

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

CPU:

```text
- IntervalJoin[13] -> Calc[14] -> ConstraintEnforcer[15] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

The fused run drops the separate Calc vertex after `IntervalJoin`; the not-fused run keeps it, as the CPU run does. Per the coverage document, the residual condition and projection are then evaluated inside the join stage. Fused vs not fused here also includes the residual push-down to the device. For persistent, no plan file was saved (not verified).

#### Evidence that the GPU ran

In batch mode (fused and not fused), the join counters are non-zero in every formal round. For persistent, as for c5-join, the command count in formal rounds is not usable as evidence (0 in formal rounds for c5 persistent; reason not verified).

#### Results

**Default** (explicit fuse / nofuse in the same job, median of 3; throughput / p50 / p99)

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join-filter | Low | 19,993 / 29.64 / 79.08 | 19,990 / 30.00 / 80.25 | 19,989 / 29.97 / 80.03 | MATCH |
| join-filter | Mid | 99,928 / 10.36 / 26.64 | 99,953 / 13.95 / 26.62 | 99,904 / 13.95 / 25.87 | MATCH |
| join-filter | Saturated | 272,760 / 20.17 / 56.22 | 254,153 / 22.89 / 54.92 | 250,751 / 23.30 / 54.38 | MATCH |

For join-filter at saturation, "not fused + device-side residual" gave 251,131, essentially the same as "not fused" (about 251k), so residual push-down alone gives almost no benefit.

GPU persistent (fused by default): join 166,494 saturated (p99 199.94 ms), join-filter 171,651 saturated (p99 197.54 ms); tail latency is about 200 ms across tiers, clearly worse than both CPU and batch.

Tuned (object-reuse on both sides; low tier: not measured):

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join-filter | Mid | 99,927 / 10.09 / 30.54 | 99,903 / 13.82 / 25.54 | 99,921 / 13.71 / 25.68 | MATCH |
| join-filter | Saturated | 307,921 / 18.80 / 63.22 | 262,297 / 23.49 / 51.55 | 259,502 / 23.67 / 52.45 | MATCH |

#### Where the time goes

Same as c5-join: the largest cost on both sides is JVM state access (the two join buffers; JFR estimate, sampled: about 52% on CPU join and 43% on GPU join), which the GPU does not replace. The join fan-out on this data is about 1.

#### Conclusion

CPU is faster on throughput: after tuning join-filter +17% (against GPU fused). Fusion gives no throughput gain after tuning (join-filter +1.1%, within noise); latency is unchanged.

<a id="c5-join-agg"></a>
### c5-join-agg

#### What it does

The join followed by a per-user_id COUNT / SUM (mini-batch 5 ms / 4096 on both sides). Modes measured: `cpu`, `gpu-batch-fuse`, `gpu-batch-nofuse`.

#### SQL

```sql
-- views imp and clk as defined above
SELECT i.user_id, COUNT(*) AS clicks, SUM(c.play_time_ms) AS play_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE
GROUP BY i.user_id;
```

#### Real execution plan

CPU, Optimized Execution Plan:

```text
GlobalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(count1$0) AS clicks, SUM(sum$1) AS play_ms])
+- Exchange(distribution=[hash[user_id]])
   +- LocalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(*) AS count1$0, SUM(play_time_ms) AS sum$1])
      +- Calc(select=[user_id, play_time_ms])
         +- IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=2, rightTimeIndex=2], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[uv, user_id, ts, uv0, play_time_ms, ts0])
            :- Exchange(distribution=[hash[uv]])
            :  +- Calc(select=[((user_id * 100000000) + video_id) AS uv, user_id, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            :     +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])(reuse_id=[1])
            :        +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
            +- Exchange(distribution=[hash[uv]])
               +- Calc(select=[((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
                  +- Reused(reference_id=[1])
```

GPU batch, fused (`gpu-batch-fuse`), Optimized Execution Plan:

```text
GpuAssistedStreamExecGlobalGroupAggregate(local=project-owned-cuda, merge=project-owned-cpu) GlobalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(count1$0) AS clicks, SUM(sum$1) AS play_ms])
+- Exchange(distribution=[hash[user_id]])
   +- GpuNativeStreamExecLocalGroupAggregate(mode=gpu-native, engine=batch, batchRows=4096, maxWaitMs=5, persistentMaxGroups=65536) LocalGroupAggregate(groupBy=[user_id], select=[user_id, COUNT(*) AS count1$0, SUM(play_time_ms) AS sum$1])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, play_time_ms])
         +- GpuNativeStreamExecIntervalJoin(mode=gpu-native, engine=batch, bounds=[-1800000,0], batchRows=4096, maxWaitMs=5) IntervalJoin(joinType=[InnerJoin], windowBounds=[isRowTime=true, leftLowerBound=-1800000, leftUpperBound=0, leftTimeIndex=2, rightTimeIndex=2], where=[((uv = uv0) AND (ts0 >= ts) AND (ts0 <= (ts + 1800000:INTERVAL MINUTE)))], select=[uv, user_id, ts, uv0, play_time_ms, ts0])
            :- Exchange(distribution=[hash[uv]])
            :  +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[((user_id * 100000000) + video_id) AS uv, user_id, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            :     +- MiniBatchAssigner(interval=[5ms], mode=[RowTime])(reuse_id=[1])
            :        +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
            +- Exchange(distribution=[hash[uv]])
               +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[((user_id * 100000000) + video_id) AS uv, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts], where=[(is_click = 1)])
                  +- Reused(reference_id=[1])
```

The Optimized Physical Plan is the same for `cpu`, `gpu-batch-fuse` and `gpu-batch-nofuse` (two-phase aggregation: `LocalGroupAggregate`, `Exchange(hash[user_id])`, `GlobalGroupAggregate`). The replacement happens in the Optimized Execution Plan. Here three nodes change: `IntervalJoin`, `LocalGroupAggregate` and `GlobalGroupAggregate`.

- `GpuNativeStreamExecLocalGroupAggregate(..., persistentMaxGroups=...)`: replaces `LocalGroupAggregate` (the Local phase of two-phase aggregation). `persistentMaxGroups` is defined by Mega-Flink. The coverage document lists Local + Global aggregation as an experimental GPU operator that also needs Flink mini-batch in the plan.
- `GpuAssistedStreamExecGlobalGroupAggregate(local=project-owned-cuda, merge=project-owned-cpu)`: replaces `GlobalGroupAggregate`. The parameter names are defined by Mega-Flink. The coverage document describes the Global phase as a JVM merge with Flink keyed state, so the state and merge stay on the JVM.

#### Plan differences between modes (fuse / nofuse)

As in join-filter, the Optimized Execution Plan of `gpu-batch-nofuse` is identical to `gpu-batch-fuse`. The difference is in the job graph (source vertex omitted):

Fused:

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18]  [parallelism=1]
```

Not fused:

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18] -> Calc[19] -> LocalGroupAggregate[20]  [parallelism=1]
```

CPU:

```text
- GlobalGroupAggregate[22] -> ConstraintEnforcer[23] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- IntervalJoin[18] -> Calc[19] -> LocalGroupAggregate[20]  [parallelism=1]
```

In the fused run, `IntervalJoin` is a vertex of its own, and the Calc and `LocalGroupAggregate` that follow it are not listed. In the not-fused run (and on CPU) they are chained behind the join. Per the coverage document, deep fusion updates COUNT/SUM in the same CUDA range probe as the join and sends only a compressed partial to the exchange. The Global phase is a separate vertex in all modes.

#### Evidence that the GPU ran

In batch mode (fused and not fused), the join counters are non-zero in every formal round.

#### Results

**Default** (explicit fuse / nofuse in the same job, median of 3; throughput / p50 / p99)

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join-agg | Low | 19,861 / 107.58 / 195.47 | 19,958 / 92.68 / 180.70 | 19,859 / 96.89 / 183.59 | MATCH |
| join-agg | Mid | 99,252 / 55.91 / 102.08 | 99,774 / 37.97 / 76.68 | 99,397 / 43.59 / 86.78 | MATCH |
| join-agg | Saturated | 268,556 / 60.56 / 121.23 | 233,707 / 44.49 / 83.84 | 232,152 / 50.97 / 98.09 | MATCH |

Tuned (object-reuse on both sides; low tier: not measured):

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join-agg | Mid | 99,141 / 55.20 / 102.33 | 99,798 / 37.57 / 76.84 | 99,377 / 42.76 / 86.52 | MATCH |
| join-agg | Saturated | 293,385 / 58.96 / 127.08 | 236,056 / 45.31 / 84.37 | 253,569 / 48.46 / 98.14 | MATCH |

For join-agg at saturation, the three CPU rounds were 293k, 257k and 300k, a wide spread.

#### Where the time goes

The largest cost on both sides is JVM state access (the two join buffers; JFR estimate, sampled: about 52% on CPU join and 43% on GPU join), which the GPU does not replace. The join fan-out on this data is about 1, so there is little data movement for fusion to save.

#### Conclusion

CPU is faster on throughput: after tuning join-agg +24% (against GPU fused). With a following aggregation, GPU latency is lower (p50 about 23% lower). Fusion: after tuning there is no throughput gain (join-agg fused is about 7% lower), only a lower join-agg p50 (about 6.5% saturated, 12% mid).

<a id="summary"></a>
## Summary of the group

| Case | CPU | GPU | Faster |
| --- | --- | --- | --- |
| c1 filter / cleaning | 2,472,421 | 2,059,561 | CPU +20% |
| c3 TopN | 1,363,629 | 1,231,075 (batch) | CPU +11% |
| c4 dedup | 1,033,538 | 850,292 (batch) | CPU +22% |
| c5 join | 291,797 | 253,233 (fused) | CPU +15% |
| c5 join-filter | 307,921 | 262,297 (fused) | CPU +17% |
| c5 join-agg | 293,385 | 236,056 (fused) | CPU +24% (GPU p50 latency about 23% lower) |

Tuned, saturated throughput (records/s), single GPU, parallelism 1. In these four cases the work the GPU takes over is small compared with serialization, JVM state access and row/column conversion, and the GPU does not replace those. Persistent and adaptive forms gave no support for choosing persistent (adaptive sends 0 rows to the GPU at low/mid rates). Limits that apply here: single GPU, mostly parallelism 1; c4 and c5 at parallelism 2/4 not measured; the tuned low tier not measured; the rewritten report is not independently re-reviewed.
