# Group 1: GPU wins (c2 window aggregation, c6 dimension-table Lookup Join)

Back to the [overview](README.md). These two cases are faster on the GPU at saturation, in both the default and the tuned tables. Setup, tiers, statistics and the meaning of "default" and "tuned" are in [README sections 2 and 3](README.md#env). Throughput is in records/s and latency in ms; "Check" compares the SHA-256 of the final table, per case and tier.

- [c2 Window aggregation (HOP and TUMBLE)](#c2-window)
- [c6 Dimension-table Lookup Join](#c6-lookup)

## How to read the plans

Flink prints several plans for a query. Two matter here.

- **Optimized Physical Plan**: Flink's planner output. It is the same on CPU and GPU.
- **Optimized Execution Plan**: the plan after ExecNodes are chosen. In GPU mode, nodes replaced by Mega-Flink carry a prefix of the form `<NodeName>(<parameters>)` in front of the original Flink node text; nodes that stay on the CPU are shown as `FlinkCpuStreamExecCalc(...)`. This is where CPU and GPU differ.

Plan text below is copied from the exported EXPLAIN of the test runs. Only the long `fields=[...]` lists of the source scan are cut and shown as `…`. The node names and parameters are defined by Mega-Flink; we explain only what its operator-coverage document supports, and write "not defined in the document" otherwise.

`FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` appears in both cases. The Mega-Flink document describes a Calc engine setting (`auto`, `cpu`, `persistent`, `batch`) and says `auto` sends a standalone Calc to the CPU for low latency. So the two parameters read as "engine used: cpu" and "configured value: auto" (our reading of the names); the Calc runs on the Flink CPU in GPU mode.

<a id="c2-window"></a>
## c2 Per-user clicks over the last N minutes: window aggregation (HOP and TUMBLE)

### What it does

Real-time statistical features such as "a user's impressions / clicks / play time over the last N minutes". Event-time windows, aggregated per user_id. HOP uses a 30-minute window with a 5-minute slide (each record lands in several windows); TUMBLE uses a 5-minute tumbling window. On CPU, Flink splits the window aggregation into Local + Global phases by default; we also measured one-phase (`1p`, a direct WindowAggregate). The GPU two-phase plan uses the experimental `LocalWindowAggregate` path; the one-phase plan corresponds to Mega-Flink's more mature path.

### SQL

```sql
-- HOP
SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt,
       SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms
FROM TABLE(HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE))
GROUP BY user_id, window_start, window_end;

-- TUMBLE: replace the HOP(...) call with
TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE))
```

### Real execution plan

Optimized Physical Plan (CPU and GPU identical; two-phase default, shown with source fields cut):

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Optimized Execution Plan, CPU (two-phase):

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Optimized Execution Plan, GPU (two-phase):

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true) GlobalWindowAggregate(groupBy=[user_id], window=[HOP(slice_end=[$slice_end], size=[30 min], slide=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- GpuNativeStreamExecLocalWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5) LocalWindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS count1$0, SUM(is_click) AS sum$1, SUM(play_time_ms) AS sum$2, slice_end('w$) AS $slice_end])
         +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
            +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

The replacement happens at the execution-plan layer, not in the physical plan. What each GPU-mode node is:

| Node and parameters | What it is |
| --- | --- |
| `GpuNativeStreamExecLocalWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5)` | Replaces the Local window aggregate. The Mega-Flink document says it supports UTC event-time TUMBLE / HOP / CUMULATE on INSERT-only input, builds Flink's partial accumulator and `$slice_end` on the GPU, and merges small GPU batches on the host by key and slice before the exchange. It is an experimental path (needs the experimental-operators switch). `batchRows=4096` and `maxWaitMs=5` are the batch size and maximum wait (section 2 of the overview). The meaning of `mode=gpu-native` is not defined in the document. |
| `GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true)` | Replaces the Global window aggregate. The document says the Global stage uses a dedicated JVM merge (no second GPU kernel), keeps unclosed windows in Flink keyed state with event-time timers, and for windows already closed by the watermark merges and writes the final row once. The three parameter values read as: Local stage is the project's own CUDA code, merge is the project's own CPU code, and watermark-closed windows are output directly. The exact value strings are not defined in the document. Also experimental. |
| `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` | The two Calc nodes stay on the Flink CPU (see above). |
| `Exchange(distribution=[hash[user_id]])`, `TableSourceScan` | Unchanged Flink nodes. |

Optimized Execution Plan for the one-phase variant (`1p`); the Optimized Physical Plan is again identical on both sides:

CPU (`cpu-1p`):

```
Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- WindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS expo_cnt, SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

GPU (`gpu-1p`):

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuNativeStreamExecWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5) WindowAggregate(groupBy=[user_id], window=[HOP(time_col=[ts], size=[30 min], slide=[5 min])], select=[user_id, COUNT(*) AS expo_cnt, SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
      +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, is_click, play_time_ms, Reinterpret(CAST(ts AS TIMESTAMP_LTZ(3) *ROWTIME*)) AS ts])
         +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

`GpuNativeStreamExecWindowAggregate(mode=gpu-native, batchRows=4096, maxWaitMs=5)` replaces the whole one-phase `WindowAggregate` in one operator. The Mega-Flink document lists TUMBLE / HOP / CUMULATE window aggregation as a default GPU path with no extra switch, and says that when the watermark reaches the end of a window it emits one final INSERT and clears the window state. The earlier text of this document also calls it Mega-Flink's more mature path. The run configuration for `gpu-1p` adds the one-phase setting `table.optimizer.agg-phase-strategy=ONE_PHASE`.

TUMBLE uses the same node structure as HOP; only the window argument changes. The GPU lines for TUMBLE (two-phase):

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[user_id, window_start, window_end, expo_cnt, click_cnt, play_ms])
+- GpuAssistedStreamExecGlobalWindowAggregate(local=project-owned-cuda, merge=project-owned-cpu, direct-watermark-output=true) GlobalWindowAggregate(groupBy=[user_id], window=[TUMBLE(slice_end=[$slice_end], size=[5 min])], select=[user_id, COUNT(count1$0) AS expo_cnt, SUM(sum$1) AS click_cnt, SUM(sum$2) AS play_ms, start('w$) AS window_start, end('w$) AS window_end])
   +- Exchange(distribution=[hash[user_id]])
```

### Plan differences per mode

| Mode | Local stage | Global / window stage | Source |
| --- | --- | --- | --- |
| `cpu` (two-phase) | `LocalWindowAggregate` | `GlobalWindowAggregate` | CPU plan above |
| `cpu-1p` | none | `WindowAggregate` | CPU one-phase plan |
| `gpu` (two-phase) | `GpuNativeStreamExecLocalWindowAggregate` | `GpuAssistedStreamExecGlobalWindowAggregate` | GPU plan above |
| `gpu-1p` | none | `GpuNativeStreamExecWindowAggregate` | GPU one-phase plan |

Modes not measured for c2: persistent and adaptive. The JobGraph (operator chains) has the same shape on CPU and GPU: two vertices, the first ending at the Local window aggregate (two-phase) and the second starting at the Global one; the operator names in the JobGraph do not carry the Gpu prefix.

CPU two-phase JobGraph:

```
JobGraph vertices (operator chains):
- GlobalWindowAggregate[11] -> Calc[12] -> ConstraintEnforcer[13] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[7] -> Calc[8] -> LocalWindowAggregate[9]  [parallelism=1]
```

### Evidence that the GPU ran

The window operators expose runtime metrics, so every GPU round can be checked. In all GPU rounds of the published runs the batch count, H2D, kernel and D2H timers are non-zero. These come from the run metrics (Mega-Flink CUDA-event timers; H2D and D2H include queuing, see overview section 3); they are not part of the numbers reviewed in the earlier text of this document. Per-round range over 3 rounds:

| Mode | Tier | Batches | H2D (ms) | Kernel (ms) | D2H (ms) |
| --- | --- | --- | --- | --- | --- |
| `gpu` HOP | Saturated | 842-951 | 12.9-13.8 | 12.2-13.8 | 8.0-9.8 |
| `gpu-1p` HOP | Saturated | 889-914 | 52.5-53.4 | 14.7 | 24.0-24.5 |
| `gpu` HOP | Mid | 6,177-6,194 | | | |
| `gpu-1p` HOP | Mid | 5,579-5,667 | | | |
| `gpu` HOP | Low | 8,482-8,486 | | | |
| `gpu-1p` HOP | Low | 3,688-3,702 | | | |
| `gpu` TUMBLE | Saturated / Mid / Low | 835-857 / 6,153-6,180 / 8,485-8,489 | | | |

There are more batches at the rate-limited tiers; the reason (batches closed by the 5 ms maximum wait instead of being filled) is our reading and not verified.

### Results

**HOP, default** ("1p" = one-phase)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU two-phase | 19,951 | 64.08 / 123.64 | MATCH |
| Low | CPU 1p | 19,951 | 30.69 / 67.84 | MATCH |
| Low | GPU two-phase | 19,971 | 54.76 / 105.96 | MATCH |
| Low | GPU 1p | 19,973 | 26.36 / 52.98 | MATCH |
| Mid | CPU two-phase | 99,752 | 68.30 / 132.81 | MATCH |
| Mid | CPU 1p | 99,747 | 25.19 / 64.30 | MATCH |
| Mid | GPU two-phase | 99,821 | 38.24 / 69.66 | MATCH |
| Mid | GPU 1p | 99,870 | 17.28 / 40.35 | MATCH |
| Saturated | CPU two-phase | 744,721 | 53.80 / 117.34 | MATCH |
| Saturated | CPU 1p | 628,892 | 26.11 / 78.29 | MATCH |
| Saturated | GPU two-phase | 898,493 | 37.33 / 83.32 | MATCH |
| Saturated | GPU 1p | 892,390 | 16.62 / 57.67 | MATCH |

**HOP, tuned**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU two-phase | 99,824 | 62.89 / 124.43 | MATCH |
| Mid | CPU 1p | 99,828 | 22.46 / 65.07 | MATCH |
| Mid | GPU two-phase | 99,924 | 37.16 / 64.68 | MATCH |
| Mid | GPU 1p | 99,920 | 16.06 / 34.12 | MATCH |
| Saturated | CPU two-phase | 1,475,619 | 48.34 / 97.40 | MATCH |
| Saturated | CPU 1p | 1,079,916 | 25.01 / 86.26 | MATCH |
| Saturated | GPU two-phase | 1,861,833 | 36.44 / 99.76 | MATCH |
| Saturated | GPU 1p | 1,184,784 | 16.02 / 51.10 | MATCH |

Tuned low tier: not measured.

**TUMBLE, default**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | 19,992 | 73.37 / 119.64 | 19,993 | 52.67 / 105.76 | MATCH |
| Mid | 99,961 | 59.86 / 122.12 | 99,967 | 34.96 / 77.36 | MATCH |
| Saturated | 724,616 | 43.42 / 105.17 | 965,671 | 32.37 / 94.38 | MATCH |

**TUMBLE, tuned**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Mid | 99,972 | 61.94 / 121.82 | 99,976 | 34.66 / 63.30 | MATCH |
| Saturated | 1,613,516 | 37.31 / 107.71 | 2,020,259 | 33.91 / 86.85 | MATCH |

### Time composition

- Sampled (JFR) shares are estimates; the Mega-Flink CUDA-event timers in the evidence table are measurements (overview section 3). Breakdown jobs ran 1 round with profiling on, so they are not used for speed.
- State access, timers, changelog merging and Global aggregation stay on the JVM and are not replaced by the GPU (overview section 6). On the GPU side, state read and merge segments are still present in the breakdown; "GPU batching removes state access" has no measured support.
- Sampled per-stage percentages for c2 are not listed in this document: not verified here.

### Takeaway

The GPU is better on both throughput and latency. After tuning, HOP two-phase GPU +26% and TUMBLE +25%; one-phase vs one-phase is GPU +10% (the CPU one-phase plan is itself slower than its two-phase plan, so best-vs-best is +26%). The "GPU +42% (vs CPU one-phase)" figure under default settings compares against a non-optimal CPU plan and should not be quoted. CPU one-phase also varies between rounds under default settings (0.59-0.69M).

<a id="c6-lookup"></a>
## c6 Feature dimension-table Lookup Join

### What it does

Real-time item-side feature joins: each event looks up a video dimension table by video_id to fetch author_id. Only INNER was measured. The CPU uses a fully in-memory HashMap (a favorable case, equivalent to Flink's full cache), and the GPU loads the table into a device hash table. Note that when a batch has fewer than 1,024 rows the GPU path falls back to a JVM lookup, so the `gpu` mode at low/mid rates mostly does not use the GPU; `gpu-min1` (force the GPU on every batch) is included as a control. The dimension table has 1,039,412 rows (413,436 at the low tier), not the full 32 million. The dimension table is declared with `PRIMARY KEY (video_id) NOT ENFORCED` and a custom test connector; both CPU and GPU read the same data.

### SQL

```sql
SELECT e.seq, e.video_id, d.author_id
FROM events AS e
JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d
  ON e.video_id = d.video_id
```

### Real execution plan

Optimized Physical Plan (CPU and GPU identical):

```
Calc(select=[seq, video_id, author_id])
+- LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Optimized Execution Plan, CPU:

```
Calc(select=[seq, video_id, author_id])
+- LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

Optimized Execution Plan, GPU (`gpu`):

```
FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, video_id, author_id])
+- GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024) LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
   +- FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto) Calc(select=[seq, video_id])
      +- TableSourceScan(table=[[*anonymous_datastream_source$1*, metadata=[rowtime], watermark=[SOURCE_WATERMARK()]]], fields=[…])
```

The replacement is again only in the execution plan. What each GPU-mode node is:

| Node and parameters | What it is |
| --- | --- |
| `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024)` | Replaces `LookupJoin`. The Mega-Flink document lists the Lookup Join on a bounded read-only snapshot (INNER / LEFT) as an experimental CPU/GPU-by-tier path with a dedicated snapshot provider: unique non-null BIGINT key, two BIGINT dimension columns, INSERT-only input, no async / retry / residual condition, and the experimental-operators switch on; its default for low and mid load is CPU. `batchRows=4096` is the batch size. `gpuMinRows=1024`: batches with fewer rows use a JVM lookup instead of the GPU (overview section 2); the document does not define this parameter name. |
| `FlinkCpuStreamExecCalc(cudfEngine=cpu, configured=auto)` | The Calc nodes stay on the Flink CPU (see above). |
| `TableSourceScan` | Unchanged Flink node. |

### Plan differences per mode

| Mode | Lookup node |
| --- | --- |
| `cpu` | `LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], ...)` (full in-memory HashMap) |
| `gpu` | `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1024)` |
| `gpu-min1` (control) | `GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1)`: forces the GPU on every batch |

The `gpu-min1` line, from the plan of the `gpu-min1` run:

```
+- GpuBoundedLookupJoin(batchRows=4096, gpuMinRows=1) LookupJoin(table=[default_catalog.default_database.video_dim], joinType=[InnerJoin], lookup=[video_id=video_id], select=[seq, video_id, video_id0, author_id])
```

Persistent and adaptive were not measured for c6. The JobGraph is one chain on both sides. On the CPU it contains `Calc -> LookupJoin -> Calc`; on the GPU the trailing Calc does not appear as a separate operator (the execution plan above still lists it):

CPU:

```
JobGraph vertices (operator chains):
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[5] -> Calc[6] -> LookupJoin[7] -> Calc[8] -> ConstraintEnforcer[9] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

GPU:

```
JobGraph vertices (operator chains):
- Source: Sequence Source -> kuairand-replay -> kuairand-watermarks -> *anonymous_datastream_source$1*[5] -> Calc[6] -> LookupJoin[7] -> ConstraintEnforcer[9] -> TableToDataStream -> capture-sink: Writer  [parallelism=1]
```

The planning-time status of the GPU runs reports a fused projection, which fits the missing trailing Calc (our reading; the document mentions fused predicate / projection for this lookup):

```
LookupJoinPlannerShimStatus = gpu-replaced:gpu-native:bounded-readonly-snapshot=kuairand-video-author-1039412-…:fused=directProjectionFields=3, d2hBytesPerProbe(inner/left)=10/10
```

### Evidence that the GPU ran

The planning-time status above is `gpu-replaced` on all three tiers, for both `gpu` and `gpu-min1` (the low tier snapshot has 413,436 rows). Runtime evidence: CUDA kernel instances counted with nsys (default config, one run each):

| Tier | `gpu` kernel instances | `gpu-min1` kernel instances |
| --- | --- | --- |
| Low | 3 (almost no GPU use) | 8,484 |
| Mid | 22 (almost no GPU use) | 6,151 |
| Saturated | 878 | 848 |

So the roughly 3 ms latency of `gpu` at low/mid rates is not caused by the GPU; only the saturated-tier GPU result has runtime evidence. The formal timed runs of c6 have no runtime GPU metrics (Lookup Join exposes none); the evidence is the planning-time `shim_status` (GPU-replaced on all three tiers) plus nsys.

**At low and mid rates most batches of `gpu` use a JVM lookup** (fewer than 1,024 rows per batch), so its roughly 3 ms latency there is not caused by the GPU; `gpu-min1` is the mode that really runs the GPU on every batch.

### Results

**Default** (three modes rotated within one job)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 20,000 | <0.01 / <0.01 | MATCH |
| Low | GPU (actually JVM lookup) | 19,998 | 2.70 / 6.10 | MATCH |
| Low | GPU min1 | 19,998 | 2.71 / 6.14 | MATCH |
| Mid | CPU | 100,000 | <0.01 / 14.76 | MATCH |
| Mid | GPU (actually JVM lookup) | 99,990 | 3.18 / 9.48 | MATCH |
| Mid | GPU min1 | 99,991 | 2.96 / 8.97 | MATCH |
| Saturated | CPU | 601,930 | <0.01 / <0.01 | MATCH |
| Saturated | GPU | 838,326 | 2.33 / 4.70 | MATCH |
| Saturated | GPU min1 | 834,406 | 2.33 / 4.76 | MATCH |

**Tuned** (low tier: not measured)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 100,000 | <0.01 / 19.20 | MATCH |
| Mid | GPU | 99,990 | 2.93 / 6.28 | MATCH |
| Mid | GPU min1 | 99,994 | 2.78 / 6.20 | MATCH |
| Saturated | CPU | 1,299,187 | <0.01 / <0.01 | MATCH |
| Saturated | GPU | 2,284,322 | 0.88 / 1.75 | MATCH |
| Saturated | GPU min1 | 2,237,176 | 0.89 / 1.76 | MATCH |

### Time composition

- Lookup Join has no runtime GPU timers, so H2D / kernel / D2H are not broken down for c6. The nsys kernel instance counts above are the only runtime GPU data for this case.
- **The c6 JFR share is not used (sampled estimate, misclassified)**: the JFR classifier counts every class under the Mega-Flink package as "GPU operator JVM side", and the CPU lookup's dimension-table class happens to live in that package, so the CPU job also shows about 24% "GPU-related". This is confirmed to be a misclassification, so no conclusion is drawn from c6 JFR shares; the "GPU-related" share for other cases that use the same package (including the driver class) should also be re-checked.

### Takeaway

At saturation the GPU is faster: +39% by default (838k vs 602k) and +76% after tuning. At low/mid rates throughput is capped by the input rate and is equal; the GPU adds about 3 ms of latency from batching. Only the simplest shape was verified: a single BIGINT key with a few BIGINT columns.

**LEFT lookup**: cannot run on the GPU in this build and falls back to CPU, so there is no performance comparison. Planning-time status: with a primary key on the dimension table, the output dimension columns do not match the dimension schema (LEFT makes dimension columns nullable while the primary-key column is non-null, so the types differ); without the primary key, the lookup key is reported as not proven to contain the dimension primary key.
