<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Group 2: fusable, but too small a share of total time (Case C, Case D)

> All execution plans, fusion decisions and kernel sources below are the original text exported automatically during the runs; only line wrapping, elisions (marked `...`) and path substitution (the data directory is written as `<data>`) were applied.
> Data: KuaiRand-27K, converted to Parquet, **92,919,789 rows**.
> Environment: 1 NVIDIA L20, Spark 3.5.9 local[8], spark-rapids 26.08.1, plugin `gpu-fusion-plugin-0.1.0`.
> Performance numbers: 10 warm-up rounds and 20 timed rounds each, with the order of the two configurations alternating every round. The time breakdown comes from a separate 20-round run with DEBUG metrics (2 warm-up rounds), so its total wall time differs slightly from the performance table.

**What defines this group**: the query contains expressions that can be fused (C: a filter + 5 features; D: a filter with 3 ops), the plugin does fuse them, and the fused operators' own time does drop; but they account for a very small share of the whole query. In C most of the time goes to reading and writing Parquet; in D it goes to the shuffle after the joins.

**One-line conclusion**: in this group the plugin is **neither faster nor slower**: C median 0.722 s vs 0.711 s (plugin faster in 10/20 rounds), D median 2.346 s vs 2.349 s (10/20); both gaps are within noise, and results are bit-identical to spark-rapids.

## Case C: filter samples first, then compute features (write Parquet)

### 0. What this case computes

Drop invalid and too-short exposures, keep only feed traffic, then compute 5 features on the remaining rows. Both the filter and the features are row-wise operations; this "filter first, compute after" pattern is the most common one in sample generation.

```sql
SELECT
  user_id,
  video_id,
  least(play_time_ms / (duration_ms + 1D), 5D)                AS watch_ratio,
  log(1D + play_time_ms / 1000D)                              AS log_play_sec,
  CASE WHEN duration_ms < 15000 THEN 0
       WHEN duration_ms < 60000 THEN 1 ELSE 2 END             AS duration_bucket,
  is_like + is_follow + is_comment + is_forward               AS interact_cnt,
  is_click
FROM log
WHERE duration_ms > 0
  AND play_time_ms >= 0
  AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
```

After filtering, **45,611,516 rows** remain (out of 92,919,789).

### 1. Original Spark (CPU) physical plan

```
*(1) Project [user_id#0, video_id#1, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#124, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#125, CASE WHEN (duration_ms#13 < 15000) THEN 0 WHEN (duration_ms#13 < 60000) THEN 1 ELSE 2 END AS duration_bucket#126, (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#127, is_click#5]
+- *(1) Filter ((((((isnotnull(duration_ms#13) AND isnotnull(play_time_ms#12)) AND isnotnull(is_rand#17)) AND (duration_ms#13 > 0)) AND (play_time_ms#12 >= 0)) AND (is_rand#17 = 0)) AND ((play_time_ms#12 > 3000) OR (is_click#5 = 1)))
   +- *(1) ColumnarToRow
      +- FileScan parquet [user_id#0,video_id#1,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,play_time_ms#12,duration_ms#13,is_rand#17] Batched: true, DataFilters: [isnotnull(duration_ms#13), isnotnull(play_time_ms#12), isnotnull(is_rand#17), (duration_ms#13 > ..., Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(duration_ms), IsNotNull(play_time_ms), IsNotNull(is_rand), GreaterThan(duration_ms,0),..., ReadSchem ...

```

How to read it: `Filter` and `Project` are both inside `*(1)`, i.e. the same generated Java code; the extra `isnotnull(...)` terms in the filter are added automatically by Spark (`>` and `=` are already unsatisfied on NULL).

### 2. GPU physical plan translated by spark-rapids (baseline)

The baseline plan is the plan from step 4 with the "Fused" wording removed: `FileScan -> GpuFilter -> GpuCoalesceBatches -> GpuProject -> GpuColumnarToRow`
(the full original text is in the exported `plan_gpu.txt` in the same directory). `GpuFilter` calls cuDF once per operation in the condition to get a boolean column, then selects rows;
`GpuProject` likewise makes one cuDF call per operation.

### 3. What the plugin saw and what it decided to fuse

```
== Fusion decisions (SAFE) ==
FUSED filter: fused 15 ops -> 1 mask kernel filt_10d674684724f049; 4 inputs, 0 cuDF-evaluated sub-expr(s) 
FUSED project: fused 14 ops -> 1 kernel fused_727b555473fc3846; 4 kernel outputs, 6 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s) 
```

- **Filter**: the 15 operations in the condition (3 `IS NOT NULL`, 4 comparisons, `AND`/`OR`) are fused into **1 "mask kernel"** that reads only 4 columns and writes a single boolean mask column directly; selecting rows is still done by cuDF.
- **Project**: 14 operations that produce 4 of the 5 features (`watch_ratio`, `log_play_sec`, `duration_bucket`, `interact_cnt`) are fused into **1 kernel** reading 6 columns; `user_id`, `video_id` and `is_click` are passed through unchanged.
- No sub-expressions had to be handed to cuDF.

### 4. Fused plan and kernel

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#146, LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#147, CASE WHEN (duration_ms#13 < 15000) THEN 0 WHEN (duration_ms#13 < 60000) THEN 1 ELSE 2 END AS duration_bucket#148, (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#149, is_click#5], true, fused 14 ops -> 1 kernel fused_727b555473fc3846; 4 kernel outputs, 6 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuCoalesceBatches targetsize(1073741824)
      +- FusedGpuFilterExec ((((((gpuisnotnull(duration_ms#13) AND gpuisnotnull(play_time_ms#12)) AND gpuisnotnull(is_rand#17)) AND (duration_ms#13 > 0)) AND (play_time_ms#12 >= 0)) AND (is_rand#17 = 0)) AND ((play_time_ms#12 > 3000) OR (is_click#5 = 1))), fused 15 ops -> 1 mask kernel filt_10d674684724f049; 4 inputs, 0 cuDF-evaluated sub-expr(s)
         +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,play_time_ms#12,duration_ms#13,is_rand#17] Batched: true, DataFilters: [isnotnull(duration_ms#13), isnotnull(play_time_ms#12), isnotnull(is_rand#17), (duration_ms#13 > ..., Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(duration_ms), IsNotNull(play_time_ms), IsNotNull(is_rand), GreaterThan(duration_ms,0),..., ReadSchema: struct<user_id:int,video_id:int,is_click:int,is_like:int,is_follow:int,is_comment:int,is_forward:...
```

Filter mask kernel (`kernels/filt_10d674684724f049.cu`, 281 lines in total: device function definitions first, then the kernel body; the middle is elided here):

```cuda
extern "C" __global__ void filt_10d674684724f049(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    const i32* __restrict__ c1, const u32* __restrict__ m1,
    const i32* __restrict__ c2, const u32* __restrict__ m2,
    const i32* __restrict__ c3, const u32* __restrict__ m3,
    bool* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  bool v2 = n1;
  bool n2 = true;
  i32 v3 = ld_i32(c1, i, inb);
  bool n3 = inb && vld(m1, i);
  bool v4 = n3;
  bool n4 = true;
  bool v5 = (n2 && n4 && v2 && v4);
  bool n5 = ((n2 && !v2) || (n4 && !v4) || (n2 && n4));
  i32 v6 = ld_i32(c2, i, inb);
  bool n6 = inb && vld(m2, i);
  bool v7 = n6;
  bool n7 = true;
  bool v8 = (n5 && n7 && v5 && v7);
  bool n8 = ((n5 && !v5) || (n7 && !v7) || (n5 && n7));
  i32 v9 = (i32)0;
  bool n9 = true;
  ...   // 15 ops: 3 IS NOT NULL, 4 comparisons, AND / OR, all done in registers
  if (inb) o0[i] = st_bool(v23);
  { u32 b = __ballot_sync(0xffffffffu, inb && n23); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
}
```

Compare with step 2: in the baseline every comparison and every AND is a cuDF call that writes out a full boolean column; here the intermediate values of all 15 operations stay in registers and only one mask column is written at the end.
The project kernel has the same structure as the Case A kernel in the [Group 1 document](group1-fusion-pays-off.md), only shorter (`kernels/fused_727b555473fc3846.cu`, 306 lines).

### 5. Performance and result consistency

**Consistency** (45,611,516 rows after filtering, compared row by row): spark-rapids vs plugin **0 rows differ**; CPU vs GPU differ in 2,093,458 rows, from the same source as Case A in Group 1 (the last digit of `log_play_sec`).

**Performance** (read Parquet + filter + features + write Parquet):

| Configuration | Rounds | Min | Median | Max | vs spark-rapids baseline |
| --- | --- | --- | --- | --- | --- |
| CPU Spark (reference) | 5 | 5.241 | 5.472 | 6.125 | spark-rapids is 7.7x faster |
| spark-rapids (baseline) | 20 | 0.699 | 0.711 | 0.775 | - |
| spark-rapids + plugin | 20 | 0.692 | 0.722 | 0.828 | -1.4% (positive = plugin faster); plugin faster in 10 of 20 rounds |

Times in seconds, wall time of the whole query. spark-rapids and the plugin each ran 20 times, with **10 warm-up rounds not counted** (with only 2 warm-up rounds the first few timed rounds were slow, hence 10), and **the order of the two alternates every round** (baseline first in even rounds, plugin first in odd rounds); same Spark session, same data, same spark-rapids settings, differing only in `spark.rapids.sql.fusion.mode` (OFF / SAFE). CPU is only a reference and was run separately up front, 5 times (plus 1 warm-up).

### 6. Time breakdown (prepare / read / compute / write)

| Phase (wall seconds, median) | spark-rapids | + plugin | Difference (plugin - baseline) | Measured or estimated |
| --- | --- | --- | --- | --- |
| Prepare (planning, listing files, job scheduling, submission) | 0.157 | 0.163 | +0.006 | **Measured**: total wall time - all Spark job time |
| Read | 0.340 | 0.333 | -0.007 | Estimated (apportioned): job time x scan operator time / total task run time |
| Compute | 0.024 | 0.009 | -0.015 | Estimated (apportioned): same, using the time of all operators other than scan/write/exchange |
| Write/output | 0.196 | 0.203 | +0.006 | Estimated (apportioned): same, using the write-file operator time |
| Other (task time not explained by the metrics above) | 0.001 | 0.001 | +0.000 | Estimated (apportioned): the remainder |
| **Total wall time** | **0.733** | **0.716** | **-0.017** | **Measured** |

| Task-seconds (summed over the 8 task threads, median) | spark-rapids | + plugin | Difference |
| --- | --- | --- | --- |
| Read | 2.037 | 1.937 | -0.100 |
| Compute | 0.145 | 0.053 | -0.092 |
| Write/output | 1.154 | 1.175 | +0.021 |

The job runs 8 task threads in parallel, and "task-seconds" is the sum of the per-thread times. The wall times for read / compute / write are **estimates (apportioned), not measured**: job time is split by each operator class's share of total task time. "Prepare" and total wall time are measured.

- **What was fused:** filter 15 ops -> 1 mask kernel, project 14 ops -> 1 kernel. Measured for the compute part: `GpuFilter` 0.050 + `GpuProject` 0.081 = 0.130 (task-seconds) -> `FusedGpuFilterExec` 0.016 + `FusedGpuProjectExec` 0.024 = 0.039 (median per-round difference -0.090, lower in 20/20 rounds). nsys: kernel count in the compute part 1001 -> 220, kernel GPU time for this part 0.046 -> 0.022 s, stream sync count in the compute part 2453 -> 583 (waiting 0.082 -> 0.033 s).
- **The most that can be saved is this 0.09 (task-seconds), which is only 0.015 s of wall time** (compute: 0.024 -> 0.009). C is dominated by "read" (scan 2.0, of which `filter time` - row-group filtering inside the reader - alone is 1.5), so the small fusion gain is swamped.
- Read part: scan time -0.069 (median per-round), -0.046 after removing GPU semaphore wait, but the interquartile range [-0.33, -0.01] is wide and only 16/20 rounds are lower; GPU decode is unchanged (+0.005). **This read-part difference is within noise and cannot be credited to the plugin**; unlike A, the compute-part saving in C is too small to make other threads queue less.
- Write part +0.026 (plugin lower in 6/20 rounds), within noise.
- Reconciliation: median per-round difference in total task run time is -0.197; compute -0.09 + read -0.07 + write +0.03 = about -0.13, which roughly adds up (the remainder is operators not listed separately).

### 7. Conclusion

- Results are bit-identical; plugin median 0.722 s vs baseline 0.711 s, plugin faster in 10 of 20 rounds, **no measurable difference** (an earlier run with the same setup showed the plugin about 2% faster; the two runs disagree in direction, which shows the gap is within noise).
- **C's content is fusable (filter 15 ops and project 14 ops, each fused into 1 kernel), and the fused operators' own time did drop** (filter + project from 0.13 to 0.04 task-seconds), but this part is too small a share of the whole query: most of the time goes to reading Parquet (scan) and writing Parquet, so nothing shows up end to end, and nothing gets slower either.
- So C is "fusable but too small a share" and **not a case that demonstrates a plugin benefit**. A benefit requires compute to take a larger share (see Group 1).

## Case D: joins + string feature + group-by aggregation

### 0. What this case computes

Join the exposure log with the user table and the video table, take the first two characters of the video tag, and group by (user activity level, video type, tag prefix) to compute exposure count, click-through rate and long-view rate.
This is the typical shape of "joins + aggregation in feature engineering", with **almost no row-wise numeric operations to fuse**; it checks that the plugin **does not slow things down** in this scenario.

```sql
-- Expected: little or nothing for the plugin to fuse.
SELECT
  u.user_active_degree,
  v.video_type,
  substring(v.tag, 1, 2)                                      AS tag_prefix,
  count(*)                                                    AS exposures,
  avg(l.is_click)                                             AS ctr,
  avg(l.long_view)                                            AS long_view_rate
FROM log l
JOIN user u        ON l.user_id = u.user_id
JOIN video_basic v ON l.video_id = v.video_id
GROUP BY u.user_active_degree, v.video_type, substring(v.tag, 1, 2)
```

### 1. Original Spark (CPU) physical plan

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[user_active_degree#39, video_type#102, _groupingexpression#137], functions=[count(1), avg(is_click#5), avg(long_view#11)], output=[user_active_degree#39, video_type#102, tag_prefix#124, exposures#125L, ctr#126, long_view_rate#127])
   +- Exchange hashpartitioning(user_active_degree#39, video_type#102, _groupingexpression#137, 8), ENSURE_REQUIREMENTS, [plan_id=81]
      +- HashAggregate(keys=[user_active_degree#39, video_type#102, _groupingexpression#137], functions=[partial_count(1), partial_avg(is_click#5), partial_avg(long_view#11)], output=[user_active_degree#39, video_type#102, _groupingexpression#137, count#143L, sum#144, count#145L, sum#146, count#147L])
         +- Project [is_click#5, long_view#11, user_active_degree#39, video_type#102, substring(tag#111, 1, 2) AS _groupingexpression#137]
            +- SortMergeJoin [video_id#1], [video_id#100], Inner
               :- Sort [video_id#1 ASC NULLS FIRST], false, 0
               :  +- Exchange hashpartitioning(video_id#1, 8), ENSURE_REQUIREMENTS, [plan_id=73]
               :     +- Project [video_id#1, is_click#5, long_view#11, user_active_degree#39]
               :        +- BroadcastHashJoin [user_id#0], [user_id#38], Inner, BuildRight, false
               :           :- Filter (isnotnull(user_id#0) AND isnotnull(video_id#1))
               :           :  +- FileScan parquet [user_id#0,video_id#1,is_click#5,long_view#11] Batched: true, DataFilters: [isnotnull(user_id#0), isnotnull(video_id#1)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(user_id), IsNotNull(video_id)], ReadSc ...
               :           +- BroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false] as bigint)),false), [plan_id=68]
               :              +- Filter isnotnull(user_id#38)
               :                 +- FileScan parquet [user_id#38,user_active_degree#39] Batched: true, DataFilters: [isnotnull(user_id#38)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/user], PartitionFilters: [], PushedFilters: [IsNotNull(user_id)], ReadSchema: struct<user_id:int,user_active_degree:string>
               +- Sort [video_id#100 ASC NULLS FIRST], false, 0
                  +- Exchange hashpartitioning(video_id#100, 8), ENSURE_REQUIREMENTS, [plan_id=74]
                     +- Filter isnotnull(video_id#100)
                        +- FileScan parquet [video_id#100,video_type#102,tag#111] Batched: true, DataFilters: [isnotnull(video_id#100)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/video_basic], PartitionFilters: [], PushedFilters: [IsNotNull(video_id)], ReadSchema: struct<video_id:int,video_type:string,tag ...

```

How to read it: two joins (log x user table uses a broadcast join, x video table uses a sort-merge join), a string `substring` in the Project, and finally a group-by aggregation on three keys.
The only "fusion" the CPU can do is the `*(n)` Java code generation.

### 2. GPU physical plan translated by spark-rapids (baseline)

All the main operators can run on the GPU: `GpuBroadcastHashJoin`, `GpuShuffledSymmetricHashJoin`, `GpuHashAggregate`, `GpuProject` (including `gpusubstring`), `GpuFilter`.
The full original text is in step 4 (the plugin replaces only one operator; the rest is identical to the baseline).

### 3. What the plugin saw and what it decided to fuse

```
== Fusion decisions (SAFE) ==
FUSED filter: fused 3 ops -> 1 mask kernel filt_804b8531df4b562e; 2 inputs, 0 cuDF-evaluated sub-expr(s) 
NOT FUSED GpuFilter (u.user_id IS NOT NULL): region too small (1 op(s) < minOps=2)
NOT FUSED GpuFilter (v.video_id IS NOT NULL): region too small (1 op(s) < minOps=2)
NOT FUSED GpuProject [is_click, long_view, user_active_degree, video_type, gpusubstring(v.tag, 1, 2) AS _groupingexpression]: no fusable expression; output 4 (gpusubstring(v.tag, 1, 2) AS _groupingexpression): GpuSubstring is not in the fusion whitelist
```

Item by item:

| Location | Decision | Reason |
| --- | --- | --- |
| Filter on the log table `user_id IS NOT NULL AND video_id IS NOT NULL` | **Fused** (3 ops -> 1 mask kernel) | Op count reaches the threshold |
| Filter on the user table `user_id IS NOT NULL` | Not fused | Only 1 op, below the threshold `minOps=2`: fusing saves nothing and still pays compile and launch cost |
| Filter on the video table `video_id IS NOT NULL` | Not fused | Same, 1 op |
| `GpuProject [..., substring(tag,1,2)]` | Not fused | The only computed column is `substring`; string functions are not in the plugin's supported list; the other columns are just passed through |
| join, aggregation, shuffle | Untouched | Not row-wise expressions; outside the plugin's scope |

### 4. Fused plan and kernel

```
GpuColumnarToRow false
+- GpuHashAggregate (keys=[user_active_degree#39, video_type#102, _groupingexpression#319], functions=[gpucount(1, false), avg(is_click#5, DoubleType, false), avg(long_view#11, DoubleType, false)], output=[user_active_degree#39, video_type#102, tag_prefix#232, exposures#233L, ctr#234, long_view_rate#235]) 
   +- GpuShuffleCoalesce 1073741824
      +- GpuColumnarExchange gpuhashpartitioning(user_active_degree#39, video_type#102, _groupingexpression#319, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=952]
         +- GpuHashAggregate (keys=[user_active_degree#39, video_type#102, _groupingexpression#319], functions=[partial_gpucount(1, false), partial_avg(is_click#5, DoubleType, false), partial_avg(long_view#11, DoubleType, false)], output=[user_active_degree#39, video_type#102, _groupingexpression#319, count#325L, sum#326, count# ...
            +- GpuProject [is_click#5, long_view#11, user_active_degree#39, video_type#102, gpusubstring(tag#111, 1, 2) AS _groupingexpression#319], true
               +- GpuShuffledSymmetricHashJoin Inner, [video_id#1], [video_id#100], false, 1073741824, 1.0, CoalesceReadOption(true,GPU,Never,None,false), false
                  :- GpuColumnarExchange gpuhashpartitioning(video_id#1, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=948]
                  :  +- GpuProject [video_id#1, is_click#5, long_view#11, user_active_degree#39], true
                  :     +- GpuBroadcastHashJoin [user_id#0], [user_id#38], Inner, GpuBuildRight, false
                  :        :- GpuCoalesceBatches targetsize(1073741824)
                  :        :  +- FusedGpuFilterExec (gpuisnotnull(user_id#0) AND gpuisnotnull(video_id#1)), fused 3 ops -> 1 mask kernel filt_804b8531df4b562e; 2 inputs, 0 cuDF-evaluated sub-expr(s)
                  :        :     +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,long_view#11] Batched: true, DataFilters: [isnotnull(user_id#0), isnotnull(video_id#1)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(user_id), Is ...
                  :        +- GpuBroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false] as bigint)),false), [plan_id=910]
                  :           +- GpuCoalesceBatches targetsize(1073741824)
                  :              +- GpuFilter gpuisnotnull(user_id#38)
                  :                 +- GpuFileGpuScan parquet [user_id#38,user_active_degree#39] Batched: true, DataFilters: [isnotnull(user_id#38)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/user], PartitionFilters: [], PushedFilters: [IsNotNull(user_id)], ReadSchema: struct<user_id:int, ...
                  +- GpuColumnarExchange gpuhashpartitioning(video_id#100, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=915]
                     +- GpuCoalesceBatches targetsize(1073741824)
                        +- GpuFilter gpuisnotnull(video_id#100)
                           +- GpuFileGpuScan parquet [video_id#100,video_type#102,tag#111] Batched: true, DataFilters: [isnotnull(video_id#100)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/video_basic], PartitionFilters: [], PushedFilters: [IsNotNull(video_id)], ReadSchema: struct<video_id ...
```

Only one thing changes in the whole plan: the `GpuFilter` on the log table becomes `FusedGpuFilterExec`; all other operators are unchanged.

The only kernel generated is this filter's mask kernel (`kernels/filt_804b8531df4b562e.cu`, 1 line in total: device function definitions first, then the kernel body):

```cuda
extern "C" __global__ void filt_804b8531df4b562e(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    const i32* __restrict__ c1, const u32* __restrict__ m1,
    bool* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  bool v2 = n1;
  bool n2 = true;
  i32 v3 = ld_i32(c1, i, inb);
  bool n3 = inb && vld(m1, i);
  bool v4 = n3;
  bool n4 = true;
  bool v5 = (n2 && n4 && v2 && v4);
  bool n5 = ((n2 && !v2) || (n4 && !v4) || (n2 && n4));
  bool v6 = (v5 && n5);
  bool n6 = true;
  if (inb) o0[i] = st_bool(v6);
  { u32 b = __ballot_sync(0xffffffffu, inb && n6); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  }
}
```

3 ops (two `IS NOT NULL` and one `AND`) are fused into one kernel that reads 2 columns and writes 1 boolean mask column; selecting rows is still done by cuDF.

### 5. Performance and result consistency

**Consistency**: the group-by result has 1,021 rows; spark-rapids vs plugin, CPU vs spark-rapids, and CPU vs plugin all have **0 rows differing**.

**Performance** (read three tables + joins + aggregation + write Parquet):

| Configuration | Rounds | Min | Median | Max | vs spark-rapids baseline |
| --- | --- | --- | --- | --- | --- |
| CPU Spark (reference) | 5 | 12.490 | 12.806 | 15.411 | spark-rapids is 5.5x faster |
| spark-rapids (baseline) | 20 | 2.269 | 2.349 | 2.693 | - |
| spark-rapids + plugin | 20 | 2.298 | 2.346 | 2.544 | +0.1% (positive = plugin faster); plugin faster in 10 of 20 rounds |

Times in seconds, wall time of the whole query. spark-rapids and the plugin each ran 20 times, with **10 warm-up rounds not counted** (with only 2 warm-up rounds the first few timed rounds were slow, hence 10), and **the order of the two alternates every round** (baseline first in even rounds, plugin first in odd rounds); same Spark session, same data, same spark-rapids settings, differing only in `spark.rapids.sql.fusion.mode` (OFF / SAFE). CPU is only a reference and was run separately up front, 5 times (plus 1 warm-up).

### 6. Time breakdown (prepare / read / compute / write)

| Phase (wall seconds, median) | spark-rapids | + plugin | Difference (plugin - baseline) | Measured or estimated |
| --- | --- | --- | --- | --- |
| Prepare (planning, listing files, job scheduling, submission) | 0.111 | 0.114 | +0.003 | **Measured**: total wall time - all Spark job time |
| Read | 0.218 | 0.217 | -0.001 | Estimated (apportioned): job time x scan operator time / total task run time |
| Compute | 0.104 | 0.099 | -0.005 | Estimated (apportioned): same, using the time of all operators other than scan/write/exchange |
| Exchange (shuffle) | 1.397 | 1.404 | +0.007 | Estimated (apportioned): same, using shuffle read + partitioning + serialization |
| Write/output | 0.004 | 0.004 | +0.000 | Estimated (apportioned): same, using the write-file operator time |
| Other (task time not explained by the metrics above) | 0.581 | 0.582 | +0.000 | Estimated (apportioned): the remainder |
| **Total wall time** | **2.421** | **2.418** | **-0.002** | **Measured** |

| Task-seconds (summed over the 8 task threads, median) | spark-rapids | + plugin | Difference |
| --- | --- | --- | --- |
| Read | 1.301 | 1.294 | -0.007 |
| Compute | 0.626 | 0.593 | -0.033 |
| Exchange (shuffle) | 8.354 | 8.393 | +0.039 |
| Write/output | 0.022 | 0.022 | -0.000 |

The job runs 8 task threads in parallel, and "task-seconds" is the sum of the per-thread times. The wall times for read / compute / write are **estimates (apportioned), not measured**: job time is split by each operator class's share of total task time. "Prepare" and total wall time are measured.

- **The plugin fused only one thing:** the 3-op filter on the log table. Measured `GpuFilter` op time 0.026 -> `FusedGpuFilterExec` 0.009 (task-seconds, saving 0.019); the other two 1-op filters and the `substring` project were not fused (`GpuProject` 0.064 vs 0.064, unchanged), consistent with the fusion decisions.
- **This saving is negligible relative to the whole:** the whole query takes 13.8 task-seconds (8 threads in parallel, 2.4 s wall time), and 0.019 is 0.1%. The bulk of D is the **exchange (shuffle)**: serialization 4.65 + deserialization read 3.12 + partitioning 0.59 (task-seconds; these overlap, so they can only be called the bulk), with 1.5 GB of shuffle output; next come read 1.3 and join/aggregation operators 0.6. Apportioned wall time: read 0.218, compute 0.104, exchange 1.397, other 0.58.
- nsys comparison (8 rounds): total kernel count 4289 vs 4256 (33 fewer, exactly the fused filter), total GPU time 0.374 vs 0.371 s, sync count 6718 vs 6704, and both sides have identical H2D 3.8 GB / D2H 3.25 GB.
- Wall time 2.421 vs 2.418 (40 rounds): no measurable difference, and not slower (plugin faster in 21 of 40 rounds).

### 7. Conclusion

- D shows that **the plugin does not slow down queries it is not suited to** (nor does it speed them up): median 2.346 s vs 2.349 s, a gap under 0.2%, plugin faster in 10 of 20 rounds, no measurable difference.
- The main operators in the query are joins, shuffle and aggregation, none of which are in the plugin's scope (shuffle is the largest piece); the plugin fused only one 3-op filter, and the other two 1-op filters were deliberately skipped because of `minOps=2`.
- The basis for every decision can be looked up in `fusion_decisions.txt`: each place that was not fused states its reason.
