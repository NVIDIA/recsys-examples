<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Group 1: fusion pays off (Case A, Case B)

> All execution plans, fusion decisions and kernel sources in this document are originals exported automatically during the actual runs; only line wrapping, elisions (marked `...`) and path substitution (the data directory is written as `<data>`) were applied.
> Data: KuaiRand-27K, converted to Parquet: **92,919,789 rows**.
> Environment: 1 NVIDIA L20, Spark 3.5.9 local[8], spark-rapids 26.08.1, plugin `gpu-fusion-plugin-0.1.0`.
> Performance numbers: 10 warm-up rounds and 20 timed rounds per configuration, with the order of the two configurations alternating each round. The time breakdown comes from a separate 20-round run with DEBUG metrics (2 warm-up rounds), so its total wall time differs slightly from the performance table.

**What this group has in common**: each row computes many features (10 features, 27 operations, including type casts, division, `least`, `log`, `CASE`, `pmod`, and so on). spark-rapids makes one cuDF call per operation, so every step writes a whole column to GPU memory and the next step reads it back, and each call is often followed by a synchronization wait. The plugin fuses these 27 operations into **1 kernel**, keeping all intermediate values in registers.

**One-sentence conclusion**: in this group the plugin is **consistently faster**. Case A, which writes Parquet, is about **6.6%** faster (0.759 s vs 0.809 s, faster in 17 of 20 rounds); Case B, which writes no file and keeps the result on the GPU, is about **21%** faster (0.299 s vs 0.361 s, faster in all 20 rounds). B removes the file read/write interference and can be seen as the upper bound for this dataset on this GPU. Results are bit-identical to spark-rapids.

## Case A: Common sample features (read from Parquet, write back to Parquet)

### 0. What this case computes

For every impression log row, compute 10 features and labels. This is the most common kind of per-row computation in recommendation sample generation (see the [README](README.md#where-the-features-come-from) for where each feature comes from):

```sql
SELECT
  user_id, video_id,
  CAST(hourmin / 100 AS INT)                              AS hour_of_day,     -- hour
  dayofweek(CAST(time_ms DIV 1000 AS TIMESTAMP))          AS day_of_week,     -- day of week
  least(play_time_ms / (duration_ms + 1D), 5D)            AS watch_ratio,     -- watch ratio, capped at 5
  log(1D + play_time_ms / 1000D)                          AS log_play_sec,    -- log of play seconds
  CASE WHEN duration_ms < 7000 THEN 0 ... ELSE 5 END      AS duration_bucket, -- duration in 6 buckets
  CASE WHEN play_time_ms >= duration_ms THEN 1 ELSE 0 END AS complete_play,   -- fully watched or not
  is_like + is_follow + is_comment + is_forward           AS interact_cnt,    -- interaction count
  is_click, long_view,                                                        -- labels passed through unchanged
  pmod(user_id * 1000003 + tab, 1048576)                  AS user_tab_cross   -- user x tab hash cross
FROM log
```

### 1. Original Spark (CPU) physical plan

```
*(1) Project [user_id#0, video_id#1,
              cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#124,
              dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#125,
              least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#126,
              LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#127,
              CASE WHEN (duration_ms#13 < 7000) THEN 0 ... ELSE 5 END AS duration_bucket#128,
              CASE WHEN (play_time_ms#12 >= duration_ms#13) THEN 1 ELSE 0 END AS complete_play#129,
              (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#130,
              is_click#5, long_view#11,
              pmod(((user_id#0 * 1000003) + tab#18), 1048576) AS user_tab_cross#131]
+- *(1) ColumnarToRow
   +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] Batched: true, Format: Parquet, ...
```

How to read it:
- Execution runs bottom-up. `FileScan` reads Parquet, `ColumnarToRow` converts columns to rows, and `Project` computes all the new columns row by row.
- `*(1)` means Spark put `Project` and the row conversion into the same generated Java code (WholeStageCodegen), which is the **"fusion" on the CPU**.
- Spark inserted many type casts itself: `play_time_ms` is an int, so it is cast `cast(... as double)` before dividing by a double; `log(x)` is rewritten as `LOG(e, x)`.

### 2. GPU physical plan after spark-rapids (baseline)

```
GpuColumnarToRow false
+- GpuProject [user_id#0, video_id#1,
               cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#144,
               gpudayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#145,
               gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#146,
               LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#147,
               CASE WHEN (duration_ms#13 < 7000) THEN 0 ... ELSE 5 END AS duration_bucket#148,
               CASE WHEN (play_time_ms#12 >= duration_ms#13) THEN 1 ELSE 0 END AS complete_play#149,
               (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#150,
               is_click#5, long_view#11,
               (((user_id#0 * 1000003) + tab#18) pmod 1048576) AS user_tab_cross#151], true
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] Batched: true, Format: Parquet, ...
```

Compared with step 1, spark-rapids did three things:
1. `FileScan` -> `GpuFileGpuScan`: Parquet is decoded on the GPU, and the data goes straight into GPU memory in cuDF column format.
2. `Project` -> `GpuProject`: each expression is replaced by its GPU version (e.g. `dayofweek` -> `gpudayofweek`, `least` -> `gpuleast`, `LOG` -> `LOG_BASE`).
3. The column-to-row step in the middle is gone; `GpuColumnarToRow` appears only at the end, when the result is handed back to Spark.

**The problem is how `GpuProject` executes**: for every operation in the expression tree (cast, division, addition, comparison, CASE, ...) it makes one cuDF call,
and each step writes its result as a whole column in GPU memory, which the next step then reads back.

### 3. What the plugin saw and what it decided to fuse

The plugin hooks in after spark-rapids (`postColumnarTransitions`), sees the `GpuProject` from step 2, and translates it output column by output column. The exported decision text:

```
FUSED project: fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs,
  1 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
  cut: CAST((log.time_ms div 1000L) AS TIMESTAMP) evaluated by cuDF (cast i64 -> ts not supported)
```

Item by item:

| Output column | Handling | Reason |
| --- | --- | --- |
| `user_id`, `video_id`, `is_click`, `long_view` | Passed through as is, not in the kernel | Just column pass-through, no computation |
| `hour_of_day`, `watch_ratio`, `log_play_sec`, `duration_bucket`, `complete_play`, `interact_cnt`, `user_tab_cross` | In the kernel | Every operation has a matching device function |
| `day_of_week` | In the kernel, except that the small piece `CAST(time_ms DIV 1000 AS TIMESTAMP)` is handed to cuDF first | The plugin does not yet support bigint -> timestamp casts. cuDF computes that piece and feeds it to the kernel as an input column; `dayofweek` itself is still computed in the kernel |

Result: **8 output columns, 27 operations -> 1 kernel**; the kernel reads 10 input columns (one of which is the timestamp precomputed by cuDF).

### 4. Fused plan and kernel

In the execution plan `GpuProject` is replaced by `FusedGpuProjectExec`; the other nodes are unchanged:

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, ... same expressions as step 2 ...], true,
     fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs, 1 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] ...
```

The CUDA source the plugin generated on the fly at runtime (`kernels/fused_88a94965d48afb79.cu`, 381 lines in total; the first 219 lines are the definitions of all device functions;
below is the kernel body, with some repetitive lines omitted):

```cuda
extern "C" __global__ void fused_88a94965d48afb79(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,     // input column hourmin + NULL bitmap
    const i64* __restrict__ c1, const u32* __restrict__ m1,     // timestamp precomputed by cuDF
    ...                                                         // c2..c9: play_time_ms, duration_ms, is_like ... tab
    i32* __restrict__ o0, u32* __restrict__ om0,                // output hour_of_day + NULL bitmap
    ...                                                         // o1..o7: the other 7 features
{
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {   // one thread per row
  const int i = base + threadIdx.x;
  // hour_of_day = CAST(hourmin / 100.0 AS INT)
  i32 v1 = ld_i32(c0, i, inb);            bool n1 = inb && vld(m0, i);        // load value + load NULL flag
  f64 v2 = (n1 ? (f64)v1 : (f64)0);       bool n2 = n1;
  f64 v3 = 100.0;                         bool n3 = true;                     // the original is a constant written as __longlong_as_double(...)
  f64 v4 = ((n2 && n3 && !is_zero(v3)) ? d_div(v2, v3) : (f64)0);             // division by 0 gives NULL, as in Spark
  i32 v5 = (n4 ? f2i_i32(v4) : (i32)0);
  // day_of_week: timestamp -> days -> weekday; if the date is out of range, raise the "out of bounds" flag and this batch takes the unfused path instead
  i64 v6 = ld_ts(c1, i, inb);
  i32 v7 = (n6 ? (i32)fdiv64(v6, 86400000000LL) : (i32)0);
  if (inb && (n7 && (v7 > 2146000000 || v7 < -2146000000))) bail(flag);
  i32 v8 = ((n7) ? dt_dow(v7) : (i32)0);
  // watch_ratio = least(play_time_ms / (duration_ms + 1.0), 5.0)
  f64 v14 = ((n12 && n13) ? w_add(v12, v13) : (f64)0);
  f64 v15 = ((n10 && n14 && !is_zero(v14)) ? d_div(v10, v14) : (f64)0);
  f64 v17 = ((n15 && n16) ? sel_f64(c_lt(v15, v16), v15, v16) : (n16 ? v16 : v15));
  // log_play_sec: v10 (play_time_ms cast to double) was already loaded above, so it is reused here with no further GPU memory read
  f64 v19 = ((n10 && n18 && !is_zero(v18)) ? d_div(v10, v18) : (f64)0);
  f64 v20 = ((n13 && n19) ? w_add(v13, v19) : (f64)0);
  f64 v22 = ((n20 && n21 && !(v20 <= 0.0) && !(v21 <= 0.0)) ? d_logbase(v20, v21) : 0.0);
  // duration_bucket: 5 comparisons + CASE, all in registers
  bool v24 = ((n11 && n23) ? c_lt(v11, v23) : (bool)0);
  ...
  i32 v43 = ((n24 && v24) ? v25 : v42);
  // complete_play, interact_cnt, user_tab_cross
  bool v44 = ((n9 && n11) ? c_ge(v9, v11) : (bool)0);
  i32 v52 = ((n50 && n51) ? w_add(v50, v51) : (i32)0);
  i32 v59 = ((n57 && n58 && !is_zero(v58)) ? d_pmod(v57, v58) : (i32)0);
  // write the 8 output columns; every 32 rows use one __ballot_sync to assemble the NULL bitmap
  if (inb) o0[i] = st_i32(v5);
  { u32 b = __ballot_sync(0xffffffffu, inb && n5); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  ...
  }
}
```

Compared with step 2: spark-rapids makes one cuDF call per operation and every intermediate result lands in GPU memory; here **each row reads each of the 10 columns it needs exactly once**,
the intermediate results of all 27 operations stay in registers (`v1`...`v59`), and only the 8 output columns are written at the end.

### 5. Performance and consistency

**Consistency** (92,919,789 rows, compared row by row):

| Comparison | Rows only on the left | Rows only on the right |
| --- | --- | --- |
| spark-rapids vs plugin | 0 | 0 |
| CPU vs spark-rapids | 5,067,411 | 5,067,411 |
| CPU vs plugin | 5,067,411 | 5,067,411 |

The plugin and spark-rapids results are **bit-identical**. The CPU-vs-GPU difference all comes from one place: the last digit of `log_play_sec` (the CPU uses Java `StrictMath`, the GPU uses the CUDA math library; this is a known difference already present in spark-rapids itself). All other columns match.

**Performance** (writing Parquet, wall time of the whole query):

| Configuration | Rounds | Min | Median | Max | Relative to the spark-rapids baseline |
| --- | --- | --- | --- | --- | --- |
| CPU Spark (reference) | 5 | 15.358 | 15.793 | 16.126 | spark-rapids is 19.5x faster |
| spark-rapids (baseline) | 20 | 0.788 | 0.809 | 0.959 | - |
| spark-rapids + plugin | 20 | 0.718 | 0.759 | 0.820 | +6.6% (positive = plugin faster); the plugin was faster in 17 of 20 rounds |

Times are in seconds, wall time of the whole query. spark-rapids and the plugin each ran 20 times, with **10 warm-up rounds not counted** (with only 2 warm-up rounds the first few timed rounds came out slow, hence 10), and **the order of the two alternates every round** (baseline first in even rounds, plugin first in odd rounds). Same Spark session, same data, same spark-rapids settings; the only difference is `spark.rapids.sql.fusion.mode` (OFF / SAFE). CPU is a reference only, run 5 times up front on its own (plus 1 warm-up round).

### 6. Time breakdown (prepare / read / compute / write)

| Phase (wall seconds, median) | spark-rapids | + plugin | Diff (plugin - baseline) | Measured or estimated |
| --- | --- | --- | --- | --- |
| Prepare (planning, listing files, job scheduling, submission) | 0.193 | 0.196 | +0.003 | **Measured**: total wall time - time of all Spark jobs |
| Read | 0.242 | 0.217 | -0.025 | Estimated (apportioned): job time x scan operator time / total task run time |
| Compute | 0.069 | 0.026 | -0.044 | Estimated (apportioned): same, using the time of the operators other than scan/write/exchange |
| Write / output | 0.298 | 0.312 | +0.014 | Estimated (apportioned): same, using the time of the file-write operator |
| Other (task time not explained by the metrics above) | 0.001 | 0.001 | -0.000 | Estimated (apportioned): the remainder |
| **Total wall time** | **0.798** | **0.751** | **-0.047** | **Measured** |

| Task-seconds (summed over the 8 task threads, median) | spark-rapids | + plugin | Diff |
| --- | --- | --- | --- |
| Read | 1.511 | 1.349 | -0.162 |
| Compute | 0.433 | 0.161 | -0.272 |
| Write / output | 1.847 | 1.960 | +0.113 |

Eight task threads run in parallel within a job, and "task-seconds" is the sum of the time over those threads. The wall times for read / compute / write are **estimated (apportioned), not measured**: the job time is split in proportion to each operator class's share of total task time. "Prepare" and total wall time are measured.

- **The gain is mainly in the "compute" phase, which matches what the plugin fuses.** 27 operations fused into 1 kernel: the `GpuProject` op time median goes from 0.433 to `FusedGpuProjectExec` 0.161 (task-seconds; it dropped in all 20 of 20 rounds, median per-round difference -0.274). In nsys (8 rounds) the number of kernels in the compute phase goes from 1001 to 209 (per round), the GPU time of those kernels from 0.108 to 0.045 s, the number of `cudaStreamSynchronize` calls in this phase from 3234 to 1232, and the total wait from 0.312 to 0.133 s. In other words, what is saved is mainly "many small cuDF calls + the synchronization wait after each call", not the GPU computing faster.
- **The "read" phase dropped by 0.16 to 0.19 (task-seconds): reading itself did not get faster; mainly the time spent waiting for the GPU went down.**
  - Source code: `scan time` is the whole duration of the scan operator's `hasNext()`, which includes the reader's wait to acquire the GPU semaphore (`GpuSemaphore.acquireIfNecessary`); at DEBUG level a version with the semaphore wait removed is also reported.
  - Measured (DEBUG metrics, median): scan time 1.511 -> 1.349 (-0.162); with the semaphore wait removed, 0.846 -> 0.800 (-0.046). **Of the 0.162 reduction, about 0.116 is GPU semaphore wait and only about 0.046 is anything else**; GPU decode time 0.413 -> 0.388 (-0.025).
  - nsys corroboration: the number and duration of read-phase kernels (decoding) are identical on both sides (1356 kernels, 0.135 vs 0.128 s), the CPU wall time of the `read_chunk` range is 0.433 vs 0.436 s, and the synchronization wait within the read phase is 0.310 vs 0.303 s.
  - Explanation (consistent with the data, though the time the GPU semaphore is held was not measured directly): each task thread holds the GPU for a shorter time in the "compute" phase, so the other task threads queue for the GPU for less time, and that wait is recorded in their scan time. **So this 0.16 is not an extra gain; it is another view of the gain in the compute phase and must not be added to the 0.27 from compute.**
- **The write phase actually got about 0.08 longer (task-seconds)**: the op time of the file-write node went from 1.847 to 1.960, and in only 2 of 20 rounds was the plugin lower; within that, GPU encode and buffer went from 0.747 to 0.812. In nsys the write-phase kernel count and duration are the same (580 kernels, 0.423 vs 0.416 s), and the write-phase CPU wall time is 0.703 vs 0.740 s. **The cause was not found.**
- Reconciliation: compute -0.27 + read -0.16 + write +0.11 (difference of medians) ~= -0.33, and the median per-round difference in total task run time is -0.350, which matches. Converted to wall time: job time -0.05 s, consistent with the total wall time difference (0.798 -> 0.751).

### 7. Conclusion

- Results are bit-identical; the plugin is **about 6.6%** faster than spark-rapids (median 0.759 s vs 0.809 s, plugin faster in 17 of 20 rounds). The margin is modest: the whole query still has to read and write 93 million rows of Parquet.
- The gain comes from the fused part: in the per-operator timings, the projection operator's time drops from 0.43 to 0.16 (task-seconds summed over the 8 task threads, median; it dropped in all 20 of 20 rounds). The rest (read, write) is not fused and did not get faster.
- To see the gain from fusion itself, remove the I/O: see the Case B section of this document (same SQL, result aggregated on the GPU, no file written).

## Case B: Same SQL as A, result aggregated on the GPU (no trip back to the CPU, no file written)

### 0. What this case computes

The SQL is **exactly the same** as Case A in this document (10 sample features; see section 0 of Case A). The only difference is what happens after the computation:
instead of writing Parquet, each feature column is summed on the GPU and only a single summary row is fetched back. This simulates "feed the features straight into GPU training once computed",
that is, keeping the time outside file reads and writes as low as possible to see how much fusion itself can bring. (The sum is only there to make sure every column is actually computed and not optimized away.)

### 1. Original Spark (CPU) physical plan

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[], functions=[count(1), sum(cast(user_id#0 as double)), sum(cast(video_id#1 as double)), sum(cast(hour_of_day#124 as double)), sum(cast(day_of_week#125 as double)), sum(watch_ratio#126), sum(log_play_sec#127), sum(cast(duration_bucket#128 as double)), sum(cast(complete_play#129 as double)), sum(cast(interact_cnt#130 as double)), sum(cast(is_click#5 as double)), sum(cast(long_view#11 as double)) ...
   +- Exchange SinglePartition, ENSURE_REQUIREMENTS, [plan_id=28]
      +- HashAggregate(keys=[], functions=[partial_count(1), partial_sum(cast(user_id#0 as double)), partial_sum(cast(video_id#1 as double)), partial_sum(cast(hour_of_day#124 as double)), partial_sum(cast(day_of_week#125 as double)), partial_sum(watch_ratio#126), partial_sum(log_play_sec#127), partial_sum(cast(duration_bucket#128 as double)), partial_sum(cast(complete_play#129 as double)), partial_sum(cast(interact_c ...
         +- Project [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#124, dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#125, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#126, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#127, CASE WHEN (duratio ...
            +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_li ...

```

How to read it: at the top is the aggregation (`HashAggregate`, first a partial sum inside each partition, then an `Exchange` to one partition for the total sum);
the `Project` in the middle is the long list of feature expressions from Case A; at the bottom is the file read.

### 2. GPU physical plan after spark-rapids (baseline)

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[], functions=[count(1), sum(cast(user_id#0 as double)), sum(cast(video_id#1 as double)), sum(cast(hour_of_day#209 as double)), sum(cast(day_of_week#210 as double)), sum(watch_ratio#211), sum(log_play_sec#212), sum(cast(duration_bucket#213 as double)), sum(cast(complete_play#214 as double)), sum(cast(interact_cnt#215 as double)), sum(cast(is_click#5 as double)), sum(cast(long_view#11 as double)) ...
   +- Exchange SinglePartition, ENSURE_REQUIREMENTS, [plan_id=79]
      +- HashAggregate(keys=[], functions=[partial_count(1), partial_sum(cast(user_id#0 as double)), partial_sum(cast(video_id#1 as double)), partial_sum(cast(hour_of_day#209 as double)), partial_sum(cast(day_of_week#210 as double)), partial_sum(watch_ratio#211), partial_sum(log_play_sec#212), partial_sum(cast(duration_bucket#213 as double)), partial_sum(cast(complete_play#214 as double)), partial_sum(cast(interact_c ...
         +- Project [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#209, dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#210, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#211, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#212, CASE WHEN (duratio ...
            +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_li ...

```

(What is exported here is the plan shell before adaptive query execution (AQE) starts; the operator names are still the original Spark ones, and the GPU operators are substituted in only at execution time. For the structure after substitution see step 4: treat `FusedGpuProjectExec` as `GpuProject` and you have the baseline; the other operators are the same.)

### 3. What the plugin saw and what it decided to fuse

```
== Fusion decisions (SAFE) ==
FUSED project: fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs, 1 cuDF-evaluated sub-expr(s), 0 cuDF output(s) cut: CAST((log.time_ms div 1000L) AS TIMESTAMP) evaluated by cuDF (cast i64 -> ts not supported)
```

Exactly the same as Case A: 8 output columns, 27 operations fused into 1 kernel (the kernel id `fused_88a94965d48afb79` is also the same; see section 4 of Case A for the source),
`CAST(time_ms DIV 1000 AS TIMESTAMP)` is not supported yet, so cuDF computes it first and feeds it to the kernel.

### 4. Fused plan and kernel

```
GpuColumnarToRow false
+- GpuHashAggregate (keys=[], functions=[gpucount(1, false), gpubasicsum(cast(user_id#0 as double), DoubleType, false), gpubasicsum(cast(video_id#1 as double), DoubleType, false), gpubasicsum(cast(hour_of_day#450 as double), DoubleType, false), gpubasicsum(cast(day_of_week#451 as double), DoubleType, false), gpubasicsum(watch_ratio#452, DoubleType, false), gpubasicsum(log_play_sec#453, DoubleType, false), gpubasicsum(cast(duration_bucket#454 as double), DoubleType, false), gpubasicsum(cast(complete_play#455 as double), DoubleType, false), gpubasicsum(cast(interact_cnt#456 as double), DoubleType, false), gpubasicsum(cast(is_click#5 as double), DoubleType, false), gpubasicsum(cast(long_view#11 ...
   +- GpuShuffleCoalesce 1073741824
      +- GpuColumnarExchange gpusinglepartitioning$(), ENSURE_REQUIREMENTS, [plan_id=244]
         +- GpuHashAggregate (keys=[], functions=[partial_gpucount(1, false), partial_gpubasicsum(cast(user_id#0 as double), DoubleType, false), partial_gpubasicsum(cast(video_id#1 as double), DoubleType, false), partial_gpubasicsum(cast(hour_of_day#450 as double), DoubleType, false), partial_gpubasicsum(cast(day_of_week#451 as double), DoubleType, false), partial_gpubasicsum(watch_ratio#452, DoubleType, false), partial_gpubasicsum(log_play_sec#453, DoubleType, false), partial_gpubasicsum(cast(duration_bucket#454 as double), DoubleType, false), partial_gpubasicsum(cast(complete_play#455 as double), DoubleType, false), partial_gpubasicsum(cast(interact_cnt#456 as double), DoubleType, false),  ...
            +- FusedGpuProjectExec [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#450, gpudayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#451, gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#452, LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#453, CASE WHEN (duration_ms#13 < 7000) THEN 0 WHEN (duration_ms#13 < 15000) THEN 1 WHEN (duration_ms#13 < 30000) THEN 2 WHEN (duration_ms#13 < 60000) THEN 3 WHEN (duration_ms#13 < 120000) THEN 4 ELSE 5 END AS duration_bucket#454, CASE WHEN (play_time_ms#12 >= duration_ms#13) T ...
               +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_like:int,is_follow:int...
```

`GpuProject` became `FusedGpuProjectExec`; the aggregation operator above it (`GpuHashAggregate`) and the `GpuFileGpuScan` below it are untouched.

The generated kernel is **the same one** as in Case A (`kernels/fused_88a94965d48afb79.cu`, 381 lines, same source); for the full source with per-section comments see section 4 of Case A. The kernel's inputs (10 columns), outputs (8 columns) and operations (27) are identical to Case A, so it is not pasted again.

### 5. Performance and consistency

**Consistency**: compared row by row over 92,919,789 rows, spark-rapids vs plugin shows **0 differing rows** (the CPU vs GPU difference is the same as Case A: the last digit of `log_play_sec`).

**Performance** (read Parquet + feature computation + GPU aggregation, fetching back a single row):

| Configuration | Rounds | Min | Median | Max | Relative to the spark-rapids baseline |
| --- | --- | --- | --- | --- | --- |
| CPU Spark (reference) | 5 | 3.650 | 3.773 | 3.950 | spark-rapids is 10.5x faster |
| spark-rapids (baseline) | 20 | 0.336 | 0.361 | 0.408 | - |
| spark-rapids + plugin | 20 | 0.267 | 0.299 | 0.354 | +20.7% (positive = plugin faster); the plugin was faster in 20 of 20 rounds |

Times are in seconds, wall time of the whole query. spark-rapids and the plugin each ran 20 times, with **10 warm-up rounds not counted** (with only 2 warm-up rounds the first few timed rounds came out slow, hence 10), and **the order of the two alternates every round** (baseline first in even rounds, plugin first in odd rounds). Same Spark session, same data, same spark-rapids settings; the only difference is `spark.rapids.sql.fusion.mode` (OFF / SAFE). CPU is a reference only, run 5 times up front on its own (plus 1 warm-up round).

### 6. Time breakdown (prepare / read / compute / write)

| Phase (wall seconds, median) | spark-rapids | + plugin | Diff (plugin - baseline) | Measured or estimated |
| --- | --- | --- | --- | --- |
| Prepare (planning, listing files, job scheduling, submission) | 0.073 | 0.080 | +0.007 | **Measured**: total wall time - time of all Spark jobs |
| Read | 0.187 | 0.180 | -0.007 | Estimated (apportioned): job time x scan operator time / total task run time |
| Compute | 0.118 | 0.061 | -0.058 | Estimated (apportioned): same, using the time of the operators other than scan/write/exchange |
| Exchange (shuffle) | 0.001 | 0.001 | -0.000 | Estimated (apportioned): same, using shuffle read + partitioning + serialization |
| Other (task time not explained by the metrics above) | 0.002 | 0.002 | +0.000 | Estimated (apportioned): the remainder |
| **Total wall time** | **0.388** | **0.321** | **-0.067** | **Measured** |

| Task-seconds (summed over the 8 task threads, median) | spark-rapids | + plugin | Diff |
| --- | --- | --- | --- |
| Read | 1.228 | 1.100 | -0.128 |
| Compute | 0.792 | 0.394 | -0.398 |
| Exchange (shuffle) | 0.006 | 0.006 | -0.000 |

Eight task threads run in parallel within a job, and "task-seconds" is the sum of the time over those threads. The wall times for read / compute / write are **estimated (apportioned), not measured**: the job time is split in proportion to each operator class's share of total task time. "Prepare" and total wall time are measured.

- **The gain is in the "compute" phase, and it matches.** Same 27 operations in 1 kernel as A; `GpuProject` op time 0.547 -> `FusedGpuProjectExec` 0.143 (dropped in 20 of 20 rounds, median difference -0.404). The aggregation (`GpuHashAggregate`) is 0.250 vs 0.247, unchanged.
- In the read phase, scan time goes 1.228 -> 1.100 (-0.100), of which GPU semaphore wait is 0.322 -> 0.207 (-0.115); with the semaphore wait removed the difference is only -0.013; GPU decode -0.018. **As in A: reading itself did not change, the time spent waiting for the GPU went down, which is an indirect reflection of the gain in the compute phase.**
- Estimated wall time: compute 0.118 -> 0.061 (-0.057), read 0.187 -> 0.180 (-0.007), which roughly matches the total wall time difference 0.388 -> 0.321 (-0.067).
- B has no write phase; the fixed driver-side overhead is 0.073 / 0.080 (smaller than A's 0.2 because no file is written).

### 7. Conclusion

- Results are bit-identical to spark-rapids.
- With the Parquet write removed and the result left on the GPU, the plugin is **about 21% faster** than spark-rapids (median 0.299 s vs 0.361 s, plugin faster in all 20 rounds). This is the largest gain among the cases in this group.
- Even so the gain is only about 20%: all the plugin can save is the `Project` phase (0.55 down to 0.14 after fusion, task-seconds summed over the task threads); reading the file and everything else is unchanged.
- This is the setting most favorable to the plugin in this group (heavy computation, no file write), and can be seen as a rough upper bound for this dataset on this GPU.
