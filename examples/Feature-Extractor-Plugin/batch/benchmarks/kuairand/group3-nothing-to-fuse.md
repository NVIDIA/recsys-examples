<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Group 3: almost nothing to fuse (Case E)

> All execution plans, fusion decisions and kernel sources in this document are originals exported automatically during the actual runs; only line wrapping, elisions (marked `...`) and path substitution (the data directory is written as `<data>`) were applied.
> Data: KuaiRand-27K, converted to Parquet: **92,919,789 rows**.
> Environment: 1 NVIDIA L20, Spark 3.5.9 local[8], spark-rapids 26.08.1, plugin `gpu-fusion-plugin-0.1.0`.
> Performance numbers: 10 warm-up rounds and 20 timed rounds per configuration, with the order of the two configurations alternating each round. The time breakdown comes from a separate 20-round run with DEBUG metrics (2 warm-up rounds), so its total wall time differs slightly from the performance table.

**What this group has in common**: the expression has only 2 operations (one type cast, one division), and the whole query just reads a file, computes almost nothing, and writes it back out. The plugin happens to reach the fusion threshold (`minOps=2`) and so it fuses, but the GPU memory reads and writes that fusion can save are far smaller than the time spent reading and writing Parquet.

**One-sentence conclusion**: this kind of query is **not worth using the plugin for, and it neither slows down nor speeds up**: median 0.682 s vs 0.713 s (plugin 4.6% faster, faster in 13 of 20 rounds, but other runs measured the same way gave -0.6% to +1.2%, and ±5% is treated as noise); results are bit-identical to spark-rapids.

## Case E: Only read the file, do a single division

### 0. What this case computes

Read the log, pass through only 3 columns, and compute 1 new column `play_time_ms / 1000`. The amount of computation is tiny, which illustrates that **fusion is pointless when there are too few operations**.

```sql
-- Expected: the plugin may fuse it, but the Parquet scan dominates, so no gain.
SELECT
  user_id,
  video_id,
  is_click,
  play_time_ms / 1000D                                        AS play_sec
FROM log
```

### 1. Original Spark (CPU) physical plan

```
*(1) Project [user_id#0, video_id#1, is_click#5, (cast(play_time_ms#12 as double) / 1000.0) AS play_sec#124]
+- *(1) ColumnarToRow
   +- FileScan parquet [user_id#0,video_id#1,is_click#5,play_time_ms#12] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,is_click:int,play_time_ms:int>

```

### 2. GPU physical plan after spark-rapids (baseline)

`GpuColumnarToRow -> GpuProject [user_id, video_id, is_click, (cast(play_time_ms as double) / 1000.0) AS play_sec] -> GpuFileGpuScan`.
`GpuProject` has only 2 operations (one type cast, one division); spark-rapids makes one cuDF call for each anyway, 2 calls in total.

### 3. What the plugin saw and what it decided to fuse

```
== Fusion decisions (SAFE) ==
FUSED project: fused 2 ops -> 1 kernel fused_4e611515e875cbba; 1 kernel outputs, 1 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s) 
```

2 operations just reach the `minOps=2` threshold, so it **fused**: cast + division -> 1 kernel, reading 1 column and writing 1 column.

### 4. Fused plan and kernel

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, is_click#5, (cast(play_time_ms#12 as double) / 1000.0) AS play_sec#134], true, fused 2 ops -> 1 kernel fused_4e611515e875cbba; 1 kernel outputs, 1 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,play_time_ms#12] Batched: true, DataFilters: [], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,is_click:int,play_time_ms:int>
```

The kernel body (`kernels/fused_4e611515e875cbba.cu`, 240 lines in total, preceded by the device function definitions):

```cuda
extern "C" __global__ void fused_4e611515e875cbba(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    f64* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  f64 v2 = (n1 ? (f64)v1 : (f64)0);
  bool n2 = n1;
  f64 v3 = __longlong_as_double(4652007308841189376LL);
  bool n3 = true;
  f64 v4 = ((n2 && n3 && !is_zero(v3)) ? d_div(v2, v3) : (f64)0);
  bool n4 = n2 && n3 && !is_zero(v3);
  if (inb) o0[i] = st_f64(v4);
  { u32 b = __ballot_sync(0xffffffffu, inb && n4); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  }
}
```

Fusion only merges the "cast" and "division" cuDF calls into one, saving one GPU memory read/write of the intermediate double column. But that column is only 92,919,789 x 8 bytes ~= 0.7 GB,
which takes milliseconds at GPU memory bandwidth, while the whole query takes 0.7 s.

### 5. Performance and consistency

**Consistency**: 92,919,789 rows, spark-rapids vs plugin shows **0 differing rows**; CPU vs GPU and CPU vs plugin are also 0 differing rows.

**Performance** (read Parquet + 1 division + write Parquet):

| Configuration | Rounds | Min | Median | Max | Relative to the spark-rapids baseline |
| --- | --- | --- | --- | --- | --- |
| CPU Spark (reference) | 5 | 4.793 | 5.139 | 6.511 | spark-rapids is 7.2x faster |
| spark-rapids (baseline) | 20 | 0.667 | 0.713 | 0.815 | - |
| spark-rapids + plugin | 20 | 0.645 | 0.682 | 0.905 | +4.6% (positive = plugin faster); the plugin was faster in 13 of 20 rounds |

Times are in seconds, wall time of the whole query. spark-rapids and the plugin each ran 20 times, with **10 warm-up rounds not counted** (with only 2 warm-up rounds the first few timed rounds came out slow, hence 10), and **the order of the two alternates every round** (baseline first in even rounds, plugin first in odd rounds). Same Spark session, same data, same spark-rapids settings; the only difference is `spark.rapids.sql.fusion.mode` (OFF / SAFE). CPU is a reference only, run 5 times up front on its own (plus 1 warm-up round).

### 6. Time breakdown (prepare / read / compute / write)

| Phase (wall seconds, median) | spark-rapids | + plugin | Diff (plugin - baseline) | Measured or estimated |
| --- | --- | --- | --- | --- |
| Prepare (planning, listing files, job scheduling, submission) | 0.129 | 0.129 | +0.001 | **Measured**: total wall time - time of all Spark jobs |
| Read | 0.213 | 0.201 | -0.012 | Estimated (apportioned): job time x scan operator time / total task run time |
| Compute | 0.016 | 0.002 | -0.014 | Estimated (apportioned): same, using the time of the operators other than scan/write/exchange |
| Write / output | 0.359 | 0.378 | +0.019 | Estimated (apportioned): same, using the time of the file-write operator |
| Other (task time not explained by the metrics above) | 0.002 | 0.002 | +0.000 | Estimated (apportioned): the remainder |
| **Total wall time** | **0.716** | **0.713** | **-0.003** | **Measured** |

| Task-seconds (summed over the 8 task threads, median) | spark-rapids | + plugin | Diff |
| --- | --- | --- | --- |
| Read | 1.231 | 1.226 | -0.005 |
| Compute | 0.096 | 0.014 | -0.082 |
| Write / output | 2.114 | 2.248 | +0.133 |

Eight task threads run in parallel within a job, and "task-seconds" is the sum of the time over those threads. The wall times for read / compute / write are **estimated (apportioned), not measured**: the job time is split in proportion to each operator class's share of total task time. "Prepare" and total wall time are measured.

- **What was fused:** 2 operations (cast + division) -> 1 kernel. Measured for the compute phase: `GpuProject` op time 0.096 -> `FusedGpuProjectExec` 0.014 (task-seconds, median per-round difference -0.083, dropped in 19 of 20 rounds); estimated wall time 0.016 -> 0.002. nsys: compute-phase kernels 33 -> 22 per round, GPU time 0.005 -> 0.004 s.
- **But that 0.08 is cancelled out by the write phase:** the file-write node's op time went 2.114 -> 2.248 (+0.13; median per-round difference +0.067, plugin lower in 8 of 20 rounds), mainly GPU encode and buffer 1.430 -> 1.560. Net result: the median per-round difference in total task run time is +0.008, ~= 0. Read-phase scan time is 1.231 vs 1.226, unchanged.
- Why the write phase got longer: **not found** (in nsys the write-phase kernel count and duration are the same: 462 kernels, 1.005 vs 1.002 s; write-phase CPU wall time is 1.319 vs 1.389 s and write-phase synchronization wait is 1.281 vs 1.225 s, so the directions disagree). A also shows the same "write phase a bit longer" effect (+0.08), and C only +0.03.
- In short, what the plugin can save in E (0.08 task-seconds ~= 0.016 s of wall time) is under 3% of the 0.7 s total wall time, and is lost in the noise.

### 7. Conclusion

- The plugin's median is 0.682 s against the baseline's 0.713 s, about 4.6% faster, faster in 13 of 20 rounds; but in other runs measured the same way it was -0.6% to +1.2%, and the range is wide (both configurations span 0.65 to 0.9 s from min to max), so **under the rule "within ±5% counts as noise" it is not treated as a gain**.
- This case shows that with only two or three operations, the GPU memory reads and writes saved by fusion are far smaller than the time spent reading and writing Parquet, so it is **not worth using** and brings no stable, visible change.
- A rough rule for whether fusion is worthwhile: look at the number of operations in the expression and at the share of the whole query's time that is not I/O. The more operations and the less I/O, the more likely there is a gain.
