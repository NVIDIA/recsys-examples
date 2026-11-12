<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KuaiRand feature cases: spark-rapids vs spark-rapids + fusion

Baseline: **spark-rapids only**. Compared: **the same spark-rapids + the expression-fusion plugin**. Every other setting
is identical. Plain CPU Spark is shown for reference.

## When it fits, when it does not

**Fits**: many row-wise numeric expressions in one projection or filter (arithmetic, casts, comparisons, CASE, least/greatest,
log, pmod, ...) whose results are consumed on the GPU without a heavy file write in the timed path.
The plugin turns the chain of per-operator cuDF calls into one generated kernel, so intermediates stay in registers.

**Does not fit**: queries dominated by Parquet read/write, joins, shuffles, aggregation or string functions; expressions with
only one or two operators. The plugin leaves those parts to spark-rapids (and says why in `fusion_decisions.txt`).

**What we measured** (1 L20, 92.9M rows, 20 timed runs each after 10 warm-up runs, order of the two GPU configurations alternated every round; results are bit-identical to spark-rapids in every case; details in the three group documents):
case B (features computed on the GPU, nothing written) is **about 21% faster** (0.299 s vs 0.361 s); case A (same SQL, writes Parquet) is **about 7% faster** (0.759 s vs 0.809 s);
cases C, D, E (filter + features dominated by Parquet read/write; joins/shuffle; almost no computation) show **no stable difference** (within about +/-5%, which is the noise level of this setup). In this setup the plugin never hurt,
and helped only as much as the I/O left room for.

## Data

[KuaiRand-27K](https://kuairand.com) (Gao et al., CIKM 2022, CC BY-SA 4.0): exposure logs of the Kuaishou short-video
app with 12 feedback signals, plus user and video feature tables. We use one log file and the user / video-basic
tables; see `prepare_data.py`. The dataset does not ship feature-processing code, so the features below are written
by us following common practice; the source of each is given.

## Cases (grouped by what the plugin can do)

| group | cases | what the plugin does | result (median, 20 timed runs, 10 warm-up) | doc |
| --- | --- | --- | --- | --- |
| 1. fusion pays off | A: per-exposure sample features, read and write Parquet. B: same SQL, features summed on the GPU, nothing written | 27 operators in one projection become 1 kernel | A **~6.6% faster** (0.759 s vs 0.809 s), B **~21% faster** (0.299 s vs 0.361 s, upper bound) | [group1-fusion-pays-off.md](group1-fusion-pays-off.md) |
| 2. fusable, but too small a share | C: sample filtering + features, write Parquet. D: joins, string feature, group-by aggregation | C: filter and projection each fused into 1 kernel; D: one 3-operator filter fused | no measurable difference, not slower (C 0.722 s vs 0.711 s, D 2.346 s vs 2.349 s) | [group2-fusable-but-small.md](group2-fusable-but-small.md) |
| 3. almost nothing to fuse | E: read the log, one conversion + one division | 2 operators (just reaches the threshold) | no stable difference (0.682 s vs 0.713 s, within noise), not worth using | [group3-nothing-to-fuse.md](group3-nothing-to-fuse.md) |

Each case section walks through the real flow: original CPU physical plan, the plan after spark-rapids translation, what the plugin
decided to fuse (verbatim), the fused plan with the generated kernel source, performance and consistency, and a time breakdown
(prepare / read / compute / write; the split inside the job is an estimate, the prepare time and total are measured).

## Where the features come from

| feature | expression | source |
| --- | --- | --- |
| `watch_ratio` | `play_time / duration`, clipped at 5 | defined as `play_duration / video_duration` by the KuaiRec dataset (same authors, [github.com/chongminggao/KuaiRec](https://github.com/chongminggao/KuaiRec)); widely used as an engagement signal in short-video recommendation. Values above 1 are replays; the clip bound 5 is our choice |
| `log_play_sec` | `log(1 + play_time_s)` | log transform of heavy-tailed counts/durations, as in NVIDIA DLRM's Criteo preprocessing (`log(x + 3)` / `log(x + 2)`, [DLRM docs](https://catalog.ngc.nvidia.com/orgs/nvidia/dle/resources/dlrm_pyt/-/advanced)) |
| `duration_bucket` | CASE on video length | bucketizing a continuous feature into an embedding id; bucket bounds are our choice |
| `hour_of_day`, `day_of_week` | from `hourmin`, `time_ms` | standard time-context features |
| `complete_play`, `interact_cnt`, `is_click`, `long_view` | from the feedback columns | multi-task labels (KuaiRand's 12 feedback signals) |
| `user_tab_cross` | `pmod(user_id * p + tab, 2^20)` | hashed cross feature into a fixed-size embedding table |

## How to run

Step-by-step instructions (hardware and software versions, where to download every dependency, how to build the plugin,
how to get and convert the data, the exact command and settings of every case, the numbers we got and how much deviation is
normal) are in [../REPRODUCE.md](../REPRODUCE.md). In short, after the build and the one-off data conversion:

```bash
# once, CPU only: CSV -> Parquet
plugin/scripts/run-spark.sh -n 32 spark-submit benchmarks/kuairand/prepare_data.py \
    --src <KuaiRand-27K/data> --log-file log_standard_4_22_to_5_08_27k_part2.csv --dst <data dir>
# all five cases, one GPU (about 25 minutes)
DATA=<data dir> OUT=<output dir> benchmarks/kuairand/run_all.sh
```
