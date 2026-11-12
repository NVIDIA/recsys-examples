<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing the KuaiRand results

This page lets you rebuild everything from public sources and re-run the five KuaiRand cases
([group 1](kuairand/group1-fusion-pays-off.md), [group 2](kuairand/group2-fusable-but-small.md),
[group 3](kuairand/group3-nothing-to-fuse.md)): spark-rapids only (the baseline) against spark-rapids plus the
expression-fusion plugin, on one GPU. Plan on about 1.5 hours the first time, most of it downloading (about 11 GB of data and 2 GB of dependencies);
building the plugin takes about 2 minutes, converting the CSV files about 1 minute and the five cases about 25 minutes.

All paths below are relative to `examples/Feature-Extractor-Plugin/batch` of this repository (called `$BATCH`) or to
directories you choose (`$WORK`, `$DATA_ROOT`).

> **Status of the steps.** Steps 1-3 and 5-7 were executed from a fresh clone and an empty toolchain directory when this page was written
> (results in step 7).
> **Step 4 (downloading the KuaiRand archive) was not run for this verification**: the CSV files had been extracted from the
> official archive earlier, and the verification started from those files. The download address and checksum in step 4
> come from the dataset page and were not exercised.

## 0. What was used

| | tested with |
| --- | --- |
| GPU | 1 x NVIDIA L20 (Ada, sm_89, 46 GB), driver 550.127.08 (CUDA 12.4 capable) |
| CPU / RAM | Intel Xeon Platinum 8458P, 176 hardware threads, about 1 TB RAM (the runs use `local[8]` and a 96 GB driver heap) |
| OS | Ubuntu 22.04, Linux 5.15 |
| CUDA toolkit | 12.4 installed on the host (only used as the default NVRTC search path, see below) |
| NVRTC | `libnvrtc.so.12` 12.9.86 from the pip wheel `nvidia-cuda-nvrtc-cu12` (any NVRTC 12.x with `libnvrtc.so.12` is expected to work) |
| JDK | OpenJDK 11 (11.0.25) |
| Python | 3.10 (for PySpark and the helper scripts) |
| Spark | 3.5.9, prebuilt for Hadoop 3 (Scala 2.12) |
| spark-rapids | `rapids-4-spark_2.12-26.08.1-cuda12.jar` |
| Maven | 3.9.9 (build only) |
| Plugin | `gpu-fusion-plugin-0.1.0` built from `plugin` (Scala 2.12.18, JCuda 12.0.0 shaded in) |

Disk: about 10 GB for the archive, 7 GB for the three CSV files used, 2 GB for the Parquet tables, plus scratch space
for Spark and the per-case outputs (the cases write their results, about 1-2 GB per case). Use a fast local disk for the data and for `FUSION_WORK`.

## 1. Get the code

```bash
export WORK=<an empty directory>
cd $WORK
git clone -b feature/feature-extractor-plugin https://github.com/cb521/recsys-examples.git
export BATCH=$WORK/recsys-examples/examples/Feature-Extractor-Plugin/batch
```

## 2. Download the dependencies

Everything goes into one toolchain directory; the scripts find it through `TOOLCHAIN_DIR`.

```bash
export TOOLCHAIN_DIR=$WORK/toolchain; mkdir -p $TOOLCHAIN_DIR; cd $TOOLCHAIN_DIR
# Spark 3.5.9 (prebuilt for Hadoop 3)
curl -fLO https://archive.apache.org/dist/spark/spark-3.5.9/spark-3.5.9-bin-hadoop3.tgz
tar xzf spark-3.5.9-bin-hadoop3.tgz
# spark-rapids 26.08.1 (CUDA 12 build)
curl -fLO https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/26.08.1/rapids-4-spark_2.12-26.08.1-cuda12.jar
# Maven 3.9.9 (only needed to build the plugin)
curl -fLO https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz
tar xzf apache-maven-3.9.9-bin.tar.gz
# NVRTC 12.9 from PyPI (the plugin compiles its kernels at run time with NVRTC)
pip download --no-deps -d nvrtc-wheel nvidia-cuda-nvrtc-cu12==12.9.86
mkdir -p nvrtc && cd nvrtc-wheel && unzip -q -o nvidia_cuda_nvrtc_cu12-*.whl && cp nvidia/cuda_nvrtc/lib/* ../nvrtc/ && cd ..
```

You also need JDK 11 (for example `apt install openjdk-11-jdk-headless`), Python 3.10, `git`, `curl`, `tar` and `unzip`.
Export `JAVA_HOME` if the JDK is not at `/usr/lib/jvm/java-11-openjdk-amd64`:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$TOOLCHAIN_DIR/apache-maven-3.9.9/bin:$PATH
```

## 3. Build the plugin

spark-rapids keeps its Spark-version-specific classes inside the distribution jar under `spark-shared/` and `spark359/`.
The Scala compiler needs them at their normal package path, so first repack them into a small "shim jar" (compile time only; the
plugin itself runs against the normal spark-rapids jar), then build with Maven (downloads Scala, JCuda and the Maven plugins from Maven Central):

```bash
cd $BATCH
plugin/scripts/make-shim-jar.sh       # writes $TOOLCHAIN_DIR/rapids-shim359.jar
plugin/scripts/build.sh               # writes plugin/target/gpu-fusion-plugin-0.1.0.jar
plugin/scripts/selftest.sh            # optional, no GPU: planner checks + every kernel compiles with NVRTC for sm_75..sm_120
```

Behind a proxy, give Maven the proxy through `MAVEN_OPTS` (see the header of `build.sh`).

## 4. Download the data (not re-run for this page)

[KuaiRand](https://kuairand.com) (Gao et al., CIKM 2022, CC BY-SA 4.0) is published on Zenodo. The checksum below is from the dataset page.

```bash
export DATA_ROOT=<a fast local disk with about 25 GB free>; mkdir -p $DATA_ROOT; cd $DATA_ROOT
wget https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz       # about 9.9 GB, md5 3e3c799a24e2d23a4d2c757fbf9adf59
md5sum KuaiRand-27K.tar.gz
tar -xzf KuaiRand-27K.tar.gz KuaiRand-27K/data/log_standard_4_22_to_5_08_27k_part2.csv \
    KuaiRand-27K/data/user_features_27k.csv KuaiRand-27K/data/video_features_basic_27k.csv
```

Only these three files are used: one exposure log (`part2`, about 7 GB, 92,919,789 rows), the user features and the basic video features.

## 5. Convert the CSV files to Parquet (CPU only, once)

```bash
cd $BATCH
export FUSION_WORK=$DATA_ROOT/work          # scratch directory for Spark (spark.local.dir); fast disk
export SPARK_DRIVER_MEM=200g                # we used 200g and 32 local threads for this one-off conversion
plugin/scripts/run-spark.sh -n 32 spark-submit benchmarks/kuairand/prepare_data.py \
    --src $DATA_ROOT/KuaiRand-27K/data --log-file log_standard_4_22_to_5_08_27k_part2.csv --dst $DATA_ROOT/data
```

Expected output (`== <table>: <rows>` lines; the three Parquet directories are about 1.1 GB, 592 KB and 659 MB):

```
== log: 92919789 rows
== user: 27285 rows
== video_basic: 32038725 rows
```

(Spark may print a `java.lang.NullPointerException` from its RPC layer during this step; it did not affect the tables, whose row counts match the above.)

## 6. Run the five cases

One command runs all of them in order (about 25 minutes on one L20 with the settings below):

```bash
cd $BATCH
export FUSION_WORK=$DATA_ROOT/work
export CUDA_VISIBLE_DEVICES=0               # use exactly one GPU
export DATA=$DATA_ROOT/data OUT=$DATA_ROOT/out
benchmarks/kuairand/run_all.sh
```

`run_all.sh` is a thin loop around `plugin/scripts/run-spark.sh` + `benchmarks/kuairand/run_case.py`. For one case
(here case A) the full command is:

```bash
FUSION_DUMP_DIR=$OUT/a_sample_features/kernels plugin/scripts/run-spark.sh -n 8 spark-submit \
    --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
    --conf spark.rapids.memory.pinnedPool.size=8g \
    benchmarks/kuairand/run_case.py --case benchmarks/kuairand/cases/a_sample_features.sql --data $DATA --out $OUT
# case B: same SQL file, add  --sink gpu_agg        (output directory a_sample_features_gpuagg)
# case C: cases/c_filter_features.sql      case D: cases/d_join_string_agg.sql      case E: cases/e_light_scan.sql
# FUSION_DUMP_DIR must be <OUT>/<output directory name>/kernels
```

Settings that are in effect (all set by `run-spark.sh` unless noted; `SPARK_DRIVER_MEM` defaults to 24g, we used 96g):

| setting | value |
| --- | --- |
| master | `local[8]` (8 task threads in one JVM) |
| driver memory | 96 GB (`SPARK_DRIVER_MEM=96g`, set by `run_all.sh`) |
| plugins / extensions | `com.nvidia.spark.SQLPlugin`, `com.nvidia.spark.rapids.fusion.FusionExtension` (fusion is switched on and off per run by `spark.rapids.sql.fusion.mode`) |
| GPU memory pool | `spark.rapids.memory.gpu.allocFraction=0.5`, `maxAllocFraction=0.5` |
| pinned memory pool | `spark.rapids.memory.pinnedPool.size=8g` (**not** the 1g default of `run-spark.sh`: with 1g the pinned pool overflows ("maximum pool size exceeded" in the log) and case A's baseline measured 1.13 s instead of 0.83 s) |
| shuffle partitions | `spark.sql.shuffle.partitions=8` |
| time zone | `spark.session.timeZone=UTC` (set in `run_case.py`) |
| UI | `spark.ui.enabled=false` |
| spark-rapids metrics | default level |

Timing protocol (in `run_case.py`): CPU Spark is run first on its own (1 untimed + 5 timed runs, reference only). Then
spark-rapids and spark-rapids + plugin run in the same Spark session, 10 untimed warm-up rounds and 20 timed rounds, and **inside every
round the order of the two alternates** (the first one of an even round is the baseline, of an odd round the plugin). Only
`spark.rapids.sql.fusion.mode` differs (`OFF` / `SAFE`). The timed part is the whole query including writing the output
(Parquet for A, C, D, E; for B, a GPU-side sum of every column and one collected row).

## 7. What you get

Per case, under `$OUT/<case>/`: `plan_cpu.txt`, `plan_gpu.txt`, `plan_fused.txt` (physical plans), `fusion_decisions.txt` (what the
plugin fused and why not), `kernels/*.cu` (generated CUDA source), `timings.json` (all timings, medians, number of rounds where the plugin
was faster), `check.txt` (consistency).

**Consistency (`check.txt`).** Each of the three configurations writes its result (dictionary encoding off) and the results are compared
row by row on the CPU. Expected: `rows only in gpu: 0   rows only in fused: 0` for every case (the plugin matches spark-rapids bit for bit).
The CPU rows differ from the GPU rows only in the last digit of `log_play_sec` (Java `StrictMath` vs the CUDA math library, a known spark-rapids
difference), so for cases A, B and C the two `cpu` lines show a non-zero count (A, B: 5,067,411 rows of 92,919,789; C: 2,093,458 rows of 45,611,516) and for D, E they show 0.

**Results we got** (median seconds, 1 x L20; ratio = spark-rapids / plugin; "wins" = rounds out of 20 where the plugin was faster):

| case | published run: spark-rapids | + plugin | ratio | wins | re-run on a fresh clone: spark-rapids | + plugin | ratio | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (write Parquet) | 0.809 | 0.759 | 1.066 | 17/20 | 0.799 | 0.729 | 1.096 | 18/20 |
| B (sum on GPU, nothing written) | 0.361 | 0.299 | 1.207 | 20/20 | 0.350 | 0.292 | 1.201 | 18/20 |
| C (filter + features) | 0.711 | 0.722 | 0.986 | 10/20 | 0.722 | 0.730 | 0.989 | 8/20 |
| D (joins, string, aggregation) | 2.349 | 2.346 | 1.001 | 10/20 | 2.366 | 2.376 | 0.996 | 10/20 |
| E (almost no computation) | 0.713 | 0.682 | 1.046 | 13/20 | 0.715 | 0.702 | 1.017 | 10/20 |

CPU Spark (5 timed runs, reference only): A 15.8 s, B 3.8 s, C 5.5 s, D 12.8 s, E 5.1 s in the published run; 18.0 / 3.9 / 6.5 / 12.5 / 5.2 s in the re-run.
Per-case min-max ranges are in each `timings.json`; the group pages list those of the published run.

**How much difference is normal.** Absolute times move with CPU, disk and GPU: expect them to scale by roughly 10-20% on other
machines, and the CPU line even more. What should reproduce is the ratio. Run to run on the same machine the
ratio varied by a few percent: A was 1.060, 1.066 and 1.096 in three runs, B 1.211, 1.207 and 1.201, C 1.023, 0.986 and 0.989, D 1.009, 1.001 and 0.996, E 0.994, 1.004, 1.012, 1.046 and 1.017.
So: A should come out about 1.04-1.10 and B about 1.18-1.24 (plugin faster in at least 17 of 20 rounds for A and 18 of 20 for B), and C, D, E
between 0.95 and 1.05 with the plugin winning roughly half of the rounds. An A or B ratio near 1.0, or a C/D/E ratio far from 1.0, means something differs from our setup
(first thing to check: the pinned pool size and that `FUSION_DUMP_DIR` kernels were produced).
