<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing the streaming benchmarks

This page tells you how to re-run the six streaming cases (c1-c6, see [README.md](README.md)) with the driver and scripts in this directory: build the driver, prepare the KuaiRand data, run the cases, check that every mode returns the same result, and read the numbers.

- **The CPU baseline needs nothing but open-source Flink 2.2.1.** No plugin is involved.
- **The GPU side is not self-contained.** The c2 (window aggregation) and c6 (dimension-table lookup) GPU side can be run with the Feature-Extractor library through the example jobs in `../examples` (`WindowFeaturesJob`, `LookupAuthorJob`; see `../examples/README.md`); they compare the final table with the CPU run in the same way. The benchmark scripts in this directory, however, drive the GPU modes only through a Mega-Flink build, and the published c2 and c6 GPU numbers come from it. The c1, c3, c4 and c5 GPU numbers come from a Mega-Flink build as well. **The GPU side of these four cases needs a Mega-Flink build ([https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink)); this directory provides the driver and scripts, and the library has no operators for them.**

All paths below are written as environment variables: `$BENCH` is this directory (`examples/Feature-Extractor-Plugin/streaming/benchmarks` of the repository), `$FLINK_HOME` an unpacked Flink distribution, `$FEP_DATA_DIR` the prepared data, `$OUT` the results.

> **Status of the steps.** Verified: step 3 (the driver builds against the open-source Flink 2.2.1 binary distribution with JDK 11); step 4, conversion half (the converter produced files byte-identical to the independent Java reference converter on a small synthetic CSV; the 17 reference checksums in `data-sha256.txt` come from the full data made earlier by that Java converter); step 5 (smoke test on one L20, see below); the Mega-Flink mode of the scripts for c1, c2-hop, c3, c4, c5-join and c6 with one local Mega-Flink build (final-table SHA-256 equal to the CPU run). **Not run: the download in step 4** (the archive address and checksum come from the dataset page and a `HEAD` request confirmed the file exists, 9,892,191,178 bytes), **the Python converter on the full 92.9 million row file** (so `sha256sum -c` against `data-sha256.txt` is the check for your copy), **the full matrix of step 6 on open-source Flink** and the breakdown / nsys scripts of step 9. The published numbers come from a Mega-Flink build in its plain CPU mode, not from open-source Flink; both gave the same final tables on the smoke slice, but the full matrix was not compared.

## 0. What is needed

| | tested with |
| --- | --- |
| GPU (Mega-Flink runs only) | 1 x NVIDIA L20 (Ada, sm_89), driver for CUDA 12.4. The CPU baseline needs no GPU. |
| CPU / RAM | the published runs used an exclusive node with 16 GB JVM heap; host CPU model and core count: not verified. Give the job several free cores: the replay source, the Flink tasks and the sink run in one JVM |
| Disk | about 10 GB for the archive (deletable after step 4), about 7 GB for the CSV files, about 11 GB for the converted data; local disk is expected |
| OS | Linux (Ubuntu 22.04 was used) |
| JDK | OpenJDK 11. JDK 17 or newer was not tried; it needs the `--add-opens` flags that Flink's `bin/config.sh` lists, passed through `FEP_JVM_ARGS` |
| Flink | 2.2.1 binary distribution, Scala 2.12 (`flink-2.2.1-bin-scala_2.12.tgz`) |
| Python | 3.8 or newer with `numpy`; `pandas` optional (several times faster conversion) |
| CUDA | only for Mega-Flink; the toolkit's runtime libraries are put on `LD_LIBRARY_PATH` through `CUDA_LIB_DIR` |
| Mega-Flink | only for the GPU modes, see step 7 |

Memory for step 4: about 25 GB for the full log. The driver keeps `warm-up + records` rows of 14 int64 columns on the heap (a few hundred MB for the published sizes) plus the captured sink rows.

## 1. Get the code

```bash
export WORK=<an empty directory>
cd $WORK
git clone -b feature/feature-extractor-plugin https://github.com/cb521/recsys-examples.git
export BENCH=$WORK/recsys-examples/examples/Feature-Extractor-Plugin/streaming/benchmarks
```

## 2. Get Flink

```bash
export FLINK_HOME=$WORK/flink-2.2.1
cd $WORK
curl -fLO https://archive.apache.org/dist/flink/flink-2.2.1/flink-2.2.1-bin-scala_2.12.tgz
tar xzf flink-2.2.1-bin-scala_2.12.tgz          # creates $WORK/flink-2.2.1
export JAVA_HOME=<JDK 11 directory>             # or leave unset to use `java` from PATH
```

You do not start a Flink cluster. The driver runs Flink's `MiniCluster` inside its own JVM and only needs the jars: `build.sh` and `run-one.sh` take `lib/*.jar` of the distribution, except the planner loader, plus `opt/flink-table-planner_2.12-2.2.1.jar`.

## 3. Build the driver

```bash
export FEP_BUILD_DIR=$WORK/build                # optional, default $BENCH/driver/build
$BENCH/driver/build.sh
```

Expected output: `built $FEP_BUILD_DIR/classes`. It compiles `driver/src/main/java/fep/*.java` with `javac` (or the JDK's compiler module if only a JRE launcher is on the path). Nothing in `src/main` refers to Mega-Flink.

## 4. Prepare the data

The driver reads the KuaiRand-27K exposure log (the `part2` file, 92,919,789 rows) as int64 column files, sorted by `time_ms`, and a video table (video_id, author_id). [KuaiRand](https://kuairand.com) (Gao et al., CIKM 2022, CC BY-SA 4.0) is public; please follow the dataset page for citation and license.

```bash
export FEP_DATA_DIR=<a fast local disk with about 30 GB free>/kuairand-stream
pip install numpy pandas                       # pandas is optional
python3 $BENCH/prepare_stream_data.py --out $FEP_DATA_DIR
```

This downloads `KuaiRand-27K.tar.gz` from Zenodo (about 9.9 GB, md5 `3e3c799a24e2d23a4d2c757fbf9adf59`, resumable), checks the md5, extracts the log and video CSV files and converts them. If you already have the CSV files: `--src <KuaiRand-27K/data directory>`. Expected output ends with:

```
== log: 92919789 rows
== video: 32038725 rows
```

Layout: `$FEP_DATA_DIR/log/<column>.i64` (14 columns), `$FEP_DATA_DIR/video/{video_id,author_id,video_duration}.i64`, a `meta.txt` in each.

**Check the data** (this is the check that the conversion matches the one behind the published numbers):

```bash
cd $FEP_DATA_DIR && sha256sum -c $BENCH/data-sha256.txt     # 17 lines, all OK
```

`driver/src/main/java/fep/KuaiRandConvert.java` is an independent Java converter with the same rules (it made the reference data): `java -cp $FEP_BUILD_DIR/classes fep.KuaiRandConvert log <log csv> <out>/log` and `... video <video csv> <out>/video`.

Small data set for a quick test: `--limit-rows 500000` reads only the first 500,000 CSV lines (a different, smaller data set; its checksums are not in `data-sha256.txt`). With the pure-Python parser (used when pandas is missing) expect minutes per 10 million rows.

## 5. Smoke test (a few minutes)

One saturated run per case on a 400,000-row slice (100,000 warm-up + 300,000 measured), CPU baseline on open-source Flink:

```bash
export HEAP=16g WARMUP=100000
for c in c1-filter c2-hop c2-tumble c3-topn c4-dedup c5-join c5-join-filter c5-join-agg c6-lookup; do
  $BENCH/scripts/bench.sh $c cpu 0 300000 $OUT/smoke --dump-final true
done
```

Each run prints one line (the full driver output is in `<run dir>/stdout.log`, which ends with `rc=0`):

```
RESULT case=c1-filter label=cpu rate=0 thr=<records/s> p50=<ms> p95=<ms> p99=<ms> out=199254 final=199254 sha=2f29e3f499b32bbf
```

`thr` and the latencies depend on your machine; **`out`, `final` and `sha` do not**. On the data of step 4 you should get exactly these (SHA-256 of the final table, first 16 hex digits; the full digest is in `result.json`):

| case | final rows | sha256 (first 16) |
| --- | ---: | --- |
| c1-filter | 199,254 | 2f29e3f499b32bbf |
| c2-hop | 85,727 | 528ec5b56b375126 |
| c2-tumble | 34,625 | ef46e2ce28a2ff15 |
| c3-topn | 58,028 | 21d7d0915cc5edfa |
| c4-dedup | 392,746 | 8593325de68ad677 |
| c5-join | 157,446 | d9600de61e056926 |
| c5-join-filter | 65,810 | 20e9ede74afdef3a |
| c5-join-agg | 6,246 | eff1aaeab5eb2b48 |
| c6-lookup | 400,000 | 07800d9c2c117f9f |

These were obtained with the data behind `data-sha256.txt` and were equal across open-source Flink, a Mega-Flink build in CPU mode, and (c1, c2-hop, c3, c4, c5-join, c6) in GPU mode. If yours differ, check `sha256sum -c` first. The changelog row count `out` of c3, c4 and c5-join-agg can differ between runs and between CPU and GPU (the final table cannot); for example c5-join-agg printed 94,666 and 95,176 in two CPU runs with the same final SHA. The throughput of this smoke test is not a benchmark and is not published (it ran on a node where the job saw 2 CPU cores).

Check that the SQL that ran is the SQL in `cases/`:

```bash
python3 $BENCH/scripts/check_sql.py $OUT/smoke              # expect: checked N runs, 0 differ
```

## 6. Run the CPU baseline matrix

`scripts/matrix.sh` runs tier x case x round, rotating the mode order each round and dumping the final table in round 1:

```bash
export HEAP=16g
# saturated tier, tuned (pipeline.object-reuse=true), 3 rounds - the "tuned CPU" column of the README
TIERS=high $BENCH/scripts/matrix.sh $OUT/run 3 \
    "c1-filter:cpu-tuned" "c2-hop:cpu-tuned,cpu-1p-tuned" "c2-tumble:cpu-tuned" "c3-topn:cpu-tuned" \
    "c4-dedup:cpu-tuned,cpu-mb-tuned" "c5-join:cpu-tuned" "c5-join-filter:cpu-tuned" "c5-join-agg:cpu-tuned" \
    "c6-lookup:cpu-tuned"
# default configuration, all three tiers: use the mode names without -tuned and drop TIERS
```

| tier | input rate | measured records | warm-up |
| --- | --- | --- | --- |
| low | 20,000 rec/s | 600,000 | 300,000 at the same rate |
| mid | 100,000 rec/s | 3,000,000 | 300,000 |
| high (saturated) | unthrottled | 3,000,000 | 300,000 |

The two rate-limited tiers take wall-clock time by construction: replaying (600,000 + 300,000) rows at 20,000 rec/s is 45 s, (3,000,000 + 300,000) rows at 100,000 rec/s is 33 s, per run and per round, plus start-up. The whole matrix was not timed. One run is killed after `RUN_TIMEOUT` seconds (default 900). Every run starts its own JVM and writes `$OUT/<root>/<case>/<mode>/r<rate>-n<records>-t<round>/`.

Modes of the CPU baseline (also see the header of `scripts/bench.sh`):

- `cpu`: Flink's default plans. `cpu-1p`: one-phase aggregation (c2). `cpu-mb`: mini-batch 5 ms / 4,096 rows (c4; `c5-join-agg` always uses mini-batch on both sides, because its two-phase aggregate needs it).
- `-tuned`: `pipeline.object-reuse=true` and `table.exec.rank.topn-cache-size=100000` (matters for c3 only). The README's methodology takes each side's best CPU mode per case; the case pages say which modes were measured. Tune the CPU side before comparing.
- `--parallelism p` (through `XARGS="--parallelism 2"`) replays with p subtasks; `warm-up + records` must be divisible by p. For c6, `-Dfep.dim.target=4000000` (through `JAVA_OPTS`) pads the dimension table with unprobed videos to that many rows.

## 7. The GPU side (Mega-Flink)

You need a Mega-Flink build for Flink 2.2.1 from [https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink) (section 7a below builds it with one command at the pinned commit and gives the env file; the manual route follows). Then:

```bash
export MEGA_FLINK_CP=<class path: Flink 2.2.1 jars + Mega-Flink runtime + its Flink 2.2.1 shim + its benchmark harness classes>
export MEGA_FLINK_NATIVE_DIR=<directory with the Mega-Flink native libraries>
export CUDA_LIB_DIR=<CUDA runtime library directory, e.g. /usr/local/cuda/lib64>
$BENCH/driver/build.sh                        # also builds driver/src/megaflink into $FEP_BUILD_DIR/classes-megaflink
```

With `MEGA_FLINK_CP` set, the scripts use it instead of `$FLINK_HOME`, pass `-Dfep.megaflink=true` to the driver, and the GPU modes become available. The harness class `io.github.cb521.cudfflink.benchmark.BenchmarkGpuResource` must be on the class path; the driver calls it by reflection to apply the benchmark GPU settings. The dimension table of c6 is then served by `FepVideoDimGpuFactory`, which adds the snapshot contract the GPU lookup needs; the CPU lookup function is the same.

```bash
# the GPU modes of the README: gpu-batch (c1, c3, c4, c5), gpu / gpu-1p (c2, c6), -fuse / -nofuse (c5), -min1 (c6)
TIERS=high $BENCH/scripts/matrix.sh $OUT/run 3 "c1-filter:cpu-tuned,gpu-batch-tuned" "c2-hop:cpu-tuned,gpu-tuned" "c6-lookup:cpu-tuned,gpu-tuned"
```

Everything that touches Mega-Flink is isolated in three places: the reflection call `configureMegaFlink` and the `fep.megaflink` switch in `FepKuaiRandBench.java`, `driver/src/megaflink/`, and the `cudf.flink.*` options added by `scripts/bench.sh` only when `MEGA_FLINK_CP` is set. Without it, `bench.sh` refuses `gpu*` modes with a message.

## 7a. One command for the Mega-Flink build, one command for all six cases

This section replaces "follow its README" and "ask the owners" of step 7: the build is scripted and pinned.

**Pinned version.** `build_megaflink.sh` builds Mega-Flink commit `f7f06845cb6b457c83d424d6424632fbf929e230` (the `main` branch of [https://github.com/cb521/Mega-Flink](https://github.com/cb521/Mega-Flink) at "Merge Spark expression-fusion plugin into main"; the commit is in the public history, a fresh `git clone` has it). Basis: the published runs were measured with one build of exactly this commit, made on an L20 with CUDA 12.4 and the native libraries compiled for sm_89 (`CUDF_FLINK_CUDA_ARCH=89`), and run from a class path (Maven `target/classes` of the three modules plus their dependency jars), not from an installed Flink. The only source difference of that working tree against the commit is a change in the native build script that moves its temporary build directory away instead of deleting it; it does not affect the compiled code. Not verified: the compiled classes and libraries of a fresh build were not compared byte by byte with the original ones; what was compared is the final-table SHA-256 (below).

**Patches: none are required.** Upstream admits only a device whose name contains `H20` (sm_90) unless the build defines `CUDF_FLINK_EXPERIMENTAL_DEVICE_MAJOR` (`native/gpu_arch_compat.h`). Without it, an L20 build would stop at runtime with "does not admit selected device". The upstream native build script sets that macro itself when it is given another architecture, so the script only has to pass `CUDF_FLINK_CUDA_ARCH=89` (its default `ARCHS=89`); the check then admits any device of compute-capability major 8. If you build for an H20 use `ARCHS=90`, which keeps the stock H20-name rule. `benchmarks/patches/` is the place for source patches (applied in order, checked, idempotent); the directory does not exist at this commit.

**Build** (needs: git, network access to GitHub and Maven Central, a JDK 11 with `javac`, CUDA toolkit 12.x with `nvcc`; the Maven wrapper of the repository downloads Maven 3.9.11 and the Flink 2.2.1 jars into the clone, and the JDK's `jni.h` is used, or fetched with a checksum when only a JRE is present; CUB comes with the CUDA toolkit):

```bash
WORK_DIR=$WORK/megaflink CUDA_TOOLKIT_ROOT=/usr/local/cuda JAVA_HOME=<JDK 11 directory> \
    $BENCH/scripts/build_megaflink.sh
```

Variables: `MEGA_FLINK_REF` (default: the commit above), `MEGA_FLINK_REPO`, `WORK_DIR`, `CUDA_TOOLKIT_ROOT`, `ARCHS` (one architecture number, default 89), `JAVA_HOME`. `--java-only` and `--native-only` split the build if the machine with network has no CUDA and the GPU node has no network (run `--java-only` first, then `--native-only` with the same `WORK_DIR`). An existing clone is reused after its commit is checked. The script ends with `[build_megaflink] wrote .../megaflink.env`; the native step also prints `MEGA_FLINK_NATIVE_BUILD_OK native_dsos=14 ...` (compiler warnings from `nvcc` are normal).

**Nothing is installed.** Your Flink installation (`FLINK_HOME`) is neither read nor modified: the benchmark driver runs Flink's `MiniCluster` in its own JVM, takes Mega-Flink from the class path in `megaflink.env` (`MEGA_FLINK_CP`, `MEGA_FLINK_NATIVE_DIR`, `CUDA_LIB_DIR`) and the Flink 2.2.1 jars from the Maven cache of the clone. Mega-Flink's planner shim overrides internal Flink planner classes, so putting its jar into a real cluster needs its own installer; that is a different, optional job, and this script does not do it. If you want it: `target/distribution` is made by Mega-Flink's `scripts/build-distribution.sh` (an H20/sm_90 packaging) and installed by the `install-mega-flink.sh` it contains (`install`, `verify`, and an exact backup that its rollback restores); read its README before using it. Not tried here.

**Run all six cases** (CPU baseline and Mega-Flink GPU, final-table SHA-256 compared, one table):

```bash
source $WORK/megaflink/megaflink.env
export FEP_DATA_DIR=<prepared data of step 4> FEP_TMP_DIR=<fast local disk>
$BENCH/scripts/run_all_six.sh --quick --out $OUT/six      # smoke: 100,000 warm-up + 300,000 records, the slice of step 5
$BENCH/scripts/run_all_six.sh --out $OUT/six-full         # one round of the saturated tier: 300,000 warm-up + 3,000,000 records
```

Each case runs once on `cpu-tuned` and once on its Mega-Flink GPU mode: `gpu-batch-tuned` (c1, c3, c4, c5-join), `gpu-1p-tuned` (c2-hop; one-phase aggregation, the mature window path), `gpu-tuned` (c6-lookup). `--cases` selects a subset; `--records`, `--warmup` override the sizes. It builds the driver first if needed. Exit code 0 only if every case is MATCH, the GPU operator really replaced the CPU one (`shim_status` in `result.json`) and, in `--quick` on the standard data, the digest equals the reference of step 5. Expected output of `--quick` on the data of step 4 (the throughputs are in each `result.json` and are not a benchmark):

```
| case | CPU final rows | GPU final rows | CPU sha256 (16) | GPU sha256 (16) | CPU vs GPU | GPU operator replaced | vs REPRODUCE reference |
| c1-filter | 199,254 | 199,254 | 2f29e3f499b32bbf | 2f29e3f499b32bbf | MATCH | yes | equal |
| c2-hop | 85,727 | 85,727 | 528ec5b56b375126 | 528ec5b56b375126 | MATCH | yes | equal |
| c3-topn | 58,028 | 58,028 | 21d7d0915cc5edfa | 21d7d0915cc5edfa | MATCH | yes | equal |
| c4-dedup | 392,746 | 392,746 | 8593325de68ad677 | 8593325de68ad677 | MATCH | yes | equal |
| c5-join | 157,446 | 157,446 | d9600de61e056926 | d9600de61e056926 | MATCH | yes | equal |
| c6-lookup | 400,000 | 400,000 | 07800d9c2c117f9f | 07800d9c2c117f9f | MATCH | yes | equal |
```

**Verified:** this script pair, run from an empty directory on one L20 (driver for CUDA 12.4, CUDA toolkit 12.4, JDK 11): the Java and the native build from a fresh clone of the pinned commit, then `run_all_six.sh --quick`: all six cases MATCH and equal the reference digests. **Not verified:** other GPUs, other CUDA or JDK versions, an H20 build (`ARCHS=90`), `--native-only` on a node without network, the full (non-quick) run of `run_all_six.sh`, and the installer path.

## 8. Check the results

Every run writes `result.json` with `final_sha256`, the SHA-256 of the final table (the changelog applied, rows sorted). Compare per case and tier:

```bash
python3 $BENCH/scripts/summarize.py $OUT/run                # medians table, then MATCH / MISMATCH per case and tier
```

All published results are MATCH. **Do not report a speed for a MISMATCH.** The comparison is on the final table; intermediate changelog row counts differ between CPU and GPU (see README, section 3). CPU and GPU runs are compared with each other when they share one result root (the examples above use `$OUT/run` for both). For a row-level diff, round 1 also keeps `final.tsv`.

## 9. Read the results

- `summarize.py` prints, per case x tier x mode, the median throughput over the rounds, the median of the per-round p50 / p95 / p99, the changelog row count and the final row count, plus the per-round values. This is the statistic of the README (median of 3 rounds; percentiles are per-round, then the median across rounds).
- Throughput is measured records divided by (last sink arrival minus first measured send time). Latency is sink arrival minus the scheduled send time of the input row that made the output visible; for window cases the reference is the first row at or after the window end; for `c5-join-agg` it is the k-th matching pair of a user. In the saturated tier the source is back-pressured, so latency there is only good for relative comparison. See README, section 3.
- `plan.txt` is the `EXPLAIN` text (a CPU run with `--forbid-gpu true`, which `bench.sh` sets for `cpu*` modes, fails if the plan contains a GPU node); `plan-full.txt` is the JSON execution plan and `jobgraph.txt` lists the operator chains. `result.json` also lists `gpu_in_plan`, `shim_status` and `metrics` (the Mega-Flink phase timers; empty on open-source Flink).
- Breakdown: `scripts/breakdown.sh ROOT TIER "case:modes"` makes one JFR-sampled run per mode (works on CPU; adds the phase timers in Mega-Flink mode) and `scripts/digest.py ROOT` prints it. `scripts/nsys-study.sh`, `scripts/nsys_digest.py` (Nsight Systems cross-check) and `scripts/c3-study.sh`, `scripts/c3_digest.py` (the c3 investigation) need Mega-Flink. JFR shares are sampled estimates; the phase timers are measurements (README, sections 3 and 6).

## 10. Known limitations

- One GPU, parallelism 1 for almost everything; there is no multi-GPU run. Parallelism 2 and 4 exist for c2, c3 and c6 only (README, section 5).
- No Kafka or other message system: the log is replayed in-process by one source map function, and the sink captures rows in memory.
- The reported numbers come from a rewritten internal report that has **not been independently re-reviewed** (README status note). Nobody outside the authors has re-run them.
- Open-source Flink for the CPU baseline was checked only on the smoke slice (same plans without GPU nodes, same final tables as a Mega-Flink build in CPU mode); the full matrix was not compared between the two.
- The converter in `prepare_stream_data.py` was verified against the Java reference converter on a small synthetic file, not on the downloaded archive.
- Absolute numbers move with CPU, disk, JDK and GPU; run on an exclusive machine. Concurrent jobs on one node changed throughput by a few percent in screening.
- Do not edit a script while a job is reading it (bash reads scripts incrementally; this once mislabeled runs as failed). The scripts create one temporary directory per run under `FEP_TMP_DIR` and do not remove it; Flink cleans its own files inside.
- Host CPU model: not documented here. The Mega-Flink commit and native build flags of the published GPU numbers are in section 7a (the compiled artifacts of a fresh build were not byte-compared with the original ones).
