<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flink GPU operator examples (KuaiRand replay)

Two DataStream jobs, each runnable with `--engine cpu` (Flink SQL, CPU operators) or `--engine gpu` (the library in `../library`): `WindowFeaturesJob` (per-user HOP 30 min / 5 min slide or 5 min TUMBLE; exposure count, click count, play time; `--window hop|tumble`, `--two-phase true|false`) and `LookupAuthorJob` (INNER lookup of `author_id` per video).

```bash
(cd ../library && mvn -q install -DskipTests)              # library jar
CUDA_TOOLKIT_ROOT=/path/to/cuda ../library/native/build.sh --archs "<your sm, e.g. 89>" --out "$PWD/native-build"
mvn -q package && mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt
export KUAIRAND_DIR=/path/to/converted/data                 # log/*.i64 and video/*.i64, see KuaiRandConvert
java -Xmx16g -Djava.library.path="$PWD/native-build" -cp "target/*:$(cat cp.txt)" \
  com.nvidia.fep.examples.WindowFeaturesJob --engine gpu --window hop --two-phase true --records 3000000 --out out
```

Both engines replay the same log in event-time order through an embedded Flink mini cluster (parallelism 1), capture the result changelog in memory and print the SHA-256 of the sorted final table; run both engines and compare the digests (`--dump-final true` also writes the table as TSV into `--out`). Other options: `--warmup`, `--rate` (rows/s, 0 = as fast as possible), `--data`.

Batch options of the GPU operators (GPU engine only):

| Option | Job | Default | Meaning |
| --- | --- | --- | --- |
| `--batch-rows N` | both | 4096 | rows an operator collects before it launches a kernel |
| `--gpu-min-batch-rows N` | `LookupAuthorJob` | 1024 | a lookup batch with fewer rows is answered by a JVM lookup, not by the GPU; 1 puts every batch on the GPU; must not exceed `--batch-rows` |

The defaults are the values of the benchmark runs (`batchRows=4096`, `gpuMinRows=1024` in the lookup plan; 4096 and a 5 ms wait for the window operators). The library's own lookup-join defaults are `gpuMinRows=1024` (the same) and `batchRows=1,048,576`; the example sets `batchRows` to 4096 as in the benchmark runs.

**Check that the GPU really ran.** After the job, both jobs print one line with the library's operator counters, and write it to `<job>-<engine>-<variant>.metrics.txt` in `--out`:

```
OPERATOR_METRICS job=lookup engine=gpu variant=inner cudfLookupJoinCpuBatches=<n> cudfLookupJoinGpuBatches=<n> cudfLookupJoinInputRows=<n> ...
```

`cudfLookupJoinGpuBatches` is the number of batches that ran the GPU kernel and `cudfLookupJoinCpuBatches` the number answered by the JVM. The window operators report `cudfLocalWindowBatches` (two-phase: one batch is one kernel round), `cudfWindowInputRows` and the output counters. Counters come from the embedded cluster through `MetricsCapture`; on a real cluster use Flink's metric reporters. For kernel counts independent of the library, run the job under `nsys profile -t cuda` and `nsys stats -r cuda_gpu_kern_sum`.

Measured on one L20 (parallelism 1, 300,000 warm-up and 3,000,000 measured rows, saturated replay, 3,300,000 rows through the operator):

| Case | Setting | GPU batches (counter) | JVM batches | kernel launches (nsys) | final table SHA-256 equals CPU |
| --- | --- | --- | --- | --- | --- |
| c6 lookup | `--gpu-min-batch-rows 1024` | 809 | 2 | 809 `lookup_kernel` + 1 table build | yes (`fe6b958a`) |
| c6 lookup | `--gpu-min-batch-rows 1` | 812 | 0 | 812 `lookup_kernel` + 1 table build | yes (`fe6b958a`) |
| c2 HOP, two-phase | `--batch-rows 4096` | 861 (`cudfLocalWindowBatches`) | - | 861 `pack_output`, 860 `packed_global_groupby` + 1 `packed_shared_groupby` | yes (`60fccd33`) |
| c2 TUMBLE 5 min | `--batch-rows 4096` | not counted by this operator (3,300,000 input rows) | - | 889 `pack_output`, 889 `packed_global_groupby` | yes (`6dcf4c11`) |

At a saturated rate the batches fill to 4,096 rows, so both lookup settings run almost every batch on the GPU; the 2 JVM batches are batches under 1,024 rows (partial batches; whether the wait timer or the end of the input flushed them was not checked). At low or mid offered rates batches are smaller and the setting matters. One run per row. `KuaiRandConvert log|video <csv> <dir>` writes the column files from the KuaiRand-27K CSV files. The GPU path needs one CUDA GPU of a compiled-in architecture and `LD_LIBRARY_PATH` or `java.library.path` that finds the `libcudf_flink_*.so` files.

Output (`<...>` stands for values that depend on the run and the machine):

```
RESULT job=window engine=gpu variant=hop-2p throughput=<rows/s> output=<n> final_rows=<n> sha256=<64 hex>
```

`<job>-<engine>-<variant>.result.json` in `--out` has the same fields plus the changelog row counts. Throughput is measured rows divided by the time from the first measured row to the last output row, with source, state, format conversion and sink inside the interval; the two engines start from different row objects (`Row` for SQL, `RowData` for the library), so treat a throughput comparison as indicative.
