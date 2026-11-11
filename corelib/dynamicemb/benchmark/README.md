# Dynamic Embedding Benchmark

## Overview

This folder contains benchmarks about dynamicemb.

## 1.Benchmark EmbeddingCollection

In this benchmark, we provide a simple performance test for dynamic embedding using 8 GPUs. The test utilizes the embedding table from DLRM and performs embedding table fusion to create a large embedding table, followed by lookups for 26 features.

### How to run

```bash
bash ./benchmark/benchmark_embedding_collection.sh <use_index_dedup> <use_dynamic_embedding> <batch_size>
```

#### Parameters

- `<use_index_dedup>`: A boolean flag to enable or disable index deduplication before data distribution.
  - **True**: Enables index deduplication, reducing communication overhead.
  - **False**: Disables index deduplication.
  - **Default**: True.

- `<use_dynamic_embedding>`: A boolean flag to enable or disable the use of dynamic embedding tables.
  - **True**: Enables dynamic embedding tables.
  - **False**: Uses static embedding tables from TorchREC.
  - **Default**: True.

- `<batch_size>`: The global batch size for processing during the benchmark.
  - **Default**: 65536.

### Test Results

In this benchmark, we primarily focus on the performance of embedding collection and deduplication. The tests were conducted on a single node with 8 H100 GPUs connected via NVSwitch. Below are the performance results:

| Configuration               | TorchREC Raw Table (ms) | Dynamic Embedding Table (ms) |
|-----------------------------|-------------------------|-------------------------------|
| Open Dedup, Batch Size 65536 | 14.88                   | 21.56                         |
| Close Dedup, Batch Size 65536 | 23.99                   | 28.47                         |

These results indicate the time taken to perform the embedding collection and deduplication operations under the specified configuration.

During the embedding lookup process, dynamic embedding incurs some performance overhead compared to TorchREC's raw table. However, these overheads diminish when considered within the context of the entire end-to-end model.

## 2.Benchmark BatchedDynamicEmbeddingTables

This benchmark measures forward / backward / evaluation overhead of
`BatchedDynamicEmbeddingTablesV2` against the TorchRec/FBGEMM
`SplitTableBatchedEmbeddingBagsCodegen` baseline on a single GPU.

It is structured as a pytest suite (`benchmark_batched_dynamicemb_tables.py`)
wrapped by a thin shell launcher (`benchmark_batched_dynamicemb_tables.sh`).
All extra flags after the shell script are forwarded to pytest, so you can
use pytest's `-k`, `-x`, `--co`, etc. to select / inspect tests.

### Test suites

Four suites parametrize over `(batch_size, num_tables, optimizer, pooling_mode)`
plus `cache_footprint_ratio` for `TestCaching`.  Defaults: `num_tables=10`
with per-table cap = `_GPU_TOTAL_CAP // nt` (`TestGpu`) or
`_CACHE_TOTAL_CAP // nt` (everything else), so total HBM stays fixed
as `nt` changes.

| Suite           | gpu_ratio | caching | Notes                              |
| --------------- | --------- | ------- | ---------------------------------- |
| `TestGpu`       | 1.0       | False   | Full table in HBM                  |
| `TestCaching`   | 0.1       | True    | 10% HBM as LRU cache               |
| `TestNoCaching` | 0.1       | False   | 10% HBM in HybridStorage           |
| `TestNoHbm`     | 0.0       | False   | Pure UVM                           |

Default per-config knobs: `num_iterations=100`, `embedding_dim=128`,
`feature_distribution="pow-law"` with `alpha=1.05`, `emb_precision=fp32`,
Adam (or SGD) with `learning_rate=0.1`, `eps=1e-8`.

### Run everything

```bash
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh
```

Output goes to stdout and a per-config JSON entry is appended to
`benchmark_results.json` (override via `BENCHMARK_RESULTS_FILE=...`).

### Run a subset

The shell script forwards `"$@"` to pytest, so any pytest selector works:

```bash
# Only the full-HBM suite
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh -k TestGpu

# Only Adam configs (any suite)
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh -k adam

# Combine
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh -k "TestCaching and adam"

# List configs without running anything
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh --co
```

Config labels look like
`T10_totalB1048570_D128_adam_caching_pool=none_cap=25M_cfr=1.0`
(`T<num_tables>_totalB<total_batch>_..._cap=<per-table>M`; uniform
per-table caps collapse to a single `cap=<N>M` token); you can match
any substring of that with `-k`.

### Correctness mode

Correctness mode compares the per-iter forward output of DynamicEmb
against the TorchRec/FBGEMM TBE baseline.  It:

1. Restricts the sparse-feature sampler to `[0, cap/2)` for each table.
2. Mirrors TorchRec's `[0, cap/2)` weight slice into DynamicEmb so every
   lookup hits a key with identical initial values on both backends.
3. Runs the timing loop (`benchmark_train_eval`) with `check_forward=True`
   so every train / eval iter asserts `torch.allclose` with a
   precision-aware tolerance (`atol=1e-4, rtol=1e-3` for fp32).

Two ways to enable it:

```bash
# CLI override: force-enable on every parametrized config
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh --correctness

# Per-config: set `correctness=True` on the BenchmarkConfig in
# _gpu_configs / _caching_configs / etc.
```

Correctness is automatically disabled (with a `UserWarning`) when any
`--profile` mode is set, because profiling captures workloads rather than
validating them.

### Nsight Systems (nsys) profiling

`--profile nsys` switches the run to a dedicated nsys-friendly path:
`run_reporting_loop` runs as untimed warmup, then `benchmark_with_nsys`
wraps the actual sampled iterations in a `cudaProfilerStart` /
`cudaProfilerStop` window.  The benchmark itself only annotates NVTX
ranges; the actual capture happens externally via `nsys profile`.

`benchmark_with_nsys` emits **one Start/Stop window per config**.  We
always use `--capture-range-end=repeat-shutdown:N` (with `N >=` the
number of configs) so nsys records every window it sees -- with `-k
<label>` that's a single window, without filter it's the full suite.
Setup / reporting-loop warmup / between-config gaps stay out of the
recording in both modes.

**Note on the `:N`.** Older nsys releases (including the one on the
cluster's `distributed-recommender` container) reject bare
`repeat-shutdown` and require an explicit cycle count.  Bare `repeat`
is accepted, but its default behavior is a *sampled* capture -- in
practice it keeps only a few cycles when many fire in sequence, so
some configs silently end up missing from the trace.  Always pass an
explicit `:N` and size it to the number of `cudaProfilerStart` calls
your script will issue (10 for the full benchmark suite; use a larger
N like 20 if unsure).

**Single config** -- pick one config with `-k <label>`:

```bash
nsys profile \
    --output=trace_dyn \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt,mpi,ucx \
    --cuda-graph-trace=node \
    --cuda-flush-interval=100 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat-shutdown:10 \
    --target-processes=all \
    bash ./benchmark/benchmark_batched_dynamicemb_tables.sh \
        --profile nsys -k "TestGpu and adam"
```

**Multi config** -- whole suite into one trace:

```bash
nsys profile \
    --output=trace_all \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt,mpi,ucx \
    --cuda-graph-trace=node \
    --cuda-flush-interval=100 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat-shutdown:10 \
    --target-processes=all \
    bash ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile nsys
```

Flag breakdown (shared by both invocations):

| Flag | Meaning |
| --- | --- |
| `--sample=none` | disable periodic CPU stack sampling -- cuts overhead and trace size when only GPU/NVTX is needed |
| `--cpuctxsw=none` | drop OS context-switch records -- noisy on multi-core, rarely needed for kernel-level perf |
| `--trace=cuda,nvtx,osrt,mpi,ucx` | trace CUDA API + GPU work, NVTX markers, OS-runtime syscalls, MPI calls, UCX calls (the latter two are no-ops on single-node single-GPU and can be dropped) |
| `--cuda-graph-trace=node` | expand CUDA Graph launches into per-node events so kernels inside a captured graph stay individually visible |
| `--cuda-flush-interval=100` | flush the CUDA event buffer every 100 ms so long runs don't overflow the ring buffer and drop events |
| `--capture-range=cudaProfilerApi` | start recording only when the process calls `cudaProfilerStart` (skips table/data setup and the reporting-loop warmup) |
| `--capture-range-end=repeat-shutdown:N` | record up to `N` Start/Stop windows into the same trace (skip gaps in between), end the session at process shutdown.  Set `N` ≥ number of `cudaProfilerStart` calls -- e.g. `:10` for the full suite.  Bare `repeat-shutdown` is rejected on older nsys; bare `repeat` is accepted but silently drops cycles when many fire back-to-back |
| `--target-processes=all` | follow children/forks too -- needed because `torchrun` spawns a worker process to host pytest |
| `--force-overwrite=true` | overwrite an existing output file instead of erroring out |
| `--output=<name>` | output file (`.nsys-rep` is the current extension; `.qdrep` from older releases still works) |

NVTX layout inside each window:
```
<cfg.label()>                          # e.g. T10_totalB1048570_D128_adam_gpu_...
└─ nsys_iter_0                         # per-iter
   ├─ dyn → forward / backward
   └─ trc → forward / backward
└─ nsys_iter_1
└─ ...
```

Browse the trace with `nsys-ui` (locally) or summarize on the cluster:

```bash
# pick a kernel-time top-N across all configs
nsys stats --report cuda_gpu_kern_sum trace_all.nsys-rep | head -30

# group kernel time by NVTX range (= cfg.label()) to compare configs
nsys stats --report cuda_gpu_kern_gb_sum trace_all.nsys-rep | head -30
```

#### Per-op breakdown pipeline (`.nsys-rep` → sunburst PNG)

End-to-end pipeline that reduces each config's nsys trace into a CSV of
per-op GPU time and renders it as a sunburst chart.  Tools:
`nsys_breakdown.py` (trace → CSV) and `nsys_sunburst.py` (CSV → PNG).

**Op attribution.**  `batched_dynamicemb_function.py` wraps the
embedding-side hot path in `torch.cuda.nvtx.range("op:<name>")` markers
(13 ops: `cache_lookup`, `storage_find`, `gather_embedding`,
`reduce_grads`, `optimizer_update_fused`, ...).  Each launch falls into
the innermost `op:*` range active on the CPU side at launch time
(CUPTI_ACTIVITY_KIND_RUNTIME — *not* GPU exec time, which would miss
async backward kernels).  Launches with no `op:*` parent above them
are kept individually under their kernel name so they remain visible
in the chart.

**Step 1 — capture one `.nsys-rep` per config.**  The recommended
approach is one `nsys profile` invocation per config (one
`--capture-range-end=stop` window per process).  This sidesteps a
buffer-rotation bug in older nsys releases that silently drops cycles
when many `repeat`/`repeat-shutdown:N` windows fire back-to-back in a
single process.  On the cluster, drive it from a loop like:

```bash
# pseudocode — see run_nsys_loop.sbatch for the working version
for TEST_ID in $(pytest --collect-only -q ...); do
    LABEL=$(echo "$TEST_ID" | sed -E 's/.*\[(.+)\]$/\1/')
    nsys profile \
        --output=nsys_run/trace_${LABEL} \
        --sample=none --cpuctxsw=none \
        --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        torchrun --nnodes 1 --nproc_per_node 1 \
            -m pytest -svv --profile nsys "$TEST_ID"
done
```

After the loop you have one `trace_<NN>_<cfg-label>.nsys-rep` per
config.

**Step 2 — `nsys_breakdown.py`: trace → per-config CSV.**  Exports
each `.nsys-rep` to a sqlite via `nsys export`, then joins
`CUPTI_ACTIVITY_KIND_RUNTIME` (CPU launch times) with
`CUPTI_ACTIVITY_KIND_KERNEL` (GPU exec times) and `NVTX_EVENTS` to
attribute every kernel to the innermost `op:*` range live at launch:

```bash
python nsys_breakdown.py nsys_run/*.nsys-rep --out-dir nsys_run/
# -> nsys_run/opcalls__<cfg-label>.csv  (one CSV per .nsys-rep)
```

Each CSV row is one op (or one bare kernel) with columns:
`phase,parent_stages,op,calls_per_iter,avg_ms_per_iter,total_ms,num_iters,kernel_names,fallback_category`.

**Step 3 — `nsys_sunburst.py`: CSV → PNG.**  Renders a 3-ring
sunburst per CSV (phase / stage / op) with a legend table listing
letter codes, op names, and percentage of total iteration time:

```bash
python nsys_sunburst.py nsys_run/opcalls__*.csv
# -> nsys_run/opcalls__<cfg-label>.sunburst.png  (next to each CSV)
```

The PNG drops next to its CSV by default; pass paths individually to
control output location.  Example outputs live under
[`plots/breakdown_sgd_gpu.png`](./plots/breakdown_sgd_gpu.png) and
[`plots/breakdown_adam_caching_cfr1.0.png`](./plots/breakdown_adam_caching_cfr1.0.png).

Working artifacts (`.nsys-rep`, `.sqlite`, intermediate CSV/PNG) are
expected to land under `local/` (gitignored).  Only the curated
figures committed under `plots/` are tracked.

### Nsight Compute (ncu) profiling

`--profile ncu` profiles per-kernel hardware counters (unlike nsys, which is a
timeline).  It is a two-step workflow:

1. **`--profile ncu-gen`** prints the exact `ncu` command for a config (single-
   config and whole-suite variants) and exits without running anything.  This
   is the source of truth for the literal command.  Optionally pass
   `--ncu-kernel-regex` to name the kernel(s) to profile (emitted verbatim as
   `--kernel-name 'regex:<value>'`); omit it to profile **every** kernel within
   the `--nvtx-include` scope (e.g. all kernels of an op range).
2. **`--profile ncu`** is the inner workload the printed command wraps: it warms
   up via the reporting loop, runs every iteration inside a single
   `cudaProfilerStart/Stop` window (outer NVTX range `cfg.label()`, inner range
   `ncu_iter`, per-iter range `iter_{i}`), and **by default only iteration 0 is
   actually captured** -- the printed command's `--nvtx-include` scopes it.  Use
   `--ncu-iterations` to widen.

So you generate the command, then run it:

```bash
# Step 1 — print the command for one TestGpu config.  -k takes any pytest
# expression matched against the test id (class + cfg.label()); the concise
# "TestGpu and adam" uniquely selects the adam config today.  --ncu-kernel-regex
# is optional (omit to profile all kernels in the nvtx scope).  ncu-gen prints a
# command whose inner -k is the exact, lossless label.
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh \
    --profile ncu-gen -k "TestGpu and adam" \
    --ncu-kernel-regex 'segmented_unique|table_insert'

# Step 2 — run the printed command (shape shown here; ncu-gen emits the exact
# -k label and your --ncu-kernel-regex verbatim):
ncu -f --target-processes all \
    --profile-from-start off \
    --nvtx --nvtx-include 'ncu_iter/iter_0/' \
    --kernel-name 'regex:segmented_unique|table_insert' \
    --set full \
    --import-source=yes \
    --csv --page raw \
    -o ncu_T10_totalB1048570_D128_adam_gpu_pool=none_cap=1M \
    bash ./benchmark/benchmark_batched_dynamicemb_tables.sh \
        --profile ncu -k 'T10_totalB1048570_D128_adam_gpu_pool=none_cap=1M'
```

To profile **all** TestGpu configs (currently `adam` + `sgd`) into one report,
drop the exact label and select the class: `-k TestGpu` on both the `ncu-gen`
and the wrapped `--profile ncu` invocation; the outer `cfg.label()` NVTX range
attributes kernels per config.

| ncu flag | why |
| -------- | --- |
| `--target-processes all` | ncu launches `bash`, which `torchrun`-spawns the python worker that runs the CUDA kernels; without this ncu would only watch the kernel-less `bash`. |
| `--profile-from-start off` | gate profiling to the `cudaProfilerStart/Stop` window (skips table build + warmup).  Must stay paired with `cudaProfilerStart` -- dropping the call profiles nothing. |
| `--nvtx --nvtx-include 'ncu_iter/iter_0/'` | belt-and-suspenders NVTX-range gate, and the iteration selector: only kernels under the listed `iter_{i}` ranges are profiled (default `iter_0`; `--ncu-iterations` emits one include per selected iter). |
| `--kernel-name 'regex:...'` | restrict to the kernel(s) named via the optional `--ncu-kernel-regex`; omitted entirely when that flag is not given, so all kernels in the NVTX scope are profiled (e.g. every kernel of an op range like `op:segmented_unique`). |
| `--set full` | full metric set (heaviest; most replay passes). |
| `--import-source=yes` | embed CUDA source so the Source page works offline (relies on `-lineinfo`, which the build now sets). |

`--num-iterations N` overrides `BenchmarkConfig.num_iterations` (default 100) on
every config.  This is the number of sampled batches, so it also bounds the
warmup/reporting loop and how many iterations each profile mode covers.  Lower
it to keep `--profile ncu` tractable (ncu replays every matched kernel launch
with the full metric set), e.g. add `--num-iterations 3` to the wrapped
`--profile ncu` invocation.

`--ncu-iterations` (only with `--profile ncu`) selects *which* iterations ncu
captures, widening the default of iteration 0 only.  It accepts either a comma
list or a Python-style `begin:end:step` slice (end exclusive; parts optional):

```bash
--ncu-iterations 0,3,7      # capture iterations 0, 3, 7
--ncu-iterations ::2        # every other iteration
--ncu-iterations 2:20:3     # 2,5,8,11,... (clamped to num_iterations)
--ncu-iterations 90:        # the last iterations of a 100-iter run
```

The selection is applied in the **ncu command**, not the workload: a single
`cudaProfilerStart/Stop` pair still wraps the whole run (ncu reliably honors
only one window), and `ncu-gen` turns the selection into one
`--nvtx-include 'ncu_iter/iter_{i}/'` per selected iteration (OR-ed together).
Unselected iterations still run (to advance the table / keep the cache warm)
but their kernels are filtered out by NVTX.  This differs from
`--num-iterations`: that shrinks the *total* iteration count (and warmup);
`--ncu-iterations` keeps the full run but only profiles a subset -- the right
tool when you need a warmed, steady-state iteration rather than the first few.
Pass it to `ncu-gen` so the printed command carries the right `--nvtx-include`
filters.

#### Capturing all kernels of an op range

A single dynamicemb op launches several **differently-named** kernels
(`segmented_unique` runs `expand_table_ids`, `segmented_init`,
`segmented_unique_kernel`, two `cub::DeviceScan*`, `compact_keys_and_freq`,
`adjust_output_indices`, plus a couple of torch fill kernels).  A
`--kernel-name` regex can only match one name, so to profile the **whole op**
you scope by its NVTX range instead and drop the kernel-name filter.

The prefetch/forward path wraps each op in an `op:<name>` NVTX range (e.g.
`op:segmented_unique`, `op:cache_lookup`, `op:gather_embedding`,
`op:reduce_grads`, `op:store_to_flat`, ...).  Their nesting inside a profiled
iteration is:

```
ncu_iter / iter_0 / forward / dynamicemb_prefetch / op:segmented_unique / <kernels>
```

To capture every kernel of `op:segmented_unique` in iteration 0, point
`--nvtx-include` at that nested path and omit `--ncu-kernel-regex` (so no
`--kernel-name` filter is emitted):

```bash
ncu -f --target-processes all --profile-from-start off \
    --nvtx --nvtx-include 'ncu_iter/iter_0/forward/dynamicemb_prefetch/op:segmented_unique/' \
    --set full --import-source=yes --csv --page raw \
    -o ncu_sgd_segunique_all \
    bash ./benchmark/benchmark_batched_dynamicemb_tables.sh --profile ncu -k "TestGpu and sgd"
```

The trailing `/` includes the whole subtree, so all kernels launched inside the
range are profiled regardless of name.  (Anchoring on just `op:segmented_unique/`
would match the op in *every* iteration; the `ncu_iter/iter_0/.../` prefix pins
it to iteration 0.  If the intermediate nesting changes, replace the middle
levels with wildcards: `ncu_iter/iter_0/*/*/op:segmented_unique/`.)

Other profile modes:

| `--profile` value | What it does                                                          |
| ----------------- | --------------------------------------------------------------------- |
| (omitted)         | Normal timing path; reports avg fwd/bwd/train/eval (ms) per config.    |
| `torch`           | Runs each backend under `torch.profiler`; exports Chrome trace + bandwidth report. |
| `nsys`            | NVTX-annotated profile path described above.                          |
| `ncu-gen`         | Prints the matching `ncu` command for the config and exits.           |
| `ncu`             | Warms up via the reporting loop, runs all iterations inside one `cudaProfilerStart/Stop` window, and captures iteration 0 by default (widen with `--ncu-iterations`). |

### Cache footprint sizing (TestCaching)

`TestCaching` is parametrized by `cache_footprint_ratio` (currently
`[0.8, 1.0]`).  At runtime the harness:

1. Generates the full sparse-feature stream before any table is built.
2. Counts distinct keys touched per table across all iterations -- the
   workload's true HBM footprint.
3. Resizes the cache so it holds
   `footprint × cache_footprint_ratio` rows per table
   (DynamicEmb `local_hbm_for_values`, FBGEMM `cache_load_factor`).

This means TestCaching emits one config per ratio in the suite, with
labels like `..._caching_..._cfr=0.8` / `..._cfr=1.0`.  The other three
suites (`TestGpu`, `TestNoCaching`, `TestNoHbm`) leave
`cache_footprint_ratio=None` and keep their construction-time
`hbm_for_embeddings` / `gpu_ratio`.

Cache state from the warmup (`run_reporting_loop`) is **kept** for the
subsequent timed `benchmark_train_eval` — only the hit-rate recorder is
disabled.  The timed numbers therefore measure steady-state cache
behavior, not a cold start.

### Plotting results

`benchmark_results.json` produced above is visualized with
`plot_benchmark_results.py`.  The script writes a single figure
(`benchmark_bdet_plot.png`) into the output directory (default
`./plots/`).  The caching column auto-fans-out into one panel per
`cache_footprint_ratio` present in the JSON, so cfr=0.8 and cfr=1.0 sit
side-by-side in the same image (the figure width scales with column
count).

```bash
# Main figure into ./plots/benchmark_bdet_plot.png
python plot_benchmark_results.py

# Different output directory / results file
python plot_benchmark_results.py \
    --results /path/to/results.json \
    --out-dir /tmp/bdet_plots

# Log y-axis (one suite dominates the range)
python plot_benchmark_results.py --log

# Hide bar value labels
python plot_benchmark_results.py --no-values
```

Each figure carries a two-line subtitle auto-derived from the result
dict:

```
NVIDIA H100 80GB HBM3  ·  D=128  ·  batch=1,048,570
pow-law(α=1.05)  ·  hotness=10  ·  pool=none
```

`cache_footprint_ratio` is intentionally not in the subtitle — it shows
up in the per-panel headers on the top secondary x-axis instead
(`Caching cfr=0.8`, `Caching cfr=1.0`).
Fields populated by `run_single_benchmark`: `gpu_name`, `embedding_dim`,
`batch_size`/`num_tables`, `feature_distribution`, `alpha`, `max_hotness`,
`pooling_mode`, `cache_footprint_ratio`.  Any field missing from a legacy
JSON is silently skipped.

Layout: one row per optimizer; within each row, every panel (`GPU`,
one per caching ratio, `NoCaching`, `NoHBM`) is drawn side-by-side on a
**shared y-axis**.  Panel names sit on a secondary axis at the top,
train/eval labels under the bars, dotted separators between panel
regions.  Each panel shows a stacked `train (fwd + bwd)` bar plus a
separate `eval` bar for DynamicEmb vs TorchRec.  Because modes span
~0.5 ms (GPU) to ~40 ms (NoHBM), GPU bars look small under the shared
scale — pass `--log` to expand the low end.

### Test Results

The figures below were collected on **NVIDIA H100 80GB HBM3** (single GPU)
with `pow-law(alpha=1.05)` index distribution.

Run configuration:
- hardware: NVIDIA H100 80GB HBM3 (single GPU)
- embedding_dtype: float32
- embedding_dim: 128
- num_tables: 10 (per-table batch = total / 10 = 104,857; total batch = 1,048,570)
- cache_algorithm: lru
- gpu_ratio: 1.0 (`TestGpu`) / footprint × `cache_footprint_ratio`
  (`TestCaching`) / 0.1 (`TestNoCaching`) / 0.0 (`TestNoHbm`)
- per-table capacity: 1.6M (`TestGpu`, `_GPU_TOTAL_CAP // nt` = 16M / 10, sized so DynamicEmb + TorchRec TBE both fit in 80 GB), 25.6M otherwise (`_CACHE_TOTAL_CAP // nt` = 256M / 10)
- optimizers: adam (`eps=1e-8`) and sgd
- num_iterations: 100

Latency by suite (DynamicEmb vs TorchRec TBE, lower is better):

![benchmark of BatchedDynamicEmbeddingTables vs TorchRec TBE](./plots/benchmark_bdet_plot.png)

#### Per-op breakdown (DynamicEmb side)

Two representative configs profiled via `nsys` and reduced with
`nsys_breakdown.py` + `nsys_sunburst.py` (see [nsys profiling](#nsight-systems-nsys-profiling)).
The sunburst rings are: phase (forward / backward) → stage
(prefetch / forward / backward) → op (NVTX `op:*` range, or kernel name
when no wrapper is present).  Percentages are share of the full
iteration time; letter codes index the legend table below each chart.

`sgd + GPU (full HBM)` — 1.59 ms/iter; cost dominated by
`segmented_unique` (~24%), `gather_embedding` (~19%), and
`optimizer_update_fused` (~12%):

![sgd gpu per-op breakdown](./plots/breakdown_sgd_gpu.png)

`adam + Caching cfr=1.0` — 11.63 ms/iter; cache traffic dominates:
`storage_insert` (~32%), `storage_find` (~25%), `cache_insert_evict`
(~15%), `cache_lookup` (~8%), with the Adam update (`optimizer_update_fused`)
only ~5%:

![adam caching cfr=1.0 per-op breakdown](./plots/breakdown_adam_caching_cfr1.0.png)

## 3. Benchmark the lookup of many embedding collections

This benchmark measures what **blocking host syncs inside DynamicEmb** cost when a
model has many separate embedding collections and very little GPU work per step.
Each collection is its own `BatchedDynamicEmbeddingTablesV2`, so every step it does
its own unique / lookup / update, and every host sync (reading the unique count,
compaction counts, eviction counts, ...) stalls the CPU until the GPU has drained.
The timed region holds nothing but the lookups (no dense part, no loss, no data
creation), so the GPU work per step is small and launch latency and sync stalls
dominate the step time: that is where removing a sync shows.

The script `benchmark_many_collections.py` (launcher `benchmark_many_collections.sh`)
uses only public DynamicEmb APIs, so the same file runs unchanged on any tree.
It needs one GPU, no `torchrun` and no process group.

### What one step does

- K collections (default 32), each with T tables (default 2) of dim D (default 64),
  sum-pooled, fp32, fused optimizer in the backward (`--optimizer sgd|adam`).
- A step is the lookup of one input: the K collections' forward, then their K
  backwards (`torch.autograd.backward` with gradients built beforehand). The backward
  is part of the lookup. In the `caching` and `hbm` layouts a training forward pins
  every key it reads with a reference count that only the backward releases (a pinned
  key is never evicted); in every layout the backward makes the fused update of the
  rows, and in `hybrid` and `host` writes them back. Nothing else is timed: no dense
  part, no loss, no data creation and no host-to-device copy.
- `--inputs` (default 2) inputs are generated on the GPU up front and used in turn:
  input 0, input 1, input 0, and so on. Every collection gets B samples (default 1024)
  x H keys per sample (default 4) per table, drawn from a pow-law distribution
  with exponent `--alpha` (default 1.05; larger means more repeated keys, which makes
  the unique/dedup stage matter more) over its own slice of the `--num-embeddings`
  rows (default 1M), so the inputs share no key. A step never finds in a cache the
  keys of the step before it; it finds those of its own earlier use, `--inputs` steps
  back, as far as the cache still holds them: all of them if it holds all the inputs,
  `(rows - U) / U` of them if it holds one input of U unique keys and part of another,
  hardly any below that. A cache that holds less than all the inputs is therefore
  refreshed at every step (`--inputs 1` uses one input at every step: after the first,
  all hits).

### How to run

```bash
cd corelib/dynamicemb

# One point with the defaults (32 collections x 2 tables, 1024 samples, HBM, SGD)
bash ./benchmark/benchmark_many_collections.sh

# Other storage layouts: caching, hybrid, host (default: hbm)
bash ./benchmark/benchmark_many_collections.sh --storage caching

# Also count blocking syncs per step and the GPU-busy fraction, and save JSON
bash ./benchmark/benchmark_many_collections.sh --sync-count --json hbm.json

# K in {1,4,16,64} x B in {256,1024,4096} in one process, one row per point
bash ./benchmark/benchmark_many_collections.sh --sweep --storage hbm --sync-count --json hbm_sweep.json

# Before/after table of two result files (needs no GPU and no torch)
bash ./benchmark/benchmark_many_collections.sh --compare main.json nosync.json
```

`python benchmark/benchmark_many_collections.py <flags>` is equivalent. Main flags
(`--help` lists all of them with their defaults):

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--collections` / `--tables` / `--dim` | 32 / 2 / 64 | K, T and D |
| `--batch-size` / `--hotness` | 1024 / 4 | B samples and H keys per sample and table |
| `--num-embeddings` / `--alpha` | 1000000 / 1.05 | rows per table, pow-law exponent |
| `--optimizer` | `sgd` | `sgd` or `adam` (Adam keeps 2*D state per row: 3x the memory) |
| `--storage` | `hbm` | `hbm`, `caching`, `hybrid` or `host`, see below |
| `--hbm-ratio` | `caching`: from the inputs, `hybrid`: 0.1 | share of each table's rows in HBM for `caching` (the cache) and `hybrid` (the HBM tier); a caching layout is given the smallest cache that holds the unique keys of an input, see the notes |
| `--warmup` / `--steps` / `--windows` | 30 / 200 / 5 | untimed steps, timed steps, timed windows (`--sweep`: 20 / 100) |
| `--inputs` / `--seed` | 2 / 0 | inputs used in turn, each on its own slice of the rows; RNG seed |
| `--sync-count` / `--sync-steps` | off / 3 | profile extra steps with `torch.profiler`, see below |
| `--profile-steps` | 0 | bracket this many extra steps for nsys, see below |
| `--sweep` | off | the grid K in {1,4,16,64} x B in {256,1024,4096} (`SWEEP_*` constants in the script) |
| `--json PATH` / `--compare A B` | | write results / compare two result files |
| `--device` | current | CUDA device index (0-9) |

Storage layouts (the script checks that the library really built the requested one):

| `--storage` | Layout | Values live in |
| ----------- | ------ | -------------- |
| `hbm` | `HBM_ONLY` | HBM |
| `caching` | `CACHING` | HBM cache of `--hbm-ratio` of the rows over a full host copy |
| `hybrid` | `HYBRID` | `--hbm-ratio` of the rows in HBM, the rest in host memory (no cache) |
| `host` | `HOST_ONLY` | host memory |

The planned memory is printed, and the run refuses a point whose planned HBM exceeds
90% of the free device memory (in a sweep it is reported as `FAILED` and the sweep
continues). With 1M rows per table the planned HBM is 17.2 GiB for the defaults
(SGD), 47.7 GiB for Adam, 34.4 GiB for SGD with 64 collections, and 95.4 GiB for
Adam with 64 collections (too much for one 80 GB GPU: use
`--storage caching|hybrid|host` or a smaller `--num-embeddings`).
`caching` and `host` keep every value in host memory, 15.3 GiB for the defaults with
SGD, and `hybrid` keeps the share that is not in HBM (13.7 GiB at the default 0.1);
the hash tables always stay in HBM (about 1.9 GiB).

### How to read the output

```
storage    K      B |        wall ms/step        |        host ms/step        | host/wall | gpu busy |  syncs/step (per coll) | cache hit
```

Every number is the median over the timed windows, followed by `[min, max]`.

- **wall ms/step**: all steps of a window are launched back to back and the device
  is synchronized once at the end. This is the throughput of a real training loop
  that does not sync by itself, and the number to compare between trees.
- **host ms/step**: CPU time spent inside the step function, measured without any
  synchronization. Blocking syncs show up here: with them the CPU waits for the GPU
  in the middle of the step and the two serialize (`host/wall` near 100%, and the
  step takes about CPU time plus GPU time). On a sync-free tree `host` is the pure
  launch cost; if `wall` is still close to `host` the step is bound by CPU launch
  overhead, if `wall` is larger the GPU is the bottleneck and the CPU runs ahead.
- **gpu busy** (with `--sync-count`): summed duration of GPU activities divided by the
  span from the first step to the last GPU activity of the profiled steps. The
  profiler's own CPU overhead lowers it, so compare it between trees and do not
  compare it with the timed numbers.
- **syncs/step** (with `--sync-count`): blocking sync API calls per step, and per
  collection in parentheses. It counts `cudaStreamSynchronize`, `cudaDeviceSynchronize`,
  `cudaEventSynchronize` and synchronous `cudaMemcpy`; every `.item()`, `.cpu()` or
  boolean-mask indexing is one `cudaStreamSynchronize`. The single-point run also
  prints the breakdown and the number of D2H copies into pageable memory
  (`d2h_pageable`: they block the host too, but are normally followed by a counted
  stream sync, so they are not added). A tree without syncs reads 0. If the profiler
  returns no CUDA runtime events the run prints a note instead of numbers.
- **cache hit** (`caching` only): the share of the unique keys that the cache of
  collection 0 found, over one round of the inputs in steady state (the cache metrics
  of the library, read after the timed windows). `--compare` prints it for both sides:
  two trees are only comparable in the same regime.

A single-point run also prints the layout, the planned memory, the unique keys of an
input and the distinct keys of all inputs and, for `caching`, what the cache of
collection 0 did at the step of each input of one round: how many of the unique keys
(over its tables) it found, inserted and evicted (or could not place). The pooled
outputs are checked for being finite after every window.

### Comparing two trees

Run the same command on each tree and compare the JSON files. The script imports
whatever `dynamicemb` is installed (or first on `PYTHONPATH`), so install or select
the whole intended tree first, the Python package and its compiled extension together
(the two trees of a comparison differ in both); the first lines of the output, and
`meta` in the JSON, give the imported path, the path of the extension and the git
commit (and a dirty flag) of that tree.

```bash
# tree A (for example origin/main), with its dynamicemb installed
bash ./benchmark/benchmark_many_collections.sh --sweep --storage hbm --sync-count --json main_hbm.json
# tree B (the change under test), with its dynamicemb installed
bash ./benchmark/benchmark_many_collections.sh --sweep --storage hbm --sync-count --json nosync_hbm.json

bash ./benchmark/benchmark_many_collections.sh --compare main_hbm.json nosync_hbm.json
```

`--compare` matches points by configuration (everything but the seed) and prints
wall ms/step, host ms/step and syncs/step of A and B, the speedup A/B (above 1 means
B is faster) and the geometric mean over the points. Use the same flags on both
sides; repeat and alternate the runs (A, B, A, B) to see the run-to-run spread, since
this benchmark is CPU-bound by design and sensitive to other work on the node.

### Profiling with nsys

Every step is wrapped in NVTX ranges: `step_<i>`, one `collection_<k>` per collection
(its forward: prefetch with unique, find and insert, then the pooled gather), and
`backward`, which runs each collection's `DynamicEmbeddingFunction.backward` (reduce
gradients and fused update, in the library's own ranges on the autograd thread).
`--profile-steps N` runs N extra steps between `cudaProfilerStart` and
`cudaProfilerStop`, so nsys records only those:

```bash
nsys profile \
    --output=many_collections --force-overwrite=true \
    --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python benchmark/benchmark_many_collections.py \
        --storage hbm --collections 32 --warmup 30 --steps 10 --windows 1 --profile-steps 3

# call counts and time per CUDA API; divide by --profile-steps for per step
nsys stats --report cuda_api_sum many_collections.nsys-rep | head -20
```

In the timeline, blocking syncs are long `cudaStreamSynchronize` blocks on the CPU
thread inside the `collection_<k>` ranges, with the GPU idle around them; on a tree
without them the launches run back to back and the GPU rows lag behind the CPU.
With `--sweep`, `--profile-steps` opens one capture window per point: use
`--capture-range-end=repeat-shutdown:N` with N at least the number of points.

### Notes

- TorchRec `EmbeddingCollection` / `DistributedModelParallel` mode is not covered on
  purpose: the point is to isolate DynamicEmb's own syncs from the ones TorchRec's
  input distribution adds. There is no collective in this benchmark.
- Tables much larger than the keys they see hold every key of the inputs after the
  first pass over them, so for `hbm` and `host` the timed steady state is the all-hit
  path (every lookup finds its key: no new key is admitted, nothing is evicted), and so
  it is for a `hybrid` whose HBM tier (`--hbm-ratio 0.1`, 100K rows) holds the keys of
  all inputs. What `caching` does depends on the size of its cache, which holds a
  multiple of 1024 rows (its buckets hold 1024). The library raises (`Outstanding
  prefetched keys ... exceed cache capacity`) when the unique keys of one step do not
  fit, so the script sizes the cache from the inputs unless `--hbm-ratio` is given: the
  smallest cache that holds the unique keys of an input in any table, that is the
  smallest one for which a step is valid. At the defaults an input has about 2.1K
  unique keys per table (the script prints the number), so the cache gets 3072 rows
  and the two inputs together, about 4.2K keys, do not fit. Measured on an H100 with
  the defaults, for collection 0 (two tables) and one step:

  | cache rows per table | `--hbm-ratio` | keys found | inserted and evicted |
  | -------------------- | ------------- | ---------- | -------------------- |
  | 2048 | 0.002 | 3% | 4096 |
  | 3072 (the default, from the inputs) | 0.0025 | 45% | 2295 |
  | 4096 | 0.004 | 93% | 299 |
  | 5120 and more | 0.005 and more | 100% | 0 |

  With 3072 rows the cache holds one input and 45% of the other, so every step misses
  on about half of its keys and replaces about a third of the cache: the miss, insert
  and eviction paths run in every step. A cache of 2048 rows is smaller than the keys
  of one input: all of it is replaced at every step and the step takes about 2.5 times
  longer than at 3072 rows, which is not a regime to compare trees in. For each
  `caching` run the script prints these numbers (and `cache hit` in the table), so the
  regime can be read off the output. The cache is sized for each point of a `--sweep`,
  so its hit rate changes with the batch size.
- A caching layout checks that the keys a step pins fit its cache. The check compares
  the number of keys of the step (B x H per table, a bound) with the capacity of the
  cache, its rows plus an overflow area of 3072 per table. A tree that does not know
  the number of unique keys on the host (the one of #545) reads it from the device in
  that step only when the bound exceeds the capacity: at the defaults it never does
  (`--sync-count` reads 0), but with a larger batch (B = 4096, say) every forward of a
  collection makes one blocking read, and `--sync-count` shows it.
- Keep `--warmup` at least `--inputs`: the first pass over the inputs inserts the keys.
- The profiler pass (`--sync-count`) runs after the timed windows and does not affect
  them. The sync counts are exact API call counts; the GPU-busy fraction is not
  comparable with the timed numbers (see above).
- Every collection prints a memory table when it is built; only the one of
  collection 0 is shown (single-point runs). After each point the modules are freed,
  and a warning is printed if more than 1 GiB of GPU memory is still held.
