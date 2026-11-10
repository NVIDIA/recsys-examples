# gpu-fusion-plugin: expression fusion for spark-rapids

An independent Spark plugin jar. After spark-rapids has turned the plan into GPU operators, it replaces chains of row-wise
expressions in `GpuProject`, `GpuFilter` and the aggregate inputs of `GpuHashAggregate` by **one generated CUDA kernel**
(compiled at run time with NVRTC) instead of one cuDF call per expression with every intermediate column written to GPU
memory. spark-rapids itself is not modified. If anything goes wrong the original (unfused) operator runs instead.

Tested with: Spark 3.5.9 (Scala 2.12), spark-rapids 26.08.1 (`cuda12` jar, cuDF 26.08.0), Java 11, GPUs L20 (sm_89),
RTX 6000D Blackwell (sm_120), H20 (sm_90).

## 1. Use it

```
spark-submit \
  --jars rapids-4-spark_2.12-26.08.1-cuda12.jar,gpu-fusion-plugin-0.1.0.jar \
  --conf spark.plugins=com.nvidia.spark.SQLPlugin \
  --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
  --conf spark.rapids.sql.fusion.mode=SAFE \
  ...
```

* Put **both** jars in `--jars` (or `spark.jars`). Do not also put them on `--driver-class-path` (the shim classes of
  spark-rapids are loaded in the same class loader as the plugin; mixing loaders gives `NoClassDefFoundError`).
* The plugin is a Spark SQL extension. It must be loaded together with the RAPIDS plugin. Its rule runs in
  `postColumnarTransitions`, i.e. after spark-rapids' own transition rules (also inside AQE query stages).
* NVRTC: every executor needs `libnvrtc.so.12` (CUDA 12.x runtime compiler, 12.8+ for Blackwell sm_120) reachable by the dynamic
  loader (e.g. `LD_LIBRARY_PATH`) or set `spark.rapids.sql.fusion.nvrtcPath=/path/to/libnvrtc.so.12`. If NVRTC cannot be
  loaded the plugin logs one warning and everything runs unfused.
* Scala UDFs: add `com.nvidia.spark.udf.Plugin` to `spark.sql.extensions` and set
  `spark.rapids.sql.udfCompiler.enabled=true`; UDFs that spark-rapids' udf-compiler translates to Catalyst expressions are
  fused like any other expression.
* For local testing use `scripts/run-spark.sh` (Spark local mode, optional srun via `SLURM_PARTITION`); `scripts/build.sh` builds the jar.

### Quick start (paths relative to `batch/plugin/`)

```
scripts/build.sh                                      # -> target/gpu-fusion-plugin-0.1.0.jar (shades JCuda; no GPU needed)
scripts/selftest.sh                                   # no-GPU check: every operator kernel compiles with NVRTC for sm_75..sm_120
examples/run-demo.sh                                  # PySpark demo on one GPU (SQL + DataFrame, explain, stats)
scripts/run-spark.sh spark-shell --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
        --conf spark.rapids.sql.fusion.mode=SAFE       # interactive; then: Fusion.explain(df)
```

Python/PySpark: nothing special; `spark.sparkContext._jvm.com.nvidia.spark.rapids.fusion.Fusion.explain(df._jdf)` prints the
fusion decisions and plan (see `examples/demo.py`).

## 2. Configuration (prefix `spark.rapids.sql.fusion.`)

| key | default | meaning |
|---|---|---|
| `mode` | `OFF` | `OFF` / `SAFE` / `AGGRESSIVE` (larger region limits; ops marked experimental). Read at planning time of every query, so it can be changed with `spark.conf.set` between queries. |
| `kill` | `false` | emergency switch, new plans are not rewritten. Env var `SPARK_RAPIDS_FUSION_KILL=1` additionally stops fused execution on running executors at the next batch. |
| `explain` | `NONE` | `NOT_FUSED` / `ALL`: driver log lines `[fusion] FUSED ...` / `[fusion] NOT FUSED ...: reason` (WARN level). |
| `shadow.every` | `1` | in shadow mode compare only every N-th batch (JVM-wide), for cheap validation on production data. |
| `shadow` | `false` | run fused **and** unfused for every batch, compare bit-for-bit (NaN-insensitive), return the unfused result, log `SHADOW MISMATCH` (ERROR), count it and blacklist the region. For validation on real data. |
| `minOps` | `2` | a region needs at least this many fused operators (otherwise not worth a kernel). |
| `force` | `false` | ignore `minOps` (testing). |
| `maxNodes` / `maxInputs` | 256 (1024 aggressive) / 48 | region size limits. |
| `allowOpaque` | `true` | an unsupported sub-expression (e.g. `length(str)`) is evaluated by cuDF and fed to the kernel as a column, instead of blocking the whole expression. |
| `project.enabled` / `filter.enabled` / `aggregate.enabled` | `true` | per operator kind. |
| `disabledExpressions` | empty | comma list of spark-rapids expression class names, e.g. `GpuCast,GpuPow`. Also `expression.<GpuName>.enabled=false`. |
| `syncAfterLaunch` | `true` | synchronize the stream after each kernel (safe default). |
| `asyncCompile` | `false` | experimental: run a new region unfused while NVRTC compiles it in the background (see `optimization-trials.md`; not recommended). |
| `nvrtcPath` | `$FUSION_NVRTC` | path of `libnvrtc.so.12` (loaded before JCuda). |
| `cacheDir` | `$TMPDIR/gpu-fusion-cubin-$USER` | on-disk cubin cache (key = source hash + arch + NVRTC version), `""` disables. |

## 3. What you see

* Plan: `FusedGpuProjectExec [...] fused 11 ops -> 1 kernel fused_xxxx; 3 kernel outputs, 4 inputs, ...`,
  `FusedGpuFilterExec`, and inside aggregates `fused[expr_xxxx](...)` expressions.
* Spark UI SQL metrics on fused operators: `fused batches`, `fallback (unfused) batches`, `fused kernel time`.
* In-JVM (local mode / tests): `com.nvidia.spark.rapids.fusion.FusionStats` (counters) and `FusionReport.dump()`
  (one line per planning decision with the reason when not fused).
* Logs: `fusion: ... failed for region X; blacklisting it and using the unfused path` when a kernel fails.

## 4. How it works

```
 spark-rapids plan (GpuProject/GpuFilter/GpuHashAggregate)
   -> FusionRule (ColumnarRule.postColumnarTransitions)
        Translator   Gpu* expression -> IR (whitelist by class name; ANSI/failOnError variants rejected;
                     unsupported subtrees become opaque cuDF-evaluated inputs)
        CodeGen/Ops  IR -> CUDA C++ (one device-function definition per operator in Prelude/Ops/MathOps)
        FusedGpuProjectExec / FusedGpuFilterExec / FusedGpuExpression
   -> executors: NativeBridge (JCuda NVRTC + cuLaunchKernel on the per-thread default stream cuDF uses)
        inputs : cuDF device buffers (data + validity bitmask) read in place
        outputs: RMM allocated DeviceMemoryBuffer -> cuDF ColumnVector -> GpuColumnVector
        retry  : spark-rapids split-and-retry (withRetry / runStreamingWithSplitRetry), original tiered project is the fallback
```

Semantics are mirrored from the unfused spark-rapids implementation (which is the reference), so fused results are
bit-identical to unfused GPU results: three-valued AND/OR, divisor 0 -> NULL, NaN is greatest and equals NaN, `-x = 0 - x`,
integer wrap-around, `x div -1`, shifts masked, `log(x<=0)` NULL, ...

**FMA:** kernels are compiled with `--fmad=true` but all floating point `+ - *` use the `__fadd_rn/__dmul_rn/...`
intrinsics, which are never contracted. `--fmad=false` would break the bitwise match of libdevice functions (e.g. `pow`)
with cuDF, whose own device code is compiled with the default.

## 5. Supported expressions (fused)

arithmetic `+ - * / div % pmod`, unary minus/plus/abs, comparisons incl. `<=>`, `AND OR NOT`, `IS [NOT] NULL`, `isnan`,
`IF`, `CASE WHEN`, `COALESCE`/`nvl`/`ifnull`/`nullif`/`nvl2`, `greatest`/`least`, `nanvl`, `IN (int literals <= 64)`,
casts between boolean/byte/short/int/long/float/double (and date<->timestamp, timestamp->integral/double, int->timestamp; UTC only),
bit ops (`& | ^ ~`, shifts, `bit_count`), `sqrt exp expm1 log log(b,x) pow hypot sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh cot cbrt rint
signum degrees radians`, `round/bround` (float/double, literal scale), `floor/ceil` (double->long), `year month day dayofmonth quarter dayofyear dayofweek weekday last_day
date_add date_sub datediff hour minute second` (UTC), `NormalizeNaNAndZero`.
Types: boolean, byte, short, int, long, float, double, date, timestamp. Not fused (run as before): strings, decimals, nested types,
ANSI mode, decimal arithmetic (incl. Spark turning `int * 1.5` into DECIMAL math), non-UTC time zones, non-deterministic expressions, expressions marked as having side effects. See `LIMITS-plugin.md`.

## 6. Things to know

* **First fused batch in every executor JVM costs ~0.7-1.5 s** (JCuda natives, CUDA context, NVRTC start; each distinct region
  then compiles in ~20 ms and is cached in memory and on disk).
* Results are bit-identical to *unfused spark-rapids* (the reference), not necessarily to CPU Spark: spark-rapids itself differs from
  the CPU for `-x` of 0.0 (GPU gives +0.0), `least/greatest` zero-sign ties, transcendental functions (1 ulp), out-of-range dates.
* GPU `sum` of doubles is not deterministic across runs (parallel reduction order); compare aggregates with a tolerance.
* `spark.rapids.sql.fusion.shadow=true` re-runs every region unfused and compares: use it for a validation run on real data.
* Date parts of dates beyond |days| > 2.146e9 and the like trip a *domain guard*: that batch is silently re-run unfused.
* Spark turns `int_col * 1.5` and `long_col / 10.0` into DECIMAL arithmetic (decimal literal). Decimal operators are not fused; write
  `1.5D` / `10.0D` (or cast the column to double) to keep such expressions fusable.
* Tested Spark 3.5.9 (default), 3.5.1, 3.4.3 with spark-rapids 26.08.1 `cuda12`; GPUs sm_75/86/89/90/120. Other spark-rapids versions are untested
  (the plugin links against its internals; a linkage problem makes the planner skip fusion with a warning, but executor-side
  mismatches could fail tasks).

## 7. Tests

`scripts/smoke/Battery.scala` (spark-shell script): ~190 expression templates over random data with NULLs and special
values, fused vs unfused GPU vs CPU Spark. `tests/` holds the larger differential/fuzz framework.
