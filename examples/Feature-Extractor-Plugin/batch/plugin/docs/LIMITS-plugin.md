# Known limits of the fusion plugin (plugin side)

Honest list; "not done" means not implemented, "refused" means the planner deliberately leaves it unfused (the query still runs on spark-rapids as before).

## What is fused, where
* Operators: `GpuProjectExec`, `GpuFilterExec`, and the *expression trees* inside `GpuHashAggregateExec` (aggregate-function inputs, grouping keys).
  Not done: Sort order expressions, join keys / conditions, `Expand`, `Window`, `Generate`, writers. Adjacent projects are merged (vertical fusion);
  a filter is **not** merged with the project above it (they stay two kernels with a cuDF gather in between).
* Types: boolean, byte, short, int, long, float, double, date, timestamp. Refused: string, decimal, binary, interval, array/map/struct.
  A sub-expression of an unsupported type (e.g. `length(str)`, a decimal division) is evaluated by spark-rapids and fed to the kernel as a column
  ("cut"), as long as the sub-expression itself returns a supported type; an *output* of an unsupported type is simply not fused.
* Spark SQL, DataFrame and (translated) Scala UDFs behave identically (same physical plan). Java UDFs, Dataset typed lambdas (`ds.map`), RDD code, Python/pandas UDFs and
  `RapidsUDF` are black boxes: not translated, not fused (Dataset typed lambdas run on the CPU altogether).

## Semantics
* Reference = unfused spark-rapids on the GPU (bit-identical, tested). It is **not** always identical to CPU Spark: spark-rapids itself differs for `-x` of 0.0
  (GPU +0.0), `least/greatest` ties of 0.0/-0.0, 1-ulp differences of transcendental functions, out-of-range dates. The fused kernels copy the GPU behaviour on purpose.
* Refused: ANSI mode (`failOnError`/`ansiMode`), expressions flagged `hasSideEffects`, non-deterministic expressions (`rand`, `monotonically_increasing_id`, ...), non-UTC time zone for
  timestamp<->date casts and `hour/minute/second`, `IN` with more than 64 values or non-integral values, `round` with a non-literal scale, `round/bround` on integers.
* Not fused expressions (by name): everything string/regexp/json/url, decimal arithmetic and casts (`GpuDecimalMultiply/Divide`, casts to/from decimal or string),
  `timestamp_seconds/millis/micros`, `unix_timestamp`, `date_format`, `trunc`, `add_months`, `months_between`, `next_day`, `make_date`, `date_from_unix_date`, `factorial`, `sec/csc`
  and other expressions spark-rapids runs through its CPU bridge. Spark turns `int_col * 1.5` / `long_col / 10.0` into DECIMAL arithmetic; write `1.5D`/`10.0D` to avoid it.
* Domain guards: date parts of dates beyond +-2.146e9 days make that batch fall back to the unfused path (cuDF's behaviour there is an overflow artefact); the region is
  blacklisted after 3 such batches.
* Transcendental functions are bit-identical to cuDF only because both use the same CUDA libdevice math; a different NVRTC/libdevice version than the one cuDF was built with could differ in the last ulp (shadow mode detects it).
* The first fused batch of every executor JVM pays ~0.7-1.5 s (native libraries, CUDA context, NVRTC start); each new region compiles in ~20 ms.

## Memory / robustness
* Output memory comes from RMM and spark-rapids' split-and-retry covers it (tested with injected retry OOMs and splits). The `PreProjectSplitIterator` of the original project (splitting inputs whose
  projected output would be huge) is **not** reproduced: very wide/large outputs rely on retry splitting only.
* A kernel failure (compile error, launch error) blacklists the region for the JVM and the batch is recomputed unfused; sticky CUDA errors (illegal address) cannot be recovered by anyone.
* Blacklist and statistics are per JVM (executor); `FusionStats` is not aggregated to the driver (per-operator SQL metrics are).

## Platform / testing
* Needs `libnvrtc.so.12` (CUDA 12.x; 12.8+ for Blackwell) on every executor, Linux x86_64, JDK 8/11 (JCuda 12.0.0 natives are bundled in the jar).
* Tested: Spark 3.5.9 (main), 3.5.1, 3.4.3 with spark-rapids 26.08.1 cuda12 (cuDF 26.08.0, which exposes the column_view layout our offset check relies on); GPUs L20, H20,
  RTX 6000D (Blackwell), A10, Quadro RTX 8000 (Turing, T4-class). **Not tested**: T4 itself, RTX PRO 5000, Spark 4, Databricks, YARN/Kubernetes multi-node (a standalone master+worker
  cluster on one node was tested), spark-rapids versions other than 26.08.1, GPUs without a CUDA-12-capable driver.
* The plugin links against spark-rapids internals (`GpuProjectExecLike`, `runStreamingWithSplitRetry`, ...). A different spark-rapids version can fail to link: the planner then disables itself with a warning;
  executor-side mismatches could fail tasks.
* Shadow mode doubles the work of fused regions (use `shadow.every=N` for production validation).
