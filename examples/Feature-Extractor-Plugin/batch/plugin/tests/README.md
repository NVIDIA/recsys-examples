# tests/ differential and fuzz tests

One command: `tests/run-all.sh [-c random-case-count] [-s seed] [-p partition] [-o output-dir] [-n skip-fused]` (1 GPU, about 20-30 minutes; with `-p` it requests resources via srun).

| File | Purpose |
|---|---|
| `fz/gen_data.py` | Generates a deterministic 1M-row test table: about 12% NULL, about 45% extreme values (MIN/MAX/0/-1/NaN/+-Inf/-0.0/subnormals...), covering bool/byte/short/int/long/float/double/date/timestamp |
| `fz/exprgen.py` | Typed random expression-tree generator (arithmetic/comparison/logic/CASE/COALESCE/greatest/least/Cast/bitwise/math/date-time); renders to both SQL and the DataFrame API |
| `fz/directed.py` | Directed grid: every operator x the full Cartesian product of special values, compared bit-for-bit three ways (cpu / gpu unfused / fused) |
| `fz/fuzz.py` `fz/fuzz_loop.sh` | Random differential testing; failures are auto-minimized and saved to `OUT/fuzz/failures/*.json`; hangs are detected automatically and the run resumes |
| `fz/diff.py` | Bitwise comparison (integers/booleans/dates must match exactly; floats bitwise with NaN normalized, -0.0 distinguished from 0.0) |
| `fz/report.py` | Summary table; exits with code 1 when fused != unfused GPU |
| `udf/` | Coverage experiments for Scala/Java/RapidsUDF/Python/typed Dataset styles (`udf/build.sh` compiles with the scalac bundled with Spark, no network needed) |
| `gpu-run.sh` | GPU run wrapper: uses srun if `SLURM_PARTITION` is set (or `-p` is given), otherwise runs locally (on a cluster, scripts and data must be on a shared filesystem, not /tmp) |

Comparison pairs: `gpu-vs-cpu` (candidate known differences, see `semantics/KNOWN-DIFFS.md`), `FUSION-vs-unfused` (**must be 0**), `fused-vs-cpu` (reference).
Pitfalls (already handled): Parquet dictionary encoding merges 0.0/-0.0 (disabled); the plugin jar must go in `--jars`/`spark.jars` only, not also in `--driver-class-path`.
