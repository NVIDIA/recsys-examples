# Known differences: spark-rapids (unfused) GPU vs CPU Spark

These differences exist with fusion **turned off**; they are not introduced by the fusion plugin. The plugin's hard requirement is "bit-identical to the unfused GPU", so fused kernels must **copy the GPU's behavior** (not the CPU's).
Environment: L20, spark-rapids 26.08.1 (cuDF 26.08.0), Spark 3.5.9, local mode, ANSI off, session time zone UTC.
Evidence: `tests/fz/directed.py` (518 templates x the full Cartesian product of special values; results in `tests/out/base/directed/directed.json`) and
`tests/fz/fuzz.py` (200 random expression trees x 1M rows, auto-minimized; `tests/out/base/fuzz/`). Comparison: integers/booleans/dates must match exactly; floats must match bit-for-bit (any NaN is normalized).

| # | Expression | CPU | GPU (unfused) | Trigger | What fusion should do |
|---|---|---|---|---|---|
| 1 | float/double unary minus `-a` | Flips the sign bit: `-(0.0) = -0.0` | `0 - a`: `-(0.0) = +0.0`, `-(-0.0) = +0.0` | a is +-0.0 | Follow the GPU: `0 - a`. (-0.0 flowing into 1/x gives NULL in Spark, so the impact is small, but a bitwise comparison will see it) |
| 2 | float/double `greatest/least` | On a tie, keeps the earlier argument | On a tie, takes the **later** argument (cuDF NULL_MAX/NULL_MIN: `x>y ? x : y`) | 0.0 and -0.0 compare equal | Follow the GPU: `acc=(acc>c)?acc:c` (`<` for least); NaN is handled separately: NaN wins in greatest, loses in least |
| 3 | `date_add(date, int)` (and downstream such as `dayofmonth(...)`) | Day count wraps around as an Int | Different result after overflow (with i=Int.MinValue, `dayofmonth` gives 13 vs 25) | Extreme day-count argument | Exclude it, or follow the GPU (v2); fusion v1 should not accept extreme-value semantics for date_add |
| 4 | `CAST(date_add(dt,i) AS TIMESTAMP)` | CPU throws an overflow exception (TASK_WRITE_FAILED) | Does not throw; returns a value | Date + extreme day count | Fusion must not change the fact that the GPU does not throw; this combination stays off the whitelist |
| 5 | `dayofyear(date)` (a reliability problem, not a numeric difference) | Works | The **second** run of the same query hangs (Java thread spins, GPU idle); reproducible in this environment | Appears in the same batch as other columns | Not in the v1 whitelist; excluded from random tests by default (enable with `FZ_HANG_PRONE=1`) |

No differences found (all directed tests pass): integer + - * wraparound, / and % and pmod by zero giving NULL (including float 0.0/-0.0 divisors), Long.MinValue div -1, integer % -1, NaN comparison (NaN=NaN is true, NaN is the largest),
`<=>`, three-valued AND/OR logic, IF/CASE WHEN/COALESCE, IN, isnan/nanvl, bitwise ops and shifts (including shift-count masking), bit_count, Casts between all numeric types (narrowing wraps, float->int saturates, NaN->0),
sqrt/floor/ceil/signum/rint/degrees/radians, year/month/day/quarter/dayofweek/weekday/last_day/datediff, hour/minute/second, timestamp<->date/long/double conversions.

## Mismatches between fused and unfused GPU (plugin bugs, all fixed)
| Expression | Symptom | Status |
|---|---|---|
| byte/short `pmod(a,b)` | a=-127, b=-128: unfused gives -127, fused gives 1 (`(r+n)%n` must be computed in int, not wrapped in byte) | **Fixed** (final jar: 518 directed templates + 300 random cases, 0 mismatches) |
| float/double `greatest/least` +-0.0 tie | Fused took the first argument (CPU behavior); unfused GPU takes the later one | **Fixed** |
