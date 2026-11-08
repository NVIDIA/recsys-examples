# Cast fusion notes (numeric / boolean / date / timestamp)

Source: spark-rapids `GpuCast.scala` (doCast at :295, default-branch cuDF castTo at :641, hasSideEffects at :1666); Spark 3.5 semantics; see cast.json for details.

1. **Integer narrowing = two's-complement wraparound** (CPU and the GPU default branch agree). Under ANSI, out-of-range values throw and GpuCast.hasSideEffects=true, so v1 excludes ANSI.
2. **float/double -> int/long**: truncate toward zero, saturate, NaN->0. For (Float,Long)(Double,Int|Long) the GPU first sets NaN to 0 (:544); the other cases rely on CUDA conversion saturation. The device function must write the saturation explicitly (note that the float value of 2147483647 is 2^31, so the check must use `>=`).
3. **float/double -> byte/short**: the CPU first converts to int and then wraps (300.0f -> 44); the GPU goes through cuDF static_cast, which most likely saturates, so the two may differ. Put it in v2, confirm with a differential test first, and record it in KNOWN-DIFFS. The fusion baseline is the unfused GPU result.
4. **int/long -> float/double** and **double -> float** must be round-to-nearest-even; use `__ll2float_rn` and `__double2float_rn`, no fast-math, `--fmad=false`.
5. **Any numeric -> bool** is `x != 0` (NaN is true, -0.0 is false).
6. **timestamp -> long** is floorDiv(micros, 1e6), rounding toward negative infinity for negative values (not C's `/`). **timestamp -> int/short/byte** floors first, then wraps; Spark 4.0 / db143 changed this to yield NULL on overflow (CastTimeToIntShim), so it branches by Spark version. **timestamp -> double** is a double division of micros/1e6.
7. **long -> timestamp**: multiply by 1e6; on overflow it saturates to Long.Min/Max (not wraparound), with the lower bound checked against -(Long.Max/1e6) (:1611). byte/short/int -> timestamp cannot overflow.
8. **float/double -> timestamp**: NaN/Inf -> NULL, and the Spark 3.3+ shim marks hasSideEffects=true, so output nullability changes. Not fused in v1.
9. **timestamp <-> date**: simple division (floorDiv 86400e6) works only with a UTC session time zone; non-UTC needs a time zone table (GpuTimeZoneDB), so v1 rejects non-UTC.
10. **date -> numeric/bool**: GpuCast returns all NULL directly (:314); not fused.
11. Nulls: Cast is NullIntolerant, so NULL input gives NULL output; but float->timestamp above, Spark 4 timestamp->narrow integer, etc. "produce NULL from non-NULL input", so the fused output validity bits must be kept in sync with `Cast.forceNullable`.
12. String, decimal, nested, interval: not in v1 (variable-length / 128-bit / overflow-to-NULL / config switches such as castFloatToDecimal differ).
