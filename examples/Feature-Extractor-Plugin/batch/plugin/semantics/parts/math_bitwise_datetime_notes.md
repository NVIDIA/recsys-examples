# Pitfalls found (math / bitwise / datetime)

Source: spark-rapids org/apache/spark/sql/rapids/{mathExpressions,bitwise,datetimeExpressions}.scala and GpuMathAndDateExpression{Overrides,RuleMetas}.scala. Semantic claims about Spark CPU come from my knowledge of Spark 3.5, not re-verified against Spark source; test them in the differential suite.

1. Transcendental math (sin/cos/tan/exp/log/pow/cbrt/hypot/sinh...): CPU uses StrictMath (fdlibm), GPU uses CUDA libm. Not bit-identical. Fused vs unfused-GPU identity also depends on NVRTC flags (--fmad=false may change libm inline code, and cuDF itself is built with default fmad). Put all in v2 "approx" with ulp-tolerant comparison; v1 only exact ops (sqrt, rint, floor/ceil, signum, mul by constant, bit ops, integer calendar math).
2. spark-rapids GPU emulations that are NOT faithful to Spark (known CPU/GPU diffs, put in KNOWN-DIFFS): expm1 = exp(x)-1; log1p = log(x+1); log10 = log(x)/log(10) (log10(1000) may be 2.9999999999999996); log2 = log(x)/log(2) (Spark's own log2 is also log(x)/log(2)); hypot emulation; asinh/acosh compat-vs-improved depends on spark.rapids.sql.improvedFloatOps.enabled.
3. log family returns NULL (not -Inf/NaN) for x<=0 (Log1p: x<=-1). Output becomes nullable; the fused kernel must clear the validity bit. log(NaN)=NaN.
4. Domain errors (sqrt(-1), asin(2), acosh(0.5)) give NaN, never NULL and never an exception, in both ANSI modes.
5. GpuFloor/GpuCeil declare hasSideEffects=true (decimal overflow path) even for double/long inputs. GpuIf/CaseWhen/Coalesce then evaluate such branches differently. The fusion legality check must decide purity per input type, not per class flag. double->long: NaN->0, saturating (Java cast). Needs edge tests (1e19, -1e19, +-Inf, NaN).
6. Float->double implicit casts: all math functions take double; float inputs are widened exactly (no rounding), int64 widening rounds RN.
7. Signum/Rint/NaNvl preserve -0.0 and NaN payload selection; compare after NaN normalization only.
8. Shifts: Java masking (n&31 / n&63), byte/short operands are widened to int (output int). In C, signed left shift and shifts >= width are UB: use unsigned types and the mask.
9. bit_count on signed ints sign-extends to 64 bits (bit_count(-1::int)=64). Verify against CPU in tests.
10. Bitwise ops on byte/short: C integer promotion yields int, cast the result back to the narrow type.
11. Date functions are pure integer calendar arithmetic on int32 days (proleptic Gregorian); use int64 intermediates; test Int.Min/Max days. date_add/date_sub/datediff wrap on int32 overflow (ANSI mode in newer Spark may throw -> exclude ANSI).
12. Hour/Minute/Second/Cast(timestamp->date)/Year(timestamp) depend on session time zone: v1 only when the zone is UTC (GpuOverrides.isUTCTimezone); negative micros need floor division, not C `/`.
13. unix_timestamp(timestamp) uses truncating Long division (toward zero) per GPU code; timestamp_seconds/millis throw on overflow regardless of ANSI -> not fusable.
14. round/bround: Spark rounds the decimal representation (BigDecimal of Double.toString) HALF_UP/HALF_EVEN; cuDF differs for doubles in edge cases; ANSI throws on integral overflow. Not v1.
15. Not covered here (belong to arithmetic/predicates/cast notes): Abs, UnaryMinus, Pmod, Add/Sub/Mul/Div, Cast. Remember GpuAbs/UnaryMinus/Add/Sub have hasSideEffects when failOnError (ANSI) && integral.
