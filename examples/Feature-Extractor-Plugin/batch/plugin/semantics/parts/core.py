#!/usr/bin/env python3
"""Generates core.json: arithmetic / predicates / conditional / null-handling contracts."""
import json
import os

NUM = ["byte", "short", "int", "long", "float", "double"]
INTS = ["byte", "short", "int", "long"]
FP = ["float", "double"]
E = []


def add(
    spark,
    gpu,
    cat,
    sql,
    types,
    null,
    non_ansi,
    ansi,
    notes,
    wl,
    why,
    dev,
    det=True,
    rowwise=True,
    src="",
):
    E.append(
        dict(
            spark_class=spark,
            gpu_class=gpu,
            category=cat,
            sql=sql,
            deterministic=det,
            rowwise=rowwise,
            types=types,
            null_semantics=null,
            non_ansi_behavior=non_ansi,
            ansi_behavior=ansi,
            rapids_notes=notes,
            whitelist=wl,
            whitelist_reason=why,
            device_function_hint=dev,
            source=src,
        )
    )


AR = "sql-plugin/.../org/apache/spark/sql/rapids/arithmetic.scala"
PR = "sql-plugin/.../org/apache/spark/sql/rapids/predicates.scala"
CO = "sql-plugin/.../com/nvidia/spark/rapids/conditionalExpressions.scala"
NU = "sql-plugin/.../com/nvidia/spark/rapids/nullExpressions.scala"

add(
    "Literal",
    "GpuLiteral",
    "leaf",
    "1, 2.5D, NULL",
    {
        "in": [],
        "out": "bool,byte,short,int,long,float,double,date,timestamp (string/decimal excluded in v1)",
    },
    "typed NULL literal -> all-null column; GPU evaluates literals as GpuScalar (not a column)",
    "none",
    "none",
    "spark-rapids keeps scalars unexpanded and uses scalar-specialised paths (e.g. GpuGreaterThanOrEqual lhs NaN scalar); results equal the column path.",
    "v1",
    "needed by every expression tree",
    "Emit as a compile-time constant. Floats must be emitted with exact bit pattern (use hex float or __longlong_as_double(0x...)) incl. NaN/Inf/-0.0; never print with %f. Typed NULL -> constant valid=false.",
    src="literals.scala",
)
add(
    "AttributeReference/BoundReference",
    "GpuBoundReference",
    "leaf",
    "col_a",
    {"in": [], "out": "any fixed-width column type"},
    "validity bit from input cudf bitmask (bit i of word i/32; null_count may be 0 and mask pointer may be NULL => all valid)",
    "none",
    "none",
    "cudf columns: data buffer + optional validity bitmask; offset/slices (column view offset) must be honoured; mask words are bitmask_type=uint32.",
    "v1",
    "needed",
    "Read data[i+offset]; valid = mask==nullptr ? true : (mask[(i+offset)/32] >> ((i+offset)%32)) & 1. BOOL8 is 1 byte per value (cudf), not bit-packed.",
    src="GpuBoundAttribute.scala",
)
add(
    "Alias",
    "GpuAlias",
    "leaf-wrapper",
    "a AS b",
    {"in": ["any"], "out": "same"},
    "transparent",
    "none",
    "none",
    "Pure rename; stripped when binding.",
    "v1",
    "transparent",
    "identity",
)

# ---- arithmetic
add(
    "UnaryMinus",
    "GpuUnaryMinus",
    "arithmetic",
    "-a",
    {"in": NUM, "out": "same"},
    "NULL propagates",
    "int wraps: -MIN_VALUE = MIN_VALUE; float: flips sign bit (-0.0 <-> 0.0, -NaN stays NaN)",
    "int types: throws ArithmeticException on MIN_VALUE (hasSideEffects=true when failOnError)",
    "GPU computes 0 - x via Scalar.sub for all types (incl. float!): 0 - 0.0 = +0.0 but Spark/Java -(0.0) = -0.0. Check: GPU result for float -x on x=+0.0 is +0.0 (0.0-0.0) vs CPU -0.0 => known diff candidate; fused kernel should replicate whichever the unfused GPU path does only if tests show it, else exclude float UnaryMinus or define as 0-x (v1 decision: bit-match the CPU: -x via sign flip; document as diff vs unfused GPU if confirmed).",
    "v1",
    "simple; ANSI form (failOnError) excluded",
    "int: (T)(0 - (U)a) (use unsigned arithmetic to avoid UB); float: -a (sign flip).",
    src=AR,
)
add(
    "UnaryPositive",
    "GpuUnaryPositive",
    "arithmetic",
    "+a",
    {"in": NUM, "out": "same"},
    "propagates",
    "none",
    "none",
    "identity (incRefCount)",
    "v1",
    "trivial",
    "identity",
)
add(
    "Abs",
    "GpuAbs",
    "arithmetic",
    "abs(a)",
    {"in": NUM, "out": "same"},
    "propagates",
    "int: abs(MIN_VALUE)=MIN_VALUE (wraps); float: clears sign bit (abs(-0.0)=0.0, abs(NaN)=NaN)",
    "int: throws on MIN_VALUE",
    "cuDF UnaryOp.ABS. Non-ANSI wraps.",
    "v1",
    "simple",
    "int: a<0 ? (T)(0-(U)a) : a ; float: fabs (bit clear sign).",
    src=AR,
)


def addsub(nm, g, op, sql):
    add(
        nm,
        g,
        "arithmetic",
        sql,
        {"in": NUM, "out": "same (after Spark type coercion both sides same type)"},
        "NULL if either NULL",
        "int/long/short/byte: two's-complement wraparound (byte/short: result truncated to type). float/double: IEEE (Inf, NaN, -0.0: (-0.0)+(-0.0)=-0.0, (-0.0)+0.0=0.0 in round-to-nearest). No FMA contraction allowed (--fmad=false).",
        "int types: throws ArithmeticException on overflow (hasSideEffects). float: same as non-ANSI. Interval/decimal variants separate.",
        "cuDF BinaryOp %s; GPU only checks overflow if failOnError. Decimal has separate overflow->NULL logic (excluded v1)."
        % op,
        "v1",
        "simple; ANSI(failOnError=true) integral instances are NOT fusible (side effects)",
        "int: (T)((U)a %s (U)b) using unsigned arithmetic for defined wrap; float: a %s b."
        % (sql.split()[1], sql.split()[1]),
        src=AR,
    )


addsub("Add", "GpuAdd", "ADD", "a + b")
addsub("Subtract", "GpuSubtract", "SUB", "a - b")
add(
    "Multiply",
    "GpuMultiply",
    "arithmetic",
    "a * b",
    {"in": NUM, "out": "same"},
    "NULL if either NULL",
    "int wraps (low bits), byte/short truncated; float IEEE; no FMA with following add (--fmad=false and no __fmaf contraction)",
    "int: throws on overflow",
    "Uses spark-rapids-jni Arithmetic.multiply(lhs,rhs,ansi,tryMode); non-ANSI wraps like CPU. Decimal: separate GpuDecimalMultiply (excluded).",
    "v1",
    "simple",
    "int: (T)((U)a * (U)b) in unsigned 32/64-bit; for byte/short compute in int then truncate; float: a*b",
    src=AR,
)
add(
    "Divide",
    "GpuDivide",
    "arithmetic",
    "a / b",
    {
        "in": [
            "double (Spark coerces int/long/float operands to double; float/float is NOT coerced? Divide requires DoubleType|DecimalType: float inputs are cast to double)"
        ],
        "out": "double",
    },
    "NULL if either NULL; ALSO NULL if divisor == 0 (including 0.0 and -0.0) when ANSI off",
    "x/0 => NULL (not Inf/NaN) for double as well. NaN/y = NaN; y/NaN = NaN; Inf/Inf=NaN. If b != 0: IEEE division.",
    "b==0 throws DIVIDE_BY_ZERO (hasSideEffects). Excluded from v1.",
    "GpuDivModLike: non-ANSI replaces zeros in rhs with NULL via findAndReplaceAll (0.0 == -0.0 both replaced) then TRUE_DIV. scalar zero rhs => all-NULL column. nullable=true.",
    "v1",
    "non-ANSI semantic is branch-free: valid = valid_a & valid_b & (b != 0)",
    "if (b == 0) {valid=false} else r = a / b. Output validity must AND (b!=0) even when result is garbage. Inputs already cast to double by planner.",
    src=AR,
)
add(
    "IntegralDivide",
    "GpuIntegralDivide",
    "arithmetic",
    "a div b",
    {"in": ["byte", "short", "int", "long"], "out": "long"},
    "NULL if either NULL, NULL if b==0 (ANSI off)",
    "truncation toward zero (C semantics). Long.MIN div -1: CPU Java gives Long.MIN (wraps) in non-ANSI. NOTE GPU passes it to cuDF DIV (C++ UB) -> unfused GPU may differ (known-diff candidate; verify)",
    "b==0 throws; Long.MIN div -1 throws ArithmeticException(overflow) (shouldFailDivOverflow = ansiEnabled)",
    "GpuDivModLike.checkDivideOverflow only when LongType && ansi. Result type always Long (int operands are widened first in cuDF with outputTypeOverride=INT64).",
    "v1",
    "need explicit INT_MIN/-1 special case in device fn",
    "if (b==0) null; else if (b==-1) r = (long)(0-(ulong)a); else r = (long)a / (long)b (operands widened to long first).",
    src=AR,
)
add(
    "Remainder",
    "GpuRemainder",
    "arithmetic",
    "a % b",
    {"in": NUM, "out": "same"},
    "NULL if either NULL; NULL if b==0 (ints AND floats) when ANSI off",
    "int: result has sign of dividend (C/Java %); MIN % -1 = 0 (Java) - cuDF MOD for b==-1 is C++ UB territory => special-case. float: Java % == C fmod: x%Inf=x, Inf%x=NaN, x%NaN=NaN, result keeps sign of dividend incl. -0.0 ((-4.0) % 2.0 = -0.0).",
    "b==0 throws (int) ; for float ANSI also throws on 0.",
    "Shares GpuDivModLike zero->NULL logic. cuDF BinaryOp.MOD: for floats uses fmod/fmodf (matches Java) but verify -0.0 outputs in differential test.",
    "v1",
    "need b==-1 special case for ints; floats via fmod (note: CUDA fmod is exact so equal to Java)",
    "if (b==0) null; ints: b==-1 ? 0 : a % b ; float: fmod(a,b).",
    src=AR,
)
add(
    "Pmod",
    "GpuPmod",
    "arithmetic",
    "pmod(a, b)",
    {"in": NUM, "out": "same as left"},
    "NULL if either NULL, NULL if b==0",
    "Spark: r = a % b; if (r < 0) (r + b) % b else r. Note for negative b the result can still be negative ((-7) pmod (-3) -> r=-1 -> (-1+-3)%-3=-1). Float NaN: r<0 false -> NaN stays NaN. int overflow in r+b wraps.",
    "b==0 throws",
    "cuDF BinaryOp.PMOD; semantic equivalence with Spark formula for negative divisor and float edge cases is a differential-test target.",
    "v1",
    "same machinery as Remainder; implement literal Spark formula (not cuDF's)",
    "r = rem(a,b) [as Remainder]; if (r < 0) r = rem(r + b, b); return r;  (r+b computed wrapping)",
    src=AR,
)
add(
    "Greatest",
    "GpuGreatest",
    "arithmetic",
    "greatest(a,b,...)",
    {"in": NUM + ["date", "timestamp", "boolean"], "out": "same; >=2 args"},
    "NULLs are SKIPPED; result NULL only if all args NULL",
    "NaN is greater than any other value (Greatest returns NaN if any non-null arg is NaN). -0.0 vs 0.0 tie: Spark keeps the earlier (first) arg because comparison is a>b strictly with -0.0 == 0.0 => result sign depends on arg order. cuDF binaryOp NULL_MAX on ties unspecified => differential-test target.",
    "same",
    "GpuGreatestLeastBase folds pairwise cuDF binaryOp (NULL_MAX/NULL_MIN) then fixes NaN with isNan masks (shouldNanWin=true for Greatest, false for Least).",
    "v1",
    "needs tie/-0.0 verification; keep behind test",
    "acc=first non-null; for next valid x: if (!acc_valid || gt(x,acc)) acc=x, where gt(x,y) = isnan(x) ? !isnan(y) : (!isnan(y) && x>y). Least: lt(x,y) = isnan(y) ? !isnan(x) : (!isnan(x) && x<y).",
    src=AR,
)
add(
    "Least",
    "GpuLeast",
    "arithmetic",
    "least(a,b,...)",
    {"in": NUM + ["date", "timestamp", "boolean"], "out": "same"},
    "NULLs skipped; NULL only if all NULL",
    "NaN is the LARGEST value so Least ignores NaN unless all non-null args are NaN; ties keep earlier arg.",
    "same",
    "see Greatest",
    "v1",
    "see Greatest",
    "see Greatest",
    src=AR,
)


# ---- comparisons
def cmp(nm, g, sql, spark_rule, rapids, dev):
    add(
        nm,
        g,
        "comparison",
        sql,
        {"in": ["boolean"] + INTS + FP + ["date", "timestamp"], "out": "boolean"},
        "NULL if either NULL (except <=>)",
        spark_rule,
        "same",
        rapids,
        "v1",
        "simple; Spark NaN ordering must be replicated",
        dev,
        src=PR,
    )


nanrule = "Spark total order for floating point: NaN == NaN is TRUE, NaN > every non-NaN (incl. +Inf), -0.0 == 0.0 is TRUE. (SQLOrderingUtil.compareDoubles). IEEE would give NaN==NaN false."
cmp(
    "EqualTo",
    "GpuEqualTo",
    "a = b",
    nanrule,
    "cuDF EQUAL then ORs (isNan(l) AND isNan(r)). Null rows: result NULL.",
    "float: (a==b) || (isnan(a)&&isnan(b)); else a==b",
)
cmp(
    "GreaterThan",
    "GpuGreaterThan",
    "a > b",
    nanrule,
    "cuDF GREATER OR (isNan(l) AND isNotNan(r)).",
    "float: isnan(a) ? !isnan(b) : (a > b)   [a>b false if b NaN]",
)
cmp(
    "GreaterThanOrEqual",
    "GpuGreaterThanOrEqual",
    "a >= b",
    nanrule,
    "cuDF GREATER_EQUAL OR isNan(l).",
    "float: isnan(a) || (a >= b)",
)
cmp(
    "LessThan",
    "GpuLessThan",
    "a < b",
    nanrule,
    "cuDF LESS OR (isNotNan(l) AND isNan(r)).",
    "float: isnan(b) ? !isnan(a) : (a < b)",
)
cmp(
    "LessThanOrEqual",
    "GpuLessThanOrEqual",
    "a <= b",
    nanrule,
    "cuDF LESS_EQUAL OR isNan(r).",
    "float: isnan(b) || (a <= b)",
)
add(
    "EqualNullSafe",
    "GpuEqualNullSafe",
    "comparison",
    "a <=> b",
    {
        "in": ["boolean"] + INTS + FP + ["date", "timestamp"],
        "out": "boolean (never NULL)",
    },
    "both NULL -> true; exactly one NULL -> false; else as EqualTo (NaN==NaN true)",
    "as EqualTo",
    "same",
    "cuDF NULL_EQUALS + NaN fix. nullable=false.",
    "v1",
    "simple",
    "valid=true; r = (va&&vb) ? eq(a,b) : (!va && !vb)",
    src=PR,
)
add(
    "And",
    "GpuAnd",
    "logical",
    "a AND b",
    {"in": ["boolean"], "out": "boolean"},
    "3-valued: F AND NULL = F; NULL AND F = F; T AND NULL = NULL; NULL AND NULL = NULL",
    "Spark CPU short-circuits (right not evaluated if left is false) - only matters for errors/side effects",
    "same",
    "cuDF NULL_LOGICAL_AND. If right child hasSideEffects (ANSI errors) GPU evaluates right only on rows where left is not false (filter/gather). v1 forbids side-effecting children so plain both-sides evaluation is equivalent.",
    "v1",
    "simple (children must be side-effect free)",
    "if ((va && !a) || (vb && !b)) {r=false; valid=true} else if (va && vb) {r=true; valid=true} else valid=false",
    src=PR,
)
add(
    "Or",
    "GpuOr",
    "logical",
    "a OR b",
    {"in": ["boolean"], "out": "boolean"},
    "T OR NULL = T; NULL OR T = T; F OR NULL = NULL",
    "short-circuit on CPU",
    "same",
    "cuDF NULL_LOGICAL_OR; side-effect branch handling as And",
    "v1",
    "simple",
    "if ((va && a) || (vb && b)) {r=true; valid=true} else if (va && vb) {r=false; valid=true} else valid=false",
    src=PR,
)
add(
    "Not",
    "GpuNot",
    "logical",
    "NOT a",
    {"in": ["boolean"], "out": "boolean"},
    "NULL -> NULL",
    "",
    "",
    "cuDF UnaryOp.NOT",
    "v1",
    "simple",
    "valid=va; r=!a",
    src=PR,
)
add(
    "In",
    "GpuIn / GpuInSet",
    "predicate",
    "a IN (1,2,3)",
    {
        "in": [
            "bool,int types,float,double,date,timestamp (literal lists; strings excluded v1)"
        ],
        "out": "boolean",
    },
    "a NULL -> NULL; a found -> TRUE; not found and list contains NULL -> NULL; not found -> FALSE",
    "float: NaN IN (NaN) is TRUE (NaN equality); -0.0 matches 0.0",
    "same",
    "GpuInSet uses cudf contains with NaN handling; In with non-literal list (<=256) uses AST-fused equality+OR. Large literal lists (OptimizeIn -> InSet) use hash set semantic - cap list length in v1 (e.g. <=16 unrolled compares).",
    "v1",
    "expand to OR chain of EqualTo; cap list size",
    "r=false; any_null=!va... : if(!va) null; else {found = OR(eq(a, lit_i)); if (found) true else if (list_has_null) null else false}",
    src="GpuIn.scala",
)

# ---- conditional
add(
    "If",
    "GpuIf",
    "conditional",
    "IF(p, t, f)",
    {"in": ["boolean", "any fixed-width t/f of same type"], "out": "type of t/f"},
    "predicate NULL -> take FALSE branch; result is validity of the chosen branch",
    "Only the chosen branch is observable. If either branch hasSideEffects (ANSI errors) GPU filters rows per branch to avoid spurious errors.",
    "same",
    "GpuIf: no side effects => evaluates BOTH branches on all rows then cuDF ifElse (pred null treated as false). With side effects: isAllTrue/isAllFalse shortcuts + filter/gather (conditionalExpressions.scala:177+).",
    "v1",
    "branch-per-row in kernel is strictly safer than GPU unfused evaluation; v1 requires both branches side-effect-free so result identical",
    "valid_p && p ? eval(t) : eval(f)  (evaluate ONLY chosen branch in kernel, branch divergence acceptable).",
    src=CO,
)
add(
    "CaseWhen",
    "GpuCaseWhen",
    "conditional",
    "CASE WHEN c1 THEN v1 ... ELSE e END",
    {"in": ["boolean conds", "same-typed values"], "out": "value type"},
    "first branch whose condition is TRUE (NULL cond = not taken) wins; no ELSE -> NULL",
    "short-circuit by order; side-effect branches filtered on GPU",
    "same",
    "GpuCaseWhen has an optional 'fuse' path (caseWhenFuseEnabled, >2 branches, all-literal values) and a side-effect-aware path. Output nullable if any branch nullable or no else.",
    "v1",
    "lowered to chain of Ifs",
    "for each i: if (valid_ci && c_i) return v_i; return else or NULL.",
    src=CO,
)
add(
    "Coalesce",
    "GpuCoalesce",
    "null-handling",
    "coalesce(a,b,...)",
    {"in": ["any same-typed"], "out": "same"},
    "first non-NULL argument; NULL if all NULL",
    "CPU stops at first non-null; GPU evaluates all children (GpuNvl folded right-to-left) - difference only visible with side effects (excluded)",
    "same",
    "GpuNvl: replaceNulls fold.",
    "v1",
    "simple",
    "for each child in order: if valid -> return; return NULL",
    src=NU,
)
add(
    "IsNull",
    "GpuIsNull",
    "null-handling",
    "a IS NULL",
    {"in": ["any"], "out": "boolean (non-null)"},
    "never NULL",
    "",
    "",
    "",
    "v1",
    "simple",
    "valid=true; r=!va",
    src=NU,
)
add(
    "IsNotNull",
    "GpuIsNotNull",
    "null-handling",
    "a IS NOT NULL",
    {"in": ["any"], "out": "boolean (non-null)"},
    "never NULL",
    "",
    "",
    "",
    "v1",
    "simple",
    "valid=true; r=va",
    src=NU,
)
add(
    "IsNaN",
    "GpuIsNan",
    "null-handling",
    "isnan(a)",
    {"in": ["float", "double"], "out": "boolean (non-null)"},
    "NULL input -> FALSE (never NULL)",
    "",
    "",
    "cuDF isNan; nulls produce false in Spark (IsNaN null-> false).",
    "v1",
    "simple",
    "valid=true; r=va && isnan(a)",
    src=NU,
)
add(
    "NaNvl",
    "GpuNaNvl",
    "null-handling",
    "nanvl(a,b)",
    {"in": ["float", "double"], "out": "same"},
    "a NULL -> NULL(!) (Spark: if a null -> null; if a NaN -> b else a); b only used when a is NaN",
    "",
    "",
    "cuDF: isNotNan(a).ifElse(a,b). NOTE cuDF isNotNan(null) semantic: verify (null a must give NULL result).",
    "v1",
    "simple (verify null)",
    "if(!va) null; else if (isnan(a)) b(with validity) else a",
    src=NU,
)
add(
    "AtLeastNNonNulls",
    "GpuAtLeastNNonNulls",
    "null-handling",
    "DataFrame.na.drop(minNonNulls)",
    {"in": ["any"], "out": "boolean"},
    "non-null & non-NaN counted",
    "NaN counted as null for float/double",
    "",
    "sum of casts",
    "v2",
    "used by na.drop only; low priority",
    "count(valid && !isnan) >= n",
    src=NU,
)
add(
    "NormalizeNaNAndZero",
    "GpuNormalizeNaNAndZero",
    "float",
    "(internal) normalize NaN/-0.0",
    {"in": ["float", "double"], "out": "same"},
    "propagates",
    "-0.0 -> 0.0; any NaN -> canonical NaN",
    "",
    "Inserted by Spark for join/group keys.",
    "v1",
    "cheap",
    "r = isnan(a) ? CANONICAL_NAN : (a == 0 ? 0.0 : a)",
    src="GpuMathAndDateExpressionRuleMetas.scala:667",
)
add(
    "KnownFloatingPointNormalized/KnownNotNull",
    "GpuKnown*",
    "tag",
    "(internal tag)",
    {"in": ["any"], "out": "same"},
    "identity (tag)",
    "",
    "",
    "ShimTaggingExpression: pure pass-through",
    "v1",
    "identity",
    "identity",
    src="constraintExpressions.scala",
)
add(
    "Nvl/IfNull/Nvl2/NullIf (RuntimeReplaceable)",
    "(rewritten to Coalesce/If/CaseWhen by Spark optimizer)",
    "null-handling",
    "nvl(a,b), nullif(a,b)",
    {"in": ["any"], "out": "same"},
    "see Coalesce/If/EqualTo",
    "",
    "",
    "Never appear in physical plan; covered by rewrite targets.",
    "v1",
    "covered",
    "n/a",
)

# ---- excluded
for nm, why in [
    (
        "Rand/Randn/Uuid/Shuffle",
        "non-deterministic, stateful per partition (GpuNondeterministic.initializeAll)",
    ),
    (
        "MonotonicallyIncreasingID/SparkPartitionID/InputFileName/InputFileBlockStart/Length",
        "depend on task/partition state; spark-rapids has special rules (InputFileBlockRule)",
    ),
    (
        "ScalaUDF / PythonUDF / RapidsUDF (opaque)",
        "black box; Scala UDFs only fusible after udf-compiler translated them to Catalyst expressions (done at logical plan level)",
    ),
    (
        "Decimal arithmetic (CheckOverflow, MakeDecimal, UnscaledValue, DecimalDivide/Multiply, PromotePrecision)",
        "128-bit math, overflow->NULL rules, precision/scale intermediate casts (see arithmetic.scala DecimalMultiplyChecks); v2",
    ),
    (
        "String / array / map / struct expressions",
        "variable length / nested layouts; out of scope v1",
    ),
    (
        "ANSI-mode arithmetic & casts (failOnError=true)",
        "hasSideEffects=true (throws). GPU reports error only after whole column checked; row index semantics differ. v2 via error flag",
    ),
    (
        "Hash functions (Murmur3Hash, XxHash64, Md5, Sha*)",
        "not-in-scope for v1; murmur3 on numeric would be a v2 candidate (needs identical seed/null handling)",
    ),
]:
    add(nm, "-", "excluded", "", {"in": [], "out": ""}, "", "", "", "", "no", why, "")
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
json.dump(
    E,
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.json"), "w"),
    indent=1,
    ensure_ascii=False,
)
print(len(E), "entries")
