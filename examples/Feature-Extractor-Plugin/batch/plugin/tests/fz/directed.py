#!/usr/bin/env python3
"""Directed edge-value grid tests: every operator x full cross product of special values (NULL, MIN/MAX, 0, -0.0, NaN, +-Inf,
denormals, ...), compared bit-exactly across modes.  Complements the random fuzzer (systematic instead of random).

  python3 directed.py --out OUTDIR --modes cpu,gpu[,fused] [--chunk 40]
Result: OUTDIR/directed.json = {template: {type: {pair: n_mismatches, sample...}}} and a printed table.
"""
import argparse
import collections
import itertools
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff as D
from common import *
from pyspark.sql import types as T

I8 = [-128, -127, -1, 0, 1, 2, 7, 63, 64, 126, 127]
I16 = [-32768, -32767, -256, -1, 0, 1, 2, 31, 255, 32766, 32767]
I32 = [
    -2147483648,
    -2147483647,
    -65536,
    -33,
    -32,
    -2,
    -1,
    0,
    1,
    2,
    3,
    31,
    32,
    33,
    65535,
    2147483646,
    2147483647,
]
I64 = [
    -9223372036854775808,
    -9223372036854775807,
    -4294967296,
    -2147483649,
    -2147483648,
    -64,
    -1,
    0,
    1,
    2,
    63,
    64,
    2147483647,
    2147483648,
    9223372036854775806,
    9223372036854775807,
]
FP = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    1.5,
    2.5,
    -2.5,
    3.0,
    7.0,
    1e-3,
    1e10,
    float("nan"),
    float("inf"),
    float("-inf"),
    1.7976931348623157e308,
    -1.7976931348623157e308,
    4.9e-324,
    -4.9e-324,
    2.2250738585072014e-308,
    2147483647.0,
    2147483648.0,
    -2147483648.0,
    -2147483649.0,
    9.2233720368547758e18,
    -9.2233720368547758e18,
    1.8446744073709552e19,
    255.5,
    300.7,
    -300.7,
    32768.5,
]
FL = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    1.5,
    2.5,
    float("nan"),
    float("inf"),
    float("-inf"),
    3.4028234663852886e38,
    -3.4028234663852886e38,
    1.401298464324817e-45,
    1.1754943508222875e-38,
    16777216.0,
    16777217.0,
    2147483648.0,
    2147483520.0,
    9.2233720368547758e18,
    300.7,
    -300.7,
    0.1,
]
DATES = [
    "0001-01-01",
    "1582-10-04",
    "1582-10-15",
    "1900-02-28",
    "1900-03-01",
    "1969-12-31",
    "1970-01-01",
    "2000-02-29",
    "2024-02-28",
    "2024-12-31",
    "9999-12-31",
]
TS = [
    "0001-01-01 00:00:00",
    "1969-12-31 23:59:59.999999",
    "1969-12-31 23:59:59",
    "1970-01-01 00:00:00",
    "1970-01-01 00:00:00.000001",
    "2000-02-29 12:34:56.789123",
    "2024-10-19 23:20:00",
    "9999-12-31 23:59:59.999999",
    "1900-01-01 00:00:00",
]

TY = {
    "byte": (T.ByteType(), I8),
    "short": (T.ShortType(), I16),
    "int": (T.IntegerType(), I32),
    "long": (T.LongType(), I64),
    "float": (T.FloatType(), FL),
    "double": (T.DoubleType(), FP),
}

NUMTARGETS = ["tinyint", "smallint", "int", "bigint", "float", "double", "boolean"]


def templates(t):
    """SQL templates over columns a, b (and c = a typed same) for numeric type t"""
    ints = t in ("byte", "short", "int", "long")
    L = [
        "a + b",
        "a - b",
        "a * b",
        "a / b",
        "a % b",
        "pmod(a, b)",
        "a = b",
        "a <> b",
        "a < b",
        "a <= b",
        "a > b",
        "a >= b",
        "a <=> b",
        "greatest(a, b)",
        "least(a, b)",
        "coalesce(a, b)",
        "-a",
        "abs(a)",
        "a IS NULL",
        "a IS NOT NULL",
        "if(a > b, a, b)",
        "CASE WHEN a < b THEN a WHEN a = b THEN b ELSE -a END",
        "if(a IS NULL, b, a)",
        "coalesce(a / b, a)",
        "(a + b) * a",
        "(a * b) + a",
        "(a - b) / (a + b)",
        "a * b + a * b",
        "if(a / b > 1, a % b, pmod(a, b))",
        "greatest(a, b, -a)",
        "least(abs(a), abs(b))",
        "CASE WHEN a > b AND b IS NOT NULL THEN a ELSE b END",
        "CASE WHEN a <=> b THEN 1 WHEN a < b THEN 2 ELSE 3 END",
        "a IN (b, a, 0)",
        "a IN (0, 1, -1)",
        "sqrt(a)",
        "floor(a)",
        "ceil(a)",
        "signum(a)",
        "rint(a)",
        "degrees(a)",
        "radians(a)",
        "bit_count(CAST(a AS BIGINT))" if False else "a - a",
    ]
    for tt in NUMTARGETS:
        L.append(f"CAST(a AS {tt})")
    for tt in ("int", "bigint", "double"):
        L.append(f"CAST(a AS {tt}) + CAST(b AS {tt})")
    if ints:
        L += [
            "a div b",
            "a & b",
            "a | b",
            "a ^ b",
            "~a",
            "a & ~b",
            "bit_count(a)",
            "CAST(a AS TIMESTAMP)",
            "CAST(a AS DATE)" if False else "a + 1",
            "shiftleft(a, CAST(b AS INT))",
            "shiftright(a, CAST(b AS INT))",
            "shiftrightunsigned(a, CAST(b AS INT))",
            "shiftleft(a, 1)",
            "shiftright(a, 33)",
            "shiftrightunsigned(a, 64)",
        ]
        if t in ("int", "long"):
            L += [
                "abs(a) % b",
                "(a % b) = 0",
                "pmod(-a, b)",
                "a div b + a % b",
                "(a div b) * b + (a % b)",
            ]
    else:
        L += [
            "isnan(a)",
            "nanvl(a, b)",
            "a / b * b",
            "a % 1",
            "a % 0.0",
            "a / 0.0",
            "a / -0.0",
            "0.0 / a",
            "CAST(a AS TIMESTAMP)",
            "isnan(a) OR a > b",
            "a = CAST('NaN' AS DOUBLE)",
            "a > CAST('NaN' AS DOUBLE)",
            "a < CAST('Infinity' AS DOUBLE)",
            "(a + b) - b",
            "a * 0.1 + b",
            "a * 2.0 + 1.0",
            "(a * b) + a",
            "-a + a",
            "a - a",
            "abs(-a)",
            "sqrt(abs(a))",
            "floor(a) + ceil(b)",
            "CAST(floor(a) AS DOUBLE)",
        ]
    return L


def grid_df(spark, t):
    dt, vals = TY[t]
    vals = vals + [None]
    rows = [(i, a, b) for i, (a, b) in enumerate(itertools.product(vals, vals))]
    return spark.createDataFrame(
        rows,
        T.StructType(
            [
                T.StructField("id", T.LongType()),
                T.StructField("a", dt),
                T.StructField("b", dt),
            ]
        ),
    )


def bool_df(spark):
    vals = [True, False, None]
    rows = [
        (i, a, b, c) for i, (a, b, c) in enumerate(itertools.product(vals, vals, vals))
    ]
    return spark.createDataFrame(
        rows,
        T.StructType(
            [
                T.StructField("id", T.LongType()),
                T.StructField("a", T.BooleanType()),
                T.StructField("b", T.BooleanType()),
                T.StructField("c", T.BooleanType()),
            ]
        ),
    )


BOOL_T = [
    "a AND b",
    "a OR b",
    "NOT a",
    "a <=> b",
    "a = b",
    "a AND (b OR c)",
    "(a OR b) AND (NOT c)",
    "a OR (b AND c)",
    "NOT (a AND b)",
    "(a AND b) OR (NOT a AND c)",
    "if(a, b, c)",
    "CASE WHEN a THEN b WHEN b THEN c ELSE a END",
    "coalesce(a, b, c)",
    "a IS NULL OR b",
    "CAST(a AS INT) + CAST(b AS INT)",
    "greatest(a, b)",
    "least(a, b, c)",
    "a > b",
    "a < b",
    "a >= b",
    "a <= b",
    "if(a IS NULL, c, b)",
    "(a AND b) IS NULL",
    "(a OR b) IS NOT NULL",
    "if(a AND b, 1, if(c, 2, 3))",
    "CAST(a AS TINYINT)",
    "CAST(a AS DOUBLE)",
]


def date_df(spark):
    import datetime

    vals = [datetime.date.fromisoformat(x) for x in DATES] + [None]
    rows = [
        (i, a, b, k)
        for i, ((a, b), k) in enumerate(
            itertools.product(
                itertools.product(vals, vals),
                [-2147483648, -366, -1, 0, 1, 59, 365, 2147483647],
            )
        )
    ]
    return spark.createDataFrame(
        rows,
        T.StructType(
            [
                T.StructField("id", T.LongType()),
                T.StructField("a", T.DateType()),
                T.StructField("b", T.DateType()),
                T.StructField("k", T.IntegerType()),
            ]
        ),
    )


DATE_T = [
    "year(a)",
    "month(a)",
    "dayofmonth(a)",
    "dayofweek(a)",
    "quarter(a)",
    "dayofyear(a)",
    "weekday(a)",
    "last_day(a)",
    "datediff(a, b)",
    "datediff(b, a)",
    "date_add(a, CAST(k AS TINYINT))",
    "date_add(a, k)",
    "date_sub(a, k)",
    "date_add(a, 30)",
    "a = b",
    "a < b",
    "a <= b",
    "a > b",
    "a >= b",
    "a <=> b",
    "greatest(a, b)",
    "least(a, b)",
    "coalesce(a, b)",
    "if(a < b, a, b)",
    "CAST(a AS TIMESTAMP)",
    "CAST(CAST(a AS TIMESTAMP) AS DATE)",
    "datediff(date_add(a, k), b)",
    "year(date_add(a, 1)) + month(a)",
    "CASE WHEN a > b THEN datediff(a, b) ELSE datediff(b, a) END",
    "a IS NULL",
    "a IN (b, DATE'1970-01-01')",
]


def ts_df(spark):
    import datetime

    vals = [datetime.datetime.fromisoformat(x) for x in TS] + [None]
    rows = [(i, a, b) for i, (a, b) in enumerate(itertools.product(vals, vals))]
    return spark.createDataFrame(
        rows,
        T.StructType(
            [
                T.StructField("id", T.LongType()),
                T.StructField("a", T.TimestampType()),
                T.StructField("b", T.TimestampType()),
            ]
        ),
    )


TS_T = [
    "hour(a)",
    "minute(a)",
    "second(a)",
    "CAST(a AS DATE)",
    "CAST(a AS BIGINT)",
    "CAST(a AS DOUBLE)",
    "CAST(a AS INT)",
    "CAST(a AS FLOAT)",
    "a = b",
    "a < b",
    "a <= b",
    "a >= b",
    "a <=> b",
    "greatest(a, b)",
    "least(a, b)",
    "coalesce(a, b)",
    "if(a < b, a, b)",
    "year(a)",
    "month(a)",
    "dayofmonth(a)",
    "datediff(CAST(a AS DATE), CAST(b AS DATE))",
    "CAST(CAST(a AS BIGINT) AS TIMESTAMP)",
    "CAST(CAST(a AS DATE) AS TIMESTAMP)",
    "CAST(CAST(a AS DOUBLE) AS TIMESTAMP)",
    "CAST(a AS BIGINT) - CAST(b AS BIGINT)",
    "a IS NULL",
]


def run_mode(spark, mode, df, exprs, outdir):
    set_mode(spark, mode)
    try:
        q = df.selectExpr("id", *[f"{e} AS c{i}" for i, e in enumerate(exprs)])
        flags = plan_flags(q)

        def once():
            shutil.rmtree(outdir, ignore_errors=True)
            q.write.mode("overwrite").parquet(outdir)
            return D.load(outdir)

        return (
            "ok",
            run_with_async_warm(
                spark, mode, once, lambda a, b: D.is_clean(D.compare(a, b))
            ),
            flags,
        )
    except Exception as e:  # noqa
        return ("err", D.short_err(e))
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def valid(spark, df, exprs):
    ok = []
    for e in exprs:
        try:
            df.selectExpr(e).queryExecution if False else df.selectExpr(
                e
            )._jdf.queryExecution().analyzed()
            ok.append(e)
        except Exception:
            pass
    return ok


def sample_inputs(df, ids, cols, k=3):
    from pyspark.sql import functions as F

    rows = {
        r["id"]: r
        for r in df.filter(F.col("id").isin(ids[:k])).select("id", *cols).collect()
    }
    return [{c: D.fmt_val(rows[i][c]) for c in cols} for i in ids[:k] if i in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="cpu,gpu")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument(
        "--only",
        default="",
        help="comma list of groups: byte,short,int,long,float,double,bool,date,timestamp",
    )
    ap.add_argument("--local-n", type=int, default=4)
    ap.add_argument("--tz", default="UTC")
    a = ap.parse_args()
    modes = a.modes.split(",")
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "tmp")
    os.makedirs(tmp, exist_ok=True)
    spark = build_session(
        use_gpu=any(m not in ("cpu", "cpuansi") for m in modes),
        local_n=a.local_n,
        tz=a.tz,
    )
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    groups = []
    for t in TY:
        groups.append((t, grid_df(spark, t), templates(t), ["a", "b"]))
    groups.append(("bool", bool_df(spark), BOOL_T, ["a", "b", "c"]))
    groups.append(("date", date_df(spark), DATE_T, ["a", "b", "k"]))
    groups.append(("timestamp", ts_df(spark), TS_T, ["a", "b"]))
    only = set(x for x in a.only.split(",") if x)
    pairs = [("cpu", "gpu", "gpu-vs-cpu")] if {"cpu", "gpu"} <= set(modes) else []
    if {"gpu", "fused"} <= set(modes):
        pairs.append(("gpu", "fused", "FUSION-vs-unfused"))
    if {"cpu", "fused"} <= set(modes):
        pairs.append(("cpu", "fused", "fused-vs-cpu"))
    if {"gpu", "ast"} <= set(modes):
        pairs.append(("gpu", "ast", "ast-vs-gpu"))
    res = {p[2]: [] for p in pairs}
    tot = collections.Counter()
    cov = collections.Counter()
    for name, df, tl, cols in groups:
        if only and name not in only:
            continue
        set_mode(spark, "cpu")
        gp = os.path.join(
            tmp, f"grid_{name}"
        )  # materialise the grid through parquet (written by CPU)
        df.write.mode("overwrite").parquet(gp)
        df = spark.read.parquet(gp)
        ex = valid(spark, df, tl)
        print(
            f"[{now()}] group {name}: rows={df.count()} templates={len(ex)}/{len(tl)}",
            flush=True,
        )
        for s in range(0, len(ex), a.chunk):
            chunk = ex[s : s + a.chunk]
            runs = {
                m: run_mode(spark, m, df, chunk, os.path.join(tmp, f"{name}_{s}_{m}"))
                for m in modes
            }
            for m, r in runs.items():
                if r[0] == "ok":
                    cov[m + ".queries"] += 1
                    if r[2]["gpu_project"]:
                        cov[m + ".gpu_project"] += 1
                    if r[2]["fused_project"]:
                        cov[m + ".fused_project"] += 1
                else:
                    cov[m + ".chunk_errors"] += 1
            for ma, mb, label in pairs:
                ra, rb = runs[ma], runs[mb]
                if ra[0] == "ok" and rb[0] == "ok":
                    c = D.compare(ra[1], rb[1])
                    tot[label + ".exprs"] += len(chunk)
                    if c["rowset"]:
                        res[label].append(
                            {
                                "group": name,
                                "template": "<rowset>",
                                "info": str(c["rowset"]),
                            }
                        )
                    for cn, info in c["cols"].items():
                        e = chunk[int(cn[1:])]
                        res[label].append(
                            {
                                "group": name,
                                "template": e,
                                "n_mismatch_rows": info["n"],
                                "types": info["type"],
                                "sample_inputs": sample_inputs(df, info["ids"], cols),
                                "outputs": {mm: None for mm in (ma, mb)},
                            }
                        )
                        # attach outputs for the first sample id
                        for mm, rr in ((ma, ra), (mb, rb)):
                            import numpy as np

                            i = int(
                                np.searchsorted(
                                    rr[1].column("id").to_numpy(), info["ids"][0]
                                )
                            )
                            res[label][-1]["outputs"][mm] = D.fmt_val(
                                rr[1].column(cn)[i].as_py()
                            )
                    tot[label + ".bad_exprs"] += len(c["cols"])
                else:
                    # per-expression fallback to find which expression errors in only one mode
                    for e in chunk:
                        r1 = run_mode(spark, ma, df, [e], os.path.join(tmp, f"x_{ma}"))
                        r2 = run_mode(spark, mb, df, [e], os.path.join(tmp, f"x_{mb}"))
                        tot[label + ".exprs"] += 1
                        if r1[0] != r2[0]:
                            res[label].append(
                                {
                                    "group": name,
                                    "template": e,
                                    "error": {
                                        ma: r1[1] if r1[0] == "err" else "ok",
                                        mb: r2[1] if r2[0] == "err" else "ok",
                                    },
                                }
                            )
                            tot[label + ".bad_exprs"] += 1
                        elif r1[0] == "ok":
                            c = D.compare(r1[1], r2[1])
                            if not D.is_clean(c):
                                info = c["cols"].get(
                                    "c0", {"n": -1, "ids": [], "type": ()}
                                )
                                res[label].append(
                                    {
                                        "group": name,
                                        "template": e,
                                        "n_mismatch_rows": info["n"],
                                        "sample_inputs": sample_inputs(
                                            df, info["ids"], cols
                                        ),
                                    }
                                )
                                tot[label + ".bad_exprs"] += 1
        shutil.rmtree(gp, ignore_errors=True)
    json.dump(
        {"results": res, "totals": dict(tot), "coverage": dict(cov), "modes": modes},
        open(os.path.join(a.out, "directed.json"), "w"),
        indent=1,
        default=str,
    )
    print("TOTALS", dict(tot))
    print("COVERAGE", dict(cov))
    for label, lst in res.items():
        print(f"== {label}: {len(lst)} mismatching templates")
        for r in lst[:400]:
            print(
                "  ",
                r["group"],
                r["template"],
                r.get("n_mismatch_rows", ""),
                json.dumps(r.get("sample_inputs", [])[:1]),
                json.dumps(r.get("outputs", r.get("error", ""))),
            )
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
