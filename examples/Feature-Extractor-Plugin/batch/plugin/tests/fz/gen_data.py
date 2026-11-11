"""Generate the deterministic fuzz input table (parquet) with Spark SQL only: every value is a pure function of (id, column).
Columns: id long; b1,b2 boolean; t1,t2 byte; s1,s2 short; i1,i2,i3 int; l1,l2 long; f1,f2 float; d1,d2,d3 double; dt1,dt2 date; ts1,ts2 timestamp.
~12% NULL in every column; ~45% 'special' values (MIN/MAX/0/-1/NaN/Inf/-0.0/denormals/...), rest random.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def h(k, mod=100):  # bucket in [0,mod)
    return f"pmod(xxhash64(id, {k}), {mod})"


def rnd(k):  # random 64-bit
    return f"xxhash64(id, {k + 7000})"


def int_col(name, k, sql_t, mn, mx):
    return (
        f"CASE WHEN {h(k)}<12 THEN CAST(NULL AS {sql_t}) WHEN {h(k)}<20 THEN CAST({mn} AS {sql_t}) WHEN {h(k)}<28 THEN CAST({mx} AS {sql_t}) "
        f"WHEN {h(k)}<36 THEN CAST(0 AS {sql_t}) WHEN {h(k)}<42 THEN CAST(1 AS {sql_t}) WHEN {h(k)}<48 THEN CAST(-1 AS {sql_t}) "
        f"WHEN {h(k)}<56 THEN CAST(2 AS {sql_t}) "
        f"WHEN {h(k)}<75 THEN CAST(pmod({rnd(k)}, 41) - 20 AS {sql_t}) ELSE CAST({rnd(k)} AS {sql_t}) END AS {name}"
    )


def fp_col(name, k, sql_t):
    mx, mn_pos, mn_norm = (
        ("3.4028234663852886E38", "1.401298464324817E-45", "1.1754943508222875E-38")
        if sql_t == "FLOAT"
        else ("1.7976931348623157E308", "4.9E-324", "2.2250738585072014E-308")
    )
    c = lambda v: f"CAST('{v}' AS {sql_t})"
    mag = f"pow(10.0D, CAST(pmod({rnd(k + 1)}, {'60' if sql_t=='FLOAT' else '600'}) - {'30' if sql_t=='FLOAT' else '300'} AS DOUBLE))"
    rnd_wide = (
        f"CAST((CAST({rnd(k)} AS DOUBLE) / 9.223372036854775807E18) * {mag} AS {sql_t})"
    )
    rnd_small = (
        f"CAST((CAST({rnd(k)} AS DOUBLE) / 9.223372036854775807E18) * 10.0D AS {sql_t})"
    )
    rnd_half = f"CAST((pmod({rnd(k)}, 41) - 20) / 2.0D AS {sql_t})"  # halves: x.5 ties
    return (
        f"CASE WHEN {h(k)}<12 THEN CAST(NULL AS {sql_t}) WHEN {h(k)}<17 THEN {c('NaN')} WHEN {h(k)}<21 THEN {c('Infinity')} WHEN {h(k)}<25 THEN {c('-Infinity')} "
        f"WHEN {h(k)}<30 THEN {c('0.0')} WHEN {h(k)}<35 THEN {c('-0.0')} WHEN {h(k)}<38 THEN {c(mx)} WHEN {h(k)}<40 THEN {c('-' + mx)} "
        f"WHEN {h(k)}<42 THEN {c(mn_pos)} WHEN {h(k)}<44 THEN {c('-' + mn_pos)} WHEN {h(k)}<46 THEN {c(mn_norm)} WHEN {h(k)}<48 THEN {c('1.0')} WHEN {h(k)}<50 THEN {c('-1.0')} "
        f"WHEN {h(k)}<60 THEN {rnd_half} WHEN {h(k)}<80 THEN {rnd_small} ELSE {rnd_wide} END AS {name}"
    )


def date_col(name, k):
    return (
        f"CASE WHEN {h(k)}<12 THEN CAST(NULL AS DATE) WHEN {h(k)}<20 THEN DATE'1970-01-01' WHEN {h(k)}<26 THEN DATE'1969-12-31' "
        f"WHEN {h(k)}<32 THEN DATE'0001-01-01' WHEN {h(k)}<38 THEN DATE'9999-12-31' WHEN {h(k)}<44 THEN DATE'2000-02-29' WHEN {h(k)}<48 THEN DATE'1900-03-01' "
        f"ELSE date_add(DATE'1970-01-01', CAST(pmod({rnd(k)}, 4000000) - 719162 AS INT)) END AS {name}"
    )


def ts_col(name, k):
    return (
        f"CASE WHEN {h(k)}<12 THEN CAST(NULL AS TIMESTAMP) WHEN {h(k)}<20 THEN timestamp_micros(0) WHEN {h(k)}<26 THEN timestamp_micros(-1) "
        f"WHEN {h(k)}<32 THEN timestamp_micros(1) WHEN {h(k)}<36 THEN TIMESTAMP'0001-01-01 00:00:00' WHEN {h(k)}<40 THEN TIMESTAMP'9999-12-31 23:59:59.999999' "
        f"WHEN {h(k)}<46 THEN timestamp_micros(-86400000000) WHEN {h(k)}<50 THEN timestamp_micros(86399999999) "
        f"ELSE timestamp_micros(pmod({rnd(k)}, 504911232000000000) - 252455616000000000 + 0) END AS {name}"
    )


def bool_col(name, k):
    return (
        f"CASE WHEN {h(k)}<12 THEN CAST(NULL AS BOOLEAN) ELSE {h(k)}<56 END AS {name}"
    )


def columns():
    c = ["id"]
    k = 100
    for n in ("b1", "b2"):
        c.append(bool_col(n, k))
        k += 10
    for n in ("t1", "t2"):
        c.append(int_col(n, k, "TINYINT", -128, 127))
        k += 10
    for n in ("s1", "s2"):
        c.append(int_col(n, k, "SMALLINT", -32768, 32767))
        k += 10
    for n in ("i1", "i2", "i3"):
        c.append(int_col(n, k, "INT", -2147483648, 2147483647))
        k += 10
    for n in ("l1", "l2"):
        c.append(int_col(n, k, "BIGINT", "-9223372036854775808", "9223372036854775807"))
        k += 10
    for n in ("f1", "f2"):
        c.append(fp_col(n, k, "FLOAT"))
        k += 10
    for n in ("d1", "d2", "d3"):
        c.append(fp_col(n, k, "DOUBLE"))
        k += 10
    for n in ("dt1", "dt2"):
        c.append(date_col(n, k))
        k += 10
    for n in ("ts1", "ts2"):
        c.append(ts_col(n, k))
        k += 10
    return c


def make(spark, rows, out, parts=8):
    cols = columns()
    df = spark.range(rows, numPartitions=parts).selectExpr(*cols)
    df.write.mode("overwrite").parquet(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = build_session(use_gpu=False, local_n=4, driver_mem="8g")
    make(sp, a.rows, a.out)
    d = sp.read.parquet(a.out)
    print("rows", d.count())
    d.printSchema()
    d.selectExpr(
        "sum(CASE WHEN isnan(d1) THEN 1 ELSE 0 END) nan_d1",
        "sum(CASE WHEN d1 = 0 AND CAST(d1 AS STRING)='-0.0' THEN 1 ELSE 0 END) negz",
        "sum(CASE WHEN f1 = 0 AND CAST(f1 AS STRING)='-0.0' THEN 1 ELSE 0 END) negz_f1",
        "min(i1)",
        "max(l1)",
        "min(dt1)",
        "max(ts1)",
    ).show(truncate=False)
