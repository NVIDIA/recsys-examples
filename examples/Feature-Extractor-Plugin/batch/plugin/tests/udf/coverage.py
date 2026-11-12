#!/usr/bin/env python3
"""'Writing style coverage' experiment: for each way a user can express the same computation, does the plan run on the GPU,
was the UDF translated into Catalyst expressions (udf-compiler), is it fused by the fusion plugin, and is the result equal to CPU?

  python3 coverage.py --data DIR --out OUT.json [--modes cpu,gpu,fused]
Needs tests/udf/udfs.jar (build with tests/udf/build.sh).  Writes OUT.json and prints a markdown table.
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fz"))
os.environ["EXTRA_JARS"] = os.path.join(HERE, "udfs.jar")
import diff as D
from common import *


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="cpu,gpu,fused")
    ap.add_argument("--no-udf-compiler", action="store_true")
    a = ap.parse_args()
    modes = a.modes.split(",")
    ext = [
        "com.nvidia.spark.udf.Plugin",
        os.environ.get(
            "FUSION_EXTENSIONS", "com.nvidia.spark.rapids.fusion.FusionExtension"
        ),
    ]
    ext = [e for e in ext if e]
    os.environ["FUSION_EXTENSIONS"] = ",".join(dict.fromkeys(ext))
    extra = {
        "spark.rapids.sql.udfCompiler.enabled": "false" if a.no_udf_compiler else "true"
    }
    spark = build_session(use_gpu=True, local_n=4, extra=extra)
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf, udf
    from pyspark.sql.types import DoubleType, IntegerType

    jv = spark._jvm
    jv.fusiontest.Udfs.register(spark._jsparkSession)
    base = (
        spark.read.parquet(a.data)
        .select("id", "i1", "i2", "d1", "d2", "l1", "l2", "s1", "f1")
        .filter("id < 100000")
    )
    base.createOrReplaceTempView("t")

    py_arith = udf(
        lambda a_, b_: None
        if a_ is None or b_ is None
        else ((a_ * b_ + 1 + 2**31) % 2**32) - 2**31,
        IntegerType(),
    )

    @pandas_udf(DoubleType())
    def pd_u(x):
        return x * 2.0 + 1.0

    cases = collections.OrderedDict()
    cases["SQL expression"] = lambda: spark.sql(
        "SELECT id, i1 * i2 + 1 AS r, if(d1 > 0, d1 * 2.0, -d1) AS q, sqrt(d1*d1+1.0) AS s FROM t"
    )
    cases["DataFrame expression"] = lambda: base.select(
        "id",
        (F.col("i1") * F.col("i2") + 1).alias("r"),
        F.when(F.col("d1") > 0, F.col("d1") * 2.0).otherwise(-F.col("d1")).alias("q"),
        F.sqrt(F.col("d1") * F.col("d1") + 1.0).alias("s"),
    )
    cases["DataFrame filter+project"] = lambda: base.filter(
        (F.col("i1") > 10) & (F.col("d1") < 100.0)
    ).select("id", (F.col("d1") * 2.0 + F.col("i1")).alias("r"))
    for nm, call in [
        ("Scala UDF: a*b+1", "u_arith(i1, i2)"),
        ("Scala UDF: if/else on double", "u_branch(d1)"),
        ("Scala UDF: nested if/else/&&", "u_branchy(i1, i2)"),
        ("Scala UDF: math.sqrt", "u_math(d1)"),
        ("Scala UDF: math.max/abs", "u_absmax(l1, l2)"),
        ("Scala UDF: boolean ||", "u_bool(i1, d1)"),
        ("Scala UDF: while loop", "u_loop(i2)"),
        ("Scala UDF: String.length", "u_strlen(CAST(i1 AS STRING))"),
        ("Scala UDF: captured val", "u_closure(i1)"),
        ("Scala UDF: boxed Integer null check", "u_opt(i1)"),
        ("Scala UDF: mixed int/long/double", "u_mixed(i1, l1, d1)"),
        ("Java UDF (UDF2 class): a*b+1", "j_arith(i1, i2)"),
        ("RapidsUDF (hand-written GPU UDF)", "r_square(d1)"),
    ]:
        cases[nm] = lambda c=call: spark.sql(f"SELECT id, {c} AS r FROM t")
    cases["Scala UDF via DataFrame API (call_udf)"] = lambda: base.select(
        "id", F.call_udf("u_arith", F.col("i1"), F.col("i2")).alias("r")
    )
    cases["Dataset typed map (ds.map)"] = lambda: DataFrame(
        jv.fusiontest.Udfs.typedMapDF(base._jdf, spark._jsparkSession), spark
    )
    cases["Dataset typed filter (ds.filter)"] = lambda: DataFrame(
        jv.fusiontest.Udfs.typedFilterDF(base._jdf, spark._jsparkSession), spark
    )
    cases["Python UDF (row at a time)"] = lambda: base.select(
        "id", py_arith("i1", "i2").alias("r")
    )
    cases["pandas UDF (vectorised)"] = lambda: base.select("id", pd_u("d1").alias("r"))
    cases["Expression around a black box: SQL + RapidsUDF + SQL"] = lambda: spark.sql(
        "SELECT id, (i1 * 3 + 7) % 11 + CAST(r_square(d1) AS DOUBLE) * 2.0 - d2 AS r FROM t"
    )
    cases["Expression around a Python UDF: SQL + pyUDF + SQL"] = lambda: base.select(
        "id", (py_arith("i1", "i2") * 3 + F.col("i2") % 7).alias("r")
    )

    tmp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "cov_tmp")
    os.makedirs(tmp, exist_ok=True)
    rows = []
    for name, mk in cases.items():
        rec = {"style": name, "modes": {}}
        outs = {}
        for m in modes:
            try:
                set_mode(spark, m)
                df = mk()
                p = plan_str(df)
                fl = plan_flags(df)
                cpu_proj = (
                    re.search(r"(?<![A-Za-z])Project\b|(?<![A-Za-z])Filter\b", p)
                    is not None
                )  # CPU row Project/Filter present
                rec["modes"][m] = {
                    "gpu_project": fl["gpu_project"],
                    "fused_project": fl["fused_project"],
                    "gpu_filter": fl["gpu_filter"],
                    "fused_filter": fl["fused_filter"],
                    "cpu_project_or_filter": cpu_proj,
                    "scala_udf_left": "ScalaUDF" in p,
                    "python_udf_left": "BatchEvalPython" in p or "ArrowEvalPython" in p,
                    "deserialize_obj": "SerializeFromObject" in p
                    or "MapElements" in p
                    or "DeserializeToObject" in p,
                    "plan": p[:1500],
                }
                o = os.path.join(tmp, re.sub(r"\W", "_", name) + "_" + m)
                df.write.mode("overwrite").parquet(o)
                outs[m] = D.load(o)
                shutil.rmtree(o, ignore_errors=True)
            except Exception as e:  # noqa
                rec["modes"][m] = {"error": D.short_err(e), "error_full": str(e)[:4000]}
        if "cpu" in outs:
            for m in modes:
                if m != "cpu" and m in outs:
                    c = D.compare(outs["cpu"], outs[m])
                    rec["modes"][m]["equals_cpu"] = D.is_clean(c)
                    if not D.is_clean(c):
                        rec["modes"][m]["mismatch"] = {
                            k: v["n"] for k, v in c["cols"].items()
                        }
        rows.append(rec)
        g = rec["modes"].get("gpu", {})
        print(
            f"[{now()}] {name}: gpu={g.get('gpu_project') or g.get('gpu_filter')} udf_left={g.get('scala_udf_left')} err={g.get('error')}",
            flush=True,
        )
    json.dump(rows, open(a.out, "w"), indent=1)

    def yn(b):
        return "yes" if b else "no"

    print(
        "\n| Style | On GPU | UDF translated | Fused | Matches CPU | Notes |\n|---|---|---|---|---|---|"
    )
    for r in rows:
        g = r["modes"].get("gpu", {})
        f = r["modes"].get("fused", {})
        if "error" in g:
            print(f"| {r['style']} | error | - | - | - | {g['error'][:80]} |")
            continue
        gpu = g.get("gpu_project") or g.get("gpu_filter")
        note = []
        if g.get("cpu_project_or_filter"):
            note.append("CPU Project/Filter remains")
        if g.get("deserialize_obj"):
            note.append("object serialize/deserialize")
        if g.get("python_udf_left"):
            note.append("Python worker")
        print(
            f"| {r['style']} | {yn(gpu)} | {yn(not g.get('scala_udf_left'))} | {yn(f.get('fused_project') or f.get('fused_filter')) if f else 'not tested'} | {yn(f.get('equals_cpu', g.get('equals_cpu')))} | {'; '.join(note)} |"
        )


if __name__ == "__main__":
    main()
