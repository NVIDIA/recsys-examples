# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One recsys feature query, three ways: CPU Spark, spark-rapids, spark-rapids + expression fusion.

Prints the physical plan of each, the fusion decisions, times each (noop write, median of N), and checks that
the three outputs are identical. Set FUSION_DUMP_DIR to also get the generated CUDA source of every fused kernel.

    scripts/run-spark.sh spark-submit \
        --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension \
        examples/feature_walkthrough.py --rows 20000000 --out $FUSION_WORK/walkthrough
"""
import argparse
import os
import statistics
import time

from pyspark.sql import SparkSession

FEATURE_SQL = """
SELECT
  user_id,
  click_cnt / (show_cnt + 1D)                         AS ctr,          -- smoothed click-through rate
  CASE WHEN age < 18 THEN 0 WHEN age < 35 THEN 1
       WHEN age < 55 THEN 2 ELSE 3 END                AS age_bucket,   -- bucketize
  hour(event_ts)                                      AS hour_of_day,  -- time feature
  least(greatest(price, 0D), 1000D)                   AS price_clip,   -- clip outliers
  pmod(user_id * 31 + item_cat, 100000)               AS user_x_cat,   -- hashed cross feature
  (click_cnt > 0 AND show_cnt >= 10)                  AS is_active
FROM events
"""


def gen_data(spark, path, rows):
    if os.path.exists(os.path.join(path, "_SUCCESS")):
        return
    spark.conf.set("spark.rapids.sql.enabled", "false")
    spark.range(rows).selectExpr(
        "id AS user_id",
        "CAST(pmod(id * 7919, 500) AS INT) AS item_cat",
        "CAST(pmod(id * 104729, 80) AS INT) AS age",
        "CASE WHEN pmod(id, 17) = 0 THEN NULL ELSE pmod(id * 13, 50) END AS click_cnt",
        "pmod(id * 31, 2000) AS show_cnt",
        "CASE WHEN pmod(id, 23) = 0 THEN -5.0D ELSE pmod(id * 37, 150000) / 100.0D END AS price",
        "CAST(1700000000L + pmod(id * 7, 86400 * 30) AS TIMESTAMP) AS event_ts",
    ).write.mode("overwrite").parquet(path)


def configure(spark, mode):
    spark.conf.set("spark.rapids.sql.enabled", "false" if mode == "cpu" else "true")
    spark.conf.set("spark.rapids.sql.fusion.mode", "SAFE" if mode == "fused" else "OFF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    spark = SparkSession.builder.appName("feature-walkthrough").getOrCreate()
    spark.conf.set("spark.session.timeZone", "UTC")
    spark.conf.set("spark.sql.parquet.enableVectorizedReader", "true")
    data = os.path.join(a.out, "events")
    gen_data(spark, data, a.rows)
    spark.read.parquet(data).createOrReplaceTempView("events")
    jvm = spark.sparkContext._jvm

    times = {}
    for mode in ["cpu", "gpu", "fused"]:
        configure(spark, mode)
        df = spark.sql(FEATURE_SQL)
        print(f"\n===== [{mode}] physical plan =====")
        print(df._jdf.queryExecution().executedPlan().toString())
        if mode == "fused":
            print("===== [fused] fusion decisions =====")
            print(jvm.com.nvidia.spark.rapids.fusion.Fusion.explain(df._jdf))
        for _ in range(a.warmup):
            df.write.format("noop").mode("overwrite").save()
        ts = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            df.write.format("noop").mode("overwrite").save()
            ts.append(time.perf_counter() - t0)
        times[mode] = statistics.median(ts)
        df.write.mode("overwrite").option("parquet.enable.dictionary", "false").parquet(
            os.path.join(a.out, f"res_{mode}")
        )

    # compare outputs on the CPU (exact, NaN-insensitive via Spark equality semantics)
    configure(spark, "cpu")
    res = {m: spark.read.parquet(os.path.join(a.out, f"res_{m}")) for m in times}
    print("\n===== results =====")
    for m in ["gpu", "fused"]:
        d1 = res["cpu"].exceptAll(res[m]).count()
        d2 = res[m].exceptAll(res["cpu"]).count()
        print(f"rows differing cpu vs {m}: {d1} / {d2}")
    d = (
        res["gpu"].exceptAll(res["fused"]).count()
        + res["fused"].exceptAll(res["gpu"]).count()
    )
    print(f"rows differing gpu vs fused: {d}")
    print(f"rows: {res['cpu'].count()}")
    for m, t in times.items():
        print(
            f"median wall time [{m}]: {t:.3f} s  (reps={a.reps}, noop write, includes Parquet scan)"
        )
    spark.stop()


if __name__ == "__main__":
    main()
