# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the KuaiRand-27K CSV files to Parquet tables used by the cases (CPU Spark, run once).

    log          one standard-log CSV file (--log-file), the exposure/feedback log
    user         user_features_27k.csv
    video_basic  video_features_basic_27k.csv

KuaiRand: https://kuairand.com (CC BY-SA 4.0), Gao et al., CIKM 2022.
"""
import argparse
import os

from pyspark.sql import SparkSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, help="extracted KuaiRand-27K/data directory"
    )
    ap.add_argument(
        "--log-file", required=True, help="which log CSV to use, relative to --src"
    )
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    spark = SparkSession.builder.appName("kuairand-prepare").getOrCreate()
    spark.conf.set("spark.rapids.sql.enabled", "false")
    spark.conf.set("spark.session.timeZone", "UTC")
    tables = {
        "log": a.log_file,
        "user": "user_features_27k.csv",
        "video_basic": "video_features_basic_27k.csv",
    }
    for name, f in tables.items():
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(os.path.join(a.src, f))
        )
        df.write.mode("overwrite").parquet(os.path.join(a.dst, name))
        df = spark.read.parquet(os.path.join(a.dst, name))
        print(f"== {name}: {df.count()} rows")
        df.printSchema()
    spark.stop()


if __name__ == "__main__":
    main()
