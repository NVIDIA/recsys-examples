# PySpark demo: SQL + DataFrame expression fusion.  Run with examples/run-demo.sh (or add the confs below to any spark-submit).
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("fusion-demo")
    .config("spark.sql.extensions", "com.nvidia.spark.rapids.fusion.FusionExtension")
    .config("spark.rapids.sql.fusion.mode", "SAFE")
    .config(
        "spark.rapids.sql.fusion.explain", "NOT_FUSED"
    )  # prints [fusion] ... reasons in the driver log
    .getOrCreate()
)

df = spark.range(0, 5_000_000).select(
    "id",
    (F.col("id") * 0.37).alias("a"),
    (F.col("id") % 97).cast("double").alias("b"),
    (F.col("id") % 5).cast("int").alias("c"),
)
df.write.mode("overwrite").parquet(
    "/tmp/fusion_demo_%s" % spark.sparkContext.applicationId
)  # parquet so the scan runs on the GPU
t = spark.read.parquet("/tmp/fusion_demo_%s" % spark.sparkContext.applicationId)

# DataFrame API
q = t.where((F.col("a") * F.col("b") > 100.0) & (F.col("c") < 4)).select(
    (F.col("a") * F.col("b") + F.col("c")).alias("x"),
    F.when(F.col("c") > 2, F.sqrt(F.col("a"))).otherwise(-F.col("b")).alias("y"),
)
# what would be fused and why not (works from Python through py4j)
print(spark.sparkContext._jvm.com.nvidia.spark.rapids.fusion.Fusion.explain(q._jdf))
print(q.agg(F.sum("x"), F.sum("y")).collect())

# SQL
t.createOrReplaceTempView("t")
print(spark.sql("select sum(a*b + c) from t where a*b > 100 and c < 4").collect())
print(spark.sparkContext._jvm.com.nvidia.spark.rapids.fusion.Fusion.stats())
