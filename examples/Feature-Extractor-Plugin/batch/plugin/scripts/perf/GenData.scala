// spark-shell -i: writes the e2e benchmark table (parquet) -- rows/dir via -Dperf.rows / -Dperf.dir
val rows = sys.props.getOrElse("perf.rows", "30000000").toLong
val dir = sys.props.getOrElse("perf.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/data/perf_t")
val df = spark.range(0, rows, 1, 32).selectExpr(
  "id",
  "cast(id % 1000 as int) as k",
  "cast(id % 97 as int) as i1",
  "cast((id * 7) % 10007 as bigint) as l1",
  "rand(1) * 100 as a", "rand(2) * 100 as b", "rand(3) * 100 as c", "rand(4) * 100 + 1 as d",
  "rand(5) * 10 as e", "rand(6) * 10 + 0.5 as f", "rand(7) * 1000 as g", "rand(8) as h",
  "cast(rand(9) * 100 as float) as x", "cast(rand(10) * 100 as float) as y")
df.write.mode("overwrite").parquet(dir)
println("WROTE " + dir + " rows=" + spark.read.parquet(dir).count())
sys.exit(0)
