// spark-shell -i: per-operator GPU op-time breakdown (DEBUG metrics, AQE off) for each mode.
import com.nvidia.spark.rapids.fusion._
import org.apache.spark.sql.execution.SparkPlan
val dir = sys.props.getOrElse("perf.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/data/perf_t")
spark.read.parquet(dir).createOrReplaceTempView("t")
spark.conf.set("spark.sql.adaptive.enabled", "false")
val qs = Seq(
  "Q1" -> "select sum(((a*b + c) * d - e) / (f + 1.5) + (g - a) * (h + 2.0) - (b*c - d*e) + a*a*0.001 + ((x*y) + x - y)) from t",
  "Q3" -> "select count(*), sum(a*b + c*d - e) from t where a*b > 100.0 and c < 90.0 and (d + e) > 20.0 and h < 0.9",
  "Q4" -> "select k, sum(a*b + c) s1, avg((d - e) * (f + 1)) s2, max(g * h + a) m from t group by k",
  "Q7" -> "select a*b + c as r1, (a - b) * (c + d) as r2, case when e > 5 then f else g end as r3, sqrt(abs(a - b)) as r4 from t")
def collectTimes(p: SparkPlan): Seq[(String, Double)] = {
  val b = scala.collection.mutable.ArrayBuffer[(String, Double)]()
  p.foreach { n =>
    val m = n.metrics.filter { case (k, _) => k.toLowerCase.contains("optime") || k == "fusedKernelTime" || k == "scanTime" || k == "gpuDecodeTime" || k == "computeAggTime" }
    m.foreach { case (k, v) => b += ((s"${n.nodeName}.$k", v.value / 1e6)) }
  }
  b.toSeq
}
for ((name, q) <- qs; mode <- Seq("OFF", "FUSED")) {
  spark.conf.set("spark.rapids.sql.fusion.mode", if (mode == "FUSED") "SAFE" else "OFF")
  spark.conf.set("spark.rapids.sql.metrics.level", "DEBUG")
  val runs = (0 until 4).map { _ =>
    val df = spark.sql(if (name == "Q7") q else q)
    val t0 = System.nanoTime()
    if (name == "Q7") df.write.format("noop").mode("overwrite").save() else df.collect()
    val wall = (System.nanoTime() - t0) / 1e9
    (wall, collectTimes(df.queryExecution.executedPlan))
  }
  val last = runs.last
  println(f"BREAKDOWN $name $mode wall(last)=${last._1}%.3fs")
  last._2.filter(_._2 > 1.0).foreach { case (k, v) => println(f"BREAKDOWN   $k%-55s $v%9.1f ms") }
}
sys.exit(0)
