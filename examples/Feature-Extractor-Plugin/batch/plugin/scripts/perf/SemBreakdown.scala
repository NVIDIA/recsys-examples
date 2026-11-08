// per-operator op time for selected E/N benchmark queries, OFF vs FUSED (DEBUG metrics, AQE off). -Dsb.queries=E6,E5
import com.nvidia.spark.rapids.fusion._
import org.apache.spark.sql.execution.SparkPlan
val dir = sys.props.getOrElse("sb.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/data/semdata")
val only = sys.props.getOrElse("sb.queries", "E6").split(",").toSet
val queries = scala.io.Source.fromFile(sys.env.getOrElse("FUSION_HOME", ".") + "/scripts/perf/sem_queries.tsv").getLines().map(l => { val i = l.indexOf('\t'); (l.substring(0, i), l.substring(i + 1)) }).toSeq.sortBy(_._1)
spark.read.parquet(s"$dir/wide").createOrReplaceTempView("wide")
spark.conf.set("spark.sql.adaptive.enabled", "false"); spark.conf.set("spark.rapids.sql.metrics.level", "DEBUG")
def times(p: SparkPlan): Seq[(String, Double)] = {
  val b = scala.collection.mutable.ArrayBuffer[(String, Double)]()
  p.foreach { n => n.metrics.foreach { case (k, v) => if ((k == "opTimeLegacy" || k == "fusedKernelTime" || k == "scanTime" || k == "gpuDecodeTime") && v.value > 0) b += ((s"${n.nodeName}.$k", v.value / 1e6)) } }
  b.toSeq
}
for ((name, q) <- queries if only.exists(name.startsWith); mode <- Seq("OFF", "FUSED")) {
  spark.conf.set("spark.rapids.sql.fusion.mode", if (mode == "FUSED") "SAFE" else "OFF")
  var last: (Double, Seq[(String, Double)]) = null
  for (it <- 0 until 4) {
    FusionStats.reset()
    val df = spark.sql(q); val t0 = System.nanoTime(); df.queryExecution.toRdd.foreach(_ => ())
    last = ((System.nanoTime() - t0) / 1e9, times(df.queryExecution.executedPlan))
  }
  println(f"\nSBD $name $mode wall=${last._1}%.3fs  ${FusionStats.timing}")
  if (mode == "FUSED") { val it = com.nvidia.spark.rapids.fusion.NativeBridge.kernelInfo.entrySet().iterator(); while (it.hasNext) { val e = it.next(); println("\nSBD   kernel " + e.getKey + " " + e.getValue) } }
  last._2.foreach { case (k, v) => println(f"SBD    $k%-50s $v%9.1f ms") }
}
sys.exit(0)
