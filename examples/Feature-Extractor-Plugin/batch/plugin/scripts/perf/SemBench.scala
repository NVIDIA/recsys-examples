// spark-shell -i: expression-heavy benchmark queries (E*) and controls (N*), OFF vs AST vs FUSED. -Dsb.dir=... -Dsb.reps=5 -Dsb.queries=E1,E2
import com.nvidia.spark.rapids.fusion._
val dir = sys.props.getOrElse("sb.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/data/semdata")
val reps = sys.props.getOrElse("sb.reps", "5").toInt
val only = sys.props.get("sb.queries").map(_.split(",").toSet)
val queries = scala.io.Source.fromFile(sys.env.getOrElse("FUSION_HOME", ".") + "/scripts/perf/sem_queries.tsv").getLines().map(l => { val i = l.indexOf('\t'); (l.substring(0, i), l.substring(i + 1)) }).toSeq.sortBy(_._1)
for (t <- Seq("wide", "lineitem", "orders", "customer", "part", "supplier", "nation", "region"))
  if (new java.io.File(s"$dir/$t").exists) spark.read.parquet(s"$dir/$t").createOrReplaceTempView(t)
def setMode(m: String): Unit = {
  spark.conf.set("spark.rapids.sql.projectAstEnabled", (m == "AST").toString)
  spark.conf.set("spark.rapids.sql.fusion.mode", if (m == "FUSED") "SAFE" else "OFF")
}
def run(sql: String): Double = { val t0 = System.nanoTime(); spark.sql(sql).write.format("noop").mode("overwrite").save(); (System.nanoTime() - t0) / 1e9 }
val sel = queries.filter { case (k, _) => only.forall(s => s.exists(p => k.startsWith(p))) }
val tab = scala.collection.mutable.ArrayBuffer[String]()
for ((name, q) <- sel) {
  setMode("FUSED"); spark.conf.set("spark.rapids.sql.fusion.explain", "ALL"); FusionReport.clear()
  try { spark.sql(q).queryExecution.executedPlan } catch { case t: Throwable => println(s"SB-PLAN-ERROR $name $t") }
  FusionReport.dump().filter(_.startsWith("NOT")).distinct.take(6).foreach(s => println(s"SB-NOTFUSED $name: " + s.take(260)))
  spark.conf.set("spark.rapids.sql.fusion.explain", "NONE")
  val med = scala.collection.mutable.LinkedHashMap[String, Double]()
  val times = scala.collection.mutable.Map[String, scala.collection.mutable.ArrayBuffer[Double]]("OFF" -> new scala.collection.mutable.ArrayBuffer[Double], "AST" -> new scala.collection.mutable.ArrayBuffer[Double], "FUSED" -> new scala.collection.mutable.ArrayBuffer[Double])
  var failed = false
  // interleaved: round 0 warms every mode, then reps rounds of OFF, AST, FUSED (cache/ordering effects hit all modes alike)
  for (round <- 0 to reps; m <- Seq("OFF", "AST", "FUSED") if !failed) {
    setMode(m)
    try { val t = run(q); if (round > 0) times(m) += t }
    catch { case t: Throwable => println(s"SB-ERROR $name $m $t"); failed = true }
  }
  if (!failed) for (m <- Seq("OFF", "AST", "FUSED")) {
    val ts = times(m).sorted; med(m) = ts(ts.length / 2)
    println(f"SB $name%-22s $m%-5s median ${med(m)}%7.3f s (min ${ts.head}%.3f max ${ts.last}%.3f)")
  }
  setMode("FUSED"); FusionStats.reset(); try { run(q) } catch { case _: Throwable => }
  println(s"SB $name fusedBatches(one run)=${FusionStats.fusedBatches.get} fallback=${FusionStats.fallbackBatches.get} rtFail=${FusionStats.runtimeFailures.get}")
  if (med.size == 3) tab += f"$name%-22s OFF ${med("OFF")}%7.3f  AST ${med("AST")}%7.3f  FUSED ${med("FUSED")}%7.3f  fused-vs-off ${100 * (med("OFF") - med("FUSED")) / med("OFF")}%6.1f%%  fused-vs-ast ${100 * (med("AST") - med("FUSED")) / med("AST")}%6.1f%%"
}
println("SB-TABLE"); tab.foreach(l => println("SB-TABLE " + l))
sys.exit(0)
