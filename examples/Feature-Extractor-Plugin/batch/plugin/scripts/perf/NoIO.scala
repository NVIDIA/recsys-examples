// End-to-end with NO file scan: data generated on the GPU (GpuRange), so expression work dominates. OFF vs AST vs FUSED, interleaved, medians.
import com.nvidia.spark.rapids.fusion._
val n = sys.props.getOrElse("noio.rows", "300000000").toLong
val reps = sys.props.getOrElse("noio.reps", "5").toInt
val base = "select id, cast(id % 1000 as double) * 0.37 as a, cast(id % 977 as double) as b, cast((id * 7) % 1013 as double) as c, cast(id % 17 as double) + 1.5 as d, cast(id % 5 as int) as k from range(0, " + n + ")"
spark.sql(base).createOrReplaceTempView("t0")
val queries = Seq(
  "A arith chain sum" -> "select sum(((a*b + c) * d - a) / (d + 1.5) + (b - a) * (c + 2.0) - (b*c - d*a) + a*a*0.001) from t0",
  "B case-when sum" -> "select sum(case when a > 300 then a*2 when a > 200 then b*3 when a > 100 then c + d else a - b end + case when b > 500 then 1 else 0 end * a) from t0",
  "C math sum" -> "select sum(sqrt(a + 1.0) * exp(-b / 1000.0) + log(c + 1.0) + sin(d) * cos(a)) from t0",
  "D filter+agg" -> "select count(*), sum(a*b + c*d - a) from t0 where a*b > 100.0 and c < 900.0 and (d + a) > 20.0",
  "E groupby arith" -> "select k, sum(a*b + c), avg((d - a) * (b + 1)), max(c * d + a) from t0 group by k")
def setConf(mode: String): Unit = {
  spark.conf.set("spark.rapids.sql.projectAstEnabled", (mode == "AST").toString)
  spark.conf.set("spark.rapids.sql.fusion.mode", if (mode == "FUSED") "SAFE" else "OFF")
}
def runQ(q: String): Double = { val t0 = System.nanoTime(); spark.sql(q).collect(); (System.nanoTime() - t0) / 1e9 }
println(s"NOIO rows=$n reps=$reps")
// correctness: same results (sums compared with a relative tolerance: GPU double sums are not deterministic in order)
def flat(rows: Array[org.apache.spark.sql.Row]): Seq[Double] = rows.sortBy(_.toString).flatMap(r => (0 until r.length).map(i => r.get(i) match { case d: java.lang.Double => d.doubleValue; case n: java.lang.Number => n.doubleValue; case _ => Double.NaN })).toSeq
for ((name, q) <- queries) {
  setConf("OFF"); val u = flat(spark.sql(q).collect()); setConf("FUSED"); val f = flat(spark.sql(q).collect())
  val ok = u.size == f.size && u.zip(f).forall { case (x, y) => (x.isNaN && y.isNaN) || math.abs(x - y) <= 1e-9 * math.max(1.0, math.abs(x)) }
  println(s"NOIO-CHECK $name results equal (tol 1e-9): $ok")
}
val tab = scala.collection.mutable.ArrayBuffer[String]()
for ((name, q) <- queries) {
  val times = Map("OFF" -> new scala.collection.mutable.ArrayBuffer[Double], "AST" -> new scala.collection.mutable.ArrayBuffer[Double], "FUSED" -> new scala.collection.mutable.ArrayBuffer[Double])
  for (round <- 0 to reps; m <- Seq("OFF", "AST", "FUSED")) { setConf(m); val t = runQ(q); if (round > 0) times(m) += t }
  val med = Seq("OFF", "AST", "FUSED").map { m => val ts = times(m).sorted; (m, ts(ts.length / 2)) }.toMap
  tab += f"$name%-20s OFF ${med("OFF")}%7.3f  AST ${med("AST")}%7.3f  FUSED ${med("FUSED")}%7.3f  fused-vs-off ${100 * (med("OFF") - med("FUSED")) / med("OFF")}%6.1f%%  fused-vs-ast ${100 * (med("AST") - med("FUSED")) / med("AST")}%6.1f%%  speedup ${med("OFF") / med("FUSED")}%.2fx"
  println("NOIO-ROW " + tab.last)
}
sys.exit(0)
