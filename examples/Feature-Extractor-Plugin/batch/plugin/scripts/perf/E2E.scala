// spark-shell -i: end-to-end timing. Modes: OFF (plain spark-rapids), AST (projectAstEnabled), FUSED. -Dperf.reps=N
import com.nvidia.spark.rapids.fusion._
val dir = sys.props.getOrElse("perf.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/data/perf_t")
val reps = sys.props.getOrElse("perf.reps", "7").toInt
spark.read.parquet(dir).createOrReplaceTempView("t")
val queries = Seq(
  "Q1 arith-heavy agg" -> "select sum(((a*b + c) * d - e) / (f + 1.5) + (g - a) * (h + 2.0) - (b*c - d*e) + a*a*0.001 + ((x*y) + x - y)) from t",
  "Q2 case-when chain" -> "select sum(case when a > 80 then a*2 when a > 60 then b*3 when a > 40 then c + d when a > 20 then e - f else g + h end + case when b > 50 then 1 else 0 end * a) from t",
  "Q3 filter+project agg" -> "select count(*), sum(a*b + c*d - e) from t where a*b > 100.0 and c < 90.0 and (d + e) > 20.0 and h < 0.9",
  "Q4 groupby w/ exprs" -> "select k, sum(a*b + c) s1, avg((d - e) * (f + 1)) s2, max(g * h + a) m from t group by k",
  "Q5 math-heavy agg" -> "select sum(sqrt(a) * exp(-b / 100.0) + log(c + 1.0) + sin(d) * cos(e) + pow(f, 1.5)) from t",
  "Q6 light (a+1) agg" -> "select sum(a + 1.0) from t",
  "Q8 wide 12-term expr agg" -> "select sum((a*1.0 + b*c - d/2.0) * (e + 0.5) - sqrt(abs(f - g/3.0)) + (a*2.0 + b*c - d/3.0) * (e + 1.5) - sqrt(abs(f - g/4.0)) + (a*3.0 + b*c - d/4.0) * (e + 2.5) - sqrt(abs(f - g/5.0)) + (a*4.0 + b*c - d/5.0) * (e + 3.5) - sqrt(abs(f - g/6.0)) + (a*5.0 + b*c - d/6.0) * (e + 4.5) - sqrt(abs(f - g/7.0)) + (a*6.0 + b*c - d/7.0) * (e + 5.5) - sqrt(abs(f - g/8.0)) + (a*7.0 + b*c - d/8.0) * (e + 6.5) - sqrt(abs(f - g/9.0)) + (a*8.0 + b*c - d/9.0) * (e + 7.5) - sqrt(abs(f - g/10.0)) + (a*9.0 + b*c - d/10.0) * (e + 8.5) - sqrt(abs(f - g/11.0)) + (a*10.0 + b*c - d/11.0) * (e + 9.5) - sqrt(abs(f - g/12.0)) + (a*11.0 + b*c - d/12.0) * (e + 10.5) - sqrt(abs(f - g/13.0)) + (a*12.0 + b*c - d/13.0) * (e + 11.5) - sqrt(abs(f - g/14.0))) from t",
  "Q7 project only (noop write)" -> "select a*b + c as r1, (a - b) * (c + d) as r2, case when e > 5 then f else g end as r3, sqrt(abs(a - b)) as r4 from t"
)
def setConf(mode: String): Unit = {
  spark.conf.set("spark.rapids.sql.projectAstEnabled", (mode == "AST").toString)
  spark.conf.set("spark.rapids.sql.fusion.mode", if (mode == "FUSED") "SAFE" else "OFF")
}
def runQ(sql: String, noop: Boolean): Double = {
  val t0 = System.nanoTime()
  if (noop) spark.sql(sql).write.format("noop").mode("overwrite").save() else spark.sql(sql).collect()
  (System.nanoTime() - t0) / 1e9
}
println(s"E2E rows=${spark.read.parquet(dir).count()} reps=$reps")
// warm the file cache + JIT once
setConf("OFF"); runQ("select sum(a+b+c+d+e+f+g+h+x+y) from t", false)
val res = scala.collection.mutable.LinkedHashMap[String, Map[String, Double]]()
for ((name, q) <- queries) {
  val noop = name.startsWith("Q7")
  val times = scala.collection.mutable.Map[String, scala.collection.mutable.ArrayBuffer[Double]]("OFF" -> new scala.collection.mutable.ArrayBuffer[Double], "AST" -> new scala.collection.mutable.ArrayBuffer[Double], "FUSED" -> new scala.collection.mutable.ArrayBuffer[Double])
  for (round <- 0 to reps; mode <- Seq("OFF", "AST", "FUSED")) {   // interleaved; round 0 = warm-up for every mode
    setConf(mode); val t = runQ(q, noop); if (round > 0) times(mode) += t
  }
  var out = Map[String, Double]()
  for (mode <- Seq("OFF", "AST", "FUSED")) {
    val ts = times(mode).sorted; val med = ts(ts.length / 2); out += (mode -> med)
    println(f"E2E-RESULT $name%-30s $mode%-6s median ${med}%7.3f s  min ${ts.head}%7.3f  max ${ts.last}%7.3f")
  }
  res(name) = out
}
println("E2E-TABLE")
for ((n, m) <- res) println(f"$n%-30s OFF ${m("OFF")}%7.3f  AST ${m("AST")}%7.3f  FUSED ${m("FUSED")}%7.3f  fused-vs-off ${100 * (m("OFF") - m("FUSED")) / m("OFF")}%6.1f%%  fused-vs-ast ${100 * (m("AST") - m("FUSED")) / m("AST")}%6.1f%%")
sys.exit(0)
