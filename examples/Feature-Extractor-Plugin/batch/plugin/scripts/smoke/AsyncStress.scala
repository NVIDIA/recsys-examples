// asyncCompile stress (spark-shell -i): injected compile failure, slow compile (mixed fused/unfused batches in one query),
// concurrent queries / concurrent tasks asking for the same new kernel, retry-OOM injection. Everything is compared bit-for-bit with unfused GPU.
// Run:  run-spark.sh spark-shell --conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension -n 8 -i AsyncStress.scala
import com.nvidia.spark.rapids.fusion._
import org.apache.spark.sql.Row
import scala.concurrent._, scala.concurrent.duration._
import ExecutionContext.Implicits.global

val src = sys.env.getOrElse("FUSION_WORK", "work") + "/data/perf_t"
val tmp = s"/tmp/${sys.env.getOrElse("USER", "u")}-asyncstress"
spark.read.parquet(src).limit(600000).repartition(240).write.mode("overwrite").parquet(tmp)
spark.read.parquet(tmp).createOrReplaceTempView("t")
println("ASYNCSTRESS columns: " + spark.table("t").columns.mkString(","))
spark.conf.set("spark.rapids.sql.fusion.force", "true")
spark.conf.set("spark.rapids.sql.fusion.cacheDir", "")       // no disk cache: every new region really compiles
spark.conf.set("spark.sql.adaptive.enabled", "false")
spark.conf.set("spark.rapids.sql.reader.batchSizeRows", "20000")   // many batches per task => batches before/after the compile finishes
def setMode(m: String, async: Boolean): Unit = { spark.conf.set("spark.rapids.sql.fusion.mode", m); spark.conf.set("spark.rapids.sql.fusion.asyncCompile", async.toString) }
def bitsEq(a: Any, b: Any): Boolean = (a, b) match {
  case (null, null) => true
  case (null, _) | (_, null) => false
  case (x: java.lang.Double, y: java.lang.Double) => (x.isNaN && y.isNaN) || java.lang.Double.doubleToRawLongBits(x) == java.lang.Double.doubleToRawLongBits(y)
  case (x: java.lang.Float, y: java.lang.Float) => (x.isNaN && y.isNaN) || java.lang.Float.floatToRawIntBits(x) == java.lang.Float.floatToRawIntBits(y)
  case _ => a == b
}
def same(x: Array[Row], y: Array[Row]): Boolean = x.length == y.length && x.indices.forall(i => (0 until x(i).length).forall(c => bitsEq(x(i).get(c), y(i).get(c))))
def run(q: String): Array[Row] = spark.sql(q).collect().sortBy(_.get(0).asInstanceOf[Number].longValue)
def mk(k: Int): String = k match {
  case 0 => "select id, a*b + c as r1, (a - b) * (c + d) as r2 from t"
  case 1 => "select id, case when e > 5 then f else g end as r1, sqrt(abs(a - b)) as r2 from t"
  case 2 => "select id, (a + b) * (a - b) - c * c + 7.0 as r1, abs(c - d) + abs(d - e) as r2 from t"
  case 3 => "select id, if(a > b, a - b, b - a) * 2.5 as r1, greatest(a, b, c) - least(a, b) as r2 from t"
  case 4 => "select id, a * 1.5D + b * 2.5D - c as r1, (d + e) / (f + 1.0D) as r2 from t"
  case 5 => "select id, exp(-a / 1000.0D) * c as r1, floor(b * 0.37D) + ceil(c * 0.11D) as r2 from t"
  case 6 => "select id, (a > b) and (c < 50.0D) as r1, (d < e) or (f > g) as r2 from t where a * b > 10.0D"
  case 7 => "select id, a - b * c + d * e - f * g + h as r1, a * a - b * b as r2 from t"
  case 8 => "select id, pow(abs(a) + 1.0D, 0.5D) + log(abs(b) + 2.0D) as r1, hypot(a, b) as r2 from t"
  case _ => s"select id, a * ${k}.25D + b * ${k}.5D - c * 0.${k}1D as r1, d + e * ${k} as r2 from t"
}
var bad = 0
def check(name: String, ok: Boolean, extra: String = ""): Unit = { if (!ok) bad += 1; println(s"ASYNCSTRESS ${if (ok) "PASS" else "FAIL"} $name $extra") }
setMode("OFF", false)
val ref = (0 until 20).map(k => run(mk(k)))

// P1: slow compile -> batches of ONE query run partly unfused (compile pending), partly fused; result identical, one compile per kernel
var mixedSeen = false
for ((delay, k) <- Seq(40, 100, 200, 400).zip(Seq(10, 11, 12, 13))) {
  System.setProperty("fusion.test.compileDelayMs", delay.toString); setMode("SAFE", true)
  FusionStats.reset(); val s0 = NativeBridge.asyncSubmitted.get
  val r = run(mk(k)); val f = FusionStats.fusedBatches.get; val fb = FusionStats.fallbackBatches.get
  val mixed = f > 0 && fb > 0; mixedSeen ||= mixed
  check(s"P1 slow-compile delay=${delay}ms equal-to-unfused", same(r, ref(k)), s"fused=$f fallback=$fb asyncFallbacks=${FusionStats.asyncFallbacks.get} compilesSubmitted=${NativeBridge.asyncSubmitted.get - s0} mixedInOneQuery=$mixed")
  check(s"P1 delay=${delay}ms one-compile-per-kernel", NativeBridge.asyncSubmitted.get - s0 == 1)
  NativeBridge.awaitCompiles(60000)
  FusionStats.reset(); val r2 = run(mk(k)); check(s"P1 delay=${delay}ms warm run fully fused & equal", same(r2, ref(k)) && FusionStats.fallbackBatches.get == 0 && FusionStats.fusedBatches.get > 0, s"fused=${FusionStats.fusedBatches.get} fallback=${FusionStats.fallbackBatches.get}")
}
System.clearProperty("fusion.test.compileDelayMs")
check("P1 at least one query had both fused and unfused batches", mixedSeen)

// P2: injected compile failure -> region blacklisted through the normal path, results unchanged, no repeated compile attempts
System.setProperty("fusion.test.failCompile", ".*"); setMode("SAFE", true)
FusionStats.reset(); val sub0 = NativeBridge.asyncSubmitted.get
val r1 = run(mk(14)); NativeBridge.awaitCompiles(60000)
val r2 = run(mk(14)); val r3 = run(mk(14))
check("P2 injected compile failure: results equal unfused (3 runs)", same(r1, ref(14)) && same(r2, ref(14)) && same(r3, ref(14)), s"stats: $FusionStats blacklist=${FusionRuntime.blacklist.size} asyncFailures=${NativeBridge.asyncFailures.size}")
check("P2 failure recorded, region blacklisted, compile attempted once", NativeBridge.asyncFailures.size >= 1 && !FusionRuntime.blacklist.isEmpty && NativeBridge.asyncSubmitted.get - sub0 == 1 && FusionStats.compileFailures.get >= 1, s"submitted=${NativeBridge.asyncSubmitted.get - sub0}")
// recovery after fixing the cause (test only: clear blacklist and failure memory)
System.clearProperty("fusion.test.failCompile"); FusionRuntime.blacklist.clear(); NativeBridge.asyncFailures.clear()
run(mk(14)); NativeBridge.awaitCompiles(60000); FusionStats.reset(); val r4 = run(mk(14))
check("P2 recovery after clearing the failure: fused again & equal", same(r4, ref(14)) && FusionStats.fusedBatches.get > 0 && FusionStats.fallbackBatches.get == 0)

// P3a: 8 queries with different new regions at the same time (8 driver threads x 8 tasks), results equal; P3b: same new kernel from 8 threads: ONE compile
setMode("SAFE", true)
val s3 = NativeBridge.asyncSubmitted.get; val ki3 = NativeBridge.kernelInfo.size; FusionStats.reset()
val futs = (0 until 8).map(i => Future { (i, (0 until 3).map(_ => run(mk(i)))) })
val res = Await.result(Future.sequence(futs), 300.seconds)
check("P3a concurrent distinct regions: all results equal unfused", res.forall { case (i, rs) => rs.forall(r => same(r, ref(i))) }, s"compilesSubmitted=${NativeBridge.asyncSubmitted.get - s3} fused=${FusionStats.fusedBatches.get} fallback=${FusionStats.fallbackBatches.get}")
NativeBridge.awaitCompiles(60000)
check("P3a exactly one compile per distinct kernel (compiles == new kernels; 9 because query 6 has a filter kernel and a project kernel)", NativeBridge.asyncSubmitted.get - s3 == NativeBridge.kernelInfo.size - ki3 && NativeBridge.asyncSubmitted.get - s3 == 9, s"submitted=${NativeBridge.asyncSubmitted.get - s3} newKernels=${NativeBridge.kernelInfo.size - ki3}")
NativeBridge.awaitCompiles(60000)
val s3b = NativeBridge.asyncSubmitted.get; FusionStats.reset()
System.setProperty("fusion.test.compileDelayMs", "150")
val futs2 = (0 until 8).map(_ => Future { run(mk(15)) })
val res2 = Await.result(Future.sequence(futs2), 300.seconds); System.clearProperty("fusion.test.compileDelayMs")
check("P3b same new kernel requested by 8 threads: one compile, equal results", NativeBridge.asyncSubmitted.get - s3b == 1 && res2.forall(r => same(r, ref(15))), s"compilesSubmitted=${NativeBridge.asyncSubmitted.get - s3b} fused=${FusionStats.fusedBatches.get} fallback=${FusionStats.fallbackBatches.get}")
NativeBridge.awaitCompiles(60000)
FusionStats.reset(); val warm = (0 until 8).forall(i => same(run(mk(i)), ref(i)))
check("P3 warm re-run of all 8 regions: equal, no fallback", warm && FusionStats.fallbackBatches.get == 0, s"fused=${FusionStats.fusedBatches.get} fallback=${FusionStats.fallbackBatches.get}")

// P4: retry / split-and-retry OOM injection while async fallback and fused batches interleave (new regions)
for ((inject, k) <- Seq("num_ooms=1,skip=0,type=GPU,split=false", "num_ooms=2,skip=1,type=GPU,split=true").zip(Seq(16, 17))) {
  spark.conf.set("spark.rapids.sql.test.injectRetryOOM", inject)
  setMode("OFF", false); val b = run(mk(k)); setMode("SAFE", true); FusionStats.reset()
  val c1 = run(mk(k)); NativeBridge.awaitCompiles(60000); val c2 = run(mk(k))
  check(s"P4 retry inject [$inject] cold+warm equal unfused", same(c1, b) && same(c2, b), s"stats: $FusionStats")
}
spark.conf.set("spark.rapids.sql.test.injectRetryOOM", "false")
println(s"ASYNCSTRESS SUMMARY failures=$bad asyncSubmitted=${NativeBridge.asyncSubmitted.get} asyncFailuresNow=${NativeBridge.asyncFailures.size}")
sys.exit(if (bad == 0) 0 else 1)
