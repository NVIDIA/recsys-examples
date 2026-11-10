// Differential smoke battery (spark-shell -i). Compares, per expression: fused GPU vs unfused GPU (strict bits)
// and fused GPU vs CPU Spark (bits, NaN-insensitive). Prints a table; exits non-zero via println MISMATCH lines.
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.types._
import com.nvidia.spark.rapids.fusion._

val batteryDir = sys.props.getOrElse("battery.dir", sys.env.getOrElse("FUSION_WORK", "work") + "/scratch/battery_data_" + java.lang.management.ManagementFactory.getRuntimeMXBean.getName.replace("@", "_"))
val N = sys.props.getOrElse("battery.rows", "60000").toInt
val rnd = new scala.util.Random(42)
def nul[T](p: Double)(f: => T): Any = if (rnd.nextDouble() < p) null else f
val i8s = Seq[Byte](Byte.MinValue, Byte.MaxValue, 0, -1, 1, 7, -7, 100, -100)
val i16s = Seq[Short](Short.MinValue, Short.MaxValue, 0, -1, 1, 300, -300)
val i32s = Seq[Int](Int.MinValue, Int.MaxValue, 0, -1, 1, 1000000, -1000000, 65536)
val i64s = Seq[Long](Long.MinValue, Long.MaxValue, 0L, -1L, 1L, 1L << 40, -(1L << 40), 1000000000000L)
val f32s = Seq[Float](Float.NaN, Float.PositiveInfinity, Float.NegativeInfinity, 0f, -0f, 1f, -1f, Float.MinPositiveValue, Float.MaxValue, -Float.MaxValue, 0.5f, -0.5f, 1.5f, 2.5f, 3.5f, -2.5f, 1e-30f, 123456.789f)
val f64s = Seq[Double](Double.NaN, Double.PositiveInfinity, Double.NegativeInfinity, 0.0, -0.0, 1.0, -1.0, Double.MinPositiveValue, Double.MaxValue, -Double.MaxValue, 0.5, -0.5, 1.5, 2.5, 3.5, -2.5, 1e-300, 123456.789, 1e18, -1e18, 9.3e18, 2147483648.0, -2147483649.0)
def pick[T](s: Seq[T]): T = s(rnd.nextInt(s.size))
val rows = (0 until N).map { i =>
  Row(i,
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(i8s) else rnd.nextInt(256).toByte),
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(i16s) else rnd.nextInt(65536).toShort),
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(i32s) else rnd.nextInt()),
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(i64s) else rnd.nextLong()),
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(f32s) else (rnd.nextGaussian() * 1000).toFloat),
    nul(0.15)(if (rnd.nextDouble() < 0.4) pick(f64s) else rnd.nextGaussian() * 1000),
    nul(0.15)(rnd.nextBoolean()),
    nul(0.15)(java.sql.Date.valueOf(java.time.LocalDate.ofEpochDay(rnd.nextInt(40000) - 5000L))),
    nul(0.15)(new java.sql.Timestamp(Math.floorMod(rnd.nextLong(), 4000000000000L) + 1L)),
    nul(0.0)(rnd.nextInt(1000).toString)  // string col: forces opaque/passthrough paths
  )
}
val schema = StructType(Seq(StructField("id", IntegerType, false), StructField("i8", ByteType), StructField("i16", ShortType),
  StructField("i32", IntegerType), StructField("i64", LongType), StructField("f32", FloatType), StructField("f64", DoubleType),
  StructField("b", BooleanType), StructField("d", DateType), StructField("ts", TimestampType), StructField("s", StringType)))
val base = spark.createDataFrame(spark.sparkContext.parallelize(rows, 4), schema)
base.write.mode("overwrite").parquet(batteryDir)
val t = spark.read.parquet(batteryDir).repartition(4)
t.createOrReplaceTempView("t")
if (sys.props.getOrElse("battery.nonull", "false") == "true") {
  // same data without any NULLs: no validity buffers on the inputs (exercises the "skip validity" paths)
  spark.sql("select id, coalesce(i8, cast(1 as tinyint)) i8, coalesce(i16, cast(2 as smallint)) i16, coalesce(i32, 3) i32, coalesce(i64, 4L) i64, coalesce(f32, cast(0.5 as float)) f32, coalesce(f64, 0.25) f64, coalesce(b, true) b, coalesce(d, date'2000-01-01') d, coalesce(ts, timestamp'2000-01-01 00:00:00') ts, s from t").write.mode("overwrite").parquet(batteryDir + "_nn")
  spark.read.parquet(batteryDir + "_nn").repartition(4).createOrReplaceTempView("t")
}

// -Dbattery.async=true: asyncCompile on. Per template: cold run (new region: batches fall back to the unfused operator while NVRTC compiles in the
// background) is compared with unfused too; then we wait for the background compile and the warm run (really fused) is compared as usual.
val asyncMode = sys.props.getOrElse("battery.async", "false") == "true"
var nAsyncCold = 0; var nAsyncColdBad = 0
def setMode(m: String, gpu: Boolean = true, extra: Map[String, String] = Map()): Unit = {
  spark.conf.set("spark.rapids.sql.enabled", gpu.toString)
  spark.conf.set("spark.rapids.sql.fusion.mode", m)
  spark.conf.set("spark.rapids.sql.fusion.force", "true")
  spark.conf.set("spark.rapids.sql.fusion.asyncCompile", asyncMode.toString)
  extra.foreach { case (k, v) => spark.conf.set(k, v) }
}
def bitsEq(a: Any, b: Any, nanEq: Boolean): Boolean = (a, b) match {
  case (null, null) => true
  case (null, _) | (_, null) => false
  case (x: java.lang.Double, y: java.lang.Double) => (x.isNaN && y.isNaN) || java.lang.Double.doubleToRawLongBits(x) == java.lang.Double.doubleToRawLongBits(y)
  case (x: java.lang.Float, y: java.lang.Float) => (x.isNaN && y.isNaN) || java.lang.Float.floatToRawIntBits(x) == java.lang.Float.floatToRawIntBits(y)
  case (x: Array[Byte], y: Array[Byte]) => x.sameElements(y)
  case _ => a == b
}
def run(sql: String): Array[Row] = spark.sql(sql).collect()
def cmp(x: Array[Row], y: Array[Row]): Option[String] = {
  if (x.length != y.length) return Some(s"row count ${x.length} vs ${y.length}")
  def key(r: Row): String = String.valueOf(r.get(0))
  val xs = x.sortBy(key); val ys = y.sortBy(key)
  var i = 0
  while (i < xs.length) {
    var c = 0
    while (c < xs(i).length) {
      if (!bitsEq(xs(i).get(c), ys(i).get(c), true)) return Some(s"id=${xs(i).get(0)} col$c ${xs(i).get(c)} vs ${ys(i).get(c)}")
      c += 1
    }
    i += 1
  }
  None
}

val exprs: Seq[String] = scala.io.Source.fromFile(sys.props.getOrElse("battery.exprs", sys.env.getOrElse("FUSION_HOME", ".") + "/scripts/smoke/exprs.txt")).getLines().map(_.trim).filter(l => l.nonEmpty && !l.startsWith("#")).toSeq
var nFusedOk = 0; var nNotFused = 0; var nBad = 0; var nCpuDiff = 0
val sb = new StringBuilder
var bprog = 0
for (e <- exprs) {
  bprog += 1; println(s"BPROG $bprog/${exprs.size} $e")
  val where = if (e.contains(" WHERE ")) "" else ""
  val q = if (e.startsWith("FILTER ")) s"select id from t where ${e.stripPrefix("FILTER ")}"
          else if (e.startsWith("AGG ")) { val x = e.stripPrefix("AGG "); s"select count(*) as c, count($x) as ce, min($x) as mn, max($x) as mx, sum(case when ($x) is null then 0 else 1 end) as nn from t" }
          else if (e.startsWith("GRP ")) { val x = e.stripPrefix("GRP "); s"select cast(abs(id) % 7 as long) as g, count(*) as c, min($x) as mn, max($x) as mx from t group by cast(abs(id) % 7 as long) order by g" }
          else if (e.startsWith("GRPKEY ")) { val x = e.stripPrefix("GRPKEY "); s"select ($x) as k, count(*) as c from t group by ($x) order by k" }
          else s"select id, $e from t"
  try {
    println("BPHASE off"); setMode("OFF"); val unf = run(q)
    setMode(sys.props.getOrElse("battery.mode", "SAFE"))
    if (asyncMode) {
      println("BPHASE cold"); FusionStats.reset(); val cold = run(q); nAsyncCold += 1
      val dc = cmp(cold, unf); if (dc.isDefined) { nAsyncColdBad += 1; nBad += 1; println(s"MISMATCH[async-cold!=unfused] $e :: ${dc.get}") }
      if (!NativeBridge.awaitCompiles(60000)) println(s"ERROR async compile did not finish for $e")
    }
    println("BPHASE warm"); FusionStats.reset(); val fus = run(q); val st = (FusionStats.fusedBatches.get, FusionStats.fallbackBatches.get, FusionStats.runtimeFailures.get)
    setMode("OFF", gpu = false); val cpu = run(q)
    val d1 = cmp(fus, unf); val d2 = cmp(fus, cpu)
    val fusedFlag = if (st._1 > 0 && st._2 == 0) "FUSED" else if (st._1 > 0) "PARTIAL" else "not-fused"
    if (fusedFlag == "not-fused") nNotFused += 1 else nFusedOk += 1
    if (d1.isDefined) { nBad += 1; println(s"MISMATCH[fused!=unfused] $e :: ${d1.get}") }
    if (d2.isDefined) { nCpuDiff += 1; println(s"cpu-diff[fused!=cpu] $e :: ${d2.get}" + (if (d1.isEmpty && unf.length >= 0 && cmp(unf, cpu).isDefined) " (unfused GPU also differs)" else "")) }
    sb.append(f"$fusedFlag%-9s unf=${d1.isEmpty}%-5s cpu=${d2.isEmpty}%-5s $e\n")
  } catch { case t: Throwable => nBad += 1; println(s"ERROR $e :: $t") }
}
println(sb.toString)
println(s"BATTERY SUMMARY exprs=${exprs.size} fused=$nFusedOk notFused=$nNotFused fusedVsUnfusedMismatch=$nBad fusedVsCpuDiff=$nCpuDiff")
println(s"ASYNC cold runs=$nAsyncCold coldMismatch=$nAsyncColdBad asyncSubmitted=${NativeBridge.asyncSubmitted.get} asyncFailures=${NativeBridge.asyncFailures.size}")
println("STATS " + FusionStats)
sys.exit(0)
