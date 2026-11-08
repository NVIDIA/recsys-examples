// Round/BRound stress: fused (AGGRESSIVE) vs unfused GPU, bit-exact, wide magnitude range and many scales.
import com.nvidia.spark.rapids.fusion._
val rnd = new scala.util.Random(sys.props.getOrElse("rs.seed", "7").toLong)
val N = sys.props.getOrElse("rs.n", "300000").toInt
val vals = (0 until N).map { i =>
  val kind = rnd.nextInt(10)
  val d: java.lang.Double = kind match {
    case 0 => Double.NaN; case 1 => Double.PositiveInfinity; case 2 => Double.NegativeInfinity; case 3 => -0.0; case 4 => 0.0
    case 5 => rnd.nextGaussian() * math.pow(10, rnd.nextInt(40) - 20)
    case 6 => (rnd.nextInt(2000000) - 1000000) / 1000.0 + 0.0005   // many exact .5 ties after scaling
    case 7 => (rnd.nextInt(200000) - 100000) * 0.125
    case 8 => (if (rnd.nextBoolean()) 1 else -1) * math.pow(10, rnd.nextInt(600) - 300) * rnd.nextDouble()
    case _ => if (rnd.nextInt(20) == 0) null else rnd.nextGaussian() * 1000
  }
  (i, d, if (d == null) null.asInstanceOf[java.lang.Float] else java.lang.Float.valueOf(d.floatValue()))
}
import spark.implicits._
val dir = sys.env.getOrElse("FUSION_WORK", "work") + "/scratch/roundstress_" + java.lang.management.ManagementFactory.getRuntimeMXBean.getName.replace("@", "_")
vals.toDF("id", "d", "f").repartition(3).write.mode("overwrite").parquet(dir)   // parquet: a LocalRelation would be folded on the CPU
spark.read.parquet(dir).createOrReplaceTempView("t")
val scales = Seq(-6, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 17, 25)
val cols = scales.flatMap(s => Seq(s"round(d, $s)", s"bround(d, $s)", s"round(f, $s)", s"bround(f, $s)")).zipWithIndex.map { case (e, i) => s"$e as c$i" }
val q = "select id, " + cols.mkString(", ") + " from t"
def bits(a: Any): Any = a match { case x: java.lang.Double => if (x.isNaN) "NaN" else x.doubleValue.toString + "/" + java.lang.Double.doubleToRawLongBits(x); case x: java.lang.Float => if (x.isNaN) "NaN" else x.floatValue.toString + "/" + java.lang.Float.floatToRawIntBits(x); case o => o }
spark.conf.set("spark.rapids.sql.fusion.force", "true")
spark.conf.set("spark.rapids.sql.fusion.mode", "OFF"); val u = spark.sql(q).collect().map(r => r.getInt(0) -> (1 until r.length).map(i => bits(r.get(i)))).toMap
spark.conf.set("spark.rapids.sql.fusion.mode", "AGGRESSIVE"); FusionStats.reset(); val f = spark.sql(q).collect().map(r => r.getInt(0) -> (1 until r.length).map(i => bits(r.get(i)))).toMap
var bad = 0; val badCols = scala.collection.mutable.Map[Int, Int]()
for ((id, uv) <- u; (x, c) <- uv.zip(f(id)).zipWithIndex if x._1 != x._2) { bad += 1; badCols(c) = badCols.getOrElse(c, 0) + 1; if (bad <= 5) println(s"ROUNDSTRESS mismatch id=$id col=${cols(c)} unfused=${x._1} fused=${x._2} input=${vals(id)._2}") }
println(s"ROUNDSTRESS rows=${u.size} cols=${cols.size} mismatches=$bad byCol=${badCols.toSeq.sortBy(-_._2).take(8).map { case (c, n) => cols(c).split(" as ")(0) + ":" + n }} ${FusionStats}")
sys.exit(0)
