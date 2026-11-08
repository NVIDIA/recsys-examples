// Retry / memory-pressure test: injected GPU (split-and-)retry OOMs + tiny pool; fused must equal unfused.
import com.nvidia.spark.rapids.fusion._
val dir = sys.env.getOrElse("FUSION_WORK", "work") + "/data/perf_t"
spark.read.parquet(dir).createOrReplaceTempView("t")
val q = "select id, a*b + c as r1, (a - b) * (c + d) as r2, case when e > 5 then f else g end as r3, sqrt(abs(a - b)) as r4 from t where a*b > 50.0 and h < 0.95"
def checksum(mode: String, inject: String): (Long, Double, Double) = {
  spark.conf.set("spark.rapids.sql.fusion.mode", mode)
  spark.conf.set("spark.rapids.sql.test.injectRetryOOM", inject)
  val r = spark.sql(q).repartition(4).selectExpr("count(*)", "sum(r1)", "sum(r2 + r3 + r4)").collect()(0)
  (r.getLong(0), r.getDouble(1), r.getDouble(2))
}
for (inject <- Seq("false", "num_ooms=1,skip=0,type=GPU,split=false", "num_ooms=2,skip=1,type=GPU,split=true")) {
  FusionStats.reset()
  val base = checksum("OFF", inject)
  FusionStats.reset()
  val fus = checksum("SAFE", inject)
  println(s"RETRY inject=[$inject] unfused=$base fused=$fus equal=${base == fus} stats: $FusionStats")
}
sys.exit(0)
