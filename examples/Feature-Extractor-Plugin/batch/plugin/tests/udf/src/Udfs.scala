package fusiontest

import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions.{udf, col}
import org.apache.spark.sql.api.java.UDF2

case class Rec(id: Long, i1: Int, i2: Int, d1: Double, d2: Double, l1: Long)
case class Out(id: Long, v: Double)

/** Scala UDFs / typed Dataset code used by the 'writing styles' coverage experiment (tests/udf/coverage.py).
 *  Every UDF is also registered under a SQL name so that SQL and DataFrame styles can both call it. */
object Udfs {
  val arith = (a: Int, b: Int) => a * b + 1
  val branch = (x: Double) => if (x > 0) x * 2.0 else -x
  val branchy = (a: Int, b: Int) => if (a > b && b != 0) a - b else if (a < 0) -a else b
  val mathy = (x: Double) => math.sqrt(x * x + 1.0)
  val absmax = (a: Long, b: Long) => math.max(math.abs(a), math.abs(b))
  val boolu = (a: Int, b: Double) => a > 5 || b < 0.5
  val loopy = (n: Int) => { var s = 0; var i = 0; while (i < n && i < 10) { s += i; i += 1 }; s }
  val strlen = (s: String) => s.length
  val cap = 7
  val closure = (a: Int) => a + cap
  val optu = (a: Integer) => if (a == null) -1 else a + 1
  val mixedcast = (a: Int, b: Long, c: Double) => (a + b).toDouble * c

  def register(spark: SparkSession): Unit = {
    spark.udf.register("u_arith", arith)
    spark.udf.register("u_branch", branch)
    spark.udf.register("u_branchy", branchy)
    spark.udf.register("u_math", mathy)
    spark.udf.register("u_absmax", absmax)
    spark.udf.register("u_bool", boolu)
    spark.udf.register("u_loop", loopy)
    spark.udf.register("u_strlen", strlen)
    spark.udf.register("u_closure", closure)
    spark.udf.register("u_opt", optu)
    spark.udf.register("u_mixed", mixedcast)
    // Java UDF (class compiled from Java) registered through the Java API
    spark.udf.register("r_square", new RapidsSquare(), org.apache.spark.sql.types.DataTypes.DoubleType)
    spark.udf.register("j_arith", new JavaUdfArith(), org.apache.spark.sql.types.DataTypes.IntegerType)
  }

  // typed Dataset API (lambda goes through map/filter, not through Catalyst expressions)
  def typedMap(ds: Dataset[Rec], spark: SparkSession): Dataset[Out] = {
    import spark.implicits._
    ds.map(r => Out(r.id, r.d1 * 2.0 + r.i1))
  }
  def typedFilter(ds: Dataset[Rec], spark: SparkSession): Dataset[Rec] = {
    import spark.implicits._
    ds.filter(r => r.i1 > 10 && r.d1 < 100.0)
  }
  def typedMapDF(df: org.apache.spark.sql.DataFrame, spark: SparkSession) = typedMap(typedDs(df, spark), spark).toDF()
  def typedFilterDF(df: org.apache.spark.sql.DataFrame, spark: SparkSession) = typedFilter(typedDs(df, spark), spark).toDF()
  def typedDs(df: org.apache.spark.sql.DataFrame, spark: SparkSession): Dataset[Rec] = {
    import spark.implicits._
    df.selectExpr("id", "coalesce(i1,0) i1", "coalesce(i2,0) i2", "coalesce(d1,0.0) d1", "coalesce(d2,0.0) d2", "coalesce(l1,0L) l1").as[Rec]
  }
}
