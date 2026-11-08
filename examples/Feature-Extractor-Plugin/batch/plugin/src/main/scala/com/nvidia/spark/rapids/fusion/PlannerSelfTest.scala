package com.nvidia.spark.rapids.fusion

import com.nvidia.spark.rapids.{GpuAlias, GpuIf, GpuLiteral, GpuCast, GpuIsNull, GpuCoalesce}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.types._

/**
 * No-GPU unit tests of the planner side: spark-rapids expression trees -> IR (Translator), refusal reasons and the
 * code generator's structural properties. Run: scripts/selftest.sh (runs this too).
 */
object PlannerSelfTest {
  private var failures = 0
  private var checks = 0
  private def check(name: String)(cond: => Boolean): Unit = {
    checks += 1
    val ok = try cond catch { case t: Throwable => println(s"  exception: $t"); false }
    if (!ok) { failures += 1; println(s"FAIL $name") }
  }

  private val cfg = FusionConf(mode = "SAFE", kill = false, explain = "NONE", shadow = false, shadowFailOnMismatch = false,
    shadowEvery = 1, minOps = 2, maxNodes = 256, maxInputs = 48, allowOpaque = true, fuseProject = true, fuseFilter = true,
    fuseAggregate = true, mergeProjects = true, syncAfterLaunch = true, nvrtcPath = "", cacheDir = "", disabledExprs = Set.empty,
    disabledOps = Set.empty, forceFuse = false)

  private def attr(n: String, t: DataType, nullable: Boolean = true) = AttributeReference(n, t, nullable)()
  private val a = attr("a", DoubleType); private val b = attr("b", DoubleType); private val i = attr("i", IntegerType)
  private val s = attr("s", StringType); private val d = attr("d", DateType); private val l = attr("l", LongType)
  private val child = Seq(a, b, i, s, d, l)
  private def lit(v: Any, t: DataType) = GpuLiteral(v, t)

  private def translate(e: Expression, c: FusionConf = cfg): (Option[Ex], Option[String], Translator) = {
    val tr = new Translator(child, c)
    try { (Some(tr.translateRoot(e)), None, tr) } catch { case nf: NotFusible => (None, Some(nf.reason), tr) }
  }

  def main(args: Array[String]): Unit = {
    // 1. supported trees translate to the expected IR
    val add = GpuAdd(a, b, false)(org.apache.spark.sql.catalyst.trees.Origin())
    check("add(a,b) -> Call(add)") { translate(add)._1.exists { case Call("add", Seq(InCol(0, FT.F64), InCol(1, FT.F64)), FT.F64, _) => true; case _ => false } }
    val mul = GpuMultiply(add, lit(2.0, DoubleType), false)(org.apache.spark.sql.catalyst.trees.Origin())
    check("alias(mul(add(a,b),2.0)) nested") { translate(GpuAlias(mul, "x")())._1.exists { case Call("mul", Seq(Call("add", _, _, _), Lit(_, FT.F64)), _, _) => true; case _ => false } }
    check("comparison, and/or, not") { translate(GpuAnd(GpuGreaterThan(a, b), GpuNot(GpuIsNull(a))))._1.exists(_.ft == FT.Bool) }
    check("if / coalesce / cast / isnull") {
      translate(GpuIf(GpuGreaterThan(a, b), a, GpuCoalesce(Seq(b, lit(1.0, DoubleType)))))._1.isDefined &&
        translate(GpuCast(i, DoubleType, false, None, false, false)(org.apache.spark.sql.catalyst.trees.Origin()))._1.isDefined
    }
    // 2. refusals carry readable reasons
    val ansiAdd = GpuAdd(i, i, true)(org.apache.spark.sql.catalyst.trees.Origin())
    check("ANSI add refused") { translate(ansiAdd)._2.exists(_.contains("ANSI")) }
    check("string column not fusable alone") { translate(s)._1.isEmpty }
    check("int add with string operand becomes opaque input or is refused") {
      val e = GpuAdd(i, GpuCast(GpuLength(s), IntegerType, false, None, false, false)(org.apache.spark.sql.catalyst.trees.Origin()), false)(org.apache.spark.sql.catalyst.trees.Origin())
      val (ex, why, tr) = translate(e)
      ex.isDefined || why.isDefined
    }
    check("expression disabled by conf") { translate(add, cfg.copy(disabledExprs = Set("GpuAdd")))._2.exists(_.contains("disabled")) }
    check("unsupported class reported by name") { translate(GpuLength(s))._2.exists(_.contains("GpuLength")) }
    // 3. CSE and node budget count distinct operators only
    val sub1 = GpuMultiply(add, add, false)(org.apache.spark.sql.catalyst.trees.Origin())
    val big = (1 to 40).foldLeft[Expression](a) { (acc, _) => GpuAdd(GpuMultiply(add, add, false)(org.apache.spark.sql.catalyst.trees.Origin()), acc, false)(org.apache.spark.sql.catalyst.trees.Origin()) }
    check("repeated sub-trees do not exhaust the budget") { translate(big, cfg.copy(maxNodes = 90))._1.isDefined }
    check("budget exhausted by distinct operators") { translate(big, cfg.copy(maxNodes = 10))._2.exists(_.contains("exceeds")) }
    // 4. literal identity is bitwise (-0.0 != 0.0, NaN == NaN)
    check("Lit(-0.0) != Lit(0.0)") { Lit(java.lang.Double.valueOf(-0.0), FT.F64) != Lit(java.lang.Double.valueOf(0.0), FT.F64) }
    check("Lit(NaN) == Lit(NaN)") { Lit(java.lang.Double.valueOf(Double.NaN), FT.F64) == Lit(java.lang.Double.valueOf(Double.NaN), FT.F64) }
    // 5. codegen structure
    val spec1 = CodeGen.build(Seq(Call("add", Seq(InCol(0, FT.F64), InCol(1, FT.F64)), FT.F64)))
    check("kernel has 2 inputs 1 output") { spec1.inputs.size == 2 && spec1.outputs.size == 1 && spec1.source.contains("__dadd_rn") }
    check("same IR -> same kernel name (cache key)") { spec1.name == CodeGen.build(Seq(Call("add", Seq(InCol(0, FT.F64), InCol(1, FT.F64)), FT.F64))).name }
    check("different literal sign -> different kernel") {
      CodeGen.build(Seq(Call("add", Seq(InCol(0, FT.F64), Lit(java.lang.Double.valueOf(0.0), FT.F64)), FT.F64))).name !=
        CodeGen.build(Seq(Call("add", Seq(InCol(0, FT.F64), Lit(java.lang.Double.valueOf(-0.0), FT.F64)), FT.F64))).name
    }
    check("guard ops flag the kernel") { CodeGen.build(Seq(Call("year", Seq(Call("civil", Seq(InCol(0, FT.Date)), FT.I64)), FT.I32))).usesGuard }
    check("narrow ints use 32-bit registers") { CodeGen.build(Seq(Call("neg", Seq(InCol(0, FT.I16)), FT.I16))).source.contains("norm16") }
    check("filter mask output never carries validity") { val sp = CodeGen.build(Seq(Call("tomask", Seq(Call("gt", Seq(InCol(0, FT.F64), InCol(1, FT.F64)), FT.Bool)), FT.Bool)), "filt"); !sp.outputs.head.selfNull && sp.outputs.head.deps.isEmpty }
    check("division creates nulls") { CodeGen.build(Seq(Call("div", Seq(InCol(0, FT.F64), InCol(1, FT.F64)), FT.F64))).outputs.head.selfNull }
    println(s"PLANNER-SELFTEST checks=$checks failures=$failures")
    sys.exit(if (failures == 0) 0 else 1)
  }
}
