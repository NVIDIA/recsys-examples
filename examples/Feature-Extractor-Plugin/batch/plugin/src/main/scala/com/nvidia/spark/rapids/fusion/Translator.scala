package com.nvidia.spark.rapids.fusion

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types._

/**
 * Turns spark-rapids GPU expression trees into the fusion IR. Matching is by class simple name so the plugin does
 * not link against individual expression classes; every supported expression has a "fusion contract" encoded here
 * (types, null semantics via the op table, no ANSI/overflow-throwing variants).
 */
final class Translator(childOutput: Seq[Attribute], cfg: FusionConf) {
  /** Sub-expressions evaluated by cuDF and fed to the kernel as columns. Slot index = position. */
  val opaques = ArrayBuffer[Expression]()
  private val opaqueSlots = scala.collection.mutable.HashMap[Expression, Int]()
  /** Reasons recorded for every cut / rejection (for explain output). */
  val notes = ArrayBuffer[String]()
  private var nodeBudget = cfg.maxNodes

  private def name(e: Expression): String = e.getClass.getSimpleName

  private def ft(e: Expression): FT = FT.fromSpark(e.dataType).getOrElse(
    throw new NotFusible(s"type ${e.dataType.simpleString} of ${name(e)}"))

  private def boolField(e: Expression, f: String): Option[Boolean] =
    try Some(e.getClass.getMethod(f).invoke(e).asInstanceOf[Boolean]) catch { case _: Throwable => None }

  private def isUtc(e: Expression): Boolean = {
    val z = try Some(e.getClass.getMethod("zoneId").invoke(e).asInstanceOf[java.time.ZoneId]) catch { case _: Throwable => None }
    z.forall(x => x.getRules.isFixedOffset && x.getRules.getOffset(java.time.Instant.EPOCH).getTotalSeconds == 0)
  }

  /** Translate a top-level output expression. Throws NotFusible if the whole root cannot be a fused op. */
  def translateRoot(e: Expression): Ex = fuse(e, underCond = false)

  /** Translate `e`; failures below the root become opaque inputs when allowed. */
  private def sub(e: Expression, underCond: Boolean): Ex =
    try fuse(e, underCond)
    catch {
      case nf: NotFusible =>
        if (!cfg.allowOpaque) throw nf
        FT.fromSpark(e.dataType) match {
          case Some(t) if e.deterministic && !hasLeafOnlyType(e) && opaqueOk(e, underCond) =>
            notes += s"cut: ${e.sql.take(80)} evaluated by cuDF (${nf.reason})"
            val slot = opaqueSlots.getOrElseUpdate(e.canonicalized, { opaques += e; opaques.size - 1 })
            Opaque(slot, t)
          case _ => throw nf
        }
    }

  private def hasLeafOnlyType(e: Expression): Boolean = false

  /**
   * spark-rapids flags some non-throwing expressions as "has side effects" (GpuFloor/GpuCeil on double, decimal ops in
   * non-ANSI mode inherit it from children, ...). Recompute: a node is truly side-effecting if it is flagged and either a
   * child truly is, or it carries an ANSI/failOn* switch that is on, or it is flagged without any known benign reason.
   */
  private def trulySideEffecting(e: Expression): Boolean = {
    if (!boolField(e, "hasSideEffects").getOrElse(false)) return false
    if (e.children.exists(trulySideEffecting)) return true
    val switches = Seq("failOnError", "ansiMode", "failOnDivideByZero").flatMap(f => boolField(e, f))
    if (switches.nonEmpty) switches.exists(identity)
    else name(e) match {
      case "GpuFloor" | "GpuCeil" => e.children.head.dataType match {
        case org.apache.spark.sql.types.DoubleType | org.apache.spark.sql.types.FloatType | org.apache.spark.sql.types.LongType => false
        case _ => true
      }
      case _ => true
    }
  }

  private def opaqueOk(e: Expression, underCond: Boolean): Boolean = {
    val side = trulySideEffecting(e)
    val retry = boolField(e, "retryable").getOrElse(true)
    retry && !(underCond && side)
  }

  private def attrIndex(a: AttributeReference): Int = {
    val i = childOutput.indexWhere(_.exprId == a.exprId)
    if (i < 0) throw new NotFusible(s"attribute ${a.name} not found in child output")
    i
  }

  /** Structurally identical nodes are one node of the kernel (CSE), so only distinct nodes count against the budget. */
  private val uniqueNodes = scala.collection.mutable.HashSet[Ex]()
  private def call(op: String, args: Seq[Ex], t: FT, attr: Seq[Any] = Nil): Ex = {
    val c = Call(op, args, t, attr)
    if (uniqueNodes.add(c)) {
      nodeBudget -= 1
      if (nodeBudget < 0) throw new NotFusible(s"region exceeds ${cfg.maxNodes} distinct operators")
    }
    c
  }

  private def sameNumeric(args: Seq[Ex], what: String): Unit = {
    if (args.map(_.ft).distinct.size != 1) throw new NotFusible(s"$what: mixed operand types ${args.map(_.ft)}")
  }

  private def noAnsi(e: Expression, what: String): Unit = {
    if (boolField(e, "failOnError").getOrElse(false)) throw new NotFusible(s"$what in ANSI mode (overflow/error check)")
  }

  private def fuse(e: Expression, underCond: Boolean): Ex = {
    val n = name(e)
    if (cfg.disabledExprs.contains(n)) throw new NotFusible(s"$n disabled by configuration")
    val ch = e.children
    def s(x: Expression): Ex = sub(x, underCond)
    n match {
      case "AttributeReference" => InCol(attrIndex(e.asInstanceOf[AttributeReference]), ft(e))
      case "GpuBoundReference" | "BoundReference" =>
        InCol(e.getClass.getMethod("ordinal").invoke(e).asInstanceOf[Int], ft(e))
      case "GpuAlias" | "Alias" => fuse(ch.head, underCond)
      case "GpuKnownFloatingPointNormalized" | "KnownFloatingPointNormalized" | "GpuKnownNotNull" | "KnownNotNull" |
           "GpuKnownNullable" | "KnownNullable" => fuse(ch.head, underCond)
      case "GpuLiteral" | "Literal" =>
        val v = e.getClass.getMethod("value").invoke(e)
        val t = ft(e)
        Lit(v, t)
      case "GpuAdd" | "GpuSubtract" | "GpuMultiply" =>
        noAnsi(e, n)
        val a = ch.map(s); sameNumeric(a, n)
        if (!a.head.ft.isNumeric) throw new NotFusible(s"$n on ${a.head.ft}")
        call(n.drop(3).toLowerCase.replace("subtract", "sub").replace("multiply", "mul"), a, ft(e))
      case "GpuUnaryMinus" | "GpuAbs" =>
        noAnsi(e, n)
        val a = ch.map(s)
        if (!a.head.ft.isNumeric) throw new NotFusible(s"$n on ${a.head.ft}")
        call(if (n == "GpuAbs") "abs" else "neg", a, ft(e))
      case "GpuUnaryPositive" => s(ch.head)
      case "GpuDivide" =>
        noAnsi(e, n)
        val a = ch.map(s)
        if (a.exists(_.ft != FT.F64)) throw new NotFusible("divide on non-double")
        call("div", a, FT.F64)
      case "GpuIntegralDivide" =>
        noAnsi(e, n)
        val a = ch.map(s); sameNumeric(a, n)
        if (!a.head.ft.isIntegral) throw new NotFusible("integral divide on non-integral")
        call("idiv", a, FT.I64)
      case "GpuRemainder" | "GpuPmod" =>
        noAnsi(e, n)
        val a = ch.map(s); sameNumeric(a, n)
        if (!a.head.ft.isNumeric) throw new NotFusible(s"$n on ${a.head.ft}")
        call(if (n == "GpuPmod") "pmod" else "rem", a, ft(e))
      case "GpuEqualTo" | "GpuLessThan" | "GpuLessThanOrEqual" | "GpuGreaterThan" | "GpuGreaterThanOrEqual" |
           "GpuEqualNullSafe" =>
        val a = ch.map(s); sameNumeric(a, n)
        val op = n match {
          case "GpuEqualTo" => "eq"; case "GpuLessThan" => "lt"; case "GpuLessThanOrEqual" => "le"
          case "GpuGreaterThan" => "gt"; case "GpuGreaterThanOrEqual" => "ge"; case _ => "eqns"
        }
        call(op, a, FT.Bool)
      case "GpuAnd" => call("and", ch.map(s), FT.Bool)
      case "GpuOr" => call("or", ch.map(s), FT.Bool)
      case "GpuNot" => call("not", ch.map(s), FT.Bool)
      case "GpuIsNull" => call("isnull", Seq(s(ch.head)), FT.Bool)
      case "GpuIsNotNull" => call("isnotnull", Seq(s(ch.head)), FT.Bool)
      case "GpuIsNan" =>
        val a = s(ch.head)
        if (!a.ft.isFloat) throw new NotFusible("isnan on non-float")
        call("isnan", Seq(a), FT.Bool)
      case "GpuIf" =>
        val p = s(ch(0)); val t = sub(ch(1), underCond = true); val f = sub(ch(2), underCond = true)
        if (t.ft != f.ft) throw new NotFusible("if branches differ in type")
        call("if", Seq(p, t, f), ft(e))
      case "GpuCaseWhen" =>
        val branches = e.getClass.getMethod("branches").invoke(e).asInstanceOf[Seq[(Expression, Expression)]]
        val elseV = e.getClass.getMethod("elseValue").invoke(e).asInstanceOf[Option[Expression]]
        val args = branches.flatMap { case (p, v) => Seq(sub(p, underCond), sub(v, underCond = true)) } ++
          elseV.map(sub(_, underCond = true))
        val t = ft(e)
        if (args.zipWithIndex.exists { case (x, i) => (i % 2 == 1 || (elseV.isDefined && i == args.size - 1)) && x.ft != t })
          throw new NotFusible("case-when branches differ in type")
        call("casewhen", args, t)
      case "GpuCoalesce" =>
        val a = ch.map(x => sub(x, underCond = underCond))
        if (a.map(_.ft).distinct.size != 1) throw new NotFusible("coalesce args differ in type")
        call("coalesce", a, ft(e))
      case "GpuGreatest" | "GpuLeast" =>
        val a = ch.map(s); sameNumeric(a, n)
        call(n.drop(3).toLowerCase, a, ft(e))
      case "GpuNormalizeNaNAndZero" =>
        val a = s(ch.head)
        if (!a.ft.isFloat) throw new NotFusible("normalize on non-float")
        call("normnz", Seq(a), a.ft)
      case "GpuCast" =>
        if (boolField(e, "ansiMode").getOrElse(false)) throw new NotFusible("ANSI cast")
        val a = s(ch.head); val to = ft(e)
        (a.ft, to) match {
          case (FT.Date, FT.Ts) | (FT.Ts, FT.Date) | (FT.Ts, _) | (_, FT.Ts) =>
            if (!isUtc(e)) throw new NotFusible("timestamp cast with non-UTC time zone")
          case _ =>
        }
        if (!Ops.castSupported(a.ft, to)) throw new NotFusible(s"cast ${a.ft} -> $to not supported")
        if (a.ft == to) a else call("cast", Seq(a), to)
      case "GpuBitwiseAnd" | "GpuBitwiseOr" | "GpuBitwiseXor" =>
        val a = ch.map(s); sameNumeric(a, n)
        if (!a.head.ft.isIntegral && a.head.ft != FT.Bool) throw new NotFusible(s"$n on ${a.head.ft}")
        call("bit_" + n.drop(10).toLowerCase, a, ft(e))
      case "GpuBitwiseNot" =>
        val a = s(ch.head)
        if (!a.ft.isIntegral) throw new NotFusible("bitwise not on non-integral")
        call("bit_not", Seq(a), a.ft)
      case "GpuShiftLeft" | "GpuShiftRight" | "GpuShiftRightUnsigned" =>
        val a = ch.map(s)
        if (!(a(0).ft == FT.I32 || a(0).ft == FT.I64) || a(1).ft != FT.I32) throw new NotFusible(s"$n operand types")
        call(n match { case "GpuShiftLeft" => "shl"; case "GpuShiftRight" => "shr"; case _ => "ushr" }, a, a(0).ft)
      case "GpuSqrt" | "GpuExp" | "GpuExpm1" | "GpuSin" | "GpuCos" | "GpuTan" | "GpuAsin" | "GpuAcos" | "GpuAtan" |
           "GpuSinh" | "GpuCosh" | "GpuTanh" | "GpuCbrt" | "GpuRint" | "GpuSignum" | "GpuLog" | "GpuPow" | "GpuHypot" |
           "GpuAcoshImproved" | "GpuAsinhImproved" | "GpuAtanh" | "GpuCot" =>
        val a = ch.map(s)
        if (a.exists(_.ft != FT.F64)) throw new NotFusible(s"$n on non-double input")
        call(n.drop(3).toLowerCase.stripSuffix("improved"), a, FT.F64)
      case "GpuRound" | "GpuBRound" =>
        val a = s(ch.head)
        val scaleLit = ch(1).getClass.getSimpleName match {
          case "GpuLiteral" | "Literal" => ch(1).getClass.getMethod("value").invoke(ch(1))
          case _ => throw new NotFusible(s"$n with non-literal scale")
        }
        if (!a.ft.isFloat) throw new NotFusible(s"$n on ${a.ft} (only float/double handled)")
        if (scaleLit == null) throw new NotFusible(s"$n with NULL scale")
        call(if (n == "GpuRound") "round" else "bround", Seq(a), a.ft, Seq(scaleLit.asInstanceOf[Number].intValue()))
      case "GpuInSet" =>
        val a = s(ch.head)
        if (!(a.ft.isIntegral || a.ft == FT.Date || a.ft == FT.Ts)) throw new NotFusible(s"IN on ${a.ft}")
        val list = e.getClass.getMethod("list").invoke(e).asInstanceOf[Seq[Any]]
        if (list.isEmpty) throw new NotFusible("IN with empty list")
        if (list.size > 64) throw new NotFusible(s"IN list too long (${list.size} > 64)")
        val vals = list.filter(_ != null).map { case x: Number => java.lang.Long.valueOf(x.longValue()); case other => throw new NotFusible(s"IN value ${other.getClass}") }
        call("inset", Seq(a), FT.Bool, vals.toSeq ++ Seq(java.lang.Boolean.valueOf(list.contains(null))))
      case "GpuLogarithm" =>
        val a = ch.map(s)
        if (a.exists(_.ft != FT.F64)) throw new NotFusible("log on non-double input")
        call("logbase", a, FT.F64)
      case "GpuNaNvl" =>
        val a = ch.map(s); sameNumeric(a, n)
        if (!a.head.ft.isFloat) throw new NotFusible("nanvl on non-float")
        call("nanvl", a, a.head.ft)
      case "GpuToDegrees" | "GpuToRadians" =>
        val a = s(ch.head)
        if (a.ft != FT.F64) throw new NotFusible(s"$n on non-double")
        val k = if (n == "GpuToDegrees") 180d / Math.PI else Math.PI / 180d
        call("mul", Seq(a, Lit(java.lang.Double.valueOf(k), FT.F64)), FT.F64)
      case "GpuFloor" | "GpuCeil" =>
        val a = s(ch.head)
        if (ft(e) != FT.I64) throw new NotFusible(s"$n with non-long result")
        a.ft match {
          case FT.I64 => a                      // spark floor/ceil of a long is the identity
          case FT.F64 => call(if (n == "GpuFloor") "floor" else "ceil", Seq(a), FT.I64)
          case other => throw new NotFusible(s"$n on $other")
        }
      case "GpuBitwiseCount" =>
        val a = s(ch.head)
        if (!(a.ft.isIntegral || a.ft == FT.Bool)) throw new NotFusible("bit_count input type")
        call("bitcount", Seq(a), FT.I32)
      case "GpuYear" | "GpuMonth" | "GpuDayOfMonth" | "GpuQuarter" | "GpuDayOfYear" | "GpuDayOfWeek" | "GpuWeekDay" | "GpuLastDay" =>
        val a = s(ch.head)
        if (a.ft != FT.Date) throw new NotFusible(s"$n on ${a.ft}")
        val op = n.drop(3).toLowerCase
        op match {
          case "dayofweek" | "weekday" => call(op, Seq(a), ft(e))
          case "dayofyear" => call(op, Seq(call("civil", Seq(a), FT.I64), a), ft(e))
          case _ => call(op, Seq(call("civil", Seq(a), FT.I64)), ft(e))   // civil (y,m,d) computed once and shared
        }
      case "GpuHour" | "GpuMinute" | "GpuSecond" =>
        val a = s(ch.head)
        if (a.ft != FT.Ts) throw new NotFusible(s"$n on ${a.ft}")
        if (!isUtc(e)) throw new NotFusible(s"$n with non-UTC time zone")
        call(n.drop(3).toLowerCase, Seq(a), FT.I32)
      case "GpuDateAdd" | "GpuDateSub" =>
        val a = ch.map(s)
        if (a(0).ft != FT.Date || !(a(1).ft == FT.I32 || a(1).ft == FT.I16 || a(1).ft == FT.I8)) throw new NotFusible(s"$n operand types")
        call(if (n == "GpuDateAdd") "dateadd" else "datesub", a, FT.Date)
      case "GpuDateDiff" =>
        val a = ch.map(s)
        if (a.exists(_.ft != FT.Date)) throw new NotFusible("datediff operand types")
        call("datediff", a, FT.I32)
      case _ =>
        throw new NotFusible(s"$n is not in the fusion whitelist")
    }
  }
}
