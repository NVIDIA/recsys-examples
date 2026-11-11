package com.nvidia.spark.rapids.fusion

import com.nvidia.spark.rapids.fusion.CodeGen.Gen

/**
 * Operator table: IR op name -> device-side semantics. Every op has one definition here; the same text is used
 * whether the op is alone in a region or fused with others.
 *
 * Convention: a value is (v, n) where n == "valid". Value of an invalid row is unspecified (kept 0 when cheap).
 * Functions are total: no traps, no UB, so evaluating a not-taken branch is harmless.
 */
object Ops {
  type V = (String, String)

  /** Ops whose result can be NULL even if all inputs are valid. */
  def createsNulls(c: Call): Boolean = c.op match {
    case "div" | "idiv" | "rem" | "pmod" | "log" | "logbase" => true
    case "inset" => c.attr.lastOption.contains(java.lang.Boolean.TRUE)   // NULL when not found and the list contains NULL
    case _ => false
  }

  /** Re-narrow a 32-bit intermediate to the (wrapped) value range of an 8/16-bit result type. */
  private def narrow(ft: FT, e: String): String = ft match { case FT.I8 => s"norm8($e)"; case FT.I16 => s"norm16($e)"; case _ => e }

  private def strict(g: Gen, c: Call, a: Seq[V])(f: Seq[String] => String): V = {
    val ok = if (a.isEmpty) "true" else a.map(_._2).mkString(" && ")
    val vals = a.map(_._1)
    g.define(c.ft, s"(($ok) ? ${narrow(c.ft, f(vals))} : (${c.ft.vtype})0)", ok)
  }

  private def bin(g: Gen, c: Call, a: Seq[V], fn: String): V = strict(g, c, a)(v => s"$fn(${v(0)}, ${v(1)})")
  private def un(g: Gen, c: Call, a: Seq[V], fn: String): V = strict(g, c, a)(v => s"$fn(${v(0)})")

  private val binFns = Map(
    "add" -> "w_add", "sub" -> "w_sub", "mul" -> "w_mul",
    "lt" -> "c_lt", "le" -> "c_le", "gt" -> "c_gt", "ge" -> "c_ge", "eq" -> "c_eq",
    "bit_and" -> "b_and", "bit_or" -> "b_or", "bit_xor" -> "b_xor",
    "shl" -> "s_shl", "shr" -> "s_shr", "ushr" -> "s_ushr")

  def emit(g: Gen, c: Call, a: Seq[V]): V = c.op match {
    case op if binFns.contains(op) => bin(g, c, a, binFns(op))
    case "neg" => un(g, c, a, "w_neg")
    case "abs" => un(g, c, a, "w_abs")
    case "pos" => a.head
    case "nanvl" =>
      val (l, nl) = a(0); val (r, nr) = a(1)
      val keep = s"(!($nl && is_nan($l)))"      // null is not NaN -> keep left (null)
      g.define(c.ft, s"($keep ? $l : $r)", s"($keep ? $nl : $nr)")
    case "logbase" =>
      val ok = s"(${a(0)._2} && ${a(1)._2} && !(${a(0)._1} <= 0.0) && !(${a(1)._1} <= 0.0))"
      g.define(FT.F64, s"($ok ? d_logbase(${a(0)._1}, ${a(1)._1}) : 0.0)", ok)
    case "floor" => strict(g, c, a)(v => s"fl_f64(${v(0)})")
    case "ceil" => strict(g, c, a)(v => s"ce_f64(${v(0)})")
    case "bitcount" =>
      strict(g, c, a)(v => if (c.args.head.ft == FT.Bool) s"(${v(0)} ? 1 : 0)" else s"bit_cnt((i64)${v(0)})")
    case "civil" =>
      // cuDF's date extraction deviates from the textbook algorithm where days + 719468 overflows int32: refuse those batches
      g.bailIf(s"${a(0)._2} && (${a(0)._1} > 2146000000 || ${a(0)._1} < -2146000000)")
      un(g, c, a, "dt_civil")
    case "year" => un(g, c, a, "dc_year")
    case "month" => un(g, c, a, "dc_month")
    case "dayofmonth" => un(g, c, a, "dc_day")
    case "quarter" => un(g, c, a, "dc_quarter")
    case "dayofyear" => bin(g, c, a, "dc_doy")
    case "dayofweek" =>
      g.bailIf(s"${a(0)._2} && (${a(0)._1} > 2146000000 || ${a(0)._1} < -2146000000)")
      un(g, c, a, "dt_dow")
    case "weekday" =>
      g.bailIf(s"${a(0)._2} && (${a(0)._1} > 2146000000 || ${a(0)._1} < -2146000000)")
      un(g, c, a, "dt_weekday")
    case "lastday" => un(g, c, a, "dc_lastday")
    case "hour" => un(g, c, a, "dt_hour")
    case "minute" => un(g, c, a, "dt_minute")
    case "second" => un(g, c, a, "dt_second")
    case "dateadd" => strict(g, c, a)(v => s"dt_add(${v(0)}, (i32)${v(1)})")
    case "datesub" => strict(g, c, a)(v => s"dt_sub(${v(0)}, (i32)${v(1)})")
    case "datediff" => bin(g, c, a, "dt_diff")
    case "round" | "bround" =>
      val sc = c.attr(0).asInstanceOf[Int]
      val maxDigits = if (c.ft == FT.F32) 39 else 309
      val fn = if (c.op == "round") "rnd_half_up" else "rnd_half_even"
      if (-sc >= maxDigits) strict(g, c, a)(v => s"rnd_zero_fix(${v(0)}, ${v(0)})")
      else if (sc >= maxDigits) a.head
      else strict(g, c, a)(v => s"$fn(${v(0)}, $sc)")
    case "inset" =>
      val vals = c.attr.dropRight(1).map(_.asInstanceOf[Long]); val hasNull = c.attr.last.asInstanceOf[Boolean]
      val (x, nx) = a.head
      val found = vals.map(v => s"((i64)$x == ${CodeGen.literal(java.lang.Long.valueOf(v), FT.I64)})").mkString(" || ")
      g.define(FT.Bool, s"($nx && ($found))", if (hasNull) s"($nx && ($found))" else nx)
    case "tomask" => g.define(FT.Bool, s"(${a(0)._1} && ${a(0)._2})", "true")
    case "bit_not" => un(g, c, a, "b_not")
    case "not" => strict(g, c, a)(v => s"(!${v(0)})")
    case "ne" => strict(g, c, a)(v => s"(!c_eq(${v(0)}, ${v(1)}))")
    // division family: zero divisor -> NULL (non-ANSI Spark semantics, same as spark-rapids' replaceZeroWithNull)
    case "div" =>
      val ok = s"${a(0)._2} && ${a(1)._2} && !is_zero(${a(1)._1})"
      g.define(c.ft, s"(($ok) ? d_div(${a(0)._1}, ${a(1)._1}) : (${c.ft.vtype})0)", ok)
    case "idiv" =>
      val ok = s"${a(0)._2} && ${a(1)._2} && !is_zero(${a(1)._1})"
      g.define(c.ft, s"(($ok) ? d_idiv(${a(0)._1}, ${a(1)._1}) : (${c.ft.vtype})0)", ok)
    case "rem" =>
      val ok = s"${a(0)._2} && ${a(1)._2} && !is_zero(${a(1)._1})"
      g.define(c.ft, s"(($ok) ? ${narrow(c.ft, s"d_rem(${a(0)._1}, ${a(1)._1})")} : (${c.ft.vtype})0)", ok)
    case "pmod" =>
      val ok = s"${a(0)._2} && ${a(1)._2} && !is_zero(${a(1)._1})"
      g.define(c.ft, s"(($ok) ? ${narrow(c.ft, s"d_pmod(${a(0)._1}, ${a(1)._1})")} : (${c.ft.vtype})0)", ok)
    // three-valued logic
    case "and" =>
      val (x, nx) = a(0); val (y, ny) = a(1)
      g.define(FT.Bool, s"($nx && $ny && $x && $y)", s"(($nx && !$x) || ($ny && !$y) || ($nx && $ny))")
    case "or" =>
      val (x, nx) = a(0); val (y, ny) = a(1)
      g.define(FT.Bool, s"(($nx && $x) || ($ny && $y))", s"(($nx && $x) || ($ny && $y) || ($nx && $ny))")
    case "isnull" => g.define(FT.Bool, s"(!${a(0)._2})", "true")
    case "isnotnull" => g.define(FT.Bool, a(0)._2, "true")
    case "isnan" => g.define(FT.Bool, s"(${a(0)._2} && is_nan(${a(0)._1}))", "true")
    case "eqns" =>
      val (x, nx) = a(0); val (y, ny) = a(1)
      g.define(FT.Bool, s"(($nx && $ny) ? c_eq($x, $y) : (!$nx && !$ny))", "true")
    case "if" =>
      val (p, np) = a(0); val (t, nt) = a(1); val (f, nf) = a(2)
      val take = s"($np && $p)"
      g.define(c.ft, s"($take ? $t : $f)", s"($take ? $nt : $nf)")
    case "casewhen" =>
      // args: p1, v1, p2, v2, ..., [else]; built as nested ifs from the back
      val hasElse = a.size % 2 == 1
      var (accV, accN) = if (hasElse) a.last else g.define(c.ft, s"(${c.ft.vtype})0", "false")
      val pairs = a.dropRight(if (hasElse) 1 else 0).grouped(2).toList.reverse
      for (Seq((p, np), (t, nt)) <- pairs) {
        val take = s"($np && $p)"
        val r = g.define(c.ft, s"($take ? $t : $accV)", s"($take ? $nt : $accN)")
        accV = r._1; accN = r._2
      }
      (accV, accN)
    case "coalesce" =>
      var (accV, accN) = a.last
      for ((v, n) <- a.dropRight(1).reverse) {
        val r = g.define(c.ft, s"($n ? $v : $accV)", s"($n || $accN)")
        accV = r._1; accN = r._2
      }
      (accV, accN)
    case "cast" => castOp(g, c, a)
    case "least" | "greatest" => leastGreatest(g, c, a, c.op == "least")
    case "normnz" => un(g, c, a, "norm_nz")
    case m if MathOps.handles(m) => MathOps.emit(g, c, a)
    case other => throw new NotFusible(s"no device definition for op '$other'")
  }

  private def leastGreatest(g: Gen, c: Call, a: Seq[V], isLeast: Boolean): V = {
    val strict = if (isLeast) "c_lt" else "c_gt"       // NaN-aware (NaN greatest)
    var (accV, accN) = a.head
    for ((v, n) <- a.tail) {
      // cuDF NULL_MAX/NULL_MIN: `x > y ? x : y`  -> on a tie (0.0 vs -0.0) the LATER operand wins, whether it is a column
      // or a scalar literal (verified with the smoke battery in scripts/smoke: least/greatest over col/lit combinations). An invalid side is
      // skipped; the result is invalid only if all are.
      val pick = if (c.ft.isFloat) s"sel_${c.ft.name}($strict($accV, $v), $accV, $v)" else s"($strict($accV, $v) ? $accV : $v)"
      val r = g.define(c.ft, s"(($accN && $n) ? $pick : ($n ? $v : $accV))", s"($accN || $n)")
      accV = r._1; accN = r._2
    }
    (accV, accN)
  }

  /** Casts castOp can generate; the translator checks this so an unsupported cast stays out of the kernel. */
  def castSupported(from: FT, to: FT): Boolean = (from, to) match {
    case (f, t) if f == t => true
    case (FT.Bool, t) if t.isNumeric => true
    case (f, FT.Bool) if f.isNumeric => true
    case (f, t) if f.isIntegral && (t.isIntegral || t.isFloat) => true
    case (FT.F32, FT.F64) | (FT.F64, FT.F32) => true
    case (f, t) if f.isFloat && t.isIntegral => true
    case (FT.Date, FT.Ts) | (FT.Ts, FT.Date) | (FT.Ts, FT.I64) => true
    case (FT.Ts, t) if t == FT.I32 || t == FT.I16 || t == FT.I8 || t.isFloat => true
    case (FT.I8 | FT.I16 | FT.I32, FT.Ts) => true
    case _ => false
  }

  def castOp(g: Gen, c: Call, a: Seq[V]): V = {
    val from = c.args.head.ft; val to = c.ft
    val (x, nx) = a.head
    val expr: String = (from, to) match {
      case (f, t) if f == t => x
      case (FT.Bool, t) if t.isNumeric => s"(${t.ctype})($x ? 1 : 0)"
      case (f, FT.Bool) if f.isNumeric => s"($x != 0)"
      case (f, t) if f.isIntegral && t.isIntegral => if (t.isNarrow) narrow(t, s"(int)$x") else s"(${t.ctype})$x"
      case (f, t) if f.isIntegral && t.isFloat => s"(${t.ctype})$x"
      case (FT.F32, FT.F64) => s"(f64)$x"
      case (FT.F64, FT.F32) => s"(f32)$x"
      case (f, t) if f.isFloat && t.isIntegral => s"f2i_${t.name}($x)"
      case (FT.Date, FT.Ts) => s"((i64)$x * 86400000000LL)"
      case (FT.Ts, FT.Date) => s"(i32)fdiv64($x, 86400000000LL)"
      case (FT.Ts, FT.I64) => s"fdiv64($x, 1000000LL)"
      case (FT.Ts, t) if t == FT.I32 || t == FT.I16 || t == FT.I8 => if (t.isNarrow) narrow(t, s"(int)fdiv64($x, 1000000LL)") else s"(${t.ctype})fdiv64($x, 1000000LL)"
      case (FT.I8 | FT.I16 | FT.I32, FT.Ts) => s"((i64)$x * 1000000LL)"
      case (FT.Ts, t) if t.isFloat => s"((${t.ctype})((f64)$x / 1000000.0))"
      case _ => throw new NotFusible(s"cast ${from} -> ${to} not supported")
    }
    g.define(to, s"($nx ? $expr : (${to.vtype})0)", nx)
  }
}
