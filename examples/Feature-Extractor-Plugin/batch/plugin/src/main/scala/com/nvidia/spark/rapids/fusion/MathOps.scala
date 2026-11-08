package com.nvidia.spark.rapids.fusion

import com.nvidia.spark.rapids.fusion.CodeGen.Gen

/** f64 -> f64 math functions, mirrored from the cuDF unary/binary ops spark-rapids uses (same CUDA math library). */
object MathOps {
  private val unary = Map(
    "sqrt" -> "sqrt", "exp" -> "exp", "expm1" -> "m_expm1", "sin" -> "sin", "cos" -> "cos", "tan" -> "tan",
    "asin" -> "asin", "acos" -> "acos", "atan" -> "atan", "sinh" -> "sinh", "cosh" -> "cosh", "tanh" -> "tanh",
    "cbrt" -> "cbrt", "rint" -> "rint", "signum" -> "m_signum",
    "acosh" -> "acosh", "asinh" -> "asinh", "atanh" -> "atanh", "cot" -> "m_cot")
  private val binary = Map("pow" -> "pow", "hypot" -> "m_hypot")
  def handles(op: String): Boolean = unary.contains(op) || binary.contains(op) || op == "log"

  def emit(g: Gen, c: Call, a: Seq[(String, String)]): (String, String) = {
    val ok = a.map(_._2).mkString(" && ")
    if (c.op == "log") {
      // Spark: log(x) is NULL for x <= 0 (spark-rapids: nullOutNegatives)
      val ok2 = s"($ok && !(${a(0)._1} <= 0.0))"
      return g.define(FT.F64, s"($ok2 ? log(${a(0)._1}) : 0.0)", ok2)
    }
    unary.get(c.op) match {
      case Some(fn) => g.define(FT.F64, s"($ok ? $fn(${a(0)._1}) : 0.0)", ok)
      case None =>
        val fn = binary(c.op)
        g.define(FT.F64, s"($ok ? $fn(${a(0)._1}, ${a(1)._1}) : 0.0)", ok)
    }
  }
}
