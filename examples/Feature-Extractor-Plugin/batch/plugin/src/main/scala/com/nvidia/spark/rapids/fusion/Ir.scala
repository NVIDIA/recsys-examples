package com.nvidia.spark.rapids.fusion

import org.apache.spark.sql.catalyst.expressions.Expression

/** Minimal expression IR the code generator understands. Structural equality is used for CSE. */
sealed trait Ex extends Serializable { def ft: FT }

/** Column `idx` of the input batch. */
final case class InCol(idx: Int, ft: FT) extends Ex

/** A sub-expression the planner does not fuse; evaluated with cuDF before the kernel and fed in as a column. */
final case class Opaque(slot: Int, ft: FT) extends Ex

/** Literal; value is a boxed Java value (Boolean/Byte/Short/Integer/Long/Float/Double) or null. */
final case class Lit(value: Any, ft: FT) extends Ex {
  // bitwise identity for floating point: Scala's == on boxed numbers says -0.0 == 0.0 (that merged different literals via CSE)
  private def key: Any = value match {
    case d: java.lang.Double => ("d", java.lang.Double.doubleToRawLongBits(d))
    case f: java.lang.Float => ("f", java.lang.Float.floatToRawIntBits(f))
    case n: java.lang.Number => ("n", n.longValue())
    case other => other
  }
  override def equals(o: Any): Boolean = o match { case l: Lit => ft == l.ft && key == l.key; case _ => false }
  override def hashCode: Int = 31 * ft.hashCode + (if (key == null) 0 else key.hashCode)
}

/** Operator application. `attr` holds op-specific immutable parameters (e.g. rounding scale). */
final case class Call(op: String, args: Seq[Ex], ft: FT, attr: Seq[Any] = Nil) extends Ex

/** Thrown by the translator when an expression/region cannot be fused; message is the user-facing reason. */
final class NotFusible(val reason: String) extends RuntimeException(reason, null, false, false)
