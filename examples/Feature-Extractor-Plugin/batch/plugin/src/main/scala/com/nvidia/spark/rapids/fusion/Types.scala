package com.nvidia.spark.rapids.fusion

import ai.rapids.cudf.DType
import org.apache.spark.sql.types._

/** Fusable value types: the kernel-side C type, byte width and Spark/cuDF mappings. */
sealed abstract class FT(val name: String, val ctype: String, val size: Int, val isFloat: Boolean,
                         val isIntegral: Boolean) extends Serializable {
  /** Register/value type inside kernels: narrow integers live in 32-bit ints (re-narrowed after every op) because ptxas
   *  mis-compiles chains of 16-bit PTX arithmetic + compare (neg.s16 then setp.s16 loses the wrap-around). */
  def vtype: String = if (name == "i8" || name == "i16") "int" else ctype
  def isNarrow: Boolean = name == "i8" || name == "i16"
  def isNumeric: Boolean = isFloat || isIntegral
  override def toString: String = name
  def cudf: DType
  def spark: DataType
}

object FT {
  case object Bool extends FT("bool", "bool", 1, false, false) { def cudf = DType.BOOL8; def spark = BooleanType }
  case object I8 extends FT("i8", "i8", 1, false, true) { def cudf = DType.INT8; def spark = ByteType }
  case object I16 extends FT("i16", "i16", 2, false, true) { def cudf = DType.INT16; def spark = ShortType }
  case object I32 extends FT("i32", "i32", 4, false, true) { def cudf = DType.INT32; def spark = IntegerType }
  case object I64 extends FT("i64", "i64", 8, false, true) { def cudf = DType.INT64; def spark = LongType }
  case object F32 extends FT("f32", "f32", 4, true, false) { def cudf = DType.FLOAT32; def spark = FloatType }
  case object F64 extends FT("f64", "f64", 8, true, false) { def cudf = DType.FLOAT64; def spark = DoubleType }
  /** days since epoch, stored as int32 */
  case object Date extends FT("date", "i32", 4, false, false) { def cudf = DType.TIMESTAMP_DAYS; def spark = DateType }
  /** microseconds since epoch, stored as int64 */
  case object Ts extends FT("ts", "i64", 8, false, false) { def cudf = DType.TIMESTAMP_MICROSECONDS; def spark = TimestampType }

  def fromSpark(dt: DataType): Option[FT] = dt match {
    case BooleanType => Some(Bool)
    case ByteType => Some(I8)
    case ShortType => Some(I16)
    case IntegerType => Some(I32)
    case LongType => Some(I64)
    case FloatType => Some(F32)
    case DoubleType => Some(F64)
    case DateType => Some(Date)
    case TimestampType => Some(Ts)
    case _ => None
  }
}
