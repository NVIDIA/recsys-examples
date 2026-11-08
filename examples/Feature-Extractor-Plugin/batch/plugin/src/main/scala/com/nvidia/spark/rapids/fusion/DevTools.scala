package com.nvidia.spark.rapids.fusion

import ai.rapids.cudf._

/** Developer utilities (not used at runtime). */
object DevTools {
  def main(args: Array[String]): Unit = {
    val f = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe"); f.setAccessible(true)
    val u = f.get(null).asInstanceOf[sun.misc.Unsafe]
    val col = ColumnVector.fromBoxedInts(1, null, 3, 4, 5, 6, 7, 8, 9, 10)
    val h = col.getNativeView
    println(s"data=${col.getData.getAddress.toHexString} valid=${col.getValid.getAddress.toHexString}")
    for (o <- 0 until 80 by 4) println(s"+$o int=${u.getInt(h + o)} long@=${if (o % 8 == 0) u.getLong(h + o).toHexString else ""}")
    val sl = col.subVector(2, 8)
    println("slice: data=" + sl.getData.getAddress.toHexString + " valid=" + (if (sl.getValid == null) "null" else sl.getValid.getAddress.toHexString))
  }
}
