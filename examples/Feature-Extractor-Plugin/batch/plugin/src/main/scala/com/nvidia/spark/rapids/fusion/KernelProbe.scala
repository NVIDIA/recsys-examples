package com.nvidia.spark.rapids.fusion

import ai.rapids.cudf._

/** Dev tool: run a tiny IR expression on a few literal values and print the results (kernel-level debugging). */
object KernelProbe {
  def main(args: Array[String]): Unit = {
    NativeBridge.ensureLoaded(sys.env.getOrElse("FUSION_NVRTC", ""))
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 0L)
    NativeBridge.ensureContext()
    val cfg = FusionConf.fromSQLConf(new org.apache.spark.sql.internal.SQLConf())
    val col = ColumnVector.fromShorts((-32768).toShort, 5.toShort, (-5).toShort, 32767.toShort)
    val e = Call("ge", Seq(Call("neg", Seq(InCol(0, FT.I16)), FT.I16), Lit(java.lang.Short.valueOf(1.toShort), FT.I16)), FT.Bool)
    val spec = CodeGen.build(Seq(e))
    val out = FusionRuntime.run(spec, Array(col.asInstanceOf[ColumnView]), 4, cfg)(0)
    val h = out.copyToHost()
    println("PROBE " + (0 until 4).map(i => h.getBoolean(i)).mkString(","))
    val e2 = Call("neg", Seq(InCol(0, FT.I16)), FT.I16)
    val out2 = FusionRuntime.run(CodeGen.build(Seq(e2)), Array(col.asInstanceOf[ColumnView]), 4, cfg)(0).copyToHost()
    println("PROBE neg " + (0 until 4).map(i => out2.getShort(i)).mkString(","))
    // signed zero handling of min/max-like patterns and normalization
    val dcol = ColumnVector.fromDoubles(0.0, -0.0, Double.NaN, 1.0, -0.0)
    def bits(h: HostColumnVector, i: Int) = java.lang.Long.toHexString(java.lang.Double.doubleToRawLongBits(h.getDouble(i)))
    for ((name, ex) <- Seq(
        ("least(x,+0)", Call("least", Seq(InCol(0, FT.F64), Lit(java.lang.Double.valueOf(0.0), FT.F64)), FT.F64)),
        ("greatest(x,+0)", Call("greatest", Seq(InCol(0, FT.F64), Lit(java.lang.Double.valueOf(0.0), FT.F64)), FT.F64)),
        ("greatest(x,-0)", Call("greatest", Seq(InCol(0, FT.F64), Lit(java.lang.Double.valueOf(-0.0), FT.F64)), FT.F64)),
        ("normnz(x)", Call("normnz", Seq(InCol(0, FT.F64)), FT.F64)),
        ("abs(x)", Call("abs", Seq(InCol(0, FT.F64)), FT.F64)),
        ("neg(x)", Call("neg", Seq(InCol(0, FT.F64)), FT.F64)))) {
      val h = FusionRuntime.run(CodeGen.build(Seq(ex)), Array(dcol.asInstanceOf[ColumnView]), 5, cfg)(0).copyToHost()
      println("PROBE " + name + " for x=(+0,-0,NaN,1,-0): " + (0 until 5).map(bits(h, _)).mkString(" "))
    }
    sys.exit(0)
  }
}
