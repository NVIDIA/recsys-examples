package com.nvidia.spark.rapids.fusion

import ai.rapids.cudf._

/** Dev tool: sem's E6 'deep chain' shape as a kernel micro benchmark, with variants to find what makes it slow. */
object DeepBench {
  def main(args: Array[String]): Unit = {
    val n = 20000000
    NativeBridge.ensureLoaded(sys.env.getOrElse("FUSION_NVRTC", ""))
    Rmm.initialize(RmmAllocationMode.POOL, null, (Cuda.memGetInfo().total * 0.6).toLong)
    NativeBridge.ensureContext()
    val cfg = FusionConf.fromSQLConf(new org.apache.spark.sql.internal.SQLConf())
    val rnd = new java.util.Random(1)
    val cols = (0 until 4).map { _ =>
      val h = HostColumnVector.build(DType.FLOAT64, n, (b: HostColumnVector.Builder) => { var i = 0; while (i < n) { b.append(rnd.nextDouble() * 1000 - 500); i += 1 } })
      try h.copyToDevice() finally h.close()
    }
    def lit(d: Double) = Lit(java.lang.Double.valueOf(d), FT.F64)
    def c(op: String, ft: FT, args: Ex*) = Call(op, args, ft)
    val ops = Seq(("mul", 1.0001), ("add", 0.5), ("sub", 0.25), ("mul", 0.999))
    def chain(k: Int, len: Int): Ex = { var e: Ex = InCol(k, FT.F64); for (j <- 0 until len) { val (op, v) = ops(j % 4); e = c(op, FT.F64, e, lit(v)) }; e }
    def deep(k: Int, len: Int): Ex = { val e = chain(k, len); c("if", FT.F64, c("gt", FT.Bool, e, lit(0.0)), c("mul", FT.F64, e, lit(2.0)), c("sub", FT.F64, e, lit(1.0))) }
    for ((name, outs) <- Seq(("1 output x 30", Seq(deep(0, 30))), ("4 outputs x 30", (0 until 4).map(deep(_, 30))), ("4 outputs x 10", (0 until 4).map(deep(_, 10))))) {
      val spec = CodeGen.build(outs)
      val views: Array[ColumnView] = spec.inputs.map(in => cols(in.ex.asInstanceOf[InCol].idx).asInstanceOf[ColumnView]).toArray
      println(s"$name: ${spec.exprCount} ops, ${spec.inputs.size} inputs, ${spec.outputs.size} outputs")
      MicroBench.timeIt("fused kernel", 3, 10) { () => FusionRuntime.run(spec, views, n, cfg).foreach(_.close()) }
      // unfused cuDF equivalent of the first output
      MicroBench.timeIt("unfused cuDF (1 out x len)", 2, 5) { () =>
        val len = if (name.endsWith("30")) 30 else 10
        var cur: ColumnVector = cols(0).incRefCount()
        for (j <- 0 until len) { val (op, v) = ops(j % 4); val s = Scalar.fromDouble(v); val nx = op match { case "mul" => cur.mul(s); case "add" => cur.add(s); case _ => cur.sub(s) }; s.close(); cur.close(); cur = nx }
        val z = Scalar.fromDouble(0.0); val g = cur.greaterThan(z); z.close()
        val two = Scalar.fromDouble(2.0); val a = cur.mul(two); two.close(); val one = Scalar.fromDouble(1.0); val b = cur.sub(one); one.close()
        val r = g.ifElse(a, b); Seq(cur, g, a, b, r).foreach(_.close())
      }
    }
    sys.exit(0)
  }
}
