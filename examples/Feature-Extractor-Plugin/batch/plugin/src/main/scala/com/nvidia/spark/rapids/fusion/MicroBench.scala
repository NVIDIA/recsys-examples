package com.nvidia.spark.rapids.fusion

import ai.rapids.cudf._

/** Kernel-level micro benchmark: unfused cuDF op chain vs one fused kernel on device-resident columns. */
object MicroBench {
  def timeIt(name: String, warm: Int, reps: Int)(f: () => Unit): Double = {
    var i = 0
    while (i < warm) { f(); i += 1 }
    Cuda.DEFAULT_STREAM.sync()
    val ts = new Array[Double](reps)
    i = 0
    while (i < reps) {
      val t0 = System.nanoTime(); f(); Cuda.DEFAULT_STREAM.sync(); ts(i) = (System.nanoTime() - t0) / 1e6; i += 1
    }
    val sorted = ts.sorted
    val med = sorted(reps / 2)
    println(f"  $name%-28s median ${med}%8.3f ms  (min ${sorted(0)}%.3f max ${sorted(reps - 1)}%.3f)")
    med
  }

  def main(args: Array[String]): Unit = {
    val n = if (args.length > 0) args(0).toInt else 20000000
    val reps = if (args.length > 1) args(1).toInt else 15
    NativeBridge.ensureLoaded(sys.env.getOrElse("FUSION_NVRTC", ""))
    val poolBytes = (Cuda.memGetInfo().total * 0.6).toLong
    Rmm.initialize(RmmAllocationMode.POOL, null, poolBytes)
    NativeBridge.ensureContext()
    val cfg = FusionConf.fromSQLConf(new org.apache.spark.sql.internal.SQLConf())
    println(s"rows=$n arch=${NativeBridge.currentArch()} nvrtc=${NativeBridge.nvrtcVersion()} fmad=${sys.env.getOrElse("FUSION_FMAD", "true")}")
    val rnd = new java.util.Random(1)
    def dcol(nulls: Boolean): ColumnVector = {
      val h = HostColumnVector.build(DType.FLOAT64, n, (b: HostColumnVector.Builder) => {
        var i = 0
        while (i < n) { if (nulls && i % 17 == 3) b.appendNull() else b.append(rnd.nextDouble() * 100 + 0.5); i += 1 }
      })
      try h.copyToDevice() finally h.close()
    }
    val cols = (0 until 6).map(_ => dcol(false))
    val colsN = (0 until 6).map(_ => dcol(true))
    def in(i: Int) = InCol(i, FT.F64)
    def c(op: String, ft: FT, args: Ex*) = Call(op, args, ft)
    def lit(d: Double) = Lit(java.lang.Double.valueOf(d), FT.F64)

    case class Case(name: String, ex: Ex, numIn: Int, unfused: Seq[ColumnVector] => ColumnVector)
    val cases = Seq(
      Case("A (a*b+c)*d-e [4 ops]", c("sub", FT.F64, c("mul", FT.F64, c("add", FT.F64, c("mul", FT.F64, in(0), in(1)), in(2)), in(3)), in(4)), 5,
        x => { val t1 = x(0).mul(x(1)); val t2 = t1.add(x(2)); val t3 = t2.mul(x(3)); val r = t3.sub(x(4)); t1.close(); t2.close(); t3.close(); r }),
      Case("B 12-op arithmetic chain", {
        var e: Ex = in(0)
        for (i <- 1 until 6) e = c(if (i % 2 == 0) "add" else "mul", FT.F64, e, in(i))
        for (i <- 0 until 6) e = c(if (i % 2 == 0) "sub" else "add", FT.F64, e, lit(i + 0.5))
        e }, 6,
        x => { var cur: ColumnVector = x(0).incRefCount()
          for (i <- 1 until 6) { val nx = if (i % 2 == 0) cur.add(x(i)) else cur.mul(x(i)); cur.close(); cur = nx }
          for (i <- 0 until 6) { val s = Scalar.fromDouble(i + 0.5); val nx = if (i % 2 == 0) cur.sub(s) else cur.add(s); s.close(); cur.close(); cur = nx }
          cur }),
      Case("C sqrt(a)*exp(-b/100)+c*d", c("add", FT.F64, c("mul", FT.F64, c("sqrt", FT.F64, in(0)), c("exp", FT.F64, c("div", FT.F64, c("neg", FT.F64, in(1)), lit(100)))), c("mul", FT.F64, in(2), in(3))), 4,
        x => { val s = x(0).sqrt(); val z0 = Scalar.fromDouble(0); val nb = z0.sub(x(1)); z0.close(); val d100 = Scalar.fromDouble(100); val dv = nb.div(d100); d100.close(); val e = dv.exp(); val cd = x(2).mul(x(3)); val m = s.mul(e); val r = m.add(cd); Seq(s, nb, dv, e, cd, m).foreach(_.close()); r }),
      Case("D if(a>b, a-b, b-a)*c", c("mul", FT.F64, c("if", FT.F64, c("gt", FT.Bool, in(0), in(1)), c("sub", FT.F64, in(0), in(1)), c("sub", FT.F64, in(1), in(0))), in(2)), 3,
        x => { val g = x(0).greaterThan(x(1)); val ab = x(0).sub(x(1)); val ba = x(1).sub(x(0)); val sel = g.ifElse(ab, ba); val r = sel.mul(x(2)); Seq(g, ab, ba, sel).foreach(_.close()); r })
    )
    for (inputs <- Seq(("no nulls", cols), ("with nulls (6%)", colsN))) {
      println(s"== inputs: ${inputs._1}")
      for (cs <- cases) {
        val spec = CodeGen.build(Seq(cs.ex))
        val views: Array[ColumnView] = spec.inputs.map(in => inputs._2(in.ex.asInstanceOf[InCol].idx).asInstanceOf[ColumnView]).toArray
        println(s"${cs.name}: ${spec.exprCount} ops, ${spec.inputs.size} inputs")
        val ins = inputs._2.take(cs.numIn)
        val tu = timeIt("unfused (cuDF ops)", 3, reps) { () => cs.unfused(ins).close() }
        val tf = timeIt("fused (NVRTC kernel)", 3, reps) { () => FusionRuntime.run(spec, views, n, cfg).foreach(_.close()) }
        // correctness
        val ref = cs.unfused(ins); val fus = FusionRuntime.run(spec, views, n, cfg)(0)
        val hr = ref.copyToHost(); val hf = fus.copyToHost()
        var bad = 0; var i = 0
        while (i < n) {
          val nr = hr.isNull(i); val nf = hf.isNull(i)
          if (nr != nf || (!nr && java.lang.Double.doubleToRawLongBits(hr.getDouble(i)) != java.lang.Double.doubleToRawLongBits(hf.getDouble(i)) && !(hr.getDouble(i).isNaN && hf.getDouble(i).isNaN))) bad += 1
          i += 1
        }
        hr.close(); hf.close(); ref.close(); fus.close()
        println(f"  => speedup ${tu / tf}%.2fx   mismatching rows: $bad")
      }
    }
    sys.exit(0)
  }
}
