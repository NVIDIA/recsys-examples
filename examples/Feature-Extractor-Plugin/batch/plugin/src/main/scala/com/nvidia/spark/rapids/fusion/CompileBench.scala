package com.nvidia.spark.rapids.fusion

/** Measures NVRTC compile latency of typical regions (dev tool). */
object CompileBench {
  def main(args: Array[String]): Unit = {
    NativeBridge.ensureLoaded(sys.env.getOrElse("FUSION_NVRTC", ""))
    NativeBridge.ensureContext()
    def in(i: Int) = InCol(i, FT.F64)
    def c(op: String, args: Ex*) = Call(op, args, FT.F64)
    val small = c("add", c("mul", in(0), in(1)), in(2))
    var big: Ex = in(0)
    for (i <- 1 until 60) big = c(if (i % 3 == 0) "add" else if (i % 3 == 1) "mul" else "sub", big, in(i % 8))
    val math = c("add", c("sqrt", in(0)), c("mul", c("exp", in(1)), c("pow", in(2), in(3))))
    println(s"arch=${NativeBridge.currentArch()} nvrtc=${NativeBridge.nvrtcVersion()}")
    for ((name, e) <- Seq(("small(2 ops)", small), ("math(sqrt,exp,pow)", math), ("big(59 ops)", big))) {
      val spec = CodeGen.build(Seq(e))
      val ts = (0 until 4).map { _ =>
        val t0 = System.nanoTime(); NativeBridge.compileToCubin(spec.source, NativeBridge.currentArch()); (System.nanoTime() - t0) / 1e6
      }
      println(f"  $name%-22s compile ms: ${ts.map(x => f"$x%.0f").mkString(", ")}  (source ${spec.source.length} chars)")
    }
    sys.exit(0)
  }
}
