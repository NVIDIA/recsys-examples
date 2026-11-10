package com.nvidia.spark.rapids.fusion

/**
 * No-GPU self test: builds a kernel for (almost) every operator x type the Translator/Ops can emit and compiles it with
 * NVRTC for every supported architecture. Catches generator mistakes (invalid C++, missing overloads) without a GPU.
 * Run: NOSRUN=1 scripts/run-java.sh com.nvidia.spark.rapids.fusion.CodegenSelfTest
 */
object CodegenSelfTest {
  def main(args: Array[String]): Unit = {
    NativeBridge.ensureNvrtcOnly(sys.env.getOrElse("FUSION_NVRTC", ""))
    val archs = Seq("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120")
    val tests = scala.collection.mutable.ArrayBuffer[(String, Ex)]()
    def in(i: Int, t: FT) = InCol(i, t)
    def c(op: String, ft: FT, args: Ex*) = Call(op, args, ft)
    val nums = Seq(FT.I8, FT.I16, FT.I32, FT.I64, FT.F32, FT.F64)
    for (t <- nums) {
      for (op <- Seq("add", "sub", "mul", "rem", "pmod")) tests += ((s"$op/$t", c(op, t, in(0, t), in(1, t))))
      for (op <- Seq("lt", "le", "gt", "ge", "eq", "ne", "eqns")) tests += ((s"$op/$t", c(op, FT.Bool, in(0, t), in(1, t))))
      for (op <- Seq("neg", "abs")) tests += ((s"$op/$t", c(op, t, in(0, t))))
      tests += ((s"least/$t", c("least", t, in(0, t), in(1, t), Lit(null, t))))
      tests += ((s"greatest/$t", c("greatest", t, in(0, t), in(1, t), in(2, t))))
      tests += ((s"if/$t", c("if", t, c("gt", FT.Bool, in(0, t), in(1, t)), in(0, t), in(1, t))))
      tests += ((s"coalesce/$t", c("coalesce", t, in(0, t), in(1, t))))
      tests += ((s"casewhen/$t", c("casewhen", t, c("gt", FT.Bool, in(0, t), in(1, t)), in(0, t), c("lt", FT.Bool, in(0, t), in(1, t)), in(1, t))))
      tests += ((s"isnull/$t", c("isnull", FT.Bool, in(0, t))))
      for (to <- nums ++ Seq(FT.Bool)) if (to != t) tests += ((s"cast/$t->$to", c("cast", to, in(0, t))))
    }
    for (t <- Seq(FT.I8, FT.I16, FT.I32, FT.I64)) {
      for (op <- Seq("bit_and", "bit_or", "bit_xor")) tests += ((s"$op/$t", c(op, t, in(0, t), in(1, t))))
      tests += ((s"bit_not/$t", c("bit_not", t, in(0, t))))
      tests += ((s"idiv/$t", c("idiv", FT.I64, in(0, t), in(1, t))))
      tests += ((s"bitcount/$t", c("bitcount", FT.I32, in(0, t))))
      tests += ((s"inset/$t", Call("inset", Seq(in(0, t)), FT.Bool, Seq(java.lang.Long.valueOf(1L), java.lang.Long.valueOf(7L), java.lang.Boolean.TRUE))))
    }
    for (t <- Seq(FT.I32, FT.I64)) for (op <- Seq("shl", "shr", "ushr")) tests += ((s"$op/$t", c(op, t, in(0, t), in(1, FT.I32))))
    for (op <- Seq("and", "or")) tests += ((op, c(op, FT.Bool, in(0, FT.Bool), in(1, FT.Bool))))
    tests += (("not", c("not", FT.Bool, in(0, FT.Bool))))
    tests += (("div", c("div", FT.F64, in(0, FT.F64), in(1, FT.F64))))
    for (t <- Seq(FT.F32, FT.F64)) { tests += ((s"isnan/$t", c("isnan", FT.Bool, in(0, t)))); tests += ((s"nanvl/$t", c("nanvl", t, in(0, t), in(1, t)))); tests += ((s"normnz/$t", c("normnz", t, in(0, t)))) }
    for (op <- Seq("sqrt", "exp", "expm1", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "cbrt", "rint", "signum", "log"))
      tests += ((s"math/$op", c(op, FT.F64, in(0, FT.F64))))
    for (t <- Seq(FT.F32, FT.F64)) for (op <- Seq("round", "bround")) for (sc <- Seq(-3, 0, 2, 40, 400, -400)) tests += ((s"$op/$t/$sc", Call(op, Seq(in(0, t)), t, Seq(Integer.valueOf(sc)))))
    for (op <- Seq("hypot", "atanh", "asinh", "acosh", "cot")) tests += ((s"math/$op", if (op == "hypot") c(op, FT.F64, in(0, FT.F64), in(1, FT.F64)) else c(op, FT.F64, in(0, FT.F64))))
    tests += (("math/pow", c("pow", FT.F64, in(0, FT.F64), in(1, FT.F64))))
    tests += (("logbase", c("logbase", FT.F64, in(0, FT.F64), in(1, FT.F64))))
    tests += (("floor", c("floor", FT.I64, in(0, FT.F64)))); tests += (("ceil", c("ceil", FT.I64, in(0, FT.F64))))
    for (op <- Seq("year", "month", "dayofmonth", "quarter", "lastday")) tests += ((s"date/$op", c(op, if (op == "lastday") FT.Date else FT.I32, c("civil", FT.I64, in(0, FT.Date)))))
    tests += (("date/dayofyear", c("dayofyear", FT.I32, c("civil", FT.I64, in(0, FT.Date)), in(0, FT.Date))))
    for (op <- Seq("dayofweek", "weekday")) tests += ((s"date/$op", c(op, FT.I32, in(0, FT.Date))))
    for (op <- Seq("hour", "minute", "second")) tests += ((s"ts/$op", c(op, FT.I32, in(0, FT.Ts))))
    tests += (("dateadd", c("dateadd", FT.Date, in(0, FT.Date), in(1, FT.I16)))); tests += (("datesub", c("datesub", FT.Date, in(0, FT.Date), in(1, FT.I32))))
    tests += (("datediff", c("datediff", FT.I32, in(0, FT.Date), in(1, FT.Date))))
    tests += (("cast/date->ts", c("cast", FT.Ts, in(0, FT.Date)))); tests += (("cast/ts->date", c("cast", FT.Date, in(0, FT.Ts))))
    tests += (("cast/ts->i64", c("cast", FT.I64, in(0, FT.Ts)))); tests += (("cast/ts->f64", c("cast", FT.F64, in(0, FT.Ts)))); tests += (("cast/i32->ts", c("cast", FT.Ts, in(0, FT.I32))))
    tests += (("cmp/date", c("lt", FT.Bool, in(0, FT.Date), in(1, FT.Date)))); tests += (("cmp/ts", c("ge", FT.Bool, in(0, FT.Ts), in(1, FT.Ts))))
    tests += (("tomask", c("tomask", FT.Bool, c("and", FT.Bool, c("gt", FT.Bool, in(0, FT.F64), Lit(java.lang.Double.valueOf(1.5), FT.F64)), c("isnotnull", FT.Bool, in(1, FT.I32))))))
    var bad = 0; var n = 0
    val allSpecs = tests.map { case (name, ex) => (name, try Right(CodeGen.build(Seq(ex))) catch { case t: Throwable => Left(t) }) }
    // one big kernel with everything too (register pressure / size)
    val big = CodeGen.build(tests.map(_._2).filter(_.ft != FT.Bool || true).take(60).toSeq)
    for (arch <- archs; (name, spec) <- allSpecs) {
      spec match {
        case Left(t) => if (arch == archs.head) { bad += 1; println(s"CODEGEN-FAIL $name: $t") }
        case Right(sp) =>
          n += 1
          try NativeBridge.compileToCubin(sp.source, arch)
          catch { case t: Throwable => bad += 1; println(s"NVRTC-FAIL $arch $name: ${t.getMessage.take(300)}") }
      }
    }
    for (arch <- archs) try NativeBridge.compileToCubin(big.source, arch) catch { case t: Throwable => bad += 1; println(s"NVRTC-FAIL $arch BIG: ${t.getMessage.take(300)}") }
    println(s"CODEGEN-SELFTEST kernels=${allSpecs.size} compiles=$n archs=${archs.mkString(",")} nvrtc=${NativeBridge.nvrtcVersion()} failures=$bad")
    sys.exit(if (bad == 0) 0 else 1)
  }
}
