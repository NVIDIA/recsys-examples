package com.nvidia.spark.rapids.fusion

import java.security.MessageDigest

import scala.collection.mutable

/** One kernel input: either a column of the input batch or a pre-evaluated opaque sub-expression. */
final case class KIn(ex: Ex, ft: FT) extends Serializable
/** One kernel output column. `selfNull`: op can create nulls from valid inputs. `deps`: indices into inputs. */
final case class KOut(ft: FT, selfNull: Boolean, deps: Seq[Int]) extends Serializable

/** Everything needed at runtime to compile and launch (serializable; kernel is compiled lazily on executors). */
final case class KernelSpec(name: String, source: String, inputs: Seq[KIn], outputs: Seq[KOut],
                            exprCount: Int, usesGuard: Boolean = false) extends Serializable {
  lazy val key: String = {
    val md = MessageDigest.getInstance("SHA-256")
    md.update(source.getBytes("UTF-8"))
    md.digest().map("%02x".format(_)).take(10).mkString
  }
}

/** Per-op semantics. Each op has exactly ONE device-side definition (in [[Prelude]] or inline below). */
object CodeGen {

  /** Generator state for one kernel. */
  final class Gen {
    val lines = mutable.ArrayBuffer[String]()
    private var counter = 0
    private val memo = mutable.HashMap[Ex, (String, String)]()
    val inputs = mutable.ArrayBuffer[KIn]()
    private val inputIdx = mutable.HashMap[Ex, Int]()
    val depsOf = mutable.HashMap[Ex, Set[Int]]()
    val selfNull = mutable.HashMap[Ex, Boolean]()
    var exprCount = 0
    var usesGuard = false
    /** Emit "if the condition holds for an in-bounds row, tell the host to discard this batch and run unfused". */
    def bailIf(cond: String): Unit = { usesGuard = true; line(s"  if (inb && ($cond)) bail(flag);") }

    def fresh(): Int = { counter += 1; counter }
    def line(s: String): Unit = lines += s

    def inputIndex(e: Ex): Int = inputIdx.getOrElseUpdate(e, { inputs += KIn(e, e.ft); inputs.size - 1 })

    /** Declare a new value/valid pair. */
    def define(ft: FT, value: String, valid: String): (String, String) = {
      val k = fresh()
      line(s"  ${ft.vtype} v$k = $value;")
      line(s"  bool n$k = $valid;")
      (s"v$k", s"n$k")
    }

    def gen(e: Ex): (String, String) = memo.get(e) match {
      case Some(r) => r
      case None =>
        val r = genNew(e)
        memo(e) = r
        r
    }

    private def genNew(e: Ex): (String, String) = e match {
      case i @ (_: InCol | _: Opaque) =>
        val idx = inputIndex(i)
        depsOf(e) = Set(idx)
        selfNull(e) = false
        val k = fresh()
        line(s"  ${i.ft.vtype} v$k = ld_${i.ft.name}(c$idx, i, inb);")
        line(s"  bool n$k = inb && vld(m$idx, i);")
        (s"v$k", s"n$k")
      case Lit(v, ft) =>
        depsOf(e) = Set.empty
        selfNull(e) = v == null
        if (v == null) define(ft, s"(${ft.ctype})0", "false")
        else define(ft, literal(v, ft), "true")
      case c @ Call(op, args, ft, attr) =>
        val a = args.map(gen)
        val isMask = op == "tomask"   // filter mask: null -> false, so the output never carries validity
        depsOf(e) = if (isMask) Set.empty else args.flatMap(x => depsOf(x)).toSet
        selfNull(e) = !isMask && (args.exists(x => selfNull(x)) || Ops.createsNulls(c))
        exprCount += 1
        Ops.emit(this, c, a)
    }
  }

  def literal(v: Any, ft: FT): String = (v, ft) match {
    case (b: java.lang.Boolean, FT.Bool) => if (b) "true" else "false"
    case (x: Number, FT.I8) => s"(int)${x.byteValue()}"
    case (x: Number, FT.I16) => s"(int)${x.shortValue()}"
    case (x: Number, FT.I32 | FT.Date) => if (x.intValue() == Int.MinValue) "(i32)(-2147483647-1)" else s"(i32)${x.intValue()}"
    case (x: Number, FT.I64 | FT.Ts) =>
      if (x.longValue() == Long.MinValue) "(i64)(-9223372036854775807LL-1)" else s"(i64)${x.longValue()}LL"
    case (x: Number, FT.F32) => s"__int_as_float(${java.lang.Float.floatToRawIntBits(x.floatValue())})"
    case (x: Number, FT.F64) => s"__longlong_as_double(${java.lang.Double.doubleToRawLongBits(x.doubleValue())}LL)"
    case _ => throw new NotFusible(s"literal $v of type $ft")
  }

  /** Build a kernel for the given outputs. `nullableOut`: outputs for which a validity buffer may be written. */
  def build(outs: Seq[Ex], nameHint: String = "fused"): KernelSpec = {
    val g = new Gen
    val results = outs.map(o => (o, g.gen(o)))
    val body = new StringBuilder
    val numIn = g.inputs.size
    val params = new StringBuilder
    params.append("int n, int* __restrict__ flag")
    g.inputs.zipWithIndex.foreach { case (in, i) =>
      params.append(s",\n    const ${in.ft.ctype}* __restrict__ c$i, const u32* __restrict__ m$i")
    }
    outs.zipWithIndex.foreach { case (o, j) =>
      params.append(s",\n    ${o.ft.ctype}* __restrict__ o$j, u32* __restrict__ om$j")
    }
    body.append("// fused region: ").append(outs.size).append(" outputs, ").append(g.exprCount).append(" ops\n")
    body.append(Prelude.text).append("\n")
    body.append("extern \"C\" __global__ void KNAME(").append(params.toString).append(")\n{\n")
    body.append("  const int stride = gridDim.x * blockDim.x;\n")
    body.append("  const int lane = threadIdx.x & 31;\n")
    body.append("  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {\n")
    body.append("  const int i = base + threadIdx.x;\n  const bool inb = i < n;\n")
    g.lines.foreach(l => body.append(l).append("\n"))
    results.zipWithIndex.foreach { case ((o, (v, ok)), j) =>
      body.append(s"  if (inb) o$j[i] = st_${o.ft.name}($v);\n")
      body.append(s"  { u32 b = __ballot_sync(0xffffffffu, inb && $ok); if (om$j != nullptr && lane == 0 && inb) om$j[i >> 5] = b; }\n")
    }
    body.append("  }\n}\n")
    val src0 = body.toString
    // kernel name derives from the hash of the source with a placeholder, so identical regions share a module
    val h = MessageDigest.getInstance("SHA-256").digest(src0.getBytes("UTF-8")).map("%02x".format(_)).take(8).mkString
    val name = s"${nameHint}_$h"
    val outSpecs = results.map { case (o, _) =>
      KOut(o.ft, g.selfNull.getOrElse(o, false), g.depsOf.getOrElse(o, Set.empty[Int]).toSeq.sorted)
    }
    KernelSpec(name, src0.replace("KNAME", name), g.inputs.toSeq, outSpecs, g.exprCount, g.usesGuard)
  }
}
