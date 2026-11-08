package com.nvidia.spark.rapids.fusion

import java.util.Optional
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import ai.rapids.cudf._
import jcuda.Pointer

import org.apache.spark.internal.Logging

/** JVM-wide counters; also what tests/benchmarks read to prove a region really ran fused. */
object FusionStats {
  val fusedBatches = new AtomicLong
  val fallbackBatches = new AtomicLong
  val compileFailures = new AtomicLong
  val runtimeFailures = new AtomicLong
  val shadowMismatches = new AtomicLong
  val shadowChecked = new AtomicLong
  val kernelNanos = new AtomicLong
  val allocNanos = new AtomicLong
  val launchNanos = new AtomicLong
  val syncNanos = new AtomicLong
  val wrapNanos = new AtomicLong
  val evalNanos = new AtomicLong
  val buildNanos = new AtomicLong
  val asyncFallbacks = new AtomicLong   // batches run unfused because their kernel was still compiling in the background
  def reset(): Unit = Seq(fusedBatches, fallbackBatches, compileFailures, runtimeFailures, shadowMismatches,
    shadowChecked, asyncFallbacks, kernelNanos, allocNanos, launchNanos, syncNanos, wrapNanos, evalNanos, buildNanos).foreach(_.set(0))
  def timing: String = f"allocMs=${allocNanos.get / 1e6}%.1f launchMs=${launchNanos.get / 1e6}%.1f syncMs=${syncNanos.get / 1e6}%.1f kernelPhaseMs=${kernelNanos.get / 1e6}%.1f preMs=${wrapNanos.get / 1e6}%.1f evalKernelMs=${evalNanos.get / 1e6}%.1f buildFusedMs=${buildNanos.get / 1e6}%.1f"
  override def toString: String =
    s"fused=${fusedBatches.get} fallback=${fallbackBatches.get} compileFail=${compileFailures.get} " +
      s"runtimeFail=${runtimeFailures.get} shadowChecked=${shadowChecked.get} shadowMismatch=${shadowMismatches.get}"
}

/** Signals "run the original (unfused) path for this batch"; not an error. */
final class UseFallback(msg: String) extends RuntimeException(msg, null, false, false)

object FusionRuntime extends Logging {
  /** Regions (kernel keys) that failed once: never tried again in this JVM. */
  val blacklist: java.util.Set[String] = ConcurrentHashMap.newKeySet[String]()
  private val guardTrips = new ConcurrentHashMap[String, AtomicLong]()

  @volatile private var initState = 0 // 0 unknown, 1 ok, -1 failed
  @volatile private var initError: String = ""

  /** Idempotent. Returns true if the NVRTC/driver layer is usable in this JVM. */
  def ensureInit(cfg: FusionConf): Boolean = {
    if (initState == 1) return true
    if (initState == -1) return false
    synchronized {
      if (initState != 0) return initState == 1
      try {
        NativeBridge.cacheDir = cfg.cacheDir
        NativeBridge.ensureLoaded(cfg.nvrtcPath)
        NativeBridge.ensureContext()
        val ok = ColumnLayout.calibrate()
        if (!ok) throw new IllegalStateException("cuDF column_view layout check failed (offset detection unavailable)")
        logInfo(s"fusion runtime ready: arch=${NativeBridge.currentArch()} nvrtc=${NativeBridge.nvrtcVersion()}")
        initState = 1
      } catch {
        case t: Throwable =>
          initError = t.toString
          logWarning(s"fusion runtime unavailable, falling back to unfused execution: $t")
          initState = -1
      }
      initState == 1
    }
  }
  def initFailure: String = initError

  /** Emergency stop that also works on executors: env SPARK_RAPIDS_FUSION_KILL=1 (checked per batch). */
  def killed: Boolean = {
    val v = System.getenv("SPARK_RAPIDS_FUSION_KILL")
    v != null && (v == "1" || v.equalsIgnoreCase("true"))
  }

  /** asyncCompile: make sure a background compile of this kernel is under way (no-op if compiled/failed). Never throws. */
  def prefetch(spec: KernelSpec): Unit =
    try NativeBridge.getKernelAsync(spec.key + "@" + spec.name, spec.name, spec.source) catch { case _: Throwable => }

  private def maskBytes(n: Int): Long = {
    val words = (n.toLong + 31) / 32
    ((words * 4 + 63) / 64) * 64
  }

  /**
   * Run the kernel over `inputs` (cuDF views, all `rows` long) and return freshly allocated output columns.
   * Throws UseFallback for conditions that are legitimate "not this batch" situations.
   */
  def run(spec: KernelSpec, inputs: Array[ColumnView], rows: Int, cfg: FusionConf): Array[ColumnVector] = {
    val tPre0 = System.nanoTime()
    if (rows == 0) throw new UseFallback("empty batch")
    var i = 0
    while (i < inputs.length) {
      val c = inputs(i)
      val want = spec.inputs(i).ft
      if (c.getType != want.cudf) throw new UseFallback(s"input $i has type ${c.getType}, expected ${want.cudf}")
      if (c.getRowCount != rows) throw new UseFallback("input row count mismatch")
      if (c.getData == null) throw new UseFallback(s"input $i has no data buffer")
      if (ColumnLayout.offsetOf(c) != 0) throw new UseFallback(s"input $i is a sliced view (offset != 0)")
      i += 1
    }
    val kernel = try {
      if (cfg.asyncCompile) {
        val k = NativeBridge.getKernelAsync(spec.key + "@" + spec.name, spec.name, spec.source)
        if (k == null) { FusionStats.asyncFallbacks.incrementAndGet(); throw new UseFallback("kernel is compiling in the background") }
        k
      } else NativeBridge.getKernel(spec.key + "@" + spec.name, spec.name, spec.source)
    } catch { case u: UseFallback => throw u; case t: Throwable => FusionStats.compileFailures.incrementAndGet(); throw t }
    FusionStats.wrapNanos.addAndGet(System.nanoTime() - tPre0)   // pre-launch checks + module lookup

    val tAlloc0 = System.nanoTime()
    val dataBufs = new Array[DeviceMemoryBuffer](spec.outputs.length)
    val validBufs = new Array[DeviceMemoryBuffer](spec.outputs.length)
    var guardBuf: DeviceMemoryBuffer = null
    try {
      var j = 0
      while (j < spec.outputs.length) {
        val o = spec.outputs(j)
        dataBufs(j) = DeviceMemoryBuffer.allocate(rows.toLong * o.ft.size)
        val needValid = o.selfNull || o.deps.exists(d => inputs(d).getValid != null)
        if (needValid) {
          val mb = maskBytes(rows)
          validBufs(j) = DeviceMemoryBuffer.allocate(mb)
          // the kernel writes words [0, ceil(rows/32)); define the padding words too so no consumer ever reads garbage
          val used = ((rows.toLong + 31) / 32) * 4
          if (mb > used) Cuda.asyncMemset(validBufs(j).getAddress + used, 0.toByte, mb - used)
        }
        j += 1
      }
      FusionStats.allocNanos.addAndGet(System.nanoTime() - tAlloc0)
      val params = new Array[Pointer](2 + 2 * inputs.length + 2 * spec.outputs.length)
      var p = 0
      params(p) = Pointer.to(Array(rows)); p += 1
      val flagBuf: DeviceMemoryBuffer = if (spec.usesGuard) { val b = DeviceMemoryBuffer.allocate(4L); Cuda.asyncMemset(b.getAddress, 0.toByte, 4L); b } else null
      guardBuf = flagBuf
      params(p) = Pointer.to(Array(if (flagBuf == null) 0L else flagBuf.getAddress)); p += 1
      i = 0
      while (i < inputs.length) {
        params(p) = Pointer.to(Array(inputs(i).getData.getAddress)); p += 1
        val v = inputs(i).getValid
        params(p) = Pointer.to(Array(if (v == null) 0L else v.getAddress)); p += 1
        i += 1
      }
      j = 0
      while (j < spec.outputs.length) {
        params(p) = Pointer.to(Array(dataBufs(j).getAddress)); p += 1
        params(p) = Pointer.to(Array(if (validBufs(j) == null) 0L else validBufs(j).getAddress)); p += 1
        j += 1
      }
      // very register-hungry kernels may not allow 256 threads per block
      val block = math.max(32, math.min(256, (kernel.maxThreads / 32) * 32))
      val grid = math.max(1, math.min((rows + block - 1) / block, NativeBridge.maxGrid))
      val t0 = System.nanoTime()
      val nvtx = new NvtxRange("fused " + spec.name, NvtxColor.GREEN)
      try NativeBridge.launch(kernel, grid, block, params, cfg.syncAfterLaunch || spec.usesGuard) finally nvtx.close()
      FusionStats.kernelNanos.addAndGet(System.nanoTime() - t0)
      if (guardBuf != null) {
        val tripped = { val h = HostMemoryBuffer.allocate(4L); try { h.copyFromDeviceBuffer(guardBuf); h.getInt(0L) != 0 } finally h.close() }
        guardBuf.close(); guardBuf = null
        if (tripped) {
          // a region whose guard trips again and again only wastes work: stop fusing it in this JVM
          val n = guardTrips.computeIfAbsent(spec.name, _ => new AtomicLong).incrementAndGet()
          if (n >= 3 && blacklist.add(spec.name)) logInfo(s"fusion: region ${spec.name} hit its domain guard $n times; using the unfused path from now on")
          throw new UseFallback("domain guard tripped (input outside the range verified to match cuDF)")
        }
      }
      val out = new Array[ColumnVector](spec.outputs.length)
      j = 0
      try {
        while (j < out.length) {
          // the ColumnVector takes ownership of (does not re-reference) the buffers
          out(j) = new ColumnVector(spec.outputs(j).ft.cudf, rows.toLong, Optional.empty[java.lang.Long](),
            dataBufs(j), validBufs(j), null)
          dataBufs(j) = null; validBufs(j) = null
          j += 1
        }
      } catch { case t: Throwable => out.foreach(c => if (c != null) c.close()); throw t }
      out
    } catch {
      case t: Throwable =>
        if (guardBuf != null) try guardBuf.close() catch { case _: Throwable => }
        dataBufs.foreach(b => if (b != null) try b.close() catch { case _: Throwable => })
        validBufs.foreach(b => if (b != null) try b.close() catch { case _: Throwable => })
        throw t
    }
  }
}

/** Reads `offset` of a cuDF column_view through its native handle (calibrated at startup against known layout). */
object ColumnLayout {
  private lazy val unsafe: sun.misc.Unsafe = {
    val f = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe"); f.setAccessible(true)
    f.get(null).asInstanceOf[sun.misc.Unsafe]
  }
  // cudf::column_view_base (cuDF 26.08): vptr(8) data_type(8) size(4)+pad(4) data*(8) null_mask*(8) null_count(4) offset(4)
  private val OFF_SIZE = 16L; private val OFF_DATA = 24L; private val OFF_MASK = 32L; private val OFF_OFFSET = 44L

  def offsetOf(c: ColumnView): Int = unsafe.getInt(c.getNativeView + OFF_OFFSET)

  def calibrate(): Boolean = {
    val col = ColumnVector.fromBoxedInts(1, null, 3, 4, 5, 6, 7, 8, 9, 10)
    try {
      val h = col.getNativeView
      val ok = unsafe.getInt(h + OFF_SIZE) == 10 &&
        unsafe.getLong(h + OFF_DATA) == col.getData.getAddress &&
        unsafe.getLong(h + OFF_MASK) == col.getValid.getAddress &&
        unsafe.getInt(h + OFF_OFFSET) == 0
      ok
    } finally col.close()
  }
}
