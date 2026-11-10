package com.nvidia.spark.rapids.fusion

import java.util.concurrent.ConcurrentHashMap

import jcuda.{Pointer, Sizeof}
import jcuda.driver._
import jcuda.nvrtc._

/**
 * Thin NVRTC + CUDA driver API layer (JCuda). One compiled module per (source hash, arch) per JVM.
 * Streams: kernels are launched on the CUDA per-thread default stream (handle 2), which is what
 * cuDF/RMM use as the default stream in spark-rapids builds (PTDS). To stay safe regardless of the
 * stream cuDF used, callers can request a synchronize after launch (see FusionConf.syncAfterLaunch).
 */
object NativeBridge {
  private val loaded = new java.util.concurrent.atomic.AtomicBoolean(false)
  @volatile private var initError: Throwable = _

  def ensureLoaded(nvrtcPath: String): Unit = {
    if (loaded.get()) return
    synchronized {
      if (loaded.get()) return
      try {
        if (nvrtcPath != null && nvrtcPath.nonEmpty) {
          val f = new java.io.File(nvrtcPath)
          val lib = if (f.isDirectory) new java.io.File(f, "libnvrtc.so.12") else f
          if (lib.exists()) {
            val b = new java.io.File(lib.getParentFile, "libnvrtc-builtins.so.12.9")
            // builtins is dlopen'ed lazily by libnvrtc itself, found via rpath/LD_LIBRARY_PATH or same dir
            System.load(lib.getAbsolutePath)
          }
        }
        JCudaDriver.setExceptionsEnabled(true)
        JNvrtc.setExceptionsEnabled(true)
        JCudaDriver.cuInit(0)
        loaded.set(true)
      } catch {
        case t: Throwable => initError = t; throw t
      }
    }
  }

  /** NVRTC only (no driver/GPU needed): for compile-only tests on machines without a GPU. */
  def ensureNvrtcOnly(nvrtcPath: String): Unit = {
    if (nvrtcPath != null && nvrtcPath.nonEmpty) { val f = new java.io.File(nvrtcPath); if (f.exists()) System.load(f.getAbsolutePath) }
    JNvrtc.setExceptionsEnabled(true)
  }

  /** Make sure the calling thread has a current CUDA context (the primary context cuDF uses). */
  def ensureContext(): Unit = {
    val cur = new CUcontext()
    JCudaDriver.cuCtxGetCurrent(cur)
    if (cur == new CUcontext()) {
      val dev = new CUdevice()
      val devIdx = new Array[Int](1)
      // runtime API (cuDF) always uses the device chosen via cudaSetDevice; pick current if known
      JCudaDriver.cuDeviceGet(dev, 0)
      val ctx = new CUcontext()
      JCudaDriver.cuDevicePrimaryCtxRetain(ctx, dev)
      JCudaDriver.cuCtxSetCurrent(ctx)
    }
  }

  def currentArch(): String = {
    ensureContext()
    val dev = new CUdevice()
    JCudaDriver.cuCtxGetDevice(dev)
    val maj = new Array[Int](1); val min = new Array[Int](1)
    JCudaDriver.cuDeviceGetAttribute(maj, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    JCudaDriver.cuDeviceGetAttribute(min, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    s"sm_${maj(0)}${min(0)}"
  }

  def nvrtcVersion(): String = {
    val a = new Array[Int](1); val b = new Array[Int](1)
    JNvrtc.nvrtcVersion(a, b); s"${a(0)}.${b(0)}"
  }

  /** Compile CUDA C++ to a cubin for the given real arch. Throws with the NVRTC log on failure. */
  def compileToCubin(src: String, arch: String, extraOpts: Seq[String] = Nil): Array[Byte] = {
    val prog = new nvrtcProgram()
    JNvrtc.nvrtcCreateProgram(prog, src, "fused.cu", 0, null, null)
    try {
      val opts = Array(s"--gpu-architecture=$arch", s"--fmad=${sys.env.getOrElse("FUSION_FMAD", sys.props.getOrElse("fusion.fmad", "true"))}", "-default-device", "--std=c++14",
        "-prec-div=true", "-prec-sqrt=true", "-ftz=false") ++ extraOpts
      val rc = JNvrtc.nvrtcCompileProgram(prog, opts.length, opts)
      val logSize = new Array[Long](1)
      JNvrtc.nvrtcGetProgramLogSize(prog, logSize)
      val logArr = new Array[String](1)
      JNvrtc.nvrtcGetProgramLog(prog, logArr)
      val log = Option(logArr(0)).getOrElse("").getBytes
      if (rc != nvrtcResult.NVRTC_SUCCESS) {
        throw new RuntimeException(s"NVRTC compile failed (${nvrtcResult.stringFor(rc)}): ${new String(log)}")
      }
      val sz = new Array[Long](1)
      JNvrtc.nvrtcGetCUBINSize(prog, sz)
      val cubin = new Array[Byte](sz(0).toInt)
      JNvrtc.nvrtcGetCUBIN(prog, cubin)
      cubin
    } finally {
      JNvrtc.nvrtcDestroyProgram(prog)
    }
  }

  final class Kernel(val module: CUmodule, val function: CUfunction, val name: String, val maxThreads: Int = 256)

  private val modules = new ConcurrentHashMap[String, Kernel]()
  val kernelInfo = new ConcurrentHashMap[String, String]()

  /** Directory for the on-disk cubin cache ("" disables). Keyed by source hash + arch + NVRTC version + fmad flag. */
  @volatile var cacheDir: String = ""
  val compileNanos = new java.util.concurrent.atomic.AtomicLong
  val compileCount = new java.util.concurrent.atomic.AtomicLong
  val cacheHits = new java.util.concurrent.atomic.AtomicLong

  private def fmadFlag: String = sys.env.getOrElse("FUSION_FMAD", sys.props.getOrElse("fusion.fmad", "true"))

  private def cubinFor(key: String, src: String, arch: String): Array[Byte] = {
    val dir = if (cacheDir == null || cacheDir.isEmpty) null else new java.io.File(cacheDir)
    val file = if (dir == null) null else new java.io.File(dir, s"$arch-nvrtc${nvrtcVersion()}-fmad$fmadFlag-$key.cubin")
    if (file != null && file.isFile) {
      try { val b = java.nio.file.Files.readAllBytes(file.toPath); if (b.length > 0) { cacheHits.incrementAndGet(); return b } }
      catch { case _: Throwable => }
    }
    val t0 = System.nanoTime()
    val cubin = compileToCubin(src, arch)
    compileNanos.addAndGet(System.nanoTime() - t0); compileCount.incrementAndGet()
    if (file != null) {
      try {
        if (!dir.exists()) {   // private directory: cached cubins are loaded and executed on the GPU
          try java.nio.file.Files.createDirectories(dir.toPath, java.nio.file.attribute.PosixFilePermissions.asFileAttribute(
            java.nio.file.attribute.PosixFilePermissions.fromString("rwx------")))
          catch { case _: Throwable => dir.mkdirs() }
        }
        val tmp = new java.io.File(dir, file.getName + "." + java.lang.Long.toHexString(System.nanoTime()) + ".tmp")
        java.nio.file.Files.write(tmp.toPath, cubin)
        java.nio.file.Files.move(tmp.toPath, file.toPath, java.nio.file.StandardCopyOption.ATOMIC_MOVE,
          java.nio.file.StandardCopyOption.REPLACE_EXISTING)
      } catch { case _: Throwable => /* cache is best effort */ }
    }
    cubin
  }

  // ---- asynchronous compilation (spark.rapids.sql.fusion.asyncCompile) ----
  private val pending = new ConcurrentHashMap[String, java.lang.Boolean]()
  /** Compile failures of background compiles, by kernel key; rethrown to the first batch that asks again (-> normal blacklist path). */
  val asyncFailures = new ConcurrentHashMap[String, Throwable]()
  val asyncSubmitted = new java.util.concurrent.atomic.AtomicLong
  private lazy val compilePool = java.util.concurrent.Executors.newSingleThreadExecutor(new java.util.concurrent.ThreadFactory {
    def newThread(r: Runnable): Thread = { val t = new Thread(r, "fusion-nvrtc-compile"); t.setDaemon(true); t }
  })
  def pendingCompiles: Int = pending.size
  /** Test/bench helper: wait until no background compile is running. */
  def awaitCompiles(timeoutMs: Long): Boolean = {
    val end = System.currentTimeMillis() + timeoutMs
    while (!pending.isEmpty && System.currentTimeMillis() < end) Thread.sleep(2)
    pending.isEmpty
  }

  /**
   * Non-blocking variant of [[getKernel]]: returns the kernel if it is already compiled (or its cubin is in the on-disk cache,
   * where loading costs milliseconds), otherwise schedules ONE background compile per key and returns null (caller runs the batch unfused).
   * A failed background compile is rethrown here so the caller's usual failure handling (blacklist the region) applies.
   */
  def getKernelAsync(key: String, name: String, src: String): Kernel = {
    val k = modules.get(key)
    if (k != null) return k
    val f = asyncFailures.get(key)
    if (f != null) throw new RuntimeException(s"background compile of $name failed: $f", f)
    if (cubinOnDisk(key)) return getKernel(key, name, src)
    if (pending.putIfAbsent(key, java.lang.Boolean.TRUE) == null) {
      try {
        asyncSubmitted.incrementAndGet()
        compilePool.execute(new Runnable { def run(): Unit = {
          try getKernel(key, name, src)
          catch { case t: Throwable => asyncFailures.put(key, t) }
          finally pending.remove(key)
        } })
      } catch { case t: Throwable => pending.remove(key); throw t }
    }
    null
  }

  private def cubinOnDisk(key: String): Boolean = {
    if (cacheDir == null || cacheDir.isEmpty) return false
    new java.io.File(cacheDir, s"${currentArch()}-nvrtc${nvrtcVersion()}-fmad$fmadFlag-${key.takeWhile(_ != '@')}.cubin").isFile
  }

  /** Test-only fault injection: -Dfusion.test.failCompile=<regex on kernel name>, -Dfusion.test.compileDelayMs=<ms> (applies to every NVRTC compile). */
  private def testHooks(name: String): Unit = {
    val d = sys.props.get("fusion.test.compileDelayMs").map(_.toLong).getOrElse(0L)
    if (d > 0) Thread.sleep(d)
    sys.props.get("fusion.test.failCompile").foreach { re => if (name.matches(re)) throw new RuntimeException("injected compile failure for " + name) }
  }

  /** Compile + load (cached per JVM by key). The module is loaded into the current context. */
  def getKernel(key: String, name: String, src: => String): Kernel = {
    val k = modules.get(key)
    if (k != null) return k
    synchronized {
      val k2 = modules.get(key)
      if (k2 != null) return k2
      ensureContext()
      sys.env.get("FUSION_DUMP_DIR").foreach { d =>   // developer aid: dump generated CUDA source of every kernel
        try { new java.io.File(d).mkdirs(); java.nio.file.Files.write(new java.io.File(d, name + ".cu").toPath, src.getBytes("UTF-8")) } catch { case _: Throwable => }
      }
      testHooks(name)
      val cubin = cubinFor(key.takeWhile(_ != '@'), src, currentArch())
      val m = new CUmodule()
      JCudaDriver.cuModuleLoadData(m, cubin)
      val f = new CUfunction()
      JCudaDriver.cuModuleGetFunction(f, m, name)
      val maxT = new Array[Int](1)
      try JCudaDriver.cuFuncGetAttribute(maxT, CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f) catch { case _: Throwable => maxT(0) = 256 }
      val kk = new Kernel(m, f, name, maxT(0))
      try {
        val regs = new Array[Int](1); val loc = new Array[Int](1)
        JCudaDriver.cuFuncGetAttribute(regs, CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, f)
        JCudaDriver.cuFuncGetAttribute(loc, CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)
        JCudaDriver.cuFuncGetAttribute(maxT, CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f)
        kernelInfo.put(name, s"regs=${regs(0)} localBytes=${loc(0)} maxThreads=${maxT(0)} cubinBytes=${cubin.length}")
      } catch { case _: Throwable => }
      modules.put(key, kk)
      kk
    }
  }

  /** CU_STREAM_PER_THREAD (handle 2): the stream cuDF's PTDS build uses as its "default" stream. */
  lazy val perThreadStream: CUstream = {
    val c = classOf[CUstream].getDeclaredConstructor(classOf[Long])
    c.setAccessible(true)
    c.newInstance(java.lang.Long.valueOf(2L))
  }

  /** Upper bound for the launch grid: a small multiple of the SM count (grid-stride loop covers the rest). */
  lazy val maxGrid: Int = {
    ensureContext()
    val dev = new CUdevice(); JCudaDriver.cuCtxGetDevice(dev)
    val sms = new Array[Int](1)
    JCudaDriver.cuDeviceGetAttribute(sms, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)
    math.max(1, sms(0) * 32)
  }

  def launch(k: Kernel, grid: Int, block: Int, params: Array[Pointer], sync: Boolean): Unit = {
    val kp = Pointer.to(params: _*)
    val t0 = System.nanoTime()
    JCudaDriver.cuLaunchKernel(k.function, grid, 1, 1, block, 1, 1, 0, perThreadStream, kp, null)
    val t1 = System.nanoTime()
    FusionStats.launchNanos.addAndGet(t1 - t0)
    if (sync) { JCudaDriver.cuStreamSynchronize(perThreadStream); FusionStats.syncNanos.addAndGet(System.nanoTime() - t1) }
  }
}
