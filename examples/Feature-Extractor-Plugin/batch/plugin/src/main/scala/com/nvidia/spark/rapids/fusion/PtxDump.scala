package com.nvidia.spark.rapids.fusion

import jcuda.nvrtc._

/** Dev tool: print PTX of a kernel source file (FUSION_DUMP_DIR output). Usage: PtxDump file.cu [arch=compute_89] */
object PtxDump {
  def main(args: Array[String]): Unit = {
    NativeBridge.ensureNvrtcOnly(sys.env.getOrElse("FUSION_NVRTC", ""))
    val src = new String(java.nio.file.Files.readAllBytes(new java.io.File(args(0)).toPath), "UTF-8")
    val arch = if (args.length > 1) args(1) else "compute_89"
    val prog = new nvrtcProgram()
    JNvrtc.nvrtcCreateProgram(prog, src, "k.cu", 0, null, null)
    val opts = Array(s"--gpu-architecture=$arch", "--fmad=true", "-default-device", "--std=c++14", "-prec-div=true", "-prec-sqrt=true", "-ftz=false")
    JNvrtc.nvrtcCompileProgram(prog, opts.length, opts)
    val sz = new Array[Long](1); JNvrtc.nvrtcGetPTXSize(prog, sz)
    val ptx = new Array[String](1); JNvrtc.nvrtcGetPTX(prog, ptx)
    println(ptx(0))
    sys.exit(0)
  }
}
