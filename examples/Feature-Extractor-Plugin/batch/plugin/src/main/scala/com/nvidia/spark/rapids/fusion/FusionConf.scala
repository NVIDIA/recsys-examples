package com.nvidia.spark.rapids.fusion

import org.apache.spark.sql.internal.SQLConf

/** Snapshot of fusion settings; taken once per query planning so behavior cannot drift mid-query. */
final case class FusionConf(
    mode: String,                 // OFF | SAFE | AGGRESSIVE
    kill: Boolean,
    explain: String,              // NONE | NOT_FUSED | ALL
    shadow: Boolean,
    shadowFailOnMismatch: Boolean,
    shadowEvery: Int,
    minOps: Int,
    maxNodes: Int,
    maxInputs: Int,
    allowOpaque: Boolean,
    fuseProject: Boolean,
    fuseFilter: Boolean,
    fuseAggregate: Boolean,
    mergeProjects: Boolean,
    syncAfterLaunch: Boolean,
    nvrtcPath: String,
    cacheDir: String,
    disabledExprs: Set[String],
    disabledOps: Set[String],
    forceFuse: Boolean,           // ignore the minOps/benefit threshold (testing)
    asyncCompile: Boolean = false // unseen region: run this batch unfused, compile in a background thread, switch when ready
) extends Serializable {
  def enabled: Boolean = !kill && (mode == "SAFE" || mode == "AGGRESSIVE")
  def aggressive: Boolean = mode == "AGGRESSIVE"
  /** In shadow mode: check every `shadowEvery`-th batch only (JVM-wide counter), to validate on production data cheaply. */
  def shadowNow: Boolean = shadow && (shadowEvery <= 1 || FusionConf.shadowCounter.incrementAndGet() % shadowEvery == 0L)
}

object FusionConf {
  val shadowCounter = new java.util.concurrent.atomic.AtomicLong
  @volatile private var warnedMode = false
  val P = "spark.rapids.sql.fusion."
  val MODE = P + "mode"
  val KILL = P + "kill"
  val EXPLAIN = P + "explain"
  val SHADOW = P + "shadow"
  val SHADOW_FAIL = P + "shadow.failOnMismatch"
  val SHADOW_EVERY = P + "shadow.every"
  val MIN_OPS = P + "minOps"
  val MAX_NODES = P + "maxNodes"
  val MAX_INPUTS = P + "maxInputs"
  val ALLOW_OPAQUE = P + "allowOpaque"
  val FUSE_PROJECT = P + "project.enabled"
  val FUSE_FILTER = P + "filter.enabled"
  val FUSE_AGG = P + "aggregate.enabled"
  val MERGE_PROJECTS = P + "project.merge"
  val SYNC = P + "syncAfterLaunch"
  val NVRTC = P + "nvrtcPath"
  val CACHE_DIR = P + "cacheDir"
  val DISABLED_EXPRS = P + "disabledExpressions"   // comma separated Gpu class names, e.g. GpuAdd,GpuCast
  val FORCE = P + "force"
  val ASYNC_COMPILE = P + "asyncCompile"
  val EXPR_PREFIX = P + "expression."              // spark.rapids.sql.fusion.expression.<GpuName>.enabled=false

  def fromSQLConf(c: SQLConf = SQLConf.get): FusionConf = {
    def s(k: String, d: String) = c.getConfString(k, d)
    def b(k: String, d: Boolean) = s(k, d.toString).trim.toBoolean
    def i(k: String, d: Int) = s(k, d.toString).trim.toInt
    val mode0 = s(MODE, "OFF").trim.toUpperCase
    val mode = if (Set("OFF", "SAFE", "AGGRESSIVE").contains(mode0)) mode0 else {
      if (!warnedMode) { warnedMode = true; System.err.println(s"[fusion] unknown $MODE=$mode0 (expected OFF|SAFE|AGGRESSIVE); fusion stays OFF") }
      "OFF"
    }
    val perExpr = c.getAllConfs.collect {
      case (k, v) if k.startsWith(EXPR_PREFIX) && k.endsWith(".enabled") && !v.trim.toBoolean =>
        k.stripPrefix(EXPR_PREFIX).stripSuffix(".enabled")
    }.toSet
    val csv = s(DISABLED_EXPRS, "").split(",").map(_.trim).filter(_.nonEmpty).toSet
    val killEnv = sys.env.get("SPARK_RAPIDS_FUSION_KILL").exists(v => v == "1" || v.equalsIgnoreCase("true"))
    val aggressive = mode == "AGGRESSIVE"
    FusionConf(
      mode = mode,
      kill = b(KILL, false) || killEnv,
      explain = s(EXPLAIN, "NOT_FUSED").trim.toUpperCase,
      shadow = b(SHADOW, false),
      shadowFailOnMismatch = b(SHADOW_FAIL, false),
      shadowEvery = math.max(1, i(SHADOW_EVERY, 1)),
      minOps = i(MIN_OPS, if (aggressive) 1 else 2),
      maxNodes = i(MAX_NODES, if (aggressive) 1024 else 256),
      maxInputs = i(MAX_INPUTS, if (aggressive) 96 else 48),
      allowOpaque = b(ALLOW_OPAQUE, true),
      fuseProject = b(FUSE_PROJECT, true),
      fuseFilter = b(FUSE_FILTER, true),
      fuseAggregate = b(FUSE_AGG, true),
      mergeProjects = b(MERGE_PROJECTS, true),
      syncAfterLaunch = b(SYNC, true),
      nvrtcPath = s(NVRTC, sys.env.getOrElse("FUSION_NVRTC", "")),
      cacheDir = s(CACHE_DIR, sys.env.getOrElse("FUSION_CACHE_DIR", new java.io.File(System.getProperty("java.io.tmpdir"), "gpu-fusion-cubin-" + System.getProperty("user.name")).getPath)),
      disabledExprs = perExpr ++ csv,
      disabledOps = Set.empty,
      forceFuse = b(FORCE, false),
      asyncCompile = b(ASYNC_COMPILE, false))
  }
}
