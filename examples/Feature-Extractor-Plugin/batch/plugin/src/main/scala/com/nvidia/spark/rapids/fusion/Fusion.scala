package com.nvidia.spark.rapids.fusion

import org.apache.spark.sql.DataFrame

/** Small public API (Scala; PySpark via `sc._jvm.com.nvidia.spark.rapids.fusion.Fusion.explain(df._jdf)`). */
object Fusion {
  /**
   * Plans `df` statically (AQE temporarily off, so the whole plan is visible) and returns the fusion decisions
   * (FUSED regions and "NOT FUSED ...: reason" lines) followed by the physical plan. Does not execute the query.
   * Needs `spark.rapids.sql.fusion.mode` != OFF to show anything.
   */
  def explain(df: DataFrame): String = {
    val s = df.sparkSession
    val key = "spark.sql.adaptive.enabled"
    val old = s.conf.getOption(key)
    s.conf.set(key, "false")
    try {
      FusionReport.clear()
      val qe = s.sessionState.executePlan(df.queryExecution.logical)
      val plan = qe.executedPlan
      val decisions = FusionReport.dump().map(_.replaceAll("\\s+", " ")).distinct
      s"== Fusion decisions (${FusionConf.fromSQLConf(s.sessionState.conf).mode}) ==\n" +
        (if (decisions.isEmpty) "(none: no GpuProject/GpuFilter/GpuHashAggregate found or fusion disabled)" else decisions.mkString("\n")) +
        "\n== Physical plan ==\n" + plan.treeString
    } finally {
      old match { case Some(v) => s.conf.set(key, v); case None => s.conf.unset(key) }
    }
  }

  /** Planner statistics: how often each "not fused" reason occurred in this JVM (most frequent first). */
  def reasonSummary: String = {
    val b = new scala.collection.mutable.ArrayBuffer[(String, Long)]()
    val it = FusionReport.reasons.entrySet().iterator()
    while (it.hasNext) { val e = it.next(); b += ((e.getKey, e.getValue.get())) }
    if (b.isEmpty) "(no 'not fused' events recorded)" else b.sortBy(-_._2).map { case (r, n) => f"$n%8d  $r" }.mkString("\n")
  }

  /** JVM-wide counters of this process (driver in local mode; executors keep their own). */
  def stats: String = FusionStats.toString + " " + FusionStats.timing
  def resetStats(): Unit = FusionStats.reset()
}
