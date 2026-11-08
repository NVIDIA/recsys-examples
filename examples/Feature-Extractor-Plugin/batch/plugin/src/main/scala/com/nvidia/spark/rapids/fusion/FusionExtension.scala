package com.nvidia.spark.rapids.fusion

import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.execution.{ColumnarRule, SparkPlan}
import org.apache.spark.sql.catalyst.rules.Rule

/**
 * Entry point: `--conf spark.sql.extensions=com.nvidia.spark.rapids.fusion.FusionExtension`.
 * The rule only acts in postColumnarTransitions, which Spark runs after spark-rapids' GpuTransitionOverrides.
 */
class FusionExtension extends (SparkSessionExtensions => Unit) {
  override def apply(ext: SparkSessionExtensions): Unit = {
    ext.injectColumnar(_ => new ColumnarRule {
      private val rule = new FusionRule
      override def postColumnarTransitions: Rule[SparkPlan] = rule
    })
  }
}
