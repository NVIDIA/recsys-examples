package com.nvidia.spark.rapids.fusion

import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.shims.ShimExpression

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * A whole expression tree replaced by one NVRTC kernel, usable wherever spark-rapids evaluates expression
 * trees (aggregate inputs, etc.). `children` are the kernel inputs (attributes or cuDF-evaluated sub-trees) and are bound
 * by spark-rapids like any other children. `fallback` is the unfused tree over the kernel inputs, expressed with
 * already-bound references to input column i, so it can be evaluated on a synthetic batch of the evaluated inputs no
 * matter how spark-rapids rewrote/tiered the children. `original` is the unrewritten tree (display only).
 */
case class FusedGpuExpression(kernel: KernelSpec, inputExprs: Seq[Expression], fallback: Expression,
                              original: Expression, cfg: FusionConf)
  extends ShimExpression with GpuExpression with Logging {

  override def children: Seq[Expression] = inputExprs
  override def dataType: DataType = original.dataType
  override def nullable: Boolean = original.nullable
  override def prettyName: String = "fused"
  override def toString: String = s"fused[${kernel.name}](${original})"
  override def sql: String = original.sql
  override def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression =
    copy(inputExprs = newChildren)

  private def runFallback(cols: Seq[GpuColumnVector], batch: ColumnarBatch): GpuColumnVector = {
    // evaluate the unfused tree over the already evaluated inputs (no re-evaluation of the children)
    val synthetic = new ColumnarBatch(cols.toArray[org.apache.spark.sql.vectorized.ColumnVector], batch.numRows())
    fallback.asInstanceOf[GpuExpression].columnarEval(synthetic)   // synthetic batch does not own the columns: not closed
  }

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    val key = kernel.name
    val temps = new scala.collection.mutable.ArrayBuffer[GpuColumnVector]()
    try {
      var i = 0
      while (i < inputExprs.length) {
        temps += inputExprs(i).asInstanceOf[GpuExpression].columnarEval(batch)
        i += 1
      }
      if (FusionRuntime.killed || FusionRuntime.blacklist.contains(key) || !FusionRuntime.ensureInit(cfg) || batch.numRows() == 0) {
        FusionStats.fallbackBatches.incrementAndGet()
        return runFallback(temps.toSeq, batch)
      }
      try {
        val views = temps.map(_.getBase.asInstanceOf[ai.rapids.cudf.ColumnView]).toArray
        val out = FusionRuntime.run(kernel, views, batch.numRows(), cfg)
        val fusedCol = GpuColumnVector.from(out(0), dataType)
        if (cfg.shadowNow) {
          // shadow mode: also run the unfused tree, compare, and return the UNFUSED result
          val ref = try runFallback(temps.toSeq, batch) catch { case t: Throwable => fusedCol.close(); throw t }
          try {
            FusionStats.shadowChecked.incrementAndGet()
            FusedSupport.sameColumn(fusedCol, ref).foreach { d =>
              FusionStats.shadowMismatches.incrementAndGet()
              logError(s"fusion SHADOW MISMATCH in expression region $key: $d")
              FusionRuntime.blacklist.add(key)
            }
          } catch { case t: Throwable => ref.close(); throw t } finally fusedCol.close()
          ref
        } else {
          FusionStats.fusedBatches.incrementAndGet()
          fusedCol
        }
      } catch {
        case _: UseFallback =>
          FusionStats.fallbackBatches.incrementAndGet()
          runFallback(temps.toSeq, batch)
        case t: Throwable if !FusedSupport.isFatal(t) =>
          FusionRuntime.blacklist.add(key)
          FusionStats.runtimeFailures.incrementAndGet()
          FusedSupport.logFallback(key, t, "fused expression")
          FusionStats.fallbackBatches.incrementAndGet()
          runFallback(temps.toSeq, batch)
      }
    } finally temps.foreach(_.close())
  }
}
