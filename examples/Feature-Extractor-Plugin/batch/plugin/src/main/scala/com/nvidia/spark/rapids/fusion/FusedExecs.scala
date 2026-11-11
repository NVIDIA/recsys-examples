package com.nvidia.spark.rapids.fusion

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf
import ai.rapids.cudf.{ColumnView, DType, Scalar}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.{splitSpillableInHalfByRows, withRetry}
import com.nvidia.spark.rapids.shims._

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

/** Where a kernel input comes from. */
sealed trait InSrc extends Serializable
final case class FromCol(idx: Int) extends InSrc
/** Column `idx` of the "side batch": expressions spark-rapids evaluates for us (opaque sub-trees, unfusable outputs) through
 *  ONE tiered project, so common sub-expressions between them are still computed once. */
final case class FromSide(idx: Int) extends InSrc

/** How one project output is produced. */
sealed trait OutSpec extends Serializable
final case class OutCopy(idx: Int) extends OutSpec
final case class OutKernel(part: Int, k: Int) extends OutSpec
final case class OutSide(idx: Int) extends OutSpec

/** The serializable "fusion decision" attached to a fused exec. */
/** One generated kernel and where its inputs come from. A project that exceeds the region limits is split into several. */
final case class KernelPart(spec: KernelSpec, inputs: Seq[InSrc]) extends Serializable
final case class FusedProjectPlan(parts: Seq[KernelPart], outs: Seq[OutSpec],
                                  outTypes: Seq[DataType], sideExprs: Seq[Expression], cfg: FusionConf,
                                  desc: String) extends Serializable {
  def keys: Seq[String] = parts.map(_.spec.name)
}
final case class FusedFilterPlan(kernel: KernelSpec, inputs: Seq[InSrc], sideExprs: Seq[Expression], cfg: FusionConf,
                                 desc: String) extends Serializable

object FusedSupport extends Logging {
  def isFatal(t: Throwable): Boolean = t match {
    case _: InterruptedException | _: VirtualMachineError => true
    case _ =>
      val n = t.getClass.getName
      // memory-pressure signals must reach spark-rapids' retry framework untouched
      n.contains("RetryOOM") || n.contains("SplitAndRetry") || n.contains("OutOfMemory") ||
        n.contains("RmmException") || n.contains("TaskKilled") || n.contains("TaskCompletion")
  }

  /** Run the kernel; inputs come from the batch or from the side batch (cuDF-evaluated expressions). */
  def evalKernel(spec: KernelSpec, inputs: Seq[InSrc], cb: ColumnarBatch, side: Option[ColumnarBatch],
                 cfg: FusionConf): Array[cudf.ColumnVector] = {
    val views = new Array[ColumnView](inputs.length)
    var i = 0
    while (i < inputs.length) {
      inputs(i) match {
        case FromCol(idx) => views(i) = cb.column(idx).asInstanceOf[GpuColumnVector].getBase
        case FromSide(j) => views(i) = side.get.column(j).asInstanceOf[GpuColumnVector].getBase
      }
      i += 1
    }
    FusionRuntime.run(spec, views, cb.numRows(), cfg)
  }

  def bindSide(sideExprs: Seq[Expression], child: SparkPlan, conf: org.apache.spark.sql.internal.SQLConf,
               metrics: Map[String, GpuMetric]): Option[GpuTieredProject] =
    if (sideExprs.isEmpty) None
    else Some(GpuBindReferences.bindGpuReferencesTiered(sideExprs.toList, child.output, conf, metrics))

  def logFallback(key: String, t: Throwable, what: String): Unit = {
    logWarning(s"fusion: $what failed for region $key; blacklisting it and using the unfused path. Cause: $t")
  }

  /** Bitwise comparison of two columns, NaNs equal each other; used only by shadow mode. */
  def sameColumn(a: GpuColumnVector, b: GpuColumnVector): Option[String] = {
    val ha = a.getBase.copyToHost(); val hb = b.getBase.copyToHost()
    try {
      if (a.getBase.getType != b.getBase.getType) return Some(s"type ${a.getBase.getType} vs ${b.getBase.getType}")
      val n = a.getRowCount.toInt
      var i = 0
      while (i < n) {
        val na = ha.isNull(i); val nb = hb.isNull(i)
        if (na != nb) return Some(s"row $i validity differs: fused null=$na, ref null=$nb")
        if (!na) {
          val same = a.getBase.getType match {
            case DType.BOOL8 | DType.INT8 => ha.getByte(i) == hb.getByte(i)
            case DType.INT16 => ha.getShort(i) == hb.getShort(i)
            case DType.INT32 | DType.TIMESTAMP_DAYS => ha.getInt(i) == hb.getInt(i)
            case DType.INT64 | DType.TIMESTAMP_MICROSECONDS => ha.getLong(i) == hb.getLong(i)
            case DType.FLOAT32 =>
              val x = ha.getFloat(i); val y = hb.getFloat(i)
              (x.isNaN && y.isNaN) || java.lang.Float.floatToRawIntBits(x) == java.lang.Float.floatToRawIntBits(y)
            case DType.FLOAT64 =>
              val x = ha.getDouble(i); val y = hb.getDouble(i)
              (x.isNaN && y.isNaN) || java.lang.Double.doubleToRawLongBits(x) == java.lang.Double.doubleToRawLongBits(y)
            case _ => true
          }
          if (!same) return Some(s"row $i value differs (type ${a.getBase.getType})")
        }
        i += 1
      }
      None
    } finally { ha.close(); hb.close() }
  }

  def compareBatches(fused: ColumnarBatch, ref: ColumnarBatch): Option[String] = {
    var c = 0
    while (c < fused.numCols()) {
      val r = sameColumn(fused.column(c).asInstanceOf[GpuColumnVector], ref.column(c).asInstanceOf[GpuColumnVector])
      if (r.isDefined) return Some(s"column $c: ${r.get}")
      c += 1
    }
    None
  }
}

case class FusedGpuProjectExec(
    projectList: List[NamedExpression],
    child: SparkPlan,
    enablePreSplit: Boolean,
    fusedPlan: FusedProjectPlan) extends GpuProjectExecLike with Logging {

  override def output: Seq[Attribute] = projectList.map(_.toAttribute)

  override def outputBatching: CoalesceGoal =
    if (enablePreSplit) TargetSize(PreProjectSplitIterator.getSplitUntilSize) else super.outputBatching

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    "fusedBatches" -> createMetric(MODERATE_LEVEL, "fused batches"),
    "fallbackBatches" -> createMetric(MODERATE_LEVEL, "fallback (unfused) batches"),
    "fusedKernelTime" -> createNanoTimingMetric(MODERATE_LEVEL, "fused kernel time"))

  override def nodeName: String = "FusedGpuProjectExec"

  override def stringArgs: Iterator[Any] = Iterator(projectList, enablePreSplit, fusedPlan.desc)

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val numOutputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val opTime = gpuLongMetric(OP_TIME_LEGACY)
    val fusedM = gpuLongMetric("fusedBatches")
    val fallbackM = gpuLongMetric("fallbackBatches")
    val kernelM = gpuLongMetric("fusedKernelTime")
    // the original (unfused) implementation, always available as fallback
    val boundProjectList = GpuBindReferences.bindGpuReferencesTiered(projectList, child.output, conf, allMetrics)
    val boundSide = FusedSupport.bindSide(fusedPlan.sideExprs, child, conf, allMetrics)
    val fp = fusedPlan
    val cfg = fp.cfg
    val rdd = child.executeColumnar()
    rdd.mapPartitionsWithIndex { (partIndex, iter) =>
      GpuNondeterministic.initializeAll(boundProjectList.exprTiers.flatten, partIndex)
      iter.flatMap { cb =>
        val pieces = NvtxIdWithMetrics(NvtxRegistry.PROJECT_EXEC, opTime) {
          val sb = SpillableColumnarBatch(cb, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
          GpuProjectExec.runStreamingWithSplitRetry(sb, Seq.empty[com.nvidia.spark.Retryable], { b =>
            projectOne(b, fp, boundProjectList, boundSide, fusedM, fallbackM, kernelM)
          })
        }
        pieces.map { ret =>
          numOutputBatches += 1
          numOutputRows += ret.numRows()
          ret
        }
      }
    }
  }

  private def projectOne(cb: ColumnarBatch, fp: FusedProjectPlan, orig: GpuTieredProject,
                         boundSide: Option[GpuTieredProject],
                         fusedM: GpuMetric, fallbackM: GpuMetric, kernelM: GpuMetric): ColumnarBatch = {
    val key = fp.keys.mkString("+")
    if (FusionRuntime.killed || fp.keys.exists(FusionRuntime.blacklist.contains) || !FusionRuntime.ensureInit(fp.cfg)) {
      fallbackM += 1; FusionStats.fallbackBatches.incrementAndGet()
      return orig.project(cb)
    }
    val fused: Option[ColumnarBatch] = try {
      val t0 = System.nanoTime()
      val r = buildFused(cb, fp, boundSide)
      kernelM += (System.nanoTime() - t0)
      Some(r)
    } catch {
      case _: UseFallback => None
      case t: Throwable if !FusedSupport.isFatal(t) =>
        fp.keys.foreach(FusionRuntime.blacklist.add)
        FusionStats.runtimeFailures.incrementAndGet()
        FusedSupport.logFallback(key, t, "fused execution")
        None
    }
    fused match {
      case None =>
        fallbackM += 1; FusionStats.fallbackBatches.incrementAndGet()
        orig.project(cb)
      case Some(f) if fp.cfg.shadowNow =>
        // run the original too, compare, return the ORIGINAL result
        closeOnExcept(f) { _ =>
          val ref = orig.project(cb)
          closeOnExcept(ref) { _ =>
            FusionStats.shadowChecked.incrementAndGet()
            val diff = FusedSupport.compareBatches(f, ref)
            f.close()
            diff.foreach { d =>
              FusionStats.shadowMismatches.incrementAndGet()
              logError(s"fusion SHADOW MISMATCH in region $key: $d")
              fp.keys.foreach(FusionRuntime.blacklist.add)
            }
            fallbackM += 0
            ref
          }
        }
      case Some(f) =>
        fusedM += 1; FusionStats.fusedBatches.incrementAndGet()
        f
    }
  }

  private def buildFused(cb: ColumnarBatch, fp: FusedProjectPlan, boundSide: Option[GpuTieredProject]): ColumnarBatch = {
    val tAll = System.nanoTime()
    val side: Option[ColumnarBatch] = boundSide.map(_.project(cb))
    try buildFused0(cb, fp, side) finally {
      side.foreach(_.close())
      FusionStats.buildNanos.addAndGet(System.nanoTime() - tAll)
    }
  }

  private def buildFused0(cb: ColumnarBatch, fp: FusedProjectPlan, side: Option[ColumnarBatch]): ColumnarBatch = {
    val tb0 = System.nanoTime()
    val kouts = new ArrayBuffer[Array[cudf.ColumnVector]]()
    if (fp.cfg.asyncCompile) fp.parts.foreach(part => FusionRuntime.prefetch(part.spec))   // start every part's background compile at once
    try {
      fp.parts.foreach(part => kouts += FusedSupport.evalKernel(part.spec, part.inputs, cb, side, fp.cfg))
    } catch { case t: Throwable => kouts.foreach(_.foreach(_.close())); throw t }
    FusionStats.evalNanos.addAndGet(System.nanoTime() - tb0)
    val kcols = kouts.map(a => new Array[GpuColumnVector](a.length))
    val usedOnce = kouts.map(a => new Array[Boolean](a.length))
    val made = new ArrayBuffer[GpuColumnVector]()
    try {
      val cols = new Array[org.apache.spark.sql.vectorized.ColumnVector](fp.outs.length)
      var i = 0
      while (i < cols.length) {
        cols(i) = fp.outs(i) match {
          case OutCopy(idx) => val c = cb.column(idx).asInstanceOf[GpuColumnVector].incRefCount(); made += c; c
          case OutKernel(pi, k) =>
            if (!usedOnce(pi)(k)) {
              usedOnce(pi)(k) = true
              kcols(pi)(k) = GpuColumnVector.from(kouts(pi)(k), fp.outTypes(i)); made += kcols(pi)(k); kcols(pi)(k)
            } else { val c = kcols(pi)(k).incRefCount(); made += c; c }
          case OutSide(j) => val c = side.get.column(j).asInstanceOf[GpuColumnVector].incRefCount(); made += c; c
        }
        i += 1
      }
      // kernel outputs never wrapped (not referenced by any output) must still be freed
      var pi = 0
      while (pi < kouts.length) { var k = 0; while (k < kouts(pi).length) { if (!usedOnce(pi)(k)) kouts(pi)(k).close(); k += 1 }; pi += 1 }
      new ColumnarBatch(cols, cb.numRows())
    } catch {
      case t: Throwable =>
        made.foreach(c => try c.close() catch { case _: Throwable => })
        var pi = 0   // kernel outputs that were not wrapped yet still belong to us
        while (pi < kouts.length) { var k = 0; while (k < kouts(pi).length) { if (!usedOnce(pi)(k)) try kouts(pi)(k).close() catch { case _: Throwable => }; k += 1 }; pi += 1 }
        throw t
    }
  }
}

case class FusedGpuFilterExec(
    condition: Expression,
    child: SparkPlan,
    outputAttrs: Seq[Attribute],
    fusedPlan: FusedFilterPlan)(
    override val coalesceAfter: Boolean = true)
  extends ShimUnaryExecNode with GpuExec with Logging {

  override def otherCopyArgs: Seq[AnyRef] = Seq[AnyRef](coalesceAfter.asInstanceOf[java.lang.Boolean])
  override def output: Seq[Attribute] = outputAttrs
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException(s"Row-based execution should not occur for $this")
  override val outputRowsLevel: MetricsLevel = ESSENTIAL_LEVEL
  override val outputBatchesLevel: MetricsLevel = MODERATE_LEVEL
  override def nodeName: String = "FusedGpuFilterExec"
  override def stringArgs: Iterator[Any] = Iterator(condition, fusedPlan.desc)

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    "fusedBatches" -> createMetric(MODERATE_LEVEL, "fused batches"),
    "fallbackBatches" -> createMetric(MODERATE_LEVEL, "fallback (unfused) batches"),
    "fusedKernelTime" -> createNanoTimingMetric(MODERATE_LEVEL, "fused kernel time"))

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = gpuLongMetric(NUM_OUTPUT_ROWS)
    val numOutputBatches = gpuLongMetric(NUM_OUTPUT_BATCHES)
    val opTime = gpuLongMetric(OP_TIME_LEGACY)
    val fusedM = gpuLongMetric("fusedBatches")
    val fallbackM = gpuLongMetric("fallbackBatches")
    val kernelM = gpuLongMetric("fusedKernelTime")
    val boundCondition = GpuBindReferences.bindGpuReferencesTiered(Seq(condition), child.output, conf, allMetrics)
    val boundSide = FusedSupport.bindSide(fusedPlan.sideExprs, child, conf, allMetrics)
    val fp = fusedPlan
    val rdd = child.executeColumnar()
    rdd.mapPartitionsWithIndex { (partIndex, iter) =>
      GpuNondeterministic.initializeAll(boundCondition.exprTiers.flatten, partIndex)
      iter.flatMap { batch =>
        val sb = SpillableColumnarBatch(batch, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
        val ret = withRetry(sb, splitSpillableInHalfByRows) { sbIn =>
          withResource(sbIn.getColumnarBatch()) { cb =>
            NvtxIdWithMetrics(NvtxRegistry.FILTER_BATCH, opTime) {
              filterOne(cb, fp, boundCondition, boundSide, fusedM, fallbackM, kernelM)
            }
          }
        }
        ret.map { cb =>
          numOutputRows += cb.numRows()
          numOutputBatches += 1
          cb
        }
      }
    }
  }

  private def applyMask(mask: cudf.ColumnVector, cb: ColumnarBatch): ColumnarBatch = {
    val allTrue = withResource(mask.all()) { a => a.isValid && a.getBoolean }
    if (allTrue) GpuColumnVector.incRefCounts(cb)
    else GpuColumnVector.filter(cb, GpuColumnVector.extractTypes(cb), mask)
  }

  private def filterOne(cb: ColumnarBatch, fp: FusedFilterPlan, orig: GpuTieredProject, boundSide: Option[GpuTieredProject],
                        fusedM: GpuMetric, fallbackM: GpuMetric, kernelM: GpuMetric): ColumnarBatch = {
    val key = fp.kernel.name
    if (FusionRuntime.killed || FusionRuntime.blacklist.contains(key) || !FusionRuntime.ensureInit(fp.cfg)) {
      fallbackM += 1; FusionStats.fallbackBatches.incrementAndGet()
      return GpuFilter(cb, orig)
    }
    val fusedRes: Option[ColumnarBatch] = try {
      val t0 = System.nanoTime()
      val side: Option[ColumnarBatch] = boundSide.map(_.project(cb))
      val kout = try FusedSupport.evalKernel(fp.kernel, fp.inputs, cb, side, fp.cfg) finally side.foreach(_.close())
      val r = try applyMask(kout(0), cb) finally kout.foreach(_.close())
      kernelM += (System.nanoTime() - t0)
      Some(r)
    } catch {
      case _: UseFallback => None
      case t: Throwable if !FusedSupport.isFatal(t) =>
        FusionRuntime.blacklist.add(key)
        FusionStats.runtimeFailures.incrementAndGet()
        FusedSupport.logFallback(key, t, "fused filter")
        None
    }
    fusedRes match {
      case None =>
        fallbackM += 1; FusionStats.fallbackBatches.incrementAndGet()
        GpuFilter(cb, orig)
      case Some(f) if fp.cfg.shadowNow =>
        closeOnExcept(f) { _ =>
          val ref = GpuFilter(cb, orig)
          closeOnExcept(ref) { _ =>
            FusionStats.shadowChecked.incrementAndGet()
            val diff = if (f.numRows() != ref.numRows()) Some(s"row count ${f.numRows()} vs ${ref.numRows()}")
              else FusedSupport.compareBatches(f, ref)
            f.close()
            diff.foreach { d =>
              FusionStats.shadowMismatches.incrementAndGet()
              logError(s"fusion SHADOW MISMATCH in filter region $key: $d")
              FusionRuntime.blacklist.add(key)
            }
            ref
          }
        }
      case Some(f) =>
        fusedM += 1; FusionStats.fusedBatches.incrementAndGet()
        f
    }
  }
}
