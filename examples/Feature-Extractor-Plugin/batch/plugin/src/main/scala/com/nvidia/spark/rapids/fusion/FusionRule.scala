package com.nvidia.spark.rapids.fusion

import scala.collection.mutable.ArrayBuffer

import com.nvidia.spark.rapids._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.rapids.aggregate.GpuAggregateExpression

/** Collected decisions, readable by tests and by `FusionExplain`. */
object FusionReport {
  /** Histogram of "why not fused" reasons (reason text -> number of planning events); see Fusion.reasonSummary. */
  val reasons = new java.util.concurrent.ConcurrentHashMap[String, java.util.concurrent.atomic.AtomicLong]()
  def reason(r: String): Unit = reasons.computeIfAbsent(r.take(160), _ => new java.util.concurrent.atomic.AtomicLong).incrementAndGet()
  val entries = new java.util.concurrent.ConcurrentLinkedQueue[String]()
  def add(s: String): Unit = { entries.add(s); while (entries.size() > 2000) entries.poll() }
  def dump(): Seq[String] = { val b = ArrayBuffer[String](); val it = entries.iterator(); while (it.hasNext) b += it.next(); b.toSeq }
  def clear(): Unit = entries.clear()
}

/** The planner: decides, for one spark-rapids Project/Filter node, whether and how to fuse. */
object Planner extends Logging {

  private def attrIdx(child: SparkPlan, a: Expression): Option[Int] = a match {
    case ar: AttributeReference =>
      val i = child.output.indexWhere(_.exprId == ar.exprId); if (i >= 0) Some(i) else None
    case _ => None
  }

  private def stripAlias(e: Expression): Expression = e.getClass.getSimpleName match {
    case "GpuAlias" | "Alias" => stripAlias(e.children.head)
    case _ => e
  }

  /** Kernel inputs: batch columns, or columns of the side batch (index into `side`). */
  private def mapInputs(k: KernelSpec, tr: Translator, side: ArrayBuffer[Expression]): Seq[InSrc] =
    k.inputs.map { in =>
      in.ex match {
        case InCol(idx, _) => FromCol(idx)
        case Opaque(slot, _) => FromSide(slot)   // opaque slots are the first side expressions, in slot order
        case other => throw new IllegalStateException(s"unexpected kernel input $other")
      }
    }

  /** Distinct operator nodes of an IR tree (what counts against the region budget). */
  private def callNodes(ex: Ex, acc: scala.collection.mutable.LinkedHashSet[Ex] = scala.collection.mutable.LinkedHashSet[Ex]()): scala.collection.mutable.LinkedHashSet[Ex] = {
    ex match { case c: Call => acc += c; c.args.foreach(callNodes(_, acc)); case _ => }
    acc
  }

  /** One kernel under construction: outputs + the opaque (cuDF-evaluated) inputs they share. */
  private final class Group {
    val exs = ArrayBuffer[Ex]()
    val outputs = ArrayBuffer[Int]()                       // project output index of each entry in `exs` (first user)
    val nodes = scala.collection.mutable.HashSet[Ex]()
    val opaques = ArrayBuffer[Expression]()
    val inputKeys = scala.collection.mutable.HashSet[Any]()
    def slotOf(e: Expression): Int = {
      val i = opaques.indexWhere(_.semanticEquals(e)); if (i >= 0) i else { opaques += e; opaques.size - 1 }
    }
  }

  private def inputKeysOf(ex: Ex, opq: Seq[Expression], acc: scala.collection.mutable.Set[Any]): Unit = ex match {
    case InCol(i, _) => acc += (("c", i))
    case Opaque(slot, _) => acc += (("o", opq(slot).canonicalized))
    case Call(_, args, _, _) => args.foreach(inputKeysOf(_, opq, acc))
    case _ =>
  }

  private def remapOpaque(ex: Ex, opq: Seq[Expression], g: Group): Ex = ex match {
    case Opaque(slot, t) => Opaque(g.slotOf(opq(slot)), t)
    case Call(op, args, t, attr) => Call(op, args.map(remapOpaque(_, opq, g)), t, attr)
    case other => other
  }

  def planProject(p: GpuProjectExec, cfg: FusionConf): scala.util.Either[String, FusedGpuProjectExec] = {
    val child = p.child
    // non-deterministic / stateful expressions are initialised per partition on the original bound list only
    if (p.projectList.exists(!_.deterministic)) return scala.util.Left("project list has non-deterministic expressions")
    val notes = ArrayBuffer[String]()
    val n = p.projectList.length
    val outs = new Array[OutSpec](n)                         // null = not decided / not fused
    // 1. translate every output on its own; outputs that cannot be fused go to the "side" (spark-rapids evaluates them)
    final case class Tr(i: Int, ex: Ex, opq: Seq[Expression], nodes: Seq[Ex])
    val translated = ArrayBuffer[Tr]()
    p.projectList.zipWithIndex.foreach { case (e, i) =>
      attrIdx(child, stripAlias(e)) match {
        case Some(idx) => outs(i) = OutCopy(idx)
        case None =>
          val tr = new Translator(child.output, cfg)
          try {
            tr.translateRoot(e) match {
              case InCol(idx, _) => outs(i) = OutCopy(idx)
              case c: Call => translated += Tr(i, c, tr.opaques.toSeq, callNodes(c).toSeq); notes ++= tr.notes
              case _ => // literal / bare opaque: nothing to fuse
            }
          } catch {
            case nf: NotFusible =>
              notes += s"output ${i} (${e.sql.take(60)}): ${nf.reason}"
              FusionReport.reason(nf.reason)
          }
      }
    }
    if (translated.isEmpty) {
      // a project that only passes columns through (column pruning) is nothing to fuse and nothing to report
      if (notes.isEmpty && outs.forall(_ != null)) return scala.util.Left("")
      return scala.util.Left(("no fusable expression" +: notes).mkString("; "))
    }
    // 2. greedily pack outputs into kernels, each within the distinct-operator / input / parameter limits
    val groups = ArrayBuffer[Group]()
    val assignment = scala.collection.mutable.HashMap[Int, (Group, Int)]()   // project output -> (kernel group, output index in it)
    var cur = new Group
    translated.foreach { t =>
      val ins = scala.collection.mutable.HashSet[Any](); inputKeysOf(t.ex, t.opq, ins)
      val tooBigNodes = cur.exs.nonEmpty && (cur.nodes ++ t.nodes).size > cfg.maxNodes
      val tooBigCols = cur.exs.nonEmpty && ((cur.inputKeys ++ ins).size > cfg.maxInputs || (cur.inputKeys ++ ins).size + cur.exs.size + 1 > 120)
      if (tooBigNodes || tooBigCols) { groups += cur; cur = new Group }
      val remapped = remapOpaque(t.ex, t.opq, cur)
      var k = cur.exs.indexOf(remapped)
      if (k < 0) { cur.exs += remapped; cur.outputs += t.i; k = cur.exs.size - 1 }
      assignment(t.i) = (cur, k)
      cur.nodes ++= t.nodes; cur.inputKeys ++= ins
    }
    groups += cur
    // outputs -> (group, index in group); a group too small to pay off is dropped (its outputs run unfused)
    val specs = groups.map(g => CodeGen.build(g.exs.toSeq))
    val keep = groups.indices.filter(gi => cfg.forceFuse || specs(gi).exprCount >= cfg.minOps)
    if (keep.isEmpty) return scala.util.Left(s"region too small (${specs.map(_.exprCount).sum} op(s) < minOps=${cfg.minOps})")
    val parts = ArrayBuffer[KernelPart]()
    val side = ArrayBuffer[Expression]()
    val groupOffset = scala.collection.mutable.HashMap[Int, Int]()
    keep.foreach { gi => groupOffset(gi) = side.size; side ++= groups(gi).opaques }
    keep.zipWithIndex.foreach { case (gi, pi) =>
      val g = groups(gi); val sp = specs(gi); val off = groupOffset(gi)
      val inputs = sp.inputs.map { in => in.ex match {
        case InCol(idx, _) => FromCol(idx)
        case Opaque(slot, _) => FromSide(off + slot)
        case other => throw new IllegalStateException(s"unexpected kernel input $other")
      } }
      parts += KernelPart(sp, inputs)
    }
    // map each fused project output to its kernel output (outputs of dropped groups fall through to the side batch)
    val partIndexOf = keep.zipWithIndex.map { case (gi, pi) => (groups(gi), pi) }.toMap
    val partOfOutput = scala.collection.mutable.HashMap[Int, (Int, Int)]()
    assignment.foreach { case (outIdx, (g, k)) => partIndexOf.get(g).foreach(pi => partOfOutput(outIdx) = (pi, k)) }
    for (i <- 0 until n if outs(i) == null) {
      partOfOutput.get(i) match {
        case Some((pi, k)) => outs(i) = OutKernel(pi, k)
        case None => side += p.projectList(i); outs(i) = OutSide(side.size - 1)
      }
    }
    val outTypes = p.projectList.map(_.dataType)
    val nSide = outs.count(_.isInstanceOf[OutSide])
    val totalOps = keep.map(gi => specs(gi).exprCount).sum
    val nOpaque = keep.map(gi => groups(gi).opaques.size).sum
    val desc = s"fused $totalOps ops -> ${parts.size} kernel${if (parts.size > 1) "s" else ""} ${parts.map(_.spec.name).mkString(",")}; " +
      s"${parts.map(_.spec.outputs.size).sum} kernel outputs, ${parts.map(_.spec.inputs.size).sum} inputs, $nOpaque cuDF-evaluated sub-expr(s), $nSide cuDF output(s)"
    val plan = FusedProjectPlan(parts.toSeq, outs.toSeq, outTypes, side.toSeq, cfg, desc)
    FusionReport.add(s"FUSED project: $desc ${notes.mkString("; ")}")
    scala.util.Right(FusedGpuProjectExec(p.projectList, child, p.enablePreSplit, plan))
  }

  /** Try to replace one expression tree (not a NamedExpression) by a [[FusedGpuExpression]]. */
  def planExpression(e: Expression, inputAttrs: Seq[Attribute], cfg: FusionConf): Option[FusedGpuExpression] = {
    val tr = new Translator(inputAttrs, cfg)
    val ex = try tr.translateRoot(e) catch { case _: NotFusible => return None }
    ex match { case _: Call => case _ => return None }
    val spec = CodeGen.build(Seq(ex), "expr")
    if (spec.exprCount < cfg.minOps && !cfg.forceFuse) return None
    if (spec.inputs.size > cfg.maxInputs) return None
    val inputs = spec.inputs.map { in =>
      in.ex match {
        case InCol(idx, _) => inputAttrs(idx).asInstanceOf[Expression]
        case Opaque(slot, _) => tr.opaques(slot)
        case other => throw new IllegalStateException(s"unexpected kernel input $other")
      }
    }
    // fallback: the original tree with every kernel input replaced by a bound reference to that input column
    val opaqueIdx: Seq[(Expression, Int)] = spec.inputs.zipWithIndex.collect { case (KIn(Opaque(slot, _), _), i) => (tr.opaques(slot), i) }
    val attrIdx: Map[org.apache.spark.sql.catalyst.expressions.ExprId, Int] = spec.inputs.zipWithIndex.collect {
      case (KIn(InCol(idx, _), _), i) => (inputAttrs(idx).exprId, i)
    }.toMap
    def ref(i: Int, like: Expression): Expression = GpuBoundReference(i, like.dataType, like.nullable)(
      like match { case n: NamedExpression => n.exprId; case _ => org.apache.spark.sql.catalyst.expressions.NamedExpression.newExprId },
      like match { case n: NamedExpression => n.name; case _ => "in" + i })
    val fallback = e.transformDown {
      case x if opaqueIdx.exists(_._1.semanticEquals(x)) => ref(opaqueIdx.find(_._1.semanticEquals(x)).get._2, x)
      case a: AttributeReference if attrIdx.contains(a.exprId) => ref(attrIdx(a.exprId), a)
    }
    FusionReport.add(s"FUSED expression: ${e.sql.take(120)} -> ${spec.name} (${spec.exprCount} ops, ${spec.inputs.size} inputs)")
    Some(FusedGpuExpression(spec, inputs, fallback, e, cfg))
  }

  /**
   * Vertical merge of two adjacent projects: `upper` is evaluated over the outputs of `lower`. The merged list is
   * expressed over lower's child output, with lower's expressions inlined (common sub-expressions are shared by the
   * fused kernel / spark-rapids' tiered project). Returns None when not applicable.
   */
  def mergeProjects(upperList: Seq[NamedExpression], lowerList: Seq[NamedExpression]): Option[List[NamedExpression]] = {
    if (upperList.exists(!_.deterministic) || lowerList.exists(!_.deterministic)) return None
    // keep every lower expression that could throw / has side effects evaluated exactly as before
    if (lowerList.exists(e => try e.getClass.getMethod("hasSideEffects").invoke(e).asInstanceOf[Boolean] catch { case _: Throwable => false })) return None
    val inlined: Map[org.apache.spark.sql.catalyst.expressions.ExprId, NamedExpression] =
      lowerList.map(ne => ne.exprId -> ne).toMap
    def body(ne: NamedExpression): Expression = stripAlias(ne)
    val merged = upperList.map { e =>
      e match {
        case ar: AttributeReference if inlined.contains(ar.exprId) => inlined(ar.exprId)
        case other =>
          other.transform {
            case a: AttributeReference if inlined.contains(a.exprId) => body(inlined(a.exprId))
          }.asInstanceOf[NamedExpression]
      }
    }
    Some(merged.toList)
  }

  def planFilter(f: GpuFilterExec, cfg: FusionConf): scala.util.Either[String, FusedGpuFilterExec] = {
    val child = f.child
    if (!f.condition.deterministic) return scala.util.Left("filter condition is non-deterministic")
    val tr = new Translator(child.output, cfg)
    val ex = try tr.translateRoot(f.condition) catch { case nf: NotFusible => FusionReport.reason(nf.reason); return scala.util.Left(nf.reason) }
    ex match { case _: Call => case _ => return scala.util.Left("filter condition is not an expression tree") }
    val spec = CodeGen.build(Seq(Call("tomask", Seq(ex), FT.Bool)), "filt")
    if (spec.exprCount - 1 < cfg.minOps && !cfg.forceFuse)
      return scala.util.Left(s"region too small (${spec.exprCount - 1} op(s) < minOps=${cfg.minOps})")
    if (spec.inputs.size > cfg.maxInputs) return scala.util.Left(s"region has ${spec.inputs.size} inputs > ${cfg.maxInputs}")
    val side = ArrayBuffer[Expression]()
    side ++= tr.opaques
    val inputs = mapInputs(spec, tr, side)
    val desc = s"fused ${spec.exprCount - 1} ops -> 1 mask kernel ${spec.name}; ${spec.inputs.size} inputs, " +
      s"${tr.opaques.size} cuDF-evaluated sub-expr(s)"
    FusionReport.add(s"FUSED filter: $desc ${tr.notes.mkString("; ")}")
    scala.util.Right(FusedGpuFilterExec(f.condition, child, f.output, FusedFilterPlan(spec, inputs, side.toSeq, cfg, desc))(f.coalesceAfter))
  }
}

object FusionRule { @volatile var linkageBroken = false }

/** Runs after spark-rapids' own post-transition rules (verified empirically). */
class FusionRule extends Rule[SparkPlan] with Logging {
  /** (project list, child, enablePreSplit) of an adjacent lower project this one can be merged with. */
  private def mergeable(p: GpuProjectExec): Option[(Seq[NamedExpression], SparkPlan, Boolean)] = p.child match {
    case f: FusedGpuProjectExec => Some((f.projectList, f.child, f.enablePreSplit))
    case g: GpuProjectExec => Some((g.projectList, g.child, g.enablePreSplit))
    case _ => None
  }

  private def planSingleProject(p: GpuProjectExec, cfg: FusionConf, explain: String => Unit): SparkPlan = {
    try Planner.planProject(p, cfg) match {
      case scala.util.Right(fp) =>
        if (cfg.explain == "ALL") explain(s"FUSED ${p.projectList.map(_.sql).mkString("[", ", ", "]").take(200)}: ${fp.fusedPlan.desc}")
        fp
      case scala.util.Left(why) =>
        if (why.nonEmpty) explain(s"NOT FUSED GpuProject ${p.projectList.map(_.sql).mkString("[", ", ", "]").take(200)}: $why"); p
    } catch { case t: Throwable => explain(s"NOT FUSED GpuProject: planner error $t"); logDebug("planner error", t); p }
  }

  /** Never let a planner problem (e.g. an incompatible spark-rapids version -> LinkageError) fail a query. */
  override def apply(plan: SparkPlan): SparkPlan =
    if (FusionRule.linkageBroken) plan
    else try applyInternal(plan)
    catch {
      case t: LinkageError =>
        // incompatible spark-rapids/Spark version: disable the plugin for this JVM, once, loudly
        FusionRule.linkageBroken = true
        logWarning(s"[fusion] disabled: this plugin build is not compatible with the running spark-rapids/Spark ($t)")
        plan
      case t: Throwable if !t.isInstanceOf[InterruptedException] =>
        logWarning(s"[fusion] planning failed, running the plan unfused: $t")
        plan
    }

  private def applyInternal(plan: SparkPlan): SparkPlan = {
    val cfg = try FusionConf.fromSQLConf() catch { case t: Throwable => logWarning(s"fusion config error: $t"); return plan }
    if (!cfg.enabled) return plan
    def explain(msg: String): Unit = {
      FusionReport.add(msg)
      if (cfg.explain != "NONE") logWarning(s"[fusion] $msg")
    }
    plan.transformUp {
      case a: GpuHashAggregateExec if cfg.fuseAggregate =>
        try {
          val inAttrs = a.child.output
          var changed = false
          val newAggs = a.aggregateExpressions.map { ae =>
            ae.transformDown {
              case e: Expression if !e.isInstanceOf[NamedExpression] && !e.isInstanceOf[LeafExpression] &&
                  !e.isInstanceOf[FusedGpuExpression] && e.getClass.getSimpleName != "GpuAggregateExpression" =>
                Planner.planExpression(e, inAttrs, cfg) match {
                  case Some(f) => changed = true; f
                  case None => e
                }
            }.asInstanceOf[GpuAggregateExpression]
          }
          val newGroup = a.groupingExpressions.map { ge =>
            ge.transformDown {
              case e: Expression if !e.isInstanceOf[NamedExpression] && !e.isInstanceOf[LeafExpression] &&
                  !e.isInstanceOf[FusedGpuExpression] =>
                Planner.planExpression(e, inAttrs, cfg) match {
                  case Some(f) => changed = true; f
                  case None => e
                }
            }.asInstanceOf[NamedExpression]
          }
          if (changed) {
            explain(s"FUSED expressions inside GpuHashAggregate (${a.aggregateExpressions.size} aggregates)")
            a.copy(groupingExpressions = newGroup, aggregateExpressions = newAggs)
          } else a
        } catch { case t: Throwable => explain(s"NOT FUSED GpuHashAggregate: planner error $t"); logDebug("planner error", t); a }
      case p: GpuProjectExec if cfg.fuseProject && cfg.mergeProjects && mergeable(p).isDefined =>
        val (lowerList, lowerChild, lowerPre) = mergeable(p).get
        val mergedOpt = Planner.mergeProjects(p.projectList, lowerList)
        val attempt = mergedOpt.flatMap { m =>
          try Planner.planProject(GpuProjectExec(m, lowerChild, p.enablePreSplit && lowerPre), cfg).toOption
          catch { case t: Throwable => logDebug("merge planner error", t); None }
        }
        attempt match {
          case Some(fp) => explain(s"FUSED vertical merge of two projects: ${fp.fusedPlan.desc}"); fp
          case None => planSingleProject(p, cfg, explain)
        }
      case p: GpuProjectExec if cfg.fuseProject =>
        planSingleProject(p, cfg, explain)
      case f: GpuFilterExec if cfg.fuseFilter =>
        try Planner.planFilter(f, cfg) match {
          case scala.util.Right(ff) => if (cfg.explain == "ALL") explain(s"FUSED filter ${f.condition.sql.take(200)}: ${ff.fusedPlan.desc}"); ff
          case scala.util.Left(why) => explain(s"NOT FUSED GpuFilter ${f.condition.sql.take(200)}: $why"); f
        } catch { case t: Throwable => explain(s"NOT FUSED GpuFilter: planner error $t"); logDebug("planner error", t); f }
    }
  }
}
