// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;


import java.io.Serializable;
import java.util.ArrayList;

/** Adapts GPU window partials to Flink's local-window accumulator plus slice-end schema. */
public final class GpuLocalWindowAggregatePlan implements Serializable {
    private static final long serialVersionUID = 1L;

    private final GpuLocalAggregatePlan aggregatePlan;
    private final int outputArity;

    private GpuLocalWindowAggregatePlan(
            GpuLocalAggregatePlan aggregatePlan, int outputArity) {
        this.aggregatePlan = aggregatePlan;
        this.outputArity = outputArity;
    }

    /** Shape-admission compiler isolated from the serialized runtime plan. */
    static final class PlanCompiler {
        private PlanCompiler() {}

    static CompileResult tryCompile(
            int[] grouping,
            AggregateCallSpec[] calls,
            boolean needRetraction,
            RowType inputType,
            RowType outputType) {
        if (outputType.getFieldCount() < 2) {
            return CompileResult.unsupported(
                    "local Window output must contain accumulators and slice end");
        }
        int sliceEndOrdinal = outputType.getFieldCount() - 1;
        if (GpuType.from(outputType.getTypeAt(sliceEndOrdinal)).orElse(null)
                != GpuType.INT64) {
            return CompileResult.unsupported("local Window slice end must be BIGINT");
        }
        RowType accumulatorType =
                new RowType(
                        outputType.isNullable(),
                        new ArrayList<>(
                                outputType.getFields().subList(0, sliceEndOrdinal)));
        GpuLocalAggregatePlan.CompileResult aggregate =
                GpuLocalAggregatePlan.PlanCompiler.tryCompile(
                        grouping,
                        calls,
                        new boolean[calls.length],
                        needRetraction,
                        inputType,
                        accumulatorType);
        if (!aggregate.isSupported()) {
            return CompileResult.unsupported(aggregate.reason());
        }
        return CompileResult.supported(
                new GpuLocalWindowAggregatePlan(
                        aggregate.plan(), outputType.getFieldCount()));
    }

    }

    public GpuAggregatePlan computePlan() {
        return aggregatePlan.computePlan();
    }

    /** Reuses the proven local accumulator layout for producer-side Calc fusion. */
    public GpuLocalAggregatePlan localAggregatePlan() {
        return aggregatePlan;
    }

    public int coreOutputArity() {
        return aggregatePlan.computePlan().outputTypes().length;
    }

    public boolean matchesCoreOutput(RowType outputType) {
        GpuType[] expected = aggregatePlan.computePlan().outputTypes();
        if (outputType.getFieldCount() != expected.length) {
            return false;
        }
        for (int i = 0; i < expected.length; i++) {
            LogicalType logicalType = outputType.getTypeAt(i);
            if (GpuType.from(logicalType).orElse(null) != expected[i]) {
                return false;
            }
        }
        return true;
    }

    void writePartial(
            WindowAggregatePartial partial,
            BinaryRowWriter output) {
        aggregatePlan.writePartial(
                new AggregatePartial(
                        partial.groupValues,
                        partial.inputRows,
                        partial.aggregateValues),
                output);
        output.writeLong(outputArity - 1, partial.windowEnd);
    }

    WindowAggregatePartial stableCopy(WindowAggregatePartial partial) {
        return new WindowAggregatePartial(
                partial.windowEnd,
                partial.groupValues.clone(),
                partial.inputRows,
                partial.aggregateValues.clone());
    }

    WindowAggregatePartial partialFromRow(org.apache.flink.table.data.RowData row) {
        if (row.isNullAt(outputArity - 1)) {
            throw new IllegalArgumentException("local Window slice end must not be NULL");
        }
        AggregatePartial partial = aggregatePlan.partialFromRow(row);
        return new WindowAggregatePartial(
                row.getLong(outputArity - 1),
                partial.groupValues,
                partial.inputRows,
                partial.aggregateValues);
    }

    WindowAggregatePartial merge(
            WindowAggregatePartial left, WindowAggregatePartial right) {
        AggregatePartial merged =
                aggregatePlan.mergePartials(
                        new AggregatePartial(
                                left.groupValues,
                                left.inputRows,
                                left.aggregateValues),
                        new AggregatePartial(
                                right.groupValues,
                                right.inputRows,
                                right.aggregateValues));
        return new WindowAggregatePartial(
                left.windowEnd,
                merged.groupValues,
                merged.inputRows,
                merged.aggregateValues);
    }

    int outputArity() {
        return outputArity;
    }

    public String summary() {
        return aggregatePlan.summary() + ", sliceEnd=BIGINT";
    }

    public static final class CompileResult {
        private final GpuLocalWindowAggregatePlan plan;
        private final String reason;

        private CompileResult(GpuLocalWindowAggregatePlan plan, String reason) {
            this.plan = plan;
            this.reason = reason;
        }

        static CompileResult supported(GpuLocalWindowAggregatePlan plan) {
            return new CompileResult(plan, null);
        }

        static CompileResult unsupported(String reason) {
            return new CompileResult(null, reason);
        }

        public boolean isSupported() {
            return plan != null;
        }

        public GpuLocalWindowAggregatePlan plan() {
            return plan;
        }

        public String reason() {
            return reason;
        }
    }
}
