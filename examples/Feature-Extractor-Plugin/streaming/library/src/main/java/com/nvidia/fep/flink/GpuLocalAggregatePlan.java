// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/** Exact adapter between the CUDA partials and Flink's flattened local accumulator schema. */
public final class GpuLocalAggregatePlan implements Serializable {
    private static final long serialVersionUID = 1L;

    private final GpuAggregatePlan computePlan;
    private final boolean[] averageCalls;
    private final GpuType[] averageSumTypes;
    private final int localOutputArity;

    private GpuLocalAggregatePlan(
            GpuAggregatePlan computePlan,
            boolean[] averageCalls,
            GpuType[] averageSumTypes,
            int localOutputArity) {
        this.computePlan = computePlan;
        this.averageCalls = averageCalls;
        this.averageSumTypes = averageSumTypes;
        this.localOutputArity = localOutputArity;
    }

    /**
     * Compatibility constructor for serialized/test fixtures created before AVG partials carried
     * their explicit sum type.  Non-AVG plans were the only legal shape at that point; keeping
     * this exact signature avoids making those frozen plans depend on the newer metadata field.
     */
    @SuppressWarnings("unused")
    private GpuLocalAggregatePlan(
            GpuAggregatePlan computePlan, boolean[] averageCalls, int localOutputArity) {
        this(
                computePlan,
                averageCalls,
                legacyAverageSumTypes(computePlan, averageCalls),
                localOutputArity);
    }

    private static GpuType[] legacyAverageSumTypes(
            GpuAggregatePlan computePlan, boolean[] averageCalls) {
        if (computePlan == null || averageCalls == null) {
            throw new NullPointerException("local aggregate compatibility metadata is null");
        }
        if (computePlan.aggregates().length != averageCalls.length) {
            throw new IllegalArgumentException(
                    "local aggregate compatibility metadata length mismatch");
        }
        GpuType[] sumTypes = new GpuType[averageCalls.length];
        GpuAggregatePlan.AggregateSpec[] specs = computePlan.aggregates();
        for (int index = 0; index < averageCalls.length; index++) {
            if (!averageCalls[index]) {
                continue;
            }
            GpuType argument = specs[index].argumentType;
            if (argument == null || !argument.isNumeric()) {
                throw new IllegalArgumentException(
                        "legacy AVG partial has no supported numeric input at call " + index);
            }
            sumTypes[index] =
                    argument == GpuType.FLOAT32 || argument == GpuType.FLOAT64
                            ? GpuType.FLOAT64
                            : GpuType.INT64;
        }
        return sumTypes;
    }

    /** Shape-admission compiler isolated from the serialized runtime plan. */
    static final class PlanCompiler {
        private PlanCompiler() {}

    static CompileResult tryCompile(
            int[] grouping,
            AggregateCallSpec[] calls,
            boolean[] callNeedsRetraction,
            boolean needRetraction,
            RowType inputType,
            RowType localOutputType) {
        try {
            if (calls.length != callNeedsRetraction.length) {
                return CompileResult.unsupported("aggregate retraction metadata length mismatch");
            }
            List<LogicalType> finalTypes = new ArrayList<>(grouping.length + calls.length);
            int localOrdinal = 0;
            for (int groupingOrdinal : grouping) {
                if (groupingOrdinal < 0 || groupingOrdinal >= inputType.getFieldCount()) {
                    return CompileResult.unsupported(
                            "invalid local GROUP BY input ordinal " + groupingOrdinal);
                }
                if (localOrdinal >= localOutputType.getFieldCount()
                        || !inputType
                                .getTypeAt(groupingOrdinal)
                                .copy(false)
                                .equals(localOutputType.getTypeAt(localOrdinal).copy(false))) {
                    return CompileResult.unsupported(
                            "local accumulator key schema does not match GROUP BY key "
                                    + localOrdinal);
                }
                finalTypes.add(inputType.getTypeAt(groupingOrdinal));
                localOrdinal++;
            }

            boolean[] averageCalls = new boolean[calls.length];
            GpuType[] averageSumTypes = new GpuType[calls.length];
            for (int callIndex = 0; callIndex < calls.length; callIndex++) {
                AggregateCallSpec call = calls[callIndex];
                GpuAggregatePlan.AggregateKind kind = call.getKind();
                if (kind == GpuAggregatePlan.AggregateKind.AVG) {
                    averageCalls[callIndex] = true;
                    if (call.getArgList().size() != 1) {
                        return CompileResult.unsupported("AVG local accumulator needs one input");
                    }
                    GpuType inputGpuType =
                            GpuType.from(inputType.getTypeAt(call.getArgList().get(0)))
                                    .orElse(null);
                    if (inputGpuType == null || !inputGpuType.isNumeric()) {
                        return CompileResult.unsupported(
                                "two-phase AVG requires a supported numeric input");
                    }
                    GpuType expectedSumType =
                            inputGpuType == GpuType.FLOAT32 || inputGpuType == GpuType.FLOAT64
                                    ? GpuType.FLOAT64
                                    : GpuType.INT64;
                    if (localOrdinal + 1 >= localOutputType.getFieldCount()
                            || GpuType.from(localOutputType.getTypeAt(localOrdinal)).orElse(null)
                                    != expectedSumType
                            || GpuType.from(localOutputType.getTypeAt(localOrdinal + 1)).orElse(null)
                                    != GpuType.INT64) {
                        return CompileResult.unsupported(
                                "Flink AVG local accumulator is not "
                                        + expectedSumType
                                        + " sum + BIGINT count");
                    }
                    averageSumTypes[callIndex] = expectedSumType;
                    finalTypes.add(inputType.getTypeAt(call.getArgList().get(0)));
                    localOrdinal += 2;
                } else {
                    if (localOrdinal >= localOutputType.getFieldCount()) {
                        return CompileResult.unsupported(
                                "local accumulator has fewer fields than aggregate calls");
                    }
                    if (!GpuType.from(localOutputType.getTypeAt(localOrdinal)).isPresent()) {
                        return CompileResult.unsupported(
                                "unsupported local accumulator type "
                                        + localOutputType.getTypeAt(localOrdinal).asSummaryString());
                    }
                    finalTypes.add(localOutputType.getTypeAt(localOrdinal));
                    localOrdinal++;
                }
            }
            if (localOrdinal != localOutputType.getFieldCount()) {
                return CompileResult.unsupported(
                        "unexpected local accumulator arity "
                                + localOutputType.getFieldCount()
                                + ", consumed "
                                + localOrdinal);
            }

            RowType syntheticFinalType = RowType.of(finalTypes.toArray(new LogicalType[0]));
            GpuAggregatePlan.CompileResult compute =
                    GpuAggregatePlan.PlanCompiler.tryCompile(
                            grouping,
                            calls,
                            callNeedsRetraction,
                            needRetraction,
                            inputType,
                            syntheticFinalType);
            if (!compute.isSupported()) {
                return CompileResult.unsupported(compute.reason());
            }
            return CompileResult.supported(
                    new GpuLocalAggregatePlan(
                            compute.plan(),
                            averageCalls,
                            averageSumTypes,
                            localOutputType.getFieldCount()));
        } catch (RuntimeException malformed) {
            return CompileResult.unsupported(
                    "could not compile local aggregate accumulator: " + malformed.getMessage());
        }
    }

    }

    public GpuAggregatePlan computePlan() {
        return computePlan;
    }

    public GpuLocalAggregatePlan remapInput(int newInputArity, int[] outputToInput) {
        return new GpuLocalAggregatePlan(
                computePlan.remapInput(newInputArity, outputToInput),
                averageCalls.clone(),
                averageSumTypes.clone(),
                localOutputArity);
    }

    GenericRowData partialRow(AggregatePartial partial) {
        GenericRowData output = new GenericRowData(localOutputArity);
        writePartial(partial, output);
        return output;
    }

    GenericRowData newPartialRow() {
        return new GenericRowData(localOutputArity);
    }

    void writePartial(AggregatePartial partial, GenericRowData output) {
        int outputOrdinal = 0;
        GpuType[] groupingTypes = computePlan.groupingTypes();
        for (int group = 0; group < partial.groupValues.length; group++) {
            output.setField(
                    outputOrdinal++,
                    groupingTypes[group].toFlinkInternal(partial.groupValues[group]));
        }
        GpuAggregatePlan.AggregateSpec[] specs = computePlan.aggregates();
        for (int aggregate = 0; aggregate < specs.length; aggregate++) {
            Object value = partial.aggregateValues[aggregate];
            if (averageCalls[aggregate]) {
                AverageAccumulator average = (AverageAccumulator) value;
                if (averageSumTypes[aggregate] == GpuType.INT64) {
                    output.setField(outputOrdinal++, average == null ? 0L : average.integralSum);
                } else {
                    output.setField(outputOrdinal++, average == null ? 0.0d : average.sum);
                }
                output.setField(outputOrdinal++, average == null ? 0L : average.count);
            } else {
                output.setField(
                        outputOrdinal++,
                        specs[aggregate]
                                .resultType
                                .toFlinkInternal(specs[aggregate].normalizePartial(value)));
            }
        }
        if (outputOrdinal != localOutputArity) {
            throw new IllegalStateException(
                    "local accumulator output arity changed after planning: " + outputOrdinal);
        }
    }

    void writePartial(AggregatePartial partial, BinaryRowWriter output) {
        int outputOrdinal = 0;
        GpuType[] groupingTypes = computePlan.groupingTypes();
        for (int group = 0; group < partial.groupValues.length; group++) {
            groupingTypes[group].writeValue(
                    partial.groupValues[group], output, outputOrdinal++);
        }
        GpuAggregatePlan.AggregateSpec[] specs = computePlan.aggregates();
        for (int aggregate = 0; aggregate < specs.length; aggregate++) {
            Object value = partial.aggregateValues[aggregate];
            if (averageCalls[aggregate]) {
                AverageAccumulator average = (AverageAccumulator) value;
                if (averageSumTypes[aggregate] == GpuType.INT64) {
                    output.writeLong(outputOrdinal++, average == null ? 0L : average.integralSum);
                } else {
                    output.writeDouble(outputOrdinal++, average == null ? 0.0d : average.sum);
                }
                output.writeLong(outputOrdinal++, average == null ? 0L : average.count);
            } else {
                specs[aggregate]
                        .resultType
                        .writeValue(
                                specs[aggregate].normalizePartial(value),
                                output,
                                outputOrdinal++);
            }
        }
        if (outputOrdinal != localOutputArity) {
            throw new IllegalStateException(
                    "local accumulator output arity changed after planning: "
                            + outputOrdinal);
        }
    }

    AggregatePartial partialFromRow(RowData row) {
        Object[] groupValues = new Object[computePlan.groupCount()];
        int inputOrdinal = 0;
        GpuType[] groupingTypes = computePlan.groupingTypes();
        for (int group = 0; group < groupValues.length; group++) {
            groupValues[group] = groupingTypes[group].copyFrom(row, inputOrdinal++);
        }
        Object[] aggregateValues = new Object[averageCalls.length];
        GpuAggregatePlan.AggregateSpec[] specs = computePlan.aggregates();
        for (int aggregate = 0; aggregate < aggregateValues.length; aggregate++) {
            if (averageCalls[aggregate]) {
                boolean integral = averageSumTypes[aggregate] == GpuType.INT64;
                long integralSum = integral ? row.getLong(inputOrdinal++) : 0L;
                double sum = integral ? 0.0d : row.getDouble(inputOrdinal++);
                long count = row.getLong(inputOrdinal++);
                if (count < 0L) {
                    throw new IllegalArgumentException("AVG partial count is negative: " + count);
                }
                aggregateValues[aggregate] =
                        count == 0L
                                ? null
                                : integral
                                        ? AverageAccumulator.integral(integralSum, count)
                                        : new AverageAccumulator(sum, count);
            } else {
                aggregateValues[aggregate] =
                        specs[aggregate].resultType.copyFrom(row, inputOrdinal++);
            }
        }
        if (inputOrdinal != localOutputArity) {
            throw new IllegalStateException(
                    "local accumulator input arity changed after planning: " + inputOrdinal);
        }
        return new AggregatePartial(groupValues, 1L, aggregateValues);
    }

    AggregatePartial mergePartials(AggregatePartial left, AggregatePartial right) {
        Object[] values = new Object[left.aggregateValues.length];
        GpuAggregatePlan.AggregateSpec[] specs = computePlan.aggregates();
        for (int aggregate = 0; aggregate < values.length; aggregate++) {
            values[aggregate] =
                    specs[aggregate]
                            .merge(
                                    left.aggregateValues[aggregate],
                                    right.aggregateValues[aggregate]);
        }
        return new AggregatePartial(
                left.groupValues,
                Math.addExact(left.inputRows, right.inputRows),
                values);
    }

    public String summary() {
        int averages = 0;
        for (boolean average : averageCalls) {
            if (average) {
                averages++;
            }
        }
        return computePlan.summary() + ", flattenedAvgAccumulators=" + averages;
    }

    public static final class CompileResult {
        private final GpuLocalAggregatePlan plan;
        private final String reason;

        private CompileResult(GpuLocalAggregatePlan plan, String reason) {
            this.plan = plan;
            this.reason = reason;
        }

        static CompileResult supported(GpuLocalAggregatePlan plan) {
            return new CompileResult(plan, null);
        }

        static CompileResult unsupported(String reason) {
            return new CompileResult(null, reason);
        }

        public boolean isSupported() {
            return plan != null;
        }

        public GpuLocalAggregatePlan plan() {
            return plan;
        }

        public String reason() {
            return reason;
        }
    }
}
