// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Correctness fallback for exceptional rows inside a project-owned aggregate batch.
 *
 * <p>This deliberately does not call cuDF. Normal supported batches run on the native GPU kernel;
 * a batch containing a value that the specialized kernel cannot safely process is reduced here and
 * then merged into the same Flink keyed state.
 */
final class CpuAggregateFallback {
    private CpuAggregateFallback() {}

    static List<AggregatePartial> execute(
            GpuAggregatePlan plan, List<AggregateBufferedRow> rows) {
        Map<GroupKey, Accumulator> groups = new LinkedHashMap<>();
        for (AggregateBufferedRow row : rows) {
            Object[] values = groupValues(plan, row);
            Accumulator accumulator =
                    groups.computeIfAbsent(
                            new GroupKey(values, Long.MIN_VALUE),
                            ignored -> new Accumulator(values, Long.MIN_VALUE, plan.aggregates().length));
            accumulator.add(plan, row);
        }
        List<AggregatePartial> result = new ArrayList<>(groups.size());
        for (Accumulator accumulator : groups.values()) {
            result.add(
                    new AggregatePartial(
                            accumulator.groupValues,
                            accumulator.inputRows,
                            accumulator.finishedValues(plan)));
        }
        return result;
    }

    static List<WindowAggregatePartial> executeWindows(
            GpuAggregatePlan plan, List<WindowBufferedRow> rows) {
        Map<GroupKey, Accumulator> groups = new LinkedHashMap<>();
        for (WindowBufferedRow windowRow : rows) {
            AggregateBufferedRow row = windowRow.aggregateRow;
            Object[] values = groupValues(plan, row);
            Accumulator accumulator =
                    groups.computeIfAbsent(
                            new GroupKey(values, windowRow.windowEnd),
                            ignored ->
                                    new Accumulator(
                                            values,
                                            windowRow.windowEnd,
                                            plan.aggregates().length));
            accumulator.add(plan, row);
        }
        List<WindowAggregatePartial> result = new ArrayList<>(groups.size());
        for (Accumulator accumulator : groups.values()) {
            result.add(
                    new WindowAggregatePartial(
                            accumulator.windowEnd,
                            accumulator.groupValues,
                            accumulator.inputRows,
                            accumulator.finishedValues(plan)));
        }
        return result;
    }

    private static Object[] groupValues(GpuAggregatePlan plan, AggregateBufferedRow row) {
        int[] ordinals = plan.groupingPackedOrdinals();
        Object[] values = new Object[ordinals.length];
        for (int i = 0; i < ordinals.length; i++) {
            values[i] = row.field(ordinals[i]);
        }
        return values;
    }

    private static final class Accumulator {
        private final Object[] groupValues;
        private final long windowEnd;
        private final Object[] values;
        private long inputRows;

        private Accumulator(Object[] groupValues, long windowEnd, int aggregateCount) {
            this.groupValues = groupValues;
            this.windowEnd = windowEnd;
            this.values = new Object[aggregateCount];
        }

        private void add(GpuAggregatePlan plan, AggregateBufferedRow row) {
            int weight = row.weight();
            inputRows = Math.addExact(inputRows, weight);
            GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
            for (int i = 0; i < specs.length; i++) {
                GpuAggregatePlan.AggregateSpec spec = specs[i];
                if (GpuAggregatePlan.isDistinctFilterOrdinal(spec.filterPackedOrdinal)) {
                    if (!row.distinctAccepted(spec.distinctIndex)) {
                        continue;
                    }
                } else if (spec.filterPackedOrdinal >= 0) {
                    Object filter = row.field(spec.filterPackedOrdinal);
                    if (!Boolean.TRUE.equals(filter)) {
                        continue;
                    }
                }
                Object input =
                        spec.inputPackedOrdinal < 0
                                ? null
                                : row.field(spec.inputPackedOrdinal);
                if (spec.kind == GpuAggregatePlan.AggregateKind.COUNT) {
                    if (spec.inputPackedOrdinal < 0 || input != null) {
                        long count = values[i] == null ? 0L : (Long) values[i];
                        values[i] = Math.addExact(count, weight);
                    }
                } else if (input != null) {
                    Object contribution;
                    if (spec.kind == GpuAggregatePlan.AggregateKind.AVG) {
                        contribution =
                                AverageAccumulator.fromWeightedValue(
                                        (Number) input, spec.argumentType, weight);
                    } else if (weight < 0
                            && (spec.kind == GpuAggregatePlan.AggregateKind.SUM
                                    || spec.kind == GpuAggregatePlan.AggregateKind.SUM0)) {
                        contribution = spec.negate(input);
                    } else {
                        contribution = input;
                    }
                    values[i] = spec.merge(values[i], contribution);
                }
            }
        }

        private Object[] finishedValues(GpuAggregatePlan plan) {
            Object[] result = values.clone();
            GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
            for (int i = 0; i < specs.length; i++) {
                if (specs[i].kind == GpuAggregatePlan.AggregateKind.COUNT && result[i] == null) {
                    result[i] = 0L;
                } else {
                    result[i] = specs[i].normalizePartial(result[i]);
                }
            }
            return result;
        }
    }

    private static final class GroupKey {
        private final Object[] values;
        private final long windowEnd;
        private final int hash;

        private GroupKey(Object[] values, long windowEnd) {
            this.values = values;
            this.windowEnd = windowEnd;
            this.hash = 31 * Arrays.deepHashCode(values) + Long.hashCode(windowEnd);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof GroupKey)) {
                return false;
            }
            GroupKey that = (GroupKey) other;
            return windowEnd == that.windowEnd && Arrays.deepEquals(values, that.values);
        }
    }
}
