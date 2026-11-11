// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.RowData;

/** Safe copy of the aggregate input fields while Flink reuses RowData instances. */
final class AggregateBufferedRow implements PackedRow {
    private static final long serialVersionUID = 1L;

    private final Object[] fields;
    private final int weight;
    private final int distinctAcceptedMask;

    private AggregateBufferedRow(Object[] fields, int weight, int distinctAcceptedMask) {
        this.fields = fields;
        if (weight != 1 && weight != -1) {
            throw new IllegalArgumentException("aggregate row weight must be +1 or -1");
        }
        this.weight = weight;
        this.distinctAcceptedMask = distinctAcceptedMask;
    }

    static AggregateBufferedRow fromPacked(Object[] fields, int weight) {
        return new AggregateBufferedRow(fields, weight, 0);
    }

    static AggregateBufferedRow fromPacked(
            Object[] fields, int weight, int distinctAcceptedMask) {
        return new AggregateBufferedRow(fields, weight, distinctAcceptedMask);
    }

    static AggregateBufferedRow copy(RowData row, GpuAggregatePlan plan) {
        return copy(row, plan, 0);
    }

    static AggregateBufferedRow copy(
            RowData row, GpuAggregatePlan plan, int distinctAcceptedMask) {
        int[] ordinals = plan.inputOrdinals();
        Object[] fields = new Object[ordinals.length];
        for (int i = 0; i < fields.length; i++) {
            fields[i] = plan.copyInputValue(row, i);
        }
        int weight;
        switch (row.getRowKind()) {
            case INSERT:
            case UPDATE_AFTER:
                weight = 1;
                break;
            case UPDATE_BEFORE:
            case DELETE:
                weight = -1;
                break;
            default:
                throw new IllegalArgumentException(
                        "unsupported aggregate RowKind " + row.getRowKind());
        }
        return new AggregateBufferedRow(fields, weight, distinctAcceptedMask);
    }

    @Override
    public Object field(int packedOrdinal) {
        return fields[packedOrdinal];
    }

    int fieldCount() {
        return fields.length;
    }

    int weight() {
        return weight;
    }

    boolean distinctAccepted(int distinctIndex) {
        return (distinctAcceptedMask & (1 << distinctIndex)) != 0;
    }
}
