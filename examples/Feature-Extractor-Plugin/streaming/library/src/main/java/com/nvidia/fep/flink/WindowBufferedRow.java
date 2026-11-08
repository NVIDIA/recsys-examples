// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** Copied aggregate fields plus the event-time tumbling-window end. */
final class WindowBufferedRow implements PackedRow {
    final AggregateBufferedRow aggregateRow;
    final long windowEnd;

    WindowBufferedRow(AggregateBufferedRow aggregateRow, long windowEnd) {
        this.aggregateRow = aggregateRow;
        this.windowEnd = windowEnd;
    }

    @Override
    public Object field(int packedOrdinal) {
        if (packedOrdinal == aggregateRow.fieldCount()) {
            return windowEnd;
        }
        return aggregateRow.field(packedOrdinal);
    }
}
