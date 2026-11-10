// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** One per-key-and-window partial produced by a project-owned GPU group-by batch. */
final class WindowAggregatePartial {
    final long windowEnd;
    final Object[] groupValues;
    final long inputRows;
    final Object[] aggregateValues;

    WindowAggregatePartial(
            long windowEnd,
            Object[] groupValues,
            long inputRows,
            Object[] aggregateValues) {
        this.windowEnd = windowEnd;
        this.groupValues = groupValues;
        this.inputRows = inputRows;
        this.aggregateValues = aggregateValues;
    }
}
