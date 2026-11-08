// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** One per-key partial result produced by a project-owned GPU group-by batch. */
final class AggregatePartial {
    final Object[] groupValues;
    final long inputRows;
    final Object[] aggregateValues;

    AggregatePartial(Object[] groupValues, long inputRows, Object[] aggregateValues) {
        this.groupValues = groupValues;
        this.inputRows = inputRows;
        this.aggregateValues = aggregateValues;
    }
}
