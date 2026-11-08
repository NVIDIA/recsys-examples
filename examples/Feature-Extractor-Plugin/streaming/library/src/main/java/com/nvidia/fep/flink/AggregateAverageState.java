// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;

/** Checkpointed hidden SUM/COUNT state for AVG calls whose public Flink row stores only the mean. */
public final class AggregateAverageState implements Serializable {
    private static final long serialVersionUID = 1L;

    public double[] sums;
    public long[] integralSums;
    public long[] counts;

    public AggregateAverageState() {}

    AggregateAverageState(int aggregateCount) {
        sums = new double[aggregateCount];
        integralSums = new long[aggregateCount];
        counts = new long[aggregateCount];
    }

    AverageAccumulator get(int index) {
        return counts[index] == 0L ? null : new AverageAccumulator(sums[index], counts[index]);
    }

    AverageAccumulator get(int index, GpuType argumentType) {
        if (counts[index] == 0L) {
            return null;
        }
        boolean integral =
                argumentType == GpuType.INT8
                        || argumentType == GpuType.INT16
                        || argumentType == GpuType.INT32
                        || argumentType == GpuType.INT64;
        return integral
                ? AverageAccumulator.integral(
                        integralSums == null ? (long) sums[index] : integralSums[index],
                        counts[index])
                : new AverageAccumulator(sums[index], counts[index]);
    }

    void set(int index, AverageAccumulator accumulator) {
        if (accumulator != null) {
            if (accumulator.integral) {
                if (integralSums == null) {
                    integralSums = new long[sums.length];
                }
                integralSums[index] = accumulator.integralSum;
            } else {
                sums[index] = accumulator.sum;
            }
            counts[index] = accumulator.count;
        }
    }
}
