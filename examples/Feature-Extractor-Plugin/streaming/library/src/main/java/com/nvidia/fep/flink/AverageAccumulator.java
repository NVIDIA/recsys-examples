// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;

/** Host-side exact AVG partial: the CUDA kernel produces the sum and non-null count separately. */
final class AverageAccumulator implements Serializable {
    private static final long serialVersionUID = 1L;

    final double sum;
    final long integralSum;
    final long count;
    final boolean integral;

    AverageAccumulator(double sum, long count) {
        this.sum = sum;
        this.integralSum = 0L;
        this.count = count;
        this.integral = false;
    }

    private AverageAccumulator(long integralSum, long count) {
        this.sum = 0.0d;
        this.integralSum = integralSum;
        this.count = count;
        this.integral = true;
    }

    static AverageAccumulator fromValue(Number value, GpuType argumentType) {
        return fromWeightedValue(value, argumentType, 1);
    }

    static AverageAccumulator fromWeightedValue(
            Number value, GpuType argumentType, int weight) {
        return argumentType == GpuType.INT8
                        || argumentType == GpuType.INT16
                        || argumentType == GpuType.INT32
                        || argumentType == GpuType.INT64
                ? integral(value.longValue() * weight, weight)
                : new AverageAccumulator(value.doubleValue() * weight, weight);
    }

    static AverageAccumulator integral(long sum, long count) {
        return new AverageAccumulator(sum, count);
    }

    AverageAccumulator merge(AverageAccumulator other) {
        if (integral != other.integral) {
            throw new IllegalArgumentException("cannot merge integral and floating AVG states");
        }
        long mergedCount = Math.addExact(count, other.count);
        return integral
                ? integral(integralSum + other.integralSum, mergedCount)
                : new AverageAccumulator(sum + other.sum, mergedCount);
    }

    Object value(GpuType resultType) {
        if (count <= 0L) {
            throw new IllegalStateException(
                    "materialized AVG accumulator count must be positive, got " + count);
        }
        if (integral) {
            long value = integralSum / count;
            switch (resultType) {
                case INT8:
                    return (byte) value;
                case INT16:
                    return (short) value;
                case INT32:
                    return (int) value;
                case INT64:
                    return value;
                default:
                    throw new IllegalArgumentException(
                            "integral AVG cannot return " + resultType);
            }
        }
        double value = sum / count;
        if (resultType == GpuType.FLOAT32) {
            return (float) value;
        }
        if (resultType == GpuType.FLOAT64) {
            return value;
        }
        throw new IllegalArgumentException("floating AVG cannot return " + resultType);
    }
}
