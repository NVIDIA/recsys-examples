// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LogicalType;

import java.io.Serializable;
import java.util.Objects;

/** Serializable precision/scale carried beside a DECIMAL64 or DECIMAL128 value. */
public final class GpuDecimalType implements Serializable {
    private static final long serialVersionUID = 1L;
    public static final int MAX_COMPACT_PRECISION = 18;
    public static final int MAX_PRECISION = 38;

    private final int precision;
    private final int scale;

    public GpuDecimalType(int precision, int scale) {
        if (precision < 1
                || precision > MAX_PRECISION
                || scale < 0
                || scale > precision) {
            throw new IllegalArgumentException(
                    "DECIMAL requires 1 <= precision <= 38 and 0 <= scale <= precision; got "
                            + precision
                            + ","
                            + scale);
        }
        this.precision = precision;
        this.scale = scale;
    }

    static GpuDecimalType from(LogicalType type) {
        DecimalType decimal = (DecimalType) type;
        return new GpuDecimalType(decimal.getPrecision(), decimal.getScale());
    }

    public int precision() {
        return precision;
    }

    public int scale() {
        return scale;
    }

    public boolean isCompact() {
        return precision <= MAX_COMPACT_PRECISION;
    }

    /** Native instruction encoding: low byte scale, next byte precision. */
    int packed() {
        return (precision << 8) | scale;
    }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof GpuDecimalType)) {
            return false;
        }
        GpuDecimalType that = (GpuDecimalType) other;
        return precision == that.precision && scale == that.scale;
    }

    @Override
    public int hashCode() {
        return Objects.hash(precision, scale);
    }

    @Override
    public String toString() {
        return "DECIMAL(" + precision + "," + scale + ")";
    }
}
