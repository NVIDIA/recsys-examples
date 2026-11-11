// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.types.logical.LocalZonedTimestampType;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.TimestampType;

import java.io.Serializable;
import java.util.Optional;

/** Scalar types understood by the project-owned GPU plan and native kernels. */
public enum GpuType implements Serializable {
    BOOLEAN,
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING,
    /** Compact Flink TIMESTAMP/TIMESTAMP_LTZ payload (precision 0..3), copied but not computed. */
    TIMESTAMP3,
    /** Recursive ARRAY root carried only by nested DeviceBatch formats. */
    ARRAY,
    /** Recursive MAP root carried only by nested DeviceBatch formats. */
    MAP,
    /** Compact DECIMAL payload; precision/scale travel in {@link GpuDecimalType}. */
    DECIMAL64,
    /** Wide DECIMAL payload with a signed 128-bit unscaled value. */
    DECIMAL128;

    public static Optional<GpuType> from(LogicalType type) {
        switch (type.getTypeRoot()) {
            case BOOLEAN:
                return Optional.of(BOOLEAN);
            case TINYINT:
                return Optional.of(INT8);
            case SMALLINT:
                return Optional.of(INT16);
            case INTEGER:
                return Optional.of(INT32);
            case BIGINT:
                return Optional.of(INT64);
            case FLOAT:
                return Optional.of(FLOAT32);
            case DOUBLE:
                return Optional.of(FLOAT64);
            case CHAR:
            case VARCHAR:
                return Optional.of(STRING);
            case TIMESTAMP_WITHOUT_TIME_ZONE:
                return ((TimestampType) type).getPrecision() <= 3
                        ? Optional.of(TIMESTAMP3)
                        : Optional.empty();
            case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
                return ((LocalZonedTimestampType) type).getPrecision() <= 3
                        ? Optional.of(TIMESTAMP3)
                        : Optional.empty();
            default:
                return Optional.empty();
        }
    }

    /**
     * Scalar conversion used by operators that preserve the original row and only need a key
     * representation. Unlike {@link #from(LogicalType)}, this includes DECIMAL because sort and
     * exact byte-key kernels do not need to materialize a new decimal value.
     */
    public static Optional<GpuType> fromIncludingDecimal(LogicalType type) {
        if (type.getTypeRoot() == org.apache.flink.table.types.logical.LogicalTypeRoot.DECIMAL) {
            int precision = ((DecimalType) type).getPrecision();
            if (precision < 1 || precision > GpuDecimalType.MAX_PRECISION) {
                return Optional.empty();
            }
            return Optional.of(
                    precision <= GpuDecimalType.MAX_COMPACT_PRECISION
                            ? DECIMAL64
                            : DECIMAL128);
        }
        return from(type);
    }

    public boolean isNumeric() {
        switch (this) {
            case INT8:
            case INT16:
            case INT32:
            case INT64:
            case FLOAT32:
            case FLOAT64:
            case DECIMAL64:
            case DECIMAL128:
                return true;
            default:
                return false;
        }
    }

    public boolean isDecimal() {
        return this == DECIMAL64 || this == DECIMAL128;
    }

    /** Whether values of this type fit entirely in a {@link BinaryRowData} fixed-width slot. */
    public boolean supportsInPlaceBinaryRowWrite() {
        switch (this) {
            case BOOLEAN:
            case INT8:
            case INT16:
            case INT32:
            case INT64:
            case FLOAT32:
            case FLOAT64:
            case TIMESTAMP3:
                return true;
            default:
                return false;
        }
    }

    public Object copyFrom(RowData row, int ordinal) {
        if (row.isNullAt(ordinal)) {
            return null;
        }
        switch (this) {
            case BOOLEAN:
                return row.getBoolean(ordinal);
            case INT8:
                return row.getByte(ordinal);
            case INT16:
                return row.getShort(ordinal);
            case INT32:
                return row.getInt(ordinal);
            case INT64:
                return row.getLong(ordinal);
            case FLOAT32:
                return row.getFloat(ordinal);
            case FLOAT64:
                return row.getDouble(ordinal);
            case STRING:
                return row.getString(ordinal).toString();
            case TIMESTAMP3:
                return row.getTimestamp(ordinal, 3);
            default:
                throw new IllegalStateException("Unhandled GPU type: " + this);
        }
    }

    /** Copies one internal Flink value into a reusable binary output row without boxing it. */
    public void writeFromRow(
            RowData source, int sourceOrdinal, BinaryRowWriter target, int targetOrdinal) {
        if (source.isNullAt(sourceOrdinal)) {
            target.setNullAt(targetOrdinal);
            return;
        }
        switch (this) {
            case BOOLEAN:
                target.writeBoolean(targetOrdinal, source.getBoolean(sourceOrdinal));
                break;
            case INT8:
                target.writeByte(targetOrdinal, source.getByte(sourceOrdinal));
                break;
            case INT16:
                target.writeShort(targetOrdinal, source.getShort(sourceOrdinal));
                break;
            case INT32:
                target.writeInt(targetOrdinal, source.getInt(sourceOrdinal));
                break;
            case INT64:
                target.writeLong(targetOrdinal, source.getLong(sourceOrdinal));
                break;
            case FLOAT32:
                target.writeFloat(targetOrdinal, source.getFloat(sourceOrdinal));
                break;
            case FLOAT64:
                target.writeDouble(targetOrdinal, source.getDouble(sourceOrdinal));
                break;
            case STRING:
                target.writeString(targetOrdinal, source.getString(sourceOrdinal));
                break;
            case TIMESTAMP3:
                target.writeTimestamp(targetOrdinal, source.getTimestamp(sourceOrdinal, 3), 3);
                break;
            default:
                throw new IllegalStateException("Unhandled GPU type: " + this);
        }
    }

    public Object toFlinkInternal(Object value) {
        return value != null && this == STRING ? StringData.fromString((String) value) : value;
    }

    /** Writes a project-owned partial value directly to a reusable Flink binary row. */
    public void writeValue(Object value, BinaryRowWriter target, int ordinal) {
        if (value == null) {
            target.setNullAt(ordinal);
            return;
        }
        switch (this) {
            case BOOLEAN:
                target.writeBoolean(ordinal, (Boolean) value);
                break;
            case INT8:
                target.writeByte(ordinal, ((Number) value).byteValue());
                break;
            case INT16:
                target.writeShort(ordinal, ((Number) value).shortValue());
                break;
            case INT32:
                target.writeInt(ordinal, ((Number) value).intValue());
                break;
            case INT64:
                target.writeLong(ordinal, ((Number) value).longValue());
                break;
            case FLOAT32:
                target.writeFloat(ordinal, ((Number) value).floatValue());
                break;
            case FLOAT64:
                target.writeDouble(ordinal, ((Number) value).doubleValue());
                break;
            case STRING:
                target.writeString(
                        ordinal,
                        value instanceof StringData
                                ? (StringData) value
                                : StringData.fromString((String) value));
                break;
            case TIMESTAMP3:
                target.writeTimestamp(ordinal, (TimestampData) value, 3);
                break;
            default:
                throw new IllegalStateException("Unhandled GPU type: " + this);
        }
    }

    /** Writes a fixed-width project value directly into an already allocated binary row. */
    public void writeValue(Object value, BinaryRowData target, int ordinal) {
        if (!supportsInPlaceBinaryRowWrite()) {
            throw new IllegalStateException("GPU type requires a variable-width row writer: " + this);
        }
        if (value == null) {
            target.setNullAt(ordinal);
            return;
        }
        switch (this) {
            case BOOLEAN:
                target.setBoolean(ordinal, (Boolean) value);
                break;
            case INT8:
                target.setByte(ordinal, ((Number) value).byteValue());
                break;
            case INT16:
                target.setShort(ordinal, ((Number) value).shortValue());
                break;
            case INT32:
                target.setInt(ordinal, ((Number) value).intValue());
                break;
            case INT64:
                target.setLong(ordinal, ((Number) value).longValue());
                break;
            case FLOAT32:
                target.setFloat(ordinal, ((Number) value).floatValue());
                break;
            case FLOAT64:
                target.setDouble(ordinal, ((Number) value).doubleValue());
                break;
            case TIMESTAMP3:
                target.setTimestamp(ordinal, (TimestampData) value, 3);
                break;
            default:
                throw new IllegalStateException("Unhandled fixed-width GPU type: " + this);
        }
    }

}
