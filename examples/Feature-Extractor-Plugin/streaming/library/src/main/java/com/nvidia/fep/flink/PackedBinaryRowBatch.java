// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.core.memory.MemorySegment;
import org.apache.flink.core.memory.MemorySegmentFactory;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.binary.BinarySegmentUtils;
import org.apache.flink.table.data.binary.BinaryStringData;

import java.nio.ByteBuffer;

/**
 * Owns a batch of {@link BinaryRowData} views backed by one contiguous heap allocation.
 *
 * <p>Each row has its own non-overlapping slice, so callers may update individual fixed-width rows
 * in place. Keeping the slices in one segment avoids the byte array and memory-segment allocation
 * that {@code BinaryRowWriter} or {@code BinaryRowData.copy()} would otherwise perform per row.
 */
public final class PackedBinaryRowBatch {
    private final byte[] bytes;
    private final MemorySegment segment;
    private final BinaryRowData[] rows;
    private final int arity;
    private final int stride;

    private PackedBinaryRowBatch(int rowCount, int arity, int stride) {
        if (rowCount < 0 || arity < 0 || stride < 0) {
            throw new IllegalArgumentException("negative packed-row batch dimension");
        }
        long totalBytes = (long) rowCount * stride;
        if (totalBytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "packed-row batch exceeds the maximum Java array size: " + totalBytes);
        }
        this.bytes = new byte[(int) totalBytes];
        this.rows = new BinaryRowData[rowCount];
        this.arity = arity;
        this.stride = stride;
        segment = MemorySegmentFactory.wrap(bytes);
        MemorySegment[] sharedSegments = {segment};
        for (int index = 0; index < rowCount; index++) {
            BinaryRowData row = new BinaryRowData(arity);
            row.pointTo(sharedSegments, index * stride, stride);
            rows[index] = row;
        }
    }

    private PackedBinaryRowBatch(int arity, int[] variableBytesByRow) {
        if (arity < 0 || variableBytesByRow == null) {
            throw new IllegalArgumentException("invalid variable-width packed-row dimensions");
        }
        int fixedSize = BinaryRowData.calculateFixPartSizeInBytes(arity);
        long totalBytes = 0L;
        for (int variableBytes : variableBytesByRow) {
            if (variableBytes < 0 || (variableBytes & 7) != 0) {
                throw new IllegalArgumentException(
                        "variable-width row storage must be a non-negative multiple of eight");
            }
            totalBytes += fixedSize + (long) variableBytes;
            if (totalBytes > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "packed-row batch exceeds the maximum Java array size: " + totalBytes);
            }
        }
        this.bytes = new byte[(int) totalBytes];
        this.rows = new BinaryRowData[variableBytesByRow.length];
        this.arity = arity;
        this.stride = -1;
        segment = MemorySegmentFactory.wrap(bytes);
        MemorySegment[] sharedSegments = {segment};
        int cursor = 0;
        for (int index = 0; index < rows.length; index++) {
            int size = fixedSize + variableBytesByRow[index];
            BinaryRowData row = new BinaryRowData(arity);
            row.pointTo(sharedSegments, cursor, size);
            rows[index] = row;
            cursor += size;
        }
    }

    /** Allocates fixed-width rows that can be populated through {@code BinaryRowData.setXxx}. */
    public static PackedBinaryRowBatch fixed(int rowCount, int arity) {
        return new PackedBinaryRowBatch(
                rowCount, arity, BinaryRowData.calculateFixPartSizeInBytes(arity));
    }

    /**
     * Allocates rows with a fixed part plus the specified, already word-aligned variable storage.
     */
    public static PackedBinaryRowBatch variable(int arity, int[] variableBytesByRow) {
        return new PackedBinaryRowBatch(arity, variableBytesByRow);
    }

    /** Returns the aligned variable storage needed for one UTF-8 string value. */
    public static int stringVariableBytes(StringData value) {
        if (value == null) {
            return 0;
        }
        int length = binaryString(value).getSizeInBytes();
        return length <= 7 ? 0 : Math.addExact(length, 7) & ~7;
    }

    /** Returns a mutable view onto the indexed, non-overlapping row slice. */
    public BinaryRowData row(int index) {
        return rows[index];
    }

    /** Returns all row views. The backing storage remains owned by this batch. */
    public BinaryRowData[] rows() {
        return rows;
    }

    /** Copies one fixed-width row into the indexed slice without a per-row allocation. */
    public void copyFixedFrom(int index, BinaryRowData source) {
        if (source == null
                || source.getArity() != arity
                || source.getSizeInBytes() != stride) {
            throw new IllegalArgumentException("source is not a matching fixed-width binary row");
        }
        BinarySegmentUtils.copyToBytes(
                source.getSegments(),
                source.getOffset(),
                bytes,
                Math.multiplyExact(index, stride),
                stride);
    }

    /** Copies fixed-width source fields into the leading fields of wider fixed-width rows. */
    public void copyFixedPrefixesFrom(BinaryRowData[] source) {
        copyFixedPrefixesFrom(source, null);
    }

    /** Gathers fixed-width source fields into the leading fields of wider fixed-width rows. */
    public void copyFixedPrefixesFrom(BinaryRowData[] source, int[] ordinals) {
        if (source == null
                || stride < 0
                || (ordinals == null ? source.length : ordinals.length) != rows.length) {
            throw new IllegalArgumentException("invalid fixed-prefix source batch");
        }
        for (int target = 0; target < rows.length; target++) {
            int sourceIndex = ordinals == null ? target : ordinals[target];
            if (sourceIndex < 0 || sourceIndex >= source.length) {
                throw new IllegalArgumentException("row ordinal is outside the source batch");
            }
            BinaryRowData value = source[sourceIndex];
            if (value == null
                    || value.getArity() > arity
                    || value.getSizeInBytes()
                            != BinaryRowData.calculateFixPartSizeInBytes(value.getArity())
                    || BinaryRowData.calculateBitSetWidthInBytes(value.getArity())
                            != BinaryRowData.calculateBitSetWidthInBytes(arity)) {
                throw new IllegalArgumentException(
                        "source row is not a compatible fixed-width prefix");
            }
            BinarySegmentUtils.copyToBytes(
                    value.getSegments(),
                    value.getOffset(),
                    bytes,
                    Math.multiplyExact(target, stride),
                    value.getSizeInBytes());
        }
    }

    /**
     * Copies prefixes from another packed fixed-width batch using the backing arrays directly.
     * This is the fast path for adding trailing fields to an existing fixed-width row batch.
     */
    public void copyFixedPrefixesFrom(PackedBinaryRowBatch source) {
        copyFixedPrefixesFrom(source, null);
    }

    /** Gathers prefixes from another packed fixed-width batch using direct heap copies. */
    public void copyFixedPrefixesFrom(PackedBinaryRowBatch source, int[] ordinals) {
        if (source == null
                || stride < 0
                || source.stride < 0
                || source.arity > arity
                || (ordinals == null ? source.rows.length : ordinals.length) != rows.length
                || BinaryRowData.calculateBitSetWidthInBytes(source.arity)
                        != BinaryRowData.calculateBitSetWidthInBytes(arity)) {
            throw new IllegalArgumentException("invalid packed fixed-prefix source batch");
        }
        for (int target = 0; target < rows.length; target++) {
            int sourceIndex = ordinals == null ? target : ordinals[target];
            if (sourceIndex < 0 || sourceIndex >= source.rows.length) {
                throw new IllegalArgumentException("row ordinal is outside the source batch");
            }
            System.arraycopy(
                    source.bytes,
                    Math.multiplyExact(sourceIndex, source.stride),
                    bytes,
                    Math.multiplyExact(target, stride),
                    source.stride);
        }
    }

    /** Copies one compatible packed fixed-width prefix after shape validation by the caller. */
    void copyFixedPrefixFromUnchecked(
            int targetRow, PackedBinaryRowBatch source, int sourceRow) {
        System.arraycopy(
                source.bytes,
                sourceRow * source.stride,
                bytes,
                targetRow * stride,
                source.stride);
    }

    /** Copies packed prefixes and appends one native BIGINT result in the same row pass. */
    void copyFixedPrefixesAndLongColumnFrom(
            PackedBinaryRowBatch source, int field, ByteBuffer column, int rowCount) {
        if (source == null
                || stride < 0
                || source.stride < 0
                || source.arity > arity
                || source.rows.length != rows.length
                || rowCount != rows.length
                || field < source.arity
                || field >= arity
                || column == null
                || column.capacity() < Math.multiplyExact(rowCount, Long.BYTES)
                || BinaryRowData.calculateBitSetWidthInBytes(source.arity)
                        != BinaryRowData.calculateBitSetWidthInBytes(arity)) {
            throw new IllegalArgumentException("invalid packed prefix/result copy");
        }
        int fieldOffset =
                BinaryRowData.calculateBitSetWidthInBytes(arity) + field * Long.BYTES;
        for (int row = 0; row < rowCount; row++) {
            int targetOffset = Math.multiplyExact(row, stride);
            System.arraycopy(
                    source.bytes,
                    Math.multiplyExact(row, source.stride),
                    bytes,
                    targetOffset,
                    source.stride);
            segment.putLong(targetOffset + fieldOffset, column.getLong(row * Long.BYTES));
        }
    }

    /** Copies one fixed-width BIGINT field into an absolute-position host column buffer. */
    public void copyLongColumnTo(int field, ByteBuffer target) {
        requireLongColumn(field, target, rows.length);
        int fieldOffset =
                BinaryRowData.calculateBitSetWidthInBytes(arity) + field * Long.BYTES;
        for (int row = 0; row < rows.length; row++) {
            target.putLong(
                    row * Long.BYTES,
                    segment.getLong(Math.multiplyExact(row, stride) + fieldOffset));
        }
    }

    /** Returns the fixed-row header word containing row-kind and NULL bits. */
    long fixedHeaderWord(int row) {
        if (stride < 0 || row < 0 || row >= rows.length) {
            throw new IllegalArgumentException("fixed row header is unavailable");
        }
        return fixedHeaderWordUnchecked(row);
    }

    /** Package-local fast path after a caller has validated the fixed batch shape. */
    long fixedHeaderWordUnchecked(int row) {
        return segment.getLong(row * stride);
    }

    /** Package-local raw header writer after a caller has validated the fixed batch shape. */
    void setFixedHeaderWordUnchecked(int row, long value) {
        segment.putLong(row * stride, value);
    }

    int rowCount() {
        return rows.length;
    }

    int arity() {
        return arity;
    }

    /** Returns the raw eight-byte fixed slot for package-local native adapters. */
    long fixedSlotBits(int row, int field) {
        if (stride < 0
                || row < 0
                || row >= rows.length
                || field < 0
                || field >= arity) {
            throw new IllegalArgumentException("fixed row slot is unavailable");
        }
        return fixedSlotBitsUnchecked(row, field);
    }

    /** Package-local fast path after a caller has validated row and field bounds. */
    long fixedSlotBitsUnchecked(int row, int field) {
        int fieldOffset =
                BinaryRowData.calculateBitSetWidthInBytes(arity) + field * Long.BYTES;
        return segment.getLong(row * stride + fieldOffset);
    }

    /** Package-local raw fixed-slot writer after a caller has validated the batch shape. */
    void setFixedSlotBitsUnchecked(int row, int field, long value) {
        int fieldOffset =
                BinaryRowData.calculateBitSetWidthInBytes(arity) + field * Long.BYTES;
        segment.putLong(row * stride + fieldOffset, value);
    }

    /**
     * Copies every fixed value slot, excluding the row header, into a row-major direct buffer.
     */
    void copyFixedValuesTo(ByteBuffer target, int targetRowOffset) {
        if (stride < 0
                || target == null
                || !target.isDirect()
                || targetRowOffset < 0) {
            throw new IllegalArgumentException("invalid fixed-row value copy target");
        }
        int bitsetBytes = BinaryRowData.calculateBitSetWidthInBytes(arity);
        int valueBytes = Math.multiplyExact(arity, Long.BYTES);
        long required =
                Math.multiplyExact((long) targetRowOffset + rows.length, valueBytes);
        if (required > target.capacity()) {
            throw new IllegalArgumentException("fixed-row values exceed target buffer");
        }
        MemorySegment targetSegment = MemorySegmentFactory.wrapOffHeapMemory(target);
        for (int row = 0; row < rows.length; row++) {
            segment.copyTo(
                    Math.multiplyExact(row, stride) + bitsetBytes,
                    targetSegment,
                    Math.multiplyExact(targetRowOffset + row, valueBytes),
                    valueBytes);
        }
    }

    /** Copies row-major fixed value slots from a direct buffer into this packed batch. */
    void copyFixedValuesFrom(ByteBuffer source, int rowCount) {
        if (stride < 0
                || source == null
                || !source.isDirect()
                || rowCount < 0
                || rowCount > rows.length) {
            throw new IllegalArgumentException("invalid fixed-row value copy source");
        }
        int bitsetBytes = BinaryRowData.calculateBitSetWidthInBytes(arity);
        int valueBytes = Math.multiplyExact(arity, Long.BYTES);
        if ((long) rowCount * valueBytes > source.capacity()) {
            throw new IllegalArgumentException("fixed-row source values are incomplete");
        }
        MemorySegment sourceSegment = MemorySegmentFactory.wrapOffHeapMemory(source);
        for (int row = 0; row < rowCount; row++) {
            sourceSegment.copyTo(
                    Math.multiplyExact(row, valueBytes),
                    segment,
                    Math.multiplyExact(row, stride) + bitsetBytes,
                    valueBytes);
        }
    }

    /** Copies an absolute-position host BIGINT column into one fixed-width output field. */
    public void copyLongColumnFrom(int field, ByteBuffer source, int rowCount) {
        requireLongColumn(field, source, rowCount);
        int fieldOffset =
                BinaryRowData.calculateBitSetWidthInBytes(arity) + field * Long.BYTES;
        for (int row = 0; row < rowCount; row++) {
            segment.putLong(
                    Math.multiplyExact(row, stride) + fieldOffset,
                    source.getLong(row * Long.BYTES));
        }
    }

    private void requireLongColumn(int field, ByteBuffer buffer, int rowCount) {
        if (stride < 0
                || field < 0
                || field >= arity
                || rowCount < 0
                || rowCount > rows.length
                || buffer == null
                || buffer.capacity() < Math.multiplyExact(rowCount, Long.BYTES)) {
            throw new IllegalArgumentException("invalid packed BIGINT column copy");
        }
    }

    /** Writes one string into a variable-width row without allocating a per-row writer. */
    public void writeString(int rowIndex, int field, StringData value, int variableOffset) {
        BinaryRowData row = rows[rowIndex];
        if (field < 0 || field >= arity) {
            throw new IllegalArgumentException("string field is outside the packed row");
        }
        if (value == null) {
            row.setNullAt(field);
            return;
        }

        BinaryStringData string = binaryString(value);
        int length = string.getSizeInBytes();
        if (length <= 7) {
            long data = 0L;
            for (int index = 0; index < length; index++) {
                long unsigned = string.byteAt(index) & 0xffL;
                data |=
                        unsigned
                                << (BinaryRowData.LITTLE_ENDIAN
                                        ? index * 8
                                        : (6 - index) * 8);
            }
            row.setLong(field, (((long) length | 0x80L) << 56) | data);
            return;
        }

        int roundedLength = Math.addExact(length, 7) & ~7;
        int fixedSize = BinaryRowData.calculateFixPartSizeInBytes(arity);
        if (variableOffset < fixedSize
                || variableOffset > row.getSizeInBytes() - roundedLength) {
            throw new IllegalArgumentException("string does not fit in the packed row slice");
        }
        BinarySegmentUtils.copyToBytes(
                string.getSegments(),
                string.getOffset(),
                bytes,
                Math.addExact(row.getOffset(), variableOffset),
                length);
        row.setLong(field, ((long) variableOffset << 32) | (length & 0xffff_ffffL));
    }

    /** Makes an independently owned, contiguous copy of all source rows. */
    public static BinaryRowData[] copyOf(BinaryRowData[] source) {
        if (source == null) {
            throw new IllegalArgumentException("source rows must not be null");
        }
        return copySelected(source, null, 0, source.length);
    }

    /** Copies a contiguous source range into one independently owned allocation. */
    public static BinaryRowData[] copyOfRange(BinaryRowData[] source, int from, int to) {
        if (source == null) {
            throw new IllegalArgumentException("source rows must not be null");
        }
        if (from < 0 || to < from || to > source.length) {
            throw new IllegalArgumentException("invalid packed-row source range");
        }
        return copySelected(source, null, from, to - from);
    }

    /** Copies a contiguous range from an already validated packed fixed-width batch. */
    public static BinaryRowData[] copyOfRange(
            PackedBinaryRowBatch source, int from, int to) {
        if (source == null || source.stride < 0 || from < 0 || to < from || to > source.rows.length) {
            throw new IllegalArgumentException("invalid packed-row source range");
        }
        PackedBinaryRowBatch output = fixed(to - from, source.arity);
        System.arraycopy(
                source.bytes,
                Math.multiplyExact(from, source.stride),
                output.bytes,
                0,
                Math.multiplyExact(to - from, source.stride));
        return output.rows;
    }

    /** Gathers rows in ordinal order into one independently owned contiguous allocation. */
    public static BinaryRowData[] gather(BinaryRowData[] source, int[] ordinals) {
        if (source == null || ordinals == null) {
            throw new IllegalArgumentException("source rows and ordinals must not be null");
        }
        return copySelected(source, ordinals, 0, ordinals.length);
    }

    /** Gathers a packed fixed-width batch using direct backing-array copies. */
    public static BinaryRowData[] gather(
            PackedBinaryRowBatch source, int[] ordinals) {
        if (source == null || ordinals == null || source.stride < 0) {
            throw new IllegalArgumentException(
                    "source must be a packed fixed-width row batch");
        }
        PackedBinaryRowBatch output = fixed(ordinals.length, source.arity);
        for (int target = 0; target < ordinals.length; target++) {
            int sourceIndex = ordinals[target];
            if (sourceIndex < 0 || sourceIndex >= source.rows.length) {
                throw new IllegalArgumentException("row ordinal is outside the source batch");
            }
            System.arraycopy(
                    source.bytes,
                    Math.multiplyExact(sourceIndex, source.stride),
                    output.bytes,
                    Math.multiplyExact(target, output.stride),
                    source.stride);
        }
        return output.rows;
    }

    private static BinaryRowData[] copySelected(
            BinaryRowData[] source, int[] ordinals, int sourceOffset, int rowCount) {
        if (rowCount == 0) {
            return new BinaryRowData[0];
        }

        int arity = -1;
        long totalBytes = 0L;
        MemorySegment[] contiguousSegments = null;
        int contiguousOffset = 0;
        int nextOffset = 0;
        boolean contiguousFixedRows = ordinals == null;
        for (int position = 0; position < rowCount; position++) {
            BinaryRowData row = selectedRow(source, ordinals, sourceOffset, position);
            if (row == null) {
                throw new IllegalArgumentException("source row must not be null");
            }
            if (arity < 0) {
                arity = row.getArity();
            } else if (row.getArity() != arity) {
                throw new IllegalArgumentException("all packed rows must have the same arity");
            }
            if (contiguousFixedRows) {
                int fixedSize = BinaryRowData.calculateFixPartSizeInBytes(row.getArity());
                if (row.getSizeInBytes() != fixedSize) {
                    contiguousFixedRows = false;
                } else if (position == 0) {
                    contiguousSegments = row.getSegments();
                    contiguousOffset = row.getOffset();
                    nextOffset = row.getOffset() + fixedSize;
                } else if (row.getSegments() != contiguousSegments
                        || row.getOffset() != nextOffset) {
                    contiguousFixedRows = false;
                } else {
                    nextOffset += fixedSize;
                }
            }
            totalBytes += row.getSizeInBytes();
            if (totalBytes > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "packed-row copy exceeds the maximum Java array size: " + totalBytes);
            }
        }

        if (contiguousFixedRows) {
            PackedBinaryRowBatch output = PackedBinaryRowBatch.fixed(rowCount, arity);
            BinarySegmentUtils.copyToBytes(
                    contiguousSegments,
                    contiguousOffset,
                    output.bytes,
                    0,
                    Math.toIntExact(totalBytes));
            return output.rows;
        }

        byte[] outputBytes = new byte[(int) totalBytes];
        MemorySegment[] sharedSegments = {MemorySegmentFactory.wrap(outputBytes)};
        BinaryRowData[] output = new BinaryRowData[rowCount];
        int cursor = 0;
        for (int position = 0; position < rowCount; position++) {
            BinaryRowData sourceRow = selectedRow(source, ordinals, sourceOffset, position);
            int size = sourceRow.getSizeInBytes();
            BinarySegmentUtils.copyToBytes(
                    sourceRow.getSegments(), sourceRow.getOffset(), outputBytes, cursor, size);
            BinaryRowData outputRow = new BinaryRowData(arity);
            outputRow.pointTo(sharedSegments, cursor, size);
            output[position] = outputRow;
            cursor += size;
        }
        return output;
    }

    private static BinaryRowData selectedRow(
            BinaryRowData[] source, int[] ordinals, int sourceOffset, int position) {
        int sourceIndex = ordinals == null ? sourceOffset + position : ordinals[position];
        if (sourceIndex < 0 || sourceIndex >= source.length) {
            throw new IllegalArgumentException("row ordinal is outside the source batch");
        }
        return source[sourceIndex];
    }

    private static BinaryStringData binaryString(StringData value) {
        return value instanceof BinaryStringData
                ? (BinaryStringData) value
                : BinaryStringData.fromString(value.toString());
    }
}
