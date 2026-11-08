// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.types.RowKind;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PackedBinaryRowBatchTest {
    @Test
    void fixedRowsUseIndependentSlices() {
        PackedBinaryRowBatch batch = PackedBinaryRowBatch.fixed(2, 3);
        BinaryRowData first = batch.row(0);
        BinaryRowData second = batch.row(1);

        first.setLong(0, 11L);
        first.setNullAt(1);
        first.setDouble(2, 1.5);
        second.setLong(0, 22L);
        second.setInt(1, 7);
        second.setDouble(2, 2.5);

        assertEquals(RowKind.INSERT, first.getRowKind());
        assertEquals(RowKind.INSERT, second.getRowKind());
        assertEquals(11L, first.getLong(0));
        assertEquals(22L, second.getLong(0));
        assertEquals(7, second.getInt(1));
        assertEquals(1.5, first.getDouble(2));
        assertEquals(2.5, second.getDouble(2));
        assertFalse(second.isNullAt(1));
        assertEquals(true, first.isNullAt(1));
    }

    @Test
    void gatherCopiesVariableRowsInRequestedOrder() {
        BinaryRowData first = stringRow(10L, "a string longer than seven bytes");
        BinaryRowData second = stringRow(20L, null);

        BinaryRowData[] gathered =
                PackedBinaryRowBatch.gather(
                        new BinaryRowData[] {first, second}, new int[] {1, 0, 1});

        assertEquals(3, gathered.length);
        assertEquals(20L, gathered[0].getLong(0));
        assertEquals(true, gathered[0].isNullAt(1));
        assertEquals(10L, gathered[1].getLong(0));
        assertEquals("a string longer than seven bytes", gathered[1].getString(1).toString());
        assertNotSame(first.getSegments(), gathered[1].getSegments());
        first.setLong(0, 99L);
        assertEquals(10L, gathered[1].getLong(0));
        gathered[0].setLong(0, 30L);
        assertEquals(20L, gathered[2].getLong(0));
    }

    @Test
    void writesInlineAndVariableStringsIntoPreSizedRows() {
        StringData inline = StringData.fromString("🔥");
        StringData variable = StringData.fromString("a string longer than seven bytes");
        int[] variableBytes = {
            PackedBinaryRowBatch.stringVariableBytes(inline),
            PackedBinaryRowBatch.stringVariableBytes(variable)
        };
        PackedBinaryRowBatch batch = PackedBinaryRowBatch.variable(2, variableBytes);
        int fixedSize = BinaryRowData.calculateFixPartSizeInBytes(2);

        batch.row(0).setLong(0, 1L);
        batch.writeString(0, 1, inline, fixedSize);
        batch.row(1).setLong(0, 2L);
        batch.writeString(1, 1, variable, fixedSize);

        assertEquals("🔥", batch.row(0).getString(1).toString());
        assertEquals("a string longer than seven bytes", batch.row(1).getString(1).toString());
    }

    @Test
    void copiesFixedValuesAndLongColumnsWithoutChangingNullBits() {
        PackedBinaryRowBatch source = PackedBinaryRowBatch.fixed(2, 3);
        source.row(0).setLong(0, 11L);
        source.row(0).setNullAt(1);
        source.row(0).setDouble(2, 1.5d);
        source.row(1).setLong(0, 22L);
        source.row(1).setInt(1, 7);
        source.row(1).setDouble(2, 2.5d);

        ByteBuffer values =
                ByteBuffer.allocateDirect(2 * 3 * Long.BYTES)
                        .order(ByteOrder.nativeOrder());
        source.copyFixedValuesTo(values, 0);
        PackedBinaryRowBatch restored = PackedBinaryRowBatch.fixed(2, 3);
        restored.copyFixedValuesFrom(values, 2);
        assertEquals(11L, restored.row(0).getLong(0));
        assertEquals(7, restored.row(1).getInt(1));
        assertEquals(2.5d, restored.row(1).getDouble(2));

        ByteBuffer column =
                ByteBuffer.allocateDirect(2 * Long.BYTES).order(ByteOrder.nativeOrder());
        source.copyLongColumnTo(0, column);
        PackedBinaryRowBatch copiedColumn = PackedBinaryRowBatch.fixed(2, 1);
        copiedColumn.copyLongColumnFrom(0, column, 2);
        assertEquals(11L, copiedColumn.row(0).getLong(0));
        assertEquals(22L, copiedColumn.row(1).getLong(0));

        // Value-area transfer deliberately leaves the destination header untouched.
        assertFalse(restored.row(0).isNullAt(1));
    }

    @Test
    void packedPrefixAndGatherKeepCompleteIndependentRows() {
        PackedBinaryRowBatch source = PackedBinaryRowBatch.fixed(2, 3);
        source.row(0).setLong(0, 10L);
        source.row(0).setInt(1, 1);
        source.row(0).setDouble(2, 1.25d);
        source.row(1).setLong(0, 20L);
        source.row(1).setNullAt(1);
        source.row(1).setDouble(2, 2.5d);

        PackedBinaryRowBatch prefixed = PackedBinaryRowBatch.fixed(2, 4);
        prefixed.copyFixedPrefixesFrom(source, new int[] {1, 0});
        prefixed.row(0).setLong(3, 101L);
        prefixed.row(1).setLong(3, 202L);
        assertEquals(20L, prefixed.row(0).getLong(0));
        assertEquals(true, prefixed.row(0).isNullAt(1));
        assertEquals(101L, prefixed.row(0).getLong(3));
        assertEquals(10L, prefixed.row(1).getLong(0));

        BinaryRowData[] gathered =
                PackedBinaryRowBatch.gather(source, new int[] {1, 0, 1});
        assertEquals(20L, gathered[0].getLong(0));
        assertEquals(10L, gathered[1].getLong(0));
        source.row(1).setLong(0, 99L);
        assertEquals(20L, gathered[0].getLong(0));
        assertEquals(20L, gathered[2].getLong(0));
    }

    @Test
    void copiesPrefixAndNativeResultInOnePass() {
        PackedBinaryRowBatch source = PackedBinaryRowBatch.fixed(2, 2);
        source.row(0).setLong(0, 11L);
        source.row(0).setNullAt(1);
        source.row(1).setLong(0, 22L);
        source.row(1).setInt(1, 7);
        ByteBuffer result =
                ByteBuffer.allocateDirect(2 * Long.BYTES).order(ByteOrder.nativeOrder());
        result.putLong(0, 101L);
        result.putLong(Long.BYTES, 202L);

        PackedBinaryRowBatch output = PackedBinaryRowBatch.fixed(2, 3);
        output.copyFixedPrefixesAndLongColumnFrom(source, 2, result, 2);

        assertEquals(11L, output.row(0).getLong(0));
        assertEquals(true, output.row(0).isNullAt(1));
        assertEquals(101L, output.row(0).getLong(2));
        assertEquals(22L, output.row(1).getLong(0));
        assertEquals(7, output.row(1).getInt(1));
        assertEquals(202L, output.row(1).getLong(2));
    }

    @Test
    void rejectsOversizedFixedBatchBeforeAllocating() {
        assertThrows(
                IllegalArgumentException.class,
                () -> PackedBinaryRowBatch.fixed(Integer.MAX_VALUE, 2));
    }

    private static BinaryRowData stringRow(long id, String value) {
        BinaryRowData row = new BinaryRowData(2);
        BinaryRowWriter writer = new BinaryRowWriter(row);
        writer.writeLong(0, id);
        if (value == null) {
            writer.setNullAt(1);
        } else {
            writer.writeString(1, StringData.fromString(value));
        }
        writer.complete();
        return row;
    }
}
