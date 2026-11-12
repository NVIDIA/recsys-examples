// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.VarCharType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FieldRowDataKeySelectorTest {
    private static final RowType TYPE =
            RowType.of(
                    new LogicalType[] {new BigIntType(), new VarCharType(8), new IntType()},
                    new String[] {"a", "b", "c"});

    @Test
    void projectsKeyFieldsIntoAnImmutableKeyWithStableEqualityAndHash() throws Exception {
        FieldRowDataKeySelector selector = new FieldRowDataKeySelector(TYPE, new int[] {1, 0});
        GenericRowData first = GenericRowData.of(7L, StringData.fromString("x"), 1);
        GenericRowData second = GenericRowData.of(7L, StringData.fromString("x"), 99);
        GenericRowData other = GenericRowData.of(8L, StringData.fromString("x"), 1);
        RowData k1 = selector.getKey(first);
        RowData k2 = selector.copy().getKey(second);
        assertEquals(k1, k2);
        assertEquals(k1.hashCode(), k2.hashCode());
        assertNotEquals(k1, selector.getKey(other));
        assertEquals(2, k1.getArity());
        assertEquals("x", k1.getString(0).toString());
        assertEquals(7L, k1.getLong(1));
        first.setField(0, 100L);
        assertEquals(7L, k1.getLong(1), "the key must not alias the input row");
    }

    @Test
    void nullKeyFieldsAreKept() throws Exception {
        FieldRowDataKeySelector selector = new FieldRowDataKeySelector(TYPE, new int[] {0});
        RowData key = selector.getKey(GenericRowData.of(null, StringData.fromString("x"), 1));
        assertTrue(key.isNullAt(0));
    }
}
