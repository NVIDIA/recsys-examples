// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.runtime.keyselector.RowDataKeySelector;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.runtime.typeutils.RowDataSerializer;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;

/**
 * Key selector that projects the given input fields into an immutable {@link BinaryRowData} key.
 * It stands in for the planner's generated key selector, which a DataStream job cannot use: the
 * planner (and its code generator) is not on the user classpath.
 */
final class FieldRowDataKeySelector implements RowDataKeySelector {
    private static final long serialVersionUID = 1L;

    private final int[] ordinals;
    private final LogicalType[] fieldTypes;
    private final InternalTypeInfo<RowData> keyType;
    private transient RowData.FieldGetter[] getters;
    private transient RowDataSerializer keySerializer;

    FieldRowDataKeySelector(RowType inputType, int[] ordinals) {
        this.ordinals = ordinals.clone();
        this.fieldTypes = new LogicalType[ordinals.length];
        LogicalType[] keyTypes = new LogicalType[ordinals.length];
        String[] names = new String[ordinals.length];
        for (int i = 0; i < ordinals.length; i++) {
            fieldTypes[i] = inputType.getTypeAt(ordinals[i]);
            keyTypes[i] = fieldTypes[i];
            names[i] = "f" + i;
        }
        this.keyType = InternalTypeInfo.of(RowType.of(keyTypes, names));
    }

    private FieldRowDataKeySelector(
            int[] ordinals, LogicalType[] fieldTypes, InternalTypeInfo<RowData> keyType) {
        this.ordinals = ordinals;
        this.fieldTypes = fieldTypes;
        this.keyType = keyType;
    }

    @Override
    public RowData getKey(RowData value) {
        if (getters == null) {
            getters = new RowData.FieldGetter[ordinals.length];
            for (int i = 0; i < ordinals.length; i++) {
                getters[i] = RowData.createFieldGetter(fieldTypes[i], ordinals[i]);
            }
            keySerializer = keyType.toRowSerializer();
        }
        GenericRowData key = new GenericRowData(ordinals.length);
        for (int i = 0; i < ordinals.length; i++) {
            key.setField(i, getters[i].getFieldOrNull(value));
        }
        return keySerializer.toBinaryRow(key, true);
    }

    @Override
    public InternalTypeInfo<RowData> getProducedType() {
        return keyType;
    }

    @Override
    public RowDataKeySelector copy() {
        return new FieldRowDataKeySelector(ordinals, fieldTypes, keyType);
    }
}
