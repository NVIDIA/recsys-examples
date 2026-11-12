// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.runtime.keyselector.RowDataKeySelector;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Bounded cache for the immutable BinaryRowData keys produced by Flink's generated key selector.
 * The GPU kernel returns one partial per group per batch, so recreating the same binary key for
 * every batch is pure host overhead. Eviction changes only performance, never state semantics.
 */
final class AggregateKeyCache {
    static final String MAX_ENTRIES_PROPERTY = "cudf.flink.aggregate.key-cache.entries";
    static final int DEFAULT_MAX_ENTRIES = 65_536;

    private final int inputArity;
    private final int[] groupingOrdinals;
    private final GpuType[] groupingTypes;
    private final GpuDecimalType[] groupingDecimalTypes;
    private final RowDataKeySelector selector;
    private final Map<Object, RowData> keys;
    private final LongIntLongKeyCache longIntLongKeys;
    private final GenericRowData selectorInput;

    AggregateKeyCache(GpuAggregatePlan plan, RowDataKeySelector selector) {
        this(
                plan.inputArity(),
                plan.groupingInputOrdinals(),
                plan.groupingTypes(),
                plan.groupingDecimalTypes(),
                selector);
    }

    AggregateKeyCache(
            int inputArity,
            int[] groupingOrdinals,
            GpuType[] groupingTypes,
            RowDataKeySelector selector) {
        this(
                inputArity,
                groupingOrdinals,
                groupingTypes,
                new GpuDecimalType[groupingTypes.length],
                selector);
    }

    AggregateKeyCache(
            int inputArity,
            int[] groupingOrdinals,
            GpuType[] groupingTypes,
            GpuDecimalType[] groupingDecimalTypes,
            RowDataKeySelector selector) {
        if (groupingOrdinals.length != groupingTypes.length) {
            throw new IllegalArgumentException("grouping ordinal/type length mismatch");
        }
        this.inputArity = inputArity;
        this.groupingOrdinals = groupingOrdinals.clone();
        this.groupingTypes = groupingTypes.clone();
        this.groupingDecimalTypes = groupingDecimalTypes.clone();
        this.selector = selector;
        int maxEntries = Integer.getInteger(MAX_ENTRIES_PROPERTY, DEFAULT_MAX_ENTRIES);
        if (maxEntries <= 0) {
            throw new IllegalArgumentException(MAX_ENTRIES_PROPERTY + " must be positive");
        }
        boolean primitiveLongIntLong =
                groupingTypes.length == 3
                        && groupingTypes[0] == GpuType.INT64
                        && groupingTypes[1] == GpuType.INT32
                        && groupingTypes[2] == GpuType.INT64;
        longIntLongKeys =
                primitiveLongIntLong ? new LongIntLongKeyCache(maxEntries) : null;
        keys =
                primitiveLongIntLong
                        ? null
                        : new LinkedHashMap<Object, RowData>(
                                Math.min(maxEntries, 1024), 0.75f, false) {
                            private static final long serialVersionUID = 1L;

                            @Override
                            protected boolean removeEldestEntry(
                                    Map.Entry<Object, RowData> eldest) {
                                return size() > maxEntries;
                            }
                        };
        selectorInput = new GenericRowData(inputArity);
    }

    RowData keyFor(Object[] groupValues) throws Exception {
        if (longIntLongKeys != null) {
            if (groupValues.length != 3) {
                throw new IllegalArgumentException("group value count changed after planning");
            }
            int nullMask =
                    (groupValues[0] == null ? 1 : 0)
                            | (groupValues[1] == null ? 2 : 0)
                            | (groupValues[2] == null ? 4 : 0);
            long value0 = groupValues[0] == null ? 0L : ((Number) groupValues[0]).longValue();
            int value1 = groupValues[1] == null ? 0 : ((Number) groupValues[1]).intValue();
            long value2 = groupValues[2] == null ? 0L : ((Number) groupValues[2]).longValue();
            RowData key = longIntLongKeys.get(value0, value1, value2, nullMask);
            if (key != null) {
                return key;
            }
            key = createKey(groupValues);
            longIntLongKeys.put(value0, value1, value2, nullMask, key);
            return key;
        }
        Object cacheKey =
                groupValues.length == 1
                        ? groupValues[0]
                        : new CompositeGroupKey(groupValues);
        RowData key = keys.get(cacheKey);
        if (key != null) {
            return key;
        }
        if (groupValues.length != groupingOrdinals.length) {
            throw new IllegalArgumentException("group value count changed after planning");
        }
        key = createKey(groupValues);
        keys.put(cacheKey, key);
        return key;
    }

    private RowData createKey(Object[] groupValues) throws Exception {
        for (int i = 0; i < groupValues.length; i++) {
            selectorInput.setField(
                    groupingOrdinals[i],
                    GpuAggregatePlan.toFlinkInternal(
                            groupValues[i], groupingTypes[i], groupingDecimalTypes[i]));
        }
        return selector.getKey(selectorInput);
    }

    void clear() {
        if (keys != null) {
            keys.clear();
        }
        if (longIntLongKeys != null) {
            longIntLongKeys.clear();
        }
    }

    private static final class CompositeGroupKey {
        private final Object[] values;
        private final int hash;

        CompositeGroupKey(Object[] values) {
            this.values = values.clone();
            this.hash = Arrays.deepHashCode(this.values);
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof CompositeGroupKey
                    && Arrays.deepEquals(values, ((CompositeGroupKey) other).values);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    /** Primitive bounded cache for the common GROUPING SETS key shape. */
    private static final class LongIntLongKeyCache {
        private final long[] value0;
        private final int[] value1;
        private final long[] value2;
        private final int[] nullMasks;
        private final RowData[] values;
        private final int mask;
        private final int maxEntries;
        private int size;

        private LongIntLongKeyCache(int maxEntries) {
            if (maxEntries > (1 << 28)) {
                throw new IllegalArgumentException(
                        MAX_ENTRIES_PROPERTY + " is too large for the primitive key cache");
            }
            int tableSize = 1;
            while (tableSize < maxEntries * 2) {
                tableSize <<= 1;
            }
            value0 = new long[tableSize];
            value1 = new int[tableSize];
            value2 = new long[tableSize];
            nullMasks = new int[tableSize];
            values = new RowData[tableSize];
            mask = tableSize - 1;
            this.maxEntries = maxEntries;
        }

        private RowData get(long first, int second, long third, int nullMask) {
            int slot = slot(first, second, third, nullMask);
            while (values[slot] != null) {
                if (value0[slot] == first
                        && value1[slot] == second
                        && value2[slot] == third
                        && nullMasks[slot] == nullMask) {
                    return values[slot];
                }
                slot = (slot + 1) & mask;
            }
            return null;
        }

        private void put(
                long first, int second, long third, int nullMask, RowData key) {
            if (size >= maxEntries) {
                clear();
            }
            int slot = slot(first, second, third, nullMask);
            while (values[slot] != null) {
                if (value0[slot] == first
                        && value1[slot] == second
                        && value2[slot] == third
                        && nullMasks[slot] == nullMask) {
                    values[slot] = key;
                    return;
                }
                slot = (slot + 1) & mask;
            }
            value0[slot] = first;
            value1[slot] = second;
            value2[slot] = third;
            nullMasks[slot] = nullMask;
            values[slot] = key;
            size++;
        }

        private int slot(long first, int second, long third, int nullMask) {
            long hash = mix64(first ^ Long.rotateLeft(third, 21));
            hash = mix64(hash ^ Integer.toUnsignedLong(second) ^ ((long) nullMask << 56));
            return (int) hash & mask;
        }

        private void clear() {
            Arrays.fill(values, null);
            size = 0;
        }

        private static long mix64(long value) {
            long z = value + 0x9e3779b97f4a7c15L;
            z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
            z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
            return z ^ (z >>> 31);
        }
    }
}
