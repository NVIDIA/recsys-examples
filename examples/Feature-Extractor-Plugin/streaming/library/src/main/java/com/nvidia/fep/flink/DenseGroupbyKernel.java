// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.TimestampData;

/** GPU numeric aggregate kernel with a per-batch dense key dictionary. */
public final class DenseGroupbyKernel implements AutoCloseable {
    static final int OP_ROW_COUNT = 1 << 0;
    static final int OP_METRIC_COUNT = 1 << 1;
    static final int OP_INT_COUNT = 1 << 2;
    static final int OP_LONG0_COUNT = 1 << 3;
    static final int OP_LONG1_COUNT = 1 << 4;
    static final int OP_FILTER0_COUNT = 1 << 5;
    static final int OP_FILTER1_COUNT = 1 << 6;
    static final int OP_FILTER2_COUNT = 1 << 7;
    static final int OP_METRIC_SUM = 1 << 8;
    static final int OP_MIN_INT = 1 << 9;
    static final int OP_MAX_INT = 1 << 10;
    static final int OP_LONG0_SUM = 1 << 11;
    static final int OP_LONG1_SUM = 1 << 12;
    static final int OP_MIN_LONG0 = 1 << 13;
    static final int OP_MAX_LONG0 = 1 << 14;
    static final int OP_MIN_LONG1 = 1 << 15;
    static final int OP_MAX_LONG1 = 1 << 16;
    static final int OP_INT_SUM = 1 << 17;
    static final int OP_MIN_METRIC = 1 << 18;
    static final int OP_MAX_METRIC = 1 << 19;
    private static final int ALL_OPERATION_MASK = (1 << 20) - 1;
    private static final String FORCE_ALL_OPERATIONS_PROPERTY =
            "cudf.flink.groupby.force-all-operations";
    private static final int INPUT_ROW_BYTES = 64;
    private static final int OUTPUT_HEADER_BYTES = 64;
    private static final int OUTPUT_ROW_BYTES = 160;
    private static final int FLEXIBLE_RESULT_BYTES = 16;
    private static final int FLEXIBLE_MAX_AGGREGATES = 16;
    private static final int FLEX_KIND_COUNT = 0;
    private static final int FLEX_KIND_SUM = 1;
    private static final int FLEX_KIND_AVG = 2;
    private static final int FLEX_KIND_MIN = 3;
    private static final int FLEX_KIND_MAX = 4;
    private static final int FLEX_VALUE_ROW = 0;
    private static final int FLEX_VALUE_METRIC = 1;
    private static final int FLEX_VALUE_INT = 2;
    private static final int FLEX_VALUE_LONG0 = 3;
    private static final int FLEX_VALUE_LONG1 = 4;
    private static final int FILTER_UNBOUND = -2;

    private static final int INPUT_GROUP_OFFSET = 0;
    private static final int INPUT_INT_OFFSET = 4;
    private static final int INPUT_METRIC_OFFSET = 8;
    private static final int INPUT_SEQUENCE_OFFSET = 16;
    private static final int INPUT_SENT_NANOS_OFFSET = 24;
    private static final int INPUT_METRIC_VALID_OFFSET = 32;
    private static final int INPUT_INT_VALID_OFFSET = 36;
    private static final int INPUT_SEQUENCE_VALID_OFFSET = 40;
    private static final int INPUT_SENT_NANOS_VALID_OFFSET = 44;
    private static final int[] INPUT_FILTER_OFFSETS = {48, 52, 56};
    private static final int INPUT_WEIGHT_OFFSET = 60;

    private static final int HEADER_H2D_NANOS_OFFSET = 0;
    private static final int HEADER_COMPUTE_NANOS_OFFSET = 8;
    private static final int HEADER_D2H_NANOS_OFFSET = 16;
    private static final int HEADER_INPUT_BYTES_OFFSET = 24;
    private static final int HEADER_OUTPUT_BYTES_OFFSET = 32;

    private static final int OUTPUT_ROW_COUNT_OFFSET = 0;
    private static final int OUTPUT_METRIC_COUNT_OFFSET = 8;
    private static final int OUTPUT_INT_COUNT_OFFSET = 16;
    private static final int OUTPUT_LONG0_COUNT_OFFSET = 24;
    private static final int OUTPUT_LONG1_COUNT_OFFSET = 32;
    private static final int[] OUTPUT_FILTER_COUNT_OFFSETS = {40, 48, 56};
    private static final int OUTPUT_METRIC_SUM_OFFSET = 64;
    private static final int OUTPUT_MIN_INT_OFFSET = 72;
    private static final int OUTPUT_MAX_INT_OFFSET = 76;
    private static final int OUTPUT_INT_SUM_OFFSET = 80;
    private static final int OUTPUT_LONG0_SUM_OFFSET = 88;
    private static final int OUTPUT_LONG1_SUM_OFFSET = 96;
    private static final int OUTPUT_MIN_LONG0_OFFSET = 104;
    private static final int OUTPUT_MAX_LONG0_OFFSET = 112;
    private static final int OUTPUT_MIN_LONG1_OFFSET = 120;
    private static final int OUTPUT_MAX_LONG1_OFFSET = 128;
    private static final int OUTPUT_MIN_METRIC_OFFSET = 136;
    private static final int OUTPUT_MAX_METRIC_OFFSET = 144;

    private final GpuAggregatePlan plan;
    private final GpuNativeDeviceGuard deviceGuard;
    private final int capacity;
    private final int[] groupingPackedOrdinals;
    private final int[] groupingInputOrdinals;
    private final GpuType[] groupingTypes;
    private final int metricPackedOrdinal;
    private final int intPackedOrdinal;
    private final int sequencePackedOrdinal;
    private final int sentNanosPackedOrdinal;
    private final int metricInputOrdinal;
    private final int intInputOrdinal;
    private final int sequenceInputOrdinal;
    private final int sentNanosInputOrdinal;
    private final GpuType metricInputType;
    private final GpuType intInputType;
    private final GpuType sequenceInputType;
    private final GpuType sentNanosInputType;
    private final int[] filterPackedOrdinals;
    private final int[] filterInputOrdinals;
    private final OutputLane[] outputLanes;
    private final boolean flexible;
    private final int outputRowBytes;
    private final ByteBuffer input;
    private final ByteBuffer output;
    private final boolean fastLongKey;
    private final boolean fastLongStringKey;
    private final boolean fastLongIntLongKey;
    private final DenseKeyMap keyMap;
    private final LongStringDenseKeyMap longStringKeyMap;
    private final LongIntLongDenseKeyMap longIntLongKeyMap;
    private final GenericDenseKeyMap genericKeyMap;
    private final long[] windowEndsByRow;
    private final String[] long0StringValues;
    private final String[] long1StringValues;
    private String[] long0RankedStrings;
    private String[] long1RankedStrings;
    private final GpuNativeHandleResourceLifecycle nativeResources;
    private int bufferedRows;
    private boolean bufferedWindowed;
    private boolean batchNeedsFallback;
    private long bufferedPackNanos;
    private long handle;

    static DenseGroupbyKernel createRequired(GpuAggregatePlan plan, int capacity) {
        String unsupported = unsupportedReason(plan);
        if (unsupported != null) {
            throw new IllegalArgumentException(
                    "project-owned dense GPU aggregate is unsupported: " + unsupported);
        }
        NativeLibrary.ensureLoaded();
        return new DenseGroupbyKernel(plan, capacity);
    }

    public static String unsupportedReason(GpuAggregatePlan plan) {
        GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
        if (specs.length == 0) {
            return "needs at least one aggregate call";
        }
        try {
            layoutFor(specs);
            return null;
        } catch (UnsupportedLayout fixedLayoutUnsupported) {
            try {
                flexibleLayoutFor(specs);
                return null;
            } catch (UnsupportedLayout flexibleLayoutUnsupported) {
                return flexibleLayoutUnsupported.getMessage();
            }
        }
    }

    static String fixedLayoutUnsupportedReason(GpuAggregatePlan plan) {
        try {
            layoutFor(plan.aggregates());
            return null;
        } catch (UnsupportedLayout unsupported) {
            return unsupported.getMessage();
        }
    }

    private DenseGroupbyKernel(GpuAggregatePlan plan, int capacity) {
        this.plan = plan;
        this.nativeResources = GpuNativeHandleResourceLifecycle.standalone(this);
        this.deviceGuard = GpuNativeDeviceGuard.capture();
        deviceGuard.bind();
        this.capacity = capacity;
        SupportLayout layout = null;
        FlexibleLayout flexibleLayout = null;
        try {
            layout = layoutFor(plan.aggregates());
        } catch (UnsupportedLayout fixedLayoutUnsupported) {
            flexibleLayout = flexibleLayoutFor(plan.aggregates());
        }
        flexible = flexibleLayout != null;
        outputLanes = flexible ? null : layout.outputLanes;
        groupingPackedOrdinals = plan.groupingPackedOrdinals().clone();
        groupingInputOrdinals = plan.groupingInputOrdinals().clone();
        groupingTypes = plan.groupingTypes().clone();
        metricPackedOrdinal =
                flexible ? flexibleLayout.metricPackedOrdinal : layout.metricPackedOrdinal;
        intPackedOrdinal =
                flexible ? flexibleLayout.intPackedOrdinal : layout.intPackedOrdinal;
        sequencePackedOrdinal =
                flexible ? flexibleLayout.long0PackedOrdinal : layout.long0PackedOrdinal;
        sentNanosPackedOrdinal =
                flexible ? flexibleLayout.long1PackedOrdinal : layout.long1PackedOrdinal;
        filterPackedOrdinals =
                (flexible ? flexibleLayout.filterPackedOrdinals : layout.filterPackedOrdinals)
                        .clone();
        int[] inputOrdinals = plan.inputOrdinals();
        metricInputOrdinal = inputOrdinal(inputOrdinals, metricPackedOrdinal);
        intInputOrdinal = inputOrdinal(inputOrdinals, intPackedOrdinal);
        sequenceInputOrdinal = inputOrdinal(inputOrdinals, sequencePackedOrdinal);
        sentNanosInputOrdinal = inputOrdinal(inputOrdinals, sentNanosPackedOrdinal);
        GpuType[] packedInputTypes = plan.inputTypes();
        metricInputType = packedInputType(packedInputTypes, metricPackedOrdinal);
        intInputType = packedInputType(packedInputTypes, intPackedOrdinal);
        sequenceInputType = packedInputType(packedInputTypes, sequencePackedOrdinal);
        sentNanosInputType = packedInputType(packedInputTypes, sentNanosPackedOrdinal);
        filterInputOrdinals = new int[filterPackedOrdinals.length];
        for (int i = 0; i < filterPackedOrdinals.length; i++) {
            filterInputOrdinals[i] =
                    filterPackedOrdinals[i] >= 0
                            ? inputOrdinal(inputOrdinals, filterPackedOrdinals[i])
                            : -1;
        }
        if (flexible) {
            outputRowBytes = FLEXIBLE_RESULT_BYTES * (plan.aggregates().length + 1);
            handle =
                    nativeCreateFlexible(
                            capacity,
                            flexibleLayout.kinds,
                            flexibleLayout.valueLanes,
                            flexibleLayout.filterLanes);
        } else {
            outputRowBytes = OUTPUT_ROW_BYTES;
            int operationMask =
                    Boolean.getBoolean(FORCE_ALL_OPERATIONS_PROPERTY)
                            ? ALL_OPERATION_MASK
                            : layout.operationMask;
            handle =
                    nativeCreate(
                            capacity,
                            operationMask,
                            nativeFilterLane(layout.metricFilterLane),
                            nativeFilterLane(layout.intFilterLane),
                            nativeFilterLane(layout.long0FilterLane),
                            nativeFilterLane(layout.long1FilterLane));
        }
        if (handle == 0) {
            throw new IllegalStateException("dense groupby native context returned null");
        }
        try {
            nativeResources.opened(this::readNativeResourceStats);
        } catch (RuntimeException | Error accountingFailure) {
            nativeResources.rollbackFailedOpen(
                    accountingFailure, this::closeNativeContext);
            throw accountingFailure;
        }
        ByteBuffer nativeInput;
        ByteBuffer nativeOutput;
        try {
            nativeInput =
                    flexible ? nativeFlexibleInputBuffer(handle) : nativeInputBuffer(handle);
            nativeOutput =
                    flexible ? nativeFlexibleOutputBuffer(handle) : nativeOutputBuffer(handle);
        } catch (RuntimeException | Error bufferFailure) {
            cleanupFailedOpen(bufferFailure);
            throw bufferFailure;
        }
        if (nativeInput == null
                || nativeOutput == null
                || !nativeInput.isDirect()
                || !nativeOutput.isDirect()
                || nativeInput.capacity() < capacity * INPUT_ROW_BYTES
                || nativeOutput.capacity()
                        < OUTPUT_HEADER_BYTES
                                + capacity * outputRowBytes) {
            IllegalStateException failure =
                    new IllegalStateException("dense groupby returned invalid native buffers");
            cleanupFailedOpen(failure);
            throw failure;
        }
        input = nativeInput.order(ByteOrder.nativeOrder());
        output = nativeOutput.order(ByteOrder.nativeOrder());
        try {
            fastLongKey = groupingTypes.length == 1 && groupingTypes[0] == GpuType.INT64;
            fastLongStringKey =
                    groupingTypes.length == 2
                            && groupingTypes[0] == GpuType.INT64
                            && groupingTypes[1] == GpuType.STRING;
            fastLongIntLongKey =
                    groupingTypes.length == 3
                            && groupingTypes[0] == GpuType.INT64
                            && groupingTypes[1] == GpuType.INT32
                            && groupingTypes[2] == GpuType.INT64;
            keyMap = fastLongKey ? new DenseKeyMap(capacity) : null;
            longStringKeyMap = fastLongStringKey ? new LongStringDenseKeyMap(capacity) : null;
            longIntLongKeyMap =
                    fastLongIntLongKey ? new LongIntLongDenseKeyMap(capacity) : null;
            genericKeyMap =
                    fastLongKey || fastLongStringKey || fastLongIntLongKey
                            ? null
                            : new GenericDenseKeyMap(capacity);
            windowEndsByRow = new long[capacity];
            long0StringValues =
                    sequenceInputType == GpuType.STRING ? new String[capacity] : null;
            long1StringValues =
                    sentNanosInputType == GpuType.STRING ? new String[capacity] : null;
            resetBatch();
        } catch (RuntimeException | Error initializationFailure) {
            try {
                // close() obtains a final native pool stats observation before marking the owner
                // closed. If cleanup itself fails, its live owner remains visible in the process
                // release snapshot rather than being reported as zero.
                close();
            } catch (RuntimeException | Error cleanupFailure) {
                initializationFailure.addSuppressed(cleanupFailure);
            }
            throw initializationFailure;
        }
    }

    private int filterPackedOrdinal(int lane) {
        return lane < filterPackedOrdinals.length ? filterPackedOrdinals[lane] : -1;
    }

    private static GpuType packedInputType(GpuType[] inputTypes, int packedOrdinal) {
        return packedOrdinal < 0 ? null : inputTypes[packedOrdinal];
    }

    private long[] readNativeResourceStats() {
        return flexible
                ? nativeFlexibleResourceStats(handle)
                : nativeResourceStats(handle);
    }

    private void cleanupFailedOpen(Throwable openFailure) {
        try {
            nativeResources.close(this::closeNativeContext);
        } catch (RuntimeException | Error cleanupFailure) {
            if (cleanupFailure != openFailure) {
                openFailure.addSuppressed(cleanupFailure);
            }
        }
    }

    private void closeNativeContext() {
        if (flexible) {
            nativeCloseFlexible(handle);
        } else {
            nativeClose(handle);
        }
        handle = 0L;
    }

    private static int inputOrdinal(int[] inputOrdinals, int packedOrdinal) {
        return packedOrdinal < 0 ? -1 : inputOrdinals[packedOrdinal];
    }

    private static int nativeFilterLane(int filterLane) {
        return filterLane == FILTER_UNBOUND ? -1 : filterLane;
    }

    void append(RowData row) {
        append(row, 1);
    }

    /** Packs an ordinary aggregate batch while sampling adapter time only once. */
    void appendBatch(RowData[] rows) {
        if (rows == null || rows.length > capacity - bufferedRows) {
            throw new IllegalStateException("dense groupby input exceeded capacity " + capacity);
        }
        long started = System.nanoTime();
        for (RowData row : rows) {
            appendInternalUntimed(row, Long.MIN_VALUE, false, 1, 0);
        }
        bufferedPackNanos += System.nanoTime() - started;
    }

    /** Fast path for the aligned nullable BIGINT key/value aggregate row layout. */
    void appendPackedLongKeyLongValueBatch(PackedBinaryRowBatch batch) {
        BinaryRowData[] rows = batch == null ? null : batch.rows();
        if (rows == null
                || rows.length == 0
                || rows.length > capacity - bufferedRows
                || rows[0].getArity() != 2
                || !fastLongKey
                || groupingInputOrdinals[0] != 0
                || metricInputOrdinal >= 0
                || intInputOrdinal >= 0
                || sequenceInputOrdinal != 1
                || sequenceInputType != GpuType.INT64
                || sentNanosInputOrdinal >= 0
                || long0StringValues != null
                || long1StringValues != null) {
            throw new IllegalArgumentException(
                    "dense groupby packed BIGINT batch has the wrong layout");
        }
        for (int ordinal : filterInputOrdinals) {
            if (ordinal >= 0) {
                throw new IllegalArgumentException(
                        "dense groupby packed BIGINT batch cannot carry filters");
            }
        }
        if (bufferedRows != 0 && bufferedWindowed) {
            throw new IllegalStateException(
                    "cannot mix packed BIGINT Aggregate and other input modes");
        }

        long started = System.nanoTime();
        for (int source = 0; source < rows.length; source++) {
            long header = batch.fixedHeaderWordUnchecked(source);
            boolean keyNull = (header & (1L << 8)) != 0L;
            long key = keyNull ? 0L : batch.fixedSlotBitsUnchecked(source, 0);
            boolean valueValid = (header & (1L << 9)) == 0L;
            long value = valueValid ? batch.fixedSlotBitsUnchecked(source, 1) : 0L;
            int group = keyMap.idFor(key, keyNull, Long.MIN_VALUE, false);
            int target = bufferedRows + source;
            int base = target * INPUT_ROW_BYTES;
            input.putInt(base + INPUT_GROUP_OFFSET, group);
            input.putLong(base + INPUT_SEQUENCE_OFFSET, value);
            input.putInt(base + INPUT_SEQUENCE_VALID_OFFSET, valueValid ? 1 : 0);
            input.putInt(base + INPUT_WEIGHT_OFFSET, 1);
            windowEndsByRow[target] = Long.MIN_VALUE;
        }
        bufferedRows += rows.length;
        bufferedWindowed = false;
        bufferedPackNanos += System.nanoTime() - started;
    }

    void append(RowData row, int weight) {
        append(row, weight, 0);
    }

    void append(RowData row, int weight, int distinctAcceptedMask) {
        if (weight != 1 && weight != -1) {
            throw new IllegalArgumentException("aggregate row weight must be +1 or -1");
        }
        appendInternal(row, Long.MIN_VALUE, false, weight, distinctAcceptedMask);
    }

    /** Packs an already projected row without materializing an intermediate Flink RowData. */
    void appendPacked(PackedRow row, int weight) {
        appendPacked(row, weight, 0);
    }

    /** Packs an ordinary aggregate row plus synthetic COUNT(DISTINCT) acceptance bits. */
    void appendPacked(PackedRow row, int weight, int distinctAcceptedMask) {
        appendPacked(row, Long.MIN_VALUE, false, weight, distinctAcceptedMask);
    }

    /** Packs a projected row while adding the Flink-assigned slice end to its dense key. */
    void appendPackedWindow(PackedRow row, long windowEnd, int weight) {
        appendPacked(row, windowEnd, true, weight, 0);
    }

    private void appendPacked(
            PackedRow row,
            long windowEnd,
            boolean windowed,
            int weight,
            int distinctAcceptedMask) {
        long started = System.nanoTime();
        if (weight != 1 && weight != -1) {
            throw new IllegalArgumentException("aggregate row weight must be +1 or -1");
        }
        if (bufferedRows >= capacity) {
            throw new IllegalStateException("dense groupby input exceeded capacity " + capacity);
        }
        if (bufferedRows != 0 && bufferedWindowed != windowed) {
            throw new IllegalStateException("mixed Aggregate and Window rows in one dense batch");
        }
        bufferedWindowed = windowed;

        int group = groupFor(row, windowEnd, windowed);
        writePackedInputAt(row, group, bufferedRows, weight, distinctAcceptedMask);
        windowEndsByRow[bufferedRows] = windowEnd;
        bufferedRows++;
        bufferedPackNanos += System.nanoTime() - started;
    }

    private int groupFor(PackedRow row, long windowEnd, boolean windowed) {
        if (fastLongKey) {
            Object value = row.field(groupingPackedOrdinals[0]);
            return keyMap.idFor(
                    value == null ? null : ((Number) value).longValue(),
                    windowEnd,
                    windowed);
        }
        if (fastLongStringKey) {
            Object longField = row.field(groupingPackedOrdinals[0]);
            Object stringField = row.field(groupingPackedOrdinals[1]);
            StringData stringValue =
                    stringField == null
                            ? null
                            : stringField instanceof StringData
                                    ? (StringData) stringField
                                    : StringData.fromString((String) stringField);
            return longStringKeyMap.idFor(
                    longField == null ? 0L : ((Number) longField).longValue(),
                    longField == null,
                    stringValue,
                    windowEnd,
                    windowed);
        }
        if (fastLongIntLongKey) {
            Object field0 = row.field(groupingPackedOrdinals[0]);
            Object field1 = row.field(groupingPackedOrdinals[1]);
            Object field2 = row.field(groupingPackedOrdinals[2]);
            int nullMask =
                    (field0 == null ? 1 : 0)
                            | (field1 == null ? 2 : 0)
                            | (field2 == null ? 4 : 0);
            return longIntLongKeyMap.idFor(
                    field0 == null ? 0L : ((Number) field0).longValue(),
                    field1 == null ? 0 : ((Number) field1).intValue(),
                    field2 == null ? 0L : ((Number) field2).longValue(),
                    nullMask,
                    windowEnd,
                    windowed);
        }
        Object[] values = new Object[groupingTypes.length];
        for (int index = 0; index < values.length; index++) {
            values[index] = row.field(groupingPackedOrdinals[index]);
        }
        return genericKeyMap.idFor(values, windowEnd, windowed);
    }

    private void writePackedInputAt(
            PackedRow row,
            int group,
            int rowIndex,
            int weight,
            int distinctAcceptedMask) {
        Object metricField =
                metricPackedOrdinal < 0 ? null : row.field(metricPackedOrdinal);
        Object intField = intPackedOrdinal < 0 ? null : row.field(intPackedOrdinal);
        Object sequenceField =
                sequencePackedOrdinal < 0 ? null : row.field(sequencePackedOrdinal);
        Object sentField =
                sentNanosPackedOrdinal < 0 ? null : row.field(sentNanosPackedOrdinal);
        boolean metricValid = metricField != null;
        double metric = metricValid ? ((Number) metricField).doubleValue() : 0.0d;
        boolean intValid = intField != null;
        int intValue = intValid ? ((Number) intField).intValue() : 0;
        boolean sequenceValid = sequenceField != null;
        long sequence = sequenceValid ? ((Number) sequenceField).longValue() : 0L;
        boolean sentValid = sentField != null;
        long sentNanos = sentValid ? ((Number) sentField).longValue() : 0L;
        if (metricValid && !Double.isFinite(metric) && !plan.onlyCounts()) {
            batchNeedsFallback = true;
        }
        int filterBits = 0;
        for (int index = 0; index < filterPackedOrdinals.length; index++) {
            int ordinal = filterPackedOrdinals[index];
            if (ordinal >= 0 && Boolean.TRUE.equals(row.field(ordinal))) {
                filterBits |= 1 << index;
            } else if (GpuAggregatePlan.isDistinctFilterOrdinal(ordinal)
                    && (distinctAcceptedMask
                                    & (1
                                            << GpuAggregatePlan.distinctIndexForFilterOrdinal(
                                                    ordinal)))
                            != 0) {
                filterBits |= 1 << index;
            }
        }
        writeInputRowAt(
                rowIndex,
                group,
                intValue,
                metric,
                sequence,
                sentNanos,
                metricValid,
                intValid,
                sequenceValid,
                sentValid,
                filterBits,
                weight);
    }

    void appendWindow(RowData row, long windowEnd) {
        appendInternal(row, windowEnd, true, 1, 0);
    }

    /** Packs one input row into several HOP windows while decoding its payload only once. */
    void appendWindows(RowData row, long[] windowEnds, int offset, int count) {
        long started = System.nanoTime();
        if (count <= 0 || offset < 0 || offset + count > windowEnds.length) {
            throw new IllegalArgumentException("invalid HOP window slice");
        }
        if (bufferedRows + count > capacity) {
            throw new IllegalStateException("dense groupby input exceeded capacity " + capacity);
        }
        if (bufferedRows != 0 && !bufferedWindowed) {
            throw new IllegalStateException("mixed Aggregate and Window rows in one dense batch");
        }
        bufferedWindowed = true;

        boolean longKeyNull = false;
        long longKey = 0L;
        boolean longIsNull = false;
        long longValue = 0L;
        StringData stringValue = null;
        int longIntLongNullMask = 0;
        long longIntLong0 = 0L;
        int longIntLong1 = 0;
        long longIntLong2 = 0L;
        Object[] genericValues = null;
        if (fastLongKey) {
            int ordinal = groupingInputOrdinals[0];
            longKeyNull = row.isNullAt(ordinal);
            longKey = longKeyNull ? 0L : row.getLong(ordinal);
        } else if (fastLongStringKey) {
            int longOrdinal = groupingInputOrdinals[0];
            int stringOrdinal = groupingInputOrdinals[1];
            longIsNull = row.isNullAt(longOrdinal);
            longValue = longIsNull ? 0L : row.getLong(longOrdinal);
            stringValue = row.isNullAt(stringOrdinal) ? null : row.getString(stringOrdinal);
        } else if (fastLongIntLongKey) {
            int ordinal0 = groupingInputOrdinals[0];
            int ordinal1 = groupingInputOrdinals[1];
            int ordinal2 = groupingInputOrdinals[2];
            if (row.isNullAt(ordinal0)) {
                longIntLongNullMask |= 1;
            } else {
                longIntLong0 = row.getLong(ordinal0);
            }
            if (row.isNullAt(ordinal1)) {
                longIntLongNullMask |= 2;
            } else {
                longIntLong1 = row.getInt(ordinal1);
            }
            if (row.isNullAt(ordinal2)) {
                longIntLongNullMask |= 4;
            } else {
                longIntLong2 = row.getLong(ordinal2);
            }
        } else {
            genericValues = new Object[groupingTypes.length];
            for (int i = 0; i < genericValues.length; i++) {
                genericValues[i] =
                        groupingTypes[i].copyFrom(row, groupingInputOrdinals[i]);
            }
        }

        boolean metricValid = metricInputOrdinal >= 0 && !row.isNullAt(metricInputOrdinal);
        double metric =
                metricValid ? floatingValue(row, metricInputOrdinal, metricInputType) : 0.0;
        boolean intValid = intInputOrdinal >= 0 && !row.isNullAt(intInputOrdinal);
        int intValue = intValid ? narrowIntegerValue(row, intInputOrdinal, intInputType) : 0;
        boolean sequenceValid =
                sequenceInputOrdinal >= 0 && !row.isNullAt(sequenceInputOrdinal);
        long sequence =
                sequenceValid ? longLaneValue(row, sequenceInputOrdinal, sequenceInputType) : 0L;
        boolean sentValid =
                sentNanosInputOrdinal >= 0 && !row.isNullAt(sentNanosInputOrdinal);
        long sentNanos =
                sentValid ? longLaneValue(row, sentNanosInputOrdinal, sentNanosInputType) : 0L;
        if (metricValid && !Double.isFinite(metric)) {
            batchNeedsFallback = true;
        }
        int filterBits = 0;
        for (int filter = 0; filter < filterInputOrdinals.length; filter++) {
            int ordinal = filterInputOrdinals[filter];
            if (ordinal >= 0 && !row.isNullAt(ordinal) && row.getBoolean(ordinal)) {
                filterBits |= 1 << filter;
            }
        }

        for (int index = 0; index < count; index++) {
            long windowEnd = windowEnds[offset + index];
            int group;
            if (fastLongKey) {
                group = keyMap.idFor(longKey, longKeyNull, windowEnd, true);
            } else if (fastLongStringKey) {
                group =
                        longStringKeyMap.idFor(
                                longValue,
                                longIsNull,
                                stringValue,
                                windowEnd,
                                true);
            } else if (fastLongIntLongKey) {
                group =
                        longIntLongKeyMap.idFor(
                                longIntLong0,
                                longIntLong1,
                                longIntLong2,
                                longIntLongNullMask,
                                windowEnd,
                                true);
            } else {
                group = genericKeyMap.idFor(genericValues, windowEnd, true);
            }
            int packedRow = bufferedRows;
            writeInputRow(
                    group,
                    intValue,
                    metric,
                    sequence,
                    sentNanos,
                    metricValid,
                    intValid,
                    sequenceValid,
                    sentValid,
                    filterBits,
                    windowEnd,
                    1);
            captureStringLanes(row, packedRow);
        }
        bufferedPackNanos += System.nanoTime() - started;
    }

    int bufferedRows() {
        return bufferedRows;
    }

    List<AggregatePartial> executeBuffered() {
        if (batchNeedsFallback) {
            return null;
        }
        int rows = bufferedRows;
        int groups = keyMapSize();
        executeNative(rows, groups);
        return decodeBufferedAggregatePartials(rows, groups);
    }

    /**
     * Finalizes a flexible-layout, single-BIGINT-key batch straight into fixed-width Flink rows.
     * This avoids one AggregatePartial plus two Object arrays per output group.
     */
    int executeBufferedMaterializedLongKeyRows(PackedBinaryRowBatch outputBatch) {
        if (!fastLongKey || !flexible) {
            throw new IllegalStateException(
                    "direct materialized Aggregate requires a flexible single-BIGINT-key plan");
        }
        if (batchNeedsFallback) {
            return -1;
        }
        int rows = bufferedRows;
        int groups = keyMapSize();
        if (outputBatch == null
                || outputBatch.rowCount() != groups
                || outputBatch.arity() != groupingTypes.length + plan.aggregates().length
                || outputBatch.arity() > 56) {
            throw new IllegalArgumentException("direct Aggregate output has the wrong shape");
        }
        if (groupingTypes.length != 1 || groupingTypes[0] != GpuType.INT64) {
            throw new IllegalStateException("direct Aggregate key is not one BIGINT");
        }
        for (GpuType type : plan.outputTypes()) {
            if (!type.supportsInPlaceBinaryRowWrite()) {
                throw new IllegalStateException(
                        "direct Aggregate output cannot hold " + type);
            }
        }
        executeNative(rows, groups);
        long decodeStarted = System.nanoTime();
        for (int group = 0; group < groups; group++) {
            long header = 0L;
            if (keyMap.keyNull(group)) {
                header |= 1L << 8;
            } else {
                outputBatch.setFixedSlotBitsUnchecked(group, 0, keyMap.keyBits(group));
            }
            header = writeFlexibleMaterializedValues(group, outputBatch, group, 1, header);
            outputBatch.setFixedHeaderWordUnchecked(group, header);
        }
        recordMetrics(rows, groups, bufferedPackNanos, System.nanoTime() - decodeStarted);
        return groups;
    }

    private long writeFlexibleMaterializedValues(
            int group,
            PackedBinaryRowBatch target,
            int targetRow,
            int targetOffset,
            long header) {
        int groupBase = outputBase(group);
        GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
        for (int index = 0; index < specs.length; index++) {
            GpuAggregatePlan.AggregateSpec spec = specs[index];
            int resultBase = groupBase + (index + 1) * FLEXIBLE_RESULT_BYTES;
            long count = output.getLong(resultBase);
            int ordinal = targetOffset + index;
            boolean isNull = false;
            long bits;
            switch (spec.kind) {
                case COUNT:
                    bits = count;
                    break;
                case SUM:
                case SUM0:
                    isNull =
                            count == 0L
                                    && !plan.consumesRetractions()
                                    && spec.kind != GpuAggregatePlan.AggregateKind.SUM0;
                    if (count == 0L) {
                        bits = rawIntegralOrFloating(spec, 0L, 0.0d);
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        0L,
                                        output.getDouble(resultBase + Long.BYTES));
                    } else {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        output.getLong(resultBase + Long.BYTES),
                                        0.0d);
                    }
                    break;
                case AVG:
                    isNull = count == 0L && !plan.consumesRetractions();
                    if (isNull) {
                        bits = 0L;
                    } else if (isIntegral(spec.argumentType)) {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        output.getLong(resultBase + Long.BYTES) / count,
                                        0.0d);
                    } else {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        0L,
                                        output.getDouble(resultBase + Long.BYTES) / count);
                    }
                    break;
                case MIN:
                case MAX:
                    isNull = count == 0L;
                    if (isNull) {
                        bits = 0L;
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        0L,
                                        output.getDouble(resultBase + Long.BYTES));
                    } else {
                        bits =
                                rawIntegralOrFloating(
                                        spec,
                                        output.getLong(resultBase + Long.BYTES),
                                        0.0d);
                    }
                    break;
                default:
                    throw new IllegalStateException(
                            "Unhandled flexible aggregate kind " + spec.kind);
            }
            if (isNull) {
                header |= 1L << (ordinal + 8);
            } else {
                target.setFixedSlotBitsUnchecked(targetRow, ordinal, bits);
            }
        }
        return header;
    }

    private static long rawIntegralOrFloating(
            GpuAggregatePlan.AggregateSpec spec, long integral, double floating) {
        switch (spec.resultType) {
            case INT8:
                return (byte) integral;
            case INT16:
                return (short) integral;
            case INT32:
                return (int) integral;
            case INT64:
            case TIMESTAMP3:
                return integral;
            case FLOAT32:
                return Integer.toUnsignedLong(Float.floatToRawIntBits((float) floating));
            case FLOAT64:
                return Double.doubleToRawLongBits(floating);
            default:
                throw new IllegalStateException(
                        "direct Aggregate output cannot write " + spec.resultType);
        }
    }

    private void writeFlexibleMaterializedValues(
            int group, BinaryRowData target, int targetOffset) {
        int groupBase = outputBase(group);
        GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
        for (int index = 0; index < specs.length; index++) {
            GpuAggregatePlan.AggregateSpec spec = specs[index];
            int resultBase = groupBase + (index + 1) * FLEXIBLE_RESULT_BYTES;
            long count = output.getLong(resultBase);
            int ordinal = targetOffset + index;
            switch (spec.kind) {
                case COUNT:
                    target.setLong(ordinal, count);
                    break;
                case SUM:
                case SUM0:
                    if (count == 0L && !plan.consumesRetractions()) {
                        if (spec.kind == GpuAggregatePlan.AggregateKind.SUM0) {
                            if (spec.argumentType == GpuType.FLOAT32
                                    || spec.argumentType == GpuType.FLOAT64) {
                                writeFloatingResult(target, ordinal, spec.resultType, 0.0);
                            } else {
                                writeIntegralResult(target, ordinal, spec.resultType, 0L);
                            }
                        } else {
                            target.setNullAt(ordinal);
                        }
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        writeFloatingResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getDouble(resultBase + Long.BYTES));
                    } else {
                        writeIntegralResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getLong(resultBase + Long.BYTES));
                    }
                    break;
                case AVG:
                    if (count == 0L && !plan.consumesRetractions()) {
                        target.setNullAt(ordinal);
                    } else if (isIntegral(spec.argumentType)) {
                        writeIntegralResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getLong(resultBase + Long.BYTES) / count);
                    } else {
                        writeFloatingResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getDouble(resultBase + Long.BYTES) / count);
                    }
                    break;
                case MIN:
                case MAX:
                    if (count == 0L) {
                        target.setNullAt(ordinal);
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        writeFloatingResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getDouble(resultBase + Long.BYTES));
                    } else {
                        writeIntegralResult(
                                target,
                                ordinal,
                                spec.resultType,
                                output.getLong(resultBase + Long.BYTES));
                    }
                    break;
                default:
                    throw new IllegalStateException(
                            "Unhandled flexible aggregate kind " + spec.kind);
            }
        }
    }

    private static void writeIntegralResult(
            BinaryRowData target, int ordinal, GpuType type, long value) {
        switch (type) {
            case INT8:
                target.setByte(ordinal, (byte) value);
                break;
            case INT16:
                target.setShort(ordinal, (short) value);
                break;
            case INT32:
                target.setInt(ordinal, (int) value);
                break;
            case INT64:
            case TIMESTAMP3:
                target.setLong(ordinal, value);
                break;
            default:
                throw new IllegalStateException(
                        "direct Aggregate integral output cannot write " + type);
        }
    }

    private static void writeFloatingResult(
            BinaryRowData target, int ordinal, GpuType type, double value) {
        if (type == GpuType.FLOAT32) {
            target.setFloat(ordinal, (float) value);
        } else if (type == GpuType.FLOAT64) {
            target.setDouble(ordinal, value);
        } else {
            throw new IllegalStateException(
                    "direct Aggregate floating output cannot write " + type);
        }
    }

    private List<AggregatePartial> decodeBufferedAggregatePartials(int rows, int groups) {
        long decodeStarted = System.nanoTime();
        List<AggregatePartial> partials = new ArrayList<>(groups);
        for (int group = 0; group < groups; group++) {
            long inputRows = rowCount(group);
            partials.add(
                    new AggregatePartial(
                            groupValues(group),
                            inputRows,
                            aggregateValues(group)));
        }
        recordMetrics(rows, groups, bufferedPackNanos, System.nanoTime() - decodeStarted);
        return partials;
    }

    List<WindowAggregatePartial> executeBufferedWindows() {
        if (batchNeedsFallback) {
            return null;
        }
        int rows = bufferedRows;
        int groups = keyMapSize();
        executeNative(rows, groups);
        return decodeBufferedWindowPartials(rows, groups);
    }

    /**
     * Finalizes a one-shot BIGINT-keyed Window batch directly into its downstream consumer.
     * Unlike {@link #executeBufferedWindows()}, this path does not build an intermediate list of
     * mergeable partials and is intended for bounded batches that will not be merged again.
     */
    boolean executeBufferedMaterializedLongKeyWindows(
            MaterializedLongKeyWindowConsumer consumer) {
        if (!fastLongKey || flexible) {
            throw new IllegalStateException(
                    "materialized Window output requires the fixed layout and exactly one BIGINT grouping key");
        }
        if (batchNeedsFallback) {
            return false;
        }
        int rows = bufferedRows;
        int groups = keyMapSize();
        executeNative(rows, groups);
        return decodeBufferedMaterializedLongKeyWindows(rows, groups, consumer);
    }

    private boolean decodeBufferedMaterializedLongKeyWindows(
            int rows, int groups, MaterializedLongKeyWindowConsumer consumer) {
        long decodeStarted = System.nanoTime();
        MaterializedAggregateRow values = new MaterializedAggregateRow();
        for (int group = 0; group < groups; group++) {
            long inputRows = rowCount(group);
            values.group = group;
            consumer.accept(
                    keyMap.keyBits(group),
                    keyMap.keyNull(group),
                    windowEnd(group),
                    inputRows,
                    values);
        }
        recordMetrics(rows, groups, bufferedPackNanos, System.nanoTime() - decodeStarted);
        return true;
    }

    private List<WindowAggregatePartial> decodeBufferedWindowPartials(int rows, int groups) {
        long decodeStarted = System.nanoTime();
        List<WindowAggregatePartial> partials = new ArrayList<>(groups);
        for (int group = 0; group < groups; group++) {
            long inputRows = rowCount(group);
            partials.add(
                    new WindowAggregatePartial(
                            windowEnd(group),
                            groupValues(group),
                            inputRows,
                            aggregateValues(group)));
        }
        recordMetrics(rows, groups, bufferedPackNanos, System.nanoTime() - decodeStarted);
        return partials;
    }

    @FunctionalInterface
    interface MaterializedLongKeyWindowConsumer {
        void accept(
                long key,
                boolean keyNull,
                long windowEnd,
                long inputRows,
                MaterializedAggregateRow aggregateValues);
    }

    /** Allocation-free typed view over one finalized native aggregate output row. */
    final class MaterializedAggregateRow {
        private int group;

        private MaterializedAggregateRow() {}

        boolean isNullAt(int aggregateIndex) {
            OutputLane lane = outputLanes[aggregateIndex];
            int base = outputBase(group);
            switch (lane) {
                case ROW_COUNT:
                case FILTER_COUNT0:
                case FILTER_COUNT1:
                case FILTER_COUNT2:
                case METRIC_COUNT:
                case INT_COUNT:
                case LONG0_COUNT:
                case LONG1_COUNT:
                    return false;
                case METRIC_SUM:
                    return output.getLong(base + OUTPUT_METRIC_COUNT_OFFSET) == 0L
                            && !plan.consumesRetractions()
                            && plan.aggregates()[aggregateIndex].normalizePartial(null) == null;
                case METRIC_AVG:
                case MIN_METRIC:
                case MAX_METRIC:
                    return output.getLong(base + OUTPUT_METRIC_COUNT_OFFSET) == 0L;
                case SUM_INT:
                    return output.getLong(base + OUTPUT_INT_COUNT_OFFSET) == 0L
                            && !plan.consumesRetractions()
                            && plan.aggregates()[aggregateIndex].normalizePartial(null) == null;
                case MIN_INT:
                case MAX_INT:
                    return output.getLong(base + OUTPUT_INT_COUNT_OFFSET) == 0L;
                case SUM_LONG0:
                    return output.getLong(base + OUTPUT_LONG0_COUNT_OFFSET) == 0L
                            && !plan.consumesRetractions()
                            && plan.aggregates()[aggregateIndex].normalizePartial(null) == null;
                case MIN_LONG0:
                case MAX_LONG0:
                    return output.getLong(base + OUTPUT_LONG0_COUNT_OFFSET) == 0L;
                case SUM_LONG1:
                    return output.getLong(base + OUTPUT_LONG1_COUNT_OFFSET) == 0L
                            && !plan.consumesRetractions()
                            && plan.aggregates()[aggregateIndex].normalizePartial(null) == null;
                case MIN_LONG1:
                case MAX_LONG1:
                    return output.getLong(base + OUTPUT_LONG1_COUNT_OFFSET) == 0L;
                default:
                    throw new IllegalStateException("Unhandled output lane " + lane);
            }
        }

        long getLong(int aggregateIndex) {
            OutputLane lane = outputLanes[aggregateIndex];
            int base = outputBase(group);
            switch (lane) {
                case ROW_COUNT:
                    return output.getLong(base + OUTPUT_ROW_COUNT_OFFSET);
                case FILTER_COUNT0:
                    return output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[0]);
                case FILTER_COUNT1:
                    return output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[1]);
                case FILTER_COUNT2:
                    return output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[2]);
                case METRIC_COUNT:
                    return output.getLong(base + OUTPUT_METRIC_COUNT_OFFSET);
                case INT_COUNT:
                    return output.getLong(base + OUTPUT_INT_COUNT_OFFSET);
                case SUM_INT:
                    return output.getLong(base + OUTPUT_INT_SUM_OFFSET);
                case MIN_INT:
                    return output.getInt(base + OUTPUT_MIN_INT_OFFSET);
                case MAX_INT:
                    return output.getInt(base + OUTPUT_MAX_INT_OFFSET);
                case LONG0_COUNT:
                    return output.getLong(base + OUTPUT_LONG0_COUNT_OFFSET);
                case SUM_LONG0:
                    return output.getLong(base + OUTPUT_LONG0_SUM_OFFSET);
                case MIN_LONG0:
                    return output.getLong(base + OUTPUT_MIN_LONG0_OFFSET);
                case MAX_LONG0:
                    return output.getLong(base + OUTPUT_MAX_LONG0_OFFSET);
                case LONG1_COUNT:
                    return output.getLong(base + OUTPUT_LONG1_COUNT_OFFSET);
                case SUM_LONG1:
                    return output.getLong(base + OUTPUT_LONG1_SUM_OFFSET);
                case MIN_LONG1:
                    return output.getLong(base + OUTPUT_MIN_LONG1_OFFSET);
                case MAX_LONG1:
                    return output.getLong(base + OUTPUT_MAX_LONG1_OFFSET);
                default:
                    throw new IllegalArgumentException(lane + " is not an integral output lane");
            }
        }

        int getInt(int aggregateIndex) {
            return Math.toIntExact(getLong(aggregateIndex));
        }

        double getDouble(int aggregateIndex) {
            OutputLane lane = outputLanes[aggregateIndex];
            int base = outputBase(group);
            switch (lane) {
                case METRIC_SUM:
                    return output.getDouble(base + OUTPUT_METRIC_SUM_OFFSET);
                case METRIC_AVG:
                    return output.getDouble(base + OUTPUT_METRIC_SUM_OFFSET)
                            / output.getLong(base + OUTPUT_METRIC_COUNT_OFFSET);
                case MIN_METRIC:
                    return output.getDouble(base + OUTPUT_MIN_METRIC_OFFSET);
                case MAX_METRIC:
                    return output.getDouble(base + OUTPUT_MAX_METRIC_OFFSET);
                default:
                    throw new IllegalArgumentException(lane + " is not a floating output lane");
            }
        }
    }

    List<AggregateBufferedRow> fallbackRows() {
        List<AggregateBufferedRow> rows = new ArrayList<>(bufferedRows);
        for (int row = 0; row < bufferedRows; row++) {
            rows.add(rebuildRow(row));
        }
        return rows;
    }

    List<WindowBufferedRow> fallbackWindowRows() {
        List<WindowBufferedRow> rows = new ArrayList<>(bufferedRows);
        for (int row = 0; row < bufferedRows; row++) {
            rows.add(new WindowBufferedRow(rebuildRow(row), windowEndsByRow[row]));
        }
        return rows;
    }

    void resetBatch() {
        bufferedRows = 0;
        bufferedWindowed = false;
        batchNeedsFallback = false;
        bufferedPackNanos = 0L;
        long0RankedStrings = null;
        long1RankedStrings = null;
        if (fastLongKey) {
            keyMap.reset();
        } else if (fastLongStringKey) {
            longStringKeyMap.reset();
        } else if (fastLongIntLongKey) {
            longIntLongKeyMap.reset();
        } else {
            genericKeyMap.reset();
        }
    }

    private void appendInternal(
            RowData row,
            long windowEnd,
            boolean windowed,
            int weight,
            int distinctAcceptedMask) {
        long started = System.nanoTime();
        appendInternalUntimed(row, windowEnd, windowed, weight, distinctAcceptedMask);
        bufferedPackNanos += System.nanoTime() - started;
    }

    private void appendInternalUntimed(
            RowData row,
            long windowEnd,
            boolean windowed,
            int weight,
            int distinctAcceptedMask) {
        if (bufferedRows >= capacity) {
            throw new IllegalStateException("dense groupby input exceeded capacity " + capacity);
        }
        if (bufferedRows != 0 && bufferedWindowed != windowed) {
            throw new IllegalStateException("mixed Aggregate and Window rows in one dense batch");
        }
        bufferedWindowed = windowed;
        int group;
        if (fastLongKey) {
            int groupInputOrdinal = groupingInputOrdinals[0];
            boolean groupNull = row.isNullAt(groupInputOrdinal);
            long groupValue = groupNull ? 0L : row.getLong(groupInputOrdinal);
            group = keyMap.idFor(groupValue, groupNull, windowEnd, windowed);
        } else if (fastLongStringKey) {
            int longOrdinal = groupingInputOrdinals[0];
            int stringOrdinal = groupingInputOrdinals[1];
            boolean longIsNull = row.isNullAt(longOrdinal);
            long longValue = longIsNull ? 0L : row.getLong(longOrdinal);
            StringData stringValue =
                    row.isNullAt(stringOrdinal) ? null : row.getString(stringOrdinal);
            group =
                    longStringKeyMap.idFor(
                            longValue,
                            longIsNull,
                            stringValue,
                            windowEnd,
                            windowed);
        } else if (fastLongIntLongKey) {
            int ordinal0 = groupingInputOrdinals[0];
            int ordinal1 = groupingInputOrdinals[1];
            int ordinal2 = groupingInputOrdinals[2];
            int nullMask = 0;
            long value0 = 0L;
            int value1 = 0;
            long value2 = 0L;
            if (row.isNullAt(ordinal0)) {
                nullMask |= 1;
            } else {
                value0 = row.getLong(ordinal0);
            }
            if (row.isNullAt(ordinal1)) {
                nullMask |= 2;
            } else {
                value1 = row.getInt(ordinal1);
            }
            if (row.isNullAt(ordinal2)) {
                nullMask |= 4;
            } else {
                value2 = row.getLong(ordinal2);
            }
            group =
                    longIntLongKeyMap.idFor(
                            value0,
                            value1,
                            value2,
                            nullMask,
                            windowEnd,
                            windowed);
        } else {
            Object[] values = new Object[groupingTypes.length];
            for (int i = 0; i < values.length; i++) {
                values[i] = groupingTypes[i].copyFrom(row, groupingInputOrdinals[i]);
            }
            group = genericKeyMap.idFor(values, windowEnd, windowed);
        }
        boolean metricValid =
                metricInputOrdinal >= 0 && !row.isNullAt(metricInputOrdinal);
        double metric =
                metricValid ? floatingValue(row, metricInputOrdinal, metricInputType) : 0.0;
        boolean intValid = intInputOrdinal >= 0 && !row.isNullAt(intInputOrdinal);
        int intValue = intValid ? narrowIntegerValue(row, intInputOrdinal, intInputType) : 0;
        boolean sequenceValid =
                sequenceInputOrdinal >= 0 && !row.isNullAt(sequenceInputOrdinal);
        long sequence =
                sequenceValid ? longLaneValue(row, sequenceInputOrdinal, sequenceInputType) : 0L;
        boolean sentValid =
                sentNanosInputOrdinal >= 0 && !row.isNullAt(sentNanosInputOrdinal);
        long sentNanos =
                sentValid ? longLaneValue(row, sentNanosInputOrdinal, sentNanosInputType) : 0L;
        if (metricValid && !Double.isFinite(metric) && !plan.onlyCounts()) {
            batchNeedsFallback = true;
        }

        int filterBits = 0;
        for (int i = 0; i < filterInputOrdinals.length; i++) {
            int ordinal = filterInputOrdinals[i];
            int packedOrdinal = filterPackedOrdinals[i];
            if (ordinal >= 0 && !row.isNullAt(ordinal) && row.getBoolean(ordinal)) {
                filterBits |= 1 << i;
            } else if (GpuAggregatePlan.isDistinctFilterOrdinal(packedOrdinal)
                    && (distinctAcceptedMask
                                    & (1
                                            << GpuAggregatePlan.distinctIndexForFilterOrdinal(
                                                    packedOrdinal)))
                            != 0) {
                filterBits |= 1 << i;
            }
        }
        int packedRow = bufferedRows;
        writeInputRow(
                group,
                intValue,
                metric,
                sequence,
                sentNanos,
                metricValid,
                intValid,
                sequenceValid,
                sentValid,
                filterBits,
                windowEnd,
                weight);
        captureStringLanes(row, packedRow);
    }

    private void captureStringLanes(RowData row, int packedRow) {
        if (long0StringValues != null) {
            long0StringValues[packedRow] =
                    row.isNullAt(sequenceInputOrdinal)
                            ? null
                            : row.getString(sequenceInputOrdinal).toString();
        }
        if (long1StringValues != null) {
            long1StringValues[packedRow] =
                    row.isNullAt(sentNanosInputOrdinal)
                            ? null
                            : row.getString(sentNanosInputOrdinal).toString();
        }
    }

    private void writeInputRow(
            int group,
            int intValue,
            double metric,
            long sequence,
            long sentNanos,
            boolean metricValid,
            boolean intValid,
            boolean sequenceValid,
            boolean sentValid,
            int filterBits,
            long windowEnd,
            int weight) {
        writeInputRowAt(
                bufferedRows,
                group,
                intValue,
                metric,
                sequence,
                sentNanos,
                metricValid,
                intValid,
                sequenceValid,
                sentValid,
                filterBits,
                weight);
        windowEndsByRow[bufferedRows] = windowEnd;
        bufferedRows++;
    }

    private static double floatingValue(RowData row, int ordinal, GpuType type) {
        if (type == GpuType.FLOAT32) {
            return row.getFloat(ordinal);
        }
        if (type == GpuType.FLOAT64) {
            return row.getDouble(ordinal);
        }
        throw new IllegalArgumentException("aggregate floating lane cannot read " + type);
    }

    private static int narrowIntegerValue(RowData row, int ordinal, GpuType type) {
        switch (type) {
            case INT8:
                return row.getByte(ordinal);
            case INT16:
                return row.getShort(ordinal);
            case INT32:
                return row.getInt(ordinal);
            default:
                throw new IllegalArgumentException("aggregate INT lane cannot read " + type);
        }
    }

    private static long longLaneValue(RowData row, int ordinal, GpuType type) {
        switch (type) {
            case INT64:
                return row.getLong(ordinal);
            case TIMESTAMP3:
                return row.getTimestamp(ordinal, 3).getMillisecond();
            case BOOLEAN:
            case STRING:
                // COUNT only needs validity. The value bits are deliberately ignored.
                return 0L;
            default:
                throw new IllegalArgumentException("aggregate BIGINT lane cannot read " + type);
        }
    }

    private static Object decodedLaneValue(GpuType type, long bits) {
        switch (type) {
            case BOOLEAN:
                return Boolean.TRUE;
            case INT8:
                return (byte) bits;
            case INT16:
                return (short) bits;
            case INT32:
                return (int) bits;
            case INT64:
                return bits;
            case FLOAT32:
                return (float) Double.longBitsToDouble(bits);
            case FLOAT64:
                return Double.longBitsToDouble(bits);
            case STRING:
                return "";
            case TIMESTAMP3:
                return TimestampData.fromEpochMillis(bits);
            default:
                throw new IllegalArgumentException("aggregate lane cannot decode " + type);
        }
    }

    private void writeInputRowAt(
            int rowIndex,
            int group,
            int intValue,
            double metric,
            long sequence,
            long sentNanos,
            boolean metricValid,
            boolean intValid,
            boolean sequenceValid,
            boolean sentValid,
            int filterBits,
            int weight) {
        int base = rowIndex * INPUT_ROW_BYTES;
        input.putInt(base + INPUT_GROUP_OFFSET, group);
        input.putInt(base + INPUT_INT_OFFSET, intValue);
        input.putDouble(base + INPUT_METRIC_OFFSET, metric);
        input.putLong(base + INPUT_SEQUENCE_OFFSET, sequence);
        input.putLong(base + INPUT_SENT_NANOS_OFFSET, sentNanos);
        input.putInt(base + INPUT_METRIC_VALID_OFFSET, metricValid ? 1 : 0);
        input.putInt(base + INPUT_INT_VALID_OFFSET, intValid ? 1 : 0);
        input.putInt(base + INPUT_SEQUENCE_VALID_OFFSET, sequenceValid ? 1 : 0);
        input.putInt(base + INPUT_SENT_NANOS_VALID_OFFSET, sentValid ? 1 : 0);
        for (int i = 0; i < filterInputOrdinals.length; i++) {
            input.putInt(base + INPUT_FILTER_OFFSETS[i], (filterBits & (1 << i)) == 0 ? 0 : 1);
        }
        input.putInt(base + INPUT_WEIGHT_OFFSET, weight);
    }

    private AggregateBufferedRow rebuildRow(int row) {
        int base = row * INPUT_ROW_BYTES;
        int group = input.getInt(base + INPUT_GROUP_OFFSET);
        return rebuildRow(row, group);
    }

    private AggregateBufferedRow rebuildRow(int row, int group) {
        int base = row * INPUT_ROW_BYTES;
        Object[] fields = new Object[plan.inputTypes().length];
        Object[] groupValues = groupValues(group);
        for (int i = 0; i < groupValues.length; i++) {
            fields[groupingPackedOrdinals[i]] = groupValues[i];
        }
        if (metricPackedOrdinal >= 0) {
            fields[metricPackedOrdinal] =
                    input.getInt(base + INPUT_METRIC_VALID_OFFSET) == 0
                            ? null
                            : decodedLaneValue(
                                    metricInputType,
                                    Double.doubleToRawLongBits(
                                            input.getDouble(base + INPUT_METRIC_OFFSET)));
        }
        if (intPackedOrdinal >= 0) {
            fields[intPackedOrdinal] =
                    input.getInt(base + INPUT_INT_VALID_OFFSET) == 0
                            ? null
                            : decodedLaneValue(
                                    intInputType, input.getInt(base + INPUT_INT_OFFSET));
        }
        if (sequencePackedOrdinal >= 0) {
            fields[sequencePackedOrdinal] =
                    input.getInt(base + INPUT_SEQUENCE_VALID_OFFSET) == 0
                            ? null
                            : sequenceInputType == GpuType.STRING
                                    ? long0StringValues[row]
                                    : decodedLaneValue(
                                            sequenceInputType,
                                            input.getLong(base + INPUT_SEQUENCE_OFFSET));
        }
        if (sentNanosPackedOrdinal >= 0) {
            fields[sentNanosPackedOrdinal] =
                    input.getInt(base + INPUT_SENT_NANOS_VALID_OFFSET) == 0
                            ? null
                            : sentNanosInputType == GpuType.STRING
                                    ? long1StringValues[row]
                                    : decodedLaneValue(
                                            sentNanosInputType,
                                            input.getLong(base + INPUT_SENT_NANOS_OFFSET));
        }
        for (int i = 0; i < filterPackedOrdinals.length; i++) {
            if (filterPackedOrdinals[i] >= 0) {
                fields[filterPackedOrdinals[i]] =
                        input.getInt(base + INPUT_FILTER_OFFSETS[i]) != 0;
            }
        }
        int distinctAcceptedMask = 0;
        for (int i = 0; i < filterPackedOrdinals.length; i++) {
            int packedOrdinal = filterPackedOrdinals[i];
            if (GpuAggregatePlan.isDistinctFilterOrdinal(packedOrdinal)
                    && input.getInt(base + INPUT_FILTER_OFFSETS[i]) != 0) {
                distinctAcceptedMask |=
                        1
                                << GpuAggregatePlan.distinctIndexForFilterOrdinal(
                                        packedOrdinal);
            }
        }
        return AggregateBufferedRow.fromPacked(
                fields,
                input.getInt(base + INPUT_WEIGHT_OFFSET),
                distinctAcceptedMask);
    }

    private int keyMapSize() {
        if (fastLongKey) {
            return keyMap.size();
        }
        if (fastLongStringKey) {
            return longStringKeyMap.size();
        }
        return fastLongIntLongKey ? longIntLongKeyMap.size() : genericKeyMap.size();
    }

    private Object[] groupValues(int group) {
        if (fastLongKey) {
            return new Object[] {keyMap.keyValue(group)};
        }
        if (fastLongStringKey) {
            return longStringKeyMap.keyValues(group);
        }
        return fastLongIntLongKey
                ? longIntLongKeyMap.keyValues(group)
                : genericKeyMap.keyValues(group);
    }

    private long windowEnd(int group) {
        if (fastLongKey) {
            return keyMap.windowEnd(group);
        }
        if (fastLongStringKey) {
            return longStringKeyMap.windowEnd(group);
        }
        return fastLongIntLongKey
                ? longIntLongKeyMap.windowEnd(group)
                : genericKeyMap.windowEnd(group);
    }

    private long rowCount(int group) {
        return output.getLong(outputBase(group) + OUTPUT_ROW_COUNT_OFFSET);
    }

    private Object[] aggregateValues(int group) {
        if (flexible) {
            return flexibleAggregateValues(group);
        }
        int base = outputBase(group);
        long metricCount = output.getLong(base + OUTPUT_METRIC_COUNT_OFFSET);
        long intCount = output.getLong(base + OUTPUT_INT_COUNT_OFFSET);
        long long0Count = output.getLong(base + OUTPUT_LONG0_COUNT_OFFSET);
        long long1Count = output.getLong(base + OUTPUT_LONG1_COUNT_OFFSET);
        Object[] values = new Object[outputLanes.length];
        GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
        for (int i = 0; i < outputLanes.length; i++) {
            switch (outputLanes[i]) {
                case ROW_COUNT:
                    values[i] = output.getLong(base + OUTPUT_ROW_COUNT_OFFSET);
                    break;
                case FILTER_COUNT0:
                    values[i] = output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[0]);
                    break;
                case FILTER_COUNT1:
                    values[i] = output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[1]);
                    break;
                case FILTER_COUNT2:
                    values[i] = output.getLong(base + OUTPUT_FILTER_COUNT_OFFSETS[2]);
                    break;
                case METRIC_COUNT:
                    values[i] = metricCount;
                    break;
                case METRIC_SUM:
                    values[i] =
                            metricCount == 0 && !plan.consumesRetractions()
                                    ? specs[i].normalizePartial(null)
                                    : output.getDouble(base + OUTPUT_METRIC_SUM_OFFSET);
                    break;
                case METRIC_AVG:
                    values[i] =
                            metricCount == 0 && !plan.consumesRetractions()
                                    ? null
                                    : new AverageAccumulator(
                                            output.getDouble(base + OUTPUT_METRIC_SUM_OFFSET),
                                            metricCount);
                    break;
                case MIN_METRIC:
                    values[i] =
                            metricCount == 0
                                    ? null
                                    : output.getDouble(base + OUTPUT_MIN_METRIC_OFFSET);
                    break;
                case MAX_METRIC:
                    values[i] =
                            metricCount == 0
                                    ? null
                                    : output.getDouble(base + OUTPUT_MAX_METRIC_OFFSET);
                    break;
                case INT_COUNT:
                    values[i] = intCount;
                    break;
                case SUM_INT:
                    values[i] =
                            intCount == 0 && !plan.consumesRetractions()
                                    ? specs[i].normalizePartial(null)
                                    : specs[i].normalizePartial(
                                            output.getLong(base + OUTPUT_INT_SUM_OFFSET));
                    break;
                case MIN_INT:
                    values[i] =
                            intCount == 0 ? null : output.getInt(base + OUTPUT_MIN_INT_OFFSET);
                    break;
                case MAX_INT:
                    values[i] =
                            intCount == 0 ? null : output.getInt(base + OUTPUT_MAX_INT_OFFSET);
                    break;
                case LONG0_COUNT:
                    values[i] = long0Count;
                    break;
                case SUM_LONG0:
                    values[i] =
                            long0Count == 0 && !plan.consumesRetractions()
                                    ? specs[i].normalizePartial(null)
                                    : output.getLong(base + OUTPUT_LONG0_SUM_OFFSET);
                    break;
                case MIN_LONG0:
                    values[i] =
                            long0Count == 0
                                    ? null
                                    : output.getLong(base + OUTPUT_MIN_LONG0_OFFSET);
                    break;
                case MAX_LONG0:
                    values[i] =
                            long0Count == 0 ? null : output.getLong(base + OUTPUT_MAX_LONG0_OFFSET);
                    break;
                case LONG1_COUNT:
                    values[i] = long1Count;
                    break;
                case SUM_LONG1:
                    values[i] =
                            long1Count == 0 && !plan.consumesRetractions()
                                    ? specs[i].normalizePartial(null)
                                    : output.getLong(base + OUTPUT_LONG1_SUM_OFFSET);
                    break;
                case MIN_LONG1:
                    values[i] =
                            long1Count == 0
                                    ? null
                                    : output.getLong(base + OUTPUT_MIN_LONG1_OFFSET);
                    break;
                case MAX_LONG1:
                    values[i] =
                            long1Count == 0 ? null : output.getLong(base + OUTPUT_MAX_LONG1_OFFSET);
                    break;
                default:
                    throw new IllegalStateException("Unhandled output lane " + outputLanes[i]);
            }
        }
        return values;
    }

    private Object[] flexibleAggregateValues(int group) {
        int groupBase = outputBase(group);
        GpuAggregatePlan.AggregateSpec[] specs = plan.aggregates();
        Object[] values = new Object[specs.length];
        for (int i = 0; i < specs.length; i++) {
            GpuAggregatePlan.AggregateSpec spec = specs[i];
            int resultBase = groupBase + (i + 1) * FLEXIBLE_RESULT_BYTES;
            long count = output.getLong(resultBase);
            switch (spec.kind) {
                case COUNT:
                    values[i] = count;
                    break;
                case SUM:
                case SUM0:
                    if (count == 0 && !plan.consumesRetractions()) {
                        values[i] = spec.normalizePartial(null);
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        values[i] =
                                spec.normalizePartial(
                                        output.getDouble(resultBase + Long.BYTES));
                    } else {
                        values[i] =
                                spec.normalizePartial(
                                        output.getLong(resultBase + Long.BYTES));
                    }
                    break;
                case AVG:
                    values[i] =
                            count == 0 && !plan.consumesRetractions()
                                    ? null
                                    : isIntegral(spec.argumentType)
                                            ? AverageAccumulator.integral(
                                                    output.getLong(resultBase + Long.BYTES), count)
                                            : new AverageAccumulator(
                                                    output.getDouble(resultBase + Long.BYTES),
                                                    count);
                    break;
                case MIN:
                case MAX:
                    if (count == 0) {
                        values[i] = null;
                    } else if (spec.argumentType == GpuType.FLOAT32
                            || spec.argumentType == GpuType.FLOAT64) {
                        values[i] =
                                spec.normalizePartial(
                                        output.getDouble(resultBase + Long.BYTES));
                    } else if (spec.argumentType == GpuType.TIMESTAMP3) {
                        values[i] =
                                TimestampData.fromEpochMillis(
                                        output.getLong(resultBase + Long.BYTES));
                    } else if (spec.argumentType == GpuType.STRING) {
                        values[i] =
                                decodeStringRank(
                                        spec.inputPackedOrdinal,
                                        output.getLong(resultBase + Long.BYTES));
                    } else {
                        values[i] =
                                spec.normalizePartial(
                                        output.getLong(resultBase + Long.BYTES));
                    }
                    break;
                default:
                    throw new IllegalStateException(
                            "Unhandled flexible aggregate kind " + spec.kind);
            }
        }
        return values;
    }

    private String decodeStringRank(int packedOrdinal, long rank) {
        String[] values;
        if (packedOrdinal == sequencePackedOrdinal) {
            values = long0RankedStrings;
        } else if (packedOrdinal == sentNanosPackedOrdinal) {
            values = long1RankedStrings;
        } else {
            throw new IllegalStateException("STRING aggregate is not bound to a native lane");
        }
        if (values == null || rank < 0L || rank >= values.length) {
            throw new IllegalStateException("native STRING aggregate returned invalid rank " + rank);
        }
        return values[(int) rank];
    }

    private static boolean isIntegral(GpuType type) {
        return type == GpuType.INT8
                || type == GpuType.INT16
                || type == GpuType.INT32
                || type == GpuType.INT64;
    }

    private int outputBase(int group) {
        return OUTPUT_HEADER_BYTES + group * outputRowBytes;
    }

    private void executeNative(int rows, int groups) {
        prepareStringRanks(rows);
        deviceGuard.bind();
        if (flexible) {
            nativeExecuteFlexible(handle, rows, groups);
        } else {
            nativeExecute(handle, rows, groups);
        }
    }

    private void prepareStringRanks(int rows) {
        if (long0StringValues != null) {
            long0RankedStrings =
                    encodeStringRanks(long0StringValues, rows, INPUT_SEQUENCE_OFFSET);
        }
        if (long1StringValues != null) {
            long1RankedStrings =
                    encodeStringRanks(long1StringValues, rows, INPUT_SENT_NANOS_OFFSET);
        }
    }

    private String[] encodeStringRanks(String[] values, int rows, int valueOffset) {
        List<String> ordered = new ArrayList<>(new HashSet<>(Arrays.asList(values).subList(0, rows)));
        ordered.remove(null);
        ordered.sort(DenseGroupbyKernel::compareUtf8);
        Map<String, Long> ranks = new HashMap<>(Math.max(4, ordered.size() * 2));
        for (int rank = 0; rank < ordered.size(); rank++) {
            ranks.put(ordered.get(rank), (long) rank);
        }
        for (int row = 0; row < rows; row++) {
            if (values[row] != null) {
                input.putLong(row * INPUT_ROW_BYTES + valueOffset, ranks.get(values[row]));
            }
        }
        return ordered.toArray(new String[0]);
    }

    private static int compareUtf8(String left, String right) {
        byte[] leftBytes = left.getBytes(StandardCharsets.UTF_8);
        byte[] rightBytes = right.getBytes(StandardCharsets.UTF_8);
        int common = Math.min(leftBytes.length, rightBytes.length);
        for (int index = 0; index < common; index++) {
            int comparison =
                    Integer.compare(Byte.toUnsignedInt(leftBytes[index]), Byte.toUnsignedInt(rightBytes[index]));
            if (comparison != 0) {
                return comparison;
            }
        }
        return Integer.compare(leftBytes.length, rightBytes.length);
    }

    private void recordMetrics(int rows, int groups, long packNanos, long decodeNanos) {
        AggregateRuntimeMetrics.record(
                rows,
                groups,
                packNanos,
                output.getLong(HEADER_H2D_NANOS_OFFSET),
                output.getLong(HEADER_COMPUTE_NANOS_OFFSET),
                output.getLong(HEADER_D2H_NANOS_OFFSET),
                decodeNanos,
                output.getLong(HEADER_INPUT_BYTES_OFFSET),
                output.getLong(HEADER_OUTPUT_BYTES_OFFSET));
    }

    @Override
    public void close() {
        if (handle == 0L) {
            return;
        }

        // A failed bind leaves the native owner untouched so a later retry cannot close it on the
        // wrong CUDA device.
        deviceGuard.bind();
        nativeResources.close(this::closeNativeContext);
    }

    static SupportLayout layoutFor(GpuAggregatePlan.AggregateSpec[] specs) {
        SupportLayout layout = new SupportLayout(specs.length);
        for (int i = 0; i < specs.length; i++) {
            GpuAggregatePlan.AggregateSpec spec = specs[i];
            int filterLane =
                    spec.filterPackedOrdinal == -1
                            ? -1
                            : assignFilterLane(layout, spec.filterPackedOrdinal);
            switch (spec.kind) {
                case COUNT:
                    if (spec.inputPackedOrdinal < 0 && filterLane >= 0) {
                        layout.outputLanes[i] =
                                filterLane == 0
                                        ? OutputLane.FILTER_COUNT0
                                        : filterLane == 1
                                                ? OutputLane.FILTER_COUNT1
                                                : OutputLane.FILTER_COUNT2;
                        layout.operationMask |=
                                filterLane == 0
                                        ? OP_FILTER0_COUNT
                                        : filterLane == 1 ? OP_FILTER1_COUNT : OP_FILTER2_COUNT;
                    } else if (spec.inputPackedOrdinal < 0) {
                        layout.outputLanes[i] = OutputLane.ROW_COUNT;
                    } else if (spec.argumentType == GpuType.FLOAT64) {
                        layout.metricPackedOrdinal =
                                sameInput(
                                        "DOUBLE COUNT/SUM",
                                        layout.metricPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.metricFilterLane =
                                bindFilter(
                                        "DOUBLE aggregate",
                                        layout.metricFilterLane,
                                        filterLane);
                        layout.outputLanes[i] = OutputLane.METRIC_COUNT;
                        layout.operationMask |= OP_METRIC_COUNT;
                    } else if (spec.argumentType == GpuType.INT32) {
                        layout.intPackedOrdinal =
                                sameInput(
                                        "INT COUNT/MIN/MAX",
                                        layout.intPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.intFilterLane =
                                bindFilter(
                                        "INT aggregate", layout.intFilterLane, filterLane);
                        layout.outputLanes[i] = OutputLane.INT_COUNT;
                        layout.operationMask |= OP_INT_COUNT;
                    } else if (spec.argumentType == GpuType.INT64) {
                        int lane = assignLongLane(layout, spec.inputPackedOrdinal);
                        bindLongFilter(layout, lane, filterLane);
                        layout.outputLanes[i] =
                                lane == 0 ? OutputLane.LONG0_COUNT : OutputLane.LONG1_COUNT;
                        layout.operationMask |= lane == 0 ? OP_LONG0_COUNT : OP_LONG1_COUNT;
                    } else {
                        throw new UnsupportedLayout(
                                "COUNT currently supports *, DOUBLE, INT, or BIGINT input");
                    }
                    break;
                case SUM:
                case SUM0:
                    if (spec.argumentType == GpuType.FLOAT64
                            && spec.resultType == GpuType.FLOAT64) {
                        layout.metricPackedOrdinal =
                                sameInput(
                                        "DOUBLE COUNT/SUM",
                                        layout.metricPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.metricFilterLane =
                                bindFilter(
                                        "DOUBLE aggregate",
                                        layout.metricFilterLane,
                                        filterLane);
                        layout.outputLanes[i] = OutputLane.METRIC_SUM;
                        layout.operationMask |= OP_METRIC_COUNT | OP_METRIC_SUM;
                    } else if (spec.argumentType == GpuType.INT64
                            && spec.resultType == GpuType.INT64) {
                        int lane = assignLongLane(layout, spec.inputPackedOrdinal);
                        bindLongFilter(layout, lane, filterLane);
                        layout.outputLanes[i] =
                                lane == 0 ? OutputLane.SUM_LONG0 : OutputLane.SUM_LONG1;
                        layout.operationMask |=
                                lane == 0
                                        ? OP_LONG0_COUNT | OP_LONG0_SUM
                                        : OP_LONG1_COUNT | OP_LONG1_SUM;
                    } else if (spec.argumentType == GpuType.INT32
                            && (spec.resultType == GpuType.INT32
                                    || spec.resultType == GpuType.INT64)) {
                        layout.intPackedOrdinal =
                                sameInput(
                                        "INT COUNT/SUM/MIN/MAX",
                                        layout.intPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.intFilterLane =
                                bindFilter(
                                        "INT aggregate", layout.intFilterLane, filterLane);
                        layout.outputLanes[i] = OutputLane.SUM_INT;
                        layout.operationMask |= OP_INT_COUNT | OP_INT_SUM;
                    } else {
                        throw new UnsupportedLayout(
                                "SUM currently needs DOUBLE, BIGINT, or INT input/result; got "
                                        + spec.argumentType
                                        + " -> "
                                        + spec.resultType);
                    }
                    break;
                case AVG:
                    if (spec.argumentType == GpuType.FLOAT64
                            && spec.resultType == GpuType.FLOAT64) {
                        layout.metricPackedOrdinal =
                                sameInput(
                                        "DOUBLE COUNT/SUM/AVG",
                                        layout.metricPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.metricFilterLane =
                                bindFilter(
                                        "DOUBLE aggregate",
                                        layout.metricFilterLane,
                                        filterLane);
                        layout.outputLanes[i] = OutputLane.METRIC_AVG;
                        layout.operationMask |= OP_METRIC_COUNT | OP_METRIC_SUM;
                    } else {
                        throw new UnsupportedLayout(
                                "AVG currently needs DOUBLE input/output");
                    }
                    break;
                case MIN:
                    if (spec.argumentType == GpuType.FLOAT64
                            && spec.resultType == GpuType.FLOAT64) {
                        layout.metricPackedOrdinal =
                                sameInput(
                                        "DOUBLE COUNT/SUM/AVG/MIN/MAX",
                                        layout.metricPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.metricFilterLane =
                                bindFilter(
                                        "DOUBLE aggregate",
                                        layout.metricFilterLane,
                                        filterLane);
                        layout.outputLanes[i] = OutputLane.MIN_METRIC;
                        layout.operationMask |= OP_METRIC_COUNT | OP_MIN_METRIC;
                    } else if (spec.argumentType == GpuType.INT32
                            && spec.resultType == GpuType.INT32) {
                        layout.intPackedOrdinal =
                                sameInput(
                                        "INT COUNT/MIN/MAX",
                                        layout.intPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.intFilterLane =
                                bindFilter(
                                        "INT aggregate", layout.intFilterLane, filterLane);
                        layout.outputLanes[i] = OutputLane.MIN_INT;
                        layout.operationMask |= OP_INT_COUNT | OP_MIN_INT;
                    } else if (spec.argumentType == GpuType.INT64
                            && spec.resultType == GpuType.INT64) {
                        int lane = assignLongLane(layout, spec.inputPackedOrdinal);
                        bindLongFilter(layout, lane, filterLane);
                        layout.outputLanes[i] =
                                lane == 0 ? OutputLane.MIN_LONG0 : OutputLane.MIN_LONG1;
                        layout.operationMask |=
                                lane == 0
                                        ? OP_LONG0_COUNT | OP_MIN_LONG0
                                        : OP_LONG1_COUNT | OP_MIN_LONG1;
                    } else {
                        throw new UnsupportedLayout(
                                "MIN currently needs DOUBLE, INT, or BIGINT input/output");
                    }
                    break;
                case MAX:
                    if (spec.argumentType == GpuType.FLOAT64
                            && spec.resultType == GpuType.FLOAT64) {
                        layout.metricPackedOrdinal =
                                sameInput(
                                        "DOUBLE COUNT/SUM/AVG/MIN/MAX",
                                        layout.metricPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.metricFilterLane =
                                bindFilter(
                                        "DOUBLE aggregate",
                                        layout.metricFilterLane,
                                        filterLane);
                        layout.outputLanes[i] = OutputLane.MAX_METRIC;
                        layout.operationMask |= OP_METRIC_COUNT | OP_MAX_METRIC;
                    } else if (spec.argumentType == GpuType.INT32
                            && spec.resultType == GpuType.INT32) {
                        layout.intPackedOrdinal =
                                sameInput(
                                        "INT COUNT/MIN/MAX",
                                        layout.intPackedOrdinal,
                                        spec.inputPackedOrdinal);
                        layout.intFilterLane =
                                bindFilter(
                                        "INT aggregate", layout.intFilterLane, filterLane);
                        layout.outputLanes[i] = OutputLane.MAX_INT;
                        layout.operationMask |= OP_INT_COUNT | OP_MAX_INT;
                    } else if (spec.argumentType == GpuType.INT64
                            && spec.resultType == GpuType.INT64) {
                        int lane = assignLongLane(layout, spec.inputPackedOrdinal);
                        bindLongFilter(layout, lane, filterLane);
                        layout.outputLanes[i] =
                                lane == 0 ? OutputLane.MAX_LONG0 : OutputLane.MAX_LONG1;
                        layout.operationMask |=
                                lane == 0
                                        ? OP_LONG0_COUNT | OP_MAX_LONG0
                                        : OP_LONG1_COUNT | OP_MAX_LONG1;
                    } else {
                        throw new UnsupportedLayout(
                                "MAX currently needs DOUBLE, INT, or BIGINT input/output");
                    }
                    break;
                default:
                    throw new UnsupportedLayout("unsupported aggregate kind " + spec.kind);
            }
        }
        return layout;
    }

    /**
     * Builds a compact aggregate program for cases that cannot share one fixed result lane, such
     * as SUM(metric) together with SUM(metric) FILTER (WHERE flag). Every aggregate owns a result
     * slot, while one fused kernel still reads each input row only once.
     */
    static FlexibleLayout flexibleLayoutFor(GpuAggregatePlan.AggregateSpec[] specs) {
        if (specs.length == 0 || specs.length > FLEXIBLE_MAX_AGGREGATES) {
            throw new UnsupportedLayout(
                    "flexible aggregate supports 1 to "
                            + FLEXIBLE_MAX_AGGREGATES
                            + " aggregate calls");
        }
        FlexibleLayout layout = new FlexibleLayout(specs.length);
        for (int i = 0; i < specs.length; i++) {
            GpuAggregatePlan.AggregateSpec spec = specs[i];
            layout.filterLanes[i] =
                    spec.filterPackedOrdinal == -1
                            ? -1
                            : assignFlexibleFilterLane(layout, spec.filterPackedOrdinal);
            switch (spec.kind) {
                case COUNT:
                    layout.kinds[i] = FLEX_KIND_COUNT;
                    layout.valueLanes[i] =
                            spec.inputPackedOrdinal < 0
                                    ? FLEX_VALUE_ROW
                                    : assignFlexibleValueLane(layout, spec);
                    break;
                case SUM:
                case SUM0:
                    validateFlexibleSum(spec);
                    layout.kinds[i] = FLEX_KIND_SUM;
                    layout.valueLanes[i] = assignFlexibleValueLane(layout, spec);
                    break;
                case AVG:
                    if (!spec.argumentType.isNumeric()
                            || spec.argumentType != spec.resultType) {
                        throw new UnsupportedLayout(
                                "AVG needs matching numeric input/output");
                    }
                    layout.kinds[i] = FLEX_KIND_AVG;
                    layout.valueLanes[i] = assignFlexibleValueLane(layout, spec);
                    break;
                case MIN:
                case MAX:
                    if (spec.argumentType != spec.resultType
                            || (spec.argumentType != GpuType.FLOAT32
                                    && spec.argumentType != GpuType.FLOAT64
                                    && spec.argumentType != GpuType.INT8
                                    && spec.argumentType != GpuType.INT16
                                    && spec.argumentType != GpuType.INT32
                                    && spec.argumentType != GpuType.INT64
                                    && spec.argumentType != GpuType.TIMESTAMP3
                                    && spec.argumentType != GpuType.STRING)) {
                        throw new UnsupportedLayout(
                                spec.kind
                                        + " needs a matching numeric, STRING, or TIMESTAMP input/output");
                    }
                    layout.kinds[i] =
                            spec.kind == GpuAggregatePlan.AggregateKind.MIN
                                    ? FLEX_KIND_MIN
                                    : FLEX_KIND_MAX;
                    layout.valueLanes[i] = assignFlexibleValueLane(layout, spec);
                    break;
                default:
                    throw new UnsupportedLayout("unsupported aggregate kind " + spec.kind);
            }
        }
        return layout;
    }

    private static void validateFlexibleSum(GpuAggregatePlan.AggregateSpec spec) {
        boolean supported =
                ((spec.argumentType == GpuType.FLOAT32
                                || spec.argumentType == GpuType.FLOAT64)
                                && spec.resultType == spec.argumentType)
                        || ((spec.argumentType == GpuType.INT8
                                        || spec.argumentType == GpuType.INT16
                                        || spec.argumentType == GpuType.INT32
                                        || spec.argumentType == GpuType.INT64)
                                && spec.resultType == spec.argumentType)
                        || (spec.argumentType == GpuType.FLOAT64
                                && spec.resultType == GpuType.FLOAT64)
                        || (spec.argumentType == GpuType.INT64
                                && spec.resultType == GpuType.INT64)
                        || (spec.argumentType == GpuType.INT32
                                && (spec.resultType == GpuType.INT32
                                        || spec.resultType == GpuType.INT64));
        if (!supported) {
            throw new UnsupportedLayout(
                    "SUM needs matching numeric input/result; got "
                            + spec.argumentType
                            + " -> "
                            + spec.resultType);
        }
    }

    private static int assignFlexibleValueLane(
            FlexibleLayout layout, GpuAggregatePlan.AggregateSpec spec) {
        if (spec.inputPackedOrdinal < 0) {
            throw new UnsupportedLayout(spec.kind + " requires an input column");
        }
        switch (spec.argumentType) {
            case FLOAT32:
            case FLOAT64:
                layout.metricPackedOrdinal =
                        sameInput(
                                "flexible floating aggregates",
                                layout.metricPackedOrdinal,
                                spec.inputPackedOrdinal);
                return FLEX_VALUE_METRIC;
            case INT8:
            case INT16:
            case INT32:
                layout.intPackedOrdinal =
                        sameInput(
                                "flexible narrow-integer aggregates",
                                layout.intPackedOrdinal,
                                spec.inputPackedOrdinal);
                return FLEX_VALUE_INT;
            case INT64:
            case TIMESTAMP3:
            case BOOLEAN:
            case STRING:
                return assignFlexibleLongLane(layout, spec.inputPackedOrdinal) == 0
                        ? FLEX_VALUE_LONG0
                        : FLEX_VALUE_LONG1;
            default:
                throw new UnsupportedLayout(
                        spec.kind + " does not support input " + spec.argumentType);
        }
    }

    private static int assignFlexibleLongLane(FlexibleLayout layout, int packedOrdinal) {
        if (layout.long0PackedOrdinal == packedOrdinal) {
            return 0;
        }
        if (layout.long1PackedOrdinal == packedOrdinal) {
            return 1;
        }
        if (layout.long0PackedOrdinal < 0) {
            layout.long0PackedOrdinal = packedOrdinal;
            return 0;
        }
        if (layout.long1PackedOrdinal < 0) {
            layout.long1PackedOrdinal = packedOrdinal;
            return 1;
        }
        throw new UnsupportedLayout(
                "flexible aggregate supports at most two distinct BIGINT inputs");
    }

    private static int assignFlexibleFilterLane(FlexibleLayout layout, int packedOrdinal) {
        for (int i = 0; i < layout.filterPackedOrdinals.length; i++) {
            if (layout.filterPackedOrdinals[i] == packedOrdinal) {
                return i;
            }
        }
        for (int i = 0; i < layout.filterPackedOrdinals.length; i++) {
            if (layout.filterPackedOrdinals[i] == -1) {
                layout.filterPackedOrdinals[i] = packedOrdinal;
                return i;
            }
        }
        throw new UnsupportedLayout("at most three distinct aggregate FILTER inputs are supported");
    }

    private static int sameInput(String label, int current, int requested) {
        if (current >= 0 && current != requested) {
            throw new UnsupportedLayout(label + " calls must reference the same input column");
        }
        return requested;
    }

    private static int bindFilter(String label, int current, int requested) {
        if (current == FILTER_UNBOUND) {
            return requested;
        }
        if (current != requested) {
            throw new UnsupportedLayout(
                    label
                            + " calls sharing one native value lane must use the same FILTER");
        }
        return current;
    }

    private static void bindLongFilter(SupportLayout layout, int lane, int requested) {
        if (lane == 0) {
            layout.long0FilterLane =
                    bindFilter("BIGINT lane 0 aggregate", layout.long0FilterLane, requested);
        } else {
            layout.long1FilterLane =
                    bindFilter("BIGINT lane 1 aggregate", layout.long1FilterLane, requested);
        }
    }

    private static int assignLongLane(SupportLayout layout, int packedOrdinal) {
        if (layout.long0PackedOrdinal == packedOrdinal) {
            return 0;
        }
        if (layout.long1PackedOrdinal == packedOrdinal) {
            return 1;
        }
        if (layout.long0PackedOrdinal < 0) {
            layout.long0PackedOrdinal = packedOrdinal;
            return 0;
        }
        if (layout.long1PackedOrdinal < 0) {
            layout.long1PackedOrdinal = packedOrdinal;
            return 1;
        }
        throw new UnsupportedLayout("at most two distinct BIGINT aggregate inputs are supported");
    }

    private static int assignFilterLane(SupportLayout layout, int packedOrdinal) {
        for (int i = 0; i < layout.filterPackedOrdinals.length; i++) {
            if (layout.filterPackedOrdinals[i] == packedOrdinal) {
                return i;
            }
        }
        for (int i = 0; i < layout.filterPackedOrdinals.length; i++) {
            if (layout.filterPackedOrdinals[i] == -1) {
                layout.filterPackedOrdinals[i] = packedOrdinal;
                return i;
            }
        }
        throw new UnsupportedLayout("at most three distinct aggregate FILTER inputs are supported");
    }

    enum OutputLane {
        ROW_COUNT,
        FILTER_COUNT0,
        FILTER_COUNT1,
        FILTER_COUNT2,
        METRIC_COUNT,
        METRIC_SUM,
        METRIC_AVG,
        MIN_METRIC,
        MAX_METRIC,
        INT_COUNT,
        SUM_INT,
        MIN_INT,
        MAX_INT,
        LONG0_COUNT,
        SUM_LONG0,
        MIN_LONG0,
        MAX_LONG0,
        LONG1_COUNT,
        SUM_LONG1,
        MIN_LONG1,
        MAX_LONG1
    }

    static final class SupportLayout {
        final OutputLane[] outputLanes;
        int operationMask = OP_ROW_COUNT;
        int metricPackedOrdinal = -1;
        int intPackedOrdinal = -1;
        int long0PackedOrdinal = -1;
        int long1PackedOrdinal = -1;
        int metricFilterLane = FILTER_UNBOUND;
        int intFilterLane = FILTER_UNBOUND;
        int long0FilterLane = FILTER_UNBOUND;
        int long1FilterLane = FILTER_UNBOUND;
        final int[] filterPackedOrdinals = {-1, -1, -1};

        private SupportLayout(int aggregateCount) {
            outputLanes = new OutputLane[aggregateCount];
        }
    }

    static final class FlexibleLayout {
        final int[] kinds;
        final int[] valueLanes;
        final int[] filterLanes;
        int metricPackedOrdinal = -1;
        int intPackedOrdinal = -1;
        int long0PackedOrdinal = -1;
        int long1PackedOrdinal = -1;
        final int[] filterPackedOrdinals = {-1, -1, -1};

        private FlexibleLayout(int aggregateCount) {
            kinds = new int[aggregateCount];
            valueLanes = new int[aggregateCount];
            filterLanes = new int[aggregateCount];
        }
    }

    private static final class UnsupportedLayout extends IllegalArgumentException {
        private static final long serialVersionUID = 1L;

        private UnsupportedLayout(String message) {
            super(message);
        }
    }

    private static final class DenseKeyMap {
        private final long[] tableKeys;
        private final long[] tableWindows;
        private final int[] tableIds;
        private final int[] epochs;
        private final long[] keysById;
        private final boolean[] keyNullById;
        private final long[] windowsById;
        private final int[] denseIds;
        private final int[] denseEpochs;
        private final int mask;
        private int epoch = 1;
        private int size;

        DenseKeyMap(int capacity) {
            int tableSize = 1;
            while (tableSize < capacity * 2) {
                tableSize <<= 1;
            }
            tableKeys = new long[tableSize];
            tableWindows = new long[tableSize];
            tableIds = new int[tableSize];
            epochs = new int[tableSize];
            keysById = new long[capacity];
            keyNullById = new boolean[capacity];
            windowsById = new long[capacity];
            denseIds = new int[capacity];
            denseEpochs = new int[capacity];
            mask = tableSize - 1;
        }

        void reset() {
            size = 0;
            if (++epoch == 0) {
                Arrays.fill(epochs, 0);
                Arrays.fill(denseEpochs, 0);
                epoch = 1;
            }
        }

        int idFor(Long key, long windowEnd, boolean windowed) {
            return idFor(key == null ? 0L : key, key == null, windowEnd, windowed);
        }

        int idFor(long keyBits, boolean keyNull, long windowEnd, boolean windowed) {
            if (!windowed && !keyNull && keyBits >= 0L && keyBits < denseIds.length) {
                int denseKey = (int) keyBits;
                if (denseEpochs[denseKey] == epoch) {
                    return denseIds[denseKey];
                }
                int id = size++;
                denseEpochs[denseKey] = epoch;
                denseIds[denseKey] = id;
                keysById[id] = keyBits;
                keyNullById[id] = false;
                windowsById[id] = windowEnd;
                return id;
            }
            long discriminator = windowed ? windowEnd : 0L;
            long hash =
                    mix64(
                            keyBits
                                    ^ Long.rotateLeft(discriminator, 23)
                                    ^ (keyNull ? 1L : 0L));
            int slot = (int) hash & mask;
            while (epochs[slot] == epoch) {
                int id = tableIds[slot];
                if (tableKeys[slot] == keyBits
                        && tableWindows[slot] == discriminator
                        && keyNullById[id] == keyNull) {
                    return id;
                }
                slot = (slot + 1) & mask;
            }
            int id = size++;
            epochs[slot] = epoch;
            tableKeys[slot] = keyBits;
            tableWindows[slot] = discriminator;
            tableIds[slot] = id;
            keysById[id] = keyBits;
            keyNullById[id] = keyNull;
            windowsById[id] = windowEnd;
            return id;
        }

        int size() {
            return size;
        }

        Object keyValue(int id) {
            return keyNullById[id] ? null : keysById[id];
        }

        long keyBits(int id) {
            return keysById[id];
        }

        boolean keyNull(int id) {
            return keyNullById[id];
        }

        long windowEnd(int id) {
            return windowsById[id];
        }

        private static long mix64(long value) {
            long z = value + 0x9e3779b97f4a7c15L;
            z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
            z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
            return z ^ (z >>> 31);
        }
    }

    /** Allocation-light exact dictionary for the common BIGINT + VARCHAR composite key. */
    private static final class LongStringDenseKeyMap {
        private final Map<LongStringGroupKey, Integer> ids;
        private final Object[][] keysById;
        private final long[] windowsById;
        private final LongStringGroupKey probe = new LongStringGroupKey();
        private int size;

        private LongStringDenseKeyMap(int capacity) {
            ids = new HashMap<>(Math.max(16, capacity * 2));
            keysById = new Object[capacity][];
            windowsById = new long[capacity];
        }

        private void reset() {
            ids.clear();
            size = 0;
        }

        private int idFor(
                long longValue,
                boolean longIsNull,
                StringData stringValue,
                long windowEnd,
                boolean windowed) {
            long discriminator = windowed ? windowEnd : 0L;
            probe.reset(longValue, longIsNull, stringValue, discriminator);
            Integer existing = ids.get(probe);
            if (existing != null) {
                return existing;
            }
            int id = size++;
            String stableString = stringValue == null ? null : stringValue.toString();
            StringData stableStringData =
                    stableString == null ? null : StringData.fromString(stableString);
            LongStringGroupKey stable =
                    new LongStringGroupKey(
                            longValue, longIsNull, stableStringData, discriminator);
            ids.put(stable, id);
            keysById[id] =
                    new Object[] {longIsNull ? null : longValue, stableString};
            windowsById[id] = windowEnd;
            return id;
        }

        private int size() {
            return size;
        }

        private Object[] keyValues(int id) {
            return keysById[id];
        }

        private long windowEnd(int id) {
            return windowsById[id];
        }
    }

    private static final class LongStringGroupKey {
        private long longValue;
        private boolean longIsNull;
        private StringData stringValue;
        private long discriminator;
        private int hash;

        private LongStringGroupKey() {}

        private LongStringGroupKey(
                long longValue,
                boolean longIsNull,
                StringData stringValue,
                long discriminator) {
            reset(longValue, longIsNull, stringValue, discriminator);
        }

        private void reset(
                long longValue,
                boolean longIsNull,
                StringData stringValue,
                long discriminator) {
            this.longValue = longValue;
            this.longIsNull = longIsNull;
            this.stringValue = stringValue;
            this.discriminator = discriminator;
            int current = longIsNull ? 0 : Long.hashCode(longValue);
            current = 31 * current + (stringValue == null ? 0 : stringValue.hashCode());
            hash = 31 * current + Long.hashCode(discriminator);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof LongStringGroupKey)) {
                return false;
            }
            LongStringGroupKey that = (LongStringGroupKey) other;
            return longIsNull == that.longIsNull
                    && (longIsNull || longValue == that.longValue)
                    && discriminator == that.discriminator
                    && Objects.equals(stringValue, that.stringValue);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    /**
     * Allocation-free lookup table for GROUPING SETS' common BIGINT + INT + BIGINT key.
     *
     * <p>The generic path allocates an {@code Object[]} and a wrapper key for every input row.
     * This table keeps primitive columns and a NULL mask in flat arrays; boxing happens only once
     * per output group after the CUDA reduction has finished.
     */
    private static final class LongIntLongDenseKeyMap {
        private final long[] tableValue0;
        private final int[] tableValue1;
        private final long[] tableValue2;
        private final int[] tableNullMasks;
        private final long[] tableWindows;
        private final int[] tableIds;
        private final int[] epochs;
        private final long[] value0ById;
        private final int[] value1ById;
        private final long[] value2ById;
        private final int[] nullMasksById;
        private final long[] windowsById;
        private final int mask;
        private int epoch = 1;
        private int size;

        private LongIntLongDenseKeyMap(int capacity) {
            int tableSize = 1;
            while (tableSize < capacity * 2) {
                tableSize <<= 1;
            }
            tableValue0 = new long[tableSize];
            tableValue1 = new int[tableSize];
            tableValue2 = new long[tableSize];
            tableNullMasks = new int[tableSize];
            tableWindows = new long[tableSize];
            tableIds = new int[tableSize];
            epochs = new int[tableSize];
            value0ById = new long[capacity];
            value1ById = new int[capacity];
            value2ById = new long[capacity];
            nullMasksById = new int[capacity];
            windowsById = new long[capacity];
            mask = tableSize - 1;
        }

        private void reset() {
            size = 0;
            if (++epoch == 0) {
                Arrays.fill(epochs, 0);
                epoch = 1;
            }
        }

        private int idFor(
                long value0,
                int value1,
                long value2,
                int nullMask,
                long windowEnd,
                boolean windowed) {
            long discriminator = windowed ? windowEnd : 0L;
            long hash = mix64(value0 ^ Long.rotateLeft(value2, 21));
            hash = mix64(hash ^ Integer.toUnsignedLong(value1) ^ ((long) nullMask << 56));
            hash = mix64(hash ^ Long.rotateLeft(discriminator, 39));
            int slot = (int) hash & mask;
            while (epochs[slot] == epoch) {
                if (tableValue0[slot] == value0
                        && tableValue1[slot] == value1
                        && tableValue2[slot] == value2
                        && tableNullMasks[slot] == nullMask
                        && tableWindows[slot] == discriminator) {
                    return tableIds[slot];
                }
                slot = (slot + 1) & mask;
            }
            int id = size++;
            epochs[slot] = epoch;
            tableValue0[slot] = value0;
            tableValue1[slot] = value1;
            tableValue2[slot] = value2;
            tableNullMasks[slot] = nullMask;
            tableWindows[slot] = discriminator;
            tableIds[slot] = id;
            value0ById[id] = value0;
            value1ById[id] = value1;
            value2ById[id] = value2;
            nullMasksById[id] = nullMask;
            windowsById[id] = windowEnd;
            return id;
        }

        private int size() {
            return size;
        }

        private Object[] keyValues(int id) {
            int nullMask = nullMasksById[id];
            return new Object[] {
                (nullMask & 1) == 0 ? value0ById[id] : null,
                (nullMask & 2) == 0 ? value1ById[id] : null,
                (nullMask & 4) == 0 ? value2ById[id] : null
            };
        }

        private long windowEnd(int id) {
            return windowsById[id];
        }

        private static long mix64(long value) {
            long z = value + 0x9e3779b97f4a7c15L;
            z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
            z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
            return z ^ (z >>> 31);
        }
    }

    /** Host-side dictionary for composite and non-BIGINT keys; the GPU sees only dense int ids. */
    private static final class GenericDenseKeyMap {
        private final Map<GenericGroupKey, Integer> ids;
        private final Object[][] keysById;
        private final long[] windowsById;
        private int size;

        private GenericDenseKeyMap(int capacity) {
            ids = new HashMap<>(Math.max(16, capacity * 2));
            keysById = new Object[capacity][];
            windowsById = new long[capacity];
        }

        private void reset() {
            ids.clear();
            size = 0;
        }

        private int idFor(Object[] values, long windowEnd, boolean windowed) {
            long discriminator = windowed ? windowEnd : 0L;
            GenericGroupKey candidate = new GenericGroupKey(values, discriminator);
            Integer existing = ids.get(candidate);
            if (existing != null) {
                return existing;
            }
            int id = size++;
            ids.put(candidate, id);
            keysById[id] = values;
            windowsById[id] = windowEnd;
            return id;
        }

        private int size() {
            return size;
        }

        private Object[] keyValues(int id) {
            return keysById[id];
        }

        private long windowEnd(int id) {
            return windowsById[id];
        }
    }

    private static final class GenericGroupKey {
        private final Object[] values;
        private final long discriminator;
        private final int hash;

        private GenericGroupKey(Object[] values, long discriminator) {
            this.values = values;
            this.discriminator = discriminator;
            this.hash = 31 * Arrays.deepHashCode(values) + Long.hashCode(discriminator);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof GenericGroupKey)) {
                return false;
            }
            GenericGroupKey that = (GenericGroupKey) other;
            return discriminator == that.discriminator && Arrays.deepEquals(values, that.values);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    /** Loads the project-owned JNI library lazily on the TaskManager that owns the GPU operator. */
    private static final class NativeLibrary {
        static {
            // The groupby DSO links against the common device-batch DSO; load that one first.
            System.loadLibrary("cudf_flink_device_batch");
            System.loadLibrary("cudf_flink_groupby");
        }

        static void ensureLoaded() {
            // Class initialization performs the load once.
        }
    }

    private static native long nativeCreate(
            int capacity,
            int operationMask,
            int metricFilterLane,
            int intFilterLane,
            int long0FilterLane,
            int long1FilterLane);

    private static native ByteBuffer nativeInputBuffer(long handle);

    private static native ByteBuffer nativeOutputBuffer(long handle);

    private static native long[] nativeResourceStats(long handle);

    private static native void nativeExecute(long handle, int rows, int groups);

    private static native void nativeClose(long handle);

    private static native long nativeCreateFlexible(
            int capacity, int[] kinds, int[] valueLanes, int[] filterLanes);

    private static native ByteBuffer nativeFlexibleInputBuffer(long handle);

    private static native ByteBuffer nativeFlexibleOutputBuffer(long handle);

    private static native long[] nativeFlexibleResourceStats(long handle);

    private static native void nativeExecuteFlexible(long handle, int rows, int groups);

    private static native void nativeCloseFlexible(long handle);
}
