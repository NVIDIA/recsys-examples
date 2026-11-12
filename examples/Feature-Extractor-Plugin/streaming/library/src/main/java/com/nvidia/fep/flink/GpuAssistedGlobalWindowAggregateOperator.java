// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.state.internal.InternalValueState;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.InternalTimer;
import org.apache.flink.streaming.api.operators.InternalTimerService;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.operators.Triggerable;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.runtime.keyselector.RowDataKeySelector;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.types.RowKind;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Slicing-window global merge for partials emitted by {@link GpuLocalWindowAggregateOperator}.
 *
 * <p>The raw rows have already been reduced by the project-owned GPU kernel. Sending the sparse
 * local partials through another GPU launch is counterproductive, so this operator performs an
 * exact host merge, keeps Flink keyed state/timers/checkpoints, and writes final binary rows
 * directly. Closed in-memory windows bypass a state write/read/timer round trip at a watermark;
 * checkpoints first drain the cache so recovery remains authoritative.
 */
public final class GpuAssistedGlobalWindowAggregateOperator
        extends AbstractStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>,
                BoundedOneInput,
                Triggerable<RowData, Long> {
    private static final long serialVersionUID = 1L;
    private static final String PARTIAL_CACHE_ENTRIES_PROPERTY =
            "cudf.flink.global-window.partial-cache.entries";
    private static final int DEFAULT_PARTIAL_CACHE_ENTRIES = 65_536;

    private final GpuLocalWindowAggregatePlan plan;
    private final long windowSizeMs;
    private final long windowSlideMs;
    private final long windowOffsetMs;
    private final boolean cumulativeWindow;
    private final int timeAttributePrecision;
    private final int[] windowPropertyKinds;
    private final int outputArity;
    private final int keyInputArity;
    private final int[] groupingOrdinals;
    private final GpuType[] groupingTypes;
    private final RowDataKeySelector keySelector;
    private final InternalTypeInfo<RowData> stateTypeInfo;

    private transient AggregateKeyCache keyCache;
    private transient InternalValueState<RowData, Long, RowData> windowState;
    private transient InternalValueState<RowData, Long, AggregateAverageState> averageState;
    private transient InternalTimerService<Long> timerService;
    private transient LinkedHashMap<WindowKey, WindowAggregatePartial> partialCache;
    private transient int maxPartialCacheEntries;
    private transient Counter inputCounter;
    private transient Counter outputCounter;
    private transient Counter lateCounter;
    private transient Counter directWatermarkCounter;
    private transient StreamRecord<RowData> reusableOutputRecord;
    private transient BinaryRowData reusableOutputRow;
    private transient BinaryRowWriter reusableOutputWriter;
    private transient long[] reusableWindowEnds;

    public GpuAssistedGlobalWindowAggregateOperator(
            GpuLocalWindowAggregatePlan plan,
            long windowSizeMs,
            long windowSlideMs,
            long windowOffsetMs,
            boolean cumulativeWindow,
            int timeAttributePrecision,
            int[] windowPropertyKinds,
            int outputArity,
            int keyInputArity,
            int[] groupingOrdinals,
            GpuType[] groupingTypes,
            RowDataKeySelector keySelector,
            InternalTypeInfo<RowData> stateTypeInfo) {
        this.plan = Objects.requireNonNull(plan);
        if (windowSizeMs <= 0L
                || windowSlideMs <= 0L
                || windowSizeMs % windowSlideMs != 0L
                || windowSizeMs / windowSlideMs > 64L
                || timeAttributePrecision < 0
                || timeAttributePrecision > 3) {
            throw new IllegalArgumentException("invalid slicing-window configuration");
        }
        if (groupingOrdinals.length != groupingTypes.length) {
            throw new IllegalArgumentException("grouping ordinal/type length mismatch");
        }
        int expectedOutputArity =
                plan.computePlan().outputTypes().length + windowPropertyKinds.length;
        if (outputArity != expectedOutputArity) {
            throw new IllegalArgumentException("global Window output arity mismatch");
        }
        this.windowSizeMs = windowSizeMs;
        this.windowSlideMs = windowSlideMs;
        this.windowOffsetMs = windowOffsetMs;
        this.cumulativeWindow = cumulativeWindow;
        this.timeAttributePrecision = timeAttributePrecision;
        this.windowPropertyKinds = windowPropertyKinds.clone();
        this.outputArity = outputArity;
        this.keyInputArity = keyInputArity;
        this.groupingOrdinals = groupingOrdinals.clone();
        this.groupingTypes = groupingTypes.clone();
        this.keySelector = Objects.requireNonNull(keySelector);
        this.stateTypeInfo = Objects.requireNonNull(stateTypeInfo);
    }

    @Override
    public void open() throws Exception {
        super.open();
        keyCache =
                new AggregateKeyCache(
                        keyInputArity,
                        groupingOrdinals,
                        groupingTypes,
                        keySelector);
        ValueStateDescriptor<RowData> descriptor =
                new ValueStateDescriptor<>("cudfGlobalTumblingWindowStateV1", stateTypeInfo);
        @SuppressWarnings("unchecked")
        InternalValueState<RowData, Long, RowData> namespacedState =
                (InternalValueState<RowData, Long, RowData>)
                        (ValueState<RowData>)
                                getOrCreateKeyedState(LongSerializer.INSTANCE, descriptor);
        windowState = namespacedState;
        if (plan.computePlan().hasAverage()) {
            ValueStateDescriptor<AggregateAverageState> averageDescriptor =
                    new ValueStateDescriptor<>(
                            "cudfGlobalTumblingWindowAverageStateV1",
                            AggregateAverageState.class);
            @SuppressWarnings("unchecked")
            InternalValueState<RowData, Long, AggregateAverageState> namespacedAverageState =
                    (InternalValueState<RowData, Long, AggregateAverageState>)
                            (ValueState<AggregateAverageState>)
                                    getOrCreateKeyedState(
                                            LongSerializer.INSTANCE,
                                            averageDescriptor);
            averageState = namespacedAverageState;
        }
        timerService =
                getInternalTimerService(
                        "cudfGlobalTumblingWindowTimers",
                        LongSerializer.INSTANCE,
                        this);
        maxPartialCacheEntries =
                Integer.getInteger(
                        PARTIAL_CACHE_ENTRIES_PROPERTY,
                        DEFAULT_PARTIAL_CACHE_ENTRIES);
        if (maxPartialCacheEntries <= 0) {
            throw new IllegalArgumentException(
                    PARTIAL_CACHE_ENTRIES_PROPERTY + " must be positive");
        }
        partialCache = new LinkedHashMap<>();
        inputCounter =
                getRuntimeContext().getMetricGroup().counter("cudfGlobalWindowPartialRows");
        outputCounter =
                getRuntimeContext().getMetricGroup().counter("cudfGlobalWindowOutputRows");
        lateCounter =
                getRuntimeContext().getMetricGroup().counter("cudfGlobalWindowLatePartials");
        directWatermarkCounter =
                getRuntimeContext()
                        .getMetricGroup()
                        .counter("cudfGlobalWindowDirectWatermarkEmits");
        reusableOutputRecord = new StreamRecord<>(null);
        reusableOutputRow = new BinaryRowData(outputArity);
        reusableOutputWriter = new BinaryRowWriter(reusableOutputRow);
        reusableWindowEnds = new long[64];
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (row.getRowKind() != RowKind.INSERT) {
            throw new IllegalArgumentException(
                    "GPU-assisted global Window accepts INSERT partials only, got "
                            + row.getRowKind());
        }
        WindowAggregatePartial slice = plan.partialFromRow(row);
        int windows =
                SlicingWindowExpansion.assignWindowEnds(
                        slice.windowEnd,
                        windowSizeMs,
                        windowSlideMs,
                        windowOffsetMs,
                        cumulativeWindow,
                        reusableWindowEnds);
        for (int i = 0; i < windows; i++) {
            addWindowPartial(slice, reusableWindowEnds[i]);
        }
        inputCounter.inc();
    }

    private void addWindowPartial(WindowAggregatePartial slice, long windowEnd)
            throws Exception {
        if (timerService.currentWatermark() >= windowEnd - 1L) {
            lateCounter.inc();
            return;
        }
        WindowAggregatePartial incoming =
                new WindowAggregatePartial(
                        windowEnd,
                        slice.groupValues,
                        slice.inputRows,
                        slice.aggregateValues);
        WindowKey probe = new WindowKey(windowEnd, incoming.groupValues);
        WindowAggregatePartial cached = partialCache.get(probe);
        if (cached == null) {
            WindowAggregatePartial stable = plan.stableCopy(incoming);
            partialCache.put(new WindowKey(stable.windowEnd, stable.groupValues), stable);
        } else {
            partialCache.put(probe, plan.merge(cached, incoming));
        }
        if (partialCache.size() >= maxPartialCacheEntries) {
            drainPartialCache();
        }
    }

    private void drainPartialCache() throws Exception {
        Iterator<Map.Entry<WindowKey, WindowAggregatePartial>> iterator =
                partialCache.entrySet().iterator();
        while (iterator.hasNext()) {
            WindowAggregatePartial partial = iterator.next().getValue();
            setCurrentKey(keyCache.keyFor(partial.groupValues));
            mergeState(partial);
            iterator.remove();
        }
    }

    private void mergeState(WindowAggregatePartial partial) throws Exception {
        GpuAggregatePlan compute = plan.computePlan();
        windowState.setCurrentNamespace(partial.windowEnd);
        if (averageState != null) {
            averageState.setCurrentNamespace(partial.windowEnd);
        }
        RowData oldState = windowState.value();
        AggregateAverageState oldAverage = averageState == null ? null : averageState.value();
        Object[] accumulators =
                compute.mergeAccumulatorValues(
                        oldState,
                        oldAverage,
                        partial.aggregateValues);
        GenericRowData newState = compute.outputRow(partial.groupValues, accumulators);
        newState.setRowKind(RowKind.INSERT);
        windowState.update(newState);
        if (averageState != null) {
            averageState.update(compute.averageState(accumulators));
        }
        if (oldState == null) {
            timerService.registerEventTimeTimer(
                    partial.windowEnd,
                    partial.windowEnd - 1L);
        }
    }

    private void emitClosedCachedWindows(long watermark) throws Exception {
        Iterator<Map.Entry<WindowKey, WindowAggregatePartial>> iterator =
                partialCache.entrySet().iterator();
        while (iterator.hasNext()) {
            WindowAggregatePartial partial = iterator.next().getValue();
            if (partial.windowEnd - 1L > watermark) {
                continue;
            }
            setCurrentKey(keyCache.keyFor(partial.groupValues));
            windowState.setCurrentNamespace(partial.windowEnd);
            if (averageState != null) {
                averageState.setCurrentNamespace(partial.windowEnd);
            }
            RowData oldState = windowState.value();
            AggregateAverageState oldAverage =
                    averageState == null ? null : averageState.value();
            GpuAggregatePlan compute = plan.computePlan();
            Object[] accumulators =
                    compute.mergeAccumulatorValues(
                            oldState,
                            oldAverage,
                            partial.aggregateValues);
            GenericRowData finalState = compute.outputRow(partial.groupValues, accumulators);
            finalState.setRowKind(RowKind.INSERT);
            if (oldState != null) {
                windowState.clear();
                if (averageState != null) {
                    averageState.clear();
                }
                timerService.deleteEventTimeTimer(
                        partial.windowEnd,
                        partial.windowEnd - 1L);
            }
            emitWindow(finalState, partial.windowEnd);
            directWatermarkCounter.inc();
            iterator.remove();
        }
    }

    @Override
    public void onEventTime(InternalTimer<RowData, Long> timer) throws Exception {
        long windowEnd = timer.getNamespace();
        windowState.setCurrentNamespace(windowEnd);
        if (averageState != null) {
            averageState.setCurrentNamespace(windowEnd);
        }
        RowData state = windowState.value();
        if (state == null) {
            return;
        }
        emitWindow(state, windowEnd);
        windowState.clear();
        if (averageState != null) {
            averageState.clear();
        }
    }

    private void emitWindow(RowData state, long windowEnd) {
        reusableOutputWriter.reset();
        GpuAggregatePlan compute = plan.computePlan();
        long windowStart =
                SlicingWindowExpansion.windowStart(
                        windowEnd,
                        windowSizeMs,
                        windowOffsetMs,
                        cumulativeWindow);
        compute.copyCoreOutput(state, reusableOutputWriter);
        int propertyStart = compute.outputTypes().length;
        for (int i = 0; i < windowPropertyKinds.length; i++) {
            int outputOrdinal = propertyStart + i;
            if (windowPropertyKinds[i] == CudfTumblingWindowOperator.PROPERTY_START) {
                reusableOutputWriter.writeTimestamp(
                        outputOrdinal,
                        TimestampData.fromEpochMillis(windowStart),
                        timeAttributePrecision);
            } else if (windowPropertyKinds[i]
                    == CudfTumblingWindowOperator.PROPERTY_END) {
                reusableOutputWriter.writeTimestamp(
                        outputOrdinal,
                        TimestampData.fromEpochMillis(windowEnd),
                        timeAttributePrecision);
            } else if (windowPropertyKinds[i]
                    == CudfTumblingWindowOperator.PROPERTY_ROWTIME) {
                reusableOutputWriter.writeTimestamp(
                        outputOrdinal,
                        TimestampData.fromEpochMillis(windowEnd - 1L),
                        timeAttributePrecision);
            } else {
                throw new IllegalStateException(
                        "unsupported global Window property "
                                + windowPropertyKinds[i]);
            }
        }

        reusableOutputWriter.complete();
        reusableOutputRow.setRowKind(RowKind.INSERT);
        output.collect(reusableOutputRecord.replace(reusableOutputRow, windowEnd - 1L));
        outputCounter.inc();
    }

    @Override
    public void onProcessingTime(InternalTimer<RowData, Long> timer) {
        throw new IllegalStateException("unexpected processing-time global Window timer");
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        emitClosedCachedWindows(mark.getTimestamp());
        super.processWatermark(mark);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        drainPartialCache();
        super.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void endInput() throws Exception {
        drainPartialCache();
    }

    @Override
    public void finish() throws Exception {
        drainPartialCache();
        super.finish();
    }

    @Override
    public void close() throws Exception {
        try {
            if (partialCache != null) {
                partialCache.clear();
                partialCache = null;
            }
            if (keyCache != null) {
                keyCache.clear();
                keyCache = null;
            }
        } finally {
            super.close();
        }
    }

    private static final class WindowKey {
        private final long windowEnd;
        private final Object[] groupValues;
        private final int hash;

        private WindowKey(long windowEnd, Object[] groupValues) {
            this.windowEnd = windowEnd;
            this.groupValues = groupValues;
            this.hash = 31 * Long.hashCode(windowEnd) + Arrays.deepHashCode(groupValues);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof WindowKey)) {
                return false;
            }
            WindowKey that = (WindowKey) other;
            return windowEnd == that.windowEnd
                    && Arrays.deepEquals(groupValues, that.groupValues);
        }
    }
}
