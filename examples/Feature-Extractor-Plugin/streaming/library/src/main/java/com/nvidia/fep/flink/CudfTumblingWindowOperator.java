// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.operators.ProcessingTimeService.ProcessingTimeCallback;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeutils.base.LongSerializer;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.state.internal.InternalValueState;
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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ScheduledFuture;

/** Event-time TUMBLE/HOP/CUMULATE Aggregate using project-owned CUDA partials and Flink state. */
public final class CudfTumblingWindowOperator extends AbstractGpuStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>,
                BoundedOneInput,
                ProcessingTimeCallback,
                Triggerable<RowData, Long> {
    private static final long serialVersionUID = 1L;
    public static final int PROPERTY_START = 0;
    public static final int PROPERTY_END = 1;
    public static final int PROPERTY_ROWTIME = 2;
    private static final String PARTIAL_CACHE_ENTRIES_PROPERTY =
            "cudf.flink.window.partial-cache.entries";
    private static final int DEFAULT_PARTIAL_CACHE_ENTRIES = 65_536;

    private final GpuAggregatePlan plan;
    private final int batchRows;
    private final long maxWaitMs;
    private final int timeAttributeIndex;
    private final int timeAttributePrecision;
    private final long windowSizeMs;
    private final long windowSlideMs;
    private final long windowOffsetMs;
    private final boolean cumulativeWindow;
    private final int[] windowPropertyKinds;
    private final int outputArity;
    private final RowDataKeySelector keySelector;
    private final InternalTypeInfo<RowData> stateTypeInfo;

    private transient DenseGroupbyKernel denseKernel;
    private transient AggregateKeyCache keyCache;
    private transient InternalValueState<RowData, Long, RowData> windowState;
    private transient InternalValueState<RowData, Long, AggregateAverageState> averageState;
    private transient InternalTimerService<Long> timerService;
    private transient long activeBatchTimer;
    private transient ScheduledFuture<?> activeBatchTimerFuture;
    private transient Counter inputCounter;
    private transient Counter outputCounter;
    private transient Counter lateCounter;
    private transient StreamRecord<RowData> reusableOutputRecord;
    private transient BinaryRowData reusableWindowOutput;
    private transient BinaryRowWriter reusableWindowWriter;
    private transient long[] reusableWindowEnds;
    private transient LinkedHashMap<WindowCacheKey, CachedWindowPartial> partialCache;
    private transient int maxPartialCacheEntries;

    public CudfTumblingWindowOperator(
            ExecutionMode mode,
            GpuAggregatePlan plan,
            int batchRows,
            long maxWaitMs,
            int timeAttributeIndex,
            int timeAttributePrecision,
            long windowSizeMs,
            long windowSlideMs,
            long windowOffsetMs,
            boolean cumulativeWindow,
            int[] windowPropertyKinds,
            int outputArity,
            RowDataKeySelector keySelector,
            InternalTypeInfo<RowData> stateTypeInfo) {
        if (mode != ExecutionMode.GPU_NATIVE) {
            throw new IllegalArgumentException("Window operator requires gpu-native mode");
        }
        if (batchRows <= 0 || maxWaitMs <= 0 || windowSizeMs <= 0 || windowSlideMs <= 0) {
            throw new IllegalArgumentException("Invalid batch/window configuration");
        }
        this.plan = Objects.requireNonNull(plan);
        long windowsPerInput = Math.floorDiv(windowSizeMs - 1L, windowSlideMs) + 1L;
        if (windowsPerInput > 64L) {
            throw new IllegalArgumentException("window produces more than 64 contributions per row");
        }
        if (cumulativeWindow && windowSizeMs % windowSlideMs != 0L) {
            throw new IllegalArgumentException("CUMULATE max size must be a multiple of step");
        }
        long expandedBatchRows = Math.multiplyExact((long) batchRows, windowsPerInput);
        if (expandedBatchRows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("expanded window batch exceeds integer capacity");
        }
        // The public batch setting counts source rows. HOP/CUMULATE can expand one source row into
        // several window contributions; sizing the native buffer by the maximum factor keeps its
        // batching policy comparable with TUMBLE and with Flink's CPU input mini-batch.
        this.batchRows = (int) expandedBatchRows;
        this.maxWaitMs = maxWaitMs;
        this.timeAttributeIndex = timeAttributeIndex;
        this.timeAttributePrecision = timeAttributePrecision;
        this.windowSizeMs = windowSizeMs;
        this.windowSlideMs = windowSlideMs;
        this.windowOffsetMs = windowOffsetMs;
        this.cumulativeWindow = cumulativeWindow;
        this.windowPropertyKinds = windowPropertyKinds.clone();
        this.outputArity = outputArity;
        this.keySelector = Objects.requireNonNull(keySelector);
        this.stateTypeInfo = Objects.requireNonNull(stateTypeInfo);
    }

    @Override
    public void open() throws Exception {
        super.open();
        denseKernel = DenseGroupbyKernel.createRequired(plan, batchRows);
        keyCache = new AggregateKeyCache(plan, keySelector);
        activeBatchTimer = Long.MIN_VALUE;
        ValueStateDescriptor<RowData> descriptor =
                new ValueStateDescriptor<>("cudfTumblingWindowStateV2", stateTypeInfo);
        @SuppressWarnings("unchecked")
        InternalValueState<RowData, Long, RowData> namespacedState =
                (InternalValueState<RowData, Long, RowData>)
                        (ValueState<RowData>)
                                getOrCreateKeyedState(LongSerializer.INSTANCE, descriptor);
        windowState = namespacedState;
        if (plan.hasAverage()) {
            ValueStateDescriptor<AggregateAverageState> averageDescriptor =
                    new ValueStateDescriptor<>(
                            "cudfWindowAverageStateV1", AggregateAverageState.class);
            @SuppressWarnings("unchecked")
            InternalValueState<RowData, Long, AggregateAverageState> namespacedAverageState =
                    (InternalValueState<RowData, Long, AggregateAverageState>)
                            (ValueState<AggregateAverageState>)
                                    getOrCreateKeyedState(
                                            LongSerializer.INSTANCE, averageDescriptor);
            averageState = namespacedAverageState;
        }
        timerService =
                getInternalTimerService(
                        "cudfTumblingWindowTimers", LongSerializer.INSTANCE, this);
        inputCounter = getRuntimeContext().getMetricGroup().counter("cudfWindowInputRows");
        outputCounter = getRuntimeContext().getMetricGroup().counter("cudfWindowOutputRows");
        lateCounter = getRuntimeContext().getMetricGroup().counter("cudfWindowLateRowsDropped");
        reusableOutputRecord = new StreamRecord<>(null);
        reusableWindowOutput = new BinaryRowData(outputArity);
        reusableWindowWriter = new BinaryRowWriter(reusableWindowOutput);
        reusableWindowEnds = new long[64];
        maxPartialCacheEntries =
                Integer.getInteger(
                        PARTIAL_CACHE_ENTRIES_PROPERTY, DEFAULT_PARTIAL_CACHE_ENTRIES);
        if (maxPartialCacheEntries <= 0) {
            throw new IllegalArgumentException(
                    PARTIAL_CACHE_ENTRIES_PROPERTY + " must be positive");
        }
        partialCache = new LinkedHashMap<>();
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (row.getRowKind() != RowKind.INSERT) {
            throw new IllegalArgumentException(
                    "GPU Window Aggregate accepts INSERT-only input, got " + row.getRowKind());
        }
        if (row.isNullAt(timeAttributeIndex)) {
            throw new IllegalArgumentException("Window time attribute must not be NULL");
        }
        long inputCopyStarted = AggregateRuntimeMetrics.phaseStart();
        long timestamp =
                row.getTimestamp(timeAttributeIndex, timeAttributePrecision).getMillisecond();
        int accepted = 0;
        if (cumulativeWindow) {
            long windowStart =
                    Math.floorDiv(timestamp - windowOffsetMs, windowSizeMs) * windowSizeMs
                            + windowOffsetMs;
            long firstStep = Math.floorDiv(timestamp - windowStart, windowSlideMs) + 1L;
            long windowsPerRow = windowSizeMs / windowSlideMs;
            for (long step = firstStep; step <= windowsPerRow; step++) {
                long windowEnd =
                        Math.addExact(
                                windowStart, Math.multiplyExact(step, windowSlideMs));
                if (timerService.currentWatermark() >= windowEnd - 1L) {
                    lateCounter.inc();
                    AggregateRuntimeMetrics.recordWindowLateContribution();
                    continue;
                }
                reusableWindowEnds[accepted++] = windowEnd;
            }
        } else {
            long lastWindowStart =
                    Math.floorDiv(timestamp - windowOffsetMs, windowSlideMs) * windowSlideMs
                            + windowOffsetMs;
            long windowsPerRow = Math.floorDiv(windowSizeMs - 1L, windowSlideMs) + 1L;
            for (long window = 0L; window < windowsPerRow; window++) {
                long windowStart =
                        Math.subtractExact(
                                lastWindowStart, Math.multiplyExact(window, windowSlideMs));
                long windowEnd = Math.addExact(windowStart, windowSizeMs);
                if (timestamp >= windowEnd) {
                    continue;
                }
                if (timerService.currentWatermark() >= windowEnd - 1L) {
                    lateCounter.inc();
                    AggregateRuntimeMetrics.recordWindowLateContribution();
                    continue;
                }
                reusableWindowEnds[accepted++] = windowEnd;
            }
        }
        int offset = 0;
        while (offset < accepted) {
            if (bufferSize() >= batchRows) {
                flush();
            }
            int append = Math.min(accepted - offset, batchRows - bufferSize());
            boolean wasEmpty = bufferSize() == 0;
            denseKernel.appendWindows(row, reusableWindowEnds, offset, append);
            offset += append;
            if (wasEmpty) {
                activeBatchTimer =
                        getProcessingTimeService().getCurrentProcessingTime() + maxWaitMs;
                activeBatchTimerFuture =
                        getProcessingTimeService().registerTimer(activeBatchTimer, this);
            }
        }
        AggregateRuntimeMetrics.recordInputCopy(inputCopyStarted);
        inputCounter.inc();
        if (accepted > 0 && bufferSize() >= batchRows) {
            flush();
        }
    }

    private void runDense() throws Exception {
        List<WindowAggregatePartial> partials = denseKernel.executeBufferedWindows();
        if (partials == null) {
            List<WindowBufferedRow> fallbackRows = denseKernel.fallbackWindowRows();
            partials = CpuAggregateFallback.executeWindows(plan, fallbackRows);
        }
        denseKernel.resetBatch();
        processPartials(partials);
    }

    private void processPartials(List<WindowAggregatePartial> partials) throws Exception {
        long postKernelStarted = AggregateRuntimeMetrics.phaseStart();
        for (WindowAggregatePartial windowPartial : partials) {
            WindowCacheKey probe =
                    new WindowCacheKey(
                            windowPartial.windowEnd, windowPartial.groupValues);
            CachedWindowPartial cached = partialCache.get(probe);
            if (cached == null) {
                Object[] stableGroupValues = windowPartial.groupValues.clone();
                WindowCacheKey stableKey =
                        new WindowCacheKey(windowPartial.windowEnd, stableGroupValues);
                partialCache.put(
                        stableKey,
                        new CachedWindowPartial(
                                windowPartial.windowEnd,
                                stableGroupValues,
                                windowPartial.inputRows,
                                windowPartial.aggregateValues.clone()));
            } else {
                cached.merge(plan, windowPartial);
            }
        }
        AggregateRuntimeMetrics.recordPostKernel(postKernelStarted);
        if (partialCache.size() >= maxPartialCacheEntries) {
            drainPartialCache();
        }
    }

    /** Writes host-coalesced partials to Flink state before watermark/checkpoint visibility. */
    private void drainPartialCache() throws Exception {
        Iterator<Map.Entry<WindowCacheKey, CachedWindowPartial>> iterator =
                partialCache.entrySet().iterator();
        while (iterator.hasNext()) {
            CachedWindowPartial cached = iterator.next().getValue();
            long keySelectionStarted = AggregateRuntimeMetrics.phaseStart();
            RowData key = keyCache.keyFor(cached.groupValues);
            setCurrentKey(key);
            AggregateRuntimeMetrics.recordKeySelection(keySelectionStarted);
            mergeWindow(cached.toPartial());
            iterator.remove();
        }
    }

    /**
     * Emits complete windows straight from the coalesced host partials.
     *
     * <p>A watermark is an ordering boundary: after the buffered GPU work has finished, a cached
     * window whose end is covered by that watermark cannot receive another on-time row. Writing
     * such a partial to keyed state, registering a timer, immediately reading it back, and firing
     * the timer only adds Flink state/timer overhead. Future windows remain cached; checkpoints
     * still call {@link #drainPartialCache()} so recovery always has an authoritative state copy.
     */
    private void emitClosedCachedWindows(long watermark) throws Exception {
        Iterator<Map.Entry<WindowCacheKey, CachedWindowPartial>> iterator =
                partialCache.entrySet().iterator();
        while (iterator.hasNext()) {
            CachedWindowPartial cached = iterator.next().getValue();
            if (cached.windowEnd - 1L > watermark) {
                continue;
            }
            long finalizationStarted = AggregateRuntimeMetrics.phaseStart();
            long keySelectionStarted = AggregateRuntimeMetrics.phaseStart();
            RowData key = keyCache.keyFor(cached.groupValues);
            setCurrentKey(key);
            AggregateRuntimeMetrics.recordKeySelection(keySelectionStarted);

            long stateReadStarted = AggregateRuntimeMetrics.phaseStart();
            windowState.setCurrentNamespace(cached.windowEnd);
            if (averageState != null) {
                averageState.setCurrentNamespace(cached.windowEnd);
            }
            RowData oldState = windowState.value();
            AggregateAverageState oldAverageState =
                    averageState == null ? null : averageState.value();
            AggregateRuntimeMetrics.recordStateRead(stateReadStarted);

            long mergeStarted = AggregateRuntimeMetrics.phaseStart();
            Object[] values =
                    plan.mergeAccumulatorValues(
                            oldState, oldAverageState, cached.aggregateValues);
            GenericRowData finalState = plan.outputRow(cached.groupValues, values);
            finalState.setRowKind(RowKind.INSERT);
            AggregateRuntimeMetrics.recordMerge(mergeStarted);

            if (oldState != null) {
                long stateWriteStarted = AggregateRuntimeMetrics.phaseStart();
                windowState.clear();
                if (averageState != null) {
                    averageState.clear();
                }
                timerService.deleteEventTimeTimer(
                        cached.windowEnd, cached.windowEnd - 1L);
                AggregateRuntimeMetrics.recordStateWrite(stateWriteStarted);
            }
            emitWindow(finalState, cached.windowEnd);
            AggregateRuntimeMetrics.recordEventTimer(finalizationStarted);
            iterator.remove();
        }
    }

    private void mergeWindow(WindowAggregatePartial partial) throws Exception {
        long stateReadStarted = AggregateRuntimeMetrics.phaseStart();
        windowState.setCurrentNamespace(partial.windowEnd);
        if (averageState != null) {
            averageState.setCurrentNamespace(partial.windowEnd);
        }
        RowData oldState = windowState.value();
        AggregateAverageState oldAverageState =
                averageState == null ? null : averageState.value();
        AggregateRuntimeMetrics.recordStateRead(stateReadStarted);
        long mergeStarted = AggregateRuntimeMetrics.phaseStart();
        Object[] values =
                plan.mergeAccumulatorValues(
                        oldState, oldAverageState, partial.aggregateValues);
        GenericRowData newState = plan.outputRow(partial.groupValues, values);
        newState.setRowKind(RowKind.INSERT);
        AggregateRuntimeMetrics.recordMerge(mergeStarted);
        long stateWriteStarted = AggregateRuntimeMetrics.phaseStart();
        windowState.update(newState);
        if (averageState != null) {
            averageState.update(plan.averageState(values));
        }
        if (oldState == null) {
            timerService.registerEventTimeTimer(
                    partial.windowEnd, partial.windowEnd - 1L);
        }
        AggregateRuntimeMetrics.recordStateWrite(stateWriteStarted);
    }

    @Override
    public void onEventTime(InternalTimer<RowData, Long> timer) throws Exception {
        long eventTimerStarted = AggregateRuntimeMetrics.phaseStart();
        long windowEnd = timer.getNamespace();
        windowState.setCurrentNamespace(windowEnd);
        if (averageState != null) {
            averageState.setCurrentNamespace(windowEnd);
        }
        RowData state = windowState.value();
        if (state == null) {
            AggregateRuntimeMetrics.recordEventTimer(eventTimerStarted);
            return;
        }
        emitWindow(state, windowEnd);
        windowState.clear();
        if (averageState != null) {
            averageState.clear();
        }
        AggregateRuntimeMetrics.recordEventTimer(eventTimerStarted);
    }

    private void emitWindow(RowData state, long windowEnd) {
        reusableWindowWriter.reset();
        long windowStart = windowStart(windowEnd);
        plan.copyCoreOutput(state, reusableWindowWriter);
        int propertyStart = plan.outputTypes().length;
        for (int i = 0; i < windowPropertyKinds.length; i++) {
            if (windowPropertyKinds[i] == PROPERTY_START) {
                reusableWindowWriter.writeTimestamp(
                        i + propertyStart,
                        TimestampData.fromEpochMillis(windowStart),
                        timeAttributePrecision);
            } else if (windowPropertyKinds[i] == PROPERTY_END) {
                reusableWindowWriter.writeTimestamp(
                        i + propertyStart,
                        TimestampData.fromEpochMillis(windowEnd),
                        timeAttributePrecision);
            } else if (windowPropertyKinds[i] == PROPERTY_ROWTIME) {
                reusableWindowWriter.writeTimestamp(
                        i + propertyStart,
                        TimestampData.fromEpochMillis(windowEnd - 1L),
                        timeAttributePrecision);
            } else {
                throw new IllegalStateException(
                        "Unsupported window property kind " + windowPropertyKinds[i]);
            }
        }

        reusableWindowWriter.complete();
        reusableWindowOutput.setRowKind(RowKind.INSERT);
        long emitStarted = AggregateRuntimeMetrics.phaseStart();
        output.collect(reusableOutputRecord.replace(reusableWindowOutput, windowEnd - 1L));
        outputCounter.inc();
        AggregateRuntimeMetrics.recordEmit(emitStarted);
    }

    private long windowStart(long windowEnd) {
        if (!cumulativeWindow) {
            return windowEnd - windowSizeMs;
        }
        // The -1 keeps an end exactly on a max-window boundary in the preceding window cycle.
        return Math.floorDiv(windowEnd - windowOffsetMs - 1L, windowSizeMs) * windowSizeMs
                + windowOffsetMs;
    }

    @Override
    public void onProcessingTime(InternalTimer<RowData, Long> timer) {
        throw new IllegalStateException("Unexpected processing-time window timer");
    }

    @Override
    public void onProcessingTime(long timestamp) throws Exception {
        if (timestamp == activeBatchTimer) {
            activeBatchTimerFuture = null;
            activeBatchTimer = Long.MIN_VALUE;
            flush();
        }
    }

    private void flush() throws Exception {
        if (bufferSize() == 0) {
            return;
        }
        cancelBatchTimer();
        runDense();
    }

    private int bufferSize() {
        return denseKernel.bufferedRows();
    }

    private void cancelBatchTimer() {
        if (activeBatchTimerFuture != null) {
            activeBatchTimerFuture.cancel(false);
            activeBatchTimerFuture = null;
        }
        activeBatchTimer = Long.MIN_VALUE;
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        flush();
        emitClosedCachedWindows(mark.getTimestamp());
        super.processWatermark(mark);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        flush();
        drainPartialCache();
        super.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void endInput() throws Exception {
        flush();
        drainPartialCache();
    }

    @Override
    public void finish() throws Exception {
        flush();
        drainPartialCache();
        super.finish();
    }

    @Override
    public void close() throws Exception {
        try {
            cancelBatchTimer();
            if (denseKernel != null) {
                denseKernel.close();
                denseKernel = null;
            }
            if (keyCache != null) {
                keyCache.clear();
                keyCache = null;
            }
            if (partialCache != null) {
                partialCache.clear();
                partialCache = null;
            }
        } finally {
            super.close();
        }
    }

    private static final class WindowCacheKey {
        private final long windowEnd;
        private final Object[] groupValues;
        private final int hash;

        private WindowCacheKey(long windowEnd, Object[] groupValues) {
            this.windowEnd = windowEnd;
            this.groupValues = groupValues;
            this.hash = 31 * Long.hashCode(windowEnd) + Arrays.deepHashCode(groupValues);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof WindowCacheKey)) {
                return false;
            }
            WindowCacheKey that = (WindowCacheKey) other;
            return windowEnd == that.windowEnd
                    && Arrays.deepEquals(groupValues, that.groupValues);
        }

        @Override
        public int hashCode() {
            return hash;
        }
    }

    private static final class CachedWindowPartial {
        private final long windowEnd;
        private final Object[] groupValues;
        private long inputRows;
        private final Object[] aggregateValues;

        private CachedWindowPartial(
                long windowEnd,
                Object[] groupValues,
                long inputRows,
                Object[] aggregateValues) {
            this.windowEnd = windowEnd;
            this.groupValues = groupValues;
            this.inputRows = inputRows;
            this.aggregateValues = aggregateValues;
        }

        private void merge(GpuAggregatePlan plan, WindowAggregatePartial partial) {
            inputRows += partial.inputRows;
            for (int aggregate = 0; aggregate < aggregateValues.length; aggregate++) {
                aggregateValues[aggregate] =
                        plan.aggregates()[aggregate].merge(
                                aggregateValues[aggregate],
                                partial.aggregateValues[aggregate]);
            }
        }

        private WindowAggregatePartial toPartial() {
            return new WindowAggregatePartial(
                    windowEnd, groupValues, inputRows, aggregateValues);
        }
    }
}
