// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.operators.ProcessingTimeService.ProcessingTimeCallback;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.event.WatermarkEvent;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.LatencyMarker;
import org.apache.flink.streaming.runtime.streamrecord.RecordAttributes;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.watermarkstatus.WatermarkStatus;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.runtime.operators.window.tvf.common.ClockService;
import org.apache.flink.table.runtime.operators.window.tvf.slicing.SliceAssigner;
import org.apache.flink.types.RowKind;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ScheduledFuture;

/** Stateless GPU local combiner for Flink's two-phase event-time Window Aggregate. */
public final class GpuLocalWindowAggregateOperator extends AbstractGpuStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>,
                BoundedOneInput,
                ProcessingTimeCallback {
    private static final long serialVersionUID = 1L;
    private static final ClockService CLOCK_SERVICE = ClockService.ofSystem();
    private static final String PARTIAL_CACHE_ENTRIES_PROPERTY =
            "cudf.flink.local-window.partial-cache.entries";
    private static final int DEFAULT_PARTIAL_CACHE_ENTRIES = 65_536;

    private final GpuLocalWindowAggregatePlan plan;
    private final SliceAssigner sliceAssigner;
    private final int batchRows;
    private final long maxWaitMs;

    private transient DenseGroupbyKernel denseKernel;
    private transient long activeTimer;
    private transient ScheduledFuture<?> activeTimerFuture;
    private transient Counter inputCounter;
    private transient Counter outputCounter;
    private transient Counter batchCounter;
    private transient StreamRecord<RowData> reusableOutputRecord;
    private transient BinaryRowData reusableOutputRow;
    private transient BinaryRowWriter reusableOutputWriter;
    private transient LinkedHashMap<SliceKey, WindowAggregatePartial> partialCache;
    private transient int maxPartialCacheEntries;

    public GpuLocalWindowAggregateOperator(
            GpuLocalWindowAggregatePlan plan,
            SliceAssigner sliceAssigner,
            int batchRows,
            long maxWaitMs) {
        this.plan = Objects.requireNonNull(plan);
        this.sliceAssigner = Objects.requireNonNull(sliceAssigner);
        if (batchRows <= 0 || maxWaitMs <= 0L) {
            throw new IllegalArgumentException("batchRows and maxWaitMs must be positive");
        }
        this.batchRows = batchRows;
        this.maxWaitMs = maxWaitMs;
    }

    @Override
    public void open() throws Exception {
        super.open();
        denseKernel = DenseGroupbyKernel.createRequired(plan.computePlan(), batchRows);
        activeTimer = Long.MIN_VALUE;
        inputCounter =
                getRuntimeContext().getMetricGroup().counter("cudfLocalWindowInputRows");
        outputCounter =
                getRuntimeContext().getMetricGroup().counter("cudfLocalWindowPartialRows");
        batchCounter =
                getRuntimeContext().getMetricGroup().counter("cudfLocalWindowBatches");
        reusableOutputRecord = new StreamRecord<>(null);
        reusableOutputRow = new BinaryRowData(plan.outputArity());
        reusableOutputWriter = new BinaryRowWriter(reusableOutputRow);
        partialCache = new LinkedHashMap<>();
        maxPartialCacheEntries =
                Integer.getInteger(
                        PARTIAL_CACHE_ENTRIES_PROPERTY,
                        DEFAULT_PARTIAL_CACHE_ENTRIES);
        if (maxPartialCacheEntries <= 0) {
            throw new IllegalArgumentException(
                    PARTIAL_CACHE_ENTRIES_PROPERTY + " must be positive");
        }
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (row.getRowKind() != RowKind.INSERT) {
            throw new IllegalArgumentException(
                    "GPU local Window Aggregate accepts INSERT-only input, got "
                            + row.getRowKind());
        }
        long copyStarted = AggregateRuntimeMetrics.phaseStart();
        long sliceEnd = sliceAssigner.assignSliceEnd(row, CLOCK_SERVICE);
        denseKernel.appendWindow(row, sliceEnd);
        AggregateRuntimeMetrics.recordInputCopy(copyStarted);
        inputCounter.inc();
        if (denseKernel.bufferedRows() == 1) {
            activeTimer = getProcessingTimeService().getCurrentProcessingTime() + maxWaitMs;
            activeTimerFuture = getProcessingTimeService().registerTimer(activeTimer, this);
        }
        if (denseKernel.bufferedRows() >= batchRows) {
            flush();
        }
    }

    private void flush() throws Exception {
        if (denseKernel.bufferedRows() == 0) {
            return;
        }
        cancelTimer();
        List<WindowAggregatePartial> partials = denseKernel.executeBufferedWindows();
        if (partials == null) {
            partials =
                    CpuAggregateFallback.executeWindows(
                            plan.computePlan(), denseKernel.fallbackWindowRows());
        }
        denseKernel.resetBatch();
        batchCounter.inc();
        for (WindowAggregatePartial partial : partials) {
            if (partial.inputRows <= 0L) {
                throw new IllegalStateException(
                        "local Window Aggregate produced an empty partial slice");
            }
            SliceKey probe = new SliceKey(partial.windowEnd, partial.groupValues);
            WindowAggregatePartial cached = partialCache.get(probe);
            if (cached == null) {
                WindowAggregatePartial stable = plan.stableCopy(partial);
                partialCache.put(
                        new SliceKey(stable.windowEnd, stable.groupValues), stable);
            } else {
                partialCache.put(probe, plan.merge(cached, partial));
            }
        }
        if (partialCache.size() >= maxPartialCacheEntries) {
            drainPartialCache();
        }
    }

    private void drainPartialCache() {
        if (partialCache.isEmpty()) {
            return;
        }
        long postKernelStarted = AggregateRuntimeMetrics.phaseStart();
        for (WindowAggregatePartial partial : partialCache.values()) {
            reusableOutputWriter.reset();
            plan.writePartial(partial, reusableOutputWriter);
            reusableOutputWriter.complete();
            reusableOutputRow.setRowKind(RowKind.INSERT);
            output.collect(reusableOutputRecord.replace(reusableOutputRow));
            outputCounter.inc();
        }
        partialCache.clear();
        AggregateRuntimeMetrics.recordPostKernel(postKernelStarted);
    }

    private void cancelTimer() {
        if (activeTimerFuture != null) {
            activeTimerFuture.cancel(false);
            activeTimerFuture = null;
        }
        activeTimer = Long.MIN_VALUE;
    }

    @Override
    public void onProcessingTime(long timestamp) throws Exception {
        if (timestamp == activeTimer) {
            activeTimerFuture = null;
            activeTimer = Long.MIN_VALUE;
            flush();
        }
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        flush();
        drainPartialCache();
        super.processWatermark(mark);
    }

    @Override
    public void processWatermarkStatus(WatermarkStatus watermarkStatus) throws Exception {
        flush();
        drainPartialCache();
        super.processWatermarkStatus(watermarkStatus);
    }

    @Override
    public void processLatencyMarker(LatencyMarker latencyMarker) throws Exception {
        flush();
        drainPartialCache();
        super.processLatencyMarker(latencyMarker);
    }

    @Override
    public void processRecordAttributes(RecordAttributes recordAttributes) throws Exception {
        flush();
        drainPartialCache();
        super.processRecordAttributes(recordAttributes);
    }

    @Override
    public void processWatermark(WatermarkEvent watermarkEvent) throws Exception {
        flush();
        drainPartialCache();
        super.processWatermark(watermarkEvent);
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
            cancelTimer();
            if (denseKernel != null) {
                denseKernel.close();
                denseKernel = null;
            }
            if (partialCache != null) {
                partialCache.clear();
            partialCache = null;
            }
        } finally {
            super.close();
        }
    }

    private static final class SliceKey {
        private final long sliceEnd;
        private final Object[] groupValues;
        private final int hash;

        private SliceKey(long sliceEnd, Object[] groupValues) {
            this.sliceEnd = sliceEnd;
            this.groupValues = groupValues;
            this.hash = 31 * Long.hashCode(sliceEnd) + Arrays.deepHashCode(groupValues);
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof SliceKey)) {
                return false;
            }
            SliceKey that = (SliceKey) other;
            return sliceEnd == that.sliceEnd
                    && Arrays.deepEquals(groupValues, that.groupValues);
        }
    }
}
