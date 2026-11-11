// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.io.InputFormat;
import org.apache.flink.api.common.operators.ProcessingTimeService.ProcessingTimeCallback;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.metrics.Counter;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.BoundedOneInput;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.api.watermark.Watermark;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.utils.JoinedRowData;
import org.apache.flink.table.runtime.typeutils.RowDataSerializer;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.RowKind;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ScheduledFuture;

/**
 * Product operator for a bounded, immutable, unique BIGINT-key Lookup Join snapshot.
 *
 * <p>Checkpoint barriers drain probe rows before operator state is snapshotted. Union operator
 * state stores an identity plus a SHA-256 content digest; every restored subtask reloads the
 * connector snapshot and refuses to activate it if any previous replica or the new content differs.
 */
public final class GpuBoundedLookupJoinOperator extends AbstractGpuStreamOperator<RowData>
        implements OneInputStreamOperator<RowData, RowData>,
                BoundedOneInput,
                ProcessingTimeCallback {
    private static final long serialVersionUID = 1L;
    private static final String SNAPSHOT_STATE_NAME = "cudfLookupJoinDimensionSnapshotV1";

    private final InputFormat<RowData, ?> dimensionInputFormat;
    private final String snapshotIdentity;
    private final RowType inputType;
    private final int probeKeyIndex;
    private final LookupJoinKernel.JoinType joinType;
    private final int batchRows;
    private final int gpuMinimumBatchRows;
    private final long maximumWaitMs;
    private final int maximumDimensionRows;

    private transient RowDataSerializer inputSerializer;
    private transient LookupJoinSnapshot activeSnapshot;
    private transient LookupJoinSnapshotActivation snapshotActivation;
    private transient LookupJoinKernel kernel;
    private transient List<BufferedProbe> buffered;
    private transient StreamRecord<RowData> reusableOutputRecord;
    private transient long activeTimer;
    private transient ScheduledFuture<?> activeTimerFuture;
    private transient ListState<String> snapshotState;
    private transient List<String> restoredMarkers;
    private transient boolean restored;
    private transient Counter inputCounter;
    private transient Counter outputCounter;
    private transient Counter cpuBatchCounter;
    private transient Counter gpuBatchCounter;
    private transient Counter checkpointDrainCounter;

    public GpuBoundedLookupJoinOperator(
            InputFormat<RowData, ?> dimensionInputFormat,
            String snapshotIdentity,
            RowType inputType,
            int probeKeyIndex,
            LookupJoinKernel.JoinType joinType,
            int batchRows,
            int gpuMinimumBatchRows,
            long maximumWaitMs,
            int maximumDimensionRows) {
        this.dimensionInputFormat =
                Objects.requireNonNull(dimensionInputFormat, "dimensionInputFormat");
        this.snapshotIdentity =
                Objects.requireNonNull(snapshotIdentity, "snapshotIdentity").trim();
        this.inputType = Objects.requireNonNull(inputType, "inputType");
        this.joinType = Objects.requireNonNull(joinType, "joinType");
        if (this.snapshotIdentity.isEmpty()) {
            throw new IllegalArgumentException("Lookup Join snapshot identity must be non-empty");
        }
        if (probeKeyIndex < 0 || probeKeyIndex >= inputType.getFieldCount()) {
            throw new IllegalArgumentException("Lookup Join probe key index is out of range");
        }
        if (batchRows <= 0
                || gpuMinimumBatchRows <= 0
                || gpuMinimumBatchRows > batchRows
                || maximumWaitMs <= 0L
                || maximumDimensionRows <= 0) {
            throw new IllegalArgumentException("invalid bounded Lookup Join configuration");
        }
        this.probeKeyIndex = probeKeyIndex;
        this.batchRows = batchRows;
        this.gpuMinimumBatchRows = gpuMinimumBatchRows;
        this.maximumWaitMs = maximumWaitMs;
        this.maximumDimensionRows = maximumDimensionRows;
    }

    @Override
    public void open() throws Exception {
        super.open();
        inputSerializer = new RowDataSerializer(inputType);
        snapshotActivation = new LookupJoinSnapshotActivation();
        buffered = new ArrayList<>(batchRows);
        reusableOutputRecord = new StreamRecord<>(null);
        activeTimer = Long.MIN_VALUE;
        activeTimerFuture = null;

        LookupJoinSnapshot loaded =
                LookupJoinSnapshotLoader.load(
                        dimensionInputFormat,
                        snapshotIdentity,
                        maximumDimensionRows,
                        getRuntimeContext(),
                        getRuntimeContext().getUserCodeClassLoader());
        if (restored && (restoredMarkers == null || restoredMarkers.isEmpty())) {
            throw new IllegalStateException(
                    "Lookup Join recovery has no dimension snapshot checkpoint marker");
        }
        LookupJoinSnapshot.requireCompatibleRestore(
                restoredMarkers == null ? Collections.emptyList() : restoredMarkers,
                loaded.checkpointMarker());
        activateSnapshot(loaded);

        inputCounter = getRuntimeContext().getMetricGroup().counter("cudfLookupJoinInputRows");
        outputCounter = getRuntimeContext().getMetricGroup().counter("cudfLookupJoinOutputRows");
        cpuBatchCounter = getRuntimeContext().getMetricGroup().counter("cudfLookupJoinCpuBatches");
        gpuBatchCounter = getRuntimeContext().getMetricGroup().counter("cudfLookupJoinGpuBatches");
        checkpointDrainCounter =
                getRuntimeContext().getMetricGroup().counter("cudfLookupJoinCheckpointDrains");
        final int snapshotRows = activeSnapshot.rowCount();
        final int matchableRows = activeSnapshot.matchableRowCount();
        final long generation = kernel.lastBuildMetrics().generation();
        getRuntimeContext().getMetricGroup().gauge("cudfLookupJoinDimensionRows", () -> snapshotRows);
        getRuntimeContext()
                .getMetricGroup()
                .gauge("cudfLookupJoinMatchableDimensionRows", () -> matchableRows);
        getRuntimeContext()
                .getMetricGroup()
                .gauge("cudfLookupJoinDimensionGeneration", () -> generation);
    }

    /** Installs only after the native inactive table has rebuilt successfully. */
    private void activateSnapshot(LookupJoinSnapshot candidate) {
        LookupJoinKernel target = kernel;
        boolean newlyAllocated = false;
        if (target == null) {
            target = new LookupJoinKernel(maximumDimensionRows, batchRows);
            newlyAllocated = true;
        }
        try {
            final LookupJoinKernel rebuildTarget = target;
            snapshotActivation.activate(
                    candidate,
                    snapshot ->
                            rebuildTarget.rebuild(
                                    snapshot.copyKeys(),
                                    snapshot.copyKeyValidity(),
                                    snapshot.copyPayloads(),
                                    snapshot.copyPayloadValidity()));
        } catch (RuntimeException | Error failure) {
            if (newlyAllocated) {
                target.close();
            }
            throw failure;
        }
        // LookupJoinKernel rebuilds an inactive table and switches its generation only on success.
        // Assigning these references last preserves the previous active snapshot on every failure.
        kernel = target;
        activeSnapshot = snapshotActivation.active();
    }

    @Override
    public void processElement(StreamRecord<RowData> element) throws Exception {
        RowData row = element.getValue();
        if (row.getRowKind() != RowKind.INSERT) {
            throw new IllegalArgumentException(
                    "bounded Lookup Join accepts INSERT-only input, got " + row.getRowKind());
        }
        if (buffered.size() >= batchRows) {
            flush();
        }
        RowData copy = inputSerializer.copy(row);
        buffered.add(
                new BufferedProbe(
                        copy,
                        element.hasTimestamp(),
                        element.hasTimestamp() ? element.getTimestamp() : 0L));
        inputCounter.inc();
        if (buffered.size() == 1) {
            activeTimer = getProcessingTimeService().getCurrentProcessingTime() + maximumWaitMs;
            activeTimerFuture = getProcessingTimeService().registerTimer(activeTimer, this);
        }
        if (buffered.size() >= batchRows) {
            flush();
        }
    }

    private void flush() throws Exception {
        if (buffered == null || buffered.isEmpty()) {
            return;
        }
        cancelTimer();
        List<BufferedProbe> batch = buffered;
        if (batch.size() >= gpuMinimumBatchRows) {
            emitNative(batch);
            gpuBatchCounter.inc();
        } else {
            emitCpu(batch);
            cpuBatchCounter.inc();
        }
        buffered.clear();
    }

    private void emitNative(List<BufferedProbe> batch) {
        long[] keys = new long[batch.size()];
        boolean[] keyValid = new boolean[batch.size()];
        for (int row = 0; row < batch.size(); row++) {
            RowData input = batch.get(row).row;
            keyValid[row] = !input.isNullAt(probeKeyIndex);
            keys[row] = keyValid[row] ? input.getLong(probeKeyIndex) : 0L;
        }
        kernel.probeInto(
                keys,
                keyValid,
                joinType,
                LookupJoinKernel.ProbeFilter.none(),
                LookupJoinKernel.ProbeTransfer.all(),
                (probeIndex,
                                probeKey,
                                probeKeyValid,
                                matched,
                                dimensionPayload,
                                dimensionPayloadValid) ->
                        emit(
                                batch.get(probeIndex),
                                matched,
                                probeKey,
                                dimensionPayload,
                                dimensionPayloadValid));
    }

    private void emitCpu(List<BufferedProbe> batch) {
        for (BufferedProbe probe : batch) {
            boolean keyValid = !probe.row.isNullAt(probeKeyIndex);
            long key = keyValid ? probe.row.getLong(probeKeyIndex) : 0L;
            LookupJoinSnapshot.Match match = activeSnapshot.lookup(key, keyValid);
            if (match == null && joinType == LookupJoinKernel.JoinType.INNER) {
                continue;
            }
            emit(
                    probe,
                    match != null,
                    key,
                    match == null ? 0L : match.payload(),
                    match != null && match.payloadValid());
        }
    }

    private void emit(
            BufferedProbe probe,
            boolean matched,
            long key,
            long payload,
            boolean payloadValid) {
        GenericRowData right = new GenericRowData(2);
        if (matched) {
            right.setField(0, key);
            right.setField(1, payloadValid ? payload : null);
        } else {
            right.setField(0, null);
            right.setField(1, null);
        }
        JoinedRowData joined = new JoinedRowData(RowKind.INSERT, probe.row, right);
        output.collect(
                probe.hasTimestamp
                        ? reusableOutputRecord.replace(joined, probe.timestamp)
                        : reusableOutputRecord.replace(joined));
        outputCounter.inc();
    }

    @Override
    public void onProcessingTime(long timestamp) throws Exception {
        if (timestamp == activeTimer) {
            activeTimer = Long.MIN_VALUE;
            activeTimerFuture = null;
            flush();
        }
    }

    private void cancelTimer() {
        if (activeTimerFuture != null) {
            activeTimerFuture.cancel(false);
            activeTimerFuture = null;
        }
        activeTimer = Long.MIN_VALUE;
    }

    @Override
    public void processWatermark(Watermark mark) throws Exception {
        flush();
        output.emitWatermark(mark);
    }

    @Override
    public void prepareSnapshotPreBarrier(long checkpointId) throws Exception {
        if (buffered != null && !buffered.isEmpty()) {
            checkpointDrainCounter.inc();
        }
        flush();
        super.prepareSnapshotPreBarrier(checkpointId);
    }

    @Override
    public void snapshotState(StateSnapshotContext context) throws Exception {
        if (activeSnapshot == null) {
            throw new IllegalStateException("Lookup Join cannot checkpoint before snapshot activation");
        }
        if (buffered != null && !buffered.isEmpty()) {
            throw new IllegalStateException("Lookup Join checkpoint captured undrained probe rows");
        }
        snapshotState.clear();
        snapshotState.add(activeSnapshot.checkpointMarker());
        super.snapshotState(context);
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
        snapshotState =
                context.getOperatorStateStore()
                        .getUnionListState(
                                new ListStateDescriptor<>(SNAPSHOT_STATE_NAME, String.class));
        restored = context.isRestored();
        restoredMarkers = new ArrayList<>();
        if (restored) {
            for (String marker : snapshotState.get()) {
                restoredMarkers.add(marker);
            }
        }
    }

    @Override
    public void endInput() throws Exception {
        flush();
    }

    @Override
    public void finish() throws Exception {
        flush();
        super.finish();
    }

    @Override
    public void close() throws Exception {
        try {
            cancelTimer();
            if (kernel != null) {
                kernel.close();
                kernel = null;
            }
            activeSnapshot = null;
            snapshotActivation = null;
            buffered = null;
            reusableOutputRecord = null;
        } finally {
            super.close();
        }
    }

    private static final class BufferedProbe {
        private final RowData row;
        private final boolean hasTimestamp;
        private final long timestamp;

        private BufferedProbe(RowData row, boolean hasTimestamp, long timestamp) {
            this.row = row;
            this.hasTimestamp = hasTimestamp;
            this.timestamp = timestamp;
        }
    }
}
