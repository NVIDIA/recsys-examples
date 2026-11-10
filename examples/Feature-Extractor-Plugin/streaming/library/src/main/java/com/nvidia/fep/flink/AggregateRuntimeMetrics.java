// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/** Process-local timing counters dedicated to the native GPU Group Aggregate operator. */
public final class AggregateRuntimeMetrics {
    public static final String PHASE_PROFILE_PROPERTY = "cudf.flink.profile.phases";
    private static final boolean PHASE_PROFILE_ENABLED =
            Boolean.getBoolean(PHASE_PROFILE_PROPERTY);

    private static final LongAdder INPUT_ROWS = new LongAdder();
    private static final LongAdder PARTIAL_GROUPS = new LongAdder();
    private static final LongAdder BATCHES = new LongAdder();
    private static final LongAdder ROW_TO_COLUMN_NANOS = new LongAdder();
    private static final LongAdder H2D_NANOS = new LongAdder();
    private static final LongAdder COMPUTE_NANOS = new LongAdder();
    private static final LongAdder D2H_NANOS = new LongAdder();
    private static final LongAdder COLUMN_TO_ROW_NANOS = new LongAdder();
    private static final LongAdder INPUT_COPY_NANOS = new LongAdder();
    private static final LongAdder POST_KERNEL_NANOS = new LongAdder();
    private static final LongAdder EVENT_TIMER_NANOS = new LongAdder();
    private static final LongAdder KEY_SELECTION_NANOS = new LongAdder();
    private static final LongAdder STATE_READ_NANOS = new LongAdder();
    private static final LongAdder MERGE_NANOS = new LongAdder();
    private static final LongAdder STATE_WRITE_NANOS = new LongAdder();
    private static final LongAdder EMIT_NANOS = new LongAdder();
    private static final LongAdder INPUT_DEVICE_BYTES = new LongAdder();
    private static final LongAdder OUTPUT_DEVICE_BYTES = new LongAdder();
    private static final LongAdder FUSED_CALC_NATIVE_FALLBACK_BATCHES = new LongAdder();
    private static final LongAdder PERSISTENT_COMMANDS = new LongAdder();
    private static final LongAdder PERSISTENT_ACK_RESPONSES = new LongAdder();
    private static final LongAdder PERSISTENT_MATERIALIZED_RESPONSES = new LongAdder();
    private static final LongAdder PERSISTENT_SNAPSHOT_CALLS = new LongAdder();
    private static final LongAdder PERSISTENT_SNAPSHOT_GROUPS = new LongAdder();
    private static final LongAdder PERSISTENT_SNAPSHOT_NANOS = new LongAdder();
    private static final LongAdder ADAPTIVE_CPU_ROWS = new LongAdder();
    private static final LongAdder ADAPTIVE_PERSISTENT_ROWS = new LongAdder();
    private static final LongAdder ADAPTIVE_BATCH_ROWS = new LongAdder();
    private static final LongAdder ADAPTIVE_SWITCHES = new LongAdder();
    private static final LongAdder ADAPTIVE_SWITCH_NANOS = new LongAdder();
    private static final LongAdder ADAPTIVE_BACKPRESSURE_SAMPLES = new LongAdder();
    private static final LongAdder SESSION_WINDOW_MERGES = new LongAdder();
    private static final LongAdder WINDOW_LATE_CONTRIBUTIONS = new LongAdder();
    private static final AtomicLong MAX_BATCH_ROWS = new AtomicLong();
    private static final AtomicLong MAX_INPUT_DEVICE_BYTES = new AtomicLong();
    private static final AtomicLong MAX_OUTPUT_DEVICE_BYTES = new AtomicLong();

    private AggregateRuntimeMetrics() {}

    public static void reset() {
        INPUT_ROWS.reset();
        PARTIAL_GROUPS.reset();
        BATCHES.reset();
        ROW_TO_COLUMN_NANOS.reset();
        H2D_NANOS.reset();
        COMPUTE_NANOS.reset();
        D2H_NANOS.reset();
        COLUMN_TO_ROW_NANOS.reset();
        INPUT_COPY_NANOS.reset();
        POST_KERNEL_NANOS.reset();
        EVENT_TIMER_NANOS.reset();
        KEY_SELECTION_NANOS.reset();
        STATE_READ_NANOS.reset();
        MERGE_NANOS.reset();
        STATE_WRITE_NANOS.reset();
        EMIT_NANOS.reset();
        INPUT_DEVICE_BYTES.reset();
        OUTPUT_DEVICE_BYTES.reset();
        FUSED_CALC_NATIVE_FALLBACK_BATCHES.reset();
        PERSISTENT_COMMANDS.reset();
        PERSISTENT_ACK_RESPONSES.reset();
        PERSISTENT_MATERIALIZED_RESPONSES.reset();
        PERSISTENT_SNAPSHOT_CALLS.reset();
        PERSISTENT_SNAPSHOT_GROUPS.reset();
        PERSISTENT_SNAPSHOT_NANOS.reset();
        ADAPTIVE_CPU_ROWS.reset();
        ADAPTIVE_PERSISTENT_ROWS.reset();
        ADAPTIVE_BATCH_ROWS.reset();
        ADAPTIVE_SWITCHES.reset();
        ADAPTIVE_SWITCH_NANOS.reset();
        ADAPTIVE_BACKPRESSURE_SAMPLES.reset();
        SESSION_WINDOW_MERGES.reset();
        WINDOW_LATE_CONTRIBUTIONS.reset();
        MAX_BATCH_ROWS.set(0);
        MAX_INPUT_DEVICE_BYTES.set(0);
        MAX_OUTPUT_DEVICE_BYTES.set(0);
    }

    static void record(
            int inputRows,
            int partialGroups,
            long rowToColumnNanos,
            long h2dNanos,
            long computeNanos,
            long d2hNanos,
            long columnToRowNanos,
            long inputDeviceBytes,
            long outputDeviceBytes) {
        INPUT_ROWS.add(inputRows);
        PARTIAL_GROUPS.add(partialGroups);
        BATCHES.increment();
        ROW_TO_COLUMN_NANOS.add(rowToColumnNanos);
        H2D_NANOS.add(h2dNanos);
        COMPUTE_NANOS.add(computeNanos);
        D2H_NANOS.add(d2hNanos);
        COLUMN_TO_ROW_NANOS.add(columnToRowNanos);
        INPUT_DEVICE_BYTES.add(inputDeviceBytes);
        OUTPUT_DEVICE_BYTES.add(outputDeviceBytes);
        MAX_BATCH_ROWS.accumulateAndGet(inputRows, Math::max);
        MAX_INPUT_DEVICE_BYTES.accumulateAndGet(inputDeviceBytes, Math::max);
        MAX_OUTPUT_DEVICE_BYTES.accumulateAndGet(outputDeviceBytes, Math::max);
    }

    static long phaseStart() {
        return PHASE_PROFILE_ENABLED ? System.nanoTime() : 0L;
    }

    static void recordInputCopy(long startedNanos) {
        recordSince(INPUT_COPY_NANOS, startedNanos);
    }

    static void recordPostKernel(long startedNanos) {
        recordSince(POST_KERNEL_NANOS, startedNanos);
    }

    static void recordEventTimer(long startedNanos) {
        recordSince(EVENT_TIMER_NANOS, startedNanos);
    }

    static void recordKeySelection(long startedNanos) {
        recordSince(KEY_SELECTION_NANOS, startedNanos);
    }

    static void recordStateRead(long startedNanos) {
        recordSince(STATE_READ_NANOS, startedNanos);
    }

    static void recordMerge(long startedNanos) {
        recordSince(MERGE_NANOS, startedNanos);
    }

    static void recordStateWrite(long startedNanos) {
        recordSince(STATE_WRITE_NANOS, startedNanos);
    }

    static void recordEmit(long startedNanos) {
        recordSince(EMIT_NANOS, startedNanos);
    }

    static void recordPersistentCommand() {
        PERSISTENT_COMMANDS.increment();
    }

    static void recordFusedCalcNativeFallbackBatch() {
        FUSED_CALC_NATIVE_FALLBACK_BATCHES.increment();
    }

    static void recordPersistentResponse(boolean materialized) {
        (materialized ? PERSISTENT_MATERIALIZED_RESPONSES : PERSISTENT_ACK_RESPONSES).increment();
    }

    static void recordPersistentSnapshot(int groups, long nanos) {
        PERSISTENT_SNAPSHOT_CALLS.increment();
        PERSISTENT_SNAPSHOT_GROUPS.add(groups);
        if (nanos > 0L) {
            PERSISTENT_SNAPSHOT_NANOS.add(nanos);
        }
    }

    static void recordAdaptiveSwitch(long nanos) {
        ADAPTIVE_SWITCHES.increment();
        ADAPTIVE_SWITCH_NANOS.add(nanos);
    }

    static void recordAdaptiveBackpressureSample() {
        ADAPTIVE_BACKPRESSURE_SAMPLES.increment();
    }

    static void recordSessionWindowMerge() {
        SESSION_WINDOW_MERGES.increment();
    }

    static void recordWindowLateContribution() {
        WINDOW_LATE_CONTRIBUTIONS.increment();
    }

    static void recordColumnToRow(long nanos) {
        if (nanos > 0L) {
            COLUMN_TO_ROW_NANOS.add(nanos);
        }
    }

    private static void recordSince(LongAdder target, long startedNanos) {
        if (startedNanos != 0L) {
            target.add(System.nanoTime() - startedNanos);
        }
    }

    public static Snapshot snapshot() {
        return new Snapshot(
                INPUT_ROWS.sum(),
                PARTIAL_GROUPS.sum(),
                BATCHES.sum(),
                ROW_TO_COLUMN_NANOS.sum(),
                H2D_NANOS.sum(),
                COMPUTE_NANOS.sum(),
                D2H_NANOS.sum(),
                COLUMN_TO_ROW_NANOS.sum(),
                INPUT_COPY_NANOS.sum(),
                POST_KERNEL_NANOS.sum(),
                EVENT_TIMER_NANOS.sum(),
                KEY_SELECTION_NANOS.sum(),
                STATE_READ_NANOS.sum(),
                MERGE_NANOS.sum(),
                STATE_WRITE_NANOS.sum(),
                EMIT_NANOS.sum(),
                INPUT_DEVICE_BYTES.sum(),
                OUTPUT_DEVICE_BYTES.sum(),
                FUSED_CALC_NATIVE_FALLBACK_BATCHES.sum(),
                PERSISTENT_COMMANDS.sum(),
                PERSISTENT_ACK_RESPONSES.sum(),
                PERSISTENT_MATERIALIZED_RESPONSES.sum(),
                PERSISTENT_SNAPSHOT_CALLS.sum(),
                PERSISTENT_SNAPSHOT_GROUPS.sum(),
                PERSISTENT_SNAPSHOT_NANOS.sum(),
                ADAPTIVE_CPU_ROWS.sum(),
                ADAPTIVE_PERSISTENT_ROWS.sum(),
                ADAPTIVE_BATCH_ROWS.sum(),
                ADAPTIVE_SWITCHES.sum(),
                ADAPTIVE_SWITCH_NANOS.sum(),
                ADAPTIVE_BACKPRESSURE_SAMPLES.sum(),
                SESSION_WINDOW_MERGES.sum(),
                WINDOW_LATE_CONTRIBUTIONS.sum(),
                MAX_BATCH_ROWS.get(),
                MAX_INPUT_DEVICE_BYTES.get(),
                MAX_OUTPUT_DEVICE_BYTES.get());
    }

    public static final class Snapshot {
        public final long inputRows;
        public final long partialGroups;
        public final long batches;
        public final long rowToColumnNanos;
        public final long h2dNanos;
        public final long computeNanos;
        public final long d2hNanos;
        public final long columnToRowNanos;
        public final long inputCopyNanos;
        public final long postKernelNanos;
        public final long eventTimerNanos;
        public final long keySelectionNanos;
        public final long stateReadNanos;
        public final long mergeNanos;
        public final long stateWriteNanos;
        public final long emitNanos;
        public final long inputDeviceBytes;
        public final long outputDeviceBytes;
        public final long fusedCalcNativeFallbackBatches;
        public final long persistentCommands;
        public final long persistentAckResponses;
        public final long persistentMaterializedResponses;
        public final long persistentSnapshotCalls;
        public final long persistentSnapshotGroups;
        public final long persistentSnapshotNanos;
        public final long adaptiveCpuRows;
        public final long adaptivePersistentRows;
        public final long adaptiveBatchRows;
        public final long adaptiveSwitches;
        public final long adaptiveSwitchNanos;
        public final long adaptiveBackpressureSamples;
        public final long sessionWindowMerges;
        public final long windowLateContributions;
        public final long maxBatchRows;
        public final long maxInputDeviceBytes;
        public final long maxOutputDeviceBytes;

        Snapshot(
                long inputRows,
                long partialGroups,
                long batches,
                long rowToColumnNanos,
                long h2dNanos,
                long computeNanos,
                long d2hNanos,
                long columnToRowNanos,
                long inputCopyNanos,
                long postKernelNanos,
                long eventTimerNanos,
                long keySelectionNanos,
                long stateReadNanos,
                long mergeNanos,
                long stateWriteNanos,
                long emitNanos,
                long inputDeviceBytes,
                long outputDeviceBytes,
                long fusedCalcNativeFallbackBatches,
                long persistentCommands,
                long persistentAckResponses,
                long persistentMaterializedResponses,
                long persistentSnapshotCalls,
                long persistentSnapshotGroups,
                long persistentSnapshotNanos,
                long adaptiveCpuRows,
                long adaptivePersistentRows,
                long adaptiveBatchRows,
                long adaptiveSwitches,
                long adaptiveSwitchNanos,
                long adaptiveBackpressureSamples,
                long sessionWindowMerges,
                long windowLateContributions,
                long maxBatchRows,
                long maxInputDeviceBytes,
                long maxOutputDeviceBytes) {
            this.inputRows = inputRows;
            this.partialGroups = partialGroups;
            this.batches = batches;
            this.rowToColumnNanos = rowToColumnNanos;
            this.h2dNanos = h2dNanos;
            this.computeNanos = computeNanos;
            this.d2hNanos = d2hNanos;
            this.columnToRowNanos = columnToRowNanos;
            this.inputCopyNanos = inputCopyNanos;
            this.postKernelNanos = postKernelNanos;
            this.eventTimerNanos = eventTimerNanos;
            this.keySelectionNanos = keySelectionNanos;
            this.stateReadNanos = stateReadNanos;
            this.mergeNanos = mergeNanos;
            this.stateWriteNanos = stateWriteNanos;
            this.emitNanos = emitNanos;
            this.inputDeviceBytes = inputDeviceBytes;
            this.outputDeviceBytes = outputDeviceBytes;
            this.fusedCalcNativeFallbackBatches = fusedCalcNativeFallbackBatches;
            this.persistentCommands = persistentCommands;
            this.persistentAckResponses = persistentAckResponses;
            this.persistentMaterializedResponses = persistentMaterializedResponses;
            this.persistentSnapshotCalls = persistentSnapshotCalls;
            this.persistentSnapshotGroups = persistentSnapshotGroups;
            this.persistentSnapshotNanos = persistentSnapshotNanos;
            this.adaptiveCpuRows = adaptiveCpuRows;
            this.adaptivePersistentRows = adaptivePersistentRows;
            this.adaptiveBatchRows = adaptiveBatchRows;
            this.adaptiveSwitches = adaptiveSwitches;
            this.adaptiveSwitchNanos = adaptiveSwitchNanos;
            this.adaptiveBackpressureSamples = adaptiveBackpressureSamples;
            this.sessionWindowMerges = sessionWindowMerges;
            this.windowLateContributions = windowLateContributions;
            this.maxBatchRows = maxBatchRows;
            this.maxInputDeviceBytes = maxInputDeviceBytes;
            this.maxOutputDeviceBytes = maxOutputDeviceBytes;
        }

        public double averageBatchRows() {
            return batches == 0 ? 0.0 : (double) inputRows / batches;
        }
    }
}
