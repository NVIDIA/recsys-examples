// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.RowData;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;

/**
 * Bounded CUDA lookup join for a read-only dimension with unique BIGINT keys.
 *
 * <p>The dimension is rebuilt atomically into an inactive device table and becomes visible only
 * after its uniqueness check succeeds. Probe output is materialized in probe-input order. SQL NULL
 * keys never match; a matched dimension payload can independently be NULL.
 */
public final class LookupJoinKernel implements AutoCloseable {
    public static final String LIBRARY_PATH_PROPERTY = "cudf.flink.lookup-join.library";

    private static final int METRICS_BYTES = 64;
    private static final int BUILD_H2D_NANOS = 0;
    private static final int BUILD_COMPUTE_NANOS = 8;
    private static final int PROBE_H2D_NANOS = 16;
    private static final int PROBE_COMPUTE_NANOS = 24;
    private static final int PROBE_D2H_NANOS = 32;
    private static final int GENERATION = 40;
    private static final int DIMENSION_ROWS = 48;
    private static final int PROBE_ROWS = 52;
    private static final int MATCH_COUNT = 56;
    private static final int DEVICE_STATUS = 60;
    private static final int MAX_LONG_BUFFER_ROWS = Integer.MAX_VALUE / Long.BYTES;

    private final int dimensionCapacity;
    private final int probeCapacity;
    private final GpuNativeDeviceGuard deviceGuard;
    private final GpuNativeKernelHandleLifecycle nativeHandleLifecycle =
            new GpuNativeKernelHandleLifecycle();
    private final ByteBuffer dimensionKeys;
    private final ByteBuffer dimensionKeyValid;
    private final ByteBuffer dimensionPayloads;
    private final ByteBuffer dimensionPayloadValid;
    private final ByteBuffer probeKeys;
    private final ByteBuffer probeKeyValid;
    private final ByteBuffer outputPayloads;
    private final ByteBuffer outputPayloadValid;
    private final ByteBuffer outputMatched;
    private final ByteBuffer metrics;

    private long handle;
    private volatile boolean closePending;
    private boolean injectNextProbeDeviceError;
    private BuildMetrics lastBuildMetrics;

    public enum JoinType {
        INNER,
        LEFT;

        public static JoinType parse(String value) {
            return valueOf(value.trim().toUpperCase(Locale.ROOT));
        }
    }

    public LookupJoinKernel(int dimensionCapacity, int probeCapacity) {
        if (dimensionCapacity <= 0 || probeCapacity <= 0) {
            throw new IllegalArgumentException("Lookup Join capacities must be positive");
        }
        if (dimensionCapacity > MAX_LONG_BUFFER_ROWS || probeCapacity > MAX_LONG_BUFFER_ROWS) {
            throw new IllegalArgumentException("Lookup Join capacity exceeds Java direct-buffer range");
        }
        NativeLibrary.ensureLoaded();
        this.deviceGuard = GpuNativeDeviceGuard.capture();
        deviceGuard.bind();
        this.dimensionCapacity = dimensionCapacity;
        this.probeCapacity = probeCapacity;
        handle = nativeCreate(dimensionCapacity, probeCapacity);
        if (handle == 0L) {
            throw new IllegalStateException("native Lookup Join returned a null context");
        }
        nativeHandleLifecycle.opened(handle, this::closeNativeHandle);
        try {
            ByteBuffer nativeDimensionKeys = nativeDimensionKeysBuffer(handle);
            ByteBuffer nativeDimensionKeyValid = nativeDimensionKeyValidBuffer(handle);
            ByteBuffer nativeDimensionPayloads = nativeDimensionPayloadsBuffer(handle);
            ByteBuffer nativeDimensionPayloadValid = nativeDimensionPayloadValidBuffer(handle);
            ByteBuffer nativeProbeKeys = nativeProbeKeysBuffer(handle);
            ByteBuffer nativeProbeKeyValid = nativeProbeKeyValidBuffer(handle);
            ByteBuffer nativeOutputPayloads = nativeOutputPayloadsBuffer(handle);
            ByteBuffer nativeOutputPayloadValid = nativeOutputPayloadValidBuffer(handle);
            ByteBuffer nativeOutputMatched = nativeOutputMatchedBuffer(handle);
            ByteBuffer nativeMetrics = nativeMetricsBuffer(handle);
            if (!valid(nativeDimensionKeys, bytes(dimensionCapacity, Long.BYTES))
                    || !valid(nativeDimensionKeyValid, dimensionCapacity)
                    || !valid(nativeDimensionPayloads, bytes(dimensionCapacity, Long.BYTES))
                    || !valid(nativeDimensionPayloadValid, dimensionCapacity)
                    || !valid(nativeProbeKeys, bytes(probeCapacity, Long.BYTES))
                    || !valid(nativeProbeKeyValid, probeCapacity)
                    || !valid(nativeOutputPayloads, bytes(probeCapacity, Long.BYTES))
                    || !valid(nativeOutputPayloadValid, probeCapacity)
                    || !valid(nativeOutputMatched, probeCapacity)
                    || !valid(nativeMetrics, METRICS_BYTES)) {
                throw new IllegalStateException("native Lookup Join returned invalid direct buffers");
            }
            ByteOrder order = ByteOrder.nativeOrder();
            dimensionKeys = nativeDimensionKeys.order(order);
            dimensionKeyValid = nativeDimensionKeyValid.order(order);
            dimensionPayloads = nativeDimensionPayloads.order(order);
            dimensionPayloadValid = nativeDimensionPayloadValid.order(order);
            probeKeys = nativeProbeKeys.order(order);
            probeKeyValid = nativeProbeKeyValid.order(order);
            outputPayloads = nativeOutputPayloads.order(order);
            outputPayloadValid = nativeOutputPayloadValid.order(order);
            outputMatched = nativeOutputMatched.order(order);
            metrics = nativeMetrics.order(order);
        } catch (RuntimeException | Error constructionFailure) {
            handle =
                    nativeHandleLifecycle.rollbackFailedOpen(
                            handle, constructionFailure, this::closeNativeHandle);
            throw constructionFailure;
        }
    }

    public int dimensionCapacity() {
        return dimensionCapacity;
    }

    public int probeCapacity() {
        return probeCapacity;
    }

    /**
     * Atomically replaces the resident dimension.
     *
     * <p>Rows with a NULL key are retained only as input accounting and are not inserted because SQL
     * equality cannot match them. Duplicate non-NULL keys are rejected before native state changes.
     */
    public synchronized BuildMetrics rebuild(
            long[] keys, boolean[] keyValid, long[] payloads, boolean[] payloadValid) {
        requireParallelArrays(keys, keyValid, payloads, payloadValid, "dimension");
        ensureOpen();
        if (keys.length > dimensionCapacity) {
            throw new IllegalArgumentException(
                    "Lookup Join dimension rows "
                            + keys.length
                            + " exceed capacity "
                            + dimensionCapacity);
        }
        long started = System.nanoTime();
        Set<Long> uniqueKeys = new HashSet<>(hashSetCapacity(keys.length));
        for (int row = 0; row < keys.length; row++) {
            if (keyValid[row] && !uniqueKeys.add(keys[row])) {
                throw new IllegalArgumentException(
                        "Lookup Join dimension contains duplicate key " + keys[row]);
            }
        }
        for (int row = 0; row < keys.length; row++) {
            dimensionKeys.putLong(row * Long.BYTES, keyValid[row] ? keys[row] : 0L);
            dimensionKeyValid.put(row, (byte) (keyValid[row] ? 1 : 0));
            dimensionPayloads.putLong(row * Long.BYTES, payloadValid[row] ? payloads[row] : 0L);
            dimensionPayloadValid.put(row, (byte) (payloadValid[row] ? 1 : 0));
        }
        deviceGuard.bind();
        nativeRebuild(handle, keys.length);
        long wallNanos = System.nanoTime() - started;
        if (metrics.getInt(DIMENSION_ROWS) != keys.length || metrics.getInt(DEVICE_STATUS) != 0) {
            throw new IllegalStateException("native Lookup Join returned inconsistent build metrics");
        }
        lastBuildMetrics =
                new BuildMetrics(
                        metrics.getLong(BUILD_H2D_NANOS),
                        metrics.getLong(BUILD_COMPUTE_NANOS),
                        wallNanos,
                        metrics.getLong(GENERATION),
                        metrics.getInt(DIMENSION_ROWS));
        return lastBuildMetrics;
    }

    public synchronized BuildMetrics lastBuildMetrics() {
        ensureOpen();
        if (lastBuildMetrics == null) {
            throw new IllegalStateException("Lookup Join dimension has not been built");
        }
        return lastBuildMetrics;
    }

    public synchronized Result probe(long[] keys, boolean[] keyValid, JoinType joinType) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(joinType, "joinType");
        List<OutputRow> rows = new ArrayList<>(keys.length);
        ProbeMetrics probeMetrics =
                probeInto(
                        keys,
                        keyValid,
                        joinType,
                        ProbeFilter.none(),
                        ProbeTransfer.all(),
                        (probeIndex,
                                        probeKey,
                                        probeKeyValid,
                                        matched,
                                        dimensionPayload,
                                        dimensionPayloadValid) ->
                                rows.add(
                                        new OutputRow(
                                                probeIndex,
                                                probeKey,
                                                probeKeyValid,
                                                matched,
                                                dimensionPayload,
                                                dimensionPayloadValid)));
        return new Result(joinType, rows, probeMetrics);
    }

    /**
     * Probes directly into a caller-owned materializer without allocating {@link OutputRow}
     * objects or copying an intermediate result list.
     *
     * <p>The callback is ordered by probe index. INNER misses are omitted; LEFT misses are passed
     * with {@code matched=false} and an invalid payload. Device status and aggregate match-count
     * validation complete before this method returns.
     */
    public synchronized ProbeMetrics probeInto(
            long[] keys,
            boolean[] keyValid,
            JoinType joinType,
            ProbeConsumer consumer) {
        return probeInto(keys, keyValid, joinType, ProbeFilter.none(), consumer);
    }

    /** Probe overload that evaluates a proven-safe post-join filter in the CUDA lookup kernel. */
    public synchronized ProbeMetrics probeInto(
            long[] keys,
            boolean[] keyValid,
            JoinType joinType,
            ProbeFilter filter,
            ProbeConsumer consumer) {
        return probeInto(keys, keyValid, joinType, filter, ProbeTransfer.all(), consumer);
    }

    /**
     * Probe overload with native field-transfer pruning for a fused final projection.
     *
     * <p>Payload transfer implies flags. Omitting flags is legal only for LEFT + no filter, where
     * the caller has proven that the final projection references no dimension field.
     */
    public synchronized ProbeMetrics probeInto(
            long[] keys,
            boolean[] keyValid,
            JoinType joinType,
            ProbeFilter filter,
            ProbeTransfer transfer,
            ProbeConsumer consumer) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(keyValid, "keyValid");
        Objects.requireNonNull(joinType, "joinType");
        Objects.requireNonNull(filter, "filter");
        Objects.requireNonNull(transfer, "transfer");
        Objects.requireNonNull(consumer, "consumer");
        if (!transfer.flags()
                && (joinType != JoinType.LEFT || !filter.isNone())) {
            throw new IllegalArgumentException(
                    "Lookup Join may omit result flags only for unfiltered LEFT projection");
        }
        ensureOpen();
        if (lastBuildMetrics == null) {
            throw new IllegalStateException("Lookup Join dimension has not been built");
        }
        if (keys.length != keyValid.length) {
            throw new IllegalArgumentException("probe key/value validity lengths differ");
        }
        if (keys.length > probeCapacity) {
            throw new IllegalArgumentException(
                    "Lookup Join probe rows " + keys.length + " exceed capacity " + probeCapacity);
        }

        long started = System.nanoTime();
        for (int row = 0; row < keys.length; row++) {
            probeKeys.putLong(row * Long.BYTES, keyValid[row] ? keys[row] : 0L);
            probeKeyValid.put(row, (byte) (keyValid[row] ? 1 : 0));
        }
        boolean injectDeviceError = injectNextProbeDeviceError;
        injectNextProbeDeviceError = false;
        deviceGuard.bind();
        nativeProbe(
                handle,
                keys.length,
                injectDeviceError,
                filter.nativeOperation(),
                filter.literal(),
                transfer.nativeMask());
        long columnWallNanos = System.nanoTime() - started;

        int matchCount = metrics.getInt(MATCH_COUNT);
        if (metrics.getInt(PROBE_ROWS) != keys.length
                || matchCount < 0
                || matchCount > keys.length
                || metrics.getInt(DEVICE_STATUS) != 0) {
            throw new IllegalStateException("native Lookup Join returned inconsistent probe metrics");
        }
        if (transfer.flags()) {
            int observedMatches = 0;
            for (int probeRow = 0; probeRow < keys.length; probeRow++) {
                int flags = outputMatched.get(probeRow) & 0xff;
                if ((flags & ~3) != 0) {
                    throw new IllegalStateException(
                            "native Lookup Join returned invalid output flags");
                }
                boolean matched = (flags & 1) != 0;
                if (matched) observedMatches++;
                if (transfer.payloads()) {
                    boolean resultPayloadValid = outputPayloadValid.get(probeRow) != 0;
                    if (!matched && resultPayloadValid) {
                        throw new IllegalStateException(
                                "LEFT miss unexpectedly carried a valid payload");
                    }
                }
            }
            if (observedMatches != matchCount) {
                throw new IllegalStateException(
                        "native Lookup Join match count differs from output validity");
            }
        }
        for (int probeRow = 0; probeRow < keys.length; probeRow++) {
            int flags = transfer.flags() ? outputMatched.get(probeRow) & 0xff : 2;
            boolean matched = (flags & 1) != 0;
            boolean selected = (flags & 2) != 0;
            if (!matched && joinType == JoinType.INNER) continue;
            if (!selected) continue;
            boolean resultPayloadValid =
                    transfer.payloads() && outputPayloadValid.get(probeRow) != 0;
            consumer.accept(
                    probeRow,
                    keyValid[probeRow] ? keys[probeRow] : 0L,
                    keyValid[probeRow],
                    matched,
                    resultPayloadValid
                            ? outputPayloads.getLong(probeRow * Long.BYTES)
                            : 0L,
                    resultPayloadValid);
        }
        long wallNanos = System.nanoTime() - started;
        return new ProbeMetrics(
                metrics.getLong(PROBE_H2D_NANOS),
                metrics.getLong(PROBE_COMPUTE_NANOS),
                metrics.getLong(PROBE_D2H_NANOS),
                0L,
                columnWallNanos,
                wallNanos,
                metrics.getLong(GENERATION),
                keys.length,
                matchCount);
    }

    /**
     * Probes BIGINT keys straight from Flink rows and emits source rows without temporary columns.
     *
     * <p>This is the row-input counterpart of {@link #probeInto(long[], boolean[], JoinType,
     * ProbeConsumer)}. It removes two large temporary Java arrays and the second host copy while
     * retaining the same native buffers, SQL NULL handling, and ordered output contract.
     */
    public synchronized ProbeMetrics probeRowsInto(
            RowData[] rows,
            int keyOrdinal,
            JoinType joinType,
            RowProbeConsumer consumer) {
        Objects.requireNonNull(rows, "rows");
        Objects.requireNonNull(joinType, "joinType");
        Objects.requireNonNull(consumer, "consumer");
        ensureOpen();
        if (lastBuildMetrics == null) {
            throw new IllegalStateException("Lookup Join dimension has not been built");
        }
        if (rows.length > probeCapacity || keyOrdinal < 0) {
            throw new IllegalArgumentException("Lookup Join row probe shape is invalid");
        }

        long started = System.nanoTime();
        for (int row = 0; row < rows.length; row++) {
            RowData input = Objects.requireNonNull(rows[row], "probe row");
            if (keyOrdinal >= input.getArity()) {
                throw new IllegalArgumentException("Lookup Join key ordinal is outside probe row");
            }
            boolean valid = !input.isNullAt(keyOrdinal);
            probeKeys.putLong(row * Long.BYTES, valid ? input.getLong(keyOrdinal) : 0L);
            probeKeyValid.put(row, valid ? (byte) 1 : (byte) 0);
        }
        long adapted = System.nanoTime();
        boolean injectDeviceError = injectNextProbeDeviceError;
        injectNextProbeDeviceError = false;
        deviceGuard.bind();
        nativeProbe(
                handle,
                rows.length,
                injectDeviceError,
                ProbeFilter.none().nativeOperation(),
                ProbeFilter.none().literal(),
                ProbeTransfer.all().nativeMask());
        long nativeFinished = System.nanoTime();

        int matchCount = metrics.getInt(MATCH_COUNT);
        if (metrics.getInt(PROBE_ROWS) != rows.length
                || matchCount < 0
                || matchCount > rows.length
                || metrics.getInt(DEVICE_STATUS) != 0) {
            throw new IllegalStateException("native Lookup Join returned inconsistent probe metrics");
        }
        int observedMatches = 0;
        for (int probeRow = 0; probeRow < rows.length; probeRow++) {
            int flags = outputMatched.get(probeRow) & 0xff;
            if ((flags & ~3) != 0) {
                throw new IllegalStateException("native Lookup Join returned invalid output flags");
            }
            boolean matched = (flags & 1) != 0;
            boolean selected = (flags & 2) != 0;
            if (matched) observedMatches++;
            boolean payloadValid = outputPayloadValid.get(probeRow) != 0;
            if (!matched && payloadValid) {
                throw new IllegalStateException("LEFT miss unexpectedly carried a valid payload");
            }
            if ((!matched && joinType == JoinType.INNER) || !selected) {
                continue;
            }
            consumer.accept(
                    probeRow,
                    rows[probeRow],
                    matched,
                    payloadValid
                            ? outputPayloads.getLong(probeRow * Long.BYTES)
                            : 0L,
                    payloadValid);
        }
        if (observedMatches != matchCount) {
            throw new IllegalStateException(
                    "native Lookup Join match count differs from output validity");
        }
        long finished = System.nanoTime();
        return new ProbeMetrics(
                metrics.getLong(PROBE_H2D_NANOS),
                metrics.getLong(PROBE_COMPUTE_NANOS),
                metrics.getLong(PROBE_D2H_NANOS),
                adapted - started,
                nativeFinished - adapted,
                finished - started,
                metrics.getLong(GENERATION),
                rows.length,
                matchCount);
    }

    /**
     * Fixed-width BinaryRowData fast path that preserves the same ordered probe contract while
     * avoiding per-field RowData dispatch.
     */
    public synchronized ProbeMetrics probePackedRowsInto(
            PackedBinaryRowBatch batch,
            int idOrdinal,
            int keyOrdinal,
            JoinType joinType,
            ProbeConsumer consumer) {
        Objects.requireNonNull(batch, "batch");
        Objects.requireNonNull(joinType, "joinType");
        Objects.requireNonNull(consumer, "consumer");
        ensureOpen();
        if (lastBuildMetrics == null) {
            throw new IllegalStateException("Lookup Join dimension has not been built");
        }
        RowData[] rows = batch.rows();
        int arity = rows.length == 0 ? -1 : rows[0].getArity();
        if (rows.length <= 0
                || rows.length > probeCapacity
                || arity > 56
                || idOrdinal < 0
                || idOrdinal >= arity
                || keyOrdinal < 0
                || keyOrdinal >= arity) {
            throw new IllegalArgumentException("Lookup Join packed probe shape is invalid");
        }

        long started = System.nanoTime();
        for (int row = 0; row < rows.length; row++) {
            long header = batch.fixedHeaderWordUnchecked(row);
            if ((header & (1L << (idOrdinal + 8))) != 0L
                    || batch.fixedSlotBitsUnchecked(row, idOrdinal) != row) {
                throw new IllegalArgumentException(
                        "Lookup Join packed probe id is missing or out of order");
            }
            boolean valid = (header & (1L << (keyOrdinal + 8))) == 0L;
            probeKeys.putLong(
                    row * Long.BYTES,
                    valid ? batch.fixedSlotBitsUnchecked(row, keyOrdinal) : 0L);
            probeKeyValid.put(row, valid ? (byte) 1 : (byte) 0);
        }
        long adapted = System.nanoTime();
        boolean injectDeviceError = injectNextProbeDeviceError;
        injectNextProbeDeviceError = false;
        deviceGuard.bind();
        nativeProbe(
                handle,
                rows.length,
                injectDeviceError,
                ProbeFilter.none().nativeOperation(),
                ProbeFilter.none().literal(),
                ProbeTransfer.all().nativeMask());
        long nativeFinished = System.nanoTime();

        int matchCount = metrics.getInt(MATCH_COUNT);
        if (metrics.getInt(PROBE_ROWS) != rows.length
                || matchCount < 0
                || matchCount > rows.length
                || metrics.getInt(DEVICE_STATUS) != 0) {
            throw new IllegalStateException("native Lookup Join returned inconsistent probe metrics");
        }
        int observedMatches = 0;
        for (int probeRow = 0; probeRow < rows.length; probeRow++) {
            int flags = outputMatched.get(probeRow) & 0xff;
            if ((flags & ~3) != 0) {
                throw new IllegalStateException("native Lookup Join returned invalid output flags");
            }
            boolean matched = (flags & 1) != 0;
            boolean selected = (flags & 2) != 0;
            if (matched) observedMatches++;
            boolean payloadValid = outputPayloadValid.get(probeRow) != 0;
            if (!matched && payloadValid) {
                throw new IllegalStateException("LEFT miss unexpectedly carried a valid payload");
            }
            if ((!matched && joinType == JoinType.INNER) || !selected) {
                continue;
            }
            long header = batch.fixedHeaderWordUnchecked(probeRow);
            boolean keyValid = (header & (1L << (keyOrdinal + 8))) == 0L;
            consumer.accept(
                    probeRow,
                    keyValid ? batch.fixedSlotBitsUnchecked(probeRow, keyOrdinal) : 0L,
                    keyValid,
                    matched,
                    payloadValid
                            ? outputPayloads.getLong(probeRow * Long.BYTES)
                            : 0L,
                    payloadValid);
        }
        if (observedMatches != matchCount) {
            throw new IllegalStateException(
                    "native Lookup Join match count differs from output validity");
        }
        long finished = System.nanoTime();
        return new ProbeMetrics(
                metrics.getLong(PROBE_H2D_NANOS),
                metrics.getLong(PROBE_COMPUTE_NANOS),
                metrics.getLong(PROBE_D2H_NANOS),
                adapted - started,
                nativeFinished - adapted,
                finished - started,
                metrics.getLong(GENERATION),
                rows.length,
                matchCount);
    }

    /** Probes packed rows and writes the complete four-field lookup result directly to a batch. */
    public ProbeMetrics probePackedRowsInto(
            PackedBinaryRowBatch batch,
            int idOrdinal,
            int keyOrdinal,
            JoinType joinType,
            PackedBinaryRowBatch output) {
        Objects.requireNonNull(output, "output");
        if (output.arity() != 4 || output.arity() > 56) {
            throw new IllegalArgumentException("Lookup Join packed output has the wrong shape");
        }
        int[] position = {0};
        ProbeMetrics probeMetrics =
                probePackedRowsInto(
                        batch,
                        idOrdinal,
                        keyOrdinal,
                        joinType,
                        (probeIndex,
                                        probeKey,
                                        probeKeyValid,
                                        matched,
                                        dimensionPayload,
                                        dimensionPayloadValid) -> {
                            int target = position[0]++;
                            if (target >= output.rowCount()) {
                                throw new IllegalStateException(
                                        "Lookup Join returned more packed rows than expected");
                            }
                            long header = 0L;
                            output.setFixedSlotBitsUnchecked(target, 0, probeIndex);
                            if (probeKeyValid) {
                                output.setFixedSlotBitsUnchecked(target, 1, probeKey);
                            } else {
                                header |= 1L << 9;
                            }
                            if (matched) {
                                output.setFixedSlotBitsUnchecked(target, 2, probeKey);
                            } else {
                                header |= 1L << 10;
                            }
                            if (dimensionPayloadValid) {
                                output.setFixedSlotBitsUnchecked(
                                        target, 3, dimensionPayload);
                            } else {
                                header |= 1L << 11;
                            }
                            output.setFixedHeaderWordUnchecked(target, header);
                        });
        if (position[0] != output.rowCount()) {
            throw new IllegalStateException(
                    "Lookup Join returned "
                            + position[0]
                            + " packed rows, expected "
                            + output.rowCount());
        }
        return probeMetrics;
    }

    /** Makes the next non-empty probe report a controlled device-side failure. Testing only. */
    public synchronized void injectNextProbeDeviceErrorForTesting() {
        ensureOpen();
        injectNextProbeDeviceError = true;
    }

    @Override
    public synchronized void close() {
        closePending = true;
        try {
            handle = nativeHandleLifecycle.close(handle, this::closeNativeHandle);
            closePending = false;
        } catch (RuntimeException | Error failure) {
            // A failed native close may already have released some direct-buffer storage.
            // Preserve the handle for close() retry, but reject every buffer-backed operation.
            throw failure;
        }
    }

    private void closeNativeHandle(long closingHandle) {
        deviceGuard.bind();
        nativeClose(closingHandle);
    }

    private void ensureOpen() {
        if (closePending) {
            throw new IllegalStateException("Lookup Join native close retry is pending");
        }
        if (handle == 0L) {
            throw new IllegalStateException("Lookup Join kernel is closed");
        }
    }

    private static int bytes(int rows, int width) {
        return Math.multiplyExact(rows, width);
    }

    private static boolean valid(ByteBuffer buffer, int minimumBytes) {
        return buffer != null && buffer.isDirect() && buffer.capacity() >= minimumBytes;
    }

    private static int hashSetCapacity(int rows) {
        if (rows < 3) return 4;
        long requested = (long) Math.ceil(rows / 0.75D);
        return (int) Math.min(Integer.MAX_VALUE - 8L, requested);
    }

    private static void requireParallelArrays(
            long[] keys,
            boolean[] keyValid,
            long[] payloads,
            boolean[] payloadValid,
            String label) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(keyValid, "keyValid");
        Objects.requireNonNull(payloads, "payloads");
        Objects.requireNonNull(payloadValid, "payloadValid");
        if (keys.length != keyValid.length
                || keys.length != payloads.length
                || keys.length != payloadValid.length) {
            throw new IllegalArgumentException(label + " array lengths differ");
        }
    }

    public static long semanticDigest(List<OutputRow> rows) {
        Objects.requireNonNull(rows, "rows");
        long count = 0L;
        long sum = 0L;
        long xor = 0L;
        for (OutputRow output : rows) {
            long flags =
                    (output.probeKeyValid ? 1L : 0L)
                            | (output.matched ? 2L : 0L)
                            | (output.dimensionPayloadValid ? 4L : 0L);
            long row =
                    mix64(output.probeIndex)
                            ^ Long.rotateLeft(mix64(output.probeKey), 11)
                            ^ Long.rotateLeft(mix64(output.dimensionPayload), 29)
                            ^ Long.rotateLeft(mix64(flags), 47);
            count++;
            sum += row;
            xor ^= Long.rotateLeft(row, (int) (row & 63));
        }
        return mix64(count) ^ mix64(sum) ^ mix64(xor);
    }

    public static long mix64(long value) {
        long mixed = value;
        mixed = (mixed ^ (mixed >>> 30)) * 0xbf58476d1ce4e5b9L;
        mixed = (mixed ^ (mixed >>> 27)) * 0x94d049bb133111ebL;
        return mixed ^ (mixed >>> 31);
    }

    public static final class OutputRow {
        private final int probeIndex;
        private final long probeKey;
        private final boolean probeKeyValid;
        private final boolean matched;
        private final long dimensionPayload;
        private final boolean dimensionPayloadValid;

        public OutputRow(
                int probeIndex,
                long probeKey,
                boolean probeKeyValid,
                boolean matched,
                long dimensionPayload,
                boolean dimensionPayloadValid) {
            if (probeIndex < 0) throw new IllegalArgumentException("negative probe index");
            if (matched && !probeKeyValid) {
                throw new IllegalArgumentException("NULL probe key cannot match");
            }
            if (!matched && dimensionPayloadValid) {
                throw new IllegalArgumentException("miss cannot carry a valid dimension payload");
            }
            this.probeIndex = probeIndex;
            this.probeKey = probeKeyValid ? probeKey : 0L;
            this.probeKeyValid = probeKeyValid;
            this.matched = matched;
            this.dimensionPayload = dimensionPayloadValid ? dimensionPayload : 0L;
            this.dimensionPayloadValid = dimensionPayloadValid;
        }

        public int probeIndex() {
            return probeIndex;
        }

        public long probeKey() {
            return probeKey;
        }

        public boolean probeKeyValid() {
            return probeKeyValid;
        }

        public boolean matched() {
            return matched;
        }

        public long dimensionPayload() {
            return dimensionPayload;
        }

        public boolean dimensionPayloadValid() {
            return dimensionPayloadValid;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) return true;
            if (!(other instanceof OutputRow)) return false;
            OutputRow that = (OutputRow) other;
            return probeIndex == that.probeIndex
                    && probeKey == that.probeKey
                    && probeKeyValid == that.probeKeyValid
                    && matched == that.matched
                    && dimensionPayload == that.dimensionPayload
                    && dimensionPayloadValid == that.dimensionPayloadValid;
        }

        @Override
        public int hashCode() {
            return Objects.hash(
                    probeIndex,
                    probeKey,
                    probeKeyValid,
                    matched,
                    dimensionPayload,
                    dimensionPayloadValid);
        }

        @Override
        public String toString() {
            return "OutputRow{"
                    + "probeIndex="
                    + probeIndex
                    + ", probeKey="
                    + (probeKeyValid ? Long.toString(probeKey) : "NULL")
                    + ", matched="
                    + matched
                    + ", dimensionPayload="
                    + (dimensionPayloadValid ? Long.toString(dimensionPayload) : "NULL")
                    + '}';
        }
    }

    /** Allocation-free ordered probe output callback used by the Flink operator. */
    @FunctionalInterface
    public interface ProbeConsumer {
        void accept(
                int probeIndex,
                long probeKey,
                boolean probeKeyValid,
                boolean matched,
                long dimensionPayload,
                boolean dimensionPayloadValid);
    }

    /** Allocation-free callback for the direct Flink-row probe path. */
    @FunctionalInterface
    public interface RowProbeConsumer {
        void accept(
                int probeIndex,
                RowData probeRow,
                boolean matched,
                long dimensionPayload,
                boolean dimensionPayloadValid);
    }

    /**
     * Serializable, SQL-null-aware filter subset fused into the lookup probe kernel.
     *
     * <p>Comparisons with NULL select no row. IS NULL/IS NOT NULL distinguish LEFT misses from a
     * matched NULL payload exactly. The numeric operation codes are an ABI shared with
     * {@code lookup_join_jni.cu}.
     */
    public static final class ProbeFilter implements Serializable {
        private static final long serialVersionUID = 1L;
        private static final ProbeFilter NONE = new ProbeFilter(0, 0L, "none");

        private final int nativeOperation;
        private final long literal;
        private final String summary;

        private ProbeFilter(int nativeOperation, long literal, String summary) {
            this.nativeOperation = nativeOperation;
            this.literal = literal;
            this.summary = summary;
        }

        public static ProbeFilter none() {
            return NONE;
        }

        public static ProbeFilter isNull(FilterField field) {
            Objects.requireNonNull(field, "field");
            return new ProbeFilter(
                    field == FilterField.DIMENSION_KEY ? 1 : 3,
                    0L,
                    field.summary + " IS NULL");
        }

        public static ProbeFilter isNotNull(FilterField field) {
            Objects.requireNonNull(field, "field");
            return new ProbeFilter(
                    field == FilterField.DIMENSION_KEY ? 2 : 4,
                    0L,
                    field.summary + " IS NOT NULL");
        }

        public static ProbeFilter compare(
                FilterField field, FilterComparison comparison, long literal) {
            Objects.requireNonNull(field, "field");
            Objects.requireNonNull(comparison, "comparison");
            int base = field == FilterField.DIMENSION_KEY ? 5 : 11;
            return new ProbeFilter(
                    base + comparison.offset,
                    literal,
                    field.summary + ' ' + comparison.symbol + ' ' + literal);
        }

        public boolean test(
                boolean matched, long key, long payload, boolean payloadValid) {
            switch (nativeOperation) {
                case 0:
                    return true;
                case 1:
                    return !matched;
                case 2:
                    return matched;
                case 3:
                    return !matched || !payloadValid;
                case 4:
                    return matched && payloadValid;
                default:
                    if (nativeOperation >= 5 && nativeOperation <= 10) {
                        return matched
                                && compareValue(key, literal, nativeOperation - 5);
                    }
                    return matched
                            && payloadValid
                            && compareValue(payload, literal, nativeOperation - 11);
            }
        }

        public boolean isNone() {
            return nativeOperation == 0;
        }

        public String summary() {
            return summary;
        }

        private int nativeOperation() {
            return nativeOperation;
        }

        private long literal() {
            return literal;
        }

        private static boolean compareValue(long value, long literal, int operation) {
            switch (operation) {
                case 0:
                    return value == literal;
                case 1:
                    return value != literal;
                case 2:
                    return value < literal;
                case 3:
                    return value <= literal;
                case 4:
                    return value > literal;
                case 5:
                    return value >= literal;
                default:
                    throw new IllegalStateException("invalid Lookup Join filter operation");
            }
        }
    }

    public enum FilterField {
        DIMENSION_KEY("dimensionKey"),
        DIMENSION_PAYLOAD("dimensionPayload");

        private final String summary;

        FilterField(String summary) {
            this.summary = summary;
        }
    }

    public enum FilterComparison {
        EQUAL(0, "="),
        NOT_EQUAL(1, "<>"),
        LESS(2, "<"),
        LESS_EQUAL(3, "<="),
        GREATER(4, ">"),
        GREATER_EQUAL(5, ">=");

        private final int offset;
        private final String symbol;

        FilterComparison(int offset, String symbol) {
            this.offset = offset;
            this.symbol = symbol;
        }
    }

    /** Per-batch D2H projection decided by the fused Calc proof. */
    public static final class ProbeTransfer {
        private static final ProbeTransfer NONE = new ProbeTransfer(false, false);
        private static final ProbeTransfer FLAGS = new ProbeTransfer(true, false);
        private static final ProbeTransfer ALL = new ProbeTransfer(true, true);

        private final boolean flags;
        private final boolean payloads;

        private ProbeTransfer(boolean flags, boolean payloads) {
            this.flags = flags;
            this.payloads = payloads;
        }

        public static ProbeTransfer projected(boolean flags, boolean payloads) {
            if (payloads && !flags) {
                throw new IllegalArgumentException("Lookup Join payload transfer requires flags");
            }
            return payloads ? ALL : flags ? FLAGS : NONE;
        }

        public static ProbeTransfer all() {
            return ALL;
        }

        public boolean flags() {
            return flags;
        }

        public boolean payloads() {
            return payloads;
        }

        public int bytesPerProbe() {
            return (flags ? 1 : 0) + (payloads ? Long.BYTES + 1 : 0);
        }

        private int nativeMask() {
            return (flags ? 1 : 0) | (payloads ? 2 : 0);
        }
    }

    public static final class Result {
        private final JoinType joinType;
        private final List<OutputRow> rows;
        private final ProbeMetrics metrics;
        private final long semanticDigest;

        private Result(JoinType joinType, List<OutputRow> rows, ProbeMetrics metrics) {
            this.joinType = joinType;
            this.rows = Collections.unmodifiableList(new ArrayList<>(rows));
            this.metrics = metrics;
            semanticDigest = LookupJoinKernel.semanticDigest(this.rows);
        }

        public JoinType joinType() {
            return joinType;
        }

        public List<OutputRow> rows() {
            return rows;
        }

        public ProbeMetrics metrics() {
            return metrics;
        }

        public long semanticDigest() {
            return semanticDigest;
        }
    }

    public static final class BuildMetrics {
        private final long h2dNanos;
        private final long computeNanos;
        private final long wallNanos;
        private final long generation;
        private final int dimensionRows;

        private BuildMetrics(
                long h2dNanos,
                long computeNanos,
                long wallNanos,
                long generation,
                int dimensionRows) {
            this.h2dNanos = h2dNanos;
            this.computeNanos = computeNanos;
            this.wallNanos = wallNanos;
            this.generation = generation;
            this.dimensionRows = dimensionRows;
        }

        public long h2dNanos() {
            return h2dNanos;
        }

        public long computeNanos() {
            return computeNanos;
        }

        public long wallNanos() {
            return wallNanos;
        }

        public long generation() {
            return generation;
        }

        public int dimensionRows() {
            return dimensionRows;
        }
    }

    public static final class ProbeMetrics {
        private final long h2dNanos;
        private final long computeNanos;
        private final long d2hNanos;
        private final long rowToColumnNanos;
        private final long columnWallNanos;
        private final long wallNanos;
        private final long generation;
        private final int probeRows;
        private final int matchCount;

        private ProbeMetrics(
                long h2dNanos,
                long computeNanos,
                long d2hNanos,
                long rowToColumnNanos,
                long columnWallNanos,
                long wallNanos,
                long generation,
                int probeRows,
                int matchCount) {
            this.h2dNanos = h2dNanos;
            this.computeNanos = computeNanos;
            this.d2hNanos = d2hNanos;
            this.rowToColumnNanos = rowToColumnNanos;
            this.columnWallNanos = columnWallNanos;
            this.wallNanos = wallNanos;
            this.generation = generation;
            this.probeRows = probeRows;
            this.matchCount = matchCount;
        }

        public long h2dNanos() {
            return h2dNanos;
        }

        public long computeNanos() {
            return computeNanos;
        }

        public long d2hNanos() {
            return d2hNanos;
        }

        public long rowToColumnNanos() {
            return rowToColumnNanos;
        }

        /** Host-column probe input through complete host-column output buffers. */
        public long columnWallNanos() {
            return columnWallNanos;
        }

        public long wallNanos() {
            return wallNanos;
        }

        public long generation() {
            return generation;
        }

        public int probeRows() {
            return probeRows;
        }

        public int matchCount() {
            return matchCount;
        }
    }

    private static final class NativeLibrary {
        static {
            String explicitPath = System.getProperty(LIBRARY_PATH_PROPERTY, "").trim();
            if (explicitPath.isEmpty()) {
                System.loadLibrary("cudf_flink_lookup_join");
            } else {
                System.load(explicitPath);
            }
        }

        private static void ensureLoaded() {}
    }

    private static native long nativeCreate(int dimensionCapacity, int probeCapacity);

    private static native ByteBuffer nativeDimensionKeysBuffer(long handle);

    private static native ByteBuffer nativeDimensionKeyValidBuffer(long handle);

    private static native ByteBuffer nativeDimensionPayloadsBuffer(long handle);

    private static native ByteBuffer nativeDimensionPayloadValidBuffer(long handle);

    private static native ByteBuffer nativeProbeKeysBuffer(long handle);

    private static native ByteBuffer nativeProbeKeyValidBuffer(long handle);

    private static native ByteBuffer nativeOutputPayloadsBuffer(long handle);

    private static native ByteBuffer nativeOutputPayloadValidBuffer(long handle);

    private static native ByteBuffer nativeOutputMatchedBuffer(long handle);

    private static native ByteBuffer nativeMetricsBuffer(long handle);

    private static native void nativeRebuild(long handle, int rows);

    private static native void nativeProbe(
            long handle,
            int rows,
            boolean injectDeviceError,
            int filterOperation,
            long filterLiteral,
            int transferMask);

    private static native void nativeClose(long handle);
}
