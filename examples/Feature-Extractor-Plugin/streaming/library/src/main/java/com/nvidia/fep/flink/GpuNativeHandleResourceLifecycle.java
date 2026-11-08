// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Retry-safe process accounting for one JNI context owned by a Java kernel wrapper.
 *
 * <p>The native stats ABI is exactly {@code [active device bytes, active references, active
 * batches]}. Ordinary contexts report one reference and zero batches while open; a context may
 * later transition in place to an authoritative DeviceBatch chain ID. A close failure deliberately
 * leaves both the Java handle and this owner registered, so teardown evidence cannot report a
 * false zero and the caller can retry on the correct CUDA device.
 */
final class GpuNativeHandleResourceLifecycle {
    static final int NATIVE_STATS_FIELDS = 3;

    private static final AtomicLong NEXT_STANDALONE_OWNER_ID =
            new AtomicLong(Long.MIN_VALUE);

    private final Object owner;
    private final GpuProcessNativeCleanupRegistry.Family cleanupFamily;
    private long ownerId;
    private boolean opened;
    private boolean closed;
    private boolean nativeCloseCompleted;

    private GpuNativeHandleResourceLifecycle(
            Object owner,
            long ownerId,
            GpuProcessNativeCleanupRegistry.Family cleanupFamily) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.cleanupFamily = Objects.requireNonNull(cleanupFamily, "cleanupFamily");
        if (ownerId == 0L) {
            throw new IllegalArgumentException("native owner ID must be non-zero");
        }
        this.ownerId = ownerId;
    }

    static GpuNativeHandleResourceLifecycle standalone(Object owner) {
        return standalone(
                owner, GpuProcessNativeCleanupRegistry.Family.NATIVE_POOL_OWNER);
    }

    static GpuNativeHandleResourceLifecycle standalone(
            Object owner, GpuProcessNativeCleanupRegistry.Family cleanupFamily) {
        long ownerId = NEXT_STANDALONE_OWNER_ID.getAndIncrement();
        if (ownerId == 0L) {
            throw new IllegalStateException("standalone native owner ID space exhausted");
        }
        return new GpuNativeHandleResourceLifecycle(owner, ownerId, cleanupFamily);
    }

    static GpuNativeHandleResourceLifecycle owned(Object owner, long ownerId) {
        return owned(
                owner,
                ownerId,
                GpuProcessNativeCleanupRegistry.Family.NATIVE_POOL_OWNER);
    }

    static GpuNativeHandleResourceLifecycle owned(
            Object owner,
            long ownerId,
            GpuProcessNativeCleanupRegistry.Family cleanupFamily) {
        return new GpuNativeHandleResourceLifecycle(owner, ownerId, cleanupFamily);
    }

    GpuProcessNativeCleanupRegistry.ConstructionPermit beginConstruction() {
        return GpuProcessNativeCleanupRegistry.acquireConstructionPermit(cleanupFamily);
    }

    synchronized void opened(NativeStats nativeStats) {
        Objects.requireNonNull(nativeStats, "nativeStats");
        if (opened) {
            throw new IllegalStateException("native owner is already open");
        }
        GpuProcessResourceMetrics.recordNativePoolOpened(owner, ownerId);
        opened = true;
        observe(nativeStats);
    }

    synchronized void observe(NativeStats nativeStats) {
        Objects.requireNonNull(nativeStats, "nativeStats");
        if (!opened || closed) {
            throw new IllegalStateException("native owner is not open");
        }
        Stats stats = Stats.requireLive(nativeStats.read());
        if (!GpuProcessResourceMetrics.recordNativePoolStatsChecked(
                owner, stats.activeBytes, stats.activeReferences, stats.activeBatches)) {
            throw new IllegalStateException("native owner stats accounting was rejected");
        }
    }

    /**
     * Transitions this still-live context entry to an authoritative DeviceBatch owner ID.
     *
     * <p>JNI stats are read and validated before the process registry is touched. The registry
     * then changes the ID and stats in one critical section, so a failed or retried transition
     * never creates an unowned interval.
     */
    synchronized void transitionToOwner(long newOwnerId, NativeStats nativeStats) {
        Objects.requireNonNull(nativeStats, "nativeStats");
        if (!opened || closed) {
            throw new IllegalStateException("native owner is not open");
        }
        if (newOwnerId == 0L) {
            throw new IllegalArgumentException("native owner ID must be non-zero");
        }
        Stats stats = Stats.requireLive(nativeStats.read());
        long expectedOwnerId = ownerId;
        if (!GpuProcessResourceMetrics.recordNativePoolOwnerTransitioned(
                owner,
                expectedOwnerId,
                newOwnerId,
                stats.activeBytes,
                stats.activeReferences,
                stats.activeBatches)) {
            throw new IllegalStateException("native owner transition was rejected");
        }
        ownerId = newOwnerId;
    }

    synchronized void close(NativeClose nativeClose) {
        Objects.requireNonNull(nativeClose, "nativeClose");
        if (closed || !opened) {
            return;
        }
        closeNativeStage(nativeClose);
        closeAccountingStage();
    }

    /** Preserves the original open failure and attaches any retryable cleanup failure. */
    synchronized void rollbackFailedOpen(Throwable openFailure, NativeClose nativeClose) {
        rollbackFailedOpen(openFailure, nativeClose, null);
    }

    /** Permit-aware construction rollback for family-fenced native creation. */
    synchronized void rollbackFailedOpen(
            Throwable openFailure,
            NativeClose nativeClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit) {
        Objects.requireNonNull(openFailure, "openFailure");
        Objects.requireNonNull(nativeClose, "nativeClose");
        try {
            close(nativeClose);
        } catch (RuntimeException | Error cleanupFailure) {
            suppress(openFailure, cleanupFailure);
            retainFailedConstruction(nativeClose, constructionPermit, openFailure);
        }
    }

    private void retainFailedConstruction(
            NativeClose nativeClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit,
            Throwable openFailure) {
        GpuProcessNativeCleanupRegistry.CleanupStage nativeCleanup =
                () -> retryNativeStage(nativeClose);
        GpuProcessNativeCleanupRegistry.CleanupStage accountingCleanup =
                this::retryAccountingStage;
        try {
            if (constructionPermit == null) {
                GpuProcessNativeCleanupRegistry.retainAfterFailedConstruction(
                        cleanupFamily, this, nativeCleanup, accountingCleanup);
            } else {
                constructionPermit.transfer(this, nativeCleanup, accountingCleanup);
            }
        } catch (RuntimeException | Error transferFailure) {
            suppress(openFailure, transferFailure);
        }
    }

    private synchronized void retryNativeStage(NativeClose nativeClose) {
        closeNativeStage(nativeClose);
    }

    private synchronized void retryAccountingStage() {
        closeAccountingStage();
    }

    private void closeNativeStage(NativeClose nativeClose) {
        if (nativeCloseCompleted) {
            return;
        }
        if (!opened || closed) {
            throw new IllegalStateException("native owner cleanup lost its open owner");
        }
        nativeClose.close();
        nativeCloseCompleted = true;
    }

    private void closeAccountingStage() {
        if (!opened || closed || !nativeCloseCompleted) {
            throw new IllegalStateException(
                    "native owner accounting ran before native close");
        }
        if (!GpuProcessResourceMetrics.recordNativePoolStatsChecked(
                owner, 0L, 0L, 0L)) {
            throw new IllegalStateException("native owner zero-stats accounting was rejected");
        }
        if (!GpuProcessResourceMetrics.recordNativePoolOwnerClosedChecked(owner)) {
            throw new IllegalStateException("native owner close accounting was rejected");
        }
        closed = true;
    }

    private static void suppress(Throwable original, Throwable later) {
        if (later != original) {
            original.addSuppressed(later);
        }
    }

    synchronized boolean isOpenForTests() {
        return opened && !closed;
    }

    synchronized long ownerIdForTests() {
        return ownerId;
    }

    @FunctionalInterface
    interface NativeStats {
        long[] read();
    }

    @FunctionalInterface
    interface NativeClose {
        void close();
    }

    private static final class Stats {
        private final long activeBytes;
        private final long activeReferences;
        private final long activeBatches;

        private Stats(long activeBytes, long activeReferences, long activeBatches) {
            this.activeBytes = activeBytes;
            this.activeReferences = activeReferences;
            this.activeBatches = activeBatches;
        }

        private static Stats requireLive(long[] values) {
            if (values == null || values.length != NATIVE_STATS_FIELDS) {
                throw new IllegalStateException(
                        "native owner stats must contain exactly "
                                + NATIVE_STATS_FIELDS
                                + " fields");
            }
            long activeBytes = values[0];
            long activeReferences = values[1];
            long activeBatches = values[2];
            if (activeBytes <= 0L
                    || activeReferences <= 0L
                    || activeBatches < 0L
                    || activeBatches > activeReferences) {
                throw new IllegalStateException(
                        "native owner stats violate the live bytes/reference/batch invariant");
            }
            return new Stats(activeBytes, activeReferences, activeBatches);
        }
    }
}
