// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.Objects;

/**
 * Identity-backed process accounting for one ordinary persistent JNI handle.
 *
 * <p>This lifecycle is only for native contexts that do not already expose authoritative native
 * pool or {@link GpuDeviceBatch} ownership. The Java helper itself is the identity owner. Native
 * creation must be followed immediately by {@link #opened(long, NativeHandleClose)}, before any
 * buffer lookup or configuration JNI call. A native close failure deliberately preserves both the
 * registered handle and its process owner so release evidence cannot report a false zero.
 */
final class GpuNativeKernelHandleLifecycle {
    private final GpuProcessNativeCleanupRegistry.Family cleanupFamily;
    private long registeredHandle;
    private long nativeCloseCompletedHandle;

    GpuNativeKernelHandleLifecycle() {
        this(GpuProcessNativeCleanupRegistry.Family.ORDINARY_NATIVE_KERNEL);
    }

    GpuNativeKernelHandleLifecycle(
            GpuProcessNativeCleanupRegistry.Family cleanupFamily) {
        this.cleanupFamily = Objects.requireNonNull(cleanupFamily, "cleanupFamily");
    }

    GpuProcessNativeCleanupRegistry.ConstructionPermit beginConstruction() {
        return GpuProcessNativeCleanupRegistry.acquireConstructionPermit(cleanupFamily);
    }

    /** Registers a newly created handle, cleaning it if process registration is rejected. */
    synchronized void opened(long nativeHandle, NativeHandleClose rejectedHandleClose) {
        opened(nativeHandle, rejectedHandleClose, null);
    }

    /** Permit-aware registration that retains an unregistered handle if rejection cleanup fails. */
    synchronized void opened(
            long nativeHandle,
            NativeHandleClose rejectedHandleClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit) {
        Objects.requireNonNull(rejectedHandleClose, "rejectedHandleClose");
        if (nativeHandle == 0L) {
            throw new IllegalArgumentException("native kernel handle must be non-zero");
        }
        if (registeredHandle != 0L) {
            throw new IllegalStateException("native kernel lifecycle is already open");
        }
        if (GpuProcessResourceMetrics.recordNativeKernelOpened(this, nativeHandle)) {
            registeredHandle = nativeHandle;
            return;
        }

        IllegalStateException rejected =
                new IllegalStateException(
                        "native kernel handle lifecycle registration was rejected");
        try {
            rejectedHandleClose.close(nativeHandle);
        } catch (RuntimeException | Error closeFailure) {
            suppress(rejected, closeFailure);
            retainUnregisteredFailedConstruction(
                    nativeHandle,
                    rejectedHandleClose,
                    constructionPermit,
                    rejected);
        }
        throw rejected;
    }

    private void retainUnregisteredFailedConstruction(
            long nativeHandle,
            NativeHandleClose nativeHandleClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit,
            Throwable constructionFailure) {
        GpuProcessNativeCleanupRegistry.CleanupStage nativeCleanup =
                () -> nativeHandleClose.close(nativeHandle);
        GpuProcessNativeCleanupRegistry.CleanupStage accountingCleanup = () -> {};
        try {
            if (constructionPermit == null) {
                GpuProcessNativeCleanupRegistry.retainAfterFailedConstruction(
                        cleanupFamily, this, nativeCleanup, accountingCleanup);
            } else {
                constructionPermit.transfer(this, nativeCleanup, accountingCleanup);
            }
        } catch (RuntimeException | Error transferFailure) {
            suppress(constructionFailure, transferFailure);
        }
    }

    /** Closes and unregisters the exact handle, retaining it unchanged when native close fails. */
    synchronized long close(long currentHandle, NativeHandleClose nativeHandleClose) {
        Objects.requireNonNull(nativeHandleClose, "nativeHandleClose");
        if (currentHandle == 0L && registeredHandle == 0L) {
            return 0L;
        }
        if (currentHandle == 0L || registeredHandle != currentHandle) {
            throw new IllegalStateException(
                    "native kernel lifecycle handle does not match its registered owner");
        }
        closeNativeStage(currentHandle, nativeHandleClose);
        closeAccountingStage(currentHandle);
        return 0L;
    }

    /** Preserves a construction failure and attaches a retry-visible cleanup failure. */
    synchronized long rollbackFailedOpen(
            long currentHandle,
            Throwable constructionFailure,
            NativeHandleClose nativeHandleClose) {
        return rollbackFailedOpen(
                currentHandle, constructionFailure, nativeHandleClose, null);
    }

    /**
     * Permit-aware construction rollback for wrappers that fence native creation by family.
     *
     * <p>The compatibility overload remains available while wrappers are migrated one family at
     * a time. New wiring must acquire the permit before native creation and pass that same permit
     * here when immediate cleanup fails.
     */
    synchronized long rollbackFailedOpen(
            long currentHandle,
            Throwable constructionFailure,
            NativeHandleClose nativeHandleClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit) {
        Objects.requireNonNull(constructionFailure, "constructionFailure");
        Objects.requireNonNull(nativeHandleClose, "nativeHandleClose");
        try {
            return close(currentHandle, nativeHandleClose);
        } catch (RuntimeException | Error closeFailure) {
            suppress(constructionFailure, closeFailure);
            retainFailedConstruction(
                    currentHandle,
                    nativeHandleClose,
                    constructionPermit,
                    constructionFailure);
            return currentHandle;
        }
    }

    private void retainFailedConstruction(
            long currentHandle,
            NativeHandleClose nativeHandleClose,
            GpuProcessNativeCleanupRegistry.ConstructionPermit constructionPermit,
            Throwable constructionFailure) {
        GpuProcessNativeCleanupRegistry.CleanupStage nativeCleanup =
                () -> retryNativeStage(currentHandle, nativeHandleClose);
        GpuProcessNativeCleanupRegistry.CleanupStage accountingCleanup =
                () -> retryAccountingStage(currentHandle);
        try {
            if (constructionPermit == null) {
                GpuProcessNativeCleanupRegistry.retainAfterFailedConstruction(
                        cleanupFamily, this, nativeCleanup, accountingCleanup);
            } else {
                constructionPermit.transfer(this, nativeCleanup, accountingCleanup);
            }
        } catch (RuntimeException | Error transferFailure) {
            suppress(constructionFailure, transferFailure);
        }
    }

    private synchronized void retryNativeStage(
            long currentHandle, NativeHandleClose nativeHandleClose) {
        closeNativeStage(currentHandle, nativeHandleClose);
    }

    private synchronized void retryAccountingStage(long currentHandle) {
        closeAccountingStage(currentHandle);
    }

    private void closeNativeStage(
            long currentHandle, NativeHandleClose nativeHandleClose) {
        if (nativeCloseCompletedHandle == currentHandle) {
            return;
        }
        if (registeredHandle != currentHandle || nativeCloseCompletedHandle != 0L) {
            throw new IllegalStateException(
                    "native kernel cleanup lost its exact registered handle");
        }
        nativeHandleClose.close(currentHandle);
        nativeCloseCompletedHandle = currentHandle;
    }

    private void closeAccountingStage(long currentHandle) {
        if (registeredHandle != currentHandle
                || nativeCloseCompletedHandle != currentHandle) {
            throw new IllegalStateException(
                    "native kernel accounting ran before exact native close");
        }
        if (!GpuProcessResourceMetrics.recordNativeKernelClosedChecked(
                this, currentHandle)) {
            throw new IllegalStateException(
                    "native kernel close accounting was rejected");
        }
        registeredHandle = 0L;
        nativeCloseCompletedHandle = 0L;
    }

    private static void suppress(Throwable original, Throwable later) {
        if (later != original) {
            original.addSuppressed(later);
        }
    }

    synchronized boolean isOpenForTests() {
        return registeredHandle != 0L;
    }

    synchronized long registeredHandleForTests() {
        return registeredHandle;
    }

    @FunctionalInterface
    interface NativeHandleClose {
        void close(long nativeHandle);
    }
}
