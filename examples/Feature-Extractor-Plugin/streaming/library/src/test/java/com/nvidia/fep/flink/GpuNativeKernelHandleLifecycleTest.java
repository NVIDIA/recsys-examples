// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuNativeKernelHandleLifecycleTest {
    @BeforeEach
    void resetMetrics() {
        GpuProcessResourceMetrics.resetForTests();
        GpuDeviceManager.resetResourceMetricsForTests();
        GpuMemoryManager.resetForTests();
    }

    @AfterEach
    void clearMetrics() {
        assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeKernelActive);
        GpuProcessResourceMetrics.resetForTests();
        GpuDeviceManager.resetResourceMetricsForTests();
        GpuMemoryManager.resetForTests();
        GpuProcessResourceMetrics.snapshot().assertFullyReleased();
    }

    @Test
    void registersImmediatelyAfterNativeCreate() {
        GpuNativeKernelHandleLifecycle lifecycle = new GpuNativeKernelHandleLifecycle();
        long handle = 0x9101L;

        lifecycle.opened(handle, ignored -> {});

        GpuProcessResourceMetrics.Snapshot active = GpuProcessResourceMetrics.snapshot();
        assertTrue(lifecycle.isOpenForTests());
        assertEquals(handle, lifecycle.registeredHandleForTests());
        assertEquals(1L, active.nativeKernelActive);
        assertEquals(1L, active.nativeKernelOpens);
        assertEquals(0L, active.nativeKernelCleanCloses);
        assertFalse(active.fullyReleased);

        assertEquals(0L, lifecycle.close(handle, ignored -> {}));
    }

    @Test
    void registrationRejectionClosesCandidateAndFailsAccountingClosed() {
        Object existingOwner = new Object();
        long collidingHandle = 0x9201L;
        assertTrue(
                GpuProcessResourceMetrics.recordNativeKernelOpened(
                        existingOwner, collidingHandle));
        GpuNativeKernelHandleLifecycle rejectedLifecycle =
                new GpuNativeKernelHandleLifecycle();
        AtomicInteger rejectedCloses = new AtomicInteger();

        IllegalStateException rejection =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                rejectedLifecycle.opened(
                                        collidingHandle,
                                        ignored -> rejectedCloses.incrementAndGet()));

        assertEquals(
                "native kernel handle lifecycle registration was rejected",
                rejection.getMessage());
        assertEquals(1, rejectedCloses.get());
        assertFalse(rejectedLifecycle.isOpenForTests());
        GpuProcessResourceMetrics.Snapshot rejected =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(1L, rejected.nativeKernelActive);
        assertEquals(1L, rejected.accountingErrors);
        assertFalse(rejected.fullyObservable);

        GpuProcessResourceMetrics.recordNativeKernelClosed(
                existingOwner, collidingHandle);
    }

    @Test
    void registrationRejectionCloseFailureIsRetainedByConstructionPermit() {
        Object existingOwner = new Object();
        long collidingHandle = 0x9202L;
        assertTrue(
                GpuProcessResourceMetrics.recordNativeKernelOpened(
                        existingOwner, collidingHandle));
        GpuNativeKernelHandleLifecycle rejectedLifecycle =
                new GpuNativeKernelHandleLifecycle();
        AtomicInteger rejectedCloses = new AtomicInteger();

        IllegalStateException rejection;
        try (GpuProcessNativeCleanupRegistry.ConstructionPermit permit =
                rejectedLifecycle.beginConstruction()) {
            rejection =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    rejectedLifecycle.opened(
                                            collidingHandle,
                                            ignored -> {
                                                if (rejectedCloses.incrementAndGet() == 1) {
                                                    throw new IllegalStateException(
                                                            "candidate close failed");
                                                }
                                            },
                                            permit));
        }

        assertEquals(1, rejectedCloses.get());
        assertEquals(1, rejection.getSuppressed().length);
        assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
        GpuProcessResourceMetrics.retryPendingNativeCleanupOnce();
        assertEquals(2, rejectedCloses.get());
        assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
        GpuProcessResourceMetrics.recordNativeKernelClosed(existingOwner, collidingHandle);
    }

    @Test
    void halfConstructedHandleClosesAndPreservesOriginalFailure() {
        GpuNativeKernelHandleLifecycle lifecycle = new GpuNativeKernelHandleLifecycle();
        long handle = 0x9301L;
        AtomicInteger nativeCloses = new AtomicInteger();
        IllegalStateException constructionFailure =
                new IllegalStateException("buffer validation failed");
        lifecycle.opened(handle, ignored -> {});

        long[] remaining = {handle};
        IllegalStateException observed =
                assertThrows(
                        IllegalStateException.class,
                        () -> {
                            try {
                                throw constructionFailure;
                            } catch (IllegalStateException failure) {
                                remaining[0] =
                                        lifecycle.rollbackFailedOpen(
                                                handle,
                                                failure,
                                                ignored -> nativeCloses.incrementAndGet());
                                throw failure;
                            }
                        });

        assertEquals(0L, remaining[0]);
        assertSame(constructionFailure, observed);
        assertEquals(0, constructionFailure.getSuppressed().length);
        assertEquals(1, nativeCloses.get());
        assertFalse(lifecycle.isOpenForTests());
        GpuProcessResourceMetrics.snapshot().assertFullyReleased();
    }

    @Test
    void halfConstructedCloseFailureRetainsOwnerAndIsSuppressed() {
        GpuNativeKernelHandleLifecycle lifecycle = new GpuNativeKernelHandleLifecycle();
        long handle = 0x9401L;
        AtomicInteger nativeCloses = new AtomicInteger();
        IllegalStateException constructionFailure =
                new IllegalStateException("configuration failed");
        IllegalStateException closeFailure =
                new IllegalStateException("native close failed");
        lifecycle.opened(handle, ignored -> {});

        long remaining =
                lifecycle.rollbackFailedOpen(
                        handle,
                        constructionFailure,
                        ignored -> {
                            if (nativeCloses.incrementAndGet() == 1) {
                                throw closeFailure;
                            }
                        });

        assertEquals(handle, remaining);
        assertEquals(1, constructionFailure.getSuppressed().length);
        assertSame(closeFailure, constructionFailure.getSuppressed()[0]);
        assertTrue(lifecycle.isOpenForTests());
        GpuProcessResourceMetrics.Snapshot retained =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(1L, retained.nativeKernelActive);
        assertEquals(1L, retained.nativeCleanupPending);
        assertTrue(retained.nativeCleanupObservable);
        assertFalse(retained.fullyReleased);

        GpuProcessResourceMetrics.retryPendingNativeCleanupOnce();
        assertEquals(2, nativeCloses.get());
        assertFalse(lifecycle.isOpenForTests());
        assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
        GpuProcessResourceMetrics.snapshot().assertFullyReleased();
    }

    @Test
    void retainedCleanupFailsFirstSweepAndSucceedsOnTheNext() {
        AtomicInteger nativeAttempts = new AtomicInteger();
        AtomicInteger accountingAttempts = new AtomicInteger();
        try (GpuProcessNativeCleanupRegistry.ConstructionPermit permit =
                GpuProcessNativeCleanupRegistry.acquireConstructionPermit(
                        GpuProcessNativeCleanupRegistry.Family.FUSED_FILTER_PROJECT)) {
            permit.transfer(
                    new Object(),
                    () -> {
                        if (nativeAttempts.incrementAndGet() == 1) {
                            throw new IllegalStateException("first retry failed");
                        }
                    },
                    accountingAttempts::incrementAndGet);
        }

        GpuProcessResourceMetrics.Snapshot retained =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(1L, retained.nativeCleanupPending);
        assertTrue(retained.nativeCleanupObservable);
        assertFalse(retained.fullyReleased);
        assertEquals(0, nativeAttempts.get(), "snapshot must remain read-only");

        IllegalStateException firstFailure =
                assertThrows(
                        IllegalStateException.class,
                        GpuProcessResourceMetrics::retryPendingNativeCleanupOnce);
        assertEquals("first retry failed", firstFailure.getMessage());
        assertEquals(1, nativeAttempts.get());
        assertEquals(0, accountingAttempts.get());
        assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);

        GpuProcessResourceMetrics.retryPendingNativeCleanupOnce();
        assertEquals(2, nativeAttempts.get());
        assertEquals(1, accountingAttempts.get());
        GpuProcessResourceMetrics.Snapshot released =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(0L, released.nativeCleanupPending);
        released.assertFullyReleased();
    }

    @Test
    void accountingRetryNeverRepeatsSuccessfulNativeCleanup() {
        AtomicInteger nativeAttempts = new AtomicInteger();
        AtomicInteger accountingAttempts = new AtomicInteger();
        try (GpuProcessNativeCleanupRegistry.ConstructionPermit permit =
                GpuProcessNativeCleanupRegistry.acquireConstructionPermit(
                        GpuProcessNativeCleanupRegistry.Family.DENSE_GROUPBY)) {
            permit.transfer(
                    new Object(),
                    nativeAttempts::incrementAndGet,
                    () -> {
                        if (accountingAttempts.incrementAndGet() == 1) {
                            throw new IllegalStateException("accounting failed");
                        }
                    });
        }

        assertThrows(
                IllegalStateException.class,
                GpuProcessResourceMetrics::retryPendingNativeCleanupOnce);
        assertEquals(1, nativeAttempts.get());
        assertEquals(1, accountingAttempts.get());
        assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);

        GpuProcessResourceMetrics.retryPendingNativeCleanupOnce();
        assertEquals(1, nativeAttempts.get());
        assertEquals(2, accountingAttempts.get());
        GpuProcessResourceMetrics.snapshot().assertFullyReleased();
    }

    @Test
    void sameFamilyConstructionPermitsAreMutuallyExclusive() throws Exception {
        GpuProcessNativeCleanupRegistry.ConstructionPermit first =
                GpuProcessNativeCleanupRegistry.acquireConstructionPermit(
                        GpuProcessNativeCleanupRegistry.Family.CALC_EXPAND_LOCAL_DEVICE_CHAIN);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CountDownLatch secondStarted = new CountDownLatch(1);
        CountDownLatch secondAcquired = new CountDownLatch(1);
        try {
            Future<?> second =
                    executor.submit(
                            () -> {
                                secondStarted.countDown();
                                try (GpuProcessNativeCleanupRegistry.ConstructionPermit ignored =
                                        GpuProcessNativeCleanupRegistry
                                                .acquireConstructionPermit(
                                                        GpuProcessNativeCleanupRegistry.Family
                                                                .CALC_EXPAND_LOCAL_DEVICE_CHAIN)) {
                                    secondAcquired.countDown();
                                }
                            });
            assertTrue(secondStarted.await(5L, TimeUnit.SECONDS));
            assertFalse(secondAcquired.await(100L, TimeUnit.MILLISECONDS));
            assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);

            first.close();
            assertTrue(secondAcquired.await(5L, TimeUnit.SECONDS));
            second.get(5L, TimeUnit.SECONDS);
            assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
        } finally {
            first.close();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5L, TimeUnit.SECONDS));
        }
    }

    @Test
    void concurrentSweepsSerializeOnePendingCallback() throws Exception {
        AtomicInteger activeCallbacks = new AtomicInteger();
        AtomicInteger maximumCallbacks = new AtomicInteger();
        AtomicInteger nativeAttempts = new AtomicInteger();
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch releaseCallback = new CountDownLatch(1);
        try (GpuProcessNativeCleanupRegistry.ConstructionPermit permit =
                GpuProcessNativeCleanupRegistry.acquireConstructionPermit(
                        GpuProcessNativeCleanupRegistry.Family
                                .JOIN_CALC_LOCAL_DEVICE_CHAIN)) {
            permit.transfer(
                    new Object(),
                    () -> {
                        int active = activeCallbacks.incrementAndGet();
                        maximumCallbacks.accumulateAndGet(active, Math::max);
                        nativeAttempts.incrementAndGet();
                        callbackEntered.countDown();
                        try {
                            if (!releaseCallback.await(5L, TimeUnit.SECONDS)) {
                                throw new IllegalStateException("callback release timed out");
                            }
                        } catch (InterruptedException interrupted) {
                            Thread.currentThread().interrupt();
                            throw new IllegalStateException(
                                    "callback retry was interrupted", interrupted);
                        } finally {
                            activeCallbacks.decrementAndGet();
                        }
                    },
                    () -> {});
        }

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<?> first =
                    executor.submit(
                            GpuProcessResourceMetrics::retryPendingNativeCleanupOnce);
            assertTrue(callbackEntered.await(5L, TimeUnit.SECONDS));
            assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
            Future<?> second =
                    executor.submit(
                            GpuProcessResourceMetrics::retryPendingNativeCleanupOnce);
            assertFalse(second.isDone());

            releaseCallback.countDown();
            first.get(5L, TimeUnit.SECONDS);
            second.get(5L, TimeUnit.SECONDS);
            assertEquals(1, nativeAttempts.get());
            assertEquals(1, maximumCallbacks.get());
            assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeCleanupPending);
        } finally {
            releaseCallback.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5L, TimeUnit.SECONDS));
        }
    }

    @Test
    void normalCloseFailureIsRetryableAndRepeatedCloseIsIdempotent() {
        GpuNativeKernelHandleLifecycle lifecycle = new GpuNativeKernelHandleLifecycle();
        long handle = 0x9501L;
        AtomicInteger nativeCloses = new AtomicInteger();
        lifecycle.opened(handle, ignored -> {});

        IllegalStateException closeFailure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                lifecycle.close(
                                        handle,
                                        ignored -> {
                                            nativeCloses.incrementAndGet();
                                            throw new IllegalStateException("first close failed");
                                        }));
        assertEquals("first close failed", closeFailure.getMessage());
        assertTrue(lifecycle.isOpenForTests());
        assertEquals(handle, lifecycle.registeredHandleForTests());
        assertEquals(1L, GpuProcessResourceMetrics.snapshot().nativeKernelActive);
        assertEquals(0L, GpuProcessResourceMetrics.snapshot().nativeKernelCleanCloses);

        long closed =
                lifecycle.close(
                        handle, ignored -> nativeCloses.incrementAndGet());
        assertEquals(0L, closed);
        assertEquals(
                0L,
                lifecycle.close(
                        closed, ignored -> nativeCloses.incrementAndGet()));
        assertEquals(2, nativeCloses.get());
        GpuProcessResourceMetrics.Snapshot released =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(0L, released.nativeKernelActive);
        assertEquals(1L, released.nativeKernelOpens);
        assertEquals(1L, released.nativeKernelCleanCloses);
        released.assertFullyReleased();
    }

    @Test
    void migratedKernelsUseTheSharedLifecycle()
            throws Exception {
        assertEquals(
                GpuNativeKernelHandleLifecycle.class,
                LookupJoinKernel.class
                        .getDeclaredField("nativeHandleLifecycle")
                        .getType());
    }

    @Test
    void ordinaryKernelsRegisterBeforeBufferJniAndRollbackConstructionFailures()
            throws IOException {
        assertOrdinaryKernelLifecycleWiring(
                "LookupJoinKernel.java", "nativeDimensionKeysBuffer(handle)");
    }

    @Test
    void lookupJoinWrapperFencesNativeBuffersWhileCloseRetryIsPending()
            throws Exception {
        assertRetryFence("LookupJoinKernel.java", LookupJoinKernel.class, "ensureOpen();", 4);
    }

    private static void assertRetryFence(
            String fileName,
            Class<?> kernelClass,
            String guardCall,
            int minimumGuardCalls)
            throws Exception {
        assertTrue(
                Modifier.isVolatile(kernelClass.getDeclaredField("closePending").getModifiers()),
                fileName + " close-pending state must be visible before native release");
        String source =
                readRepositoryFile("src/main/java/com/nvidia/fep/flink/" + fileName);
        int close = source.indexOf("public synchronized void close()");
        int pending = source.indexOf("closePending = true;", close);
        String closeAttemptText;
        if (fileName.equals("NativeHashJoinKernel.java")
                || fileName.equals("PersistentHashJoinKernel.java")
                || fileName.equals("LookupJoinKernel.java")) {
            closeAttemptText = "nativeHandleLifecycle.close(";
        } else {
            closeAttemptText = "nativeResources.close(";
        }
        int closeAttempt = source.indexOf(closeAttemptText, pending);
        int cleared = source.indexOf("closePending = false;", closeAttempt);
        int retryFailure =
                source.indexOf("catch (RuntimeException | Error failure)", closeAttempt);
        assertTrue(close >= 0, fileName + " close must be synchronized");
        assertTrue(
                pending > close && closeAttempt > pending && cleared > closeAttempt,
                fileName + " must clear close-pending only after native close succeeds");
        assertTrue(
                retryFailure > cleared,
                fileName + " must propagate a retry-visible native close failure");
        {
            int closeCallback = source.indexOf("private void closeNative", retryFailure);
            String nativeCloseText = "nativeClose(handle);";
            if (fileName.equals("PersistentTopNKernel.java")) {
                nativeCloseText = "nativePersistentClose(handle);";
            } else if (fileName.equals("PersistentHashJoinKernel.java")) {
                nativeCloseText = "nativePersistentClose(closingHandle);";
            } else if (fileName.equals("NativeHashJoinKernel.java")) {
                nativeCloseText = "nativeClose(closingHandle);";
            } else if (fileName.equals("LookupJoinKernel.java")) {
                nativeCloseText = "nativeClose(closingHandle);";
            }
            int bind = source.indexOf("deviceGuard.bind();", closeCallback);
            int nativeClose = source.indexOf(nativeCloseText, bind);
            assertTrue(
                    closeCallback > retryFailure && bind > closeCallback && nativeClose > bind,
                    fileName + " retry callback must bind the device before native close");
        }
        assertTrue(
                countOccurrences(source, guardCall) >= minimumGuardCalls,
                fileName + " must fence every direct-buffer execution entry point");
        assertTrue(
                source.contains("if (closePending)"),
                fileName + " guard must reject close-retry-pending access");
    }

    private static int countOccurrences(String source, String expected) {
        int count = 0;
        int offset = 0;
        while ((offset = source.indexOf(expected, offset)) >= 0) {
            count++;
            offset += expected.length();
        }
        return count;
    }

    private static void assertOrdinaryKernelLifecycleWiring(
            String fileName, String firstBufferLookup) throws IOException {
        assertOrdinaryKernelLifecycleWiring(
                fileName,
                "nativeCreate(",
                firstBufferLookup,
                "nativeClose");
    }

    private static void assertOrdinaryKernelLifecycleWiring(
            String fileName,
            String nativeCreateCall,
            String firstBufferLookup,
            String nativeCloseMethod) throws IOException {
        String source =
                readRepositoryFile(
                        "src/main/java/com/nvidia/fep/flink/" + fileName);
        int create = source.indexOf(nativeCreateCall);
        int nullGuard = source.indexOf("if (handle == 0L)", create);
        int opened = source.indexOf("nativeHandleLifecycle.opened(", nullGuard);
        int guardedConstruction = source.indexOf("try {", opened);
        int bufferLookup = source.indexOf(firstBufferLookup, opened);
        int failureCatch =
                source.indexOf(
                        "} catch (RuntimeException | Error constructionFailure)",
                        bufferLookup);
        int rollback =
                source.indexOf(
                        "nativeHandleLifecycle.rollbackFailedOpen(", failureCatch);
        int exactClose =
                source.indexOf(
                        "handle = nativeHandleLifecycle.close(handle, this::closeNativeHandle)",
                        rollback);
        int closeCallback =
                source.indexOf(
                        "private void closeNativeHandle(long closingHandle)", exactClose);

        assertTrue(create >= 0, fileName + " must create a native handle");
        assertTrue(nullGuard > create, fileName + " must reject a null native handle");
        assertTrue(
                opened > nullGuard,
                fileName + " must register immediately after the null-handle guard");
        assertTrue(
                guardedConstruction > opened && bufferLookup > guardedConstruction,
                fileName + " must register before the first JNI buffer lookup");
        assertTrue(
                failureCatch > bufferLookup && rollback > failureCatch,
                fileName + " must roll back every post-registration construction failure");
        assertTrue(
                exactClose > rollback,
                fileName + " must preserve the handle until exact lifecycle close succeeds");
        assertTrue(
                closeCallback > exactClose,
                fileName + " must close through its device-bound native callback");
        int closeBind = source.indexOf("deviceGuard.bind();", closeCallback);
        int nativeClose =
                source.indexOf(nativeCloseMethod + "(closingHandle);", closeCallback);
        assertTrue(
                closeBind > closeCallback && nativeClose > closeBind,
                fileName + " close callback must bind the device before native close");
        assertFalse(
                source.contains(nativeCloseMethod + "(handle);"),
                fileName + " must not bypass retry-visible lifecycle ownership");
    }

    private static String readRepositoryFile(String relativePath) throws IOException {
        Path directory = Paths.get("").toAbsolutePath().normalize();
        for (int depth = 0; depth < 5 && directory != null; depth++) {
            Path candidate = directory.resolve(relativePath);
            if (Files.isRegularFile(candidate)) {
                return new String(Files.readAllBytes(candidate), StandardCharsets.UTF_8);
            }
            directory = directory.getParent();
        }
        throw new IllegalStateException("repository file is unavailable: " + relativePath);
    }
}
