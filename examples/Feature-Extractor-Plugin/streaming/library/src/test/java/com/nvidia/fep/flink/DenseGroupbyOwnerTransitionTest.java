// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DenseGroupbyOwnerTransitionTest {
    private static final long DEVICE_BATCH_CHAIN_ID = 0x6d6567617701L;

    @BeforeEach
    void resetMetrics() {
        GpuProcessResourceMetrics.resetForTests();
        GpuDeviceManager.resetResourceMetricsForTests();
        GpuMemoryManager.resetForTests();
    }

    @AfterEach
    void requireCleanTestProcess() {
        GpuProcessResourceMetrics.snapshot().assertFullyReleased();
        GpuProcessResourceMetrics.resetForTests();
        GpuDeviceManager.resetResourceMetricsForTests();
        GpuMemoryManager.resetForTests();
    }

    @Test
    void contextToDeviceBatchOwnerTransitionNeverDropsOwnerAndIsRetryable() {
        Object owner = new Object();
        GpuNativeHandleResourceLifecycle lifecycle =
                GpuNativeHandleResourceLifecycle.standalone(owner);
        long contextOwnerId = lifecycle.ownerIdForTests();

        lifecycle.opened(() -> new long[] {64L, 1L, 0L});
        assertSingleLiveOwner(64L, 1L, 0L);
        assertEquals(contextOwnerId, lifecycle.ownerIdForTests());

        IllegalStateException statsFailure =
                new IllegalStateException("injected post-pool stats failure");
        IllegalStateException observed =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                lifecycle.transitionToOwner(
                                        DEVICE_BATCH_CHAIN_ID,
                                        () -> {
                                            throw statsFailure;
                                        }));
        assertSame(statsFailure, observed);
        assertEquals(contextOwnerId, lifecycle.ownerIdForTests());
        assertSingleLiveOwner(64L, 1L, 0L);

        lifecycle.transitionToOwner(
                DEVICE_BATCH_CHAIN_ID, () -> new long[] {192L, 1L, 0L});
        assertEquals(DEVICE_BATCH_CHAIN_ID, lifecycle.ownerIdForTests());
        assertSingleLiveOwner(192L, 1L, 0L);

        // Retrying an already completed transition updates stats without opening a second owner.
        lifecycle.transitionToOwner(
                DEVICE_BATCH_CHAIN_ID, () -> new long[] {224L, 2L, 1L});
        assertSingleLiveOwner(224L, 2L, 1L);

        AtomicInteger nativeCloses = new AtomicInteger();
        IllegalStateException closeFailure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                lifecycle.close(
                                        () -> {
                                            if (nativeCloses.incrementAndGet() == 1) {
                                                throw new IllegalStateException(
                                                        "injected native close failure");
                                            }
                                        }));
        assertEquals("injected native close failure", closeFailure.getMessage());
        assertTrue(lifecycle.isOpenForTests());
        assertSingleLiveOwner(224L, 2L, 1L);

        lifecycle.close(nativeCloses::incrementAndGet);
        lifecycle.close(nativeCloses::incrementAndGet);
        assertEquals(2, nativeCloses.get());
        GpuProcessResourceMetrics.Snapshot released = GpuProcessResourceMetrics.snapshot();
        assertEquals(0L, released.nativePoolOwners);
        assertEquals(1L, released.nativePoolOpens);
        assertEquals(1L, released.nativePoolCleanCloses);
        assertEquals(0L, released.accountingErrors);
        released.assertFullyReleased();
    }

    @Test
    void denseWiringRegistersBeforeJniLookupsAndClosesThroughTheOwner() throws IOException {
        String source =
                readRepositoryFile(
                        "src/main/java/com/nvidia/fep/flink/DenseGroupbyKernel.java");
        int lifecycle =
                source.indexOf(
                        "this.nativeResources = GpuNativeHandleResourceLifecycle.standalone(this)");
        int nullGuard = source.indexOf("if (handle == 0)", lifecycle);
        int opened =
                source.indexOf(
                        "nativeResources.opened(this::readNativeResourceStats)", nullGuard);
        int firstBufferLookup = source.indexOf("nativeInputBuffer(handle)", opened);
        int cleanup = source.indexOf("private void cleanupFailedOpen", firstBufferLookup);
        int cleanupThroughOwner =
                source.indexOf("nativeResources.close(this::closeNativeContext)", cleanup);
        int publicClose = source.indexOf("public void close()", cleanupThroughOwner);
        int ordinaryCloseThroughOwner =
                source.indexOf("nativeResources.close(this::closeNativeContext)", publicClose);

        assertTrue(lifecycle >= 0, "DenseGroupby must always create an ordinary context owner");
        assertTrue(nullGuard > lifecycle, "native create must precede its null-handle guard");
        assertTrue(opened > nullGuard, "the context owner must register after create succeeds");
        assertTrue(
                firstBufferLookup > opened,
                "the context owner must register before the first JNI buffer lookup");
        assertTrue(
                cleanupThroughOwner > cleanup && cleanupThroughOwner < publicClose,
                "failed construction must close through the registered owner");
        assertTrue(
                ordinaryCloseThroughOwner > publicClose,
                "close must go through the registered owner");
        assertFalse(
                source.contains("GpuProcessResourceMetrics.recordNativePoolOpened("),
                "DenseGroupby must not open a second registry owner after configuration");
        assertFalse(
                source.contains("GpuProcessResourceMetrics.recordNativePoolOwnerClosed("),
                "DenseGroupby must not claim a clean close outside the retryable lifecycle");
        assertEquals(
                source.indexOf("nativeClose(handle);"),
                source.lastIndexOf("nativeClose(handle);"),
                "native context close must have one lifecycle-owned call site");
        assertEquals(
                source.indexOf("nativeCloseFlexible(handle);"),
                source.lastIndexOf("nativeCloseFlexible(handle);"),
                "flexible context close must have one lifecycle-owned call site");
    }

    private static void assertSingleLiveOwner(
            long bytes, long references, long batches) {
        GpuProcessResourceMetrics.Snapshot snapshot =
                GpuProcessResourceMetrics.snapshot();
        assertEquals(1L, snapshot.nativePoolOwners);
        assertEquals(1L, snapshot.nativePoolOpens);
        assertEquals(0L, snapshot.nativePoolCleanCloses);
        assertEquals(bytes, snapshot.nativePoolInUseBytes);
        assertEquals(references, snapshot.nativePoolActiveReferences);
        assertEquals(batches, snapshot.nativePoolInUseBatches);
        assertTrue(snapshot.nativePoolInUseBatchesObservable);
        assertEquals(0L, snapshot.accountingErrors);
        assertFalse(snapshot.fullyReleased);
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
