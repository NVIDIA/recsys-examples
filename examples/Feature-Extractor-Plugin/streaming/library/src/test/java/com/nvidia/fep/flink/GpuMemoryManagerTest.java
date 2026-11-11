// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuMemoryManagerTest {
    private static final GpuResourceConfig LIMITS =
            new GpuResourceConfig(
                    GpuResourceMode.EXTERNAL_RESOURCE, "gpu", 0, 400L, 250L, 100L);

    @AfterEach
    void clearBudgets() {
        GpuMemoryManager.resetForTests();
    }

    @Test
    void accountsChainedOperatorsAndReleasesReservations() {
        GpuMemoryReservation first =
                GpuMemoryManager.reserve(0, "slot-a", "Calc", LIMITS);
        GpuMemoryReservation second =
                GpuMemoryManager.reserve(0, "slot-a", "Aggregate", LIMITS);
        assertEquals(200L, first.taskReservedBytes());
        assertEquals(200L, first.deviceReservedBytes());

        IllegalStateException rejected =
                assertThrows(
                        IllegalStateException.class,
                        () -> GpuMemoryManager.reserve(0, "slot-a", "Join", LIMITS));
        assertTrue(rejected.getMessage().contains("task=slot-a reserved=200/250"));

        first.close();
        assertEquals(100L, second.taskReservedBytes());
        try (GpuMemoryReservation replacement =
                GpuMemoryManager.reserve(0, "slot-a", "Join", LIMITS)) {
            assertEquals(200L, replacement.taskReservedBytes());
        }
        second.close();
    }

    @Test
    void enforcesTaskManagerLimitAcrossTasksWithDiagnostic() {
        GpuResourceConfig deviceLimited =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE, "gpu", 0, 250L, 200L, 100L);
        GpuMemoryReservation first =
                GpuMemoryManager.reserve(0, "slot-a", "Calc", deviceLimited);
        GpuMemoryReservation second =
                GpuMemoryManager.reserve(0, "slot-b", "Join", deviceLimited);
        try {
            IllegalStateException rejected =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    GpuMemoryManager.reserve(
                                            0, "slot-c", "Aggregate", deviceLimited));
            assertTrue(rejected.getMessage().contains("TaskManager reserved=200/250"));
            assertTrue(rejected.getMessage().contains("requested=100 bytes"));
        } finally {
            first.close();
            second.close();
        }
    }

    @Test
    void rejectsConflictingTaskManagerBudgets() {
        GpuMemoryReservation active =
                GpuMemoryManager.reserve(0, "slot-a", "Calc", LIMITS);
        try {
            GpuResourceConfig conflicting =
                    new GpuResourceConfig(
                            GpuResourceMode.EXTERNAL_RESOURCE, "gpu", 0, 500L, 250L, 100L);
            IllegalStateException rejected =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    GpuMemoryManager.reserve(
                                            0, "slot-b", "Join", conflicting));
            assertTrue(rejected.getMessage().contains("Conflicting TaskManager GPU memory limits"));
        } finally {
            active.close();
        }
    }

    @Test
    void validatesLimitHierarchy() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new GpuResourceConfig(
                                GpuResourceMode.EXTERNAL_RESOURCE,
                                "gpu",
                                0,
                                100L,
                                101L,
                                1L));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new GpuResourceConfig(
                                GpuResourceMode.EXTERNAL_RESOURCE,
                                "gpu",
                                0,
                                100L,
                                50L,
                                51L));
    }
}
