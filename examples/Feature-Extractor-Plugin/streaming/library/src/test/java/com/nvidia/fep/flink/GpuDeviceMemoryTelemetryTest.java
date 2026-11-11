// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class GpuDeviceMemoryTelemetryTest {
    @BeforeEach
    void initialize() {
        GpuDeviceMemoryTelemetry.resetForTests();
    }

    @AfterEach
    void reset() {
        GpuDeviceMemoryTelemetry.resetForTests();
    }

    @Test
    void tracksWholeDeviceHighWaterIndependentlyPerDevice() {
        GpuDeviceMemoryTelemetry.DeviceSample first =
                GpuDeviceMemoryTelemetry.recordSampleForTests(0, 900L, 1_000L, 1L);
        GpuDeviceMemoryTelemetry.DeviceSample peak =
                GpuDeviceMemoryTelemetry.recordSampleForTests(0, 600L, 1_000L, 2L);
        GpuDeviceMemoryTelemetry.DeviceSample released =
                GpuDeviceMemoryTelemetry.recordSampleForTests(0, 800L, 1_000L, 3L);
        GpuDeviceMemoryTelemetry.DeviceSample other =
                GpuDeviceMemoryTelemetry.recordSampleForTests(1, 950L, 1_000L, 4L);

        assertEquals(100L, first.usedBytes);
        assertEquals(400L, peak.highWaterBytes);
        assertEquals(200L, released.usedBytes);
        assertEquals(400L, released.highWaterBytes);
        assertEquals(50L, other.highWaterBytes);
    }

    @Test
    void rejectsImpossibleSamples() {
        assertThrows(
                IllegalArgumentException.class,
                () -> GpuDeviceMemoryTelemetry.recordSampleForTests(0, 1_001L, 1_000L, 1L));
        assertThrows(
                IllegalArgumentException.class,
                () -> GpuDeviceMemoryTelemetry.recordSampleForTests(-1, 1L, 1_000L, 1L));
    }
}
