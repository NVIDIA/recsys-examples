// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.HashMap;
import java.util.Map;

/** Best-effort whole-device CUDA memory telemetry shared by operators in one TaskManager JVM. */
final class GpuDeviceMemoryTelemetry {
    private static final long CACHE_NANOS = 100_000_000L;
    private static final Map<Integer, DeviceSample> DEVICES = new HashMap<>();

    private GpuDeviceMemoryTelemetry() {}

    static long freeBytes(int deviceIndex) {
        return metricValue(deviceIndex, Value.FREE);
    }

    static long totalBytes(int deviceIndex) {
        return metricValue(deviceIndex, Value.TOTAL);
    }

    static long usedBytes(int deviceIndex) {
        return metricValue(deviceIndex, Value.USED);
    }

    static long highWaterBytes(int deviceIndex) {
        return metricValue(deviceIndex, Value.HIGH_WATER);
    }

    private static long metricValue(int deviceIndex, Value value) {
        try {
            DeviceSample sample = sample(deviceIndex);
            switch (value) {
                case FREE:
                    return sample.freeBytes;
                case TOTAL:
                    return sample.totalBytes;
                case USED:
                    return sample.usedBytes;
                case HIGH_WATER:
                    return sample.highWaterBytes;
                default:
                    throw new AssertionError(value);
            }
        } catch (RuntimeException | LinkageError unavailable) {
            // Metrics must not make an otherwise healthy operator fail. The CUDA operation itself
            // still reports errors through its normal JNI call; -1 marks telemetry unavailable.
            return -1L;
        }
    }

    private static synchronized DeviceSample sample(int deviceIndex) {
        long now = System.nanoTime();
        DeviceSample previous = DEVICES.get(deviceIndex);
        if (previous != null && now - previous.sampledNanos < CACHE_NANOS) {
            return previous;
        }
        long[] memory = CudaDeviceBinding.memoryInfo(deviceIndex);
        return recordSample(deviceIndex, memory[0], memory[1], now);
    }

    static synchronized DeviceSample recordSampleForTests(
            int deviceIndex, long freeBytes, long totalBytes, long sampledNanos) {
        return recordSample(deviceIndex, freeBytes, totalBytes, sampledNanos);
    }

    private static DeviceSample recordSample(
            int deviceIndex, long freeBytes, long totalBytes, long sampledNanos) {
        if (deviceIndex < 0 || freeBytes < 0L || totalBytes <= 0L || freeBytes > totalBytes) {
            throw new IllegalArgumentException(
                    "invalid CUDA memory sample device="
                            + deviceIndex
                            + " free="
                            + freeBytes
                            + " total="
                            + totalBytes);
        }
        long usedBytes = totalBytes - freeBytes;
        DeviceSample previous = DEVICES.get(deviceIndex);
        long highWaterBytes =
                previous == null
                        ? usedBytes
                        : Math.max(previous.highWaterBytes, usedBytes);
        DeviceSample next =
                new DeviceSample(
                        freeBytes, totalBytes, usedBytes, highWaterBytes, sampledNanos);
        DEVICES.put(deviceIndex, next);
        return next;
    }

    static synchronized void resetForTests() {
        DEVICES.clear();
    }

    static final class DeviceSample {
        final long freeBytes;
        final long totalBytes;
        final long usedBytes;
        final long highWaterBytes;
        final long sampledNanos;

        private DeviceSample(
                long freeBytes,
                long totalBytes,
                long usedBytes,
                long highWaterBytes,
                long sampledNanos) {
            this.freeBytes = freeBytes;
            this.totalBytes = totalBytes;
            this.usedBytes = usedBytes;
            this.highWaterBytes = highWaterBytes;
            this.sampledNanos = sampledNanos;
        }
    }

    private enum Value {
        FREE,
        TOTAL,
        USED,
        HIGH_WATER
    }
}
