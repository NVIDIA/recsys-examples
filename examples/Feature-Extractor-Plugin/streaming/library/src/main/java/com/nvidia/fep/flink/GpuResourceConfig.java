// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.configuration.ReadableConfig;

import java.io.Serializable;
import java.util.Objects;

/** Serializable deployment settings used by every project-owned GPU operator. */
public final class GpuResourceConfig implements Serializable {
    private static final long serialVersionUID = 1L;

    private final GpuResourceMode mode;
    private final String resourceName;
    private final int legacyDeviceIndex;
    private final int maxActiveSlotsPerDevice;
    private final long taskManagerMemoryLimitBytes;
    private final long taskMemoryLimitBytes;
    private final long operatorMemoryReservationBytes;

    public GpuResourceConfig(
            GpuResourceMode mode, String resourceName, int legacyDeviceIndex) {
        this(
                mode,
                resourceName,
                legacyDeviceIndex,
                CudfFlinkOptions.GPU_MAX_ACTIVE_SLOTS_PER_DEVICE.defaultValue(),
                CudfFlinkOptions.GPU_MEMORY_TASKMANAGER_LIMIT_BYTES.defaultValue(),
                CudfFlinkOptions.GPU_MEMORY_TASK_LIMIT_BYTES.defaultValue(),
                CudfFlinkOptions.GPU_MEMORY_OPERATOR_RESERVATION_BYTES.defaultValue());
    }

    public GpuResourceConfig(
            GpuResourceMode mode,
            String resourceName,
            int legacyDeviceIndex,
            long taskManagerMemoryLimitBytes,
            long taskMemoryLimitBytes,
            long operatorMemoryReservationBytes) {
        this(
                mode,
                resourceName,
                legacyDeviceIndex,
                CudfFlinkOptions.GPU_MAX_ACTIVE_SLOTS_PER_DEVICE.defaultValue(),
                taskManagerMemoryLimitBytes,
                taskMemoryLimitBytes,
                operatorMemoryReservationBytes);
    }

    public GpuResourceConfig(
            GpuResourceMode mode,
            String resourceName,
            int legacyDeviceIndex,
            int maxActiveSlotsPerDevice,
            long taskManagerMemoryLimitBytes,
            long taskMemoryLimitBytes,
            long operatorMemoryReservationBytes) {
        this.mode = Objects.requireNonNull(mode, "mode");
        this.resourceName = Objects.requireNonNull(resourceName, "resourceName").trim();
        if (this.resourceName.isEmpty()) {
            throw new IllegalArgumentException("GPU external-resource name must not be blank");
        }
        if (legacyDeviceIndex < 0) {
            throw new IllegalArgumentException("legacy CUDA device index must be non-negative");
        }
        this.legacyDeviceIndex = legacyDeviceIndex;
        if (maxActiveSlotsPerDevice <= 0 || maxActiveSlotsPerDevice > 64) {
            throw new IllegalArgumentException(
                    "maximum active GPU slots per device must be in [1, 64]");
        }
        this.maxActiveSlotsPerDevice = maxActiveSlotsPerDevice;
        if (taskManagerMemoryLimitBytes <= 0L) {
            throw new IllegalArgumentException("TaskManager GPU memory limit must be positive");
        }
        if (taskMemoryLimitBytes <= 0L
                || taskMemoryLimitBytes > taskManagerMemoryLimitBytes) {
            throw new IllegalArgumentException(
                    "task GPU memory limit must be positive and no larger than the TaskManager limit");
        }
        if (operatorMemoryReservationBytes <= 0L
                || operatorMemoryReservationBytes > taskMemoryLimitBytes) {
            throw new IllegalArgumentException(
                    "operator GPU memory reservation must be positive and no larger than the task limit");
        }
        this.taskManagerMemoryLimitBytes = taskManagerMemoryLimitBytes;
        this.taskMemoryLimitBytes = taskMemoryLimitBytes;
        this.operatorMemoryReservationBytes = operatorMemoryReservationBytes;
    }

    public static GpuResourceConfig from(ReadableConfig config) {
        return new GpuResourceConfig(
                GpuResourceMode.parse(config.get(CudfFlinkOptions.GPU_RESOURCE_MODE)),
                config.get(CudfFlinkOptions.GPU_RESOURCE_NAME),
                config.get(CudfFlinkOptions.GPU_LEGACY_DEVICE_INDEX),
                config.get(CudfFlinkOptions.GPU_MAX_ACTIVE_SLOTS_PER_DEVICE),
                config.get(CudfFlinkOptions.GPU_MEMORY_TASKMANAGER_LIMIT_BYTES),
                config.get(CudfFlinkOptions.GPU_MEMORY_TASK_LIMIT_BYTES),
                config.get(CudfFlinkOptions.GPU_MEMORY_OPERATOR_RESERVATION_BYTES));
    }

    public static GpuResourceConfig externalResourceDefaults() {
        return new GpuResourceConfig(GpuResourceMode.EXTERNAL_RESOURCE, "gpu", 0);
    }

    public GpuResourceMode mode() {
        return mode;
    }

    public String resourceName() {
        return resourceName;
    }

    public int legacyDeviceIndex() {
        return legacyDeviceIndex;
    }

    public int maxActiveSlotsPerDevice() {
        return maxActiveSlotsPerDevice;
    }

    public long taskManagerMemoryLimitBytes() {
        return taskManagerMemoryLimitBytes;
    }

    public long taskMemoryLimitBytes() {
        return taskMemoryLimitBytes;
    }

    public long operatorMemoryReservationBytes() {
        return operatorMemoryReservationBytes;
    }

    private Object readResolve() {
        // Jobs compiled before slot or memory admission existed deserialize the added fields as
        // zero. Preserve those job graphs/savepoints while restoring the safe single-slot default.
        if (maxActiveSlotsPerDevice > 0
                && taskManagerMemoryLimitBytes > 0L
                && taskMemoryLimitBytes > 0L
                && operatorMemoryReservationBytes > 0L) {
            return this;
        }
        if (taskManagerMemoryLimitBytes > 0L
                && taskMemoryLimitBytes > 0L
                && operatorMemoryReservationBytes > 0L) {
            return new GpuResourceConfig(
                    mode,
                    resourceName,
                    legacyDeviceIndex,
                    CudfFlinkOptions.GPU_MAX_ACTIVE_SLOTS_PER_DEVICE.defaultValue(),
                    taskManagerMemoryLimitBytes,
                    taskMemoryLimitBytes,
                    operatorMemoryReservationBytes);
        }
        return new GpuResourceConfig(mode, resourceName, legacyDeviceIndex);
    }

    @Override
    public String toString() {
        return mode == GpuResourceMode.EXTERNAL_RESOURCE
                ? mode.configValue()
                        + ':'
                        + resourceName
                        + ":max-active-slots-per-device="
                        + maxActiveSlotsPerDevice
                : mode.configValue() + ':' + legacyDeviceIndex;
    }
}
