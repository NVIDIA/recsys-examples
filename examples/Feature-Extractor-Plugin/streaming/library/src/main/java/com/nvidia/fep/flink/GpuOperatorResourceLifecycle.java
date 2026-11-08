// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.functions.RuntimeContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Objects;

/** Serializable configuration plus transient acquire/bind/release state for a GPU operator. */
final class GpuOperatorResourceLifecycle implements Serializable {
    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(GpuOperatorResourceLifecycle.class);

    private GpuResourceConfig config = GpuResourceConfig.externalResourceDefaults();
    private transient GpuDeviceLease lease;
    private transient GpuMemoryReservation memoryReservation;

    void configure(GpuResourceConfig newConfig) {
        if (lease != null) {
            throw new IllegalStateException("GPU resource cannot be changed after operator open");
        }
        config = Objects.requireNonNull(newConfig, "newConfig");
    }

    void open(RuntimeContext runtimeContext, String operatorName) {
        GpuDeviceLease acquired = GpuDeviceManager.acquire(config, runtimeContext);
        try {
            CudaDeviceBinding.bind(acquired.deviceIndex());
            memoryReservation =
                    GpuMemoryManager.reserve(
                            acquired.deviceIndex(),
                            acquired.allocationId(),
                            operatorName,
                            config);
        } catch (RuntimeException | Error error) {
            if (memoryReservation != null) {
                memoryReservation.close();
                memoryReservation = null;
            }
            acquired.close();
            throw error;
        }
        lease = acquired;
        final int deviceIndex = acquired.deviceIndex();
        final String resourceMode = config.mode().configValue();
        final String resourceName = config.resourceName();
        final GpuDeviceLease deviceLease = acquired;
        final int maxActiveSlotsPerDevice = config.maxActiveSlotsPerDevice();
        runtimeContext.getMetricGroup().gauge("cudfGpuDeviceIndex", () -> deviceIndex);
        runtimeContext.getMetricGroup().gauge("cudfGpuResourceMode", () -> resourceMode);
        runtimeContext.getMetricGroup().gauge("cudfGpuResourceName", () -> resourceName);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuActiveSlotsOnDevice", deviceLease::activeAssignmentsOnDevice);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuMaxActiveSlotsPerDevice", () -> maxActiveSlotsPerDevice);
        final GpuMemoryReservation reservation = memoryReservation;
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuMemoryReservedBytes", reservation::reservedBytes);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuTaskMemoryReservedBytes", reservation::taskReservedBytes);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuTaskMemoryLimitBytes", reservation::taskLimitBytes);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuTaskManagerMemoryReservedBytes", reservation::deviceReservedBytes);
        runtimeContext
                .getMetricGroup()
                .gauge("cudfGpuTaskManagerMemoryLimitBytes", reservation::deviceLimitBytes);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessActiveChains",
                        () -> GpuProcessResourceMetrics.snapshot().activeChains);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessActiveDeviceLeases",
                        () -> GpuProcessResourceMetrics.snapshot().activeDeviceLeases);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessActiveExternalAssignments",
                        () -> GpuProcessResourceMetrics.snapshot().activeExternalAssignments);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessActiveReservations",
                        () -> GpuProcessResourceMetrics.snapshot().activeReservations);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessReservedBytes",
                        () -> GpuProcessResourceMetrics.snapshot().reservedBytes);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessNativePoolOwners",
                        () -> GpuProcessResourceMetrics.snapshot().nativePoolOwners);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessNativePoolInUseBytes",
                        () -> GpuProcessResourceMetrics.snapshot().nativePoolInUseBytes);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessNativePoolActiveReferences",
                        () -> GpuProcessResourceMetrics.snapshot().nativePoolActiveReferences);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessNativePoolInUseBatches",
                        () -> GpuProcessResourceMetrics.snapshot().nativePoolInUseBatches);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessDeviceBatchActiveReferences",
                        () -> GpuProcessResourceMetrics.snapshot().deviceBatchActiveReferences);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessDeviceBatchActiveBatches",
                        () -> GpuProcessResourceMetrics.snapshot().deviceBatchActiveBatches);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessDeviceBatchInUseBytes",
                        () -> GpuProcessResourceMetrics.snapshot().deviceBatchInUseBytes);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuProcessResourceAccountingErrors",
                        () -> GpuProcessResourceMetrics.snapshot().accountingErrors);
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuDeviceMemoryFreeBytes",
                        () -> GpuDeviceMemoryTelemetry.freeBytes(deviceIndex));
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuDeviceMemoryTotalBytes",
                        () -> GpuDeviceMemoryTelemetry.totalBytes(deviceIndex));
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuDeviceMemoryUsedBytes",
                        () -> GpuDeviceMemoryTelemetry.usedBytes(deviceIndex));
        runtimeContext
                .getMetricGroup()
                .gauge(
                        "cudfGpuDeviceMemoryHighWaterBytes",
                        () -> GpuDeviceMemoryTelemetry.highWaterBytes(deviceIndex));
        long deviceMemoryUsed = GpuDeviceMemoryTelemetry.usedBytes(deviceIndex);
        long deviceMemoryHighWater = GpuDeviceMemoryTelemetry.highWaterBytes(deviceIndex);
        long deviceMemoryTotal = GpuDeviceMemoryTelemetry.totalBytes(deviceIndex);
        int activeSlotsOnDevice = acquired.activeAssignmentsOnDevice();
        LOG.info(
                "Bound GPU operator {} to CUDA device {} using {}; reserved {} bytes "
                        + "(task {}/{}, TaskManager {}/{}); active GPU slots on device {}/{}; "
                        + "CUDA device memory used/high-water/total {}/{}/{} bytes",
                operatorName,
                deviceIndex,
                config,
                reservation.reservedBytes(),
                reservation.taskReservedBytes(),
                reservation.taskLimitBytes(),
                reservation.deviceReservedBytes(),
                reservation.deviceLimitBytes(),
                activeSlotsOnDevice,
                maxActiveSlotsPerDevice,
                deviceMemoryUsed,
                deviceMemoryHighWater,
                deviceMemoryTotal);
    }

    void close() {
        if (memoryReservation != null) {
            memoryReservation.close();
            memoryReservation = null;
        }
        if (lease != null) {
            lease.close();
            lease = null;
        }
    }
}
