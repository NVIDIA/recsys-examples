// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.TaskInfo;
import org.apache.flink.api.common.externalresource.ExternalResourceInfo;
import org.apache.flink.api.common.functions.RuntimeContext;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/** Resolves Flink external resources and enforces explicit per-device slot admission. */
final class GpuDeviceManager {
    private static final Map<String, DevicePool> DEVICE_POOLS = new HashMap<>();
    private static long legacyLeaseReferences;
    private static long accountingErrors;

    private GpuDeviceManager() {}

    static GpuDeviceLease acquire(GpuResourceConfig config, RuntimeContext runtimeContext) {
        if (config.mode() == GpuResourceMode.LEGACY_DEVICE) {
            return acquireLegacy(config.legacyDeviceIndex());
        }

        Set<ExternalResourceInfo> infos =
                runtimeContext.getExternalResourceInfos(config.resourceName());
        List<Integer> devices = parseDeviceIndexes(config.resourceName(), infos);
        TaskInfo taskInfo = runtimeContext.getTaskInfo();
        String allocationId = taskInfo.getAllocationIDAsString();
        if (allocationId == null
                || allocationId.trim().isEmpty()
                || "UNKNOWN".equals(allocationId)) {
            allocationId = taskInfo.getTaskNameWithSubtasks();
        }
        return acquireExternal(
                config.resourceName(),
                allocationId,
                devices,
                config.maxActiveSlotsPerDevice());
    }

    static List<Integer> parseDeviceIndexes(
            String resourceName, Set<ExternalResourceInfo> resourceInfos) {
        if (resourceInfos == null || resourceInfos.isEmpty()) {
            throw new IllegalStateException(
                    "GPU mode requires Flink external resource '"
                            + resourceName
                            + "', but this TaskManager reported none. Configure external-resources, "
                            + "external-resource."
                            + resourceName
                            + ".amount, and the Flink GPU driver; use legacy-device only for the "
                            + "embedded benchmark harness.");
        }

        Set<Integer> unique = new LinkedHashSet<>();
        for (ExternalResourceInfo info : resourceInfos) {
            Optional<String> value = info.getProperty("index");
            if (!value.isPresent()) {
                throw new IllegalStateException(
                        "Flink external resource '"
                                + resourceName
                                + "' did not expose the required GPU property 'index'; keys="
                                + info.getKeys());
            }
            try {
                int index = Integer.parseInt(value.get().trim());
                if (index < 0) {
                    throw new NumberFormatException("negative index");
                }
                unique.add(index);
            } catch (NumberFormatException error) {
                throw new IllegalStateException(
                        "Flink external resource '"
                                + resourceName
                                + "' returned invalid CUDA device index '"
                                + value.get()
                                + "'",
                        error);
            }
        }
        List<Integer> devices = new ArrayList<>(unique);
        Collections.sort(devices);
        return devices;
    }

    private static synchronized GpuDeviceLease acquireExternal(
            String resourceName,
            String allocationId,
            List<Integer> devices,
            int maxActiveSlotsPerDevice) {
        for (Map.Entry<String, DevicePool> entry : DEVICE_POOLS.entrySet()) {
            if (entry.getValue().maxActiveSlotsPerDevice != maxActiveSlotsPerDevice) {
                throw new IllegalStateException(
                        "Conflicting maximum active GPU slots per device in one TaskManager: "
                                + "active external resource '"
                                + entry.getKey()
                                + "' uses "
                                + entry.getValue().maxActiveSlotsPerDevice
                                + ", requested external resource '"
                                + resourceName
                                + "' uses "
                                + maxActiveSlotsPerDevice
                                + ". Every GPU operator in one TaskManager must use one fixed value.");
            }
        }
        DevicePool pool = DEVICE_POOLS.get(resourceName);
        if (pool == null) {
            pool = new DevicePool(maxActiveSlotsPerDevice);
            DEVICE_POOLS.put(resourceName, pool);
        }
        if (pool.maxActiveSlotsPerDevice != maxActiveSlotsPerDevice) {
            throw new IllegalStateException(
                    "Conflicting maximum active GPU slots per device for Flink external resource '"
                            + resourceName
                            + "': active="
                            + pool.maxActiveSlotsPerDevice
                            + ", requested="
                            + maxActiveSlotsPerDevice
                            + ". Every GPU operator in one TaskManager must use one fixed value.");
        }
        Assignment existing = pool.byAllocation.get(allocationId);
        if (existing != null) {
            if (!devices.contains(existing.deviceIndex)) {
                throw new IllegalStateException(
                        "Flink GPU assignment changed while slot "
                                + allocationId
                                + " was running; device "
                                + existing.deviceIndex
                                + " is no longer reported");
            }
            existing.references++;
            return GpuDeviceLease.external(resourceName, allocationId, existing.deviceIndex);
        }

        int selectedDevice = -1;
        int selectedAssignments = Integer.MAX_VALUE;
        int start = Math.floorMod(allocationId.hashCode(), devices.size());
        for (int offset = 0; offset < devices.size(); offset++) {
            int device = devices.get((start + offset) % devices.size());
            Set<String> active = pool.allocationsByDevice.get(device);
            int activeAssignments = active == null ? 0 : active.size();
            if (activeAssignments < maxActiveSlotsPerDevice
                    && activeAssignments < selectedAssignments) {
                selectedDevice = device;
                selectedAssignments = activeAssignments;
            }
        }
        if (selectedDevice >= 0) {
            pool.allocationsByDevice
                    .computeIfAbsent(selectedDevice, ignored -> new LinkedHashSet<>())
                    .add(allocationId);
            pool.byAllocation.put(allocationId, new Assignment(selectedDevice));
            return GpuDeviceLease.external(resourceName, allocationId, selectedDevice);
        }
        throw new IllegalStateException(
                "No admitted GPU slot from Flink external resource '"
                        + resourceName
                        + "' for slot "
                        + allocationId
                        + "; reported devices="
                        + devices
                        + ", max active slots per device="
                        + maxActiveSlotsPerDevice
                        + ", active slots="
                        + pool.allocationsByDevice);
    }

    static synchronized int activeAssignmentsOnDevice(GpuDeviceLease lease) {
        if (!lease.externalResource()) {
            return 0;
        }
        DevicePool pool = DEVICE_POOLS.get(lease.resourceName());
        if (pool == null) {
            return 0;
        }
        Set<String> active = pool.allocationsByDevice.get(lease.deviceIndex());
        return active == null ? 0 : active.size();
    }

    private static synchronized GpuDeviceLease acquireLegacy(int deviceIndex) {
        legacyLeaseReferences++;
        return GpuDeviceLease.legacy(deviceIndex);
    }

    static synchronized void release(GpuDeviceLease lease) {
        if (!lease.externalResource()) {
            if (legacyLeaseReferences <= 0L) {
                accountingErrors++;
            } else {
                legacyLeaseReferences--;
            }
            return;
        }
        DevicePool pool = DEVICE_POOLS.get(lease.resourceName());
        if (pool == null) {
            accountingErrors++;
            return;
        }
        Assignment assignment = pool.byAllocation.get(lease.allocationId());
        if (assignment == null || assignment.deviceIndex != lease.deviceIndex()) {
            accountingErrors++;
            return;
        }
        assignment.references--;
        if (assignment.references == 0) {
            pool.byAllocation.remove(lease.allocationId());
            Set<String> active = pool.allocationsByDevice.get(lease.deviceIndex());
            if (active == null || !active.remove(lease.allocationId())) {
                accountingErrors++;
            } else if (active.isEmpty()) {
                pool.allocationsByDevice.remove(lease.deviceIndex());
            }
        }
        if (pool.byAllocation.isEmpty()) {
            DEVICE_POOLS.remove(lease.resourceName());
        }
    }

    static synchronized ResourceSnapshot resourceSnapshot() {
        long leaseReferences = legacyLeaseReferences;
        long externalAssignments = 0L;
        long activeExternalDevices = 0L;
        long maximumActiveAssignmentsPerDevice = 0L;
        long configuredMaxActiveSlotsPerDevice = 0L;
        boolean consistent = accountingErrors == 0L && legacyLeaseReferences >= 0L;
        for (DevicePool pool : DEVICE_POOLS.values()) {
            externalAssignments += pool.byAllocation.size();
            if (configuredMaxActiveSlotsPerDevice == 0L) {
                configuredMaxActiveSlotsPerDevice = pool.maxActiveSlotsPerDevice;
            } else if (configuredMaxActiveSlotsPerDevice != pool.maxActiveSlotsPerDevice) {
                consistent = false;
            }
            int deviceAssignments = 0;
            for (Set<String> allocations : pool.allocationsByDevice.values()) {
                deviceAssignments += allocations.size();
                activeExternalDevices++;
                maximumActiveAssignmentsPerDevice =
                        Math.max(maximumActiveAssignmentsPerDevice, allocations.size());
                if (allocations.isEmpty()
                        || allocations.size() > pool.maxActiveSlotsPerDevice) {
                    consistent = false;
                }
            }
            if (pool.byAllocation.size() != deviceAssignments) {
                consistent = false;
            }
            for (Map.Entry<String, Assignment> entry : pool.byAllocation.entrySet()) {
                Assignment assignment = entry.getValue();
                leaseReferences += assignment.references;
                if (assignment.references <= 0
                        || !pool.allocationsByDevice
                                .getOrDefault(assignment.deviceIndex, Collections.emptySet())
                                .contains(entry.getKey())) {
                    consistent = false;
                }
            }
        }
        return new ResourceSnapshot(
                leaseReferences,
                externalAssignments,
                activeExternalDevices,
                maximumActiveAssignmentsPerDevice,
                configuredMaxActiveSlotsPerDevice,
                legacyLeaseReferences,
                accountingErrors,
                consistent);
    }

    static synchronized void resetResourceMetricsForTests() {
        if (!DEVICE_POOLS.isEmpty() || legacyLeaseReferences != 0L) {
            throw new IllegalStateException(
                    "cannot reset GPU device metrics while leases are active");
        }
        accountingErrors = 0L;
    }

    static final class ResourceSnapshot {
        final long activeLeaseReferences;
        final long activeExternalAssignments;
        final long activeExternalDevices;
        final long maximumActiveAssignmentsPerDevice;
        final long configuredMaxActiveSlotsPerDevice;
        final long activeLegacyLeases;
        final long accountingErrors;
        final boolean consistent;

        private ResourceSnapshot(
                long activeLeaseReferences,
                long activeExternalAssignments,
                long activeExternalDevices,
                long maximumActiveAssignmentsPerDevice,
                long configuredMaxActiveSlotsPerDevice,
                long activeLegacyLeases,
                long accountingErrors,
                boolean consistent) {
            this.activeLeaseReferences = activeLeaseReferences;
            this.activeExternalAssignments = activeExternalAssignments;
            this.activeExternalDevices = activeExternalDevices;
            this.maximumActiveAssignmentsPerDevice = maximumActiveAssignmentsPerDevice;
            this.configuredMaxActiveSlotsPerDevice = configuredMaxActiveSlotsPerDevice;
            this.activeLegacyLeases = activeLegacyLeases;
            this.accountingErrors = accountingErrors;
            this.consistent = consistent;
        }
    }

    private static final class DevicePool {
        private final Map<String, Assignment> byAllocation = new HashMap<>();
        private final Map<Integer, Set<String>> allocationsByDevice = new HashMap<>();
        private final int maxActiveSlotsPerDevice;

        private DevicePool(int maxActiveSlotsPerDevice) {
            this.maxActiveSlotsPerDevice = maxActiveSlotsPerDevice;
        }
    }

    private static final class Assignment {
        private final int deviceIndex;
        private int references = 1;

        private Assignment(int deviceIndex) {
            this.deviceIndex = deviceIndex;
        }
    }
}

/** One operator's reference to a TaskManager-local CUDA device assignment. */
final class GpuDeviceLease implements AutoCloseable {
    private final String resourceName;
    private final String allocationId;
    private final int deviceIndex;
    private final boolean externalResource;
    private final AtomicBoolean closed = new AtomicBoolean();

    private GpuDeviceLease(
            String resourceName,
            String allocationId,
            int deviceIndex,
            boolean externalResource) {
        this.resourceName = resourceName;
        this.allocationId = allocationId;
        this.deviceIndex = deviceIndex;
        this.externalResource = externalResource;
    }

    static GpuDeviceLease external(String resourceName, String allocationId, int deviceIndex) {
        return new GpuDeviceLease(resourceName, allocationId, deviceIndex, true);
    }

    static GpuDeviceLease legacy(int deviceIndex) {
        return new GpuDeviceLease("legacy", "shared", deviceIndex, false);
    }

    String resourceName() {
        return resourceName;
    }

    String allocationId() {
        return allocationId;
    }

    int deviceIndex() {
        return deviceIndex;
    }

    boolean externalResource() {
        return externalResource;
    }

    int activeAssignmentsOnDevice() {
        return GpuDeviceManager.activeAssignmentsOnDevice(this);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            GpuDeviceManager.release(this);
        }
    }
}
