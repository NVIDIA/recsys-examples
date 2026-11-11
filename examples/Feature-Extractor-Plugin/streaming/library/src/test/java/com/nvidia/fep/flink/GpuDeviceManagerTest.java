// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.TaskInfo;
import org.apache.flink.api.common.externalresource.ExternalResourceInfo;
import org.apache.flink.api.common.functions.RuntimeContext;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Proxy;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GpuDeviceManagerTest {
    private static final GpuResourceConfig EXTERNAL =
            new GpuResourceConfig(GpuResourceMode.EXTERNAL_RESOURCE, "gpu", 0);

    @AfterEach
    void resetDeviceAccounting() {
        GpuDeviceManager.resetResourceMetricsForTests();
    }

    @Test
    void parsesAndSortsFlinkGpuIndexes() {
        Set<ExternalResourceInfo> infos =
                Set.of(resource("2"), resource("0"), resource("1"));
        assertEquals(Arrays.asList(0, 1, 2), GpuDeviceManager.parseDeviceIndexes("gpu", infos));
    }

    @Test
    void rejectsMissingOrInvalidFlinkGpuInformation() {
        IllegalStateException missing =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                GpuDeviceManager.acquire(
                                        EXTERNAL, runtime("slot-a", Collections.emptySet())));
        assertTrue(missing.getMessage().contains("reported none"));

        assertThrows(
                IllegalStateException.class,
                () ->
                        GpuDeviceManager.parseDeviceIndexes(
                                "gpu", Collections.singleton(resource("GPU-0"))));

        ExternalResourceInfo withoutIndex =
                new ExternalResourceInfo() {
                    @Override
                    public Optional<String> getProperty(String key) {
                        return Optional.empty();
                    }

                    @Override
                    public Collection<String> getKeys() {
                        return Collections.singleton("uuid");
                    }
                };
        assertThrows(
                IllegalStateException.class,
                () ->
                        GpuDeviceManager.parseDeviceIndexes(
                                "gpu", Collections.singleton(withoutIndex)));
    }

    @Test
    void sharesOneDeviceInsideSlotAndPreventsCrossSlotCollisionByDefault() {
        Set<ExternalResourceInfo> oneGpu = Collections.singleton(resource("0"));
        GpuDeviceLease first = GpuDeviceManager.acquire(EXTERNAL, runtime("slot-a", oneGpu));
        GpuDeviceLease chained = GpuDeviceManager.acquire(EXTERNAL, runtime("slot-a", oneGpu));
        try {
            assertEquals(0, first.deviceIndex());
            assertEquals(first.deviceIndex(), chained.deviceIndex());
            assertThrows(
                    IllegalStateException.class,
                    () -> GpuDeviceManager.acquire(EXTERNAL, runtime("slot-b", oneGpu)));
            first.close();
            assertThrows(
                    IllegalStateException.class,
                    () -> GpuDeviceManager.acquire(EXTERNAL, runtime("slot-b", oneGpu)));
        } finally {
            first.close();
            chained.close();
        }

        try (GpuDeviceLease next =
                GpuDeviceManager.acquire(EXTERNAL, runtime("slot-b", oneGpu))) {
            assertEquals(0, next.deviceIndex());
        }
    }

    @Test
    void explicitAdmissionSharesOneDeviceAcrossBoundedSlotsAndReleasesExactly() {
        GpuResourceConfig shared =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE,
                        "gpu",
                        0,
                        2,
                        400L,
                        250L,
                        100L);
        Set<ExternalResourceInfo> oneGpu = Collections.singleton(resource("0"));
        GpuDeviceLease first = GpuDeviceManager.acquire(shared, runtime("slot-a", oneGpu));
        GpuDeviceLease chained = GpuDeviceManager.acquire(shared, runtime("slot-a", oneGpu));
        GpuDeviceLease second = GpuDeviceManager.acquire(shared, runtime("slot-b", oneGpu));
        try {
            assertEquals(0, first.deviceIndex());
            assertEquals(0, chained.deviceIndex());
            assertEquals(0, second.deviceIndex());
            GpuDeviceManager.ResourceSnapshot active = GpuDeviceManager.resourceSnapshot();
            assertEquals(3L, active.activeLeaseReferences);
            assertEquals(2L, active.activeExternalAssignments);
            assertEquals(1L, active.activeExternalDevices);
            assertEquals(2L, active.maximumActiveAssignmentsPerDevice);
            assertEquals(2L, active.configuredMaxActiveSlotsPerDevice);
            assertEquals(2, first.activeAssignmentsOnDevice());
            assertTrue(active.consistent);

            assertThrows(
                    IllegalStateException.class,
                    () -> GpuDeviceManager.acquire(shared, runtime("slot-c", oneGpu)));
            first.close();
            assertEquals(2, chained.activeAssignmentsOnDevice());
            assertThrows(
                    IllegalStateException.class,
                    () -> GpuDeviceManager.acquire(shared, runtime("slot-c", oneGpu)));
            chained.close();
            assertEquals(1, second.activeAssignmentsOnDevice());
            try (GpuDeviceLease replacement =
                    GpuDeviceManager.acquire(shared, runtime("slot-c", oneGpu))) {
                assertEquals(0, replacement.deviceIndex());
                assertEquals(
                        2L,
                        GpuDeviceManager.resourceSnapshot().activeExternalAssignments);
            }
        } finally {
            first.close();
            chained.close();
            second.close();
        }
        GpuDeviceManager.ResourceSnapshot released = GpuDeviceManager.resourceSnapshot();
        assertEquals(0L, released.activeLeaseReferences);
        assertEquals(0L, released.activeExternalAssignments);
        assertEquals(0L, released.activeExternalDevices);
        assertEquals(0L, released.maximumActiveAssignmentsPerDevice);
        assertEquals(0L, released.configuredMaxActiveSlotsPerDevice);
        assertTrue(released.consistent);
    }

    @Test
    void rejectsConflictingActiveSlotAdmissionLimits() {
        GpuResourceConfig shared =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE,
                        "gpu",
                        0,
                        2,
                        400L,
                        250L,
                        100L);
        Set<ExternalResourceInfo> oneGpu = Collections.singleton(resource("0"));
        try (GpuDeviceLease ignored =
                GpuDeviceManager.acquire(shared, runtime("slot-a", oneGpu))) {
            IllegalStateException conflict =
                    assertThrows(
                            IllegalStateException.class,
                            () -> GpuDeviceManager.acquire(EXTERNAL, runtime("slot-b", oneGpu)));
            assertTrue(conflict.getMessage().contains("Conflicting maximum active GPU slots"));
        }
    }

    @Test
    void rejectsConflictingAdmissionLimitsAcrossResourceNamesInOneTaskManager() {
        GpuResourceConfig shared =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE,
                        "gpu",
                        0,
                        2,
                        400L,
                        250L,
                        100L);
        GpuResourceConfig otherResource =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE,
                        "other-gpu",
                        0,
                        1,
                        400L,
                        250L,
                        100L);
        Set<ExternalResourceInfo> oneGpu = Collections.singleton(resource("0"));
        try (GpuDeviceLease ignored =
                GpuDeviceManager.acquire(shared, runtime("slot-a", oneGpu))) {
            IllegalStateException conflict =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    GpuDeviceManager.acquire(
                                            otherResource, runtime("slot-b", oneGpu)));
            assertTrue(conflict.getMessage().contains("in one TaskManager"));
            assertTrue(conflict.getMessage().contains("other-gpu"));
        }

        try (GpuDeviceLease admitted =
                GpuDeviceManager.acquire(otherResource, runtime("slot-b", oneGpu))) {
            assertEquals(1, admitted.activeAssignmentsOnDevice());
        }
    }

    @Test
    void validatesActiveSlotAdmissionBounds() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new GpuResourceConfig(
                                GpuResourceMode.EXTERNAL_RESOURCE,
                                "gpu",
                                0,
                                0,
                                400L,
                                250L,
                                100L));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new GpuResourceConfig(
                                GpuResourceMode.EXTERNAL_RESOURCE,
                                "gpu",
                                0,
                                65,
                                400L,
                                250L,
                                100L));
    }

    @Test
    void legacySerializedConfigRestoresSafeSingleSlotDefault() throws Exception {
        GpuResourceConfig legacySerializedShape =
                new GpuResourceConfig(
                        GpuResourceMode.EXTERNAL_RESOURCE,
                        "gpu",
                        0,
                        12,
                        400L,
                        250L,
                        100L);
        Field field = GpuResourceConfig.class.getDeclaredField("maxActiveSlotsPerDevice");
        field.setAccessible(true);
        field.setInt(legacySerializedShape, 0);

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream output = new ObjectOutputStream(bytes)) {
            output.writeObject(legacySerializedShape);
        }
        GpuResourceConfig restored;
        try (ObjectInputStream input =
                new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            restored = (GpuResourceConfig) input.readObject();
        }

        assertEquals(1, restored.maxActiveSlotsPerDevice());
        assertEquals(400L, restored.taskManagerMemoryLimitBytes());
        assertEquals(250L, restored.taskMemoryLimitBytes());
        assertEquals(100L, restored.operatorMemoryReservationBytes());
    }

    @Test
    void rejectedReleaseIsObservableAndDuplicateCloseIsIdempotent() {
        GpuDeviceLease missing = GpuDeviceLease.external("gpu", "missing-slot", 0);
        missing.close();
        missing.close();

        GpuDeviceManager.ResourceSnapshot rejected = GpuDeviceManager.resourceSnapshot();
        assertEquals(1L, rejected.accountingErrors);
        assertEquals(0L, rejected.activeLeaseReferences);
        assertTrue(!rejected.consistent);
    }

    @Test
    void legacyModeUsesConfiguredDeviceWithoutFlinkResource() {
        GpuResourceConfig legacy =
                new GpuResourceConfig(GpuResourceMode.LEGACY_DEVICE, "gpu", 3);
        try (GpuDeviceLease lease =
                GpuDeviceManager.acquire(legacy, runtime("slot-a", Collections.emptySet()))) {
            assertEquals(3, lease.deviceIndex());
        }
    }

    @Test
    void fallsBackToTaskIdentityWhenAllocationIdIsUnknown() {
        Set<ExternalResourceInfo> oneGpu = Collections.singleton(resource("0"));
        try (GpuDeviceLease first =
                GpuDeviceManager.acquire(
                        EXTERNAL, runtime("UNKNOWN", "gpu-task-a (1/1)#0", oneGpu))) {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            GpuDeviceManager.acquire(
                                    EXTERNAL,
                                    runtime("UNKNOWN", "gpu-task-b (1/1)#0", oneGpu)));
        }
    }

    private static ExternalResourceInfo resource(String index) {
        Map<String, String> properties = new HashMap<>();
        properties.put("index", index);
        return new ExternalResourceInfo() {
            @Override
            public Optional<String> getProperty(String key) {
                return Optional.ofNullable(properties.get(key));
            }

            @Override
            public Collection<String> getKeys() {
                return properties.keySet();
            }
        };
    }

    private static RuntimeContext runtime(
            String allocationId, Set<ExternalResourceInfo> resourceInfos) {
        return runtime(allocationId, "test-task (1/1)#0", resourceInfos);
    }

    private static RuntimeContext runtime(
            String allocationId,
            String taskNameWithSubtasks,
            Set<ExternalResourceInfo> resourceInfos) {
        TaskInfo taskInfo =
                (TaskInfo)
                        Proxy.newProxyInstance(
                                TaskInfo.class.getClassLoader(),
                                new Class<?>[] {TaskInfo.class},
                                (proxy, method, args) -> {
                                    if (method.getName().equals("getAllocationIDAsString")) {
                                        return allocationId;
                                    }
                                    if (method.getName().equals("getTaskNameWithSubtasks")) {
                                        return taskNameWithSubtasks;
                                    }
                                    throw new UnsupportedOperationException(method.getName());
                                });
        return (RuntimeContext)
                Proxy.newProxyInstance(
                        RuntimeContext.class.getClassLoader(),
                        new Class<?>[] {RuntimeContext.class},
                        (proxy, method, args) -> {
                            if (method.getName().equals("getExternalResourceInfos")) {
                                return resourceInfos;
                            }
                            if (method.getName().equals("getTaskInfo")) {
                                return taskInfo;
                            }
                            throw new UnsupportedOperationException(method.getName());
                        });
    }
}
