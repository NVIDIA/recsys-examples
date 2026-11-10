// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/** TaskManager-local admission control for independently allocated native GPU contexts. */
final class GpuMemoryManager {
    private static final Map<Integer, DeviceBudget> DEVICES = new HashMap<>();
    private static long accountingErrors;

    private GpuMemoryManager() {}

    static synchronized GpuMemoryReservation reserve(
            int deviceIndex,
            String taskIdentity,
            String operatorName,
            GpuResourceConfig config) {
        long requested = config.operatorMemoryReservationBytes();
        DeviceBudget device = DEVICES.get(deviceIndex);
        if (device == null) {
            device = new DeviceBudget(config.taskManagerMemoryLimitBytes());
            DEVICES.put(deviceIndex, device);
        } else if (device.limitBytes != config.taskManagerMemoryLimitBytes()) {
            throw new IllegalStateException(
                    "Conflicting TaskManager GPU memory limits on CUDA device "
                            + deviceIndex
                            + ": active="
                            + device.limitBytes
                            + ", requested="
                            + config.taskManagerMemoryLimitBytes()
                            + ". Configure one TaskManager-wide value for every job.");
        }

        TaskBudget task = device.tasks.get(taskIdentity);
        if (task == null) {
            task = new TaskBudget(config.taskMemoryLimitBytes());
            device.tasks.put(taskIdentity, task);
        } else if (task.limitBytes != config.taskMemoryLimitBytes()) {
            throw new IllegalStateException(
                    "Conflicting task GPU memory limits for "
                            + taskIdentity
                            + ": active="
                            + task.limitBytes
                            + ", requested="
                            + config.taskMemoryLimitBytes());
        }

        long nextTask = addExact(task.reservedBytes, requested, "task GPU reservation overflow");
        long nextDevice =
                addExact(device.reservedBytes, requested, "TaskManager GPU reservation overflow");
        if (nextTask > task.limitBytes || nextDevice > device.limitBytes) {
            if (task.reservations.isEmpty()) {
                device.tasks.remove(taskIdentity);
            }
            if (device.tasks.isEmpty()) {
                DEVICES.remove(deviceIndex);
            }
            throw new IllegalStateException(
                    "GPU memory admission rejected operator "
                            + operatorName
                            + " on CUDA device "
                            + deviceIndex
                            + ": requested="
                            + requested
                            + " bytes, task="
                            + taskIdentity
                            + " reserved="
                            + task.reservedBytes
                            + "/"
                            + task.limitBytes
                            + ", TaskManager reserved="
                            + device.reservedBytes
                            + "/"
                            + device.limitBytes
                            + ", active="
                            + task.reservations.values());
        }

        long id = ++device.nextReservationId;
        String description = operatorName + '#' + id + '=' + requested;
        task.reservations.put(id, description);
        task.reservedBytes = nextTask;
        device.reservedBytes = nextDevice;
        return new GpuMemoryReservation(
                deviceIndex,
                taskIdentity,
                id,
                requested,
                task.limitBytes,
                device.limitBytes);
    }

    static synchronized long taskReservedBytes(GpuMemoryReservation reservation) {
        DeviceBudget device = DEVICES.get(reservation.deviceIndex());
        if (device == null) {
            return 0L;
        }
        TaskBudget task = device.tasks.get(reservation.taskIdentity());
        return task == null ? 0L : task.reservedBytes;
    }

    static synchronized long deviceReservedBytes(GpuMemoryReservation reservation) {
        DeviceBudget device = DEVICES.get(reservation.deviceIndex());
        return device == null ? 0L : device.reservedBytes;
    }

    static synchronized void release(GpuMemoryReservation reservation) {
        DeviceBudget device = DEVICES.get(reservation.deviceIndex());
        if (device == null) {
            accountingErrors++;
            return;
        }
        TaskBudget task = device.tasks.get(reservation.taskIdentity());
        if (task == null) {
            accountingErrors++;
            return;
        }
        if (task.reservations.remove(reservation.id()) == null) {
            accountingErrors++;
            return;
        }
        task.reservedBytes -= reservation.reservedBytes();
        device.reservedBytes -= reservation.reservedBytes();
        if (task.reservations.isEmpty()) {
            device.tasks.remove(reservation.taskIdentity());
        }
        if (device.tasks.isEmpty()) {
            DEVICES.remove(reservation.deviceIndex());
        }
    }

    static synchronized ResourceSnapshot resourceSnapshot() {
        long activeReservations = 0L;
        long reservedBytes = 0L;
        long activeTasks = 0L;
        boolean consistent = accountingErrors == 0L;
        for (DeviceBudget device : DEVICES.values()) {
            long deviceReservationBytes = 0L;
            activeTasks += device.tasks.size();
            for (TaskBudget task : device.tasks.values()) {
                activeReservations += task.reservations.size();
                reservedBytes += task.reservedBytes;
                deviceReservationBytes += task.reservedBytes;
                if (task.reservations.isEmpty()
                        || task.reservedBytes < 0L
                        || task.reservedBytes > task.limitBytes) {
                    consistent = false;
                }
            }
            if (device.tasks.isEmpty()
                    || device.reservedBytes != deviceReservationBytes
                    || device.reservedBytes < 0L
                    || device.reservedBytes > device.limitBytes) {
                consistent = false;
            }
        }
        return new ResourceSnapshot(
                activeReservations,
                reservedBytes,
                DEVICES.size(),
                activeTasks,
                accountingErrors,
                consistent);
    }

    static synchronized void resetForTests() {
        if (!DEVICES.isEmpty()) {
            throw new IllegalStateException(
                    "cannot reset GPU memory metrics while reservations are active");
        }
        DEVICES.clear();
        accountingErrors = 0L;
    }

    static final class ResourceSnapshot {
        final long activeReservations;
        final long reservedBytes;
        final long activeDevices;
        final long activeTasks;
        final long accountingErrors;
        final boolean consistent;

        private ResourceSnapshot(
                long activeReservations,
                long reservedBytes,
                long activeDevices,
                long activeTasks,
                long accountingErrors,
                boolean consistent) {
            this.activeReservations = activeReservations;
            this.reservedBytes = reservedBytes;
            this.activeDevices = activeDevices;
            this.activeTasks = activeTasks;
            this.accountingErrors = accountingErrors;
            this.consistent = consistent;
        }
    }

    private static long addExact(long left, long right, String message) {
        try {
            return Math.addExact(left, right);
        } catch (ArithmeticException overflow) {
            throw new IllegalStateException(message, overflow);
        }
    }

    private static final class DeviceBudget {
        private final long limitBytes;
        private final Map<String, TaskBudget> tasks = new HashMap<>();
        private long reservedBytes;
        private long nextReservationId;

        private DeviceBudget(long limitBytes) {
            this.limitBytes = limitBytes;
        }
    }

    private static final class TaskBudget {
        private final long limitBytes;
        private final Map<Long, String> reservations = new LinkedHashMap<>();
        private long reservedBytes;

        private TaskBudget(long limitBytes) {
            this.limitBytes = limitBytes;
        }
    }
}

/** One operator's revocable TaskManager-local GPU memory reservation. */
final class GpuMemoryReservation implements AutoCloseable {
    private final int deviceIndex;
    private final String taskIdentity;
    private final long id;
    private final long reservedBytes;
    private final long taskLimitBytes;
    private final long deviceLimitBytes;
    private final AtomicBoolean closed = new AtomicBoolean();

    GpuMemoryReservation(
            int deviceIndex,
            String taskIdentity,
            long id,
            long reservedBytes,
            long taskLimitBytes,
            long deviceLimitBytes) {
        this.deviceIndex = deviceIndex;
        this.taskIdentity = taskIdentity;
        this.id = id;
        this.reservedBytes = reservedBytes;
        this.taskLimitBytes = taskLimitBytes;
        this.deviceLimitBytes = deviceLimitBytes;
    }

    int deviceIndex() {
        return deviceIndex;
    }

    String taskIdentity() {
        return taskIdentity;
    }

    long id() {
        return id;
    }

    long reservedBytes() {
        return reservedBytes;
    }

    long taskLimitBytes() {
        return taskLimitBytes;
    }

    long deviceLimitBytes() {
        return deviceLimitBytes;
    }

    long taskReservedBytes() {
        return GpuMemoryManager.taskReservedBytes(this);
    }

    long deviceReservedBytes() {
        return GpuMemoryManager.deviceReservedBytes(this);
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            GpuMemoryManager.release(this);
        }
    }
}
