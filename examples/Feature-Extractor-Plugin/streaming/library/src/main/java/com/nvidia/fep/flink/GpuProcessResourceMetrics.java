// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Process-local, owner-backed accounting for resources that must be gone after GPU teardown.
 *
 * <p>This registry is deliberately fed only from real owner transitions. It does not infer a
 * release from log messages. Device assignments and memory reservations are sampled directly from
 * their authoritative managers; chains, Java {@link GpuDeviceBatch} leases and native pool/context
 * owners register their actual object identities. Registered persistent native kernel wrappers
 * contribute the exact JNI handle they own; those handles are separate from DeviceBatch chains
 * and native pools. Native byte/reference/batch gauges come from each pool owner's JNI stats
 * response.
 */
public final class GpuProcessResourceMetrics {
    private static final String SCHEMA = "gpu-process-resource-release/v2";
    private static final Object LOCK = new Object();
    private static final IdentityHashMap<Object, Long> CHAINS = new IdentityHashMap<>();
    private static final IdentityHashMap<Object, BatchKey> DEVICE_BATCH_LEASES =
            new IdentityHashMap<>();
    private static final Map<BatchKey, BatchState> DEVICE_BATCHES = new LinkedHashMap<>();
    private static final IdentityHashMap<Object, NativePoolState> NATIVE_POOLS =
            new IdentityHashMap<>();
    private static final IdentityHashMap<Object, Long> NATIVE_KERNELS =
            new IdentityHashMap<>();

    private static long nativePoolOpens;
    private static long nativePoolCleanCloses;
    private static long nativeKernelOpens;
    private static long nativeKernelCleanCloses;
    private static long accountingErrors;

    private GpuProcessResourceMetrics() {}

    static void recordChainOpened(Object owner, long ownerChainId) {
        if (owner == null || ownerChainId == 0L) {
            recordAccountingError();
            return;
        }
        synchronized (LOCK) {
            if (CHAINS.put(owner, ownerChainId) != null) {
                accountingErrors++;
            }
        }
    }

    static void recordChainClosed(Object owner) {
        if (owner == null) {
            recordAccountingError();
            return;
        }
        synchronized (LOCK) {
            if (CHAINS.remove(owner) == null) {
                accountingErrors++;
            }
        }
    }

    /** Registers one persistent JNI kernel handle immediately after native creation succeeds. */
    static boolean recordNativeKernelOpened(Object owner, long nativeHandle) {
        if (owner == null || nativeHandle == 0L) {
            recordAccountingError();
            return false;
        }
        synchronized (LOCK) {
            boolean handleAlreadyRegistered = false;
            for (Long registeredHandle : NATIVE_KERNELS.values()) {
                if (registeredHandle.longValue() == nativeHandle) {
                    handleAlreadyRegistered = true;
                    break;
                }
            }
            if (NATIVE_KERNELS.containsKey(owner) || handleAlreadyRegistered) {
                accountingErrors++;
                return false;
            }
            NATIVE_KERNELS.put(owner, nativeHandle);
            nativeKernelOpens++;
            return true;
        }
    }

    /** Removes one owner only after the exact persistent JNI handle closed successfully. */
    static void recordNativeKernelClosed(Object owner, long nativeHandle) {
        recordNativeKernelClosedChecked(owner, nativeHandle);
    }

    /** Checked variant used by retry state machines that must retain a rejected owner. */
    static boolean recordNativeKernelClosedChecked(Object owner, long nativeHandle) {
        if (owner == null || nativeHandle == 0L) {
            recordAccountingError();
            return false;
        }
        synchronized (LOCK) {
            Long registeredHandle = NATIVE_KERNELS.get(owner);
            if (registeredHandle == null || registeredHandle.longValue() != nativeHandle) {
                accountingErrors++;
                return false;
            }
            NATIVE_KERNELS.remove(owner);
            nativeKernelCleanCloses++;
            return true;
        }
    }

    static void recordDeviceBatchLeaseOpened(
            Object lease, Object nativeAccess, long token, long dataBytes) {
        if (lease == null || nativeAccess == null || token == 0L || dataBytes < 0L) {
            recordAccountingError();
            return;
        }
        synchronized (LOCK) {
            BatchKey key = new BatchKey(nativeAccess, token);
            if (DEVICE_BATCH_LEASES.put(lease, key) != null) {
                accountingErrors++;
                return;
            }
            BatchState state = DEVICE_BATCHES.get(key);
            if (state == null) {
                DEVICE_BATCHES.put(key, new BatchState(dataBytes));
                return;
            }
            if (state.dataBytes != dataBytes) {
                accountingErrors++;
            }
            state.references++;
        }
    }

    static void recordDeviceBatchLeaseClosed(Object lease) {
        if (lease == null) {
            recordAccountingError();
            return;
        }
        synchronized (LOCK) {
            BatchKey key = DEVICE_BATCH_LEASES.remove(lease);
            if (key == null) {
                accountingErrors++;
                return;
            }
            BatchState state = DEVICE_BATCHES.get(key);
            if (state == null || state.references <= 0L) {
                accountingErrors++;
                return;
            }
            state.references--;
            if (state.references == 0L) {
                DEVICE_BATCHES.remove(key);
            }
        }
    }

    static void recordNativePoolOpened(Object owner, long ownerChainId) {
        if (owner == null || ownerChainId == 0L) {
            recordAccountingError();
            return;
        }
        synchronized (LOCK) {
            if (NATIVE_POOLS.containsKey(owner)) {
                accountingErrors++;
                return;
            }
            NATIVE_POOLS.put(owner, new NativePoolState(ownerChainId));
            nativePoolOpens++;
        }
    }

    /**
     * Atomically re-identifies one live native context as its configured DeviceBatch pool owner.
     *
     * <p>The owner object and registry entry never disappear during this transition. The caller
     * must first obtain a live JNI stats snapshot after native pool configuration; invalid stats
     * or a stale expected owner ID leave the original entry untouched and poison release evidence.
     */
    static boolean recordNativePoolOwnerTransitioned(
            Object owner,
            long expectedOwnerChainId,
            long newOwnerChainId,
            long activeBytes,
            long activeReferences,
            long activeBatches) {
        if (owner == null
                || expectedOwnerChainId == 0L
                || newOwnerChainId == 0L
                || activeBytes <= 0L
                || activeReferences <= 0L
                || activeBatches < 0L
                || activeBatches > activeReferences) {
            recordAccountingError();
            return false;
        }
        synchronized (LOCK) {
            NativePoolState state = NATIVE_POOLS.get(owner);
            // Accept the already-transitioned ID as an idempotent retry. No registry mutation can
            // make the owner temporarily disappear, even if a caller retries after an ambiguous
            // higher-level failure.
            if (state == null
                    || state.ownerClosed
                    || (state.ownerChainId != expectedOwnerChainId
                            && state.ownerChainId != newOwnerChainId)) {
                accountingErrors++;
                return false;
            }
            state.ownerChainId = newOwnerChainId;
            state.observed = true;
            state.activeBytes = activeBytes;
            state.activeReferences = activeReferences;
            state.activeBatchesObserved = true;
            state.activeBatches = activeBatches;
            return true;
        }
    }

    static void recordNativePoolStats(
            Object owner, long activeBytes, long activeReferences) {
        recordNativePoolStatsChecked(owner, activeBytes, activeReferences, -1L);
    }

    /** Records an exact native-owner snapshot, including active batches when the ABI exposes it. */
    static void recordNativePoolStats(
            Object owner, long activeBytes, long activeReferences, long activeBatches) {
        recordNativePoolStatsChecked(
                owner, activeBytes, activeReferences, activeBatches);
    }

    /** Checked variant used by retry state machines that must retain a rejected owner. */
    static boolean recordNativePoolStatsChecked(
            Object owner, long activeBytes, long activeReferences, long activeBatches) {
        if (owner == null
                || activeBytes < 0L
                || activeReferences < 0L
                || activeBatches < -1L
                || (activeBatches >= 0L && activeBatches > activeReferences)) {
            recordAccountingError();
            return false;
        }
        synchronized (LOCK) {
            NativePoolState state = NATIVE_POOLS.get(owner);
            if (state == null || state.ownerClosed) {
                accountingErrors++;
                return false;
            }
            state.observed = true;
            state.activeBytes = activeBytes;
            state.activeReferences = activeReferences;
            state.activeBatchesObserved = activeBatches >= 0L;
            state.activeBatches = activeBatches >= 0L ? activeBatches : 0L;
            if (activeReferences == 0L && activeBytes != 0L) {
                state.invalidReferenceInvariant = true;
                accountingErrors++;
                return false;
            }
            return true;
        }
    }

    static void recordNativePoolOwnerClosed(Object owner) {
        recordNativePoolOwnerClosedChecked(owner);
    }

    /** Checked variant used by retry state machines that must retain a rejected owner. */
    static boolean recordNativePoolOwnerClosedChecked(Object owner) {
        if (owner == null) {
            recordAccountingError();
            return false;
        }
        synchronized (LOCK) {
            NativePoolState state = NATIVE_POOLS.get(owner);
            if (state == null || state.ownerClosed) {
                accountingErrors++;
                return false;
            }
            if (!state.observed
                    || state.invalidReferenceInvariant
                    || state.activeBytes != 0L
                    || state.activeReferences != 0L) {
                accountingErrors++;
                return false;
            }
            state.ownerClosed = true;
            NATIVE_POOLS.remove(owner);
            nativePoolCleanCloses++;
            return true;
        }
    }

    /** Explicitly retries each retained native cleanup owner once. */
    public static void retryPendingNativeCleanupOnce() {
        GpuProcessNativeCleanupRegistry.retryPendingOnce();
    }

    /** Captures the current TaskManager-JVM release state. */
    public static Snapshot snapshot() {
        GpuDeviceManager.ResourceSnapshot deviceResources =
                GpuDeviceManager.resourceSnapshot();
        GpuMemoryManager.ResourceSnapshot memoryResources =
                GpuMemoryManager.resourceSnapshot();
        GpuProcessNativeCleanupRegistry.Snapshot cleanup =
                GpuProcessNativeCleanupRegistry.snapshot();
        LocalSnapshot local;
        synchronized (LOCK) {
            local = localSnapshot();
        }
        return new Snapshot(deviceResources, memoryResources, cleanup, local);
    }

    /**
     * Fails closed unless every tracked owner is gone and native pool zero is observable.
     *
     * <p>Call this only after the job/operator teardown path is quiescent. It intentionally does
     * not wait for concurrent owners to disappear.
     */
    public static void assertFullyReleased() {
        snapshot().assertFullyReleased();
    }

    static void resetForTests() {
        GpuProcessNativeCleanupRegistry.resetForTests();
        synchronized (LOCK) {
            if (!CHAINS.isEmpty()
                    || !DEVICE_BATCH_LEASES.isEmpty()
                    || !DEVICE_BATCHES.isEmpty()
                    || !NATIVE_POOLS.isEmpty()
                    || !NATIVE_KERNELS.isEmpty()) {
                throw new IllegalStateException(
                        "cannot reset GPU process resource metrics while owners are active");
            }
            nativePoolOpens = 0L;
            nativePoolCleanCloses = 0L;
            nativeKernelOpens = 0L;
            nativeKernelCleanCloses = 0L;
            accountingErrors = 0L;
        }
    }

    private static void recordAccountingError() {
        synchronized (LOCK) {
            accountingErrors++;
        }
    }

    private static LocalSnapshot localSnapshot() {
        long javaBatchBytes = 0L;
        for (BatchState state : DEVICE_BATCHES.values()) {
            javaBatchBytes += state.dataBytes;
        }

        long nativePoolOwners = 0L;
        long nativePoolBytes = 0L;
        long nativePoolReferences = 0L;
        boolean nativePoolStatsObservable = true;
        long nativePoolBatches = 0L;
        boolean nativePoolBatchesObservable = true;
        boolean exactNativePoolBatchesObserved = false;
        List<String> unresolved = new ArrayList<>();
        for (NativePoolState state : NATIVE_POOLS.values()) {
            if (!state.ownerClosed) {
                nativePoolOwners++;
            }
            if (!state.observed) {
                nativePoolStatsObservable = false;
                unresolved.add(
                        "native pool "
                                + Long.toUnsignedString(state.ownerChainId)
                                + " has no JNI stats observation");
                nativePoolBatchesObservable = false;
                continue;
            }
            nativePoolBytes += state.activeBytes;
            nativePoolReferences += state.activeReferences;
            if (state.invalidReferenceInvariant) {
                nativePoolStatsObservable = false;
                unresolved.add(
                        "native pool "
                                + Long.toUnsignedString(state.ownerChainId)
                                + " reported zero references with non-zero bytes");
            }
            if (state.ownerClosed
                    && (state.activeBytes != 0L || state.activeReferences != 0L)) {
                unresolved.add(
                        "closed native pool "
                                + Long.toUnsignedString(state.ownerChainId)
                                + " retained resources");
            }
            if (state.activeBatchesObserved) {
                nativePoolBatches += state.activeBatches;
                exactNativePoolBatchesObserved = true;
            } else if (state.activeReferences != 0L || state.activeBytes != 0L) {
                nativePoolBatchesObservable = false;
                unresolved.add(
                        "native pool "
                                + Long.toUnsignedString(state.ownerChainId)
                                + " ABI omits exact active batch count while references remain");
            }
        }

        String nativePoolBatchCountSource;
        if (!nativePoolStatsObservable || !nativePoolBatchesObservable) {
            nativePoolBatches = -1L;
            nativePoolBatchesObservable = false;
            nativePoolBatchCountSource = "unresolved-native-abi-omits-active-batches";
        } else if (exactNativePoolBatchesObserved) {
            nativePoolBatchCountSource = "exact-native-owner-stats";
        } else {
            // The native registry gives every non-free slot at least one reference and clears both
            // active_batches and active_bytes when its final reference is released. Therefore a
            // fully observed zero reference count proves zero in-use batches without guessing.
            nativePoolBatches = 0L;
            nativePoolBatchCountSource = "proven-zero-by-native-reference-invariant";
        }

        return new LocalSnapshot(
                CHAINS.size(),
                DEVICE_BATCH_LEASES.size(),
                DEVICE_BATCHES.size(),
                javaBatchBytes,
                nativePoolOwners,
                nativePoolOpens,
                nativePoolCleanCloses,
                nativePoolStatsObservable,
                nativePoolBytes,
                nativePoolReferences,
                nativePoolBatches,
                nativePoolBatchesObservable,
                nativePoolBatchCountSource,
                NATIVE_KERNELS.size(),
                nativeKernelOpens,
                nativeKernelCleanCloses,
                accountingErrors,
                unresolved);
    }

    private static final class BatchKey {
        private final Object nativeAccess;
        private final long token;

        private BatchKey(Object nativeAccess, long token) {
            this.nativeAccess = nativeAccess;
            this.token = token;
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) {
                return true;
            }
            if (!(other instanceof BatchKey)) {
                return false;
            }
            BatchKey that = (BatchKey) other;
            return nativeAccess == that.nativeAccess && token == that.token;
        }

        @Override
        public int hashCode() {
            return 31 * System.identityHashCode(nativeAccess) + Long.hashCode(token);
        }
    }

    private static final class BatchState {
        private final long dataBytes;
        private long references = 1L;

        private BatchState(long dataBytes) {
            this.dataBytes = dataBytes;
        }
    }

    private static final class NativePoolState {
        private long ownerChainId;
        private boolean observed;
        private boolean ownerClosed;
        private boolean invalidReferenceInvariant;
        private boolean activeBatchesObserved;
        private long activeBytes;
        private long activeReferences;
        private long activeBatches;

        private NativePoolState(long ownerChainId) {
            this.ownerChainId = ownerChainId;
        }
    }

    private static final class LocalSnapshot {
        private final long activeChains;
        private final long deviceBatchActiveReferences;
        private final long deviceBatchActiveBatches;
        private final long deviceBatchInUseBytes;
        private final long nativePoolOwners;
        private final long nativePoolOpens;
        private final long nativePoolCleanCloses;
        private final boolean nativePoolStatsObservable;
        private final long nativePoolInUseBytes;
        private final long nativePoolActiveReferences;
        private final long nativePoolInUseBatches;
        private final boolean nativePoolInUseBatchesObservable;
        private final String nativePoolBatchCountSource;
        private final long nativeKernelActive;
        private final long nativeKernelOpens;
        private final long nativeKernelCleanCloses;
        private final long accountingErrors;
        private final List<String> unresolved;

        private LocalSnapshot(
                long activeChains,
                long deviceBatchActiveReferences,
                long deviceBatchActiveBatches,
                long deviceBatchInUseBytes,
                long nativePoolOwners,
                long nativePoolOpens,
                long nativePoolCleanCloses,
                boolean nativePoolStatsObservable,
                long nativePoolInUseBytes,
                long nativePoolActiveReferences,
                long nativePoolInUseBatches,
                boolean nativePoolInUseBatchesObservable,
                String nativePoolBatchCountSource,
                long nativeKernelActive,
                long nativeKernelOpens,
                long nativeKernelCleanCloses,
                long accountingErrors,
                List<String> unresolved) {
            this.activeChains = activeChains;
            this.deviceBatchActiveReferences = deviceBatchActiveReferences;
            this.deviceBatchActiveBatches = deviceBatchActiveBatches;
            this.deviceBatchInUseBytes = deviceBatchInUseBytes;
            this.nativePoolOwners = nativePoolOwners;
            this.nativePoolOpens = nativePoolOpens;
            this.nativePoolCleanCloses = nativePoolCleanCloses;
            this.nativePoolStatsObservable = nativePoolStatsObservable;
            this.nativePoolInUseBytes = nativePoolInUseBytes;
            this.nativePoolActiveReferences = nativePoolActiveReferences;
            this.nativePoolInUseBatches = nativePoolInUseBatches;
            this.nativePoolInUseBatchesObservable = nativePoolInUseBatchesObservable;
            this.nativePoolBatchCountSource = nativePoolBatchCountSource;
            this.nativeKernelActive = nativeKernelActive;
            this.nativeKernelOpens = nativeKernelOpens;
            this.nativeKernelCleanCloses = nativeKernelCleanCloses;
            this.accountingErrors = accountingErrors;
            this.unresolved = Collections.unmodifiableList(new ArrayList<>(unresolved));
        }
    }

    /** Immutable release evidence suitable for an in-process assertion or JSON artifact. */
    public static final class Snapshot {
        public final String schema;
        public final long activeChains;
        public final long activeDeviceLeases;
        public final long activeExternalAssignments;
        public final long activeExternalDevices;
        public final long maximumActiveAssignmentsPerDevice;
        public final long configuredMaxActiveSlotsPerDevice;
        public final long activeLegacyLeases;
        public final long activeReservations;
        public final long reservedBytes;
        public final long reservationDevices;
        public final long reservationTasks;
        public final long nativePoolOwners;
        public final long nativePoolOpens;
        public final long nativePoolCleanCloses;
        public final boolean nativePoolStatsObservable;
        public final long nativePoolInUseBytes;
        public final long nativePoolActiveReferences;
        public final long nativePoolInUseBatches;
        public final boolean nativePoolInUseBatchesObservable;
        public final String nativePoolBatchCountSource;
        public final long nativeKernelActive;
        public final long nativeKernelOpens;
        public final long nativeKernelCleanCloses;
        public final long nativeCleanupPending;
        public final boolean nativeCleanupObservable;
        public final long deviceBatchActiveReferences;
        public final long deviceBatchActiveBatches;
        public final long deviceBatchInUseBytes;
        public final long forcedLeakedReferences;
        public final long accountingErrors;
        public final List<String> unresolved;
        public final boolean fullyObservable;
        public final boolean fullyReleased;

        private Snapshot(
                GpuDeviceManager.ResourceSnapshot device,
                GpuMemoryManager.ResourceSnapshot memory,
                GpuProcessNativeCleanupRegistry.Snapshot cleanup,
                LocalSnapshot local) {
            schema = SCHEMA;
            activeChains = local.activeChains;
            activeDeviceLeases = device.activeLeaseReferences;
            activeExternalAssignments = device.activeExternalAssignments;
            activeExternalDevices = device.activeExternalDevices;
            maximumActiveAssignmentsPerDevice = device.maximumActiveAssignmentsPerDevice;
            configuredMaxActiveSlotsPerDevice = device.configuredMaxActiveSlotsPerDevice;
            activeLegacyLeases = device.activeLegacyLeases;
            activeReservations = memory.activeReservations;
            reservedBytes = memory.reservedBytes;
            reservationDevices = memory.activeDevices;
            reservationTasks = memory.activeTasks;
            nativePoolOwners = local.nativePoolOwners;
            nativePoolOpens = local.nativePoolOpens;
            nativePoolCleanCloses = local.nativePoolCleanCloses;
            nativePoolStatsObservable = local.nativePoolStatsObservable;
            nativePoolInUseBytes =
                    nativePoolStatsObservable ? local.nativePoolInUseBytes : -1L;
            nativePoolActiveReferences =
                    nativePoolStatsObservable ? local.nativePoolActiveReferences : -1L;
            nativePoolInUseBatches = local.nativePoolInUseBatches;
            nativePoolInUseBatchesObservable =
                    local.nativePoolInUseBatchesObservable;
            nativePoolBatchCountSource = local.nativePoolBatchCountSource;
            nativeKernelActive = local.nativeKernelActive;
            nativeKernelOpens = local.nativeKernelOpens;
            nativeKernelCleanCloses = local.nativeKernelCleanCloses;
            nativeCleanupPending = cleanup.pending;
            nativeCleanupObservable = cleanup.observable;
            deviceBatchActiveReferences = local.deviceBatchActiveReferences;
            deviceBatchActiveBatches = local.deviceBatchActiveBatches;
            deviceBatchInUseBytes = local.deviceBatchInUseBytes;
            // No DeviceBatch leases exist in this library, so no forced-leak count is carried.
            forcedLeakedReferences = 0L;
            accountingErrors =
                    local.accountingErrors
                            + device.accountingErrors
                            + memory.accountingErrors
                            + (!device.consistent && device.accountingErrors == 0L ? 1L : 0L)
                            + (!memory.consistent && memory.accountingErrors == 0L ? 1L : 0L);
            List<String> reasons = new ArrayList<>(local.unresolved);
            if (!device.consistent) {
                reasons.add("GPU device manager accounting is inconsistent");
            }
            if (!memory.consistent) {
                reasons.add("GPU memory manager accounting is inconsistent");
            }
            unresolved = Collections.unmodifiableList(reasons);
            fullyObservable =
                    nativePoolStatsObservable
                            && nativePoolInUseBatchesObservable
                            && nativeCleanupObservable
                            && accountingErrors == 0L
                            && unresolved.isEmpty();
            fullyReleased =
                    fullyObservable
                            && activeChains == 0L
                            && activeDeviceLeases == 0L
                            && activeExternalAssignments == 0L
                            && activeExternalDevices == 0L
                            && maximumActiveAssignmentsPerDevice == 0L
                            && configuredMaxActiveSlotsPerDevice == 0L
                            && activeLegacyLeases == 0L
                            && activeReservations == 0L
                            && reservedBytes == 0L
                            && reservationDevices == 0L
                            && reservationTasks == 0L
                            && nativePoolOwners == 0L
                            && nativePoolInUseBytes == 0L
                            && nativePoolActiveReferences == 0L
                            && nativePoolInUseBatches == 0L
                            && nativeKernelActive == 0L
                            && nativeKernelOpens == nativeKernelCleanCloses
                            && nativeCleanupPending == 0L
                            && deviceBatchActiveReferences == 0L
                            && deviceBatchActiveBatches == 0L
                            && deviceBatchInUseBytes == 0L
                            && forcedLeakedReferences == 0L;
        }

        public void assertFullyReleased() {
            if (!fullyReleased) {
                throw new IllegalStateException(
                        "GPU process resources are not fully released: " + toJson());
            }
        }

        public String toJson() {
            StringBuilder json = new StringBuilder(768);
            json.append('{');
            append(json, "schema", schema).append(',');
            append(json, "active_chains", activeChains).append(',');
            append(json, "active_device_leases", activeDeviceLeases).append(',');
            append(json, "active_external_assignments", activeExternalAssignments).append(',');
            append(json, "active_external_devices", activeExternalDevices).append(',');
            append(
                            json,
                            "maximum_active_assignments_per_device",
                            maximumActiveAssignmentsPerDevice)
                    .append(',');
            append(
                            json,
                            "configured_max_active_slots_per_device",
                            configuredMaxActiveSlotsPerDevice)
                    .append(',');
            append(json, "active_legacy_leases", activeLegacyLeases).append(',');
            append(json, "active_reservations", activeReservations).append(',');
            append(json, "reserved_bytes", reservedBytes).append(',');
            append(json, "reservation_devices", reservationDevices).append(',');
            append(json, "reservation_tasks", reservationTasks).append(',');
            append(json, "native_pool_owners", nativePoolOwners).append(',');
            append(json, "native_pool_opens", nativePoolOpens).append(',');
            append(json, "native_pool_clean_closes", nativePoolCleanCloses).append(',');
            append(json, "native_pool_stats_observable", nativePoolStatsObservable).append(',');
            append(json, "native_pool_in_use_bytes", nativePoolInUseBytes).append(',');
            append(json, "native_pool_active_references", nativePoolActiveReferences).append(',');
            append(json, "native_pool_in_use_batches", nativePoolInUseBatches).append(',');
            append(
                            json,
                            "native_pool_in_use_batches_observable",
                            nativePoolInUseBatchesObservable)
                    .append(',');
            append(json, "native_pool_batch_count_source", nativePoolBatchCountSource).append(',');
            append(json, "native_kernel_active", nativeKernelActive).append(',');
            append(json, "native_kernel_opens", nativeKernelOpens).append(',');
            append(json, "native_kernel_clean_closes", nativeKernelCleanCloses).append(',');
            append(json, "native_cleanup_pending", nativeCleanupPending).append(',');
            append(json, "native_cleanup_observable", nativeCleanupObservable).append(',');
            append(json, "device_batch_active_references", deviceBatchActiveReferences).append(',');
            append(json, "device_batch_active_batches", deviceBatchActiveBatches).append(',');
            append(json, "device_batch_in_use_bytes", deviceBatchInUseBytes).append(',');
            append(json, "forced_leaked_references", forcedLeakedReferences).append(',');
            append(json, "accounting_errors", accountingErrors).append(',');
            json.append("\"unresolved\":[");
            for (int index = 0; index < unresolved.size(); index++) {
                if (index > 0) {
                    json.append(',');
                }
                quote(json, unresolved.get(index));
            }
            json.append("],");
            append(json, "fully_observable", fullyObservable).append(',');
            append(json, "fully_released", fullyReleased);
            return json.append('}').toString();
        }

        private static StringBuilder append(StringBuilder json, String key, long value) {
            quote(json, key).append(':').append(value);
            return json;
        }

        private static StringBuilder append(StringBuilder json, String key, boolean value) {
            quote(json, key).append(':').append(value);
            return json;
        }

        private static StringBuilder append(StringBuilder json, String key, String value) {
            quote(json, key).append(':');
            quote(json, value);
            return json;
        }

        private static StringBuilder quote(StringBuilder json, String value) {
            json.append('"');
            for (int index = 0; index < value.length(); index++) {
                char character = value.charAt(index);
                switch (character) {
                    case '"':
                        json.append("\\\"");
                        break;
                    case '\\':
                        json.append("\\\\");
                        break;
                    case '\b':
                        json.append("\\b");
                        break;
                    case '\f':
                        json.append("\\f");
                        break;
                    case '\n':
                        json.append("\\n");
                        break;
                    case '\r':
                        json.append("\\r");
                        break;
                    case '\t':
                        json.append("\\t");
                        break;
                    default:
                        if (character < 0x20) {
                            json.append(String.format("\\u%04x", (int) character));
                        } else {
                            json.append(character);
                        }
                }
            }
            return json.append('"');
        }
    }
}
