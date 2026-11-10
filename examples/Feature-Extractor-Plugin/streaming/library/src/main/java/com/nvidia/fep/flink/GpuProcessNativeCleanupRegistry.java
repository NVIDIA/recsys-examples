// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Process-local handoff for native cleanup that outlives a failed Java construction.
 *
 * <p>The registry is deliberately fixed-capacity: every {@link Family} owns one slot and at most
 * one retry owner. Callers that wire a family into production must hold its {@link
 * ConstructionPermit} from before native creation until Java publication or failed-construction
 * cleanup has been transferred. A later construction first retries the existing owner and is
 * rejected while that owner remains unresolved.
 *
 * <p>Cleanup has two explicit stages. Once native cleanup succeeds, an accounting failure retains
 * only the accounting stage; a retry can never invoke native cleanup again. No registry lock is
 * held while either callback runs. Device binding remains part of the family callback, because
 * only that callback knows the authoritative CUDA device for its native owner.
 */
final class GpuProcessNativeCleanupRegistry {
    enum Family {
        ORDINARY_NATIVE_KERNEL,
        NATIVE_POOL_OWNER,
        FUSED_FILTER_PROJECT,
        DENSE_GROUPBY,
        CALC_EXPAND_LOCAL_DEVICE_CHAIN,
        JOIN_CALC_LOCAL_DEVICE_CHAIN,
        DEVICE_BATCH_POOL_CONSTRUCTION
    }

    private static final Map<Family, Slot> SLOTS = new EnumMap<>(Family.class);
    private static final ReentrantLock SWEEP_STATE_LOCK = new ReentrantLock();
    private static final Condition SWEEP_STATE_CHANGED = SWEEP_STATE_LOCK.newCondition();

    private static boolean sweepActive;
    private static Thread sweepThread;

    static {
        for (Family family : Family.values()) {
            SLOTS.put(family, new Slot(family));
        }
    }

    private GpuProcessNativeCleanupRegistry() {}

    static ConstructionPermit acquireConstructionPermit(Family family) {
        return slot(family).acquireConstructionPermit();
    }

    /**
     * Compatibility path for lifecycle callers that have not yet moved their whole construction
     * window under a permit. New family wiring must acquire the permit before native creation.
     */
    static void retainAfterFailedConstruction(
            Family family,
            Object owner,
            CleanupStage nativeCleanup,
            CleanupStage accountingCleanup) {
        try (ConstructionPermit permit = acquireConstructionPermit(family)) {
            permit.transfer(owner, nativeCleanup, accountingCleanup);
        }
    }

    /** Runs at most one attempt for every fixed family and serializes concurrent sweeps. */
    static void retryPendingOnce() {
        beginSweep();
        Throwable failure = null;
        try {
            for (Slot slot : SLOTS.values()) {
                try {
                    slot.retryPending();
                } catch (Throwable retryFailure) {
                    failure = accumulate(failure, retryFailure);
                }
            }
        } finally {
            endSweep();
        }
        throwFailure(failure);
    }

    static Snapshot snapshot() {
        long pending = 0L;
        boolean observable = true;
        for (Slot slot : SLOTS.values()) {
            SlotSnapshot slotSnapshot = slot.snapshot();
            if (slotSnapshot.pending) {
                pending++;
            }
            observable &= slotSnapshot.observable;
        }
        return new Snapshot(pending, observable);
    }

    static void resetForTests() {
        if (!SWEEP_STATE_LOCK.tryLock()) {
            throw new IllegalStateException(
                    "cannot reset native cleanup registry during a sweep transition");
        }
        try {
            if (sweepActive) {
                throw new IllegalStateException(
                        "cannot reset native cleanup registry during a sweep");
            }
        } finally {
            SWEEP_STATE_LOCK.unlock();
        }
        for (Slot slot : SLOTS.values()) {
            slot.resetForTests();
        }
    }

    private static void beginSweep() {
        Thread current = Thread.currentThread();
        SWEEP_STATE_LOCK.lock();
        try {
            while (sweepActive) {
                if (sweepThread == current) {
                    throw new IllegalStateException(
                            "native cleanup sweep cannot recursively retry itself");
                }
                SWEEP_STATE_CHANGED.awaitUninterruptibly();
            }
            sweepActive = true;
            sweepThread = current;
        } finally {
            SWEEP_STATE_LOCK.unlock();
        }
    }

    private static void endSweep() {
        SWEEP_STATE_LOCK.lock();
        try {
            if (!sweepActive || sweepThread != Thread.currentThread()) {
                throw new IllegalStateException("native cleanup sweep lost its owner thread");
            }
            sweepActive = false;
            sweepThread = null;
            SWEEP_STATE_CHANGED.signalAll();
        } finally {
            SWEEP_STATE_LOCK.unlock();
        }
    }

    private static Slot slot(Family family) {
        Slot slot = SLOTS.get(Objects.requireNonNull(family, "family"));
        if (slot == null) {
            throw new IllegalArgumentException("unknown native cleanup family " + family);
        }
        return slot;
    }

    private static Throwable accumulate(Throwable first, Throwable later) {
        if (first == null) {
            return later;
        }
        if (first != later) {
            first.addSuppressed(later);
        }
        return first;
    }

    private static void throwFailure(Throwable failure) {
        if (failure == null) {
            return;
        }
        if (failure instanceof RuntimeException) {
            throw (RuntimeException) failure;
        }
        if (failure instanceof Error) {
            throw (Error) failure;
        }
        throw new IllegalStateException("native cleanup retry failed", failure);
    }

    @FunctionalInterface
    interface CleanupStage {
        void run();
    }

    static final class ConstructionPermit implements AutoCloseable {
        private final Slot slot;
        private final Thread ownerThread;
        private boolean transferred;
        private boolean closed;

        private ConstructionPermit(Slot slot, Thread ownerThread) {
            this.slot = slot;
            this.ownerThread = ownerThread;
        }

        void transfer(
                Object owner,
                CleanupStage nativeCleanup,
                CleanupStage accountingCleanup) {
            requireOwnerThread();
            if (closed || transferred) {
                throw new IllegalStateException("native construction permit is already terminal");
            }
            slot.transfer(
                    ownerThread,
                    new PendingCleanup(owner, nativeCleanup, accountingCleanup));
            transferred = true;
        }

        @Override
        public void close() {
            requireOwnerThread();
            if (closed) {
                return;
            }
            try {
                slot.releaseConstruction(ownerThread);
            } finally {
                closed = true;
            }
        }

        private void requireOwnerThread() {
            if (Thread.currentThread() != ownerThread) {
                throw new IllegalStateException(
                        "native construction permit must stay on its acquiring thread");
            }
        }
    }

    static final class Snapshot {
        final long pending;
        final boolean observable;

        private Snapshot(long pending, boolean observable) {
            this.pending = pending;
            this.observable = observable;
        }
    }

    private static final class Slot {
        private final Family family;
        private final ReentrantLock stateLock = new ReentrantLock(true);
        private final Condition stateChanged = stateLock.newCondition();
        private boolean constructionActive;
        private Thread constructionThread;
        private boolean retrying;
        private Thread retryThread;
        private boolean observable = true;
        private PendingCleanup pending;

        private Slot(Family family) {
            this.family = family;
        }

        private ConstructionPermit acquireConstructionPermit() {
            Thread current = Thread.currentThread();
            while (true) {
                PendingCleanup candidate;
                stateLock.lock();
                try {
                    awaitIdle(current);
                    candidate = pending;
                    if (candidate == null) {
                        constructionActive = true;
                        constructionThread = current;
                        return new ConstructionPermit(this, current);
                    }
                    retrying = true;
                    retryThread = current;
                } finally {
                    stateLock.unlock();
                }
                completeRetry(candidate);
            }
        }

        private void retryPending() {
            PendingCleanup candidate;
            Thread current = Thread.currentThread();
            stateLock.lock();
            try {
                awaitIdle(current);
                candidate = pending;
                if (candidate == null) {
                    return;
                }
                retrying = true;
                retryThread = current;
            } finally {
                stateLock.unlock();
            }
            completeRetry(candidate);
        }

        private void completeRetry(PendingCleanup candidate) {
            Throwable failure = null;
            try {
                candidate.retry();
            } catch (Throwable retryFailure) {
                failure = retryFailure;
            }

            stateLock.lock();
            try {
                if (!retrying
                        || retryThread != Thread.currentThread()
                        || pending != candidate) {
                    observable = false;
                    failure =
                            accumulate(
                                    failure,
                                    new IllegalStateException(
                                            "native cleanup retry owner changed for " + family));
                } else if (failure == null) {
                    pending = null;
                }
                retrying = false;
                retryThread = null;
                stateChanged.signalAll();
            } finally {
                stateLock.unlock();
            }
            throwFailure(failure);
        }

        private void awaitIdle(Thread current) {
            while (constructionActive || retrying) {
                if (constructionThread == current || retryThread == current) {
                    observable = false;
                    throw new IllegalStateException(
                            "native cleanup family cannot recursively acquire " + family);
                }
                stateChanged.awaitUninterruptibly();
            }
        }

        private void transfer(Thread ownerThread, PendingCleanup candidate) {
            stateLock.lock();
            try {
                if (!constructionActive
                        || constructionThread != ownerThread
                        || pending != null) {
                    observable = false;
                    throw new IllegalStateException(
                            "native cleanup slot already owns a pending " + family);
                }
                pending = candidate;
            } finally {
                stateLock.unlock();
            }
        }

        private void releaseConstruction(Thread ownerThread) {
            stateLock.lock();
            try {
                if (!constructionActive || constructionThread != ownerThread) {
                    observable = false;
                    throw new IllegalStateException(
                            "native cleanup construction permit lost its active slot");
                }
                constructionActive = false;
                constructionThread = null;
                stateChanged.signalAll();
            } finally {
                stateLock.unlock();
            }
        }

        private SlotSnapshot snapshot() {
            stateLock.lock();
            try {
                return new SlotSnapshot(
                        pending != null || constructionActive || retrying, observable);
            } finally {
                stateLock.unlock();
            }
        }

        private void resetForTests() {
            if (!stateLock.tryLock()) {
                throw new IllegalStateException(
                        "cannot reset native cleanup registry during a state transition for "
                                + family);
            }
            try {
                if (constructionActive || retrying || pending != null) {
                    throw new IllegalStateException(
                            "cannot reset native cleanup registry with a pending owner for "
                                    + family);
                }
                observable = true;
            } finally {
                stateLock.unlock();
            }
        }
    }

    private static final class SlotSnapshot {
        private final boolean pending;
        private final boolean observable;

        private SlotSnapshot(boolean pending, boolean observable) {
            this.pending = pending;
            this.observable = observable;
        }
    }

    private static final class PendingCleanup {
        @SuppressWarnings("unused")
        private final Object owner;
        private final CleanupStage nativeCleanup;
        private final CleanupStage accountingCleanup;
        private boolean nativeCompleted;
        private boolean completed;

        private PendingCleanup(
                Object owner,
                CleanupStage nativeCleanup,
                CleanupStage accountingCleanup) {
            this.owner = Objects.requireNonNull(owner, "owner");
            this.nativeCleanup = Objects.requireNonNull(nativeCleanup, "nativeCleanup");
            this.accountingCleanup =
                    Objects.requireNonNull(accountingCleanup, "accountingCleanup");
        }

        private void retry() {
            if (completed) {
                return;
            }
            if (!nativeCompleted) {
                nativeCleanup.run();
                nativeCompleted = true;
            }
            accountingCleanup.run();
            completed = true;
        }
    }
}
