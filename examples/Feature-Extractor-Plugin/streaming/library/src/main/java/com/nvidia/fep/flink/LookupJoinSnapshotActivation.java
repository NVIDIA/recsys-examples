// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.Objects;

/** Transactional active-snapshot pointer used around the native inactive-table rebuild. */
public final class LookupJoinSnapshotActivation {
    private LookupJoinSnapshot active;

    /**
     * Runs the complete rebuild before publishing the candidate. If validation, capacity, JNI or
     * device work fails, {@link #active()} continues to return the previous snapshot.
     */
    public void activate(LookupJoinSnapshot candidate, Rebuilder rebuilder) {
        Objects.requireNonNull(candidate, "candidate");
        Objects.requireNonNull(rebuilder, "rebuilder").rebuild(candidate);
        active = candidate;
    }

    public LookupJoinSnapshot active() {
        if (active == null) {
            throw new IllegalStateException("Lookup Join has no active dimension snapshot");
        }
        return active;
    }

    @FunctionalInterface
    public interface Rebuilder {
        void rebuild(LookupJoinSnapshot candidate);
    }
}
