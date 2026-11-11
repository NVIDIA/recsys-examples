// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** Exact host-side slice-to-window mapping used by the specialized global Window merge. */
final class SlicingWindowExpansion {
    private SlicingWindowExpansion() {}

    static int assignWindowEnds(
            long sliceEnd,
            long sizeMs,
            long slideMs,
            long offsetMs,
            boolean cumulative,
            long[] target) {
        if (sizeMs <= 0L
                || slideMs <= 0L
                || sizeMs % slideMs != 0L
                || sizeMs / slideMs > target.length) {
            throw new IllegalArgumentException("invalid slicing-window expansion");
        }
        if (Math.floorMod(Math.subtractExact(sliceEnd, offsetMs), slideMs) != 0L) {
            throw new IllegalArgumentException(
                    "slice end is not aligned to the configured slide/step");
        }
        long lastWindowEnd;
        if (cumulative) {
            long cycleStart = alignedStart(Math.subtractExact(sliceEnd, 1L), sizeMs, offsetMs);
            lastWindowEnd = Math.addExact(cycleStart, sizeMs);
        } else {
            lastWindowEnd =
                    Math.addExact(
                            Math.subtractExact(sliceEnd, slideMs),
                            sizeMs);
        }
        int count = 0;
        long windowEnd = sliceEnd;
        while (true) {
            if (count >= target.length || windowEnd > lastWindowEnd) {
                throw new IllegalArgumentException("slice expansion exceeded its validated bound");
            }
            target[count++] = windowEnd;
            if (windowEnd == lastWindowEnd) {
                return count;
            }
            windowEnd = Math.addExact(windowEnd, slideMs);
        }
    }

    static long windowStart(
            long windowEnd,
            long sizeMs,
            long offsetMs,
            boolean cumulative) {
        return cumulative
                ? alignedStart(Math.subtractExact(windowEnd, 1L), sizeMs, offsetMs)
                : Math.subtractExact(windowEnd, sizeMs);
    }

    private static long alignedStart(long timestamp, long sizeMs, long offsetMs) {
        long shifted = Math.subtractExact(timestamp, offsetMs);
        return Math.addExact(
                Math.multiplyExact(Math.floorDiv(shifted, sizeMs), sizeMs),
                offsetMs);
    }
}
