// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SlicingWindowExpansionTest {
    @Test
    void expandsTumbleAndHopSlicesWithOffset() {
        long[] ends = new long[64];
        int tumble =
                SlicingWindowExpansion.assignWindowEnds(
                        1_100L, 1_000L, 1_000L, 100L, false, ends);
        assertArrayEquals(new long[] {1_100L}, Arrays.copyOf(ends, tumble));
        assertEquals(
                100L,
                SlicingWindowExpansion.windowStart(
                        1_100L, 1_000L, 100L, false));

        int hop =
                SlicingWindowExpansion.assignWindowEnds(
                        1_100L, 2_000L, 1_000L, 100L, false, ends);
        assertArrayEquals(new long[] {1_100L, 2_100L}, Arrays.copyOf(ends, hop));
        assertEquals(
                -900L,
                SlicingWindowExpansion.windowStart(
                        1_100L, 2_000L, 100L, false));
    }

    @Test
    void expandsEveryCumulativeSliceWithoutDoubleCounting() {
        long[] ends = new long[64];
        int first =
                SlicingWindowExpansion.assignWindowEnds(
                        1_100L, 3_000L, 1_000L, 100L, true, ends);
        assertArrayEquals(
                new long[] {1_100L, 2_100L, 3_100L},
                Arrays.copyOf(ends, first));

        int second =
                SlicingWindowExpansion.assignWindowEnds(
                        2_100L, 3_000L, 1_000L, 100L, true, ends);
        assertArrayEquals(new long[] {2_100L, 3_100L}, Arrays.copyOf(ends, second));

        int last =
                SlicingWindowExpansion.assignWindowEnds(
                        3_100L, 3_000L, 1_000L, 100L, true, ends);
        assertArrayEquals(new long[] {3_100L}, Arrays.copyOf(ends, last));
        assertEquals(
                100L,
                SlicingWindowExpansion.windowStart(
                        3_100L, 3_000L, 100L, true));
    }

    @Test
    void handlesNegativeEpochsAndRejectsMisalignedSlices() {
        long[] ends = new long[64];
        int count =
                SlicingWindowExpansion.assignWindowEnds(
                        -900L, 3_000L, 1_000L, 100L, true, ends);
        assertArrayEquals(new long[] {-900L, 100L}, Arrays.copyOf(ends, count));
        assertEquals(
                -2_900L,
                SlicingWindowExpansion.windowStart(
                        -900L, 3_000L, 100L, true));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        SlicingWindowExpansion.assignWindowEnds(
                                1_050L,
                                2_000L,
                                1_000L,
                                100L,
                                false,
                                ends));
    }
}
