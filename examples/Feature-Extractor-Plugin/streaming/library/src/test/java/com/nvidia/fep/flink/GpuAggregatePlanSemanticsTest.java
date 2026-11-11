// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class GpuAggregatePlanSemanticsTest {
    @Test
    void floatingExtremaUseFlinkOrderedComparisonForNanAndSignedZero() {
        GpuAggregatePlan.AggregateSpec minimum =
                new GpuAggregatePlan.AggregateSpec(
                        GpuAggregatePlan.AggregateKind.MIN,
                        0,
                        -1,
                        GpuType.FLOAT64,
                        GpuType.FLOAT64,
                        false);
        GpuAggregatePlan.AggregateSpec maximum =
                new GpuAggregatePlan.AggregateSpec(
                        GpuAggregatePlan.AggregateKind.MAX,
                        0,
                        -1,
                        GpuType.FLOAT64,
                        GpuType.FLOAT64,
                        false);

        assertTrue(Double.isNaN((Double) minimum.merge(Double.NaN, 1.0d)));
        assertEquals(1.0d, minimum.merge(1.0d, Double.NaN));
        assertTrue(Double.isNaN((Double) maximum.merge(Double.NaN, 1.0d)));
        assertEquals(1.0d, maximum.merge(1.0d, Double.NaN));

        assertEquals(
                Double.doubleToRawLongBits(0.0d),
                Double.doubleToRawLongBits((Double) minimum.merge(0.0d, -0.0d)));
        assertEquals(
                Double.doubleToRawLongBits(-0.0d),
                Double.doubleToRawLongBits((Double) maximum.merge(-0.0d, 0.0d)));
    }
}
