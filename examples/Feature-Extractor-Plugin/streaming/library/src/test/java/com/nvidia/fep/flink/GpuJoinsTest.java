// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.junit.jupiter.api.Test;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Build-time shape validation of {@link GpuJoins} and the in-memory {@link LookupDimension}. */
class GpuJoinsTest {
    private static final RowType PROBE =
            RowType.of(
                    new LogicalType[] {new BigIntType(false), new BigIntType(), new IntType()},
                    new String[] {"user_id", "video_id", "kind"});
    private static final LookupDimension DIM =
            LookupDimension.ofLongPairs("dim-v1", new long[] {1L, 2L}, new long[] {10L, 20L});

    private static DataStream<RowData> stream(RowType type) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        return env.fromData(
                Collections.<RowData>singletonList(new GenericRowData(type.getFieldCount())),
                InternalTypeInfo.of(type));
    }

    @Test
    void outputIsTheProbeRowPlusTwoBigintColumns() {
        GpuJoins.LookupInnerBuilder b =
                GpuJoins.lookupInner(stream(PROBE), PROBE, "video_id", DIM)
                        .dimensionColumns("dim_video_id", "author_id");
        assertEquals(
                "ROW<`user_id` BIGINT NOT NULL, `video_id` BIGINT, `kind` INT, "
                        + "`dim_video_id` BIGINT, `author_id` BIGINT> NOT NULL",
                b.outputType().asSummaryString());
        assertEquals(InternalTypeInfo.of(b.outputType()), b.build().getType());
    }

    @Test
    void probeKeyMustBeABigintColumn() {
        GpuUnsupportedShapeException e =
                assertThrows(
                        GpuUnsupportedShapeException.class,
                        () -> GpuJoins.lookupInner(stream(PROBE), PROBE, "kind", DIM).build());
        assertTrue(e.getMessage().contains("probe lookup key must be BIGINT"), e.getMessage());
        assertThrows(
                IllegalArgumentException.class,
                () -> GpuJoins.lookupInner(stream(PROBE), PROBE, "nope", DIM).build());
    }

    @Test
    void batchConfigurationIsChecked() {
        GpuUnsupportedShapeException e =
                assertThrows(
                        GpuUnsupportedShapeException.class,
                        () ->
                                GpuJoins.lookupInner(stream(PROBE), PROBE, "video_id", DIM)
                                        .batchRows(100)
                                        .gpuMinBatchRows(200)
                                        .build());
        assertTrue(e.getMessage().contains("invalid bounded Lookup Join configuration"));
        assertThrows(
                GpuUnsupportedShapeException.class,
                () ->
                        GpuJoins.lookupInner(stream(PROBE), PROBE, "video_id", DIM)
                                .maxDimensionRows(0)
                                .build());
    }

    @Test
    void outputNamesMustNotClash() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        GpuJoins.lookupInner(stream(PROBE), PROBE, "video_id", DIM)
                                .dimensionColumns("kind", "x")
                                .build());
    }

    @Test
    void dimensionIdentityMustBeNonBlank() {
        assertThrows(
                IllegalArgumentException.class,
                () -> LookupDimension.ofLongPairs("  ", new long[0], new long[0]));
        assertThrows(
                IllegalArgumentException.class,
                () -> LookupDimension.ofLongPairs("x", new long[1], new long[2]));
    }

    @Test
    void inMemoryDimensionLoadsAsASnapshotAndEnforcesTheRowBound() throws Exception {
        LookupJoinSnapshot snapshot =
                LookupJoinSnapshotLoader.loadForTesting(DIM.format(), DIM.identity(), 2);
        assertEquals(2, snapshot.rowCount());
        assertEquals(10L, snapshot.lookup(1L, true).payload());
        assertEquals(20L, snapshot.lookup(2L, true).payload());
        assertNull(snapshot.lookup(3L, true));
        assertNull(snapshot.lookup(1L, false));
        assertFalse(snapshot.checkpointMarker().isEmpty());
        assertThrows(
                IllegalStateException.class,
                () -> LookupJoinSnapshotLoader.loadForTesting(DIM.format(), DIM.identity(), 1));
    }
}
