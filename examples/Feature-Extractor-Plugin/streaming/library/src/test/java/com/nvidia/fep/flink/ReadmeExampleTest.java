// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.TimestampType;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Keeps the example in README.md compiling and building. */
class ReadmeExampleTest {
    @Test
    void readmeExampleBuilds() {
        RowType eventType =
                RowType.of(
                        new LogicalType[] {
                            new BigIntType(false), new BigIntType(), new TimestampType(3)
                        },
                        new String[] {"user_id", "play_ms", "ts"});
        DataStream<RowData> events =
                StreamExecutionEnvironment.getExecutionEnvironment()
                        .fromData(
                                Collections.<RowData>singletonList(new GenericRowData(3)),
                                InternalTypeInfo.of(eventType));

        DataStream<RowData> stats =
                GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
                        .timeField("ts")
                        .keyBy("user_id")
                        .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_ms_sum"))
                        .build();

        assertEquals(
                java.util.Arrays.asList(
                        "user_id", "cnt", "play_ms_sum", "window_start", "window_end"),
                ((RowType) ((InternalTypeInfo<?>) stats.getType()).toLogicalType()).getFieldNames());
    }
}
