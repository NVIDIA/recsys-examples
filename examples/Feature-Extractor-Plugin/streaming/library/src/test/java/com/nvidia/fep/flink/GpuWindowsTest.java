// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.IntType;
import org.apache.flink.table.types.logical.LocalZonedTimestampType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.TimestampType;
import org.apache.flink.table.types.logical.VarCharType;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Build-time shape validation of {@link GpuWindows}; needs no GPU because nothing is executed. */
class GpuWindowsTest {
    private static final RowType EVENT =
            RowType.of(
                    new LogicalType[] {
                        new BigIntType(false),
                        new IntType(),
                        new BigIntType(),
                        new VarCharType(16),
                        new TimestampType(3)
                    },
                    new String[] {"user_id", "kind", "play_ms", "tag", "ts"});

    private static DataStream<RowData> stream(RowType type) {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        return env.fromData(
                Collections.<RowData>singletonList(new GenericRowData(type.getFieldCount())),
                InternalTypeInfo.of(type));
    }

    private static GpuWindows.Builder tumble() {
        return GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                .timeField("ts")
                .keyBy("user_id")
                .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_sum"));
    }

    private static GpuUnsupportedShapeException rejects(GpuWindows.Builder builder) {
        return assertThrows(GpuUnsupportedShapeException.class, builder::build);
    }

    @Test
    void tumbleOutputTypeFollowsFlinkSqlRules() {
        RowType out =
                GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                        .timeField("ts")
                        .keyBy("user_id", "kind")
                        .aggregate(
                                Agg.count(),
                                Agg.avg("kind"),
                                Agg.min("play_ms").as("min_play"),
                                Agg.max("tag").as("max_tag"))
                        .windowProperties(WindowProperty.START, WindowProperty.END, WindowProperty.ROWTIME)
                        .outputType();
        assertEquals(
                "ROW<`user_id` BIGINT NOT NULL, `kind` INT, `count` BIGINT NOT NULL, "
                        + "`avg_kind` INT, `min_play` BIGINT, `max_tag` VARCHAR(16), "
                        + "`window_start` TIMESTAMP(3) NOT NULL, `window_end` TIMESTAMP(3) NOT NULL, "
                        + "`window_time` TIMESTAMP(3) NOT NULL> NOT NULL",
                out.asSummaryString());
    }

    @Test
    void ltzTimeColumnYieldsLtzWindowColumns() {
        RowType type =
                RowType.of(
                        new LogicalType[] {new BigIntType(), new LocalZonedTimestampType(3)},
                        new String[] {"k", "t"});
        RowType out =
                GpuWindows.tumble(stream(type), type, Duration.ofSeconds(10))
                        .timeField("t")
                        .keyBy("k")
                        .aggregate(Agg.count())
                        .outputType();
        assertEquals(
                new LocalZonedTimestampType(false, 3), out.getTypeAt(out.getFieldCount() - 1));
    }

    @Test
    void validShapesBuildAndTwoPhaseAddsALocalOperator() {
        tumble().build();
        DataStream<RowData> hop =
                GpuWindows.hop(stream(EVENT), EVENT, Duration.ofMinutes(30), Duration.ofMinutes(5))
                        .timeField("ts")
                        .keyBy("user_id")
                        .aggregate(Agg.count(), Agg.avg("play_ms"), Agg.sum("kind"))
                        .build();
        assertEquals(
                InternalTypeInfo.of(
                        GpuWindows.hop(
                                        stream(EVENT),
                                        EVENT,
                                        Duration.ofMinutes(30),
                                        Duration.ofMinutes(5))
                                .timeField("ts")
                                .keyBy("user_id")
                                .aggregate(Agg.count(), Agg.avg("play_ms"), Agg.sum("kind"))
                                .outputType()),
                hop.getType());
        DataStream<RowData> two =
                GpuWindows.hop(stream(EVENT), EVENT, Duration.ofMinutes(30), Duration.ofMinutes(5))
                        .timeField("ts")
                        .keyBy("user_id")
                        .aggregate(Agg.count(), Agg.avg("play_ms"), Agg.min("play_ms"))
                        .twoPhase()
                        .name("hop2")
                        .build();
        boolean hasLocal =
                two.getExecutionEnvironment().getStreamGraph().getStreamNodes().stream()
                        .anyMatch(n -> "hop2-local".equals(n.getOperatorName()));
        assertTrue(hasLocal);
    }

    @Test
    void missingPiecesFailEarlyWithAClearMessage() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                                .keyBy("user_id")
                                .aggregate(Agg.count())
                                .build());
        assertTrue(
                rejects(
                                GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                                        .timeField("ts")
                                        .aggregate(Agg.count()))
                        .getMessage()
                        .contains("at least one key column"));
        assertTrue(
                rejects(
                                GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                                        .timeField("ts")
                                        .keyBy("user_id"))
                        .getMessage()
                        .contains("no aggregate calls"));
    }

    @Test
    void timeColumnMustBeATimestamp() {
        GpuUnsupportedShapeException e =
                rejects(
                        GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                                .timeField("play_ms")
                                .keyBy("user_id")
                                .aggregate(Agg.count()));
        assertTrue(e.getMessage().contains("only TIMESTAMP/TIMESTAMP_LTZ rowtime is supported"), e.getMessage());
    }

    @Test
    void windowParametersAreChecked() {
        assertTrue(
                rejects(
                                GpuWindows.tumble(stream(EVENT), EVENT, Duration.ZERO)
                                        .timeField("ts")
                                        .keyBy("user_id")
                                        .aggregate(Agg.count()))
                        .getMessage()
                        .contains("size must be positive"));
        assertTrue(
                rejects(
                                GpuWindows.hop(
                                                stream(EVENT),
                                                EVENT,
                                                Duration.ofMinutes(10),
                                                Duration.ofSeconds(1))
                                        .timeField("ts")
                                        .keyBy("user_id")
                                        .aggregate(Agg.count()))
                        .getMessage()
                        .contains("more than 64 windows per input row"));
        assertTrue(
                rejects(
                                GpuWindows.hop(
                                                stream(EVENT),
                                                EVENT,
                                                Duration.ofMinutes(7),
                                                Duration.ofMinutes(3))
                                        .timeField("ts")
                                        .keyBy("user_id")
                                        .aggregate(Agg.count())
                                        .twoPhase())
                        .getMessage()
                        .contains("integral multiple of slide"));
        // one phase accepts a size that is not a multiple of the slide
        GpuWindows.hop(stream(EVENT), EVENT, Duration.ofMinutes(7), Duration.ofMinutes(3))
                .timeField("ts")
                .keyBy("user_id")
                .aggregate(Agg.count())
                .build();
        assertTrue(
                rejects(tumble().offset(Duration.ofMinutes(5))).getMessage().contains("abs(offset) < size"));
        assertThrows(
                IllegalArgumentException.class, () -> tumble().offset(Duration.ofNanos(1)).build());
    }

    @Test
    void aggregateAndKeyTypesAreChecked() {
        GpuUnsupportedShapeException e =
                rejects(
                        GpuWindows.tumble(stream(EVENT), EVENT, Duration.ofMinutes(5))
                                .timeField("ts")
                                .keyBy("user_id")
                                .aggregate(Agg.sum("tag")));
        assertTrue(e.getMessage().contains("SUM"), e.getMessage());
        RowType dec =
                RowType.of(
                        new LogicalType[] {new DecimalType(10, 2), new BigIntType(), new TimestampType(3)},
                        new String[] {"d", "v", "ts"});
        // a DECIMAL key is accepted in one phase but not by the two-phase global merge
        GpuWindows.tumble(stream(dec), dec, Duration.ofMinutes(5))
                .timeField("ts")
                .keyBy("d")
                .aggregate(Agg.sum("v"))
                .build();
        String why =
                rejects(
                                GpuWindows.tumble(stream(dec), dec, Duration.ofMinutes(5))
                                        .timeField("ts")
                                        .keyBy("d")
                                        .aggregate(Agg.sum("v"))
                                        .twoPhase())
                        .getMessage();
        assertTrue(why.contains("unsupported global Window key type"), why);
    }

    @Test
    void twoPhaseRejectsSubMillisecondTimePrecision() {
        RowType type =
                RowType.of(
                        new LogicalType[] {new BigIntType(), new TimestampType(6)},
                        new String[] {"k", "t"});
        assertTrue(
                rejects(
                                GpuWindows.tumble(stream(type), type, Duration.ofSeconds(10))
                                        .timeField("t")
                                        .keyBy("k")
                                        .aggregate(Agg.count())
                                        .twoPhase())
                        .getMessage()
                        .contains("above millisecond precision"));
    }

    @Test
    void duplicateAndUnknownNamesAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> tumble().aggregate(Agg.count().as("x"), Agg.sum("play_ms").as("x")).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> tumble().aggregate(Agg.sum("nope")).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> tumble().keyBy("user_id", "user_id").build());
    }

    @Test
    void declaredRowTypeMustMatchTheStream() {
        RowType other =
                RowType.of(
                        new LogicalType[] {new BigIntType(), new TimestampType(3)},
                        new String[] {"user_id", "ts"});
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        GpuWindows.tumble(stream(EVENT), other, Duration.ofMinutes(5))
                                .timeField("ts")
                                .keyBy("user_id")
                                .aggregate(Agg.count())
                                .build());
    }

    @Test
    void jobGraphsWithTheOperatorsSerialize() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        DataStream<RowData> in =
                env.fromData(
                        Collections.<RowData>singletonList(new GenericRowData(EVENT.getFieldCount())),
                        InternalTypeInfo.of(EVENT));
        GpuWindows.tumble(in, EVENT, Duration.ofMinutes(5))
                .timeField("ts")
                .keyBy("user_id")
                .aggregate(Agg.count(), Agg.avg("kind"))
                .build();
        GpuWindows.hop(in, EVENT, Duration.ofMinutes(30), Duration.ofMinutes(5))
                .timeField("ts")
                .keyBy("user_id", "kind")
                .aggregate(Agg.count(), Agg.sum("play_ms"), Agg.avg("play_ms"))
                .twoPhase()
                .build();
        GpuJoins.lookupInner(
                        in,
                        EVENT,
                        "play_ms",
                        LookupDimension.ofLongPairs("d", new long[] {1L}, new long[] {2L}))
                .build();
        // Generating the job graph serializes every operator factory: one-phase window,
        // two-phase window (local + global) and lookup join make four GPU vertices.
        assertEquals(
                4,
                env.getStreamGraph()
                        .getJobGraph(Thread.currentThread().getContextClassLoader())
                        .getVerticesSortedTopologicallyFromSources()
                        .stream()
                        .filter(v -> v.getName().contains("Gpu"))
                        .count());
    }
}
