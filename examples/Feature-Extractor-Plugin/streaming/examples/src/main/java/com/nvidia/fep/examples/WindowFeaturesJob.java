// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.examples;

import com.nvidia.fep.flink.Agg;
import com.nvidia.fep.flink.GpuResourceConfig;
import com.nvidia.fep.flink.GpuResourceMode;
import com.nvidia.fep.flink.GpuWindows;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.LocalZonedTimestampType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.Row;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Per-user window features over the KuaiRand log: exposure count, click count and total play time
 * per (user_id, window), for a HOP (30 min size, 5 min slide) or a 5 min TUMBLE window.
 *
 * <pre>
 * --engine cpu   Flink SQL window table-valued function (CPU operators)
 * --engine gpu   com.nvidia.fep.flink.GpuWindows (GPU operators)
 * --window hop|tumble   (default hop)
 * --two-phase true|false   local+global aggregation on both engines (default false)
 * --records N   measured rows (default 3000000)   --warmup N   rows before them (default 300000)
 * --rate R   offered rows/s, 0 = as fast as possible (default 0)
 * --data DIR (or env KUAIRAND_DIR)   --out DIR   --dump-final true|false
 * --batch-rows N   rows the GPU operator collects before a kernel launch (default 4096, the
 *     library default and the value of the benchmark runs; GPU engine only)
 * </pre>
 *
 * Output columns, in this order for both engines: user_id, window_start, window_end, expo_cnt,
 * click_cnt, play_ms. The result line carries the SHA-256 of the sorted final result table.
 */
public final class WindowFeaturesJob {
    private static final String[] OUT_NAMES = {
        "user_id", "window_start", "window_end", "expo_cnt", "click_cnt", "play_ms"
    };

    private WindowFeaturesJob() {}

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = Replay.parseArgs(args);
        String engine = opt.getOrDefault("engine", "cpu");
        String window = opt.getOrDefault("window", "hop");
        boolean twoPhase = Boolean.parseBoolean(opt.getOrDefault("two-phase", "false"));
        int records = Integer.parseInt(opt.getOrDefault("records", "3000000"));
        int warmup = Integer.parseInt(opt.getOrDefault("warmup", "300000"));
        double rate = Double.parseDouble(opt.getOrDefault("rate", "0"));
        if (!window.equals("hop") && !window.equals("tumble")) {
            throw new IllegalArgumentException("--window must be hop or tumble");
        }
        if (!engine.equals("cpu") && !engine.equals("gpu")) {
            throw new IllegalArgumentException("--engine must be cpu or gpu");
        }
        Replay.load(Replay.dataDir(opt), warmup, records, rate);
        String variant = window + (twoPhase ? "-2p" : "-1p");

        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment(
                        MetricsCapture.install(new Configuration()));
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);
        env.setParallelism(1);
        env.getConfig().setAutoWatermarkInterval(20);
        DataStream<Long> seq = env.fromSequence(0, Replay.total - 1).setParallelism(1);

        if (engine.equals("cpu")) {
            buildSql(env, seq, window, twoPhase);
        } else {
            buildGpu(seq, window, twoPhase, Integer.parseInt(opt.getOrDefault("batch-rows", "4096")));
        }
        long t0 = System.nanoTime();
        env.execute("window-features-" + engine + "-" + variant);
        long wall = System.nanoTime() - t0;
        Replay.finish("window", engine, variant, opt, wall);
        MetricsCapture.dump("window", engine, variant, opt);
        System.exit(0);
    }

    // ---- CPU: Flink SQL ------------------------------------------------------------------

    private static void buildSql(
            StreamExecutionEnvironment env, DataStream<Long> seq, String window, boolean twoPhase) {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        Configuration tc = tEnv.getConfig().getConfiguration();
        tc.setString("table.local-time-zone", "UTC");
        if (twoPhase) {
            // local-global (two-phase) window aggregation needs mini-batch in Flink SQL
            tc.setString("table.exec.mini-batch.enabled", "true");
            tc.setString("table.exec.mini-batch.allow-latency", "5 ms");
            tc.setString("table.exec.mini-batch.size", "4096");
            tc.setString("table.optimizer.agg-phase-strategy", "TWO_PHASE");
        } else {
            tc.setString("table.optimizer.agg-phase-strategy", "ONE_PHASE");
        }

        String[] names = new String[Replay.LOG_COLS.length + 1];
        TypeInformation<?>[] types = new TypeInformation<?>[names.length];
        names[0] = "seq";
        types[0] = Types.LONG;
        for (int i = 0; i < Replay.LOG_COLS.length; i++) {
            names[i + 1] = Replay.LOG_COLS[i];
            types[i + 1] = Types.LONG;
        }
        DataStream<Row> rows =
                seq.map(new Replay.RowReplay())
                        .returns(new RowTypeInfo(types, names))
                        .name("kuairand-replay")
                        .assignTimestampsAndWatermarks(
                                WatermarkStrategy.<Row>forMonotonousTimestamps()
                                        .withTimestampAssigner((r, ts) -> (Long) r.getField(3)))
                        .name("kuairand-watermarks");
        Schema.Builder sb = Schema.newBuilder();
        for (String n : names) {
            sb.column(n, DataTypes.BIGINT().notNull());
        }
        sb.columnByMetadata("ts", DataTypes.TIMESTAMP_LTZ(3), "rowtime");
        sb.watermark("ts", "SOURCE_WATERMARK()");
        tEnv.createTemporaryView("events", tEnv.fromDataStream(rows, sb.build()));

        String tvf =
                window.equals("hop")
                        ? "HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE)"
                        : "TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE)";
        Table result =
                tEnv.sqlQuery(
                        "SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt, "
                                + "SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms "
                                + "FROM TABLE("
                                + tvf
                                + ") GROUP BY user_id, window_start, window_end");
        tEnv.toChangelogStream(result)
                .sinkTo(
                        new Replay.CaptureSink<Row>(
                                new Replay.Converter<Row>() {
                                    private static final long serialVersionUID = 1L;

                                    @Override
                                    public long[] values(Row r) {
                                        long[] f = new long[r.getArity()];
                                        for (int i = 0; i < f.length; i++) {
                                            f[i] = Replay.toLong(r.getField(i));
                                        }
                                        return f;
                                    }

                                    @Override
                                    public byte kind(Row r) {
                                        return r.getKind().toByteValue();
                                    }
                                }))
                .name("capture-sink")
                .setParallelism(1);
    }

    // ---- GPU: this library -----------------------------------------------------------------

    private static void buildGpu(
            DataStream<Long> seq, String window, boolean twoPhase, int batchRows) {
        // seq, the 14 log columns, then ts (TIMESTAMP_LTZ(3) = time_ms)
        List<LogicalType> t = new ArrayList<>();
        List<String> n = new ArrayList<>();
        t.add(new BigIntType(false));
        n.add("seq");
        for (String c : Replay.LOG_COLS) {
            t.add(new BigIntType(false));
            n.add(c);
        }
        t.add(new LocalZonedTimestampType(false, 3));
        n.add("ts");
        RowType eventType = RowType.of(t.toArray(new LogicalType[0]), n.toArray(new String[0]));
        int tsIndex = n.indexOf("ts");

        DataStream<RowData> events =
                seq.map(new Replay.RowDataReplay())
                        .returns(InternalTypeInfo.of(eventType))
                        .name("kuairand-replay")
                        .assignTimestampsAndWatermarks(
                                WatermarkStrategy.<RowData>forMonotonousTimestamps()
                                        .withTimestampAssigner(
                                                (r, ts) -> r.getTimestamp(tsIndex, 3).getMillisecond()))
                        .name("kuairand-watermarks");

        GpuWindows.Builder b =
                window.equals("hop")
                        ? GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
                        : GpuWindows.tumble(events, eventType, Duration.ofMinutes(5));
        b.timeField("ts")
                .batchRows(batchRows)
                .keyBy("user_id")
                .aggregate(
                        Agg.count().as("expo_cnt"),
                        Agg.sum("is_click").as("click_cnt"),
                        Agg.sum("play_time_ms").as("play_ms"))
                // run on the single local GPU of this example (device 0); on a cluster, drop this
                // line and let Flink's external-resource framework assign the device
                .gpuResource(new GpuResourceConfig(GpuResourceMode.LEGACY_DEVICE, "gpu", 0));
        if (twoPhase) {
            b.twoPhase();
        }
        RowType outType = b.outputType();
        // library column order: keys, aggregates, window_start, window_end; map to the SQL order
        final int[] perm = new int[OUT_NAMES.length];
        for (int i = 0; i < perm.length; i++) {
            perm[i] = outType.getFieldNames().indexOf(OUT_NAMES[i]);
            if (perm[i] < 0) {
                throw new IllegalStateException(
                        "missing output column " + OUT_NAMES[i] + " in " + outType.getFieldNames());
            }
        }
        final RowData.FieldGetter[] getters = new RowData.FieldGetter[perm.length];
        for (int i = 0; i < perm.length; i++) {
            getters[i] = RowData.createFieldGetter(outType.getTypeAt(perm[i]), perm[i]);
        }
        b.build()
                .sinkTo(
                        new Replay.CaptureSink<RowData>(
                                new Replay.Converter<RowData>() {
                                    private static final long serialVersionUID = 1L;

                                    @Override
                                    public long[] values(RowData r) {
                                        long[] f = new long[getters.length];
                                        for (int i = 0; i < f.length; i++) {
                                            f[i] = Replay.toLong(getters[i].getFieldOrNull(r));
                                        }
                                        return f;
                                    }

                                    @Override
                                    public byte kind(RowData r) {
                                        return r.getRowKind().toByteValue();
                                    }
                                }))
                .name("capture-sink")
                .setParallelism(1);
    }
}
