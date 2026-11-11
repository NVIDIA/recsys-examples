// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.examples;

import com.nvidia.fep.flink.GpuJoins;
import com.nvidia.fep.flink.GpuResourceConfig;
import com.nvidia.fep.flink.GpuResourceMode;
import com.nvidia.fep.flink.LookupDimension;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.LookupTableSource;
import org.apache.flink.table.connector.source.lookup.LookupFunctionProvider;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.LookupFunction;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.Row;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * INNER lookup of the video author for every event of the KuaiRand log:
 * {@code events JOIN video_dim ON video_id}, output (seq, video_id, author_id).
 *
 * <pre>
 * --engine cpu   Flink SQL lookup join (the dimension is a full in-memory hash map)
 * --engine gpu   com.nvidia.fep.flink.GpuJoins.lookupInner (dimension snapshot on the GPU)
 * --records N   measured rows (default 3000000)   --warmup N   rows before them (default 300000)
 * --rate R   offered rows/s, 0 = as fast as possible (default 0)
 * --data DIR (or env KUAIRAND_DIR)   --out DIR   --dump-final true|false
 * --batch-rows N   rows the GPU operator collects per batch (default 4096)
 * --gpu-min-batch-rows N   batches with fewer rows are answered by the JVM instead of the GPU
 *     (default 1024; 1 forces every batch onto the GPU; must not exceed --batch-rows)
 * </pre>
 *
 * The defaults 4096 / 1024 are the values of the benchmark runs; the library's own defaults are
 * 1,048,576 for the batch size and 1,024 for the GPU minimum. After a GPU run
 * the job prints {@code OPERATOR_METRICS} with {@code cudfLookupJoinGpuBatches} and
 * {@code cudfLookupJoinCpuBatches}.
 */
public final class LookupAuthorJob {
    private LookupAuthorJob() {}

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = Replay.parseArgs(args);
        String engine = opt.getOrDefault("engine", "cpu");
        int records = Integer.parseInt(opt.getOrDefault("records", "3000000"));
        int warmup = Integer.parseInt(opt.getOrDefault("warmup", "300000"));
        double rate = Double.parseDouble(opt.getOrDefault("rate", "0"));
        if (!engine.equals("cpu") && !engine.equals("gpu")) {
            throw new IllegalArgumentException("--engine must be cpu or gpu");
        }
        Replay.load(Replay.dataDir(opt), warmup, records, rate);
        Replay.loadDim(Replay.dataDir(opt));

        int batchRows = Integer.parseInt(opt.getOrDefault("batch-rows", "4096"));
        int gpuMinBatchRows = Integer.parseInt(opt.getOrDefault("gpu-min-batch-rows", "1024"));
        StreamExecutionEnvironment env =
                StreamExecutionEnvironment.getExecutionEnvironment(
                        MetricsCapture.install(new Configuration()));
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);
        env.setParallelism(1);
        DataStream<Long> seq = env.fromSequence(0, Replay.total - 1).setParallelism(1);
        if (engine.equals("cpu")) {
            buildSql(env, seq);
        } else {
            buildGpu(seq, batchRows, gpuMinBatchRows);
        }
        long t0 = System.nanoTime();
        env.execute("lookup-author-" + engine);
        long wall = System.nanoTime() - t0;
        Replay.finish("lookup", engine, "inner", opt, wall);
        MetricsCapture.dump("lookup", engine, "inner", opt);
        System.exit(0);
    }

    // ---- CPU: Flink SQL lookup join ---------------------------------------------------------

    private static void buildSql(StreamExecutionEnvironment env, DataStream<Long> seq) {
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        tEnv.getConfig().getConfiguration().setString("table.local-time-zone", "UTC");
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
                        .name("kuairand-replay");
        Schema.Builder sb = Schema.newBuilder();
        for (String n : names) {
            sb.column(n, DataTypes.BIGINT().notNull());
        }
        sb.columnByExpression("proc_time", "PROCTIME()");
        tEnv.createTemporaryView("events", tEnv.fromDataStream(rows, sb.build()));
        tEnv.executeSql(
                "CREATE TEMPORARY TABLE video_dim (video_id BIGINT, author_id BIGINT, "
                        + "PRIMARY KEY (video_id) NOT ENFORCED) WITH ('connector'='"
                        + DimFactory.IDENTIFIER
                        + "')");
        Table result =
                tEnv.sqlQuery(
                        "SELECT e.seq, e.video_id, d.author_id FROM events AS e "
                                + "JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d "
                                + "ON e.video_id = d.video_id");
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

    private static void buildGpu(DataStream<Long> seq, int batchRows, int gpuMinBatchRows) {
        List<LogicalType> t = new ArrayList<>();
        List<String> n = new ArrayList<>();
        t.add(new BigIntType(false));
        n.add("seq");
        for (String c : Replay.LOG_COLS) {
            t.add(new BigIntType(false));
            n.add(c);
        }
        RowType eventType = RowType.of(t.toArray(new LogicalType[0]), n.toArray(new String[0]));
        DataStream<RowData> events =
                seq.map(new Replay.RowDataBigintReplay())
                        .returns(InternalTypeInfo.of(eventType))
                        .name("kuairand-replay");

        // identity must change whenever the dimension content changes
        long h = 1469598103934665603L;
        for (int i = 0; i < Replay.dimKeys.length; i++) {
            h = (h ^ Replay.dimKeys[i]) * 1099511628211L;
            h = (h ^ Replay.dimValues[i]) * 1099511628211L;
        }
        LookupDimension dim =
                LookupDimension.ofLongPairs(
                        "kuairand-video-author-" + Replay.dimKeys.length + "-" + Long.toHexString(h),
                        Replay.dimKeys,
                        Replay.dimValues);
        GpuJoins.LookupInnerBuilder b =
                GpuJoins.lookupInner(events, eventType, "video_id", dim)
                        .dimensionColumns("dim_video_id", "author_id")
                        .batchRows(batchRows)
                        .gpuMinBatchRows(gpuMinBatchRows)
                        // run on the single local GPU of this example (device 0)
                        .gpuResource(new GpuResourceConfig(GpuResourceMode.LEGACY_DEVICE, "gpu", 0));
        RowType outType = b.outputType();
        final int[] perm = {
            outType.getFieldNames().indexOf("seq"),
            outType.getFieldNames().indexOf("video_id"),
            outType.getFieldNames().indexOf("author_id")
        };
        final RowData.FieldGetter[] getters = new RowData.FieldGetter[3];
        for (int i = 0; i < 3; i++) {
            getters[i] = RowData.createFieldGetter(outType.getTypeAt(perm[i]), perm[i]);
        }
        b.build()
                .sinkTo(
                        new Replay.CaptureSink<RowData>(
                                new Replay.Converter<RowData>() {
                                    private static final long serialVersionUID = 1L;

                                    @Override
                                    public long[] values(RowData r) {
                                        long[] f = new long[3];
                                        for (int i = 0; i < 3; i++) {
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

    // ---- the CPU dimension table (a plain Flink lookup source) -----------------------------

    /** video_id -> author_id as a bounded, read-only, in-memory table (full in-memory hash map). */
    public static final class DimFactory implements DynamicTableSourceFactory {
        public static final String IDENTIFIER = "fep-video-dim";

        @Override
        public String factoryIdentifier() {
            return IDENTIFIER;
        }

        @Override
        public Set<ConfigOption<?>> requiredOptions() {
            return Collections.emptySet();
        }

        @Override
        public Set<ConfigOption<?>> optionalOptions() {
            return Collections.emptySet();
        }

        @Override
        public DynamicTableSource createDynamicTableSource(Context context) {
            return new Source();
        }
    }

    static final class Source implements LookupTableSource {
        @Override
        public LookupRuntimeProvider getLookupRuntimeProvider(LookupContext context) {
            return LookupFunctionProvider.of(new DimLookup());
        }

        @Override
        public DynamicTableSource copy() {
            return new Source();
        }

        @Override
        public String asSummaryString() {
            return DimFactory.IDENTIFIER;
        }
    }

    /** Lookup function over the in-memory dimension. */
    public static final class DimLookup extends LookupFunction {
        private static final long serialVersionUID = 1L;
        private transient HashMap<Long, RowData> map;

        @Override
        public void open(FunctionContext context) {
            long[] k = Replay.dimKeys;
            long[] v = Replay.dimValues;
            map = new HashMap<>(k.length * 2);
            for (int i = 0; i < k.length; i++) {
                map.put(k[i], GenericRowData.of(k[i], v[i]));
            }
        }

        @Override
        public Collection<RowData> lookup(RowData keyRow) {
            if (keyRow.isNullAt(0)) {
                return Collections.emptyList();
            }
            RowData r = map.get(keyRow.getLong(0));
            return r == null ? Collections.emptyList() : Collections.singletonList(r);
        }
    }
}
