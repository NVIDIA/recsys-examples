// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package fep;

import org.apache.flink.api.common.RuntimeExecutionMode;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.connector.sink2.Sink;
import org.apache.flink.api.connector.sink2.SinkWriter;
import org.apache.flink.api.connector.sink2.WriterInitContext;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.types.RowKind;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.LockSupport;

/**
 * KuaiRand-27K log replayed as an event-time ordered stream into Flink SQL.
 *
 * <p>With plain open-source Flink on the classpath this is the CPU baseline. When the JVM property
 * {@code fep.megaflink=true} is set (by scripts/run-one.sh when MEGA_FLINK_CP is given) the driver also
 * applies the benchmark GPU resource settings of a Mega-Flink build and uses a GPU-aware dimension table
 * factory, both looked up by reflection so that this source compiles and runs without any plugin.
 *
 * <p>Timing contract: an in-JVM replay source stamps every input with its scheduled send time
 * (fixed-rate tiers) or its actual emit time (saturated tier). Latency of an output row is sink
 * arrival time minus the send time of the input that made the row visible. Throughput is measured
 * inputs divided by (last sink arrival - first measured input send time). Source, Flink runtime,
 * state, row/column conversion, H2D/D2H and the sink are all inside the measured path for every
 * mode. Result rows are captured in memory and materialized (changelog applied) after the job,
 * outside the timed region.
 */
public final class FepKuaiRandBench {
    static final String[] LOG_COLS = {
        "user_id", "video_id", "time_ms", "is_click", "is_like", "is_follow", "is_comment",
        "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "is_rand", "tab"
    };

    // ---- shared JVM state (MiniCluster runs in this JVM) ----
    static long[][] COLS; // [col][row] for the replay slice
    static long[] SEND_NANOS; // per seq
    static volatile long SOURCE_DONE_NANOS;
    static int TOTAL;
    static int WARMUP;
    static double RATE;
    public static long[] DIM_KEYS = new long[0];
    public static long[] DIM_VALUES = new long[0];
    /** True when a Mega-Flink build is on the classpath and requested (-Dfep.megaflink=true). */
    static final boolean MEGA = Boolean.getBoolean("fep.megaflink");

    // sink capture
    static final Object SINK_LOCK = new Object();
    static List<long[]> OUT_ROWS = new ArrayList<>();
    static byte[] OUT_KINDS = new byte[1 << 20];
    static long[] OUT_NANOS = new long[1 << 20];
    static int OUT_COUNT;

    private FepKuaiRandBench() {}

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = new LinkedHashMap<>();
        Map<String, String> conf = new LinkedHashMap<>();
        List<String> expectPlan = new ArrayList<>();
        for (int i = 0; i < args.length; i += 2) {
            String k = args[i];
            String v = args[i + 1];
            if (k.equals("--conf")) {
                int eq = v.indexOf('=');
                conf.put(v.substring(0, eq), v.substring(eq + 1));
            } else if (k.equals("--expect-plan")) {
                expectPlan.add(v);
            } else {
                opt.put(k.substring(2), v);
            }
        }
        String caseId = opt.get("case");
        String label = opt.get("label");
        String dataDir = opt.getOrDefault("data", System.getenv("FEP_DATA_DIR"));
        if (dataDir == null || dataDir.isEmpty()) {
            throw new IllegalArgumentException(
                    "pass --data DIR (the output of prepare_stream_data.py) or set FEP_DATA_DIR");
        }
        Path data = Paths.get(dataDir);
        Path out = Paths.get(opt.get("out"));
        Files.createDirectories(out);
        int offset = Integer.parseInt(opt.getOrDefault("offset", "0"));
        WARMUP = Integer.parseInt(opt.getOrDefault("warmup", "200000"));
        int records = Integer.parseInt(opt.get("records"));
        RATE = Double.parseDouble(opt.getOrDefault("rate", "0"));
        boolean dumpFinal = Boolean.parseBoolean(opt.getOrDefault("dump-final", "false"));
        boolean forbidGpu = Boolean.parseBoolean(opt.getOrDefault("forbid-gpu", "false"));
        int parallelism = Integer.parseInt(opt.getOrDefault("parallelism", "1"));
        TOTAL = WARMUP + records;
        if (TOTAL % parallelism != 0) {
            throw new IllegalArgumentException("warmup+records must be divisible by parallelism");
        }

        long tLoad = System.nanoTime();
        COLS = loadColumns(data.resolve("log"), offset, TOTAL);
        SEND_NANOS = new long[TOTAL];
        CaseDef def = CaseDef.of(caseId);
        if (def.aggTriggers) {
            def.triggers = joinTriggers(1_800_000L);
        }
        if (def.needsDim) {
            loadDim(data.resolve("video"));
        }
        System.out.printf(
                Locale.ROOT,
                "loaded %d rows (offset %d) in %.1fs%n",
                TOTAL,
                offset,
                (System.nanoTime() - tLoad) / 1e9);

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setRuntimeMode(RuntimeExecutionMode.STREAMING);
        env.setParallelism(parallelism);
        if (opt.containsKey("buffer-timeout-ms")) {
            env.setBufferTimeout(Long.parseLong(opt.get("buffer-timeout-ms")));
        }
        env.getConfig()
                .setAutoWatermarkInterval(Long.parseLong(opt.getOrDefault("watermark-ms", "20")));
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);
        if (MEGA) {
            configureMegaFlink(tableEnv);
        }
        Configuration tc = tableEnv.getConfig().getConfiguration();
        tc.setString("table.local-time-zone", "UTC");
        if (MEGA) {
            tc.setString("cudf.flink.fallback.enabled", "true");
        }
        for (Map.Entry<String, String> e : conf.entrySet()) {
            tc.setString(e.getKey(), e.getValue());
        }

        String[] names = new String[LOG_COLS.length + 1];
        TypeInformation<?>[] types = new TypeInformation<?>[names.length];
        names[0] = "seq";
        types[0] = Types.LONG;
        for (int i = 0; i < LOG_COLS.length; i++) {
            names[i + 1] = LOG_COLS[i];
            types[i + 1] = Types.LONG;
        }
        RowTypeInfo rowType = new RowTypeInfo(types, names);
        DataStream<Long> seqStream = env.fromSequence(0, TOTAL - 1).setParallelism(1);
        if (parallelism > 1) {
            // single cheap Long source; replay (Row building + pacing) runs at `parallelism`, subtask i
            // gets global rows i, i+p, i+2p, ... so every subtask keeps time order
            seqStream = seqStream.partitionCustom((k, n) -> (int) (k % n), v -> v);
        }
        DataStream<Row> rows =
                seqStream
                        .map(new ReplayMap())
                        .returns(rowType)
                        .name("kuairand-replay")
                        .assignTimestampsAndWatermarks(
                                WatermarkStrategy.<Row>forMonotonousTimestamps()
                                        .withTimestampAssigner(
                                                (r, ts) -> (Long) r.getField(3)))
                        .name("kuairand-watermarks");
        Schema.Builder sb = Schema.newBuilder();
        for (String n : names) {
            sb.column(n, DataTypes.BIGINT().notNull());
        }
        sb.columnByMetadata("ts", DataTypes.TIMESTAMP_LTZ(3), "rowtime");
        sb.watermark("ts", "SOURCE_WATERMARK()");
        sb.columnByExpression("proc_time", "PROCTIME()");
        tableEnv.createTemporaryView("events", tableEnv.fromDataStream(rows, sb.build()));
        if (def.needsDim) {
            // LEFT join output carries nullable dimension columns; Mega-Flink's GPU lookup checks the
            // output type equals the dimension schema, so the LEFT case declares nullable columns (no PK)
            boolean leftCase = false; // PK-less nullable variant fails "lookup key is not proven to contain the PK"
            tableEnv.executeSql(
                    "CREATE TEMPORARY TABLE video_dim (video_id BIGINT, author_id BIGINT"
                            + (leftCase ? "" : ", PRIMARY KEY (video_id) NOT ENFORCED")
                            + ") WITH ('connector'='"
                            + (MEGA ? "fep-video-dim-gpu" : FepVideoDimFactory.IDENTIFIER)
                            + "')");
        }
        for (String view : def.views) {
            tableEnv.executeSql(view);
        }
        Table result = tableEnv.sqlQuery(def.sql);
        String plan = result.explain();
        Files.write(out.resolve("plan.txt"), plan.getBytes(StandardCharsets.UTF_8));
        for (String p : expectPlan) {
            if (!plan.contains(p)) {
                throw new IllegalStateException("plan does not contain '" + p + "'\n" + plan);
            }
        }
        if (forbidGpu && plan.contains("Gpu")) {
            throw new IllegalStateException("CPU run unexpectedly contains a GPU node\n" + plan);
        }
        tableEnv.toChangelogStream(result)
                .sinkTo(new CaptureSink())
                .name("capture-sink")
                .setParallelism(1);
        try {
            Files.write(
                    out.resolve("plan-full.txt"),
                    result.explain(org.apache.flink.table.api.ExplainDetail.JSON_EXECUTION_PLAN)
                            .getBytes(StandardCharsets.UTF_8));
            org.apache.flink.streaming.api.graph.StreamGraph sg = env.getStreamGraph(false);
            StringBuilder jg = new StringBuilder("JobGraph vertices (operator chains):\n");
            for (org.apache.flink.runtime.jobgraph.JobVertex v : sg.getJobGraph().getVertices()) {
                jg.append("- ").append(v.getName()).append("  [parallelism=")
                        .append(v.getParallelism()).append("]\n");
            }
            Files.write(out.resolve("jobgraph.txt"), jg.toString().getBytes(StandardCharsets.UTF_8));
        } catch (Throwable t) {
            Files.write(out.resolve("jobgraph.txt"), ("failed: " + t).getBytes(StandardCharsets.UTF_8));
        }

        long jobStart = System.nanoTime();
        env.execute("fep-" + caseId + "-" + label);
        long jobEnd = System.nanoTime();

        // ---------------- post-processing (not timed) ----------------
        long firstMeasured = SEND_NANOS[WARMUP];
        long lastSink = 0;
        for (int i = 0; i < OUT_COUNT; i++) {
            lastSink = Math.max(lastSink, OUT_NANOS[i]);
        }
        long[] lat = new long[OUT_COUNT];
        int nLat = 0;
        int noRef = 0;
        long[] kindCounts = new long[4];
        for (int i = 0; i < OUT_COUNT; i++) {
            byte kind = OUT_KINDS[i];
            kindCounts[kind]++;
            if (kind == 1 || kind == 3) { // -U / -D carry no new visibility
                continue;
            }
            int ref = def.latencyRef(OUT_ROWS.get(i));
            if (ref < 0) {
                noRef++;
                continue;
            }
            if (ref < WARMUP) {
                continue;
            }
            lat[nLat++] = OUT_NANOS[i] - SEND_NANOS[ref];
        }
        Arrays.sort(lat, 0, nLat);
        double seconds = (lastSink - firstMeasured) / 1e9;
        double throughput = records / seconds;
        double sourceSeconds = (SOURCE_DONE_NANOS - firstMeasured) / 1e9;

        // materialize final table
        Map<String, Integer> table = new HashMap<>();
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < OUT_COUNT; i++) {
            long[] r = OUT_ROWS.get(i);
            line.setLength(0);
            for (int c = 0; c < r.length; c++) {
                if (c > 0) {
                    line.append('\t');
                }
                if (r[c] == Long.MIN_VALUE) {
                    line.append("NULL");
                } else {
                    line.append(r[c]);
                }
            }
            String key = line.toString();
            int delta = (OUT_KINDS[i] == 0 || OUT_KINDS[i] == 2) ? 1 : -1;
            table.merge(key, delta, Integer::sum);
        }
        List<String> finalRows = new ArrayList<>(table.size());
        long negative = 0;
        for (Map.Entry<String, Integer> e : table.entrySet()) {
            int c = e.getValue();
            if (c < 0) {
                negative++;
            }
            for (int k = 0; k < c; k++) {
                finalRows.add(e.getKey());
            }
        }
        finalRows.sort(null);
        MessageDigest sha = MessageDigest.getInstance("SHA-256");
        for (String r : finalRows) {
            sha.update(r.getBytes(StandardCharsets.UTF_8));
            sha.update((byte) '\n');
        }
        String digest = hex(sha.digest());
        if (dumpFinal) {
            try (BufferedWriter w =
                    Files.newBufferedWriter(out.resolve("final.tsv"), StandardCharsets.UTF_8)) {
                for (String r : finalRows) {
                    w.write(r);
                    w.write('\n');
                }
            }
        }

        StringBuilder js = new StringBuilder("{\n");
        kv(js, "timestamp", Instant.now().toString());
        kv(js, "case", caseId);
        kv(js, "label", label);
        kv(js, "sql", def.sql);
        kv(js, "views", String.join(" ; ", def.views));
        kv(js, "conf", conf.toString());
        kv(js, "offset", offset);
        kv(js, "warmup_records", WARMUP);
        kv(js, "measured_records", records);
        kv(js, "offered_rate", RATE);
        kv(js, "parallelism", parallelism);
        kv(js, "throughput_records_per_s", throughput);
        kv(js, "measured_seconds", seconds);
        kv(js, "source_seconds_measured", sourceSeconds);
        kv(js, "job_wall_seconds", (jobEnd - jobStart) / 1e9);
        kv(js, "latency_samples", nLat);
        kv(js, "latency_rows_without_trigger", noRef);
        kv(js, "latency_p50_ms", pct(lat, nLat, 0.50) / 1e6);
        kv(js, "latency_p95_ms", pct(lat, nLat, 0.95) / 1e6);
        kv(js, "latency_p99_ms", pct(lat, nLat, 0.99) / 1e6);
        kv(js, "latency_max_ms", (nLat == 0 ? 0 : lat[nLat - 1]) / 1e6);
        kv(js, "out_changelog_rows", OUT_COUNT);
        kv(js, "out_insert", kindCounts[0]);
        kv(js, "out_update_before", kindCounts[1]);
        kv(js, "out_update_after", kindCounts[2]);
        kv(js, "out_delete", kindCounts[3]);
        kv(js, "final_rows", finalRows.size());
        kv(js, "final_negative_keys", negative);
        kv(js, "final_sha256", digest);
        kv(js, "gpu_in_plan", plan.contains("Gpu"));
        kv(js, "oracle_join_pairs", ORACLE_PAIRS);
        js.append("  \"shim_status\": ").append(shimStatusJson()).append(",\n");
        js.append("  \"metrics\": ").append(metricsJson()).append("\n}\n");
        Files.write(out.resolve("result.json"), js.toString().getBytes(StandardCharsets.UTF_8));
        System.out.printf(
                Locale.ROOT,
                "RESULT case=%s label=%s rate=%.0f thr=%.0f/s p50=%.2fms p95=%.2fms p99=%.2fms "
                        + "out=%d final=%d sha=%s%n",
                caseId,
                label,
                RATE,
                throughput,
                pct(lat, nLat, 0.50) / 1e6,
                pct(lat, nLat, 0.95) / 1e6,
                pct(lat, nLat, 0.99) / 1e6,
                OUT_COUNT,
                finalRows.size(),
                digest.substring(0, 16));
        System.exit(0);
    }

    /**
     * Mega-Flink only: the build's benchmark harness class sets the GPU resource and batching defaults
     * used for the published GPU numbers. It is package-private in that build, hence reflection.
     */
    static void configureMegaFlink(StreamTableEnvironment tableEnv) throws Exception {
        Class<?> c = Class.forName("io.github.cb521.cudfflink.benchmark.BenchmarkGpuResource");
        Method m = c.getDeclaredMethod("configure", StreamTableEnvironment.class);
        m.setAccessible(true);
        m.invoke(null, tableEnv);
    }

    // ------------------------------------------------------------------ cases

    static final class CaseDef {
        String sql;
        List<String> views = new ArrayList<>();
        boolean needsDim;
        // latency reference: list of output field indexes holding input seqs (max is used), or
        // windowEndField >= 0 for window outputs.
        int[] seqFields = new int[0];
        int windowEndField = -1;

        boolean aggTriggers;
        HashMap<Long, long[]> triggers;

        int latencyRef(long[] row) {
            if (aggTriggers) {
                long[] t = triggers.get(row[0]);
                int k = (int) row[1];
                return (t == null || k < 1 || k > t.length) ? -1 : (int) t[k - 1];
            }
            if (windowEndField >= 0) {
                return firstAtOrAfter(row[windowEndField]);
            }
            long m = -1;
            for (int f : seqFields) {
                m = Math.max(m, row[f]);
            }
            return (int) m;
        }

        static CaseDef of(String id) {
            CaseDef d = new CaseDef();
            switch (id) {
                case "c1-filter":
                    // Same feature logic as the batch-side case C, minus LOG/LEAST (not GPU Calc).
                    d.sql =
                            "SELECT seq, user_id, video_id, time_ms, "
                                    + "CAST(play_time_ms AS DOUBLE) / CAST(duration_ms + 1 AS DOUBLE) AS watch_ratio, "
                                    + "CASE WHEN duration_ms < 15000 THEN 0 WHEN duration_ms < 60000 THEN 1 ELSE 2 END AS duration_bucket, "
                                    + "is_like + is_follow + is_comment + is_forward AS interact_cnt, "
                                    + "is_click "
                                    + "FROM events WHERE duration_ms > 0 AND play_time_ms >= 0 AND is_rand = 0 "
                                    + "AND (play_time_ms > 3000 OR is_click = 1)";
                    d.seqFields = new int[] {0};
                    break;
                case "c2-hop":
                    d.sql =
                            "SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt, "
                                    + "SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms "
                                    + "FROM TABLE(HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE)) "
                                    + "GROUP BY user_id, window_start, window_end";
                    d.windowEndField = 2;
                    break;
                case "c0-pass":
                    // ceiling probe: no real work, only source replay + exchange + sink
                    d.sql = "SELECT seq, user_id, video_id, play_time_ms FROM events";
                    d.seqFields = new int[] {0};
                    break;
                case "c2-tumble":
                    d.sql =
                            "SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt, "
                                    + "SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms "
                                    + "FROM TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE)) "
                                    + "GROUP BY user_id, window_start, window_end";
                    d.windowEndField = 2;
                    break;
                case "c3-topn":
                    d.sql =
                            "SELECT seq, user_id, video_id, play_time_ms FROM ("
                                    + "SELECT seq, user_id, video_id, play_time_ms, "
                                    + "ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn "
                                    + "FROM events) WHERE rn <= 10";
                    d.seqFields = new int[] {0};
                    break;
                case "c3-topn-rn":
                    d.sql =
                            "SELECT seq, user_id, video_id, play_time_ms, rn FROM ("
                                    + "SELECT seq, user_id, video_id, play_time_ms, "
                                    + "ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn "
                                    + "FROM events) WHERE rn <= 10";
                    d.seqFields = new int[] {0};
                    break;
                case "c4-dedup":
                    d.sql =
                            "SELECT seq, user_id, video_id, play_time_ms, time_ms FROM ("
                                    + "SELECT seq, user_id, video_id, play_time_ms, time_ms, "
                                    + "ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY ts DESC) AS rn "
                                    + "FROM events) WHERE rn = 1";
                    d.seqFields = new int[] {0};
                    break;
                case "c5-join":
                case "c5-join-filter":
                case "c5-join-agg":
                    d.views.add(
                            "CREATE TEMPORARY VIEW imp AS SELECT seq, user_id * 100000000 + video_id AS uv, "
                                    + "user_id, duration_ms, ts FROM events");
                    d.views.add(
                            "CREATE TEMPORARY VIEW clk AS SELECT seq, user_id * 100000000 + video_id AS uv, "
                                    + "play_time_ms, ts FROM events WHERE is_click = 1");
                    String join =
                            "FROM imp i JOIN clk c ON i.uv = c.uv "
                                    + "AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE";
                    if (id.equals("c5-join")) {
                        d.sql =
                                "SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms "
                                        + join;
                        d.seqFields = new int[] {0, 1};
                    } else if (id.equals("c5-join-filter")) {
                        d.sql =
                                "SELECT i.seq, c.seq, i.uv, c.play_time_ms "
                                        + join
                                        + " WHERE c.play_time_ms >= i.duration_ms";
                        d.seqFields = new int[] {0, 1};
                    } else {
                        // COUNT/SUM only: the shape Mega-Flink can fuse Join->Local Aggregate.
                        // Latency reference: output (user, clicks=k) becomes visible with the
                        // k-th join match of that user (matches ordered by trigger seq).
                        d.sql =
                                "SELECT i.user_id, COUNT(*) AS clicks, SUM(c.play_time_ms) AS play_ms "
                                        + join
                                        + " GROUP BY i.user_id";
                        d.aggTriggers = true;
                    }
                    break;
                case "c6-lookup":
                case "c6-lookup-left":
                    d.needsDim = true;
                    d.sql =
                            "SELECT e.seq, e.video_id, d.author_id FROM events AS e "
                                    + (id.endsWith("left") ? "LEFT " : "")
                                    + "JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d "
                                    + "ON e.video_id = d.video_id";
                    d.seqFields = new int[] {0};
                    break;
                default:
                    throw new IllegalArgumentException("unknown case " + id);
            }
            return d;
        }
    }

    /**
     * Exact oracle of the c5 interval join (c.ts BETWEEN i.ts AND i.ts + bound, same uv, clicks are
     * rows with is_click=1): per impression user, the sorted trigger seqs max(i.seq, c.seq).
     */
    static HashMap<Long, long[]> joinTriggers(long boundMs) {
        long[] user = COLS[0];
        long[] video = COLS[1];
        long[] time = COLS[2];
        long[] click = COLS[3];
        HashMap<Long, long[]> imps = new HashMap<>();
        HashMap<Long, Integer> impCount = new HashMap<>();
        for (int s = 0; s < TOTAL; s++) {
            long uv = user[s] * 100000000L + video[s];
            long[] l = imps.get(uv);
            int n = impCount.getOrDefault(uv, 0);
            if (l == null) {
                l = new long[2];
                imps.put(uv, l);
            } else if (n == l.length) {
                l = Arrays.copyOf(l, n * 2);
                imps.put(uv, l);
            }
            l[n] = s;
            impCount.put(uv, n + 1);
        }
        HashMap<Long, long[]> per = new HashMap<>();
        HashMap<Long, Integer> perCount = new HashMap<>();
        long pairs = 0;
        for (int c = 0; c < TOTAL; c++) {
            if (click[c] != 1) {
                continue;
            }
            long uv = user[c] * 100000000L + video[c];
            long[] l = imps.get(uv);
            int n = impCount.get(uv);
            for (int k = 0; k < n; k++) {
                int i = (int) l[k];
                if (time[i] <= time[c] && time[c] <= time[i] + boundMs) {
                    long u = user[i];
                    long[] t = per.get(u);
                    int m = perCount.getOrDefault(u, 0);
                    if (t == null) {
                        t = new long[4];
                        per.put(u, t);
                    } else if (m == t.length) {
                        t = Arrays.copyOf(t, m * 2);
                        per.put(u, t);
                    }
                    t[m] = Math.max(i, c);
                    perCount.put(u, m + 1);
                    pairs++;
                }
            }
        }
        HashMap<Long, long[]> out = new HashMap<>();
        for (Map.Entry<Long, long[]> e : per.entrySet()) {
            long[] t = Arrays.copyOf(e.getValue(), perCount.get(e.getKey()));
            Arrays.sort(t);
            out.put(e.getKey(), t);
        }
        System.out.printf("join oracle: %d matched pairs, %d users%n", pairs, out.size());
        ORACLE_PAIRS = pairs;
        return out;
    }

    static long ORACLE_PAIRS = -1;

    /** Index of the first replayed row with time_ms >= t (the row whose watermark closes t). */
    static int firstAtOrAfter(long t) {
        long[] time = COLS[2];
        int lo = 0;
        int hi = TOTAL;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (time[mid] < t) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo >= TOTAL ? -1 : lo;
    }

    // ------------------------------------------------------------------ source

    static final class ReplayMap extends RichMapFunction<Long, Row> {
        private static final long serialVersionUID = 1L;
        private transient long start;
        private transient double nanosPerRow;

        @Override
        public Row map(Long index) {
            int s = (int) index.longValue();
            long now = System.nanoTime();
            if (start == 0) {
                start = now + 1_000_000L;
                nanosPerRow = RATE > 0 ? 1e9 / RATE : 0;
            }
            long send;
            if (RATE > 0) {
                long target = start + (long) (s * nanosPerRow);
                while (now < target) {
                    long gap = target - now;
                    if (gap > 200_000L) {
                        LockSupport.parkNanos(gap - 100_000L);
                    } else {
                        Thread.onSpinWait();
                    }
                    now = System.nanoTime();
                }
                send = target;
            } else {
                send = now;
            }
            SEND_NANOS[s] = send;
            if (s == TOTAL - 1) {
                SOURCE_DONE_NANOS = System.nanoTime();
            }
            Row r = new Row(LOG_COLS.length + 1);
            r.setField(0, (long) s);
            for (int c = 0; c < LOG_COLS.length; c++) {
                r.setField(c + 1, COLS[c][s]);
            }
            return r;
        }
    }

    // ------------------------------------------------------------------ sink

    static final class CaptureSink implements Sink<Row> {
        private static final long serialVersionUID = 1L;

        @Override
        public SinkWriter<Row> createWriter(WriterInitContext context) {
            return new SinkWriter<Row>() {
                @Override
                public void write(Row value, Context context) {
                    long now = System.nanoTime();
                    int n = value.getArity();
                    long[] f = new long[n];
                    for (int i = 0; i < n; i++) {
                        f[i] = toLong(value.getField(i));
                    }
                    byte kind = (byte) value.getKind().toByteValue();
                    synchronized (SINK_LOCK) {
                        if (OUT_COUNT == OUT_KINDS.length) {
                            OUT_KINDS = Arrays.copyOf(OUT_KINDS, OUT_COUNT * 2);
                            OUT_NANOS = Arrays.copyOf(OUT_NANOS, OUT_COUNT * 2);
                        }
                        OUT_KINDS[OUT_COUNT] = kind;
                        OUT_NANOS[OUT_COUNT] = now;
                        OUT_ROWS.add(f);
                        OUT_COUNT++;
                    }
                }

                @Override
                public void flush(boolean endOfInput) {}

                @Override
                public void close() {}
            };
        }
    }

    static long toLong(Object o) {
        if (o == null) {
            return Long.MIN_VALUE;
        }
        if (o instanceof Double) {
            return Double.doubleToLongBits((Double) o);
        }
        if (o instanceof Float) {
            return Double.doubleToLongBits(((Float) o).doubleValue());
        }
        if (o instanceof Number) {
            return ((Number) o).longValue();
        }
        if (o instanceof Boolean) {
            return ((Boolean) o) ? 1 : 0;
        }
        if (o instanceof Instant) {
            return ((Instant) o).toEpochMilli();
        }
        if (o instanceof LocalDateTime) {
            return ((LocalDateTime) o).toInstant(ZoneOffset.UTC).toEpochMilli();
        }
        throw new IllegalArgumentException("unsupported output type " + o.getClass());
    }

    // ------------------------------------------------------------------ data

    static long[][] loadColumns(Path dir, int offset, int n) throws IOException {
        long[][] cols = new long[LOG_COLS.length][];
        for (int c = 0; c < LOG_COLS.length; c++) {
            cols[c] = readLongs(dir.resolve(LOG_COLS[c] + ".i64"), offset, n);
        }
        return cols;
    }

    static long[] readLongs(Path file, long offset, int n) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long avail = ch.size() / 8 - offset;
            if (n < 0) {
                n = (int) avail;
            }
            if (avail < n) {
                throw new IllegalArgumentException("not enough rows in " + file);
            }
            MappedByteBuffer mb = ch.map(FileChannel.MapMode.READ_ONLY, offset * 8, (long) n * 8);
            LongBuffer lb = mb.order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
            long[] out = new long[n];
            lb.get(out);
            return out;
        }
    }

    /** Dimension restricted to the videos that occur in the replay slice (unique keys). */
    static void loadDim(Path dir) throws IOException {
        long[] vid = readLongs(dir.resolve("video_id.i64"), 0, -1);
        long[] author = readLongs(dir.resolve("author_id.i64"), 0, -1);
        long maxId = 0;
        for (long v : vid) {
            maxId = Math.max(maxId, v);
        }
        boolean[] used = new boolean[(int) maxId + 1];
        long[] sliceVideos = COLS[1];
        for (int i = 0; i < TOTAL; i++) {
            long v = sliceVideos[i];
            if (v >= 0 && v <= maxId) {
                used[(int) v] = true;
            }
        }
        int n = 0;
        for (long v : vid) {
            if (used[(int) v]) {
                n++;
            }
        }
        // optional: pad the dimension with extra (unprobed) videos up to -Dfep.dim.target rows
        int target = Integer.getInteger("fep.dim.target", 0);
        for (int i = 0; i < vid.length && n < target; i++) {
            int v = (int) vid[i];
            if (!used[v]) {
                used[v] = true;
                n++;
            }
        }
        DIM_KEYS = new long[n];
        DIM_VALUES = new long[n];
        int j = 0;
        boolean[] seen = new boolean[used.length];
        for (int i = 0; i < vid.length; i++) {
            int v = (int) vid[i];
            if (used[v]) {
                if (seen[v]) {
                    throw new IllegalStateException("duplicate video id " + v);
                }
                seen[v] = true;
                DIM_KEYS[j] = v;
                DIM_VALUES[j] = author[i];
                j++;
            }
        }
        System.out.printf("dimension rows (videos present in slice): %d%n", n);
    }

    // ------------------------------------------------------------------ util

    static long pct(long[] a, int n, double p) {
        if (n == 0) {
            return 0;
        }
        int idx = (int) Math.ceil(p * n) - 1;
        return a[Math.max(0, Math.min(n - 1, idx))];
    }

    static String hex(byte[] b) {
        StringBuilder s = new StringBuilder();
        for (byte x : b) {
            s.append(String.format("%02x", x));
        }
        return s.toString();
    }

    static void kv(StringBuilder js, String k, Object v) {
        js.append("  \"").append(k).append("\": ");
        if (v instanceof Number || v instanceof Boolean) {
            js.append(v);
        } else {
            js.append('"')
                    .append(String.valueOf(v).replace("\\", "\\\\").replace("\"", "\\\""))
                    .append('"');
        }
        js.append(",\n");
    }

    static String shimStatusJson() {
        String[] names = {
            "PlannerShimStatus", "WindowPlannerShimStatus", "TwoPhaseWindowPlannerShimStatus",
            "TopNPlannerShimStatus", "DeduplicatePlannerShimStatus",
            "IntervalJoinPlannerShimStatus", "TwoPhaseAggregatePlannerShimStatus",
            "AggregatePlannerShimStatus", "LookupJoinPlannerShimStatus"
        };
        StringBuilder s = new StringBuilder("{");
        boolean first = true;
        for (String n : names) {
            try {
                Object v = Class.forName("io.github.cb521.cudfflink." + n).getMethod("status").invoke(null);
                if (!first) {
                    s.append(", ");
                }
                first = false;
                s.append('"').append(n).append("\": \"")
                        .append(String.valueOf(v).replace("\\", "/").replace("\"", "'").replace("\n", " "))
                        .append('"');
            } catch (Throwable t) {
                // not present
            }
        }
        return s.append('}').toString();
    }

    static String metricsJson() {
        String[] classes = {
            "io.github.cb521.cudfflink.RuntimeMetrics",
            "io.github.cb521.cudfflink.CalcRuntimeMetrics",
            "io.github.cb521.cudfflink.AggregateRuntimeMetrics",
            "io.github.cb521.cudfflink.TopNRuntimeMetrics",
            "io.github.cb521.cudfflink.DeduplicateRuntimeMetrics",
            "io.github.cb521.cudfflink.JoinRuntimeMetrics",
            "io.github.cb521.cudfflink.GpuDeviceBatchRuntimeMetrics"
        };
        StringBuilder s = new StringBuilder("{");
        boolean first = true;
        for (String cn : classes) {
            try {
                Class<?> c = Class.forName(cn);
                Method m = c.getMethod("snapshot");
                Object snap = m.invoke(null);
                if (!first) {
                    s.append(", ");
                }
                first = false;
                s.append('"').append(c.getSimpleName()).append("\": {");
                boolean f2 = true;
                for (Field f : snap.getClass().getFields()) {
                    if (Modifier.isStatic(f.getModifiers())) {
                        continue;
                    }
                    Object v = f.get(snap);
                    if (!(v instanceof Number) && !(v instanceof Boolean)) {
                        continue;
                    }
                    if (!f2) {
                        s.append(", ");
                    }
                    f2 = false;
                    s.append('"').append(f.getName()).append("\": ").append(v);
                }
                s.append('}');
            } catch (Throwable t) {
                // metrics class not present or no snapshot(); skip
            }
        }
        return s.append('}').toString();
    }
}
