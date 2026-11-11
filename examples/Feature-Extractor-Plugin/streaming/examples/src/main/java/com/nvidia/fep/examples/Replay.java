// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.examples;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.connector.sink2.Sink;
import org.apache.flink.api.connector.sink2.SinkWriter;
import org.apache.flink.api.connector.sink2.WriterInitContext;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.types.Row;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.locks.LockSupport;

/**
 * Shared pieces of the example jobs: an in-JVM replay of the KuaiRand-27K log (as event-time
 * ordered rows), a capture sink, and the SHA-256 of the final result table.
 *
 * <p>The jobs run on an embedded Flink mini cluster in one JVM, so the replayed columns and the
 * captured output live in static fields. Throughput is the number of measured input rows divided by
 * the time from the emission of the first measured row to the arrival of the last output row at
 * the sink; the replay, Flink runtime, state, format conversion, device transfers and the sink are
 * all inside that interval. The result table is built after the job, outside the timed region.
 */
public final class Replay {
    /** Log columns, in the order of the converted column files. */
    public static final String[] LOG_COLS = {
        "user_id", "video_id", "time_ms", "is_click", "is_like", "is_follow", "is_comment",
        "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "is_rand", "tab"
    };

    static long[][] cols;
    static int total;
    static int warmup;
    static double rate;
    static volatile long firstMeasuredNanos;
    static long[] dimKeys = new long[0];
    static long[] dimValues = new long[0];

    private static final Object SINK_LOCK = new Object();
    private static final List<long[]> OUT_ROWS = new ArrayList<>();
    private static byte[] outKinds = new byte[1 << 16];
    private static long[] outNanos = new long[1 << 16];
    private static int outCount;

    private Replay() {}

    /** Parses "--key value" pairs; repeated keys keep the last value. */
    public static Map<String, String> parseArgs(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            if (!args[i].startsWith("--")) {
                throw new IllegalArgumentException("expected --option, got " + args[i]);
            }
            m.put(args[i].substring(2), args[i + 1]);
        }
        return m;
    }

    /** Data directory: --data, else the KUAIRAND_DIR environment variable. */
    public static Path dataDir(Map<String, String> opt) {
        String d = opt.getOrDefault("data", System.getenv("KUAIRAND_DIR"));
        if (d == null) {
            throw new IllegalArgumentException(
                    "set KUAIRAND_DIR (or --data) to the directory holding log/ and video/");
        }
        return java.nio.file.Paths.get(d);
    }

    /** Loads the first {@code warmup + records} log rows (time ordered) into memory. */
    public static void load(Path data, int warmupRows, int records, double offeredRate)
            throws IOException {
        warmup = warmupRows;
        total = warmupRows + records;
        rate = offeredRate;
        cols = new long[LOG_COLS.length][];
        for (int c = 0; c < LOG_COLS.length; c++) {
            cols[c] = readLongs(data.resolve("log").resolve(LOG_COLS[c] + ".i64"), 0, total);
        }
    }

    /**
     * Loads the dimension (video_id, author_id), restricted to the videos that occur in the loaded
     * slice, as in the benchmark this example is derived from.
     */
    public static void loadDim(Path data) throws IOException {
        long[] vid = readLongs(data.resolve("video").resolve("video_id.i64"), 0, -1);
        long[] author = readLongs(data.resolve("video").resolve("author_id.i64"), 0, -1);
        long maxId = 0;
        for (long v : vid) {
            maxId = Math.max(maxId, v);
        }
        boolean[] used = new boolean[(int) maxId + 1];
        long[] sliceVideos = cols[1];
        for (int i = 0; i < total; i++) {
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
        dimKeys = new long[n];
        dimValues = new long[n];
        int j = 0;
        for (int i = 0; i < vid.length; i++) {
            if (used[(int) vid[i]]) {
                dimKeys[j] = vid[i];
                dimValues[j] = author[i];
                j++;
            }
        }
        System.out.printf(Locale.ROOT, "dimension rows (videos present in slice): %d%n", n);
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

    // ---------------------------------------------------------------- replay sources

    /** Paced (--rate rows/s) or saturated (--rate 0) replay of row index to a value. */
    abstract static class PacedMap<T> extends RichMapFunction<Long, T> {
        private static final long serialVersionUID = 1L;
        private transient long start;
        private transient double nanosPerRow;

        final void pace(int s) {
            long now = System.nanoTime();
            if (start == 0) {
                start = now + 1_000_000L;
                nanosPerRow = rate > 0 ? 1e9 / rate : 0;
            }
            if (rate > 0) {
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
                now = target;
            }
            if (s == warmup) {
                firstMeasuredNanos = now;
            }
        }
    }

    /** Row 0 = seq, rows 1..14 = LOG_COLS (all BIGINT), for the Flink SQL path. */
    public static final class RowReplay extends PacedMap<Row> {
        private static final long serialVersionUID = 1L;

        @Override
        public Row map(Long index) {
            int s = (int) index.longValue();
            pace(s);
            Row r = new Row(LOG_COLS.length + 1);
            r.setField(0, (long) s);
            for (int c = 0; c < LOG_COLS.length; c++) {
                r.setField(c + 1, cols[c][s]);
            }
            return r;
        }
    }

    /** Same columns as {@link RowReplay} plus a final TIMESTAMP_LTZ(3) column (time_ms), as RowData. */
    public static final class RowDataReplay extends PacedMap<RowData> {
        private static final long serialVersionUID = 1L;

        @Override
        public RowData map(Long index) {
            int s = (int) index.longValue();
            pace(s);
            GenericRowData r = new GenericRowData(LOG_COLS.length + 2);
            r.setField(0, (long) s);
            for (int c = 0; c < LOG_COLS.length; c++) {
                r.setField(c + 1, cols[c][s]);
            }
            r.setField(LOG_COLS.length + 1, TimestampData.fromEpochMillis(cols[2][s]));
            return r;
        }
    }

    /** Like {@link RowDataReplay} but without the timestamp column (for jobs without event time). */
    public static final class RowDataBigintReplay extends PacedMap<RowData> {
        private static final long serialVersionUID = 1L;

        @Override
        public RowData map(Long index) {
            int s = (int) index.longValue();
            pace(s);
            GenericRowData r = new GenericRowData(LOG_COLS.length + 1);
            r.setField(0, (long) s);
            for (int c = 0; c < LOG_COLS.length; c++) {
                r.setField(c + 1, cols[c][s]);
            }
            return r;
        }
    }

    // ---------------------------------------------------------------- sink

    /** Turns one output record into long values (NULL = Long.MIN_VALUE) and a changelog kind. */
    public interface Converter<T> extends java.io.Serializable {
        long[] values(T record);

        /** 0 = +I, 1 = -U, 2 = +U, 3 = -D (Flink RowKind byte values). */
        byte kind(T record);
    }

    /** Collects every record in memory with its arrival time. */
    public static final class CaptureSink<T> implements Sink<T> {
        private static final long serialVersionUID = 1L;
        private final Converter<T> converter;

        public CaptureSink(Converter<T> converter) {
            this.converter = converter;
        }

        @Override
        public SinkWriter<T> createWriter(WriterInitContext context) {
            return new SinkWriter<T>() {
                @Override
                public void write(T value, Context context) {
                    long now = System.nanoTime();
                    long[] f = converter.values(value);
                    byte kind = converter.kind(value);
                    synchronized (SINK_LOCK) {
                        if (outCount == outKinds.length) {
                            outKinds = Arrays.copyOf(outKinds, outCount * 2);
                            outNanos = Arrays.copyOf(outNanos, outCount * 2);
                        }
                        outKinds[outCount] = kind;
                        outNanos[outCount] = now;
                        OUT_ROWS.add(f);
                        outCount++;
                    }
                }

                @Override
                public void flush(boolean endOfInput) {}

                @Override
                public void close() {}
            };
        }
    }

    /** Converts one field value of a Row / RowData to a long. */
    public static long toLong(Object o) {
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
        if (o instanceof TimestampData) {
            return ((TimestampData) o).getMillisecond();
        }
        throw new IllegalArgumentException("unsupported output type " + o.getClass());
    }

    // ---------------------------------------------------------------- result

    /** Prints the result line, writes result.json (and final.tsv if requested), returns the digest. */
    public static String finish(String job, String engine, String variant, Map<String, String> opt,
            long jobWallNanos) throws IOException {
        int records = total - warmup;
        long lastSink = 0;
        long[] kindCounts = new long[4];
        for (int i = 0; i < outCount; i++) {
            lastSink = Math.max(lastSink, outNanos[i]);
            kindCounts[outKinds[i]]++;
        }
        double seconds = (lastSink - firstMeasuredNanos) / 1e9;
        double throughput = records / seconds;

        // materialize the final table (changelog applied), sorted, then hash
        Map<String, Integer> table = new HashMap<>();
        StringBuilder line = new StringBuilder();
        for (int i = 0; i < outCount; i++) {
            long[] r = OUT_ROWS.get(i);
            line.setLength(0);
            for (int c = 0; c < r.length; c++) {
                if (c > 0) {
                    line.append('\t');
                }
                line.append(r[c] == Long.MIN_VALUE ? "NULL" : Long.toString(r[c]));
            }
            int delta = (outKinds[i] == 0 || outKinds[i] == 2) ? 1 : -1;
            table.merge(line.toString(), delta, Integer::sum);
        }
        List<String> finalRows = new ArrayList<>(table.size());
        long negative = 0;
        for (Map.Entry<String, Integer> e : table.entrySet()) {
            if (e.getValue() < 0) {
                negative++;
            }
            for (int k = 0; k < e.getValue(); k++) {
                finalRows.add(e.getKey());
            }
        }
        finalRows.sort(null);
        MessageDigest sha;
        try {
            sha = MessageDigest.getInstance("SHA-256");
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        for (String r : finalRows) {
            sha.update(r.getBytes(StandardCharsets.UTF_8));
            sha.update((byte) '\n');
        }
        StringBuilder hex = new StringBuilder();
        for (byte x : sha.digest()) {
            hex.append(String.format(Locale.ROOT, "%02x", x));
        }
        String digest = hex.toString();

        Path out = java.nio.file.Paths.get(opt.getOrDefault("out", "."));
        Files.createDirectories(out);
        if (Boolean.parseBoolean(opt.getOrDefault("dump-final", "false"))) {
            try (BufferedWriter w =
                    Files.newBufferedWriter(
                            out.resolve(job + "-" + engine + "-" + variant + ".final.tsv"),
                            StandardCharsets.UTF_8)) {
                for (String r : finalRows) {
                    w.write(r);
                    w.write('\n');
                }
            }
        }
        String json =
                String.format(
                        Locale.ROOT,
                        "{\"job\": \"%s\", \"engine\": \"%s\", \"variant\": \"%s\","
                                + " \"warmup_records\": %d, \"measured_records\": %d,"
                                + " \"offered_rate\": %.0f, \"throughput_records_per_s\": %.1f,"
                                + " \"measured_seconds\": %.3f, \"job_wall_seconds\": %.3f,"
                                + " \"out_changelog_rows\": %d, \"out_insert\": %d,"
                                + " \"out_update_before\": %d, \"out_update_after\": %d,"
                                + " \"out_delete\": %d, \"final_rows\": %d,"
                                + " \"final_negative_keys\": %d, \"final_sha256\": \"%s\"}%n",
                        job, engine, variant, warmup, records, rate, throughput, seconds,
                        jobWallNanos / 1e9, outCount, kindCounts[0], kindCounts[1], kindCounts[2],
                        kindCounts[3], finalRows.size(), negative, digest);
        Files.write(
                out.resolve(job + "-" + engine + "-" + variant + ".result.json"),
                json.getBytes(StandardCharsets.UTF_8));
        System.out.printf(
                Locale.ROOT,
                "RESULT job=%s engine=%s variant=%s throughput=%.0f rows/s output=%d final_rows=%d"
                        + " sha256=%s%n",
                job, engine, variant, throughput, outCount, finalRows.size(), digest);
        return digest;
    }
}
