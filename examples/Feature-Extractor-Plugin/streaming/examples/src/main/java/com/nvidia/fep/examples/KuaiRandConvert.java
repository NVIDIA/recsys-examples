// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.examples;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

/**
 * One-off conversion of KuaiRand-27K CSV to fixed-width little-endian int64 column files.
 *
 * <p>log: rows are globally sorted by (time_ms, original CSV line) so a replay in file order is an
 * event-time ordered stream. video: video_id, author_id, video_duration_ms (as read).
 *
 * <p>Usage: KuaiRandConvert log|video SRC_CSV DST_DIR
 */
public final class KuaiRandConvert {
    // Columns kept from the log CSV (by header name) in output order.
    static final String[] LOG_COLS = {
        "user_id", "video_id", "time_ms", "is_click", "is_like", "is_follow", "is_comment",
        "is_forward", "is_hate", "long_view", "play_time_ms", "duration_ms", "is_rand", "tab"
    };
    static final String[] VIDEO_COLS = {"video_id", "author_id", "video_duration"};

    public static void main(String[] args) throws Exception {
        String kind = args[0];
        Path src = Paths.get(args[1]);
        Path dst = Paths.get(args[2]);
        Files.createDirectories(dst);
        String[] want = kind.equals("log") ? LOG_COLS : VIDEO_COLS;
        long t0 = System.nanoTime();
        Parsed p = parse(src, want);
        System.out.printf("parsed %d rows in %.1fs%n", p.rows, (System.nanoTime() - t0) / 1e9);
        if (kind.equals("log")) {
            int timeIdx = 2;
            long[] t = p.cols[timeIdx];
            long min = Long.MAX_VALUE;
            long max = Long.MIN_VALUE;
            for (int i = 0; i < p.rows; i++) {
                min = Math.min(min, t[i]);
                max = Math.max(max, t[i]);
            }
            if (max - min >= (1L << 35) || p.rows >= (1 << 28)) {
                throw new IllegalStateException("packing range exceeded");
            }
            long[] keys = new long[p.rows];
            for (int i = 0; i < p.rows; i++) {
                keys[i] = ((t[i] - min) << 28) | i;
            }
            Arrays.parallelSort(keys);
            int[] perm = new int[p.rows];
            for (int i = 0; i < p.rows; i++) {
                perm[i] = (int) (keys[i] & ((1L << 28) - 1));
            }
            keys = null;
            for (int c = 0; c < p.cols.length; c++) {
                long[] in = p.cols[c];
                long[] out = new long[p.rows];
                for (int i = 0; i < p.rows; i++) {
                    out[i] = in[perm[i]];
                }
                p.cols[c] = out;
            }
            System.out.printf("sorted, time range %d..%d%n", min, max);
        }
        for (int c = 0; c < want.length; c++) {
            write(dst.resolve(want[c] + ".i64"), p.cols[c], p.rows);
        }
        Files.write(
                dst.resolve("meta.txt"),
                ("rows=" + p.rows + "\ncolumns=" + String.join(",", want) + "\nsource="
                                + src.getFileName() + "\n")
                        .getBytes());
        System.out.printf("done in %.1fs%n", (System.nanoTime() - t0) / 1e9);
    }

    static final class Parsed {
        long[][] cols;
        int rows;
    }

    static Parsed parse(Path src, String[] want) throws IOException {
        try (InputStream in = new FileInputStream(src.toFile())) {
            byte[] buf = new byte[1 << 24];
            int len = 0;
            int pos = 0;
            // header
            StringBuilder header = new StringBuilder();
            int b;
            while ((b = in.read()) != '\n') {
                header.append((char) b);
            }
            String[] names = header.toString().trim().split(",");
            int[] map = new int[names.length];
            Arrays.fill(map, -1);
            for (int i = 0; i < names.length; i++) {
                for (int j = 0; j < want.length; j++) {
                    if (names[i].equals(want[j])) {
                        map[i] = j;
                    }
                }
            }
            int cap = 1 << 20;
            long[][] cols = new long[want.length][cap];
            int rows = 0;
            int field = 0;
            long value = 0;
            boolean neg = false;
            boolean frac = false;
            boolean any = false;
            while (true) {
                if (pos >= len) {
                    len = in.read(buf, 0, buf.length);
                    pos = 0;
                    if (len <= 0) {
                        break;
                    }
                }
                byte ch = buf[pos++];
                if (ch == ',' || ch == '\n') {
                    if (field < map.length && map[field] >= 0) {
                        if (rows == cap) {
                            cap *= 2;
                            for (int c = 0; c < cols.length; c++) {
                                cols[c] = Arrays.copyOf(cols[c], cap);
                            }
                        }
                        cols[map[field]][rows] = any ? (neg ? -value : value) : Long.MIN_VALUE;
                    }
                    value = 0;
                    neg = false;
                    frac = false;
                    any = false;
                    field++;
                    if (ch == '\n') {
                        rows++;
                        field = 0;
                    }
                } else if (ch >= '0' && ch <= '9') {
                    if (!frac) {
                        value = value * 10 + (ch - '0');
                        any = true;
                    }
                } else if (ch == '-') {
                    neg = true;
                } else if (ch == '.') {
                    frac = true; // integral part only (video_duration is "87433.0")
                } else if (ch == '\r') {
                    // ignore
                } else {
                    // non-numeric field (string column we do not keep); mark as null if kept
                    if (field < map.length && map[field] >= 0) {
                        any = false;
                    }
                }
            }
            Parsed p = new Parsed();
            p.cols = cols;
            p.rows = rows;
            return p;
        }
    }

    static void write(Path path, long[] data, int rows) throws IOException {
        try (FileChannel ch =
                FileChannel.open(
                        path,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer bb = ByteBuffer.allocateDirect(1 << 23).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < rows; i++) {
                if (!bb.hasRemaining()) {
                    bb.flip();
                    while (bb.hasRemaining()) {
                        ch.write(bb);
                    }
                    bb.clear();
                }
                bb.putLong(data[i]);
            }
            bb.flip();
            while (bb.hasRemaining()) {
                ch.write(bb);
            }
        }
    }
}
