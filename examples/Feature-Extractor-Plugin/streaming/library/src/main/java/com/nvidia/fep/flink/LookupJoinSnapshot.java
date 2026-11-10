// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Canonical immutable dimension snapshot shared by CPU probing, CUDA rebuild and recovery checks. */
public final class LookupJoinSnapshot {
    private static final String STATE_VERSION = "v1";

    private final long[] keys;
    private final boolean[] keyValid;
    private final long[] payloads;
    private final boolean[] payloadValid;
    private final Map<Long, Integer> nonNullKeyRows;
    private final String checkpointMarker;

    private LookupJoinSnapshot(
            long[] keys,
            boolean[] keyValid,
            long[] payloads,
            boolean[] payloadValid,
            Map<Long, Integer> nonNullKeyRows,
            String checkpointMarker) {
        this.keys = keys;
        this.keyValid = keyValid;
        this.payloads = payloads;
        this.payloadValid = payloadValid;
        this.nonNullKeyRows = nonNullKeyRows;
        this.checkpointMarker = checkpointMarker;
    }

    /** Canonicalizes input order and rejects duplicate non-NULL keys before any active rebuild. */
    public static LookupJoinSnapshot create(
            String snapshotIdentity,
            long[] keys,
            boolean[] keyValid,
            long[] payloads,
            boolean[] payloadValid) {
        requireParallelArrays(keys, keyValid, payloads, payloadValid);
        String identity = Objects.requireNonNull(snapshotIdentity, "snapshotIdentity").trim();
        if (identity.isEmpty()) {
            throw new IllegalArgumentException("Lookup Join snapshot identity must be non-empty");
        }

        List<DimensionRow> rows = new ArrayList<>(keys.length);
        for (int row = 0; row < keys.length; row++) {
            rows.add(
                    new DimensionRow(
                            keyValid[row] ? keys[row] : 0L,
                            keyValid[row],
                            payloadValid[row] ? payloads[row] : 0L,
                            payloadValid[row]));
        }
        rows.sort(DimensionRow.CANONICAL_ORDER);

        long[] canonicalKeys = new long[rows.size()];
        boolean[] canonicalKeyValid = new boolean[rows.size()];
        long[] canonicalPayloads = new long[rows.size()];
        boolean[] canonicalPayloadValid = new boolean[rows.size()];
        Map<Long, Integer> keyRows = new HashMap<>(hashMapCapacity(rows.size()));
        for (int row = 0; row < rows.size(); row++) {
            DimensionRow value = rows.get(row);
            canonicalKeys[row] = value.key;
            canonicalKeyValid[row] = value.keyValid;
            canonicalPayloads[row] = value.payload;
            canonicalPayloadValid[row] = value.payloadValid;
            if (value.keyValid && keyRows.put(value.key, row) != null) {
                throw new IllegalArgumentException(
                        "Lookup Join dimension contains duplicate key " + value.key);
            }
        }
        String marker =
                checkpointMarker(
                        identity,
                        canonicalKeys,
                        canonicalKeyValid,
                        canonicalPayloads,
                        canonicalPayloadValid,
                        keyRows.size());
        return new LookupJoinSnapshot(
                canonicalKeys,
                canonicalKeyValid,
                canonicalPayloads,
                canonicalPayloadValid,
                Collections.unmodifiableMap(keyRows),
                marker);
    }

    public int rowCount() {
        return keys.length;
    }

    public int matchableRowCount() {
        return nonNullKeyRows.size();
    }

    public String checkpointMarker() {
        return checkpointMarker;
    }

    public long[] copyKeys() {
        return Arrays.copyOf(keys, keys.length);
    }

    public boolean[] copyKeyValidity() {
        return Arrays.copyOf(keyValid, keyValid.length);
    }

    public long[] copyPayloads() {
        return Arrays.copyOf(payloads, payloads.length);
    }

    public boolean[] copyPayloadValidity() {
        return Arrays.copyOf(payloadValid, payloadValid.length);
    }

    /** Returns null for SQL NULL or misses; matched NULL payload remains distinguishable. */
    public Match lookup(long key, boolean valid) {
        if (!valid) {
            return null;
        }
        Integer row = nonNullKeyRows.get(key);
        if (row == null) {
            return null;
        }
        return new Match(payloads[row], payloadValid[row]);
    }

    /** Validates all union-state replicas and the freshly loaded snapshot. */
    public static void requireCompatibleRestore(
            Iterable<String> restoredMarkers, String loadedMarker) {
        Objects.requireNonNull(restoredMarkers, "restoredMarkers");
        Objects.requireNonNull(loadedMarker, "loadedMarker");
        String expected = null;
        int count = 0;
        for (String marker : restoredMarkers) {
            if (marker == null || marker.isEmpty()) {
                throw new IllegalStateException("Lookup Join restored an empty snapshot marker");
            }
            if (expected == null) {
                expected = marker;
            } else if (!expected.equals(marker)) {
                throw new IllegalStateException(
                        "Lookup Join restored inconsistent snapshot replicas: "
                                + abbreviate(expected)
                                + " != "
                                + abbreviate(marker));
            }
            count++;
        }
        if (count > 0 && !loadedMarker.equals(expected)) {
            throw new IllegalStateException(
                    "Lookup Join snapshot changed across recovery: checkpoint="
                            + abbreviate(expected)
                            + ", loaded="
                            + abbreviate(loadedMarker));
        }
    }

    private static String checkpointMarker(
            String identity,
            long[] keys,
            boolean[] keyValid,
            long[] payloads,
            boolean[] payloadValid,
            int matchableRows) {
        MessageDigest identityDigest = sha256();
        identityDigest.update(identity.getBytes(StandardCharsets.UTF_8));
        MessageDigest contentDigest = sha256();
        ByteBuffer scratch = ByteBuffer.allocate(17).order(ByteOrder.BIG_ENDIAN);
        for (int row = 0; row < keys.length; row++) {
            scratch.clear();
            scratch.put((byte) ((keyValid[row] ? 1 : 0) | (payloadValid[row] ? 2 : 0)));
            scratch.putLong(keyValid[row] ? keys[row] : 0L);
            scratch.putLong(payloadValid[row] ? payloads[row] : 0L);
            contentDigest.update(scratch.array());
        }
        return STATE_VERSION
                + ':'
                + hex(identityDigest.digest())
                + ':'
                + hex(contentDigest.digest())
                + ':'
                + keys.length
                + ':'
                + matchableRows;
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
    }

    private static String hex(byte[] bytes) {
        char[] output = new char[bytes.length * 2];
        char[] alphabet = "0123456789abcdef".toCharArray();
        for (int index = 0; index < bytes.length; index++) {
            int value = bytes[index] & 0xff;
            output[index * 2] = alphabet[value >>> 4];
            output[index * 2 + 1] = alphabet[value & 0x0f];
        }
        return new String(output);
    }

    private static String abbreviate(String value) {
        return value.length() <= 32 ? value : value.substring(0, 32) + "...";
    }

    private static int hashMapCapacity(int rows) {
        if (rows < 3) {
            return 4;
        }
        return (int) Math.min(Integer.MAX_VALUE - 8L, (long) Math.ceil(rows / 0.75D));
    }

    private static void requireParallelArrays(
            long[] keys, boolean[] keyValid, long[] payloads, boolean[] payloadValid) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(keyValid, "keyValid");
        Objects.requireNonNull(payloads, "payloads");
        Objects.requireNonNull(payloadValid, "payloadValid");
        if (keys.length != keyValid.length
                || keys.length != payloads.length
                || keys.length != payloadValid.length) {
            throw new IllegalArgumentException("Lookup Join dimension array lengths differ");
        }
    }

    public static final class Match {
        private final long payload;
        private final boolean payloadValid;

        private Match(long payload, boolean payloadValid) {
            this.payload = payloadValid ? payload : 0L;
            this.payloadValid = payloadValid;
        }

        public long payload() {
            return payload;
        }

        public boolean payloadValid() {
            return payloadValid;
        }
    }

    private static final class DimensionRow {
        private static final Comparator<DimensionRow> CANONICAL_ORDER =
                Comparator.comparing((DimensionRow row) -> row.keyValid)
                        .thenComparingLong(row -> row.key)
                        .thenComparing(row -> row.payloadValid)
                        .thenComparingLong(row -> row.payload);

        private final long key;
        private final boolean keyValid;
        private final long payload;
        private final boolean payloadValid;

        private DimensionRow(long key, boolean keyValid, long payload, boolean payloadValid) {
            this.key = key;
            this.keyValid = keyValid;
            this.payload = payload;
            this.payloadValid = payloadValid;
        }
    }
}
