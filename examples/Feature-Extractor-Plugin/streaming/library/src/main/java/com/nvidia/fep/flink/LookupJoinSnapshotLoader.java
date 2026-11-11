// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.functions.RuntimeContext;
import org.apache.flink.api.common.io.InputFormat;
import org.apache.flink.api.common.io.RichInputFormat;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.io.InputSplit;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.binary.BinaryRowData;
import org.apache.flink.types.RowKind;
import org.apache.flink.util.InstantiationUtil;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Objects;

/** Loads the complete connector-owned bounded snapshot before activating a Lookup Join operator. */
public final class LookupJoinSnapshotLoader {
    private LookupJoinSnapshotLoader() {}

    @SuppressWarnings("unchecked")
    public static LookupJoinSnapshot load(
            InputFormat<RowData, ?> initialFormat,
            String snapshotIdentity,
            int maximumRows,
            RuntimeContext runtimeContext,
            ClassLoader userCodeClassLoader)
            throws Exception {
        Objects.requireNonNull(initialFormat, "initialFormat");
        Objects.requireNonNull(userCodeClassLoader, "userCodeClassLoader");
        if (maximumRows <= 0) {
            throw new IllegalArgumentException("Lookup Join maximum dimension rows must be positive");
        }

        InputFormat<RowData, InputSplit> format =
                (InputFormat<RowData, InputSplit>)
                        InstantiationUtil.clone(initialFormat, userCodeClassLoader);
        MutableRows rows = new MutableRows(Math.min(maximumRows, 1024));
        RichInputFormat<RowData, InputSplit> rich =
                format instanceof RichInputFormat
                        ? (RichInputFormat<RowData, InputSplit>) format
                        : null;
        boolean richOpened = false;
        try {
            format.configure(new Configuration());
            if (rich != null) {
                Objects.requireNonNull(
                        runtimeContext, "Rich snapshot InputFormat requires a RuntimeContext");
                rich.setRuntimeContext(runtimeContext);
                rich.openInputFormat();
                richOpened = true;
            }
            InputSplit[] splits = format.createInputSplits(1);
            if (splits == null || splits.length == 0) {
                throw new IllegalStateException(
                        "Lookup Join snapshot InputFormat returned no input splits");
            }
            Arrays.sort(splits, Comparator.comparingInt(InputSplit::getSplitNumber));
            for (InputSplit split : splits) {
                boolean splitOpened = false;
                try {
                    format.open(split);
                    splitOpened = true;
                    RowData reuse = new BinaryRowData(2);
                    while (!format.reachedEnd()) {
                        RowData next = format.nextRecord(reuse);
                        if (next == null) {
                            break;
                        }
                        if (next.getRowKind() != RowKind.INSERT) {
                            throw new IllegalStateException(
                                    "Lookup Join snapshot must be INSERT-only, got "
                                            + next.getRowKind());
                        }
                        if (next.getArity() != 2) {
                            throw new IllegalStateException(
                                    "Lookup Join snapshot rows must have arity 2, got "
                                            + next.getArity());
                        }
                        if (rows.size == maximumRows) {
                            throw new IllegalStateException(
                                    "Lookup Join dimension exceeds configured maximum "
                                            + maximumRows);
                        }
                        boolean keyValid = !next.isNullAt(0);
                        boolean payloadValid = !next.isNullAt(1);
                        rows.add(
                                keyValid ? next.getLong(0) : 0L,
                                keyValid,
                                payloadValid ? next.getLong(1) : 0L,
                                payloadValid);
                    }
                } finally {
                    if (splitOpened) {
                        format.close();
                    }
                }
            }
        } finally {
            if (richOpened) {
                rich.closeInputFormat();
            }
        }
        return LookupJoinSnapshot.create(
                snapshotIdentity,
                rows.keys(),
                rows.keyValidity(),
                rows.payloads(),
                rows.payloadValidity());
    }

    /** Local semantic-test entry point for a non-rich serializable InputFormat. */
    public static LookupJoinSnapshot loadForTesting(
            InputFormat<RowData, ?> inputFormat, String snapshotIdentity, int maximumRows)
            throws Exception {
        if (inputFormat instanceof RichInputFormat) {
            throw new IllegalArgumentException(
                    "loadForTesting does not support RichInputFormat without a RuntimeContext");
        }
        return load(
                inputFormat,
                snapshotIdentity,
                maximumRows,
                null,
                LookupJoinSnapshotLoader.class.getClassLoader());
    }

    private static final class MutableRows {
        private long[] keys;
        private boolean[] keyValid;
        private long[] payloads;
        private boolean[] payloadValid;
        private int size;

        private MutableRows(int initialCapacity) {
            int capacity = Math.max(1, initialCapacity);
            keys = new long[capacity];
            keyValid = new boolean[capacity];
            payloads = new long[capacity];
            payloadValid = new boolean[capacity];
        }

        private void add(long key, boolean keyIsValid, long payload, boolean payloadIsValid) {
            ensureCapacity(size + 1);
            keys[size] = key;
            keyValid[size] = keyIsValid;
            payloads[size] = payload;
            payloadValid[size] = payloadIsValid;
            size++;
        }

        private void ensureCapacity(int required) {
            if (required <= keys.length) {
                return;
            }
            int next = Math.max(required, Math.multiplyExact(keys.length, 2));
            keys = Arrays.copyOf(keys, next);
            keyValid = Arrays.copyOf(keyValid, next);
            payloads = Arrays.copyOf(payloads, next);
            payloadValid = Arrays.copyOf(payloadValid, next);
        }

        private long[] keys() {
            return Arrays.copyOf(keys, size);
        }

        private boolean[] keyValidity() {
            return Arrays.copyOf(keyValid, size);
        }

        private long[] payloads() {
            return Arrays.copyOf(payloads, size);
        }

        private boolean[] payloadValidity() {
            return Arrays.copyOf(payloadValid, size);
        }
    }
}
