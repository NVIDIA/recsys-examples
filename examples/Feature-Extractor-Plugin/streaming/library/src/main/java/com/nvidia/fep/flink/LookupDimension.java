// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.api.common.io.DefaultInputSplitAssigner;
import org.apache.flink.api.common.io.statistics.BaseStatistics;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.io.InputSplitAssigner;
import org.apache.flink.api.common.io.InputFormat;
import org.apache.flink.core.io.GenericInputSplit;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;

import java.io.IOException;
import java.io.Serializable;
import java.util.Objects;

/**
 * The bounded dimension table of a {@link GpuJoins#lookupInner} join: a complete, read-only
 * snapshot that is loaded once per parallel subtask when the operator opens.
 *
 * <p>Contract (the join does not and cannot verify content-level promises for you):
 *
 * <ul>
 *   <li>the rows have exactly two columns, {@code BIGINT} key and {@code BIGINT} payload, both
 *       nullable; a NULL key never matches;
 *   <li>non-NULL keys are unique;
 *   <li>every invocation enumerates the same complete table; the table is bounded (see {@link
 *       GpuJoins.LookupInnerBuilder#maxDimensionRows(int)});
 *   <li>the rows are {@code RowKind.INSERT};
 *   <li>the identity string is non-blank and changes whenever the content changes (for example a
 *       content-addressed manifest id). It is stored in checkpoints together with a content digest;
 *       a restore refuses to start if the reloaded content differs.
 * </ul>
 */
public final class LookupDimension implements Serializable {
    private static final long serialVersionUID = 1L;

    private final InputFormat<RowData, ?> format;
    private final String identity;

    private LookupDimension(InputFormat<RowData, ?> format, String identity) {
        this.format = Objects.requireNonNull(format, "format");
        this.identity = Objects.requireNonNull(identity, "snapshotIdentity").trim();
        if (this.identity.isEmpty()) {
            throw new IllegalArgumentException("Lookup Join snapshot identity must be non-empty");
        }
    }

    /**
     * A snapshot read through a serializable Flink {@link InputFormat} that produces two-column
     * {@code RowData} rows (BIGINT key, BIGINT payload). The format is cloned and read on every
     * subtask.
     */
    public static LookupDimension of(InputFormat<RowData, ?> format, String snapshotIdentity) {
        return new LookupDimension(format, snapshotIdentity);
    }

    /** An in-memory snapshot: {@code payloads[i]} belongs to {@code keys[i]}. Both arrays are copied. */
    public static LookupDimension ofLongPairs(
            String snapshotIdentity, long[] keys, long[] payloads) {
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(payloads, "payloads");
        if (keys.length != payloads.length) {
            throw new IllegalArgumentException("keys and payloads must have the same length");
        }
        return new LookupDimension(
                new LongPairInputFormat(keys.clone(), payloads.clone()), snapshotIdentity);
    }

    InputFormat<RowData, ?> format() {
        return format;
    }

    String identity() {
        return identity;
    }

    private static final class LongPairInputFormat implements InputFormat<RowData, GenericInputSplit> {
        private static final long serialVersionUID = 1L;

        private final long[] keys;
        private final long[] payloads;
        private transient int next;

        private LongPairInputFormat(long[] keys, long[] payloads) {
            this.keys = keys;
            this.payloads = payloads;
        }

        @Override
        public void configure(Configuration parameters) {}

        @Override
        public BaseStatistics getStatistics(BaseStatistics cachedStatistics) {
            return null;
        }

        @Override
        public GenericInputSplit[] createInputSplits(int numSplits) {
            return new GenericInputSplit[] {new GenericInputSplit(0, 1)};
        }

        @Override
        public InputSplitAssigner getInputSplitAssigner(GenericInputSplit[] splits) {
            return new DefaultInputSplitAssigner(splits);
        }

        @Override
        public void open(GenericInputSplit split) throws IOException {
            next = 0;
        }

        @Override
        public void close() {}

        @Override
        public boolean reachedEnd() {
            return next >= keys.length;
        }

        @Override
        public RowData nextRecord(RowData reuse) {
            GenericRowData row = new GenericRowData(2);
            row.setField(0, keys[next]);
            row.setField(1, payloads[next]);
            next++;
            return row;
        }
    }
}
