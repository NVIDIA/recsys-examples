// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * GPU joins for DataStream jobs. This version has one join: {@link #lookupInner}, an INNER join of
 * a probe stream against a bounded, read-only dimension snapshot.
 *
 * <pre>{@code
 * LookupDimension videos = LookupDimension.ofLongPairs("videos-v7", videoIds, authorIds);
 * DataStream<RowData> enriched =
 *     GpuJoins.lookupInner(events, eventType, "video_id", videos)
 *         .dimensionColumns("dim_video_id", "author_id")
 *         .build();
 * }</pre>
 *
 * <p>Output rows are the complete probe row followed by the two dimension columns ({@code BIGINT},
 * {@code BIGINT}); probe rows without a match are dropped (INNER). The output keeps the probe row's
 * timestamp. Probe rows are not repartitioned: every parallel subtask loads the whole dimension.
 *
 * <h2>What the library checks at {@code build()}</h2>
 *
 * <p>These are the checks of Mega-Flink's lookup-join admission, applied to the declared shape.
 * A failed check throws {@link GpuUnsupportedShapeException} (or {@link IllegalArgumentException}
 * for an invalid argument) while the job graph is built; there is no fallback to a CPU lookup.
 *
 * <ul>
 *   <li>INNER join only (LEFT is not offered);
 *   <li>one probe key column, of type {@code BIGINT}; the dimension is {@code ROW<BIGINT key,
 *       BIGINT payload>} (see {@link LookupDimension});
 *   <li>valid batch configuration: {@code batchRows}, {@code gpuMinBatchRows}, {@code maxWaitMs},
 *       {@code maxDimensionRows} positive and {@code gpuMinBatchRows <= batchRows};
 *   <li>the stream's own type information equals the declared {@link RowType} when it is an {@code
 *       InternalTypeInfo}.
 * </ul>
 *
 * <p><b>Probe input must be append-only</b> ({@code RowKind.INSERT}); this is a property of the data
 * and is enforced when records arrive (the operator throws on a retraction or update row). The
 * dimension content contract is documented on {@link LookupDimension}; it is the caller's promise.
 *
 * <p>Batches smaller than {@code gpuMinBatchRows} are answered by an exact JVM lookup in the same
 * operator, and larger ones by the GPU kernel; the output is the same either way.
 */
public final class GpuJoins {
    private GpuJoins() {}

    /**
     * INNER lookup join of {@code probe} against {@code dimension} on a BIGINT probe column.
     *
     * @param probeKeyField name of the BIGINT column of the probe row that is looked up
     */
    public static LookupInnerBuilder lookupInner(
            DataStream<RowData> probe,
            RowType probeType,
            String probeKeyField,
            LookupDimension dimension) {
        return new LookupInnerBuilder(probe, probeType, probeKeyField, dimension);
    }

    /** Builder for {@link #lookupInner}. Not thread-safe. */
    public static final class LookupInnerBuilder {
        private final DataStream<RowData> probe;
        private final RowType probeType;
        private final String probeKeyField;
        private final LookupDimension dimension;
        private String dimensionKeyName = "dim_key";
        private String dimensionPayloadName = "dim_payload";
        private int batchRows = defaultInt(LookupJoinOptions.BATCH_ROWS.defaultValue());
        private int gpuMinBatchRows = defaultInt(LookupJoinOptions.GPU_MIN_BATCH_ROWS.defaultValue());
        private long maxWaitMs = LookupJoinOptions.BATCH_MAX_WAIT_MS.defaultValue();
        private int maxDimensionRows = defaultInt(LookupJoinOptions.MAX_DIMENSION_ROWS.defaultValue());
        private GpuResourceConfig resource = GpuResourceConfig.externalResourceDefaults();
        private String name = "GpuBoundedLookupJoin";

        private LookupInnerBuilder(
                DataStream<RowData> probe,
                RowType probeType,
                String probeKeyField,
                LookupDimension dimension) {
            this.probe = Objects.requireNonNull(probe, "probe");
            this.probeType = Objects.requireNonNull(probeType, "probeType");
            this.probeKeyField = Objects.requireNonNull(probeKeyField, "probeKeyField");
            this.dimension = Objects.requireNonNull(dimension, "dimension");
        }

        private static int defaultInt(Integer value) {
            return value;
        }

        /** Names of the two output columns taken from the dimension (default dim_key, dim_payload). */
        public LookupInnerBuilder dimensionColumns(String keyName, String payloadName) {
            this.dimensionKeyName = Objects.requireNonNull(keyName, "keyName");
            this.dimensionPayloadName = Objects.requireNonNull(payloadName, "payloadName");
            return this;
        }

        /** Most probe rows one batch holds. Default 1,048,576 (Mega-Flink's default). */
        public LookupInnerBuilder batchRows(int rows) {
            this.batchRows = rows;
            return this;
        }

        /**
         * Smallest batch that is sent to the GPU; smaller batches use the exact JVM lookup. Default
         * 1,024 (the value of the benchmark runs). Must not exceed {@code batchRows}.
         */
        public LookupInnerBuilder gpuMinBatchRows(int rows) {
            this.gpuMinBatchRows = rows;
            return this;
        }

        /** Longest a partial batch waits before it is flushed. Default 5 ms. */
        public LookupInnerBuilder maxWaitMs(long millis) {
            this.maxWaitMs = millis;
            return this;
        }

        /**
         * Hard bound on the number of dimension rows; loading more fails the job. Default 1,048,576
         * (Mega-Flink's default).
         */
        public LookupInnerBuilder maxDimensionRows(int rows) {
            this.maxDimensionRows = rows;
            return this;
        }

        /** How the operator finds its GPU. The default is Flink external resource {@code gpu}. */
        public LookupInnerBuilder gpuResource(GpuResourceConfig resource) {
            this.resource = Objects.requireNonNull(resource, "resource");
            return this;
        }

        /** Operator name shown in the Flink UI. */
        public LookupInnerBuilder name(String name) {
            this.name = Objects.requireNonNull(name, "name");
            return this;
        }

        /** Row type of the result: the probe columns, then the two dimension columns. */
        public RowType outputType() {
            return resolve().outputType;
        }

        /**
         * Validates the shape and adds the GPU lookup-join operator to the job graph.
         *
         * @throws GpuUnsupportedShapeException if the shape is outside what the GPU operator supports
         * @throws IllegalArgumentException if an argument is invalid (unknown column, bad config)
         */
        public SingleOutputStreamOperator<RowData> build() {
            Resolved r = resolve();
            GpuBoundedLookupJoinOperator operator =
                    new GpuBoundedLookupJoinOperator(
                            dimension.format(),
                            dimension.identity(),
                            probeType,
                            r.probeKeyOrdinal,
                            LookupJoinKernel.JoinType.INNER,
                            batchRows,
                            gpuMinBatchRows,
                            maxWaitMs,
                            maxDimensionRows);
            operator.configureGpuResource(resource);
            return probe.transform(name, InternalTypeInfo.of(r.outputType), operator);
        }

        // Conditions and messages follow Flink221LookupJoinPlanner.unsupportedReasonWithoutProvider.
        private Resolved resolve() {
            if (probe.getType() instanceof InternalTypeInfo) {
                LogicalType declared = ((InternalTypeInfo<?>) probe.getType()).toLogicalType();
                if (!declared.copy(true).equals(probeType.copy(true))) {
                    throw new IllegalArgumentException(
                            "declared probeType "
                                    + probeType.asSummaryString()
                                    + " differs from the stream's row type "
                                    + declared.asSummaryString());
                }
            }
            int keyOrdinal = probeType.getFieldNames().indexOf(probeKeyField);
            if (keyOrdinal < 0) {
                throw new IllegalArgumentException(
                        "probe key column '"
                                + probeKeyField
                                + "' is not a field of the probe row type "
                                + probeType.getFieldNames());
            }
            if (probeType.getTypeAt(keyOrdinal).getTypeRoot() != LogicalTypeRoot.BIGINT) {
                throw unsupported(
                        "probe lookup key must be BIGINT, but column '"
                                + probeKeyField
                                + "' is "
                                + probeType.getTypeAt(keyOrdinal).asSummaryString());
            }
            if (batchRows <= 0
                    || gpuMinBatchRows <= 0
                    || gpuMinBatchRows > batchRows
                    || maxWaitMs <= 0L
                    || maxDimensionRows <= 0) {
                throw unsupported(
                        "invalid bounded Lookup Join configuration (batchRows="
                                + batchRows
                                + ", gpuMinBatchRows="
                                + gpuMinBatchRows
                                + ", maxWaitMs="
                                + maxWaitMs
                                + ", maxDimensionRows="
                                + maxDimensionRows
                                + ")");
            }
            if (probeType.getFieldNames().contains(dimensionKeyName)
                    || probeType.getFieldNames().contains(dimensionPayloadName)
                    || dimensionKeyName.equals(dimensionPayloadName)) {
                throw new IllegalArgumentException(
                        "dimension output column names '"
                                + dimensionKeyName
                                + "' and '"
                                + dimensionPayloadName
                                + "' must be distinct and must not clash with probe columns "
                                + probeType.getFieldNames());
            }
            List<LogicalType> types = new ArrayList<>(probeType.getChildren());
            List<String> names = new ArrayList<>(probeType.getFieldNames());
            types.add(new BigIntType(true));
            names.add(dimensionKeyName);
            types.add(new BigIntType(true));
            names.add(dimensionPayloadName);
            List<RowType.RowField> fields = new ArrayList<>(types.size());
            for (int i = 0; i < types.size(); i++) {
                fields.add(new RowType.RowField(names.get(i), types.get(i)));
            }
            return new Resolved(keyOrdinal, new RowType(false, fields));
        }

        private static GpuUnsupportedShapeException unsupported(String reason) {
            return new GpuUnsupportedShapeException(
                    "GPU INNER lookup join does not support this shape: " + reason);
        }
    }

    private static final class Resolved {
        final int probeKeyOrdinal;
        final RowType outputType;

        Resolved(int probeKeyOrdinal, RowType outputType) {
            this.probeKeyOrdinal = probeKeyOrdinal;
            this.outputType = outputType;
        }
    }
}
