// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.runtime.keyselector.RowDataKeySelector;
import org.apache.flink.table.runtime.operators.window.tvf.slicing.SliceAssigner;
import org.apache.flink.table.runtime.operators.window.tvf.slicing.SliceAssigners;
import org.apache.flink.table.runtime.typeutils.InternalTypeInfo;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DoubleType;
import org.apache.flink.table.types.logical.LocalZonedTimestampType;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.TimestampType;

import java.time.Duration;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Event-time TUMBLE and HOP window aggregation on the GPU, for DataStream jobs.
 *
 * <p>The input is a {@code DataStream<RowData>} with watermarks, plus the Flink {@link RowType}
 * that describes its rows. The result is a {@code DataStream<RowData>} with one row per key and
 * window: the key columns, then the aggregate columns, then the requested {@link WindowProperty}
 * columns (by default {@code window_start} and {@code window_end}). Rows carry the timestamp
 * {@code window_end - 1 ms}.
 *
 * <pre>{@code
 * DataStream<RowData> stats =
 *     GpuWindows.hop(events, eventType, Duration.ofMinutes(30), Duration.ofMinutes(5))
 *         .timeField("ts")
 *         .keyBy("user_id")
 *         .aggregate(Agg.count().as("cnt"), Agg.sum("play_ms").as("play_ms_sum"))
 *         .twoPhase()          // optional, see below
 *         .build();
 * }</pre>
 *
 * <h2>What the library checks at {@code build()} (and {@code outputType()})</h2>
 *
 * <p>All checks run while the job graph is built; a failed check throws {@link
 * GpuUnsupportedShapeException} (or {@link IllegalArgumentException} for plainly invalid arguments
 * such as an unknown column) with a message that names the problem. There is no silent fallback to
 * a CPU operator. The checks are the ones Mega-Flink's planner applied before it replaced a node,
 * split out of its {@code tryCompile} admission code:
 *
 * <ul>
 *   <li>the time column is {@code TIMESTAMP} or {@code TIMESTAMP_LTZ} (event time only; windows are
 *       aligned in UTC);
 *   <li>at least one key column and at least one aggregate;
 *   <li>TUMBLE: size {@code > 0} and {@code abs(offset) < size}; HOP: size and slide {@code > 0}
 *       and no more than 64 windows per input row; durations are whole milliseconds;
 *   <li>two-phase only: size is a multiple of slide, at most 64 slices per window, time column
 *       precision at most 3, key types that the GPU key lanes support;
 *   <li>the key and aggregate column types and the aggregate/type combinations the GPU kernels
 *       accept (for example SUM/AVG need a numeric column, MIN/MAX need the same type in and out);
 *   <li>the stream's own type information equals the declared {@link RowType} when it is an {@code
 *       InternalTypeInfo}.
 * </ul>
 *
 * <p><b>Input must be append-only</b> ({@code RowKind.INSERT}). Whether a stream is append-only is a
 * property of its data, so it cannot be proven at build time; the operators throw on the first
 * retraction or update row. Rows whose time column is NULL are rejected the same way. Rows that
 * arrive after their window was closed by a watermark are dropped (counted in the metric {@code
 * cudfWindowLateRowsDropped}), as in Flink SQL.
 *
 * <h2>One phase or two phases</h2>
 *
 * <p>By default one operator per key group keeps window state and fires windows (Flink SQL's
 * {@code WindowAggregate}). {@link Builder#twoPhase()} instead puts a stateless GPU combiner in front
 * of a keyed merge ({@code LocalWindowAggregate} then {@code GlobalWindowAggregate}); the combiner
 * output is repartitioned by key between the two operators. Which is faster depends on the data; the
 * library does not choose for you.
 *
 * <h2>Resources</h2>
 *
 * <p>The operators acquire a GPU through Flink's external-resource framework (resource name
 * {@code gpu}) unless {@link Builder#gpuResource(GpuResourceConfig)} says otherwise. The native
 * libraries must be loadable on the TaskManagers.
 */
public final class GpuWindows {
    private static final int DEFAULT_BATCH_ROWS = 4096;
    private static final long DEFAULT_MAX_WAIT_MS = 5L;
    private static final ZoneId UTC = ZoneId.of("UTC");

    private GpuWindows() {}

    /** Tumbling event-time window of the given size. */
    public static Builder tumble(DataStream<RowData> input, RowType inputType, Duration size) {
        return new Builder(input, inputType, false, size, size);
    }

    /** Hopping (sliding) event-time window with the given size and slide. */
    public static Builder hop(
            DataStream<RowData> input, RowType inputType, Duration size, Duration slide) {
        return new Builder(input, inputType, true, size, slide);
    }

    /** Builder for one window aggregation. Not thread-safe. */
    public static final class Builder {
        private final DataStream<RowData> input;
        private final RowType inputType;
        private final boolean hop;
        private final Duration size;
        private final Duration slide;
        private String timeField;
        private final List<String> keyFields = new ArrayList<>();
        private final List<Agg> aggs = new ArrayList<>();
        private Duration offset = Duration.ZERO;
        private boolean twoPhase;
        private WindowProperty[] properties = {WindowProperty.START, WindowProperty.END};
        private int batchRows = DEFAULT_BATCH_ROWS;
        private long maxWaitMs = DEFAULT_MAX_WAIT_MS;
        private GpuResourceConfig resource = GpuResourceConfig.externalResourceDefaults();
        private String name;

        private Builder(
                DataStream<RowData> input,
                RowType inputType,
                boolean hop,
                Duration size,
                Duration slide) {
            this.input = Objects.requireNonNull(input, "input");
            this.inputType = Objects.requireNonNull(inputType, "inputType");
            this.hop = hop;
            this.size = Objects.requireNonNull(size, "size");
            this.slide = Objects.requireNonNull(slide, "slide");
        }

        /** The input column that holds the event time ({@code TIMESTAMP(p)} or {@code TIMESTAMP_LTZ(p)}). */
        public Builder timeField(String column) {
            this.timeField = Objects.requireNonNull(column, "column");
            return this;
        }

        /** The key columns (at least one). Rows with equal keys are aggregated together. */
        public Builder keyBy(String... columns) {
            keyFields.clear();
            keyFields.addAll(Arrays.asList(columns));
            return this;
        }

        /** The aggregates (at least one). Output columns appear in this order after the keys. */
        public Builder aggregate(Agg... aggregates) {
            aggs.clear();
            aggs.addAll(Arrays.asList(aggregates));
            return this;
        }

        /** Window offset (TUMBLE: {@code abs(offset) < size}). The default is zero. */
        public Builder offset(Duration offset) {
            this.offset = Objects.requireNonNull(offset, "offset");
            return this;
        }

        /**
         * Use a stateless GPU combiner followed by a keyed merge instead of one keyed operator. In
         * Mega-Flink this local/global path is marked experimental.
         */
        public Builder twoPhase() {
            this.twoPhase = true;
            return this;
        }

        /** Window columns to append to the output. The default is START and END. */
        public Builder windowProperties(WindowProperty... properties) {
            this.properties = properties.clone();
            return this;
        }

        /**
         * Rows the GPU operator collects before it launches a kernel. Default 4096 (the default of
         * Mega-Flink's {@code cudf.flink.batch.rows}). For HOP the native buffer is sized by the
         * number of windows each row can land in.
         */
        public Builder batchRows(int rows) {
            this.batchRows = rows;
            return this;
        }

        /** Longest time a partial batch waits before it is flushed. Default 5 ms. */
        public Builder maxWaitMs(long millis) {
            this.maxWaitMs = millis;
            return this;
        }

        /** How the operators find their GPU. The default is Flink external resource {@code gpu}. */
        public Builder gpuResource(GpuResourceConfig resource) {
            this.resource = Objects.requireNonNull(resource, "resource");
            return this;
        }

        /** Operator name shown in the Flink UI. */
        public Builder name(String name) {
            this.name = Objects.requireNonNull(name, "name");
            return this;
        }

        /** Row type of the result. Runs the same checks as {@link #build()}. */
        public RowType outputType() {
            return resolve().fullOutputType;
        }

        /**
         * Validates the shape and adds the GPU operator(s) to the job graph.
         *
         * @throws GpuUnsupportedShapeException if the shape is outside what the GPU operators support
         * @throws IllegalArgumentException if an argument is invalid (unknown column, bad duration)
         */
        public SingleOutputStreamOperator<RowData> build() {
            Resolved r = resolve();
            String opName = name != null ? name : "GpuWindowAggregate(" + r.windowName + ")";
            InternalTypeInfo<RowData> outputInfo = InternalTypeInfo.of(r.fullOutputType);
            if (!twoPhase) {
                RowDataKeySelector selector = new FieldRowDataKeySelector(inputType, r.keyOrdinals);
                CudfTumblingWindowOperator operator =
                        new CudfTumblingWindowOperator(
                                ExecutionMode.GPU_NATIVE,
                                r.plan,
                                batchRows,
                                maxWaitMs,
                                r.timeOrdinal,
                                r.timePrecision,
                                r.sizeMs,
                                r.slideMs,
                                r.offsetMs,
                                false,
                                r.propertyKinds,
                                r.fullOutputType.getFieldCount(),
                                selector,
                                InternalTypeInfo.of(r.coreOutputType));
                operator.configureGpuResource(resource);
                return input.keyBy(selector, selector.getProducedType())
                        .transform(opName, outputInfo, operator);
            }
            SliceAssigner assigner;
            if (hop) {
                assigner =
                        SliceAssigners.hopping(r.timeOrdinal, UTC, size, slide)
                                .withOffset(offset);
            } else {
                assigner = SliceAssigners.tumbling(r.timeOrdinal, UTC, size).withOffset(offset);
            }
            GpuLocalWindowAggregateOperator local =
                    new GpuLocalWindowAggregateOperator(
                            r.localPlan, assigner, batchRows, maxWaitMs);
            local.configureGpuResource(resource);
            SingleOutputStreamOperator<RowData> partials =
                    input.transform(
                            opName + "-local", InternalTypeInfo.of(r.localOutputType), local);
            int[] partialKeys = new int[r.keyOrdinals.length];
            GpuType[] partialKeyTypes = new GpuType[partialKeys.length];
            for (int i = 0; i < partialKeys.length; i++) {
                partialKeys[i] = i;
                partialKeyTypes[i] = GpuType.from(r.localOutputType.getTypeAt(i)).orElse(null);
            }
            RowDataKeySelector partialSelector =
                    new FieldRowDataKeySelector(r.localOutputType, partialKeys);
            GpuAssistedGlobalWindowAggregateOperator global =
                    new GpuAssistedGlobalWindowAggregateOperator(
                            r.localPlan,
                            r.sizeMs,
                            r.slideMs,
                            r.offsetMs,
                            false,
                            r.timePrecision,
                            r.propertyKinds,
                            r.fullOutputType.getFieldCount(),
                            r.localOutputType.getFieldCount(),
                            partialKeys,
                            partialKeyTypes,
                            partialSelector,
                            InternalTypeInfo.of(r.coreOutputType));
            return partials.keyBy(partialSelector, partialSelector.getProducedType())
                    .transform(opName, outputInfo, global);
        }

        // ------------------------------------------------------------------------------------
        // Validation. Messages and conditions follow Mega-Flink's StreamExecWindowAggregate /
        // StreamExecLocalWindowAggregate / StreamExecGlobalWindowAggregate prechecks and the
        // tryCompile admission code of GpuAggregatePlan, GpuLocalAggregatePlan and
        // GpuLocalWindowAggregatePlan.
        // ------------------------------------------------------------------------------------

        private Resolved resolve() {
            final String window = hop ? "HOP" : "TUMBLE";
            if (input.getType() instanceof InternalTypeInfo) {
                LogicalType declared = ((InternalTypeInfo<?>) input.getType()).toLogicalType();
                if (!declared.copy(true).equals(inputType.copy(true))) {
                    throw new IllegalArgumentException(
                            "declared inputType "
                                    + inputType.asSummaryString()
                                    + " differs from the stream's row type "
                                    + declared.asSummaryString());
                }
            }
            if (timeField == null) {
                throw new IllegalArgumentException("timeField(...) is required");
            }
            if (batchRows <= 0 || maxWaitMs <= 0L) {
                throw new IllegalArgumentException("batchRows and maxWaitMs must be positive");
            }
            Objects.requireNonNull(offset, "offset");
            if (!distinctProperties(properties)) {
                throw new IllegalArgumentException("window properties must not repeat or be null");
            }

            // window parameters
            long sizeMs = wholeMillis("window size", size);
            long slideMs = wholeMillis("window slide", slide);
            long offsetMs = wholeMillis("window offset", offset);
            if (sizeMs <= 0L || slideMs <= 0L) {
                throw unsupported(
                        window,
                        hop
                                ? "HOP size and slide must be positive"
                                : "TUMBLE size must be positive");
            }
            if (!hop && Math.abs(offsetMs) >= sizeMs) {
                throw unsupported(
                        window,
                        "TUMBLE offset must satisfy abs(offset) < size, got size "
                                + sizeMs
                                + "ms and offset "
                                + offsetMs
                                + "ms");
            }
            long windowsPerRow = Math.floorDiv(sizeMs - 1L, slideMs) + 1L;
            if (windowsPerRow > 64L) {
                throw unsupported(window, "HOP produces more than 64 windows per input row");
            }
            if (twoPhase) {
                if (sizeMs % slideMs != 0L) {
                    throw unsupported(
                            window, "two-phase window size must be an integral multiple of slide");
                }
                if (sizeMs / slideMs > 64L) {
                    throw unsupported(window, "two-phase window needs more than 64 slice contributions");
                }
            }

            // time column
            int timeOrdinal = ordinalOf("time column", timeField);
            LogicalType timeType = inputType.getTypeAt(timeOrdinal);
            if (timeType.getTypeRoot() != LogicalTypeRoot.TIMESTAMP_WITHOUT_TIME_ZONE
                    && timeType.getTypeRoot() != LogicalTypeRoot.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
                throw unsupported(
                        window,
                        "only TIMESTAMP/TIMESTAMP_LTZ rowtime is supported, but column '"
                                + timeField
                                + "' is "
                                + timeType.asSummaryString());
            }
            int precision =
                    timeType instanceof TimestampType
                            ? ((TimestampType) timeType).getPrecision()
                            : ((LocalZonedTimestampType) timeType).getPrecision();
            if (twoPhase && precision > 3) {
                throw unsupported(
                        window,
                        "global Window timestamps above millisecond precision are unsupported");
            }

            // keys
            if (keyFields.isEmpty()) {
                throw unsupported(
                        window,
                        "at least one key column is required (global, keyless window aggregation is"
                                + " not supported)");
            }
            int[] keyOrdinals = new int[keyFields.size()];
            Set<String> outputNames = new LinkedHashSet<>();
            for (int i = 0; i < keyOrdinals.length; i++) {
                keyOrdinals[i] = ordinalOf("key column", keyFields.get(i));
                if (!outputNames.add(keyFields.get(i))) {
                    throw new IllegalArgumentException(
                            "key column '" + keyFields.get(i) + "' is listed twice");
                }
            }

            // aggregates
            if (aggs.isEmpty()) {
                throw unsupported(window, "window has no aggregate calls");
            }
            AggregateCallSpec[] calls = new AggregateCallSpec[aggs.size()];
            List<LogicalType> coreTypes = new ArrayList<>();
            List<String> coreNames = new ArrayList<>();
            for (int k : keyOrdinals) {
                coreTypes.add(inputType.getTypeAt(k));
                coreNames.add(inputType.getFieldNames().get(k));
            }
            List<LogicalType> localTypes = new ArrayList<>(coreTypes);
            List<String> localNames = new ArrayList<>(coreNames);
            for (int i = 0; i < calls.length; i++) {
                Agg agg = Objects.requireNonNull(aggs.get(i), "aggregate");
                String outName = agg.outputName();
                if (!outputNames.add(outName)) {
                    throw new IllegalArgumentException(
                            "output column name '" + outName + "' is used more than once");
                }
                List<Integer> args = new ArrayList<>();
                LogicalType argType = null;
                if (agg.column() != null) {
                    int ordinal = ordinalOf("aggregate column", agg.column());
                    args.add(ordinal);
                    argType = inputType.getTypeAt(ordinal);
                }
                calls[i] = new AggregateCallSpec(agg.kind(), args, false, -1);
                LogicalType resultType;
                if (agg.kind() == GpuAggregatePlan.AggregateKind.COUNT) {
                    resultType = new BigIntType(false);
                } else {
                    resultType = argType;
                }
                coreTypes.add(resultType);
                coreNames.add(outName);
                // Flink's flattened local accumulator: AVG = sum + count, the others one field.
                if (agg.kind() == GpuAggregatePlan.AggregateKind.AVG) {
                    GpuType argGpu = GpuType.from(argType).orElse(null);
                    boolean floating = argGpu == GpuType.FLOAT32 || argGpu == GpuType.FLOAT64;
                    localTypes.add(floating ? new DoubleType(true) : new BigIntType(true));
                    localNames.add(outName + "$sum");
                    localTypes.add(new BigIntType(false));
                    localNames.add(outName + "$count");
                } else {
                    localTypes.add(resultType);
                    localNames.add(outName);
                }
            }
            RowType coreOutputType = rowType(coreTypes, coreNames);
            List<LogicalType> fullTypes = new ArrayList<>(coreTypes);
            List<String> fullNames = new ArrayList<>(coreNames);
            int[] kinds = new int[properties.length];
            LogicalType propertyType =
                    timeType instanceof TimestampType
                            ? new TimestampType(false, precision)
                            : new LocalZonedTimestampType(false, precision);
            for (int i = 0; i < properties.length; i++) {
                String columnName = properties[i].columnName();
                if (!outputNames.add(columnName)) {
                    throw new IllegalArgumentException(
                            "output column name '" + columnName + "' is used more than once");
                }
                kinds[i] = properties[i].kind();
                fullTypes.add(propertyType);
                fullNames.add(columnName);
            }
            RowType fullOutputType = rowType(fullTypes, fullNames);

            GpuAggregatePlan plan = null;
            GpuLocalWindowAggregatePlan localPlan = null;
            RowType localOutputType = null;
            if (!twoPhase) {
                GpuAggregatePlan.CompileResult compiled =
                        GpuAggregatePlan.PlanCompiler.tryCompile(
                                keyOrdinals,
                                calls,
                                new boolean[calls.length],
                                false,
                                inputType,
                                coreOutputType);
                if (!compiled.isSupported()) {
                    throw unsupported(window, compiled.reason());
                }
                plan = compiled.plan();
                String kernelReason = DenseGroupbyKernel.unsupportedReason(plan);
                if (kernelReason != null) {
                    throw unsupported(window, kernelReason);
                }
            } else {
                localTypes.add(new BigIntType(false));
                localNames.add("$slice_end");
                localOutputType = rowType(localTypes, localNames);
                GpuLocalWindowAggregatePlan.CompileResult compiled =
                        GpuLocalWindowAggregatePlan.PlanCompiler.tryCompile(
                                keyOrdinals, calls, false, inputType, localOutputType);
                if (!compiled.isSupported()) {
                    throw unsupported(window, compiled.reason());
                }
                localPlan = compiled.plan();
                String kernelReason = DenseGroupbyKernel.unsupportedReason(localPlan.computePlan());
                if (kernelReason != null) {
                    throw unsupported(window, kernelReason);
                }
                for (int i = 0; i < keyOrdinals.length; i++) {
                    if (!GpuType.from(localOutputType.getTypeAt(i)).isPresent()) {
                        throw unsupported(
                                window,
                                "unsupported global Window key type "
                                        + localOutputType.getTypeAt(i).asSummaryString());
                    }
                }
                if (!localPlan.matchesCoreOutput(coreOutputType)) {
                    throw unsupported(
                            window, "global Window output types do not match local accumulators");
                }
            }
            return new Resolved(
                    window,
                    sizeMs,
                    slideMs,
                    offsetMs,
                    timeOrdinal,
                    precision,
                    keyOrdinals,
                    kinds,
                    coreOutputType,
                    fullOutputType,
                    plan,
                    localPlan,
                    localOutputType);
        }

        private int ordinalOf(String what, String column) {
            int ordinal = inputType.getFieldNames().indexOf(column);
            if (ordinal < 0) {
                throw new IllegalArgumentException(
                        what
                                + " '"
                                + column
                                + "' is not a field of the input row type "
                                + inputType.getFieldNames());
            }
            return ordinal;
        }

        private static long wholeMillis(String what, Duration duration) {
            Objects.requireNonNull(duration, what);
            if (duration.getNano() % 1_000_000 != 0) {
                throw new IllegalArgumentException(what + " must be a whole number of milliseconds");
            }
            return duration.toMillis();
        }

        private static boolean distinctProperties(WindowProperty[] properties) {
            if (properties == null) {
                return false;
            }
            Set<WindowProperty> seen = EnumSet.noneOf(WindowProperty.class);
            for (WindowProperty p : properties) {
                if (p == null || !seen.add(p)) {
                    return false;
                }
            }
            return true;
        }

        private static RowType rowType(List<LogicalType> types, List<String> names) {
            List<RowType.RowField> fields = new ArrayList<>(types.size());
            for (int i = 0; i < types.size(); i++) {
                fields.add(new RowType.RowField(names.get(i), types.get(i)));
            }
            return new RowType(false, fields);
        }

        private static GpuUnsupportedShapeException unsupported(String window, String reason) {
            return new GpuUnsupportedShapeException(
                    "GPU " + window + " window aggregate does not support this shape: " + reason);
        }
    }

    private static final class Resolved {
        final String windowName;
        final long sizeMs;
        final long slideMs;
        final long offsetMs;
        final int timeOrdinal;
        final int timePrecision;
        final int[] keyOrdinals;
        final int[] propertyKinds;
        final RowType coreOutputType;
        final RowType fullOutputType;
        final GpuAggregatePlan plan;
        final GpuLocalWindowAggregatePlan localPlan;
        final RowType localOutputType;

        Resolved(
                String windowName,
                long sizeMs,
                long slideMs,
                long offsetMs,
                int timeOrdinal,
                int timePrecision,
                int[] keyOrdinals,
                int[] propertyKinds,
                RowType coreOutputType,
                RowType fullOutputType,
                GpuAggregatePlan plan,
                GpuLocalWindowAggregatePlan localPlan,
                RowType localOutputType) {
            this.windowName = windowName;
            this.sizeMs = sizeMs;
            this.slideMs = slideMs;
            this.offsetMs = offsetMs;
            this.timeOrdinal = timeOrdinal;
            this.timePrecision = timePrecision;
            this.keyOrdinals = keyOrdinals;
            this.propertyKinds = propertyKinds;
            this.coreOutputType = coreOutputType;
            this.fullOutputType = fullOutputType;
            this.plan = plan;
            this.localPlan = localPlan;
            this.localOutputType = localOutputType;
        }
    }
}
