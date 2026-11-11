// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.data.writer.BinaryRowWriter;
import org.apache.flink.table.types.logical.RowType;


import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** A validated, serializable subset of a Flink streaming GROUP BY plan. */
public final class GpuAggregatePlan implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int inputArity;
    private final int[] inputOrdinals;
    private final GpuType[] inputTypes;
    private final GpuDecimalType[] inputDecimalTypes;
    private final int[] groupingInputOrdinals;
    private final int[] groupingPackedOrdinals;
    private final GpuType[] groupingTypes;
    private final GpuDecimalType[] groupingDecimalTypes;
    private final AggregateSpec[] aggregates;
    private final GpuType[] outputTypes;
    private final GpuDecimalType[] outputDecimalTypes;
    private final DistinctSpec[] distinctSpecs;
    private final boolean retractInput;

    private GpuAggregatePlan(
            int inputArity,
            int[] inputOrdinals,
            GpuType[] inputTypes,
            int[] groupingInputOrdinals,
            int[] groupingPackedOrdinals,
            GpuType[] groupingTypes,
            AggregateSpec[] aggregates,
            GpuType[] outputTypes,
            DistinctSpec[] distinctSpecs,
            boolean retractInput) {
        this(
                inputArity,
                inputOrdinals,
                inputTypes,
                new GpuDecimalType[inputTypes.length],
                groupingInputOrdinals,
                groupingPackedOrdinals,
                groupingTypes,
                new GpuDecimalType[groupingTypes.length],
                aggregates,
                outputTypes,
                new GpuDecimalType[outputTypes.length],
                distinctSpecs,
                retractInput);
    }

    private GpuAggregatePlan(
            int inputArity,
            int[] inputOrdinals,
            GpuType[] inputTypes,
            GpuDecimalType[] inputDecimalTypes,
            int[] groupingInputOrdinals,
            int[] groupingPackedOrdinals,
            GpuType[] groupingTypes,
            GpuDecimalType[] groupingDecimalTypes,
            AggregateSpec[] aggregates,
            GpuType[] outputTypes,
            GpuDecimalType[] outputDecimalTypes,
            DistinctSpec[] distinctSpecs,
            boolean retractInput) {
        this.inputArity = inputArity;
        this.inputOrdinals = inputOrdinals;
        this.inputTypes = inputTypes;
        this.inputDecimalTypes = inputDecimalTypes;
        this.groupingInputOrdinals = groupingInputOrdinals;
        this.groupingPackedOrdinals = groupingPackedOrdinals;
        this.groupingTypes = groupingTypes;
        this.groupingDecimalTypes = groupingDecimalTypes;
        this.aggregates = aggregates;
        this.outputTypes = outputTypes;
        this.outputDecimalTypes = outputDecimalTypes;
        this.distinctSpecs = distinctSpecs;
        this.retractInput = retractInput;
    }

    /** Binary/test-fixture compatibility for the former single-DISTINCT plan shape. */
    private GpuAggregatePlan(
            int inputArity,
            int[] inputOrdinals,
            GpuType[] inputTypes,
            int[] groupingInputOrdinals,
            int[] groupingPackedOrdinals,
            GpuType[] groupingTypes,
            AggregateSpec[] aggregates,
            GpuType[] outputTypes,
            int distinctInputOrdinal,
            GpuType distinctInputType,
            boolean retractInput) {
        this(
                inputArity,
                inputOrdinals,
                inputTypes,
                groupingInputOrdinals,
                groupingPackedOrdinals,
                groupingTypes,
                aggregates,
                outputTypes,
                legacyDistinctSpecs(
                        aggregates, distinctInputOrdinal, distinctInputType),
                retractInput);
    }

    private static DistinctSpec[] legacyDistinctSpecs(
            AggregateSpec[] aggregates, int inputOrdinal, GpuType inputType) {
        if (inputOrdinal < 0) {
            return new DistinctSpec[0];
        }
        int aggregateIndex = -1;
        for (int index = 0; index < aggregates.length; index++) {
            if (aggregates[index].distinct) {
                aggregateIndex = index;
                break;
            }
        }
        if (aggregateIndex < 0 || inputType == null) {
            throw new IllegalArgumentException("invalid legacy DISTINCT plan shape");
        }
        return new DistinctSpec[] {
            new DistinctSpec(inputOrdinal, inputType, aggregateIndex)
        };
    }

    /** Package-local keyed BIGINT prefix-SUM fixture for the isolated opponent harness. */
    static GpuAggregatePlan longPrefixSumReferencePlan() {
        return new GpuAggregatePlan(
                2,
                new int[] {0, 1},
                new GpuType[] {GpuType.INT64, GpuType.INT64},
                new int[] {0},
                new int[] {0},
                new GpuType[] {GpuType.INT64},
                new AggregateSpec[] {
                    new AggregateSpec(
                            AggregateKind.SUM,
                            1,
                            -1,
                            GpuType.INT64,
                            GpuType.INT64,
                            false,
                            -1)
                },
                new GpuType[] {GpuType.INT64, GpuType.INT64},
                new DistinctSpec[0],
                false);
    }

    /** Package-local full OVER fixture aligned with the locked Velox Window benchmark. */
    static GpuAggregatePlan overReferencePlan() {
        GpuType[] inputTypes = {
            GpuType.INT64, GpuType.INT64, GpuType.FLOAT64, GpuType.INT32, GpuType.INT64
        };
        AggregateSpec[] specs = {
            new AggregateSpec(AggregateKind.COUNT, -1, -1, null, GpuType.INT64, false),
            new AggregateSpec(
                    AggregateKind.COUNT, 2, -1, GpuType.FLOAT64, GpuType.INT64, false),
            new AggregateSpec(
                    AggregateKind.SUM, 2, -1, GpuType.FLOAT64, GpuType.FLOAT64, false),
            new AggregateSpec(
                    AggregateKind.AVG, 2, -1, GpuType.FLOAT64, GpuType.FLOAT64, false),
            new AggregateSpec(
                    AggregateKind.MIN, 2, -1, GpuType.FLOAT64, GpuType.FLOAT64, false),
            new AggregateSpec(
                    AggregateKind.MAX, 2, -1, GpuType.FLOAT64, GpuType.FLOAT64, false),
            new AggregateSpec(
                    AggregateKind.SUM, 3, -1, GpuType.INT32, GpuType.INT64, false),
            new AggregateSpec(
                    AggregateKind.MIN, 3, -1, GpuType.INT32, GpuType.INT32, false),
            new AggregateSpec(
                    AggregateKind.MAX, 3, -1, GpuType.INT32, GpuType.INT32, false),
            new AggregateSpec(
                    AggregateKind.MAX, 0, -1, GpuType.INT64, GpuType.INT64, false),
            new AggregateSpec(
                    AggregateKind.MAX, 4, -1, GpuType.INT64, GpuType.INT64, false)
        };
        GpuType[] outputTypes = new GpuType[1 + specs.length];
        outputTypes[0] = GpuType.INT64;
        for (int index = 0; index < specs.length; index++) {
            outputTypes[index + 1] = specs[index].resultType;
        }
        return new GpuAggregatePlan(
                5,
                new int[] {0, 1, 2, 3, 4},
                inputTypes,
                new int[] {1},
                new int[] {1},
                new GpuType[] {GpuType.INT64},
                specs,
                outputTypes,
                new DistinctSpec[0],
                false);
    }

    /** Package-local mixed COUNT/SUM/COUNT(DISTINCT) fixture for the aligned Velox harness. */
    static GpuAggregatePlan mixedDistinctReferencePlan() {
        return new GpuAggregatePlan(
                3,
                new int[] {0, 1, 2},
                new GpuType[] {GpuType.INT64, GpuType.INT32, GpuType.FLOAT64},
                new int[] {0},
                new int[] {0},
                new GpuType[] {GpuType.INT64},
                new AggregateSpec[] {
                    new AggregateSpec(
                            AggregateKind.COUNT,
                            -1,
                            -1,
                            null,
                            GpuType.INT64,
                            false,
                            -1),
                    new AggregateSpec(
                            AggregateKind.SUM,
                            2,
                            -1,
                            GpuType.FLOAT64,
                            GpuType.FLOAT64,
                            false,
                            -1),
                    new AggregateSpec(
                            AggregateKind.COUNT,
                            -1,
                            distinctFilterOrdinal(0),
                            GpuType.INT32,
                            GpuType.INT64,
                            true,
                            0),
                    new AggregateSpec(
                            AggregateKind.COUNT,
                            -1,
                            distinctFilterOrdinal(1),
                            GpuType.FLOAT64,
                            GpuType.INT64,
                            true,
                            1)
                },
                new GpuType[] {
                    GpuType.INT64,
                    GpuType.INT64,
                    GpuType.FLOAT64,
                    GpuType.INT64,
                    GpuType.INT64
                },
                new DistinctSpec[] {
                    new DistinctSpec(1, GpuType.INT32, 2),
                    new DistinctSpec(2, GpuType.FLOAT64, 3)
                },
                false);
    }

    /** Package-local primitive aggregate fixture aligned with the locked Velox benchmark. */
    static GpuAggregatePlan primitiveAggregateReferencePlan(GpuType valueType) {
        boolean numeric = valueType.isNumeric();
        boolean average = numeric && valueType != GpuType.INT8;
        List<AggregateSpec> specs = new ArrayList<>();
        List<GpuType> outputs = new ArrayList<>();
        outputs.add(GpuType.INT64);
        specs.add(
                new AggregateSpec(
                        AggregateKind.COUNT,
                        1,
                        -1,
                        valueType,
                        GpuType.INT64,
                        false,
                        -1));
        outputs.add(GpuType.INT64);
        if (numeric) {
            specs.add(
                    new AggregateSpec(
                            AggregateKind.SUM,
                            1,
                            -1,
                            valueType,
                            valueType,
                            false,
                            -1));
            outputs.add(valueType);
            if (average) {
                specs.add(
                        new AggregateSpec(
                                AggregateKind.AVG,
                                1,
                                -1,
                                valueType,
                                valueType,
                                false,
                                -1));
                outputs.add(valueType);
            }
        }
        specs.add(
                new AggregateSpec(
                        AggregateKind.MIN,
                        1,
                        -1,
                        valueType,
                        valueType,
                        false,
                        -1));
        outputs.add(valueType);
        specs.add(
                new AggregateSpec(
                        AggregateKind.MAX,
                        1,
                        -1,
                        valueType,
                        valueType,
                        false,
                        -1));
        outputs.add(valueType);
        return new GpuAggregatePlan(
                2,
                new int[] {0, 1},
                new GpuType[] {GpuType.INT64, valueType},
                new int[] {0},
                new int[] {0},
                new GpuType[] {GpuType.INT64},
                specs.toArray(new AggregateSpec[0]),
                outputs.toArray(new GpuType[0]),
                new DistinctSpec[0],
                false);
    }

    /** Package-local Filter -> Project -> GroupBy fixture aligned with the Velox plan. */
    static GpuAggregatePlan filterGroupByReferencePlan() {
        AggregateSpec[] specs = {
            new AggregateSpec(
                    AggregateKind.COUNT, -1, -1, null, GpuType.INT64, false, -1),
            new AggregateSpec(
                    AggregateKind.COUNT,
                    1,
                    -1,
                    GpuType.FLOAT64,
                    GpuType.INT64,
                    false,
                    -1),
            new AggregateSpec(
                    AggregateKind.SUM,
                    1,
                    -1,
                    GpuType.FLOAT64,
                    GpuType.FLOAT64,
                    false,
                    -1),
            new AggregateSpec(
                    AggregateKind.COUNT,
                    -1,
                    2,
                    null,
                    GpuType.INT64,
                    false,
                    -1),
            new AggregateSpec(
                    AggregateKind.SUM,
                    3,
                    -1,
                    GpuType.INT32,
                    GpuType.INT64,
                    false,
                    -1)
        };
        return new GpuAggregatePlan(
                4,
                new int[] {0, 1, 2, 3},
                new GpuType[] {
                    GpuType.INT64, GpuType.FLOAT64, GpuType.BOOLEAN, GpuType.INT32
                },
                new int[] {0},
                new int[] {0},
                new GpuType[] {GpuType.INT64},
                specs,
                new GpuType[] {
                    GpuType.INT64,
                    GpuType.INT64,
                    GpuType.INT64,
                    GpuType.FLOAT64,
                    GpuType.INT64,
                    GpuType.INT64
                },
                new DistinctSpec[0],
                false);
    }

    /** Shape-admission compiler kept in a separate nested class so runtime deserialization is pure. */
    static final class PlanCompiler {
        private PlanCompiler() {}

    static CompileResult tryCompile(
            int[] grouping,
            AggregateCallSpec[] calls,
            boolean[] callNeedsRetraction,
            boolean needRetraction,
            RowType inputType,
            RowType outputType) {
        try {
            if (calls.length != callNeedsRetraction.length) {
                return CompileResult.unsupported("aggregate retraction metadata length mismatch");
            }
            boolean retractInput = needRetraction;
            for (boolean callNeedsRetract : callNeedsRetraction) {
                retractInput |= callNeedsRetract;
            }
            if (calls.length == 0) {
                if (grouping.length == 0) {
                    return CompileResult.unsupported(
                            "aggregate plan has neither grouping keys nor aggregate calls");
                }
                if (retractInput) {
                    return CompileResult.unsupported(
                            "streaming DISTINCT currently requires INSERT-only input");
                }
            }
            if (outputType.getFieldCount() != grouping.length + calls.length) {
                return CompileResult.unsupported(
                        "unexpected aggregate output arity " + outputType.getFieldCount());
            }

            LinkedHashMap<Integer, GpuType> referenced = new LinkedHashMap<>();
            LinkedHashMap<Integer, GpuDecimalType> referencedDecimals = new LinkedHashMap<>();
            int[] groupPacked = new int[grouping.length];
            GpuType[] groupTypes = new GpuType[grouping.length];
            GpuDecimalType[] groupDecimals = new GpuDecimalType[grouping.length];
            GpuType[] allOutputTypes = new GpuType[outputType.getFieldCount()];
            GpuDecimalType[] allOutputDecimals =
                    new GpuDecimalType[outputType.getFieldCount()];
            for (int i = 0; i < grouping.length; i++) {
                int ordinal = grouping[i];
                GpuType type = requireInputKeyType(inputType, ordinal, "GROUP BY key");
                GpuType outputGpuType =
                        requireLogicalKeyType(outputType, i, "GROUP BY output");
                GpuDecimalType decimal = decimalType(inputType, ordinal);
                GpuDecimalType outputDecimal = decimalType(outputType, i);
                if (type != outputGpuType || !Objects.equals(decimal, outputDecimal)) {
                    throw new UnsupportedAggregate(
                            "GROUP BY key type changed from "
                                    + logicalTypeName(type, decimal)
                                    + " to "
                                    + logicalTypeName(outputGpuType, outputDecimal));
                }
                referenced.putIfAbsent(ordinal, type);
                referencedDecimals.putIfAbsent(ordinal, decimal);
                groupTypes[i] = type;
                groupDecimals[i] = decimal;
                allOutputTypes[i] = outputGpuType;
                allOutputDecimals[i] = outputDecimal;
            }

            List<PendingSpec> pending = new ArrayList<>(calls.length);
            List<DistinctSpec> distinctSpecs = new ArrayList<>();
            for (int i = 0; i < calls.length; i++) {
                AggregateCallSpec call = calls[i];
                boolean distinctCount = call.isDistinct();
                if (retractInput && distinctCount) {
                    throw new UnsupportedAggregate(
                            "retract input does not yet support DISTINCT aggregates");
                }
                if (call.isApproximate()) {
                    throw new UnsupportedAggregate(call + " uses approximate semantics");
                }
                if (distinctCount
                        && (call.getKind() != AggregateKind.COUNT
                                || call.getArgList().size() != 1
                                || call.hasFilter())) {
                    throw new UnsupportedAggregate(
                            "DISTINCT currently supports unfiltered COUNT(DISTINCT column) calls");
                }
                GpuType resultType =
                        requireLogicalType(outputType, grouping.length + i, "aggregate output");
                allOutputTypes[grouping.length + i] = resultType;
                GpuAggregatePlan.AggregateKind sqlKind = call.getKind();
                AggregateKind kind;
                int inputOrdinal = -1;
                int filterOrdinal = -1;
                int distinctIndex = -1;
                GpuType argumentType = null;
                switch (sqlKind.name()) {
                    case "COUNT":
                        kind = AggregateKind.COUNT;
                        if (resultType != GpuType.INT64) {
                            throw new UnsupportedAggregate("COUNT output is not BIGINT");
                        }
                        if (call.getArgList().size() > 1) {
                            throw new UnsupportedAggregate("COUNT with multiple arguments is not supported");
                        }
                        if (!call.getArgList().isEmpty()) {
                            inputOrdinal = call.getArgList().get(0);
                            argumentType =
                                    distinctCount
                                            ? requireInputKeyType(
                                                    inputType, inputOrdinal, "COUNT DISTINCT argument")
                                            : requireInputType(
                                                    inputType, inputOrdinal, "COUNT argument");
                            if (distinctCount) {
                                if (distinctSpecs.size() >= 3) {
                                    throw new UnsupportedAggregate(
                                            "at most three COUNT(DISTINCT) calls are supported per aggregate");
                                }
                                distinctIndex = distinctSpecs.size();
                                distinctSpecs.add(
                                        new DistinctSpec(
                                                inputOrdinal,
                                                argumentType,
                                                decimalType(inputType, inputOrdinal),
                                                i));
                                // The operator's checkpointed distinct state removes duplicates and
                                // NULLs. CUDA sees a synthetic per-call filter and counts accepted
                                // rows as COUNT(*), without a type-specific STRING/BOOLEAN/TIMESTAMP
                                // value lane.
                                inputOrdinal = -1;
                            } else {
                                referenced.putIfAbsent(inputOrdinal, argumentType);
                                referencedDecimals.putIfAbsent(inputOrdinal, null);
                            }
                        }
                        break;
                    case "SUM":
                        kind = AggregateKind.SUM;
                        inputOrdinal = requireOneArgument(call);
                        argumentType = requireInputType(inputType, inputOrdinal, "SUM argument");
                        if (!argumentType.isNumeric() || !resultType.isNumeric()) {
                            throw new UnsupportedAggregate("SUM requires a supported numeric type");
                        }
                        referenced.putIfAbsent(inputOrdinal, argumentType);
                        referencedDecimals.putIfAbsent(inputOrdinal, null);
                        break;
                    case "SUM0":
                        kind = AggregateKind.SUM0;
                        inputOrdinal = requireOneArgument(call);
                        argumentType = requireInputType(inputType, inputOrdinal, "SUM0 argument");
                        if (!argumentType.isNumeric() || !resultType.isNumeric()) {
                            throw new UnsupportedAggregate("SUM0 requires a supported numeric type");
                        }
                        referenced.putIfAbsent(inputOrdinal, argumentType);
                        referencedDecimals.putIfAbsent(inputOrdinal, null);
                        break;
                    case "AVG":
                        kind = AggregateKind.AVG;
                        inputOrdinal = requireOneArgument(call);
                        argumentType = requireInputType(inputType, inputOrdinal, "AVG argument");
                        requireAvgTypes(argumentType, resultType);
                        referenced.putIfAbsent(inputOrdinal, argumentType);
                        referencedDecimals.putIfAbsent(inputOrdinal, null);
                        break;
                    case "MIN":
                        kind = AggregateKind.MIN;
                        inputOrdinal = requireOneArgument(call);
                        argumentType = requireInputType(inputType, inputOrdinal, "MIN argument");
                        requireMinMaxTypes(argumentType, resultType, "MIN");
                        referenced.putIfAbsent(inputOrdinal, argumentType);
                        referencedDecimals.putIfAbsent(inputOrdinal, null);
                        break;
                    case "MAX":
                        kind = AggregateKind.MAX;
                        inputOrdinal = requireOneArgument(call);
                        argumentType = requireInputType(inputType, inputOrdinal, "MAX argument");
                        requireMinMaxTypes(argumentType, resultType, "MAX");
                        referenced.putIfAbsent(inputOrdinal, argumentType);
                        referencedDecimals.putIfAbsent(inputOrdinal, null);
                        break;
                    default:
                        throw new UnsupportedAggregate(
                                "aggregate function " + call.getName() + " is not supported");
                }
                if (call.hasFilter()) {
                    filterOrdinal = call.filterArg();
                    GpuType filterType =
                            requireInputType(inputType, filterOrdinal, "aggregate FILTER");
                    if (filterType != GpuType.BOOLEAN) {
                        throw new UnsupportedAggregate("aggregate FILTER is not BOOLEAN");
                    }
                    referenced.putIfAbsent(filterOrdinal, filterType);
                    referencedDecimals.putIfAbsent(filterOrdinal, null);
                }
                if (retractInput) {
                    if (kind == AggregateKind.MIN || kind == AggregateKind.MAX) {
                        throw new UnsupportedAggregate(
                                "retract input does not yet support MIN/MAX multiset state");
                    }
                    if ((kind == AggregateKind.SUM
                                    || kind == AggregateKind.SUM0
                                    || kind == AggregateKind.AVG)
                            && call.hasFilter()) {
                        throw new UnsupportedAggregate(
                                "retract SUM/SUM0/AVG currently does not support FILTER");
                    }
                }
                pending.add(
                        new PendingSpec(
                                kind,
                                inputOrdinal,
                                filterOrdinal,
                                argumentType,
                                resultType,
                                distinctCount,
                                distinctIndex));
            }

            int[] inputOrdinals = new int[referenced.size()];
            GpuType[] inputTypes = new GpuType[referenced.size()];
            GpuDecimalType[] inputDecimals = new GpuDecimalType[referenced.size()];
            Map<Integer, Integer> packedByInput = new LinkedHashMap<>();
            int packed = 0;
            for (Map.Entry<Integer, GpuType> entry : referenced.entrySet()) {
                inputOrdinals[packed] = entry.getKey();
                inputTypes[packed] = entry.getValue();
                inputDecimals[packed] = referencedDecimals.get(entry.getKey());
                packedByInput.put(entry.getKey(), packed);
                packed++;
            }
            for (int i = 0; i < grouping.length; i++) {
                groupPacked[i] = packedByInput.get(grouping[i]);
            }
            AggregateSpec[] specs = new AggregateSpec[pending.size()];
            for (int i = 0; i < specs.length; i++) {
                PendingSpec spec = pending.get(i);
                specs[i] =
                        new AggregateSpec(
                                spec.kind,
                                spec.inputOrdinal < 0
                                        ? -1
                                        : packedByInput.get(spec.inputOrdinal),
                                spec.filterOrdinal < 0
                                        ? (spec.distinctIndex < 0
                                                ? -1
                                                : distinctFilterOrdinal(spec.distinctIndex))
                                        : packedByInput.get(spec.filterOrdinal),
                                spec.argumentType,
                                spec.resultType,
                                spec.distinct,
                                spec.distinctIndex);
            }

            return CompileResult.supported(
                    new GpuAggregatePlan(
                            inputType.getFieldCount(),
                            inputOrdinals,
                            inputTypes,
                            inputDecimals,
                            grouping.clone(),
                            groupPacked,
                            groupTypes,
                            groupDecimals,
                            specs,
                            allOutputTypes,
                            allOutputDecimals,
                            distinctSpecs.toArray(new DistinctSpec[0]),
                            retractInput));
        } catch (UnsupportedAggregate unsupported) {
            return CompileResult.unsupported(unsupported.getMessage());
        } catch (RuntimeException malformed) {
            return CompileResult.unsupported(
                    "could not compile aggregate: " + malformed.getMessage());
        }
    }

    private static int requireOneArgument(AggregateCallSpec call) {
        if (call.getArgList().size() != 1) {
            throw new UnsupportedAggregate(call + " requires exactly one input argument");
        }
        return call.getArgList().get(0);
    }

    private static GpuType requireInputType(RowType inputType, int ordinal, String label) {
        if (ordinal < 0 || ordinal >= inputType.getFieldCount()) {
            throw new UnsupportedAggregate(label + " references invalid input $" + ordinal);
        }
        return GpuType.from(inputType.getTypeAt(ordinal))
                .orElseThrow(
                        () ->
                                new UnsupportedAggregate(
                                        label
                                                + " type "
                                                + inputType.getTypeAt(ordinal).asSummaryString()
                                                + " is not supported"));
    }

    private static GpuType requireInputKeyType(RowType inputType, int ordinal, String label) {
        if (ordinal < 0 || ordinal >= inputType.getFieldCount()) {
            throw new UnsupportedAggregate(label + " references invalid input $" + ordinal);
        }
        return GpuType.fromIncludingDecimal(inputType.getTypeAt(ordinal))
                .orElseThrow(
                        () ->
                                new UnsupportedAggregate(
                                        label
                                                + " type "
                                                + inputType.getTypeAt(ordinal).asSummaryString()
                                                + " is not supported"));
    }

    private static GpuType requireLogicalType(RowType type, int ordinal, String label) {
        return GpuType.from(type.getTypeAt(ordinal))
                .orElseThrow(
                        () ->
                                new UnsupportedAggregate(
                                        label
                                                + " type "
                                                + type.getTypeAt(ordinal).asSummaryString()
                                                + " is not supported"));
    }

    private static GpuType requireLogicalKeyType(RowType type, int ordinal, String label) {
        return GpuType.fromIncludingDecimal(type.getTypeAt(ordinal))
                .orElseThrow(
                        () ->
                                new UnsupportedAggregate(
                                        label
                                                + " type "
                                                + type.getTypeAt(ordinal).asSummaryString()
                                                + " is not supported"));
    }

    private static GpuDecimalType decimalType(RowType rowType, int ordinal) {
        GpuType type = GpuType.fromIncludingDecimal(rowType.getTypeAt(ordinal)).orElse(null);
        return type != null && type.isDecimal()
                ? GpuDecimalType.from(rowType.getTypeAt(ordinal))
                : null;
    }

    private static String logicalTypeName(GpuType type, GpuDecimalType decimal) {
        return decimal == null ? String.valueOf(type) : decimal.toString();
    }

    private static void requireMinMaxTypes(GpuType input, GpuType output, String name) {
        if (input != output || input == GpuType.BOOLEAN) {
            throw new UnsupportedAggregate(
                    name + " needs the same supported numeric/string input and output type");
        }
    }

    private static void requireAvgTypes(GpuType input, GpuType output) {
        if (!input.isNumeric() || input != output) {
            throw new UnsupportedAggregate(
                    "AVG requires matching TINYINT, SMALLINT, INT, BIGINT, FLOAT, or DOUBLE input/output");
        }
    }

    }

    int inputArity() {
        return inputArity;
    }

    GpuAggregatePlan remapInput(int newInputArity, int[] outputToInput) {
        if (newInputArity <= 0 || outputToInput.length != inputArity) {
            throw new IllegalArgumentException("invalid Aggregate input projection mapping");
        }
        int[] remappedInputs = remapOrdinals(inputOrdinals, outputToInput, newInputArity);
        int[] remappedGroups =
                remapOrdinals(groupingInputOrdinals, outputToInput, newInputArity);
        DistinctSpec[] remappedDistinct = new DistinctSpec[distinctSpecs.length];
        for (int index = 0; index < remappedDistinct.length; index++) {
            DistinctSpec spec = distinctSpecs[index];
            remappedDistinct[index] =
                    new DistinctSpec(
                            remapOrdinal(spec.inputOrdinal, outputToInput, newInputArity),
                            spec.inputType,
                            spec.decimalType,
                            spec.aggregateIndex);
        }
        return new GpuAggregatePlan(
                newInputArity,
                remappedInputs,
                inputTypes.clone(),
                inputDecimalTypes.clone(),
                remappedGroups,
                groupingPackedOrdinals.clone(),
                groupingTypes.clone(),
                groupingDecimalTypes.clone(),
                aggregates.clone(),
                outputTypes.clone(),
                outputDecimalTypes.clone(),
                remappedDistinct,
                retractInput);
    }

    private static int[] remapOrdinals(
            int[] ordinals, int[] outputToInput, int newInputArity) {
        int[] remapped = new int[ordinals.length];
        for (int index = 0; index < ordinals.length; index++) {
            remapped[index] = remapOrdinal(ordinals[index], outputToInput, newInputArity);
        }
        return remapped;
    }

    private static int remapOrdinal(
            int ordinal, int[] outputToInput, int newInputArity) {
        if (ordinal < 0 || ordinal >= outputToInput.length) {
            throw new IllegalArgumentException("Aggregate projection references invalid output");
        }
        int remapped = outputToInput[ordinal];
        if (remapped < 0 || remapped >= newInputArity) {
            throw new IllegalArgumentException("Aggregate projection maps outside its new input");
        }
        return remapped;
    }

    int[] inputOrdinals() {
        return inputOrdinals;
    }

    GpuType[] inputTypes() {
        return inputTypes;
    }

    GpuDecimalType[] inputDecimalTypes() {
        return inputDecimalTypes;
    }

    int[] groupingInputOrdinals() {
        return groupingInputOrdinals;
    }

    int[] groupingPackedOrdinals() {
        return groupingPackedOrdinals;
    }

    GpuType[] groupingTypes() {
        return groupingTypes;
    }

    GpuDecimalType[] groupingDecimalTypes() {
        return groupingDecimalTypes;
    }

    AggregateSpec[] aggregates() {
        return aggregates;
    }

    GpuType[] outputTypes() {
        return outputTypes;
    }

    public boolean hasDistinctCount() {
        return distinctSpecs.length > 0;
    }

    int distinctCount() {
        return distinctSpecs.length;
    }

    GpuType[] distinctKeyTypes(int distinctIndex) {
        DistinctSpec spec = distinctSpec(distinctIndex);
        GpuType[] types = Arrays.copyOf(groupingTypes, groupingTypes.length + 1);
        types[groupingTypes.length] = spec.inputType;
        return types;
    }

    GpuDecimalType[] distinctKeyDecimalTypes(int distinctIndex) {
        DistinctSpec spec = distinctSpec(distinctIndex);
        GpuDecimalType[] types =
                Arrays.copyOf(groupingDecimalTypes, groupingDecimalTypes.length + 1);
        types[groupingDecimalTypes.length] = spec.decimalType;
        return types;
    }

    Object[] copyDistinctKeyValues(RowData row, int distinctIndex) {
        DistinctSpec spec = distinctSpec(distinctIndex);
        Object[] values = new Object[groupingTypes.length + 1];
        for (int index = 0; index < groupingTypes.length; index++) {
            values[index] = copyValue(row, groupingInputOrdinals[index], groupingTypes[index],
                    groupingDecimalTypes[index]);
        }
        values[groupingTypes.length] =
                copyValue(row, spec.inputOrdinal, spec.inputType, spec.decimalType);
        return values;
    }

    static int distinctFilterOrdinal(int distinctIndex) {
        if (distinctIndex < 0 || distinctIndex > 29) {
            throw new IllegalArgumentException("invalid DISTINCT filter index " + distinctIndex);
        }
        return -2 - distinctIndex;
    }

    static boolean isDistinctFilterOrdinal(int ordinal) {
        return ordinal <= -2;
    }

    static int distinctIndexForFilterOrdinal(int ordinal) {
        if (!isDistinctFilterOrdinal(ordinal)) {
            throw new IllegalArgumentException("not a DISTINCT filter ordinal: " + ordinal);
        }
        return -2 - ordinal;
    }

    /** GROUP BY all projected fields with no aggregate call, i.e. streaming SELECT DISTINCT. */
    public boolean isDistinctOnly() {
        return groupingTypes.length > 0 && aggregates.length == 0;
    }

    public boolean consumesRetractions() {
        return retractInput;
    }

    /**
     * Exact, read-only predicate for a narrow device-island terminal contract.
     *
     * <p>No mutable planner arrays or package-private aggregate descriptors are exposed.
     */
    public boolean isExactCountStarGrouping(
            int[] expectedGroupingInputOrdinals, GpuType[] expectedGroupingTypes) {
        Objects.requireNonNull(expectedGroupingInputOrdinals, "expectedGroupingInputOrdinals");
        Objects.requireNonNull(expectedGroupingTypes, "expectedGroupingTypes");
        if (retractInput
                || !Arrays.equals(groupingInputOrdinals, expectedGroupingInputOrdinals)
                || !Arrays.equals(groupingTypes, expectedGroupingTypes)
                || aggregates.length != 1) {
            return false;
        }
        AggregateSpec count = aggregates[0];
        return count.kind == AggregateKind.COUNT
                && count.inputPackedOrdinal < 0
                && count.filterPackedOrdinal < 0
                && !count.distinct;
    }

    boolean onlyCounts() {
        for (AggregateSpec aggregate : aggregates) {
            if (aggregate.kind != AggregateKind.COUNT) {
                return false;
            }
        }
        return true;
    }

    boolean distinctInputIsNull(RowData row, int distinctIndex) {
        return row.isNullAt(distinctSpec(distinctIndex).inputOrdinal);
    }

    /** Stable keyed-state representation matching Flink's scalar equality for supported types. */
    String distinctStateKey(RowData row, int distinctIndex) {
        DistinctSpec spec = distinctSpec(distinctIndex);
        if (row.isNullAt(spec.inputOrdinal)) {
            return distinctIndex + ":n:";
        }
        Object value = copyValue(row, spec.inputOrdinal, spec.inputType, spec.decimalType);
        String encoded;
        switch (spec.inputType) {
            case BOOLEAN:
                encoded = (Boolean) value ? "b:1" : "b:0";
                break;
            case INT8:
            case INT16:
            case INT32:
            case INT64:
                encoded = "i:" + value;
                break;
            case FLOAT32:
                int floatBits = Float.floatToIntBits(((Number) value).floatValue());
                if ((floatBits & 0x7fffffff) == 0) {
                    floatBits = 0;
                }
                encoded = "f:"
                        + Integer.toUnsignedString(
                                floatBits, 16);
                break;
            case FLOAT64:
                long doubleBits = Double.doubleToLongBits(((Number) value).doubleValue());
                if ((doubleBits & 0x7fffffffffffffffL) == 0L) {
                    doubleBits = 0L;
                }
                encoded = "d:"
                        + Long.toUnsignedString(
                                doubleBits, 16);
                break;
            case STRING:
                encoded = "s:" + value;
                break;
            case TIMESTAMP3:
                TimestampData timestamp = (TimestampData) value;
                encoded = "t:"
                        + timestamp.getMillisecond()
                        + ':'
                        + timestamp.getNanoOfMillisecond();
                break;
            case DECIMAL64:
            case DECIMAL128:
                encoded = "m:" + ((BigDecimal) value).unscaledValue().toString(16);
                break;
            default:
                throw new IllegalStateException(
                        "Unhandled DISTINCT input type " + spec.inputType);
        }
        return distinctIndex + ":" + encoded;
    }

    private DistinctSpec distinctSpec(int distinctIndex) {
        if (distinctIndex < 0 || distinctIndex >= distinctSpecs.length) {
            throw new IllegalArgumentException("invalid DISTINCT call index " + distinctIndex);
        }
        return distinctSpecs[distinctIndex];
    }

    int groupCount() {
        return groupingTypes.length;
    }

    GenericRowData keySelectorInput(Object[] groupValues) {
        GenericRowData input = new GenericRowData(inputArity);
        for (int i = 0; i < groupValues.length; i++) {
            input.setField(
                    groupingInputOrdinals[i],
                    toFlinkInternal(
                            groupValues[i], groupingTypes[i], groupingDecimalTypes[i]));
        }
        return input;
    }

    GenericRowData outputRow(Object[] groupValues, Object[] aggregateValues) {
        GenericRowData output = new GenericRowData(outputTypes.length);
        for (int i = 0; i < groupValues.length; i++) {
            output.setField(
                    i,
                    toFlinkInternal(groupValues[i], groupingTypes[i], groupingDecimalTypes[i]));
        }
        for (int i = 0; i < aggregateValues.length; i++) {
            AggregateSpec spec = aggregates[i];
            output.setField(
                    groupingTypes.length + i,
                    spec.resultType.toFlinkInternal(spec.outputValue(aggregateValues[i])));
        }
        return output;
    }

    Object[] copyAggregateValues(RowData state) {
        Object[] values = new Object[aggregates.length];
        for (int i = 0; i < values.length; i++) {
            values[i] =
                    aggregates[i].resultType.copyFrom(state, groupingTypes.length + i);
        }
        return values;
    }

    /** Copies grouping fields directly from an input row before Flink is allowed to reuse it. */
    Object[] copyInputGroupValues(RowData input) {
        Object[] values = new Object[groupingTypes.length];
        for (int i = 0; i < values.length; i++) {
            values[i] =
                    copyValue(
                            input,
                            groupingInputOrdinals[i],
                            groupingTypes[i],
                            groupingDecimalTypes[i]);
        }
        return values;
    }

    /** Rebuilds the exact accumulator representation used to seed a resident GPU state entry. */
    Object[] copyAccumulatorValues(
            RowData outputState, AggregateAverageState hiddenAverageState) {
        if (outputState == null) {
            return null;
        }
        Object[] values = copyAggregateValues(outputState);
        for (int i = 0; i < aggregates.length; i++) {
            if (aggregates[i].kind == AggregateKind.AVG) {
                AverageAccumulator accumulator =
                        hiddenAverageState == null
                                ? null
                                : hiddenAverageState.get(i, aggregates[i].argumentType);
                if (accumulator == null) {
                    throw new IllegalStateException(
                            "restored AVG output is missing its exact SUM/COUNT state");
                }
                values[i] = accumulator;
            }
        }
        return values;
    }

    Object[] mergeAccumulatorValues(
            RowData oldOutputState,
            AggregateAverageState oldAverageState,
            Object[] partialValues) {
        Object[] values =
                oldOutputState == null
                        ? partialValues.clone()
                        : copyAggregateValues(oldOutputState);
        for (int i = 0; i < values.length; i++) {
            AggregateSpec spec = aggregates[i];
            if (spec.kind == AggregateKind.AVG) {
                AverageAccumulator previous =
                        oldAverageState == null
                                ? null
                                : oldAverageState.get(i, spec.argumentType);
                values[i] = spec.merge(previous, partialValues[i]);
            } else if (oldOutputState != null) {
                values[i] = spec.merge(values[i], partialValues[i]);
            }
        }
        return values;
    }

    Object[] mergeCachedAccumulatorValues(
            Object[] cachedAccumulatorValues, Object[] partialValues) {
        Object[] values =
                cachedAccumulatorValues == null
                        ? partialValues.clone()
                        : cachedAccumulatorValues.clone();
        if (cachedAccumulatorValues == null) {
            return values;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] = aggregates[i].merge(values[i], partialValues[i]);
        }
        return values;
    }

    boolean materializedAggregateValuesEqual(
            RowData oldOutputState, Object[] accumulatorValues) {
        for (int i = 0; i < accumulatorValues.length; i++) {
            AggregateSpec spec = aggregates[i];
            Object oldValue =
                    spec.resultType.copyFrom(oldOutputState, groupingTypes.length + i);
            Object newValue = spec.outputValue(accumulatorValues[i]);
            if (!Objects.equals(oldValue, newValue)) {
                return false;
            }
        }
        return true;
    }

    Object[] materializeAggregateValues(Object[] accumulatorValues) {
        Object[] output = new Object[accumulatorValues.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = aggregates[i].outputValue(accumulatorValues[i]);
        }
        return output;
    }

    AggregateAverageState averageState(Object[] accumulatorValues) {
        if (!hasAverage()) {
            return null;
        }
        AggregateAverageState state = new AggregateAverageState(aggregates.length);
        for (int i = 0; i < aggregates.length; i++) {
            if (aggregates[i].kind == AggregateKind.AVG) {
                state.set(i, (AverageAccumulator) accumulatorValues[i]);
            }
        }
        return state;
    }

    boolean hasAverage() {
        for (AggregateSpec aggregate : aggregates) {
            if (aggregate.kind == AggregateKind.AVG) {
                return true;
            }
        }
        return false;
    }

    Object[] copyGroupValues(RowData state) {
        Object[] values = new Object[groupingTypes.length];
        for (int i = 0; i < values.length; i++) {
            values[i] = copyValue(state, i, groupingTypes[i], groupingDecimalTypes[i]);
        }
        return values;
    }

    void copyCoreOutput(RowData source, GenericRowData target) {
        for (int i = 0; i < outputTypes.length; i++) {
            target.setField(
                    i,
                    toFlinkInternal(
                            copyValue(source, i, outputTypes[i], outputDecimalTypes[i]),
                            outputTypes[i],
                            outputDecimalTypes[i]));
        }
    }

    void copyCoreOutput(RowData source, BinaryRowWriter target) {
        for (int i = 0; i < outputTypes.length; i++) {
            if (outputDecimalTypes[i] == null) {
                outputTypes[i].writeFromRow(source, i, target, i);
            } else if (source.isNullAt(i)) {
                target.setNullAt(i);
            } else {
                GpuDecimalType decimal = outputDecimalTypes[i];
                target.writeDecimal(
                        i,
                        source.getDecimal(i, decimal.precision(), decimal.scale()),
                        decimal.precision());
            }
        }
    }

    Object copyInputValue(RowData row, int packedOrdinal) {
        return copyValue(
                row,
                inputOrdinals[packedOrdinal],
                inputTypes[packedOrdinal],
                inputDecimalTypes[packedOrdinal]);
    }

    static Object toFlinkInternal(
            Object value, GpuType type, GpuDecimalType decimalType) {
        if (value == null || decimalType == null) {
            return type.toFlinkInternal(value);
        }
        BigDecimal decimal =
                value instanceof DecimalData
                        ? ((DecimalData) value).toBigDecimal()
                        : (BigDecimal) value;
        DecimalData internal =
                DecimalData.fromBigDecimal(
                        decimal, decimalType.precision(), decimalType.scale());
        if (internal == null) {
            throw new ArithmeticException(
                    "DECIMAL value exceeds " + decimalType + ": " + decimal.toPlainString());
        }
        return internal;
    }

    private static Object copyValue(
            RowData row, int ordinal, GpuType type, GpuDecimalType decimalType) {
        if (row.isNullAt(ordinal)) {
            return null;
        }
        if (decimalType == null) {
            return type.copyFrom(row, ordinal);
        }
        return row.getDecimal(ordinal, decimalType.precision(), decimalType.scale())
                .toBigDecimal();
    }

    public String summary() {
        return "groups="
                + Arrays.toString(groupingInputOrdinals)
                + ", aggregates="
                + Arrays.toString(aggregates)
                + ", retractInput="
                + retractInput;
    }

    enum AggregateKind {
        COUNT,
        SUM,
        SUM0,
        AVG,
        MIN,
        MAX
    }

    static final class AggregateSpec implements Serializable {
        private static final long serialVersionUID = 1L;

        final AggregateKind kind;
        final int inputPackedOrdinal;
        final int filterPackedOrdinal;
        final GpuType argumentType;
        final GpuType resultType;
        final boolean distinct;
        final int distinctIndex;

        AggregateSpec(
                AggregateKind kind,
                int inputPackedOrdinal,
                int filterPackedOrdinal,
                GpuType argumentType,
                GpuType resultType,
                boolean distinct) {
            this(
                    kind,
                    inputPackedOrdinal,
                    filterPackedOrdinal,
                    argumentType,
                    resultType,
                    distinct,
                    distinct ? 0 : -1);
        }

        AggregateSpec(
                AggregateKind kind,
                int inputPackedOrdinal,
                int filterPackedOrdinal,
                GpuType argumentType,
                GpuType resultType,
                boolean distinct,
                int distinctIndex) {
            this.kind = kind;
            this.inputPackedOrdinal = inputPackedOrdinal;
            this.filterPackedOrdinal = filterPackedOrdinal;
            this.argumentType = argumentType;
            this.resultType = resultType;
            this.distinct = distinct;
            this.distinctIndex = distinctIndex;
        }

        Object merge(Object existing, Object partial) {
            if (kind == AggregateKind.AVG) {
                AverageAccumulator left = toAverage(existing);
                AverageAccumulator right = toAverage(partial);
                if (right == null) {
                    return left;
                }
                return left == null ? right : left.merge(right);
            }
            if (kind == AggregateKind.COUNT) {
                long left = existing == null ? 0L : ((Number) existing).longValue();
                long right = partial == null ? 0L : ((Number) partial).longValue();
                return left + right;
            }
            Object normalized = normalizePartial(partial);
            if (normalized == null) {
                return existing;
            }
            if (existing == null) {
                return normalized;
            }
            switch (kind) {
                case SUM:
                case SUM0:
                    return add(existing, normalized, resultType);
                case MIN:
                    return lessThan(normalized, existing, resultType) ? normalized : existing;
                case MAX:
                    return greaterThan(normalized, existing, resultType) ? normalized : existing;
                default:
                    throw new IllegalStateException("Unhandled aggregate kind " + kind);
            }
        }

        Object normalizePartial(Object partial) {
            if (kind == AggregateKind.AVG) {
                return toAverage(partial);
            }
            if (partial == null && kind == AggregateKind.SUM0) {
                return zero(resultType);
            }
            return partial == null ? null : convert(partial, resultType);
        }

        Object outputValue(Object accumulator) {
            if (kind != AggregateKind.AVG) {
                return accumulator;
            }
            AverageAccumulator average = toAverage(accumulator);
            return average == null ? null : average.value(resultType);
        }

        private AverageAccumulator toAverage(Object value) {
            if (value == null || value instanceof AverageAccumulator) {
                return (AverageAccumulator) value;
            }
            if (value instanceof Number) {
                return AverageAccumulator.fromValue((Number) value, argumentType);
            }
            throw new IllegalArgumentException(
                    "AVG partial is neither numeric nor an accumulator: " + value.getClass());
        }

        private static Object add(Object left, Object right, GpuType type) {
            switch (type) {
                case INT8:
                    return (byte) (((Number) left).byteValue() + ((Number) right).byteValue());
                case INT16:
                    return (short) (((Number) left).shortValue() + ((Number) right).shortValue());
                case INT32:
                    return ((Number) left).intValue() + ((Number) right).intValue();
                case INT64:
                    return ((Number) left).longValue() + ((Number) right).longValue();
                case FLOAT32:
                    return ((Number) left).floatValue() + ((Number) right).floatValue();
                case FLOAT64:
                    return ((Number) left).doubleValue() + ((Number) right).doubleValue();
                default:
                    throw new IllegalArgumentException("SUM cannot return " + type);
            }
        }

        Object negate(Object value) {
            switch (resultType) {
                case INT8:
                    return (byte) -((Number) value).byteValue();
                case INT16:
                    return (short) -((Number) value).shortValue();
                case INT32:
                    return -((Number) value).intValue();
                case INT64:
                    return -((Number) value).longValue();
                case FLOAT32:
                    return -((Number) value).floatValue();
                case FLOAT64:
                    return -((Number) value).doubleValue();
                default:
                    throw new IllegalArgumentException("SUM cannot negate " + resultType);
            }
        }

        private static Object zero(GpuType type) {
            switch (type) {
                case INT8:
                    return (byte) 0;
                case INT16:
                    return (short) 0;
                case INT32:
                    return 0;
                case INT64:
                    return 0L;
                case FLOAT32:
                    return 0.0f;
                case FLOAT64:
                    return 0.0d;
                default:
                    throw new IllegalArgumentException("SUM0 cannot return " + type);
            }
        }

        @SuppressWarnings({"rawtypes", "unchecked"})
        private static int compare(Object left, Object right, GpuType type) {
            if (type == GpuType.FLOAT32) {
                return Float.compare(((Number) left).floatValue(), ((Number) right).floatValue());
            }
            if (type == GpuType.FLOAT64) {
                return Double.compare(((Number) left).doubleValue(), ((Number) right).doubleValue());
            }
            return ((Comparable) left).compareTo(right);
        }

        private static boolean lessThan(Object left, Object right, GpuType type) {
            if (type == GpuType.FLOAT32) {
                return ((Number) left).floatValue() < ((Number) right).floatValue();
            }
            if (type == GpuType.FLOAT64) {
                return ((Number) left).doubleValue() < ((Number) right).doubleValue();
            }
            return compare(left, right, type) < 0;
        }

        private static boolean greaterThan(Object left, Object right, GpuType type) {
            if (type == GpuType.FLOAT32) {
                return ((Number) left).floatValue() > ((Number) right).floatValue();
            }
            if (type == GpuType.FLOAT64) {
                return ((Number) left).doubleValue() > ((Number) right).doubleValue();
            }
            return compare(left, right, type) > 0;
        }

        private static Object convert(Object value, GpuType type) {
            if (!(value instanceof Number)) {
                return value;
            }
            Number number = (Number) value;
            switch (type) {
                case INT8:
                    return number.byteValue();
                case INT16:
                    return number.shortValue();
                case INT32:
                    return number.intValue();
                case INT64:
                    return number.longValue();
                case FLOAT32:
                    return number.floatValue();
                case FLOAT64:
                    return number.doubleValue();
                default:
                    return value;
            }
        }

        @Override
        public String toString() {
            return kind
                    + "(packed="
                    + inputPackedOrdinal
                    + ", filter="
                    + filterPackedOrdinal
                    + ", input="
                    + argumentType
                    + ", output="
                    + resultType
                    + ", distinct="
                    + distinct
                    + ", distinctIndex="
                    + distinctIndex
                    + ")";
        }
    }

    public static final class CompileResult {
        private final GpuAggregatePlan plan;
        private final String reason;

        private CompileResult(GpuAggregatePlan plan, String reason) {
            this.plan = plan;
            this.reason = reason;
        }

        static CompileResult supported(GpuAggregatePlan plan) {
            return new CompileResult(Objects.requireNonNull(plan), null);
        }

        static CompileResult unsupported(String reason) {
            return new CompileResult(null, Objects.requireNonNull(reason));
        }

        public boolean isSupported() {
            return plan != null;
        }

        public GpuAggregatePlan plan() {
            if (plan == null) {
                throw new IllegalStateException("Unsupported aggregate: " + reason);
            }
            return plan;
        }

        public String reason() {
            return reason;
        }
    }

    private static final class PendingSpec {
        final AggregateKind kind;
        final int inputOrdinal;
        final int filterOrdinal;
        final GpuType argumentType;
        final GpuType resultType;
        final boolean distinct;
        final int distinctIndex;

        PendingSpec(
                AggregateKind kind,
                int inputOrdinal,
                int filterOrdinal,
                GpuType argumentType,
                GpuType resultType,
                boolean distinct,
                int distinctIndex) {
            this.kind = kind;
            this.inputOrdinal = inputOrdinal;
            this.filterOrdinal = filterOrdinal;
            this.argumentType = argumentType;
            this.resultType = resultType;
            this.distinct = distinct;
            this.distinctIndex = distinctIndex;
        }
    }

    private static final class DistinctSpec implements Serializable {
        private static final long serialVersionUID = 1L;

        final int inputOrdinal;
        final GpuType inputType;
        final GpuDecimalType decimalType;
        final int aggregateIndex;

        DistinctSpec(int inputOrdinal, GpuType inputType, int aggregateIndex) {
            this(inputOrdinal, inputType, null, aggregateIndex);
        }

        DistinctSpec(
                int inputOrdinal,
                GpuType inputType,
                GpuDecimalType decimalType,
                int aggregateIndex) {
            this.inputOrdinal = inputOrdinal;
            this.inputType = inputType;
            this.decimalType = decimalType;
            this.aggregateIndex = aggregateIndex;
        }
    }

    private static final class UnsupportedAggregate extends RuntimeException {
        private static final long serialVersionUID = 1L;

        UnsupportedAggregate(String message) {
            super(message);
        }
    }
}
