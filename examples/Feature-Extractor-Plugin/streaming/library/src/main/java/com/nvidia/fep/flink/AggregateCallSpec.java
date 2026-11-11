// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * One aggregate call as declared by the caller. This replaces the Calcite {@code AggregateCall}
 * that Mega-Flink's planner passes to {@code GpuAggregatePlan.PlanCompiler.tryCompile}: the same
 * accessors are kept so the admission code can stay as it was. Only the shapes the library's
 * window operators accept can be expressed (COUNT, SUM, SUM0, AVG, MIN, MAX over one input
 * column or COUNT(*); optional BOOLEAN FILTER column; DISTINCT is representable so that it is
 * rejected with an explicit reason instead of being silently dropped).
 */
final class AggregateCallSpec implements Serializable {
    private static final long serialVersionUID = 1L;

    private final GpuAggregatePlan.AggregateKind kind;
    private final List<Integer> args;
    private final boolean distinct;
    private final int filterArg;

    AggregateCallSpec(
            GpuAggregatePlan.AggregateKind kind,
            List<Integer> args,
            boolean distinct,
            int filterArg) {
        this.kind = Objects.requireNonNull(kind, "kind");
        this.args = Collections.unmodifiableList(new ArrayList<>(args));
        this.distinct = distinct;
        this.filterArg = filterArg;
    }

    GpuAggregatePlan.AggregateKind getKind() {
        return kind;
    }

    List<Integer> getArgList() {
        return args;
    }

    boolean isDistinct() {
        return distinct;
    }

    boolean isApproximate() {
        return false;
    }

    boolean hasFilter() {
        return filterArg >= 0;
    }

    int filterArg() {
        return filterArg;
    }

    String getName() {
        return kind.name();
    }

    @Override
    public String toString() {
        return kind.name()
                + "("
                + (distinct ? "DISTINCT " : "")
                + (args.isEmpty() ? "*" : args.toString())
                + ")"
                + (hasFilter() ? " FILTER $" + filterArg : "");
    }
}
