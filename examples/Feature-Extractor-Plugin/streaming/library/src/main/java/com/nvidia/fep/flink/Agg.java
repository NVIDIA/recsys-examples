// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;
import java.util.Objects;

/**
 * One aggregate of a {@link GpuWindows} window aggregation. Create it with one of the factory
 * methods and optionally name the output column with {@link #as(String)}.
 *
 * <pre>{@code
 * Agg.count().as("cnt")          // COUNT(*)
 * Agg.count("play_ms")           // COUNT(play_ms): non-null values only
 * Agg.sum("play_ms").as("play")  // SUM, AVG, MIN, MAX take one numeric column
 * }</pre>
 *
 * <p>Output types follow Flink SQL: COUNT is {@code BIGINT NOT NULL}; SUM, AVG, MIN and MAX have the
 * type of the input column (AVG of an integer column is an integer, as in Flink SQL), nullable when
 * the input column is nullable. DISTINCT, FILTER and other aggregate functions are not supported.
 */
public final class Agg implements Serializable {
    private static final long serialVersionUID = 1L;

    private final GpuAggregatePlan.AggregateKind kind;
    private final String column;
    private final String alias;

    private Agg(GpuAggregatePlan.AggregateKind kind, String column, String alias) {
        this.kind = kind;
        this.column = column;
        this.alias = alias;
    }

    /** {@code COUNT(*)}. */
    public static Agg count() {
        return new Agg(GpuAggregatePlan.AggregateKind.COUNT, null, null);
    }

    /** {@code COUNT(column)}: the number of rows where {@code column} is not NULL. */
    public static Agg count(String column) {
        return new Agg(
                GpuAggregatePlan.AggregateKind.COUNT,
                Objects.requireNonNull(column, "column"),
                null);
    }

    /** {@code SUM(column)} over a numeric column. */
    public static Agg sum(String column) {
        return new Agg(
                GpuAggregatePlan.AggregateKind.SUM, Objects.requireNonNull(column, "column"), null);
    }

    /** {@code AVG(column)} over a numeric column. */
    public static Agg avg(String column) {
        return new Agg(
                GpuAggregatePlan.AggregateKind.AVG, Objects.requireNonNull(column, "column"), null);
    }

    /** {@code MIN(column)}. */
    public static Agg min(String column) {
        return new Agg(
                GpuAggregatePlan.AggregateKind.MIN, Objects.requireNonNull(column, "column"), null);
    }

    /** {@code MAX(column)}. */
    public static Agg max(String column) {
        return new Agg(
                GpuAggregatePlan.AggregateKind.MAX, Objects.requireNonNull(column, "column"), null);
    }

    /** Names the output column. The default is the lower-case function name plus the column. */
    public Agg as(String alias) {
        Objects.requireNonNull(alias, "alias");
        if (alias.trim().isEmpty()) {
            throw new IllegalArgumentException("aggregate alias must not be blank");
        }
        return new Agg(kind, column, alias);
    }

    GpuAggregatePlan.AggregateKind kind() {
        return kind;
    }

    /** The input column name, or null for COUNT(*). */
    String column() {
        return column;
    }

    String outputName() {
        if (alias != null) {
            return alias;
        }
        String function = kind.name().toLowerCase(java.util.Locale.ROOT);
        return column == null ? function : function + "_" + column;
    }

    @Override
    public String toString() {
        return kind.name() + "(" + (column == null ? "*" : column) + ")"
                + (alias == null ? "" : " AS " + alias);
    }
}
