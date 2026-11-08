// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/**
 * Extra output columns that describe the window an aggregate row belongs to, as in Flink SQL's
 * {@code window_start}, {@code window_end} and {@code window_time}.
 */
public enum WindowProperty {
    /** Inclusive start of the window; column name {@code window_start}. */
    START("window_start"),
    /** Exclusive end of the window; column name {@code window_end}. */
    END("window_end"),
    /** Largest timestamp inside the window ({@code window_end - 1 ms}); column {@code window_time}. */
    ROWTIME("window_time");

    private final String columnName;

    WindowProperty(String columnName) {
        this.columnName = columnName;
    }

    /** The output column name. */
    public String columnName() {
        return columnName;
    }

    int kind() {
        switch (this) {
            case START:
                return CudfTumblingWindowOperator.PROPERTY_START;
            case END:
                return CudfTumblingWindowOperator.PROPERTY_END;
            default:
                return CudfTumblingWindowOperator.PROPERTY_ROWTIME;
        }
    }
}
