// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.util.Locale;

/** Execution modes compared by the benchmark. */
public enum ExecutionMode {
    CPU_ROW("cpu-row", false),
    /** Benchmark baseline: Flink's own mini-batch aggregation, still entirely on CPU. */
    CPU_MINI_BATCH("cpu-mini-batch", false),
    /** Only project-owned CUDA kernels are allowed. */
    GPU_NATIVE("gpu-native", true);

    private final String configValue;
    private final boolean gpu;

    ExecutionMode(String configValue, boolean gpu) {
        this.configValue = configValue;
        this.gpu = gpu;
    }

    public String configValue() {
        return configValue;
    }

    public boolean isGpu() {
        return gpu;
    }

    public static ExecutionMode parse(String value) {
        String normalized = value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        for (ExecutionMode mode : values()) {
            if (mode.configValue.equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException(
                "Unknown mode '"
                        + value
                        + "'. Expected cpu-row, cpu-mini-batch, or gpu-native.");
    }
}
