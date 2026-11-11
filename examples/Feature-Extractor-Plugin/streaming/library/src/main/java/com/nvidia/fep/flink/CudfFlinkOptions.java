// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;

/** GPU device and memory-admission options read by the GPU resource lifecycle. */
public final class CudfFlinkOptions {
    public static final ConfigOption<String> GPU_RESOURCE_MODE =
            ConfigOptions.key("cudf.flink.gpu.resource-mode")
                    .stringType()
                    .defaultValue(GpuResourceMode.EXTERNAL_RESOURCE.configValue());

    public static final ConfigOption<String> GPU_RESOURCE_NAME =
            ConfigOptions.key("cudf.flink.gpu.resource-name")
                    .stringType()
                    .defaultValue("gpu");

    public static final ConfigOption<Integer> GPU_LEGACY_DEVICE_INDEX =
            ConfigOptions.key("cudf.flink.gpu.legacy-device-index")
                    .intType()
                    .defaultValue(0);

    public static final ConfigOption<Integer> GPU_MAX_ACTIVE_SLOTS_PER_DEVICE =
            ConfigOptions.key("cudf.flink.gpu.max-active-slots-per-device")
                    .intType()
                    .defaultValue(1)
                    .withDescription(
                            "Maximum concurrently active Flink slot allocations admitted on one "
                                    + "CUDA device inside a TaskManager; values above 1 are an "
                                    + "explicit opt-in and remain bounded by GPU memory admission.");

    public static final ConfigOption<Long> GPU_MEMORY_TASKMANAGER_LIMIT_BYTES =
            ConfigOptions.key("cudf.flink.gpu.memory.taskmanager-limit-bytes")
                    .longType()
                    .defaultValue(64L * 1024L * 1024L * 1024L);

    public static final ConfigOption<Long> GPU_MEMORY_TASK_LIMIT_BYTES =
            ConfigOptions.key("cudf.flink.gpu.memory.task-limit-bytes")
                    .longType()
                    .defaultValue(16L * 1024L * 1024L * 1024L);

    public static final ConfigOption<Long> GPU_MEMORY_OPERATOR_RESERVATION_BYTES =
            ConfigOptions.key("cudf.flink.gpu.memory.operator-reservation-bytes")
                    .longType()
                    .defaultValue(1024L * 1024L * 1024L);

    private CudfFlinkOptions() {}
}
