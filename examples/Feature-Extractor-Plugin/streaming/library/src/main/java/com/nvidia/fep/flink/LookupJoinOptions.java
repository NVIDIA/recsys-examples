// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ConfigOptions;

/** Configuration for the bounded, read-only BIGINT Lookup Join product path. */
public final class LookupJoinOptions {
    /** Maximum number of probe rows retained by one operator batch. */
    public static final ConfigOption<Integer> BATCH_ROWS =
            ConfigOptions.key("cudf.flink.lookup-join.batch.rows")
                    .intType()
                    .defaultValue(1_048_576);

    /** Flush deadline for a partial batch. Partial batches use the exact JVM lookup path. */
    public static final ConfigOption<Long> BATCH_MAX_WAIT_MS =
            ConfigOptions.key("cudf.flink.lookup-join.batch.max-wait-ms")
                    .longType()
                    .defaultValue(5L);

    /** Smallest batch sent to the GPU; smaller batches use the exact JVM lookup. */
    public static final ConfigOption<Integer> GPU_MIN_BATCH_ROWS =
            ConfigOptions.key("cudf.flink.lookup-join.gpu.min-batch-rows")
                    .intType()
                    .defaultValue(1_024);

    /** Hard bound checked before a connector snapshot can allocate native state. */
    public static final ConfigOption<Integer> MAX_DIMENSION_ROWS =
            ConfigOptions.key("cudf.flink.lookup-join.dimension.max-rows")
                    .intType()
                    .defaultValue(1_048_576);

    private LookupJoinOptions() {}
}
