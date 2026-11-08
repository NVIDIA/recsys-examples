// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import java.io.Serializable;
import java.util.Locale;

/** How a GPU operator obtains its CUDA device. */
public enum GpuResourceMode implements Serializable {
    /** Resolve devices through Flink's external-resource API and fail if none was assigned. */
    EXTERNAL_RESOURCE("external-resource"),
    /** Compatibility mode for the embedded single-GPU benchmark harness. */
    LEGACY_DEVICE("legacy-device");

    private final String configValue;

    GpuResourceMode(String configValue) {
        this.configValue = configValue;
    }

    public String configValue() {
        return configValue;
    }

    public static GpuResourceMode parse(String value) {
        String normalized =
                value == null
                        ? ""
                        : value.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        for (GpuResourceMode mode : values()) {
            if (mode.configValue.equals(normalized)) {
                return mode;
            }
        }
        throw new IllegalArgumentException(
                "Unknown cudf.flink.gpu.resource-mode '"
                        + value
                        + "'; expected external-resource or legacy-device");
    }
}
