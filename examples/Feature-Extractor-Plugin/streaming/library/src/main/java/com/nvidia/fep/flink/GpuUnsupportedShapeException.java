// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/**
 * Thrown by the builders in this library when a job's shape is outside what the GPU operators
 * support. It is raised while the job graph is being built (before any record is processed) and its
 * message names the unsupported element. There is no automatic fallback to a CPU operator.
 */
public class GpuUnsupportedShapeException extends IllegalArgumentException {
    private static final long serialVersionUID = 1L;

    public GpuUnsupportedShapeException(String message) {
        super(message);
    }
}
