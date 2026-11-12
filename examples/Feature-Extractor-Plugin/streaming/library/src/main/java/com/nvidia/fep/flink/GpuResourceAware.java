// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** Implemented by runtime operators whose CUDA device is configured by the planner shim. */
public interface GpuResourceAware {
    void configureGpuResource(GpuResourceConfig config);
}
