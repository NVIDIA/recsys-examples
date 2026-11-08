// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

import org.apache.flink.streaming.api.operators.AbstractStreamOperator;

/** Common official-resource lifecycle for every CUDA-backed Flink operator. */
public abstract class AbstractGpuStreamOperator<OUT> extends AbstractStreamOperator<OUT>
        implements GpuResourceAware {
    private static final long serialVersionUID = 1L;
    private final GpuOperatorResourceLifecycle gpuResourceLifecycle =
            new GpuOperatorResourceLifecycle();

    @Override
    public final void configureGpuResource(GpuResourceConfig config) {
        gpuResourceLifecycle.configure(config);
    }

    @Override
    public void open() throws Exception {
        super.open();
        gpuResourceLifecycle.open(getRuntimeContext(), getClass().getSimpleName());
    }

    @Override
    public void close() throws Exception {
        try {
            super.close();
        } finally {
            gpuResourceLifecycle.close();
        }
    }
}
