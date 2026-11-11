// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/**
 * Remembers the CUDA device on which a native context was created and rebinds later host threads.
 *
 * <p>Flink may initialize an operator and execute it on different host threads. CUDA's current
 * device is thread-local, while streams and events belong to the device that created them. Every
 * JNI entry that touches an existing context therefore has to restore that context's device first.
 */
final class GpuNativeDeviceGuard {
    private final int deviceIndex;

    private GpuNativeDeviceGuard(int deviceIndex) {
        this.deviceIndex = deviceIndex;
    }

    static GpuNativeDeviceGuard capture() {
        return new GpuNativeDeviceGuard(CudaDeviceBinding.currentDevice());
    }

    void bind() {
        CudaDeviceBinding.bind(deviceIndex);
    }

    int deviceIndex() {
        return deviceIndex;
    }
}
