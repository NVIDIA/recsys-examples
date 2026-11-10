// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
package com.nvidia.fep.flink;

/** Minimal project-owned JNI bridge for binding the Flink task thread to a CUDA device. */
final class CudaDeviceBinding {
    static {
        try {
            System.loadLibrary("cudf_flink_device");
        } catch (UnsatisfiedLinkError error) {
            throw new IllegalStateException(
                    "Could not load libcudf_flink_device.so. Build the native libraries and put their "
                            + "directory on LD_LIBRARY_PATH (or java.library.path). Original error: "
                            + error.getMessage(),
                    error);
        }
    }

    private CudaDeviceBinding() {}

    static void bind(int deviceIndex) {
        nativeSetDevice(deviceIndex);
    }

    static int currentDevice() {
        return nativeGetDevice();
    }

    /** Returns free and total bytes for the selected physical CUDA device. */
    static long[] memoryInfo(int deviceIndex) {
        long[] values = nativeMemoryInfo(deviceIndex);
        if (values == null || values.length != 2) {
            throw new IllegalStateException("cudaMemGetInfo returned an invalid result");
        }
        return values;
    }

    static void injectFailure(int failureKind) {
        nativeInjectFailure(failureKind);
    }

    private static native void nativeSetDevice(int deviceIndex);

    private static native int nativeGetDevice();

    private static native long[] nativeMemoryInfo(int deviceIndex);

    private static native void nativeInjectFailure(int failureKind);
}
