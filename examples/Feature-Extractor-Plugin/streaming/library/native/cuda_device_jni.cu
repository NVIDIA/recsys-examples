// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Derived unchanged (except JNI symbol prefix) from Mega-Flink native/cuda_device_jni.cu.

#include <cuda_runtime.h>
#include <jni.h>

#include <cstddef>
#include <limits>
#include <string>

namespace {

void throw_java(JNIEnv* env, const char* operation, cudaError_t status) {
    int device_index = -1;
    cudaGetDevice(&device_index);
    std::string message = std::string(operation)
            + ": code=" + std::to_string(static_cast<int>(status))
            + " name=" + cudaGetErrorName(status)
            + " message=" + cudaGetErrorString(status)
            + " device=" + std::to_string(device_index);
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type != nullptr) {
        env->ThrowNew(type, message.c_str());
    }
}

__global__ void injected_noop_kernel() {}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_CudaDeviceBinding_nativeSetDevice(
        JNIEnv* env, jclass, jint device_index) {
    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        throw_java(env, "cudaSetDevice failed", status);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_nvidia_fep_flink_CudaDeviceBinding_nativeGetDevice(JNIEnv* env, jclass) {
    int device_index = -1;
    cudaError_t status = cudaGetDevice(&device_index);
    if (status != cudaSuccess) {
        throw_java(env, "cudaGetDevice failed", status);
        return -1;
    }
    return device_index;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_CudaDeviceBinding_nativeMemoryInfo(
        JNIEnv* env, jclass, jint device_index) {
    cudaError_t status = cudaSetDevice(device_index);
    if (status != cudaSuccess) {
        throw_java(env, "cudaSetDevice for memory telemetry failed", status);
        return nullptr;
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess) {
        throw_java(env, "cudaMemGetInfo failed", status);
        return nullptr;
    }
    jlong values[2] = {
            static_cast<jlong>(free_bytes), static_cast<jlong>(total_bytes)};
    jlongArray result = env->NewLongArray(2);
    if (result == nullptr) {
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 2, values);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_CudaDeviceBinding_nativeInjectFailure(
        JNIEnv* env, jclass, jint failure_kind) {
    cudaError_t status = cudaSuccess;
    const char* operation = nullptr;
    if (failure_kind == 1) {
        operation = "injected CUDA OOM";
        void* pointer = nullptr;
        status = cudaMalloc(&pointer, std::numeric_limits<std::size_t>::max());
        if (status == cudaSuccess) {
            cudaFree(pointer);
        }
    } else if (failure_kind == 2) {
        operation = "injected CUDA kernel launch error";
        injected_noop_kernel<<<1, 0>>>();
        status = cudaPeekAtLastError();
    } else {
        jclass type = env->FindClass("java/lang/IllegalArgumentException");
        if (type != nullptr) {
            env->ThrowNew(type, "unknown CUDA fault injection kind");
        }
        return;
    }
    // Both injected failures are intentionally non-poisoning. Clear the per-thread last-error
    // slot so the restarted Flink task proves recovery from this error rather than inheriting a
    // synthetic sticky status. Illegal-address/reset/ECC faults require an exclusive test node.
    cudaGetLastError();
    if (status == cudaSuccess) {
        jclass type = env->FindClass("java/lang/IllegalStateException");
        if (type != nullptr) {
            env->ThrowNew(type, "requested CUDA fault injection unexpectedly succeeded");
        }
        return;
    }
    throw_java(env, operation, status);
}
