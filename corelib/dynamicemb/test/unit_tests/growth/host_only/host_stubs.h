// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-only stand-ins for the CUDA runtime, the driver API and ATen: enough to compile
// and RUN HostVMMTensor (cut out of the real src/vmm_tensor.cu by extract_host_class.py)
// on an ordinary Linux box, with no GPU and no nvcc. The fake runtime records every
// registration and can be told to fail one, to return a device pointer that is not
// contiguous, or to take its time.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

typedef int cudaError_t;
constexpr cudaError_t cudaSuccess = 0, cudaErrorNotSupported = 801,
                      cudaErrorHostMemoryAlreadyRegistered = 712,
                      cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2;
enum { cudaDevAttrCanMapHostMemory = 1 };
constexpr unsigned cudaHostRegisterMapped = 2, cudaHostRegisterPortable = 1,
                   cudaDeviceMapHost = 8;
typedef unsigned long long CUdeviceptr;

// What the fake runtime recorded, and the failures a test injects.
struct FakeCuda {
  std::map<void *, size_t> registered; // start -> bytes, what is registered now
  std::vector<std::pair<void *, size_t>> register_calls;
  std::vector<void *> unregister_calls;
  int fail_register_at = -1; // fail the Nth cudaHostRegister (0-based), -1: never
  int calls = 0;
  long long device_offset_for_call = -1; // the Nth device pointer query gets +4096
  int queries = 0;
  cudaError_t last = cudaSuccess;
  void reset() {
    registered.clear();
    register_calls.clear();
    unregister_calls.clear();
    fail_register_at = -1;
    calls = 0;
    device_offset_for_call = -1;
    queries = 0;
    last = cudaSuccess;
  }
};
inline FakeCuda g_cuda;
// A cudaHostRegister that takes its time (a multi-GB tail does), and the flag a test
// waits for to know a thread is inside one.
inline int g_register_delay_ms = 0;
inline std::atomic<bool> g_inside_register{false};

inline cudaError_t cudaHostRegister(void *p, size_t bytes, unsigned) {
  g_inside_register = true;
  if (g_register_delay_ms > 0)
    std::this_thread::sleep_for(std::chrono::milliseconds(g_register_delay_ms));
  int n = g_cuda.calls++;
  if (reinterpret_cast<uintptr_t>(p) % 4096 != 0 || bytes % 4096 != 0)
    return cudaErrorInvalidValue;
  if (n == g_cuda.fail_register_at)
    return g_cuda.last = cudaErrorMemoryAllocation;
  if (g_cuda.registered.count(p))
    return g_cuda.last = cudaErrorHostMemoryAlreadyRegistered;
  g_cuda.registered[p] = bytes;
  g_cuda.register_calls.push_back({p, bytes});
  return cudaSuccess;
}
inline cudaError_t cudaHostUnregister(void *p) {
  g_cuda.unregister_calls.push_back(p);
  if (!g_cuda.registered.erase(p)) // must be the exact start
    return g_cuda.last = cudaErrorInvalidValue;
  return cudaSuccess;
}
inline cudaError_t cudaHostGetDevicePointer(void **d, void *h, unsigned) {
  int n = g_cuda.queries++;
  // UVA: the device pointer is the host pointer
  *d = static_cast<char *>(h) + (n == g_cuda.device_offset_for_call ? 4096 : 0);
  return cudaSuccess;
}
inline cudaError_t cudaGetLastError() {
  auto e = g_cuda.last;
  g_cuda.last = cudaSuccess;
  return e;
}
inline const char *cudaGetErrorString(cudaError_t e) {
  return e == cudaSuccess                ? "success"
         : e == cudaErrorNotSupported    ? "operation not supported"
         : e == cudaErrorMemoryAllocation ? "out of memory"
                                          : "error";
}
inline cudaError_t cudaDeviceGetAttribute(int *v, int, int) {
  *v = 1;
  return cudaSuccess;
}
inline cudaError_t cudaSetDeviceFlags(unsigned) { return cudaSuccess; }
#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e_ = (cmd);                                                    \
    if (e_ != cudaSuccess)                                                     \
      throw std::runtime_error("cuda error");                                  \
  } while (0)

namespace at {
enum ScalarType { kChar = 1, kInt = 3, kLong = 4, kHalf = 5, kFloat = 6, kCUDA = 100 };
struct TensorOptions {
  TensorOptions dtype(ScalarType) { return *this; }
  TensorOptions device(ScalarType, int) { return *this; }
};
struct Tensor {
  void *ptr = nullptr;
  int64_t numel = 0;
};
inline Tensor from_blob(void *p, std::vector<int64_t> sizes, TensorOptions) {
  return Tensor{p, sizes[0]};
}
} // namespace at
namespace torch {
using Dtype = at::ScalarType;
using ScalarType = at::ScalarType;
} // namespace torch
namespace dyn_emb {
inline int get_size(torch::ScalarType t) {
  return t == at::kHalf ? 2 : t == at::kChar ? 1 : t == at::kLong ? 8 : 4;
}
inline void wait_for_device_work(int) {}
} // namespace dyn_emb
