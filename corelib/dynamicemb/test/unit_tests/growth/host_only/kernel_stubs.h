// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Host-only stand-ins for CUDA and ATen, enough to compile and RUN the rehash kernels
// of table_rows.cu on the CPU: every "thread" of every block runs one after the other.
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__ __restrict
#define C10_UNLIKELY(x) (x)
#define CUDA_KERNEL_ASSERT(cond)                                              \
  do {                                                                        \
    if (!(cond)) throw std::runtime_error("device assert: " #cond);           \
  } while (0)
#define DEMB_CUDA_KERNEL_LAUNCH_CHECK() do {} while (0)
#define TORCH_CHECK(cond, ...)                                                \
  do {                                                                        \
    if (!(cond)) throw std::runtime_error("TORCH_CHECK failed: " #cond);      \
  } while (0)

struct Dim3 { int64_t x = 0, y = 0, z = 0; };
inline Dim3 blockIdx, blockDim, gridDim, threadIdx;

typedef void *cudaStream_t;

// CUDA's global atomics (the 64-bit unsigned ones the code uses).
inline unsigned long long atomicAdd(unsigned long long *a, unsigned long long v) {
  unsigned long long o = *a; *a = o + v; return o;
}
inline unsigned long long atomicCAS(unsigned long long *a, unsigned long long cmp,
                                    unsigned long long val) {
  unsigned long long o = *a; if (o == cmp) *a = val; return o;
}

namespace at {
enum ScalarType { kLong, kInt, kByte, kFloat };
struct Options { ScalarType dtype; };
template <class T> constexpr ScalarType scalar_of();
template <> constexpr ScalarType scalar_of<int64_t>() { return kLong; }
template <> constexpr ScalarType scalar_of<int32_t>() { return kInt; }
template <> constexpr ScalarType scalar_of<uint8_t>() { return kByte; }
inline size_t item_size(ScalarType t) { return t == kLong ? 8 : t == kInt ? 4 : t == kFloat ? 4 : 1; }

// A strided tensor over host memory.
class Tensor {
public:
  std::shared_ptr<std::vector<char>> mem;
  int64_t offset = 0; // in elements
  std::vector<int64_t> sizes_, strides_;
  ScalarType dtype_ = kLong;
  bool defined_ = false;
  Tensor() {}
  template <class T>
  static Tensor make(std::vector<int64_t> sizes, std::vector<int64_t> strides) {
    Tensor t; t.defined_ = true; t.dtype_ = scalar_of<T>();
    int64_t extent = 1;
    for (size_t i = 0; i < sizes.size(); ++i) extent += (sizes[i] - 1) * strides[i];
    t.mem = std::make_shared<std::vector<char>>(std::max<int64_t>(extent, 1) * sizeof(T));
    t.sizes_ = sizes; t.strides_ = strides; return t;
  }
  template <class T> static Tensor vec(int64_t n) { return make<T>({n}, {1}); }
  bool defined() const { return defined_; }
  bool is_cuda() const { return true; }
  ScalarType scalar_type() const { return dtype_; }
  int64_t dim() const { return sizes_.size(); }
  int64_t size(int i) const { return sizes_[i]; }
  int64_t stride(int i) const { return strides_[i]; }
  int64_t numel() const { int64_t n = 1; for (auto s : sizes_) n *= s; return n; }
  bool is_contiguous() const {
    int64_t expect = 1;
    for (int i = (int)sizes_.size() - 1; i >= 0; --i) {
      if (sizes_[i] != 1 && strides_[i] != expect) return false;
      expect *= sizes_[i];
    }
    return true;
  }
  template <class T> T *data_ptr() const {
    TORCH_CHECK(scalar_of<T>() == dtype_);
    return reinterpret_cast<T *>(mem->data()) + offset;
  }
  Options options() const { return Options{dtype_}; }
  // x[a:b] of a vector
  Tensor slice(int64_t a, int64_t b) const { Tensor t = *this; t.offset += a; t.sizes_ = {b - a}; return t; }
};
namespace cuda { struct Stream { operator cudaStream_t() const { return nullptr; } };
  inline Stream getCurrentCUDAStream() { return {}; } }
} // namespace at

namespace torch {
inline at::Tensor full(std::vector<int64_t> sizes, int64_t v, at::Options o) {
  TORCH_CHECK(o.dtype == at::kLong);
  auto t = at::Tensor::make<int64_t>(sizes, {1});
  std::fill(t.data_ptr<int64_t>(), t.data_ptr<int64_t>() + t.numel(), v);
  return t;
}
inline at::Tensor zeros(std::vector<int64_t> sizes, at::Options o) {
  TORCH_CHECK(o.dtype == at::kLong);
  auto t = at::Tensor::make<int64_t>(sizes, {1});
  std::fill(t.mem->begin(), t.mem->end(), 0);
  return t;
}
} // namespace torch

// The bucket type the kernels name; the digest is any recognisable byte.
template <class K> struct LinearBucket {
  static uint8_t reclaimed_digest() { return 0xA7; }
};

// dyn_emb has its own atomicAdd overloads, which HIDE the global ones inside the
// namespace: the code under test must say ::atomicAdd.
namespace dyn_emb {
inline float atomicAdd(float *a, float v) { float o = *a; *a = o + v; return o; }

template <class T> T *get_pointer(at::Tensor t) { return t.data_ptr<T>(); }
template <class T> T *get_pointer(const std::optional<at::Tensor> &t) {
  if (!t.has_value()) return nullptr;
  return t->data_ptr<T>();
}

// The sequential stand-in of <<<grid, block, shmem, stream>>>.
template <class F, class... A>
void emulate_launch(F f, int64_t grid, int64_t block, int shmem, cudaStream_t, A... args) {
  (void)shmem;
  gridDim.x = grid; blockDim.x = block;
  for (int64_t b = 0; b < grid; ++b) {
    blockIdx.x = b;
    for (int64_t t = 0; t < block; ++t) { threadIdx.x = t; f(args...); }
  }
}
} // namespace dyn_emb
