/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

// ---------------------------------------------------------------------------
// Where device memory comes from.
//
// Every component that owns device memory takes an allocator, so that a cache
// embedded in a PyTorch job can draw from torch's caching allocator rather than
// calling cudaMalloc behind its back. That matters for three reasons, none of
// them about speed:
//
//   * memory the cache holds shows up in torch.cuda.memory_allocated() and in
//     memory snapshots, instead of being a gap between what torch reports and
//     what nvidia-smi does
//   * an out-of-memory error names the cache, rather than surfacing somewhere
//     unrelated that happened to allocate next
//   * freed blocks return to torch's pool and get reused, instead of each
//     cudaFree handing memory back to the driver for torch to re-acquire
//
// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------
//
// An allocator provides:
//
//   using Buffer = ...;                  // owns a device allocation
//   Buffer allocate(size_t bytes) const; // zero bytes is legal, yields empty
//
// A buffer provides:
//
//   void*  data()  const;   // device pointer, nullptr when empty
//   size_t bytes() const;
//
// and releases its allocation when destroyed. It must be movable; whether it is
// copyable is up to the implementation (the torch one is, because at::Tensor is
// reference counted).
//
// Allocators may hold state — a device index, a stream, a pool — so they are
// stored rather than instantiated at each use. They only ever live on the host,
// so their size costs nothing.
//
// Alignment: buffers are aligned to at least 8 bytes, which is what the node
// arrays need. Both implementations here give considerably more.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// Detected rather than configured, so the host tests — which build with a
// stand-in for the CUDA runtime and no torch at all — need no special flags.
//
// This must sit outside the namespace: an #include inside one would drop every
// ATen declaration into embedding_cache::.
#if defined(__has_include)
#if __has_include(<ATen/ATen.h>)
#define EMBEDDING_CACHE_HAS_TORCH 1
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#endif

namespace embedding_cache {

// ---------------------------------------------------------------------------
// CudaAllocator — cudaMalloc / cudaFree
// ---------------------------------------------------------------------------
//
// The default, and what the host tests use. Fine standalone; inside a PyTorch
// process prefer TorchAllocator, so the two do not compete for memory without
// either knowing about the other.

class CudaBuffer {
 public:
  CudaBuffer() = default;

  explicit CudaBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) return;
    const cudaError_t e = cudaMalloc(&data_, bytes);
    if (e != cudaSuccess) {
      throw std::runtime_error("embedding_cache: cudaMalloc of " +
                               std::to_string(bytes) +
                               " bytes failed: " + cudaGetErrorString(e));
    }
  }

  ~CudaBuffer() {
    if (data_) cudaFree(data_);
  }

  CudaBuffer(CudaBuffer&& o) noexcept
      : data_(o.data_), bytes_(o.bytes_) {
    o.data_ = nullptr;
    o.bytes_ = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& o) noexcept {
    if (this != &o) {
      if (data_) cudaFree(data_);
      data_ = o.data_;
      bytes_ = o.bytes_;
      o.data_ = nullptr;
      o.bytes_ = 0;
    }
    return *this;
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

struct CudaAllocator {
  using Buffer = CudaBuffer;
  Buffer allocate(size_t bytes) const { return Buffer(bytes); }
};

// ---------------------------------------------------------------------------
// TorchAllocator — at::Tensor
// ---------------------------------------------------------------------------
//
// Compiled only where ATen is on the include path; see the detection above.
//
// ---------------------------------------------------------------------------
// Why this one takes a stream and CudaAllocator does not
// ---------------------------------------------------------------------------
//
// torch's caching allocator is stream-aware in a way cudaMalloc is not. Every
// block it hands out is tagged with the stream that was current when it was
// allocated, and on free it may hand that block straight back to the pool for
// reuse *on that stream* — with no synchronisation, because work on one stream
// is ordered. Memory allocated on stream A and still being read by a kernel on
// stream B can therefore be reissued to somebody else while B is mid-read.
//
// The rest of this repository never meets that problem: it allocates and uses
// on at::cuda::getCurrentCUDAStream() throughout, so the tag is always right.
// This cache is the first thing here that deliberately runs on a stream of its
// own — the prefetch stream — which makes the mismatch reachable. So the
// allocator takes that stream and makes it current for the at::empty, and the
// tag is right from the start.
//
// The structural allocations need nothing more — bucket heads, the node array,
// the row pool are taken once when the cache is built and released once when it
// is destroyed, so nothing is freed mid-flight. Per-step scratch that is freed
// while another stream still has work queued against it would also need
// at::Tensor::record_stream; nothing allocates that way yet, so there is no hook
// for it here.

#ifdef EMBEDDING_CACHE_HAS_TORCH

class TorchBuffer {
 public:
  TorchBuffer() = default;

  TorchBuffer(size_t bytes, int device_index, cudaStream_t stream) {
    if (bytes == 0) return;
    // Allocated as int64 rather than uint8 so the pointer is 8-byte aligned by
    // construction, which is what the node arrays need. Relying on the caching
    // allocator's internal alignment would work today and is not a promise.
    const int64_t elems = static_cast<int64_t>((bytes + 7) / 8);
    const auto opts =
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index);
    if (stream != nullptr) {
      // Makes the block's stream tag the cache's stream rather than whatever
      // happened to be current on the thread that built the cache.
      const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
          stream, static_cast<c10::DeviceIndex>(device_index)));
      t_ = at::empty({elems}, opts);
    } else {
      t_ = at::empty({elems}, opts);
    }
    bytes_ = bytes;
  }

  void* data() const { return t_.defined() ? t_.data_ptr() : nullptr; }
  size_t bytes() const { return bytes_; }

 private:
  at::Tensor t_;   // reference counted, so this type is copyable as well
  size_t bytes_ = 0;
};

class TorchAllocator {
 public:
  using Buffer = TorchBuffer;

  // -1 means the current device at the time of allocation. A null stream means
  // "whatever is current", which is right for a cache that lives on the default
  // stream and wrong for one that does not — see the note above.
  explicit TorchAllocator(int device_index = -1, cudaStream_t stream = nullptr)
      : device_(device_index), stream_(stream) {}

  Buffer allocate(size_t bytes) const {
    int dev = device_;
    if (dev < 0) {
      cudaGetDevice(&dev);
    }
    return Buffer(bytes, dev, stream_);
  }

  int device_index() const { return device_; }
  cudaStream_t stream() const { return stream_; }

 private:
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

#endif  // EMBEDDING_CACHE_HAS_TORCH

}  // namespace embedding_cache
