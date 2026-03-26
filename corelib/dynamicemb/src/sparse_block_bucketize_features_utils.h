/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#pragma once

#ifdef DYNAMICEMB_GPU_ENABLE_DUMMY_IA32_SERIALIZE
// Workaround the missing __builtin_ia32_serialize issue
#if defined(__NVCC__) &&                                                       \
    (__CUDACC_VER_MAJOR__ > 11 || __CUDACC_VER_MINOR__ >= 4)
#if defined(__i386__) || defined(__i686__) || defined(__x86_64__)
static __inline void __attribute__((__gnu_inline__, __always_inline__,
                                    __artificial__, __target__("serialize")))
__builtin_ia32_serialize(void) {
  abort();
}
#endif
#endif // __NVCC__
#endif // DYNAMICEMB_GPU_ENABLE_DUMMY_IA32_SERIALIZE

#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>

#define DLL_PUBLIC __attribute__((visibility("default")))

#define DYNAMICEMB_OP_DISPATCH(DISPATCH_KEY, EXPORT_NAME, FUNC_NAME)           \
  TORCH_LIBRARY_IMPL(dyn_emb, DISPATCH_KEY, m) {                               \
    m.impl(EXPORT_NAME, torch::dispatch(c10::DispatchKey::DISPATCH_KEY,        \
                                        TORCH_FN(FUNC_NAME)));                 \
  }

#define DYNAMICEMB_GPU_CUB_NS_PREFIX dyn_emb::

#define DYNAMICEMB_DISPATCH_TORCH_FLOATING_TYPES_CASE(...)                     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                          \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DYNAMICEMB_DISPATCH_TORCH_INTEGRAL_TYPES_CASE(...)                     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DYNAMICEMB_DISPATCH_TORCH_ALL_TYPES(TYPE, NAME, ...)                   \
  AT_DISPATCH_SWITCH(                                                          \
      TYPE, NAME,                                                              \
      DYNAMICEMB_DISPATCH_TORCH_FLOATING_TYPES_CASE(__VA_ARGS__)               \
          DYNAMICEMB_DISPATCH_TORCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

namespace dyn_emb {

inline std::optional<int64_t>
get_device_index_from_tensor(const at::Tensor &ten) {
  return {ten.device().index()};
}

inline std::optional<int64_t>
get_device_index_from_tensor(const c10::optional<at::Tensor> &ten) {
  if (ten) {
    return {ten->device().index()};
  } else {
    return {};
  }
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor &ten) {
  return ten.is_cuda();
}

inline bool
torch_tensor_on_cuda_gpu_check(const c10::optional<at::Tensor> &ten) {
  return !ten.has_value() || torch_tensor_on_cuda_gpu_check(ten.value());
}

inline bool torch_tensor_undefined(const at::Tensor &ten) {
  return ten.defined();
}

inline bool torch_tensor_undefined(const c10::optional<at::Tensor> &ten) {
  return !ten.has_value() || torch_tensor_undefined(ten.value());
}

inline std::string torch_tensor_device_name(const at::Tensor &ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string
torch_tensor_device_name(const c10::optional<at::Tensor> &ten) {
  if (ten.has_value()) {
    return torch_tensor_device_name(ten.value());
  } else {
    return "N/A";
  }
}

template <typename... Tensors>
std::string
tensor_on_same_gpu_if_not_optional_check(const std::string &var_names_str,
                                         const Tensors &...tensors) {
  std::optional<int64_t> gpu_index;
  bool on_same_gpu = true;

  // Collect the GPU index of the first non-empty optional tensor and make sure
  // that all tensors are on this same index.
  (
      [&](const auto &tensor) {
        if (!torch_tensor_undefined(tensor)) {
          return;
        }
        if (!torch_tensor_on_cuda_gpu_check(tensor)) {
          on_same_gpu = false;
          return;
        }
        const auto my_gpu_index = get_device_index_from_tensor(tensor);
        if (my_gpu_index) {
          if (!gpu_index) {
            gpu_index = my_gpu_index;
          } else if (*gpu_index != my_gpu_index) {
            on_same_gpu = false;
          }
        }
      }(tensors),
      ...);

  if (on_same_gpu) {
    return "";
  }

  std::vector<std::string> var_names;
  {
    std::string temp = "";
    for (const auto &x : var_names_str) {
      if (x == ',') {
        var_names.push_back(temp);
        temp = "";
      } else {
        temp.push_back(x);
      }
    }
    var_names.push_back(temp);
  }

  // Not all the tensors on a GPU or on the same GPU, generate a message.
  std::string msg = "Not all tensors were on the same GPU: ";
  size_t current_idx = 0;
  (
      [&](const auto &tensor) {
        if (current_idx > 0) {
          msg.append(", ");
        }
        msg.append(var_names.at(current_idx++) + "(" +
                   torch_tensor_device_name(tensor));
        const auto gpu_device_index = get_device_index_from_tensor(tensor);
        if (gpu_device_index) {
          msg.append(":" + std::to_string(*gpu_device_index));
        }
        msg.append(")");
      }(tensors),
      ...);

  return msg;
}

inline bool torch_tensor_on_cpu_check(const at::Tensor &ten) {
  return ten.is_cpu();
}

inline bool torch_tensor_on_cpu_check(const c10::optional<at::Tensor> &ten) {
  return !ten.has_value() || torch_tensor_on_cpu_check(ten.value());
}

#define TENSOR_ON_CPU(x)                                                       \
  TORCH_CHECK(torch_tensor_on_cpu_check(x),                                    \
              #x " must be a CPU tensor; it is currently on device ",          \
              torch_tensor_device_name(x))

#define TENSOR_ON_CUDA_GPU(x)                                                  \
  TORCH_CHECK(torch_tensor_on_cuda_gpu_check(x),                               \
              #x " must be a CUDA tensor; it is currently on device ",         \
              torch_tensor_device_name(x))

// Generate constexpr array of variable names to improve diagnostic output and
// raise a message if any non-empty tensor is not on a GPU or not on the same
// GPU as all the other non-empty tensors.
#define TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(...)                          \
  do {                                                                         \
    const auto tensors_on_same_gpu =                                           \
        tensor_on_same_gpu_if_not_optional_check(#__VA_ARGS__, __VA_ARGS__);   \
    TORCH_CHECK(tensors_on_same_gpu.empty(), tensors_on_same_gpu);             \
  } while (false)

#define CUDA_DEVICE_GUARD(TENSOR)                                              \
  at::cuda::OptionalCUDAGuard device_guard;                                    \
  device_guard.set_index(TENSOR.get_device())

#define DYNAMICEMB_AT_DISPATCH_FLOAT_ONLY(TYPE, NAME, ...)                     \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                               \
                     AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__))

static constexpr int32_t kMaxThreads = 1024;
static int const kWarpSize = 32;

/// Function is designed to facilitate run-time value checking.
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_xblock_count_base(Integer1 num_items,
                                               Integer2 threads_per_block) {
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that for compute capability 3.5-* the grid dimension of a kernel
  // launch must must be <=2^31-1.
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

// See: cuda_calc_xblock_count_base
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value &&
                               std::is_signed<Integer2>::value,
                           bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value &&
                               std::is_unsigned<Integer2>::value,
                           bool> = true>
constexpr uint32_t cuda_calc_xblock_count(Integer1 num_items,
                                          Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value &&
                               std::is_unsigned<Integer2>::value,
                           bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value &&
                               std::is_signed<Integer2>::value,
                           bool> = true>
constexpr uint32_t cuda_calc_xblock_count(Integer1 num_items,
                                          Integer2 threads_per_block) {
  TORCH_CHECK(threads_per_block >= 0, "When calculating thread counts, the "
                                      "number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value &&
                               std::is_signed<Integer2>::value,
                           bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value &&
                               std::is_signed<Integer2>::value,
                           bool> = true>
constexpr uint32_t cuda_calc_xblock_count(Integer1 num_items,
                                          Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  TORCH_CHECK(threads_per_block >= 0, "When calculating thread counts, the "
                                      "number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value &&
                               std::is_unsigned<Integer2>::value,
                           bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value &&
                               std::is_unsigned<Integer2>::value,
                           bool> = true>
constexpr uint32_t cuda_calc_xblock_count(Integer1 num_items,
                                          Integer2 threads_per_block) {
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

/// Determine an appropriate CUDA block count.
///
/// See cuda_calc_xblock_count_base() for details.
template <typename Integer1, typename Integer2,
          std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
          std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_block_count(Integer1 num_items,
                                         Integer2 threads_per_block) {
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that the grid dimension of a kernel launch must generally
  // be <=65535. (For compute capability 3.5-* the grid's x-dimension must
  // be <=2^31-1.) Because this function does not know which dimension
  // is being calculated, we use the smaller limit.
  constexpr uint32_t max_blocks = 65535;
  return std::min(cuda_calc_xblock_count(num_items, threads_per_block),
                  max_blocks);
}

} // namespace dyn_emb
