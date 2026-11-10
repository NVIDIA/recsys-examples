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

#include "check.h"
#include "counted_ops.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#ifdef DEMB_USE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef DEMB_USE_PYBIND11
namespace py = pybind11;
#endif

namespace dyn_emb {

namespace {

constexpr int kBlock = 256;

// The kernels below move bits: `Word` is an unsigned integer (or uint4) of the
// width the host picked, so none of them knows the dtype. Every loop is
// grid-stride and bounded by n = counted_limit(num_valid, capacity); entries at
// or past n, and those with idx < 0, are neither read nor written.

// One thread per element.
template <typename Word>
__global__ void
counted_gather_1d_kernel(const Word *__restrict__ src, Word *out,
                         const int64_t *__restrict__ idx,
                         const int64_t *num_valid, int64_t capacity) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const int64_t j = idx[i];
    if (j >= 0) {
      out[i] = src[j];
    }
  }
}

template <typename Word>
__global__ void
counted_scatter_1d_kernel(Word *dst, const int64_t *__restrict__ idx,
                          const Word *__restrict__ src,
                          const int64_t *num_valid, int64_t capacity) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const int64_t j = idx[i];
    if (j >= 0) {
      dst[j] = src[i];
    }
  }
}

template <typename Word>
__global__ void
counted_fill_1d_kernel(Word *dst, const int64_t *__restrict__ idx, Word value,
                       const int64_t *num_valid, int64_t capacity) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const int64_t j = idx[i];
    if (j >= 0) {
      dst[j] = value;
    }
  }
}

// 2-D: `tpr` threads (a power of two dividing the block) move one row, so a
// block moves blockDim.x / tpr rows at a time: one row of a wide tensor, many
// of a narrow one. Pitches and the row width are in words.
template <typename Word>
__global__ void counted_gather_rows_kernel(
    const Word *__restrict__ src, Word *out, const int64_t *__restrict__ idx,
    const int64_t *num_valid, int64_t capacity, int64_t row_words,
    int64_t src_pitch, int64_t out_pitch, int tpr) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t rows_per_block = blockDim.x / tpr;
  const int64_t stride = rows_per_block * gridDim.x;
  const int lane = threadIdx.x % tpr;
  for (int64_t i = blockIdx.x * rows_per_block + threadIdx.x / tpr; i < n;
       i += stride) {
    const int64_t j = idx[i];
    if (j < 0) {
      continue;
    }
    const Word *s = src + j * src_pitch;
    Word *d = out + i * out_pitch;
    for (int64_t c = lane; c < row_words; c += tpr) {
      d[c] = s[c];
    }
  }
}

template <typename Word>
__global__ void counted_scatter_rows_kernel(
    Word *dst, const int64_t *__restrict__ idx, const Word *__restrict__ src,
    const int64_t *num_valid, int64_t capacity, int64_t row_words,
    int64_t dst_pitch, int64_t src_pitch, int tpr) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t rows_per_block = blockDim.x / tpr;
  const int64_t stride = rows_per_block * gridDim.x;
  const int lane = threadIdx.x % tpr;
  for (int64_t i = blockIdx.x * rows_per_block + threadIdx.x / tpr; i < n;
       i += stride) {
    const int64_t j = idx[i];
    if (j < 0) {
      continue;
    }
    Word *d = dst + j * dst_pitch;
    const Word *s = src + i * src_pitch;
    for (int64_t c = lane; c < row_words; c += tpr) {
      d[c] = s[c];
    }
  }
}

template <typename Word>
__global__ void
counted_fill_rows_kernel(Word *dst, const int64_t *__restrict__ idx, Word value,
                         const int64_t *num_valid, int64_t capacity,
                         int64_t row_words, int64_t dst_pitch, int tpr) {
  const int64_t n = counted_limit(num_valid, capacity);
  const int64_t rows_per_block = blockDim.x / tpr;
  const int64_t stride = rows_per_block * gridDim.x;
  const int lane = threadIdx.x % tpr;
  for (int64_t i = blockIdx.x * rows_per_block + threadIdx.x / tpr; i < n;
       i += stride) {
    const int64_t j = idx[i];
    if (j < 0) {
      continue;
    }
    Word *d = dst + j * dst_pitch;
    for (int64_t c = lane; c < row_words; c += tpr) {
      d[c] = value;
    }
  }
}

// The summand of counted_count_true: 1 for a set flag among the first n.
struct CountedTrue {
  bool const *flags;
  int64_t const *num_valid;
  int64_t capacity;
  __host__ __device__ __forceinline__ int64_t operator()(int64_t i) const {
    return (i < counted_limit(num_valid, capacity) && flags[i]) ? 1 : 0;
  }
};

// ---- host side -----------------------------------------------------------

bool counted_dtype_supported(at::ScalarType t) {
  switch (t) {
  case at::kBool:
  case at::kChar:
  case at::kInt:
  case at::kLong:
  case at::kUInt64:
  case at::kHalf:
  case at::kBFloat16:
  case at::kFloat:
    return true;
  default:
    return false;
  }
}

// A 1-D tensor, or a 2-D tensor with contiguous rows, as the kernels address
// it.
struct RowsView {
  char *ptr;
  int64_t rows;
  int64_t cols;  // elements per row; 1 for a 1-D tensor
  int64_t pitch; // elements between the starts of two rows
  int elem;      // bytes per element
  bool two_d;
};

RowsView make_rows_view(const at::Tensor &t, const char *what) {
  TORCH_CHECK(t.is_cuda(), what, " must be on CUDA");
  TORCH_CHECK(t.dim() == 1 || t.dim() == 2, what, " must be 1-D or 2-D, got ",
              t.dim(), " dimensions");
  TORCH_CHECK(counted_dtype_supported(t.scalar_type()), what,
              " has an unsupported dtype ", t.scalar_type());
  RowsView v;
  v.ptr = static_cast<char *>(t.data_ptr());
  v.rows = t.size(0);
  v.two_d = t.dim() == 2;
  v.cols = v.two_d ? t.size(1) : 1;
  v.elem = static_cast<int>(t.element_size());
  if (v.two_d) {
    TORCH_CHECK(v.cols <= 1 || t.stride(1) == 1, what,
                " must have contiguous rows (stride(1) == 1)");
    v.pitch = v.rows > 1 ? t.stride(0) : v.cols;
    TORCH_CHECK(v.pitch >= v.cols, what, " must not have overlapping rows");
  } else {
    TORCH_CHECK(v.rows <= 1 || t.stride(0) == 1, what, " must be contiguous");
    v.pitch = 1;
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(v.ptr) % v.elem == 0, what,
              " is not aligned to its element size");
  return v;
}

void check_index(const at::Tensor &idx, const at::Device &device) {
  TORCH_CHECK(idx.is_cuda() && idx.device() == device,
              "idx must be on the device of the data");
  TORCH_CHECK(idx.scalar_type() == at::kLong && idx.dim() == 1 &&
                  (idx.size(0) <= 1 || idx.stride(0) == 1),
              "idx must be a contiguous 1-D int64 tensor");
}

// Blocks of a grid-stride launch over `units` threads or rows: enough to cover
// them, but capped the way the vec4 kernels cap theirs (4 waves of resident
// blocks), so that an empty batch costs one cheap launch, not a block per
// capacity entry.
int counted_grid(int64_t units, int block) {
  const cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
  const int64_t max_grid =
      static_cast<int64_t>(prop->multiProcessorCount) *
      std::max(1, prop->maxThreadsPerMultiProcessor / block) * 4;
  return static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(units, max_grid)));
}

// Launch shape of the 2-D kernels: the threads that move a row (the next power
// of two of its words, at most a block), and the grid covering `capacity` rows.
struct RowsShape {
  int tpr;
  int grid;
};

RowsShape rows_shape(int64_t row_words, int64_t capacity) {
  int tpr = 1;
  while (tpr < row_words && tpr < kBlock) {
    tpr <<= 1;
  }
  const int64_t rows_per_block = kBlock / tpr;
  return {tpr, counted_grid((capacity + rows_per_block - 1) / rows_per_block,
                            kBlock)};
}

// The widest word (16, 8, 4, 2 or 1 bytes) in which every row of the views can
// be moved: a multiple of the element size that divides the row bytes, the row
// pitch and the base address. The element size itself always qualifies.
int rows_word_bytes(std::initializer_list<const RowsView *> views) {
  for (int w = 16;; w >>= 1) {
    bool ok = true;
    for (const RowsView *v : views) {
      ok = ok && w >= v->elem && (v->cols * v->elem) % w == 0 &&
           (v->pitch * v->elem) % w == 0 &&
           reinterpret_cast<uintptr_t>(v->ptr) % w == 0;
    }
    if (ok || w == 1) {
      return w;
    }
  }
}

#define COUNTED_DISPATCH_WORD(BYTES, FUNC, ...)                                \
  switch (BYTES) {                                                             \
  case 16:                                                                     \
    FUNC<uint4>(__VA_ARGS__);                                                  \
    break;                                                                     \
  case 8:                                                                      \
    FUNC<uint64_t>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 4:                                                                      \
    FUNC<uint32_t>(__VA_ARGS__);                                               \
    break;                                                                     \
  case 2:                                                                      \
    FUNC<uint16_t>(__VA_ARGS__);                                               \
    break;                                                                     \
  default:                                                                     \
    FUNC<uint8_t>(__VA_ARGS__);                                                \
    break;                                                                     \
  }

template <typename Word>
void run_gather(const RowsView &src, const RowsView &out, const int64_t *idx,
                const int64_t *num_valid, int64_t capacity,
                cudaStream_t stream) {
  constexpr int64_t kWord = sizeof(Word);
  if (!src.two_d) {
    const int grid = counted_grid((capacity + kBlock - 1) / kBlock, kBlock);
    counted_gather_1d_kernel<Word><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<const Word *>(src.ptr),
        reinterpret_cast<Word *>(out.ptr), idx, num_valid, capacity);
  } else {
    const int64_t row_words = src.cols * src.elem / kWord;
    const RowsShape shape = rows_shape(row_words, capacity);
    counted_gather_rows_kernel<Word><<<shape.grid, kBlock, 0, stream>>>(
        reinterpret_cast<const Word *>(src.ptr),
        reinterpret_cast<Word *>(out.ptr), idx, num_valid, capacity, row_words,
        src.pitch * src.elem / kWord, out.pitch * out.elem / kWord, shape.tpr);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Word>
void run_scatter(const RowsView &dst, const RowsView &src, const int64_t *idx,
                 const int64_t *num_valid, int64_t capacity,
                 cudaStream_t stream) {
  constexpr int64_t kWord = sizeof(Word);
  if (!dst.two_d) {
    const int grid = counted_grid((capacity + kBlock - 1) / kBlock, kBlock);
    counted_scatter_1d_kernel<Word><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<Word *>(dst.ptr), idx,
        reinterpret_cast<const Word *>(src.ptr), num_valid, capacity);
  } else {
    const int64_t row_words = dst.cols * dst.elem / kWord;
    const RowsShape shape = rows_shape(row_words, capacity);
    counted_scatter_rows_kernel<Word><<<shape.grid, kBlock, 0, stream>>>(
        reinterpret_cast<Word *>(dst.ptr), idx,
        reinterpret_cast<const Word *>(src.ptr), num_valid, capacity, row_words,
        dst.pitch * dst.elem / kWord, src.pitch * src.elem / kWord, shape.tpr);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// `bits` is the element's bit pattern in its low bytes; the word repeats it.
template <typename Word> Word fill_pattern(uint64_t bits, int elem) {
  unsigned char bytes[16];
  for (int b = 0; b < 16; ++b) {
    bytes[b] = static_cast<unsigned char>(bits >> (8 * (b % elem)));
  }
  Word word;
  std::memcpy(&word, bytes, sizeof(Word));
  return word;
}

template <typename Word>
void run_fill(const RowsView &dst, const int64_t *idx, uint64_t bits,
              const int64_t *num_valid, int64_t capacity, cudaStream_t stream) {
  constexpr int64_t kWord = sizeof(Word);
  const Word value = fill_pattern<Word>(bits, dst.elem);
  if (!dst.two_d) {
    const int grid = counted_grid((capacity + kBlock - 1) / kBlock, kBlock);
    counted_fill_1d_kernel<Word><<<grid, kBlock, 0, stream>>>(
        reinterpret_cast<Word *>(dst.ptr), idx, value, num_valid, capacity);
  } else {
    const int64_t row_words = dst.cols * dst.elem / kWord;
    const RowsShape shape = rows_shape(row_words, capacity);
    counted_fill_rows_kernel<Word><<<shape.grid, kBlock, 0, stream>>>(
        reinterpret_cast<Word *>(dst.ptr), idx, value, num_valid, capacity,
        row_words, dst.pitch * dst.elem / kWord, shape.tpr);
  }
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// `value` as an integer of type T. The bounds are exact in a double except the
// top one of a 64-bit type, which rounds up to 2^63 / 2^64: that value and the
// integers that round to it become the largest T.
template <typename T> T fill_integer(double value) {
  const double lo = static_cast<double>(std::numeric_limits<T>::lowest());
  const double hi = static_cast<double>(std::numeric_limits<T>::max());
  TORCH_CHECK(value >= lo && value <= hi, "counted_scatter_fill: value ", value,
              " does not fit the dtype of dst");
  return value >= hi ? std::numeric_limits<T>::max() : static_cast<T>(value);
}

// The bit pattern, in the low bytes, of `value` converted to `dtype`.
uint64_t fill_bits(at::ScalarType dtype, double value) {
  switch (dtype) {
  case at::kBool:
    return value != 0.0 ? 1 : 0;
  case at::kChar:
    return static_cast<uint8_t>(fill_integer<int8_t>(value));
  case at::kInt:
    return static_cast<uint32_t>(fill_integer<int32_t>(value));
  case at::kLong:
    return static_cast<uint64_t>(fill_integer<int64_t>(value));
  case at::kUInt64:
    return fill_integer<uint64_t>(value);
  case at::kHalf:
    return c10::Half(static_cast<float>(value)).x;
  case at::kBFloat16:
    return c10::BFloat16(static_cast<float>(value)).x;
  case at::kFloat: {
    const float f = static_cast<float>(value);
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
  }
  default:
    TORCH_CHECK(false, "counted_scatter_fill: unsupported dtype ", dtype);
  }
  return 0;
}

} // namespace

at::Tensor counted_gather(at::Tensor src, at::Tensor idx,
                          std::optional<at::Tensor> num_valid,
                          std::optional<at::Tensor> out) {
  const RowsView s = make_rows_view(src, "src");
  const at::Device device = src.device();
  check_index(idx, device);
  c10::cuda::CUDAGuard device_guard(device);
  const int64_t capacity = idx.size(0);
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, device);

  at::Tensor result;
  if (out.has_value()) {
    result = out.value();
    TORCH_CHECK(result.device() == device, "out must be on the device of src");
    TORCH_CHECK(result.scalar_type() == src.scalar_type(),
                "out must have the dtype of src");
    TORCH_CHECK(result.dim() == src.dim(),
                "out must have as many dimensions as src");
  } else {
    std::vector<int64_t> shape = src.sizes().vec();
    shape[0] = capacity;
    result = at::empty(shape, src.options());
  }
  const RowsView o = make_rows_view(result, "out");
  TORCH_CHECK(o.rows >= capacity, "out has fewer rows (", o.rows,
              ") than idx (", capacity, ")");
  TORCH_CHECK(o.cols == s.cols, "out and src must have rows of the same width");
  if (capacity == 0 || s.cols == 0) {
    return result;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t *d_idx = idx.data_ptr<int64_t>();
  const int word_bytes = s.two_d ? rows_word_bytes({&s, &o}) : s.elem;
  COUNTED_DISPATCH_WORD(word_bytes, run_gather, s, o, d_idx, d_num_valid,
                        capacity, stream);
  return result;
}

void counted_scatter(at::Tensor dst, at::Tensor idx, at::Tensor src,
                     std::optional<at::Tensor> num_valid) {
  const RowsView d = make_rows_view(dst, "dst");
  const RowsView s = make_rows_view(src, "src");
  const at::Device device = dst.device();
  TORCH_CHECK(src.device() == device, "src must be on the device of dst");
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(),
              "src must have the dtype of dst");
  TORCH_CHECK(src.dim() == dst.dim() && s.cols == d.cols,
              "src and dst must have rows of the same width");
  check_index(idx, device);
  c10::cuda::CUDAGuard device_guard(device);
  const int64_t capacity = idx.size(0);
  TORCH_CHECK(s.rows >= capacity, "src has fewer rows (", s.rows,
              ") than idx (", capacity, ")");
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, device);
  if (capacity == 0 || d.cols == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t *d_idx = idx.data_ptr<int64_t>();
  const int word_bytes = d.two_d ? rows_word_bytes({&d, &s}) : d.elem;
  COUNTED_DISPATCH_WORD(word_bytes, run_scatter, d, s, d_idx, d_num_valid,
                        capacity, stream);
}

void counted_scatter_fill(at::Tensor dst, at::Tensor idx, double value,
                          std::optional<at::Tensor> num_valid) {
  const RowsView d = make_rows_view(dst, "dst");
  const at::Device device = dst.device();
  check_index(idx, device);
  c10::cuda::CUDAGuard device_guard(device);
  const int64_t capacity = idx.size(0);
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, device);
  const uint64_t bits = fill_bits(dst.scalar_type(), value);
  if (capacity == 0 || d.cols == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t *d_idx = idx.data_ptr<int64_t>();
  const int word_bytes = d.two_d ? rows_word_bytes({&d}) : d.elem;
  COUNTED_DISPATCH_WORD(word_bytes, run_fill, d, d_idx, bits, d_num_valid,
                        capacity, stream);
}

at::Tensor counted_count_true(at::Tensor flags,
                              std::optional<at::Tensor> num_valid) {
  TORCH_CHECK(flags.is_cuda(), "flags must be on CUDA");
  TORCH_CHECK(flags.scalar_type() == at::kBool && flags.dim() == 1 &&
                  flags.is_contiguous(),
              "flags must be a contiguous 1-D bool tensor");
  const at::Device device = flags.device();
  c10::cuda::CUDAGuard device_guard(device);
  const int64_t capacity = flags.size(0);
  const int64_t *d_num_valid = counted_num_valid_ptr(num_valid, device);
  const auto i64 = at::TensorOptions().dtype(at::kLong).device(device);
  if (capacity == 0) {
    return at::zeros({1}, i64);
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor count = at::empty({1}, i64);
  auto summands = thrust::make_transform_iterator(
      thrust::counting_iterator<int64_t>(0),
      CountedTrue{flags.data_ptr<bool>(), d_num_valid, capacity});
  void *d_temp = nullptr;
  size_t temp_bytes = 0;
  cub::DeviceReduce::Sum(d_temp, temp_bytes, summands,
                         count.data_ptr<int64_t>(), capacity, stream);
  at::Tensor temp_storage =
      at::empty({static_cast<int64_t>(std::max<size_t>(temp_bytes, 1))},
                at::TensorOptions().dtype(at::kByte).device(device));
  cub::DeviceReduce::Sum(temp_storage.data_ptr(), temp_bytes, summands,
                         count.data_ptr<int64_t>(), capacity, stream);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return count;
}

#undef COUNTED_DISPATCH_WORD

} // namespace dyn_emb

#ifdef DEMB_USE_PYBIND11
void bind_counted_ops(py::module &m) {
  m.def("counted_gather", &dyn_emb::counted_gather,
        "out[i] = src[idx[i]] for the live i (i < n, n = clamp(*num_valid, 0, "
        "len(idx)); None: len(idx)) with idx[i] >= 0. src is 1-D or 2-D (whole "
        "rows); out (allocated empty when None, shape [len(idx), ...]) is not "
        "written elsewhere. No host sync.",
        py::arg("src"), py::arg("idx"), py::arg("num_valid"),
        py::arg("out") = py::none());

  m.def(
      "counted_scatter", &dyn_emb::counted_scatter,
      "dst[idx[i]] = src[i] for the live i with idx[i] >= 0 (1-D or 2-D whole "
      "rows). A repeated index: an arbitrary source wins. No host sync.",
      py::arg("dst"), py::arg("idx"), py::arg("src"), py::arg("num_valid"));

  m.def(
      "counted_scatter_fill", &dyn_emb::counted_scatter_fill,
      "dst[idx[i]] = value for the live i with idx[i] >= 0; value (float, int "
      "or bool) is converted to the dtype of dst, an error if it does not fit. "
      "No host sync.",
      py::arg("dst"), py::arg("idx"), py::arg("value"), py::arg("num_valid"));

  m.def("counted_count_true", &dyn_emb::counted_count_true,
        "The number of true flags among the first n = clamp(*num_valid, 0, "
        "len(flags)) (None: len(flags)), as an int64[1] device tensor. No host "
        "sync.",
        py::arg("flags"), py::arg("num_valid"));
}
#endif
