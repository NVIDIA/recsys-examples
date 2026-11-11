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

#include "counted_ops.h"
#include "initializer.cuh"

namespace py = pybind11;

namespace dyn_emb {

__global__ void init_curand_state_kernel(unsigned long long seed,
                                         curandState *states) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &states[grid.thread_rank()]);
}

class CurandStateContext {

public:
  CurandStateContext() {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto &deviceProp = DeviceProp::getDeviceProp();
    num_worker_ = deviceProp.total_threads;
    CUDACHECK(
        cudaMallocAsync(&states_, sizeof(curandState) * num_worker_, stream));
    std::random_device rd;
    auto seed = rd();
    int block_size = deviceProp.max_thread_per_block;
    int grid_size = num_worker_ / block_size;
    init_curand_state_kernel<<<grid_size, block_size, 0, stream>>>(seed,
                                                                   states_);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  ~CurandStateContext() {
    // not async to avoid stream destroy case.
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(states_));
  }

  int64_t num_worker() { return num_worker_; }

  curandState *ptr() { return states_; }

private:
  curandState *states_;
  int64_t num_worker_;
};

// Entry i of the batch is row indices[i] of the buffer, or row i itself when
// `indices` is null.  With `num_valid`, only the first n = clamp(*num_valid, 0,
// num) entries are live: the rest are undefined and are neither read nor
// written.  The generator reads its per-row parameters (and a debug key) by the
// buffer row, which is why those run alongside the buffer, not the indices.
template <typename ValueT, typename IndexT, typename GeneratorT>
__global__ void initialize_with_index_addressor_kernel(
    int64_t num, int64_t dim, int64_t stide, ValueT *__restrict__ buffer,
    IndexT const *__restrict__ indices, typename GeneratorT::Args generator_args,
    int64_t const *__restrict__ num_valid) {

  GeneratorT gen(generator_args);
  int64_t num_task = counted_limit(num_valid, num) * dim;
  int64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; task_id < num_task; task_id += gridDim.x * blockDim.x) {
    int64_t emb_id = task_id / dim;
    int64_t index =
        indices != nullptr ? static_cast<int64_t>(indices[emb_id]) : emb_id;
    if (index < 0)
      continue;
    ValueT *dst = buffer + index * stide;
    auto tmp = gen.generate(index);
    dst[task_id % dim] = TypeConvertFunc<ValueT, float>::convert(tmp);
  }
  gen.destroy();
}

// Everything an initializer reads is dereferenced by the kernel, so a tensor
// left on the host, or on another device, is an illegal access rather than a
// wrong answer. Say which tensor it was before launching.
static void check_beside_buffer(const at::Tensor &buffer,
                                const at::Tensor &tensor, const char *name) {
  if (!tensor.is_cuda()) {
    throw std::invalid_argument(std::string("Initializer's ") + name +
                                " have to be a CUDA tensor.");
  }
  if (tensor.device() != buffer.device()) {
    throw std::invalid_argument(
        std::string("Initializer's ") + name +
        " have to be on the same device as the value buffer.");
  }
}

// `indices` selects the rows of `buffer` to write, in order; left out (None, or
// an undefined tensor) it means the rows 0..n-1, n = num_valid, which is then
// required.  With num_valid, only the first n entries of `indices` are used --
// the rest are undefined, possibly out of range, and never read.
template <typename GeneratorT>
void initialize_with_generator(at::Tensor buffer,
                               const std::optional<at::Tensor> &indices,
                               const std::optional<at::Tensor> &num_valid,
                               typename GeneratorT::Args generator_args,
                               int64_t num_worker = -1) {
  int64_t num_dims = buffer.dim();
  if (num_dims != 2) {
    throw std::runtime_error("Initializer'input buffer's dim have to be 2.");
  }
  if (buffer.stride(1) != 1) {
    throw std::runtime_error(
        "Initializer'input buffer has to be contiguous at dim1.");
  }
  if (!buffer.is_cuda()) {
    throw std::invalid_argument(
        "Initializer's value buffer have to be a CUDA tensor.");
  }
  const bool has_indices = indices.has_value() && indices.value().defined();
  const int64_t *d_num_valid =
      counted_num_valid_ptr(num_valid, buffer.device());
  if (has_indices) {
    check_beside_buffer(buffer, indices.value(), "indices");
  } else if (d_num_valid == nullptr) {
    throw std::invalid_argument(
        "Initializer needs indices or num_valid: without indices it writes the "
        "first num_valid rows of the buffer.");
  }
  int64_t num_total = has_indices ? indices.value().size(0) : buffer.size(0);
  int64_t dim = buffer.size(1);
  int64_t stride = buffer.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto &deviceProp = DeviceProp::getDeviceProp();

  int64_t block_size = deviceProp.max_thread_per_block;
  int64_t num_need = num_total * dim;
  if (num_worker == -1) {
    num_worker = deviceProp.total_threads;
  }
  int64_t max_grid_size = num_worker / block_size;
  if (num_worker > num_need) {
    num_worker = num_need;
  }
  int64_t grid_size = (num_worker - 1) / block_size + 1;
  if (grid_size > max_grid_size) {
    grid_size = max_grid_size;
  }

  auto value_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(buffer.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    if (!has_indices) {
      // Entry i is row i of the buffer.
      initialize_with_index_addressor_kernel<ValueType, int64_t, GeneratorT>
          <<<grid_size, block_size, 0, stream>>>(
              num_total, dim, stride,
              reinterpret_cast<ValueType *>(buffer.data_ptr()),
              static_cast<const int64_t *>(nullptr), generator_args,
              d_num_valid);
      return;
    }
    auto index_type = scalartype_to_datatype(
        convertTypeMetaToScalarType(indices.value().dtype()));
    DISPATCH_INTEGER_DATATYPE_FUNCTION(index_type, IndexType, [&] {
      initialize_with_index_addressor_kernel<ValueType, IndexType, GeneratorT>
          <<<grid_size, block_size, 0, stream>>>(
              num_total, dim, stride,
              reinterpret_cast<ValueType *>(buffer.data_ptr()),
              reinterpret_cast<IndexType *>(indices.value().data_ptr()),
              generator_args, d_num_valid);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

// The pointers a per-table generator reads. ``table_params`` is
// [num_tables, num_params]; ``table_ids`` says which table owns each row of
// the value buffer, so it runs alongside that buffer, as ``keys`` does, while
// ``indices`` picks out the rows to write.
struct TableParamPtrs {
  const float *args;
  const int64_t *ids;
};

static TableParamPtrs check_table_params(const at::Tensor &buffer,
                                         const at::Tensor &table_params,
                                         const at::Tensor &table_ids,
                                         int64_t num_params) {
  check_beside_buffer(buffer, table_params, "table_params");
  check_beside_buffer(buffer, table_ids, "table_ids");
  if (table_params.scalar_type() != at::kFloat) {
    throw std::invalid_argument(
        "Initializer's table_params have to be float32.");
  }
  if (table_params.dim() != 2 || table_params.size(1) != num_params) {
    throw std::invalid_argument(
        "Initializer's table_params have to be [num_tables, num_params].");
  }
  if (!table_params.is_contiguous()) {
    throw std::invalid_argument(
        "Initializer's table_params have to be contiguous.");
  }
  if (table_ids.scalar_type() != at::kLong) {
    throw std::invalid_argument("Initializer's table_ids have to be int64.");
  }
  if (!table_ids.is_contiguous()) {
    throw std::invalid_argument(
        "Initializer's table_ids have to be contiguous.");
  }
  if (table_ids.size(0) != buffer.size(0)) {
    throw std::invalid_argument(
        "Initializer's table_ids have to run alongside the value buffer, one "
        "per row of it -- not alongside the indices selecting rows.");
  }
  return TableParamPtrs{static_cast<const float *>(table_params.data_ptr()),
                        static_cast<const int64_t *>(table_ids.data_ptr())};
}

// Every initializer takes `indices` as optional and a trailing `num_valid`; see
// initialize_with_generator.
void normal_init(at::Tensor buffer, const std::optional<at::Tensor> &indices,
                 CurandStateContext &curand_state_context, float mean,
                 float std_dev,
                 const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = NormalEmbeddingGenerator<false>;
  typename GeneratorT::Params params{{mean, std_dev}};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void normal_init_table_params(
    at::Tensor buffer, const std::optional<at::Tensor> &indices,
    CurandStateContext &curand_state_context, at::Tensor table_params,
    at::Tensor table_ids,
    const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = NormalEmbeddingGenerator<true>;
  auto ptrs = check_table_params(buffer, table_params, table_ids,
                                 GeneratorT::kNumParams);
  typename GeneratorT::Params params{ptrs.args, ptrs.ids};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void truncated_normal_init(
    at::Tensor buffer, const std::optional<at::Tensor> &indices,
    CurandStateContext &curand_state_context, float mean, float std_dev,
    float lower, float upper,
    const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = TruncatedNormalEmbeddingGenerator<false>;
  typename GeneratorT::Params params{{mean, std_dev, lower, upper}};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void truncated_normal_init_table_params(
    at::Tensor buffer, const std::optional<at::Tensor> &indices,
    CurandStateContext &curand_state_context, at::Tensor table_params,
    at::Tensor table_ids,
    const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = TruncatedNormalEmbeddingGenerator<true>;
  auto ptrs = check_table_params(buffer, table_params, table_ids,
                                 GeneratorT::kNumParams);
  typename GeneratorT::Params params{ptrs.args, ptrs.ids};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void uniform_init(at::Tensor buffer, const std::optional<at::Tensor> &indices,
                  CurandStateContext &curand_state_context, float lower,
                  float upper,
                  const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = UniformEmbeddingGenerator<false>;
  typename GeneratorT::Params params{{lower, upper}};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void uniform_init_table_params(
    at::Tensor buffer, const std::optional<at::Tensor> &indices,
    CurandStateContext &curand_state_context, at::Tensor table_params,
    at::Tensor table_ids,
    const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = UniformEmbeddingGenerator<true>;
  auto ptrs = check_table_params(buffer, table_params, table_ids,
                                 GeneratorT::kNumParams);
  typename GeneratorT::Params params{ptrs.args, ptrs.ids};
  auto generator_args =
      typename GeneratorT::Args{curand_state_context.ptr(), params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args,
                                        curand_state_context.num_worker());
}

void const_init(at::Tensor buffer, const std::optional<at::Tensor> &indices,
                float value,
                const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = ConstEmbeddingGenerator<false>;
  typename GeneratorT::Params params{{value}};
  auto generator_args = typename GeneratorT::Args{params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args);
}

void const_init_table_params(
    at::Tensor buffer, const std::optional<at::Tensor> &indices,
    at::Tensor table_params, at::Tensor table_ids,
    const std::optional<at::Tensor> &num_valid = std::nullopt) {
  using GeneratorT = ConstEmbeddingGenerator<true>;
  auto ptrs = check_table_params(buffer, table_params, table_ids,
                                 GeneratorT::kNumParams);
  typename GeneratorT::Params params{ptrs.args, ptrs.ids};
  auto generator_args = typename GeneratorT::Args{params};
  initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                        generator_args);
}

void debug_init(at::Tensor buffer, const std::optional<at::Tensor> &indices,
                at::Tensor keys,
                const std::optional<at::Tensor> &num_valid = std::nullopt) {
  check_beside_buffer(buffer, keys, "keys");
  if (keys.size(0) != buffer.size(0)) {
    throw std::invalid_argument(
        "Initializer's keys have to run alongside the value buffer, one per "
        "row of it -- not alongside the indices selecting rows.");
  }
  auto key_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(keys.dtype()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    using GeneratorT = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename GeneratorT::Args{
        reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
    initialize_with_generator<GeneratorT>(buffer, indices, num_valid,
                                          generator_args);
  });
}

} // namespace dyn_emb

void bind_initializer_op(py::module &m) {

  py::class_<dyn_emb::CurandStateContext>(m, "CurandStateContext")
      .def(py::init<>())
      .def("ptr", &dyn_emb::CurandStateContext::ptr,
           py::return_value_policy::reference);

  // Each mode comes in two forms. The plain one takes the parameters every
  // table of a fused module shares; the ``_table_params`` one takes a
  // [num_tables, num_params] table of them plus the table each buffer row
  // belongs to. Both serve the same multi-table buffer -- only the parameters
  // differ. They are separate entry points, and so separate kernels, so the
  // shared form carries nothing of the other's.
  //
  // `indices` may be None: the rows 0..n-1 of the buffer are then written, n
  // being `num_valid`, which is required in that case.  With `num_valid`, only
  // the first n entries of `indices` are used.  The per-table `table_ids` (and
  // the debug `keys`) run alongside the rows of the buffer either way.
  m.def("normal_init", &dyn_emb::normal_init, "Normal initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"),
        py::arg("mean"), py::arg("std_dev"), py::arg("num_valid") = py::none());

  m.def("normal_init_table_params", &dyn_emb::normal_init_table_params,
        "Normal initializer, parameters one row per table", py::arg("buffer"),
        py::arg("indices"), py::arg("curand_state_context"),
        py::arg("table_params"), py::arg("table_ids"),
        py::arg("num_valid") = py::none());

  m.def("truncated_normal_init", &dyn_emb::truncated_normal_init,
        "Truncated normal initializer", py::arg("buffer"), py::arg("indices"),
        py::arg("curand_state_context"), py::arg("mean"), py::arg("std_dev"),
        py::arg("lower"), py::arg("upper"), py::arg("num_valid") = py::none());

  m.def("truncated_normal_init_table_params",
        &dyn_emb::truncated_normal_init_table_params,
        "Truncated normal initializer, parameters one row per table",
        py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"),
        py::arg("table_params"), py::arg("table_ids"),
        py::arg("num_valid") = py::none());

  m.def("uniform_init", &dyn_emb::uniform_init, "Uniform initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"),
        py::arg("lower"), py::arg("upper"), py::arg("num_valid") = py::none());

  m.def("uniform_init_table_params", &dyn_emb::uniform_init_table_params,
        "Uniform initializer, parameters one row per table", py::arg("buffer"),
        py::arg("indices"), py::arg("curand_state_context"),
        py::arg("table_params"), py::arg("table_ids"),
        py::arg("num_valid") = py::none());

  m.def("const_init", &dyn_emb::const_init, "Const initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("value"),
        py::arg("num_valid") = py::none());

  m.def("const_init_table_params", &dyn_emb::const_init_table_params,
        "Const initializer, parameters one row per table", py::arg("buffer"),
        py::arg("indices"), py::arg("table_params"), py::arg("table_ids"),
        py::arg("num_valid") = py::none());

  m.def("debug_init", &dyn_emb::debug_init, "Debug initializer",
        py::arg("buffer"), py::arg("indices"), py::arg("keys"),
        py::arg("num_valid") = py::none());
}
