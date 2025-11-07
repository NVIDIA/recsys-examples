/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

#include "initializer.cuh"

namespace py = pybind11;

namespace dyn_emb {

__global__ void init_curand_state_kernel(
  unsigned long long seed,
  curandState *states
) {
  auto grid = cooperative_groups::this_grid();
  curand_init(seed, grid.thread_rank(), 0, &states[grid.thread_rank()]);
}

class CurandStateContext {

public:
  CurandStateContext() {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto &deviceProp = DeviceProp::getDeviceProp();
    num_worker_ = deviceProp.total_threads;
    CUDACHECK(cudaMallocAsync(
      &states_, sizeof(curandState) * num_worker_, stream));
    std::random_device rd;
    auto seed = rd();
    int block_size = deviceProp.max_thread_per_block;
    int grid_size = num_worker_ / block_size;
    init_curand_state_kernel<<<grid_size, block_size, 0, stream>>>(seed, states_);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  ~CurandStateContext() {
    // not async to avoid stream destroy case.
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaFree(states_));
  }

  int64_t num_worker() {
    return num_worker_;
  }

  curandState* ptr() { return states_; }

private:
  curandState* states_;
  int64_t num_worker_;
};

template <typename ValueT, typename IndexT, typename GeneratorT>
__global__ void initialize_with_index_addressor_kernel(
  int64_t num,
  int64_t dim,
  int64_t stide,
  ValueT * __restrict__ buffer,
  IndexT const * __restrict__ indices,
  typename GeneratorT::Args generator_args
) {

  GeneratorT gen(generator_args);
  int64_t num_task = num * dim;
  int64_t task_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (; task_id < num_task; task_id += gridDim.x * blockDim.x) {
    int64_t emb_id = task_id / dim;
    int64_t index = indices[emb_id];
    ValueT * dst = buffer + index * stide;
    auto tmp = gen.generate(index);
    dst[task_id % dim] = TypeConvertFunc<ValueT, float>::convert(tmp);
  }
  gen.destroy();
}

template <typename GeneratorT>
void initialize_with_generator(
  at::Tensor buffer, 
  at::Tensor indices, 
  typename GeneratorT::Args generator_args,
  int64_t num_worker = -1
) {
  int64_t num_dims = buffer.dim();
  if (num_dims != 2) {
    throw std::runtime_error("Initializer'input buffer's dim have to be 2.");
  }
  if (buffer.stride(1) != 1) {
    throw std::runtime_error("Initializer'input buffer has to be contiguous at dim1.");
  }
  int64_t num_total = indices.size(0);
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
  auto index_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(indices.dtype()));
  DISPATCH_FLOAT_DATATYPE_FUNCTION(value_type, ValueType, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(index_type, IndexType, [&] {
      initialize_with_index_addressor_kernel<ValueType, IndexType, GeneratorT>
        <<<grid_size, block_size, 0, stream>>>(
        num_total, dim, stride, reinterpret_cast<ValueType *>(buffer.data_ptr()), 
        reinterpret_cast<IndexType *>(indices.data_ptr()), generator_args);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void normal_init(
  at::Tensor buffer, 
  at::Tensor indices, 
  CurandStateContext& curand_state_context,
  float mean,
  float std_dev
) {

  using GeneratorT = NormalEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args {curand_state_context.ptr(), mean, std_dev};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args, num_worker);
}

void truncated_normal_init(
  at::Tensor buffer, 
  at::Tensor indices, 
  CurandStateContext& curand_state_context,
  float mean,
  float std_dev,
  float lower,
  float upper
) {
  using GeneratorT = TruncatedNormalEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args {curand_state_context.ptr(), mean, std_dev, lower, upper};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args, num_worker);
}

void uniform_init(
  at::Tensor buffer, 
  at::Tensor indices, 
  CurandStateContext& curand_state_context,
  float lower,
  float upper
) {
  using GeneratorT = UniformEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args {curand_state_context.ptr(), lower, upper};
  int num_worker = curand_state_context.num_worker();
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args, num_worker);
}

void const_init(
  at::Tensor buffer, 
  at::Tensor indices, 
  float value
) {
  using GeneratorT = ConstEmbeddingGenerator;
  auto generator_args = typename GeneratorT::Args {value};
  initialize_with_generator<GeneratorT>(buffer, indices, generator_args);
}

void debug_init(
  at::Tensor buffer, 
  at::Tensor indices, 
  at::Tensor keys
) {
  auto key_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(keys.dtype()));
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    using GeneratorT = MappingEmbeddingGenerator<KeyType>;
    auto generator_args = typename GeneratorT::Args {reinterpret_cast<const KeyType *>(keys.data_ptr()), 100000};
    initialize_with_generator<GeneratorT>(buffer, indices, generator_args);
  });

}

} // namespace dyn_emb

void bind_initializer_op(py::module &m) {

  py::class_<dyn_emb::CurandStateContext>(m, "CurandStateContext")
    .def(py::init<>())
    .def("ptr", &dyn_emb::CurandStateContext::ptr,
          py::return_value_policy::reference);

  m.def("normal_init", &dyn_emb::normal_init,
    "Normal initializer",
    py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"), py::arg("mean"), py::arg("std_dev"));

  m.def("truncated_normal_init", &dyn_emb::truncated_normal_init,
    "Truncated normal initializer",
    py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"), 
    py::arg("mean"), py::arg("std_dev"), py::arg("lower"), py::arg("upper"));
  
  m.def(
    "uniform_init", &dyn_emb::uniform_init,
    "Uniform initializer", 
     py::arg("buffer"), py::arg("indices"), py::arg("curand_state_context"), 
    py::arg("lower"), py::arg("upper"));

  m.def(
    "const_init", &dyn_emb::const_init,
    "Const initializer", 
    py::arg("buffer"), py::arg("indices"), py::arg("value"));

  m.def(
    "debug_init", &dyn_emb::debug_init,
    "Debug initializer", 
    py::arg("buffer"), py::arg("indices"), py::arg("keys"));
}