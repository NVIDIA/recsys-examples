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

#include "torch_utils.h"

namespace {

template <class S> __global__ void device_nano_kernel(S *d_clk) {
  S mclk;
  asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(mclk));
  *d_clk = mclk;
}

class DeviceTimestamp {
public:
  DeviceTimestamp() {
    CUDACHECK(cudaMalloc((void **)&d_timestamp, sizeof(uint64_t)));
  }

  ~DeviceTimestamp() { CUDACHECK(cudaFree(d_timestamp)); }

  uint64_t get(const cudaStream_t &stream) {
    device_nano_kernel<uint64_t><<<1, 1, 0, stream>>>(d_timestamp);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    CUDACHECK(cudaMemcpyAsync(&h_timestamp, d_timestamp, sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    return h_timestamp;
  }

private:
  uint64_t *d_timestamp{nullptr};
  uint64_t h_timestamp{0};
};

} // namespace

namespace dyn_emb {

uint64_t device_timestamp() {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  static DeviceTimestamp timestamp;
  return timestamp.get(stream);
}

DataType scalartype_to_datatype(at::ScalarType scalar_type) {
  switch (scalar_type) {
  case at::kFloat:
    return DataType::Float32;
  case at::kHalf:
    return DataType::Float16;
  case at::kBFloat16:
    return DataType::BFloat16;
  case at::kLong:
    return DataType::Int64;
  case at::kInt:
    return DataType::Int32;
  case at::kUInt64:
    return DataType::UInt64;
  case at::kUInt32:
    return DataType::UInt32;
  default:
    throw std::invalid_argument("Unsupported data type");
  }
}

at::ScalarType datatype_to_scalartype(dyn_emb::DataType dtype) {
  switch (dtype) {
  case dyn_emb::DataType::Float32:
    return at::kFloat;
  case dyn_emb::DataType::Float16:
    return at::kHalf;
  case dyn_emb::DataType::BFloat16:
    return at::kBFloat16;
  case dyn_emb::DataType::Int64:
    return at::kLong;
  case dyn_emb::DataType::UInt64:
    return at::kUInt64;
  case dyn_emb::DataType::Int32:
    return at::kInt;
  case dyn_emb::DataType::UInt32:
    return at::kUInt32;
  case dyn_emb::DataType::Size_t:
    return at::kLong;
  default:
    throw std::invalid_argument("Unsupported DataType");
  }
}

at::ScalarType convertTypeMetaToScalarType(const caffe2::TypeMeta &typeMeta) {
  if (typeMeta == caffe2::TypeMeta::Make<float>()) {
    return at::kFloat;
  } else if (typeMeta == caffe2::TypeMeta::Make<at::Half>()) {
    return at::kHalf;
  } else if (typeMeta == caffe2::TypeMeta::Make<at::BFloat16>()) {
    return at::kBFloat16;
  } else if (typeMeta == caffe2::TypeMeta::Make<int64_t>()) {
    return at::kLong;
  } else if (typeMeta == caffe2::TypeMeta::Make<int>()) {
    return at::kInt;
  } else if (typeMeta == caffe2::TypeMeta::Make<uint64_t>()) {
    return at::kUInt64;
  } else if (typeMeta == caffe2::TypeMeta::Make<uint32_t>()) {
    return at::kUInt32;
  } else {
    throw std::invalid_argument("Unsupported DataType");
  }
}
} // namespace dyn_emb

// PYTHON WRAP
void bind_utils(py::module &m) {
  m.def("device_timestamp", &dyn_emb::device_timestamp,
        "Get device timestamp.");
}