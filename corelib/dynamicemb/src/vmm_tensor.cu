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

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>

#include "check.h"
#include "torch_utils.h"

namespace py = pybind11;

namespace dyn_emb {

class VMMTensor {

public:
  VMMTensor(std::size_t numel, torch::Dtype dtype, int device)
      : dtype_(dtype), device_(device) {

    if (numel == 0) {
      throw std::runtime_error("Can't create VMM tensor of size 0\n");
    }
    if (device < 0) {
      throw std::runtime_error("Invalid device id\n");
    }

    cuInit(0);

    auto scalar_type = static_cast<torch::ScalarType>(dtype);
    auto dtype_bytes = get_size(scalar_type);
    m_size = numel * dtype_bytes;

    auto &deviceProp = DeviceProp::getDeviceProp();
    m_reserved = deviceProp.totalGlobalMem;

    CUdevice cu_dev;
    checkCu(cuDeviceGet(&cu_dev, device), "cuDeviceGet");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    checkCu(cuMemGetAllocationGranularity(&m_page_size, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            "cuMemGetAllocationGranularity");

    m_reserved = (m_reserved + m_page_size - 1) / m_page_size * m_page_size;
    checkCu(cuMemAddressReserve(&m_addr, m_reserved, m_page_size, 0, 0),
            "cuMemAddressReserve");

    std::size_t alloc_bytes =
        (m_size + m_page_size - 1) / m_page_size * m_page_size;

    CUmemGenericAllocationHandle m_handle;
    checkCu(cuMemCreate(&m_handle, alloc_bytes, &prop, 0), "cuMemCreate");

    checkCu(cuMemMap(m_addr, alloc_bytes, 0, m_handle, 0), "cuMemMap");

    handles.push_back(m_handle);

    CUmemAccessDesc access_desc = {};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    checkCu(cuMemSetAccess(m_addr, alloc_bytes, &access_desc, 1),
            "cuMemSetAccess");

    m_size = alloc_bytes;
  }

  void extend(std::size_t numel_new) {

    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);
    if (numel_new * dtype_bytes <= m_size) {
      return;
    }

    std::size_t new_bytes =
        (numel_new * dtype_bytes + m_page_size - 1) / m_page_size * m_page_size;
    if (new_bytes > m_reserved) {
      throw std::runtime_error("Requested size exceeds reserved VA range");
    }

    // auto stream = at::cuda::getCurrentCUDAStream().stream();
    // CUDACHECK(cudaDeviceSynchronize());

    std::size_t old_size = m_size;
    CUdevice cu_dev;
    checkCu(cuDeviceGet(&cu_dev, device_), "cuDeviceGet");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    CUmemGenericAllocationHandle handle;
    std::size_t delta = new_bytes - old_size;

    checkCu(cuMemCreate(&handle, delta, &prop, 0), "cuMemCreate (extend)");

    checkCu(cuMemMap(m_addr + old_size, delta, 0, handle, 0),
            "cuMemMap (extend)");

    CUmemAccessDesc access_desc = {};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    checkCu(cuMemSetAccess(m_addr + old_size, delta, &access_desc, 1),
            "cuMemSetAccess (extend)");

    handles.push_back(handle);

    m_size = old_size + delta;
    // CUDACHECK(cudaDeviceSynchronize());
  }

  at::Tensor data() const {
    auto m_dev_ptr = reinterpret_cast<void *>(m_addr);
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);

    if (m_size % dtype_bytes != 0) {
      std::cout << "[warning(dynamicemb)]: Allocated size is not a multiple of "
                   "the data type size.\n";
    }

    auto numel_alloc = static_cast<int64_t>(m_size / dtype_bytes);

    auto data_ = at::from_blob(
        m_dev_ptr, {numel_alloc},
        at::TensorOptions().dtype(dtype_).device(at::kCUDA, device_));
    return data_;
  }

  ~VMMTensor() {
    if (m_size > 0) {
      cuMemUnmap(m_addr, m_size);
    }
    for (auto handle : handles) {
      if (handle) {
        cuMemRelease(handle);
      }
    }

    handles.clear();

    if (m_addr && m_reserved > 0) {
      cuMemAddressFree(m_addr, m_reserved);
    }
  }

private:
  VMMTensor(const VMMTensor &) = delete;
  VMMTensor &operator=(const VMMTensor &) = delete;

  torch::Dtype dtype_ = at::kChar;
  int device_ = -1;

  CUdeviceptr m_addr = 0;
  std::size_t m_size = 0;
  std::size_t m_reserved = 0;
  std::size_t m_page_size = 0;
  std::vector<CUmemGenericAllocationHandle> handles;
};

} // namespace dyn_emb

void bind_vmm_op(py::module &m) {

  py::class_<dyn_emb::VMMTensor>(m, "VMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::VMMTensor::extend, py::arg("numel_new"),
           "extend")
      .def("data", &dyn_emb::VMMTensor::data, "data");
}
