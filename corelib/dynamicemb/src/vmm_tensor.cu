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

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <errno.h>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h> // Used to set memory lock limits
#include <unistd.h>

std::size_t getTotalPhysicalMemory() {
  std::ifstream meminfo("/proc/meminfo");
  std::string key, unit;
  std::size_t value = 0; // kB
  while (meminfo >> key >> value >> unit) {
    if (key == "MemTotal:") {
      return value * 1024; // 转成字节
    }
  }
  return 0;
}

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
    CU_CHECK(cuDeviceGet(&cu_dev, device), "cuDeviceGet");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    CU_CHECK(cuMemGetAllocationGranularity(&m_page_size, &prop,
                                           CU_MEM_ALLOC_GRANULARITY_MINIMUM),
             "cuMemGetAllocationGranularity");

    m_reserved = (m_reserved + m_page_size - 1) / m_page_size * m_page_size;
    CU_CHECK(cuMemAddressReserve(&m_addr, m_reserved, m_page_size, 0, 0),
             "cuMemAddressReserve");

    std::size_t alloc_bytes =
        (m_size + m_page_size - 1) / m_page_size * m_page_size;

    CUmemGenericAllocationHandle m_handle;
    CU_CHECK(cuMemCreate(&m_handle, alloc_bytes, &prop, 0), "cuMemCreate");

    CU_CHECK(cuMemMap(m_addr, alloc_bytes, 0, m_handle, 0), "cuMemMap");

    handles.push_back(m_handle);

    CUmemAccessDesc access_desc = {};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK(cuMemSetAccess(m_addr, alloc_bytes, &access_desc, 1),
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
    // CUDA_CHECK(cudaDeviceSynchronize());

    std::size_t old_size = m_size;
    CUdevice cu_dev;
    CU_CHECK(cuDeviceGet(&cu_dev, device_), "cuDeviceGet");

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    CUmemGenericAllocationHandle handle;
    std::size_t delta = new_bytes - old_size;

    CU_CHECK(cuMemCreate(&handle, delta, &prop, 0), "cuMemCreate (extend)");

    CU_CHECK(cuMemMap(m_addr + old_size, delta, 0, handle, 0),
             "cuMemMap (extend)");

    CUmemAccessDesc access_desc = {};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK(cuMemSetAccess(m_addr + old_size, delta, &access_desc, 1),
             "cuMemSetAccess (extend)");

    handles.push_back(handle);

    m_size = old_size + delta;
    // CUDA_CHECK(cudaDeviceSynchronize());
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

class HostVMMTensor {

public:
  HostVMMTensor(std::size_t numel, torch::Dtype dtype, int device)
      : dtype_(dtype), device_(device) {

    if (numel == 0) {
      throw std::runtime_error("Can't create Host VMM tensor of size 0\n");
    }

    if (device < 0) {
      throw std::runtime_error("Invalid device id\n");
    }

    auto scalar_type = static_cast<torch::ScalarType>(dtype);
    auto dtype_bytes = get_size(scalar_type);
    m_size = numel * dtype_bytes;

    m_reserved = getTotalPhysicalMemory();

    int canMap = 0;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, device));
    if (!canMap) {
      throw std::runtime_error("Device does not support mapped host memory\n");
    }
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

    int64_t page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
      throw std::runtime_error("sysconf error\n");
    }
    m_page_size = page_size;
    m_reserved = (m_reserved + m_page_size - 1) / m_page_size * m_page_size;
    m_size = (m_size + m_page_size - 1) / m_page_size * m_page_size;

    // reserve host virtual memory
    m_addr_h = mmap(nullptr, m_reserved, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m_addr_h == MAP_FAILED) {
      throw std::runtime_error("mmap virtual memory failed\n");
    }

    // Accessing virtual addresses to trigger page loss interrupts and allocate
    // physical pages
    if (madvise(m_addr_h, m_size, MADV_WILLNEED) == -1) {
      munmap(m_addr_h, m_reserved);
      m_addr_h = nullptr;
      throw std::runtime_error(
          "madvise allocate initial physical memory failed\n");
    }

    // memset(m_addr_h, 0, m_size);
    uintptr_t aligned_ptr =
        (((uintptr_t)m_addr_h + m_page_size - 1) & ~(m_page_size - 1));
    for (uintptr_t p = aligned_ptr; p < ((uintptr_t)m_addr_h + m_size);
         p += m_page_size) {
      memset((void *)p, 0, 1);
    }

    // Lock the physical page corresponding to the virtual address
    if (mlock(m_addr_h, m_size) == -1) {
      munmap(m_addr_h, m_reserved);
      m_addr_h = nullptr;
      throw std::runtime_error("mlock initial physical memory failed");
    }

    CUDA_CHECK(cudaHostRegister(
        m_addr_h, m_size, cudaHostRegisterMapped | cudaHostRegisterPortable));

    CUDA_CHECK(
        cudaHostGetDevicePointer((void **)&m_addr_d, (void *)m_addr_h, 0));
  }

  void extend(std::size_t numel_new) {

    if (m_addr_h == nullptr) {
      throw std::runtime_error("Not initlialized.");
    }

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

    std::size_t old_size = m_size;

    uintptr_t append_start = (uintptr_t)m_addr_h + m_size;
    std::size_t delta = new_bytes - old_size;

    if (madvise(append_start, delta, MADV_WILLNEED) == -1) {
      throw std::runtime_error("madvise allocate physical memory failed\n");
    }

    uintptr_t aligned_ptr =
        (((uintptr_t)append_start + m_page_size - 1) & ~(m_page_size - 1));
    for (uintptr_t p = aligned_ptr; p < ((uintptr_t)append_start + delta);
         p += m_page_size) {
      memset((void *)p, 0, 1);
    }

    if (mlock(append_start, delta) == -1) {
      throw std::runtime_error("mlock physical memory failed");
    }

    CUDA_CHECK(cudaHostUnregister(m_addr_h));

    try {
      CUDA_CHECK(
          cudaHostRegister(m_addr_h, m_size + delta,
                           cudaHostRegisterMapped | cudaHostRegisterPortable));
    } catch (const std::runtime_error &e) {
      munlock(append_start, delta);
      /// TODO: but what will happened if it failed?
      CUDA_CHECK(cudaHostRegister(
          m_addr_h, m_size, cudaHostRegisterMapped | cudaHostRegisterPortable));

      throw;
    }

    CUdeviceptr m_addr_d_new = 0;

    try {

      CUDA_CHECK(cudaHostGetDevicePointer((void **)&m_addr_d_new,
                                          (void *)m_addr_h, 0));

      m_addr_d = m_addr_d_new;
      m_size = old_size + delta;
    } catch (const std::runtime_error &e) {
      munlock(append_start, delta);
      CUDA_CHECK(cudaHostUnregister(m_addr_h));
      CUDA_CHECK(cudaHostRegister(
          m_addr_h, m_size, cudaHostRegisterMapped | cudaHostRegisterPortable));
      throw;
    }
  }

  at::Tensor data() const {
    auto m_dev_ptr = reinterpret_cast<void *>(m_addr_d);
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);

    if (m_size % dtype_bytes != 0) {
      std::cout << "[warning(dynamicemb)]: Allocated size is not a multiple of "
                   "the data type size.\n";
    }

    auto numel_alloc = static_cast<int64_t>(m_size / dtype_bytes);

    // it still on CUDA device
    auto data_ = at::from_blob(
        m_dev_ptr, {numel_alloc},
        at::TensorOptions().dtype(dtype_).device(at::kCUDA, device_));
    return data_;
  }

  ~HostVMMTensor() {

    if (m_size > 0) {
      munlock(m_addr_h, m_size);
      CUDA_CHECK(cudaHostUnregister(m_addr_h));
      munmap(m_addr_h, m_reserved);
    }
  }

private:
  HostVMMTensor(const HostVMMTensor &) = delete;
  HostVMMTensor &operator=(const HostVMMTensor &) = delete;

  torch::Dtype dtype_ = at::kChar;
  int device_ = -1;

  void *m_addr_h = nullptr;
  CUdeviceptr m_addr_d = 0;
  std::size_t m_page_size = 0;
  std::size_t m_size = 0;
  std::size_t m_reserved = 0;
};

} // namespace dyn_emb

void bind_vmm_op(py::module &m) {

  py::class_<dyn_emb::VMMTensor>(m, "VMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::VMMTensor::extend, py::arg("numel_new"),
           "extend")
      .def("data", &dyn_emb::VMMTensor::data, "data");

  py::class_<dyn_emb::HostVMMTensor>(m, "HostVMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::HostVMMTensor::extend, py::arg("numel_new"),
           "extend")
      .def("data", &dyn_emb::HostVMMTensor::data, "data");
}
