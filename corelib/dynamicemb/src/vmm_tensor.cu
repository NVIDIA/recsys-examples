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

// Needed for mremap()/MREMAP_MAYMOVE (a GNU extension); must be defined before
// any libc header pulls in <sys/mman.h>.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

/// Total installed RAM in bytes (Linux). Used to size the host VMM reservation.
/// Tries MemTotal from /proc/meminfo (see meminfo(5), value is kB), then
/// sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE). Throws if neither works,
/// so callers never get 0 (mmap(…, 0, …) would fail).
std::size_t getTotalPhysicalMemory() {
  std::ifstream meminfo("/proc/meminfo");
  if (meminfo) {
    std::string key, unit;
    std::size_t value_kb = 0;
    while (meminfo >> key >> value_kb >> unit) {
      if (key == "MemTotal:") {
        std::size_t bytes = value_kb * 1024;
        if (bytes > 0) {
          return bytes;
        }
        break;
      }
    }
  }

  long phys_pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (phys_pages > 0 && page_size > 0) {
    return static_cast<std::size_t>(phys_pages) *
           static_cast<std::size_t>(page_size);
  }

  throw std::runtime_error(
      "Could not determine total physical memory: MemTotal not found or zero "
      "in /proc/meminfo, and sysconf(_SC_PHYS_PAGES / _SC_PAGE_SIZE) is "
      "unavailable or invalid");
}

#include <pybind11/pybind11.h>

#include "check.h"
#include "torch_utils.h"

namespace py = pybind11;

namespace dyn_emb {

class VMMTensor {

public:
  VMMTensor(std::size_t numel, torch::Dtype dtype, int device)
      : dtype_(dtype), device_(device), m_logical_numel(numel) {

    if (numel == 0) {
      throw std::runtime_error("Can't create VMM tensor of size 0\n");
    }
    if (device < 0) {
      throw std::runtime_error("Invalid device id\n");
    }

    cuInit(0);

    auto scalar_type = static_cast<torch::ScalarType>(dtype);
    auto dtype_bytes = get_size(scalar_type);
    std::size_t required_bytes = numel * dtype_bytes;

    auto &deviceProp = DeviceProp::getDeviceProp(device);
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
        (required_bytes + m_page_size - 1) / m_page_size * m_page_size;
    m_size = alloc_bytes;

    CUmemGenericAllocationHandle m_handle;
    CU_CHECK(cuMemCreate(&m_handle, alloc_bytes, &prop, 0), "cuMemCreate");

    CU_CHECK(cuMemMap(m_addr, alloc_bytes, 0, m_handle, 0), "cuMemMap");

    handles.push_back(m_handle);

    CUmemAccessDesc access_desc = {};
    access_desc.location = prop.location;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CU_CHECK(cuMemSetAccess(m_addr, alloc_bytes, &access_desc, 1),
             "cuMemSetAccess");

  }

  // Set logical size to new_total_logical_numel; uses alignment slack first,
  // allocates only when needed.
  void extend(std::size_t new_total_logical_numel) {
    if (new_total_logical_numel <= m_logical_numel) {
      return;
    }
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);
    std::size_t required_bytes = new_total_logical_numel * dtype_bytes;
    if (required_bytes <= m_size) {
      m_logical_numel = new_total_logical_numel;
      return;
    }

    std::size_t new_bytes =
        (required_bytes + m_page_size - 1) / m_page_size * m_page_size;
    if (new_bytes > m_reserved) {
      throw std::runtime_error("Requested size exceeds reserved VA range");
    }

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
    m_logical_numel = new_total_logical_numel;
  }

  at::Tensor data() const {
    auto m_dev_ptr = reinterpret_cast<void *>(m_addr);
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);

    if (m_logical_numel * dtype_bytes > m_size) {
      throw std::runtime_error(
          "VMMTensor logical numel exceeds allocated size");
    }

    auto data_ = at::from_blob(
        m_dev_ptr, {static_cast<int64_t>(m_logical_numel)},
        at::TensorOptions().dtype(dtype_).device(at::kCUDA, device_));
    return data_;
  }

  std::size_t logical_numel() const { return m_logical_numel; }

  std::size_t allocated_numel() const {
    auto dtype_bytes = get_size(static_cast<torch::ScalarType>(dtype_));
    return m_size / dtype_bytes;
  }

  /// Bytes actually mapped / backed (page-aligned); >= logical size in bytes.
  std::size_t allocated_bytes() const { return m_size; }

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
  std::size_t m_size = 0;           // allocated bytes (page-aligned)
  std::size_t m_logical_numel = 0;  // user-visible element count
  std::size_t m_reserved = 0;
  std::size_t m_page_size = 0;
  std::vector<CUmemGenericAllocationHandle> handles;
};

class HostVMMTensor {

public:
  HostVMMTensor(std::size_t numel, torch::Dtype dtype, int device)
      : dtype_(dtype), device_(device), m_logical_numel(numel) {

    if (numel == 0) {
      throw std::runtime_error("Can't create Host VMM tensor of size 0\n");
    }

    if (device < 0) {
      throw std::runtime_error("Invalid device id\n");
    }

    auto scalar_type = static_cast<torch::ScalarType>(dtype);
    auto dtype_bytes = get_size(scalar_type);
    std::size_t required_bytes = numel * dtype_bytes;

    // A HostVMMTensor is pinned (mlock + cudaHostRegister), so it can never
    // exceed installed RAM; reject early with a clear message.
    std::size_t total_ram = getTotalPhysicalMemory();
    if (required_bytes > total_ram) {
      throw std::runtime_error("Requested HostVMMTensor size exceeds total physical memory");
    }

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
    m_size =
        (required_bytes + m_page_size - 1) / m_page_size * m_page_size;
    // Map exactly the bytes we back. The previous implementation mmap'd the
    // whole machine's RAM per tensor as a "reservation"; with many tables those
    // multi-TB mappings accumulate and exhaust the process virtual address
    // space, so mmap eventually fails with ENOMEM even for tiny tensors.
    // extend() grows this mapping on demand via mremap() instead.
    m_reserved = m_size;

    m_addr_h = mmap(nullptr, m_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m_addr_h == MAP_FAILED) {
      throw std::runtime_error(std::string("mmap host memory failed (size=") +
                               std::to_string(m_size) +
                               " bytes): " + std::strerror(errno));
    }

    // MADV_WILLNEED is only a readahead hint; the memset loop below and mlock()
    // are what actually fault in and pin the pages, so a failure here is
    // non-fatal. Surface it as a warning rather than aborting (or swallowing).
    if (madvise(m_addr_h, m_size, MADV_WILLNEED) == -1) {
      std::cerr << "HostVMMTensor: madvise(MADV_WILLNEED) failed: "
                << std::strerror(errno) << " (non-fatal)\n";
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

    // On failure, free the already mmap'd + mlocked buffer before throwing: the
    // constructor is not done, so ~HostVMMTensor() will not run to clean it up.
    cudaError_t err =
        cudaHostRegister(m_addr_h, m_size,
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err == cudaSuccess) {
      err = cudaHostGetDevicePointer((void **)&m_addr_d, (void *)m_addr_h, 0);
    }
    if (err != cudaSuccess) {
      cudaGetLastError();
      munlock(m_addr_h, m_reserved);
      munmap(m_addr_h, m_reserved);
      m_addr_h = nullptr;
      throw std::runtime_error(
          std::string(
              "cudaHostRegister/cudaHostGetDevicePointer (init) failed: ") +
          cudaGetErrorString(err));
    }
  }

  // Best-effort recovery for extend(): bring the buffer back to a working,
  // self-consistent state at `restore_bytes` after a failed grow. The mapping
  // may currently be larger than restore_bytes (mremap already succeeded) and
  // may be registered at either size, so: drop any CUDA registration, shrink
  // the mapping back, then re-register and refresh the device pointer. Updates
  // m_addr_h / m_reserved / m_addr_d so that afterwards m_size == m_reserved ==
  // restore_bytes. Returns an empty string on success, or a description if the
  // recovery itself fails (the buffer is then unusable and must be destroyed).
  std::string restore_to_size(std::size_t restore_bytes) {
    // CUDA registration pins the pages, so it must be dropped before remapping.
    // The buffer may already be unregistered here; ignore that error and clear
    // the sticky CUDA error so it cannot leak into an unrelated CUDA_CHECK.
    cudaHostUnregister(m_addr_h);
    cudaGetLastError();

    if (m_reserved != restore_bytes) {
      // Shrinking never needs MREMAP_MAYMOVE and keeps the base address; the
      // truncated tail is unmapped (and thereby munlocked) for us.
      void *p = mremap(m_addr_h, m_reserved, restore_bytes, 0);
      if (p == MAP_FAILED) {
        return std::string("mremap shrink to ") +
               std::to_string(restore_bytes) +
               " bytes failed: " + std::strerror(errno);
      }
      m_addr_h = p;
      m_reserved = restore_bytes;
    }

    cudaError_t err =
        cudaHostRegister(m_addr_h, restore_bytes,
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return std::string("cudaHostRegister(") + std::to_string(restore_bytes) +
             ") failed: " + cudaGetErrorString(err);
    }
    err = cudaHostGetDevicePointer((void **)&m_addr_d, (void *)m_addr_h, 0);
    if (err != cudaSuccess) {
      cudaGetLastError();
      return std::string("cudaHostGetDevicePointer failed: ") +
             cudaGetErrorString(err);
    }
    return std::string();
  }

  // Set logical size to new_total_logical_numel; uses slack first, grows the
  // backing mapping (via mremap) only when needed. On any failure after the
  // mapping has grown, the buffer is rolled back to its previous size via
  // restore_to_size() so the object stays consistent (m_size == m_reserved)
  // and is safe to use or retry.
  void extend(std::size_t new_total_logical_numel) {
    if (m_addr_h == nullptr) {
      throw std::runtime_error("Not initlialized.");
    }
    if (new_total_logical_numel <= m_logical_numel) {
      return;
    }
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);
    std::size_t required_bytes = new_total_logical_numel * dtype_bytes;
    if (required_bytes <= m_size) {
      m_logical_numel = new_total_logical_numel;
      return;
    }

    std::size_t new_bytes =
        (required_bytes + m_page_size - 1) / m_page_size * m_page_size;
    std::size_t total_ram = getTotalPhysicalMemory();
    std::size_t total_ram_aligned =
        (total_ram + m_page_size - 1) / m_page_size * m_page_size;
    if (new_bytes > total_ram_aligned) {
      throw std::runtime_error(
          "Requested HostVMMTensor size exceeds total physical memory");
    }

    std::size_t old_size = m_size; // == m_reserved on entry (consistent state)
    std::size_t delta = new_bytes - old_size;

    // CUDA registration is tied to the current virtual address, so drop it
    // before remapping. If this throws, nothing has changed and the buffer is
    // still valid at its old size.
    CUDA_CHECK(cudaHostUnregister(m_addr_h));

    // mremap() may relocate the buffer (MREMAP_MAYMOVE); that is safe because
    // callers rebuild value-buffer pointers after extend() (get_table_ptrs()).
    void *new_addr = mremap(m_addr_h, old_size, new_bytes, MREMAP_MAYMOVE);
    if (new_addr == MAP_FAILED) {
      int saved_errno = errno; // mremap left the old mapping intact.
      std::string rec = restore_to_size(old_size);
      std::string msg = std::string("mremap (extend) failed (old=") +
                        std::to_string(old_size) + " new=" +
                        std::to_string(new_bytes) +
                        " bytes): " + std::strerror(saved_errno);
      if (!rec.empty()) {
        msg += "; buffer recovery also failed: " + rec;
      }
      throw std::runtime_error(msg);
    }
    m_addr_h = new_addr;
    // The mapping is now new_bytes large at new_addr; record the true extent
    // before anything below can throw so the destructor and restore_to_size()
    // always see the real mapping size.
    m_reserved = new_bytes;

    // MADV_WILLNEED is only a readahead hint; the memset loop and mlock() below
    // are what actually fault in and pin the pages, so a failure here is
    // non-fatal. Surface it as a warning rather than aborting or swallowing.
    uintptr_t tail_start = (uintptr_t)m_addr_h + old_size;
    if (madvise((void *)tail_start, delta, MADV_WILLNEED) == -1) {
      std::cerr << "HostVMMTensor::extend: madvise(MADV_WILLNEED) failed: "
                << std::strerror(errno) << " (non-fatal)\n";
    }
    uintptr_t aligned_ptr =
        (((uintptr_t)tail_start + m_page_size - 1) & ~(m_page_size - 1));
    for (uintptr_t p = aligned_ptr; p < ((uintptr_t)m_addr_h + new_bytes);
         p += m_page_size) {
      memset((void *)p, 0, 1);
    }

    if (mlock(m_addr_h, new_bytes) == -1) {
      int saved_errno = errno;
      std::string rec = restore_to_size(old_size);
      std::string msg =
          std::string("mlock (extend) failed: ") + std::strerror(saved_errno);
      if (!rec.empty()) {
        msg += "; buffer recovery also failed: " + rec;
      }
      throw std::runtime_error(msg);
    }

    cudaError_t err =
        cudaHostRegister(m_addr_h, new_bytes,
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err == cudaSuccess) {
      err = cudaHostGetDevicePointer((void **)&m_addr_d, (void *)m_addr_h, 0);
    }
    if (err != cudaSuccess) {
      cudaGetLastError();
      std::string primary = cudaGetErrorString(err);
      std::string rec = restore_to_size(old_size);
      std::string msg = std::string("cudaHostRegister/cudaHostGetDevicePointer "
                                     "(extend) failed: ") +
                        primary;
      if (!rec.empty()) {
        msg += "; buffer recovery also failed: " + rec;
      }
      throw std::runtime_error(msg);
    }

    m_size = new_bytes;
    m_logical_numel = new_total_logical_numel;
  }

  at::Tensor data() const {
    auto m_dev_ptr = reinterpret_cast<void *>(m_addr_d);
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);

    if (m_logical_numel * dtype_bytes > m_size) {
      throw std::runtime_error(
          "HostVMMTensor logical numel exceeds allocated size");
    }

    auto data_ = at::from_blob(
        m_dev_ptr, {static_cast<int64_t>(m_logical_numel)},
        at::TensorOptions().dtype(dtype_).device(at::kCUDA, device_));
    return data_;
  }

  std::size_t logical_numel() const { return m_logical_numel; }

  std::size_t allocated_numel() const {
    auto dtype_bytes = get_size(static_cast<torch::ScalarType>(dtype_));
    return m_size / dtype_bytes;
  }

  /// Bytes actually locked + registered (page-aligned); >= logical size in bytes.
  std::size_t allocated_bytes() const { return m_size; }

  ~HostVMMTensor() {

    if (m_size > 0) {
      munlock(m_addr_h, m_reserved);
      // Best-effort: a destructor must never throw, so do not CUDA_CHECK here.
      // Clear the sticky error so it cannot surface in an unrelated check.
      cudaHostUnregister(m_addr_h);
      cudaGetLastError();
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
  std::size_t m_size = 0;           // allocated bytes (page-aligned)
  std::size_t m_logical_numel = 0;  // user-visible element count
  std::size_t m_reserved = 0;
};

} // namespace dyn_emb

void bind_vmm_op(py::module &m) {

  py::class_<dyn_emb::VMMTensor>(m, "VMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::VMMTensor::extend,
           py::arg("new_total_logical_numel"),
           "Set logical size to new_total_logical_numel; uses slack, then allocates if needed.")
      .def("data", &dyn_emb::VMMTensor::data, "data")
      .def("logical_numel", &dyn_emb::VMMTensor::logical_numel,
           "Logical element count (user-visible size).")
      .def("allocated_numel", &dyn_emb::VMMTensor::allocated_numel,
           "Allocated element count (may be larger due to alignment).")
      .def("allocated_bytes", &dyn_emb::VMMTensor::allocated_bytes,
           "Bytes actually mapped (page-aligned); >= logical bytes.");

  py::class_<dyn_emb::HostVMMTensor>(m, "HostVMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::HostVMMTensor::extend,
           py::arg("new_total_logical_numel"),
           "Set logical size to new_total_logical_numel; uses slack, then allocates if needed.")
      .def("data", &dyn_emb::HostVMMTensor::data, "data")
      .def("logical_numel", &dyn_emb::HostVMMTensor::logical_numel,
           "Logical element count (user-visible size).")
      .def("allocated_numel", &dyn_emb::HostVMMTensor::allocated_numel,
           "Allocated element count (may be larger due to alignment).")
      .def("allocated_bytes", &dyn_emb::HostVMMTensor::allocated_bytes,
           "Bytes actually locked+registered (page-aligned); >= logical bytes.");
}