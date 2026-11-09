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

// Needed for the GNU extensions of <sys/mman.h> (MAP_ANONYMOUS, MAP_NORESERVE);
// must be defined before any libc header pulls it in.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>
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

namespace {

// Waits until the GPU has finished all work queued on `device`.
//
// cudaFree waits like this before it frees memory, but cuMemUnmap and
// cuMemRelease do not. The tensors that VMMTensor::data() returns do not own
// the memory, so without this wait, dropping the last reference to a VMMTensor
// would free the memory while kernels that use it still wait in a stream.
// Those kernels would then read or write unmapped memory and fail with an
// illegal memory access. The wait covers every stream of the device, because
// DynamicEmb kernels run on whichever stream is current, and TorchRec
// pipelines use several. PyTorch also waits for queued work before it unmaps
// an expandable segment.
//
// Like any device sync, this also waits for queued collectives, such as NCCL
// kernels. If a peer launches its side of one only after a host-side step (a
// barrier, a store wait) that this rank reaches after the drop, the ranks
// deadlock; finish pending collectives before dropping a table. A device sync
// is not allowed during CUDA graph capture, so a drop during a capture breaks
// that capture.
//
// ~VMMTensor() and ~HostVMMTensor() call this, so it never throws. If the wait
// fails, for example after a sticky error from an earlier kernel or during a
// graph capture, it logs the failed step and the caller still frees the memory.
// It never creates a CUDA context: if the
// device has no active primary context, as after CUDA has shut down at exit,
// no work can be queued on it.
void wait_for_device_work(int device) {
  // A failed wait is reported, not thrown (this runs in destructors). The
  // memory is still freed, so the log is the only trace of the cause if a later
  // kernel then faults on it.
  auto report = [device](const char *step, CUresult result) {
    const char *name = nullptr;
    cuGetErrorName(result, &name);
    std::cerr << "dynamicemb: wait for queued GPU work on device " << device
              << " failed in " << step << ": " << (name ? name : "unknown")
              << "; freeing the memory anyway\n";
  };
  CUdevice device_handle;
  CUresult result = cuDeviceGet(&device_handle, device);
  if (result != CUDA_SUCCESS) {
    report("cuDeviceGet", result);
    return;
  }
  unsigned int context_flags = 0;
  int context_active = 0;
  result = cuDevicePrimaryCtxGetState(device_handle, &context_flags,
                                      &context_active);
  if (result != CUDA_SUCCESS) {
    report("cuDevicePrimaryCtxGetState", result);
    return;
  }
  if (!context_active) {
    return;
  }
  CUcontext context = nullptr;
  result = cuDevicePrimaryCtxRetain(&context, device_handle);
  if (result != CUDA_SUCCESS) {
    report("cuDevicePrimaryCtxRetain", result);
    return;
  }
  // Push and pop leave the caller's current context and device as they were.
  result = cuCtxPushCurrent(context);
  if (result == CUDA_SUCCESS) {
    result = cuCtxSynchronize();
    if (result != CUDA_SUCCESS) {
      report("cuCtxSynchronize", result);
    }
    cuCtxPopCurrent(&context);
  } else {
    report("cuCtxPushCurrent", result);
  }
  cuDevicePrimaryCtxRelease(device_handle);
}

} // namespace

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
  // allocates only when needed. The base address never changes: new pages are
  // mapped after the old ones, and the old ones are neither unmapped nor
  // remapped, so kernels that are queued or running on them are not affected. It
  // only calls driver functions and touches the members of this object, so the
  // binding releases the GIL.
  void extend(std::size_t new_total_logical_numel) {
    std::lock_guard<std::mutex> lock(m_mutex);
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
    std::lock_guard<std::mutex> lock(m_mutex);
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

  std::size_t logical_numel() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_logical_numel;
  }

  std::size_t allocated_numel() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto dtype_bytes = get_size(static_cast<torch::ScalarType>(dtype_));
    return m_size / dtype_bytes;
  }

  /// Bytes actually mapped / backed (page-aligned); >= logical size in bytes.
  std::size_t allocated_bytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_size;
  }

  ~VMMTensor() {
    // Kernels queued before this free may still use the memory.
    wait_for_device_work(device_);
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
  mutable std::mutex m_mutex; // extend() runs without the GIL
};

// A buffer in pinned host memory that the GPU reads and writes through a mapped
// device pointer. It grows IN PLACE: the address range it can ever need is
// reserved when it is created (PROT_NONE + MAP_NORESERVE: address space only,
// nothing is committed and nothing counts against the overcommit limit), and
// growing makes the next pages of that range readable, resident, locked and
// registered with CUDA. The pages already registered are never unregistered,
// remapped or moved, so the base address (and with it the pointer table the
// kernels read) never changes and no kernel that is queued or running is
// affected. That is what lets a growth go without waiting for the device:
// unregistering a range blocks until the kernels that use it have finished, and
// moving it would leave them with a stale pointer.
//
// The range is registered in pieces, one per growth. The device pointer of each
// piece must continue the first one's (it does wherever a registered range has
// the device pointer of its host pointer, which is every Linux system with
// unified addressing); a piece for which it does not is rolled back and the
// growth fails with an error, the buffer is unchanged and can still be used.
class HostVMMTensor {

public:
  // `reserve_numel` is the number of elements the buffer can grow to (0: eight
  // times what it needs now, at least 1 GiB). It is clamped to the installed
  // memory, and halved while the kernel refuses that much address space, down
  // to what the buffer needs now.
  HostVMMTensor(std::size_t numel, torch::Dtype dtype, int device,
                std::size_t reserve_numel = 0)
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
      throw std::runtime_error(
          "Requested HostVMMTensor size exceeds total physical memory");
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
    m_size.store(round_up(required_bytes), std::memory_order_relaxed);

    // Address space only. A reservation of the whole machine's RAM per tensor,
    // as an earlier version made, exhausts the process's address space once
    // there are dozens of tables, so the caller says what the buffer can ever
    // need; the kernel can still refuse (RLIMIT_AS, a container), and the
    // reservation then halves, down to the bytes the buffer needs now.
    std::size_t reserve_bytes;
    if (reserve_numel > 0) {
      reserve_bytes = reserve_numel > total_ram / dtype_bytes
                          ? total_ram
                          : reserve_numel * dtype_bytes;
    } else {
      reserve_bytes = std::min<std::size_t>(
          std::max<std::size_t>(8 * required_bytes, std::size_t(1) << 30),
          total_ram);
    }
    const std::size_t initial_bytes = m_size.load(std::memory_order_relaxed);
    m_reserved = std::max(round_up(reserve_bytes), initial_bytes);
    for (;;) {
      m_addr_h = mmap(nullptr, m_reserved, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (m_addr_h != MAP_FAILED || m_reserved <= initial_bytes) {
        break;
      }
      m_reserved = std::max(initial_bytes, round_up(m_reserved / 2));
    }
    if (m_addr_h == MAP_FAILED) {
      m_addr_h = nullptr;
      throw std::runtime_error(
          std::string("mmap host address range failed (size=") +
          std::to_string(m_reserved) + " bytes): " + std::strerror(errno));
    }

    // The constructor is not done if this throws, so ~HostVMMTensor() will not
    // run: give the range back here.
    try {
      commit(0, initial_bytes);
    } catch (...) {
      munmap(m_addr_h, m_reserved);
      m_addr_h = nullptr;
      throw;
    }
  }

  // Set logical size to new_total_logical_numel; uses slack first, commits more
  // of the reserved range only when needed: the new tail alone is touched,
  // locked and registered. It only does host memory management and CUDA
  // registration (no wait for the device), so the binding releases the GIL. A
  // failure leaves the buffer as it was and usable, and extend() can be tried
  // again.
  //
  // m_mutex only keeps two growers apart. The commit of a multi-GB tail takes
  // seconds, and a thread that asks for data() or a size meanwhile does so with
  // the GIL held (these are plain bindings), so none of the readers takes the
  // mutex: the two sizes they read are atomics, which a growth stores in the
  // order that keeps `logical * element size <= committed bytes` for a reader
  // that loads them in the opposite order.
  void extend(std::size_t new_total_logical_numel) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_addr_h == nullptr) {
      throw std::runtime_error("Not initlialized.");
    }
    if (new_total_logical_numel <=
        m_logical_numel.load(std::memory_order_relaxed)) {
      return;
    }
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);
    std::size_t required_bytes = new_total_logical_numel * dtype_bytes;
    const std::size_t committed = m_size.load(std::memory_order_relaxed);
    if (required_bytes <= committed) {
      m_logical_numel.store(new_total_logical_numel, std::memory_order_release);
      return;
    }

    std::size_t new_bytes = round_up(required_bytes);
    std::size_t total_ram = getTotalPhysicalMemory();
    if (new_bytes > round_up(total_ram)) {
      throw std::runtime_error(
          "Requested HostVMMTensor size exceeds total physical memory");
    }
    if (new_bytes > m_reserved) {
      throw std::runtime_error(std::string("HostVMMTensor: growing to ") +
                               std::to_string(new_bytes) +
                               " bytes needs more address space than the " +
                               std::to_string(m_reserved) +
                               " bytes reserved when it was created");
    }
    commit(committed, new_bytes);
    // The committed bytes first, then the logical size that needs them.
    m_size.store(new_bytes, std::memory_order_release);
    m_logical_numel.store(new_total_logical_numel, std::memory_order_release);
  }

  at::Tensor data() const {
    auto m_dev_ptr = reinterpret_cast<void *>(m_addr_d);
    auto scalar_type = static_cast<torch::ScalarType>(dtype_);
    auto dtype_bytes = get_size(scalar_type);

    // The logical size first, then the committed bytes: a growth stores them in
    // the opposite order, so the bytes read here cover the size read here.
    const std::size_t logical = m_logical_numel.load(std::memory_order_acquire);
    const std::size_t committed = m_size.load(std::memory_order_acquire);
    if (logical * dtype_bytes > committed) {
      throw std::runtime_error(
          "HostVMMTensor logical numel exceeds allocated size");
    }

    auto data_ = at::from_blob(
        m_dev_ptr, {static_cast<int64_t>(logical)},
        at::TensorOptions().dtype(dtype_).device(at::kCUDA, device_));
    return data_;
  }

  std::size_t logical_numel() const {
    return m_logical_numel.load(std::memory_order_acquire);
  }

  std::size_t allocated_numel() const {
    auto dtype_bytes = get_size(static_cast<torch::ScalarType>(dtype_));
    return m_size.load(std::memory_order_acquire) / dtype_bytes;
  }

  /// Bytes actually locked + registered (page-aligned); >= logical size in
  /// bytes.
  std::size_t allocated_bytes() const {
    return m_size.load(std::memory_order_acquire);
  }

  /// Bytes of address space reserved: what the buffer can grow to in place
  /// (fixed when the buffer is made).
  std::size_t reserved_bytes() const { return m_reserved; }

  ~HostVMMTensor() {

    if (m_addr_h != nullptr) {
      // Kernels queued before this free may still read or write the memory
      // through the mapped device pointer. Wait for them explicitly: whether
      // cudaHostUnregister also waits is driver behavior the CUDA docs do not
      // promise.
      wait_for_device_work(device_);
      // Best-effort: a destructor must never throw, so do not CUDA_CHECK here.
      // Each piece is unregistered by the address it was registered with. Clear
      // the sticky error so it cannot surface in an unrelated check.
      for (auto const &piece : m_registered) {
        cudaHostUnregister(piece.first);
      }
      cudaGetLastError();
      munlock(m_addr_h, m_size.load(std::memory_order_relaxed));
      munmap(m_addr_h, m_reserved);
    }
  }

private:
  HostVMMTensor(const HostVMMTensor &) = delete;
  HostVMMTensor &operator=(const HostVMMTensor &) = delete;

  std::size_t round_up(std::size_t bytes) const {
    return (bytes + m_page_size - 1) / m_page_size * m_page_size;
  }

  // Makes [begin, end) of the reservation (page aligned, begin == 0 or the end
  // of what is committed) readable, resident, locked and registered with CUDA.
  // On any failure the range is address space only again, nothing else has
  // changed and the exception is thrown.
  void commit(std::size_t begin, std::size_t end) {
    char *p = static_cast<char *>(m_addr_h) + begin;
    std::size_t bytes = end - begin;

    // Drops the pages and the lock and makes the range inaccessible again.
    auto give_back = [&] {
      (void)mmap(p, bytes, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
    };

    if (mprotect(p, bytes, PROT_READ | PROT_WRITE) != 0) {
      throw std::runtime_error(
          std::string("mprotect (host buffer) failed (size=") +
          std::to_string(bytes) + " bytes): " + std::strerror(errno));
    }
    // Fault the pages in, zero filled. (Not a memset of the whole range for the
    // sake of the time: one store per page is what takes them.)
    volatile char *touch = p;
    for (std::size_t off = 0; off < bytes; off += m_page_size) {
      touch[off] = 0;
    }
    if (mlock(p, bytes) == -1) {
      int saved_errno = errno;
      give_back();
      throw std::runtime_error(
          std::string("mlock (host buffer) failed (size=") +
          std::to_string(bytes) + " bytes): " + std::strerror(saved_errno));
    }

    void *device_ptr = nullptr;
    cudaError_t err = cudaHostRegister(
        p, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
    bool registered = (err == cudaSuccess);
    if (registered) {
      err = cudaHostGetDevicePointer(&device_ptr, p, 0);
    }
    if (err == cudaSuccess && begin == 0) {
      m_addr_d = reinterpret_cast<CUdeviceptr>(device_ptr);
    }
    if (err == cudaSuccess &&
        reinterpret_cast<std::uintptr_t>(device_ptr) !=
            static_cast<std::uintptr_t>(m_addr_d) + begin) {
      // The device addresses of the pieces are not contiguous on this system: a
      // kernel addressing base + row * stride would leave the first piece.
      err = cudaErrorNotSupported;
    }
    if (err != cudaSuccess) {
      cudaGetLastError();
      std::string what = cudaGetErrorString(err);
      if (registered) {
        // The tail only: no kernel has used it, so this returns at once.
        cudaHostUnregister(p);
        cudaGetLastError();
      }
      munlock(p, bytes);
      give_back();
      throw std::runtime_error(
          std::string("cudaHostRegister/cudaHostGetDevicePointer (host buffer) "
                      "failed (size=") +
          std::to_string(bytes) + " bytes): " + what);
    }
    m_registered.emplace_back(p, bytes);
  }

  torch::Dtype dtype_ = at::kChar;
  int device_ = -1;

  // Fixed when the buffer is made, so every thread can read them without a lock:
  // the host range, the device address of its first byte (every piece continues
  // it), the page size and the size of the reservation.
  void *m_addr_h = nullptr;
  CUdeviceptr m_addr_d = 0;
  std::size_t m_page_size = 0;
  std::size_t m_reserved = 0; // address space reserved (page-aligned)
  // What a growth changes, and what the readers load (see extend()).
  std::atomic<std::size_t> m_size{0};          // committed bytes (page-aligned)
  std::atomic<std::size_t> m_logical_numel{0}; // user-visible element count
  // The pieces registered with CUDA: the address of each, as it was registered.
  // Only a grower (and the destructor) touches it.
  std::vector<std::pair<void *, std::size_t>> m_registered;
  std::mutex m_mutex; // keeps two extend() calls apart; extend() runs without the GIL
};

} // namespace dyn_emb

void bind_vmm_op(py::module &m) {

  py::class_<dyn_emb::VMMTensor>(m, "VMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int>(), py::arg("numel"),
           py::arg("dtype"), py::arg("device"))
      .def("extend", &dyn_emb::VMMTensor::extend,
           py::arg("new_total_logical_numel"),
           py::call_guard<py::gil_scoped_release>(),
           "Set logical size to new_total_logical_numel; uses slack, then allocates if needed. The base address never changes.")
      .def("data", &dyn_emb::VMMTensor::data, "data")
      .def("logical_numel", &dyn_emb::VMMTensor::logical_numel,
           "Logical element count (user-visible size).")
      .def("allocated_numel", &dyn_emb::VMMTensor::allocated_numel,
           "Allocated element count (may be larger due to alignment).")
      .def("allocated_bytes", &dyn_emb::VMMTensor::allocated_bytes,
           "Bytes actually mapped (page-aligned); >= logical bytes.");

  py::class_<dyn_emb::HostVMMTensor>(m, "HostVMMTensor")
      .def(py::init<std::size_t, torch::Dtype, int, std::size_t>(),
           py::arg("numel"), py::arg("dtype"), py::arg("device"),
           py::arg("reserve_numel") = 0)
      .def("extend", &dyn_emb::HostVMMTensor::extend,
           py::arg("new_total_logical_numel"),
           py::call_guard<py::gil_scoped_release>(),
           "Set logical size to new_total_logical_numel; uses slack, then commits "
           "more of the address range reserved at creation. The base address never changes.")
      .def("data", &dyn_emb::HostVMMTensor::data, "data")
      .def("logical_numel", &dyn_emb::HostVMMTensor::logical_numel,
           "Logical element count (user-visible size).")
      .def("allocated_numel", &dyn_emb::HostVMMTensor::allocated_numel,
           "Allocated element count (may be larger due to alignment).")
      .def("allocated_bytes", &dyn_emb::HostVMMTensor::allocated_bytes,
           "Bytes actually locked+registered (page-aligned); >= logical bytes.")
      .def("reserved_bytes", &dyn_emb::HostVMMTensor::reserved_bytes,
           "Bytes of address space reserved: what the buffer can grow to in place.");

  // Capability flag, read by callers with getattr(..., False): a HostVMMTensor
  // takes the number of elements to reserve address space for and grows in place
  // (its base address never changes; extend() never waits for the device).
  m.attr("host_vmm_in_place") = py::bool_(true);
}