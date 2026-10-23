// ---------------------------------------------------------------------------
// A HOST-ONLY STAND-IN FOR THE CUDA RUNTIME.  NOT THE REAL THING.
//
// Purpose: compile and exercise the *algorithmic* core of embedding_cache (probe
// sequences, wraparound, collision handling, rebuild) on a machine with no CUDA
// toolkit.  It maps device memory onto malloc, mapped-pinned onto malloc, and
// atomics onto single-threaded reads/writes.
//
// It therefore cannot test anything that makes CUDA interesting: real
// concurrency, memory ordering, warp behaviour, or PCIe.  Passing tests under
// this shim says the logic is right, not that the kernels are.
//
// This directory must only ever be on the include path of the host-test build
// (see run_tests.sh).  The guard below turns accidental use in a real
// device build into a compile error instead of a mystery.
// ---------------------------------------------------------------------------
#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "embedding_cache host_shim/cuda_runtime.h was included in a CUDA compilation. \
Remove test/embedding_cache/host_shim from the include path; it exists only for the \
host-only test build."
#endif

#include <cstdlib>
#include <cstring>

#define __host__
#define __device__
#define __global__
#define __forceinline__ inline

// `static` gives one instance per instantiation, which is exactly shared-memory
// semantics for a single block. A test can therefore simulate a block by
// calling a kernel once per threadIdx in ascending order — thread 0 publishes,
// the rest read. Only valid while the work fits one grid-stride iteration;
// beyond that the "threads" would run different iterations against one copy.
#define __shared__ static

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind {
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3
};
enum { cudaHostAllocDefault = 0x00, cudaHostAllocMapped = 0x02 };

// No streams here: everything is immediate, which is exactly why this shim
// cannot say anything about the stream ordering the design relies on. A
// synchronise therefore has nothing to wait for and always succeeds — which is
// also why a test passing here proves nothing about asynchrony.
typedef void* cudaStream_t;
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

inline cudaError_t cudaGetDevice(int* d) {
  *d = 0;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }

// Enough of cudaDeviceProp for dyn_emb::DeviceProp (src/utils.cpp). The values
// are a plausible mid-range GPU, not this machine — there is no device here to
// ask. A test that depends on a particular grid size is testing the shim.
struct cudaDeviceProp {
  int multiProcessorCount;
  int warpSize;
  int maxThreadsPerMultiProcessor;
  int maxThreadsPerBlock;
  size_t totalGlobalMem;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  p->multiProcessorCount = 108;
  p->warpSize = 32;
  p->maxThreadsPerMultiProcessor = 2048;
  p->maxThreadsPerBlock = 1024;
  p->totalGlobalMem = size_t(40) << 30;
  return cudaSuccess;
}

inline const char* cudaGetErrorString(cudaError_t e) {
  return e == cudaSuccess ? "no error" : "stub error";
}

// Always clean: nothing here can fail asynchronously, because nothing here is
// asynchronous. A launch failure — bad geometry, too much shared memory — is
// invisible to this shim by construction.
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

template <typename T>
inline cudaError_t cudaMalloc(T** p, size_t bytes) {
  *p = static_cast<T*>(std::malloc(bytes));
  return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void* p) {
  std::free(p);
  return cudaSuccess;
}
template <typename T>
inline cudaError_t cudaHostAlloc(T** p, size_t bytes, unsigned) {
  *p = static_cast<T*>(std::malloc(bytes));
  return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFreeHost(void* p) {
  std::free(p);
  return cudaSuccess;
}
template <typename T>
inline cudaError_t cudaHostGetDevicePointer(T** d, void* h, unsigned) {
  *d = static_cast<T*>(h);  // host-mapped: device and host share the address
  return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes,
                              cudaMemcpyKind) {
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes,
                                   cudaMemcpyKind, cudaStream_t = nullptr) {
  std::memcpy(dst, src, bytes);
  return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t bytes) {
  std::memset(p, v, bytes);
  return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t bytes,
                                   cudaStream_t = nullptr) {
  std::memset(p, v, bytes);
  return cudaSuccess;
}

// Single-threaded atomics.  Real contention is precisely what this cannot test:
// every one of these is trivially "atomic" because nothing else is running.
inline unsigned long long atomicCAS(unsigned long long* addr,
                                    unsigned long long expected,
                                    unsigned long long desired) {
  const unsigned long long old = *addr;
  if (old == expected) *addr = desired;
  return old;
}
inline unsigned int atomicCAS(unsigned int* addr, unsigned int expected,
                              unsigned int desired) {
  const unsigned int old = *addr;
  if (old == expected) *addr = desired;
  return old;
}
inline unsigned int atomicAdd(unsigned int* addr, unsigned int v) {
  const unsigned int old = *addr;
  *addr = old + v;
  return old;
}
inline unsigned int atomicSub(unsigned int* addr, unsigned int v) {
  const unsigned int old = *addr;
  *addr = old - v;
  return old;
}
inline unsigned int atomicOr(unsigned int* addr, unsigned int v) {
  const unsigned int old = *addr;
  *addr = old | v;
  return old;
}
inline unsigned int atomicAnd(unsigned int* addr, unsigned int v) {
  const unsigned int old = *addr;
  *addr = old & v;
  return old;
}
inline unsigned long long atomicAdd(unsigned long long* addr,
                                    unsigned long long v) {
  const unsigned long long old = *addr;
  *addr = old + v;
  return old;
}
// Deliberately NO 64-bit atomicSub: CUDA does not provide one (only int and
// unsigned int), and an over-permissive shim hides that until nvcc sees the
// code. Subtract by adding the two's complement instead.
inline int atomicExch(int* addr, int v) {
  const int old = *addr;
  *addr = v;
  return old;
}
inline unsigned int atomicExch(unsigned int* addr, unsigned int v) {
  const unsigned int old = *addr;
  *addr = v;
  return old;
}

inline void __threadfence() {}
inline void __syncthreads() {}
inline void __syncwarp(unsigned = 0xffffffffu) {}

// A single-thread "launch geometry". Kernels written with grid-stride loops
// process every element under this, so their bodies can be called directly from
// a host test — what stays unverified is the launch configuration, not the
// kernel logic.
struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
inline dim3 threadIdx(0, 0, 0);
inline dim3 blockIdx(0, 0, 0);
inline dim3 blockDim(1, 1, 1);
inline dim3 gridDim(1, 1, 1);
