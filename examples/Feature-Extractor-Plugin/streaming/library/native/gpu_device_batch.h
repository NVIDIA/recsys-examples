// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Derived unchanged (except JNI symbol prefix) from Mega-Flink native/gpu_device_batch.h.

#ifndef CUDF_FLINK_GPU_DEVICE_BATCH_H_
#define CUDF_FLINK_GPU_DEVICE_BATCH_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Versioned C ABI shared by all project-owned native operator libraries. Java only receives the
// opaque uint64 token; device pointers and CUDA events are visible exclusively through this native
// borrow view.
#define CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION 1u
#define CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS 16u

// Stable IDs; do not use a C++/Java enum ordinal at the ABI boundary.
enum CudfFlinkGpuDeviceBatchTypeId : std::uint32_t {
  CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN = 1,
  CUDF_FLINK_GPU_DEVICE_BATCH_INT8 = 2,
  CUDF_FLINK_GPU_DEVICE_BATCH_INT16 = 3,
  CUDF_FLINK_GPU_DEVICE_BATCH_INT32 = 4,
  CUDF_FLINK_GPU_DEVICE_BATCH_INT64 = 5,
  CUDF_FLINK_GPU_DEVICE_BATCH_FLOAT32 = 6,
  CUDF_FLINK_GPU_DEVICE_BATCH_FLOAT64 = 7,
  CUDF_FLINK_GPU_DEVICE_BATCH_STRING = 8,
  CUDF_FLINK_GPU_DEVICE_BATCH_TIMESTAMP3 = 9,
  CUDF_FLINK_GPU_DEVICE_BATCH_ARRAY = 10,
  CUDF_FLINK_GPU_DEVICE_BATCH_MAP = 11,
};

enum CudfFlinkGpuDeviceBatchStatus : std::int32_t {
  CUDF_FLINK_GPU_DEVICE_BATCH_OK = 0,
  CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT = 1,
  CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND = 2,
  CUDF_FLINK_GPU_DEVICE_BATCH_STALE = 3,
  CUDF_FLINK_GPU_DEVICE_BATCH_BUSY = 4,
  CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_DEVICE = 5,
  CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT = 6,
  CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_SCHEMA = 7,
  CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_CHAIN = 8,
  CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE = 9,
  CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR = 10,
  CUDF_FLINK_GPU_DEVICE_BATCH_OUT_OF_MEMORY = 11,
  CUDF_FLINK_GPU_DEVICE_BATCH_INTERNAL_ERROR = 12,
};

enum CudfFlinkGpuDeviceBatchDrainFlags : std::uint32_t {
  CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_NONE = 0,
  // Only legal after the owner has synchronized every producer/consumer stream. Used during
  // exceptional operator teardown to invalidate leaked leases without freeing in-flight memory.
  CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_FORCE = 1u << 0,
};

struct CudfFlinkGpuDeviceBatchDescriptor {
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  std::uint64_t format_id;
  std::uint64_t schema_fingerprint;
  std::uint64_t owner_chain_id;
  std::int32_t device_ordinal;
  std::uint32_t rows;
  std::uint32_t stride_bytes;
  std::uint32_t field_count;
  std::uint64_t data_bytes;
  std::uint32_t type_ids[CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS];
};

// stride_bytes>0 describes a fixed-width row layout and requires data_bytes=rows*stride_bytes.
// stride_bytes==0 delegates a variable-width/columnar slab to format_id; non-empty batches then
// require data_bytes>0. An empty batch in either mode has rows=0/data_bytes=0 and is represented by
// a normal non-zero registry token. A zero token is never an empty batch.

struct CudfFlinkGpuDeviceBatchBorrowedView {
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  void* device_data;
  CudfFlinkGpuDeviceBatchDescriptor descriptor;
};

struct CudfFlinkGpuDeviceBatchPoolStats {
  std::uint32_t struct_size;
  std::uint32_t abi_version;
  std::uint64_t allocated_bytes;
  std::uint64_t active_bytes;
  std::uint64_t peak_active_bytes;
  std::uint64_t active_batches;
  std::uint64_t active_references;
  std::uint64_t acquired_batches;
  std::uint64_t published_batches;
  std::uint64_t retained_references;
  std::uint64_t released_references;
  std::uint64_t consumer_waits;
  std::uint64_t consumed_batches;
  std::uint64_t drain_calls;
  std::uint64_t forced_leaked_references;
  std::uint64_t stale_token_rejections;
};

static_assert(sizeof(CudfFlinkGpuDeviceBatchDescriptor) == 120,
              "GpuDeviceBatch descriptor ABI size changed");
static_assert(alignof(CudfFlinkGpuDeviceBatchDescriptor) == 8,
              "GpuDeviceBatch descriptor ABI alignment changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchDescriptor, format_id) == 8,
              "GpuDeviceBatch descriptor format offset changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchDescriptor, device_ordinal) == 32,
              "GpuDeviceBatch descriptor device offset changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchDescriptor, data_bytes) == 48,
              "GpuDeviceBatch descriptor byte-size offset changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchDescriptor, type_ids) == 56,
              "GpuDeviceBatch descriptor type offset changed");
static_assert(sizeof(CudfFlinkGpuDeviceBatchBorrowedView) == 136,
              "GpuDeviceBatch borrowed-view ABI size changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchBorrowedView, device_data) == 8,
              "GpuDeviceBatch borrowed-view pointer offset changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchBorrowedView, descriptor) == 16,
              "GpuDeviceBatch borrowed-view descriptor offset changed");
static_assert(sizeof(CudfFlinkGpuDeviceBatchPoolStats) == 120,
              "GpuDeviceBatch stats ABI size changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchPoolStats, allocated_bytes) == 8,
              "GpuDeviceBatch stats allocation offset changed");
static_assert(offsetof(CudfFlinkGpuDeviceBatchPoolStats, stale_token_rejections) == 112,
              "GpuDeviceBatch stats stale-token offset changed");

extern "C" {

std::int32_t cudf_flink_gpu_device_batch_pool_create(
    std::int32_t device_ordinal,
    std::uint64_t owner_chain_id,
    std::uint64_t slot_bytes,
    std::uint32_t slot_count,
    std::uint64_t* pool_handle) noexcept;

// Marks the producer owner closed. Published retained batches keep the pool allocation alive until
// their final release; new acquire calls are rejected. CUDA teardown is explicit and retry-safe:
// any teardown failure for a valid handle retains the registry entry and every resource that did
// not report a successful destroy/free. The caller must retry close on the pool's device and must
// not report a clean close.
std::int32_t cudf_flink_gpu_device_batch_pool_close(
    std::uint64_t pool_handle) noexcept;

std::int32_t cudf_flink_gpu_device_batch_pool_acquire(
    std::uint64_t pool_handle,
    const CudfFlinkGpuDeviceBatchDescriptor* descriptor,
    std::uint64_t* batch_handle,
    void** producer_device_data) noexcept;

// Finalizes a data-dependent producer result while the slot is still Writing. Only rows and
// data_bytes may change; every capability/identity field from acquire remains immutable. A known
// output may skip this call. Seal freezes the descriptor and rejects later finalize calls.
std::int32_t cudf_flink_gpu_device_batch_finalize(
    std::uint64_t batch_handle,
    std::uint32_t rows,
    std::uint64_t data_bytes) noexcept;

// Records producer completion without making the token borrowable. The producer must decide any
// whole-batch fallback before publish.
std::int32_t cudf_flink_gpu_device_batch_seal(
    std::uint64_t batch_handle,
    cudaStream_t producer_stream) noexcept;

std::int32_t cudf_flink_gpu_device_batch_publish(
    std::uint64_t batch_handle) noexcept;

// Reclaims an unpublished batch. A Writing batch requires its non-default producer stream: the
// runtime records and synchronizes the slot's ready event before reuse. A Sealed batch already has
// that fence and ignores producer_stream, which may be null. This makes it impossible to reclaim a
// Writing allocation while previously launched producer work still owns it.
std::int32_t cudf_flink_gpu_device_batch_abort(
    std::uint64_t batch_handle,
    cudaStream_t producer_stream) noexcept;

std::int32_t cudf_flink_gpu_device_batch_retain(
    std::uint64_t batch_handle) noexcept;

std::int32_t cudf_flink_gpu_device_batch_descriptor(
    std::uint64_t batch_handle,
    CudfFlinkGpuDeviceBatchDescriptor* descriptor) noexcept;

std::int32_t cudf_flink_gpu_device_batch_borrow(
    std::uint64_t batch_handle,
    const CudfFlinkGpuDeviceBatchDescriptor* expected,
    cudaStream_t consumer_stream,
    CudfFlinkGpuDeviceBatchBorrowedView* view) noexcept;

std::int32_t cudf_flink_gpu_device_batch_mark_consumed(
    std::uint64_t batch_handle,
    cudaStream_t consumer_stream) noexcept;

// Exception-path equivalent of mark_consumed. It records the same stream completion event and
// drops the runtime's in-flight borrow reference, but does not count a successful consumption.
std::int32_t cudf_flink_gpu_device_batch_abort_consume(
    std::uint64_t batch_handle,
    cudaStream_t consumer_stream) noexcept;

// A CUDA error after logical final-reference reclamation leaves this exact opaque token as a
// close-retry capability. The caller must not borrow/retain it, and must retry release until OK;
// successfully destroyed pool resources are not destroyed again.
std::int32_t cudf_flink_gpu_device_batch_release(
    std::uint64_t batch_handle) noexcept;

std::int32_t cudf_flink_gpu_device_batch_pool_drain(
    std::uint64_t pool_handle,
    std::uint32_t flags,
    CudfFlinkGpuDeviceBatchPoolStats* stats) noexcept;

std::int32_t cudf_flink_gpu_device_batch_pool_stats(
    std::uint64_t pool_handle,
    CudfFlinkGpuDeviceBatchPoolStats* stats) noexcept;

const char* cudf_flink_gpu_device_batch_status_string(std::int32_t status) noexcept;

}  // extern "C"

#endif  // CUDF_FLINK_GPU_DEVICE_BATCH_H_
