// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Derived unchanged (except JNI symbol prefix) from Mega-Flink native/gpu_device_batch_jni.cu.

#include "gpu_device_batch.h"

#include <jni.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(std::int32_t target_device) {
    status_ = cudaGetDevice(&previous_device_);
    if (status_ == cudaSuccess && previous_device_ != target_device) {
      status_ = cudaSetDevice(target_device);
      restore_ = status_ == cudaSuccess;
    }
  }

  ~ScopedCudaDevice() {
    // Every successful operation calls restore() explicitly and propagates failure. This fallback
    // runs only while an earlier CUDA failure is already being returned to the caller.
    restore();
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  cudaError_t status() const { return status_; }

  cudaError_t restore() {
    if (!restore_) return cudaSuccess;
    const cudaError_t status = cudaSetDevice(previous_device_);
    if (status == cudaSuccess) restore_ = false;
    return status;
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
  cudaError_t status_ = cudaSuccess;
};

enum class SlotState {
  kFree,
  kWriting,
  kSealed,
  kReclaiming,
  kPublished,
  kBorrowed,
  kConsumed,
};

struct Slot {
  void* device_data = nullptr;
  cudaEvent_t ready = nullptr;
  cudaEvent_t done = nullptr;
  std::uint32_t generation = 0;
  std::uint32_t references = 0;
  SlotState state = SlotState::kFree;
  CudfFlinkGpuDeviceBatchDescriptor descriptor{};
};

struct Pool {
  std::uint32_t id = 0;
  std::int32_t device = -1;
  std::uint64_t owner_chain_id = 0;
  std::uint64_t slot_bytes = 0;
  bool owner_closed = false;
  // If the final Java lease release performed logical reclamation but CUDA teardown failed, only
  // this exact token may complete that release. The slot must not be made usable again because
  // some of its allocation resources may already have been destroyed successfully.
  std::uint64_t teardown_retry_batch_handle = 0;
  std::vector<Slot> slots;
  CudfFlinkGpuDeviceBatchPoolStats stats{};

  ~Pool() {
    // CUDA cleanup is deliberately never attempted from this non-reporting destructor. Every
    // registered pool is removed only after destroy_pool_resources() has reported full success.
    // A residual pointer here therefore represents an already-reported create/close failure and
    // is left to process teardown instead of risking an unobservable double destroy/free.
    for (Slot& slot : slots) {
      if (slot.ready != nullptr || slot.done != nullptr || slot.device_data != nullptr) {
        std::fprintf(stderr,
                     "GpuDeviceBatch pool %u destroyed with unreleased CUDA resources\n",
                     id);
        break;
      }
    }
  }
};

std::mutex registry_mutex;
std::unordered_map<std::uint32_t, std::shared_ptr<Pool>> pools;
std::uint32_t next_pool_id = 1;

constexpr std::uint32_t kMaximumSlots = 2;
constexpr std::uint32_t kMaximumGeneration = 0x7fffffffU;

std::uint64_t make_batch_handle(
    std::uint32_t pool_id,
    std::uint32_t slot,
    std::uint32_t generation) {
  return (static_cast<std::uint64_t>(pool_id) << 32) |
         (static_cast<std::uint64_t>(generation) << 1) |
         static_cast<std::uint64_t>(slot);
}

std::uint32_t handle_pool_id(std::uint64_t handle) {
  return static_cast<std::uint32_t>(handle >> 32);
}

std::uint32_t handle_slot(std::uint64_t handle) {
  return static_cast<std::uint32_t>(handle & 1U);
}

std::uint32_t handle_generation(std::uint64_t handle) {
  return static_cast<std::uint32_t>((handle & 0xffffffffULL) >> 1);
}

std::int32_t cuda_status(cudaError_t status) {
  return status == cudaSuccess ? CUDF_FLINK_GPU_DEVICE_BATCH_OK
                               : CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
}

bool valid_data_shape(
    std::uint32_t rows,
    std::uint32_t stride_bytes,
    std::uint64_t data_bytes) {
  if (rows == 0) return data_bytes == 0;
  if (stride_bytes == 0) return data_bytes > 0;
  return data_bytes == static_cast<std::uint64_t>(rows) * stride_bytes;
}

bool valid_descriptor(const CudfFlinkGpuDeviceBatchDescriptor* descriptor) {
  if (descriptor == nullptr ||
      descriptor->struct_size != sizeof(*descriptor) ||
      descriptor->abi_version != CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION ||
      descriptor->format_id == 0 || descriptor->schema_fingerprint == 0 ||
      descriptor->owner_chain_id == 0 || descriptor->device_ordinal < 0 ||
      descriptor->field_count == 0 ||
      descriptor->field_count > CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS) {
    return false;
  }
  if (!valid_data_shape(
          descriptor->rows,
          descriptor->stride_bytes,
          descriptor->data_bytes)) return false;
  for (std::uint32_t index = 0;
       index < CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS;
       ++index) {
    const std::uint32_t type = descriptor->type_ids[index];
    if (index < descriptor->field_count) {
      if (type < CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN ||
          type > CUDF_FLINK_GPU_DEVICE_BATCH_MAP) {
        return false;
      }
    } else if (type != 0) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<Pool> pool_locked(std::uint64_t pool_handle) {
  if (pool_handle == 0 || pool_handle > 0xffffffffULL) return nullptr;
  const auto found = pools.find(static_cast<std::uint32_t>(pool_handle));
  return found == pools.end() ? nullptr : found->second;
}

std::int32_t batch_locked(
    std::uint64_t handle,
    std::shared_ptr<Pool>* pool,
    Slot** slot) {
  const auto found = pools.find(handle_pool_id(handle));
  if (found == pools.end()) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
  const std::uint32_t slot_index = handle_slot(handle);
  if (slot_index >= found->second->slots.size()) {
    found->second->stats.stale_token_rejections++;
    return CUDF_FLINK_GPU_DEVICE_BATCH_STALE;
  }
  Slot& candidate = found->second->slots[slot_index];
  if (candidate.state == SlotState::kFree || candidate.references == 0 ||
      candidate.generation != handle_generation(handle)) {
    found->second->stats.stale_token_rejections++;
    return CUDF_FLINK_GPU_DEVICE_BATCH_STALE;
  }
  *pool = found->second;
  *slot = &candidate;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}

bool all_slots_free(const Pool& pool) {
  return std::all_of(
      pool.slots.begin(), pool.slots.end(), [](const Slot& slot) {
        return slot.state == SlotState::kFree;
      });
}

bool all_pool_resources_released(const Pool& pool) {
  return std::all_of(
      pool.slots.begin(), pool.slots.end(), [](const Slot& slot) {
        return slot.ready == nullptr && slot.done == nullptr && slot.device_data == nullptr;
      });
}

void remember_first_cuda_error(cudaError_t status, cudaError_t* first_error) {
  if (status != cudaSuccess && *first_error == cudaSuccess) *first_error = status;
}

// Explicit, retry-safe teardown. Successful items are nulled immediately and are never destroyed
// again; failed items remain addressable for a same-device retry. All independent resources and
// device restoration are attempted, while the first CUDA failure determines the returned status.
std::int32_t destroy_pool_resources(Pool* pool, bool reject_pending_cuda_error = true) {
  if (pool == nullptr || pool->device < 0) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  int previous_device = -1;
  cudaError_t status = cudaGetDevice(&previous_device);
  if (status != cudaSuccess) return cuda_status(status);

  const bool restore_device = previous_device != pool->device;
  if (restore_device) {
    status = cudaSetDevice(pool->device);
    if (status != cudaSuccess) return cuda_status(status);
  }

  cudaError_t first_error = cudaSuccess;
  // CUDA teardown APIs may otherwise surface a prior asynchronous launch error after also changing
  // resource state. Drain that error first, report it, and leave every handle untouched for retry.
  if (reject_pending_cuda_error) {
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      first_error = status;
      if (restore_device) {
        remember_first_cuda_error(cudaSetDevice(previous_device), &first_error);
      }
      return cuda_status(first_error);
    }
  }
  for (Slot& slot : pool->slots) {
    if (slot.ready != nullptr) {
      status = cudaEventDestroy(slot.ready);
      if (status == cudaSuccess) {
        slot.ready = nullptr;
      } else {
        remember_first_cuda_error(status, &first_error);
      }
    }
    if (slot.done != nullptr) {
      status = cudaEventDestroy(slot.done);
      if (status == cudaSuccess) {
        slot.done = nullptr;
      } else {
        remember_first_cuda_error(status, &first_error);
      }
    }
    if (slot.device_data != nullptr) {
      status = cudaFree(slot.device_data);
      if (status == cudaSuccess) {
        slot.device_data = nullptr;
      } else {
        remember_first_cuda_error(status, &first_error);
      }
    }
  }

  if (restore_device) {
    remember_first_cuda_error(cudaSetDevice(previous_device), &first_error);
  }
  if (first_error != cudaSuccess) return cuda_status(first_error);
  return all_pool_resources_released(*pool)
             ? CUDF_FLINK_GPU_DEVICE_BATCH_OK
             : CUDF_FLINK_GPU_DEVICE_BATCH_INTERNAL_ERROR;
}

void clear_slot_locked(Pool* pool, Slot* slot) {
  if (slot->state == SlotState::kFree) return;
  pool->stats.active_bytes -= slot->descriptor.data_bytes;
  pool->stats.active_batches--;
  slot->state = SlotState::kFree;
  slot->references = 0;
  std::memset(&slot->descriptor, 0, sizeof(slot->descriptor));
}

std::int32_t maybe_remove_closed_pool_locked(
    const std::shared_ptr<Pool>& pool,
    std::uint64_t completing_batch_handle = 0) {
  if (!pool->owner_closed || !all_slots_free(*pool)) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
  }
  const std::int32_t status = destroy_pool_resources(pool.get());
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  // A pool-close retry may finish CUDA cleanup after a final batch release failed. Retain the
  // registry tombstone until that exact Java lease retries, so Java never holds a dangling token.
  if (pool->teardown_retry_batch_handle != 0 &&
      completing_batch_handle != pool->teardown_retry_batch_handle) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
  }
  pool->teardown_retry_batch_handle = 0;
  pools.erase(pool->id);
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}

std::int32_t retry_batch_teardown_locked(std::uint64_t batch_handle) {
  const auto found = pools.find(handle_pool_id(batch_handle));
  if (found == pools.end()) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
  std::shared_ptr<Pool> pool = found->second;
  if (!pool->owner_closed || !all_slots_free(*pool) ||
      pool->teardown_retry_batch_handle != batch_handle) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_STALE;
  }
  return maybe_remove_closed_pool_locked(pool, batch_handle);
}

std::int32_t descriptor_matches(
    const CudfFlinkGpuDeviceBatchDescriptor& actual,
    const CudfFlinkGpuDeviceBatchDescriptor& expected) {
  if (actual.device_ordinal != expected.device_ordinal) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_DEVICE;
  }
  if (actual.format_id != expected.format_id ||
      actual.stride_bytes != expected.stride_bytes) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT;
  }
  if (actual.schema_fingerprint != expected.schema_fingerprint) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_SCHEMA;
  }
  if (actual.field_count != expected.field_count ||
      std::memcmp(
          actual.type_ids,
          expected.type_ids,
          sizeof(actual.type_ids)) != 0) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_SCHEMA;
  }
  if (actual.owner_chain_id != expected.owner_chain_id) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_CHAIN;
  }
  if (actual.rows != expected.rows || actual.data_bytes != expected.data_bytes) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT;
  }
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}

void copy_stats(
    const Pool& pool,
    CudfFlinkGpuDeviceBatchPoolStats* output) {
  if (output == nullptr) return;
  *output = pool.stats;
  output->struct_size = sizeof(*output);
  output->abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
}

void throw_java(JNIEnv* env, const char* operation, std::int32_t status) {
  jclass exception_class = env->FindClass("java/lang/IllegalStateException");
  if (exception_class == nullptr) return;
  const char* status_text = cudf_flink_gpu_device_batch_status_string(status);
  char message[256];
  std::snprintf(
      message, sizeof(message), "%s failed: %s (%d)", operation, status_text, status);
  env->ThrowNew(exception_class, message);
}

}  // namespace

namespace {

struct LimitCoreResources {
  long long* device_input = nullptr;
  long long* device_output = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  ~LimitCoreResources() {
    if (stream != nullptr) cudaStreamSynchronize(stream);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    if (device_output != nullptr) cudaFree(device_output);
    if (device_input != nullptr) cudaFree(device_input);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

void require_limit_core_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(status) +
        " (" + cudaGetErrorString(status) + ")");
  }
}

__global__ void copy_limit_ordinals(
    const long long* input,
    int begin,
    int rows,
    long long* output) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    output[row] = input[begin + row];
  }
}

void throw_limit_core_java(JNIEnv* env, const std::string& message) {
  if (env->ExceptionCheck()) return;
  jclass exception_class = env->FindClass("java/lang/IllegalStateException");
  if (exception_class != nullptr) env->ThrowNew(exception_class, message.c_str());
}

}  // namespace

#define CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES                           \
  catch (const std::bad_alloc&) {                                           \
    return CUDF_FLINK_GPU_DEVICE_BATCH_OUT_OF_MEMORY;                       \
  }                                                                         \
  catch (...) {                                                             \
    return CUDF_FLINK_GPU_DEVICE_BATCH_INTERNAL_ERROR;                      \
  }

extern "C" std::int32_t cudf_flink_gpu_device_batch_pool_create(
    std::int32_t device_ordinal,
    std::uint64_t owner_chain_id,
    std::uint64_t slot_bytes,
    std::uint32_t slot_count,
    std::uint64_t* pool_handle) noexcept try {
  if (device_ordinal < 0 || owner_chain_id == 0 || slot_bytes == 0 ||
      slot_count == 0 || slot_count > kMaximumSlots || pool_handle == nullptr) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  if (slot_bytes > std::numeric_limits<std::size_t>::max() ||
      slot_bytes >
          std::numeric_limits<std::uint64_t>::max() / slot_count) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  ScopedCudaDevice device_scope(device_ordinal);
  if (device_scope.status() != cudaSuccess) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
  }

  auto pool = std::make_shared<Pool>();
  pool->device = device_ordinal;
  pool->owner_chain_id = owner_chain_id;
  pool->slot_bytes = slot_bytes;
  pool->slots.resize(slot_count);
  pool->stats.struct_size = sizeof(pool->stats);
  pool->stats.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  pool->stats.allocated_bytes = slot_bytes * slot_count;
  for (Slot& slot : pool->slots) {
    cudaError_t status = cudaMalloc(&slot.device_data, slot_bytes);
    if (status != cudaSuccess) {
      const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
      return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
                 ? cuda_status(status)
                 : cleanup_status;
    }
    status = cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
      return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
                 ? cuda_status(status)
                 : cleanup_status;
    }
    status = cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming);
    if (status != cudaSuccess) {
      const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
      return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
                 ? cuda_status(status)
                 : cleanup_status;
    }
  }

  const cudaError_t restore_status = device_scope.restore();
  if (restore_status != cudaSuccess) {
    const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
    return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
               ? cuda_status(restore_status)
               : cleanup_status;
  }

  std::lock_guard<std::mutex> lock(registry_mutex);
  if (next_pool_id == 0) {
    const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
    return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
               ? CUDF_FLINK_GPU_DEVICE_BATCH_BUSY
               : cleanup_status;
  }
  pool->id = next_pool_id;
  try {
    const auto inserted = pools.emplace(pool->id, pool);
    if (!inserted.second) {
      const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
      return cleanup_status == CUDF_FLINK_GPU_DEVICE_BATCH_OK
                 ? CUDF_FLINK_GPU_DEVICE_BATCH_BUSY
                 : cleanup_status;
    }
  } catch (...) {
    const std::int32_t cleanup_status = destroy_pool_resources(pool.get(), false);
    if (cleanup_status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return cleanup_status;
    throw;
  }
  next_pool_id++;
  *pool_handle = pool->id;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_pool_close(
    std::uint64_t pool_handle) noexcept try {
  std::shared_ptr<Pool> pool;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    pool = pool_locked(pool_handle);
    if (pool == nullptr) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
    pool->owner_closed = true;
    return maybe_remove_closed_pool_locked(pool);
  }
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_pool_acquire(
    std::uint64_t pool_handle,
    const CudfFlinkGpuDeviceBatchDescriptor* descriptor,
    std::uint64_t* batch_handle,
    void** producer_device_data) noexcept try {
  if (!valid_descriptor(descriptor) || batch_handle == nullptr ||
      producer_device_data == nullptr) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool = pool_locked(pool_handle);
  if (pool == nullptr) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
  if (pool->owner_closed) return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  if (descriptor->device_ordinal != pool->device) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_DEVICE;
  }
  if (descriptor->owner_chain_id != pool->owner_chain_id) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_CHAIN;
  }
  if (descriptor->data_bytes > pool->slot_bytes) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT;
  }
  for (std::uint32_t index = 0; index < pool->slots.size(); ++index) {
    Slot& slot = pool->slots[index];
    if (slot.state != SlotState::kFree) continue;
    slot.generation =
        slot.generation == kMaximumGeneration ? 1 : slot.generation + 1;
    slot.references = 1;
    slot.state = SlotState::kWriting;
    slot.descriptor = *descriptor;
    pool->stats.active_bytes += descriptor->data_bytes;
    pool->stats.peak_active_bytes = std::max(
        pool->stats.peak_active_bytes, pool->stats.active_bytes);
    pool->stats.active_batches++;
    pool->stats.active_references++;
    pool->stats.acquired_batches++;
    *batch_handle = make_batch_handle(pool->id, index, slot.generation);
    *producer_device_data = slot.device_data;
    return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
  }
  return CUDF_FLINK_GPU_DEVICE_BATCH_BUSY;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_finalize(
    std::uint64_t batch_handle,
    std::uint32_t rows,
    std::uint64_t data_bytes) noexcept try {
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kWriting) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  if (!valid_data_shape(rows, slot->descriptor.stride_bytes, data_bytes) ||
      data_bytes > pool->slot_bytes) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT;
  }
  const std::uint64_t old_bytes = slot->descriptor.data_bytes;
  if (data_bytes >= old_bytes) {
    pool->stats.active_bytes += data_bytes - old_bytes;
  } else {
    pool->stats.active_bytes -= old_bytes - data_bytes;
  }
  pool->stats.peak_active_bytes = std::max(
      pool->stats.peak_active_bytes, pool->stats.active_bytes);
  slot->descriptor.rows = rows;
  slot->descriptor.data_bytes = data_bytes;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_seal(
    std::uint64_t batch_handle,
    cudaStream_t producer_stream) noexcept try {
  if (producer_stream == nullptr) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kWriting) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  ScopedCudaDevice device_scope(pool->device);
  if (device_scope.status() != cudaSuccess) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
  }
  const cudaError_t event_status = cudaEventRecord(slot->ready, producer_stream);
  if (event_status != cudaSuccess) return cuda_status(event_status);
  const cudaError_t restore_status = device_scope.restore();
  if (restore_status != cudaSuccess) return cuda_status(restore_status);
  slot->state = SlotState::kSealed;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_publish(
    std::uint64_t batch_handle) noexcept try {
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kSealed) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  slot->state = SlotState::kPublished;
  pool->stats.published_batches++;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_abort(
    std::uint64_t batch_handle,
    cudaStream_t producer_stream) noexcept try {
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  cudaEvent_t wait_event = nullptr;
  std::uint32_t generation = 0;
  SlotState expected_state = SlotState::kFree;
  bool record_writing_fence = false;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
    if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
      return status == CUDF_FLINK_GPU_DEVICE_BATCH_STALE
                 ? retry_batch_teardown_locked(batch_handle)
                 : status;
    }
    if ((slot->state != SlotState::kWriting && slot->state != SlotState::kSealed) ||
        slot->references != 1) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
    }
    if (slot->state == SlotState::kWriting && producer_stream == nullptr) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
    }
    wait_event = slot->ready;
    record_writing_fence = slot->state == SlotState::kWriting;
    generation = slot->generation;
    expected_state = slot->state;
    // Claim reclamation before dropping the registry lock. Seal/publish/retain/release all reject
    // this transition state, so no caller can win a race while abort waits for producer work.
    slot->state = SlotState::kReclaiming;
  }
  ScopedCudaDevice device_scope(pool->device);
  if (device_scope.status() != cudaSuccess) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (slot->generation == generation &&
        slot->state == SlotState::kReclaiming) {
      slot->state = expected_state;
    }
    return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
  }
  if (record_writing_fence) {
    const cudaError_t record_status = cudaEventRecord(wait_event, producer_stream);
    if (record_status != cudaSuccess) {
      std::lock_guard<std::mutex> lock(registry_mutex);
      if (slot->generation == generation &&
          slot->state == SlotState::kReclaiming) {
        slot->state = expected_state;
      }
      return cuda_status(record_status);
    }
  }
  const cudaError_t wait_status = cudaEventSynchronize(wait_event);
  if (wait_status != cudaSuccess) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (slot->generation == generation &&
        slot->state == SlotState::kReclaiming) {
      slot->state = expected_state;
    }
    return cuda_status(wait_status);
  }
  const cudaError_t restore_status = device_scope.restore();
  if (restore_status != cudaSuccess) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (slot->generation == generation &&
        slot->state == SlotState::kReclaiming) {
      slot->state = expected_state;
    }
    return cuda_status(restore_status);
  }
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (slot->generation != generation || slot->references != 1 ||
        slot->state != SlotState::kReclaiming) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_STALE;
    }
    pool->stats.active_references--;
    clear_slot_locked(pool.get(), slot);
    if (pool->owner_closed && all_slots_free(*pool)) {
      pool->teardown_retry_batch_handle = batch_handle;
    }
    return maybe_remove_closed_pool_locked(pool, batch_handle);
  }
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_retain(
    std::uint64_t batch_handle) noexcept try {
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kPublished &&
      slot->state != SlotState::kBorrowed &&
      slot->state != SlotState::kConsumed) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  if (slot->references == std::numeric_limits<std::uint32_t>::max()) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_BUSY;
  }
  slot->references++;
  pool->stats.active_references++;
  pool->stats.retained_references++;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_descriptor(
    std::uint64_t batch_handle,
    CudfFlinkGpuDeviceBatchDescriptor* descriptor) noexcept try {
  if (descriptor == nullptr || descriptor->struct_size != sizeof(*descriptor) ||
      descriptor->abi_version != CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kPublished &&
      slot->state != SlotState::kBorrowed &&
      slot->state != SlotState::kConsumed) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  *descriptor = slot->descriptor;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_borrow(
    std::uint64_t batch_handle,
    const CudfFlinkGpuDeviceBatchDescriptor* expected,
    cudaStream_t consumer_stream,
    CudfFlinkGpuDeviceBatchBorrowedView* view) noexcept try {
  if (!valid_descriptor(expected) || consumer_stream == nullptr || view == nullptr ||
      view->struct_size != sizeof(*view) ||
      view->abi_version != CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  std::int32_t status = batch_locked(batch_handle, &pool, &slot);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->state != SlotState::kPublished) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
  }
  status = descriptor_matches(slot->descriptor, *expected);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
  if (slot->references == std::numeric_limits<std::uint32_t>::max()) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_BUSY;
  }
  ScopedCudaDevice device_scope(pool->device);
  if (device_scope.status() != cudaSuccess) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
  }
  const cudaError_t wait_status = cudaStreamWaitEvent(consumer_stream, slot->ready, 0);
  if (wait_status != cudaSuccess) return cuda_status(wait_status);
  const cudaError_t restore_status = device_scope.restore();
  if (restore_status != cudaSuccess) return cuda_status(restore_status);
  slot->state = SlotState::kBorrowed;
  // The runtime owns an in-flight reference until mark_consumed/abort_consume records a done
  // event. Closing the Java lease concurrently therefore cannot reclaim the slot under a kernel.
  slot->references++;
  pool->stats.active_references++;
  pool->stats.consumer_waits++;
  view->device_data = slot->device_data;
  view->descriptor = slot->descriptor;
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

std::int32_t finish_consume(
    std::uint64_t batch_handle,
    cudaStream_t consumer_stream,
    bool successful) {
  if (consumer_stream == nullptr) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::int32_t status = batch_locked(batch_handle, &pool, &slot);
    if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) return status;
    if (slot->state != SlotState::kBorrowed || slot->references == 0) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
    }
    ScopedCudaDevice device_scope(pool->device);
    if (device_scope.status() != cudaSuccess) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
    }
    const cudaError_t event_status = cudaEventRecord(slot->done, consumer_stream);
    if (event_status != cudaSuccess) return cuda_status(event_status);
    const cudaError_t restore_status = device_scope.restore();
    if (restore_status != cudaSuccess) return cuda_status(restore_status);
    slot->state = SlotState::kConsumed;
  }
  // Drop only the runtime-owned in-flight reference. The ordinary Java consumer lease remains.
  const std::int32_t release_status =
      cudf_flink_gpu_device_batch_release(batch_handle);
  if (release_status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    return release_status;
  }
  if (successful) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    pool->stats.consumed_batches++;
  }
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}

extern "C" std::int32_t cudf_flink_gpu_device_batch_mark_consumed(
    std::uint64_t batch_handle,
    cudaStream_t consumer_stream) noexcept try {
  return finish_consume(batch_handle, consumer_stream, true);
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_abort_consume(
    std::uint64_t batch_handle,
    cudaStream_t consumer_stream) noexcept try {
  return finish_consume(batch_handle, consumer_stream, false);
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_release(
    std::uint64_t batch_handle) noexcept try {
  std::shared_ptr<Pool> pool;
  Slot* slot = nullptr;
  cudaEvent_t wait_event = nullptr;
  std::uint32_t generation = 0;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const std::int32_t status = batch_locked(batch_handle, &pool, &slot);
    if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
      return status == CUDF_FLINK_GPU_DEVICE_BATCH_STALE
                 ? retry_batch_teardown_locked(batch_handle)
                 : status;
    }
    if (slot->state != SlotState::kPublished &&
        slot->state != SlotState::kBorrowed &&
        slot->state != SlotState::kConsumed) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
    }
    // A sole kBorrowed reference is the runtime's in-flight protection and can only be dropped by
    // mark_consumed/abort_consume after recording the done event.
    if (slot->state == SlotState::kBorrowed && slot->references == 1) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE;
    }
    if (slot->references > 1) {
      slot->references--;
      pool->stats.active_references--;
      pool->stats.released_references++;
      return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
    }
    generation = slot->generation;
    wait_event = slot->state == SlotState::kConsumed ? slot->done : slot->ready;
  }
  ScopedCudaDevice device_scope(pool->device);
  if (device_scope.status() != cudaSuccess) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR;
  }
  const cudaError_t event_status = cudaEventSynchronize(wait_event);
  if (event_status != cudaSuccess) return cuda_status(event_status);
  const cudaError_t restore_status = device_scope.restore();
  if (restore_status != cudaSuccess) return cuda_status(restore_status);
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (slot->generation != generation || slot->references == 0) {
      return CUDF_FLINK_GPU_DEVICE_BATCH_STALE;
    }
    // A concurrent retain while the event was synchronizing keeps the batch live; release only
    // this caller's reference and leave reclamation to the eventual final release.
    slot->references--;
    pool->stats.active_references--;
    pool->stats.released_references++;
    if (slot->references != 0) return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
    clear_slot_locked(pool.get(), slot);
    if (pool->owner_closed && all_slots_free(*pool)) {
      pool->teardown_retry_batch_handle = batch_handle;
    }
    return maybe_remove_closed_pool_locked(pool, batch_handle);
  }
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_pool_drain(
    std::uint64_t pool_handle,
    std::uint32_t flags,
    CudfFlinkGpuDeviceBatchPoolStats* stats) noexcept try {
  if ((flags & ~CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_FORCE) != 0 ||
      stats == nullptr || stats->struct_size != sizeof(*stats) ||
      stats->abi_version != CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::shared_ptr<Pool> pool;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    pool = pool_locked(pool_handle);
    if (pool == nullptr) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
    pool->stats.drain_calls++;
    if (pool->stats.active_references != 0 &&
        (flags & CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_FORCE) == 0) {
      copy_stats(*pool, stats);
      return CUDF_FLINK_GPU_DEVICE_BATCH_BUSY;
    }
  }
  if ((flags & CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_FORCE) != 0) {
    // The caller contract requires all producer and consumer streams to be synchronized first.
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (Slot& slot : pool->slots) {
      if (slot.state == SlotState::kFree) continue;
      pool->stats.forced_leaked_references += slot.references;
      pool->stats.active_references -= slot.references;
      clear_slot_locked(pool.get(), &slot);
    }
  }
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    copy_stats(*pool, stats);
    return maybe_remove_closed_pool_locked(pool);
  }
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" std::int32_t cudf_flink_gpu_device_batch_pool_stats(
    std::uint64_t pool_handle,
    CudfFlinkGpuDeviceBatchPoolStats* stats) noexcept try {
  if (stats == nullptr || stats->struct_size != sizeof(*stats) ||
      stats->abi_version != CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION) {
    return CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  std::shared_ptr<Pool> pool = pool_locked(pool_handle);
  if (pool == nullptr) return CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND;
  copy_stats(*pool, stats);
  return CUDF_FLINK_GPU_DEVICE_BATCH_OK;
}
CUDF_FLINK_GPU_DEVICE_BATCH_C_ABI_CATCHES

extern "C" const char* cudf_flink_gpu_device_batch_status_string(
    std::int32_t status) noexcept {
  switch (status) {
    case CUDF_FLINK_GPU_DEVICE_BATCH_OK: return "ok";
    case CUDF_FLINK_GPU_DEVICE_BATCH_INVALID_ARGUMENT: return "invalid argument";
    case CUDF_FLINK_GPU_DEVICE_BATCH_NOT_FOUND: return "not found";
    case CUDF_FLINK_GPU_DEVICE_BATCH_STALE: return "stale token";
    case CUDF_FLINK_GPU_DEVICE_BATCH_BUSY: return "busy";
    case CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_DEVICE: return "wrong device";
    case CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_LAYOUT: return "wrong layout";
    case CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_SCHEMA: return "wrong schema";
    case CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_CHAIN: return "wrong chain";
    case CUDF_FLINK_GPU_DEVICE_BATCH_WRONG_STATE: return "wrong state";
    case CUDF_FLINK_GPU_DEVICE_BATCH_CUDA_ERROR: return "CUDA error";
    case CUDF_FLINK_GPU_DEVICE_BATCH_OUT_OF_MEMORY: return "out of memory";
    case CUDF_FLINK_GPU_DEVICE_BATCH_INTERNAL_ERROR: return "internal error";
    default: return "unknown status";
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_GpuDeviceBatch_nativeRetain(
    JNIEnv* env, jclass, jlong handle) {
  const std::int32_t status =
      cudf_flink_gpu_device_batch_retain(static_cast<std::uint64_t>(handle));
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    throw_java(env, "GpuDeviceBatch retain", status);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_GpuDeviceBatch_nativeRelease(
    JNIEnv* env, jclass, jlong handle) {
  const std::int32_t status =
      cudf_flink_gpu_device_batch_release(static_cast<std::uint64_t>(handle));
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    throw_java(env, "GpuDeviceBatch release", status);
  }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_GpuDeviceBatch_nativeDescribe(
    JNIEnv* env, jclass, jlong handle) {
  CudfFlinkGpuDeviceBatchDescriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  const std::int32_t status = cudf_flink_gpu_device_batch_descriptor(
      static_cast<std::uint64_t>(handle), &descriptor);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    throw_java(env, "GpuDeviceBatch describe", status);
    return nullptr;
  }
  constexpr jsize kDescriptorLongs = 9 + CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS;
  jlong values[kDescriptorLongs]{};
  values[0] = descriptor.abi_version;
  values[1] = static_cast<jlong>(descriptor.format_id);
  values[2] = static_cast<jlong>(descriptor.schema_fingerprint);
  values[3] = static_cast<jlong>(descriptor.owner_chain_id);
  values[4] = descriptor.device_ordinal;
  values[5] = descriptor.rows;
  values[6] = descriptor.stride_bytes;
  values[7] = static_cast<jlong>(descriptor.data_bytes);
  values[8] = descriptor.field_count;
  for (std::uint32_t index = 0;
       index < CUDF_FLINK_GPU_DEVICE_BATCH_MAX_FIELDS;
       ++index) {
    values[9 + index] = descriptor.type_ids[index];
  }
  jlongArray result = env->NewLongArray(kDescriptorLongs);
  if (result != nullptr) env->SetLongArrayRegion(result, 0, kDescriptorLongs, values);
  return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_LimitKernel_nativeCopyDeviceOrdinals(
    JNIEnv* env,
    jclass,
    jlongArray input_sequence,
    jint begin,
    jint rows,
    jlongArray output_sequence) {
  try {
    if (input_sequence == nullptr || output_sequence == nullptr || begin < 0 ||
        rows <= 0 || env->GetArrayLength(output_sequence) != rows ||
        begin > env->GetArrayLength(input_sequence) - rows) {
      throw std::invalid_argument("Limit device ordinal shape is invalid");
    }
    const jsize input_rows = env->GetArrayLength(input_sequence);
    std::vector<jlong> host_input(static_cast<std::size_t>(input_rows));
    std::vector<jlong> host_output(static_cast<std::size_t>(rows));
    env->GetLongArrayRegion(input_sequence, 0, input_rows, host_input.data());
    if (env->ExceptionCheck()) return 0;

    LimitCoreResources resources;
    require_limit_core_cuda(
        cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
        "create Limit core stream");
    require_limit_core_cuda(
        cudaEventCreate(&resources.start), "create Limit core start event");
    require_limit_core_cuda(
        cudaEventCreate(&resources.stop), "create Limit core stop event");
    require_limit_core_cuda(
        cudaMalloc(
            &resources.device_input,
            static_cast<std::size_t>(input_rows) * sizeof(long long)),
        "allocate Limit core input");
    require_limit_core_cuda(
        cudaMalloc(
            &resources.device_output,
            static_cast<std::size_t>(rows) * sizeof(long long)),
        "allocate Limit core output");
    require_limit_core_cuda(
        cudaMemcpyAsync(
            resources.device_input,
            host_input.data(),
            static_cast<std::size_t>(input_rows) * sizeof(long long),
            cudaMemcpyHostToDevice,
            resources.stream),
        "copy Limit core input");
    require_limit_core_cuda(
        cudaEventRecord(resources.start, resources.stream),
        "record Limit core start");
    constexpr int kThreads = 256;
    const int blocks = std::min(256, (rows + kThreads - 1) / kThreads);
    copy_limit_ordinals<<<blocks, kThreads, 0, resources.stream>>>(
        resources.device_input, begin, rows, resources.device_output);
    require_limit_core_cuda(cudaGetLastError(), "launch Limit core kernel");
    require_limit_core_cuda(
        cudaEventRecord(resources.stop, resources.stream),
        "record Limit core stop");
    require_limit_core_cuda(
        cudaMemcpyAsync(
            host_output.data(),
            resources.device_output,
            static_cast<std::size_t>(rows) * sizeof(long long),
            cudaMemcpyDeviceToHost,
            resources.stream),
        "copy Limit core output");
    require_limit_core_cuda(
        cudaStreamSynchronize(resources.stream), "wait for Limit core output");
    float milliseconds = 0.0f;
    require_limit_core_cuda(
        cudaEventElapsedTime(&milliseconds, resources.start, resources.stop),
        "measure Limit core kernel");
    env->SetLongArrayRegion(output_sequence, 0, rows, host_output.data());
    if (env->ExceptionCheck()) return 0;
    return std::max<jlong>(
        1, static_cast<jlong>(milliseconds * 1'000'000.0f));
  } catch (const std::exception& failure) {
    throw_limit_core_java(env, failure.what());
    return 0;
  }
}
