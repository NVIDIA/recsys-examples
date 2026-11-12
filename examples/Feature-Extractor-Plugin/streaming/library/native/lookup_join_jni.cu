// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Derived unchanged (except JNI symbol prefix) from Mega-Flink native/lookup_join_jni.cu.

#include <jni.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kThreads = 256;
constexpr int kEmptySlot = -1;
constexpr int kStatusOk = 0;
constexpr int kStatusDuplicateKey = 1;
constexpr int kStatusTableFull = 2;
constexpr int kStatusInjectedDeviceError = 3;
constexpr int kFilterNone = 0;
constexpr int kFilterKeyIsNull = 1;
constexpr int kFilterKeyIsNotNull = 2;
constexpr int kFilterPayloadIsNull = 3;
constexpr int kFilterPayloadIsNotNull = 4;
constexpr int kFilterKeyEqual = 5;
constexpr int kFilterKeyNotEqual = 6;
constexpr int kFilterKeyLess = 7;
constexpr int kFilterKeyLessEqual = 8;
constexpr int kFilterKeyGreater = 9;
constexpr int kFilterKeyGreaterEqual = 10;
constexpr int kFilterPayloadEqual = 11;
constexpr int kFilterPayloadNotEqual = 12;
constexpr int kFilterPayloadLess = 13;
constexpr int kFilterPayloadLessEqual = 14;
constexpr int kFilterPayloadGreater = 15;
constexpr int kFilterPayloadGreaterEqual = 16;
constexpr int kFilterMaximum = kFilterPayloadGreaterEqual;
constexpr int kTransferFlags = 1;
constexpr int kTransferPayload = 2;
constexpr int kTransferMaximum = kTransferFlags | kTransferPayload;
constexpr std::size_t kDeviceMemoryReserve = 64ULL * 1024ULL * 1024ULL;

struct alignas(64) Metrics {
  std::int64_t build_h2d_nanos;
  std::int64_t build_compute_nanos;
  std::int64_t probe_h2d_nanos;
  std::int64_t probe_compute_nanos;
  std::int64_t probe_d2h_nanos;
  std::int64_t generation;
  std::int32_t dimension_rows;
  std::int32_t probe_rows;
  std::int32_t match_count;
  std::int32_t device_status;
};

static_assert(sizeof(Metrics) == 64, "Lookup Join metrics must occupy one cache line");

struct Context {
  int device = -1;
  int dimension_capacity = 0;
  int probe_capacity = 0;
  int table_size = 0;
  int active_table = 0;
  bool built = false;
  bool teardown_started = false;
  bool teardown_synchronized = false;

  long long* host_dimension_keys = nullptr;
  std::uint8_t* host_dimension_key_valid = nullptr;
  long long* host_dimension_payloads = nullptr;
  std::uint8_t* host_dimension_payload_valid = nullptr;
  long long* host_probe_keys = nullptr;
  std::uint8_t* host_probe_key_valid = nullptr;
  long long* host_output_payloads = nullptr;
  std::uint8_t* host_output_payload_valid = nullptr;
  std::uint8_t* host_output_matched = nullptr;
  Metrics* host_metrics = nullptr;

  long long* device_dimension_keys[2] = {nullptr, nullptr};
  std::uint8_t* device_dimension_key_valid[2] = {nullptr, nullptr};
  long long* device_dimension_payloads[2] = {nullptr, nullptr};
  std::uint8_t* device_dimension_payload_valid[2] = {nullptr, nullptr};
  int* device_slots[2] = {nullptr, nullptr};
  long long* device_probe_keys = nullptr;
  std::uint8_t* device_probe_key_valid = nullptr;
  long long* device_output_payloads = nullptr;
  std::uint8_t* device_output_payload_valid = nullptr;
  std::uint8_t* device_output_matched = nullptr;
  int* device_match_count = nullptr;
  int* device_status = nullptr;

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t after_h2d = nullptr;
  cudaEvent_t after_compute = nullptr;
  cudaEvent_t after_d2h = nullptr;
};

void require_cuda(const char* operation, cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             cudaGetErrorName(error) + " (" +
                             cudaGetErrorString(error) + ")");
  }
}

std::size_t checked_bytes(int rows, std::size_t width, const char* label) {
  if (rows < 0 || static_cast<std::size_t>(rows) >
                      std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error(std::string(label) + " byte size overflow");
  }
  return static_cast<std::size_t>(rows) * width;
}

std::size_t checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error("Lookup Join allocation size overflow");
  }
  return left + right;
}

int table_size_for(int capacity) {
  if (capacity <= 0 || capacity > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("Lookup Join dimension capacity is out of range");
  }
  const int target = std::max(8, capacity * 2);
  int result = 1;
  while (result < target) {
    if (result > std::numeric_limits<int>::max() / 2) {
      throw std::overflow_error("Lookup Join hash table size overflow");
    }
    result <<= 1;
  }
  return result;
}

std::size_t required_device_bytes(int dimension_capacity,
                                  int probe_capacity,
                                  int table_size) {
  std::size_t bytes = 0;
  const std::size_t dimension_long_bytes =
      checked_bytes(dimension_capacity, sizeof(long long), "dimension");
  const std::size_t dimension_valid_bytes =
      checked_bytes(dimension_capacity, sizeof(std::uint8_t), "dimension validity");
  const std::size_t probe_long_bytes =
      checked_bytes(probe_capacity, sizeof(long long), "probe");
  const std::size_t probe_valid_bytes =
      checked_bytes(probe_capacity, sizeof(std::uint8_t), "probe validity");
  const std::size_t slot_bytes = checked_bytes(table_size, sizeof(int), "hash slots");
  for (int table = 0; table < 2; ++table) {
    bytes = checked_add(bytes, dimension_long_bytes);
    bytes = checked_add(bytes, dimension_valid_bytes);
    bytes = checked_add(bytes, dimension_long_bytes);
    bytes = checked_add(bytes, dimension_valid_bytes);
    bytes = checked_add(bytes, slot_bytes);
  }
  bytes = checked_add(bytes, probe_long_bytes);
  bytes = checked_add(bytes, probe_valid_bytes);
  bytes = checked_add(bytes, probe_long_bytes);
  bytes = checked_add(bytes, probe_valid_bytes);
  bytes = checked_add(bytes, probe_valid_bytes);
  bytes = checked_add(bytes, 2 * sizeof(int));
  return bytes;
}

void remember_teardown_error(
    const char* operation,
    cudaError_t error,
    std::string* failures) {
  if (error == cudaSuccess) return;
  if (!failures->empty()) failures->append("; ");
  failures->append(operation)
      .append(" failed: ")
      .append(cudaGetErrorName(error))
      .append(" (")
      .append(cudaGetErrorString(error))
      .append(")");
}

void destroy_event_for_teardown(
    const char* operation,
    cudaEvent_t* event,
    std::string* failures) {
  if (*event == nullptr) return;
  const cudaError_t error = cudaEventDestroy(*event);
  if (error == cudaSuccess) {
    *event = nullptr;
  } else {
    remember_teardown_error(operation, error, failures);
  }
}

void destroy_stream_for_teardown(
    const char* operation,
    cudaStream_t* stream,
    std::string* failures) {
  if (*stream == nullptr) return;
  const cudaError_t error = cudaStreamDestroy(*stream);
  if (error == cudaSuccess) {
    *stream = nullptr;
  } else {
    remember_teardown_error(operation, error, failures);
  }
}

template <typename T>
void free_device_for_teardown(
    const char* operation,
    T** pointer,
    std::string* failures) {
  if (*pointer == nullptr) return;
  const cudaError_t error = cudaFree(static_cast<void*>(*pointer));
  if (error == cudaSuccess) {
    *pointer = nullptr;
  } else {
    remember_teardown_error(operation, error, failures);
  }
}

template <typename T>
void free_host_for_teardown(
    const char* operation,
    T** pointer,
    std::string* failures) {
  if (*pointer == nullptr) return;
  const cudaError_t error = cudaFreeHost(static_cast<void*>(*pointer));
  if (error == cudaSuccess) {
    *pointer = nullptr;
  } else {
    remember_teardown_error(operation, error, failures);
  }
}

bool retains_owned_resources(const Context& context) {
  for (int table = 0; table < 2; ++table) {
    if (context.device_dimension_keys[table] != nullptr ||
        context.device_dimension_key_valid[table] != nullptr ||
        context.device_dimension_payloads[table] != nullptr ||
        context.device_dimension_payload_valid[table] != nullptr ||
        context.device_slots[table] != nullptr) {
      return true;
    }
  }
  return context.device_probe_keys != nullptr ||
         context.device_probe_key_valid != nullptr ||
         context.device_output_payloads != nullptr ||
         context.device_output_payload_valid != nullptr ||
         context.device_output_matched != nullptr ||
         context.device_match_count != nullptr || context.device_status != nullptr ||
         context.host_dimension_keys != nullptr ||
         context.host_dimension_key_valid != nullptr ||
         context.host_dimension_payloads != nullptr ||
         context.host_dimension_payload_valid != nullptr ||
         context.host_probe_keys != nullptr ||
         context.host_probe_key_valid != nullptr ||
         context.host_output_payloads != nullptr ||
         context.host_output_payload_valid != nullptr ||
         context.host_output_matched != nullptr || context.host_metrics != nullptr ||
         context.stream != nullptr || context.start != nullptr ||
         context.after_h2d != nullptr || context.after_compute != nullptr ||
         context.after_d2h != nullptr;
}

void destroy(Context* context, bool reject_pending_cuda_error = true) {
  if (context == nullptr) return;
  if (!retains_owned_resources(*context)) {
    delete context;
    return;
  }
  if (context->device < 0) {
    throw std::runtime_error(
        "Lookup Join teardown has resources without a CUDA device");
  }
  require_cuda(
      "bind Lookup Join teardown device", cudaSetDevice(context->device));
  context->teardown_started = true;
  if (!context->teardown_synchronized && context->stream != nullptr) {
    require_cuda(
        "synchronize Lookup Join teardown stream",
        cudaStreamSynchronize(context->stream));
    context->teardown_synchronized = true;
  } else if (context->stream == nullptr) {
    context->teardown_synchronized = true;
  }
  if (reject_pending_cuda_error) {
    require_cuda("drain pending Lookup Join teardown error", cudaGetLastError());
  } else {
    (void) cudaGetLastError();
  }

  std::string failures;
  destroy_event_for_teardown(
      "destroy Lookup Join start event", &context->start, &failures);
  destroy_event_for_teardown(
      "destroy Lookup Join H2D event", &context->after_h2d, &failures);
  destroy_event_for_teardown(
      "destroy Lookup Join compute event", &context->after_compute, &failures);
  destroy_event_for_teardown(
      "destroy Lookup Join D2H event", &context->after_d2h, &failures);
  for (int table = 0; table < 2; ++table) {
    free_device_for_teardown(
        "free Lookup Join dimension keys",
        &context->device_dimension_keys[table], &failures);
    free_device_for_teardown(
        "free Lookup Join dimension key validity",
        &context->device_dimension_key_valid[table], &failures);
    free_device_for_teardown(
        "free Lookup Join dimension payloads",
        &context->device_dimension_payloads[table], &failures);
    free_device_for_teardown(
        "free Lookup Join dimension payload validity",
        &context->device_dimension_payload_valid[table], &failures);
    free_device_for_teardown(
        "free Lookup Join hash slots", &context->device_slots[table], &failures);
  }
  free_device_for_teardown(
      "free Lookup Join probe keys", &context->device_probe_keys, &failures);
  free_device_for_teardown(
      "free Lookup Join probe key validity",
      &context->device_probe_key_valid, &failures);
  free_device_for_teardown(
      "free Lookup Join output payloads",
      &context->device_output_payloads, &failures);
  free_device_for_teardown(
      "free Lookup Join output payload validity",
      &context->device_output_payload_valid, &failures);
  free_device_for_teardown(
      "free Lookup Join output matches",
      &context->device_output_matched, &failures);
  free_device_for_teardown(
      "free Lookup Join match count", &context->device_match_count, &failures);
  free_device_for_teardown(
      "free Lookup Join device status", &context->device_status, &failures);
  destroy_stream_for_teardown(
      "destroy Lookup Join stream", &context->stream, &failures);
  free_host_for_teardown(
      "free Lookup Join host dimension keys",
      &context->host_dimension_keys, &failures);
  free_host_for_teardown(
      "free Lookup Join host dimension key validity",
      &context->host_dimension_key_valid, &failures);
  free_host_for_teardown(
      "free Lookup Join host dimension payloads",
      &context->host_dimension_payloads, &failures);
  free_host_for_teardown(
      "free Lookup Join host dimension payload validity",
      &context->host_dimension_payload_valid, &failures);
  free_host_for_teardown(
      "free Lookup Join host probe keys", &context->host_probe_keys, &failures);
  free_host_for_teardown(
      "free Lookup Join host probe key validity",
      &context->host_probe_key_valid, &failures);
  free_host_for_teardown(
      "free Lookup Join host output payloads",
      &context->host_output_payloads, &failures);
  free_host_for_teardown(
      "free Lookup Join host output payload validity",
      &context->host_output_payload_valid, &failures);
  free_host_for_teardown(
      "free Lookup Join host output matches",
      &context->host_output_matched, &failures);
  free_host_for_teardown(
      "free Lookup Join host metrics", &context->host_metrics, &failures);

  if (!failures.empty() || retains_owned_resources(*context)) {
    if (failures.empty()) {
      failures = "Lookup Join teardown retained an owned native resource";
    } else {
      failures.append(
          "; Lookup Join teardown retained owned native resources for retry");
    }
    throw std::runtime_error(failures);
  }
  delete context;
}

template <typename T>
void allocate_host(T** address, std::size_t bytes, const char* label) {
  require_cuda(label, cudaHostAlloc(reinterpret_cast<void**>(address), bytes,
                                    cudaHostAllocPortable));
}

template <typename T>
void allocate_device(T** address, std::size_t bytes, const char* label) {
  require_cuda(label, cudaMalloc(reinterpret_cast<void**>(address), bytes));
}

void allocate(Context* context) {
  const std::size_t required = required_device_bytes(
      context->dimension_capacity, context->probe_capacity, context->table_size);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  require_cuda("cudaMemGetInfo", cudaMemGetInfo(&free_bytes, &total_bytes));
  if (required > free_bytes || free_bytes - required < kDeviceMemoryReserve) {
    throw std::runtime_error(
        "Lookup Join capacity exceeds available device memory (required=" +
        std::to_string(required) + ", free=" + std::to_string(free_bytes) +
        ", reserve=" + std::to_string(kDeviceMemoryReserve) + ")");
  }

  const std::size_t dimension_long_bytes =
      checked_bytes(context->dimension_capacity, sizeof(long long), "dimension");
  const std::size_t dimension_valid_bytes =
      checked_bytes(context->dimension_capacity, sizeof(std::uint8_t), "dimension validity");
  const std::size_t probe_long_bytes =
      checked_bytes(context->probe_capacity, sizeof(long long), "probe");
  const std::size_t probe_valid_bytes =
      checked_bytes(context->probe_capacity, sizeof(std::uint8_t), "probe validity");
  const std::size_t slot_bytes =
      checked_bytes(context->table_size, sizeof(int), "hash slots");

  require_cuda("cudaStreamCreateWithFlags",
               cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
  require_cuda("cudaEventCreate start", cudaEventCreate(&context->start));
  require_cuda("cudaEventCreate after H2D", cudaEventCreate(&context->after_h2d));
  require_cuda("cudaEventCreate after compute", cudaEventCreate(&context->after_compute));
  require_cuda("cudaEventCreate after D2H", cudaEventCreate(&context->after_d2h));

  allocate_host(&context->host_dimension_keys, dimension_long_bytes,
                "cudaHostAlloc dimension keys");
  allocate_host(&context->host_dimension_key_valid, dimension_valid_bytes,
                "cudaHostAlloc dimension key validity");
  allocate_host(&context->host_dimension_payloads, dimension_long_bytes,
                "cudaHostAlloc dimension payloads");
  allocate_host(&context->host_dimension_payload_valid, dimension_valid_bytes,
                "cudaHostAlloc dimension payload validity");
  allocate_host(&context->host_probe_keys, probe_long_bytes, "cudaHostAlloc probe keys");
  allocate_host(&context->host_probe_key_valid, probe_valid_bytes,
                "cudaHostAlloc probe validity");
  allocate_host(&context->host_output_payloads, probe_long_bytes,
                "cudaHostAlloc output payloads");
  allocate_host(&context->host_output_payload_valid, probe_valid_bytes,
                "cudaHostAlloc output payload validity");
  allocate_host(&context->host_output_matched, probe_valid_bytes,
                "cudaHostAlloc output matches");
  allocate_host(&context->host_metrics, sizeof(Metrics), "cudaHostAlloc metrics");
  std::memset(context->host_metrics, 0, sizeof(Metrics));

  for (int table = 0; table < 2; ++table) {
    allocate_device(&context->device_dimension_keys[table], dimension_long_bytes,
                    "cudaMalloc dimension keys");
    allocate_device(&context->device_dimension_key_valid[table], dimension_valid_bytes,
                    "cudaMalloc dimension key validity");
    allocate_device(&context->device_dimension_payloads[table], dimension_long_bytes,
                    "cudaMalloc dimension payloads");
    allocate_device(&context->device_dimension_payload_valid[table], dimension_valid_bytes,
                    "cudaMalloc dimension payload validity");
    allocate_device(&context->device_slots[table], slot_bytes, "cudaMalloc hash slots");
  }
  allocate_device(&context->device_probe_keys, probe_long_bytes, "cudaMalloc probe keys");
  allocate_device(&context->device_probe_key_valid, probe_valid_bytes,
                  "cudaMalloc probe validity");
  allocate_device(&context->device_output_payloads, probe_long_bytes,
                  "cudaMalloc output payloads");
  allocate_device(&context->device_output_payload_valid, probe_valid_bytes,
                  "cudaMalloc output payload validity");
  allocate_device(&context->device_output_matched, probe_valid_bytes,
                  "cudaMalloc output matches");
  allocate_device(&context->device_match_count, sizeof(int), "cudaMalloc match count");
  allocate_device(&context->device_status, sizeof(int), "cudaMalloc device status");
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void build_table_kernel(const long long* keys,
                                   const std::uint8_t* key_valid,
                                   int rows,
                                   int* slots,
                                   int table_size,
                                   int* status) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || key_valid[row] == 0) return;

  const long long key = keys[row];
  const int mask = table_size - 1;
  int position = static_cast<int>(mix64(static_cast<unsigned long long>(key))) & mask;
  for (int step = 0; step < table_size; ++step) {
    const int previous = atomicCAS(&slots[position], kEmptySlot, row);
    if (previous == kEmptySlot) return;
    if (keys[previous] == key) {
      atomicCAS(status, kStatusOk, kStatusDuplicateKey);
      return;
    }
    position = (position + 1) & mask;
  }
  atomicCAS(status, kStatusOk, kStatusTableFull);
}

__device__ __forceinline__ bool compare_filter(long long value,
                                               long long literal,
                                               int operation,
                                               int equal_operation) {
  switch (operation - equal_operation) {
    case 0:
      return value == literal;
    case 1:
      return value != literal;
    case 2:
      return value < literal;
    case 3:
      return value <= literal;
    case 4:
      return value > literal;
    case 5:
      return value >= literal;
    default:
      return false;
  }
}

__device__ __forceinline__ bool select_lookup_result(
    int operation,
    long long literal,
    bool key_valid,
    long long key,
    bool payload_valid,
    long long payload) {
  switch (operation) {
    case kFilterNone:
      return true;
    case kFilterKeyIsNull:
      return !key_valid;
    case kFilterKeyIsNotNull:
      return key_valid;
    case kFilterPayloadIsNull:
      return !payload_valid;
    case kFilterPayloadIsNotNull:
      return payload_valid;
    default:
      if (operation >= kFilterKeyEqual && operation <= kFilterKeyGreaterEqual) {
        return key_valid &&
               compare_filter(key, literal, operation, kFilterKeyEqual);
      }
      if (operation >= kFilterPayloadEqual &&
          operation <= kFilterPayloadGreaterEqual) {
        return payload_valid &&
               compare_filter(payload, literal, operation, kFilterPayloadEqual);
      }
      return false;
  }
}

__global__ void lookup_kernel(const long long* probe_keys,
                              const std::uint8_t* probe_key_valid,
                              int probe_rows,
                              const long long* dimension_keys,
                              const long long* dimension_payloads,
                              const std::uint8_t* dimension_payload_valid,
                              const int* slots,
                              int table_size,
                              long long* output_payloads,
                              std::uint8_t* output_payload_valid,
                              std::uint8_t* output_matched,
                              int* match_count,
                              int* status,
                              bool inject_device_error,
                              int filter_operation,
                              long long filter_literal) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= probe_rows) return;
  if (inject_device_error) {
    if (row == 0) *status = kStatusInjectedDeviceError;
    return;
  }

  bool matched = false;
  int dimension_row = kEmptySlot;
  if (probe_key_valid[row] != 0) {
    const long long key = probe_keys[row];
    const int mask = table_size - 1;
    int position = static_cast<int>(mix64(static_cast<unsigned long long>(key))) & mask;
    for (int step = 0; step < table_size; ++step) {
      const int candidate = slots[position];
      if (candidate == kEmptySlot) break;
      if (dimension_keys[candidate] == key) {
        matched = true;
        dimension_row = candidate;
        break;
      }
      position = (position + 1) & mask;
    }
  }

  bool payload_valid = false;
  long long payload = 0;
  if (matched) {
    payload_valid = dimension_payload_valid[dimension_row] != 0;
    payload = payload_valid ? dimension_payloads[dimension_row] : 0;
    atomicAdd(match_count, 1);
  }
  output_payload_valid[row] = payload_valid ? 1 : 0;
  output_payloads[row] = payload;
  const bool selected = select_lookup_result(
      filter_operation, filter_literal, matched,
      matched ? probe_keys[row] : 0, payload_valid, payload);
  // Bit 0 preserves match state. Bit 1 carries a fused post-join filter decision without
  // increasing D2H bytes.
  output_matched[row] = (matched ? 1 : 0) | (selected ? 2 : 0);
}

std::int64_t elapsed_nanos(cudaEvent_t start, cudaEvent_t end) {
  float milliseconds = 0.0F;
  require_cuda("cudaEventElapsedTime", cudaEventElapsedTime(&milliseconds, start, end));
  return static_cast<std::int64_t>(milliseconds * 1000000.0F);
}

Context* checked_context(jlong handle) {
  if (handle == 0) throw std::invalid_argument("Lookup Join context is closed");
  Context* context = reinterpret_cast<Context*>(handle);
  if (context->teardown_started) {
    throw std::logic_error("Lookup Join native close retry is pending");
  }
  return context;
}

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
  jclass exception_class = env->FindClass(class_name);
  if (exception_class != nullptr) env->ThrowNew(exception_class, message.c_str());
}

void throw_failure(JNIEnv* env, const std::exception& failure) {
  throw_java(env, "java/lang/IllegalStateException", failure.what());
}

jobject direct_buffer(JNIEnv* env, void* address, std::size_t bytes) {
  return env->NewDirectByteBuffer(address, static_cast<jlong>(bytes));
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeCreate(
    JNIEnv* env, jclass, jint dimension_capacity, jint probe_capacity) {
  try {
    if (dimension_capacity <= 0 || probe_capacity <= 0) {
      throw std::invalid_argument("Lookup Join capacities must be positive");
    }
    const int table_size = table_size_for(dimension_capacity);
    Context* context = new Context();
    context->dimension_capacity = dimension_capacity;
    context->probe_capacity = probe_capacity;
    context->table_size = table_size;
    try {
      require_cuda("cudaGetDevice", cudaGetDevice(&context->device));
      allocate(context);
    } catch (const std::exception& failure) {
      std::string message = failure.what();
      try {
        destroy(context, false);
      } catch (const std::exception& cleanup_failure) {
        message.append("; Lookup Join construction cleanup failed: ")
            .append(cleanup_failure.what());
      } catch (...) {
        message.append(
            "; Lookup Join construction cleanup failed with an unknown error");
      }
      throw std::runtime_error(message);
    } catch (...) {
      std::string message = "Lookup Join construction failed with an unknown error";
      try {
        destroy(context, false);
      } catch (const std::exception& cleanup_failure) {
        message.append("; Lookup Join construction cleanup failed: ")
            .append(cleanup_failure.what());
      } catch (...) {
        message.append(
            "; Lookup Join construction cleanup failed with an unknown error");
      }
      throw std::runtime_error(message);
    }
    return reinterpret_cast<jlong>(context);
  } catch (const std::invalid_argument& failure) {
    throw_java(env, "java/lang/IllegalArgumentException", failure.what());
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
  }
  return 0;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeDimensionKeysBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_dimension_keys,
                         checked_bytes(context->dimension_capacity, sizeof(long long),
                                       "dimension keys"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeDimensionKeyValidBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_dimension_key_valid,
                         checked_bytes(context->dimension_capacity, sizeof(std::uint8_t),
                                       "dimension key validity"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeDimensionPayloadsBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_dimension_payloads,
                         checked_bytes(context->dimension_capacity, sizeof(long long),
                                       "dimension payloads"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeDimensionPayloadValidBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_dimension_payload_valid,
                         checked_bytes(context->dimension_capacity, sizeof(std::uint8_t),
                                       "dimension payload validity"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeProbeKeysBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_probe_keys,
                         checked_bytes(context->probe_capacity, sizeof(long long),
                                       "probe keys"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeProbeKeyValidBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_probe_key_valid,
                         checked_bytes(context->probe_capacity, sizeof(std::uint8_t),
                                       "probe key validity"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeOutputPayloadsBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_output_payloads,
                         checked_bytes(context->probe_capacity, sizeof(long long),
                                       "output payloads"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeOutputPayloadValidBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_output_payload_valid,
                         checked_bytes(context->probe_capacity, sizeof(std::uint8_t),
                                       "output payload validity"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeOutputMatchedBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_output_matched,
                         checked_bytes(context->probe_capacity, sizeof(std::uint8_t),
                                       "output matches"));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeMetricsBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    Context* context = checked_context(handle);
    return direct_buffer(env, context->host_metrics, sizeof(Metrics));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeRebuild(
    JNIEnv* env, jclass, jlong handle, jint rows) {
  try {
    Context* context = checked_context(handle);
    if (rows < 0 || rows > context->dimension_capacity) {
      throw std::invalid_argument("Lookup Join dimension rows exceed capacity");
    }
    const int next_table = context->built ? 1 - context->active_table : context->active_table;
    const std::size_t long_bytes = checked_bytes(rows, sizeof(long long), "dimension");
    const std::size_t valid_bytes = checked_bytes(rows, sizeof(std::uint8_t), "dimension validity");
    const std::size_t slot_bytes =
        checked_bytes(context->table_size, sizeof(int), "hash slots");

    require_cuda("record Lookup Join build start",
                 cudaEventRecord(context->start, context->stream));
    if (rows > 0) {
      require_cuda("copy Lookup Join dimension keys",
                   cudaMemcpyAsync(context->device_dimension_keys[next_table],
                                   context->host_dimension_keys, long_bytes,
                                   cudaMemcpyHostToDevice, context->stream));
      require_cuda("copy Lookup Join dimension key validity",
                   cudaMemcpyAsync(context->device_dimension_key_valid[next_table],
                                   context->host_dimension_key_valid, valid_bytes,
                                   cudaMemcpyHostToDevice, context->stream));
      require_cuda("copy Lookup Join dimension payloads",
                   cudaMemcpyAsync(context->device_dimension_payloads[next_table],
                                   context->host_dimension_payloads, long_bytes,
                                   cudaMemcpyHostToDevice, context->stream));
      require_cuda("copy Lookup Join dimension payload validity",
                   cudaMemcpyAsync(context->device_dimension_payload_valid[next_table],
                                   context->host_dimension_payload_valid, valid_bytes,
                                   cudaMemcpyHostToDevice, context->stream));
    }
    require_cuda("record Lookup Join build H2D",
                 cudaEventRecord(context->after_h2d, context->stream));
    require_cuda("clear Lookup Join hash slots",
                 cudaMemsetAsync(context->device_slots[next_table], 0xff, slot_bytes,
                                 context->stream));
    require_cuda("clear Lookup Join build status",
                 cudaMemsetAsync(context->device_status, 0, sizeof(int), context->stream));
    if (rows > 0) {
      const int blocks = (rows + kThreads - 1) / kThreads;
      build_table_kernel<<<blocks, kThreads, 0, context->stream>>>(
          context->device_dimension_keys[next_table],
          context->device_dimension_key_valid[next_table], rows,
          context->device_slots[next_table], context->table_size,
          context->device_status);
      require_cuda("launch Lookup Join build kernel", cudaGetLastError());
    }
    require_cuda("record Lookup Join build compute",
                 cudaEventRecord(context->after_compute, context->stream));
    context->host_metrics->device_status = kStatusOk;
    require_cuda("copy Lookup Join build status",
                 cudaMemcpyAsync(&context->host_metrics->device_status,
                                 context->device_status, sizeof(int),
                                 cudaMemcpyDeviceToHost, context->stream));
    require_cuda("synchronize Lookup Join build", cudaStreamSynchronize(context->stream));

    context->host_metrics->build_h2d_nanos =
        elapsed_nanos(context->start, context->after_h2d);
    context->host_metrics->build_compute_nanos =
        elapsed_nanos(context->after_h2d, context->after_compute);
    const int status = context->host_metrics->device_status;
    if (status == kStatusDuplicateKey) {
      throw std::invalid_argument("Lookup Join dimension contains a duplicate non-NULL key");
    }
    if (status == kStatusTableFull) {
      throw std::runtime_error("Lookup Join hash table capacity was exhausted");
    }
    if (status != kStatusOk) {
      throw std::runtime_error("Lookup Join build reported device status " +
                               std::to_string(status));
    }

    context->active_table = next_table;
    context->built = true;
    context->host_metrics->generation += 1;
    context->host_metrics->dimension_rows = rows;
    context->host_metrics->probe_rows = 0;
    context->host_metrics->match_count = 0;
  } catch (const std::invalid_argument& failure) {
    throw_java(env, "java/lang/IllegalArgumentException", failure.what());
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeProbe(
    JNIEnv* env, jclass, jlong handle, jint rows, jboolean inject_device_error,
    jint filter_operation, jlong filter_literal, jint transfer_mask) {
  try {
    Context* context = checked_context(handle);
    if (!context->built) {
      throw std::logic_error("Lookup Join dimension has not been built");
    }
    if (rows < 0 || rows > context->probe_capacity) {
      throw std::invalid_argument("Lookup Join probe rows exceed capacity");
    }
    if (filter_operation < kFilterNone || filter_operation > kFilterMaximum) {
      throw std::invalid_argument("Lookup Join probe filter operation is invalid");
    }
    if (transfer_mask < 0 || transfer_mask > kTransferMaximum ||
        ((transfer_mask & kTransferPayload) != 0 &&
         (transfer_mask & kTransferFlags) == 0)) {
      throw std::invalid_argument("Lookup Join probe transfer mask is invalid");
    }
    const std::size_t long_bytes = checked_bytes(rows, sizeof(long long), "probe");
    const std::size_t valid_bytes = checked_bytes(rows, sizeof(std::uint8_t), "probe validity");

    require_cuda("record Lookup Join probe start",
                 cudaEventRecord(context->start, context->stream));
    if (rows > 0) {
      require_cuda("copy Lookup Join probe keys",
                   cudaMemcpyAsync(context->device_probe_keys, context->host_probe_keys,
                                   long_bytes, cudaMemcpyHostToDevice, context->stream));
      require_cuda("copy Lookup Join probe validity",
                   cudaMemcpyAsync(context->device_probe_key_valid,
                                   context->host_probe_key_valid, valid_bytes,
                                   cudaMemcpyHostToDevice, context->stream));
    }
    require_cuda("record Lookup Join probe H2D",
                 cudaEventRecord(context->after_h2d, context->stream));
    require_cuda("clear Lookup Join match count",
                 cudaMemsetAsync(context->device_match_count, 0, sizeof(int),
                                 context->stream));
    require_cuda("clear Lookup Join probe status",
                 cudaMemsetAsync(context->device_status, 0, sizeof(int), context->stream));
    if (rows > 0) {
      const int blocks = (rows + kThreads - 1) / kThreads;
      const int active = context->active_table;
      lookup_kernel<<<blocks, kThreads, 0, context->stream>>>(
          context->device_probe_keys, context->device_probe_key_valid, rows,
          context->device_dimension_keys[active],
          context->device_dimension_payloads[active],
          context->device_dimension_payload_valid[active],
          context->device_slots[active], context->table_size,
          context->device_output_payloads, context->device_output_payload_valid,
          context->device_output_matched, context->device_match_count,
          context->device_status, inject_device_error == JNI_TRUE,
          filter_operation, static_cast<long long>(filter_literal));
      require_cuda("launch Lookup Join probe kernel", cudaGetLastError());
    }
    require_cuda("record Lookup Join probe compute",
                 cudaEventRecord(context->after_compute, context->stream));

    context->host_metrics->match_count = 0;
    context->host_metrics->device_status = kStatusOk;
    require_cuda("copy Lookup Join match count",
                 cudaMemcpyAsync(&context->host_metrics->match_count,
                                 context->device_match_count, sizeof(int),
                                 cudaMemcpyDeviceToHost, context->stream));
    require_cuda("copy Lookup Join probe status",
                 cudaMemcpyAsync(&context->host_metrics->device_status,
                                 context->device_status, sizeof(int),
                                 cudaMemcpyDeviceToHost, context->stream));
    if (rows > 0 && (transfer_mask & kTransferPayload) != 0) {
      require_cuda("copy Lookup Join output payloads",
                   cudaMemcpyAsync(context->host_output_payloads,
                                   context->device_output_payloads, long_bytes,
                                   cudaMemcpyDeviceToHost, context->stream));
      require_cuda("copy Lookup Join output payload validity",
                   cudaMemcpyAsync(context->host_output_payload_valid,
                                   context->device_output_payload_valid, valid_bytes,
                                   cudaMemcpyDeviceToHost, context->stream));
    }
    if (rows > 0 && (transfer_mask & kTransferFlags) != 0) {
      require_cuda("copy Lookup Join output flags",
                   cudaMemcpyAsync(context->host_output_matched,
                                   context->device_output_matched, valid_bytes,
                                   cudaMemcpyDeviceToHost, context->stream));
    }
    require_cuda("record Lookup Join probe D2H",
                 cudaEventRecord(context->after_d2h, context->stream));
    require_cuda("synchronize Lookup Join probe", cudaEventSynchronize(context->after_d2h));

    context->host_metrics->probe_h2d_nanos =
        elapsed_nanos(context->start, context->after_h2d);
    context->host_metrics->probe_compute_nanos =
        elapsed_nanos(context->after_h2d, context->after_compute);
    context->host_metrics->probe_d2h_nanos =
        elapsed_nanos(context->after_compute, context->after_d2h);
    context->host_metrics->probe_rows = rows;
    const int status = context->host_metrics->device_status;

    if (status == kStatusInjectedDeviceError) {
      throw std::runtime_error("Lookup Join probe reported injected device error");
    }
    if (status != kStatusOk) {
      throw std::runtime_error("Lookup Join probe reported device status " +
                               std::to_string(status));
    }
  } catch (const std::invalid_argument& failure) {
    throw_java(env, "java/lang/IllegalArgumentException", failure.what());
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_LookupJoinKernel_nativeClose(
    JNIEnv* env, jclass, jlong handle) {
  try {
    if (handle == 0) return;
    destroy(reinterpret_cast<Context*>(handle));
  } catch (const std::exception& failure) {
    throw_failure(env, failure);
  } catch (...) {
    throw_java(
        env, "java/lang/IllegalStateException",
        "Lookup Join native close failed with an unknown error");
  }
}
