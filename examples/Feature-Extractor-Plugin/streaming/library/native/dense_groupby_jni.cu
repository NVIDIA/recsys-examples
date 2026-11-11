// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Derived unchanged (except JNI symbol prefix) from Mega-Flink native/dense_groupby_jni.cu.

#include <jni.h>

#include <cuda_runtime.h>

#include "gpu_arch_compat.h"
#include "gpu_device_batch.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kSharedGroupLimit = 128;
constexpr int kSharedMinimumRows = 1024;
constexpr std::size_t kOutputHeaderBytes = 64;
constexpr std::size_t kDeviceAllocationAlignment = 256;
constexpr std::uint64_t kPackedInputFormatId = 0x5041434b494e3031ULL;  // PACKIN01
constexpr std::uint64_t kPackedInputSchemaFingerprint =
    0xd796593a2f314553ULL;
constexpr std::uint32_t kPackedInputFieldCount = 13;
constexpr std::uint32_t kDeviceBatchPoolSlots = 2;
constexpr int kDeviceBatchMetricFields = 21;

enum DeviceBatchMetricField : int {
  kBatchProduced = 0,
  kBatchConsumed = 1,
  kBatchProducerH2DBytes = 2,
  kBatchProducerH2DNanos = 3,
  kBatchProducerKernelNanos = 4,
  kBatchControlD2HBytes = 5,
  kBatchControlD2HNanos = 6,
  kBatchIntermediatePayloadD2HBytes = 7,
  kBatchConsumerH2DBytes = 8,
  kBatchConsumerWaitNanos = 9,
  kBatchConsumerKernelNanos = 10,
  kBatchFinalD2HBytes = 11,
  kBatchFinalD2HNanos = 12,
  kBatchDrainCalls = 13,
  kBatchForcedLeakedReferences = 14,
  kBatchStaleTokenRejections = 15,
  kBatchAllocatedBytes = 16,
  kBatchActiveBytes = 17,
  kBatchPeakActiveBytes = 18,
  kBatchActiveReferences = 19,
  kBatchMetricsAbi = 20,
};

enum OperationMask : std::uint32_t {
  kRowCount = 1u << 0,
  kMetricCount = 1u << 1,
  kIntCount = 1u << 2,
  kLong0Count = 1u << 3,
  kLong1Count = 1u << 4,
  kFilter0Count = 1u << 5,
  kFilter1Count = 1u << 6,
  kFilter2Count = 1u << 7,
  kMetricSum = 1u << 8,
  kMinInt = 1u << 9,
  kMaxInt = 1u << 10,
  kLong0Sum = 1u << 11,
  kLong1Sum = 1u << 12,
  kMinLong0 = 1u << 13,
  kMaxLong0 = 1u << 14,
  kMinLong1 = 1u << 15,
  kMaxLong1 = 1u << 16,
  kIntSum = 1u << 17,
  kMinMetric = 1u << 18,
  kMaxMetric = 1u << 19,
};
constexpr std::uint32_t kAllOperations = (1u << 20) - 1u;

struct alignas(16) PackedInputRow {
  int group;
  int int_value;
  double metric;
  long long sequence;
  long long sent_nanos;
  std::uint32_t metric_valid;
  std::uint32_t int_valid;
  std::uint32_t sequence_valid;
  std::uint32_t sent_nanos_valid;
  std::uint32_t filter0;
  std::uint32_t filter1;
  std::uint32_t filter2;
  std::int32_t weight;
};

struct alignas(16) PackedAggregateRow {
  unsigned long long row_count;
  unsigned long long metric_count;
  unsigned long long int_count;
  unsigned long long long0_count;
  unsigned long long long1_count;
  unsigned long long filter0_count;
  unsigned long long filter1_count;
  unsigned long long filter2_count;
  double metric_sum;
  int min_int;
  int max_int;
  long long int_sum;
  long long long0_sum;
  long long long1_sum;
  long long min_long0;
  long long max_long0;
  long long min_long1;
  long long max_long1;
  double min_metric;
  double max_metric;
  unsigned long long metric_negative_zero_count;
};

// Numeric/BOOLEAN subset of GenericCalcKernel's stable 24-byte bytecode ABI. The opcode and type
// values intentionally match generic_calc_jni.cu so the same Java compiler can feed either
// standalone Calc or producer-side groupby fusion.
enum CalcType : int {
  kCalcBoolean = 0,
  kCalcInt8 = 1,
  kCalcInt16 = 2,
  kCalcInt32 = 3,
  kCalcInt64 = 4,
  kCalcFloat32 = 5,
  kCalcFloat64 = 6,
};

enum CalcOpcode : int {
  kCalcInput = 1,
  kCalcLiteral = 2,
  kCalcAdd = 10,
  kCalcSub = 11,
  kCalcMul = 12,
  kCalcDiv = 13,
  kCalcMod = 14,
  kCalcPow = 15,
  kCalcEqual = 20,
  kCalcNotEqual = 21,
  kCalcLess = 22,
  kCalcLessEqual = 23,
  kCalcGreater = 24,
  kCalcGreaterEqual = 25,
  kCalcAnd = 30,
  kCalcOr = 31,
  kCalcNot = 32,
  kCalcIsNull = 33,
  kCalcIsTrue = 34,
  kCalcIsFalse = 35,
  kCalcIsNotTrue = 36,
  kCalcIsNotFalse = 37,
  kCalcCast = 40,
  kCalcSelect = 41,
  kCalcCoalesce = 42,
  kCalcAbs = 50,
  kCalcSqrt = 51,
  kCalcFloor = 52,
  kCalcCeil = 53,
  kCalcSin = 54,
  kCalcCos = 55,
  kCalcTan = 56,
  kCalcExp = 57,
  kCalcStoreCondition = 100,
  kCalcStoreOutput = 101,
};

struct alignas(8) CalcInstruction {
  std::int32_t opcode;
  std::int32_t type;
  std::int32_t arg;
  std::int32_t reserved;
  std::uint64_t bits;
};

struct alignas(64) OutputHeader {
  std::uint64_t h2d_nanos;
  std::uint64_t compute_nanos;
  std::uint64_t d2h_nanos;
  std::uint64_t input_bytes;
  std::uint64_t output_bytes;
  std::uint32_t rows;
  std::uint32_t groups;
  std::uint32_t kernel_variant;
  std::uint8_t padding[12];
};

static_assert(sizeof(PackedInputRow) == 64, "Java/native input layout mismatch");
static_assert(sizeof(PackedAggregateRow) == 160, "Java/native output layout mismatch");
static_assert(sizeof(CalcInstruction) == 24,
              "Java/native fused Calc instruction layout mismatch");
static_assert(sizeof(OutputHeader) == kOutputHeaderBytes,
              "Java/native output header mismatch");

struct AggregateColumns {
  unsigned long long* row_count = nullptr;
  unsigned long long* metric_count = nullptr;
  unsigned long long* int_count = nullptr;
  unsigned long long* long0_count = nullptr;
  unsigned long long* long1_count = nullptr;
  unsigned long long* filter0_count = nullptr;
  unsigned long long* filter1_count = nullptr;
  unsigned long long* filter2_count = nullptr;
  unsigned long long* metric_negative_zero_count = nullptr;
  double* metric_sum = nullptr;
  double* min_metric = nullptr;
  double* max_metric = nullptr;
  int* min_int = nullptr;
  int* max_int = nullptr;
  long long* int_sum = nullptr;
  long long* long0_sum = nullptr;
  long long* long1_sum = nullptr;
  long long* min_long0 = nullptr;
  long long* max_long0 = nullptr;
  long long* min_long1 = nullptr;
  long long* max_long1 = nullptr;
};

constexpr std::uint32_t kPersistentAggregateSlotCount = 1024;
constexpr std::uint32_t kPersistentAggregateSlotMask =
    kPersistentAggregateSlotCount - 1;

struct alignas(64) PersistentAggregateControl {
  volatile std::uint32_t service_active;
  volatile std::uint32_t ready;
  volatile std::uint64_t produced;
  volatile std::uint64_t consumed;
  volatile std::uint32_t error;
  std::uint8_t padding[36];
};

struct alignas(64) PersistentAggregateCommand {
  PackedInputRow input;
  PackedAggregateRow seed;
  std::int32_t seed_valid;
  volatile std::uint32_t request;
  std::int32_t materialize_response;
  std::uint8_t padding[20];
};

struct alignas(64) PersistentAggregateResponse {
  PackedAggregateRow aggregate;
  volatile std::uint32_t ticket;
  std::int32_t error;
  std::uint8_t padding[88];
};

struct alignas(64) PersistentAggregateSlot {
  PersistentAggregateCommand command;
  PersistentAggregateResponse response;
};

struct alignas(64) PersistentAggregateStorage {
  PersistentAggregateControl control;
  PersistentAggregateSlot slots[kPersistentAggregateSlotCount];
};

static_assert(sizeof(PersistentAggregateControl) == 64,
              "persistent Aggregate control layout mismatch");
static_assert(sizeof(PersistentAggregateCommand) == 256,
              "persistent Aggregate command layout mismatch");
static_assert(sizeof(PersistentAggregateResponse) == 256,
              "persistent Aggregate response layout mismatch");
static_assert(sizeof(PersistentAggregateSlot) == 512,
              "persistent Aggregate slot layout mismatch");
static_assert(offsetof(PersistentAggregateStorage, slots) == 64,
              "persistent Aggregate slots must start at Java offset 64");
static_assert(offsetof(PersistentAggregateCommand, seed) == 64,
              "persistent Aggregate seed offset mismatch");
static_assert(offsetof(PersistentAggregateCommand, seed_valid) == 224,
              "persistent Aggregate seed flag offset mismatch");
static_assert(offsetof(PersistentAggregateCommand, request) == 228,
              "persistent Aggregate request offset mismatch");
static_assert(offsetof(PersistentAggregateCommand, materialize_response) == 232,
              "persistent Aggregate response mode offset mismatch");
static_assert(offsetof(PersistentAggregateSlot, response) == 256,
              "persistent Aggregate response offset mismatch");
static_assert(offsetof(PersistentAggregateResponse, ticket) == 160,
              "persistent Aggregate ticket offset mismatch");

struct PersistentAggregateContext {
  int device = -1;
  int capacity = 0;
  std::uint32_t operation_mask = 0;
  int metric_filter_lane = -1;
  int int_filter_lane = -1;
  int long0_filter_lane = -1;
  int long1_filter_lane = -1;
  PersistentAggregateStorage* host_storage = nullptr;
  PersistentAggregateStorage* device_storage = nullptr;
  unsigned char* device_aggregate_storage = nullptr;
  std::size_t device_aggregate_storage_bytes = 0;
  AggregateColumns aggregates{};
  int* device_snapshot_groups = nullptr;
  PackedAggregateRow* device_snapshot = nullptr;
  PackedAggregateRow* host_snapshot = nullptr;
  cudaStream_t stream = nullptr;
};

struct Context {
  int device = -1;
  int capacity = 0;
  int multiprocessors = 0;
  std::uint32_t operation_mask = 0;
  int metric_filter_lane = -1;
  int int_filter_lane = -1;
  int long0_filter_lane = -1;
  int long1_filter_lane = -1;
  PackedInputRow* host_input = nullptr;
  int* host_expanded_groups = nullptr;
  unsigned char* device_storage = nullptr;
  std::size_t device_storage_bytes = 0;
  PackedInputRow* device_input = nullptr;
  AggregateColumns aggregates{};
  PackedAggregateRow* device_output = nullptr;
  unsigned char* host_output = nullptr;
  unsigned char* host_calc_input = nullptr;
  unsigned char* device_calc_input = nullptr;
  CalcInstruction* device_calc_program = nullptr;
  int calc_input_count = 0;
  int calc_output_count = 0;
  int calc_instruction_count = 0;
  int calc_max_stack = 0;
  int calc_metric_output = -1;
  int calc_int_output = -1;
  int calc_long0_output = -1;
  int calc_long1_output = -1;
  int calc_filter_outputs[3] = {-1, -1, -1};
  bool calc_only_counts = false;
  std::size_t calc_input_row_bytes = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  cudaEvent_t compute_end = nullptr;
  cudaEvent_t d2h_end = nullptr;
  bool teardown_started = false;
  std::uint64_t device_batch_pool = 0;
  std::uint64_t device_batch_chain = 0;
  cudaStream_t device_batch_producer_stream = nullptr;
  cudaEvent_t device_batch_h2d_start = nullptr;
  cudaEvent_t device_batch_h2d_end = nullptr;
  cudaEvent_t device_batch_producer_kernel_end = nullptr;
  cudaEvent_t device_batch_control_d2h_end = nullptr;
  cudaEvent_t device_batch_consumer_wait_start = nullptr;
  cudaEvent_t device_batch_consumer_wait_end = nullptr;
  cudaEvent_t device_batch_consumer_kernel_end = nullptr;
  cudaEvent_t device_batch_final_d2h_end = nullptr;
  std::uint64_t device_batch_metrics[kDeviceBatchMetricFields]{};
  std::uint64_t device_batch_observed_drain_calls = 0;
  std::uint64_t device_batch_observed_forced_leaks = 0;
  std::uint64_t device_batch_observed_stale_rejections = 0;
};

constexpr int kFlexibleMaxOperations = 16;

enum FlexibleKind : int {
  kFlexibleCount = 0,
  kFlexibleSum = 1,
  kFlexibleAverage = 2,
  kFlexibleMin = 3,
  kFlexibleMax = 4,
};

enum FlexibleValueLane : int {
  kFlexibleRow = 0,
  kFlexibleMetric = 1,
  kFlexibleInt = 2,
  kFlexibleLong0 = 3,
  kFlexibleLong1 = 4,
};

struct alignas(16) FlexibleProgramOp {
  int kind;
  int value_lane;
  int filter_lane;
  int padding;
};

struct alignas(16) FlexibleResult {
  unsigned long long count;
  unsigned long long value_bits;
};

static_assert(sizeof(FlexibleProgramOp) == 16,
              "flexible aggregate program layout mismatch");
static_assert(sizeof(FlexibleResult) == 16,
              "flexible aggregate result layout mismatch");

struct FlexibleContext {
  int device = -1;
  int capacity = 0;
  int operation_count = 0;
  int result_stride = 0;
  int multiprocessors = 0;
  PackedInputRow* host_input = nullptr;
  unsigned char* host_output = nullptr;
  unsigned char* device_storage = nullptr;
  std::size_t device_storage_bytes = 0;
  PackedInputRow* device_input = nullptr;
  FlexibleProgramOp* device_program = nullptr;
  FlexibleResult* device_results = nullptr;
  unsigned long long* device_negative_zero_counts = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t h2d_start = nullptr;
  cudaEvent_t h2d_end = nullptr;
  cudaEvent_t compute_end = nullptr;
  cudaEvent_t d2h_end = nullptr;
};

struct PrimitiveLongAggregateContext;
void destroy_primitive_long_aggregate(PrimitiveLongAggregateContext* context);

enum class ConstructionOrphanKind {
  kNone,
  kDense,
  kFlexible,
  kPrimitiveLong,
};

struct ConstructionOrphanOwner {
  ConstructionOrphanKind kind = ConstructionOrphanKind::kNone;
  void* context = nullptr;
};

// Native construction is serialized so one fixed-capacity slot is sufficient. A later create
// must retry this owner before allocating anything new; repeated CUDA failures therefore cannot
// create an unbounded list of unreachable contexts. Without a handle-free JNI retry entry point,
// a later create is also the only Java-reachable trigger after nativeCreate fails to return a
// handle.
std::mutex construction_orphan_mutex;
ConstructionOrphanOwner construction_orphan;

std::string cuda_error(const char* operation, cudaError_t error) {
  return std::string(operation) + " failed: " + cudaGetErrorName(error) + " (" +
         cudaGetErrorString(error) + ")";
}

void require_cuda(const char* operation, cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cuda_error(operation, error));
  }
}

void require_device_batch(const char* operation, std::int32_t status) {
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    throw std::runtime_error(
        std::string(operation) + " failed: " +
        cudf_flink_gpu_device_batch_status_string(status) + " (" +
        std::to_string(status) + ")");
  }
}

void synchronize_stream_for_teardown(
    const char* operation, cudaStream_t stream) {
  if (stream != nullptr) {
    require_cuda(operation, cudaStreamSynchronize(stream));
  }
}

void destroy_event_for_teardown(
    const char* operation, cudaEvent_t* event) {
  if (event == nullptr || *event == nullptr) {
    return;
  }
  require_cuda(operation, cudaEventDestroy(*event));
  *event = nullptr;
}

void destroy_stream_for_teardown(
    const char* operation, cudaStream_t* stream) {
  if (stream == nullptr || *stream == nullptr) {
    return;
  }
  require_cuda(operation, cudaStreamDestroy(*stream));
  *stream = nullptr;
}

template <typename T>
void free_device_for_teardown(const char* operation, T** pointer) {
  if (pointer == nullptr || *pointer == nullptr) {
    return;
  }
  require_cuda(operation, cudaFree(static_cast<void*>(*pointer)));
  *pointer = nullptr;
}

template <typename T>
void free_host_for_teardown(const char* operation, T** pointer) {
  if (pointer == nullptr || *pointer == nullptr) {
    return;
  }
  require_cuda(operation, cudaFreeHost(static_cast<void*>(*pointer)));
  *pointer = nullptr;
}

void close_device_batch_pool_for_teardown(
    const char* operation, std::uint64_t* pool) {
  if (pool == nullptr || *pool == 0) {
    return;
  }
  require_device_batch(
      operation, cudf_flink_gpu_device_batch_pool_close(*pool));
  *pool = 0;
}

std::string exception_message(const std::exception_ptr& failure) {
  if (failure == nullptr) {
    return "missing native failure";
  }
  try {
    std::rethrow_exception(failure);
  } catch (const std::exception& error) {
    return error.what();
  } catch (...) {
    return "unknown native failure";
  }
}

[[noreturn]] void retain_construction_orphan(
    ConstructionOrphanKind kind,
    void* context,
    const std::exception_ptr& construction_failure,
    const std::exception_ptr& teardown_failure) {
  if (kind == ConstructionOrphanKind::kNone || context == nullptr ||
      construction_orphan.kind != ConstructionOrphanKind::kNone ||
      construction_orphan.context != nullptr) {
    throw std::logic_error(
        "dense groupby construction orphan ownership invariant failed");
  }
  // Store the only remaining native owner before building an error string, which may itself
  // allocate and fail. The create mutex prevents another constructor from replacing this slot.
  construction_orphan.context = context;
  construction_orphan.kind = kind;
  throw std::runtime_error(
      std::string("dense groupby construction failed: ") +
      exception_message(construction_failure) +
      "; teardown failed; native context retained for retry: " +
      exception_message(teardown_failure));
}

std::uint64_t event_nanos(
    const char* operation, cudaEvent_t start, cudaEvent_t end) {
  float elapsed_ms = 0.0f;
  require_cuda(operation, cudaEventElapsedTime(&elapsed_ms, start, end));
  return static_cast<std::uint64_t>(
      std::llround(elapsed_ms * 1'000'000.0));
}

CudfFlinkGpuDeviceBatchDescriptor packed_input_descriptor(
    const Context* context, int rows) {
  CudfFlinkGpuDeviceBatchDescriptor descriptor{};
  descriptor.struct_size = sizeof(descriptor);
  descriptor.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  descriptor.format_id = kPackedInputFormatId;
  descriptor.schema_fingerprint = kPackedInputSchemaFingerprint;
  descriptor.owner_chain_id = context->device_batch_chain;
  descriptor.device_ordinal = context->device;
  descriptor.rows = static_cast<std::uint32_t>(rows);
  descriptor.stride_bytes = sizeof(PackedInputRow);
  descriptor.field_count = kPackedInputFieldCount;
  descriptor.data_bytes =
      static_cast<std::uint64_t>(rows) * sizeof(PackedInputRow);
  const std::uint32_t types[kPackedInputFieldCount] = {
      CUDF_FLINK_GPU_DEVICE_BATCH_INT32,
      CUDF_FLINK_GPU_DEVICE_BATCH_INT32,
      CUDF_FLINK_GPU_DEVICE_BATCH_FLOAT64,
      CUDF_FLINK_GPU_DEVICE_BATCH_INT64,
      CUDF_FLINK_GPU_DEVICE_BATCH_INT64,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_BOOLEAN,
      CUDF_FLINK_GPU_DEVICE_BATCH_INT32};
  std::copy(
      std::begin(types), std::end(types), std::begin(descriptor.type_ids));
  return descriptor;
}

void throw_java(JNIEnv* env, const std::string& message) {
  jclass exception_class = env->FindClass("java/lang/RuntimeException");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message.c_str());
  }
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

class DeviceSlabLayout {
 public:
  explicit DeviceSlabLayout(unsigned char* base) : base_(base) {}

  template <typename T>
  T* take(std::size_t count) {
    offset_ = align_up(offset_, kDeviceAllocationAlignment);
    T* pointer =
        base_ == nullptr ? nullptr : reinterpret_cast<T*>(base_ + offset_);
    offset_ += count * sizeof(T);
    return pointer;
  }

  std::size_t bytes() const {
    return align_up(offset_, kDeviceAllocationAlignment);
  }

 private:
  unsigned char* base_ = nullptr;
  std::size_t offset_ = 0;
};

std::size_t bind_device_storage(Context* context, unsigned char* storage) {
  DeviceSlabLayout layout(storage);
  const std::size_t capacity = static_cast<std::size_t>(context->capacity);
  context->device_input = layout.take<PackedInputRow>(capacity);
  context->aggregates.row_count = layout.take<unsigned long long>(capacity);
  context->aggregates.metric_count = layout.take<unsigned long long>(capacity);
  context->aggregates.int_count = layout.take<unsigned long long>(capacity);
  context->aggregates.long0_count = layout.take<unsigned long long>(capacity);
  context->aggregates.long1_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter0_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter1_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter2_count = layout.take<unsigned long long>(capacity);
  context->aggregates.metric_negative_zero_count =
      layout.take<unsigned long long>(capacity);
  context->aggregates.metric_sum = layout.take<double>(capacity);
  context->aggregates.min_metric = layout.take<double>(capacity);
  context->aggregates.max_metric = layout.take<double>(capacity);
  context->aggregates.min_int = layout.take<int>(capacity);
  context->aggregates.max_int = layout.take<int>(capacity);
  context->aggregates.int_sum = layout.take<long long>(capacity);
  context->aggregates.long0_sum = layout.take<long long>(capacity);
  context->aggregates.long1_sum = layout.take<long long>(capacity);
  context->aggregates.min_long0 = layout.take<long long>(capacity);
  context->aggregates.max_long0 = layout.take<long long>(capacity);
  context->aggregates.min_long1 = layout.take<long long>(capacity);
  context->aggregates.max_long1 = layout.take<long long>(capacity);
  context->device_output = layout.take<PackedAggregateRow>(capacity + 1);
  return layout.bytes();
}

std::size_t bind_persistent_aggregate_storage(
    PersistentAggregateContext* context, unsigned char* storage) {
  DeviceSlabLayout layout(storage);
  const std::size_t capacity = static_cast<std::size_t>(context->capacity);
  context->aggregates.row_count = layout.take<unsigned long long>(capacity);
  context->aggregates.metric_count = layout.take<unsigned long long>(capacity);
  context->aggregates.int_count = layout.take<unsigned long long>(capacity);
  context->aggregates.long0_count = layout.take<unsigned long long>(capacity);
  context->aggregates.long1_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter0_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter1_count = layout.take<unsigned long long>(capacity);
  context->aggregates.filter2_count = layout.take<unsigned long long>(capacity);
  context->aggregates.metric_negative_zero_count =
      layout.take<unsigned long long>(capacity);
  context->aggregates.metric_sum = layout.take<double>(capacity);
  context->aggregates.min_metric = layout.take<double>(capacity);
  context->aggregates.max_metric = layout.take<double>(capacity);
  context->aggregates.min_int = layout.take<int>(capacity);
  context->aggregates.max_int = layout.take<int>(capacity);
  context->aggregates.int_sum = layout.take<long long>(capacity);
  context->aggregates.long0_sum = layout.take<long long>(capacity);
  context->aggregates.long1_sum = layout.take<long long>(capacity);
  context->aggregates.min_long0 = layout.take<long long>(capacity);
  context->aggregates.max_long0 = layout.take<long long>(capacity);
  context->aggregates.min_long1 = layout.take<long long>(capacity);
  context->aggregates.max_long1 = layout.take<long long>(capacity);
  return layout.bytes();
}

__device__ void atomic_max_i64(long long* address, long long value) {
  auto* raw = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *raw;
  while (static_cast<long long>(observed) < value) {
    const unsigned long long previous =
        atomicCAS(raw, observed, static_cast<unsigned long long>(value));
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__device__ void atomic_min_i64(long long* address, long long value) {
  auto* raw = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *raw;
  while (static_cast<long long>(observed) > value) {
    const unsigned long long previous =
        atomicCAS(raw, observed, static_cast<unsigned long long>(value));
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__device__ __forceinline__ unsigned long long weighted_i64_bits(
    long long value, int weight) {
  // Flink integral SUM uses Java's two's-complement wraparound. Do the multiply in
  // unsigned space so retracting LLONG_MIN and overflowing an accumulator are defined by
  // C++, rather than relying on nvcc's treatment of signed overflow.
  return static_cast<unsigned long long>(value) *
         static_cast<unsigned long long>(static_cast<long long>(weight));
}

__device__ void atomic_add_i64(long long* address,
                               unsigned long long value_bits) {
  atomicAdd(reinterpret_cast<unsigned long long*>(address),
            value_bits);
}

__device__ __forceinline__ void wrapping_add_i64(
    long long* address, unsigned long long value_bits) {
  auto* raw = reinterpret_cast<unsigned long long*>(address);
  *raw += value_bits;
}

__device__ void atomic_min_f64(double* address, double value) {
  auto* raw = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *raw;
  while (__longlong_as_double(static_cast<long long>(observed)) > value) {
    const unsigned long long previous =
        atomicCAS(raw, observed, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__device__ void atomic_max_f64(double* address, double value) {
  auto* raw = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *raw;
  while (__longlong_as_double(static_cast<long long>(observed)) < value) {
    const unsigned long long previous =
        atomicCAS(raw, observed, static_cast<unsigned long long>(__double_as_longlong(value)));
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__device__ __forceinline__ bool is_negative_zero(double value) {
  return __double_as_longlong(value) == LLONG_MIN;
}

__device__ __forceinline__ float calc_bits_float(std::uint64_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits));
}

__device__ __forceinline__ std::uint64_t calc_float_bits(float value) {
  return static_cast<std::uint64_t>(__float_as_uint(value));
}

__device__ __forceinline__ double calc_bits_double(std::uint64_t bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ std::uint64_t calc_double_bits(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}

__device__ __forceinline__ std::int64_t calc_signed_value(
    int type, std::uint64_t bits) {
  switch (type) {
    case kCalcInt8: return static_cast<std::int8_t>(bits);
    case kCalcInt16: return static_cast<std::int16_t>(bits);
    case kCalcInt32: return static_cast<std::int32_t>(bits);
    default: return static_cast<std::int64_t>(bits);
  }
}

__device__ __forceinline__ std::uint64_t calc_narrow_integer(
    int type, std::uint64_t value) {
  switch (type) {
    case kCalcInt8:
      return static_cast<std::uint64_t>(
          static_cast<std::int8_t>(value));
    case kCalcInt16:
      return static_cast<std::uint64_t>(
          static_cast<std::int16_t>(value));
    case kCalcInt32:
      return static_cast<std::uint64_t>(
          static_cast<std::int32_t>(value));
    default:
      return value;
  }
}

__device__ __forceinline__ bool calc_compare_bits(
    int opcode, int type, std::uint64_t left, std::uint64_t right) {
  int compared = 0;
  if (type == kCalcFloat64) {
    const double a = calc_bits_double(left);
    const double b = calc_bits_double(right);
    if (a < b) compared = -1;
    else if (a > b) compared = 1;
    else {
      const std::uint64_t a_bits =
          isnan(a) ? 0x7ff8000000000000ULL : left;
      const std::uint64_t b_bits =
          isnan(b) ? 0x7ff8000000000000ULL : right;
      const auto a_signed = static_cast<std::int64_t>(a_bits);
      const auto b_signed = static_cast<std::int64_t>(b_bits);
      compared = a_signed == b_signed ? 0 : (a_signed < b_signed ? -1 : 1);
    }
  } else if (type == kCalcFloat32) {
    const float a = calc_bits_float(left);
    const float b = calc_bits_float(right);
    if (a < b) compared = -1;
    else if (a > b) compared = 1;
    else {
      const std::uint32_t a_bits =
          isnan(a) ? 0x7fc00000U : static_cast<std::uint32_t>(left);
      const std::uint32_t b_bits =
          isnan(b) ? 0x7fc00000U : static_cast<std::uint32_t>(right);
      const auto a_signed = static_cast<std::int32_t>(a_bits);
      const auto b_signed = static_cast<std::int32_t>(b_bits);
      compared = a_signed == b_signed ? 0 : (a_signed < b_signed ? -1 : 1);
    }
  } else {
    const std::int64_t a = calc_signed_value(type, left);
    const std::int64_t b = calc_signed_value(type, right);
    compared = a == b ? 0 : (a < b ? -1 : 1);
  }
  switch (opcode) {
    case kCalcEqual: return compared == 0;
    case kCalcNotEqual: return compared != 0;
    case kCalcLess: return compared < 0;
    case kCalcLessEqual: return compared <= 0;
    case kCalcGreater: return compared > 0;
    default: return compared >= 0;
  }
}

__device__ __forceinline__ bool calc_integer_minimum(
    int type, std::int64_t value) {
  switch (type) {
    case kCalcInt8: return value == INT8_MIN;
    case kCalcInt16: return value == INT16_MIN;
    case kCalcInt32: return value == INT32_MIN;
    default: return value == INT64_MIN;
  }
}

__device__ __forceinline__ std::uint64_t calc_numeric_binary(
    int opcode,
    int type,
    std::uint64_t left,
    std::uint64_t right,
    bool* valid) {
  if (type == kCalcFloat64) {
    const double a = calc_bits_double(left);
    const double b = calc_bits_double(right);
    switch (opcode) {
      case kCalcAdd: return calc_double_bits(a + b);
      case kCalcSub: return calc_double_bits(a - b);
      case kCalcMul: return calc_double_bits(a * b);
      case kCalcDiv: return calc_double_bits(a / b);
      case kCalcMod: return calc_double_bits(fmod(a, b));
      default: return calc_double_bits(pow(a, b));
    }
  }
  if (type == kCalcFloat32) {
    const float a = calc_bits_float(left);
    const float b = calc_bits_float(right);
    switch (opcode) {
      case kCalcAdd: return calc_float_bits(a + b);
      case kCalcSub: return calc_float_bits(a - b);
      case kCalcMul: return calc_float_bits(a * b);
      case kCalcDiv: return calc_float_bits(a / b);
      case kCalcMod: return calc_float_bits(fmodf(a, b));
      default: return calc_float_bits(powf(a, b));
    }
  }
  const std::int64_t a = calc_signed_value(type, left);
  const std::int64_t b = calc_signed_value(type, right);
  std::uint64_t result = 0;
  switch (opcode) {
    case kCalcAdd:
      result = static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b);
      break;
    case kCalcSub:
      result = static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b);
      break;
    case kCalcMul:
      result = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
      break;
    case kCalcDiv:
      if (b == 0) {
        *valid = false;
        return 0;
      }
      result = b == -1 && calc_integer_minimum(type, a)
                   ? static_cast<std::uint64_t>(a)
                   : static_cast<std::uint64_t>(a / b);
      break;
    case kCalcMod:
      if (b == 0) {
        *valid = false;
        return 0;
      }
      result = b == -1 && calc_integer_minimum(type, a)
                   ? 0
                   : static_cast<std::uint64_t>(a % b);
      break;
    default:
      result = static_cast<std::uint64_t>(static_cast<std::int64_t>(
          pow(static_cast<double>(a), static_cast<double>(b))));
      break;
  }
  return calc_narrow_integer(type, result);
}

__device__ __forceinline__ std::uint64_t calc_cast_bits(
    int source, int target, std::uint64_t bits) {
  double value;
  if (source == kCalcFloat64) value = calc_bits_double(bits);
  else if (source == kCalcFloat32) value = calc_bits_float(bits);
  else value = static_cast<double>(calc_signed_value(source, bits));
  if (target == kCalcFloat64) return calc_double_bits(value);
  if (target == kCalcFloat32) return calc_float_bits(static_cast<float>(value));
  return calc_narrow_integer(
      target, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
}

__device__ __forceinline__ std::uint64_t calc_unary_math(
    int opcode, int type, std::uint64_t bits) {
  if (type == kCalcFloat32) {
    const float value = calc_bits_float(bits);
    switch (opcode) {
      case kCalcAbs: return calc_float_bits(fabsf(value));
      case kCalcSqrt: return calc_float_bits(sqrtf(value));
      case kCalcFloor: return calc_float_bits(floorf(value));
      case kCalcCeil: return calc_float_bits(ceilf(value));
      case kCalcSin: return calc_float_bits(sinf(value));
      case kCalcCos: return calc_float_bits(cosf(value));
      case kCalcTan: return calc_float_bits(tanf(value));
      default: return calc_float_bits(expf(value));
    }
  }
  if (type == kCalcFloat64) {
    const double value = calc_bits_double(bits);
    switch (opcode) {
      case kCalcAbs: return calc_double_bits(fabs(value));
      case kCalcSqrt: return calc_double_bits(sqrt(value));
      case kCalcFloor: return calc_double_bits(floor(value));
      case kCalcCeil: return calc_double_bits(ceil(value));
      case kCalcSin: return calc_double_bits(sin(value));
      case kCalcCos: return calc_double_bits(cos(value));
      case kCalcTan: return calc_double_bits(tan(value));
      default: return calc_double_bits(exp(value));
    }
  }
  const std::int64_t value = calc_signed_value(type, bits);
  const std::uint64_t absolute =
      opcode == kCalcAbs && value < 0
          ? 0ULL - static_cast<std::uint64_t>(value)
          : static_cast<std::uint64_t>(value);
  return calc_narrow_integer(type, absolute);
}

__device__ __forceinline__ bool flexible_filter_passes(
    const PackedInputRow& input, int filter_lane) {
  switch (filter_lane) {
    case -1:
      return true;
    case 0:
      return input.filter0 != 0;
    case 1:
      return input.filter1 != 0;
    case 2:
      return input.filter2 != 0;
    default:
      return false;
  }
}

__device__ __forceinline__ void update_flexible_result(
    FlexibleResult* result,
    unsigned long long* negative_zero_count,
    const FlexibleProgramOp& operation,
    const PackedInputRow& input) {
  if (!flexible_filter_passes(input, operation.filter_lane)) {
    return;
  }

  bool valid = true;
  double metric_value = 0.0;
  long long integer_value = 0;
  switch (operation.value_lane) {
    case kFlexibleRow:
      break;
    case kFlexibleMetric:
      valid = input.metric_valid != 0;
      metric_value = input.metric;
      break;
    case kFlexibleInt:
      valid = input.int_valid != 0;
      integer_value = static_cast<long long>(input.int_value);
      break;
    case kFlexibleLong0:
      valid = input.sequence_valid != 0;
      integer_value = input.sequence;
      break;
    case kFlexibleLong1:
      valid = input.sent_nanos_valid != 0;
      integer_value = input.sent_nanos;
      break;
    default:
      valid = false;
      break;
  }
  if (!valid) {
    return;
  }

  const auto weight = static_cast<unsigned long long>(
      static_cast<long long>(input.weight));
  atomicAdd(&result->count, weight);
  if (operation.kind == kFlexibleCount) {
    return;
  }
  if (operation.value_lane == kFlexibleMetric) {
    double* value = reinterpret_cast<double*>(&result->value_bits);
    if (operation.kind == kFlexibleSum ||
        operation.kind == kFlexibleAverage) {
      if (is_negative_zero(metric_value)) {
        atomicAdd(negative_zero_count, weight);
      }
      atomicAdd(value, metric_value * static_cast<double>(input.weight));
    } else if (operation.kind == kFlexibleMin) {
      atomic_min_f64(value, metric_value);
    } else if (operation.kind == kFlexibleMax) {
      atomic_max_f64(value, metric_value);
    }
    return;
  }

  auto* value = reinterpret_cast<long long*>(&result->value_bits);
  if (operation.kind == kFlexibleSum ||
      operation.kind == kFlexibleAverage) {
    atomic_add_i64(value, weighted_i64_bits(integer_value, input.weight));
  } else if (operation.kind == kFlexibleMin) {
    atomic_min_i64(value, integer_value);
  } else if (operation.kind == kFlexibleMax) {
    atomic_max_i64(value, integer_value);
  }
}

__device__ __forceinline__ unsigned long long flexible_initial_value(
    const FlexibleProgramOp& operation) {
  if (operation.kind == kFlexibleMin) {
    return operation.value_lane == kFlexibleMetric
               ? static_cast<unsigned long long>(__double_as_longlong(DBL_MAX))
               : static_cast<unsigned long long>(LLONG_MAX);
  }
  if (operation.kind == kFlexibleMax) {
    return operation.value_lane == kFlexibleMetric
               ? static_cast<unsigned long long>(__double_as_longlong(-DBL_MAX))
               : static_cast<unsigned long long>(LLONG_MIN);
  }
  return 0ULL;
}

__global__ void initialize_flexible_results(
    FlexibleResult* results,
    unsigned long long* negative_zero_counts,
    const FlexibleProgramOp* program,
    int groups,
    int operation_count) {
  const int stride = operation_count + 1;
  const int total = groups * stride;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
       index += blockDim.x * gridDim.x) {
    const int slot = index % stride;
    results[index].count = 0;
    results[index].value_bits =
        slot == 0 ? 0ULL : flexible_initial_value(program[slot - 1]);
    negative_zero_counts[index] = 0;
  }
}

__global__ void flexible_global_groupby(
    const PackedInputRow* input,
    int rows,
    const FlexibleProgramOp* program,
    int operation_count,
    FlexibleResult* output,
    unsigned long long* negative_zero_counts) {
  const int stride = operation_count + 1;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    const PackedInputRow value = input[row];
    FlexibleResult* group_output = output + value.group * stride;
    atomicAdd(&group_output[0].count,
              static_cast<unsigned long long>(
                  static_cast<long long>(value.weight)));
    for (int operation = 0; operation < operation_count; operation++) {
      update_flexible_result(
          &group_output[operation + 1],
          &negative_zero_counts[value.group * stride + operation + 1],
          program[operation],
          value);
    }
  }
}

__device__ __forceinline__ void flush_flexible_result(
    const FlexibleResult& local,
    unsigned long long local_negative_zero_count,
    FlexibleResult* output,
    unsigned long long* output_negative_zero_count,
    const FlexibleProgramOp& operation) {
  if (local.count == 0) {
    return;
  }
  atomicAdd(&output->count, local.count);
  if (operation.kind == kFlexibleCount) {
    return;
  }
  if (operation.value_lane == kFlexibleMetric) {
    const double local_value =
        __longlong_as_double(static_cast<long long>(local.value_bits));
    double* output_value = reinterpret_cast<double*>(&output->value_bits);
    if (operation.kind == kFlexibleSum ||
        operation.kind == kFlexibleAverage) {
      if (local_negative_zero_count != 0) {
        atomicAdd(output_negative_zero_count, local_negative_zero_count);
      }
      atomicAdd(output_value, local_value);
    } else if (operation.kind == kFlexibleMin) {
      atomic_min_f64(output_value, local_value);
    } else if (operation.kind == kFlexibleMax) {
      atomic_max_f64(output_value, local_value);
    }
    return;
  }
  const long long local_value = static_cast<long long>(local.value_bits);
  auto* output_value = reinterpret_cast<long long*>(&output->value_bits);
  if (operation.kind == kFlexibleSum ||
      operation.kind == kFlexibleAverage) {
    atomic_add_i64(output_value, local_value);
  } else if (operation.kind == kFlexibleMin) {
    atomic_min_i64(output_value, local_value);
  } else if (operation.kind == kFlexibleMax) {
    atomic_max_i64(output_value, local_value);
  }
}

__global__ void flexible_shared_groupby(
    const PackedInputRow* input,
    int rows,
    int groups,
    const FlexibleProgramOp* program,
    int operation_count,
    FlexibleResult* output,
    unsigned long long* negative_zero_counts) {
  extern __shared__ unsigned char shared_storage[];
  const int stride = operation_count + 1;
  const int total = groups * stride;
  auto* local = reinterpret_cast<FlexibleResult*>(shared_storage);
  auto* local_negative_zero_counts = reinterpret_cast<unsigned long long*>(
      local + total);
  for (int index = threadIdx.x; index < total; index += blockDim.x) {
    const int slot = index % stride;
    local[index].count = 0;
    local[index].value_bits =
        slot == 0 ? 0ULL : flexible_initial_value(program[slot - 1]);
    local_negative_zero_counts[index] = 0;
  }
  __syncthreads();

  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    const PackedInputRow value = input[row];
    FlexibleResult* group_output = local + value.group * stride;
    atomicAdd(&group_output[0].count,
              static_cast<unsigned long long>(
                  static_cast<long long>(value.weight)));
    for (int operation = 0; operation < operation_count; operation++) {
      update_flexible_result(
          &group_output[operation + 1],
          &local_negative_zero_counts[value.group * stride + operation + 1],
          program[operation],
          value);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < total; index += blockDim.x) {
    const int slot = index % stride;
    if (slot == 0) {
      if (local[index].count != 0) {
        atomicAdd(&output[index].count, local[index].count);
      }
    } else {
      flush_flexible_result(local[index],
                            local_negative_zero_counts[index],
                            &output[index],
                            &negative_zero_counts[index],
                            program[slot - 1]);
    }
  }
}

__global__ void finalize_flexible_floating_zero(
    FlexibleResult* results,
    const unsigned long long* negative_zero_counts,
    const FlexibleProgramOp* program,
    int result_count,
    int stride) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < result_count;
       index += blockDim.x * gridDim.x) {
    const int slot = index % stride;
    if (slot == 0) continue;
    const FlexibleProgramOp operation = program[slot - 1];
    if (operation.value_lane == kFlexibleMetric &&
        (operation.kind == kFlexibleSum ||
         operation.kind == kFlexibleAverage) &&
        results[index].count != 0 &&
        results[index].count == negative_zero_counts[index]) {
      results[index].value_bits = 0x8000000000000000ULL;
    }
  }
}

__device__ __forceinline__ void update_group(
    AggregateColumns output,
    const PackedInputRow& input,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
  int long0_filter_lane,
  int long1_filter_lane) {
  const int group = input.group;
  const auto weight = static_cast<unsigned long long>(
      static_cast<long long>(input.weight));
  atomicAdd(output.row_count + group, weight);
  if ((operation_mask & kFilter0Count) != 0 && input.filter0 != 0) {
    atomicAdd(output.filter0_count + group, weight);
  }
  if ((operation_mask & kFilter1Count) != 0 && input.filter1 != 0) {
    atomicAdd(output.filter1_count + group, weight);
  }
  if ((operation_mask & kFilter2Count) != 0 && input.filter2 != 0) {
    atomicAdd(output.filter2_count + group, weight);
  }
  const std::uint32_t filters[3] = {input.filter0, input.filter1, input.filter2};
  const bool metric_passes =
      metric_filter_lane < 0 || filters[metric_filter_lane] != 0;
  const bool int_passes = int_filter_lane < 0 || filters[int_filter_lane] != 0;
  const bool long0_passes =
      long0_filter_lane < 0 || filters[long0_filter_lane] != 0;
  const bool long1_passes =
      long1_filter_lane < 0 || filters[long1_filter_lane] != 0;
  if (input.metric_valid != 0 && metric_passes) {
    if ((operation_mask & kMetricCount) != 0) {
      atomicAdd(output.metric_count + group, weight);
    }
    if ((operation_mask & kMetricSum) != 0) {
      if (is_negative_zero(input.metric)) {
        atomicAdd(output.metric_negative_zero_count + group, weight);
      }
      atomicAdd(output.metric_sum + group,
                input.metric * static_cast<double>(input.weight));
    }
    if ((operation_mask & kMinMetric) != 0) {
      atomic_min_f64(output.min_metric + group, input.metric);
    }
    if ((operation_mask & kMaxMetric) != 0) {
      atomic_max_f64(output.max_metric + group, input.metric);
    }
  }
  if (input.int_valid != 0 && int_passes) {
    if ((operation_mask & kIntCount) != 0) {
      atomicAdd(output.int_count + group, weight);
    }
    if ((operation_mask & kIntSum) != 0) {
      atomic_add_i64(
          output.int_sum + group,
          weighted_i64_bits(static_cast<long long>(input.int_value), input.weight));
    }
    if ((operation_mask & kMinInt) != 0) {
      atomicMin(output.min_int + group, input.int_value);
    }
    if ((operation_mask & kMaxInt) != 0) {
      atomicMax(output.max_int + group, input.int_value);
    }
  }
  if (input.sequence_valid != 0 && long0_passes) {
    if ((operation_mask & kLong0Count) != 0) {
      atomicAdd(output.long0_count + group, weight);
    }
    if ((operation_mask & kLong0Sum) != 0) {
      atomic_add_i64(output.long0_sum + group,
                     weighted_i64_bits(input.sequence, input.weight));
    }
    if ((operation_mask & kMinLong0) != 0) {
      atomic_min_i64(output.min_long0 + group, input.sequence);
    }
    if ((operation_mask & kMaxLong0) != 0) {
      atomic_max_i64(output.max_long0 + group, input.sequence);
    }
  }
  if (input.sent_nanos_valid != 0 && long1_passes) {
    if ((operation_mask & kLong1Count) != 0) {
      atomicAdd(output.long1_count + group, weight);
    }
    if ((operation_mask & kLong1Sum) != 0) {
      atomic_add_i64(output.long1_sum + group,
                     weighted_i64_bits(input.sent_nanos, input.weight));
    }
    if ((operation_mask & kMinLong1) != 0) {
      atomic_min_i64(output.min_long1 + group, input.sent_nanos);
    }
    if ((operation_mask & kMaxLong1) != 0) {
      atomic_max_i64(output.max_long1 + group, input.sent_nanos);
    }
  }
}

__device__ __forceinline__ std::uint32_t persistent_aggregate_load_acquire_u32(
    volatile std::uint32_t* address) {
  std::uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ std::uint64_t persistent_aggregate_load_acquire_u64(
    volatile std::uint64_t* address) {
  std::uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void persistent_aggregate_store_release_u32(
    volatile std::uint32_t* address, std::uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               :
               : "l"(address), "r"(value)
               : "memory");
}

__device__ __forceinline__ void persistent_aggregate_store_release_u64(
    volatile std::uint64_t* address, std::uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(address), "l"(value)
               : "memory");
}

__device__ __forceinline__ void persistent_seed_field(
    AggregateColumns output,
    int group,
    int lane,
    const PackedAggregateRow& seed) {
  switch (lane) {
    case 0: output.row_count[group] = seed.row_count; break;
    case 1: output.metric_count[group] = seed.metric_count; break;
    case 2: output.int_count[group] = seed.int_count; break;
    case 3: output.long0_count[group] = seed.long0_count; break;
    case 4: output.long1_count[group] = seed.long1_count; break;
    case 5: output.filter0_count[group] = seed.filter0_count; break;
    case 6: output.filter1_count[group] = seed.filter1_count; break;
    case 7: output.filter2_count[group] = seed.filter2_count; break;
    case 8:
      output.metric_sum[group] = seed.metric_sum;
      output.metric_negative_zero_count[group] =
          seed.metric_negative_zero_count;
      break;
    case 9: output.min_int[group] = seed.min_int; break;
    case 10: output.max_int[group] = seed.max_int; break;
    case 11: output.int_sum[group] = seed.int_sum; break;
    case 12: output.long0_sum[group] = seed.long0_sum; break;
    case 13: output.long1_sum[group] = seed.long1_sum; break;
    case 14: output.min_long0[group] = seed.min_long0; break;
    case 15: output.max_long0[group] = seed.max_long0; break;
    case 16: output.min_long1[group] = seed.min_long1; break;
    case 17: output.max_long1[group] = seed.max_long1; break;
    case 18: output.min_metric[group] = seed.min_metric; break;
    case 19: output.max_metric[group] = seed.max_metric; break;
    default: break;
  }
}

__device__ __forceinline__ void persistent_update_field(
    AggregateColumns output,
    int group,
    int lane,
    const PackedInputRow& input,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  const std::uint32_t filters[3] = {input.filter0, input.filter1, input.filter2};
  const bool metric_passes =
      metric_filter_lane < 0 || filters[metric_filter_lane] != 0;
  const bool int_passes =
      int_filter_lane < 0 || filters[int_filter_lane] != 0;
  const bool long0_passes =
      long0_filter_lane < 0 || filters[long0_filter_lane] != 0;
  const bool long1_passes =
      long1_filter_lane < 0 || filters[long1_filter_lane] != 0;
  switch (lane) {
    case 0:
      output.row_count[group] += static_cast<unsigned long long>(
          static_cast<long long>(input.weight));
      break;
    case 1:
      if ((operation_mask & kMetricCount) != 0 && input.metric_valid != 0 &&
          metric_passes) output.metric_count[group] += static_cast<unsigned long long>(
              static_cast<long long>(input.weight));
      break;
    case 2:
      if ((operation_mask & kIntCount) != 0 && input.int_valid != 0 &&
          int_passes) output.int_count[group] += static_cast<unsigned long long>(
              static_cast<long long>(input.weight));
      break;
    case 3:
      if ((operation_mask & kLong0Count) != 0 && input.sequence_valid != 0 &&
          long0_passes) output.long0_count[group] += static_cast<unsigned long long>(
              static_cast<long long>(input.weight));
      break;
    case 4:
      if ((operation_mask & kLong1Count) != 0 && input.sent_nanos_valid != 0 &&
          long1_passes) output.long1_count[group] += static_cast<unsigned long long>(
              static_cast<long long>(input.weight));
      break;
    case 5:
      if ((operation_mask & kFilter0Count) != 0 && input.filter0 != 0)
        output.filter0_count[group] += static_cast<unsigned long long>(
            static_cast<long long>(input.weight));
      break;
    case 6:
      if ((operation_mask & kFilter1Count) != 0 && input.filter1 != 0)
        output.filter1_count[group] += static_cast<unsigned long long>(
            static_cast<long long>(input.weight));
      break;
    case 7:
      if ((operation_mask & kFilter2Count) != 0 && input.filter2 != 0)
        output.filter2_count[group] += static_cast<unsigned long long>(
            static_cast<long long>(input.weight));
      break;
    case 8:
      if ((operation_mask & kMetricSum) != 0 && input.metric_valid != 0 &&
          metric_passes) {
        if (is_negative_zero(input.metric)) {
          output.metric_negative_zero_count[group] +=
              static_cast<unsigned long long>(
                  static_cast<long long>(input.weight));
        }
        output.metric_sum[group] +=
            input.metric * static_cast<double>(input.weight);
      }
      break;
    case 9:
      if ((operation_mask & kMinInt) != 0 && input.int_valid != 0 && int_passes)
        output.min_int[group] = min(output.min_int[group], input.int_value);
      break;
    case 10:
      if ((operation_mask & kMaxInt) != 0 && input.int_valid != 0 && int_passes)
        output.max_int[group] = max(output.max_int[group], input.int_value);
      break;
    case 11:
      if ((operation_mask & kIntSum) != 0 && input.int_valid != 0 && int_passes)
        wrapping_add_i64(
            output.int_sum + group,
            weighted_i64_bits(static_cast<long long>(input.int_value), input.weight));
      break;
    case 12:
      if ((operation_mask & kLong0Sum) != 0 && input.sequence_valid != 0 &&
          long0_passes)
        wrapping_add_i64(output.long0_sum + group,
                         weighted_i64_bits(input.sequence, input.weight));
      break;
    case 13:
      if ((operation_mask & kLong1Sum) != 0 && input.sent_nanos_valid != 0 &&
          long1_passes)
        wrapping_add_i64(output.long1_sum + group,
                         weighted_i64_bits(input.sent_nanos, input.weight));
      break;
    case 14:
      if ((operation_mask & kMinLong0) != 0 && input.sequence_valid != 0 &&
          long0_passes)
        output.min_long0[group] = min(output.min_long0[group], input.sequence);
      break;
    case 15:
      if ((operation_mask & kMaxLong0) != 0 && input.sequence_valid != 0 &&
          long0_passes)
        output.max_long0[group] = max(output.max_long0[group], input.sequence);
      break;
    case 16:
      if ((operation_mask & kMinLong1) != 0 && input.sent_nanos_valid != 0 &&
          long1_passes)
        output.min_long1[group] = min(output.min_long1[group], input.sent_nanos);
      break;
    case 17:
      if ((operation_mask & kMaxLong1) != 0 && input.sent_nanos_valid != 0 &&
          long1_passes)
        output.max_long1[group] = max(output.max_long1[group], input.sent_nanos);
      break;
    case 18:
      if ((operation_mask & kMinMetric) != 0 && input.metric_valid != 0 &&
          metric_passes)
        output.min_metric[group] = min(output.min_metric[group], input.metric);
      break;
    case 19:
      if ((operation_mask & kMaxMetric) != 0 && input.metric_valid != 0 &&
          metric_passes)
        output.max_metric[group] = max(output.max_metric[group], input.metric);
      break;
    default:
      break;
  }
}

__device__ __forceinline__ void persistent_snapshot_input(
    const volatile PackedInputRow& source, PackedInputRow& target) {
  target.group = source.group;
  target.int_value = source.int_value;
  target.metric = source.metric;
  target.sequence = source.sequence;
  target.sent_nanos = source.sent_nanos;
  target.metric_valid = source.metric_valid;
  target.int_valid = source.int_valid;
  target.sequence_valid = source.sequence_valid;
  target.sent_nanos_valid = source.sent_nanos_valid;
  target.filter0 = source.filter0;
  target.filter1 = source.filter1;
  target.filter2 = source.filter2;
  target.weight = source.weight;
}

__device__ __forceinline__ void persistent_snapshot_seed(
    const volatile PackedAggregateRow& source, PackedAggregateRow& target) {
  target.row_count = source.row_count;
  target.metric_count = source.metric_count;
  target.int_count = source.int_count;
  target.long0_count = source.long0_count;
  target.long1_count = source.long1_count;
  target.filter0_count = source.filter0_count;
  target.filter1_count = source.filter1_count;
  target.filter2_count = source.filter2_count;
  target.metric_sum = source.metric_sum;
  target.min_int = source.min_int;
  target.max_int = source.max_int;
  target.int_sum = source.int_sum;
  target.long0_sum = source.long0_sum;
  target.long1_sum = source.long1_sum;
  target.min_long0 = source.min_long0;
  target.max_long0 = source.max_long0;
  target.min_long1 = source.min_long1;
  target.max_long1 = source.max_long1;
  target.min_metric = source.min_metric;
  target.max_metric = source.max_metric;
  target.metric_negative_zero_count = source.metric_negative_zero_count;
}

__device__ __forceinline__ double finalized_metric_sum(
    AggregateColumns input, int group) {
  const unsigned long long count = input.metric_count[group];
  return count != 0 &&
                 count == input.metric_negative_zero_count[group]
             ? __longlong_as_double(LLONG_MIN)
             : input.metric_sum[group];
}

__device__ __forceinline__ void persistent_write_response_field(
    AggregateColumns input,
    int group,
    int lane,
    volatile PackedAggregateRow& output) {
  switch (lane) {
    case 0: output.row_count = input.row_count[group]; break;
    case 1: output.metric_count = input.metric_count[group]; break;
    case 2: output.int_count = input.int_count[group]; break;
    case 3: output.long0_count = input.long0_count[group]; break;
    case 4: output.long1_count = input.long1_count[group]; break;
    case 5: output.filter0_count = input.filter0_count[group]; break;
    case 6: output.filter1_count = input.filter1_count[group]; break;
    case 7: output.filter2_count = input.filter2_count[group]; break;
    case 8: output.metric_sum = finalized_metric_sum(input, group); break;
    case 9: output.min_int = input.min_int[group]; break;
    case 10: output.max_int = input.max_int[group]; break;
    case 11: output.int_sum = input.int_sum[group]; break;
    case 12: output.long0_sum = input.long0_sum[group]; break;
    case 13: output.long1_sum = input.long1_sum[group]; break;
    case 14: output.min_long0 = input.min_long0[group]; break;
    case 15: output.max_long0 = input.max_long0[group]; break;
    case 16: output.min_long1 = input.min_long1[group]; break;
    case 17: output.max_long1 = input.max_long1[group]; break;
    case 18: output.min_metric = input.min_metric[group]; break;
    case 19: output.max_metric = input.max_metric[group]; break;
    default: break;
  }
}

__global__ void persistent_streaming_aggregate(
    volatile PersistentAggregateStorage* storage,
    int capacity,
    AggregateColumns aggregates,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  if (blockIdx.x != 0 || threadIdx.x >= 32) return;
  constexpr unsigned kWarpMask = 0xffffffffu;
  const int lane = threadIdx.x;
  __shared__ int wave_groups[32];
  __shared__ int wave_seed_valid[32];
  __shared__ int wave_materialize_response[32];
  __shared__ int wave_ranks[32];
  __shared__ std::uint32_t wave_requests[32];
  __shared__ PackedInputRow wave_inputs[32];
  __shared__ PackedAggregateRow wave_seeds[32];
  __shared__ int wave_max_rank;
  unsigned long long consumed = 0;
  unsigned long long produced = 0;
  if (lane == 0) {
    consumed =
        persistent_aggregate_load_acquire_u64(&storage->control.consumed);
    produced =
        persistent_aggregate_load_acquire_u64(&storage->control.produced);
  }
  consumed = __shfl_sync(kWarpMask, consumed, 0);
  produced = __shfl_sync(kWarpMask, produced, 0);
  __syncwarp(kWarpMask);

  while (consumed < produced) {
    const int active = static_cast<int>(min(produced - consumed, 32ULL));
    // Give every command a power-of-two sub-warp. A single available command keeps all 32 lanes;
    // a backlog automatically turns into up to 32 concurrent commands without waiting to fill a
    // batch first.
    int command_count = 1;
    while (command_count < active) command_count <<= 1;
    const int lanes_per_command = 32 / command_count;
    const int command = lane / lanes_per_command;
    const int field_lane = lane % lanes_per_command;
    const bool command_active = command < active;
    volatile PersistentAggregateSlot* slot =
        command_active
            ? &storage->slots[(consumed + command) &
                              kPersistentAggregateSlotMask]
            : nullptr;

    if (command_active && field_lane == 0) {
      wave_requests[command] =
          persistent_aggregate_load_acquire_u32(&slot->command.request);
      persistent_snapshot_input(slot->command.input, wave_inputs[command]);
      wave_groups[command] = wave_inputs[command].group;
      wave_seed_valid[command] = slot->command.seed_valid;
      wave_materialize_response[command] =
          slot->command.materialize_response;
      if (wave_seed_valid[command] != 0) {
        persistent_snapshot_seed(slot->command.seed, wave_seeds[command]);
      }
    }
    __syncwarp(kWarpMask);

    if (command_active && field_lane == 0) {
      const int group = wave_groups[command];
      int rank = 0;
      if (group >= 0 && group < capacity) {
        // Commands for different keys may run together. Commands for the same key are assigned
        // successive ranks so every emitted changelog row observes exact Flink arrival order.
        for (int prior = 0; prior < command; ++prior) {
          if (wave_groups[prior] == group) ++rank;
        }
      }
      wave_ranks[command] = rank;
      if (group < 0 || group >= capacity) {
        slot->response.error = 1;
        persistent_aggregate_store_release_u32(
            &slot->response.ticket, wave_requests[command]);
      }
    }
    __syncwarp(kWarpMask);
    if (lane == 0) {
      wave_max_rank = 0;
      for (int index = 0; index < active; ++index) {
        wave_max_rank = max(wave_max_rank, wave_ranks[index]);
      }
    }
    __syncwarp(kWarpMask);

    for (int rank = 0; rank <= wave_max_rank; ++rank) {
      const int group = command_active ? wave_groups[command] : -1;
      const bool process = command_active && group >= 0 && group < capacity &&
                           wave_ranks[command] == rank;
      if (process && wave_seed_valid[command] != 0) {
        for (int field = field_lane; field < 20;
             field += lanes_per_command) {
          persistent_seed_field(
              aggregates, group, field, wave_seeds[command]);
        }
      }
      __syncwarp(kWarpMask);
      if (process) {
        for (int field = field_lane; field < 20;
             field += lanes_per_command) {
          persistent_update_field(aggregates,
                                  group,
                                  field,
                                  wave_inputs[command],
                                  operation_mask,
                                  metric_filter_lane,
                                  int_filter_lane,
                                  long0_filter_lane,
                                  long1_filter_lane);
        }
      }
      __syncwarp(kWarpMask);
      if (process && field_lane == 0) {
        // One acquiring lane publishes the complete mapped response. The release store of the
        // ticket then orders every preceding response-field store at system scope; publishing
        // fields from several lanes allowed the host to observe a new ticket with one stale field.
        if (wave_materialize_response[command] != 0) {
          for (int field = 0; field < 20; ++field) {
            persistent_write_response_field(
                aggregates, group, field, slot->response.aggregate);
          }
        }
        slot->response.error = 0;
        persistent_aggregate_store_release_u32(
            &slot->response.ticket, wave_requests[command]);
      }
      __syncwarp(kWarpMask);
    }

    consumed += active;
    if (lane == 0) {
      persistent_aggregate_store_release_u64(&storage->control.consumed,
                                             consumed);
    }
    __syncwarp(kWarpMask);
  }
  if (lane == 0) {
    persistent_aggregate_store_release_u32(
        &storage->control.service_active, 0);
  }
}

__global__ void initialize_aggregates(AggregateColumns output, int groups) {
  for (int group = blockIdx.x * blockDim.x + threadIdx.x; group < groups;
       group += blockDim.x * gridDim.x) {
    output.row_count[group] = 0;
    output.metric_count[group] = 0;
    output.int_count[group] = 0;
    output.long0_count[group] = 0;
    output.long1_count[group] = 0;
    output.filter0_count[group] = 0;
    output.filter1_count[group] = 0;
    output.filter2_count[group] = 0;
    output.metric_negative_zero_count[group] = 0;
    output.metric_sum[group] = 0.0;
    output.min_metric[group] = DBL_MAX;
    output.max_metric[group] = -DBL_MAX;
    output.min_int[group] = INT_MAX;
    output.max_int[group] = INT_MIN;
    output.int_sum[group] = 0;
    output.long0_sum[group] = 0;
    output.long1_sum[group] = 0;
    output.min_long0[group] = LLONG_MAX;
    output.max_long0[group] = LLONG_MIN;
    output.min_long1[group] = LLONG_MAX;
    output.max_long1[group] = LLONG_MIN;
  }
}

__global__ void packed_global_groupby(
    const PackedInputRow* input,
    int rows,
    AggregateColumns output,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    update_group(output, input[row], operation_mask, metric_filter_lane,
                 int_filter_lane, long0_filter_lane, long1_filter_lane);
  }
}

template <int kStackCapacity>
__device__ __forceinline__ bool evaluate_fused_calc_row(
    const std::uint64_t* input_values,
    std::uint32_t valid_inputs,
    const CalcInstruction* program,
    int instruction_count,
    std::uint64_t* output_bits,
    std::uint32_t* valid_outputs,
    bool* program_valid) {
  static_assert(kStackCapacity > 0 && kStackCapacity <= 32,
                "invalid fused Calc stack capacity");
  std::uint64_t stack_bits[kStackCapacity];
  std::uint8_t stack_valid[kStackCapacity];
  int stack_size = 0;
  bool passes = true;
  *valid_outputs = 0;
  *program_valid = true;
  for (int pc = 0; pc < instruction_count; ++pc) {
    const CalcInstruction instruction = program[pc];
    const int opcode = instruction.opcode;
    if (opcode == kCalcInput) {
      stack_bits[stack_size] = input_values[instruction.arg];
      stack_valid[stack_size] =
          (valid_inputs & (1u << instruction.arg)) != 0;
      ++stack_size;
    } else if (opcode == kCalcLiteral) {
      stack_bits[stack_size] = instruction.bits;
      stack_valid[stack_size] = instruction.arg != 0;
      ++stack_size;
    } else if (opcode >= kCalcAdd && opcode <= kCalcPow) {
      --stack_size;
      const std::uint64_t right = stack_bits[stack_size];
      const bool right_valid = stack_valid[stack_size];
      const std::uint64_t left = stack_bits[stack_size - 1];
      bool valid = stack_valid[stack_size - 1] && right_valid;
      stack_bits[stack_size - 1] =
          valid ? calc_numeric_binary(
                      opcode, instruction.type, left, right, &valid)
                : 0;
      stack_valid[stack_size - 1] = valid;
    } else if (opcode >= kCalcEqual && opcode <= kCalcGreaterEqual) {
      --stack_size;
      const bool valid =
          stack_valid[stack_size - 1] && stack_valid[stack_size];
      stack_bits[stack_size - 1] =
          valid && calc_compare_bits(
                       opcode,
                       instruction.type,
                       stack_bits[stack_size - 1],
                       stack_bits[stack_size]);
      stack_valid[stack_size - 1] = valid;
    } else if (opcode == kCalcAnd || opcode == kCalcOr) {
      --stack_size;
      const bool left_valid = stack_valid[stack_size - 1];
      const bool right_valid = stack_valid[stack_size];
      const bool left = stack_bits[stack_size - 1] != 0;
      const bool right = stack_bits[stack_size] != 0;
      if (opcode == kCalcAnd) {
        if ((left_valid && !left) || (right_valid && !right)) {
          stack_bits[stack_size - 1] = 0;
          stack_valid[stack_size - 1] = true;
        } else {
          stack_bits[stack_size - 1] = 1;
          stack_valid[stack_size - 1] = left_valid && right_valid;
        }
      } else if ((left_valid && left) || (right_valid && right)) {
        stack_bits[stack_size - 1] = 1;
        stack_valid[stack_size - 1] = true;
      } else {
        stack_bits[stack_size - 1] = 0;
        stack_valid[stack_size - 1] = left_valid && right_valid;
      }
    } else if (opcode == kCalcNot) {
      if (stack_valid[stack_size - 1]) {
        stack_bits[stack_size - 1] = stack_bits[stack_size - 1] == 0;
      }
    } else if (opcode == kCalcIsNull) {
      stack_bits[stack_size - 1] = stack_valid[stack_size - 1] ? 0 : 1;
      stack_valid[stack_size - 1] = true;
    } else if (opcode >= kCalcIsTrue && opcode <= kCalcIsNotFalse) {
      const bool valid = stack_valid[stack_size - 1];
      const bool value = stack_bits[stack_size - 1] != 0;
      if (opcode == kCalcIsTrue) {
        stack_bits[stack_size - 1] = valid && value;
      } else if (opcode == kCalcIsFalse) {
        stack_bits[stack_size - 1] = valid && !value;
      } else if (opcode == kCalcIsNotTrue) {
        stack_bits[stack_size - 1] = !valid || !value;
      } else {
        stack_bits[stack_size - 1] = !valid || value;
      }
      stack_valid[stack_size - 1] = true;
    } else if (opcode == kCalcCast) {
      if (stack_valid[stack_size - 1]) {
        stack_bits[stack_size - 1] = calc_cast_bits(
            instruction.arg,
            instruction.type,
            stack_bits[stack_size - 1]);
      }
    } else if (opcode == kCalcSelect) {
      stack_size -= 2;
      const int condition_index = stack_size - 1;
      const int true_index = stack_size;
      const int false_index = stack_size + 1;
      const bool select_true =
          stack_valid[condition_index] && stack_bits[condition_index] != 0;
      const int selected = select_true ? true_index : false_index;
      stack_bits[condition_index] = stack_bits[selected];
      stack_valid[condition_index] = stack_valid[selected];
    } else if (opcode == kCalcCoalesce) {
      --stack_size;
      const int preferred = stack_size - 1;
      const int fallback = stack_size;
      if (!stack_valid[preferred]) {
        stack_bits[preferred] = stack_bits[fallback];
        stack_valid[preferred] = stack_valid[fallback];
      }
    } else if (opcode >= kCalcAbs && opcode <= kCalcExp) {
      if (stack_valid[stack_size - 1]) {
        stack_bits[stack_size - 1] = calc_unary_math(
            opcode, instruction.type, stack_bits[stack_size - 1]);
      }
    } else if (opcode == kCalcStoreCondition) {
      --stack_size;
      passes = stack_valid[stack_size] && stack_bits[stack_size] != 0;
      if (!passes) break;
    } else if (opcode == kCalcStoreOutput) {
      --stack_size;
      if (stack_valid[stack_size]) {
        output_bits[instruction.arg] = stack_bits[stack_size];
        *valid_outputs |= 1u << instruction.arg;
      }
    } else {
      *program_valid = false;
      return false;
    }
  }
  return passes;
}

__device__ __forceinline__ void assign_calc_lane(
    int output,
    std::uint32_t valid_outputs,
    const std::uint64_t* values,
    std::uint32_t* valid,
    std::uint64_t* bits) {
  if (output >= 0 && (valid_outputs & (1u << output)) != 0) {
    *valid = 1;
    *bits = values[output];
  } else {
    *valid = 0;
    *bits = 0;
  }
}

/**
 * Evaluates Calc and materializes the aggregate input directly in device memory.
 *
 * Keeping this as a producer kernel instead of calling update_group here is intentional: the
 * following groupby kernel can use its smaller, specialized register footprint (and its shared
 * aggregation variant) without sending the projected rows through the host.
 */
template <int kStackCapacity, int kOutputCapacity>
__global__ void calc_project_packed_rows(
    const unsigned char* raw_input,
    std::size_t input_row_bytes,
    const CalcInstruction* program,
    int instruction_count,
    int rows,
    PackedInputRow* projected,
    PackedAggregateRow* control,
    int metric_output,
    int int_output,
    int long0_output,
    int long1_output,
    int filter0_output,
    int filter1_output,
    int filter2_output,
    bool only_counts) {
  extern __shared__ __align__(16) unsigned char shared_program_bytes[];
  auto* shared_program =
      reinterpret_cast<CalcInstruction*>(shared_program_bytes);
  for (int pc = threadIdx.x; pc < instruction_count; pc += blockDim.x) {
    shared_program[pc] = program[pc];
  }
  __syncthreads();
  unsigned long long selected_rows = 0;
  bool requires_fallback = false;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    const unsigned char* row_input = raw_input + row * input_row_bytes;
    const int group = *reinterpret_cast<const int*>(row_input);
    const std::uint32_t valid_inputs =
        *reinterpret_cast<const std::uint32_t*>(row_input + sizeof(int));
    const auto* input_values =
        reinterpret_cast<const std::uint64_t*>(row_input + 8);
    static_assert(kOutputCapacity > 0 && kOutputCapacity <= 16,
                  "invalid fused Calc output capacity");
    std::uint64_t values[kOutputCapacity]{};
    std::uint32_t valid_outputs = 0;
    bool program_valid = true;
    const bool passes = evaluate_fused_calc_row<kStackCapacity>(
        input_values,
        valid_inputs,
        shared_program,
        instruction_count,
        values,
        &valid_outputs,
        &program_valid);
    if (!program_valid) {
      requires_fallback = true;
      projected[row] = PackedInputRow{};
      continue;
    }
    PackedInputRow packed{};
    if (!passes) {
      // A zero-weight, all-NULL row is an exact no-op even for MIN/MAX. Writing one slot per
      // input avoids a device compaction and lets the next kernel consume this buffer directly.
      projected[row] = packed;
      continue;
    }

    ++selected_rows;
    packed.group = group;
    packed.weight = 1;
    std::uint64_t bits = 0;
    assign_calc_lane(
        metric_output,
        valid_outputs,
        values,
        &packed.metric_valid,
        &bits);
    packed.metric = calc_bits_double(bits);
    if (packed.metric_valid != 0 && !only_counts && !isfinite(packed.metric)) {
      requires_fallback = true;
    }
    assign_calc_lane(
        int_output,
        valid_outputs,
        values,
        &packed.int_valid,
        &bits);
    packed.int_value = static_cast<std::int32_t>(bits);
    assign_calc_lane(
        long0_output,
        valid_outputs,
        values,
        &packed.sequence_valid,
        &bits);
    packed.sequence = static_cast<long long>(bits);
    assign_calc_lane(
        long1_output,
        valid_outputs,
        values,
        &packed.sent_nanos_valid,
        &bits);
    packed.sent_nanos = static_cast<long long>(bits);
    const int filter_outputs[3] = {
        filter0_output, filter1_output, filter2_output};
    std::uint32_t* filters[3] = {
        &packed.filter0, &packed.filter1, &packed.filter2};
    for (int filter = 0; filter < 3; ++filter) {
      const int output_ordinal = filter_outputs[filter];
      *filters[filter] =
          output_ordinal >= 0 &&
                  (valid_outputs & (1u << output_ordinal)) != 0 &&
                  values[output_ordinal] != 0
              ? 1
              : 0;
    }
    projected[row] = packed;
  }
  // Amortize the two control atomics over all grid-stride rows handled by this thread.
  if (selected_rows != 0) {
    atomicAdd(&control->metric_count, selected_rows);
  }
  if (requires_fallback) {
    atomicExch(&control->row_count, 1ULL);
  }
}

template <int kStackCapacity, int kOutputCapacity>
void launch_calc_projection(
    Context* context,
    int rows,
    int blocks,
    PackedAggregateRow* control,
    std::size_t shared_program_bytes) {
  calc_project_packed_rows<kStackCapacity, kOutputCapacity><<<
      blocks, kThreads, shared_program_bytes, context->stream>>>(
      context->device_calc_input,
      context->calc_input_row_bytes,
      context->device_calc_program,
      context->calc_instruction_count,
      rows,
      context->device_input,
      control,
      context->calc_metric_output,
      context->calc_int_output,
      context->calc_long0_output,
      context->calc_long1_output,
      context->calc_filter_outputs[0],
      context->calc_filter_outputs[1],
      context->calc_filter_outputs[2],
      context->calc_only_counts);
}

template <int kStackCapacity>
void launch_calc_projection_for_outputs(
    Context* context,
    int rows,
    int blocks,
    PackedAggregateRow* control,
    std::size_t shared_program_bytes) {
  if (context->calc_output_count <= 4) {
    launch_calc_projection<kStackCapacity, 4>(
        context, rows, blocks, control, shared_program_bytes);
  } else if (context->calc_output_count <= 8) {
    launch_calc_projection<kStackCapacity, 8>(
        context, rows, blocks, control, shared_program_bytes);
  } else {
    launch_calc_projection<kStackCapacity, 16>(
        context, rows, blocks, control, shared_program_bytes);
  }
}

template <int kStackCapacity, int kOutputCapacity>
void launch_calc_projection_to_device_batch(
    Context* context,
    int rows,
    int blocks,
    PackedInputRow* projected,
    PackedAggregateRow* control,
    cudaStream_t producer_stream,
    std::size_t shared_program_bytes) {
  calc_project_packed_rows<kStackCapacity, kOutputCapacity><<<
      blocks, kThreads, shared_program_bytes, producer_stream>>>(
      context->device_calc_input,
      context->calc_input_row_bytes,
      context->device_calc_program,
      context->calc_instruction_count,
      rows,
      projected,
      control,
      context->calc_metric_output,
      context->calc_int_output,
      context->calc_long0_output,
      context->calc_long1_output,
      context->calc_filter_outputs[0],
      context->calc_filter_outputs[1],
      context->calc_filter_outputs[2],
      context->calc_only_counts);
}

template <int kStackCapacity>
void launch_calc_projection_for_outputs_to_device_batch(
    Context* context,
    int rows,
    int blocks,
    PackedInputRow* projected,
    PackedAggregateRow* control,
    cudaStream_t producer_stream,
    std::size_t shared_program_bytes) {
  if (context->calc_output_count <= 4) {
    launch_calc_projection_to_device_batch<kStackCapacity, 4>(
        context,
        rows,
        blocks,
        projected,
        control,
        producer_stream,
        shared_program_bytes);
  } else if (context->calc_output_count <= 8) {
    launch_calc_projection_to_device_batch<kStackCapacity, 8>(
        context,
        rows,
        blocks,
        projected,
        control,
        producer_stream,
        shared_program_bytes);
  } else {
    launch_calc_projection_to_device_batch<kStackCapacity, 16>(
        context,
        rows,
        blocks,
        projected,
        control,
        producer_stream,
        shared_program_bytes);
  }
}

/**
 * Expands only the group id on device. The payload row is copied once per original input while
 * every grouping-set projection contributes one dense group id in a compact side array.
 */
__global__ void packed_expanded_global_groupby(
    const PackedInputRow* raw_input,
    const int* expanded_groups,
    int logical_rows,
    int project_count,
    AggregateColumns output,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < logical_rows;
       logical += blockDim.x * gridDim.x) {
    PackedInputRow value = raw_input[logical / project_count];
    value.group = expanded_groups[logical];
    update_group(output, value, operation_mask, metric_filter_lane,
                 int_filter_lane, long0_filter_lane, long1_filter_lane);
  }
}

constexpr std::size_t shared_bytes_for(int groups) {
  return static_cast<std::size_t>(groups) *
         (sizeof(unsigned long long) * 9 + sizeof(double) * 3 + sizeof(int) * 2 +
          sizeof(long long) * 7);
}

__device__ AggregateColumns shared_columns(unsigned char* storage, int groups) {
  AggregateColumns columns{};
  columns.row_count = reinterpret_cast<unsigned long long*>(storage);
  columns.metric_count = columns.row_count + groups;
  columns.int_count = columns.metric_count + groups;
  columns.long0_count = columns.int_count + groups;
  columns.long1_count = columns.long0_count + groups;
  columns.filter0_count = columns.long1_count + groups;
  columns.filter1_count = columns.filter0_count + groups;
  columns.filter2_count = columns.filter1_count + groups;
  columns.metric_negative_zero_count = columns.filter2_count + groups;
  columns.metric_sum =
      reinterpret_cast<double*>(columns.metric_negative_zero_count + groups);
  columns.min_metric = columns.metric_sum + groups;
  columns.max_metric = columns.min_metric + groups;
  columns.int_sum = reinterpret_cast<long long*>(columns.max_metric + groups);
  columns.long0_sum = columns.int_sum + groups;
  columns.long1_sum = columns.long0_sum + groups;
  columns.min_long0 = columns.long1_sum + groups;
  columns.max_long0 = columns.min_long0 + groups;
  columns.min_long1 = columns.max_long0 + groups;
  columns.max_long1 = columns.min_long1 + groups;
  columns.min_int = reinterpret_cast<int*>(columns.max_long1 + groups);
  columns.max_int = columns.min_int + groups;
  return columns;
}

__device__ void initialize_shared(AggregateColumns local, int groups) {
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    local.row_count[group] = 0;
    local.metric_count[group] = 0;
    local.int_count[group] = 0;
    local.long0_count[group] = 0;
    local.long1_count[group] = 0;
    local.filter0_count[group] = 0;
    local.filter1_count[group] = 0;
    local.filter2_count[group] = 0;
    local.metric_negative_zero_count[group] = 0;
    local.metric_sum[group] = 0.0;
    local.min_metric[group] = DBL_MAX;
    local.max_metric[group] = -DBL_MAX;
    local.min_int[group] = INT_MAX;
    local.max_int[group] = INT_MIN;
    local.int_sum[group] = 0;
    local.long0_sum[group] = 0;
    local.long1_sum[group] = 0;
    local.min_long0[group] = LLONG_MAX;
    local.max_long0[group] = LLONG_MIN;
    local.min_long1[group] = LLONG_MAX;
    local.max_long1[group] = LLONG_MIN;
  }
}

__device__ void flush_shared(
    AggregateColumns local,
    AggregateColumns output,
    int group,
    std::uint32_t operation_mask) {
  const unsigned long long rows = local.row_count[group];
  if (rows == 0) {
    return;
  }
  atomicAdd(output.row_count + group, rows);
  if ((operation_mask & kFilter0Count) != 0) {
    atomicAdd(output.filter0_count + group, local.filter0_count[group]);
  }
  if ((operation_mask & kFilter1Count) != 0) {
    atomicAdd(output.filter1_count + group, local.filter1_count[group]);
  }
  if ((operation_mask & kFilter2Count) != 0) {
    atomicAdd(output.filter2_count + group, local.filter2_count[group]);
  }
  const unsigned long long metrics = local.metric_count[group];
  if (metrics != 0) {
    if ((operation_mask & kMetricCount) != 0) {
      atomicAdd(output.metric_count + group, metrics);
    }
    if ((operation_mask & kMetricSum) != 0) {
      atomicAdd(output.metric_negative_zero_count + group,
                local.metric_negative_zero_count[group]);
      atomicAdd(output.metric_sum + group, local.metric_sum[group]);
    }
    if ((operation_mask & kMinMetric) != 0) {
      atomic_min_f64(output.min_metric + group, local.min_metric[group]);
    }
    if ((operation_mask & kMaxMetric) != 0) {
      atomic_max_f64(output.max_metric + group, local.max_metric[group]);
    }
  }
  const unsigned long long ints = local.int_count[group];
  if (ints != 0) {
    if ((operation_mask & kIntCount) != 0) {
      atomicAdd(output.int_count + group, ints);
    }
    if ((operation_mask & kIntSum) != 0) {
      atomic_add_i64(output.int_sum + group, local.int_sum[group]);
    }
    if ((operation_mask & kMinInt) != 0) {
      atomicMin(output.min_int + group, local.min_int[group]);
    }
    if ((operation_mask & kMaxInt) != 0) {
      atomicMax(output.max_int + group, local.max_int[group]);
    }
  }
  const unsigned long long long0s = local.long0_count[group];
  if (long0s != 0) {
    if ((operation_mask & kLong0Count) != 0) {
      atomicAdd(output.long0_count + group, long0s);
    }
    if ((operation_mask & kLong0Sum) != 0) {
      atomic_add_i64(output.long0_sum + group, local.long0_sum[group]);
    }
    if ((operation_mask & kMinLong0) != 0) {
      atomic_min_i64(output.min_long0 + group, local.min_long0[group]);
    }
    if ((operation_mask & kMaxLong0) != 0) {
      atomic_max_i64(output.max_long0 + group, local.max_long0[group]);
    }
  }
  const unsigned long long long1s = local.long1_count[group];
  if (long1s != 0) {
    if ((operation_mask & kLong1Count) != 0) {
      atomicAdd(output.long1_count + group, long1s);
    }
    if ((operation_mask & kLong1Sum) != 0) {
      atomic_add_i64(output.long1_sum + group, local.long1_sum[group]);
    }
    if ((operation_mask & kMinLong1) != 0) {
      atomic_min_i64(output.min_long1 + group, local.min_long1[group]);
    }
    if ((operation_mask & kMaxLong1) != 0) {
      atomic_max_i64(output.max_long1 + group, local.max_long1[group]);
    }
  }
}

__global__ void packed_shared_groupby(
    const PackedInputRow* input,
    int rows,
    int groups,
    AggregateColumns output,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  extern __shared__ unsigned char storage[];
  AggregateColumns local = shared_columns(storage, groups);
  initialize_shared(local, groups);
  __syncthreads();
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows;
       row += blockDim.x * gridDim.x) {
    update_group(local, input[row], operation_mask, metric_filter_lane,
                 int_filter_lane, long0_filter_lane, long1_filter_lane);
  }
  __syncthreads();
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    flush_shared(local, output, group, operation_mask);
  }
}

__global__ void packed_expanded_shared_groupby(
    const PackedInputRow* raw_input,
    const int* expanded_groups,
    int logical_rows,
    int project_count,
    int groups,
    AggregateColumns output,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  extern __shared__ unsigned char storage[];
  AggregateColumns local = shared_columns(storage, groups);
  initialize_shared(local, groups);
  __syncthreads();
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x;
       logical < logical_rows;
       logical += blockDim.x * gridDim.x) {
    PackedInputRow value = raw_input[logical / project_count];
    value.group = expanded_groups[logical];
    update_group(local, value, operation_mask, metric_filter_lane,
                 int_filter_lane, long0_filter_lane, long1_filter_lane);
  }
  __syncthreads();
  for (int group = threadIdx.x; group < groups; group += blockDim.x) {
    flush_shared(local, output, group, operation_mask);
  }
}

__global__ void pack_output(
    AggregateColumns input, PackedAggregateRow* output, int groups) {
  for (int group = blockIdx.x * blockDim.x + threadIdx.x; group < groups;
       group += blockDim.x * gridDim.x) {
    output[group] = {input.row_count[group],
                     input.metric_count[group],
                     input.int_count[group],
                     input.long0_count[group],
                     input.long1_count[group],
                     input.filter0_count[group],
                     input.filter1_count[group],
                     input.filter2_count[group],
                     finalized_metric_sum(input, group),
                     input.min_int[group],
                     input.max_int[group],
                     input.int_sum[group],
                     input.long0_sum[group],
                     input.long1_sum[group],
                     input.min_long0[group],
                     input.max_long0[group],
                     input.min_long1[group],
                     input.max_long1[group],
                     input.min_metric[group],
                     input.max_metric[group],
                     input.metric_negative_zero_count[group]};
  }
}

__global__ void pack_selected_output(
    AggregateColumns input,
    const int* selected_groups,
    PackedAggregateRow* output,
    int selected_count) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < selected_count;
       index += blockDim.x * gridDim.x) {
    const int group = selected_groups[index];
    output[index] = {input.row_count[group],
                     input.metric_count[group],
                     input.int_count[group],
                     input.long0_count[group],
                     input.long1_count[group],
                     input.filter0_count[group],
                     input.filter1_count[group],
                     input.filter2_count[group],
                     finalized_metric_sum(input, group),
                     input.min_int[group],
                     input.max_int[group],
                     input.int_sum[group],
                     input.long0_sum[group],
                     input.long1_sum[group],
                     input.min_long0[group],
                     input.max_long0[group],
                     input.min_long1[group],
                     input.max_long1[group],
                     input.min_metric[group],
                     input.max_metric[group],
                     input.metric_negative_zero_count[group]};
  }
}

void destroy_context(Context* context) {
  if (context == nullptr) {
    return;
  }
  if (context->device >= 0) {
    require_cuda(
        "bind dense groupby teardown device",
        cudaSetDevice(context->device));
  }
  synchronize_stream_for_teardown(
      "synchronize dense groupby DeviceBatch producer stream",
      context->device_batch_producer_stream);
  synchronize_stream_for_teardown(
      "synchronize dense groupby stream", context->stream);
  context->teardown_started = true;

  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch H2D start event",
      &context->device_batch_h2d_start);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch H2D end event",
      &context->device_batch_h2d_end);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch producer kernel event",
      &context->device_batch_producer_kernel_end);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch control D2H event",
      &context->device_batch_control_d2h_end);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch consumer wait start event",
      &context->device_batch_consumer_wait_start);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch consumer wait end event",
      &context->device_batch_consumer_wait_end);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch consumer kernel event",
      &context->device_batch_consumer_kernel_end);
  destroy_event_for_teardown(
      "destroy dense groupby DeviceBatch final D2H event",
      &context->device_batch_final_d2h_end);
  destroy_event_for_teardown(
      "destroy dense groupby H2D start event", &context->h2d_start);
  destroy_event_for_teardown(
      "destroy dense groupby H2D end event", &context->h2d_end);
  destroy_event_for_teardown(
      "destroy dense groupby compute end event", &context->compute_end);
  destroy_event_for_teardown(
      "destroy dense groupby D2H end event", &context->d2h_end);

  free_device_for_teardown(
      "free dense groupby device storage", &context->device_storage);
  if (context->device_storage == nullptr) {
    context->device_storage_bytes = 0;
    context->device_input = nullptr;
    context->aggregates = AggregateColumns{};
    context->device_output = nullptr;
  }
  free_device_for_teardown(
      "free dense groupby Calc input", &context->device_calc_input);
  free_device_for_teardown(
      "free dense groupby Calc program", &context->device_calc_program);
  free_host_for_teardown(
      "free dense groupby host input", &context->host_input);
  free_host_for_teardown(
      "free dense groupby host Calc input", &context->host_calc_input);
  free_host_for_teardown(
      "free dense groupby host expanded groups",
      &context->host_expanded_groups);
  free_host_for_teardown(
      "free dense groupby host output", &context->host_output);

  // Keep both streams and the pool valid until every other fallible release succeeds. Java may
  // retry its drain before re-entering nativeClose after this function reports a failure.
  destroy_stream_for_teardown(
      "destroy dense groupby DeviceBatch producer stream",
      &context->device_batch_producer_stream);
  destroy_stream_for_teardown(
      "destroy dense groupby stream", &context->stream);
  if (context->device_batch_producer_stream != nullptr ||
      context->stream != nullptr || context->device_batch_h2d_start != nullptr ||
      context->device_batch_h2d_end != nullptr ||
      context->device_batch_producer_kernel_end != nullptr ||
      context->device_batch_control_d2h_end != nullptr ||
      context->device_batch_consumer_wait_start != nullptr ||
      context->device_batch_consumer_wait_end != nullptr ||
      context->device_batch_consumer_kernel_end != nullptr ||
      context->device_batch_final_d2h_end != nullptr ||
      context->h2d_start != nullptr || context->h2d_end != nullptr ||
      context->compute_end != nullptr || context->d2h_end != nullptr ||
      context->device_storage != nullptr ||
      context->device_calc_input != nullptr ||
      context->device_calc_program != nullptr || context->host_input != nullptr ||
      context->host_calc_input != nullptr ||
      context->host_expanded_groups != nullptr || context->host_output != nullptr) {
    throw std::logic_error(
        "dense groupby teardown retained an owned native resource");
  }
  // Pool close is the final fallible operation. Once it succeeds there is no later exception that
  // could return a Java-visible handle whose pool has already disappeared.
  close_device_batch_pool_for_teardown(
      "close dense groupby DeviceBatch pool", &context->device_batch_pool);
  delete context;
}

[[noreturn]] void handle_failed_context_construction(
    Context* context, const std::exception_ptr& construction_failure) {
  try {
    destroy_context(context);
  } catch (...) {
    retain_construction_orphan(
        ConstructionOrphanKind::kDense,
        context,
        construction_failure,
        std::current_exception());
  }
  std::rethrow_exception(construction_failure);
}

Context* create_context(int capacity,
                        std::uint32_t operation_mask,
                        int metric_filter_lane,
                        int int_filter_lane,
                        int long0_filter_lane,
                        int long1_filter_lane) {
  if (capacity <= 0 || (operation_mask & kRowCount) == 0 ||
      (operation_mask & ~kAllOperations) != 0 || metric_filter_lane < -1 ||
      metric_filter_lane > 2 || int_filter_lane < -1 || int_filter_lane > 2 ||
      long0_filter_lane < -1 || long0_filter_lane > 2 ||
      long1_filter_lane < -1 || long1_filter_lane > 2) {
    throw std::invalid_argument("dense groupby capacity must be positive");
  }
  auto* context = new Context();
  try {
    context->capacity = capacity;
    context->operation_mask = operation_mask;
    context->metric_filter_lane = metric_filter_lane;
    context->int_filter_lane = int_filter_lane;
    context->long0_filter_lane = long0_filter_lane;
    context->long1_filter_lane = long1_filter_lane;
    cudaDeviceProp properties{};
    int device = 0;
    require_cuda("cudaGetDevice", cudaGetDevice(&device));
    context->device = device;
    require_cuda("cudaGetDeviceProperties",
                 cudaGetDeviceProperties(&properties, device));
    if (!mega_flink_device_matches_native_build(properties)) {
      throw std::runtime_error(
          std::string("dense groupby build does not admit selected device: ") +
          properties.name);
    }
    context->multiprocessors = properties.multiProcessorCount;
    require_cuda("cudaHostAlloc input",
                 cudaHostAlloc(&context->host_input,
                               static_cast<std::size_t>(capacity) * sizeof(PackedInputRow),
                               cudaHostAllocDefault));
    require_cuda("cudaHostAlloc expanded groups",
                 cudaHostAlloc(&context->host_expanded_groups,
                               static_cast<std::size_t>(capacity) * sizeof(int),
                               cudaHostAllocDefault));
    require_cuda("cudaHostAlloc output",
                 cudaHostAlloc(&context->host_output,
                               kOutputHeaderBytes +
                                   static_cast<std::size_t>(capacity + 1) *
                                       sizeof(PackedAggregateRow),
                               cudaHostAllocDefault));
    context->device_storage_bytes = bind_device_storage(context, nullptr);
    require_cuda("cudaMalloc dense groupby slab",
                 cudaMalloc(&context->device_storage,
                            context->device_storage_bytes));
    const std::size_t bound_bytes =
        bind_device_storage(context, context->device_storage);
    if (bound_bytes != context->device_storage_bytes) {
      throw std::logic_error("dense groupby slab layout size changed while binding");
    }
    require_cuda("cudaStreamCreate",
                 cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
    require_cuda("cudaEventCreate h2d start", cudaEventCreate(&context->h2d_start));
    require_cuda("cudaEventCreate h2d end", cudaEventCreate(&context->h2d_end));
    require_cuda("cudaEventCreate compute end", cudaEventCreate(&context->compute_end));
    require_cuda("cudaEventCreate d2h end", cudaEventCreate(&context->d2h_end));
    require_cuda("shared kernel opt-in",
                 cudaFuncSetAttribute(
                     packed_shared_groupby,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(shared_bytes_for(kSharedGroupLimit))));
    require_cuda("expanded shared kernel opt-in",
                 cudaFuncSetAttribute(
                     packed_expanded_shared_groupby,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(shared_bytes_for(kSharedGroupLimit))));
    std::memset(context->host_output,
                0,
                kOutputHeaderBytes +
                    static_cast<std::size_t>(capacity + 1) * sizeof(PackedAggregateRow));
    return context;
  } catch (...) {
    handle_failed_context_construction(context, std::current_exception());
  }
}

void execute(Context* context, int rows, int groups) {
  if (context == nullptr) {
    throw std::invalid_argument("dense groupby context is null");
  }
  if (rows <= 0 || rows > context->capacity || groups <= 0 || groups > rows ||
      groups > context->capacity) {
    throw std::invalid_argument("invalid dense groupby rows/groups");
  }
  const std::size_t input_bytes =
      static_cast<std::size_t>(rows) * sizeof(PackedInputRow);
  const std::size_t output_bytes =
      static_cast<std::size_t>(groups) * sizeof(PackedAggregateRow);
  const int blocks = std::max(
      1, std::min((rows + kThreads - 1) / kThreads,
                  context->multiprocessors * 2));

  require_cuda("record H2D start",
               cudaEventRecord(context->h2d_start, context->stream));
  require_cuda("copy dense input",
               cudaMemcpyAsync(context->device_input,
                               context->host_input,
                               input_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("record H2D end",
               cudaEventRecord(context->h2d_end, context->stream));
  initialize_aggregates<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, groups);
  require_cuda("initialize dense aggregates", cudaGetLastError());

  int variant = 0;
  if (groups <= kSharedGroupLimit && rows >= kSharedMinimumRows) {
    variant = 1;
    packed_shared_groupby<<<
        blocks, kThreads, shared_bytes_for(groups), context->stream>>>(
        context->device_input,
        rows,
        groups,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("packed shared groupby", cudaGetLastError());
  } else {
    packed_global_groupby<<<blocks, kThreads, 0, context->stream>>>(
        context->device_input,
        rows,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("packed global groupby", cudaGetLastError());
  }
  pack_output<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, context->device_output, groups);
  require_cuda("pack dense output", cudaGetLastError());
  require_cuda("record compute end",
               cudaEventRecord(context->compute_end, context->stream));
  require_cuda("copy dense output",
               cudaMemcpyAsync(context->host_output + kOutputHeaderBytes,
                               context->device_output,
                               output_bytes,
                               cudaMemcpyDeviceToHost,
                               context->stream));
  require_cuda("record D2H end",
               cudaEventRecord(context->d2h_end, context->stream));
  require_cuda("wait dense output", cudaEventSynchronize(context->d2h_end));

  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  require_cuda("time dense H2D",
               cudaEventElapsedTime(&h2d_ms, context->h2d_start, context->h2d_end));
  require_cuda("time dense compute",
               cudaEventElapsedTime(
                   &compute_ms, context->h2d_end, context->compute_end));
  require_cuda("time dense D2H",
               cudaEventElapsedTime(&d2h_ms, context->compute_end, context->d2h_end));
  auto* header = reinterpret_cast<OutputHeader*>(context->host_output);
  header->h2d_nanos = static_cast<std::uint64_t>(std::llround(h2d_ms * 1'000'000.0));
  header->compute_nanos =
      static_cast<std::uint64_t>(std::llround(compute_ms * 1'000'000.0));
  header->d2h_nanos = static_cast<std::uint64_t>(std::llround(d2h_ms * 1'000'000.0));
  header->input_bytes = input_bytes;
  header->output_bytes = output_bytes;
  header->rows = rows;
  header->groups = groups;
  header->kernel_variant = variant;
}

void execute_expanded(Context* context,
                      int raw_rows,
                      int project_count,
                      int groups) {
  if (context == nullptr) {
    throw std::invalid_argument("expanded dense groupby context is null");
  }
  if (raw_rows <= 0 || project_count <= 1 ||
      raw_rows > context->capacity / project_count) {
    throw std::invalid_argument("invalid expanded dense groupby row shape");
  }
  const int logical_rows = raw_rows * project_count;
  if (groups <= 0 || groups > logical_rows || groups > context->capacity) {
    throw std::invalid_argument("invalid expanded dense groupby group count");
  }
  const std::size_t raw_input_bytes =
      static_cast<std::size_t>(raw_rows) * sizeof(PackedInputRow);
  const std::size_t group_input_bytes =
      static_cast<std::size_t>(logical_rows) * sizeof(int);
  const std::size_t input_bytes = raw_input_bytes + group_input_bytes;
  if (input_bytes >
      static_cast<std::size_t>(context->capacity) * sizeof(PackedInputRow)) {
    throw std::invalid_argument("expanded dense groupby packed input exceeds capacity");
  }
  const std::size_t output_bytes =
      static_cast<std::size_t>(groups) * sizeof(PackedAggregateRow);
  const int blocks = std::max(
      1, std::min((logical_rows + kThreads - 1) / kThreads,
                  context->multiprocessors * 2));

  // The fixed input slab is large enough for the raw payload followed by all compact group ids
  // whenever project_count >= 2. Coalesce them on the host so CUDA sees one H2D operation rather
  // than paying a second launch-sized memcpy overhead for a tiny side array.
  std::memcpy(reinterpret_cast<unsigned char*>(context->host_input) +
                  raw_input_bytes,
              context->host_expanded_groups,
              group_input_bytes);
  auto* expanded_device_groups = reinterpret_cast<const int*>(
      reinterpret_cast<const unsigned char*>(context->device_input) +
      raw_input_bytes);
  require_cuda("record expanded H2D start",
               cudaEventRecord(context->h2d_start, context->stream));
  require_cuda("copy expanded packed input",
               cudaMemcpyAsync(context->device_input,
                               context->host_input,
                               input_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("record expanded H2D end",
               cudaEventRecord(context->h2d_end, context->stream));
  initialize_aggregates<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, groups);
  require_cuda("initialize expanded dense aggregates", cudaGetLastError());

  int variant = 2;
  if (groups <= kSharedGroupLimit && logical_rows >= kSharedMinimumRows) {
    variant = 3;
    packed_expanded_shared_groupby<<<
        blocks, kThreads, shared_bytes_for(groups), context->stream>>>(
        context->device_input,
        expanded_device_groups,
        logical_rows,
        project_count,
        groups,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("packed expanded shared groupby", cudaGetLastError());
  } else {
    packed_expanded_global_groupby<<<blocks, kThreads, 0, context->stream>>>(
        context->device_input,
        expanded_device_groups,
        logical_rows,
        project_count,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("packed expanded global groupby", cudaGetLastError());
  }
  pack_output<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, context->device_output, groups);
  require_cuda("pack expanded dense output", cudaGetLastError());
  require_cuda("record expanded compute end",
               cudaEventRecord(context->compute_end, context->stream));
  require_cuda("copy expanded dense output",
               cudaMemcpyAsync(context->host_output + kOutputHeaderBytes,
                               context->device_output,
                               output_bytes,
                               cudaMemcpyDeviceToHost,
                               context->stream));
  require_cuda("record expanded D2H end",
               cudaEventRecord(context->d2h_end, context->stream));
  require_cuda("wait expanded dense output",
               cudaEventSynchronize(context->d2h_end));

  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  require_cuda("time expanded dense H2D",
               cudaEventElapsedTime(&h2d_ms,
                                    context->h2d_start,
                                    context->h2d_end));
  require_cuda("time expanded dense compute",
               cudaEventElapsedTime(&compute_ms,
                                    context->h2d_end,
                                    context->compute_end));
  require_cuda("time expanded dense D2H",
               cudaEventElapsedTime(&d2h_ms,
                                    context->compute_end,
                                    context->d2h_end));
  auto* header = reinterpret_cast<OutputHeader*>(context->host_output);
  header->h2d_nanos =
      static_cast<std::uint64_t>(std::llround(h2d_ms * 1'000'000.0));
  header->compute_nanos =
      static_cast<std::uint64_t>(std::llround(compute_ms * 1'000'000.0));
  header->d2h_nanos =
      static_cast<std::uint64_t>(std::llround(d2h_ms * 1'000'000.0));
  header->input_bytes = input_bytes;
  header->output_bytes = output_bytes;
  header->rows = logical_rows;
  header->groups = groups;
  header->kernel_variant = variant;
}

void configure_calc_context(
    Context* context,
    int input_count,
    int output_count,
    const CalcInstruction* host_program,
    int instruction_count,
    int max_stack,
    int metric_output,
    int int_output,
    int long0_output,
    int long1_output,
    int filter0_output,
    int filter1_output,
    int filter2_output,
    bool only_counts) {
  if (context == nullptr || context->host_calc_input != nullptr ||
      input_count <= 0 || input_count > 16 || output_count <= 0 ||
      output_count > 16 || host_program == nullptr ||
      instruction_count <= 0 || instruction_count > 256 || max_stack <= 0 ||
      max_stack > 32) {
    throw std::invalid_argument("invalid fused Calc groupby dimensions");
  }
  const int outputs[7] = {metric_output,
                          int_output,
                          long0_output,
                          long1_output,
                          filter0_output,
                          filter1_output,
                          filter2_output};
  for (int output : outputs) {
    if (output < -1 || output >= output_count) {
      throw std::invalid_argument("invalid fused Calc groupby output mapping");
    }
  }
  context->calc_input_count = input_count;
  context->calc_output_count = output_count;
  context->calc_instruction_count = instruction_count;
  context->calc_max_stack = max_stack;
  context->calc_metric_output = metric_output;
  context->calc_int_output = int_output;
  context->calc_long0_output = long0_output;
  context->calc_long1_output = long1_output;
  context->calc_filter_outputs[0] = filter0_output;
  context->calc_filter_outputs[1] = filter1_output;
  context->calc_filter_outputs[2] = filter2_output;
  context->calc_only_counts = only_counts;
  context->calc_input_row_bytes =
      8 + static_cast<std::size_t>(input_count) * sizeof(std::uint64_t);
  const std::size_t input_bytes =
      static_cast<std::size_t>(context->capacity) *
      context->calc_input_row_bytes;
  const std::size_t program_bytes =
      static_cast<std::size_t>(instruction_count) * sizeof(CalcInstruction);
  require_cuda(
      "allocate fused Calc host input",
      cudaHostAlloc(
          &context->host_calc_input, input_bytes, cudaHostAllocDefault));
  require_cuda(
      "allocate fused Calc device input",
      cudaMalloc(&context->device_calc_input, input_bytes));
  require_cuda(
      "allocate fused Calc bytecode",
      cudaMalloc(&context->device_calc_program, program_bytes));
  require_cuda(
      "copy fused Calc bytecode",
      cudaMemcpyAsync(
          context->device_calc_program,
          host_program,
          program_bytes,
          cudaMemcpyHostToDevice,
          context->stream));
  require_cuda(
      "finish fused Calc configuration", cudaStreamSynchronize(context->stream));
}

void execute_calc(Context* context, int rows, int groups) {
  if (context == nullptr || context->host_calc_input == nullptr ||
      context->device_calc_input == nullptr ||
      context->device_calc_program == nullptr) {
    throw std::invalid_argument("fused Calc groupby is not configured");
  }
  if (rows <= 0 || rows > context->capacity || groups <= 0 || groups > rows ||
      groups > context->capacity) {
    throw std::invalid_argument("invalid fused Calc groupby rows/groups");
  }
  const std::size_t input_bytes =
      static_cast<std::size_t>(rows) * context->calc_input_row_bytes;
  const std::size_t output_bytes =
      static_cast<std::size_t>(groups + 1) * sizeof(PackedAggregateRow);
  const int blocks = std::max(
      1,
      std::min(
          (rows + kThreads - 1) / kThreads,
          context->multiprocessors * 2));
  auto* control = context->device_output + groups;

  require_cuda(
      "record fused Calc H2D start",
      cudaEventRecord(context->h2d_start, context->stream));
  require_cuda(
      "copy fused Calc packed input",
      cudaMemcpyAsync(
          context->device_calc_input,
          context->host_calc_input,
          input_bytes,
          cudaMemcpyHostToDevice,
          context->stream));
  require_cuda(
      "record fused Calc H2D end",
      cudaEventRecord(context->h2d_end, context->stream));
  initialize_aggregates<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, groups);
  require_cuda("initialize fused Calc aggregates", cudaGetLastError());
  require_cuda(
      "initialize fused Calc control",
      cudaMemsetAsync(control, 0, sizeof(PackedAggregateRow), context->stream));
  const std::size_t shared_program_bytes =
      static_cast<std::size_t>(context->calc_instruction_count) *
      sizeof(CalcInstruction);
  if (context->calc_max_stack <= 4) {
    launch_calc_projection_for_outputs<4>(
        context, rows, blocks, control, shared_program_bytes);
  } else if (context->calc_max_stack <= 8) {
    launch_calc_projection_for_outputs<8>(
        context, rows, blocks, control, shared_program_bytes);
  } else if (context->calc_max_stack <= 16) {
    launch_calc_projection_for_outputs<16>(
        context, rows, blocks, control, shared_program_bytes);
  } else {
    launch_calc_projection_for_outputs<32>(
        context, rows, blocks, control, shared_program_bytes);
  }
  require_cuda("fused Calc device projection", cudaGetLastError());
  int variant = 5;
  if (groups <= kSharedGroupLimit && rows >= kSharedMinimumRows) {
    variant = 6;
    packed_shared_groupby<<<
        blocks, kThreads, shared_bytes_for(groups), context->stream>>>(
        context->device_input,
        rows,
        groups,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("fused Calc staged shared groupby", cudaGetLastError());
  } else {
    packed_global_groupby<<<blocks, kThreads, 0, context->stream>>>(
        context->device_input,
        rows,
        context->aggregates,
        context->operation_mask,
        context->metric_filter_lane,
        context->int_filter_lane,
        context->long0_filter_lane,
        context->long1_filter_lane);
    require_cuda("fused Calc staged global groupby", cudaGetLastError());
  }
  pack_output<<<
      (groups + kThreads - 1) / kThreads, kThreads, 0, context->stream>>>(
      context->aggregates, context->device_output, groups);
  require_cuda("pack fused Calc dense output", cudaGetLastError());
  require_cuda(
      "record fused Calc compute end",
      cudaEventRecord(context->compute_end, context->stream));
  require_cuda(
      "copy fused Calc dense output",
      cudaMemcpyAsync(
          context->host_output + kOutputHeaderBytes,
          context->device_output,
          output_bytes,
          cudaMemcpyDeviceToHost,
          context->stream));
  require_cuda(
      "record fused Calc D2H end",
      cudaEventRecord(context->d2h_end, context->stream));
  require_cuda(
      "wait fused Calc dense output",
      cudaEventSynchronize(context->d2h_end));

  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  require_cuda(
      "time fused Calc H2D",
      cudaEventElapsedTime(&h2d_ms, context->h2d_start, context->h2d_end));
  require_cuda(
      "time fused Calc compute",
      cudaEventElapsedTime(
          &compute_ms, context->h2d_end, context->compute_end));
  require_cuda(
      "time fused Calc D2H",
      cudaEventElapsedTime(
          &d2h_ms, context->compute_end, context->d2h_end));
  auto* header = reinterpret_cast<OutputHeader*>(context->host_output);
  header->h2d_nanos =
      static_cast<std::uint64_t>(std::llround(h2d_ms * 1'000'000.0));
  header->compute_nanos =
      static_cast<std::uint64_t>(std::llround(compute_ms * 1'000'000.0));
  header->d2h_nanos =
      static_cast<std::uint64_t>(std::llround(d2h_ms * 1'000'000.0));
  header->input_bytes = input_bytes;
  header->output_bytes = output_bytes;
  header->rows = rows;
  header->groups = groups;
  header->kernel_variant = variant;
}

void configure_calc_device_batch_context(
    Context* context, std::uint64_t owner_chain_id) {
  if (context == nullptr || context->host_calc_input == nullptr ||
      context->device_calc_input == nullptr ||
      context->device_calc_program == nullptr || owner_chain_id == 0 ||
      context->device_batch_pool != 0) {
    throw std::invalid_argument("invalid Calc device-batch configuration");
  }
  context->device_batch_chain = owner_chain_id;
  require_cuda(
      "create device-batch producer stream",
      cudaStreamCreateWithFlags(
          &context->device_batch_producer_stream, cudaStreamNonBlocking));
  require_cuda(
      "create device-batch H2D start event",
      cudaEventCreate(&context->device_batch_h2d_start));
  require_cuda(
      "create device-batch H2D end event",
      cudaEventCreate(&context->device_batch_h2d_end));
  require_cuda(
      "create device-batch producer kernel event",
      cudaEventCreate(&context->device_batch_producer_kernel_end));
  require_cuda(
      "create device-batch control D2H event",
      cudaEventCreate(&context->device_batch_control_d2h_end));
  require_cuda(
      "create device-batch consumer wait start event",
      cudaEventCreate(&context->device_batch_consumer_wait_start));
  require_cuda(
      "create device-batch consumer wait end event",
      cudaEventCreate(&context->device_batch_consumer_wait_end));
  require_cuda(
      "create device-batch consumer kernel event",
      cudaEventCreate(&context->device_batch_consumer_kernel_end));
  require_cuda(
      "create device-batch final D2H event",
      cudaEventCreate(&context->device_batch_final_d2h_end));
  require_device_batch(
      "create Calc device-batch pool",
      cudf_flink_gpu_device_batch_pool_create(
          context->device,
          owner_chain_id,
          static_cast<std::uint64_t>(context->capacity) *
              sizeof(PackedInputRow),
          kDeviceBatchPoolSlots,
          &context->device_batch_pool));
  context->device_batch_metrics[kBatchMetricsAbi] =
      CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
}

void reset_device_batch_execution_metrics(Context* context) {
  std::fill(
      context->device_batch_metrics,
      context->device_batch_metrics + kBatchMetricsAbi,
      0);
  context->device_batch_metrics[kBatchMetricsAbi] =
      CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
}

void write_device_batch_output_header(
    Context* context,
    int rows,
    int groups,
    int variant,
    bool include_final_output) {
  auto* header = reinterpret_cast<OutputHeader*>(context->host_output);
  header->h2d_nanos = context->device_batch_metrics[kBatchProducerH2DNanos];
  header->compute_nanos =
      context->device_batch_metrics[kBatchProducerKernelNanos] +
      context->device_batch_metrics[kBatchConsumerWaitNanos] +
      context->device_batch_metrics[kBatchConsumerKernelNanos];
  header->d2h_nanos =
      context->device_batch_metrics[kBatchControlD2HNanos] +
      context->device_batch_metrics[kBatchFinalD2HNanos];
  header->input_bytes = context->device_batch_metrics[kBatchProducerH2DBytes];
  header->output_bytes =
      context->device_batch_metrics[kBatchControlD2HBytes] +
      (include_final_output
           ? context->device_batch_metrics[kBatchFinalD2HBytes]
           : 0);
  header->rows = rows;
  header->groups = groups;
  header->kernel_variant = variant;
}

std::uint64_t produce_calc_device_batch(
    Context* context, int rows, int groups) {
  if (context == nullptr || context->device_batch_pool == 0 ||
      context->device_batch_producer_stream == nullptr ||
      context->host_calc_input == nullptr ||
      context->device_calc_input == nullptr ||
      context->device_calc_program == nullptr) {
    throw std::invalid_argument("Calc device-batch producer is not configured");
  }
  if (rows <= 0 || rows > context->capacity || groups <= 0 || groups > rows ||
      groups > context->capacity) {
    throw std::invalid_argument("invalid Calc device-batch rows/groups");
  }
  reset_device_batch_execution_metrics(context);
  const std::size_t input_bytes =
      static_cast<std::size_t>(rows) * context->calc_input_row_bytes;
  const int blocks = std::max(
      1,
      std::min(
          (rows + kThreads - 1) / kThreads,
          context->multiprocessors * 2));
  auto* control = context->device_output + groups;
  const auto descriptor = packed_input_descriptor(context, rows);
  std::uint64_t batch_handle = 0;
  void* producer_data = nullptr;
  bool published = false;
  try {
    require_device_batch(
        "acquire Calc device batch",
        cudf_flink_gpu_device_batch_pool_acquire(
            context->device_batch_pool,
            &descriptor,
            &batch_handle,
            &producer_data));
    auto* projected = static_cast<PackedInputRow*>(producer_data);
    cudaStream_t producer_stream = context->device_batch_producer_stream;
    require_cuda(
        "record device-batch producer H2D start",
        cudaEventRecord(context->device_batch_h2d_start, producer_stream));
    require_cuda(
        "copy device-batch Calc input",
        cudaMemcpyAsync(
            context->device_calc_input,
            context->host_calc_input,
            input_bytes,
            cudaMemcpyHostToDevice,
            producer_stream));
    require_cuda(
        "record device-batch producer H2D end",
        cudaEventRecord(context->device_batch_h2d_end, producer_stream));
    require_cuda(
        "initialize device-batch Calc control",
        cudaMemsetAsync(
            control, 0, sizeof(PackedAggregateRow), producer_stream));
    const std::size_t shared_program_bytes =
        static_cast<std::size_t>(context->calc_instruction_count) *
        sizeof(CalcInstruction);
    if (context->calc_max_stack <= 4) {
      launch_calc_projection_for_outputs_to_device_batch<4>(
          context,
          rows,
          blocks,
          projected,
          control,
          producer_stream,
          shared_program_bytes);
    } else if (context->calc_max_stack <= 8) {
      launch_calc_projection_for_outputs_to_device_batch<8>(
          context,
          rows,
          blocks,
          projected,
          control,
          producer_stream,
          shared_program_bytes);
    } else if (context->calc_max_stack <= 16) {
      launch_calc_projection_for_outputs_to_device_batch<16>(
          context,
          rows,
          blocks,
          projected,
          control,
          producer_stream,
          shared_program_bytes);
    } else {
      launch_calc_projection_for_outputs_to_device_batch<32>(
          context,
          rows,
          blocks,
          projected,
          control,
          producer_stream,
          shared_program_bytes);
    }
    require_cuda("launch device-batch Calc projection", cudaGetLastError());
    require_cuda(
        "record device-batch producer kernel end",
        cudaEventRecord(
            context->device_batch_producer_kernel_end, producer_stream));
    require_device_batch(
        "seal Calc device batch",
        cudf_flink_gpu_device_batch_seal(batch_handle, producer_stream));
    require_cuda(
        "copy device-batch Calc control",
        cudaMemcpyAsync(
            context->host_output + kOutputHeaderBytes +
                static_cast<std::size_t>(groups) *
                    sizeof(PackedAggregateRow),
            control,
            sizeof(PackedAggregateRow),
            cudaMemcpyDeviceToHost,
            producer_stream));
    require_cuda(
        "record device-batch control D2H end",
        cudaEventRecord(
            context->device_batch_control_d2h_end, producer_stream));
    require_cuda(
        "wait device-batch Calc control",
        cudaEventSynchronize(context->device_batch_control_d2h_end));

    context->device_batch_metrics[kBatchProducerH2DBytes] = input_bytes;
    context->device_batch_metrics[kBatchProducerH2DNanos] = event_nanos(
        "time device-batch producer H2D",
        context->device_batch_h2d_start,
        context->device_batch_h2d_end);
    context->device_batch_metrics[kBatchProducerKernelNanos] = event_nanos(
        "time device-batch producer kernel",
        context->device_batch_h2d_end,
        context->device_batch_producer_kernel_end);
    context->device_batch_metrics[kBatchControlD2HBytes] =
        sizeof(PackedAggregateRow);
    context->device_batch_metrics[kBatchControlD2HNanos] = event_nanos(
        "time device-batch control D2H",
        context->device_batch_producer_kernel_end,
        context->device_batch_control_d2h_end);
    // These two counters are deliberately zero: the projected payload stays on device and the
    // consumer borrows it instead of uploading it again.
    context->device_batch_metrics[kBatchIntermediatePayloadD2HBytes] = 0;
    context->device_batch_metrics[kBatchConsumerH2DBytes] = 0;

    const auto* host_control = reinterpret_cast<const PackedAggregateRow*>(
        context->host_output + kOutputHeaderBytes +
        static_cast<std::size_t>(groups) * sizeof(PackedAggregateRow));
    if (host_control->row_count != 0) {
      require_device_batch(
          "abort fallback Calc device batch",
          cudf_flink_gpu_device_batch_abort(batch_handle, producer_stream));
      write_device_batch_output_header(
          context, rows, groups, 9, false);
      return 0;
    }
    require_device_batch(
        "publish Calc device batch",
        cudf_flink_gpu_device_batch_publish(batch_handle));
    published = true;
    context->device_batch_metrics[kBatchProduced] = 1;
    return batch_handle;
  } catch (...) {
    if (batch_handle != 0 && !published) {
      // The common runtime fences both Writing and Sealed slots before making them reusable.
      cudf_flink_gpu_device_batch_abort(
          batch_handle, context->device_batch_producer_stream);
    }
    throw;
  }
}

void consume_calc_device_batch(
    Context* context,
    std::uint64_t batch_handle,
    int rows,
    int groups) {
  if (context == nullptr || context->device_batch_pool == 0 ||
      batch_handle == 0 || rows <= 0 || rows > context->capacity ||
      groups <= 0 || groups > rows || groups > context->capacity) {
    throw std::invalid_argument("invalid Calc device-batch consumer input");
  }
  CudfFlinkGpuDeviceBatchDescriptor expected =
      packed_input_descriptor(context, rows);
  CudfFlinkGpuDeviceBatchBorrowedView view{};
  view.struct_size = sizeof(view);
  view.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  const int blocks = std::max(
      1,
      std::min(
          (rows + kThreads - 1) / kThreads,
          context->multiprocessors * 2));
  bool borrowed = false;
  bool consumption_recorded = false;
  try {
    require_cuda(
        "record device-batch consumer wait start",
        cudaEventRecord(
            context->device_batch_consumer_wait_start, context->stream));
    require_device_batch(
        "borrow Calc device batch",
        cudf_flink_gpu_device_batch_borrow(
            batch_handle, &expected, context->stream, &view));
    borrowed = true;
    require_cuda(
        "record device-batch consumer wait end",
        cudaEventRecord(
            context->device_batch_consumer_wait_end, context->stream));
    initialize_aggregates<<<
        (groups + kThreads - 1) / kThreads,
        kThreads,
        0,
        context->stream>>>(context->aggregates, groups);
    require_cuda(
        "initialize device-batch consumer aggregates", cudaGetLastError());
    int variant = 7;
    auto* projected = static_cast<const PackedInputRow*>(view.device_data);
    if (groups <= kSharedGroupLimit && rows >= kSharedMinimumRows) {
      variant = 8;
      packed_shared_groupby<<<
          blocks,
          kThreads,
          shared_bytes_for(groups),
          context->stream>>>(
          projected,
          rows,
          groups,
          context->aggregates,
          context->operation_mask,
          context->metric_filter_lane,
          context->int_filter_lane,
          context->long0_filter_lane,
          context->long1_filter_lane);
      require_cuda(
          "launch device-batch shared groupby", cudaGetLastError());
    } else {
      packed_global_groupby<<<blocks, kThreads, 0, context->stream>>>(
          projected,
          rows,
          context->aggregates,
          context->operation_mask,
          context->metric_filter_lane,
          context->int_filter_lane,
          context->long0_filter_lane,
          context->long1_filter_lane);
      require_cuda(
          "launch device-batch global groupby", cudaGetLastError());
    }
    pack_output<<<
        (groups + kThreads - 1) / kThreads,
        kThreads,
        0,
        context->stream>>>(
        context->aggregates, context->device_output, groups);
    require_cuda(
        "pack device-batch aggregate output", cudaGetLastError());
    require_cuda(
        "record device-batch consumer kernel end",
        cudaEventRecord(
            context->device_batch_consumer_kernel_end, context->stream));
    require_device_batch(
        "mark Calc device batch consumed",
        cudf_flink_gpu_device_batch_mark_consumed(
            batch_handle, context->stream));
    consumption_recorded = true;
    const std::size_t final_output_bytes =
        static_cast<std::size_t>(groups) * sizeof(PackedAggregateRow);
    require_cuda(
        "copy device-batch aggregate output",
        cudaMemcpyAsync(
            context->host_output + kOutputHeaderBytes,
            context->device_output,
            final_output_bytes,
            cudaMemcpyDeviceToHost,
            context->stream));
    require_cuda(
        "record device-batch final D2H end",
        cudaEventRecord(
            context->device_batch_final_d2h_end, context->stream));
    require_cuda(
        "wait device-batch aggregate output",
        cudaEventSynchronize(context->device_batch_final_d2h_end));
    context->device_batch_metrics[kBatchConsumerWaitNanos] = event_nanos(
        "time device-batch consumer wait",
        context->device_batch_consumer_wait_start,
        context->device_batch_consumer_wait_end);
    context->device_batch_metrics[kBatchConsumerKernelNanos] = event_nanos(
        "time device-batch consumer kernel",
        context->device_batch_consumer_wait_end,
        context->device_batch_consumer_kernel_end);
    context->device_batch_metrics[kBatchFinalD2HBytes] = final_output_bytes;
    context->device_batch_metrics[kBatchFinalD2HNanos] = event_nanos(
        "time device-batch final D2H",
        context->device_batch_consumer_kernel_end,
        context->device_batch_final_d2h_end);
    context->device_batch_metrics[kBatchConsumed] = 1;
    write_device_batch_output_header(
        context, rows, groups, variant, true);
  } catch (...) {
    if (borrowed && !consumption_recorded) {
      cudf_flink_gpu_device_batch_abort_consume(
          batch_handle, context->stream);
    }
    throw;
  }
}

void update_device_batch_pool_metrics(
    Context* context,
    const CudfFlinkGpuDeviceBatchPoolStats& stats) {
  context->device_batch_metrics[kBatchDrainCalls] =
      stats.drain_calls - context->device_batch_observed_drain_calls;
  context->device_batch_metrics[kBatchForcedLeakedReferences] =
      stats.forced_leaked_references -
      context->device_batch_observed_forced_leaks;
  context->device_batch_metrics[kBatchStaleTokenRejections] =
      stats.stale_token_rejections -
      context->device_batch_observed_stale_rejections;
  context->device_batch_observed_drain_calls = stats.drain_calls;
  context->device_batch_observed_forced_leaks =
      stats.forced_leaked_references;
  context->device_batch_observed_stale_rejections =
      stats.stale_token_rejections;
  context->device_batch_metrics[kBatchAllocatedBytes] = stats.allocated_bytes;
  context->device_batch_metrics[kBatchActiveBytes] = stats.active_bytes;
  context->device_batch_metrics[kBatchPeakActiveBytes] =
      stats.peak_active_bytes;
  context->device_batch_metrics[kBatchActiveReferences] =
      stats.active_references;
}

void refresh_device_batch_pool_metrics(Context* context) {
  if (context == nullptr || context->device_batch_pool == 0) {
    throw std::invalid_argument("Calc device-batch pool is not configured");
  }
  CudfFlinkGpuDeviceBatchPoolStats stats{};
  stats.struct_size = sizeof(stats);
  stats.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  require_device_batch(
      "read Calc device-batch pool stats",
      cudf_flink_gpu_device_batch_pool_stats(
          context->device_batch_pool, &stats));
  update_device_batch_pool_metrics(context, stats);
}

void drain_calc_device_batch_context(Context* context, bool force) {
  if (context == nullptr || context->device_batch_pool == 0) {
    throw std::invalid_argument("Calc device-batch pool is not configured");
  }
  if (!context->teardown_started &&
      (context->device_batch_producer_stream == nullptr ||
       context->stream == nullptr)) {
    throw std::logic_error(
        "Calc device-batch streams disappeared before teardown");
  }
  synchronize_stream_for_teardown(
      "drain Calc device-batch producer stream",
      context->device_batch_producer_stream);
  synchronize_stream_for_teardown(
      "drain Calc device-batch consumer stream", context->stream);
  CudfFlinkGpuDeviceBatchPoolStats stats{};
  stats.struct_size = sizeof(stats);
  stats.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
  const std::int32_t status = cudf_flink_gpu_device_batch_pool_drain(
      context->device_batch_pool,
      force ? CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_FORCE
            : CUDF_FLINK_GPU_DEVICE_BATCH_DRAIN_NONE,
      &stats);
  update_device_batch_pool_metrics(context, stats);
  if (status != CUDF_FLINK_GPU_DEVICE_BATCH_OK) {
    throw std::runtime_error(
        std::string("drain Calc device-batch pool failed: ") +
        cudf_flink_gpu_device_batch_status_string(status) +
        "; active_refs=" + std::to_string(stats.active_references) +
        "; active_bytes=" + std::to_string(stats.active_bytes));
  }
}

std::size_t bind_flexible_device_storage(
    FlexibleContext* context, unsigned char* storage) {
  DeviceSlabLayout layout(storage);
  const std::size_t capacity = static_cast<std::size_t>(context->capacity);
  context->device_input = layout.take<PackedInputRow>(capacity);
  context->device_program =
      layout.take<FlexibleProgramOp>(context->operation_count);
  context->device_results = layout.take<FlexibleResult>(
      capacity * static_cast<std::size_t>(context->result_stride));
  context->device_negative_zero_counts = layout.take<unsigned long long>(
      capacity * static_cast<std::size_t>(context->result_stride));
  return layout.bytes();
}

void destroy_flexible_context(FlexibleContext* context) {
  if (context == nullptr) {
    return;
  }
  if (context->device >= 0) {
    require_cuda(
        "bind flexible dense groupby teardown device",
        cudaSetDevice(context->device));
  }
  synchronize_stream_for_teardown(
      "synchronize flexible dense groupby stream", context->stream);
  destroy_event_for_teardown(
      "destroy flexible dense groupby H2D start event", &context->h2d_start);
  destroy_event_for_teardown(
      "destroy flexible dense groupby H2D end event", &context->h2d_end);
  destroy_event_for_teardown(
      "destroy flexible dense groupby compute end event",
      &context->compute_end);
  destroy_event_for_teardown(
      "destroy flexible dense groupby D2H end event", &context->d2h_end);
  free_device_for_teardown(
      "free flexible dense groupby device storage",
      &context->device_storage);
  if (context->device_storage == nullptr) {
    context->device_storage_bytes = 0;
    context->device_input = nullptr;
    context->device_program = nullptr;
    context->device_results = nullptr;
    context->device_negative_zero_counts = nullptr;
  }
  free_host_for_teardown(
      "free flexible dense groupby host input", &context->host_input);
  free_host_for_teardown(
      "free flexible dense groupby host output", &context->host_output);
  destroy_stream_for_teardown(
      "destroy flexible dense groupby stream", &context->stream);
  if (context->stream != nullptr || context->h2d_start != nullptr ||
      context->h2d_end != nullptr || context->compute_end != nullptr ||
      context->d2h_end != nullptr || context->device_storage != nullptr ||
      context->host_input != nullptr || context->host_output != nullptr) {
    throw std::logic_error(
        "flexible dense groupby teardown retained an owned native resource");
  }
  delete context;
}

[[noreturn]] void handle_failed_flexible_context_construction(
    FlexibleContext* context,
    const std::exception_ptr& construction_failure) {
  try {
    destroy_flexible_context(context);
  } catch (...) {
    retain_construction_orphan(
        ConstructionOrphanKind::kFlexible,
        context,
        construction_failure,
        std::current_exception());
  }
  std::rethrow_exception(construction_failure);
}

void retry_construction_orphan() {
  if (construction_orphan.kind == ConstructionOrphanKind::kNone) {
    if (construction_orphan.context != nullptr) {
      throw std::logic_error(
          "dense groupby construction orphan kind/context mismatch");
    }
    return;
  }
  if (construction_orphan.context == nullptr) {
    throw std::logic_error(
        "dense groupby construction orphan context is missing");
  }

  try {
    if (construction_orphan.kind == ConstructionOrphanKind::kDense) {
      destroy_context(static_cast<Context*>(construction_orphan.context));
    } else if (construction_orphan.kind == ConstructionOrphanKind::kFlexible) {
      destroy_flexible_context(
          static_cast<FlexibleContext*>(construction_orphan.context));
    } else if (
        construction_orphan.kind == ConstructionOrphanKind::kPrimitiveLong) {
      destroy_primitive_long_aggregate(
          static_cast<PrimitiveLongAggregateContext*>(
              construction_orphan.context));
    } else {
      throw std::logic_error(
          "dense groupby construction orphan kind is invalid");
    }
  } catch (const std::exception& failure) {
    throw std::runtime_error(
        std::string("retry dense groupby construction orphan teardown failed; ") +
        "new native construction rejected: " + failure.what());
  } catch (...) {
    throw std::runtime_error(
        "retry dense groupby construction orphan teardown failed with an "
        "unknown error; new native construction rejected");
  }

  // destroy_* returned only after deleting the retained context. Clear the slot afterwards, so a
  // failed retry always keeps the exact pointer reachable for the next serialized create.
  construction_orphan.context = nullptr;
  construction_orphan.kind = ConstructionOrphanKind::kNone;
}

void validate_flexible_program(
    const FlexibleProgramOp* program, int operation_count) {
  if (operation_count <= 0 || operation_count > kFlexibleMaxOperations) {
    throw std::invalid_argument("flexible aggregate operation count is invalid");
  }
  for (int index = 0; index < operation_count; index++) {
    const FlexibleProgramOp& operation = program[index];
    if (operation.kind < kFlexibleCount || operation.kind > kFlexibleMax ||
        operation.value_lane < kFlexibleRow ||
        operation.value_lane > kFlexibleLong1 || operation.filter_lane < -1 ||
        operation.filter_lane > 2) {
      throw std::invalid_argument("flexible aggregate program is invalid");
    }
    if (operation.value_lane == kFlexibleRow &&
        operation.kind != kFlexibleCount) {
      throw std::invalid_argument(
          "only COUNT may use the flexible row value lane");
    }
    if (operation.kind == kFlexibleAverage &&
        operation.value_lane == kFlexibleRow) {
      throw std::invalid_argument("flexible AVG requires a value lane");
    }
  }
}

FlexibleContext* create_flexible_context(
    int capacity,
    const FlexibleProgramOp* program,
    int operation_count) {
  if (capacity <= 0) {
    throw std::invalid_argument(
        "flexible dense groupby capacity must be positive");
  }
  validate_flexible_program(program, operation_count);
  auto* context = new FlexibleContext();
  try {
    context->capacity = capacity;
    context->operation_count = operation_count;
    context->result_stride = operation_count + 1;
    cudaDeviceProp properties{};
    int device = 0;
    require_cuda("cudaGetDevice", cudaGetDevice(&device));
    context->device = device;
    require_cuda("cudaGetDeviceProperties",
                 cudaGetDeviceProperties(&properties, device));
    if (!mega_flink_device_matches_native_build(properties)) {
      throw std::runtime_error(
          std::string("flexible dense groupby build does not admit selected device: ") +
          properties.name);
    }
    context->multiprocessors = properties.multiProcessorCount;
    const std::size_t input_bytes =
        static_cast<std::size_t>(capacity) * sizeof(PackedInputRow);
    const std::size_t output_bytes =
        kOutputHeaderBytes + static_cast<std::size_t>(capacity) *
                                 context->result_stride * sizeof(FlexibleResult);
    require_cuda("cudaHostAlloc flexible input",
                 cudaHostAlloc(&context->host_input,
                               input_bytes,
                               cudaHostAllocDefault));
    require_cuda("cudaHostAlloc flexible output",
                 cudaHostAlloc(&context->host_output,
                               output_bytes,
                               cudaHostAllocDefault));
    context->device_storage_bytes =
        bind_flexible_device_storage(context, nullptr);
    require_cuda("cudaMalloc flexible groupby slab",
                 cudaMalloc(&context->device_storage,
                            context->device_storage_bytes));
    const std::size_t bound_bytes =
        bind_flexible_device_storage(context, context->device_storage);
    if (bound_bytes != context->device_storage_bytes) {
      throw std::logic_error(
          "flexible groupby slab layout size changed while binding");
    }
    require_cuda("copy flexible aggregate program",
                 cudaMemcpy(context->device_program,
                            program,
                            static_cast<std::size_t>(operation_count) *
                                sizeof(FlexibleProgramOp),
                            cudaMemcpyHostToDevice));
    require_cuda("cudaStreamCreate flexible",
                 cudaStreamCreateWithFlags(
                     &context->stream, cudaStreamNonBlocking));
    require_cuda("cudaEventCreate flexible H2D start",
                 cudaEventCreate(&context->h2d_start));
    require_cuda("cudaEventCreate flexible H2D end",
                 cudaEventCreate(&context->h2d_end));
    require_cuda("cudaEventCreate flexible compute end",
                 cudaEventCreate(&context->compute_end));
    require_cuda("cudaEventCreate flexible D2H end",
                 cudaEventCreate(&context->d2h_end));
    const std::size_t maximum_shared_bytes =
        static_cast<std::size_t>(kSharedGroupLimit) *
        (kFlexibleMaxOperations + 1) *
        (sizeof(FlexibleResult) + sizeof(unsigned long long));
    require_cuda("flexible shared kernel opt-in",
                 cudaFuncSetAttribute(
                     flexible_shared_groupby,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     static_cast<int>(maximum_shared_bytes)));
    std::memset(context->host_output, 0, output_bytes);
    return context;
  } catch (...) {
    handle_failed_flexible_context_construction(
        context, std::current_exception());
  }
}

void execute_flexible(FlexibleContext* context, int rows, int groups) {
  if (context == nullptr) {
    throw std::invalid_argument("flexible dense groupby context is null");
  }
  if (rows <= 0 || rows > context->capacity || groups <= 0 || groups > rows ||
      groups > context->capacity) {
    throw std::invalid_argument("invalid flexible dense groupby rows/groups");
  }
  const std::size_t input_bytes =
      static_cast<std::size_t>(rows) * sizeof(PackedInputRow);
  const std::size_t output_bytes =
      static_cast<std::size_t>(groups) * context->result_stride *
      sizeof(FlexibleResult);
  const int result_count = groups * context->result_stride;
  const int blocks = std::max(
      1, std::min((rows + kThreads - 1) / kThreads,
                  context->multiprocessors * 2));

  require_cuda("record flexible H2D start",
               cudaEventRecord(context->h2d_start, context->stream));
  require_cuda("copy flexible dense input",
               cudaMemcpyAsync(context->device_input,
                               context->host_input,
                               input_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("record flexible H2D end",
               cudaEventRecord(context->h2d_end, context->stream));
  initialize_flexible_results<<<
      (result_count + kThreads - 1) / kThreads,
      kThreads,
      0,
      context->stream>>>(context->device_results,
                         context->device_negative_zero_counts,
                         context->device_program,
                         groups,
                         context->operation_count);
  require_cuda("initialize flexible aggregate results", cudaGetLastError());

  int variant = 2;
  const std::size_t shared_bytes =
      static_cast<std::size_t>(result_count) *
      (sizeof(FlexibleResult) + sizeof(unsigned long long));
  if (groups <= kSharedGroupLimit && rows >= kSharedMinimumRows) {
    variant = 3;
    flexible_shared_groupby<<<blocks,
                              kThreads,
                              shared_bytes,
                              context->stream>>>(context->device_input,
                                                rows,
                                                groups,
                                                context->device_program,
                                                context->operation_count,
                                                context->device_results,
                                                context->device_negative_zero_counts);
    require_cuda("flexible shared groupby", cudaGetLastError());
  } else {
    flexible_global_groupby<<<blocks, kThreads, 0, context->stream>>>(
        context->device_input,
        rows,
        context->device_program,
        context->operation_count,
        context->device_results,
        context->device_negative_zero_counts);
    require_cuda("flexible global groupby", cudaGetLastError());
  }
  finalize_flexible_floating_zero<<<
      (result_count + kThreads - 1) / kThreads,
      kThreads,
      0,
      context->stream>>>(context->device_results,
                         context->device_negative_zero_counts,
                         context->device_program,
                         result_count,
                         context->result_stride);
  require_cuda("finalize flexible floating zero", cudaGetLastError());
  require_cuda("record flexible compute end",
               cudaEventRecord(context->compute_end, context->stream));
  require_cuda("copy flexible dense output",
               cudaMemcpyAsync(context->host_output + kOutputHeaderBytes,
                               context->device_results,
                               output_bytes,
                               cudaMemcpyDeviceToHost,
                               context->stream));
  require_cuda("record flexible D2H end",
               cudaEventRecord(context->d2h_end, context->stream));
  require_cuda("wait flexible dense output",
               cudaEventSynchronize(context->d2h_end));

  float h2d_ms = 0.0f;
  float compute_ms = 0.0f;
  float d2h_ms = 0.0f;
  require_cuda("time flexible dense H2D",
               cudaEventElapsedTime(
                   &h2d_ms, context->h2d_start, context->h2d_end));
  require_cuda("time flexible dense compute",
               cudaEventElapsedTime(
                   &compute_ms, context->h2d_end, context->compute_end));
  require_cuda("time flexible dense D2H",
               cudaEventElapsedTime(
                   &d2h_ms, context->compute_end, context->d2h_end));
  auto* header = reinterpret_cast<OutputHeader*>(context->host_output);
  header->h2d_nanos =
      static_cast<std::uint64_t>(std::llround(h2d_ms * 1'000'000.0));
  header->compute_nanos =
      static_cast<std::uint64_t>(std::llround(compute_ms * 1'000'000.0));
  header->d2h_nanos =
      static_cast<std::uint64_t>(std::llround(d2h_ms * 1'000'000.0));
  header->input_bytes = input_bytes;
  header->output_bytes = output_bytes;
  header->rows = rows;
  header->groups = groups;
  header->kernel_variant = variant;
}

template <typename T>
T persistent_aggregate_host_load_acquire(volatile T* address) {
  return __atomic_load_n(const_cast<T*>(address), __ATOMIC_ACQUIRE);
}

template <typename T>
void persistent_aggregate_host_store_release(volatile T* address, T value) {
  __atomic_store_n(const_cast<T*>(address), value, __ATOMIC_RELEASE);
}

void destroy_persistent_aggregate(PersistentAggregateContext* context) {
  if (context == nullptr) return;
  if (context->device >= 0) cudaSetDevice(context->device);
  if (context->stream != nullptr) cudaStreamSynchronize(context->stream);
  if (context->stream != nullptr) cudaStreamDestroy(context->stream);
  if (context->device_snapshot_groups != nullptr)
    cudaFree(context->device_snapshot_groups);
  if (context->device_snapshot != nullptr) cudaFree(context->device_snapshot);
  if (context->host_snapshot != nullptr) cudaFreeHost(context->host_snapshot);
  if (context->device_aggregate_storage != nullptr)
    cudaFree(context->device_aggregate_storage);
  if (context->host_storage != nullptr) cudaFreeHost(context->host_storage);
  delete context;
}

PersistentAggregateContext* create_persistent_aggregate(
    int capacity,
    std::uint32_t operation_mask,
    int metric_filter_lane,
    int int_filter_lane,
    int long0_filter_lane,
    int long1_filter_lane) {
  if (capacity <= 0 || (operation_mask & kRowCount) == 0 ||
      (operation_mask & ~kAllOperations) != 0 || metric_filter_lane < -1 ||
      metric_filter_lane > 2 || int_filter_lane < -1 ||
      int_filter_lane > 2 || long0_filter_lane < -1 ||
      long0_filter_lane > 2 || long1_filter_lane < -1 ||
      long1_filter_lane > 2) {
    throw std::invalid_argument("invalid persistent Aggregate configuration");
  }
  auto* context = new PersistentAggregateContext();
  try {
    context->capacity = capacity;
    context->operation_mask = operation_mask;
    context->metric_filter_lane = metric_filter_lane;
    context->int_filter_lane = int_filter_lane;
    context->long0_filter_lane = long0_filter_lane;
    context->long1_filter_lane = long1_filter_lane;

    cudaDeviceProp properties{};
    int device = 0;
    require_cuda("cudaGetDevice", cudaGetDevice(&device));
    context->device = device;
    require_cuda("cudaGetDeviceProperties",
                 cudaGetDeviceProperties(&properties, device));
    if (!mega_flink_device_matches_native_build(properties)) {
      throw std::runtime_error(
          std::string("persistent Aggregate build does not admit selected device: ") +
          properties.name);
    }

    require_cuda(
        "allocate persistent Aggregate mapped queue",
        cudaHostAlloc(&context->host_storage,
                      sizeof(PersistentAggregateStorage),
                      cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(
        context->host_storage, 0, sizeof(PersistentAggregateStorage));
    require_cuda(
        "map persistent Aggregate queue",
        cudaHostGetDevicePointer(
            &context->device_storage, context->host_storage, 0));

    context->device_aggregate_storage_bytes =
        bind_persistent_aggregate_storage(context, nullptr);
    require_cuda(
        "allocate persistent Aggregate state",
        cudaMalloc(&context->device_aggregate_storage,
                   context->device_aggregate_storage_bytes));
    const std::size_t bound_bytes = bind_persistent_aggregate_storage(
        context, context->device_aggregate_storage);
    if (bound_bytes != context->device_aggregate_storage_bytes) {
      throw std::logic_error(
          "persistent Aggregate state layout changed while binding");
    }
    require_cuda(
        "allocate persistent Aggregate snapshot group ids",
        cudaMalloc(&context->device_snapshot_groups,
                   static_cast<std::size_t>(capacity) * sizeof(int)));
    require_cuda(
        "allocate persistent Aggregate device snapshot",
        cudaMalloc(&context->device_snapshot,
                   static_cast<std::size_t>(capacity) *
                       sizeof(PackedAggregateRow)));
    require_cuda(
        "allocate persistent Aggregate host snapshot",
        cudaHostAlloc(&context->host_snapshot,
                      static_cast<std::size_t>(capacity) *
                          sizeof(PackedAggregateRow),
                      cudaHostAllocPortable));
    require_cuda(
        "create persistent Aggregate stream",
        cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
    initialize_aggregates<<<
        (capacity + kThreads - 1) / kThreads,
        kThreads,
        0,
        context->stream>>>(context->aggregates, capacity);
    require_cuda("initialize persistent Aggregate state", cudaPeekAtLastError());
    require_cuda("initialize persistent Aggregate state",
                 cudaStreamSynchronize(context->stream));
    persistent_aggregate_host_store_release(
        &context->host_storage->control.ready, std::uint32_t{1});
    return context;
  } catch (...) {
    destroy_persistent_aggregate(context);
    throw;
  }
}

void kick_persistent_aggregate(PersistentAggregateContext* context) {
  if (context == nullptr) {
    throw std::invalid_argument("persistent Aggregate context is null");
  }
  require_cuda("bind persistent Aggregate CUDA device",
               cudaSetDevice(context->device));
  std::uint32_t expected = 0;
  if (!__atomic_compare_exchange_n(
          const_cast<std::uint32_t*>(
              &context->host_storage->control.service_active),
          &expected,
          std::uint32_t{1},
          false,
          __ATOMIC_ACQ_REL,
          __ATOMIC_ACQUIRE)) {
    const cudaError_t status = cudaStreamQuery(context->stream);
    if (status == cudaErrorNotReady) return;
    if (status == cudaSuccess) {
      // The finite service completed between the host CAS and the stream query. Clear a stale
      // flag defensively so this call can schedule the remaining commands.
      persistent_aggregate_host_store_release(
          &context->host_storage->control.service_active, std::uint32_t{0});
      return kick_persistent_aggregate(context);
    }
    persistent_aggregate_host_store_release(
        &context->host_storage->control.service_active, std::uint32_t{0});
    require_cuda("query persistent Aggregate service", status);
  }
  // H20 does not preempt a never-ending kernel from the same CUDA context. A resident polling
  // warp therefore starves Calc, Join, and every other operator sharing the TaskManager GPU.
  // Keep the aggregate state resident, but launch a finite service warp only when Java publishes
  // work. Multiple launches are stream-ordered and a launch may drain several queued commands.
  persistent_streaming_aggregate<<<1, 32, 0, context->stream>>>(
      context->device_storage,
      context->capacity,
      context->aggregates,
      context->operation_mask,
      context->metric_filter_lane,
      context->int_filter_lane,
      context->long0_filter_lane,
      context->long1_filter_lane);
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) {
    persistent_aggregate_host_store_release(
        &context->host_storage->control.service_active, std::uint32_t{0});
    require_cuda("launch persistent Aggregate service", launch_status);
  }
}

void snapshot_persistent_aggregate(
    PersistentAggregateContext* context,
    const int* groups,
    int group_count) {
  if (context == nullptr) {
    throw std::invalid_argument("persistent Aggregate context is null");
  }
  if (group_count < 0 || group_count > context->capacity ||
      (group_count > 0 && groups == nullptr)) {
    throw std::invalid_argument(
        "persistent Aggregate snapshot group count is out of range");
  }
  for (int index = 0; index < group_count; ++index) {
    if (groups[index] < 0 || groups[index] >= context->capacity) {
      throw std::invalid_argument(
          "persistent Aggregate snapshot group id is out of range");
    }
  }
  require_cuda("bind persistent Aggregate CUDA device",
               cudaSetDevice(context->device));
  if (group_count == 0) return;
  require_cuda(
      "copy persistent Aggregate snapshot group ids",
      cudaMemcpyAsync(context->device_snapshot_groups,
                      groups,
                      static_cast<std::size_t>(group_count) * sizeof(int),
                      cudaMemcpyHostToDevice,
                      context->stream));
  pack_selected_output<<<(group_count + kThreads - 1) / kThreads,
                         kThreads,
                         0,
                         context->stream>>>(context->aggregates,
                                            context->device_snapshot_groups,
                                            context->device_snapshot,
                                            group_count);
  require_cuda("pack persistent Aggregate snapshot", cudaPeekAtLastError());
  require_cuda(
      "copy persistent Aggregate snapshot",
      cudaMemcpyAsync(context->host_snapshot,
                      context->device_snapshot,
                      static_cast<std::size_t>(group_count) *
                          sizeof(PackedAggregateRow),
                      cudaMemcpyDeviceToHost,
                      context->stream));
  require_cuda("finish persistent Aggregate snapshot",
               cudaStreamSynchronize(context->stream));
}

Context* context_from(jlong handle) {
  return reinterpret_cast<Context*>(static_cast<std::uintptr_t>(handle));
}

FlexibleContext* flexible_context_from(jlong handle) {
  return reinterpret_cast<FlexibleContext*>(
      static_cast<std::uintptr_t>(handle));
}

PersistentAggregateContext* persistent_aggregate_context_from(jlong handle) {
  return reinterpret_cast<PersistentAggregateContext*>(
      static_cast<std::uintptr_t>(handle));
}

jlongArray take_device_batch_metrics(
    JNIEnv* env, Context* context, bool drain, bool force) {
  if (drain) {
    drain_calc_device_batch_context(context, force);
  } else {
    refresh_device_batch_pool_metrics(context);
  }
  jlong values[kDeviceBatchMetricFields]{};
  for (int index = 0; index < kDeviceBatchMetricFields; ++index) {
    values[index] =
        static_cast<jlong>(context->device_batch_metrics[index]);
  }
  jlongArray result = env->NewLongArray(kDeviceBatchMetricFields);
  if (result == nullptr) return nullptr;
  env->SetLongArrayRegion(result, 0, kDeviceBatchMetricFields, values);
  reset_device_batch_execution_metrics(context);
  return result;
}

void add_native_owner_bytes(std::uint64_t* total, std::size_t bytes) {
  if (total == nullptr ||
      bytes > std::numeric_limits<std::uint64_t>::max() - *total) {
    throw std::overflow_error("dense groupby native owner byte count overflows");
  }
  *total += static_cast<std::uint64_t>(bytes);
}

jlongArray native_owner_stats(JNIEnv* env,
                              std::uint64_t active_bytes,
                              std::uint64_t active_references,
                              std::uint64_t active_batches) {
  if (active_bytes == 0 || active_references == 0 ||
      active_batches > active_references ||
      active_bytes > static_cast<std::uint64_t>(
                         std::numeric_limits<jlong>::max()) ||
      active_references > static_cast<std::uint64_t>(
                              std::numeric_limits<jlong>::max()) ||
      active_batches > static_cast<std::uint64_t>(
                           std::numeric_limits<jlong>::max())) {
    throw std::overflow_error("dense groupby native owner stats are invalid");
  }
  const jlong values[3] = {static_cast<jlong>(active_bytes),
                           static_cast<jlong>(active_references),
                           static_cast<jlong>(active_batches)};
  jlongArray result = env->NewLongArray(3);
  if (result != nullptr) env->SetLongArrayRegion(result, 0, 3, values);
  return result;
}

jlongArray context_native_owner_stats(JNIEnv* env, Context* context) {
  if (context == nullptr) {
    throw std::invalid_argument("dense groupby context is null");
  }
  std::uint64_t active_bytes = 0;
  add_native_owner_bytes(&active_bytes, context->device_storage_bytes);
  if (context->device_calc_input != nullptr) {
    add_native_owner_bytes(
        &active_bytes,
        static_cast<std::size_t>(context->capacity) *
            context->calc_input_row_bytes);
  }
  if (context->device_calc_program != nullptr) {
    add_native_owner_bytes(
        &active_bytes,
        static_cast<std::size_t>(context->calc_instruction_count) *
            sizeof(CalcInstruction));
  }
  std::uint64_t active_references = 1;
  std::uint64_t active_batches = 0;
  if (context->device_batch_pool != 0) {
    CudfFlinkGpuDeviceBatchPoolStats stats{};
    stats.struct_size = sizeof(stats);
    stats.abi_version = CUDF_FLINK_GPU_DEVICE_BATCH_ABI_VERSION;
    require_device_batch(
        "read dense groupby native owner stats",
        cudf_flink_gpu_device_batch_pool_stats(
            context->device_batch_pool, &stats));
    add_native_owner_bytes(&active_bytes, stats.allocated_bytes);
    active_references += stats.active_references;
    active_batches = stats.active_batches;
  }
  return native_owner_stats(
      env, active_bytes, active_references, active_batches);
}

jlongArray flexible_native_owner_stats(JNIEnv* env,
                                       FlexibleContext* context) {
  if (context == nullptr) {
    throw std::invalid_argument("flexible dense groupby context is null");
  }
  return native_owner_stats(env, context->device_storage_bytes, 1, 0);
}

struct alignas(64) PrimitiveLongAggregateHeader {
  std::uint64_t h2d_nanos;
  std::uint64_t compute_nanos;
  std::uint64_t d2h_nanos;
  std::uint64_t input_bytes;
  std::uint64_t output_bytes;
  std::int32_t rows;
  std::int32_t groups;
  std::int32_t row_capacity;
  std::int32_t group_capacity;
  std::int32_t reserved[2];
};

struct PrimitiveLongAggregateError {
  std::int32_t code;
  std::int32_t row;
};

static_assert(sizeof(PrimitiveLongAggregateHeader) == 64,
              "primitive BIGINT aggregate header ABI changed");

struct PrimitiveLongAggregateContext {
  int device = -1;
  int multiprocessors = 0;
  int row_capacity = 0;
  int group_capacity = 0;
  std::int64_t* host_keys = nullptr;
  std::int64_t* host_values = nullptr;
  std::uint8_t* host_validity = nullptr;
  std::uint8_t* host_output = nullptr;
  PrimitiveLongAggregateHeader* host_header = nullptr;
  PrimitiveLongAggregateError* host_error = nullptr;
  std::int64_t* device_keys = nullptr;
  std::int64_t* device_values = nullptr;
  std::uint8_t* device_validity = nullptr;
  std::uint8_t* device_output = nullptr;
  std::uint32_t* device_row_counts = nullptr;
  PrimitiveLongAggregateError* device_error = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t after_h2d = nullptr;
  cudaEvent_t after_compute = nullptr;
  cudaEvent_t after_d2h = nullptr;
};

std::size_t primitive_validity_bytes(int values) {
  return static_cast<std::size_t>(values / 8 + (values % 8 == 0 ? 0 : 1));
}

std::size_t primitive_padded_validity_bytes(int values) {
  return static_cast<std::size_t>((values + 31) / 32) * sizeof(std::uint32_t);
}

std::size_t primitive_output_capacity_bytes(int groups) {
  return static_cast<std::size_t>(groups) * 6u * sizeof(std::int64_t) +
         primitive_padded_validity_bytes(groups) * 4u;
}

std::uint64_t primitive_elapsed_nanos(cudaEvent_t start, cudaEvent_t end) {
  float milliseconds = 0.0f;
  require_cuda("time primitive BIGINT aggregate",
               cudaEventElapsedTime(&milliseconds, start, end));
  return static_cast<std::uint64_t>(
      std::llround(milliseconds * 1'000'000.0));
}

void destroy_primitive_long_aggregate(PrimitiveLongAggregateContext* context) {
  if (context == nullptr) return;
  if (context->device >= 0) {
    require_cuda("bind primitive BIGINT aggregate teardown device",
                 cudaSetDevice(context->device));
  }
  synchronize_stream_for_teardown(
      "synchronize primitive BIGINT aggregate stream", context->stream);
  destroy_event_for_teardown(
      "destroy primitive BIGINT aggregate start event", &context->start);
  destroy_event_for_teardown(
      "destroy primitive BIGINT aggregate H2D event", &context->after_h2d);
  destroy_event_for_teardown(
      "destroy primitive BIGINT aggregate compute event", &context->after_compute);
  destroy_event_for_teardown(
      "destroy primitive BIGINT aggregate D2H event", &context->after_d2h);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device keys", &context->device_keys);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device values", &context->device_values);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device validity", &context->device_validity);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device output", &context->device_output);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device row counts",
      &context->device_row_counts);
  free_device_for_teardown(
      "free primitive BIGINT aggregate device error", &context->device_error);
  destroy_stream_for_teardown(
      "destroy primitive BIGINT aggregate stream", &context->stream);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host keys", &context->host_keys);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host values", &context->host_values);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host validity", &context->host_validity);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host output", &context->host_output);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host header", &context->host_header);
  free_host_for_teardown(
      "free primitive BIGINT aggregate host error", &context->host_error);
  delete context;
}

[[noreturn]] void handle_failed_primitive_long_aggregate_construction(
    PrimitiveLongAggregateContext* context,
    const std::exception_ptr& construction_failure) {
  try {
    destroy_primitive_long_aggregate(context);
  } catch (...) {
    retain_construction_orphan(
        ConstructionOrphanKind::kPrimitiveLong,
        context,
        construction_failure,
        std::current_exception());
  }
  std::rethrow_exception(construction_failure);
}

PrimitiveLongAggregateContext* primitive_long_aggregate_from(jlong handle) {
  if (handle == 0) {
    throw std::invalid_argument("primitive BIGINT aggregate handle is closed");
  }
  return reinterpret_cast<PrimitiveLongAggregateContext*>(
      static_cast<std::uintptr_t>(handle));
}

__device__ bool primitive_bitmap_valid(const std::uint8_t* bitmap, int row) {
  return (bitmap[row >> 3] & (1u << (row & 7))) != 0u;
}

__device__ void record_primitive_long_aggregate_error(
    PrimitiveLongAggregateError* error, int code, int row) {
  if (atomicCAS(&error->code, 0, code) == 0) error->row = row;
}

__global__ void initialize_primitive_long_min_max(
    std::int64_t* output, int groups) {
  std::int64_t* minimum = output + static_cast<std::size_t>(groups) * 4u;
  std::int64_t* maximum = output + static_cast<std::size_t>(groups) * 5u;
  for (int group = blockIdx.x * blockDim.x + threadIdx.x;
       group < groups;
       group += blockDim.x * gridDim.x) {
    minimum[group] = LLONG_MAX;
    maximum[group] = LLONG_MIN;
  }
}

__global__ void aggregate_primitive_dense_long(
    const std::int64_t* keys,
    const std::int64_t* values,
    const std::uint8_t* validity,
    int rows,
    int groups,
    std::uint32_t* row_counts,
    std::int64_t* output,
    PrimitiveLongAggregateError* error) {
  auto* counts = reinterpret_cast<unsigned long long*>(
      output + static_cast<std::size_t>(groups));
  std::int64_t* sums = output + static_cast<std::size_t>(groups) * 2u;
  std::int64_t* minimum = output + static_cast<std::size_t>(groups) * 4u;
  std::int64_t* maximum = output + static_cast<std::size_t>(groups) * 5u;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows;
       row += blockDim.x * gridDim.x) {
    const std::int64_t key = keys[row];
    if (key < 0 || key >= groups) {
      record_primitive_long_aggregate_error(error, 1, row);
      continue;
    }
    const int group = static_cast<int>(key);
    atomicAdd(row_counts + group, 1u);
    if (!primitive_bitmap_valid(validity, row)) continue;
    const std::int64_t value = values[row];
    atomicAdd(counts + group, 1ULL);
    atomic_add_i64(
        reinterpret_cast<long long*>(sums) + group,
        static_cast<long long>(value));
    atomic_min_i64(
        reinterpret_cast<long long*>(minimum) + group,
        static_cast<long long>(value));
    atomic_max_i64(
        reinterpret_cast<long long*>(maximum) + group,
        static_cast<long long>(value));
  }
}

__global__ void finalize_primitive_dense_long(
    std::uint32_t* row_counts,
    std::int64_t* output,
    int groups,
    PrimitiveLongAggregateError* error) {
  std::int64_t* keys = output;
  auto* counts = reinterpret_cast<unsigned long long*>(
      output + static_cast<std::size_t>(groups));
  std::int64_t* sums = output + static_cast<std::size_t>(groups) * 2u;
  std::int64_t* averages = output + static_cast<std::size_t>(groups) * 3u;
  std::int64_t* minimum = output + static_cast<std::size_t>(groups) * 4u;
  std::int64_t* maximum = output + static_cast<std::size_t>(groups) * 5u;
  const int validity_words = (groups + 31) / 32;
  auto* output_validity = reinterpret_cast<std::uint32_t*>(output +
      static_cast<std::size_t>(groups) * 6u);
  for (int group = blockIdx.x * blockDim.x + threadIdx.x;
       group < groups;
       group += blockDim.x * gridDim.x) {
    if (row_counts[group] == 0u) {
      record_primitive_long_aggregate_error(error, 2, group);
    }
    keys[group] = group;
    const auto count = counts[group];
    averages[group] = count == 0u ? 0 : sums[group] / static_cast<std::int64_t>(count);
    if (count == 0u) {
      sums[group] = 0;
      minimum[group] = 0;
      maximum[group] = 0;
    }
    if ((group & 31) == 0) {
      std::uint32_t bits = 0u;
      const int end = min(groups, group + 32);
      for (int candidate = group; candidate < end; candidate++) {
        if (counts[candidate] != 0u) bits |= 1u << (candidate - group);
      }
      const int word = group >> 5;
      for (int column = 0; column < 4; column++) {
        output_validity[column * validity_words + word] = bits;
      }
    }
  }
}

PrimitiveLongAggregateContext* create_primitive_long_aggregate(
    int row_capacity, int group_capacity) {
  if (row_capacity <= 0 || group_capacity <= 0 || group_capacity > row_capacity) {
    throw std::invalid_argument(
        "primitive BIGINT aggregate capacities are invalid");
  }
  auto* context = new PrimitiveLongAggregateContext();
  try {
    context->row_capacity = row_capacity;
    context->group_capacity = group_capacity;
    require_cuda("capture primitive BIGINT aggregate device",
                 cudaGetDevice(&context->device));
    cudaDeviceProp properties{};
    require_cuda("read primitive BIGINT aggregate device",
                 cudaGetDeviceProperties(&properties, context->device));
    if (!mega_flink_device_matches_native_build(properties)) {
      throw std::runtime_error(
          std::string("primitive BIGINT aggregate build does not admit selected device: ") +
          properties.name);
    }
    context->multiprocessors = properties.multiProcessorCount;
    const std::size_t row_bytes =
        static_cast<std::size_t>(row_capacity) * sizeof(std::int64_t);
    const std::size_t input_validity_bytes =
        primitive_validity_bytes(row_capacity);
    const std::size_t output_bytes =
        primitive_output_capacity_bytes(group_capacity);
    require_cuda("allocate primitive BIGINT aggregate host keys",
                 cudaHostAlloc(&context->host_keys, row_bytes, cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate host values",
                 cudaHostAlloc(&context->host_values, row_bytes, cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate host validity",
                 cudaHostAlloc(&context->host_validity,
                               input_validity_bytes,
                               cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate host output",
                 cudaHostAlloc(&context->host_output, output_bytes, cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate host header",
                 cudaHostAlloc(&context->host_header,
                               sizeof(PrimitiveLongAggregateHeader),
                               cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate host error",
                 cudaHostAlloc(&context->host_error,
                               sizeof(PrimitiveLongAggregateError),
                               cudaHostAllocDefault));
    require_cuda("allocate primitive BIGINT aggregate device keys",
                 cudaMalloc(&context->device_keys, row_bytes));
    require_cuda("allocate primitive BIGINT aggregate device values",
                 cudaMalloc(&context->device_values, row_bytes));
    require_cuda("allocate primitive BIGINT aggregate device validity",
                 cudaMalloc(&context->device_validity, input_validity_bytes));
    require_cuda("allocate primitive BIGINT aggregate device output",
                 cudaMalloc(&context->device_output, output_bytes));
    require_cuda("allocate primitive BIGINT aggregate device row counts",
                 cudaMalloc(&context->device_row_counts,
                            static_cast<std::size_t>(group_capacity) *
                                sizeof(std::uint32_t)));
    require_cuda("allocate primitive BIGINT aggregate device error",
                 cudaMalloc(&context->device_error,
                            sizeof(PrimitiveLongAggregateError)));
    require_cuda("create primitive BIGINT aggregate stream",
                 cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking));
    require_cuda("create primitive BIGINT aggregate start event",
                 cudaEventCreate(&context->start));
    require_cuda("create primitive BIGINT aggregate H2D event",
                 cudaEventCreate(&context->after_h2d));
    require_cuda("create primitive BIGINT aggregate compute event",
                 cudaEventCreate(&context->after_compute));
    require_cuda("create primitive BIGINT aggregate D2H event",
                 cudaEventCreate(&context->after_d2h));
    std::memset(context->host_header, 0, sizeof(*context->host_header));
    return context;
  } catch (...) {
    handle_failed_primitive_long_aggregate_construction(
        context, std::current_exception());
  }
}

void execute_primitive_long_aggregate(
    PrimitiveLongAggregateContext* context, int rows, int groups) {
  if (context == nullptr || rows <= 0 || rows > context->row_capacity ||
      groups <= 0 || groups > context->group_capacity || groups > rows) {
    throw std::invalid_argument("primitive BIGINT aggregate execution shape is invalid");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(rows) * sizeof(std::int64_t);
  const std::size_t input_validity_bytes = primitive_validity_bytes(rows);
  const std::size_t output_bytes =
      static_cast<std::size_t>(groups) * 6u * sizeof(std::int64_t) +
      primitive_padded_validity_bytes(groups) * 4u;
  const int row_blocks = std::max(
      1, std::min((rows + kThreads - 1) / kThreads,
                  context->multiprocessors * 4));
  const int group_blocks = std::max(
      1, std::min((groups + kThreads - 1) / kThreads,
                  context->multiprocessors * 4));
  auto* device_output =
      reinterpret_cast<std::int64_t*>(context->device_output);

  require_cuda("record primitive BIGINT aggregate start",
               cudaEventRecord(context->start, context->stream));
  require_cuda("copy primitive BIGINT aggregate keys",
               cudaMemcpyAsync(context->device_keys,
                               context->host_keys,
                               row_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("copy primitive BIGINT aggregate values",
               cudaMemcpyAsync(context->device_values,
                               context->host_values,
                               row_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("copy primitive BIGINT aggregate validity",
               cudaMemcpyAsync(context->device_validity,
                               context->host_validity,
                               input_validity_bytes,
                               cudaMemcpyHostToDevice,
                               context->stream));
  require_cuda("record primitive BIGINT aggregate H2D",
               cudaEventRecord(context->after_h2d, context->stream));

  require_cuda("clear primitive BIGINT aggregate output",
               cudaMemsetAsync(
                   context->device_output, 0, output_bytes, context->stream));
  require_cuda("clear primitive BIGINT aggregate row counts",
               cudaMemsetAsync(context->device_row_counts,
                               0,
                               static_cast<std::size_t>(groups) *
                                   sizeof(std::uint32_t),
                               context->stream));
  require_cuda("clear primitive BIGINT aggregate error",
               cudaMemsetAsync(context->device_error,
                               0,
                               sizeof(PrimitiveLongAggregateError),
                               context->stream));
  initialize_primitive_long_min_max<<<
      group_blocks, kThreads, 0, context->stream>>>(device_output, groups);
  require_cuda("initialize primitive BIGINT aggregate min/max",
               cudaGetLastError());
  aggregate_primitive_dense_long<<<
      row_blocks, kThreads, 0, context->stream>>>(context->device_keys,
                                                 context->device_values,
                                                 context->device_validity,
                                                 rows,
                                                 groups,
                                                 context->device_row_counts,
                                                 device_output,
                                                 context->device_error);
  require_cuda("launch primitive BIGINT aggregate", cudaGetLastError());
  finalize_primitive_dense_long<<<
      group_blocks, kThreads, 0, context->stream>>>(context->device_row_counts,
                                                   device_output,
                                                   groups,
                                                   context->device_error);
  require_cuda("finalize primitive BIGINT aggregate", cudaGetLastError());
  require_cuda("record primitive BIGINT aggregate compute",
               cudaEventRecord(context->after_compute, context->stream));

  require_cuda("copy primitive BIGINT aggregate output",
               cudaMemcpyAsync(context->host_output,
                               context->device_output,
                               output_bytes,
                               cudaMemcpyDeviceToHost,
                               context->stream));
  require_cuda("copy primitive BIGINT aggregate error",
               cudaMemcpyAsync(context->host_error,
                               context->device_error,
                               sizeof(PrimitiveLongAggregateError),
                               cudaMemcpyDeviceToHost,
                               context->stream));
  require_cuda("record primitive BIGINT aggregate D2H",
               cudaEventRecord(context->after_d2h, context->stream));
  require_cuda("wait primitive BIGINT aggregate",
               cudaEventSynchronize(context->after_d2h));
  if (context->host_error->code != 0) {
    throw std::invalid_argument(
        "primitive BIGINT aggregate rejected dense key domain: code=" +
        std::to_string(context->host_error->code) + ", row=" +
        std::to_string(context->host_error->row));
  }

  context->host_header->h2d_nanos =
      primitive_elapsed_nanos(context->start, context->after_h2d);
  context->host_header->compute_nanos =
      primitive_elapsed_nanos(context->after_h2d, context->after_compute);
  context->host_header->d2h_nanos =
      primitive_elapsed_nanos(context->after_compute, context->after_d2h);
  context->host_header->input_bytes =
      row_bytes * 2u + input_validity_bytes;
  context->host_header->output_bytes = output_bytes;
  context->host_header->rows = rows;
  context->host_header->groups = groups;
  context->host_header->row_capacity = context->row_capacity;
  context->host_header->group_capacity = context->group_capacity;
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeCreate(
    JNIEnv* env, jclass, jint row_capacity, jint group_capacity) {
  try {
    std::lock_guard<std::mutex> construction_lock(
        construction_orphan_mutex);
    retry_construction_orphan();
    return reinterpret_cast<jlong>(
        create_primitive_long_aggregate(row_capacity, group_capacity));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return 0;
  } catch (...) {
    throw_java(env, "unknown primitive BIGINT aggregate construction failure");
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeKeysBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    auto* context = primitive_long_aggregate_from(handle);
    return env->NewDirectByteBuffer(
        context->host_keys,
        static_cast<jlong>(context->row_capacity) * sizeof(std::int64_t));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeValuesBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    auto* context = primitive_long_aggregate_from(handle);
    return env->NewDirectByteBuffer(
        context->host_values,
        static_cast<jlong>(context->row_capacity) * sizeof(std::int64_t));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeValidityBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    auto* context = primitive_long_aggregate_from(handle);
    return env->NewDirectByteBuffer(
        context->host_validity,
        primitive_validity_bytes(context->row_capacity));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeOutputBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    auto* context = primitive_long_aggregate_from(handle);
    return env->NewDirectByteBuffer(
        context->host_output,
        primitive_output_capacity_bytes(context->group_capacity));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeHeaderBuffer(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return env->NewDirectByteBuffer(
        primitive_long_aggregate_from(handle)->host_header,
        sizeof(PrimitiveLongAggregateHeader));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeExecute(
    JNIEnv* env, jclass, jlong handle, jint rows, jint groups) {
  try {
    execute_primitive_long_aggregate(
        primitive_long_aggregate_from(handle), rows, groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_GpuDenseLongAggregateKernel_nativeClose(
    JNIEnv* env, jclass, jlong handle) {
  try {
    destroy_primitive_long_aggregate(primitive_long_aggregate_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  } catch (...) {
    throw_java(env, "unknown primitive BIGINT aggregate teardown failure");
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeCreate(
    JNIEnv* env,
    jclass,
    jint capacity,
    jint operation_mask,
    jint metric_filter_lane,
    jint int_filter_lane,
    jint long0_filter_lane,
    jint long1_filter_lane) {
  try {
    std::lock_guard<std::mutex> construction_lock(
        construction_orphan_mutex);
    retry_construction_orphan();
    return static_cast<jlong>(
        reinterpret_cast<std::uintptr_t>(
            create_context(capacity,
                           static_cast<std::uint32_t>(operation_mask),
                           metric_filter_lane,
                           int_filter_lane,
                           long0_filter_lane,
                           long1_filter_lane)));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return 0;
  } catch (...) {
    throw_java(env, "unknown dense groupby construction failure");
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeInputBuffer(
    JNIEnv* env, jclass, jlong handle) {
  Context* context = context_from(handle);
  if (context == nullptr) {
    throw_java(env, "dense groupby context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_input,
      static_cast<jlong>(context->capacity) * sizeof(PackedInputRow));
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeExpandedGroupBuffer(
    JNIEnv* env, jclass, jlong handle) {
  Context* context = context_from(handle);
  if (context == nullptr) {
    throw_java(env, "expanded dense groupby context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_expanded_groups,
      static_cast<jlong>(context->capacity) * sizeof(int));
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentKick(
    JNIEnv* env, jclass, jlong handle) {
  try {
    kick_persistent_aggregate(persistent_aggregate_context_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeOutputBuffer(
    JNIEnv* env, jclass, jlong handle) {
  Context* context = context_from(handle);
  if (context == nullptr) {
    throw_java(env, "dense groupby context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_output,
      kOutputHeaderBytes +
          static_cast<jlong>(context->capacity + 1) * sizeof(PackedAggregateRow));
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeResourceStats(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return context_native_owner_stats(env, context_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeConfigureCalc(
    JNIEnv* env,
    jclass,
    jlong handle,
    jint input_count,
    jint output_count,
    jobject program_buffer,
    jint instruction_count,
    jint max_stack,
    jint metric_output,
    jint int_output,
    jint long0_output,
    jint long1_output,
    jint filter0_output,
    jint filter1_output,
    jint filter2_output,
    jboolean only_counts) {
  try {
    void* program = env->GetDirectBufferAddress(program_buffer);
    const jlong program_capacity =
        env->GetDirectBufferCapacity(program_buffer);
    const std::size_t required =
        static_cast<std::size_t>(instruction_count) *
        sizeof(CalcInstruction);
    if (program == nullptr || program_capacity < static_cast<jlong>(required)) {
      throw std::invalid_argument(
          "fused Calc groupby bytecode must be a direct buffer");
    }
    Context* context = context_from(handle);
    configure_calc_context(
        context,
        input_count,
        output_count,
        static_cast<const CalcInstruction*>(program),
        instruction_count,
        max_stack,
        metric_output,
        int_output,
        long0_output,
        long1_output,
        filter0_output,
        filter1_output,
        filter2_output,
        only_counts == JNI_TRUE);
    return env->NewDirectByteBuffer(
        context->host_calc_input,
        static_cast<jlong>(context->capacity) *
            static_cast<jlong>(context->calc_input_row_bytes));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeConfigureCalcDeviceBatch(
    JNIEnv* env, jclass, jlong handle, jlong owner_chain_id) {
  try {
    configure_calc_device_batch_context(
        context_from(handle),
        static_cast<std::uint64_t>(owner_chain_id));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeExecute(
    JNIEnv* env, jclass, jlong handle, jint rows, jint groups) {
  try {
    execute(context_from(handle), rows, groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeExecuteCalc(
    JNIEnv* env, jclass, jlong handle, jint rows, jint groups) {
  try {
    execute_calc(context_from(handle), rows, groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeProduceCalcDeviceBatch(
    JNIEnv* env, jclass, jlong handle, jint rows, jint groups) {
  try {
    return static_cast<jlong>(produce_calc_device_batch(
        context_from(handle), rows, groups));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return 0;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeConsumeCalcDeviceBatch(
    JNIEnv* env,
    jclass,
    jlong handle,
    jlong batch_handle,
    jint rows,
    jint groups) {
  try {
    consume_calc_device_batch(
        context_from(handle),
        static_cast<std::uint64_t>(batch_handle),
        rows,
        groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeTakeCalcDeviceBatchMetrics(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return take_device_batch_metrics(env, context_from(handle), false, false);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeDrainCalcDeviceBatch(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return take_device_batch_metrics(env, context_from(handle), true, false);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeForceDrainCalcDeviceBatch(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return take_device_batch_metrics(env, context_from(handle), true, true);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeExecuteExpanded(
    JNIEnv* env,
    jclass,
    jlong handle,
    jint raw_rows,
    jint project_count,
    jint groups) {
  try {
    execute_expanded(
        context_from(handle), raw_rows, project_count, groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeClose(
    JNIEnv* env, jclass, jlong handle) {
  try {
    destroy_context(context_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  } catch (...) {
    throw_java(env, "unknown dense groupby teardown failure");
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeCreateFlexible(
    JNIEnv* env,
    jclass,
    jint capacity,
    jintArray kinds,
    jintArray value_lanes,
    jintArray filter_lanes) {
  try {
    std::lock_guard<std::mutex> construction_lock(
        construction_orphan_mutex);
    retry_construction_orphan();
    if (kinds == nullptr || value_lanes == nullptr || filter_lanes == nullptr) {
      throw std::invalid_argument("flexible aggregate program arrays are null");
    }
    const jsize operation_count = env->GetArrayLength(kinds);
    if (env->GetArrayLength(value_lanes) != operation_count ||
        env->GetArrayLength(filter_lanes) != operation_count) {
      throw std::invalid_argument(
          "flexible aggregate program arrays have different lengths");
    }
    std::vector<jint> host_kinds(operation_count);
    std::vector<jint> host_value_lanes(operation_count);
    std::vector<jint> host_filter_lanes(operation_count);
    env->GetIntArrayRegion(kinds, 0, operation_count, host_kinds.data());
    env->GetIntArrayRegion(
        value_lanes, 0, operation_count, host_value_lanes.data());
    env->GetIntArrayRegion(
        filter_lanes, 0, operation_count, host_filter_lanes.data());
    if (env->ExceptionCheck()) {
      return 0;
    }
    std::vector<FlexibleProgramOp> program(operation_count);
    for (jsize index = 0; index < operation_count; index++) {
      program[index] = {host_kinds[index],
                        host_value_lanes[index],
                        host_filter_lanes[index],
                        0};
    }
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(
        create_flexible_context(capacity, program.data(), operation_count)));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return 0;
  } catch (...) {
    throw_java(env, "unknown flexible dense groupby construction failure");
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeFlexibleInputBuffer(
    JNIEnv* env, jclass, jlong handle) {
  FlexibleContext* context = flexible_context_from(handle);
  if (context == nullptr) {
    throw_java(env, "flexible dense groupby context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_input,
      static_cast<jlong>(context->capacity) * sizeof(PackedInputRow));
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeFlexibleOutputBuffer(
    JNIEnv* env, jclass, jlong handle) {
  FlexibleContext* context = flexible_context_from(handle);
  if (context == nullptr) {
    throw_java(env, "flexible dense groupby context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_output,
      kOutputHeaderBytes + static_cast<jlong>(context->capacity) *
                               context->result_stride * sizeof(FlexibleResult));
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeFlexibleResourceStats(
    JNIEnv* env, jclass, jlong handle) {
  try {
    return flexible_native_owner_stats(env, flexible_context_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return nullptr;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeExecuteFlexible(
    JNIEnv* env, jclass, jlong handle, jint rows, jint groups) {
  try {
    execute_flexible(flexible_context_from(handle), rows, groups);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_DenseGroupbyKernel_nativeCloseFlexible(
    JNIEnv* env, jclass, jlong handle) {
  try {
    destroy_flexible_context(flexible_context_from(handle));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  } catch (...) {
    throw_java(env, "unknown flexible dense groupby teardown failure");
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentCreate(
    JNIEnv* env,
    jclass,
    jint capacity,
    jint operation_mask,
    jint metric_filter_lane,
    jint int_filter_lane,
    jint long0_filter_lane,
    jint long1_filter_lane) {
  try {
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(
        create_persistent_aggregate(
            capacity,
            static_cast<std::uint32_t>(operation_mask),
            metric_filter_lane,
            int_filter_lane,
            long0_filter_lane,
            long1_filter_lane)));
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentBuffer(
    JNIEnv* env, jclass, jlong handle) {
  PersistentAggregateContext* context =
      persistent_aggregate_context_from(handle);
  if (context == nullptr) {
    throw_java(env, "persistent Aggregate context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_storage,
      static_cast<jlong>(sizeof(PersistentAggregateStorage)));
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentSnapshotBuffer(
    JNIEnv* env, jclass, jlong handle) {
  PersistentAggregateContext* context =
      persistent_aggregate_context_from(handle);
  if (context == nullptr) {
    throw_java(env, "persistent Aggregate context is null");
    return nullptr;
  }
  return env->NewDirectByteBuffer(
      context->host_snapshot,
      static_cast<jlong>(context->capacity) * sizeof(PackedAggregateRow));
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentSnapshot(
    JNIEnv* env, jclass, jlong handle, jintArray groups) {
  try {
    if (groups == nullptr) {
      throw std::invalid_argument(
          "persistent Aggregate snapshot groups are null");
    }
    const jsize group_count = env->GetArrayLength(groups);
    std::vector<jint> host_groups(group_count);
    env->GetIntArrayRegion(groups, 0, group_count, host_groups.data());
    if (env->ExceptionCheck()) return;
    snapshot_persistent_aggregate(
        persistent_aggregate_context_from(handle),
        host_groups.data(),
        group_count);
  } catch (const std::exception& failure) {
    throw_java(env, failure.what());
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_fep_flink_PersistentAggregateKernel_nativePersistentClose(
    JNIEnv*, jclass, jlong handle) {
  destroy_persistent_aggregate(persistent_aggregate_context_from(handle));
}
