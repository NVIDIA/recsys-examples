#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kvcache_manager {

constexpr std::array<char, 4> kFlexKVAOTIMagic{{'F', 'K', 'A', 'O'}};
constexpr uint8_t kFlexKVAOTIVersion = 1;

enum class FlexKVOpcode : uint8_t {
    RegisterDPClient = 1,
    IsReady = 2,
    Put = 3,
    Get = 4,
    GetMatch = 5,
    LaunchTasks = 6,
    CancelTask = 7,
    Wait = 8,
    TryWait = 9,
    Shutdown = 10,
    Start = 11,
    RegisterTPClient = 12,
    IsReadyResponse = 101,
    GetMatchResponse = 102,
    WaitResponse = 103,
    TryWaitResponse = 104,
    ErrorResponse = 127,
};

enum class KVResponseStatus : uint8_t {
    SUCCESS = 0,
    NOTFOUND = 1,
    UNREADY = 2,
    TIMEOUT = 3,
    CANCELLED = 4,
    FAILED = 5,
};

struct KVTaskResponse {
    int64_t task_id;
    KVResponseStatus status;
};

enum class FlexKVLayoutType : int32_t {
    LayerFirst = 0,
    BlockFirst = 1,
};

struct FlexKVCacheLayoutSpec {
    FlexKVLayoutType type;
    int32_t num_layer;
    int32_t num_block;
    int32_t tokens_per_block;
    int32_t num_head;
    int32_t head_size;
    bool is_mla;
};

class FlexKVBufferWriter {
public:
    void writeHeader(FlexKVOpcode opcode);
    void writeBytes(const void* data, size_t size);
    void writeU8(uint8_t value);
    void writeI32(int32_t value);
    void writeI64(int64_t value);
    void writeF64(double value);
    void writeBool(bool value);
    void writeString(const std::string& value);
    void writeStringList(const std::vector<std::string>& values);
    void writeInt64Tensor(const at::Tensor& value);
    void writeBoolTensor(const at::Tensor& value);
    void writeOptionalBoolTensor(const std::optional<at::Tensor>& value);
    void writeInt64TensorList(const std::vector<at::Tensor>& values);
    std::vector<uint8_t> finish() &&;

private:
    std::vector<uint8_t> buffer_;
    void append(const void* data, size_t size);
};

class FlexKVBufferReader {
public:
    explicit FlexKVBufferReader(const std::vector<uint8_t>& buffer);

    FlexKVOpcode readHeader();
    uint8_t readU8();
    int32_t readI32();
    int64_t readI64();
    double readF64();
    bool readBool();
    std::string readString();
    std::vector<std::string> readStringList();
    at::Tensor readInt64Tensor();
    at::Tensor readBoolTensor();
    std::optional<at::Tensor> readOptionalBoolTensor();
    std::vector<at::Tensor> readInt64TensorList();

private:
    const std::vector<uint8_t>& buffer_;
    size_t offset_;
    const uint8_t* consume(size_t size);
};

std::vector<uint8_t> encodeRegisterDPClientRequest(
    int32_t dpClientId,
    const std::string& clientRecvPort,
    int32_t tpSize);

std::vector<uint8_t> encodeStartRequest(int32_t dpClientId);

std::vector<uint8_t> encodeIsReadyRequest(int32_t dpClientId);

std::vector<uint8_t> encodePutRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const at::Tensor& slotMapping,
    const std::optional<at::Tensor>& tokenMask,
    const std::vector<std::string>& nameSpace);

std::vector<uint8_t> encodeGetRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const at::Tensor& slotMapping,
    const std::optional<at::Tensor>& tokenMask,
    int32_t layerGranularity,
    const std::vector<std::string>& nameSpace);

std::vector<uint8_t> encodeGetMatchRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const std::optional<at::Tensor>& tokenMask,
    int32_t layerGranularity,
    const std::vector<std::string>& nameSpace);

std::vector<uint8_t> encodeLaunchTasksRequest(
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds,
    const std::vector<at::Tensor>& slotMappings,
    bool asBatch,
    int64_t batchId);

std::vector<uint8_t> encodeCancelTaskRequest(
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds);

std::vector<uint8_t> encodeWaitRequest(
    FlexKVOpcode opcode,
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds,
    double timeoutSeconds,
    bool completely);

std::vector<uint8_t> encodeShutdownRequest(int32_t dpClientId);

std::vector<uint8_t> encodeRegisterTPClientRequest(
    int32_t dpClientId,
    int32_t deviceId,
    const std::vector<at::Tensor>& kvCaches,
    const FlexKVCacheLayoutSpec& layout);

bool decodeIsReadyResponse(const std::vector<uint8_t>& buffer);
std::pair<int64_t, at::Tensor> decodeGetMatchResponse(const std::vector<uint8_t>& buffer);
std::unordered_map<int64_t, KVTaskResponse> decodeWaitResponse(
    const std::vector<uint8_t>& buffer,
    FlexKVOpcode expectedOpcode);
std::string decodeErrorResponse(const std::vector<uint8_t>& buffer);

} // namespace kvcache_manager