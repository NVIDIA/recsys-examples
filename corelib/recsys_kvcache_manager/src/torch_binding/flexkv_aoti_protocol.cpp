#include "flexkv_aoti_protocol.h"

#include <c10/util/Exception.h>

#include <cuda_runtime_api.h>

#include <cstring>

namespace kvcache_manager {

namespace {

template <typename T>
T readScalar(const uint8_t* data) {
    T value;
    std::memcpy(&value, data, sizeof(T));
    return value;
}

at::Tensor toCpuContiguous(const at::Tensor& tensor, at::ScalarType dtype) {
    return tensor.to(at::TensorOptions().device(at::kCPU).dtype(dtype)).contiguous();
}

int32_t scalarTypeToProtocolCode(at::ScalarType scalarType) {
    switch (scalarType) {
        case at::kFloat:
            return 0;
        case at::kHalf:
            return 1;
        case at::kBFloat16:
            return 2;
        case at::kLong:
            return 3;
        case at::kInt:
            return 4;
        case at::kShort:
            return 5;
        case at::kChar:
            return 6;
        case at::kByte:
            return 7;
        case at::kBool:
            return 8;
        default:
            TORCH_CHECK(false, "Unsupported FlexKV tensor dtype for IPC export");
    }
}

void writeTensorHandle(FlexKVBufferWriter& writer, const at::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda(), "FlexKV TP registration expects CUDA tensors");
    cudaIpcMemHandle_t handle;
    const auto result = cudaIpcGetMemHandle(&handle, tensor.data_ptr());
    TORCH_CHECK(result == cudaSuccess, "cudaIpcGetMemHandle failed with code ", result);
    writer.writeI32(static_cast<int32_t>(sizeof(handle)));
    writer.writeBytes(&handle, sizeof(handle));
    writer.writeI32(static_cast<int32_t>(tensor.dim()));
    for (const auto dim : tensor.sizes()) {
        writer.writeI64(dim);
    }
    writer.writeI32(scalarTypeToProtocolCode(tensor.scalar_type()));
    writer.writeI64(0);
}

} // namespace

void FlexKVBufferWriter::append(const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
}

void FlexKVBufferWriter::writeBytes(const void* data, size_t size) { append(data, size); }

void FlexKVBufferWriter::writeHeader(FlexKVOpcode opcode) {
    append(kFlexKVAOTIMagic.data(), kFlexKVAOTIMagic.size());
    writeU8(kFlexKVAOTIVersion);
    writeU8(static_cast<uint8_t>(opcode));
}

void FlexKVBufferWriter::writeU8(uint8_t value) { append(&value, sizeof(value)); }
void FlexKVBufferWriter::writeI32(int32_t value) { append(&value, sizeof(value)); }
void FlexKVBufferWriter::writeI64(int64_t value) { append(&value, sizeof(value)); }
void FlexKVBufferWriter::writeF64(double value) { append(&value, sizeof(value)); }
void FlexKVBufferWriter::writeBool(bool value) { writeU8(value ? 1 : 0); }

void FlexKVBufferWriter::writeString(const std::string& value) {
    writeI32(static_cast<int32_t>(value.size()));
    append(value.data(), value.size());
}

void FlexKVBufferWriter::writeStringList(const std::vector<std::string>& values) {
    writeI32(static_cast<int32_t>(values.size()));
    for (const auto& value : values) {
        writeString(value);
    }
}

void FlexKVBufferWriter::writeInt64Tensor(const at::Tensor& value) {
    const auto flat = toCpuContiguous(value, at::kLong).reshape({-1});
    writeI32(static_cast<int32_t>(flat.numel()));
    append(flat.data_ptr<int64_t>(), flat.numel() * sizeof(int64_t));
}

void FlexKVBufferWriter::writeBoolTensor(const at::Tensor& value) {
    const auto flat = toCpuContiguous(value, at::kBool).reshape({-1});
    const auto uint8Flat = flat.to(at::kByte);
    writeI32(static_cast<int32_t>(uint8Flat.numel()));
    append(uint8Flat.data_ptr<uint8_t>(), uint8Flat.numel() * sizeof(uint8_t));
}

void FlexKVBufferWriter::writeOptionalBoolTensor(const std::optional<at::Tensor>& value) {
    writeBool(value.has_value());
    if (value.has_value()) {
        writeBoolTensor(*value);
    }
}

void FlexKVBufferWriter::writeInt64TensorList(const std::vector<at::Tensor>& values) {
    writeI32(static_cast<int32_t>(values.size()));
    for (const auto& value : values) {
        writeInt64Tensor(value);
    }
}

std::vector<uint8_t> FlexKVBufferWriter::finish() && { return std::move(buffer_); }

FlexKVBufferReader::FlexKVBufferReader(const std::vector<uint8_t>& buffer)
    : buffer_(buffer), offset_(0) {}

const uint8_t* FlexKVBufferReader::consume(size_t size) {
    TORCH_CHECK(offset_ + size <= buffer_.size(), "Unexpected end of FlexKV AOTI payload");
    const uint8_t* ptr = buffer_.data() + offset_;
    offset_ += size;
    return ptr;
}

FlexKVOpcode FlexKVBufferReader::readHeader() {
    const auto* magic = consume(4);
    TORCH_CHECK(std::memcmp(magic, kFlexKVAOTIMagic.data(), kFlexKVAOTIMagic.size()) == 0, "Invalid FlexKV AOTI magic");
    const auto version = readU8();
    TORCH_CHECK(version == kFlexKVAOTIVersion, "Unsupported FlexKV AOTI protocol version: ", version);
    return static_cast<FlexKVOpcode>(readU8());
}

uint8_t FlexKVBufferReader::readU8() { return readScalar<uint8_t>(consume(sizeof(uint8_t))); }
int32_t FlexKVBufferReader::readI32() { return readScalar<int32_t>(consume(sizeof(int32_t))); }
int64_t FlexKVBufferReader::readI64() { return readScalar<int64_t>(consume(sizeof(int64_t))); }
double FlexKVBufferReader::readF64() { return readScalar<double>(consume(sizeof(double))); }
bool FlexKVBufferReader::readBool() { return readU8() != 0; }

std::string FlexKVBufferReader::readString() {
    const auto size = readI32();
    const auto* ptr = consume(static_cast<size_t>(size));
    return std::string(reinterpret_cast<const char*>(ptr), static_cast<size_t>(size));
}

std::vector<std::string> FlexKVBufferReader::readStringList() {
    const auto size = readI32();
    std::vector<std::string> values;
    values.reserve(size);
    for (int32_t i = 0; i < size; ++i) {
        values.push_back(readString());
    }
    return values;
}

at::Tensor FlexKVBufferReader::readInt64Tensor() {
    const auto size = readI32();
    auto tensor = at::empty({size}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
    if (size > 0) {
        std::memcpy(tensor.data_ptr<int64_t>(), consume(static_cast<size_t>(size) * sizeof(int64_t)), static_cast<size_t>(size) * sizeof(int64_t));
    }
    return tensor;
}

at::Tensor FlexKVBufferReader::readBoolTensor() {
    const auto size = readI32();
    auto tensor = at::empty({size}, at::TensorOptions().device(at::kCPU).dtype(at::kBool));
    if (size > 0) {
        auto tmp = at::empty({size}, at::TensorOptions().device(at::kCPU).dtype(at::kByte));
        std::memcpy(tmp.data_ptr<uint8_t>(), consume(static_cast<size_t>(size)), static_cast<size_t>(size));
        tensor.copy_(tmp.to(at::kBool));
    }
    return tensor;
}

std::optional<at::Tensor> FlexKVBufferReader::readOptionalBoolTensor() {
    if (!readBool()) {
        return std::nullopt;
    }
    return readBoolTensor();
}

std::vector<at::Tensor> FlexKVBufferReader::readInt64TensorList() {
    const auto size = readI32();
    std::vector<at::Tensor> values;
    values.reserve(size);
    for (int32_t i = 0; i < size; ++i) {
        values.push_back(readInt64Tensor());
    }
    return values;
}

std::vector<uint8_t> encodeRegisterDPClientRequest(
    int32_t dpClientId,
    const std::string& clientRecvPort,
    int32_t tpSize) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::RegisterDPClient);
    writer.writeI32(dpClientId);
    writer.writeString(clientRecvPort);
    writer.writeI32(tpSize);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeIsReadyRequest(int32_t dpClientId) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::IsReady);
    writer.writeI32(dpClientId);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeStartRequest(int32_t dpClientId) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::Start);
    writer.writeI32(dpClientId);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodePutRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const at::Tensor& slotMapping,
    const std::optional<at::Tensor>& tokenMask,
    const std::vector<std::string>& nameSpace) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::Put);
    writer.writeI32(dpClientId);
    writer.writeInt64Tensor(tokenIds);
    writer.writeInt64Tensor(slotMapping);
    writer.writeOptionalBoolTensor(tokenMask);
    writer.writeI64(taskId);
    writer.writeStringList(nameSpace);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeGetRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const at::Tensor& slotMapping,
    const std::optional<at::Tensor>& tokenMask,
    int32_t layerGranularity,
    const std::vector<std::string>& nameSpace) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::Get);
    writer.writeI32(dpClientId);
    writer.writeInt64Tensor(tokenIds);
    writer.writeInt64Tensor(slotMapping);
    writer.writeOptionalBoolTensor(tokenMask);
    writer.writeI64(taskId);
    writer.writeI32(layerGranularity);
    writer.writeStringList(nameSpace);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeGetMatchRequest(
    int32_t dpClientId,
    int64_t taskId,
    const at::Tensor& tokenIds,
    const std::optional<at::Tensor>& tokenMask,
    int32_t layerGranularity,
    const std::vector<std::string>& nameSpace) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::GetMatch);
    writer.writeI32(dpClientId);
    writer.writeInt64Tensor(tokenIds);
    writer.writeOptionalBoolTensor(tokenMask);
    writer.writeI32(layerGranularity);
    writer.writeI64(taskId);
    writer.writeStringList(nameSpace);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeLaunchTasksRequest(
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds,
    const std::vector<at::Tensor>& slotMappings,
    bool asBatch,
    int64_t batchId) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::LaunchTasks);
    writer.writeI32(dpClientId);
    writer.writeI32(static_cast<int32_t>(taskIds.size()));
    for (auto taskId : taskIds) {
        writer.writeI64(taskId);
    }
    writer.writeInt64TensorList(slotMappings);
    writer.writeBool(asBatch);
    writer.writeI64(batchId);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeCancelTaskRequest(
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::CancelTask);
    writer.writeI32(dpClientId);
    writer.writeI32(static_cast<int32_t>(taskIds.size()));
    for (auto taskId : taskIds) {
        writer.writeI64(taskId);
    }
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeWaitRequest(
    FlexKVOpcode opcode,
    int32_t dpClientId,
    const std::vector<int64_t>& taskIds,
    double timeoutSeconds,
    bool completely) {
    FlexKVBufferWriter writer;
    writer.writeHeader(opcode);
    writer.writeI32(dpClientId);
    writer.writeI32(static_cast<int32_t>(taskIds.size()));
    for (auto taskId : taskIds) {
        writer.writeI64(taskId);
    }
    if (opcode == FlexKVOpcode::Wait) {
        writer.writeF64(timeoutSeconds);
        writer.writeBool(completely);
    }
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeShutdownRequest(int32_t dpClientId) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::Shutdown);
    writer.writeI32(dpClientId);
    return std::move(writer).finish();
}

std::vector<uint8_t> encodeRegisterTPClientRequest(
    int32_t dpClientId,
    int32_t deviceId,
    const std::vector<at::Tensor>& kvCaches,
    const FlexKVCacheLayoutSpec& layout) {
    FlexKVBufferWriter writer;
    writer.writeHeader(FlexKVOpcode::RegisterTPClient);
    writer.writeI32(dpClientId);
    writer.writeI32(deviceId);
    writer.writeI32(static_cast<int32_t>(kvCaches.size()));
    for (const auto& tensor : kvCaches) {
        writeTensorHandle(writer, tensor);
    }
    writer.writeI32(static_cast<int32_t>(layout.type));
    writer.writeI32(layout.num_layer);
    writer.writeI32(layout.num_block);
    writer.writeI32(layout.tokens_per_block);
    writer.writeI32(layout.num_head);
    writer.writeI32(layout.head_size);
    writer.writeBool(layout.is_mla);
    return std::move(writer).finish();
}

bool decodeIsReadyResponse(const std::vector<uint8_t>& buffer) {
    FlexKVBufferReader reader(buffer);
    const auto opcode = reader.readHeader();
    if (opcode == FlexKVOpcode::ErrorResponse) {
        TORCH_CHECK(false, decodeErrorResponse(buffer));
    }
    TORCH_CHECK(opcode == FlexKVOpcode::IsReadyResponse, "Unexpected FlexKV response opcode");
    return reader.readBool();
}

std::pair<int64_t, at::Tensor> decodeGetMatchResponse(const std::vector<uint8_t>& buffer) {
    FlexKVBufferReader reader(buffer);
    const auto opcode = reader.readHeader();
    if (opcode == FlexKVOpcode::ErrorResponse) {
        TORCH_CHECK(false, decodeErrorResponse(buffer));
    }
    TORCH_CHECK(opcode == FlexKVOpcode::GetMatchResponse, "Unexpected FlexKV response opcode");
    return {reader.readI64(), reader.readBoolTensor()};
}

std::unordered_map<int64_t, KVTaskResponse> decodeWaitResponse(
    const std::vector<uint8_t>& buffer,
    FlexKVOpcode expectedOpcode) {
    FlexKVBufferReader reader(buffer);
    const auto opcode = reader.readHeader();
    if (opcode == FlexKVOpcode::ErrorResponse) {
        TORCH_CHECK(false, decodeErrorResponse(buffer));
    }
    TORCH_CHECK(opcode == expectedOpcode, "Unexpected FlexKV response opcode");
    const auto size = reader.readI32();
    std::unordered_map<int64_t, KVTaskResponse> values;
    values.reserve(size);
    for (int32_t i = 0; i < size; ++i) {
        const auto taskId = reader.readI64();
        values.emplace(taskId, KVTaskResponse{taskId, static_cast<KVResponseStatus>(reader.readU8())});
    }
    return values;
}

std::string decodeErrorResponse(const std::vector<uint8_t>& buffer) {
    FlexKVBufferReader reader(buffer);
    const auto opcode = reader.readHeader();
    TORCH_CHECK(opcode == FlexKVOpcode::ErrorResponse, "Unexpected non-error FlexKV response");
    return reader.readString();
}

} // namespace kvcache_manager