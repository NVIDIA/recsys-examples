#include "flexkv_cpp_client.h"

#include <c10/util/Exception.h>

#include <chrono>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include <zmq.hpp>

namespace kvcache_manager {

class FlexKVCppClient::Impl {
public:
    explicit Impl(const std::string& serverAddr, int32_t dpClientId, int32_t tpSize)
        : server_addr_(serverAddr),
          dp_client_id_(dpClientId),
          tp_size_(tpSize),
          task_counter_(static_cast<int64_t>(dpClientId) * 10000000),
          task_id_end_(static_cast<int64_t>(dpClientId + 1) * 10000000),
                    registered_(false),
          context_(std::make_shared<zmq::context_t>(2)),
          send_sock_(std::make_shared<zmq::socket_t>(*context_, zmq::socket_type::push)),
          recv_sock_(std::make_shared<zmq::socket_t>(*context_, zmq::socket_type::pull)) {
        send_sock_->set(zmq::sockopt::linger, 0);
        send_sock_->set(zmq::sockopt::immediate, 1);
        send_sock_->set(zmq::sockopt::sndtimeo, 5000);
        recv_sock_->set(zmq::sockopt::linger, 0);
        send_sock_->connect(server_addr_);
        recv_port_ = create_recv_endpoint();
        recv_sock_->bind(recv_port_);
        std::cout << "[KVCACHE][flexkv_client] connected server_addr=" << server_addr_
              << " client_recv_port=" << recv_port_ << std::endl;
    }

    ~Impl() {
        if (send_sock_) {
            send_sock_->close();
        }
        if (recv_sock_) {
            recv_sock_->close();
        }
        if (!recv_path_.empty()) {
            std::remove(recv_path_.c_str());
        }
        if (!recv_dir_.empty()) {
            rmdir(recv_dir_.c_str());
        }
    }

    std::string create_recv_endpoint() {
        char dirTemplate[] = "/tmp/flexkv_aoti_client_XXXXXX";
        const char* dirPath = mkdtemp(dirTemplate);
        TORCH_CHECK(dirPath != nullptr, "Failed to allocate temporary IPC directory for FlexKV client");
        recv_dir_ = dirPath;
        recv_path_ = recv_dir_ + "/reply.sock";
        return std::string("ipc://") + recv_path_;
    }

    void send_only(const std::vector<uint8_t>& payload) {
        zmq::message_t message(payload.size());
        std::memcpy(message.data(), payload.data(), payload.size());
        const auto sent = send_sock_->send(message, zmq::send_flags::none);
        TORCH_CHECK(sent.has_value(), "Timed out sending FlexKV request to ", server_addr_);
    }

    std::vector<uint8_t> send_and_recv(const std::vector<uint8_t>& payload, int timeoutMs = 25000) {
        send_only(payload);
        std::cout << "[KVCACHE][flexkv_client] waiting for response timeout_ms="
                  << timeoutMs << std::endl;
        zmq::pollitem_t items[] = {{recv_sock_->handle(), 0, ZMQ_POLLIN, 0}};
        const auto rc = zmq::poll(items, 1, std::chrono::milliseconds(timeoutMs));
        TORCH_CHECK(rc >= 0, "FlexKV poll failed");
        TORCH_CHECK(items[0].revents & ZMQ_POLLIN, "Timed out waiting for FlexKV response");
        zmq::message_t response;
        recv_sock_->recv(response, zmq::recv_flags::none);
        std::cout << "[KVCACHE][flexkv_client] received response bytes="
                  << response.size() << std::endl;
        const auto* begin = static_cast<const uint8_t*>(response.data());
        return std::vector<uint8_t>(begin, begin + response.size());
    }

    int64_t next_task_id() {
        const auto task_id = task_counter_++;
        if (task_counter_ >= task_id_end_) {
            task_counter_ = static_cast<int64_t>(dp_client_id_) * 10000000;
        }
        return task_id;
    }

    std::string server_addr_;
    int32_t dp_client_id_;
    int32_t tp_size_;
    int64_t task_counter_;
    int64_t task_id_end_;
    bool registered_;
    std::string recv_port_;
    std::string recv_dir_;
    std::string recv_path_;
    std::shared_ptr<zmq::context_t> context_;
    std::shared_ptr<zmq::socket_t> send_sock_;
    std::shared_ptr<zmq::socket_t> recv_sock_;
};

FlexKVCppClient::FlexKVCppClient(const std::string& serverAddr, int32_t dpClientId, int32_t tpSize)
    : impl_(std::make_unique<Impl>(serverAddr, dpClientId, tpSize)) {}

FlexKVCppClient::~FlexKVCppClient() = default;

void FlexKVCppClient::register_to_server() {
    if (impl_->registered_) {
        return;
    }
    std::cout << "[KVCACHE][flexkv_client] register DP client begin dp="
              << impl_->dp_client_id_ << " recv=" << impl_->recv_port_ << std::endl;
    impl_->send_only(encodeRegisterDPClientRequest(
        impl_->dp_client_id_,
        impl_->recv_port_,
        impl_->tp_size_));
    impl_->registered_ = true;
    std::cout << "[KVCACHE][flexkv_client] register DP client sent" << std::endl;
}

void FlexKVCppClient::start_server_and_register() {
    std::cout << "[KVCACHE][flexkv_client] start request begin dp="
              << impl_->dp_client_id_ << std::endl;
    impl_->send_only(encodeStartRequest(impl_->dp_client_id_));
    std::cout << "[KVCACHE][flexkv_client] start request sent" << std::endl;
    register_to_server();
}

const std::string& FlexKVCppClient::client_recv_port() const { return impl_->recv_port_; }

std::pair<int64_t, at::Tensor> FlexKVCppClient::get_match(
    const at::Tensor& token_ids,
    const std::optional<at::Tensor>& token_mask,
    int layer_granularity,
    const std::vector<std::string>& namespace_list) {
    const auto task_id = impl_->next_task_id();
    const auto request = encodeGetMatchRequest(
        impl_->dp_client_id_,
        task_id,
        token_ids,
        token_mask,
        layer_granularity,
        namespace_list);
    return decodeGetMatchResponse(impl_->send_and_recv(request));
}

int64_t FlexKVCppClient::put_async(
    const at::Tensor& token_ids,
    const at::Tensor& slot_mapping,
    const std::optional<at::Tensor>& token_mask,
    const std::vector<std::string>& namespace_list) {
    const auto task_id = impl_->next_task_id();
    impl_->send_only(encodePutRequest(
        impl_->dp_client_id_,
        task_id,
        token_ids,
        slot_mapping,
        token_mask,
        namespace_list));
    return task_id;
}

int64_t FlexKVCppClient::get_async(
    const at::Tensor& token_ids,
    const at::Tensor& slot_mapping,
    const std::optional<at::Tensor>& token_mask,
    int layer_granularity,
    const std::vector<std::string>& namespace_list) {
    const auto task_id = impl_->next_task_id();
    impl_->send_only(encodeGetRequest(
        impl_->dp_client_id_,
        task_id,
        token_ids,
        slot_mapping,
        token_mask,
        layer_granularity,
        namespace_list));
    return task_id;
}

std::vector<int64_t> FlexKVCppClient::launch_tasks(
    const std::vector<int64_t>& task_ids,
    const std::vector<at::Tensor>& slot_mappings,
    bool as_batch) {
    int64_t batch_id = -1;
    if (as_batch) {
        batch_id = impl_->next_task_id();
    }
    impl_->send_only(encodeLaunchTasksRequest(
        impl_->dp_client_id_,
        task_ids,
        slot_mappings,
        as_batch,
        batch_id));
    if (as_batch) {
        return {batch_id};
    }
    return task_ids;
}

std::unordered_map<int, KVTaskResponse> FlexKVCppClient::wait(
    const std::vector<int64_t>& task_ids,
    float timeout_sec,
    bool completely) {
    auto decoded = decodeWaitResponse(
        impl_->send_and_recv(encodeWaitRequest(
            FlexKVOpcode::Wait,
            impl_->dp_client_id_,
            task_ids,
            timeout_sec,
            completely)),
        FlexKVOpcode::WaitResponse);
    std::unordered_map<int, KVTaskResponse> result;
    for (const auto& entry : decoded) {
        result.emplace(static_cast<int>(entry.first), entry.second);
    }
    return result;
}

std::unordered_map<int, KVTaskResponse> FlexKVCppClient::try_wait(
    const std::vector<int64_t>& task_ids) {
    auto decoded = decodeWaitResponse(
        impl_->send_and_recv(encodeWaitRequest(
            FlexKVOpcode::TryWait,
            impl_->dp_client_id_,
            task_ids,
            0.0,
            false)),
        FlexKVOpcode::TryWaitResponse);
    std::unordered_map<int, KVTaskResponse> result;
    for (const auto& entry : decoded) {
        result.emplace(static_cast<int>(entry.first), entry.second);
    }
    return result;
}

void FlexKVCppClient::cancel(const std::vector<int64_t>& task_ids) {
    impl_->send_only(encodeCancelTaskRequest(impl_->dp_client_id_, task_ids));
}

bool FlexKVCppClient::is_ready() {
    std::cout << "[KVCACHE][flexkv_client] is_ready request begin dp="
              << impl_->dp_client_id_ << std::endl;
    return decodeIsReadyResponse(
        impl_->send_and_recv(encodeIsReadyRequest(impl_->dp_client_id_)));
}

void FlexKVCppClient::shutdown() {
    impl_->send_only(encodeShutdownRequest(impl_->dp_client_id_));
}


class FlexKVGPURegistrator::Impl {
public:
    Impl(const std::string& gpuRegisterPort, int32_t dpClientId, int32_t deviceId)
        : gpu_register_port_(gpuRegisterPort),
          dp_client_id_(dpClientId),
          device_id_(deviceId),
          context_(std::make_shared<zmq::context_t>(2)),
          send_sock_(std::make_shared<zmq::socket_t>(*context_, zmq::socket_type::push)) {
        send_sock_->connect(gpu_register_port_);
    }

    void send_only(const std::vector<uint8_t>& payload) {
        zmq::message_t message(payload.size());
        std::memcpy(message.data(), payload.data(), payload.size());
        const auto sent = send_sock_->send(message, zmq::send_flags::dontwait);
        TORCH_CHECK(sent.has_value(), "Failed to send FlexKV GPU registration request to ", gpu_register_port_);
    }

    std::string gpu_register_port_;
    int32_t dp_client_id_;
    int32_t device_id_;
    std::shared_ptr<zmq::context_t> context_;
    std::shared_ptr<zmq::socket_t> send_sock_;
};

FlexKVGPURegistrator::FlexKVGPURegistrator(const std::string& gpuRegisterPort, int32_t dpClientId, int32_t deviceId)
    : impl_(std::make_unique<Impl>(gpuRegisterPort, dpClientId, deviceId)) {}

FlexKVGPURegistrator::~FlexKVGPURegistrator() = default;

void FlexKVGPURegistrator::register_to_server(
    const std::vector<at::Tensor>& kvCaches,
    const FlexKVCacheLayoutSpec& layout,
    std::optional<int32_t> overrideDeviceId) {
    impl_->send_only(encodeRegisterTPClientRequest(
        impl_->dp_client_id_,
        overrideDeviceId.value_or(impl_->device_id_),
        kvCaches,
        layout));
}

} // namespace kvcache_manager
