#include "flexkv_serialization_bridge.h"

#include <pybind11/stl.h>
#include <torch/script.h>
#include <sstream>

namespace py = pybind11;

namespace kvcache_manager {

FlexKVSerializationBridge::FlexKVSerializationBridge() {
    try {
        // Import FlexKV modules
        flexkv_server_request_ = py::module_::import("flexkv.server.request");
        flexkv_common_request_ = py::module_::import("flexkv.common.request");
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to import FlexKV Python modules: " + std::string(e.what()));
    }
}

// Helper: Tensor to numpy
py::object FlexKVSerializationBridge::tensor_to_numpy(const at::Tensor& tensor) {
    // Convert to CPU and ensure contiguous
    at::Tensor cpu_tensor = tensor.contiguous().to(at::kCPU).detach();
    
    // Use torch.from_numpy in reverse: create numpy array from torch tensor
    // by exporting to numpy via PyTorch's built-in conversion
    py::object np = py::module_::import("numpy");
    py::object torch_module = py::module_::import("torch");
    
    // Convert torch tensor to numpy
    return torch_module.attr("Tensor")(cpu_tensor).attr("numpy")();
}

// Helper: vector<Tensor> to list of numpy arrays
py::list FlexKVSerializationBridge::tensors_to_numpy_list(
    const std::vector<at::Tensor>& tensors) {
    py::list result;
    for (const auto& t : tensors) {
        result.append(tensor_to_numpy(t));
    }
    return result;
}

// Helper: numpy to Tensor
at::Tensor FlexKVSerializationBridge::numpy_to_tensor(const py::object& np_array) {
    // Convert numpy array to PyTorch tensor
    return torch::from_blob(
        np_array.attr("data").attr("_as_parameter_").cast<void*>(),
        py::list(np_array.attr("shape")).cast<std::vector<int64_t>>(),
        [](void*) {}, // no deleter
        at::kCPU);
}

// Helper: numpy to vector<bool>
std::vector<bool> FlexKVSerializationBridge::numpy_to_bool_vector(
    const py::object& np_array) {
    py::array_t<bool> arr = np_array.cast<py::array_t<bool>>();
    auto buf = arr.request();
    bool* ptr = static_cast<bool*>(buf.ptr);
    return std::vector<bool>(ptr, ptr + arr.size());
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_get_match_request(
    int dp_client_id,
    const at::Tensor& token_ids,
    const std::optional<at::Tensor>& token_mask,
    int layer_granularity,
    const std::vector<std::string>& namespace_list) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        // Create GetMatchRequest instance
        auto GetMatchRequest = flexkv_server_request_.attr("GetMatchRequest");
        
        py::object mask_obj = token_mask ? tensor_to_numpy(*token_mask) : py::none();
        py::object ns_obj = namespace_list.empty() ? py::none() : 
                            py::cast(namespace_list);
        
        py::object request = GetMatchRequest(
            dp_client_id,
            tensor_to_numpy(token_ids),
            mask_obj,
            layer_granularity,
            ns_obj);
        
        // Pickle the request
        py::bytes pickled = pickle_module.attr("dumps")(request);
        
        // Convert to vector<uint8_t>
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize GetMatchRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_put_request(
    int dp_client_id,
    const at::Tensor& token_ids,
    const at::Tensor& slot_mapping,
    const std::optional<at::Tensor>& token_mask,
    int task_id,
    const std::vector<std::string>& namespace_list) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto PutRequest = flexkv_server_request_.attr("PutRequest");
        
        py::object mask_obj = token_mask ? tensor_to_numpy(*token_mask) : py::none();
        py::object ns_obj = namespace_list.empty() ? py::none() : 
                            py::cast(namespace_list);
        
        py::object request = PutRequest(
            dp_client_id,
            tensor_to_numpy(token_ids),
            tensor_to_numpy(slot_mapping),
            mask_obj,
            task_id,
            ns_obj);
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize PutRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_get_request(
    int dp_client_id,
    const at::Tensor& token_ids,
    const at::Tensor& slot_mapping,
    const std::optional<at::Tensor>& token_mask,
    int task_id,
    int layer_granularity,
    const std::vector<std::string>& namespace_list) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto GetRequest = flexkv_server_request_.attr("GetRequest");
        
        py::object mask_obj = token_mask ? tensor_to_numpy(*token_mask) : py::none();
        py::object ns_obj = namespace_list.empty() ? py::none() : 
                            py::cast(namespace_list);
        
        py::object request = GetRequest(
            dp_client_id,
            tensor_to_numpy(token_ids),
            tensor_to_numpy(slot_mapping),
            mask_obj,
            task_id,
            layer_granularity,
            ns_obj);
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize GetRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_launch_task_request(
    int dp_client_id,
    const std::vector<int>& task_ids,
    const std::vector<at::Tensor>& slot_mappings) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto LaunchTaskRequest = flexkv_server_request_.attr("LaunchTaskRequest");
        
        py::list slot_mappings_list;
        for (const auto& t : slot_mappings) {
            slot_mappings_list.append(tensor_to_numpy(t));
        }
        
        py::object request = LaunchTaskRequest(
            dp_client_id,
            py::cast(task_ids),
            slot_mappings_list);
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize LaunchTaskRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_wait_request(
    int dp_client_id,
    const std::vector<int>& task_ids,
    float timeout_sec,
    bool completely) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto WaitRequest = flexkv_server_request_.attr("WaitRequest");
        
        py::object request = WaitRequest(
            dp_client_id,
            py::none(),  // reserved
            py::cast(task_ids),
            static_cast<double>(timeout_sec),
            completely);
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize WaitRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_try_wait_request(
    int dp_client_id,
    const std::vector<int>& task_ids) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto TryWaitRequest = flexkv_server_request_.attr("TryWaitRequest");
        
        py::object request = TryWaitRequest(
            dp_client_id,
            py::none(),
            py::cast(task_ids));
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize TryWaitRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_cancel_task_request(
    int dp_client_id,
    const std::vector<int>& task_ids) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto CancelTaskRequest = flexkv_server_request_.attr("CancelTaskRequest");
        
        py::object request = CancelTaskRequest(
            dp_client_id,
            py::cast(task_ids));
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize CancelTaskRequest: " + std::string(e.what()));
    }
}

std::vector<uint8_t> FlexKVSerializationBridge::serialize_is_ready_request(
    int dp_client_id) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        auto IsReadyRequest = flexkv_server_request_.attr("IsReadyRequest");
        
        py::object request = IsReadyRequest(dp_client_id);
        
        py::bytes pickled = pickle_module.attr("dumps")(request);
        std::string pickled_str = pickled;
        return std::vector<uint8_t>(pickled_str.begin(), pickled_str.end());
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to serialize IsReadyRequest: " + std::string(e.what()));
    }
}

std::pair<int, std::vector<bool>>
FlexKVSerializationBridge::deserialize_get_match_response(
    const std::vector<uint8_t>& pickled_response) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        // Unpickle response
        py::bytes pickled_bytes(
            reinterpret_cast<const char*>(pickled_response.data()),
            pickled_response.size());
        py::object response = pickle_module.attr("loads")(pickled_bytes);
        
        // Extract task_id and matched_mask
        int task_id = py::cast<int>(response.attr("task_ids")[0]);
        std::vector<bool> matched_mask = numpy_to_bool_vector(
            response.attr("matched_mask"));
        
        return {task_id, matched_mask};
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to deserialize GetMatch response: " + std::string(e.what()));
    }
}

std::unordered_map<int, std::pair<int, std::string>>
FlexKVSerializationBridge::deserialize_wait_response(
    const std::vector<uint8_t>& pickled_response) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        py::bytes pickled_bytes(
            reinterpret_cast<const char*>(pickled_response.data()),
            pickled_response.size());
        py::object response = pickle_module.attr("loads")(pickled_bytes);
        
        std::unordered_map<int, std::pair<int, std::string>> result;
        
        // response.status is a dict-like structure
        py::dict status_dict = response.attr("status").cast<py::dict>();
        
        for (auto item : status_dict) {
            int task_id = py::cast<int>(item.first);
            py::object kv_response = item.second;
            
            int status_code = py::cast<int>(kv_response.attr("status"));
            std::string error_msg = py::cast<std::string>(
                kv_response.attr("error_msg") ? 
                kv_response.attr("error_msg") : py::str(""));
            
            result[task_id] = {status_code, error_msg};
        }
        
        return result;
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to deserialize wait response: " + std::string(e.what()));
    }
}

bool FlexKVSerializationBridge::deserialize_is_ready_response(
    const std::vector<uint8_t>& pickled_response) {
    try {
        py::object pickle_module = py::module_::import("pickle");
        
        py::bytes pickled_bytes(
            reinterpret_cast<const char*>(pickled_response.data()),
            pickled_response.size());
        py::object response = pickle_module.attr("loads")(pickled_bytes);
        
        return py::cast<bool>(response.attr("is_ready"));
    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            "Failed to deserialize is_ready response: " + std::string(e.what()));
    }
}

} // namespace kvcache_manager
