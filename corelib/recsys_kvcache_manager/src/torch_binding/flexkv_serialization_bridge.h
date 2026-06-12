#pragma once

#include <vector>
#include <string>
#include <optional>
#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace kvcache_manager {

/**
 * @class FlexKVSerializationBridge
 * @brief Handles pickle serialization/deserialization using Python.
 *
 * This bridge isolates Python/pybind11 dependencies to one place,
 * keeping the main C++ client clean.
 */
class FlexKVSerializationBridge {
public:
    FlexKVSerializationBridge();
    ~FlexKVSerializationBridge() = default;

    // ===== Serialization (C++ -> pickle bytes) =====

    /**
     * Serialize a GetMatchRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param token_ids CPU tensor of token IDs (int64)
     * @param token_mask optional mask tensor
     * @param layer_granularity granularity level
     * @param namespace_list list of namespaces
     * @return pickled bytes ready for ZMQ transmission
     */
    std::vector<uint8_t> serialize_get_match_request(
        int dp_client_id,
        const at::Tensor& token_ids,
        const std::optional<at::Tensor>& token_mask,
        int layer_granularity,
        const std::vector<std::string>& namespace_list);

    /**
     * Serialize a PutRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param token_ids CPU int64 tensor
     * @param slot_mapping CPU int64 tensor
     * @param token_mask optional mask
     * @param task_id task identifier
     * @param namespace_list list of namespaces
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_put_request(
        int dp_client_id,
        const at::Tensor& token_ids,
        const at::Tensor& slot_mapping,
        const std::optional<at::Tensor>& token_mask,
        int task_id,
        const std::vector<std::string>& namespace_list);

    /**
     * Serialize a GetRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param token_ids CPU int64 tensor
     * @param slot_mapping CPU int64 tensor
     * @param token_mask optional mask
     * @param task_id task identifier
     * @param layer_granularity layer-level granularity
     * @param namespace_list list of namespaces
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_get_request(
        int dp_client_id,
        const at::Tensor& token_ids,
        const at::Tensor& slot_mapping,
        const std::optional<at::Tensor>& token_mask,
        int task_id,
        int layer_granularity,
        const std::vector<std::string>& namespace_list);

    /**
     * Serialize a LaunchTaskRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param task_ids list of task IDs
     * @param slot_mappings list of slot mapping tensors (CPU int64)
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_launch_task_request(
        int dp_client_id,
        const std::vector<int>& task_ids,
        const std::vector<at::Tensor>& slot_mappings);

    /**
     * Serialize a WaitRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param task_ids list of task IDs to wait for
     * @param timeout_sec timeout in seconds
     * @param completely whether to wait completely
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_wait_request(
        int dp_client_id,
        const std::vector<int>& task_ids,
        float timeout_sec,
        bool completely = false);

    /**
     * Serialize a TryWaitRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param task_ids list of task IDs to check
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_try_wait_request(
        int dp_client_id,
        const std::vector<int>& task_ids);

    /**
     * Serialize a CancelTaskRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @param task_ids list of task IDs to cancel
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_cancel_task_request(
        int dp_client_id,
        const std::vector<int>& task_ids);

    /**
     * Serialize an IsReadyRequest for pickle transmission.
     *
     * @param dp_client_id data-parallel client ID
     * @return pickled bytes
     */
    std::vector<uint8_t> serialize_is_ready_request(int dp_client_id);

    // ===== Deserialization (pickle bytes -> C++) =====

    /**
     * Deserialize a GetMatch response.
     *
     * @param pickled_response raw pickled bytes from server
     * @return {task_id, matched_mask as vector<bool>}
     */
    std::pair<int, std::vector<bool>> deserialize_get_match_response(
        const std::vector<uint8_t>& pickled_response);

    /**
     * Deserialize a wait/try_wait response containing task status.
     *
     * @param pickled_response raw pickled bytes
     * @return map of {task_id -> (status_int, error_msg)}
     */
    std::unordered_map<int, std::pair<int, std::string>> deserialize_wait_response(
        const std::vector<uint8_t>& pickled_response);

    /**
     * Deserialize is_ready response.
     *
     * @param pickled_response raw pickled bytes
     * @return true if ready
     */
    bool deserialize_is_ready_response(
        const std::vector<uint8_t>& pickled_response);

private:
    // Python module references for FlexKV classes
    pybind11::module_ flexkv_server_request_;
    pybind11::module_ flexkv_common_request_;

    // Helper: convert Tensor to numpy array for pickling
    pybind11::object tensor_to_numpy(const at::Tensor& tensor);

    // Helper: convert vector<Tensor> to list of numpy arrays
    pybind11::list tensors_to_numpy_list(const std::vector<at::Tensor>& tensors);

    // Helper: numpy array to Tensor
    at::Tensor numpy_to_tensor(const pybind11::object& np_array);

    // Helper: numpy array to vector<bool>
    std::vector<bool> numpy_to_bool_vector(const pybind11::object& np_array);
};

} // namespace kvcache_manager
