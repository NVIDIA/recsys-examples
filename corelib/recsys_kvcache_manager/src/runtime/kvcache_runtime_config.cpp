#include "kvcache_runtime_config.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace kvcache_manager {

namespace {

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string remove_yaml_comment(const std::string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t idx = 0; idx < line.size(); ++idx) {
        const char ch = line[idx];
        if (ch == '\'' && !in_double_quote) {
            in_single_quote = !in_single_quote;
        } else if (ch == '"' && !in_single_quote) {
            in_double_quote = !in_double_quote;
        } else if (ch == '#' && !in_single_quote && !in_double_quote) {
            return line.substr(0, idx);
        }
    }
    return line;
}

std::string strip_yaml_quotes(std::string value) {
    if (value.size() >= 2) {
        const char first = value.front();
        const char last = value.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            value = value.substr(1, value.size() - 2);
        }
    }
    return value;
}

} // namespace

ConfigMap load_yaml_config(const std::filesystem::path& path) {
    std::ifstream stream(path);
    TORCH_CHECK(stream.is_open(), "Failed to open KV-cache config YAML: ", path.string());

    ConfigMap config;
    std::string line;
    while (std::getline(stream, line)) {
        line = trim_copy(remove_yaml_comment(line));
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        auto key = trim_copy(line.substr(0, separator));
        auto value = trim_copy(line.substr(separator + 1));
        if (key.empty()) {
            continue;
        }
        config[key] = strip_yaml_quotes(value);
    }
    return config;
}

std::filesystem::path find_config_path() {
    const char* override_path = std::getenv("KVCACHE_MANAGER_CONFIG_FILE");
    if (override_path != nullptr && !std::string(override_path).empty()) {
        std::filesystem::path path(override_path);
        TORCH_CHECK(
            std::filesystem::exists(path),
            "KVCACHE_MANAGER_CONFIG_FILE does not exist: ",
            path.string());
        return path;
    }

    const std::vector<std::filesystem::path> candidates = {
        "inference_aoti/kvcache_cpp_runtime.yaml",
        "examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml",
        "kvcache_cpp_runtime.yaml",
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    TORCH_CHECK(false, "Unable to find kvcache_cpp_runtime.yaml. Set KVCACHE_MANAGER_CONFIG_FILE to the YAML path.");
    return {};
}

std::string get_required_config_string(const ConfigMap& config, const std::string& name) {
    auto iter = config.find(name);
    TORCH_CHECK(
        iter != config.end() && !iter->second.empty(),
        "Missing required KV-cache config value: ",
        name);
    return iter->second;
}

int get_config_int(const ConfigMap& config, const std::string& name) {
    auto value = get_required_config_string(config, name);
    return std::stoi(value);
}

at::ScalarType get_config_dtype(const ConfigMap& config) {
    auto dtype = get_required_config_string(config, "dtype");
    std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (dtype == "bfloat16" || dtype == "bf16") {
        return at::kBFloat16;
    }
    if (dtype == "float16" || dtype == "fp16" || dtype == "half") {
        return at::kHalf;
    }
    TORCH_CHECK(false, "Unsupported KV-cache dtype: ", dtype);
}

} // namespace kvcache_manager