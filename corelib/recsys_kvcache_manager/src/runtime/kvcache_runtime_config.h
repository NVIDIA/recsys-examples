#pragma once

#include <ATen/ATen.h>

#include <filesystem>
#include <string>
#include <unordered_map>

namespace kvcache_manager {

using ConfigMap = std::unordered_map<std::string, std::string>;

ConfigMap load_yaml_config(const std::filesystem::path& path);
std::filesystem::path find_config_path();
std::string get_required_config_string(const ConfigMap& config, const std::string& name);
int get_config_int(const ConfigMap& config, const std::string& name);
at::ScalarType get_config_dtype(const ConfigMap& config);

} // namespace kvcache_manager