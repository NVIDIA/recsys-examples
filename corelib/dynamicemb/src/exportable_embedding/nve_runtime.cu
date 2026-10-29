// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "dynamicemb/exportable_embedding/nve_runtime.h"
#include "dynamicemb/exportable_embedding/nve_tier_binding.h"
#include "python/pynve/torch_bindings/nve_loader.hpp"

#include <fstream>
#include <set>

namespace dynamicemb::exportable_embedding {

bool has_nve_tier_update(const std::shared_ptr<NveBinding>& layer) {
  return std::dynamic_pointer_cast<TierGPUEmbedding>(layer) ||
         std::dynamic_pointer_cast<TierUVMEmbedding>(layer) ||
         std::dynamic_pointer_cast<TierHierarchicalEmbedding>(layer);
}

void update_nve_tier(const std::shared_ptr<NveBinding>& layer, size_t count,
                     uintptr_t keys, uintptr_t values, int64_t tier, uint64_t stream) {
  if (auto p = std::dynamic_pointer_cast<TierHierarchicalEmbedding>(layer))
    p->update_tier(count, keys, values, tier, stream);
  else if (auto p = std::dynamic_pointer_cast<TierGPUEmbedding>(layer))
    p->update_tier(count, keys, values, tier, stream);
  else if (auto p = std::dynamic_pointer_cast<TierUVMEmbedding>(layer))
    p->update_tier(count, keys, values, tier, stream);
  else
    TORCH_CHECK(false, "Tier updates require a DynamicEmb NVE binding; use "
                       "create_nve_layer/load_embedding_collection_aot or NveRuntimeDirectory");
}

struct NveRuntimeDirectory::Impl {
  nve::ResourceDirectory resources;
  std::vector<NveRuntimeLayer> layers;

  ~Impl() {
    for (const auto& layer : layers)
      nve::NVELayerRegistry::instance().unregister_binding(layer.marker.data_ptr());
  }
};

NveRuntimeDirectory::NveRuntimeDirectory(const std::string& path, int device)
    : impl_(std::make_unique<Impl>()) {
  ScopedDevice guard(device);
  std::ifstream file(path + "/metadata.json");
  TORCH_CHECK(file.is_open(), "Cannot open NVE metadata in ", path);
  const auto doc = nlohmann::json::parse(file);
  TORCH_CHECK(doc.is_object() && doc.at("version") == 2,
              "DynamicEmb tier bindings require NVE 26.10 v2 metadata");
  const auto& resources = doc.at("resources");
  std::set<std::string> module_paths;
  std::set<int64_t> ids;
  for (const auto& entry : doc.at("layers")) {
    const auto name = entry.at("module_path").get<std::string>();
    const auto id = entry.at("id").get<int64_t>();
    TORCH_CHECK(module_paths.insert(name).second && ids.insert(id).second,
                "Duplicate NVE layer path/id");
    const auto kind = entry.at("layer_type").get<std::string>();
    const auto dim = entry.at("embedding_size").get<size_t>();
    const auto rows = entry.at("num_embeddings").get<size_t>();
    const auto dtype = nve::parse_dtype(entry.at("dtype").get<std::string>());
    const auto ref = entry.at("storage_ref").get<std::string>();
    const auto gpu_cache = entry.value("gpu_cache_size", size_t{0});
    const auto host_cache = entry.value("host_cache_size", size_t{0});
    const auto cfg = nve::parse_embed_config(entry);
    TORCH_CHECK(!entry.at("optimize_for_training").get<bool>(),
                "DynamicEmb runtime loads inference layers only");
    TORCH_CHECK(device >= 0 || kind == "HostLayer", "CPU execution requires HostLayer");
    std::shared_ptr<NveBinding> binding;

    if (kind == "Hierarchical") {
      const auto& ps = resources.at("remote_ps").at(ref);
      auto remote = impl_->resources.find_ps(ref);
      if (!remote) {
        TORCH_CHECK(ps.value("remote_ps_type", "") == "plugin",
                    "Expected an NVE plugin parameter server");
        remote = nve::create_table_from_plugin(
            ps.at("plugin_name").get<std::string>(),
            ps.value("factory_config", nlohmann::json::object()),
            ps.value("table_config", nlohmann::json::object()));
        TORCH_CHECK(remote->get_value_type() == dtype, "Parameter-server dtype mismatch");
        // Preserve stock NVE's optional initial PS-data loading and completion.
        if (ps.contains("remote_ps_data")) {
          const auto& data = ps.at("remote_ps_data");
          auto ctx = remote->create_execution_context(0, 0, nullptr, nullptr);
          nve::insert_keys_from_filepath(
              remote, ctx, data.at("keys").get<std::string>(),
              data.at("values").get<std::string>(),
              dim * nve::dtype_size(dtype), 1ull << 20);
          ctx->wait();
        }
        impl_->resources.insert_ps(ref, remote);
      }
      TORCH_CHECK(remote->get_value_type() == dtype, "Shared parameter-server dtype mismatch");
      binding = std::make_shared<TierHierarchicalEmbedding>(
          dim, dtype, gpu_cache, host_cache, remote,
          ps.value("num_rows", uint64_t{0}), false, device, cfg);
    } else {
      TORCH_CHECK(kind == "GPULayer" || kind == "LinearUVM" || kind == "HostLayer",
                  "Unsupported NVE layer: ", kind);
      const auto& mb = resources.at("memblocks").at(ref);
      TORCH_CHECK(mb.at("row_elements").get<size_t>() == dim &&
                      mb.at("num_rows").get<size_t>() == rows &&
                      nve::parse_dtype(mb.at("dtype").get<std::string>()) == dtype,
                  "MemBlock geometry/dtype mismatch");
      auto storage = impl_->resources.find_memblock(ref);
      const bool load_weights = !storage;
      if (!storage) {
        const auto type = mb.at("type").get<std::string>();
        TORCH_CHECK(type.find("NVL") == std::string::npos &&
                        type.find("MPI") == std::string::npos,
                    "DynamicEmb runtime does not load distributed MemBlock topology");
        storage = nve::create_memblock(type, dim, rows, dtype, device, kind == "GPULayer");
        impl_->resources.insert_memblock(ref, storage);
      }
      const auto load = [&](const auto& layer) {
        if (load_weights) {
          nve::InputFileStreamWrapper weights(path + "/weights/" + ref + ".nve");
          layer->load_tensor_from_stream(weights, id);
        }
        binding = layer;
      };
      if (kind == "GPULayer")
        load(std::make_shared<TierGPUEmbedding>(dim, rows, dtype, storage, device, cfg));
      else if (kind == "LinearUVM")
        load(std::make_shared<TierUVMEmbedding>(
            dim, rows, dtype, storage, gpu_cache, false, device, cfg));
      else
        load(std::make_shared<nve::HostEmbedding<int64_t>>(
            dim, rows, dtype, storage, device, cfg));
    }
    const auto opts = at::TensorOptions().dtype(at::kLong).device(
        device < 0 ? at::Device(at::kCPU) : at::Device(at::kCUDA, device));
    auto marker = at::tensor(std::vector<int64_t>{id}, opts);
    impl_->layers.push_back({name, binding, marker});
    nve::NVELayerRegistry::instance().register_binding(marker.data_ptr(), binding);
  }
}

NveRuntimeDirectory::~NveRuntimeDirectory() = default;
const std::vector<NveRuntimeLayer>& NveRuntimeDirectory::layers() const { return impl_->layers; }
NveBindings NveRuntimeDirectory::bindings() const {
  NveBindings result;
  for (const auto& layer : impl_->layers) result.emplace(layer.module_path, layer.binding);
  return result;
}
std::unordered_map<std::string, at::Tensor> NveRuntimeDirectory::markers() const {
  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& layer : impl_->layers)
    result.emplace(nve::marker_fqn(layer.module_path), layer.marker);
  return result;
}
void NveRuntimeDirectory::bind(torch::inductor::AOTIModelPackageLoader& loader) const {
  const auto fqns = loader.get_constant_fqns();
  const std::set<std::string> available(fqns.begin(), fqns.end());
  auto constants = markers();
  for (const auto& [name, marker] : constants)
    TORCH_CHECK(available.count(name), "NVE marker is not an updatable AOTI constant: ", name);
  loader.load_constants(constants, false, false, true);
  loader.load_constants(constants, true, false, true);
}

}  // namespace dynamicemb::exportable_embedding
