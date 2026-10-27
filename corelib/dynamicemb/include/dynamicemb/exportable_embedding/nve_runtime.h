// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dynamicemb/exportable_embedding/indexer_directory.h"

namespace nve {
template <class IndexT> class NVEmbedBinding;
}

namespace dynamicemb::exportable_embedding {

using NveBinding = nve::NVEmbedBinding<int64_t>;
using NveBindings = std::unordered_map<std::string, std::shared_ptr<NveBinding>>;

DYNAMICEMB_EXPORT bool has_nve_tier_update(const std::shared_ptr<NveBinding>& layer);
DYNAMICEMB_EXPORT void update_nve_tier(
    const std::shared_ptr<NveBinding>& layer, size_t count, uintptr_t keys,
    uintptr_t values, int64_t tier, uint64_t stream);

struct NveRuntimeLayer {
  std::string module_path;
  std::shared_ptr<NveBinding> binding;
  at::Tensor marker;
};

// Reconstructs the PR's NVE 26.10 v2 packages with local tier-capable bindings.
// Own this directory until after the AOTI loader is destroyed. Host keeps its
// stock implementation (including the separate concurrent-row NVE patch).
class DYNAMICEMB_EXPORT NveRuntimeDirectory {
 public:
  NveRuntimeDirectory(const std::string& package_dir, int device_index);
  ~NveRuntimeDirectory();
  NveRuntimeDirectory(const NveRuntimeDirectory&) = delete;
  NveRuntimeDirectory& operator=(const NveRuntimeDirectory&) = delete;

  const std::vector<NveRuntimeLayer>& layers() const;
  NveBindings bindings() const;
  std::unordered_map<std::string, at::Tensor> markers() const;
  void bind(torch::inductor::AOTIModelPackageLoader& loader) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dynamicemb::exportable_embedding
