// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "python/pynve/bindings/binding_layers.hpp"

namespace dynamicemb::exportable_embedding {

// Construct the derived type from the outset: no casts of stock objects and no
// replacement cache. NVE 26.10 already supports tier selection below its binding.
template <class StockBinding>
class TierUpdateBinding : public StockBinding {
 public:
  using StockBinding::StockBinding;

  void update_tier(size_t count, uintptr_t keys, uintptr_t values,
                   int64_t tier, uint64_t stream) {
    NVE_CHECK_(tier >= 0 && tier < this->emb_layer_ptr_->get_num_tables(),
               "Invalid update tier");
    auto ctx = this->get_exec_context(reinterpret_cast<cudaStream_t>(stream));
    this->emb_layer_ptr_->update(
        ctx, count, reinterpret_cast<const void*>(keys),
        this->row_size_in_bytes_, this->row_size_in_bytes_,
        reinterpret_cast<const void*>(values), tier);
  }
};

using TierGPUEmbedding = TierUpdateBinding<nve::GPUEmbedding<int64_t>>;
using TierUVMEmbedding = TierUpdateBinding<nve::LinearUVMEmbedding<int64_t>>;
using TierHierarchicalEmbedding = TierUpdateBinding<nve::HierarchicalEmbedding<int64_t>>;

}  // namespace dynamicemb::exportable_embedding
