// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <vector>

namespace c10d {
class ProcessGroup;
}

namespace dyn_emb::replica {
using Layout = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;
struct Table {
  at::Tensor values, buckets, row_map, sizes, fresh;
  Layout layout; // embedding/optimizer widths, optimizer bytes, bucket
                 // capacity, scores
  bool constant;
  int64_t global_capacity;
};
struct Route {
  at::Tensor keys, rows,
      slots;           // signed key bits; negative rows and slots mean absent
  bool dense = false;  // every replica holds every union key
  bool cached = false; // validated previous key/row/slot triple
};

// The peer transport owns its state format. It sets done[t] for every table
// it merged. All other tables are handled by the packed NCCL path.
std::vector<at::Tensor>
batch_peer(const std::vector<Table> &tables, const std::vector<Route> &routes,
           const std::vector<at::Tensor> &peer_state,
           const c10::intrusive_ptr<c10d::ProcessGroup> &group,
           int64_t max_staging_bytes, std::vector<bool> &done);
} // namespace dyn_emb::replica
