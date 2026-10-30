/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/
// The op's symmetric windows (ctrl, keys, rows): this rank's memory of each,
// allocated once at the sizes of the layout, then registered collectively;
// the regions' handles and offsets the kernels address them by (Win); and
// the counted views of lookup_rows the lookup writes into.
#pragma once

#include <ATen/ATen.h>
#include <nccl.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/host/layout.h"

namespace dyn_emb {
namespace hier_output_dist {

class SymmetricWindows {
 public:
  // Allocates this rank's memory of every window (local) and zeroes the
  // regions that must read 0 (kRegionZero). Throws if it cannot, with
  // nothing left allocated.
  SymmetricWindows(const Layout &layout, int device);
  SymmetricWindows(const SymmetricWindows &) = delete;
  SymmetricWindows &operator=(const SymmetricWindows &) = delete;
  // Frees nothing: the owner deregisters and frees (deregister(), free()), or
  // keeps the memory until the process exits when a peer may still read it.
  ~SymmetricWindows() = default;

  // Collective, in window order: once every rank allocated its memory.
  void register_all(ncclComm_t comm);
  const Win &win() const { return win_; }
  // This rank's memory of `region`.
  char *region(Region region) const;
  // The first `rows` rows of lookup_rows as a [rows, cols] tensor of `dtype`
  // (a row of lookup_rows), counted until its storage is freed; then
  // `on_drop` runs, on the thread that frees it.
  at::Tensor lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype,
                         std::function<void()> on_drop = {});
  int64_t live_views() const { return views_->load(); }
  // This rank's memory of each window, in whole granules of `granularity`.
  int64_t window_bytes(int window, size_t granularity) const;

  // Teardown steps, local: deregister every window, then (once the
  // communicator is gone) free their memory, except the rows window's while
  // a view of lookup_rows is alive.
  void deregister(ncclComm_t comm) noexcept;
  void free(bool keep_rows) noexcept;

 private:
  struct Window {
    char *buf = nullptr;
    uint64_t bytes = 0;
    ncclWindow_t handle = nullptr;  // nullptr until registered
  };
  void alloc(int window, const Layout &layout);

  int device_;
  int64_t row_bytes_;
  Window windows_[kNumWins];
  Win win_{};
  // Shared with the views' deleters, which may outlive this object.
  std::shared_ptr<std::atomic<int64_t>> views_ = std::make_shared<std::atomic<int64_t>>(0);
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
