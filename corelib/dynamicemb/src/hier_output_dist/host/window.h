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
// The op's window (common/window.h): this rank's copy of its symmetric
// memory, allocated once at the op's sizes, then registered collectively;
// its layout over that copy, which the host views and reads its buffers by;
// and the counted views of lookup_rows the lookup writes into.
#pragma once

#include <ATen/ATen.h>
#include <nccl.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

#include "hier_output_dist/common/sizes.h"
#include "hier_output_dist/common/window.h"

namespace dyn_emb {
namespace hier_output_dist {

class SymmetricWindow {
 public:
  // Allocates this rank's copy (local) and zeroes what must read 0 before
  // the window is registered: the ctrl section (sequence numbers start at
  // 0) and push_flags (tags start at 0). Every other buffer is written
  // before it is read, so it is never initialized. Throws if it cannot,
  // with nothing left allocated.
  SymmetricWindow(const Sizes &sizes, int device);
  SymmetricWindow(const SymmetricWindow &) = delete;
  SymmetricWindow &operator=(const SymmetricWindow &) = delete;
  // Frees nothing: the owner deregisters and frees (deregister(), free()),
  // or keeps the memory until the process exits when a peer may still read
  // it.
  ~SymmetricWindow() = default;

  // Collective: once every rank allocated its copy.
  void register_window(ncclComm_t comm);
  // The registered window, which the kernels take (nullptr before).
  ncclWindow_t handle() const { return handle_; }
  // The window laid out over this rank's copy.
  WindowLayout layout() const { return WindowLayout(sizes_, base_); }
  // The first `rows` rows of lookup_rows as a [rows, cols] tensor of `dtype`
  // (a row of lookup_rows), counted until its storage is freed; then
  // `on_drop` runs, on the thread that frees it.
  at::Tensor lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype,
                         std::function<void()> on_drop = {});
  int64_t live_views() const { return views_->load(); }
  // This rank's copy, in whole granules of `granularity`.
  int64_t bytes(size_t granularity) const;

  // Teardown steps, local: deregister the window, then (once the
  // communicator is gone) free its memory, unless `keep`: a view of
  // lookup_rows is alive, so the whole window stays until the process exits.
  void deregister(ncclComm_t comm) noexcept;
  void free(bool keep) noexcept;

 private:
  int device_;
  Sizes sizes_;
  uint64_t bytes_;                 // as allocated (WindowLayout::bytes)
  char *base_ = nullptr;           // this rank's copy
  ncclWindow_t handle_ = nullptr;  // nullptr until registered
  // Shared with the views' deleters, which may outlive this object.
  std::shared_ptr<std::atomic<int64_t>> views_ = std::make_shared<std::atomic<int64_t>>(0);
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
