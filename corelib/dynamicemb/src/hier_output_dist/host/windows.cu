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
#include <c10/cuda/CUDAGuard.h>

#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/windows.h"

namespace dyn_emb {
namespace hier_output_dist {

SymmetricWindows::SymmetricWindows(const Layout &layout, int device)
    : device_(device), row_bytes_(layout.row_bytes) {
  c10::cuda::CUDAGuard guard(device_);
  for (int region = 0; region < kNumRegions; ++region) {
    win_.off[region] = layout.region_off[region];
    win_.bytes[region] = layout.region_bytes[region];
  }
  win_.lookup_keys_cap = layout.lookup_keys;
  win_.num_proxy_rows = layout.num_proxy_rows;
  win_.rounds_cap = layout.rounds_cap;
  try {
    for (int window = 0; window < kNumWins; ++window) alloc(window, layout);
  } catch (...) {
    free(false);
    throw;
  }
}

// The regions that must read 0 are zeroed on a private stream, synchronized
// before returning: the registration that follows ends in a barrier, so no
// peer reaches them before.
void SymmetricWindows::alloc(int window, const Layout &layout) {
  const uint64_t total = layout.window_bytes[window];
  char *buf = nullptr;
  HIER_NCCL_CHECK(ncclMemAlloc((void **)&buf, total));
  windows_[window].buf = buf;
  windows_[window].bytes = total;
  bool zero = false;
  for (int region = 0; region < kNumRegions; ++region)
    zero |= kRegionWin[region] == window && kRegionZero[region] && win_.bytes[region] > 0;
  if (!zero) return;
  cudaStream_t stream;
  HIER_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaError_t err = cudaSuccess;
  for (int region = 0; region < kNumRegions && err == cudaSuccess; ++region)
    if (kRegionWin[region] == window && kRegionZero[region])
      err = cudaMemsetAsync(buf + win_.off[region], 0, win_.bytes[region], stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  HIER_CUDA_CHECK(err);
}

void SymmetricWindows::register_all(ncclComm_t comm) {
  c10::cuda::CUDAGuard guard(device_);
  for (int window = 0; window < kNumWins; ++window) {
    Window &mine = windows_[window];
    HIER_NCCL_CHECK(
        ncclCommWindowRegister(comm, mine.buf, mine.bytes, &mine.handle, NCCL_WIN_DEFAULT));
    TORCH_CHECK(mine.handle != nullptr, "hier_output_dist: symmetric window registration failed");
    for (int region = 0; region < kNumRegions; ++region)
      if (kRegionWin[region] == window) win_.handle[region] = mine.handle;
  }
}

char *SymmetricWindows::region(Region region) const {
  return windows_[kRegionWin[region]].buf + win_.off[region];
}

at::Tensor SymmetricWindows::lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype,
                                         std::function<void()> on_drop) {
  TORCH_CHECK(cols * (int64_t)c10::elementSize(dtype) == row_bytes_,
              "hier_output_dist: a row of lookup_rows is ", row_bytes_, " bytes, not ", cols, " x ",
              dtype);
  TORCH_CHECK(rows >= 0 && rows <= win_.lookup_keys_cap, "hier_output_dist: ", rows,
              " rows are more than lookup_rows holds (", win_.lookup_keys_cap, ")");
  auto views = views_;
  views->fetch_add(1);
  return at::from_blob(
      region(kLookupRows), {rows, cols},
      [views, on_drop](void *) {
        if (on_drop) on_drop();
        views->fetch_sub(1);
      },
      at::TensorOptions().device(at::kCUDA, device_).dtype(dtype));
}

int64_t SymmetricWindows::window_bytes(int window, size_t granularity) const {
  return (int64_t)((windows_[window].bytes + granularity - 1) / granularity * granularity);
}

// Deregistration is local: a node peer keeps its mapping of this rank's
// memory until it deregisters its own window.
void SymmetricWindows::deregister(ncclComm_t comm) noexcept {
  for (int window = kNumWins - 1; window >= 0; --window) {
    if (windows_[window].handle == nullptr) continue;
    ncclCommWindowDeregister(comm, windows_[window].handle);
    windows_[window].handle = nullptr;
  }
}

void SymmetricWindows::free(bool keep_rows) noexcept {
  for (int window = kNumWins - 1; window >= 0; --window) {
    if (windows_[window].buf == nullptr || (keep_rows && window == kWinRows)) continue;
    ncclMemFree(windows_[window].buf);
    windows_[window].buf = nullptr;
  }
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
