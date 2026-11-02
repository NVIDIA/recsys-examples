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
#include "hier_output_dist/host/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// What must read 0 is zeroed on a private stream, synchronized before
// returning: the registration that follows ends in a barrier, so no peer
// reaches it before.
SymmetricWindow::SymmetricWindow(const Sizes &sizes, int device)
    : device_(device), sizes_(sizes), bytes_(WindowLayout::bytes(sizes)) {
  c10::cuda::CUDAGuard guard(device_);
  char *base = nullptr;
  HIER_NCCL_CHECK(ncclMemAlloc((void **)&base, bytes_));
  base_ = base;
  const WindowLayout window = layout();
  cudaStream_t stream;
  cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (err == cudaSuccess) {
    err = cudaMemsetAsync(window.ctrl.base(), 0, window.ctrl.bytes(), stream);
    if (err == cudaSuccess && window.rows.push_flags_bytes() > 0)
      err = cudaMemsetAsync(window.rows.push_flags(0), 0, window.rows.push_flags_bytes(), stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  if (err != cudaSuccess) free(false);
  HIER_CUDA_CHECK(err);
}

void SymmetricWindow::register_window(ncclComm_t comm) {
  c10::cuda::CUDAGuard guard(device_);
  HIER_NCCL_CHECK(ncclCommWindowRegister(comm, base_, bytes_, &handle_, NCCL_WIN_DEFAULT));
  TORCH_CHECK(handle_ != nullptr, "hier_output_dist: symmetric window registration failed");
}

at::Tensor SymmetricWindow::lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype,
                                        std::function<void()> on_drop) {
  TORCH_CHECK(cols * (int64_t)c10::elementSize(dtype) == sizes_.row_bytes,
              "hier_output_dist: a row of lookup_rows is ", sizes_.row_bytes, " bytes, not ", cols,
              " x ", dtype);
  TORCH_CHECK(rows >= 0 && rows <= sizes_.lookup_keys, "hier_output_dist: ", rows,
              " rows are more than lookup_rows holds (", sizes_.lookup_keys, ")");
  auto views = views_;
  views->fetch_add(1);
  return at::from_blob(
      layout().rows.lookup_rows(), {rows, cols},
      [views, on_drop](void *) {
        if (on_drop) on_drop();
        views->fetch_sub(1);
      },
      at::TensorOptions().device(at::kCUDA, device_).dtype(dtype));
}

int64_t SymmetricWindow::bytes(size_t granularity) const {
  return (int64_t)((bytes_ + granularity - 1) / granularity * granularity);
}

// Deregistration is local: a node peer keeps its mapping of this rank's copy
// until it deregisters its own window.
void SymmetricWindow::deregister(ncclComm_t comm) noexcept {
  if (handle_ == nullptr) return;
  ncclCommWindowDeregister(comm, handle_);
  handle_ = nullptr;
}

void SymmetricWindow::free(bool keep) noexcept {
  if (base_ == nullptr || keep) return;
  ncclMemFree(base_);
  base_ = nullptr;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
