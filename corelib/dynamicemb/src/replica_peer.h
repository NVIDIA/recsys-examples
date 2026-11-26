// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

namespace dyn_emb::replica {
void peer_transfer(const at::Tensor &table, const at::Tensor &mailbox,
                   const at::Tensor &rows, const at::Tensor &slots,
                   const at::Tensor &buckets,
                   const std::optional<at::Tensor> &peer_rows,
                   std::tuple<int64_t, bool> config);
void peer_copy_merge_async(const at::Tensor &table, const at::Tensor &inbox,
                           const at::Tensor &peer, const at::Tensor &rows,
                           const at::Tensor &peer_rows, const at::Tensor &slots,
                           const at::Tensor &buckets, at::IntArrayRef ends,
                           std::tuple<int64_t, int64_t, bool> layout,
                           c10::cuda::CUDAStream compute);
} // namespace dyn_emb::replica
