// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "replica_sync_batch.h"

namespace dyn_emb::replica {
void validate(const at::Tensor &, const at::Tensor &, const at::Tensor &,
              const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t,
              const at::Tensor &);
int64_t wire_width(const Table &t);
at::Tensor count_touched(const Table &t, int64_t threshold);
void export_keys(const Table &t, int64_t threshold, const at::Tensor &output);
Route lookup(const Table &t, const at::Tensor &keys);
void pack_into(const Table &t, const at::Tensor &rows, const at::Tensor &slots,
               int64_t padded, const at::Tensor &output);
void merge_into(const Table &t, const at::Tensor &rows, const at::Tensor &slots,
                const at::Tensor &packets,
                const std::optional<at::Tensor> &indices,
                const at::Tensor &missing_values,
                const at::Tensor &missing_scores);
} // namespace dyn_emb::replica
