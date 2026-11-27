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

// Host-side views of tables of different embedding dimensions: what an entry
// point takes in place of a run of loose tensors and ints. See
// docs/mixed_dim_view_design.md.
//
//   PaddedBufferView  a padded [n, max_value_dim] buffer; only the per-table
//                     dims (an external storage has no pointers).
//   FlatTablesView    one buffer per table; also the pointers, the row widths
//                     and the element type.
//
// Both hold tensors by reference, never copy data, and check only what needs
// no device synchronization: dtype, device, contiguity, lengths. That
// max_emb_dim covers emb_dims and that all_dims_vec4 is right is the caller's
// to guarantee (MixedDimBuffers computes both from host-side lists).

#pragma once

#include "mixed_dim_params.cuh"
#include "torch_utils.h"

namespace dyn_emb {

namespace detail {

// A per-table tensor: CUDA, int64, 1-D, contiguous.
inline void check_per_table_tensor(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kLong, name, " must be int64, got ",
              t.scalar_type());
  TORCH_CHECK(t.dim() == 1, name, " must be 1-D, got ", t.dim(), "-D");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline void check_row_dims(int64_t max_emb_dim, int64_t max_value_dim) {
  TORCH_CHECK(max_emb_dim > 0, "max_emb_dim must be positive, got ",
              max_emb_dim);
  TORCH_CHECK(max_value_dim >= max_emb_dim, "max_value_dim (", max_value_dim,
              ") must cover max_emb_dim (", max_emb_dim, ")");
}

} // namespace detail

class PaddedBufferView {
public:
  PaddedBufferView(at::Tensor emb_dims, int64_t max_emb_dim,
                   int64_t max_value_dim, bool all_dims_vec4)
      : emb_dims_(std::move(emb_dims)), max_emb_dim_(max_emb_dim),
        max_value_dim_(max_value_dim), all_dims_vec4_(all_dims_vec4) {
    detail::check_per_table_tensor(emb_dims_, "emb_dims");
    TORCH_CHECK(emb_dims_.numel() > 0, "emb_dims must not be empty");
    detail::check_row_dims(max_emb_dim_, max_value_dim_);
  }

  PaddedBufferParams params() const {
    return PaddedBufferParams{get_pointer<int64_t>(emb_dims_), max_emb_dim_,
                              max_value_dim_};
  }

  const at::Tensor &emb_dims() const { return emb_dims_; }
  int64_t num_tables() const { return emb_dims_.numel(); }
  int64_t max_emb_dim() const { return max_emb_dim_; }
  int64_t max_value_dim() const { return max_value_dim_; }
  bool all_dims_vec4() const { return all_dims_vec4_; }

private:
  at::Tensor emb_dims_;
  int64_t max_emb_dim_;
  int64_t max_value_dim_;
  bool all_dims_vec4_;
};

class FlatTablesView {
public:
  FlatTablesView(at::Tensor ptrs, at::Tensor value_dims, at::Tensor emb_dims,
                 int64_t max_emb_dim, int64_t max_value_dim,
                 bool all_dims_vec4, at::ScalarType dtype)
      : ptrs_(std::move(ptrs)), value_dims_(std::move(value_dims)),
        emb_dims_(std::move(emb_dims)), max_emb_dim_(max_emb_dim),
        max_value_dim_(max_value_dim), all_dims_vec4_(all_dims_vec4),
        dtype_(dtype) {
    detail::check_per_table_tensor(ptrs_, "ptrs");
    detail::check_per_table_tensor(value_dims_, "value_dims");
    detail::check_per_table_tensor(emb_dims_, "emb_dims");
    TORCH_CHECK(ptrs_.numel() > 0, "ptrs must not be empty");
    TORCH_CHECK(value_dims_.numel() == ptrs_.numel() &&
                    emb_dims_.numel() == ptrs_.numel(),
                "ptrs, value_dims and emb_dims must have one entry per table; "
                "got ",
                ptrs_.numel(), ", ", value_dims_.numel(), ", ",
                emb_dims_.numel());
    TORCH_CHECK(value_dims_.device() == ptrs_.device() &&
                    emb_dims_.device() == ptrs_.device(),
                "ptrs, value_dims and emb_dims must be on the same device");
    detail::check_row_dims(max_emb_dim_, max_value_dim_);
    // Throws on an element type no kernel dispatches on.
    (void)scalartype_to_datatype(dtype_);
  }

  FlatTablesParams params() const {
    return FlatTablesParams{get_pointer<int64_t>(ptrs_),
                            get_pointer<int64_t>(value_dims_),
                            get_pointer<int64_t>(emb_dims_), max_emb_dim_,
                            max_value_dim_};
  }

  const at::Tensor &ptrs() const { return ptrs_; }
  const at::Tensor &value_dims() const { return value_dims_; }
  const at::Tensor &emb_dims() const { return emb_dims_; }
  int64_t num_tables() const { return ptrs_.numel(); }
  int64_t max_emb_dim() const { return max_emb_dim_; }
  int64_t max_value_dim() const { return max_value_dim_; }
  bool all_dims_vec4() const { return all_dims_vec4_; }
  // The type of the embedding part of a row (an fp8 table is Float8_e4m3fn,
  // i.e. dyn_fp8_t). The optimizer-state part is typed by the optimizer.
  at::ScalarType dtype() const { return dtype_; }
  DataType data_type() const { return scalartype_to_datatype(dtype_); }

private:
  at::Tensor ptrs_;
  at::Tensor value_dims_;
  at::Tensor emb_dims_;
  int64_t max_emb_dim_;
  int64_t max_value_dim_;
  bool all_dims_vec4_;
  at::ScalarType dtype_;
};

} // namespace dyn_emb
