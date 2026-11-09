/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef LOOKUP_FORWARD_H
#define LOOKUP_FORWARD_H
#include "lookup_kernel.cuh"
#include "pooled_layout.cuh"
#include "utils.h"

namespace dyn_emb {

// Unified pooled-gather descriptor.  One CTA/warp per bag, indexed in the
// input's order; ``layout`` maps that to the output's columns.  Source rows use
// src_stride, which exceeds the copy width when optimizer state follows the
// embedding in a row.
template <typename SrcType, typename DstType, typename IndexType>
struct ForwardMultiToOneFMLayoutDesc {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) { return offset_ptr[i]; }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    return layout.col_width(layout.feature_of_input(i));
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) {
    int pooling_factor = static_cast<int>(offset_ptr[i + 1] - offset_ptr[i]);
    return pooling_mode == PoolingMode::kMean ? pooling_factor : 1;
  }
  HOST_DEVICE_INLINE float get_weight(int i) {
    // nullptr => unweighted pooling (identical to the old path).
    return weights_ptr ? weights_ptr[i] : 1.0f;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    int idx = inverse_idx_ptr[i];
    return src_ptr + (int64_t)src_stride * idx;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    const int f = layout.feature_of_input(i);
    const int b = layout.sample_of_input(i);
    return dst_ptr + b * layout.total_D + layout.col_begin(f);
  }

  PoolingMode pooling_mode;
  PooledLayout layout;
  int src_stride; // source row stride; exceeds layout.max_D when optimizer
                  // states are appended to each row
  const IndexType *__restrict__ offset_ptr;
  // torch.unique's inverse mapping: for input position i, the row of the
  // deduplicated table that position reads.  Most of the codebase still spells
  // this "reverse"; it is the same array.
  const IndexType *__restrict__ inverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
  const float *__restrict__ weights_ptr; // nullptr -> unweighted
};

// Unified pooled gather (scatter-combine).  ``layout`` carries the shapes, the
// bag numbering and the copy width; see pooled_layout.cuh.  ``src_stride`` is
// the source row stride, which exceeds the copy width when optimizer state is
// appended to each row.  ``IndexType`` is shared by the offsets and the inverse
// indices, which the descriptor requires to have the same width.
template <typename SrcType, typename DstType, typename IndexType>
void scatter_combine(const SrcType *src_ptr, DstType *dst_ptr,
                     const IndexType *offset_ptr,
                     const IndexType *inverse_idx_ptr, PoolingMode pooling_mode,
                     const PooledLayout &layout, int src_stride,
                     bool feature_dims_vec4, cudaStream_t stream,
                     const float *weights_ptr = nullptr) {
  ForwardMultiToOneFMLayoutDesc<SrcType, DstType, IndexType> desc{
      pooling_mode, layout,  src_stride,  offset_ptr, inverse_idx_ptr,
      src_ptr,      dst_ptr, weights_ptr};
  copy_multi_to_one(desc, feature_dims_vec4, stream);
}

// Row i of dst (row out_row[i] with out_row, int32) gets row inverse_idx[i]
// of src.
void scatter_fused(void *src_ptr, void *dst_ptr, void *inverse_idx_ptr,
                   int num_emb, int ev_size, int src_stride, DataType src_type,
                   DataType dst_type, DataType offset_type, int device_num_sms,
                   cudaStream_t stream, const int32_t *out_row = nullptr);

void get_new_length_and_offsets(uint64_t *d_unique_offsets,
                                int64_t *d_table_offsets_in_feature,
                                int table_num, int64_t new_lengths_size,
                                int local_batch_size, DataType length_type,
                                DataType offset_type, void *new_offsets,
                                void *new_lenghths, cudaStream_t stream);

} // namespace dyn_emb
#endif // LOOKUP_FORWARD_H
