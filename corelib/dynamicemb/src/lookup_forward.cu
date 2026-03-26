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

#include "check.h"
#include "lookup_forward.h"
#include "lookup_kernel.cuh"

namespace dyn_emb {

template <typename SrcType, typename DstType, typename offset_t>
struct ForwardMultiToOneFMLayoutDesc {
  using SrcT = SrcType;
  using DstT = DstType;

  HOST_DEVICE_INLINE int get_offset(int i) {
    return offset_ptr[i] - offset_ptr[0];
  }
  HOST_DEVICE_INLINE int get_vec_length(int i) {
    // TODO:now only have one size
    return ev_size;
  }
  HOST_DEVICE_INLINE int get_average_pooling_factor(int i) {
    int pooling_factor = static_cast<int>(offset_ptr[i + 1] - offset_ptr[i]);
    // TODO:now use 1 = Average, 0 = SUM or None
    return combiner == 1 ? pooling_factor : 1;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    int idx = reverse_idx_ptr[i];
    return src_ptr + ev_size * idx;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    int b = i % batch_size;
    int f = i / batch_size;
    return dst_ptr + b * total_D + accum_D + f * ev_size;
  }

  int num_vec_;
  int combiner;
  int ev_size;
  const offset_t *__restrict__ offset_ptr;
  const offset_t *__restrict__ reverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
  int batch_size;
  int total_D;
  int accum_D;
};

void scatter_combine(void *src_ptr, void *dst_ptr, void *offset_ptr,
                     void *inverse_idx_ptr, int combiner, int total_D,
                     int accum_D, int ev_size, int num_vec, int batch_size,
                     DataType src_type, DataType dst_type, DataType offset_type,
                     cudaStream_t stream) {

  DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, offset_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(src_type, src_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
        using CopyDesc = ForwardMultiToOneFMLayoutDesc<src_t, dst_t, offset_t>;
        CopyDesc multi_to_one_desc{num_vec,
                                   combiner,
                                   ev_size,
                                   (offset_t *)offset_ptr,
                                   (offset_t *)inverse_idx_ptr,
                                   (src_t *)src_ptr,
                                   (dst_t *)dst_ptr,
                                   batch_size,
                                   total_D,
                                   accum_D};
        copy_multi_to_one(multi_to_one_desc, ev_size, stream);
      });
    });
  });
}

template <typename SrcType, typename DstType, typename offset_t>
struct ForwardSequenceCopyDesc {

  using SrcT = SrcType;
  using DstT = DstType;
  HOST_DEVICE_INLINE int get_vec_length() {
    // TODO:now only have one size
    return ev_size;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    offset_t idx = reverse_idx_ptr[i];
    return src_ptr + idx * ev_size;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    offset_t offset = offset_ptr[0];
    return dst_ptr + (offset + i) * ev_size;
  }

  int num_vec_;
  int ev_size;
  const offset_t *__restrict__ offset_ptr;
  const offset_t *__restrict__ reverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
};

void scatter(void *src_ptr, void *dst_ptr, void *offset_ptr,
             void *inverse_idx_ptr, int num_emb, int ev_size, DataType src_type,
             DataType dst_type, DataType offset_type, int device_num_sms,
             cudaStream_t stream) {
  DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, offset_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(src_type, src_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
        using CopyDesc = ForwardSequenceCopyDesc<src_t, dst_t, offset_t>;
        CopyDesc sequence_copy_desc{num_emb,
                                    ev_size,
                                    (offset_t *)offset_ptr,
                                    (offset_t *)inverse_idx_ptr,
                                    (src_t *)src_ptr,
                                    (dst_t *)dst_ptr};
        copy_one_to_one<CopyDesc>(sequence_copy_desc, ev_size, device_num_sms,
                                  stream);
      });
    });
  });
}

template <typename SrcType, typename DstType, typename offset_t>
struct ForwardSequenceFusedCopyDesc {

  using SrcT = SrcType;
  using DstT = DstType;
  HOST_DEVICE_INLINE int get_vec_length() {
    // TODO:now only have one size
    return ev_size;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    offset_t idx = reverse_idx_ptr[i];
    return src_ptr + idx * ev_size;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    return dst_ptr + i * ev_size;
  }

  int num_vec_;
  int ev_size;
  const offset_t *__restrict__ reverse_idx_ptr;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
};

void scatter_fused(void *src_ptr, void *dst_ptr, void *inverse_idx_ptr,
                   int num_emb, int ev_size, DataType src_type,
                   DataType dst_type, DataType offset_type, int device_num_sms,
                   cudaStream_t stream) {
  DISPATCH_INTEGER_DATATYPE_FUNCTION(offset_type, offset_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(src_type, src_t, [&] {
      DISPATCH_FLOAT_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
        using CopyDesc = ForwardSequenceFusedCopyDesc<src_t, dst_t, offset_t>;
        CopyDesc sequence_copy_desc{num_emb, ev_size,
                                    (offset_t *)inverse_idx_ptr,
                                    (src_t *)src_ptr, (dst_t *)dst_ptr};
        copy_one_to_one<CopyDesc>(sequence_copy_desc, ev_size, device_num_sms,
                                  stream);
      });
    });
  });
}

template <typename SrcType, typename DstType> struct ForwardOneToOneCopyDesc {

  using SrcT = SrcType;
  using DstT = DstType;
  HOST_DEVICE_INLINE int get_vec_length() {
    // TODO:now only have one size
    return vec_length;
  }
  HOST_DEVICE_INLINE const SrcType *get_src_ptr(int i) {
    return src_ptr + i * vec_length;
  }
  HOST_DEVICE_INLINE DstType *get_dst_ptr(int i) {
    return dst_ptr + i * vec_length;
  }
  int num_vec_;
  int vec_length;
  const SrcType *__restrict__ src_ptr;
  DstType *dst_ptr;
};

void batched_vector_copy_device(void *src_ptr, void *dst_ptr, int batch_size,
                                int vec_length, DataType src_type,
                                DataType dst_type, int device_num_sms,
                                cudaStream_t stream) {
  DISPATCH_FLOAT_DATATYPE_FUNCTION(src_type, src_t, [&] {
    DISPATCH_FLOAT_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
      using CopyDesc = ForwardOneToOneCopyDesc<src_t, dst_t>;
      CopyDesc sequence_dedup_copy_desc{batch_size, vec_length,
                                        (src_t *)src_ptr, (dst_t *)dst_ptr};
      copy_one_to_one<CopyDesc>(sequence_dedup_copy_desc, vec_length,
                                device_num_sms, stream);
    });
  });
}

void add_offset(void *src_ptr, void *dst_ptr, int idx, DataType src_type,
                DataType dst_type, cudaStream_t stream) {
  DISPATCH_INTEGER_DATATYPE_FUNCTION(src_type, src_t, [&] {
    DISPATCH_INTEGER_DATATYPE_FUNCTION(dst_type, dst_t, [&] {
      add_offset_kernel<src_t, dst_t>
          <<<1, 1, 0, stream>>>(reinterpret_cast<const src_t *>(src_ptr),
                                reinterpret_cast<dst_t *>(dst_ptr), idx);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void get_new_length_and_offsets(uint64_t *d_unique_offsets,
                                int64_t *d_table_offsets_in_feature,
                                int table_num, int64_t new_lengths_size,
                                int local_batch_size, DataType length_type,
                                DataType offset_type, void *new_offsets,
                                void *new_lenghths, cudaStream_t stream) {

  int block_size = 256;
  int grid_size = (new_lengths_size + block_size - 1) / block_size;
  DISPATCH_OFFSET_INT_TYPE(offset_type, offset_t, [&] {
    DISPATCH_OFFSET_INT_TYPE(length_type, length_t, [&] {
      get_new_length_and_offsets_kernel<offset_t, length_t>
          <<<grid_size, block_size, 0, stream>>>(
              d_unique_offsets, d_table_offsets_in_feature, table_num,
              new_lengths_size, local_batch_size,
              reinterpret_cast<offset_t *>(new_offsets),
              reinterpret_cast<length_t *>(new_lenghths));
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
