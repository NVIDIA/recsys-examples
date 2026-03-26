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
#include "utils.h"

namespace dyn_emb {

void scatter_combine(void *src_ptr, void *dst_ptr, void *offset_ptr,
                     void *inverse_idx_ptr, int combiner, int total_D,
                     int accum_D, int ev_size, int num_vec, int batch_size,
                     DataType src_type, DataType dst_type, DataType offset_type,
                     cudaStream_t stream);

void scatter(void *src_ptr, void *dst_ptr, void *offset_ptr,
             void *inverse_idx_ptr, int num_emb, int ev_size, DataType src_type,
             DataType dst_type, DataType offset_type, int device_num_sms,
             cudaStream_t stream);

void scatter_fused(void *src_ptr, void *dst_ptr, void *inverse_idx_ptr,
                   int num_emb, int ev_size, DataType src_type,
                   DataType dst_type, DataType offset_type, int device_num_sms,
                   cudaStream_t stream);

void add_offset(void *src_ptr, void *dst_ptr, int idx, DataType src_type,
                DataType dst_type, cudaStream_t stream);

void get_new_length_and_offsets(uint64_t *d_unique_offsets,
                                int64_t *d_table_offsets_in_feature,
                                int table_num, int64_t new_lengths_size,
                                int local_batch_size, DataType length_type,
                                DataType offset_type, void *new_offsets,
                                void *new_lenghths, cudaStream_t stream);

void batched_vector_copy_device(void *src_ptr, void *dst_ptr, int batch_size,
                                int vec_length, DataType src_type,
                                DataType dst_type, int num_sms,
                                cudaStream_t stream);

} // namespace dyn_emb
#endif // LOOKUP_FORWARD_H
