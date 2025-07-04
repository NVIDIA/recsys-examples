/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

namespace flash {

template <typename Kernel_traits, typename Params>
struct HstuBlockInfo {
  __device__ HstuBlockInfo(const Params& params, const int bidb)
      : sum_s_q(params.cu_seqlens_q[bidb]),
        sum_s_k(params.cu_seqlens_k[bidb]),
        sum_s_t((Kernel_traits::Paged_KV && params.cu_seqlens_t == nullptr) ? 0 : params.cu_seqlens_t[bidb]),
        sum_s_page((Kernel_traits::Paged_KV && params.page_offsets == nullptr) ? 0 : params.page_offsets[bidb]),
        actual_seqlen_c((Kernel_traits::Is_context && params.num_contexts == nullptr) ? 0 : params.num_contexts[bidb]),
        actual_seqlen_q(params.cu_seqlens_q[bidb + 1] - sum_s_q),
        actual_seqlen_k(params.cu_seqlens_k[bidb + 1] - sum_s_k),
        actual_seqlen_t((Kernel_traits::Is_target && params.num_targets == nullptr) ? 0 : params.num_targets[bidb]),
        actual_page_num((Kernel_traits::Paged_KV && params.page_offsets == nullptr) ? 0 : params.page_offsets[bidb + 1] - sum_s_page),
        last_page_seqlen((Kernel_traits::Paged_KV && params.last_page_lens == nullptr) ? 0 : params.last_page_lens[bidb]) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(const index_t row_stride) const {
    return uint32_t(sum_s_k) * row_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t kv_cache_offset(const index_t row_stride) const {
    return uint32_t(sum_s_q) * row_stride + (actual_seqlen_q - actual_seqlen_t) * row_stride; // base: (new history + target), offset: new history
  }

  const int sum_s_q;
  const int sum_s_k;
  const int sum_s_t;
  const int sum_s_page;

  const int actual_seqlen_c;
  const int actual_seqlen_q;
  const int actual_seqlen_k;
  const int actual_seqlen_t;
  const int actual_page_num;
  const int last_page_seqlen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
