/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "hstu.h"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "block_info.h"
#include "kernel_traits.h"
#include "static_switch.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock(const Params& params,
                                                   const int bidb,
                                                   const int bidh,
                                                   int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;
  constexpr bool Is_causal = Kernel_traits::Is_causal;
  constexpr bool Is_target = Kernel_traits::Is_target;
  constexpr bool Is_context = Kernel_traits::Is_context;

  constexpr bool Is_even_rab = Kernel_traits::Is_even_rab;
  constexpr bool Is_local = Kernel_traits::Is_local;
  constexpr bool Has_rab = Kernel_traits::Has_rab;

  constexpr bool Paged_KV = Kernel_traits::Paged_KV;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  // constexpr int kStages = Kernel_traits::kStages;

  const HstuBlockInfo<Kernel_traits, Params> binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) {
    return;
  }

  const int actual_seqlen_q = binfo.actual_seqlen_q;
  const int actual_seqlen_k = binfo.actual_seqlen_k;
  // Actual target length of this sequence
  const int actual_seqlen_t = Is_target ? binfo.actual_seqlen_t : 0;
  // Actual context length of this sequence
  const int actual_seqlen_c = Is_context ? binfo.actual_seqlen_c : 0;
  // Actual history length of this sequence
  const int actual_seqlen_h = Is_target ? actual_seqlen_k - actual_seqlen_t : actual_seqlen_k;
  const int actual_seqlen_offset = actual_seqlen_k - actual_seqlen_q;
  // Paged KV
  const int last_page_seqlen = binfo.last_page_seqlen;
  const int page_offset = binfo.sum_s_page;

  const bool is_jump = Is_target && m_block * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_target = Is_target && (m_block + 1) * kBlockM + actual_seqlen_offset > actual_seqlen_h;
  const bool is_in_context = Is_context && (m_block + 1) * kBlockM <= actual_seqlen_c;
  const bool is_in_mixed_context = Is_context && (m_block + 1) * kBlockM > actual_seqlen_c && m_block * kBlockM < actual_seqlen_c;

  const bool is_in_paged_target = is_in_target && Paged_KV;
  const int last_page_offset = is_in_paged_target ? kBlockN - last_page_seqlen : 0;

  const int n_block_history = cute::ceil_div(actual_seqlen_h, kBlockN);
  const int target_index = (m_block * kBlockM - actual_seqlen_h) * params.target_group_size_inv;
  const int n_block_paged = Paged_KV ? n_block_history : 0;
  const int n_block_target = cute::ceil_div(actual_seqlen_t, kBlockN);

  // calculate n_block_min and n_block_max
  int n_block_min = !Is_local ? 0
        : std::max(0, (m_block * kBlockM + actual_seqlen_offset - params.window_size_left) / kBlockN);
  int n_block_max = Paged_KV ? n_block_history + n_block_target : cute::ceil_div(actual_seqlen_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    int offset = (m_block + 1) * kBlockM + actual_seqlen_offset + params.window_size_right;
    if (is_in_paged_target) { offset += last_page_offset; }
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div(offset, kBlockN));
  }
  if constexpr (Is_context) {
    n_block_min = (is_in_context || is_in_mixed_context) ? 0 : n_block_min;
    n_block_max = (is_in_context || is_in_mixed_context) ? std::max(n_block_history, n_block_max) : n_block_max;
  }
  // calculate n_masking_block_max and n_masking_block_min
  int n_masking_block_max = cute::ceil_div(std::min(actual_seqlen_k + last_page_offset, (m_block + 1) * kBlockM + actual_seqlen_offset + last_page_offset), kBlockN);
  int n_masking_block_min = (m_block * kBlockM + actual_seqlen_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = is_jump ? (actual_seqlen_h + actual_seqlen_offset + target_index * params.target_group_size + last_page_offset) / kBlockN : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min = is_in_mixed_context ? n_block_min : n_masking_block_min;
    n_masking_block_max = is_in_mixed_context ? n_block_max : n_masking_block_max;
  }
  const int n_masking_steps = (!Is_causal || is_in_context) ? 0 : n_masking_block_max - n_masking_block_min;

  // We exit early and write 0 to gO. This also covers the case where
  // actual_seqlen_k == 0. Otherwise we might read OOB elements from gK and gV.
  if ((Is_causal || Is_local) && n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                      binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gO), size<1>(gO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, actual_seqlen_q - m_block * kBlockM);
    return;
  }

  // We iterate over the blocks in reverse order.
  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) +
                                binfo.q_offset(params.q_row_stride)),
                  make_shape(actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));
  Tensor mK =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) +
                                (Paged_KV ? binfo.kv_cache_offset(params.k_row_stride) : binfo.k_offset(params.k_row_stride))),
                  make_shape(Paged_KV ? actual_seqlen_t : actual_seqlen_k, params.h_k, params.d),
                  make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));
  Tensor mV =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) +
                                (Paged_KV ? binfo.kv_cache_offset(params.v_row_stride) : binfo.k_offset(params.v_row_stride))),
                  make_shape(Paged_KV ? actual_seqlen_t : actual_seqlen_k, params.h_k, params.d),
                  make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));

  Tensor mKV_page = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.kv_cache_ptr)),
                               make_shape(params.total_pages, 2, params.page_size, params.h_k, params.d),
                               make_stride(params.kv_cache_kvtensor_stride, params.kv_cache_page_stride, params.kv_cache_head_stride, params.kv_cache_row_stride, _1{}));

  Tensor gK_page = local_tile(mKV_page(_, 0, _, bidh / params.h_h_k_ratio, _),
                        Shape<_1, Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0, 0));

  Tensor gV_page = local_tile(mKV_page(_, 1, _, bidh / params.h_h_k_ratio, _),
                        Shape<_1, Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0, 0));

  const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
  size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride + bidh_rab * params.rab_seqlen_q_stride + params.seqlen_k_rounded * actual_seqlen_offset;
  auto mRab =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.rab_ptr) + rab_qkv_not_equal_offset),
                  make_shape(actual_seqlen_q, params.seqlen_k_rounded),
                  make_stride(params.rab_seqlen_k_stride, _1{}));
  auto gRab = local_tile(mRab(_, _),
                        make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                        make_coord(m_block, _));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK =
      make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sRab = make_tensor(sV.data() + size(sV),
                            typename Kernel_traits::SmemLayoutRab{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  Tensor tKgK_page = gmem_thr_copy_QKV.partition_S(gK_page(0, _, _, _));
  Tensor tVgV_page = gmem_thr_copy_QKV.partition_S(gV_page(0, _, _, _));

  auto tQgRab = gmem_thr_copy_Rab.partition_S(gRab);
  auto tQsRab = gmem_thr_copy_Rab.partition_D(sRab);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle(_, _, _0{}));
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  //
  // Copy Atom retiling
  //
  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  auto smem_tiled_copy_rab =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
  auto tSsRab = smem_thr_copy_rab.partition_S(sRab);

  //
  // PREDICATES
  //
  // c = coord
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

  auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
  auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);

  auto copy_if_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
    #pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      if (get<0>(tQcRab(0, m, 0)) < (actual_seqlen_q - m_block * kBlockM)) {
        #pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
          if (Is_even_rab || get<1>(tQcRab(0, m, k)) < (actual_seqlen_k - n_block_id * kBlockN)) {
            cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
          }
        }
      }
    }
  };

  auto copy_g2s_rab = [&](int n_block_id, int buffer_stage) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
    #pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      #pragma unroll
      for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
        if (Is_even_rab || get<0>(tQcRab(0, m, k)) < (actual_seqlen_q - m_block * kBlockM)) {
          cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k), tQsRab(_, m, k, buffer_stage));
        }
      }
    }
  };

  // Prologue
  int n_block = n_block_max - 1;
  int buffer_stage = 0;

  if constexpr (Has_rab) {
    copy_if_g2s_rab(n_block, buffer_stage);
  }

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  // prefill q
  flash::copy</*Is_even_MN=*/false>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ,
                          actual_seqlen_q - m_block * kBlockM);
  if constexpr (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  if constexpr (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  // prefill k
  auto tKsK_stage_view = tKsK(_, _, _, buffer_stage);
  bool is_paged_tile = (n_block < n_block_paged) && Paged_KV;
  if (!is_paged_tile) {
    flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - n_block_paged), tKsK_stage_view, tKVcKV,
        (Paged_KV ? actual_seqlen_t : actual_seqlen_k) - (n_block - n_block_paged) * kBlockN);
  } else {
    flash::copy</*Is_even_MN=*/true, /*Clear_OOB_MN=*/false>(
        gmem_tiled_copy_QKV, tKgK_page(_, _, _, params.page_ids[page_offset + n_block]), tKsK_stage_view, tKVcKV);
  }
  cute::cp_async_fence();

  if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  clear(acc_o);

  auto col_limit_right = [&](int row) {
    return std::min(
        actual_seqlen_k,
        row + 1 + params.window_size_right
    );
  };
  auto col_limit_left = [&](int row) {
    return std::max(
        0,
        row - params.window_size_left
    );
  };

  auto apply_mask = [&](auto& tensor, int n_block) {
    static constexpr int Row = 0, Col = 1;
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    const int base_row = m_block * kBlockM + actual_seqlen_offset;
    const int base_col = n_block * kBlockN;
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
      int row = int(get<Row>(tScS(i))) + base_row;
      int col = int(get<Col>(tScS(i))) + base_col;
      if (Paged_KV && row >= actual_seqlen_h) {
        col -= last_page_offset;
      }
      if constexpr (!Is_causal && !Is_local) {
        if (col >= actual_seqlen_k) {
          tensor(i) = -INFINITY;
        }
      } else {
        if constexpr (Is_context) {
          if (row < actual_seqlen_c && col < actual_seqlen_h) {
            continue;
          }
        }
        // causal mask
        if (col >= col_limit_right(row)) {
          tensor(i) = -INFINITY;
          continue;
        }
        if constexpr (Is_local) {
          if (col < col_limit_left(row)) {
            tensor(i) = -INFINITY;
            continue;
          }
        }
        if constexpr (Is_target) {
          int target_index = (row - actual_seqlen_h) * params.target_group_size_inv;
          int target_col_limit_left = actual_seqlen_h + target_index * params.target_group_size;
          if (row >= actual_seqlen_h && (col + (Paged_KV ? last_page_offset : 0)) >= actual_seqlen_h && col < target_col_limit_left) {
            tensor(i) = -INFINITY;
          }
        }
      }
    }
  };

  for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
    // When jumps occur, it is necessary to apply a mask for the mixed situation
    const bool is_masking = masking_step < n_masking_steps || (n_block + 1) * kBlockN > actual_seqlen_h;
    Tensor acc_s = partition_fragment_C(
        tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    flash::cp_async_wait<0>();
    __syncthreads();

    // async load(v)
    auto tVsV_stage_view = tVsV(_, _, _, buffer_stage);
    bool is_paged_tile = (n_block < n_block_paged) && Paged_KV;
    if (masking_step > 0) {
      flash::copy</*Is_even_MN=*/true>(
          gmem_tiled_copy_QKV, is_paged_tile ? tVgV_page(_, _, _, params.page_ids[page_offset + n_block]) : tVgV(_, _, _, n_block - n_block_paged), tVsV_stage_view, tKVcKV);
    } else {
      if (!is_paged_tile) {
        flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block - n_block_paged), tVsV_stage_view, tKVcKV, (Paged_KV ? actual_seqlen_t : actual_seqlen_k) - (n_block - n_block_paged) * kBlockN);
      } else {
        flash::copy</*Is_even_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV_page(_, _, _, params.page_ids[page_offset + n_block]), tVsV_stage_view, tKVcKV);
      }
    }
    cute::cp_async_fence();

    // compute q @ k + rab
    int n_block_prev = n_block;
    if constexpr (Has_rab) {
      Tensor rRab = make_tensor<Element>(
          partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
      auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
      cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, buffer_stage), tSrRab_view(_, _, _));
      flash::convert_type_safe(rRab, acc_s);
      if (n_block > n_block_min) {
        if (is_jump && masking_step == n_masking_steps - 1) {
          n_block = std::min(n_block, n_block_history);
        }
        copy_g2s_rab(n_block - 1, buffer_stage);
      }
    } else {
      clear(acc_s);
    }
    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
        acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_Q,
        smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    if (Is_local || is_masking) {
      apply_mask(acc_s, n_block_prev);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if constexpr (!Has_rab) {
      if (is_jump && masking_step == n_masking_steps - 1) {
        n_block = std::min(n_block, n_block_history);
      }
    }
    if (n_block > n_block_min) {
      // async load(next(k))
      bool is_paged_tile = (n_block - 1 < n_block_paged) && Paged_KV;
      auto tKsK_stage_view_next = tKsK(_, _, _, buffer_stage);
      flash::copy</*Is_even_MN=*/true>(gmem_tiled_copy_QKV,
                                      is_paged_tile ? tKgK_page(_, _, _, params.page_ids[page_offset + n_block - 1]) : tKgK(_, _, _, n_block - 1 - n_block_paged),
                                      tKsK_stage_view_next, tKVcKV);
      cute::cp_async_fence();
    }

    for (int i = 0; i < size(acc_s); ++i) {
      acc_s(i) *= params.alpha;
    }
    fast_silu(acc_s);

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = make_tensor_like<Element>(acc_s);
    flash::convert_type_safe(acc_s, rP);

    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2) if using m16n8k16
    // or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

    // compute qk @ v
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt(_, _, _, buffer_stage), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  }

  // scale acc_o
  for (int i = 0; i < size(acc_o); ++i) {
    acc_o(i) /= params.seqlen_q;
  }

  // Epilogue
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = make_tensor_like<Element>(acc_o);
  flash::convert_type_safe(acc_o, rO);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  if constexpr (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                                binfo.q_offset(params.o_row_stride)),
                  make_shape(actual_seqlen_q, params.h, params.d),
                  make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
  // Repeat the partitioning with identity layouts
  Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOrO, tOgO, tOcO, actual_seqlen_q - m_block * kBlockM);
}

template <typename Kernel_traits, typename Params>
__global__ void hstu_fwd_kernel(Params params) {
  constexpr bool Is_balance = Kernel_traits::Is_balance;
  int m_block, bidb, bidh;
  if constexpr (Is_balance) {
    m_block = gridDim.z - blockIdx.z - 1;
    bidh = blockIdx.x;
    bidb = blockIdx.y;
  } else {
    m_block = gridDim.x - blockIdx.x - 1;
    bidh = blockIdx.y;
    bidb = blockIdx.z;
  }

  hstu_compute_attn_1rowblock<Kernel_traits>(params, bidb, bidh, m_block);
}

}  // namespace flash

template <typename elem_type,
          int kHeadDim,
          int kBlockM,
          int kBlockN,
          int kNWarps,
          bool Is_delta_q,
          bool Is_causal,
          bool Is_target,
          bool Is_context,
          bool Is_local,
          bool Has_rab,
          bool Paged_KV,
          bool Is_balance = false,
          bool Is_even_rab = true,
          bool Is_Q_in_regs = false,
          bool Share_Q_K_smem = false>
void run_hstu_fwd_impl(Hstu_fwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_fwd_kernel_traits<
      kHeadDim, kBlockM, kBlockN, kNWarps, Is_delta_q, Is_causal, Is_target, Is_context, Is_local,
      Has_rab, Paged_KV, Is_balance, Is_even_rab, Is_Q_in_regs, Share_Q_K_smem, elem_type>;

  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + kBlockM - 1) / kBlockM;
  dim3 grid;
  if constexpr (Is_balance) {
    grid = dim3(params.h, params.b, num_m_block);
  } else {
    grid = dim3(num_m_block, params.h, params.b);
  }
  auto kernel = &flash::hstu_fwd_kernel<Kernel_traits, Hstu_fwd_params>;

  if constexpr (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Arch, typename elem_type, int kHeadDim, bool Has_rab, bool Is_local,
          bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_fwd_(Hstu_fwd_params& params, cudaStream_t stream) {
  // BOOL_SWITCH(params.is_balance_fwd, Is_balance, [&] {
  static constexpr bool Is_balance = false;
  INT_SWITCH(params.page_size, Page_Size, [&] {
    static constexpr bool Paged_KV = Page_Size > 0;
    static constexpr auto tile_size = flash::get_tile_size_fwd<Arch, kHeadDim, Has_rab>();
    static constexpr int kBlockM = std::get<0>(tile_size);
    static constexpr int kBlockN = Paged_KV? Page_Size : std::get<1>(tile_size);
    static constexpr int kNWarps = std::get<2>(tile_size);
    static constexpr bool Is_Q_in_regs = kHeadDim <= 128;
    static constexpr bool Share_Q_K_smem = kHeadDim <= 128;
    const bool even_rab = params.seqlen_q % kBlockM == 0 && params.seqlen_k % kBlockN == 0;
    BOOL_SWITCH(even_rab, Is_even_rab, [&] {
      run_hstu_fwd_impl<elem_type, kHeadDim, kBlockM, kBlockN, kNWarps, Is_delta_q, Is_causal, Is_target, Is_context, Is_local,
                        Has_rab, Paged_KV, Is_balance, Is_even_rab, Is_Q_in_regs, Share_Q_K_smem>(params, stream);
    });
  });
  // });
}