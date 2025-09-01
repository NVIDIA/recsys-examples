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
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "hstu.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void set_params_fprop(Hstu_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t target_group_size,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t h_rab,
                      const size_t d,
                      const float alpha,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor rab,
                      at::Tensor out,
                      void* num_contexts_d,
                      void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d,
                      void* num_targets_d,
                      bool has_rab,
                      const at::Tensor& func,
                      int window_size_left,
                      int window_size_right) {
  // Reset the parameters
  params = {};

  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
  #ifdef HSTU_DISABLE_BF16
    TORCH_CHECK(!params.is_bf16, "This hstu attention build does not support bf16.");
  #endif
  #ifdef HSTU_DISABLE_FP16
    TORCH_CHECK(q.dtype() != torch::kFloat16, "This hstu attention build does not support fp16.");
  #endif
  #ifdef HSTU_DISABLE_FP8
    TORCH_CHECK(!params.is_e4m3, "This hstu attention build does not support fp8_e4m3.");
  #endif

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  if (out.numel() > 0) {
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);
  }

  params.has_rab = has_rab;
  #ifdef HSTU_DISABLE_RAB
      TORCH_CHECK(!has_rab, "This hstu attention build does not support has_rab.");
  #endif
  if (has_rab) {
    params.rab_ptr = rab.data_ptr();
    params.rab_batch_stride = rab.stride(0);
    params.rab_row_stride = rab.stride(-2);
    params.rab_head_stride = rab.stride(-3);
    params.h_rab = h_rab;
  } else {
    params.rab_ptr = nullptr;
    params.rab_batch_stride = 0;
    params.rab_row_stride = 0;
    params.rab_head_stride = 0;
    params.h_rab = h;
  }

  params.num_contexts = static_cast<int*>(num_contexts_d);
  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.num_targets = static_cast<int*>(num_targets_d);

  TORCH_CHECK(
      bool(params.cu_seqlens_q) == bool(params.cu_seqlens_k),
      "cu_seqlens_q and cu_seqlens_k must be both null or non-null"
  );
  // Set num SM
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // Set the block scheduling
  // float coeff = 0.3;
  params.is_balance_fwd = false;
  params.is_balance_bwd = false;
  // auto dprops = at::cuda::getCurrentDeviceProperties();
  // int l2_size = dprops->l2CacheSize;
  // int sm_count = dprops->multiProcessorCount;
  // int num_KV = std::min(sm_count, int(b * h_k));
  // int kv_cache_size = 2 * seqlen_k * num_KV * d * sizeof(k.dtype()) * coeff;
  // int do_cache_size = seqlen_q * num_KV * d * sizeof(out.dtype()) * coeff;
  // if (kv_cache_size < l2_size) {
  //   params.is_balance_fwd = true;
  // }
  // if (kv_cache_size + do_cache_size < l2_size) {
  //   params.is_balance_bwd = true;
  // }

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.alpha = alpha;
  // Set the masks.
  params.is_target = num_targets_d != nullptr;
  #ifdef HSTU_DISABLE_TARGET
    TORCH_CHECK(!params.is_target, "This hstu attention build does not support target.");
  #endif
  params.target_group_size = target_group_size;
  if (params.is_target) {
    TORCH_CHECK(target_group_size > 0, "target_group_size must be greater than 0 when target is True");
  }
  params.is_context = num_contexts_d != nullptr;
  #ifdef HSTU_DISABLE_CONTEXT
    TORCH_CHECK(!params.is_context, "This hstu attention build does not support context mask.");
  #endif

  if (window_size_left < 0 || window_size_left > (int)seqlen_k) { window_size_left = seqlen_k; }
  if (window_size_right < 0 || window_size_right > (int)seqlen_k) { window_size_right = seqlen_k; }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.is_causal = window_size_left == (int)seqlen_k && window_size_right == 0;
  #ifdef HSTU_DISABLE_CAUSAL
    TORCH_CHECK(!params.is_causal, "This hstu attention build does not support causal.");
  #endif
  TORCH_CHECK(!(!params.is_causal && params.is_target), "Target mask is True, but causal mask is False, this is undefined behavior.");
  TORCH_CHECK(!(!params.is_causal && params.is_context), "Context mask is True, but causal mask is False, this is undefined behavior.");
  params.is_local = (window_size_left < (int)seqlen_k || window_size_right < (int)seqlen_k) && !params.is_causal;
  #ifdef HSTU_DISABLE_LOCAL
    TORCH_CHECK(!params.is_local, "This hstu attention build does not support local mask.");
  #endif

  params.is_arbitrary_mask = func.defined();
  #ifdef HSTU_DISABLE_ARBITRARY
    TORCH_CHECK(!params.is_arbitrary_mask, "This hstu attention build does not support arbitrary mask.");
  #endif
  if (params.is_arbitrary_mask) {
    TORCH_CHECK(func.dtype() == torch::kInt32, "func must have dtype int32");
    CHECK_DEVICE(func); // batch_size x heads x n_func x seqlen_q

    params.func_ptr = func.data_ptr();
    params.func_ids_stride = func.stride(-2);
    params.func_head_stride = func.stride(-3);
    params.n_func = func.size(-2);
    TORCH_CHECK(params.n_func == HSTU_ARBITRARY_NFUNC, "n_func must be equal to HSTU_ARBITRARY_NFUNC");
  }
}

void set_params_dgrad(Hstu_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t target_group_size,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t h_rab,
                      const size_t d,
                      const float alpha,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor dout,
                      const at::Tensor rab,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      at::Tensor drab,
                      void *num_contexts_d,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *num_targets_d,
                      void *dq_accum_d,
                      const at::Tensor& func,
                      bool has_rab,
                      bool has_drab,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic) {

  set_params_fprop(params, b, seqlen_q, seqlen_k, target_group_size, seqlen_q_rounded,
                    seqlen_k_rounded, h, h_k, h_rab, d, alpha, q, k, v, rab,
                    /*out=*/torch::Tensor(),
                    num_contexts_d, cu_seqlens_q_d, cu_seqlens_k_d, num_targets_d,
                    has_rab, func, window_size_left, window_size_right);

  params.has_drab = has_drab;
  #ifdef HSTU_DISABLE_DRAB
    TORCH_CHECK(!has_drab, "This hstu attention build does not support has_drab.");
  #endif
  if (has_drab) {
    params.drab_ptr = drab.data_ptr();
    params.drab_batch_stride = drab.stride(0);
    params.drab_row_stride = drab.stride(-2);
    params.drab_head_stride = drab.stride(-3);
  } else {
    params.drab_ptr = nullptr;
    params.drab_batch_stride = 0;
    params.drab_row_stride = 0;
    params.drab_head_stride = 0;
  }
  // Set the pointers and strides.
  params.do_ptr = dout.data_ptr();
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.dq_row_stride = dq.stride(-3);
  params.dk_row_stride = dk.stride(-3);
  params.dv_row_stride = dv.stride(-3);
  params.dq_head_stride = dq.stride(-2);
  params.dk_head_stride = dk.stride(-2);
  params.dv_head_stride = dv.stride(-2);

  params.dq_accum_ptr = dq_accum_d;

  params.deterministic = deterministic;
}

template <typename Dtype, int Headdim>
void run_hstu_fwd_mask_16(Hstu_fwd_params &params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    #ifndef HSTU_DISABLE_ARBITRARY
    if (params.is_arbitrary_mask) { run_hstu_fwd_<90, Dtype, Headdim, Has_rab, false, false, false, false, true, HSTU_ARBITRARY_NFUNC>(params, stream); return; }
    #endif
    #ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) { run_hstu_fwd_<90, Dtype, Headdim, Has_rab, true, false, false, false, false, 0>(params, stream); return; }
    #endif
    if (!params.is_causal) { run_hstu_fwd_<90, Dtype, Headdim, Has_rab, false, false, false, false, false, 0>(params, stream); return; }
    else {
      #ifndef HSTU_DISABLE_CAUSAL
      CONTEXT_SWITCH(params.is_context, Is_context, [&] {
        TARGET_SWITCH(params.is_target, Is_target, [&] {
          run_hstu_fwd_<90, Dtype, Headdim, Has_rab, false, true, Is_context, Is_target, false, 0>(params, stream);
        });
      });
      #endif
    }
  });
}

template <typename Dtype, int Headdim>
void run_hstu_fwd_mask_8(Hstu_fwd_params &params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    #ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) { run_hstu_fwd_<90, Dtype, Headdim, Has_rab, true, false, false, false, false, 0>(params, stream); return; }
    #endif
    #ifndef HSTU_DISABLE_CAUSAL
    if (params.is_causal) { run_hstu_fwd_<90, Dtype, Headdim, Has_rab, false, true, false, false, false, 0>(params, stream); return; }
    #endif
    run_hstu_fwd_<90, Dtype, Headdim, Has_rab, false, false, false, false, false, 0>(params, stream);
  });
}

void run_hstu_fwd(Hstu_fwd_params &params, cudaStream_t stream) {
  if (params.is_bf16) {
    #ifndef HSTU_DISABLE_BF16
    #ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) { run_hstu_fwd_mask_16<cutlass::bfloat16_t, 32>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) { run_hstu_fwd_mask_16<cutlass::bfloat16_t, 64>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) { run_hstu_fwd_mask_16<cutlass::bfloat16_t, 128>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) { run_hstu_fwd_mask_16<cutlass::bfloat16_t, 256>(params, stream); }
    #endif
    #else
    TORCH_CHECK(false, "This flash attention build does not support BF16.");
    #endif
  } else if (params.is_e4m3) {
    #ifndef HSTU_DISABLE_FP8
    #ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) { std::cerr << "Not support dim = 32 and dtype = float_e4m3_t for now." << std::endl; }
    #endif
    #ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) { run_hstu_fwd_mask_8<cutlass::float_e4m3_t, 64>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) { run_hstu_fwd_mask_8<cutlass::float_e4m3_t, 128>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) { run_hstu_fwd_mask_8<cutlass::float_e4m3_t, 256>(params, stream); }
    #endif
    #else
    TORCH_CHECK(false, "This flash attention build does not support FP8.");
    #endif
  } else {
    #ifndef HSTU_DISABLE_FP16
    #ifndef HSTU_DISABLE_HDIM32
    if (params.d == 32) { run_hstu_fwd_mask_16<cutlass::half_t, 32>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM64
    if (params.d == 64) { run_hstu_fwd_mask_16<cutlass::half_t, 64>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM128
    if (params.d == 128) { run_hstu_fwd_mask_16<cutlass::half_t, 128>(params, stream); }
    #endif
    #ifndef HSTU_DISABLE_HDIM256
    if (params.d == 256) { run_hstu_fwd_mask_16<cutlass::half_t, 256>(params, stream); }
    #endif
    #else
    TORCH_CHECK(false, "This flash attention build does not support FP16.");
    #endif
  }
}

std::vector<at::Tensor>
hstu_varlen_fwd(const at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               const int max_seqlen_q,
               const int max_seqlen_k,
               std::optional<const at::Tensor> &num_contexts,  // b
               std::optional<const at::Tensor> &num_targets,  // b
               const int target_group_size,
               int window_size_left,
               int window_size_right,
               const float alpha,
               std::optional<at::Tensor> &rab,
               std::optional<const at::Tensor>& func,
               std::optional<const at::Tensor> &descale_q_,
               std::optional<const at::Tensor> &descale_k_,
               std::optional<const at::Tensor> &descale_v_) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");

  auto q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
              "HSTU only supports fp16, bf16, and fp8_e4m3 data type");
  if (dprops->major < 9) {
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                "HSTU on Ampere/Ada cards only supports fp16 and bf16 data type");
  }
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(num_heads == num_heads_k, "Number of heads in key/value and query must be equal");
  if (q_type == at::ScalarType::Float8_e4m3fn) {
    TORCH_CHECK(head_size == 64 || head_size == 128 || head_size == 256,
                "For fp8, HSTU forward only supports head dimension 64, 128, or 256");
  } else {
    TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
                "HSTU forward only supports head dimension 32, 64, 128, or 256");
  }

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(num_contexts.value().dtype() == torch::kInt32,
                "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  if (num_targets.has_value()) {
    TORCH_CHECK(num_targets.value().dtype() == torch::kInt32,
                "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::Half : q_type;
  at::Tensor out = torch::empty_like(q, out_type);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q.dtype()));
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q.dtype()));

  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.get_device()};

  auto opts = q.options();
  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1,
                "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(rab.value(), batch_size, num_heads_rab, max_seqlen_k, max_seqlen_k);
    if (seqlen_k_rounded != max_seqlen_k) {
        rab = torch::nn::functional::pad(rab.value(), torch::nn::functional::PadFuncOptions(
                            {0, seqlen_k_rounded - max_seqlen_k}));
    }
  }

  Hstu_fwd_params params;
  set_params_fprop(params,
                    batch_size,
                    max_seqlen_q, max_seqlen_k, target_group_size,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k, num_heads_rab,
                    head_size, alpha,
                    q, k, v,
                    has_rab ? rab.value() : torch::Tensor(), out,
                    num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
                    cu_seqlens_q.data_ptr(),
                    cu_seqlens_k.data_ptr(),
                    num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
                    has_rab,
                    func.has_value() ? func.value() : torch::Tensor(),
                    window_size_left,
                    window_size_right);

  auto tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32)); // TODO: if arbitrary mask, do not need zeros it, just empty
  params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

  params.total_q = total_q;
  params.total_k = total_k;

  if(q_type == at::ScalarType::Float8_e4m3fn) {
    at::Tensor descale_q, descale_k, descale_v;
    if (descale_q_.has_value() && descale_k_.has_value() && descale_k_.has_value()) {
      descale_q = descale_q_.value();
      descale_k = descale_k_.value();
      descale_v = descale_v_.value();
      CHECK_DEVICE(descale_q);
      CHECK_DEVICE(descale_k);
      CHECK_DEVICE(descale_v);
      CHECK_SHAPE(descale_q, 1);
      CHECK_SHAPE(descale_k, 1);
      CHECK_SHAPE(descale_v, 1);
    } else {
      descale_q = torch::ones({1}, opts.dtype(at::kFloat));
      descale_k = torch::ones({1}, opts.dtype(at::kFloat));
      descale_v = torch::ones({1}, opts.dtype(at::kFloat));
    }
    params.descale_q_ptr = descale_q.data_ptr<float>();
    params.descale_k_ptr = descale_k.data_ptr<float>();
    params.descale_v_ptr = descale_v.data_ptr<float>();
  } else {
    params.descale_q_ptr = nullptr;
    params.descale_k_ptr = nullptr;
    params.descale_v_ptr = nullptr;
  }

  if (total_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_hstu_fwd(params, stream);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
  }

  return {out, has_rab ? rab.value() : torch::Tensor()};
}

template <typename Dtype, int Headdim>
void run_hstu_bwd_mask_16(Hstu_bwd_params &params, cudaStream_t stream) {
  RAB_DRAB_SWITCH(params.has_rab, params.has_drab, Has_rab, Has_drab, [&] {
    #ifndef HSTU_DISABLE_ARBITRARY
    if (params.is_arbitrary_mask) { run_hstu_bwd_<90, Dtype, Headdim, Has_rab, Has_drab, false, false, false, false, true, HSTU_ARBITRARY_NFUNC>(params, stream); return; }
    #endif
    #ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) { run_hstu_bwd_<90, Dtype, Headdim, Has_rab, Has_drab, true, false, false, false, false, 0>(params, stream); return; }
    #endif
    if (!params.is_causal) { run_hstu_bwd_<90, Dtype, Headdim, Has_rab, Has_drab, false, false, false, false, false, 0>(params, stream); return; }
    else {
      #ifndef HSTU_DISABLE_CAUSAL
      CONTEXT_SWITCH(params.is_context, Is_context, [&] {
        TARGET_SWITCH(params.is_target, Is_target, [&] {
            run_hstu_bwd_<90, Dtype, Headdim, Has_rab, Has_drab, false, true, Is_context, Is_target, false, 0>(params, stream);
        });
      });
      #endif
    }
  });
}

template <typename Dtype, int Headdim>
void run_hstu_bwd_mask_8(Hstu_bwd_params &params, cudaStream_t stream) {
  RAB_SWITCH(params.has_rab, Has_rab, [&] {
    #ifndef HSTU_DISABLE_LOCAL
    if (params.is_local) { run_hstu_bwd_<90, Dtype, Headdim, Has_rab, false, true, false, false, false, false, 0>(params, stream); return; }
    #endif
    #ifndef HSTU_DISABLE_CAUSAL
    if (params.is_causal) { run_hstu_bwd_<90, Dtype, Headdim, Has_rab, false, false, true, false, false, false, 0>(params, stream); return; }
    #endif
    run_hstu_bwd_<90, Dtype, Headdim, Has_rab, false, false, false, false, false, false, 0>(params, stream);
  });
}

void run_hstu_bwd(Hstu_bwd_params &params, cudaStream_t stream) {
  #ifndef HSTU_DISABLE_BACKWARD
    if (params.is_bf16) {
      #ifndef HSTU_DISABLE_BF16
      #ifndef HSTU_DISABLE_HDIM32
      if (params.d == 32) { run_hstu_bwd_mask_16<cutlass::bfloat16_t, 32>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM64
      if (params.d == 64) { run_hstu_bwd_mask_16<cutlass::bfloat16_t, 64>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM128
      if (params.d == 128) { run_hstu_bwd_mask_16<cutlass::bfloat16_t, 128>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM256
      if (params.d == 256) { run_hstu_bwd_mask_16<cutlass::bfloat16_t, 256>(params, stream); }
      #endif
      #else
      TORCH_CHECK(false, "This flash attention build does not support BF16.");
      #endif
    } else if (params.is_e4m3) {
      // #ifndef HSTU_DISABLE_FP8
      // #ifndef HSTU_DISABLE_HDIM32
      // if (params.d == 32) { std::cerr << "Not support dim = 32 and dtype = float_e4m3_t for now." << std::endl; }
      // #endif
      // #ifndef HSTU_DISABLE_HDIM64
      // if (params.d == 64) { run_hstu_bwd_mask_8<cutlass::float_e4m3_t, 64>(params, stream); }
      // #endif
      // #ifndef HSTU_DISABLE_HDIM128
      // if (params.d == 128) { run_hstu_bwd_mask_8<cutlass::float_e4m3_t, 128>(params, stream); }
      // #endif
      // #ifndef HSTU_DISABLE_HDIM256
      // if (params.d == 256) { run_hstu_bwd_mask_8<cutlass::float_e4m3_t, 256>(params, stream); }
      // #endif
      // #else
      // TORCH_CHECK(false, "This flash attention build does not support FP8.");
      // #endif
    } else {
      #ifndef HSTU_DISABLE_FP16
      #ifndef HSTU_DISABLE_HDIM32
      if (params.d == 32) { run_hstu_bwd_mask_16<cutlass::half_t, 32>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM64
      if (params.d == 64) { run_hstu_bwd_mask_16<cutlass::half_t, 64>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM128
      if (params.d == 128) { run_hstu_bwd_mask_16<cutlass::half_t, 128>(params, stream); }
      #endif
      #ifndef HSTU_DISABLE_HDIM256
      if (params.d == 256) { run_hstu_bwd_mask_16<cutlass::half_t, 256>(params, stream); }
      #endif
      #else
      TORCH_CHECK(false, "This flash attention build does not support FP16.");
      #endif
    }
  #endif
}

std::vector<at::Tensor> hstu_varlen_bwd(
    const at::Tensor &dout,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    std::optional<const at::Tensor> &dq_,
    std::optional<const at::Tensor> &dk_,
    std::optional<const at::Tensor> &dv_,
    const at::Tensor &cu_seqlens_q,  // b+1
    const at::Tensor &cu_seqlens_k,  // b+1
    const int max_seqlen_q,
    const int max_seqlen_k,
    std::optional<const at::Tensor> &num_contexts,  // b
    std::optional<const at::Tensor> &num_targets,  // b
    const int target_group_size,
    int window_size_left,
    int window_size_right,
    const float alpha,
    std::optional<at::Tensor> &rab,
    const bool has_drab,
    std::optional<const at::Tensor>& func,
    const bool deterministic) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(dprops->major >= 8, "HSTU only supports Ampere GPUs or newer.");

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "HSTU only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);
  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  CHECK_CONTIGUOUS(cu_seqlens_q);
  CHECK_CONTIGUOUS(cu_seqlens_k);

  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = q.size(0);
  const int num_heads = q.size(1);
  const int head_size = q.size(2);
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);

  CHECK_SHAPE(k, total_k, num_heads_k, head_size);
  CHECK_SHAPE(v, total_k, num_heads_k, head_size);
  CHECK_SHAPE(dout, total_q, num_heads, head_size);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size == 32 || head_size == 64 || head_size == 128 || head_size == 256,
              "head_size should be 32, 64, 128, or 256");
  TORCH_CHECK(num_heads == num_heads_k, "Number of heads in key/value and query must be equal");

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  // This should match the kernel configs
  const int kBlockM = head_size <= 64 ? 128 : 64;
  const int kBlockN = head_size <= 128 ? 128 : 64;
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, sizeof(cutlass::uint128_t) / sizeof(q_dtype));
  int const total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockN, kBlockN);

  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_contexts.has_value()) {
    TORCH_CHECK(num_contexts.value().dtype() == torch::kInt32,
                "num_contexts must have dtype int32");
    CHECK_DEVICE(num_contexts.value());
    CHECK_CONTIGUOUS(num_contexts.value());
    CHECK_SHAPE(num_contexts.value(), batch_size);
  }
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  if (num_targets.has_value()) {
    TORCH_CHECK(num_targets.value().dtype() == torch::kInt32,
                "num_targets must have dtype int32");
    CHECK_DEVICE(num_targets.value());
    CHECK_CONTIGUOUS(num_targets.value());
    CHECK_SHAPE(num_targets.value(), batch_size);
  }

  bool has_rab = rab.has_value();
  int num_heads_rab = num_heads;
  if (has_rab) {
    num_heads_rab = rab.value().size(1);
    CHECK_DEVICE(rab.value());
    TORCH_CHECK(rab.value().stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(num_heads == num_heads_rab || num_heads_rab == 1,
                "Number of heads in rab must be 1 or equal to number of heads in query");
    CHECK_SHAPE(rab.value(), batch_size, num_heads_rab, max_seqlen_k, seqlen_k_rounded);
  }

  at::Tensor dq = dq_.has_value() ? dq_.value() : torch::empty_like(q);
  at::Tensor dk = dk_.has_value() ? dk_.value() : torch::empty_like(k);
  at::Tensor dv = dv_.has_value() ? dv_.value() : torch::empty_like(v);


  // Otherwise the kernel will be launched from cuda:0 device
  at::cuda::CUDAGuard device_guard{q.get_device()};

  auto opts = q.options();
  at::Tensor dq_accum;
  dq_accum = torch::empty({num_heads, total_q_padded_rounded, head_size}, opts.dtype(at::kFloat));
  dq_accum.zero_();

  at::Tensor drab;
  TORCH_CHECK(!(!has_rab && has_drab), "has_rab must be True when has_drab=True");
  if (has_drab) {
      drab = torch::zeros_like(rab.value());
  } else {
      drab = torch::empty({seqlen_k_rounded}, opts);
  }

  Hstu_bwd_params params;

  set_params_dgrad(params,
                    batch_size,
                    max_seqlen_q, max_seqlen_k, target_group_size,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k, num_heads_rab,
                    head_size, alpha,
                    q, k, v, dout,
                    has_rab ? rab.value() : torch::Tensor(),
                    dq, dk, dv, drab,
                    num_contexts.has_value() ? num_contexts.value().data_ptr() : nullptr,
                    cu_seqlens_q.data_ptr(), cu_seqlens_k.data_ptr(),
                    num_targets.has_value() ? num_targets.value().data_ptr() : nullptr,
                    dq_accum.data_ptr(),
                    func.has_value() ? func.value() : torch::Tensor(),
                    has_rab, has_drab,
                    window_size_left, window_size_right,
                    deterministic);
  params.total_q = total_q;
  params.total_k = total_k;

  // Will be zero'ed out in the backward preprocess kernel
  at::Tensor dq_semaphore = torch::empty({(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
  dq_semaphore.zero_();
  params.dq_semaphore = dq_semaphore.data_ptr<int>();

  if (total_q > 0) {
      run_hstu_bwd(params, stream);
  } else {
      // If max_seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
      dk.zero_();
      dv.zero_();
      drab.zero_();
  }

  if (has_drab && seqlen_k_rounded != max_seqlen_k) {
      drab = drab.index({"...", torch::indexing::Slice(torch::indexing::None, max_seqlen_k)});
  }

  return { dq, dk, dv, drab };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HstuAttention";
    m.def("varlen_fwd", &hstu_varlen_fwd, "Varlen hstu forward pass");
    m.def("varlen_bwd", &hstu_varlen_bwd, "Varlen hstu backward pass");
}
