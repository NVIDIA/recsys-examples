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
// The output dist's kernels, fwd_kernel and bwd_kernel (forward.cuh,
// backward.cuh: this is the one translation unit that holds them), and the
// out channel that launches them, one per direction of a call on the
// caller's stream (output_dist.h); then the kernels' trace.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nccl_device.h>

#include <algorithm>
#include <cstdint>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/output_dist/args.cuh"
#include "hier_output_dist/output_dist/backward.cuh"
#include "hier_output_dist/output_dist/forward.cuh"
#include "hier_output_dist/output_dist/output_dist.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

const void *forward_kernel(bool multi_node) {
  return multi_node ? (const void *)fwd_kernel<true> : (const void *)fwd_kernel<false>;
}

const void *backward_kernel(at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf,
              "hier_output_dist: gradient rows must be fp32, bf16 or fp16, got ", dtype);
  return dtype == at::kFloat      ? (const void *)bwd_kernel<float>
         : dtype == at::kBFloat16 ? (const void *)bwd_kernel<__nv_bfloat16>
                                  : (const void *)bwd_kernel<__half>;
}

}  // namespace

// SMs left to NCCL's own kernels.
constexpr int kNcclReserve = 32;

OutChannel::OutChannel(const Config &config, const Sizes &sizes, ncclWindow_t window,
                       const ncclDevComm &dev, const WindowLayout &layout, int device)
    : sizes_(sizes),
      window_(window),
      dev_(dev),
      lookup_rows_(layout.rows.lookup_rows()),
      dim_(config.dim),
      dtype_(config.dtype),
      world_size_(config.world_size),
      device_(device),
      sms_(sms_of(device)),
      ctas_(config.ctas),
      reduce_ctas_(config.reduce_ctas),
      multi_node_(sizes.multi_node()),
      trace_(device) {}

// DYNAMICEMB_GIN_CTAS if set. Else on one node 32 for the forward and 64 for
// the backward: inside a node pulls saturate NVLink at about 32 CTAs, and
// pushes keep improving up to about 64. Across nodes 32 for both: InfiniBand
// bounds them under uniform keys, and under skew the hot owner's NVLink and
// the proxies' sums do (more CTAs crowd the backward's pushes). Every
// kernel that waits for peers must be resident at once: an out kernel leaves
// an SM to a comm kernel of an input dist on another stream, kNcclReserve to
// NCCL's own kernels, and one to spare. Across nodes the backward needs CTA 0
// (its issuer) and a reducing CTA at least.
int OutChannel::out_ctas(OutKernel which) const {
  const bool forward = which == OutKernel::kForward;
  const int want = ctas_ > 0 ? ctas_ : (forward || multi_node_ ? 32 : 64);
  const int least = multi_node_ && !forward ? 2 : 1;
  return std::max(least, std::min(want, sms_ - 2 - kNcclReserve));
}

int OutChannel::reducers(int ctas) const { return std::max(1, std::min(reduce_ctas_, ctas - 1)); }

// Storage that changes size only when a call needs more rows than it has or
// fewer than half of them (it then takes rows plus 1/32): batches of about
// the same size ask for one size, so the caching allocator hands each call a
// block an earlier output freed. resize_ keeps the storage and makes no view
// (autograd refuses in-place changes to a custom function's output that is a
// view).
at::Tensor OutChannel::output(int64_t rows) {
  if (rows > out_rows_ || 2 * rows < out_rows_) out_rows_ = rows + rows / 32;
  auto out =
      at::empty({out_rows_, dim_}, at::TensorOptions().device(at::kCUDA, device_).dtype(dtype_));
  return out.resize_({rows, dim_});
}

template <typename Args>
void OutChannel::launch(const void *kernel, OutKernel which, int ctas, Args &args) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
  int64_t seq = seq_ + 1, arrive0 = arrive_;
  args.trace = trace_.start_out(which, stream);
  void *params[] = {&window_, &dev_, &sizes_, &seq, &arrive0, &args};
  HIER_CUDA_CHECK(cudaLaunchCooperativeKernel(
      kernel, ctas, kThreads, params,
      out_shared_bytes(sizes_.topo.local_size, args.trace != nullptr), stream));
  trace_.launched(which);
  seq_ = seq;
  arrive_ += ctas;
}

at::Tensor OutChannel::forward(const FwdPlan &plan, const at::Tensor &lookup_out) {
  c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(
      lookup_out.numel() == 0 || (lookup_out.is_cuda() && lookup_out.data_ptr() == lookup_rows_),
      "hier_output_dist: the lookup output must be the rows lookup_rows() returned");
  at::Tensor out = output(plan.num_out());
  at::Tensor ready;  // the kernel zeroes it
  if (multi_node_) ready = at::empty({sizes_.topo.local_size}, out.options().dtype(at::kLong));
  FwdArgs args{};
  args.plan = plan.view(sizes_);
  args.out = (char *)out.data_ptr();
  args.ready = ready.defined() ? ready.data_ptr<int64_t>() : nullptr;
  launch(forward_kernel(multi_node_), OutKernel::kForward, out_ctas(OutKernel::kForward), args);
  return out;
}

void OutChannel::backward(const BwdPlan &plan, at::Tensor grad, bool divide) {
  c10::cuda::CUDAGuard guard(device_);
  grad = grad.contiguous();
  if (grad.numel() > 0 && (uintptr_t)grad.data_ptr() % 16 != 0) grad = grad.clone();
  const void *kernel = backward_kernel(grad.scalar_type());
  // The heavy rows' sub-tasks' fp32 sums, how many of each heavy row's are
  // done, and across nodes the proxy's keys reduced of each chunk of its
  // stream (the kernel zeroes the counts).
  const auto options = grad.options();
  const at::Tensor partial = at::empty({plan.heavy_subs, grad.size(1)}, options.dtype(at::kFloat));
  const at::Tensor done = at::empty({plan.fanout.pos.numel()}, options.dtype(at::kLong));
  at::Tensor ready;
  if (multi_node_) ready = at::empty({sizes_.max_chunks}, options.dtype(at::kLong));
  const int ctas = out_ctas(OutKernel::kBackward);
  BwdArgs args{};
  args.plan = plan.view(sizes_);
  args.grad = (const char *)grad.data_ptr();
  args.partial = partial.data_ptr<float>();
  args.heavy_done = (uint64_t *)done.data_ptr<int64_t>();
  args.ready = ready.defined() ? ready.data_ptr<int64_t>() : nullptr;
  args.reducers = reducers(ctas);
  // TorchRec's owner divides by world_size, a CPU scalar: PyTorch multiplies
  // by the fp32 reciprocal.
  args.scale = divide ? 1.0f / (float)world_size_ : 1.0f;
  launch(kernel, OutKernel::kBackward, ctas, args);
}

// ── the trace ──

Trace::Trace(int device) {
  if constexpr (kTrace) {
    out_ = at::zeros({2, kTraceRing, kTraceWords},
                     at::TensorOptions().device(at::kCUDA, device).dtype(at::kLong));
  }
}

// The earliest stamps start at UINT64_MAX, the rest at 0.
unsigned long long *Trace::start_out(OutKernel which, cudaStream_t stream) {
  if constexpr (kTrace) {
    const int direction = (int)which;
    const int64_t at = direction * kTraceRing + launches_[direction] % kTraceRing;
    int64_t *row = out_.data_ptr<int64_t>() + at * kTraceWords;
    HIER_CUDA_CHECK(cudaMemsetAsync(row, 0xff, kNumEarliest * sizeof(int64_t), stream));
    HIER_CUDA_CHECK(cudaMemsetAsync(row + kNumEarliest, 0,
                                    (kTraceWords - kNumEarliest) * sizeof(int64_t), stream));
    return (unsigned long long *)row;
  }
  return nullptr;
}

void Trace::launched(OutKernel which) {
  if constexpr (kTrace) ++launches_[(int)which];
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
