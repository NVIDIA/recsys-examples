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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ScopeExit.h>

#include <mutex>
#include <unordered_map>

#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/out_channel.h"
#include "hier_output_dist/kernels.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The out kernels of every op of the process run one at a time (the CTA
// budget counts one), and a lookup writes lookup_rows only after the last of
// them. A launch or a lookup on another stream than the last out kernel's
// waits for that kernel's event first; on the same stream, the stream's
// order is enough. Every rank issues them in the same order, so this adds no
// wait between ranks.

// The last out kernel of the process on a device: its event and stream.
struct LastOut {
  std::shared_ptr<at::cuda::CUDAEvent> event;
  c10::Stream stream;
};

std::mutex &last_out_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<int, LastOut> &last_out() {
  static std::unordered_map<int, LastOut> by_device;
  return by_device;
}

// `stream` waits for the last out kernel of the process on its device.
void wait_last_out(const at::cuda::CUDAStream &stream) {
  std::shared_ptr<at::cuda::CUDAEvent> event;
  {
    std::lock_guard<std::mutex> lock(last_out_mutex());
    auto it = last_out().find(stream.device_index());
    if (it == last_out().end() || it->second.stream == c10::Stream(stream)) return;
    event = it->second.event;
  }
  event->block(stream);
}

// Records an event on `stream` after an out kernel, makes it the one
// wait_last_out() waits for, and returns it.
std::shared_ptr<at::cuda::CUDAEvent> record_last_out(const at::cuda::CUDAStream &stream) {
  auto event = std::make_shared<at::cuda::CUDAEvent>();
  event->record(stream);
  std::lock_guard<std::mutex> lock(last_out_mutex());
  last_out().insert_or_assign(stream.device_index(), LastOut{event, c10::Stream(stream)});
  return event;
}

}  // namespace

LookupGrad::LookupGrad() {
  HIER_CUDA_CHECK(cudaEventCreateWithFlags(&released, cudaEventDisableTiming));
}

LookupGrad::~LookupGrad() {
  if (released != nullptr) cudaEventDestroy(released);
}

OutChannel::OutChannel(const Config &config, const Layout &layout,
                       const ChannelStatic &channel_static, SymmetricWindows &windows,
                       CtaBudget &budget, Trace &trace, int device,
                       std::shared_ptr<LookupGrad> lookup_grad)
    : static_{channel_static, (int)layout.row_bytes, (int)(layout.row_bytes / 16)},
      channel_{0, layout.ctrl_at.out_started},
      dim_(config.dim),
      dtype_(config.dtype),
      world_size_(config.world_size),
      device_(device),
      multi_node_(layout.multi_node()),
      windows_(windows),
      budget_(budget),
      trace_(trace),
      lookup_grad_(std::move(lookup_grad)) {}

void OutChannel::check_lookup_grad(const char *what) const {
  TORCH_CHECK(lookup_grad_->live.load() == 0,
              "hier_output_dist: the gradient the last backward handed the lookup's backward (a "
              "view of the lookup output window) is still alive at ",
              what, ", which would overwrite it; do not keep the gradient of the lookup output");
}

at::Tensor OutChannel::lookup_rows(const Plan &plan) {
  c10::cuda::CUDAGuard guard(device_);
  check_lookup_grad("the next lookup");
  auto stream = at::cuda::getCurrentCUDAStream(device_);
  wait_last_out(stream);
  HIER_CUDA_CHECK(cudaStreamWaitEvent(stream, lookup_grad_->released, 0));
  return windows_.lookup_rows(plan.num_recv(), dim_, dtype_);
}

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

void OutChannel::launch(const void *kernel, bool forward, OutCall &call) {
  const int ctas = budget_.out_ctas(forward);
  call.reducers = budget_.reducers(ctas);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
  const size_t smem = trace_.start_out(forward, call, stream);
  call.ch = Channel{channel_.seq + 1, channel_.started};
  call.arrive0 = arrive_;
  OutArgs args{static_, call};
  launch_out_kernel(kernel, args, ctas, smem, stream);
  trace_.launched(forward);
  channel_.seq = call.ch.seq;
  arrive_ += ctas;
}

std::tuple<at::Tensor, std::vector<at::Tensor>> OutChannel::forward(Plan &plan,
                                                                    const at::Tensor &lookup_out) {
  c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(plan.num_recv() == 0 ||
                  (lookup_out.is_cuda() && lookup_out.data_ptr() == windows_.region(kLookupRows)),
              "hier_output_dist: the lookup output must be the rows lookup_rows() returned");
  auto stream = at::cuda::getCurrentCUDAStream(device_);
  at::Tensor out = output(plan.num_out());
  wait_last_out(stream);
  plan.record_stream(stream);
  OutCall call{};
  call.plan = plan.out_plan(false);
  call.plan.out = (char *)out.data_ptr();
  launch(forward_kernel(multi_node_), true, call);
  std::vector<at::Tensor> maps = plan.after_forward();
  last_done_ = record_last_out(stream);
  return {out, std::move(maps)};
}

at::Tensor OutChannel::backward(Plan &plan, at::Tensor grad, const std::vector<at::Tensor> &maps,
                                bool divide) {
  c10::cuda::CUDAGuard guard(device_);
  check_lookup_grad("the next backward");
  plan.hold_backward_maps(maps);
  auto drop_maps = c10::make_scope_exit([&plan]() { plan.drop_backward_maps(); });
  grad = grad.contiguous();
  if (grad.numel() > 0 && (uintptr_t)grad.data_ptr() % 16 != 0) grad = grad.clone();
  const at::ScalarType dtype = grad.scalar_type();
  const void *kernel = backward_kernel(dtype);
  // The sub-tasks' fp32 sums, and how many of each heavy row's and each
  // heavy group's are done (the kernel zeroes them).
  auto options = grad.options();
  auto partial =
      at::empty({plan.heavy_subs() + plan.group_subs(), grad.size(1)}, options.dtype(at::kFloat));
  const int64_t slots = plan.heavy_slots();
  auto done = at::empty({slots + plan.heavy_groups()}, options.dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream(device_);
  wait_last_out(stream);
  HIER_CUDA_CHECK(cudaStreamWaitEvent(stream, lookup_grad_->released, 0));
  for (const at::Tensor &tensor : maps) tensor.record_stream(stream);
  OutCall call{};
  call.plan = plan.out_plan(true);
  call.plan.grad = (const char *)grad.data_ptr();
  call.plan.partial = partial.data_ptr<float>();
  call.plan.heavy_done = (uint64_t *)done.data_ptr<int64_t>();
  call.plan.group_done = call.plan.heavy_done + slots;
  call.plan.heavy_subs = plan.heavy_subs();
  call.plan.group_subs = plan.group_subs();
  // TorchRec's owner divides by world_size, a CPU scalar: PyTorch multiplies
  // by the fp32 reciprocal.
  call.plan.scale = divide ? 1.0f / (float)world_size_ : 1.0f;
  // Dropped by the lookup's backward, which read it: the release is recorded
  // on the stream of the thread that drops it.
  auto state = lookup_grad_;
  const int device = device_;
  at::Tensor lookup_grad =
      windows_.lookup_rows(plan.num_recv(), grad.size(1), dtype, [state, device]() {
        try {
          cudaEventRecord(state->released, at::cuda::getCurrentCUDAStream(device).stream());
        } catch (...) {  // a deleter must not throw
        }
        state->live.fetch_sub(1);
      });
  state->live.fetch_add(1);
  launch(kernel, false, call);
  last_done_ = record_last_out(stream);
  return lookup_grad;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
