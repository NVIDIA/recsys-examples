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
// The out channel: the output dist's kernels (one per direction of a call)
// on the caller's stream, under one sequence number (out_seq) the same on
// every rank, and the gradient views of lookup_rows the backward hands out.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include <atomic>
#include <memory>
#include <tuple>
#include <vector>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/host/config.h"
#include "hier_output_dist/host/cta_budget.h"
#include "hier_output_dist/host/layout.h"
#include "hier_output_dist/host/plan.h"
#include "hier_output_dist/host/trace.h"
#include "hier_output_dist/host/windows.h"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The gradient views of lookup_rows the backward hands out (lookup_grad):
// how many are alive, and the event the last one dropped recorded on the
// stream that dropped it (the lookup's backward's), which whatever next
// writes lookup_rows waits for.
struct LookupGrad {
  std::atomic<int64_t> live{0};
  cudaEvent_t released = nullptr;
  LookupGrad();  // creates the event; throws if it cannot
  ~LookupGrad();
};

class OutChannel {
 public:
  // `channel_static`: the part of the kernels' arguments both channels share;
  // `lookup_grad`: made by the op before its ranks agree that it can run.
  OutChannel(const Config &config, const Layout &layout, const ChannelStatic &channel_static,
             SymmetricWindows &windows, CtaBudget &budget, Trace &trace, int device,
             std::shared_ptr<LookupGrad> lookup_grad);

  // The rows the lookup of a call writes, the first num_recv rows of
  // lookup_rows as a [num_recv, dim] view. Raises if the gradient view the
  // last backward handed out is still alive, before anything is queued, so
  // the call has no side effect; it raises on every rank alike only because
  // every rank keeps or drops that view the same way. Else the current
  // stream waits for the last out kernel of the process and for that view's
  // release.
  at::Tensor lookup_rows(const Plan &plan);
  // fwd_kernel on the current stream: this rank's rows, from the
  // lookup_rows of its node's owners and (across nodes) the proxy_rows of
  // its node's proxies, into a new tensor of num_out rows. lookup_out must
  // be the view lookup_rows() returned, which the lookup wrote (zero copy).
  // Returns the rows and the backward's maps, which the plan lets go of:
  // autograd saves them for backward().
  std::tuple<at::Tensor, std::vector<at::Tensor>> forward(Plan &plan, const at::Tensor &lookup_out);
  // bwd_kernel on the current stream: the gradient of this rank's rows back
  // into the lookup_rows of their owners (across nodes through the proxies,
  // which sum each node-unique key's), times 1 / world_size if `divide`.
  // `maps` are those forward() returned, as autograd saved them. Returns
  // this rank's lookup_grad, the gradient of its lookup output, a view of
  // lookup_rows until the lookup's backward drops it.
  at::Tensor backward(Plan &plan, at::Tensor grad, const std::vector<at::Tensor> &maps,
                      bool divide);
  // The event after this op's last out kernel (nullptr before the first).
  std::shared_ptr<at::cuda::CUDAEvent> last_done() const { return last_done_; }

 private:
  void check_lookup_grad(const char *what) const;
  // The output of a forward of `rows` rows.
  at::Tensor output(int64_t rows);
  // A cooperative launch: every CTA is resident, so they may wait for each
  // other. The kernel takes its sequence number and its arrivals only once
  // it is launched: a launch that fails leaves the channel as it was.
  void launch(const void *kernel, bool forward, OutCall &call);

  OutStatic static_{};  // what every kernel takes
  Channel channel_;
  int64_t arrive_ = 0;    // Ctrl::arrive after the out kernels issued so far
  int64_t out_rows_ = 0;  // rows of the storage of the forward's outputs
  int64_t dim_;
  at::ScalarType dtype_;
  int world_size_, device_;
  bool multi_node_;
  SymmetricWindows &windows_;
  CtaBudget &budget_;
  Trace &trace_;
  // Shared with the views' deleters, which may outlive this object.
  std::shared_ptr<LookupGrad> lookup_grad_;
  std::shared_ptr<at::cuda::CUDAEvent> last_done_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
