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
// The in channel: the input dist of every call after part 1 (the first
// wait), in issue order on the op's side stream, under one sequence number
// (in_seq) the same on every rank. Its comm kernels (input/comm_kernels.cuh)
// exchange with the node and rail peers; the planner's steps (plan/) lay the
// call out between them, as ATen ops on the same stream.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include <memory>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/host/config.h"
#include "hier_output_dist/host/layout.h"
#include "hier_output_dist/host/plan.h"
#include "hier_output_dist/host/windows.h"
#include "hier_output_dist/plan/proxy.h"
#include "hier_output_dist/plan/requester.h"

namespace dyn_emb {
namespace hier_output_dist {

class InChannel {
 public:
  // channel: the part of the kernels' arguments both channels share, over
  // the registered windows.
  InChannel(const Config &config, const Layout &layout, const ChannelStatic &channel,
            const SymmetricWindows &windows, int device);

  // Queues the call on the side stream, after part 1 on the caller's stream
  // and after the out kernels that the previous start found issued (its
  // out_done; that wait bounds how far the host runs ahead): the steps of
  // the node (its counts, layout and keys), across nodes those of the proxy,
  // the rail and the requester, then the call's tables and status, whose
  // pinned copy the host reads after the plan's event done. It never reads
  // a device value on the host.
  void start(Plan &plan, const std::shared_ptr<at::cuda::CUDAEvent> &out_done);
  int64_t calls() const { return seq_; }

 private:
  ChannelStatic channel_;
  PlanSizes sizes_;
  bool node_dedup_;
  int64_t seq_ = 0;  // the calls started
  at::cuda::CUDAStream side_;
  // The out kernels the previous start found issued (its out_done).
  std::shared_ptr<at::cuda::CUDAEvent> previous_out_done_;
  // This rank's regions the steps read and write, as tensors without an
  // allocator (the windows outlive them): the own part of node_keys (and
  // node_weights), and across nodes its remote part, the cell tags, the
  // proxy's windows and rail_keys.
  at::Tensor own_keys_, own_weights_;
  at::Tensor remote_keys_, remote_weights_, remote_cells_;
  ProxyWindows proxy_windows_;
  at::Tensor rail_keys_;
  // Constants of the op across nodes: the table of each feature, the other
  // nodes, and the words of the proxy headers this requester reads.
  at::Tensor tables_, other_nodes_;
  PeerWords header_gather_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
