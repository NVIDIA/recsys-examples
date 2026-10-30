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
// The kernels of the hierarchical dist as the host classes see them: the
// functions of kernels.cu, the one translation unit that holds every kernel,
// which launch them.
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <nccl_device.h>

#include <cstdint>
#include <utility>

#include "hier_output_dist/common/channel_args.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/output/args.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// ── setup and teardown ───────────────────────────────────────────────────────

// Whether every signal and counter of every GIN context reads 0, checked on
// the current stream (synchronized).
bool gin_signals_zero(const ncclDevComm &dev, int num_nodes);
// The exit round's kernel (one warp, on the default stream): tells the node
// and the rail peers that this rank is done, waits at most limit_ns for them
// to say the same, and records in Ctrl::closed whether they all did.
cudaError_t launch_exit_kernel(const ChannelStatic &channel, uint64_t limit_ns);

// ── the in channel's comm kernels (input/comm_kernels.cuh) ───────────────────
// On the current stream (the side stream, in InChannel::start), at call seq;
// each allocates its outputs there.

// The node barrier, and every node peer's count vector: int64 [local_size]
// [my_counts.numel()].
at::Tensor launch_node_count_exchange(const ChannelStatic &channel, int64_t seq,
                                      const at::Tensor &my_counts);
// This rank's first *rows_to_push rows of part 1 (keys, weights or undefined,
// row_cell) into its targets' node_keys at destination_base[cell] + row.
void launch_node_key_push(const ChannelStatic &channel, const at::Tensor &keys,
                          const at::Tensor &weights, const at::Tensor &row_cell,
                          const at::Tensor &destination_base, const at::Tensor &rows_to_push,
                          int64_t remote_base);
// Tells the node this rank's keys are pushed, and waits until every
// requester of the node told it.
void launch_node_done(const ChannelStatic &channel, int64_t seq);
// The rail's count messages: int64 [num_nodes][message.numel()].
at::Tensor launch_rail_count_exchange(const ChannelStatic &channel, int64_t seq,
                                      const at::Tensor &message);
// Puts this proxy's key_stage slices into its rail owners' rail_keys (each
// [num_nodes], int64 but put_skip), and waits for the other nodes' key streams.
void launch_rail_key_exchange(const ChannelStatic &channel, int64_t seq, const at::Tensor &put_src,
                              const at::Tensor &put_dst, const at::Tensor &put_bytes,
                              const at::Tensor &put_skip);
// Publishes this rank's needs and cell starts to its node (publish, int64
// [kNumNeeds + local_size * num_features]); returns every node peer's needs
// (int64 [local_size][kNumNeeds]) and where each one's KJT holds this
// requester's cells (int64 [local_size][num_features]).
std::pair<at::Tensor, at::Tensor> launch_node_status_exchange(const ChannelStatic &channel,
                                                              int64_t seq,
                                                              const at::Tensor &publish);
// int32 [words][items]: word first_word[i] + k * word_stride of node peer
// peer[i]'s `region` (int32 words) at [k][i]; an item of a negative peer
// keeps what the allocation held.
at::Tensor launch_peer_gather(const ChannelStatic &channel, Region region, const at::Tensor &peer,
                              const at::Tensor &first_word, int words, int64_t word_stride);

// ── the out kernels ──────────────────────────────────────────────────────────

const void *forward_kernel(bool multi_node);
const void *backward_kernel(at::ScalarType dtype);  // fp32, bf16 or fp16 rows
// A cooperative launch (every CTA resident).
void launch_out_kernel(const void *kernel, OutArgs &args, int ctas, size_t smem,
                       cudaStream_t stream);

// ── benchmark ────────────────────────────────────────────────────────────────

// This GPU's %globaltimer into out[index], on the current stream; raises in
// a build without DYNAMICEMB_HIER_TRACE=1.
void clock_stamp(at::Tensor out, int64_t index);

}  // namespace hier_output_dist
}  // namespace dyn_emb
