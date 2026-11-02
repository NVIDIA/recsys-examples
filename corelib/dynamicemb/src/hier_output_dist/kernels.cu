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
// The one device translation unit of the hierarchical dist: every kernel of
// the op, in the order a call runs them, and the functions that launch them
// (kernels.h), which the host classes call (hier_output_dist.cu, channels.cu).
// One unit: the headers below define the kernels and the device functions
// they call, so only this file includes them.
//
// Per call there are two groups of kernels, all on the caller's stream; the
// planner's ATen ops (planner.h) run part 1 and the steps between the first
// group's kernels:
//   first wait -- the comm kernels (comm_kernels.cuh), which exchange
//                 the call's counts, keys and needs with the node and rail
//                 peers;
//   output     -- (forward.cuh, backward.cuh) fwd_kernel after
//                 the lookup, bwd_kernel in the backward, which hands the
//                 lookup's backward its gradient as a view of lookup_rows.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <nccl_device.h>
#include <torch/extension.h>

#include <cstdint>

#include "hier_output_dist/args.cuh"
#include "hier_output_dist/backward.cuh"
#include "hier_output_dist/comm_kernels.cuh"
#include "hier_output_dist/common.h"
#include "hier_output_dist/config.h"
#include "hier_output_dist/device.cuh"
#include "hier_output_dist/forward.cuh"
#include "hier_output_dist/kernels.h"
#include "hier_output_dist/window.h"

namespace dyn_emb {
namespace hier_output_dist {

// The kernels, in the order a call runs them: the first wait's comm kernels,
// then the output dist's.
// clang-format off
// clang-format on

// Teardown, 1 warp: tell the node this rank reads no window any more (its
// closing word) and, across nodes, its rail peers that it sends nothing more
// (a closing signal on the credits' context, behind every credit it sent
// them on the same QPs), both before waiting for anyone; then wait (at most
// limit_ns) until every node peer and every rail peer said the same, each
// lane taking peers in turn. Records in its closed word whether they all
// did. Its waits are loops of their own: after limit_ns they give up and
// record it, where Spin would trap.
__global__ void exit_kernel(ChannelStatic channel, uint64_t limit_ns) {
  const Topo &topo = channel.topo;
  const int lane = threadIdx.x, num_nodes = topo.num_nodes;
  const NodeWindow node(channel.window, topo.local_rank);
  const CtrlLayout ctrl = node.layout(channel).ctrl;
  if (lane == 0) st_release_sys(ctrl.closing(), 1);
  if (num_nodes > 1) {
    ncclGin cred(channel.dev, kCtxCredit);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      if (peer_node != topo.node)
        cred.signal(ncclTeamRail(channel.dev), peer_node,
                    ncclGin_StrongSignalAdd{(ncclGinSignal_t)sig_closing(num_nodes, topo.node), 1},
                    ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
                    cuda::thread_scope_system);
  }
  bool seen = true;
  const uint64_t start_ns = now_ns();
  for (int peer = lane; peer < topo.local_size && seen; peer += 32) {
    const uint64_t *closing = node.peer(ctrl.closing(), peer);
    while (ld_acquire_sys(closing) == 0) {
      if (now_ns() - start_ns > limit_ns) {
        seen = false;
        break;
      }
      __nanosleep(1000);
    }
  }
  if (num_nodes > 1) {
    ncclGin cred(channel.dev, kCtxCredit);
    for (int peer_node = lane; peer_node < num_nodes && seen; peer_node += 32)
      while (peer_node != topo.node && cred.readSignal(sig_closing(num_nodes, peer_node)) == 0) {
        if (now_ns() - start_ns > limit_ns) {
          seen = false;
          break;
        }
        __nanosleep(1000);
      }
    __syncwarp();
    cred.flush(ncclCoopWarp{});
  }
  const bool all = __all_sync(kFull, seen);
  if (lane == 0) *ctrl.closed() = all ? 1 : 0;
}

cudaError_t launch_exit_kernel(const ChannelStatic &channel, uint64_t limit_ns) {
  exit_kernel<<<1, 32>>>(channel, limit_ns);
  return cudaGetLastError();
}

// ── setup ────────────────────────────────────────────────────────────────────

// Signals and counters start at 0 (NCCL zeroes them): every comparison is
// against running totals from 0. Checked before any peer can send, on the
// current stream: the caching allocator may hand `bad` a block that kernels
// queued there still use, and only that stream's order puts these writes
// after them.
bool gin_signals_zero(const ncclDevComm &dev, int num_nodes) {
  int device;
  HIER_CUDA_CHECK(cudaGetDevice(&device));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  auto bad = at::empty({1}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kInt));
  HIER_CUDA_CHECK(cudaMemsetAsync(bad.data_ptr<int>(), 0, sizeof(int), stream));
  gin_zero_kernel<<<kGinContexts, 32, 0, stream>>>(dev, kGinSignalsPerNode * num_nodes,
                                                   kGinCounters, bad.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bad.item<int>() == 0;
}

// ── the in channel's comm kernels ────────────────────────────────────────────

namespace {

CommArgs comm_args(const ChannelStatic &channel, int64_t seq) {
  return CommArgs{channel, Channel{seq}};
}

cudaStream_t current_stream() { return at::cuda::getCurrentCUDAStream(); }

// CTAs of kThreads threads for `work` items on the current device.
int grid_of(int64_t work) {
  return grid_for(work, at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> launch_node_count_exchange(
    const ChannelStatic &channel, int64_t seq, const at::Tensor &my_counts) {
  at::Tensor totals = at::empty_like(my_counts), before = at::empty_like(my_counts);
  at::Tensor mine =
      at::empty({channel.topo.local_size, channel.num_features + 1}, my_counts.options());
  node_count_exchange_kernel<<<1, kThreads, 0, current_stream()>>>(
      comm_args(channel, seq), my_counts.data_ptr<int64_t>(), totals.data_ptr<int64_t>(),
      before.data_ptr<int64_t>(), mine.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {totals, before, mine};
}

void launch_node_key_push(const ChannelStatic &channel, const at::Tensor &keys,
                          const at::Tensor &weights, const at::Tensor &row_cell,
                          const at::Tensor &destination_base, const at::Tensor &rows_to_push) {
  node_key_push_kernel<<<grid_of(keys.numel()), kThreads, 0, current_stream()>>>(
      channel, keys.data_ptr<int64_t>(), weights.defined() ? weights.data_ptr<float>() : nullptr,
      row_cell.data_ptr<int32_t>(), destination_base.data_ptr<int64_t>(),
      rows_to_push.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_node_done(const ChannelStatic &channel, int64_t seq) {
  node_done_kernel<<<1, 32, 0, current_stream()>>>(comm_args(channel, seq));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor launch_rail_count_exchange(const ChannelStatic &channel, int64_t seq,
                                      const at::Tensor &message) {
  const int num_nodes = channel.topo.num_nodes;
  const CountMessage format{num_nodes, channel.num_features};
  at::Tensor messages = at::empty({num_nodes, format.length()}, message.options());
  rail_count_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      comm_args(channel, seq), message.data_ptr<int64_t>(), messages.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return messages;
}

void launch_rail_key_exchange(const ChannelStatic &channel, int64_t seq, const at::Tensor &put_src,
                              const at::Tensor &put_dst, const at::Tensor &put_bytes,
                              const at::Tensor &put_skip) {
  rail_key_exchange_kernel<<<1, 64, 0, current_stream()>>>(
      comm_args(channel, seq), put_src.data_ptr<int64_t>(), put_dst.data_ptr<int64_t>(),
      put_bytes.data_ptr<int64_t>(), put_skip.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::pair<at::Tensor, at::Tensor> launch_node_status_exchange(const ChannelStatic &channel,
                                                              int64_t seq,
                                                              const at::Tensor &publish) {
  const int local_size = channel.topo.local_size;
  at::Tensor peer_needs = at::empty({local_size, kNumNeeds}, publish.options());
  at::Tensor owner_cell_start = at::empty({local_size, channel.num_features}, publish.options());
  node_status_exchange_kernel<<<1, 256, 0, current_stream()>>>(
      comm_args(channel, seq), publish.data_ptr<int64_t>(), peer_needs.data_ptr<int64_t>(),
      owner_cell_start.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {peer_needs, owner_cell_start};
}

at::Tensor launch_peer_gather(const ChannelStatic &channel, GatherFrom from, const at::Tensor &peer,
                              const at::Tensor &first_word) {
  const int64_t items = peer.numel();
  const int words = gather_words(from);
  at::Tensor out = at::empty({words, items}, peer.options());
  peer_gather_kernel<<<grid_of(items * words), kThreads, 0, current_stream()>>>(
      channel, from, peer.data_ptr<int32_t>(), first_word.data_ptr<int64_t>(), items, words,
      out.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ── the out kernels ──────────────────────────────────────────────────────────

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

void launch_out_kernel(const void *kernel, OutArgs &args, int ctas, size_t smem,
                       cudaStream_t stream) {
  void *params[] = {&args};
  HIER_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, ctas, kThreads, params, smem, stream));
}

// ── benchmark ────────────────────────────────────────────────────────────────

// Benchmark tracing: this GPU's %globaltimer into out[index], on the current
// stream (each GPU's clock has its own origin; ping-pongs of these stamps
// between ranks give their offsets). The kernel exists only in a build with
// the trace.
#if DEMB_HIER_TRACE
__global__ void clock_kernel(int64_t *out) { *out = (int64_t)now_ns(); }
#endif

void clock_stamp(at::Tensor out, int64_t index) {
  TORCH_CHECK(kTrace, "hier_output_dist_clock: built without DYNAMICEMB_HIER_TRACE=1");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong && index >= 0 &&
                  index < out.numel() && out.is_contiguous(),
              "hier_output_dist_clock: a contiguous CUDA int64 tensor and an index in it");
  c10::cuda::CUDAGuard guard(out.device());
#if DEMB_HIER_TRACE
  clock_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>() + index);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
