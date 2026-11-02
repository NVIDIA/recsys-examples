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
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/StringUtil.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>

#include "hier_output_dist/hier_output_dist.h"
#include "hier_output_dist/host/common.h"
#include "hier_output_dist/kernels.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

int current_device() {
  int device;
  HIER_CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

// The test hook of the next construction (fail_next_construction).
std::atomic<bool> g_fail_alloc{false};

// What a rank tells the others when they agree whether the op can run.
enum Readiness : int { kReady = 0, kNoLsaTeam, kNoGin, kNoMemory };

std::string why_not(int readiness, int local_size) {
  switch (readiness) {
    case kNoLsaTeam:
      return c10::str("the NCCL LSA team (NVLink domain) does not hold all ", local_size,
                      " ranks of its node");
    case kNoGin:
      return "NCCL GIN does not connect the nodes' rails, gives fewer than 4 contexts, or its "
             "signals do not start at 0";
    default:  // kNoMemory
      return "it could not allocate its window or event";
  }
}

// The part of the kernels' arguments both channels share, over the
// registered window.
ChannelStatic channel_static(const Sizes &sizes, ncclWindow_t window, const ncclDevComm &dev,
                             double timeout_s) {
  return ChannelStatic{sizes, window, (uint64_t)(std::min(timeout_s, 1e9) * 1e9), dev};
}

// Every rank's readiness (`every`, in rank order) in words: those of the
// ranks that cannot run, or "" when every rank can.
std::string why_some_cannot_run(const std::vector<int> &every, int local_size) {
  std::string why;
  for (int peer = 0; peer < (int)every.size(); ++peer)
    if (every[peer] != kReady)
      why +=
          c10::str(why.empty() ? "" : "; ", "rank ", peer, ": ", why_not(every[peer], local_size));
  return why;
}

// Part 1 runs the rules present: their codes on the host.
OwnerRules owner_rules(const Config &config) {
  const at::Tensor codes = config.dist_types.to(at::kCPU, at::kInt).contiguous();
  return OwnerRules{config.dist_types, config.block_sizes,
                    std::vector<int>(codes.data_ptr<int>(), codes.data_ptr<int>() + codes.numel())};
}

}  // namespace

void fail_next_construction(bool fail_alloc) { g_fail_alloc = fail_alloc; }

std::string HierOutputDist::unique_id() { return Communicator::unique_id(); }

std::string HierOutputDist::unsupported(int world_size, int local_size, int num_features,
                                        int64_t row_bytes) {
  return hier_output_dist::unsupported(world_size, local_size, num_features, row_bytes);
}

std::string HierOutputDist::invalid(const Config &config) {
  try {
    config.validate();
  } catch (const std::invalid_argument &error) {
    return error.what();
  }
  return "";
}

HierOutputDist::HierOutputDist(const std::string &unique_id, Config config)
    : config_(std::move(config)),
      sizes_(sizes_of(config_)),
      rules_(owner_rules(config_)),
      budget_(config_, sms_of(current_device())),
      trace_(current_device()),
      device_(current_device()) {
  c10::cuda::CUDAGuard guard(device_);
  const bool fail_alloc = g_fail_alloc.exchange(false);
  communicator_ = std::make_unique<Communicator>(unique_id, sizes_.topo);
  // Nothing of the op stays on any rank once the ranks know it cannot run.
  // When they all know it (together), the communicator ends on every rank at
  // once: a rank whose part of the agreement is done may still be sending a
  // peer its last data, which an abort would cut. An error of this rank alone
  // aborts it.
  auto give_up = [this](bool together) noexcept {
    if (window_ != nullptr) window_->deregister(communicator_->comm());
    if (together)
      communicator_->finish();
    else
      communicator_->abort();
    if (window_ != nullptr) window_->free(false);
    lookup_grad_.reset();
    window_.reset();
  };
  int readiness = kReady;
  if (!communicator_->lsa_ok())
    readiness = kNoLsaTeam;
  else if (!communicator_->gin_ok())
    readiness = kNoGin;
  if (readiness == kReady) {
    try {
      TORCH_CHECK(!fail_alloc, "hier_output_dist: an allocation failure a test asked for");
      window_ = std::make_unique<SymmetricWindow>(sizes_, device_);
      lookup_grad_ = std::make_shared<LookupGrad>();
    } catch (const std::exception &) {  // out of memory included
      readiness = kNoMemory;
      if (window_ != nullptr) window_->free(false);
      lookup_grad_.reset();
      window_.reset();
    }
  }
  // Before the collective registration: a rank that cannot run must not
  // leave the others waiting for it there.
  std::string why;
  try {
    why = why_some_cannot_run(communicator_->all_gather(readiness), sizes_.topo.local_size);
  } catch (...) {
    give_up(false);
    throw;
  }
  if (!why.empty()) {
    give_up(true);
    throw std::runtime_error("[hier_output_dist] the op cannot run (" + why + ")");
  }
  try {
    window_->register_window(communicator_->comm());
  } catch (...) {  // the peers may wait in the registration: nothing to agree on
    give_up(false);
    throw;
  }
  channel_static_ =
      channel_static(sizes_, window_->handle(), communicator_->dev(), config_.timeout_s);
  in_ = std::make_unique<InChannel>(config_, sizes_, channel_static_, *window_, device_);
  out_ = std::make_unique<OutChannel>(config_, sizes_, channel_static_, *window_, budget_, trace_,
                                      device_, lookup_grad_);
  // The other ops' kernels issued so far had their CTAs counted without this
  // op (CtaBudget): let them end first.
  HIER_CUDA_CHECK(cudaDeviceSynchronize());
  budget_.count();
}

HierOutputDist::~HierOutputDist() { release(); }

// Runs exit_kernel; true when every node and rail peer reached its teardown
// within the limit.
bool HierOutputDist::exit_round() noexcept {
  const double limit_s = std::min(config_.timeout_s, 30.0);
  cudaError_t err = launch_exit_kernel(channel_static_, (uint64_t)(limit_s * 1e9));
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  uint64_t closed = 0;
  if (err == cudaSuccess)
    err = cudaMemcpy(&closed, window_->layout().ctrl.closed(), 8, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "[hier_output_dist] rank %d: CUDA error during the teardown of the op: %s;"
            " keeping the window until the process exits\n",
            sizes_.topo.rank, cudaGetErrorString(err));
    return false;
  }
  if (closed != 1)
    fprintf(stderr,
            "[hier_output_dist] rank %d: a node or rail peer did not tear down its op within"
            " the timeout; keeping the window until the process exits\n",
            sizes_.topo.rank);
  return closed == 1;
}

// Python frees the op when the last reference to its sharding goes, which
// need not happen at the same point on every rank, so the teardown is local:
// no collective deregistration or finalize, which would deadlock against
// ranks still training. Node peers may read this rank's window after its own
// kernels have ended, and rail peers may still send it the credits each out
// kernel sends for the next one at its end, which nobody waits for. So
// before freeing it every node peer and every rail peer must have reached
// its teardown too (the exit round; a rail peer's closing signal comes
// behind its credits), unless every rank quiesced the op after its last
// kernel. If the device sync reports an error (a kernel that trapped), the
// op keeps its window, its communicator and its share of the CTA budget,
// since its kernels may still hold SMs. Once the sync succeeds, its kernels
// have ended, so it leaves the budget. If the exit round then reports a CUDA
// error, or a peer does not reach its teardown in time, the op keeps its
// window and communicator until the process exits.
void HierOutputDist::release() noexcept {
  c10::cuda::OptionalCUDAGuard guard;
  bool keep_window = false;
  try {
    guard.set_device(at::Device(at::kCUDA, device_));  // Python may free the op on any thread
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr,
              "[hier_output_dist] rank %d: CUDA error during the teardown of the op: %s;"
              " keeping the window until the process exits\n",
              sizes_.topo.rank, cudaGetErrorString(err));
      return;
    }
    budget_.uncount();  // its kernels have ended
    if (!quiesced_ && !exit_round()) return;
    // A view of lookup_rows still alive points into the window: keep its memory.
    keep_window = window_->live_views() > 0;
    if (keep_window)
      fprintf(stderr,
              "[hier_output_dist] rank %d: %lld views of the lookup output (lookup_rows) outlive"
              " the op; keeping the window's memory until the process exits\n",
              sizes_.topo.rank, (long long)window_->live_views());
  } catch (...) {  // teardown must not throw: keep the window, as above
    return;
  }
  window_->deregister(communicator_->comm());
  communicator_->abort();
  window_->free(keep_window);
}

std::shared_ptr<Plan> HierOutputDist::plan(PlanInput input) {
  c10::cuda::CUDAGuard guard(device_);
  at::NoGradGuard no_grad;
  TORCH_CHECK(input.weights.has_value() == config_.weights,
              "hier_output_dist: the keys carry weights exactly when the op was made with "
              "weights (",
              config_.weights ? "with" : "without", ")");
  Partition part = partition_keys(input, rules_, sizes_.topo.world_size);
  FanoutTables fanout;
  if (input.reverse_indices.has_value()) fanout = fanout_tables(*input.reverse_indices, part.inv);
  return std::make_shared<Plan>(sizes_, std::move(input), std::move(part), std::move(fanout));
}

void HierOutputDist::start(Plan &plan) {
  quiesced_ = false;
  in_->start(plan, out_->last_done());
}

bool HierOutputDist::wait(Plan &plan) {
  const bool fits = plan.wait();
  trace_.keep_step_times(plan.in_seq_, plan.step_events_);
  return fits;
}

at::Tensor HierOutputDist::lookup_rows(Plan &plan) { return out_->lookup_rows(plan); }

std::tuple<at::Tensor, std::vector<at::Tensor>> HierOutputDist::forward(
    Plan &plan, const at::Tensor &lookup_out) {
  quiesced_ = false;
  return out_->forward(plan, lookup_out);
}

at::Tensor HierOutputDist::backward(Plan &plan, const at::Tensor &grad,
                                    const std::vector<at::Tensor> &maps, bool divide) {
  quiesced_ = false;
  return out_->backward(plan, grad, maps, divide);
}

void HierOutputDist::quiesce() {
  c10::cuda::CUDAGuard guard(device_);
  HIER_CUDA_CHECK(cudaDeviceSynchronize());
  communicator_->barrier();
  quiesced_ = true;
}

std::map<std::string, int64_t> HierOutputDist::capacities() const {
  return {{"lookup_keys", sizes_.lookup_keys},
          {"remote_keys", sizes_.remote_keys},
          {"proxy_bytes", config_.capacities.proxy_bytes},
          {"num_proxy_rows", sizes_.num_proxy_rows},
          {"rounds_cap", sizes_.rounds_cap}};
}

// The ctrl section's buffer takes whole kBufferAlign units, as the keys and
// rows sections' buffers do.
std::map<std::string, std::map<std::string, int64_t>> HierOutputDist::memory() const {
  std::map<std::string, std::map<std::string, int64_t>> bytes;
  bytes["windows"]["window"] = window_->bytes(communicator_->granularity());
  const WindowLayout window = window_->layout();
  std::map<std::string, int64_t> &buffers = bytes["buffers"];
  buffers["ctrl"] = (int64_t)buffer_end(window.ctrl.bytes());
  for (const auto &buffer : window.keys.buffers()) buffers[buffer.first] = (int64_t)buffer.second;
  for (const auto &buffer : window.rows.buffers()) buffers[buffer.first] = (int64_t)buffer.second;
  return bytes;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
