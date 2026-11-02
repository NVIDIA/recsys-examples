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
// The operator's construction, teardown and calls, its communicator, its
// window and its trace (hier_output_dist.h).

#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/StringUtil.h>
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "hier_output_dist/common.h"
#include "hier_output_dist/config.h"
#include "hier_output_dist/hier_output_dist.h"
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
bool g_fail_alloc = false;

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
      return "it could not allocate its window";
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
      trace_(current_device()),
      device_(current_device()) {
  c10::cuda::CUDAGuard guard(device_);
  const bool fail_alloc = g_fail_alloc;
  g_fail_alloc = false;
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
    } catch (const std::exception &) {  // out of memory included
      readiness = kNoMemory;
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
  out_ = std::make_unique<OutChannel>(config_, sizes_, channel_static_, *window_, trace_, device_);
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
// kernel. If the device sync reports an error (a kernel that trapped), or
// the exit round reports a CUDA error, or a peer does not reach its teardown
// in time, the op keeps its window and communicator until the process exits.
// Else it lets go of its window, whose memory goes with the last view of
// lookup_rows.
void HierOutputDist::release() noexcept {
  c10::cuda::OptionalCUDAGuard guard;
  try {
    guard.set_device(at::Device(at::kCUDA, device_));  // Python may free the op on any thread
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      fprintf(stderr,
              "[hier_output_dist] rank %d: CUDA error during the teardown of the op: %s;"
              " keeping the window until the process exits\n",
              sizes_.topo.rank, cudaGetErrorString(err));
      window_->release(/*keep=*/true);
      return;
    }
    if (!quiesced_ && !exit_round()) {
      window_->release(/*keep=*/true);
      return;
    }
  } catch (...) {  // teardown must not throw: keep the window, as above
    window_->release(/*keep=*/true);
    return;
  }
  window_->deregister(communicator_->comm());
  communicator_->abort();
  window_->release(/*keep=*/false);
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
  in_->start(plan);
}

bool HierOutputDist::wait(Plan &plan) { return plan.wait(); }

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

// ── the communicator ──

std::string Communicator::unique_id() {
  ncclUniqueId id;
  HIER_NCCL_CHECK(ncclGetUniqueId(&id));
  return std::string(id.internal, sizeof(id.internal));
}

Communicator::Communicator(const std::string &unique_id, const Topo &topo) : topo_(topo) {
  TORCH_CHECK(unique_id.size() == sizeof(ncclUniqueId::internal),
              "hier_output_dist: a unique id is ", sizeof(ncclUniqueId::internal), " bytes, got ",
              unique_id.size());
  HIER_CUDA_CHECK(cudaGetDevice(&device_));
  CUdevice cu_device;
  HIER_CU_CHECK(cuDeviceGet(&cu_device, device_));
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cu_device;
  HIER_CU_CHECK(
      cuMemGetAllocationGranularity(&granularity_, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  ncclUniqueId id;
  memcpy(id.internal, unique_id.data(), sizeof(id.internal));
  HIER_NCCL_CHECK(ncclCommInitRank(&comm_, topo.world_size, id, topo.rank));
  try {
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    HIER_NCCL_CHECK(ncclCommQueryProperties(comm_, &props));
    TORCH_CHECK(props.deviceApiSupport, "hier_output_dist: NCCL device API not supported here");
    // One node: LSA only, no GIN context. Across nodes: the rails over GIN,
    // 4 contexts (a hint: the count is checked), 3 signals per node and 2
    // counters each. Without GIN the op cannot run (gin_ok).
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    const bool gin = topo.num_nodes > 1 && props.railedGinType != NCCL_GIN_TYPE_NONE;
    reqs.ginContextCount = gin ? kGinContexts : 0;
    reqs.ginSignalCount = gin ? kGinSignalsPerNode * topo.num_nodes : 0;
    reqs.ginCounterCount = gin ? kGinCounters : 0;
    reqs.ginConnectionType = gin ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_NONE;
    reqs.ginQueueDepth = gin ? 512 : 0;
    HIER_NCCL_CHECK(ncclDevCommCreate(comm_, &reqs, &dev_));
    gin_ok_ =
        gin && (int)dev_.ginContextCount >= kGinContexts && gin_signals_zero(dev_, topo.num_nodes);
  } catch (...) {
    abort();
    throw;
  }
}

// On the current stream: the caching allocator may hand `gathered` a block
// that kernels queued there still use, and only that stream's order puts the
// gather after them.
std::vector<int> Communicator::all_gather(int value) {
  c10::cuda::CUDAGuard guard(device_);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_);
  auto gathered = at::empty({topo_.world_size + 1},
                            at::TensorOptions().device(at::kCUDA, device_).dtype(at::kInt));
  int *mine = gathered.data_ptr<int>() + topo_.world_size;
  HIER_CUDA_CHECK(cudaMemcpyAsync(mine, &value, sizeof(int), cudaMemcpyHostToDevice, stream));
  HIER_NCCL_CHECK(ncclAllGather(mine, gathered.data_ptr<int>(), 1, ncclInt32, comm_, stream));
  HIER_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto host = gathered.narrow(0, 0, topo_.world_size).cpu();
  return std::vector<int>(host.data_ptr<int>(), host.data_ptr<int>() + topo_.world_size);
}

void Communicator::finish() noexcept {
  if (comm_ == nullptr) return;
  ncclResult_t result = ncclDevCommDestroy(comm_, &dev_);
  if (result == ncclSuccess) result = ncclCommFinalize(comm_);  // blocks until quiet everywhere
  if (result == ncclSuccess) result = ncclCommDestroy(comm_);
  if (result != ncclSuccess) ncclCommAbort(comm_);
  comm_ = nullptr;
}

void Communicator::abort() {
  if (comm_ == nullptr) return;
  ncclCommAbort(comm_);  // switches to the comm's device itself
  comm_ = nullptr;
}

// ── the window ──

// What must read 0 is zeroed and synchronized before returning: the
// registration that follows ends in a barrier, so no peer reaches it before.
SymmetricWindow::SymmetricWindow(const Sizes &sizes, int device)
    : device_(device), sizes_(sizes), bytes_(WindowLayout::bytes(sizes)) {
  c10::cuda::CUDAGuard guard(device_);
  char *base = nullptr;
  HIER_NCCL_CHECK(ncclMemAlloc((void **)&base, bytes_));
  memory_ = std::shared_ptr<char>(base, [device](char *copy) {
    c10::cuda::CUDAGuard guard(device);  // its last holder may be on any thread
    ncclMemFree(copy);
  });
  const WindowLayout window = layout();
  HIER_CUDA_CHECK(cudaMemset(window.ctrl.base(), 0, window.ctrl.bytes()));
  if (window.rows.push_flags_bytes() > 0)
    HIER_CUDA_CHECK(cudaMemset(window.rows.push_flags(0), 0, window.rows.push_flags_bytes()));
  HIER_CUDA_CHECK(cudaDeviceSynchronize());
}

void SymmetricWindow::register_window(ncclComm_t comm) {
  c10::cuda::CUDAGuard guard(device_);
  HIER_NCCL_CHECK(ncclCommWindowRegister(comm, memory_.get(), bytes_, &handle_, NCCL_WIN_DEFAULT));
  TORCH_CHECK(handle_ != nullptr, "hier_output_dist: symmetric window registration failed");
}

at::Tensor SymmetricWindow::lookup_rows(int64_t rows, int64_t cols, at::ScalarType dtype) const {
  TORCH_CHECK(cols * (int64_t)c10::elementSize(dtype) == sizes_.row_bytes,
              "hier_output_dist: a row of lookup_rows is ", sizes_.row_bytes, " bytes, not ", cols,
              " x ", dtype);
  TORCH_CHECK(rows >= 0 && rows <= sizes_.lookup_keys, "hier_output_dist: ", rows,
              " rows are more than lookup_rows holds (", sizes_.lookup_keys, ")");
  std::shared_ptr<char> memory = memory_;
  return at::from_blob(
      layout().rows.lookup_rows(), {rows, cols}, [memory](void *) {},
      at::TensorOptions().device(at::kCUDA, device_).dtype(dtype));
}

int64_t SymmetricWindow::bytes(size_t granularity) const {
  return (int64_t)((bytes_ + granularity - 1) / granularity * granularity);
}

// Deregistration is local: a node peer keeps its mapping of this rank's copy
// until it deregisters its own window.
void SymmetricWindow::deregister(ncclComm_t comm) noexcept {
  if (handle_ == nullptr) return;
  ncclCommWindowDeregister(comm, handle_);
  handle_ = nullptr;
}

void SymmetricWindow::release(bool keep) noexcept {
  if (keep) new std::shared_ptr<char>(memory_);  // never freed
  memory_.reset();
}

// ── the trace ──

Trace::Trace(int device) {
  if constexpr (kTrace) {
    out_ = at::zeros({2, kTraceRing, kTraceWords},
                     at::TensorOptions().device(at::kCUDA, device).dtype(at::kLong));
  }
}

// The earliest stamps start at UINT64_MAX, the rest at 0.
bool Trace::start_out(bool forward, OutCall &call, cudaStream_t stream) {
  if constexpr (kTrace) {
    const int direction = forward ? 0 : 1;
    const int64_t at = direction * kTraceRing + launches_[direction] % kTraceRing;
    int64_t *row = out_.data_ptr<int64_t>() + at * kTraceWords;
    HIER_CUDA_CHECK(cudaMemsetAsync(row, 0xff, kNumEarliest * sizeof(int64_t), stream));
    HIER_CUDA_CHECK(cudaMemsetAsync(row + kNumEarliest, 0,
                                    (kTraceWords - kNumEarliest) * sizeof(int64_t), stream));
    call.trace = (unsigned long long *)row;
    return true;
  }
  return false;
}

void Trace::launched(bool forward) {
  if constexpr (kTrace) ++launches_[forward ? 0 : 1];
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
