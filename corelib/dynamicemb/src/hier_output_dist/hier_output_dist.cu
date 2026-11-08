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
// The operator's construction, teardown and calls, its communicator and its
// window (hier_output_dist.h), and the kernels of its setup and teardown.

#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/StringUtil.h>
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/hier_output_dist.h"

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
      return "NCCL GIN does not connect the nodes' rails, gives fewer than 4 contexts, its "
             "connections are of different backends or of none the kernels are built for, or "
             "its signals do not start at 0";
    default:  // kNoMemory
      return "it could not allocate its window";
  }
}

// Signals and counters of every GIN context start at 0: `bad` counts those
// that do not.
__global__ void gin_zero_kernel(ncclDevComm dev, int signals, int counters, int *bad) {
  ncclGin gin(dev, blockIdx.x);
  for (int signal = threadIdx.x; signal < signals; signal += blockDim.x)
    if (gin.readSignal(signal) != 0) atomicAdd(bad, 1);
  for (int counter = threadIdx.x; counter < counters; counter += blockDim.x)
    if (gin.readCounter(counter) != 0) atomicAdd(bad, 1);
}

// Teardown, 1 warp: tell the node this rank reads no window any more (its
// closing word) and, across nodes, its rail peers that it sends nothing more
// (a closing signal on the credits' context, behind every credit it sent
// them on the same QPs), both before waiting for anyone; then wait (at most
// limit_ns) until every node peer and every rail peer said the same, each
// lane taking peers in turn. Records in its closed word whether they all
// did. Its waits are loops of their own: after limit_ns they give up and
// record it, where Spin would trap.
__global__ void exit_kernel(ncclWindow_t window, ncclDevComm dev, Sizes sizes, uint64_t limit_ns) {
  const Topo &topo = sizes.topo;
  const int lane = threadIdx.x, num_nodes = topo.num_nodes;
  const NodeWindow node(window, topo.local_rank);
  const CtrlLayout ctrl = node.layout(sizes).ctrl;
  if (lane == 0) st_release_sys(ctrl.closing(), 1);
  if (num_nodes > 1) {
    ncclGin cred(dev, kCtxCredit);
    for (int peer_node = lane; peer_node < num_nodes; peer_node += 32)
      if (peer_node != topo.node)
        cred.signal(ncclTeamRail(dev), peer_node,
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
    ncclGin cred(dev, kCtxCredit);
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

cudaError_t launch_exit_kernel(ncclWindow_t window, const ncclDevComm &dev, const Sizes &sizes,
                               uint64_t limit_ns) {
  exit_kernel<<<1, 32>>>(window, dev, sizes, limit_ns);
  return cudaGetLastError();
}

// The GIN backend of every connection of `dev` (ncclGinType_t), or
// NCCL_GIN_TYPE_NONE when they differ or it is not one the out kernels are
// built for (gin_context).
ncclGinType_t gin_type_of(const ncclDevComm &dev) {
  if (dev.ginConnectionCount == 0) return NCCL_GIN_TYPE_NONE;
  const int type = dev.ginNetDeviceTypes[0];
  for (int c = 1; c < dev.ginConnectionCount; ++c)
    if (dev.ginNetDeviceTypes[c] != type) return NCCL_GIN_TYPE_NONE;
  return type == NCCL_GIN_TYPE_PROXY || type == NCCL_GIN_TYPE_GDAKI || type == NCCL_GIN_TYPE_GPI
             ? (ncclGinType_t)type
             : NCCL_GIN_TYPE_NONE;
}

// Signals and counters start at 0 (NCCL zeroes them): every comparison is
// against running totals from 0. Checked before any peer can send, on the
// current stream: the caching allocator may hand `bad` a block that kernels
// queued there still use, and only that stream's order puts these writes
// after them.
bool gin_signals_zero(const ncclDevComm &dev, int num_nodes, int counters) {
  int device;
  HIER_CUDA_CHECK(cudaGetDevice(&device));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  auto bad = at::empty({1}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kInt));
  HIER_CUDA_CHECK(cudaMemsetAsync(bad.data_ptr<int>(), 0, sizeof(int), stream));
  gin_zero_kernel<<<kGinContexts, 32, 0, stream>>>(dev, kGinSignalsPerNode * num_nodes, counters,
                                                   bad.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return bad.item<int>() == 0;
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

// `config` once it passed its checks (Config::validate throws).
Config validated(Config config) {
  config.validate();
  return config;
}

}  // namespace

void fail_next_construction(bool fail_alloc) { g_fail_alloc = fail_alloc; }

HierOutputDist::HierOutputDist(const std::string &unique_id, int64_t rank, int64_t world_size,
                               int64_t local_size, const std::vector<int64_t> &feature_tables,
                               const at::Tensor &block_sizes, const at::Tensor &dist_types,
                               int64_t dim, at::ScalarType dtype, bool weights, bool node_dedup,
                               int64_t lookup_keys, int64_t remote_keys, int64_t proxy_bytes,
                               double timeout_s, int64_t ctas, int64_t reduce_ctas,
                               int64_t chunk_kb)
    : config_(
          validated(config_of(rank, world_size, local_size, feature_tables, block_sizes, dist_types,
                              dim, dtype, weights, node_dedup, lookup_keys, remote_keys,
                              proxy_bytes, timeout_s, ctas, reduce_ctas, chunk_kb))),
      sizes_(sizes_of(config_)),
      device_(current_device()) {
  c10::cuda::CUDAGuard guard(device_);
  const bool fail_alloc = g_fail_alloc;
  g_fail_alloc = false;
  communicator_ = std::make_unique<Communicator>(unique_id, sizes_.topo, (int)sizes_.counters());
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
  in_ = std::make_unique<InChannel>(config_, sizes_, window_->handle(), communicator_->dev(),
                                    window_->layout(), device_);
  out_ = std::make_unique<OutChannel>(config_, sizes_, window_->handle(), communicator_->dev(),
                                      communicator_->gin_type(), window_->layout(), device_);
  if (sizes_.multi_node() && !communicator_->gin_railed() && sizes_.topo.rank == 0)
    fprintf(stderr,
            "[hier_output_dist] NCCL connected GIN to every rank, not only to the rails it uses, "
            "so its queues take memory that grows with the number of ranks; set "
            "NCCL_CROSS_NIC=0 for the job to connect the rails alone\n");
}

HierOutputDist::~HierOutputDist() { release(); }

// Runs exit_kernel; true when every node and rail peer reached its teardown
// within the limit.
bool HierOutputDist::exit_round() noexcept {
  const double limit_s = std::min(config_.timeout_s, 30.0);
  cudaError_t err = launch_exit_kernel(window_->handle(), communicator_->dev(), sizes_,
                                       (uint64_t)(limit_s * 1e9));
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

HierOutputDist::InDistTensors HierOutputDist::input_dist(const at::Tensor &values,
                                                         const c10::optional<at::Tensor> &offsets,
                                                         const c10::optional<at::Tensor> &lengths,
                                                         int64_t batch_size,
                                                         const c10::optional<at::Tensor> &weights,
                                                         const at::Tensor &input_offsets,
                                                         const at::Tensor &reverse_indices) {
  quiesced_ = false;
  PlanInput input;
  input.values = values;
  input.offsets = offsets;
  input.lengths = lengths;
  input.batch_size = batch_size;
  input.weights = weights;
  input.input_offsets = input_offsets;
  input.reverse_indices = reverse_indices;
  const InDist dist = in_->input_dist(input);
  c10::cuda::CUDAGuard guard(device_);
  return {dist.kjt_values,
          dist.kjt_weights,
          dist.kjt_lengths,
          dist.kjt_offsets,
          window_->lookup_rows(sizes_.lookup_keys, config_.dim, config_.dtype),
          dist.status,
          dist.fwd_plan.tensors(),
          dist.bwd_plan.tensors()};
}

at::Tensor HierOutputDist::output_dist_fwd(const std::vector<c10::optional<at::Tensor>> &fwd_plan,
                                           const at::Tensor &lookup_out) {
  quiesced_ = false;
  return out_->forward(FwdPlan::of(fwd_plan), lookup_out);
}

at::Tensor HierOutputDist::output_dist_bwd(const std::vector<c10::optional<at::Tensor>> &bwd_plan,
                                           const std::vector<int64_t> &counts,
                                           const at::Tensor &grad, bool divide) {
  quiesced_ = false;
  const BwdPlan plan = BwdPlan::of(bwd_plan, counts);
  out_->backward(plan, grad, divide);
  c10::cuda::CUDAGuard guard(device_);
  return window_->lookup_rows(plan.num_recv, grad.size(1), grad.scalar_type());
}

void HierOutputDist::quiesce() {
  c10::cuda::CUDAGuard guard(device_);
  HIER_CUDA_CHECK(cudaDeviceSynchronize());
  communicator_->barrier();
  quiesced_ = true;
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

Communicator::Communicator(const std::string &unique_id, const Topo &topo, int counters)
    : topo_(topo) {
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
    // 4 contexts (a hint: the count is checked), 3 signals per node and
    // `counters` counters each, all of one backend the out kernels are built
    // for (gin_type). Without GIN the op cannot run (gin_ok).
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    const bool gin = topo.num_nodes > 1 && props.railedGinType != NCCL_GIN_TYPE_NONE;
    reqs.ginContextCount = gin ? kGinContexts : 0;
    reqs.ginSignalCount = gin ? kGinSignalsPerNode * topo.num_nodes : 0;
    reqs.ginCounterCount = gin ? counters : 0;
    reqs.ginConnectionType = gin ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_NONE;
    reqs.ginQueueDepth = gin ? 512 : 0;
    HIER_NCCL_CHECK(ncclDevCommCreate(comm_, &reqs, &dev_));
    gin_type_ = gin ? gin_type_of(dev_) : NCCL_GIN_TYPE_NONE;
    gin_ok_ = gin_type_ != NCCL_GIN_TYPE_NONE && (int)dev_.ginContextCount >= kGinContexts &&
              gin_signals_zero(dev_, topo.num_nodes, counters);
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

}  // namespace hier_output_dist
}  // namespace dyn_emb
