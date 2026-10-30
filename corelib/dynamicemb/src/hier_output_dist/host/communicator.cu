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
#include <cuda.h>

#include <cstring>

#include "hier_output_dist/common/gin_signals.h"
#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/communicator.h"
#include "hier_output_dist/kernels.h"

namespace dyn_emb {
namespace hier_output_dist {

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

}  // namespace hier_output_dist
}  // namespace dyn_emb
