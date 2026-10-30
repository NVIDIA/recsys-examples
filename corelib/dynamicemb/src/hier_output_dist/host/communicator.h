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
// The op's own NCCL communicator: its device communicator (LSA inside the
// node, and across nodes GIN on the rails), what the device can do, and a
// collective the constructor agrees with.
#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <cstddef>
#include <string>
#include <vector>

#include "hier_output_dist/common/topology.h"

namespace dyn_emb {
namespace hier_output_dist {

class Communicator {
 public:
  // Collective: every rank of `topo` with the same unique id (unique_id()).
  // A rank that fails here raises; the others then wait in NCCL's own
  // initialization, as with any communicator.
  Communicator(const std::string &unique_id, const Topo &topo);
  Communicator(const Communicator &) = delete;
  Communicator &operator=(const Communicator &) = delete;
  // The owner ends the communicator (abort() or finish()), or keeps it until the process
  // exits when a peer may still need it: the destructor does neither.
  ~Communicator() = default;

  // A new unique id (on one rank; the caller hands it to the others).
  static std::string unique_id();

  ncclComm_t comm() const { return comm_; }
  const ncclDevComm &dev() const { return dev_; }
  // This rank's parts of whether the op can run: the LSA team is its node,
  // and across nodes GIN connects the rails with at least 4 contexts, whose
  // signals and counters start at 0.
  bool lsa_ok() const { return dev_.lsaSize == topo_.local_size; }
  bool gin_ok() const { return topo_.num_nodes == 1 || gin_ok_; }
  // Across nodes, whether NCCL connected GIN on the rails only, as asked (it
  // does so only when NCCL_CROSS_NIC is 0; else GIN's memory grows with the
  // ranks).
  bool gin_railed() const { return dev_.ginConnectionsRailed; }
  // What ncclMemAlloc rounds an allocation up to.
  size_t granularity() const { return granularity_; }
  // Every rank's `value`, in rank order (collective, synchronous).
  std::vector<int> all_gather(int value);
  // Returns once every rank has called it (collective, synchronous).
  void barrier() { all_gather(0); }
  // Ends the communicator on this rank alone (no collective).
  void abort();
  // Ends the communicator on every rank together (collective): each rank's
  // last communications reach their peers before any rank frees its
  // resources, which abort() does not wait for.
  void finish() noexcept;

 private:
  Topo topo_;
  ncclComm_t comm_ = nullptr;
  ncclDevComm dev_{};
  bool gin_ok_ = false;
  int device_ = 0;
  size_t granularity_ = 0;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
