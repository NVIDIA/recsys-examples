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
// The rail: GIN between this rank and the ranks of its local index on the
// other nodes. The credits rail peers raise each other, puts of rows, and the
// in channel's messages. A GIN warp's lane n works with the rail peer on node
// n (at most 32 nodes).
#pragma once

#include <nccl_device.h>

#include <cstdint>

#include "hier_output_dist/device/memory_order.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// Credits on the rail: payload-free signals of GIN context kCtxCredit (the
// caller's `gin`) whose value only grows. A sender adds the difference to the
// value it sent before, which it keeps in a word of its own, so its peer
// reads the latest value.
struct Credits {
  const ncclGin &gin;
  const ncclTeam &rail;
  __device__ __forceinline__ Credits(const ncclGin &context, const ncclTeam &team)
      : gin(context), rail(team) {}
  // Raises this rank's credit `sig` at the rail peer on node peer_node to
  // `value` (nothing when *sent, what it sent before, is that much already).
  // The signal makes a system-scope release of its own: what this thread did
  // before is ordered before the credit.
  __device__ __forceinline__ void raise(int peer_node, int sig, uint64_t value,
                                        uint64_t *sent) const {
    if (value <= *sent) return;
    gin.signal(rail, peer_node, ncclGin_StrongSignalAdd{(ncclGinSignal_t)sig, value - *sent},
               ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
               cuda::thread_scope_system);
    *sent = value;
  }
  // The credit `sig` that rail peers raised at this rank (acquire).
  __device__ __forceinline__ uint64_t read(int sig) const { return gin.readSignal(sig); }
  // Every lane of the warp: once it returns, no credit the warp raised is in
  // flight.
  __device__ __forceinline__ void flush_warp() const { gin.flush(ncclCoopWarp{}); }
};

// Puts of rows on the rail, from one region of this rank's window `win` into
// another of a rail peer's (rows of row_bytes; the regions at byte offsets
// src_base and dst_base of the window), each adding its rows to the peer's
// signal `sig`. The signal is strong: a peer that sees it also sees this put
// and every earlier one on the context to it. The puts make no release of
// their own: they are told the caller made a system-scope release, which is
// the scope they need. So the caller makes one system-scope fence (fence())
// after the writes they carry and before them.
struct RowPut {
  const ncclGin &gin;
  const ncclTeam &rail;
  ncclWindow_t win;
  uint64_t dst_base, src_base;
  int64_t row_bytes;
  int sig;
  __device__ __forceinline__ void fence() const { fence_sc_sys(); }
  // Rows [src_row, src_row + rows) into [dst_row, dst_row + rows) of the rail
  // peer on node peer_node; with `count`, the put also counts on this rank's
  // counter `counter` once the NIC has read its rows.
  __device__ __forceinline__ void put(int peer_node, int dst_row, int src_row, int rows, bool count,
                                      int counter) const {
    const uint64_t dst = dst_base + (uint64_t)dst_row * row_bytes;
    const uint64_t src = src_base + (uint64_t)src_row * row_bytes;
    const ncclGin_StrongSignalAdd add{(ncclGinSignal_t)sig, (uint64_t)rows};
    if (!count)
      gin.put(rail, peer_node, win, dst, win, src, (size_t)rows * row_bytes, add, ncclGin_None{},
              ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_system,
              cuda::thread_scope_system);
    else
      gin.put(rail, peer_node, win, dst, win, src, (size_t)rows * row_bytes, add,
              ncclGin_WeakCounterInc{(ncclGinCounter_t)counter}, ncclCoopThread{}, ncclGin_None{},
              cuda::thread_scope_system, cuda::thread_scope_system);
  }
};

// One kind of the in channel's messages on the rail (GIN context kCtxIn, the
// caller's `gin`): a count message or a key stream. Every call sends each
// rail peer one message of each kind, which adds 1 to the peer's signal of
// that kind for this rank's node (`sig`). So a signal counts the calls whose
// message from that node has landed, a running total that reaches in_seq
// once this call's has. A message is one put from this rank's window into the
// peer's, without a release of its own (as RowPut's: the caller makes one
// system-scope fence after the writes it carries and before the put), or,
// when there is nothing to carry, the signal alone (with a release of its
// own).
struct RailMessage {
  const ncclGin &gin;
  const ncclTeam &rail;
  const ncclGin_StrongSignalAdd sig;
  __device__ __forceinline__ RailMessage(const ncclGin &context, const ncclTeam &team, int signal)
      : gin(context), rail(team), sig{(ncclGinSignal_t)signal, 1} {}
  // Bytes [src, src + bytes) of window `win` into [dst, dst + bytes) of the
  // rail peer's on node peer_node.
  __device__ __forceinline__ void put(int peer_node, ncclWindow_t win, uint64_t dst, uint64_t src,
                                      size_t bytes) const {
    gin.put(rail, peer_node, win, dst, win, src, bytes, sig, ncclGin_None{}, ncclCoopThread{},
            ncclGin_None{}, cuda::thread_scope_system, cuda::thread_scope_system);
  }
  // Nothing to carry: the signal alone.
  __device__ __forceinline__ void signal(int peer_node) const {
    gin.signal(rail, peer_node, sig, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
               cuda::thread_scope_system);
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
