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
// A rank's control words, by name.
#pragma once

#include <cstdint>

#include "hier_output_dist/common/ctrl.h"
#include "hier_output_dist/common/windows.h"
#include "hier_output_dist/device/sym_window.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The control words of node peer lsa_peer (this rank's own included): its
// ctrl region, where Ctrl gives the offset of each word (Ctrl says what each
// one holds). Each accessor returns the address of a word, or of the first of
// an array of them. The node barriers' and flags' words (started, pushed,
// pulled, in_written, in_done) are reached through NodeBarrier and PeerFlags
// by their Ctrl offsets.
struct CtrlWords {
  uint64_t *words;
  const Ctrl &at;
  __device__ __forceinline__ CtrlWords(const Win &win, const Ctrl &offsets, int lsa_peer)
      : words(SymRegion<kCtrl, uint64_t>(win).peer(lsa_peer)), at(offsets) {}
  template <typename Args>
  __device__ __forceinline__ static CtrlWords of(const Args &args, int lsa_peer) {
    return CtrlWords(args.win, args.ctrl_at, lsa_peer);
  }
  // The word at `offset` (a channel's node barrier word, Channel::started).
  __device__ __forceinline__ uint64_t *word(int offset) const { return &words[offset]; }

  // the out channel
  __device__ __forceinline__ uint64_t *arrive() const { return &words[at.arrive]; }
  __device__ __forceinline__ uint64_t *task(int queue) const { return &words[at.tasks + queue]; }
  __device__ __forceinline__ uint64_t *landed_rows() const { return words + at.landed_rows; }
  __device__ __forceinline__ uint64_t *landed_rows(int owner_node) const {
    return &words[at.landed_rows + owner_node];
  }
  __device__ __forceinline__ uint64_t *fwd_tot(int owner_node) const {
    return &words[at.fwd_tot + owner_node];
  }
  __device__ __forceinline__ uint64_t *bwd_tot(int proxy_node) const {
    return &words[at.bwd_tot + proxy_node];
  }
  __device__ __forceinline__ uint64_t *grant_sent(int owner_node) const {
    return &words[at.grant_sent + owner_node];
  }
  __device__ __forceinline__ uint64_t *bwd_ready_sent(int proxy_node) const {
    return &words[at.bwd_ready_sent + proxy_node];
  }
  __device__ __forceinline__ uint64_t *posted(int half) const { return &words[at.posted + half]; }
  // A requester's two done words: of its even and of its odd rounds.
  __device__ __forceinline__ uint64_t *done(int requester) const {
    return words + at.done + 2 * requester;
  }
  __device__ __forceinline__ uint64_t *done(int requester, int parity) const {
    return &words[at.done + 2 * requester + parity];
  }
  __device__ __forceinline__ uint64_t *half_free() const { return &words[at.half_free]; }

  // the in channel
  __device__ __forceinline__ int64_t *cell_start() const {
    return (int64_t *)(words + at.cell_start);
  }
  // The needs (Need) after node_status_exchange's word.
  __device__ __forceinline__ uint64_t *in_done_need(int need) const {
    return &words[at.in_done + 1 + need];
  }
  __device__ __forceinline__ int64_t *count_msg_out() const {
    return (int64_t *)(words + at.count_msg_out);
  }

  // teardown
  __device__ __forceinline__ uint64_t *closing() const { return &words[at.closing]; }
  __device__ __forceinline__ uint64_t *closed() const { return &words[at.closed]; }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
