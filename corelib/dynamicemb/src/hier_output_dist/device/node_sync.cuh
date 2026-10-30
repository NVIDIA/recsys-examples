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
// Syncs between the ranks of a node through their control words: the node
// barrier (each rank publishes its own word and waits for every peer's),
// flags per pair of node peers (each rank sets its flag in every peer's words
// and waits for every peer's in its own), and words that carry a payload
// beside the sequence number (TaggedWord). The words hold the channel's
// sequence number, so none is ever reset: a later kernel waits for a larger
// one.
#pragma once

#include <cstdint>

#include "hier_output_dist/device/ctrl_words.cuh"
#include "hier_output_dist/device/memory_order.cuh"
#include "hier_output_dist/device/spin.cuh"

namespace dyn_emb {
namespace hier_output_dist {

// The node barrier on one control word, for a channel's kernel (Args, CommArgs
// or OutArgs, gives its windows, Ctrl, Channel, Topo and timeout): one thread
// publishes this rank's sequence number in its word; threads [0, local_size)
// then wait, each until one node peer's word has it. That peer has then done
// everything it did before it published (its release, our acquire): all its
// earlier work on the channel too (its kernels run in issue order).
template <typename Args>
struct NodeBarrier {
  const Args &args;
  int word;  // Ctrl offset of the word
  __device__ __forceinline__ NodeBarrier(const Args &channel_args, int offset)
      : args(channel_args), word(offset) {}
  // One thread: this rank's word (in its control words `mine`) gets the
  // sequence number.
  __device__ __forceinline__ void publish(const CtrlWords &mine) const {
    st_release_sys(mine.word(word), (uint64_t)args.ch.seq);
  }
  // Waits until node peer `peer` has published (spin reports `what`).
  __device__ __forceinline__ void wait(int peer, const char *what) const {
    Spin spin(args.timeout_ns);
    const CtrlWords peer_words = CtrlWords::of(args, peer);
    while ((int64_t)ld_acquire_sys(peer_words.word(word)) < args.ch.seq)
      spin.pause(what, peer, args.ch.seq, args.topo.rank);
  }
};

// Flags of the node's ranks, one per pair: node peer r's flag in a rank's
// control words is at word + r. Each rank tells every peer (sets its flag in
// the peer's words to the sequence number), then waits until every peer told
// it. Args as for NodeBarrier.
template <typename Args>
struct PeerFlags {
  const Args &args;
  int word;  // Ctrl offset of node peer 0's flag
  __device__ __forceinline__ PeerFlags(const Args &channel_args, int offset)
      : args(channel_args), word(offset) {}
  // This rank (node peer `me`) tells node peer `peer`.
  __device__ __forceinline__ void tell(int peer, int me) const {
    st_release_sys(CtrlWords::of(args, peer).word(word + me), (uint64_t)args.ch.seq);
  }
  // Waits until node peer `peer` told this rank (node peer `me`).
  __device__ __forceinline__ void wait(int me, int peer, const char *what) const {
    Spin spin(args.timeout_ns);
    const uint64_t *flag = CtrlWords::of(args, me).word(word + peer);
    while ((int64_t)ld_acquire_sys(flag) < args.ch.seq)
      spin.pause(what, peer, args.ch.seq, args.topo.rank);
  }
};

// A control word that holds a channel's sequence number in its high bits and
// a payload in its low PayloadBits bits, which its writer publishes (release)
// and readers wait for (acquire): landed_rows (rows, 32 bits).
template <int PayloadBits>
struct TaggedWord {
  __device__ __forceinline__ static uint64_t of(int64_t seq, uint64_t payload) {
    return ((uint64_t)seq << PayloadBits) | payload;
  }
  __device__ __forceinline__ static uint64_t payload(uint64_t word) {
    return word & ((1ull << PayloadBits) - 1);
  }
  // One thread: *word gets sequence number seq and the payload.
  __device__ __forceinline__ static void publish(uint64_t *word, int64_t seq, uint64_t payload) {
    st_release_sys(word, of(seq, payload));
  }
  // Waits until *word holds sequence number seq or a later one, and returns
  // it (a timeout reports `what`, the peer, seq and this rank).
  __device__ __forceinline__ static uint64_t wait(const uint64_t *word, int64_t seq, Spin &spin,
                                                  const char *what, int peer, int rank) {
    uint64_t value;
    while (((value = ld_acquire_sys(word)) >> PayloadBits) < (uint64_t)seq)
      spin.pause(what, peer, seq, rank);
    return value;
  }
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
