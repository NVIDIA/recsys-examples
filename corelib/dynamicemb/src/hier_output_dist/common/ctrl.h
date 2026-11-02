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
// The ctrl section of the window (CtrlLayout): the control words of both
// channels and the device state they keep across calls. The formats of two
// messages that live there, the count vector and the count message; and a
// channel's sequence number (Channel).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "hier_output_dist/common/constants.h"
#include "hier_output_dist/common/sizes.h"
#include "hier_output_dist/common/status.h"

namespace dyn_emb {
namespace hier_output_dist {

// A rank's count vector (int64 words, plan/node.h): the keys it pushes each
// GPU of its node, for their remote parts, then for each feature of their
// own parts.
struct CountVector {
  int64_t local_size, num_features;
  __host__ __device__ int64_t words() const { return local_size * (num_features + 1); }
  // Its keys for the remote part of node peer `target`.
  __host__ __device__ int64_t remote(int64_t target) const { return target; }
  // Its keys of `feature` for the own part of node peer `target`.
  __host__ __device__ int64_t own(int64_t target, int64_t feature) const {
    return local_size + target * num_features + feature;
  }
};

// A count message (int64 words, plan/rail.h), all a rail peer needs to lay
// out the keys its rail puts: a common part, the same for every peer (its
// node's largest remote part, its rounds as proxy, the keys of its own node
// it looks up as owner, and its node-unique keys per owner node), then the
// part for the peer's node alone (its node-unique keys of each feature
// there).
struct CountMessage {
  static constexpr int kNeedRemote = 0, kRounds = 1, kOwnTotal = 2;
  int64_t num_nodes, num_features;
  __host__ __device__ int64_t uniq_out(int64_t owner_node) const { return 3 + owner_node; }
  __host__ __device__ int64_t common() const { return 3 + num_nodes; }
  // A message as a peer gets it: the common part, then the part for its node.
  __host__ __device__ int64_t length() const { return common() + num_features; }
  // What a rank sends from: the common part, then the part for each node.
  __host__ __device__ int64_t out_words() const { return common() + num_nodes * num_features; }
  __host__ __device__ int64_t part(int64_t node) const { return common() + node * num_features; }
  // A slot a message lands in: whole 16-byte units.
  __host__ __device__ int64_t slot_words() const { return (length() + 1) / 2 * 2; }
};

// Kernels of one channel run one at a time, in issue order, and share a
// sequence number, the same on every rank. The in channel is the input dist
// (side stream), whose sequence number counts calls; the out channel the
// output dist's kernels (the caller's stream), whose number counts them.
struct Channel {
  int64_t seq;
};

// The ctrl section (uint64 words): every control word of both channels and
// the device state they keep across calls. Its size depends only on
// local_size, num_nodes and num_features. Every control word is tagged with
// its channel's sequence number, is a running total, or is back at its start
// value when the kernel that uses it ends. The data words (the count vectors,
// cell_start, the needs, the count messages) are written whole in each call,
// before a tagged word, the node barrier or a signal publishes them. So none
// is ever reset, and a call that overflowed leaves all of them valid for the
// next. Words marked (local) are only this rank's: a GIN warp's place with
// each peer, which it sets when it starts. Words marked (num_nodes > 1) exist
// across nodes only. Each accessor returns the address of a word, or of the
// first of an array of them, in the copy at `base`.
class CtrlLayout {
 public:
  __host__ __device__ CtrlLayout(const Sizes &sizes, void *base)
      : base_((uint64_t *)base),
        num_nodes_(sizes.topo.num_nodes),
        num_features_(sizes.num_features),
        count_{sizes.topo.local_size, sizes.num_features},
        message_{sizes.topo.num_nodes, sizes.num_features} {
    const int64_t local_size = sizes.topo.local_size, num_nodes = num_nodes_;
    const bool multi = num_nodes > 1;
    int64_t next = 0;
    auto take = [&](int64_t words) {
      const int64_t at = next;
      next += words;
      return at;
    };
    in_started_ = take(1);
    count_vectors_ = take(2 * count_.words());
    in_written_ = take(local_size);
    cell_start_ = take(multi ? local_size * num_features_ : 0);
    in_done_ = take(multi ? 1 + kNumNeeds : 0);
    count_msg_out_ = take(multi ? message_.out_words() : 0);
    count_msgs_ = take(multi ? 2 * num_nodes * message_.slot_words() : 0);
    out_started_ = take(1);
    pushed_ = take(local_size);
    pulled_ = take(1);
    arrive_ = take(1);
    tasks_ = take(kNumQueues);
    landed_rows_ = take(multi ? num_nodes : 0);
    fwd_tot_ = take(multi ? num_nodes : 0);
    bwd_tot_ = take(multi ? num_nodes : 0);
    grant_sent_ = take(multi ? num_nodes : 0);
    bwd_ready_sent_ = take(multi ? num_nodes : 0);
    put_at_ = take(multi ? 2 * num_nodes : 0);
    relay_want_ = take(multi ? 2 * local_size : 0);
    posted_ = take(multi ? 2 : 0);
    done_ = take(multi ? 2 * local_size : 0);
    half_free_ = take(multi ? 1 : 0);
    closing_ = take(1);
    closed_ = take(1);
    words_ = next;
  }
  // The section: its first word and its bytes.
  __host__ __device__ uint64_t *base() const { return base_; }
  __host__ __device__ uint64_t bytes() const { return 8 * (uint64_t)words_; }

  // ── the in channel ──
  // in_seq of this rank's current node_count_exchange (its node barrier)
  __host__ __device__ uint64_t *in_started() const { return word(in_started_); }
  // This rank's count vector (CountVector) in call in_seq, by its parity.
  __host__ __device__ int64_t *count_vector(int64_t in_seq) const {
    return (int64_t *)word(count_vectors_ + (in_seq & 1) * count_.words());
  }
  // [local_size] in_seq whose keys each requester pushed into node_keys
  __host__ __device__ uint64_t *in_written() const { return word(in_written_); }
  // (num_nodes > 1) [num_features] where the cells (feature, requester) of a
  // same-node requester start in this rank's KJT
  __host__ __device__ int64_t *cell_start(int64_t requester) const {
    return (int64_t *)word(cell_start_ + requester * num_features_);
  }
  // (num_nodes > 1) in_seq of this rank's node_status_exchange
  __host__ __device__ uint64_t *in_done() const { return word(in_done_); }
  // (num_nodes > 1) [kNumNeeds] its rail's needs, which in_done publishes
  __host__ __device__ int64_t *needs() const { return (int64_t *)word(in_done_ + 1); }
  // (num_nodes > 1) the count message this rank sends its rail peers
  // (CountMessage::out_words)
  __host__ __device__ int64_t *count_msg_out() const { return (int64_t *)word(count_msg_out_); }
  // (num_nodes > 1) the count message of call in_seq from the rail peer on
  // `node` (this rank's own too), by in_seq parity: a peer's message of call
  // seq + 2 needs this rank's of seq + 1, which it sends once call seq ended
  // here.
  __host__ __device__ int64_t *count_msg(int64_t in_seq, int64_t node) const {
    return (int64_t *)word(count_msgs_ +
                           ((in_seq & 1) * num_nodes_ + node) * message_.slot_words());
  }

  // ── the out channel ──
  // out_seq of this rank's current output kernel (its node barrier)
  __host__ __device__ uint64_t *out_started() const { return word(out_started_); }
  // [local_size] out_seq whose gradient rows each requester pushed into lookup_rows
  __host__ __device__ uint64_t *pushed() const { return word(pushed_); }
  // out_seq after this rank's last pull of a forward (exit barrier)
  __host__ __device__ uint64_t *pulled() const { return word(pulled_); }
  // (local) CTAs of this rank's out kernels that reached their end, a running total
  __host__ __device__ uint64_t *arrive() const { return word(arrive_); }
  // (local) task queue `queue` of the current out kernel, back at 0 when it ends
  __host__ __device__ uint64_t *task(int queue) const { return word(tasks_ + queue); }
  // (num_nodes > 1) (out_seq << 32) | rows of an owner node's that landed in
  // this proxy's proxy_rows during that forward (the relay)
  __host__ __device__ uint64_t *landed_rows(int64_t owner_node) const {
    return word(landed_rows_ + owner_node);
  }
  // (num_nodes > 1, local) rows an owner node put into proxy_rows in earlier
  // forwards: what its signal held when this forward began
  __host__ __device__ uint64_t *fwd_tot(int64_t owner_node) const {
    return word(fwd_tot_ + owner_node);
  }
  // (num_nodes > 1, local) the same of a proxy node's gradient rows
  __host__ __device__ uint64_t *bwd_tot(int64_t proxy_node) const {
    return word(bwd_tot_ + proxy_node);
  }
  // (num_nodes > 1, local) the grant this proxy sent a rail owner
  __host__ __device__ uint64_t *grant_sent(int64_t owner_node) const {
    return word(grant_sent_ + owner_node);
  }
  // (num_nodes > 1, local) the bwd_ready this owner sent a rail proxy
  __host__ __device__ uint64_t *bwd_ready_sent(int64_t proxy_node) const {
    return word(bwd_ready_sent_ + proxy_node);
  }
  // (num_nodes > 1, local) [2] an issuer's place in its put list for the
  // rail peer on node peer_node: its next segment, and that segment's round
  __host__ __device__ int64_t *put_at(int64_t peer_node) const {
    return (int64_t *)word(put_at_ + 2 * peer_node);
  }
  // (num_nodes > 1, local) the done word the forward's relay waits for of a
  // requester, of its even (parity 0) or its odd rounds
  __host__ __device__ uint64_t *relay_want(int64_t requester, int parity) const {
    return word(relay_want_ + 2 * requester + parity);
  }
  // (num_nodes > 1, local) backward puts this proxy posted in calls of more
  // than one round, out of a half of proxy_rows: what its put counters count
  __host__ __device__ uint64_t *posted(int half) const { return word(posted_ + half); }
  // (num_nodes > 1) rows a same-node requester pulled from this proxy's
  // proxy_rows in forwards of more than one round, of its even (parity 0) or
  // its odd rounds: running totals, which tell the relay a round's half is
  // free again
  __host__ __device__ uint64_t *done(int64_t requester, int parity) const {
    return word(done_ + 2 * requester + parity);
  }
  // (num_nodes > 1) grant_of(out_seq, rnd): this proxy's node may push the
  // gradient rows of rounds up to rnd of that backward into its proxy_rows
  // (rounds 0 and 1 need none)
  __host__ __device__ uint64_t *half_free() const { return word(half_free_); }

  // ── teardown ──
  // this rank reads no window any more
  __host__ __device__ uint64_t *closing() const { return word(closing_); }
  // every node peer, and every rail peer (closing signals), said so (written locally)
  __host__ __device__ uint64_t *closed() const { return word(closed_); }

 private:
  __host__ __device__ uint64_t *word(int64_t at) const { return base_ + at; }

  uint64_t *base_;
  int64_t num_nodes_, num_features_;
  CountVector count_;
  CountMessage message_;
  int64_t in_started_, count_vectors_, in_written_, cell_start_, in_done_, count_msg_out_,
      count_msgs_;
  int64_t out_started_, pushed_, pulled_, arrive_, tasks_, landed_rows_, fwd_tot_, bwd_tot_,
      grant_sent_, bwd_ready_sent_, put_at_, relay_want_, posted_, done_, half_free_;
  int64_t closing_, closed_, words_;
};

}  // namespace hier_output_dist
}  // namespace dyn_emb
