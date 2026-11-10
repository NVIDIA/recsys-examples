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
// The bucketize of a call: its keys by owner, as TorchRec's bucketize orders
// them, and index dedup's fan-out (planner.h).

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "hier_output_dist/input_dist/node.cuh"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"
#include "murmur_hash.cuh"
#include "utils.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace {

// The KJT's keys, offsets and lengths are each int32 or int64: the kernels
// that read them take the type as a template argument (Key, Length,
// Offset), and with_index calls `launch` with the tensor's data as a pointer
// of its dtype; an absent tensor as a null int64_t pointer, which the kernel
// does not read.
template <typename Launch>
void with_index(const at::Tensor &tensor, Launch &&launch) {
  if (tensor.scalar_type() == at::kInt)
    launch(tensor.data_ptr<int32_t>());
  else
    launch(tensor.data_ptr<int64_t>());
}
template <typename Launch>
void with_index(const c10::optional<at::Tensor> &tensor, Launch &&launch) {
  if (tensor.has_value())
    with_index(*tensor, launch);
  else
    launch(static_cast<const int64_t *>(nullptr));
}

// ── the KJT's features ──

// One CTA per feature of a KJT of lengths: its keys, the sum of its batch
// lengths.
template <typename Length>
__global__ void __launch_bounds__(kThreads)
    feature_totals_kernel(const Length *lengths, int64_t batch, int64_t *totals) {
  using Reduce = cub::BlockReduce<int64_t, kThreads>;
  __shared__ typename Reduce::TempStorage reduce;
  const int64_t feature = blockIdx.x;
  int64_t mine = 0;
  for (int64_t sample = threadIdx.x; sample < batch; sample += kThreads)
    mine += lengths[feature * batch + sample];
  const int64_t total = Reduce(reduce).Sum(mine);
  if (threadIdx.x == 0) totals[feature] = total;
}

// One CTA: where each feature's keys start in the KJT (value_offsets
// [num_features + 1], from its first key), from its offsets (every
// batch-th) or, for a KJT of lengths (offsets nullptr), from the features'
// totals; and the totals construct_jagged_tensors reads
// (input_length_per_key [num_features]), of the input KJT's offsets.
template <typename Offset, typename InputOffset>
__global__ void __launch_bounds__(kThreads)
    feature_offsets_kernel(const Offset *offsets, const int64_t *totals, int64_t batch,
                           int64_t num_features, const InputOffset *input_offsets,
                           int64_t *value_offsets, int64_t *input_length_per_key) {
  if (offsets != nullptr) {
    for (int64_t feature = threadIdx.x; feature <= num_features; feature += kThreads)
      value_offsets[feature] = (int64_t)offsets[feature * batch] - (int64_t)offsets[0];
    __syncthreads();
  } else {
    cta_offsets_of<int64_t>(
        num_features, [&](int64_t feature) { return totals[feature]; }, value_offsets);
  }
  for (int64_t feature = threadIdx.x; feature < num_features; feature += kThreads)
    input_length_per_key[feature] =
        (int64_t)input_offsets[(feature + 1) * batch] - (int64_t)input_offsets[feature * batch];
}

// ── the owner rules ──

// The owner rules the op takes (DistType, src/utils.h), as DynamicEmb's
// bucketize places a key (src/sparse_block_bucketize_features.cu): in the
// arithmetic of the key's own type (int32_t or int64_t), as unsigned numbers.
// Both keep the key. Config::validate and the bucketize's hook refuse every
// other code.

// kCyclic ("roundrobin"): key % world_size.
template <typename Key>
__device__ __forceinline__ int64_t cyclic_owner(Key key, int64_t world_size) {
  using Unsigned = std::make_unsigned_t<Key>;
  return (int64_t)((Unsigned)key % (Unsigned)world_size);
}

// kHashedCyclic ("hash_roundrobin"): murmur3_fmix64(key) % world_size, the hash
// of an int32 key (zero-extended) taken to its low 32 bits.
template <typename Key>
__device__ __forceinline__ int64_t hashed_cyclic_owner(Key key, int64_t world_size) {
  using Unsigned = std::make_unsigned_t<Key>;
  const Unsigned hash = (Unsigned)murmur3_fmix64((Unsigned)key);
  return (int64_t)(hash % (Unsigned)world_size);
}

// A key's owner by its feature's rule, read at run time (a sharding's tables
// may mix the two). A warp's keys are consecutive, so they are of one feature
// and take one branch, except at the start of a feature.
template <typename Key>
__device__ __forceinline__ int64_t owner_of(Key key, int32_t dist_type, int64_t world_size) {
  return dist_type == (int)DistType::kCyclic ? cyclic_owner(key, world_size)
                                             : hashed_cyclic_owner(key, world_size);
}

// ── the counting sort by owner ──
//
// The bucketize orders the keys as a stable sort by bucket feature (owner *
// num_features + feature, world_size * num_features past the valid keys)
// would. A KJT's offsets never fall, so neither do its keys' features: two
// keys of one owner are in the same order by (feature, index) as by index.
// So that sort is a stable sort by owner, with the keys past the valid ones
// last: world_size + 1 buckets. Two kernels sort them, each a CTA per chunk
// of consecutive keys. owner_count_kernel counts each chunk's keys of each
// bucket. owner_scatter_kernel adds up every chunk's counts, which give where
// each bucket starts and where each chunk's keys of each bucket go; then it
// ranks its chunk's keys in sub-tiles, stages them in shared memory by bucket
// and writes them out, a bucket's keys to consecutive rows. No CTA waits for
// another, and no atomic decides a row. With many owners its shared memory
// grows too large, and the radix sort by bucket feature (below) takes over.

// A sub-tile: kSortItems keys of each thread. Item j of lane l of warp w is
// key w * kWarpKeys + 32 * j + l of the sub-tile, so (warp, item, lane) order
// is the KJT's order.
constexpr int kSortItems = 8;
constexpr int kWarpKeys = 32 * kSortItems;
constexpr int kSubTile = kThreads * kSortItems;
// The most buckets of the counting sort: the owners and the keys past the
// valid ones, rounded up to 4.
constexpr int64_t kMaxCountingBuckets = 2048;
// The bucket of a lane without a key: past every bucket, in 16 bits, so a
// key's bucket and its rank (below kWarpKeys) share a word.
constexpr uint32_t kNoBucket = 0xffffu;
// The buckets of a thread in owner_scatter_kernel's scans over the buckets:
// kMaxCountingBuckets take one tile.
constexpr int kBucketScanItems = 4;
// The threads of owner_scatter_kernel that add up the counts: all but warps
// 0-2, which find the features of its chunk's bounds meanwhile.
constexpr int kReducers = kThreads - 3 * 32;

// Where each feature's keys start, as the counting sort reads it: feature f
// starts at offsets[f * stride] - offsets[0] (feature_offsets_kernel's
// value_offsets), in the KJT's offsets (stride batch_size), or in
// value_offsets (stride 1) for a KJT of lengths.
template <typename Offset>
struct FeatureStarts {
  const Offset *offsets;
  int64_t stride;
  // Feature f starts at or before key `index` when at(f) <= index + origin().
  __device__ __forceinline__ int64_t at(int64_t feature) const {
    return (int64_t)offsets[feature * stride];
  }
  __device__ __forceinline__ int64_t origin() const { return (int64_t)offsets[0]; }
};

// The feature of key `index` (num_features past the valid keys), as
// place_keys_kernel finds it, by the whole warp (warp_upper_bound). Every
// lane, with the same arguments.
template <typename Offset>
__device__ __forceinline__ int32_t warp_feature_of(const FeatureStarts<Offset> &starts,
                                                   int64_t num_features, int64_t index,
                                                   int64_t origin, int lane) {
  return (int32_t)(warp_upper_bound(
                       num_features + 1, [&](int64_t feature) { return starts.at(feature); },
                       index + origin, lane) -
                   1);
}

// The buckets of a lane's keys first + 32 * item + lane, kSortItems of them:
// a key's owner by its feature's rule, or world_size past the valid keys
// (kNoBucket at or past `end`); and their bucket features. The lane loads all
// its keys first. Then it finds their features in [lo_feature, hi_feature],
// the features of its CTA's keys, by halving steps, a step's probes for all
// its keys at once: a key's feature is the last one there that starts at or
// before it.
template <typename Key, typename Offset>
__device__ __forceinline__ void bucket_items(const Key *values, int64_t first, int64_t end,
                                             const FeatureStarts<Offset> &starts, int64_t origin,
                                             int32_t lo_feature, int32_t hi_feature,
                                             int64_t num_features, const int32_t *dist_types,
                                             int64_t world_size, int lane,
                                             uint32_t (&bucket)[kSortItems],
                                             int32_t (&bucket_feature)[kSortItems]) {
  Key key[kSortItems];
  int32_t feature[kSortItems];
#pragma unroll
  for (int item = 0; item < kSortItems; ++item) {
    const int64_t index = first + 32 * item + lane;
    key[item] = index < end ? values[index] : (Key)0;
    feature[item] = lo_feature;
  }
  const int32_t span = hi_feature - lo_feature;
  for (int32_t step = span > 0 ? 1 << (31 - __clz(span)) : 0; step > 0; step >>= 1) {
#pragma unroll
    for (int item = 0; item < kSortItems; ++item) {
      const int32_t probe = feature[item] + step;
      if (probe <= hi_feature && starts.at(probe) <= first + 32 * item + lane + origin)
        feature[item] = probe;
    }
  }
#pragma unroll
  for (int item = 0; item < kSortItems; ++item) {
    if (first + 32 * item + lane >= end) {
      bucket[item] = kNoBucket;
      bucket_feature[item] = 0;
    } else if (feature[item] < num_features) {
      const int64_t owner = owner_of(key[item], dist_types[feature[item]], world_size);
      bucket[item] = (uint32_t)owner;
      bucket_feature[item] = (int32_t)(owner * num_features + feature[item]);
    } else {
      bucket[item] = (uint32_t)world_size;
      bucket_feature[item] = (int32_t)(world_size * num_features);
    }
  }
}

// A key's rank among its warp's keys of its bucket in the sub-tile: those
// before it, of its earlier items too. counts: the warp's keys of each bucket
// so far (at most kWarpKeys), which the bucket's first lane raises by the
// item's keys of the bucket. Every lane of the warp, an item at a time; a
// lane of kNoBucket gets nothing.
__device__ __forceinline__ int32_t warp_rank(uint16_t *counts, uint32_t bucket, int lane) {
  const unsigned peers = __match_any_sync(0xffffffffu, bucket);
  const int leader = __ffs(peers) - 1;
  int32_t before = 0;
  if (lane == leader && bucket != kNoBucket) {
    before = counts[bucket];
    counts[bucket] = (uint16_t)(before + __popc(peers));
  }
  before = __shfl_sync(0xffffffffu, before, leader);
  __syncwarp();  // the next item's first lanes read the counts this item's wrote
  return before + __popc(peers & ((1u << lane) - 1));
}

// A CTA per chunk of `chunk` consecutive keys (the last chunk fewer; the
// CTAs from `ctas` on have none): counts[cta][b], its keys of each bucket b
// of `buckets` (0 past world_size), by warp-aggregated shared adds. Warps 0
// and 1 first find the features of its first and last key. On the offsets
// path (input_length_per_key not nullptr) every CTA also writes the input's
// totals, grid-strided (feature_offsets_kernel's).
template <typename Key, typename Offset, typename InputOffset>
__global__ void __launch_bounds__(kThreads, 2)
    owner_count_kernel(const Key *values, int64_t num_keys, FeatureStarts<Offset> starts,
                       int64_t num_features, const int32_t *dist_types, int64_t world_size,
                       int64_t chunk, int64_t ctas, int64_t buckets, int32_t *counts,
                       const InputOffset *input_offsets, int64_t batch,
                       int64_t *input_length_per_key) {
  extern __shared__ int32_t bucket_keys[];  // [buckets]: the chunk's keys of each bucket
  __shared__ int32_t span[2];               // the features of the chunk's first and last key
  if (input_length_per_key != nullptr)
    for (int64_t feature = first_item(); feature < num_features; feature += item_stride())
      input_length_per_key[feature] =
          (int64_t)input_offsets[(feature + 1) * batch] - (int64_t)input_offsets[feature * batch];
  const int64_t cta = blockIdx.x;
  if (cta >= ctas) return;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int64_t lo = min(cta * chunk, num_keys), hi = min(lo + chunk, num_keys);
  const int64_t origin = starts.origin();
  for (int64_t bucket = threadIdx.x; bucket < buckets; bucket += kThreads) bucket_keys[bucket] = 0;
  if (warp < 2) {
    const int32_t feature =
        lo < hi ? warp_feature_of(starts, num_features, warp == 0 ? lo : hi - 1, origin, lane) : 0;
    if (lane == 0) span[warp] = feature;
  }
  __syncthreads();
  const int32_t lo_feature = span[0], hi_feature = span[1];
  for (int64_t first = lo + warp * kWarpKeys; first < hi; first += kSubTile) {
    uint32_t bucket[kSortItems];
    int32_t bucket_feature[kSortItems];
    bucket_items(values, first, hi, starts, origin, lo_feature, hi_feature, num_features,
                 dist_types, world_size, lane, bucket, bucket_feature);
#pragma unroll
    for (int item = 0; item < kSortItems; ++item) {
      const unsigned peers = __match_any_sync(0xffffffffu, bucket[item]);
      if (bucket[item] != kNoBucket && lane == __ffs(peers) - 1)
        atomicAdd(&bucket_keys[bucket[item]], __popc(peers));
    }
  }
  __syncthreads();
  for (int64_t bucket = threadIdx.x; bucket < buckets; bucket += kThreads)
    counts[cta * buckets + bucket] = bucket_keys[bucket];
}

// The entries [from, from + length) of offset_per_bucket_feature, each
// `value`, for every lane of the warp (length 0: none): a run of one entry by
// its lane, and each longer run by the whole warp, 32 entries at a time (a
// key of a sparse owner can follow its owner's key before it by many
// features).
__device__ __forceinline__ void fill_runs(int64_t *offset_per_bucket_feature, int64_t from,
                                          int64_t length, int64_t value, int lane) {
  if (length == 1) offset_per_bucket_feature[from] = value;
  for (unsigned longer = __ballot_sync(0xffffffffu, length > 1); longer != 0;
       longer &= longer - 1) {
    const int source = __ffs(longer) - 1;
    const int64_t run_from = __shfl_sync(0xffffffffu, from, source),
                  run_length = __shfl_sync(0xffffffffu, length, source),
                  run_value = __shfl_sync(0xffffffffu, value, source);
    for (int64_t entry = lane; entry < run_length; entry += 32)
      offset_per_bucket_feature[run_from + entry] = run_value;
  }
}

// owner_scatter_kernel's dynamic shared memory, in this order: stage_row and
// stage_bucket_feature (int32 [kSubTile] each; first the sums of the rows of
// counts, and in each sub-tile its keys' buckets and ranks, before the
// stage), next_row and shift (int32 [buckets] each), staged (int32 [buckets +
// 1]), last_feature (int32 [owners]), stage_bucket (uint16 [kSubTile]) and
// warp_counts (uint16 [kWarps][buckets]). About 41 KiB at 16 owners, 62 KiB
// at 512.
inline int64_t scatter_smem(int64_t buckets, int64_t owners) {
  return 4 * (2 * kSubTile + 3 * buckets + 1 + owners) + 2 * (kSubTile + kWarps * buckets);
}

// A CTA per chunk of owner_count_kernel's, in three parts.
//
// Where its keys of each bucket go. Warps 0-2 find the features of the
// chunk's first key, its last key and the key past it, while the others add
// up the rows of counts ([ctas][buckets], int4 words of 4 buckets, the rows
// in stripes): all of them, and those of the CTAs before this one. A scan of
// the totals gives where each bucket starts, which CTA 0 writes
// (offset_per_bucket, and offset_per_bucket_feature at the first bucket
// feature of each owner outside [first_owner, first_owner + owners)); plus
// the CTAs before, the row of the chunk's first key of each bucket
// (next_row).
//
// The sub-tiles of kSubTile keys. Each warp ranks its keys in their buckets
// (warp_rank); a key's bucket, rank and bucket feature wait at its own place
// in the stage, not in registers. A thread per bucket turns the warps'
// counts of the bucket into the counts of the warps before, and a scan over
// the buckets gives where each bucket starts in the stage (staged). Each key
// goes to the stage at staged[b] + its warp's keys of b before + its rank,
// so the stage holds the sub-tile's keys by bucket, each bucket's in KJT
// order. Then a thread per staged key writes inv and bucket_feature_of_row at
// its row, next_row[b] + its place among the stage's keys of b: consecutive
// threads write consecutive rows of a bucket.
//
// The offsets of each bucket feature of the owners [first_owner,
// first_owner + owners): entry o * num_features + f is the row of owner o's
// first key of feature f or more, or the row past its last key (run_offsets'
// search). A key of owner o and feature f at row r, whose owner's key before
// it is of feature p, writes r into the entries of the features (p, f]
// (fill_runs). Before the chunk's first key of o, p is the feature of the
// chunk's first key (-1 in CTA 0); after its last key of o, the CTA writes
// next_row[o] up to the feature of the key past the chunk (num_features - 1
// in the last CTA). So CTA c writes the features after that of its first key
// (from 0 in CTA 0) up to that of the next chunk's first key, and each entry
// is written once. When most features start inside one chunk, its CTA writes
// most of the owners * num_features entries.
template <typename Key, typename Offset>
__global__ void __launch_bounds__(kThreads, 2)
    owner_scatter_kernel(const Key *values, int64_t num_keys, FeatureStarts<Offset> starts,
                         int64_t num_features, const int32_t *dist_types, int64_t world_size,
                         int64_t chunk, int64_t ctas, int64_t buckets, const int32_t *counts,
                         int64_t first_owner, int64_t owners, int32_t *inv,
                         int32_t *bucket_feature_of_row, int64_t *offset_per_bucket_feature,
                         int64_t *offset_per_bucket) {
  extern __shared__ __align__(16) unsigned char scatter_shared[];
  int32_t *stage_row = reinterpret_cast<int32_t *>(scatter_shared);  // each staged key's KJT row
  int32_t *stage_bucket_feature = stage_row + kSubTile;              // and its bucket feature
  int32_t *next_row = stage_bucket_feature + kSubTile;  // the row of each bucket's next key
  int32_t *shift = next_row + buckets;                  // a staged key's row less its place
  int32_t *staged = shift + buckets;                    // where each bucket starts in the stage
  int32_t *last_feature = staged + buckets + 1;         // the feature of each owner's key before
  uint16_t *stage_bucket = reinterpret_cast<uint16_t *>(last_feature + owners);
  uint16_t *warp_counts = stage_bucket + kSubTile;  // [kWarps][buckets]
  uint32_t *warp_count_words = reinterpret_cast<uint32_t *>(warp_counts);
  uint32_t *key_slot = reinterpret_cast<uint32_t *>(stage_row);  // by key, before the stage
  // the features of the chunk's first key, its last key and the key past it
  __shared__ int32_t bounds[3];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int64_t cta = blockIdx.x;
  const int64_t lo = min(cta * chunk, num_keys), hi = min(lo + chunk, num_keys);
  const int64_t origin = starts.origin();
  const int64_t columns = buckets / 4, stripes = max(kReducers / columns, (int64_t)1);
  // where its keys of each bucket go: stripe s of column k (int4, 4 buckets) adds up the rows s,
  // s + stripes, ... of counts, all of them and those of the CTAs before this one
  int4 *all_partials = reinterpret_cast<int4 *>(stage_row);
  int4 *before_partials = reinterpret_cast<int4 *>(stage_bucket_feature);
  if (warp < 3) {
    const bool found = warp < 2 ? lo < hi : cta + 1 < ctas;
    const int32_t feature = found ? warp_feature_of(starts, num_features,
                                                    warp == 0   ? lo
                                                    : warp == 1 ? hi - 1
                                                                : hi,
                                                    origin, lane)
                                  : 0;
    if (lane == 0) bounds[warp] = feature;
  } else {
    const int4 *count_words = reinterpret_cast<const int4 *>(counts);
    for (int64_t k = tid - 3 * 32; k < stripes * columns; k += kReducers) {
      const int64_t stripe = k / columns, column = k % columns;
      int4 all = make_int4(0, 0, 0, 0), before = make_int4(0, 0, 0, 0);
#pragma unroll 4
      for (int64_t row = stripe; row < ctas; row += stripes) {
        const int4 count = count_words[row * columns + column];
        all.x += count.x;
        all.y += count.y;
        all.z += count.z;
        all.w += count.w;
        if (row < cta) {
          before.x += count.x;
          before.y += count.y;
          before.z += count.z;
          before.w += count.w;
        }
      }
      all_partials[k] = all;
      before_partials[k] = before;
    }
  }
  __syncthreads();
  // then a thread per bucket and sum adds up the stripes: the totals into shift, those of the CTAs
  // before into next_row
  for (int64_t k = tid; k < 2 * buckets; k += kThreads) {
    const bool all = k < buckets;
    const int64_t bucket = all ? k : k - buckets;
    const int32_t *partials =
        reinterpret_cast<const int32_t *>(all ? all_partials : before_partials);
    int32_t keys = 0;
    for (int64_t stripe = 0; stripe < stripes; ++stripe)
      keys += partials[stripe * buckets + bucket];
    (all ? shift : next_row)[bucket] = keys;
  }
  __syncthreads();
  cta_scan_blocked<kBucketScanItems, int32_t>(
      buckets, [&](int64_t bucket) { return shift[bucket]; },
      [&](int64_t bucket, int32_t offset) {
        next_row[bucket] += offset;
        if (cta == 0 && bucket <= world_size) {
          offset_per_bucket[bucket] = offset;
          if (bucket < first_owner || bucket >= first_owner + owners)
            offset_per_bucket_feature[bucket * num_features] = offset;
        }
      });
  for (int64_t owner = tid; owner < owners; owner += kThreads)
    last_feature[owner] = cta == 0 ? -1 : bounds[0];
  for (int64_t word = tid; word < kWarps * buckets / 2; word += kThreads)
    warp_count_words[word] = 0;
  __syncthreads();

  // the sub-tiles
  for (int64_t tile = lo; tile < hi; tile += kSubTile) {
    // each key's bucket and rank (key_slot: the bucket in the low 16 bits, the rank above) and
    // its bucket feature, at the key's own place in the stage (KJT order) until the buckets'
    // places are known
    {
      uint32_t bucket[kSortItems];
      int32_t bucket_feature[kSortItems];
      bucket_items(values, tile + warp * kWarpKeys, hi, starts, origin, bounds[0], bounds[1],
                   num_features, dist_types, world_size, lane, bucket, bucket_feature);
#pragma unroll
      for (int item = 0; item < kSortItems; ++item) {
        const int key = warp * kWarpKeys + 32 * item + lane;
        key_slot[key] = bucket[item] |
                        (uint32_t)warp_rank(warp_counts + warp * buckets, bucket[item], lane) << 16;
        stage_bucket_feature[key] = bucket_feature[item];
      }
    }
    __syncthreads();
    // per bucket: the keys of the warps before each warp, and of the sub-tile
    for (int64_t b = tid; b < buckets; b += kThreads) {
      int32_t keys = 0;
      for (int w = 0; w < kWarps; ++w) {
        const int32_t warp_keys = warp_counts[w * buckets + b];
        warp_counts[w * buckets + b] = (uint16_t)keys;
        keys += warp_keys;
      }
      staged[b] = keys;
    }
    __syncthreads();
    const int32_t tile_keys = cta_scan_blocked<kBucketScanItems, int32_t>(
        buckets, [&](int64_t b) { return staged[b]; },
        [&](int64_t b, int32_t at) {
          staged[b] = at;
          shift[b] = next_row[b] - at;
        });
    if (tid == 0) staged[buckets] = tile_keys;
    // each key into its place in the stage, once every key's words are read
    uint32_t slot[kSortItems];
    int32_t bucket_feature[kSortItems];
#pragma unroll
    for (int item = 0; item < kSortItems; ++item) {
      const int key = warp * kWarpKeys + 32 * item + lane;
      slot[item] = key_slot[key];
      bucket_feature[item] = stage_bucket_feature[key];
    }
    __syncthreads();
#pragma unroll
    for (int item = 0; item < kSortItems; ++item) {
      const uint32_t b = slot[item] & 0xffffu;
      if (b == kNoBucket) continue;
      const int32_t at = staged[b] + warp_counts[warp * buckets + b] + (int32_t)(slot[item] >> 16);
      stage_row[at] = (int32_t)(tile + warp * kWarpKeys + 32 * item + lane);
      stage_bucket_feature[at] = bucket_feature[item];
      stage_bucket[at] = (uint16_t)b;
    }
    __syncthreads();
    // a thread per staged key (whole warps: fill_runs)
    for (int32_t from = 0; from < tile_keys; from += kThreads) {
      const int32_t at = from + tid;
      int64_t fill_from = 0, fill_length = 0, row = 0;
      if (at < tile_keys) {
        const int32_t b = stage_bucket[at], bf = stage_bucket_feature[at];
        row = at + shift[b];
        inv[row] = stage_row[at];
        bucket_feature_of_row[row] = bf;
        if (b >= first_owner && b < first_owner + owners) {
          const int64_t owner_first = (int64_t)b * num_features;
          const int64_t previous = at > staged[b] ? stage_bucket_feature[at - 1] - owner_first
                                                  : last_feature[b - first_owner];
          fill_from = owner_first + previous + 1;
          fill_length = bf - owner_first - previous;
        }
      }
      fill_runs(offset_per_bucket_feature, fill_from, fill_length, row, lane);
    }
    __syncthreads();
    for (int64_t owner = tid; owner < owners; owner += kThreads) {
      const int64_t b = first_owner + owner;
      if (staged[b + 1] > staged[b])
        last_feature[owner] = stage_bucket_feature[staged[b + 1] - 1] - (int32_t)(b * num_features);
    }
    for (int64_t b = tid; b < buckets; b += kThreads) next_row[b] += staged[b + 1] - staged[b];
    for (int64_t word = tid; word < kWarps * buckets / 2; word += kThreads)
      warp_count_words[word] = 0;
    __syncthreads();
  }
  // each owner's features past its last key of the chunk, up to the next chunk's first
  const int64_t fill_end =
      cta + 1 < ctas ? min((int64_t)bounds[2], num_features - 1) : num_features - 1;
  for (int64_t owner = 0; owner < owners; ++owner) {
    const int64_t b = first_owner + owner, row = next_row[b];
    for (int64_t feature = last_feature[owner] + 1 + tid; feature <= fill_end; feature += kThreads)
      offset_per_bucket_feature[b * num_features + feature] = row;
  }
}

// The counting sort's grid: `ctas` CTAs of `chunk` consecutive keys each, a
// multiple of 32 (the last CTA's fewer). A CTA per sub-tile of keys, so at
// most a sub-tile each, up to two CTAs per SM, or one above 128 buckets
// (each CTA of owner_scatter_kernel reads every CTA's counts); past that,
// more keys per CTA. `buckets`: the owners and the keys past the valid ones,
// rounded up to 4 (a row of counts is whole int4 words). All from host
// values: the keys, world_size and the SMs.
struct CountingGrid {
  int64_t ctas, chunk, buckets;
};
CountingGrid counting_grid(int64_t num_keys, int64_t world_size) {
  const int64_t buckets = (world_size + 1 + 3) / 4 * 4;
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t ctas =
      std::min(std::max(ceil_div<kSubTile>(num_keys), (int64_t)1), (buckets <= 128 ? 2 : 1) * sms);
  const int64_t chunk = (std::max(num_keys, (int64_t)1) + ctas - 1) / ctas;
  const int64_t whole_chunk = ceil_div<32>(chunk) * 32;
  return {num_keys > 0 ? (num_keys + whole_chunk - 1) / whole_chunk : 1, whole_chunk, buckets};
}

// A kernel's shared memory on one GPU: its static bytes (-1 until read), and
// whether its attribute lets it take the GPU's most per CTA.
struct KernelSmem {
  std::atomic<int64_t> static_bytes{-1};
  std::atomic<bool> opted_in{false};
};
// The most shared memory of a CTA (static and dynamic) without the attribute.
constexpr int64_t kDefaultSmem = 48 << 10;

// Whether `kernel` can take `bytes` of dynamic shared memory on the current
// GPU (per_gpu: its KernelSmem of each GPU). Above the default, the first
// such launch on a GPU sets the kernel's attribute
// (cudaFuncAttributeMaxDynamicSharedMemorySize) to the GPU's most per CTA,
// less the kernel's static shared memory. Host queries only: nothing waits
// for the device.
template <typename Kernel>
bool takes_smem(Kernel kernel, KernelSmem *per_gpu, int64_t bytes) {
  const int gpu = c10::cuda::current_device();
  TORCH_CHECK(gpu < C10_COMPILE_TIME_MAX_GPUS, "hier_output_dist: GPU ", gpu,
              " is past the GPUs PyTorch was built for (", C10_COMPILE_TIME_MAX_GPUS, ")");
  KernelSmem &smem = per_gpu[gpu];
  int64_t static_bytes = smem.static_bytes.load();
  if (static_bytes < 0) {
    cudaFuncAttributes attributes;
    HIER_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    static_bytes = (int64_t)attributes.sharedSizeBytes;
    smem.static_bytes.store(static_bytes);
  }
  if (static_bytes + bytes <= kDefaultSmem) return true;
  const int64_t most = (int64_t)at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
  if (static_bytes + bytes > most) return false;
  if (!smem.opted_in.load()) {
    HIER_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)(most - static_bytes)));
    smem.opted_in.store(true);
  }
  return true;
}
// owner_scatter_kernel<Key, Offset>'s KernelSmem of each GPU.
template <typename Key, typename Offset>
KernelSmem *scatter_smem_per_gpu() {
  static KernelSmem per_gpu[C10_COMPILE_TIME_MAX_GPUS];
  return per_gpu;
}

// Calls `launch` with the keys and the table of feature starts the counting
// sort reads, as pointers of their types: the KJT's offsets, or for a KJT of
// lengths value_offsets (int64; undefined: a null pointer, whose type alone
// counts).
template <typename Launch>
void with_feature_starts(const PlanInput &input, const at::Tensor &value_offsets, Launch &&launch) {
  with_index(input.values, [&](const auto *keys) {
    if (input.offsets.has_value())
      with_index(*input.offsets, [&](const auto *offsets) {
        using Offset = std::decay_t<decltype(*offsets)>;
        launch(keys, FeatureStarts<Offset>{offsets, input.batch_size});
      });
    else
      launch(keys, FeatureStarts<int64_t>{data_or_null<int64_t>(value_offsets), 1});
  });
}

// ── the radix sort by bucket feature (many owners, and the hooks' reference) ──

// A thread per key of the KJT: its bucket feature, owner * num_features +
// feature with its owner by its feature's rule, or world_size * num_features
// past the valid keys (index dedup's bound); `index` the key's own index,
// which the sort carries. The rules keep the key, so the keys stay in the
// KJT: the push gathers them from there.
template <typename Key>
__global__ void __launch_bounds__(kThreads)
    place_keys_kernel(const Key *values, int64_t num_keys, const int64_t *value_offsets,
                      int64_t num_features, const int32_t *dist_types, int64_t world_size,
                      int32_t *bucket_feature, int32_t *index) {
  for (int64_t key = first_item(); key < num_keys; key += item_stride()) {
    const int64_t feature = upper_bound(value_offsets, num_features + 1, key) - 1;
    bucket_feature[key] =
        feature < num_features
            ? (int32_t)(owner_of(values[key], dist_types[feature], world_size) * num_features +
                        feature)
            : (int32_t)(world_size * num_features);
    index[key] = (int32_t)key;
  }
}

// A thread per position of the keys sorted by bucket feature (gather_rows,
// node.cuh): the key there, widened to int64 with its sign, and its weight
// (key_weights nullptr: none).
__global__ void __launch_bounds__(kThreads)
    gather_keys_kernel(GatherRows rows, int64_t num_keys, int64_t *keys, float *key_weights) {
  gather_rows(rows, num_keys, [&](int64_t row, int64_t key, float weight) {
    keys[row] = key;
    if (key_weights != nullptr) key_weights[row] = weight;
  });
}

// ── index dedup's fan-out ──

// A thread per original row and per bucketized row (`rows` of each): an
// original row's deduplicated row as an int32 sort key, and its own index,
// which the sort carries; and where the bucketize put each deduplicated row
// (position_of, the inverse of inv).
__global__ void __launch_bounds__(kThreads)
    original_rows_kernel(const int64_t *reverse_indices, int64_t originals, const int32_t *inv,
                         int64_t rows, int32_t *row, int32_t *index, int32_t *position_of) {
  for (int64_t item = first_item(); item < max(originals, rows); item += item_stride()) {
    if (item < originals) {
      row[item] = (int32_t)reverse_indices[item];
      index[item] = (int32_t)item;
    }
    if (item < rows) position_of[inv[item]] = (int32_t)item;
  }
}

// The end of the run of `row` in sorted[0, n) (ascending) that holds
// position `at`: the first position past it. The whole warp searches: each
// round its lanes probe the 32 positions `step` apart after lo, the last
// position known in the run. While all of them are in the run the steps
// grow 32 times (galloping); then each round narrows the end 32 times. A run
// whose last position is at most 31 past `at` takes one round, a run of a
// million positions seven. Every lane of the warp, with the same arguments;
// every lane gets the end. The probes stay below hi (at most n), whatever the
// table holds.
__device__ __forceinline__ int64_t warp_run_end(const int32_t *sorted, int64_t n, int64_t at,
                                                int32_t row, int lane) {
  int64_t lo = at, hi = n, step = 1;  // the run ends in (lo, hi]
  while (hi - lo > 1) {
    const int64_t probe = lo + step * (lane + 1);
    const unsigned ballot = __ballot_sync(0xffffffffu, probe < hi && sorted[probe] == row);
    const int inside = __clz(__brev(~ballot));  // the lanes from lane 0 on whose probe is in it
    if (inside == 32) {
      lo += 32 * step;
      step *= 32;
    } else {
      hi = min(hi, lo + (inside + 1) * step);
      lo += inside * step;
      step = (hi - lo + 31) / 32;
    }
  }
  return hi;
}

// The fan-out's tables from the sorted rows (sorted: each original row's
// deduplicated row, ascending, n of them), in one kernel: a thread per sorted
// position, a warp's lanes at consecutive positions. Position q starts the
// run of row r = sorted[q] when q is 0 or sorted[q - 1] differs. Its thread
// writes exp_off[r] = q, and for the rows between the previous run's row and
// r, which have no original rows, exp_off q and heavy_slot -1. The rows
// before the first run get exp_off 0 and those after the last exp_off n
// (exp_off[rows] too), heavy_slot -1, in loops over the whole grid: index
// dedup leaves many rows past the valid ones. With no original rows, every
// row is such a row.
//
// The run is heavy when it has more than kHeavy positions: sorted[q + kHeavy]
// is r too (r not heavy: heavy_slot[r] = -1). Its warp takes it: the warp
// finds its end, and the run-start thread reserves its heavy slot and its
// ceil(length / kHeavy) sub-tasks with one 64-bit atomicAdd of
// (1 << 32) | subs on the counter word: the old word's high half is its heavy
// slot, its low half its first sub-task. So consecutive heavy slots have
// consecutive sub-task ranges, and first[s + 1] - first[s] counts slot s's
// sub-tasks (the last slot's end at count[1]). The thread writes
// heavy_slot[r], pos (where the bucketize put r) and first; then the warp
// writes the run's sub-tasks (r, q + t * kHeavy), 32 at a time, so a row of
// a million original rows (15,625 sub-tasks) takes 489 rounds of one warp,
// not 15,625 stores of one thread. Heavy slots past max_heavy_slots and
// sub-tasks past max_subs are not written (reverse indices in [0, rows) never
// reach them).
//
// The heavy slots are so numbered in the order their runs reserve them,
// which can differ between two calls on the same rows. No sum depends on the
// numbering: a row's sub-tasks stay consecutive and in original-row order, so
// the backward adds up each heavy row's original rows in the same order
// whatever its heavy slot, and the gradient keeps its bits.
//
// count ([heavy rows, sub-tasks]) is the counter once every run reserved:
// each CTA, once its threads are done, takes a ticket (words[1]), and the
// CTA that takes the last ticket writes count from the counter, then resets
// both words to 0 for the next call. No CTA waits: a CTA takes its ticket and
// exits, and the last one goes on only because every other CTA already
// arrived. It is the last one out, not a spin on other CTAs, so it needs no
// CTA to run at the same time as another, and it keeps the planner's rule
// that no kernel waits for another CTA (planner.cuh). words: [counter,
// ticket], 0 when the kernel starts (kFanoutWords).
__global__ void __launch_bounds__(kThreads)
    fanout_runs_kernel(const int32_t *sorted, int64_t n, int64_t rows, const int32_t *position_of,
                       int64_t max_heavy_slots, int64_t max_subs, int32_t *exp_off,
                       int32_t *heavy_slot, int32_t *pos, int32_t *first, int64_t *sub,
                       int64_t *count, unsigned long long *words) {
  const int lane = threadIdx.x & 31;
  // the rows before the first run and after the last
  const int64_t head = n > 0 ? min(max((int64_t)sorted[0], (int64_t)0), rows + 1) : rows + 1;
  for (int64_t row = first_item(); row < head; row += item_stride()) {
    exp_off[row] = 0;
    if (row < rows) heavy_slot[row] = -1;
  }
  const int64_t tail =
      n > 0 ? min(max((int64_t)sorted[n - 1] + 1, (int64_t)0), rows + 1) : rows + 1;
  for (int64_t row = tail + first_item(); row <= rows; row += item_stride()) {
    exp_off[row] = (int32_t)n;
    if (row < rows) heavy_slot[row] = -1;
  }
  // a warp's positions at a time, so every lane takes part in its heavy runs
  for (int64_t base = first_item() - lane; base < n; base += item_stride()) {
    const int64_t q = base + lane;
    int32_t row = 0;
    bool heavy = false;  // q starts a heavy run
    if (q < n) {
      row = sorted[q];
      if (q == 0 || sorted[q - 1] != row) {
        // r, and the rows between the previous run's row and r (q 0: the head's loop took them)
        const int64_t from = q == 0 ? row : (int64_t)sorted[q - 1] + 1;
        for (int64_t gap = max(from, (int64_t)0); gap <= min((int64_t)row, rows); ++gap) {
          exp_off[gap] = (int32_t)q;
          if (gap < row && gap < rows) heavy_slot[gap] = -1;
        }
        if (row >= 0 && row < rows) {
          heavy = q + kHeavy < n && sorted[q + kHeavy] == row;
          if (!heavy) heavy_slot[row] = -1;
        }
      }
    }
    for (unsigned runs = __ballot_sync(0xffffffffu, heavy); runs != 0; runs &= runs - 1) {
      const int starter = __ffs(runs) - 1;
      const int64_t start = base + starter;
      const int32_t run_row = __shfl_sync(0xffffffffu, row, starter);
      const int64_t subs =
          ceil_div<kHeavy>(warp_run_end(sorted, n, start + kHeavy, run_row, lane) - start);
      unsigned long long reserved = 0;
      if (lane == starter) {
        reserved = atomicAdd(&words[0], (1ull << 32) | (unsigned long long)subs);
        const int64_t run_slot = (int64_t)(reserved >> 32);
        heavy_slot[run_row] = (int32_t)run_slot;
        if (run_slot < max_heavy_slots) {
          pos[run_slot] = position_of[run_row];
          first[run_slot] = (int32_t)(reserved & 0xffffffffull);
        }
      }
      const int64_t first_sub =
          (int64_t)(__shfl_sync(0xffffffffu, reserved, starter) & 0xffffffffull);
      for (int64_t task = lane; task < subs && first_sub + task < max_subs; task += 32) {
        sub[2 * (first_sub + task)] = run_row;
        sub[2 * (first_sub + task) + 1] = start + task * kHeavy;
      }
    }
  }
  // the last CTA out: count, and the words back to 0
  __syncthreads();  // every reservation of this CTA is made
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(&words[1], 1ull) == gridDim.x - 1) {
      __threadfence();
      const unsigned long long total = atomicExch(&words[0], 0ull);
      count[0] = (int64_t)(total >> 32);
      count[1] = (int64_t)(total & 0xffffffffull);
      atomicExch(&words[1], 0ull);
    }
  }
}

// contiguous, on the GPU of the keys
bool on_gpu_of(const at::Tensor &tensor, const at::Tensor &values) {
  return tensor.is_cuda() && tensor.get_device() == values.get_device() && tensor.is_contiguous();
}
bool is_index_dtype(const at::Tensor &tensor) {
  return tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong;
}

// value_offsets, where each feature's keys start in the KJT (num_features +
// 1, from its first key), and the input's totals into input_length_per_key:
// feature_offsets_kernel, after feature_totals_kernel for a KJT of lengths.
at::Tensor feature_offsets_of(const PlanInput &input, int64_t num_features,
                              const at::Tensor &input_length_per_key) {
  const int64_t batch = input.batch_size;
  const auto long_options = input.values.options().dtype(at::kLong);
  const cudaStream_t stream = current_stream();
  const at::Tensor value_offsets = at::empty({num_features + 1}, long_options);
  at::Tensor totals;
  if (input.lengths.has_value()) {
    totals = at::empty({num_features}, long_options);
    with_index(*input.lengths, [&](const auto *lengths) {
      feature_totals_kernel<<<num_features, kThreads, 0, stream>>>(lengths, batch,
                                                                   totals.data_ptr<int64_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  with_index(input.offsets, [&](const auto *offsets) {
    with_index(input.input_offsets, [&](const auto *input_offsets) {
      feature_offsets_kernel<<<kOneCta, kThreads, 0, stream>>>(
          offsets, data_or_null<int64_t>(totals), batch, num_features, input_offsets,
          value_offsets.data_ptr<int64_t>(), input_length_per_key.data_ptr<int64_t>());
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return value_offsets;
}

// The radix sort by bucket feature: the bucket feature of each key, then a
// stable sort by bucket feature (a bucket feature's keys keep the KJT's
// order, and the keys past the valid ones stay last, in order), then a
// search of the sorted bucket features for where each bucket feature starts.
// It writes all of offset_per_bucket_feature.
BucketizedKeys bucketize_rows_sorted(const PlanInput &input, const at::Tensor &dist_types,
                                     int64_t world_size) {
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel(), num_features = dist_types.numel();
  const int64_t bucket_features = world_size * num_features;
  const auto long_options = values.options().dtype(at::kLong);
  const auto int_options = values.options().dtype(at::kInt);
  BucketizedKeys bucketized;
  bucketized.input_length_per_key = at::empty({num_features}, long_options);
  const at::Tensor value_offsets =
      feature_offsets_of(input, num_features, bucketized.input_length_per_key);
  const at::Tensor bucket_feature = at::empty({num_keys}, int_options);
  const at::Tensor index = at::empty({num_keys}, int_options);
  if (num_keys > 0) {
    with_index(values, [&](const auto *keys) {
      place_keys_kernel<<<item_grid(num_keys), kThreads, 0, current_stream()>>>(
          keys, num_keys, value_offsets.data_ptr<int64_t>(), num_features,
          dist_types.data_ptr<int32_t>(), world_size, bucket_feature.data_ptr<int32_t>(),
          index.data_ptr<int32_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  std::tie(bucketized.bucket_feature_of_row, bucketized.inv) =
      sort_pairs(bucket_feature, index, bits_for(bucket_features + 1));
  bucketized.offset_per_bucket_feature = at::empty({bucket_features + 1}, long_options);
  bucketized.offset_per_bucket = at::empty({world_size + 1}, long_options);
  run_offsets<int32_t, int64_t>(bucketized.bucket_feature_of_row, bucket_features,
                                bucketized.offset_per_bucket_feature, num_features,
                                bucketized.offset_per_bucket);
  bucketized.values = values;
  return bucketized;
}

// The counting sort on its grid: owner_count_kernel, then
// owner_scatter_kernel, which owner_scatter_fits found can take its shared
// memory.
BucketizedKeys bucketize_rows_counted(const PlanInput &input, const at::Tensor &dist_types,
                                      int64_t world_size, int64_t first_owner, int64_t owners,
                                      const CountingGrid &grid) {
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel(), num_features = dist_types.numel();
  const auto long_options = values.options().dtype(at::kLong);
  const auto int_options = values.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  BucketizedKeys bucketized;
  bucketized.input_length_per_key = at::empty({num_features}, long_options);
  // A KJT of lengths: where its features start and the input's totals first, as the radix sort
  // makes them. On the offsets path the kernels read the starts from the KJT's offsets, and
  // owner_count_kernel writes the totals.
  at::Tensor value_offsets;
  if (!input.offsets.has_value())
    value_offsets = feature_offsets_of(input, num_features, bucketized.input_length_per_key);
  // each CTA's keys of each bucket (the CTAs from grid.ctas on: the input's totals only)
  const at::Tensor counts = at::empty({grid.ctas, grid.buckets}, int_options);
  const int64_t count_ctas =
      value_offsets.defined() ? grid.ctas : std::max(grid.ctas, (int64_t)item_grid(num_features));
  bucketized.inv = at::empty({num_keys}, int_options);
  bucketized.bucket_feature_of_row = at::empty({num_keys}, int_options);
  bucketized.offset_per_bucket_feature = at::empty({world_size * num_features + 1}, long_options);
  bucketized.offset_per_bucket = at::empty({world_size + 1}, long_options);
  with_feature_starts(input, value_offsets, [&](const auto *keys, const auto starts) {
    using Key = std::decay_t<decltype(*keys)>;
    using Offset = std::decay_t<decltype(*starts.offsets)>;
    with_index(input.input_offsets, [&](const auto *input_offsets) {
      owner_count_kernel<<<count_ctas, kThreads, 4 * grid.buckets, stream>>>(
          keys, num_keys, starts, num_features, dist_types.data_ptr<int32_t>(), world_size,
          grid.chunk, grid.ctas, grid.buckets, counts.data_ptr<int32_t>(), input_offsets,
          input.batch_size,
          value_offsets.defined() ? nullptr : bucketized.input_length_per_key.data_ptr<int64_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    owner_scatter_kernel<Key, Offset>
        <<<grid.ctas, kThreads, scatter_smem(grid.buckets, owners), stream>>>(
            keys, num_keys, starts, num_features, dist_types.data_ptr<int32_t>(), world_size,
            grid.chunk, grid.ctas, grid.buckets, counts.data_ptr<int32_t>(), first_owner, owners,
            bucketized.inv.data_ptr<int32_t>(),
            bucketized.bucket_feature_of_row.data_ptr<int32_t>(),
            bucketized.offset_per_bucket_feature.data_ptr<int64_t>(),
            bucketized.offset_per_bucket.data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });
  bucketized.values = values;
  bucketized.counted = true;
  return bucketized;
}

// Whether owner_scatter_kernel, of the keys' type and that of the feature
// starts, can take its shared memory for `owners` owners on this GPU.
bool owner_scatter_fits(const PlanInput &input, const CountingGrid &grid, int64_t owners) {
  bool fits = false;
  with_feature_starts(input, at::Tensor(), [&](const auto *keys, const auto starts) {
    using Key = std::decay_t<decltype(*keys)>;
    using Offset = std::decay_t<decltype(*starts.offsets)>;
    fits = takes_smem(owner_scatter_kernel<Key, Offset>, scatter_smem_per_gpu<Key, Offset>(),
                      scatter_smem(grid.buckets, owners));
  });
  return fits;
}

}  // namespace

void check_bucketize_input(const PlanInput &input, const at::Tensor &dist_types,
                           int64_t world_size) {
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel(), batch = input.batch_size,
                num_features = dist_types.numel();
  TORCH_CHECK(num_features >= 1 && world_size >= 1,
              "hier_output_dist: a rule per feature, and some owners");
  TORCH_CHECK(on_gpu_of(dist_types, values) && dist_types.scalar_type() == at::kInt,
              "hier_output_dist: the rules' dist types (int32) must be contiguous on the GPU of "
              "the keys");
  TORCH_CHECK(
      values.is_cuda() && values.is_contiguous() && is_index_dtype(values) && num_keys < INT32_MAX,
      "hier_output_dist: keys must be contiguous int32 or int64 on the GPU, fewer than 2^31");
  TORCH_CHECK(batch > 0, "hier_output_dist: a KJT of at least one sample");
  TORCH_CHECK(input.offsets.has_value() != input.lengths.has_value(),
              "hier_output_dist: the KJT's offsets or its lengths");
  const at::Tensor &bounds = input.offsets.has_value() ? *input.offsets : *input.lengths;
  TORCH_CHECK(on_gpu_of(bounds, values) && is_index_dtype(bounds) &&
                  bounds.numel() == num_features * batch + input.offsets.has_value(),
              "hier_output_dist: offsets (num_features * batch_size + 1) or lengths "
              "(num_features * batch_size) must be contiguous int32 or int64 on the GPU of the "
              "keys");
  TORCH_CHECK(on_gpu_of(input.input_offsets, values) && is_index_dtype(input.input_offsets) &&
                  input.input_offsets.numel() == num_features * batch + 1,
              "hier_output_dist: the input KJT's offsets must be contiguous int32 or int64 on "
              "the GPU of the keys, num_features * batch_size + 1 of them");
  TORCH_CHECK(!input.weights.has_value() || (on_gpu_of(*input.weights, values) &&
                                             input.weights->scalar_type() == at::kFloat &&
                                             input.weights->numel() == num_keys),
              "hier_output_dist: weights must be contiguous fp32, one per key");
}

void check_fanout_input(const at::Tensor &reverse_indices) {
  // Original rows are int32 (exp_idx).
  TORCH_CHECK(reverse_indices.is_cuda() && reverse_indices.is_contiguous() &&
                  reverse_indices.scalar_type() == at::kLong && reverse_indices.numel() < INT32_MAX,
              "hier_output_dist: reverse indices must be contiguous int64 on the GPU, fewer "
              "than 2^31");
}

BucketizedKeys bucketize_rows(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size, int64_t first_owner, int64_t owners,
                              c10::optional<bool> sorted_path) {
  TORCH_CHECK(first_owner >= 0 && owners >= 1 && first_owner + owners <= world_size,
              "hier_output_dist: the owners of the bucketize's offsets per bucket feature must be "
              "some of its world_size owners");
  // The keys stay in the KJT either way: the push gathers them (bucketize_keys here).
  if (sorted_path.value_or(false)) return bucketize_rows_sorted(input, dist_types, world_size);
  const CountingGrid grid = counting_grid(input.values.numel(), world_size);
  const bool few_owners = grid.buckets <= kMaxCountingBuckets;
  if (few_owners && owner_scatter_fits(input, grid, owners))
    return bucketize_rows_counted(input, dist_types, world_size, first_owner, owners, grid);
  // The counting sort cannot run here: sorted_path false raises with the reason.
  TORCH_CHECK(!sorted_path.has_value() || few_owners,
              "hier_output_dist: the bucketize's counting sort takes at most ",
              kMaxCountingBuckets - 1, " owners, and this one has ", world_size);
  TORCH_CHECK(!sorted_path.has_value(), "hier_output_dist: the bucketize's counting sort needs ",
              scatter_smem(grid.buckets, owners), " bytes of dynamic shared memory per CTA for ",
              world_size, " owners (offsets per bucket feature of ", owners,
              "), and with its static ones that is more than a CTA of this GPU takes (",
              at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin, " bytes)");
  return bucketize_rows_sorted(input, dist_types, world_size);
}

BucketizedKeys bucketize_keys(const PlanInput &input, const at::Tensor &dist_types,
                              int64_t world_size, int64_t first_owner, int64_t owners,
                              c10::optional<bool> sorted_path) {
  BucketizedKeys bucketized =
      bucketize_rows(input, dist_types, world_size, first_owner, owners, sorted_path);
  const at::Tensor &values = input.values;
  const int64_t num_keys = values.numel();
  // the keys (and weights) gathered into bucketized order
  bucketized.keys = at::empty({num_keys}, values.options().dtype(at::kLong));
  if (input.weights.has_value())
    bucketized.key_weights = at::empty({num_keys}, values.options().dtype(at::kFloat));
  if (num_keys > 0) {
    const GatherRows rows =
        gather_rows_of(bucketized.inv, bucketized.values, input.weights.value_or(at::Tensor()));
    gather_keys_kernel<<<item_grid(num_keys), kThreads, 0, current_stream()>>>(
        rows, num_keys, bucketized.keys.data_ptr<int64_t>(),
        data_or_null<float>(bucketized.key_weights));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return bucketized;
}

FanoutTables fanout_tables(const at::Tensor &reverse_indices, const at::Tensor &inv,
                           const at::Tensor &words) {
  const int64_t rows = inv.numel(), originals = reverse_indices.numel();
  // Original and deduplicated rows are int32 (exp_idx, exp_off): check_fanout_input.
  TORCH_CHECK(inv.scalar_type() == at::kInt, "hier_output_dist: inv must be int32");
  TORCH_CHECK(!words.defined() || (on_gpu_of(words, inv) && words.scalar_type() == at::kLong &&
                                   words.numel() == kFanoutWords),
              "hier_output_dist: the fan-out's words must be ", kFanoutWords,
              " contiguous int64 on the GPU of inv");
  const int64_t max_heavy_slots = originals / (kHeavy + 1) + 1,
                max_subs = 2 * originals / kHeavy + 2;
  const auto long_options = inv.options().dtype(at::kLong);
  const auto int_options = inv.options().dtype(at::kInt);
  const cudaStream_t stream = current_stream();
  FanoutTables fanout;
  // where the bucketize put each row, and each row's original rows: a stable
  // sort of the reverse indices, which keeps them in increasing order (the
  // sort's inputs live until it is done)
  const at::Tensor position_of = at::empty({rows}, int_options);
  at::Tensor sorted_dedup_row;
  {
    const at::Tensor dedup_row = at::empty({originals}, int_options);
    const at::Tensor index = at::empty({originals}, int_options);
    const int64_t items = std::max(originals, rows);
    if (items > 0) {
      original_rows_kernel<<<item_grid(items), kThreads, 0, stream>>>(
          reverse_indices.data_ptr<int64_t>(), originals, inv.data_ptr<int32_t>(), rows,
          dedup_row.data_ptr<int32_t>(), index.data_ptr<int32_t>(),
          position_of.data_ptr<int32_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    std::tie(sorted_dedup_row, fanout.exp_idx) = sort_pairs(dedup_row, index, bits_for(rows));
  }
  // the runs of the sorted rows, in one kernel: exp_off, and the heavy rows'
  // slots and sub-tasks, numbered in the order the kernel reserves them
  // (fanout_runs_kernel); the hook's call gets words of its own
  fanout.exp_off = at::empty({rows + 1}, int_options);
  fanout.heavy_slot = at::empty({rows}, int_options);
  fanout.sub = at::empty({2 * max_subs}, long_options);
  fanout.count = at::empty({2}, long_options);
  fanout.pos = at::empty({max_heavy_slots}, int_options);
  fanout.first = at::empty({max_heavy_slots}, int_options);
  const at::Tensor counter = words.defined() ? words : at::zeros({kFanoutWords}, long_options);
  fanout_runs_kernel<<<item_grid(std::max(originals, rows + 1)), kThreads, 0, stream>>>(
      sorted_dedup_row.data_ptr<int32_t>(), originals, rows, position_of.data_ptr<int32_t>(),
      max_heavy_slots, max_subs, fanout.exp_off.data_ptr<int32_t>(),
      fanout.heavy_slot.data_ptr<int32_t>(), fanout.pos.data_ptr<int32_t>(),
      fanout.first.data_ptr<int32_t>(), fanout.sub.data_ptr<int64_t>(),
      fanout.count.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long *>(counter.data_ptr<int64_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return fanout;
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
