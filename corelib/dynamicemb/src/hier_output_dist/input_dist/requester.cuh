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
// The requester kernel of a call across nodes (planner.h: RequesterTables, and the owner's
// records, OwnerRecords), which the in channel launches on its node peers' copies of the window
// (WindowPeers, comm_kernels.cu) and the hooks on stand-ins for them (StandInPeers, planner.cu).
// One grid, its CTAs in four roles, in this order:
//   a table CTA per proxy p of the node  p's chunks C_p (its header's, clamped into [0,
//                                        max_chunks]), its chunk ends [0, C_p) and its stream's
//                                        end, and this requester's push table there [0, C_p], made
//                                        absolute; past C_p the words the proxy holds there;
//   the scan CTA                         the remote task offsets of the L * maxC (chunk, proxy)
//                                        entries the chunks take (maxC: the most of a proxy);
//   an owner CTA per other node          where its records start in rail_keys, and those of each
//                                        feature (the rail layout's; none in the hooks);
//   the row CTAs                         each row's src and dst, and its place in the push order.
// Every CTA first keeps in shared memory where its rows at each proxy start in its push order, and
// where each proxy's buffers are. A table CTA writes its proxy's entries past C_p from the entry
// at C_p: ends 0, and push table entries of the entry at C_p, the words the proxy holds there (its
// ends are 0 past its chunks, and its push table entries the same from its chunks on). So no CTA
// reads a proxy's header past C_p, and only the table and scan CTAs read the headers.
// owner_tables writes the remote task offsets past L * maxC.
//
// It reads its node peers' headers and remote_map after node_status_exchange of the call, earlier
// on the stream, acquired every peer's in_status, which the peer published after its proxy steps
// wrote them. A peer writes them again only in its next call's proxy steps, after that call's
// node barrier, which waits for this rank's in_started of that call: this rank releases it in its
// next node_exchange, which its stream starts only once this kernel has ended (comm_kernels.cu).
// Their loads go through L2 only (ld.global.cg).
//
// It does not include device.cuh, so planner.cu does not get nccl_device.h. The build has no
// relocatable device code: everything here is inline or a template.
#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <utility>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/input_dist/planner.cuh"
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

// What the requester kernel reads and writes but its node's proxies' buffers, which a Peers type
// gives: header(p) and remote_map(p), node peer p's proxy header and remote_map as this GPU reads
// them.
struct RequesterArgs {
  const int32_t *bucket_feature_of_row;  // [rows], the bucketize's
  const int64_t *offset_per_bucket;      // [world_size + 1], the bucketize's
  const int64_t *row_shift;              // [world_size * num_features + 1], the node layout's
  const int64_t *owner_group_shift;      // [local_size][num_groups]: node_status_exchange's
  const int32_t *group_of_feature;       // [num_features]
  int64_t rows;
  // the owner CTAs' inputs: the rail layout's slice_in and node_length_per_source_feature, and the
  // other nodes (unused without owner CTAs)
  const int64_t *slice_in, *node_length_per_source_feature, *other_nodes;
  // what it writes: RequesterTables, and OwnerRecords (with owner CTAs)
  int32_t *src, *dst, *push_order;
  int64_t *push_order_off, *proxy_num_chunks, *proxy_ends;
  int32_t *proxy_push_tables;
  int64_t *remote_task_offsets, *records_first, *record_offsets;
};

// The CTAs of each role, in the grid's order: a table CTA per proxy, the scan CTA, the owner CTAs,
// then the row CTAs.
struct RequesterGrid {
  int64_t tables, owners, rows;
  __host__ __device__ int64_t scan() const { return tables; }
  __host__ __device__ int64_t first_owner() const { return tables + 1; }
  __host__ __device__ int64_t first_row() const { return first_owner() + owners; }
  __host__ __device__ int64_t ctas() const { return first_row() + rows; }
};

// What every CTA keeps of its node's proxies in its dynamic shared memory (requester_smem_bytes):
// per proxy p, where this requester's rows there start in its push order (its rows of the owners
// with p's local rank on the other nodes; at local_size its rows at every proxy), p's header and
// remote_map as this GPU reads them, and in the scan CTA p's chunks C_p.
struct NodeProxies {
  int64_t *push_order_off;      // [local_size + 1]
  int64_t *num_chunks;          // [local_size]
  const int32_t **headers;      // [local_size]
  const int32_t **remote_maps;  // [local_size]
};
__device__ __forceinline__ NodeProxies node_proxies(int64_t local_size) {
  extern __shared__ int64_t requester_words[];
  int64_t *num_chunks = requester_words + local_size + 1;
  const int32_t **headers = (const int32_t **)(num_chunks + local_size);
  return NodeProxies{requester_words, num_chunks, headers, headers + local_size};
}
// The dynamic shared memory of each CTA of the requester kernel: 4 words per proxy and 1. With
// the kernel's static shared memory (CUB's scans, about 4.5 KiB) it stays in the default 48 KiB up
// to 1,390 GPUs per node.
inline int64_t requester_smem_bytes(int64_t local_size) { return 8 * (4 * local_size + 1); }

// Words [0, n) of a node peer's `words` (int32) to put(index, word), every thread of the CTA: the
// words before the first 16-byte boundary (0 to 3) one each, then 16-byte loads, kVectors of them
// in flight per thread (32 KiB a CTA), then the words past the last whole vector (0 to 3).
template <typename Put>
__device__ __forceinline__ void copy_peer_words(const int32_t *words, int64_t n, Put put) {
  constexpr int kVectors = 4;
  const int64_t head = min(n, (int64_t)((16 - ((uintptr_t)words & 15)) & 15) / 4);
  const int64_t vectors = (n - head) / 4, tail = head + 4 * vectors;
  const int4 *aligned = (const int4 *)(words + head);
  if (threadIdx.x < head) put(threadIdx.x, __ldcg(words + threadIdx.x));
  if (tail + threadIdx.x < n) put(tail + threadIdx.x, __ldcg(words + tail + threadIdx.x));
  for (int64_t first = threadIdx.x; first < vectors; first += (int64_t)kThreads * kVectors) {
    int4 got[kVectors];
#pragma unroll
    for (int k = 0; k < kVectors; ++k)
      if (first + k * kThreads < vectors) got[k] = __ldcg(aligned + first + k * kThreads);
#pragma unroll
    for (int k = 0; k < kVectors; ++k) {
      const int64_t at = head + 4 * (first + k * kThreads);
      if (at >= tail) continue;
      put(at, got[k].x);
      put(at + 1, got[k].y);
      put(at + 2, got[k].z);
      put(at + 3, got[k].w);
    }
  }
}

// Every CTA, all threads: where this requester's rows at each proxy start in its push order (CTA 0
// writes them to push_order_off too) and each proxy's header and remote_map, into shared memory.
// Of node n its rows of the owners (n, q) for q < p are offset_per_bucket[n * local_size + p] less
// offset_per_bucket[n * local_size]: a warp per proxy, its lanes over the other nodes, summed by
// shuffles.
template <class Peers>
__device__ __noinline__ void requester_proxies(const Peers &peers, const int64_t *offset_per_bucket,
                                               const Topo &topo, int64_t *push_order_off) {
  const NodeProxies proxies = node_proxies(topo.local_size);
  const int64_t local_size = topo.local_size;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (int64_t proxy = warp; proxy <= local_size; proxy += kWarps) {
    int64_t rows = 0;
    for (int64_t node = lane; node < topo.num_nodes; node += 32) {
      const int64_t first = node * local_size;
      if (node != topo.node) rows += offset_per_bucket[first + proxy] - offset_per_bucket[first];
    }
    rows = warp_sum(rows);
    if (lane == 0) proxies.push_order_off[proxy] = rows;
    if (lane == 0 && blockIdx.x == 0) push_order_off[proxy] = rows;
  }
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads) {
    proxies.headers[proxy] = peers.header((int)proxy);
    proxies.remote_maps[proxy] = peers.remote_map((int)proxy);
  }
}

// Where this requester's push table starts in a proxy header (int32 words).
__device__ __forceinline__ int64_t own_table_word(const Sizes &sizes) {
  return ProxyHeader::of(sizes).push_table_word(sizes.topo.local_rank);
}

// A proxy's chunks C_p: its header's word 0, clamped into [0, max_chunks].
__device__ __forceinline__ int64_t proxy_chunks(const int32_t *header, int64_t max_chunks) {
  return min(max((int64_t)__ldcg(header + ProxyHeader::kNumChunksWord), (int64_t)0), max_chunks);
}

// A table CTA, proxy `proxy`: its chunks C_p (every thread reads them), its chunk ends [0, C_p)
// and its stream's end (uint32 words, into int64), and this requester's push table there [0, C_p],
// made absolute (its rows at the proxies before it come first in its push order). Then the entries
// past C_p, a thread per entry: ends 0 in [C_p, max_chunks), and in (C_p, max_chunks] of its push
// table the entry at C_p, which every thread reads beside its copy's loads.
__device__ __noinline__ inline void requester_table(int64_t proxy, const RequesterArgs &args,
                                                    const Sizes &sizes) {
  const NodeProxies proxies = node_proxies(sizes.topo.local_size);
  const int64_t max_chunks = sizes.max_chunks, push_offset = proxies.push_order_off[proxy];
  const int32_t *header = proxies.headers[proxy], *published = header + own_table_word(sizes);
  const int64_t chunks = proxy_chunks(header, max_chunks);
  const int32_t table_end = (int32_t)(__ldcg(published + chunks) + push_offset);
  int64_t *ends = args.proxy_ends + proxy * (max_chunks + 1);
  int32_t *table = args.proxy_push_tables + proxy * (max_chunks + 1);
  if (threadIdx.x == 0) {
    args.proxy_num_chunks[proxy] = chunks;
    ends[max_chunks] = (int64_t)(uint32_t)__ldcg(header + ProxyHeader::kEndWord);
  }
  copy_peer_words(header + ProxyHeader::kEndsWord, chunks,
                  [&](int64_t chunk, int32_t end) { ends[chunk] = (int64_t)(uint32_t)end; });
  copy_peer_words(published, chunks + 1, [&](int64_t chunk, int32_t entry) {
    table[chunk] = (int32_t)(entry + push_offset);
  });
  for (int64_t chunk = chunks + threadIdx.x; chunk < max_chunks; chunk += kThreads) {
    ends[chunk] = 0;
    table[chunk + 1] = table_end;
  }
}

// The quotients of values in [0, 2^63) by a divisor in [1, 2^63), without a division per value:
// the high word of value * floor((2^64 - 1) / divisor), which is the quotient or one less, then
// one correction.
struct Divider {
  uint64_t divisor, multiplier;
  __device__ explicit Divider(int64_t by) : divisor(by), multiplier(~0ull / (uint64_t)by) {}
  __device__ __forceinline__ int64_t quotient(int64_t value) const {
    uint64_t estimate = __umul64hi((uint64_t)value, multiplier);
    if ((uint64_t)value - estimate * divisor >= divisor) ++estimate;
    return (int64_t)estimate;
  }
};

// The scan CTA: the remote task offsets, chunk after chunk, in a chunk the proxies in turn from
// this rank's own one (me), kTaskRows positions of its push order a task; a proxy has none past
// its chunks, and the push tables' differences are the same relative or absolute. First a thread
// per proxy its chunks C_p, into shared memory; then a blocked scan over the local_size * maxC
// entries of the chunks there are, then the total there and at local_size * max_chunks;
// owner_tables writes it into the entries between.
__device__ __noinline__ inline void requester_scan(const RequesterArgs &args, const Sizes &sizes) {
  const int64_t local_size = sizes.topo.local_size, me = sizes.topo.local_rank,
                table_word = own_table_word(sizes);
  const NodeProxies proxies = node_proxies(local_size);
  for (int64_t proxy = threadIdx.x; proxy < local_size; proxy += kThreads)
    proxies.num_chunks[proxy] = proxy_chunks(proxies.headers[proxy], sizes.max_chunks);
  __syncthreads();
  int64_t most = 0;  // maxC
  for (int64_t proxy = 0; proxy < local_size; ++proxy) most = max(most, proxies.num_chunks[proxy]);
  const int64_t entries = local_size * most;
  const Divider by_proxies(local_size);
  const int64_t total =
      cta_scan_blocked<scan_items<int64_t>(), int64_t, cub::BLOCK_SCAN_WARP_SCANS>(
          entries,
          [&](int64_t index) -> int64_t {
            const int64_t chunk = by_proxies.quotient(index),
                          turn = me + index - chunk * local_size;
            const int64_t proxy = turn < local_size ? turn : turn - local_size;
            if (chunk >= proxies.num_chunks[proxy]) return 0;
            const int32_t *table = proxies.headers[proxy] + table_word;
            return ceil_div<kTaskRows>((int64_t)__ldcg(table + chunk + 1) - __ldcg(table + chunk));
          },
          [&](int64_t index, int64_t offset) { args.remote_task_offsets[index] = offset; });
  if (threadIdx.x == 0) {
    args.remote_task_offsets[entries] = total;
    args.remote_task_offsets[local_size * sizes.max_chunks] = total;
  }
}

// An owner CTA, other node `other` (its index in other_nodes): where that node's records start in
// rail_keys (records_first) and its records of each feature (record_offsets [feature][other
// node]), counted in records of record_bytes (a slice and its chunks are whole records): after
// its chunks, and its records of the features before (a blocked scan of its keys per feature).
__device__ __noinline__ inline void owner_records_of(int64_t other, const int64_t *slice_in,
                                                     const int64_t *node_length_per_source_feature,
                                                     const int64_t *other_nodes, int64_t num_nodes,
                                                     int64_t num_features, int64_t record_bytes,
                                                     int64_t *records_first,
                                                     int64_t *record_offsets) {
  const int64_t source = other_nodes[other];
  const int64_t first =
      (slice_in[2 * source] + (int64_t)sizeof(OwnerChunk) * slice_in[2 * source + 1]) /
      record_bytes;
  if (threadIdx.x == 0) records_first[other] = first;
  cta_scan_blocked<scan_items<int64_t>(), int64_t>(
      num_features,
      [&](int64_t feature) {
        return node_length_per_source_feature[source * num_features + feature];
      },
      [&](int64_t feature, int64_t offset) {
        record_offsets[feature * (num_nodes - 1) + other] = first + offset;
      });
}

// The rows a thread of the row CTAs takes at a time: their loads go out before their stores.
constexpr int kRowsInFlight = 4;

// The row CTAs: a thread per bucketized row (first, then every stride-th: the row CTAs'), in steps
// of kRowsInFlight rows. A row of an owner on this node is in the owner's lookup_rows at its place
// in the owner's own part (where this requester pushed it, row + row_shift) plus the shift of its
// feature's lookup group there (owner_group_shift [owner][group]), both forward and backward. A
// valid row of another node's owner reads its key's row in proxy_rows and pushes its gradient to
// its slot: entry occ of its proxy's remote_map (its place in the proxy's remote part, where it
// pushed it, row + row_shift less lookup_keys), with its position in its push order there. A row
// of another node's owner past the valid rows gets -1, which nothing reads.
__device__ __noinline__ inline void requester_rows(const RequesterArgs &args, const Sizes &sizes,
                                                   int64_t first, int64_t stride) {
  const NodeProxies proxies = node_proxies(sizes.topo.local_size);
  // int32 like the bucket features (world_size * num_features is below 2^31, config.cu)
  const int32_t local_size = sizes.topo.local_size, world_size = sizes.topo.world_size,
                num_features = (int32_t)sizes.num_features;
  const int64_t rows = args.rows, valid_rows = args.offset_per_bucket[world_size];
  for (int64_t base = first; base < rows; base += kRowsInFlight * stride) {
    // per row: its src and dst, and of a valid row of another node's owner its proxy and its
    // position in its push order there (peer -1: none)
    int32_t src[kRowsInFlight], dst[kRowsInFlight], peer[kRowsInFlight], position[kRowsInFlight];
#pragma unroll
    for (int k = 0; k < kRowsInFlight; ++k) {
      const int64_t row = base + k * stride;
      peer[k] = -1;
      if (row >= rows) continue;
      const int32_t bucket_feature = args.bucket_feature_of_row[row];
      const int32_t owner = min(bucket_feature / num_features, world_size - 1);
      const int64_t shift = args.row_shift[bucket_feature];
      if (owner / local_size == sizes.topo.node) {
        const int32_t group = args.group_of_feature[bucket_feature % num_features];
        src[k] = dst[k] =
            (int32_t)(row + shift +
                      args.owner_group_shift[(int64_t)(owner % local_size) * sizes.num_groups +
                                             group]);
      } else if (row >= valid_rows) {
        src[k] = dst[k] = -1;
      } else {
        peer[k] = owner % local_size;
        const int32_t *map = proxies.remote_maps[peer[k]] +
                             clamp_index(shift + row - sizes.lookup_keys, sizes.remote_keys);
        src[k] = __ldcg(map);
        dst[k] = __ldcg(map + sizes.remote_keys);
        position[k] = __ldcg(map + 2 * sizes.remote_keys);
      }
    }
#pragma unroll
    for (int k = 0; k < kRowsInFlight; ++k) {
      const int64_t row = base + k * stride;
      if (row >= rows) continue;
      args.src[row] = src[k];
      args.dst[row] = dst[k];
      if (peer[k] < 0) continue;
      // a call that does not fit may read words no one wrote: a position out of range is dropped
      const int64_t at = proxies.push_order_off[peer[k]] + position[k];
      if (at >= 0 && at < rows) args.push_order[at] = (int32_t)row;
    }
  }
}

// The requester kernel: its CTAs in RequesterGrid's roles, at most two CTAs per SM (64 registers a
// thread), each role a function that is not inlined, so it has the registers to itself. No CTA
// waits for another, and the owner and row CTAs read no proxy header.
template <class Peers>
__global__ void __launch_bounds__(kThreads, 2)
    requester_kernel(const __grid_constant__ Peers peers,
                     const __grid_constant__ RequesterArgs args,
                     const __grid_constant__ Sizes sizes, RequesterGrid grid) {
  const int64_t cta = blockIdx.x;
  requester_proxies(peers, args.offset_per_bucket, sizes.topo, args.push_order_off);
  __syncthreads();
  if (cta < grid.tables) {
    requester_table(cta, args, sizes);
  } else if (cta == grid.scan()) {
    requester_scan(args, sizes);
  } else if (cta < grid.first_row()) {
    owner_records_of(cta - grid.first_owner(), args.slice_in, args.node_length_per_source_feature,
                     args.other_nodes, sizes.topo.num_nodes, sizes.num_features, sizes.record_bytes,
                     args.records_first, args.record_offsets);
  } else {
    requester_rows(args, sizes, (cta - grid.first_row()) * kThreads + threadIdx.x,
                   grid.rows * kThreads);
  }
}

// The requester kernel's tables of a call on `peers` (WindowPeers in the in channel, StandInPeers
// in the hooks), in new tensors on the current stream; with `rail` (and other_nodes, int64
// [num_nodes - 1]) also the owner's records, on a CTA per other node (none without). Its grid and
// shared memory follow from the sizes, the rows and the SMs (requester_row_ctas: the row CTAs
// take the CTA slots the other roles leave, at least one per SM).
template <class Peers>
std::pair<RequesterTables, OwnerRecords> requester_tables_of(
    const Peers &peers, const at::Tensor &bucket_feature_of_row,
    const at::Tensor &offset_per_bucket, const at::Tensor &row_shift,
    const at::Tensor &owner_group_shift, const at::Tensor &group_of_feature, const RailLayout *rail,
    const at::Tensor &other_nodes, const Sizes &sizes) {
  const int64_t rows = bucket_feature_of_row.numel(), local_size = sizes.topo.local_size,
                max_chunks = sizes.max_chunks;
  const auto long_options = offset_per_bucket.options().dtype(at::kLong);
  const auto int_options = offset_per_bucket.options().dtype(at::kInt);
  RequesterTables tables;
  tables.src = at::empty({rows}, int_options);
  tables.dst = at::empty({rows}, int_options);
  tables.push_order = at::empty({rows}, int_options);
  tables.push_order_off = at::empty({local_size + 1}, long_options);
  tables.proxy_num_chunks = at::empty({local_size}, long_options);
  tables.proxy_ends = at::empty({local_size, max_chunks + 1}, long_options);
  tables.proxy_push_tables = at::empty({local_size, max_chunks + 1}, int_options);
  tables.remote_task_offsets = at::empty({max_chunks * local_size + 1}, long_options);
  OwnerRecords records;
  if (rail != nullptr) {
    records.records_first = at::empty({sizes.topo.num_nodes - 1}, long_options);
    records.record_offsets =
        at::empty({sizes.num_features * (sizes.topo.num_nodes - 1)}, long_options);
  }
  RequesterArgs args{
      bucket_feature_of_row.data_ptr<int32_t>(),
      offset_per_bucket.data_ptr<int64_t>(),
      row_shift.data_ptr<int64_t>(),
      owner_group_shift.data_ptr<int64_t>(),
      group_of_feature.data_ptr<int32_t>(),
      rows,
      rail != nullptr ? rail->slice_in.data_ptr<int64_t>() : nullptr,
      rail != nullptr ? rail->node_length_per_source_feature.data_ptr<int64_t>() : nullptr,
      rail != nullptr ? other_nodes.data_ptr<int64_t>() : nullptr,
      tables.src.data_ptr<int32_t>(),
      tables.dst.data_ptr<int32_t>(),
      tables.push_order.data_ptr<int32_t>(),
      tables.push_order_off.data_ptr<int64_t>(),
      tables.proxy_num_chunks.data_ptr<int64_t>(),
      tables.proxy_ends.data_ptr<int64_t>(),
      tables.proxy_push_tables.data_ptr<int32_t>(),
      tables.remote_task_offsets.data_ptr<int64_t>(),
      data_or_null<int64_t>(records.records_first),
      data_or_null<int64_t>(records.record_offsets)};
  const int64_t owners = rail != nullptr ? sizes.topo.num_nodes - 1 : 0;
  const RequesterGrid grid{local_size, owners, requester_row_ctas(rows, local_size + 1 + owners)};
  requester_kernel<Peers>
      <<<(int)grid.ctas(), kThreads, requester_smem_bytes(local_size), current_stream()>>>(
          peers, args, sizes, grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {tables, records};
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
