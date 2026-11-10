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
// Test-only bindings of the hierarchical dist. Most run the planner's steps
// outside an op, on plain tensors: the bucketize, the node layout, a
// requester's push into stand-ins for its targets' node_keys, and one rank's
// proxy, rail and requester steps across nodes. Last, a hook that makes the
// next construction fail, a hook that delays the reads of node_offsets and
// node_keys of later input dists, hooks that choose the paths of later ones
// (node_exchange's, and the CTAs of the proxy's scans), a hook that fills
// tables of the proxy's with garbage before its kernels write them, and the
// GPU clock the benchmark aligns timelines by (a build with
// DYNAMICEMB_HIER_TRACE=1).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "hier_output_dist/common/common.h"
#include "hier_output_dist/common/config.h"
#include "hier_output_dist/common/device.cuh"
#include "hier_output_dist/common/window.h"
#include "hier_output_dist/hier_output_dist.h"
#include "hier_output_dist/input_dist/input_dist.h"
#include "hier_output_dist/input_dist/planner.h"
#include "utils.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace py = pybind11;

namespace {

// The bucketize of a KJT (bucketize_keys), by name, with each owner's feature
// offsets relative to its block (feature_offsets [world_size][num_features + 1]),
// which TorchRec's bucketize gives. The keys by owner are the KJT's keys
// gathered at inv, widened to int64 with their sign. dist_types are the
// rules' codes on the host, kCyclic or kHashedCyclic, the op's (it raises for
// any other, as Config::validate refuses them); their copy to the GPU does
// not wait for the device (the tests run the planner under the sync debug
// mode "error"). num_nodes, local_size and node (all or none): the bucketize
// of the in channel of a rank of that node, whose offset_per_bucket_feature
// holds each owner's first entry and every entry of its node's owners, and
// garbage elsewhere (then no feature_offsets); none, every entry.
// sorted_path: bucketize_rows' (the radix sort by bucket feature, the
// counting sort, or none: as the sizes choose); counted (bool, on the host):
// whether the counting sort ran.
std::map<std::string, at::Tensor> bucketize_test(
    at::Tensor values, c10::optional<at::Tensor> offsets, c10::optional<at::Tensor> lengths,
    int64_t batch_size, c10::optional<at::Tensor> weights, std::vector<int> dist_types,
    int64_t world_size, at::Tensor input_offsets, c10::optional<int64_t> num_nodes,
    c10::optional<int64_t> local_size, c10::optional<int64_t> node,
    c10::optional<bool> sorted_path) {
  for (const int code : dist_types)
    TORCH_CHECK(code == (int)DistType::kCyclic || code == (int)DistType::kHashedCyclic,
                "hier_output_dist: the bucketize takes dist_type codes ", (int)DistType::kCyclic,
                " (roundrobin) and ", (int)DistType::kHashedCyclic, " (hash_roundrobin), got ",
                code);
  const bool of_node = node.has_value();
  TORCH_CHECK(num_nodes.has_value() == of_node && local_size.has_value() == of_node &&
                  (!of_node ||
                   (*num_nodes >= 1 && *local_size >= 1 && *num_nodes * *local_size == world_size &&
                    *node >= 0 && *node < *num_nodes)),
              "hier_output_dist: the bucketize's node: num_nodes, local_size and node of "
              "world_size ranks, or none of them");
  c10::cuda::CUDAGuard guard(values.device());
  PlanInput input;
  input.values = values;
  input.offsets = offsets;
  input.lengths = lengths;
  input.batch_size = batch_size;
  input.weights = weights;
  input.input_offsets = input_offsets;
  const at::Tensor rules = at::tensor(dist_types, at::kInt)
                               .pin_memory()
                               .to(values.options().dtype(at::kInt), /*non_blocking=*/true);
  check_bucketize_input(input, rules, world_size);
  const int64_t first_owner = of_node ? *node * *local_size : 0,
                owners = of_node ? *local_size : world_size;
  const BucketizedKeys bucketized =
      bucketize_keys(input, rules, world_size, first_owner, owners, sorted_path);
  const int64_t num_features = (int64_t)dist_types.size();
  std::map<std::string, at::Tensor> named{
      {"keys", bucketized.keys},
      {"inv", bucketized.inv},
      {"bucket_feature_of_row", bucketized.bucket_feature_of_row},
      {"offset_per_bucket_feature", bucketized.offset_per_bucket_feature},
      {"offset_per_bucket", bucketized.offset_per_bucket},
      {"input_length_per_key", bucketized.input_length_per_key},
      {"counted", at::scalar_tensor(bucketized.counted, at::kBool)}};
  if (!of_node)
    named["feature_offsets"] =
        (bucketized.offset_per_bucket_feature.unfold(0, num_features + 1, num_features) -
         bucketized.offset_per_bucket.narrow(0, 0, world_size).unsqueeze(1))
            .reshape({-1});
  if (bucketized.key_weights.defined()) named["weights"] = bucketized.key_weights;
  return named;
}

// Index dedup's fan-out (fanout_tables), by name. words: the kernel's words
// (int64 [kFanoutWords]), as the in channel keeps them across calls; none, a
// zeroed tensor of this call's.
std::map<std::string, at::Tensor> fanout_test(at::Tensor reverse_indices, at::Tensor inv,
                                              c10::optional<at::Tensor> words) {
  check_fanout_input(reverse_indices);
  c10::cuda::CUDAGuard guard(inv.device());
  const FanoutTables fanout = fanout_tables(reverse_indices, inv, words.value_or(at::Tensor()));
  return {
      {"exp_off", fanout.exp_off}, {"exp_idx", fanout.exp_idx}, {"heavy_slot", fanout.heavy_slot},
      {"sub", fanout.sub},         {"count", fanout.count},     {"pos", fanout.pos},
      {"first", fanout.first}};
}

// The ranks of a sharding of num_nodes nodes of local_size GPUs, as rank
// local_rank of node `node` sees them.
Topo topo_of(int64_t num_nodes, int64_t local_size, int64_t node, int64_t local_rank) {
  TORCH_CHECK(num_nodes >= 1 && local_size >= 1 && node >= 0 && node < num_nodes &&
                  local_rank >= 0 && local_rank < local_size,
              "hier_output_dist: the ranks of a node (num_nodes, local_size, node, local_rank)");
  return Topo{(int)(num_nodes * local_size),
              (int)local_size,
              (int)num_nodes,
              (int)(local_size + num_nodes - 1),
              (int)(node * local_size + local_rank),
              (int)local_rank,
              (int)node};
}

// One rank of a node: its node layout (node_layout) on the node's offsets (node_offsets: int64
// [local_size][world_size * num_features + 1], row r requester r's offset_per_bucket_feature, of
// which node_exchange gathers the rows they push), by name (remote_offsets and need_remote across
// nodes, kjt_offsets and length_per_key on one node). items_in_cta: lay the items out on one CTA
// or on a grid (none: as node_exchange chooses). On one node, given the own part of its node_keys
// once every requester pushed (node_keys, and node_key_weights with weights: [lookup_keys]) and
// the src its own push wrote, also its KJT and src (one_node_owner).
std::map<std::string, at::Tensor> node_layout_test(
    at::Tensor node_offsets, int64_t num_nodes, int64_t local_size, int64_t node,
    int64_t local_rank, int64_t num_features, int64_t lookup_keys, int64_t remote_keys,
    c10::optional<at::Tensor> node_keys, c10::optional<at::Tensor> node_key_weights,
    c10::optional<at::Tensor> src, c10::optional<bool> items_in_cta) {
  c10::cuda::CUDAGuard guard(node_offsets.device());
  const Topo topo = topo_of(num_nodes, local_size, node, local_rank);
  TORCH_CHECK(num_features >= 1 && lookup_keys >= 0 && remote_keys >= 0 &&
                  (num_nodes > 1 || remote_keys == 0),
              "hier_output_dist: node_layout's sizes (remote_keys 0 on one node)");
  const NodeLayout layout =
      node_layout(node_offsets, topo, num_features, lookup_keys, remote_keys, items_in_cta);
  std::map<std::string, at::Tensor> named{{"own_lengths", layout.own_lengths},
                                          {"own_offsets", layout.own_offsets},
                                          {"own_total", layout.own_total},
                                          {"need_lookup", layout.need_lookup},
                                          {"skip", layout.skip},
                                          {"row_shift", layout.row_shift}};
  if (num_nodes > 1) {
    named["remote_offsets"] = layout.remote_offsets;
    named["need_remote"] = layout.need_remote;
  } else {
    named["kjt_offsets"] = layout.kjt_offsets;
    named["length_per_key"] = layout.length_per_key;
  }
  TORCH_CHECK(node_keys.has_value()
                  ? num_nodes == 1 && is_cuda_array(*node_keys, at::kLong, lookup_keys) &&
                        (!node_key_weights.has_value() ||
                         is_cuda_array(*node_key_weights, at::kFloat, lookup_keys)) &&
                        src.has_value() && src->is_cuda() && src->scalar_type() == at::kInt
                  : !node_key_weights.has_value() && !src.has_value(),
              "hier_output_dist: node_layout's own part (one node: node_keys and src)");
  if (!node_keys.has_value()) return named;
  const OneNodeOwner owner =
      one_node_owner(layout, *node_keys, node_key_weights.value_or(at::Tensor()), *src, topo);
  named["kjt_values"] = owner.kjt_values;
  if (owner.kjt_weights.defined()) named["kjt_weights"] = owner.kjt_weights;
  named["kjt_lengths"] = owner.kjt_lengths;
  named["kjt_offsets"] = owner.kjt_offsets;
  named["num_recv"] = owner.num_recv;
  named["length_per_key"] = owner.length_per_key;
  named["src"] = owner.src;
  return named;
}

// One requester's push (push_to_node_keys: node_key_push's push_rows) of its KJT's keys (values,
// int32 or int64) and weights (or None), on what its bucketize gives (bucketize_rows: inv,
// bucket_feature_of_row and offset_per_bucket) and its node layout (row_shift, skip),
// into stand-ins for its targets' node_keys: row t of node_keys ([local_size][lookup_keys +
// remote_keys]), of node_key_weights (given exactly with weights) and across nodes of
// node_bucket_feature_of_row ([local_size][remote_keys]); on one node also src (int32 [rows]).
// It writes them in place: an entry it pushes nothing to keeps what it held, and a place outside
// a row is not stored.
void node_push_test(at::Tensor values, at::Tensor inv, c10::optional<at::Tensor> weights,
                    at::Tensor bucket_feature_of_row, at::Tensor offset_per_bucket,
                    at::Tensor row_shift, at::Tensor skip, int64_t num_nodes, int64_t local_size,
                    int64_t node, int64_t local_rank, int64_t num_features, int64_t lookup_keys,
                    int64_t remote_keys, at::Tensor node_keys,
                    c10::optional<at::Tensor> node_key_weights,
                    c10::optional<at::Tensor> node_bucket_feature_of_row,
                    c10::optional<at::Tensor> src) {
  c10::cuda::CUDAGuard guard(inv.device());
  const Topo topo = topo_of(num_nodes, local_size, node, local_rank);
  const int64_t rows = inv.numel();
  TORCH_CHECK(num_features >= 1 && values.is_cuda() && values.is_contiguous() &&
                  (values.scalar_type() == at::kInt || values.scalar_type() == at::kLong) &&
                  values.numel() == rows && is_cuda_array(inv, at::kInt, rows) &&
                  (!weights.has_value() || is_cuda_array(*weights, at::kFloat, rows)) &&
                  is_cuda_array(bucket_feature_of_row, at::kInt, rows) &&
                  is_cuda_array(offset_per_bucket, at::kLong, topo.world_size + 1) &&
                  is_cuda_array(row_shift, at::kLong, topo.world_size * num_features + 1) &&
                  is_cuda_array(skip, at::kBool, 1) &&
                  node_bucket_feature_of_row.has_value() == (num_nodes > 1),
              "hier_output_dist: node_push's inputs (node_bucket_feature_of_row exactly across "
              "nodes)");
  NodeLayout layout;
  layout.row_shift = row_shift;
  layout.skip = skip;
  push_to_node_keys(values, inv, weights.value_or(at::Tensor()), bucket_feature_of_row,
                    offset_per_bucket, layout, topo, num_features, lookup_keys, remote_keys,
                    node_keys, node_key_weights.value_or(at::Tensor()),
                    node_bucket_feature_of_row.value_or(at::Tensor()), src.value_or(at::Tensor()));
}

// The sizes of a call across nodes (Sizes) by name: num_nodes, local_size,
// node, local_rank, num_features, lookup_keys, remote_keys, num_proxy_rows,
// max_chunk_rows, record_bytes, and payload_rows (the rows of a chunk's
// payload; max_chunk_rows when not given); and the lookup groups (the
// number of features of each, Config::lookup_groups: empty, one group). rows_per_payload is
// rows_per_payload_for(max_chunk_rows, payload_rows): here max_chunk_rows
// stands for sizes_of's chunk_ring_rows. The two differ only when sizes_of's
// least of 2 * max_key_rows sets max_chunk_rows. Then max_chunks follows, and
// weighted with record_bytes. No step of the planner reads the others
// (row_bytes), which stay 0.
Sizes plan_sizes(const std::map<std::string, int64_t> &named,
                 const std::vector<int64_t> &lookup_groups) {
  const int num_nodes = (int)named.at("num_nodes"), local_size = (int)named.at("local_size"),
            node = (int)named.at("node"), local_rank = (int)named.at("local_rank");
  Sizes sizes{};
  sizes.topo = topo_of(num_nodes, local_size, node, local_rank);
  sizes.num_features = named.at("num_features");
  sizes.num_groups = lookup_groups.empty() ? 1 : (int64_t)lookup_groups.size();
  sizes.lookup_keys = named.at("lookup_keys");
  sizes.remote_keys = named.at("remote_keys");
  sizes.num_proxy_rows = named.at("num_proxy_rows");
  sizes.max_chunk_rows = named.at("max_chunk_rows");
  const auto payload = named.find("payload_rows");
  const int64_t payload_rows = payload != named.end() ? payload->second : sizes.max_chunk_rows;
  TORCH_CHECK(payload_rows >= 1, "hier_output_dist: a chunk's payload of at least one row");
  sizes.rows_per_payload = rows_per_payload_for(sizes.max_chunk_rows, payload_rows);
  sizes.record_bytes = named.at("record_bytes");
  sizes.weighted = sizes.weighted_records();
  // a chunk holds two keys of the most counted rows (sizes_of's do)
  TORCH_CHECK(num_nodes >= 2 && local_size >= 1 && node >= 0 && node < num_nodes &&
                  local_rank >= 0 && local_rank < local_size && sizes.num_features >= 1 &&
                  num_nodes * local_size * sizes.num_features < INT32_MAX &&
                  sizes.lookup_keys >= 1 && sizes.lookup_keys < INT32_MAX &&
                  sizes.remote_keys >= 1 && sizes.remote_keys < INT32_MAX &&
                  sizes.max_chunk_rows >= 2 * sizes.max_counted_rows() &&
                  sizes.num_proxy_rows >= 2 * sizes.max_chunk_rows &&
                  (sizes.record_bytes == 8 || sizes.record_bytes == 16),
              "hier_output_dist: the sizes of a call across nodes");
  sizes.max_chunks = max_chunks_for(sizes);
  int64_t grouped = 0;
  for (const int64_t group : lookup_groups) {
    TORCH_CHECK(group >= 1, "hier_output_dist: a lookup group of at least one feature");
    grouped += group;
  }
  TORCH_CHECK(lookup_groups.empty() || grouped == sizes.num_features,
              "hier_output_dist: lookup groups of num_features features in all");
  return sizes;
}

// The op's constants of the lookup groups (InChannel's): the group of each
// feature (int32 [num_features]) and each group's first feature, then
// num_features (int64 [num_groups + 1]), on the device of `options`; copies
// that do not wait for the device (the tests run the planner under the sync
// debug mode "error").
std::pair<at::Tensor, at::Tensor> group_tables(const std::vector<int64_t> &lookup_groups,
                                               int64_t num_features,
                                               const at::TensorOptions &options) {
  std::vector<int64_t> starts{0};
  if (lookup_groups.empty()) starts.push_back(num_features);
  for (const int64_t group : lookup_groups) starts.push_back(starts.back() + group);
  std::vector<int> group_of_feature;
  for (int64_t group = 0; group + 1 < (int64_t)starts.size(); ++group)
    group_of_feature.insert(group_of_feature.end(), starts[group + 1] - starts[group], (int)group);
  const at::Device device = options.device();
  return {at::tensor(group_of_feature, at::kInt).pin_memory().to(device, at::kInt, true),
          at::tensor(starts, at::kLong).pin_memory().to(device, at::kLong, true)};
}

// One proxy's steps of a call across nodes (proxy_keys,
// proxy_chunks_and_slots, proxy_push_order) on its remote part (node_keys,
// node_key_weights, node_bucket_feature_of_row: what those buffers hold
// past lookup_keys; weights exactly with 16-byte records) and what the node
// layout says of it (remote_offsets, skip, need_remote, own_lengths: its own
// part's lengths [feature][requester]), and the count messages it sends its
// rail (count_message: [num_nodes][CountMessage::length], row n to the peer on
// node n), by name. It writes remote_map, key_stage and proxy_header (a proxy
// header, ProxyHeader) as the steps write those buffers of its window. Also
// the node-unique key of each occurrence (node_reverse_indices, garbage past
// the valid ones) and the node-unique keys (node_num_uniques), for the test's
// reference.
std::map<std::string, at::Tensor> proxy_plan_test(
    at::Tensor node_keys, c10::optional<at::Tensor> node_key_weights,
    at::Tensor node_bucket_feature_of_row, at::Tensor remote_offsets, at::Tensor skip,
    at::Tensor tables, at::Tensor need_remote, at::Tensor own_lengths, at::Tensor remote_map,
    at::Tensor key_stage, at::Tensor proxy_header, bool node_dedup,
    std::map<std::string, int64_t> named_sizes, std::vector<int64_t> lookup_groups) {
  c10::cuda::CUDAGuard guard(node_keys.device());
  const Sizes sizes = plan_sizes(named_sizes, lookup_groups);
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                num_features = sizes.num_features;
  ProxyHeader header = ProxyHeader::of(sizes);  // over proxy_header once checked
  TORCH_CHECK(is_cuda_array(node_keys, at::kLong, entries) &&
                  node_key_weights.has_value() == sizes.weighted_records() &&
                  (!node_key_weights.has_value() ||
                   is_cuda_array(*node_key_weights, at::kFloat, entries)) &&
                  is_cuda_array(node_bucket_feature_of_row, at::kInt, entries) &&
                  is_cuda_array(remote_offsets, at::kLong, local_size + 1) &&
                  is_cuda_array(skip, at::kBool, 1) &&
                  is_cuda_array(tables, at::kInt, num_features) &&
                  is_cuda_array(need_remote, at::kLong, 1) &&
                  is_cuda_array(own_lengths, at::kLong, num_features * local_size) &&
                  is_cuda_array(remote_map, at::kInt, RemoteMap::kArrays * entries) &&
                  is_cuda_array(key_stage, at::kLong,
                                (int64_t)KeysLayout(sizes, nullptr).key_stage_bytes() / 8) &&
                  is_cuda_array(proxy_header, at::kInt, header.words()),
              "hier_output_dist: proxy_plan's inputs");
  const ProxyInput input{node_keys,
                         node_key_weights.has_value() ? *node_key_weights : at::Tensor(),
                         node_bucket_feature_of_row,
                         remote_offsets,
                         skip.reshape({}),
                         tables};
  header.base = proxy_header.data_ptr<int32_t>();
  const ProxyBuffers buffers = proxy_buffers(remote_map.view({RemoteMap::kArrays, entries}),
                                             key_stage, header, proxy_header.options());
  ProxyKeys proxy = proxy_keys(input, sizes, node_dedup);
  const at::Tensor node_reverse_indices = proxy.node_reverse_indices;  // the push order drops it
  const at::Tensor node_num_uniques = proxy.node_num_uniques;          // the plan does not keep it
  ProxyChunks chunks = proxy_chunks_and_slots(proxy, remote_offsets, buffers, sizes);
  const ProxyPlan plan =
      proxy_push_order(std::move(proxy), std::move(chunks), remote_offsets, buffers, sizes);
  return {{"num_chunks", plan.num_chunks},
          {"occurrence_count", plan.occurrence_count},
          {"need_key_rows", plan.need_key_rows},
          {"node_length_per_bucket_feature", plan.node_length_per_bucket_feature},
          {"node_offset_per_bucket", plan.node_offset_per_bucket},
          {"node_length_per_bucket", plan.node_length_per_bucket},
          {"chunks", plan.chunks},
          {"chunk_length_per_bucket", plan.chunk_length_per_bucket},
          {"chunk_offset_per_bucket", plan.chunk_offset_per_bucket},
          {"chunk_inv", plan.chunk_inv},
          {"stage_offset_per_bucket", plan.stage_offset_per_bucket},
          {"reduce_task_offsets", plan.reduce_task_offsets},
          {"node_exp_off", plan.node_exp_off},
          {"message",
           count_message(need_remote.reshape({}), own_lengths, plan.need_key_rows,
                         plan.chunk_length_per_bucket, plan.node_length_per_bucket_feature,
                         group_tables(lookup_groups, num_features, node_keys.options()).second)},
          {"node_reverse_indices", node_reverse_indices},
          {"node_num_uniques", node_num_uniques}};
}

// One rank's rail steps of a call on its rail's count messages to it (int64
// [num_nodes][CountMessage::length], row n the message of the rail peer on node
// n, this rank's own too, as rail_exchange gets them), by name: the rail
// layout (rail_layout), its needs and group shifts the ones the rank publishes
// to its node. Given its rail_keys (int64) and the own part of its node_keys
// (and node_key_weights) once the rail's proxies and its node's requesters put
// their keys, also its owner tables (owner_tables: its KJT, the chunks its
// forward puts and its row map).
std::map<std::string, at::Tensor> rail_tables_test(at::Tensor messages, at::Tensor own_lengths,
                                                   at::Tensor own_offsets,
                                                   c10::optional<at::Tensor> rail_keys,
                                                   c10::optional<at::Tensor> node_keys,
                                                   c10::optional<at::Tensor> node_key_weights,
                                                   std::map<std::string, int64_t> named_sizes,
                                                   std::vector<int64_t> lookup_groups) {
  c10::cuda::CUDAGuard guard(messages.device());
  const Sizes sizes = plan_sizes(named_sizes, lookup_groups);
  const int64_t local_size = sizes.topo.local_size, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, lookup_keys = sizes.lookup_keys;
  const int64_t own_columns = num_features * local_size;
  TORCH_CHECK(
      is_cuda_array(messages, at::kLong,
                    num_nodes * CountMessage{num_nodes, num_features, sizes.num_groups}.length()) &&
          is_cuda_array(own_lengths, at::kLong, own_columns) &&
          is_cuda_array(own_offsets, at::kLong, own_columns) &&
          (!rail_keys.has_value() ||
           (is_cuda_array(*rail_keys, at::kLong,
                          (int64_t)KeysLayout(sizes, nullptr).rail_keys_bytes() / 8) &&
            node_keys.has_value() && is_cuda_array(*node_keys, at::kLong, lookup_keys))) &&
          (!node_key_weights.has_value() ||
           is_cuda_array(*node_key_weights, at::kFloat, lookup_keys)),
      "hier_output_dist: rail_tables' inputs");
  const RailLayout rail = rail_layout(messages.view({num_nodes, -1}), own_lengths, sizes);
  std::map<std::string, at::Tensor> named{
      {"put_dst", rail.put_dst},
      {"put_bytes", rail.put_bytes},
      {"put_skip", rail.put_skip},
      {"bwd_groups", rail.bwd_groups},
      {"kjt_lengths", rail.kjt_lengths},
      {"kjt_offsets", rail.kjt_offsets},
      {"num_recv", rail.num_recv},
      {"length_per_key", rail.length_per_key},
      {"slice_in", rail.slice_in},
      {"node_length_per_source", rail.node_length_per_source},
      {"node_length_per_source_feature", rail.node_length_per_source_feature},
      {"fwd_groups", rail.fwd_groups},
      {"group_offsets", rail.group_offsets},
      {"group_shift", rail.group_shift},
      {"needs", rail.needs}};
  if (rail_keys.has_value()) {
    const OwnerTables owner =
        owner_tables(rail, own_offsets, *node_keys,
                     node_key_weights.has_value() ? *node_key_weights : at::Tensor(), *rail_keys,
                     other_nodes_of(sizes.topo, messages.options()),
                     group_tables(lookup_groups, num_features, messages.options()).first, sizes);
    named["kjt_values"] = owner.kjt_values;
    if (owner.kjt_weights.defined()) named["kjt_weights"] = owner.kjt_weights;
    named["fwd_chunks"] = owner.fwd_chunks;
    named["fwd_chunk_offsets"] = owner.fwd_chunk_offsets;
    if (owner.row_map.defined()) named["row_map"] = owner.row_map;
  }
  return named;
}

// One requester's tables of a call across nodes (requester_tables), by name,
// from what its node's proxies published (remote_maps: int32 [local_size][3]
// [remote_keys], RemoteMap; proxy_headers: int32 [local_size]
// [ProxyHeader::words]) and its node's owners' group shifts (owner_group_shift
// [local_size][num_groups], as node_status_exchange gives them). The two
// gathers read them as peer_gather would, on the word lists of
// remote_map_gather and header_gather; an item a gather skips holds -1.
std::map<std::string, at::Tensor> requester_tables_test(
    at::Tensor remote_maps, at::Tensor proxy_headers, at::Tensor owner_group_shift,
    at::Tensor bucket_feature_of_row, at::Tensor offset_per_bucket, at::Tensor row_shift,
    std::map<std::string, int64_t> named_sizes, std::vector<int64_t> lookup_groups) {
  c10::cuda::CUDAGuard guard(bucket_feature_of_row.device());
  const Sizes sizes = plan_sizes(named_sizes, lookup_groups);
  const int64_t local_size = sizes.topo.local_size, num_features = sizes.num_features,
                bucket_features = sizes.topo.world_size * num_features;
  const ProxyHeader header = ProxyHeader::of(sizes);  // its words
  TORCH_CHECK(
      is_cuda_array(remote_maps, at::kInt, local_size * RemoteMap::kArrays * sizes.remote_keys) &&
          is_cuda_array(proxy_headers, at::kInt, local_size * header.words()) &&
          is_cuda_array(owner_group_shift, at::kLong, local_size * sizes.num_groups) &&
          bucket_feature_of_row.is_cuda() && bucket_feature_of_row.is_contiguous() &&
          bucket_feature_of_row.scalar_type() == at::kInt &&
          is_cuda_array(offset_per_bucket, at::kLong, sizes.topo.world_size + 1) &&
          is_cuda_array(row_shift, at::kLong, bucket_features + 1),
      "hier_output_dist: requester_tables' inputs");
  // peer_gather: out[k][item] = word first_word[item] + k * stride of its
  // peer's buffer (a proxy header: one word; remote_map: entry first_word of
  // each array, remote_keys words apart)
  auto gather = [](const at::Tensor &buffers, const PeerWords &items, int64_t words,
                   int64_t stride) {
    const at::Tensor first =
        items.peer.clamp_min(0).to(at::kLong) * buffers.size(1) + items.first_word;
    const at::Tensor word =
        at::arange(words, first.options()).view({words, 1}) * stride + first.view({1, -1});
    return at::where(items.peer.ge(0), buffers.view({-1}).take(word), -1);
  };
  const at::Tensor words =
      gather(remote_maps.view({local_size, -1}),
             remote_map_gather(bucket_feature_of_row, offset_per_bucket, row_shift, sizes),
             RemoteMap::kArrays, sizes.remote_keys);
  const at::Tensor headers = gather(proxy_headers.view({local_size, -1}),
                                    header_gather(sizes, bucket_feature_of_row.options()), 1, 0)
                                 .view({local_size, header.gathered_words()});
  const RequesterTables tables = requester_tables(
      words, headers, owner_group_shift.view({local_size, sizes.num_groups}), bucket_feature_of_row,
      row_shift, group_tables(lookup_groups, num_features, bucket_feature_of_row.options()).first,
      offset_per_bucket, sizes);
  return {{"src", tables.src},
          {"dst", tables.dst},
          {"push_order", tables.push_order},
          {"push_order_off", tables.push_order_off},
          {"proxy_push_tables", tables.proxy_push_tables},
          {"proxy_num_chunks", tables.proxy_num_chunks},
          {"proxy_ends", tables.proxy_ends},
          {"remote_task_offsets", tables.remote_task_offsets}};
}

// Benchmark tracing: this GPU's %globaltimer into out[index], on the current
// stream (each GPU's clock has its own origin; ping-pongs of these stamps
// between ranks give their offsets). The kernel exists only in a build with
// the trace.
#if DEMB_HIER_TRACE
__global__ void clock_kernel(int64_t *out) { *out = (int64_t)now_ns(); }
#endif

void clock_stamp(at::Tensor out, int64_t index) {
  TORCH_CHECK(kTrace, "hier_output_dist_clock: built without DYNAMICEMB_HIER_TRACE=1");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong && index >= 0 &&
                  index < out.numel() && out.is_contiguous(),
              "hier_output_dist_clock: a contiguous CUDA int64 tensor and an index in it");
  c10::cuda::CUDAGuard guard(out.device());
#if DEMB_HIER_TRACE
  clock_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int64_t>() + index);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#endif
}

}  // namespace

void bind_testing(py::module &py_module) {
  py_module.def("hier_output_dist_fail_next_construction", &fail_next_construction,
                "The next construction of an op in this process fails to allocate",
                py::arg("fail_alloc"));
  py_module.def("hier_output_dist_bucketize", &bucketize_test,
                "The keys bucketized by owner (widened to int64 with their sign), their rows in "
                "the KJT, their bucket features and offsets, and the input's per-feature totals, "
                "by name; given num_nodes, local_size and node, as the in channel of that node "
                "writes the offsets; sorted_path: the radix sort (True), the counting sort "
                "(False), or as the sizes choose (None); counted: whether the counting sort ran",
                py::arg("values"), py::arg("offsets"), py::arg("lengths"), py::arg("batch_size"),
                py::arg("weights"), py::arg("dist_types"), py::arg("world_size"),
                py::arg("input_offsets"), py::arg("num_nodes") = py::none(),
                py::arg("local_size") = py::none(), py::arg("node") = py::none(),
                py::arg("sorted_path") = py::none());
  py_module.def("hier_output_dist_fanout", &fanout_test,
                "Index dedup's fan-out of the bucketized rows, by name; words: the kernel's "
                "counter and CTA ticket (int64 [2], zero), which it leaves at zero, or None",
                py::arg("reverse_indices"), py::arg("inv"), py::arg("words") = py::none());
  py_module.def("hier_output_dist_node_layout", &node_layout_test,
                "One rank's node layout on its node's offsets, and on one node given its own "
                "part of node_keys and its src its KJT, by name",
                py::arg("node_offsets"), py::arg("num_nodes"), py::arg("local_size"),
                py::arg("node"), py::arg("local_rank"), py::arg("num_features"),
                py::arg("lookup_keys"), py::arg("remote_keys"), py::arg("node_keys") = py::none(),
                py::arg("node_key_weights") = py::none(), py::arg("src") = py::none(),
                py::arg("items_in_cta") = py::none());
  py_module.def("hier_output_dist_node_push", &node_push_test,
                "One requester's push of its bucketized rows into stand-ins for its targets' "
                "node_keys ([local_size] rows), in place",
                py::arg("values"), py::arg("inv"), py::arg("weights"),
                py::arg("bucket_feature_of_row"), py::arg("offset_per_bucket"),
                py::arg("row_shift"), py::arg("skip"), py::arg("num_nodes"), py::arg("local_size"),
                py::arg("node"), py::arg("local_rank"), py::arg("num_features"),
                py::arg("lookup_keys"), py::arg("remote_keys"), py::arg("node_keys"),
                py::arg("node_key_weights") = py::none(),
                py::arg("node_bucket_feature_of_row") = py::none(), py::arg("src") = py::none());
  py_module.def("hier_output_dist_delay_node_reads", &delay_node_reads,
                "Each later input dist of this process waits ns on the GPU twice (0: none): in "
                "node_exchange between the node barrier and the read of node_offsets, and between "
                "node_done and the first read of node_keys. Its rank reads the node's offsets and "
                "keys late",
                py::arg("ns"));
  py_module.def("hier_output_dist_node_paths", &force_node_paths,
                "Each later input dist of this process pushes its row of node_offsets (push) and "
                "lays out the node's items (items) on node_exchange's CTA (1) or on a grid (0), or "
                "as their sizes choose (-1)",
                py::arg("push"), py::arg("items"));
  py_module.def("hier_output_dist_scan_ctas", &force_scan_ctas,
                "Each later proxy step of this process runs its scans over a device count (of the "
                "representatives and of the keys' rows) on ctas CTAs, or as their sizes choose (0)",
                py::arg("ctas"));
  py_module.def("hier_output_dist_proxy_garbage", &fill_proxy_garbage,
                "With fill, each later proxy step of this process sets every bit of the rows "
                "before each key, its chunk tables and reduce_task_offsets before its kernels "
                "write them, so a word they leave out shows (False: as the caching allocator "
                "gives them)",
                py::arg("fill"));
  py_module.def(
      "hier_output_dist_proxy_plan", &proxy_plan_test,
      "One proxy's steps of a call across nodes on plain tensors, which it writes as "
      "its buffers of the window, and the count message its rank sends its rail, by name",
      py::arg("node_keys"), py::arg("node_key_weights"), py::arg("node_bucket_feature_of_row"),
      py::arg("remote_offsets"), py::arg("skip"), py::arg("tables"), py::arg("need_remote"),
      py::arg("own_lengths"), py::arg("remote_map"), py::arg("key_stage"), py::arg("proxy_header"),
      py::arg("node_dedup"), py::arg("sizes"), py::arg("lookup_groups") = std::vector<int64_t>{});
  py_module.def("hier_output_dist_rail_tables", &rail_tables_test,
                "One rank's rail layout on its rail's count messages to it, and given its "
                "rail_keys and node_keys its owner tables, by name",
                py::arg("messages"), py::arg("own_lengths"), py::arg("own_offsets"),
                py::arg("rail_keys") = py::none(), py::arg("node_keys") = py::none(),
                py::arg("node_key_weights") = py::none(), py::arg("sizes"),
                py::arg("lookup_groups") = std::vector<int64_t>{});
  py_module.def("hier_output_dist_requester_tables", &requester_tables_test,
                "One requester's tables across nodes from its node's proxies' remote_map and "
                "headers, by name",
                py::arg("remote_maps"), py::arg("proxy_headers"), py::arg("owner_group_shift"),
                py::arg("bucket_feature_of_row"), py::arg("offset_per_bucket"),
                py::arg("row_shift"), py::arg("sizes"),
                py::arg("lookup_groups") = std::vector<int64_t>{});
  py_module.def(
      "hier_output_dist_clock", &clock_stamp,
      "Writes this GPU's %globaltimer (ns) into out[index] on the current stream, from one "
      "thread (benchmark tracing: clock offsets between GPUs); raises in a build without "
      "DYNAMICEMB_HIER_TRACE=1",
      py::arg("out"), py::arg("index"));
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
