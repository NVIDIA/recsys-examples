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
// outside an op, on plain tensors: the bucketize, the node layout, and one
// rank's proxy, rail and requester steps across nodes. Last, a hook that makes
// the next construction fail, and the GPU clock the benchmark aligns timelines
// by (a build with DYNAMICEMB_HIER_TRACE=1).

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
#include "hier_output_dist/input_dist/planner.h"

namespace dyn_emb {
namespace hier_output_dist {

namespace py = pybind11;

namespace {

// The bucketize of a KJT (bucketize_keys), by name, with each owner's feature
// offsets relative to its block (feature_start [world_size][num_features + 1]),
// which TorchRec's bucketize gives. dist_types are the rules' codes on the
// host; their copy to the GPU does not wait for the device (the tests run the
// planner under the sync debug mode "error").
std::map<std::string, at::Tensor> bucketize_test(
    at::Tensor values, c10::optional<at::Tensor> offsets, c10::optional<at::Tensor> lengths,
    int64_t batch_size, c10::optional<at::Tensor> weights, at::Tensor block_sizes,
    std::vector<int> dist_types, int64_t world_size, at::Tensor input_offsets) {
  c10::cuda::CUDAGuard guard(values.device());
  PlanInput input;
  input.values = values;
  input.offsets = offsets;
  input.lengths = lengths;
  input.batch_size = batch_size;
  input.weights = weights;
  input.input_offsets = input_offsets;
  const OwnerRules rules{at::tensor(dist_types, at::kInt)
                             .pin_memory()
                             .to(values.options().dtype(at::kInt), /*non_blocking=*/true),
                         block_sizes, dist_types};
  const BucketizedKeys bucketized = bucketize_keys(input, rules, world_size);
  const int64_t num_features = (int64_t)dist_types.size();
  std::map<std::string, at::Tensor> named{
      {"keys", bucketized.keys},
      {"inv", bucketized.inv},
      {"bucket_feature_of_row", bucketized.bucket_feature_of_row},
      {"offset_per_bucket_feature", bucketized.offset_per_bucket_feature},
      {"offset_per_bucket", bucketized.offset_per_bucket},
      {"feature_start",
       (bucketized.offset_per_bucket_feature.unfold(0, num_features + 1, num_features) -
        bucketized.offset_per_bucket.narrow(0, 0, world_size).unsqueeze(1))
           .reshape({-1})},
      {"input_length_per_key", bucketized.input_length_per_key}};
  if (bucketized.key_weights.defined()) named["weights"] = bucketized.key_weights;
  return named;
}

// Index dedup's fan-out (fanout_tables), by name.
std::map<std::string, at::Tensor> fanout_test(at::Tensor reverse_indices, at::Tensor inv) {
  c10::cuda::CUDAGuard guard(inv.device());
  const FanoutTables fanout = fanout_tables(reverse_indices, inv);
  return {{"exp_off", fanout.exp_off}, {"exp_idx", fanout.exp_idx}, {"slot", fanout.slot},
          {"sub", fanout.sub},         {"count", fanout.count},     {"pos", fanout.pos},
          {"first", fanout.first}};
}

// One rank of a node: its count vector (node_count_vector), and its node
// layout (node_layout) on the node's counts (totals, before, mine) as
// node_count_exchange would give them, by name (requester_start and
// need_remote across nodes). On one node, with the owner's node_keys (and
// node_weights) after every requester pushed, also its KJT and src
// (one_node_owner).
std::map<std::string, at::Tensor> node_layout_test(
    at::Tensor totals, at::Tensor before, at::Tensor mine, at::Tensor offset_per_bucket_feature,
    at::Tensor offset_per_bucket, at::Tensor bucket_feature_of_row, int64_t num_nodes,
    int64_t local_size, int64_t node, int64_t local_rank, int64_t num_features, int64_t lookup_keys,
    int64_t remote_keys, c10::optional<at::Tensor> node_keys,
    c10::optional<at::Tensor> node_weights) {
  c10::cuda::CUDAGuard guard(offset_per_bucket_feature.device());
  const int world_size = (int)(num_nodes * local_size);
  const Topo topo{world_size,
                  (int)local_size,
                  (int)num_nodes,
                  (int)(local_size + num_nodes - 1),
                  (int)(node * local_size + local_rank),
                  (int)local_rank,
                  (int)node};
  const int64_t words = local_size * (num_features + 1);
  TORCH_CHECK(is_cuda_array(totals, at::kLong, words) && is_cuda_array(before, at::kLong, words) &&
                  is_cuda_array(mine, at::kLong, words),
              "hier_output_dist: node_layout's counts");
  const NodeLayout layout = node_layout(
      NodeCounts{totals, before, mine.view({local_size, num_features + 1})},
      offset_per_bucket_feature, offset_per_bucket, topo, num_features, lookup_keys, remote_keys);
  std::map<std::string, at::Tensor> named{
      {"count_vector", node_count_vector(offset_per_bucket_feature, topo, num_features)},
      {"own_lengths", layout.own_lengths},
      {"own_offsets", layout.own_offsets},
      {"own_total", layout.own_total},
      {"need_lookup", layout.need_lookup},
      {"skip", layout.skip},
      {"rows_to_push", layout.rows_to_push},
      {"destination_base", layout.destination_base}};
  if (num_nodes > 1) {
    named["requester_start"] = layout.requester_start;
    named["need_remote"] = layout.need_remote;
  }
  if (node_keys.has_value()) {
    const OneNodeOwner owner =
        one_node_owner(layout, bucket_feature_of_row, *node_keys,
                       node_weights.has_value() ? *node_weights : at::Tensor(), topo, num_features);
    named["kjt_values"] = owner.kjt_values;
    if (owner.kjt_weights.defined()) named["kjt_weights"] = owner.kjt_weights;
    named["kjt_lengths"] = owner.kjt_lengths;
    named["kjt_offsets"] = owner.kjt_offsets;
    named["num_recv"] = owner.num_recv;
    named["length_per_key"] = owner.length_per_key;
    named["src"] = owner.src;
  }
  return named;
}

// The sizes of a call across nodes (Sizes) by name: num_nodes, local_size,
// node, local_rank, num_features, num_tables, lookup_keys, remote_keys,
// num_proxy_rows, chunk_rows, record_bytes; max_chunks follows, and weights
// with record_bytes. No step of the planner reads the others (row_bytes),
// which stay 0.
Sizes plan_sizes(const std::map<std::string, int64_t> &named) {
  const int num_nodes = (int)named.at("num_nodes"), local_size = (int)named.at("local_size"),
            node = (int)named.at("node"), local_rank = (int)named.at("local_rank");
  Sizes sizes{};
  sizes.topo =
      Topo{num_nodes * local_size,         local_size, num_nodes, local_size + num_nodes - 1,
           node * local_size + local_rank, local_rank, node};
  sizes.num_features = named.at("num_features");
  sizes.num_tables = named.at("num_tables");
  sizes.lookup_keys = named.at("lookup_keys");
  sizes.remote_keys = named.at("remote_keys");
  sizes.num_proxy_rows = named.at("num_proxy_rows");
  sizes.chunk_rows = named.at("chunk_rows");
  sizes.max_chunks = max_chunks_for(sizes);
  sizes.record_bytes = named.at("record_bytes");
  sizes.weights = sizes.weighted_records();
  TORCH_CHECK(num_nodes >= 2 && local_size >= 1 && node >= 0 && node < num_nodes &&
                  local_rank >= 0 && local_rank < local_size && sizes.num_features >= 1 &&
                  num_nodes * local_size * sizes.num_features < INT32_MAX &&
                  sizes.num_tables >= 1 && sizes.num_tables <= sizes.num_features &&
                  sizes.lookup_keys >= 1 && sizes.lookup_keys < INT32_MAX &&
                  sizes.remote_keys >= 1 && sizes.remote_keys < INT32_MAX &&
                  sizes.chunk_rows >= 2 * sizes.max_key_rows() &&
                  sizes.num_proxy_rows >= 2 * sizes.chunk_rows &&
                  (sizes.record_bytes == 8 || sizes.record_bytes == 16),
              "hier_output_dist: the sizes of a call across nodes");
  return sizes;
}

// One proxy's steps of a call across nodes (proxy_keys,
// proxy_chunks_and_slots, proxy_push_order) on its remote part (keys,
// weights, bucket_feature_of_occ: what node_keys, node_weights and
// bucket_feature_of_occ hold past lookup_keys; weights exactly with 16-byte
// records) and what the node layout says of it (requester_start, skip,
// need_remote, own_total: its own part), and its count messages
// (count_message), by name. It writes remote_map, key_stage and
// proxy_header (a proxy header, ProxyHeader) as the steps write those
// buffers of its window. Also the node-unique key of each occurrence
// (key_of_occurrence), for the test's reference.
std::map<std::string, at::Tensor> proxy_plan_test(
    at::Tensor keys, c10::optional<at::Tensor> weights, at::Tensor bucket_feature_of_occ,
    at::Tensor requester_start, at::Tensor skip, at::Tensor tables, at::Tensor need_remote,
    at::Tensor own_total, at::Tensor remote_map, at::Tensor key_stage, at::Tensor proxy_header,
    bool node_dedup, std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(keys.device());
  const Sizes sizes = plan_sizes(named_sizes);
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                num_features = sizes.num_features;
  ProxyHeader header = ProxyHeader::of(sizes);  // over proxy_header once checked
  TORCH_CHECK(
      is_cuda_array(keys, at::kLong, entries) && weights.has_value() == sizes.weighted_records() &&
          (!weights.has_value() || is_cuda_array(*weights, at::kFloat, entries)) &&
          is_cuda_array(bucket_feature_of_occ, at::kInt, entries) &&
          is_cuda_array(requester_start, at::kLong, local_size + 1) &&
          is_cuda_array(skip, at::kBool, 1) && is_cuda_array(tables, at::kInt, num_features) &&
          is_cuda_array(need_remote, at::kLong, 1) && is_cuda_array(own_total, at::kLong, 1) &&
          is_cuda_array(remote_map, at::kInt, RemoteMap::kArrays * entries) &&
          is_cuda_array(key_stage, at::kLong,
                        (int64_t)KeysLayout(sizes, nullptr).key_stage_bytes() / 8) &&
          is_cuda_array(proxy_header, at::kInt, header.words()),
      "hier_output_dist: proxy_plan's inputs");
  const ProxyInput input{keys,
                         weights.has_value() ? *weights : at::Tensor(),
                         bucket_feature_of_occ,
                         requester_start,
                         skip.reshape({}),
                         tables};
  header.base = proxy_header.data_ptr<int32_t>();
  const ProxyBuffers buffers = proxy_buffers(remote_map.view({RemoteMap::kArrays, entries}),
                                             key_stage, header, proxy_header.options());
  ProxyKeys proxy = proxy_keys(input, sizes, node_dedup);
  const at::Tensor key_of_occurrence =
      at::empty_like(proxy.key_at).index_copy_(0, proxy.order.to(at::kLong), proxy.key_at);
  ProxyChunks chunks = proxy_chunks_and_slots(proxy, requester_start, buffers, sizes);
  const ProxyPlan plan =
      proxy_push_order(std::move(proxy), std::move(chunks), requester_start, buffers, sizes);
  return {{"num_chunks", plan.num_chunks},
          {"occurrence_count", plan.occurrence_count},
          {"key_rows", plan.key_rows},
          {"uniq_per_bucket_feature", plan.uniq_per_bucket_feature},
          {"uniq_off", plan.uniq_off},
          {"uniq_out", plan.uniq_out},
          {"chunks", plan.chunks},
          {"chunks_out", plan.chunks_out},
          {"node_chunk_start", plan.node_chunk_start},
          {"node_chunks", plan.node_chunks},
          {"stage_slice", plan.stage_slice},
          {"reduce_task_starts", plan.reduce_task_starts},
          {"reducer_start", plan.reducer_start},
          {"reducer_flags", plan.reducer_flags},
          {"message", count_message(need_remote.reshape({}), own_total.reshape({}), plan.key_rows,
                                    plan.uniq_out, plan.chunks_out, plan.uniq_per_bucket_feature)},
          {"key_of_occurrence", key_of_occurrence}};
}

// One rank's rail steps of a call on the count messages it got (int64
// [num_nodes][CountMessage::length], as rail_count_exchange gives them), by
// name: the rail layout (rail_layout) and what the rank publishes to its node
// (status_publish). Given its rail_keys (int64) and the own part of its
// node_keys (and node_weights) once the rail's proxies and its node's
// requesters put their keys, and where its keys of each feature start at its
// rail's owners (owner_offsets, int64 [num_nodes][num_features], as the owners
// put their source_offsets rows), also its owner tables (owner_tables) and the
// put lists (put_lists, on its chunks, node_chunk_start, node_chunks, uniq_off
// and uniq_per_bucket_feature as proxy: proxy_plan's).
std::map<std::string, at::Tensor> rail_tables_test(
    at::Tensor messages, at::Tensor own_lengths, at::Tensor own_offsets, at::Tensor chunks,
    at::Tensor node_chunk_start, at::Tensor node_chunks, at::Tensor uniq_off,
    at::Tensor uniq_per_bucket_feature, c10::optional<at::Tensor> owner_offsets,
    c10::optional<at::Tensor> rail_keys, c10::optional<at::Tensor> node_keys,
    c10::optional<at::Tensor> node_weights, std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(messages.device());
  const Sizes sizes = plan_sizes(named_sizes);
  const int64_t local_size = sizes.topo.local_size, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, max_chunks = sizes.max_chunks,
                lookup_keys = sizes.lookup_keys;
  const int64_t own_columns = num_features * local_size;
  TORCH_CHECK(
      is_cuda_array(messages, at::kLong,
                    num_nodes * CountMessage{num_nodes, num_features}.length()) &&
          is_cuda_array(own_lengths, at::kLong, own_columns) &&
          is_cuda_array(own_offsets, at::kLong, own_columns) &&
          is_cuda_array(chunks, at::kInt, max_chunks * 8) &&
          is_cuda_array(node_chunk_start, at::kLong, num_nodes + 1) &&
          is_cuda_array(node_chunks, at::kInt, max_chunks) &&
          is_cuda_array(uniq_off, at::kLong, num_nodes + 1) &&
          is_cuda_array(uniq_per_bucket_feature, at::kLong, num_nodes * num_features) &&
          (!owner_offsets.has_value() ||
           is_cuda_array(*owner_offsets, at::kLong, num_nodes * num_features)) &&
          (!rail_keys.has_value() ||
           (is_cuda_array(*rail_keys, at::kLong,
                          (int64_t)KeysLayout(sizes, nullptr).rail_keys_bytes() / 8) &&
            node_keys.has_value() && is_cuda_array(*node_keys, at::kLong, lookup_keys))) &&
          (!node_weights.has_value() || is_cuda_array(*node_weights, at::kFloat, lookup_keys)),
      "hier_output_dist: rail_tables' inputs");
  const at::Tensor other_nodes = other_nodes_of(sizes.topo, messages.options());
  const RailLayout rail =
      rail_layout(messages.view({num_nodes, -1}), own_lengths, other_nodes, sizes);
  std::map<std::string, at::Tensor> named{
      {"put_dst", rail.put_dst},
      {"put_bytes", rail.put_bytes},
      {"put_skip", rail.put_skip},
      {"kjt_lengths", rail.kjt_lengths},
      {"kjt_offsets", rail.kjt_offsets},
      {"num_recv", rail.num_recv},
      {"length_per_key", rail.length_per_key},
      {"slice_in", rail.slice_in},
      {"uniq_in", rail.uniq_in},
      {"length_per_source_feature", rail.length_per_source_feature},
      {"source_offsets", rail.source_offsets},
      {"requester_offsets", rail.requester_offsets},
      {"needs", rail.needs},
      {"publish", status_publish(rail)}};
  if (rail_keys.has_value()) {
    TORCH_CHECK(owner_offsets.has_value(),
                "hier_output_dist: rail_tables' put lists need owner_offsets");
    const OwnerTables owner = owner_tables(rail, own_offsets, *node_keys,
                                           node_weights.has_value() ? *node_weights : at::Tensor(),
                                           *rail_keys, other_nodes, sizes);
    const PutLists lists = put_lists(
        chunks.view({max_chunks, 8}), node_chunk_start, node_chunks, uniq_off,
        uniq_per_bucket_feature, owner_offsets->view({num_nodes, num_features}), owner.fwd_chunks,
        owner.fwd_chunk_start, rail.length_per_source_feature, rail.source_offsets, sizes);
    named["kjt_values"] = owner.kjt_values;
    if (owner.kjt_weights.defined()) named["kjt_weights"] = owner.kjt_weights;
    named["fwd_chunks"] = owner.fwd_chunks;
    named["fwd_chunk_start"] = owner.fwd_chunk_start;
    named["put_list"] = lists.list;
    named["put_starts"] = lists.starts;
  }
  return named;
}

// One requester's tables of a call across nodes (requester_tables), by name,
// from what its node's proxies published (remote_maps: int32 [local_size][3]
// [remote_keys], RemoteMap; proxy_headers: int32 [local_size]
// [ProxyHeader::words]) and where its node's owners put its keys of each
// feature (offset_per_owner_feature, as node_status_exchange gives it). The two
// gathers read them as peer_gather would, on the word lists of remote_rows and
// header_gather; an item a gather skips holds -1.
std::map<std::string, at::Tensor> requester_tables_test(
    at::Tensor remote_maps, at::Tensor proxy_headers, at::Tensor offset_per_owner_feature,
    at::Tensor bucket_feature_of_row, at::Tensor offset_per_bucket_feature,
    at::Tensor offset_per_bucket, at::Tensor destination_base,
    std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(bucket_feature_of_row.device());
  const Sizes sizes = plan_sizes(named_sizes);
  const int64_t local_size = sizes.topo.local_size, num_features = sizes.num_features,
                bucket_features = sizes.topo.world_size * num_features;
  const ProxyHeader header = ProxyHeader::of(sizes);  // its words
  TORCH_CHECK(
      is_cuda_array(remote_maps, at::kInt, local_size * RemoteMap::kArrays * sizes.remote_keys) &&
          is_cuda_array(proxy_headers, at::kInt, local_size * header.words()) &&
          is_cuda_array(offset_per_owner_feature, at::kLong, local_size * num_features) &&
          bucket_feature_of_row.is_cuda() && bucket_feature_of_row.is_contiguous() &&
          bucket_feature_of_row.scalar_type() == at::kInt &&
          is_cuda_array(offset_per_bucket_feature, at::kLong, bucket_features + 1) &&
          is_cuda_array(offset_per_bucket, at::kLong, sizes.topo.world_size + 1) &&
          is_cuda_array(destination_base, at::kLong, bucket_features + 1),
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
             remote_rows(bucket_feature_of_row, offset_per_bucket, destination_base, sizes),
             RemoteMap::kArrays, sizes.remote_keys);
  const at::Tensor headers = gather(proxy_headers.view({local_size, -1}),
                                    header_gather(sizes, bucket_feature_of_row.options()), 1, 0)
                                 .view({local_size, header.gathered_words()});
  const RequesterTables tables =
      requester_tables(words, headers, offset_per_owner_feature.view({local_size, num_features}),
                       bucket_feature_of_row, offset_per_bucket_feature, offset_per_bucket, sizes);
  return {{"src", tables.src},
          {"dst", tables.dst},
          {"push_order", tables.push_order},
          {"push_order_off", tables.push_order_off},
          {"push_tables", tables.push_tables},
          {"proxy_chunks", tables.proxy_chunks},
          {"proxy_ends", tables.proxy_ends},
          {"task_starts", tables.task_starts}};
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
                "The keys bucketized by owner, their bucket features and offsets, and the input's "
                "per-feature totals, by name",
                py::arg("values"), py::arg("offsets"), py::arg("lengths"), py::arg("batch_size"),
                py::arg("weights"), py::arg("block_sizes"), py::arg("dist_types"),
                py::arg("world_size"), py::arg("input_offsets"));
  py_module.def("hier_output_dist_fanout", &fanout_test,
                "Index dedup's fan-out of the bucketized rows, by name", py::arg("reverse_indices"),
                py::arg("inv"));
  py_module.def("hier_output_dist_node_layout", &node_layout_test,
                "One rank's count vector and node layout on its node's counts, and on one node "
                "with its node_keys its KJT and src, by name",
                py::arg("totals"), py::arg("before"), py::arg("mine"),
                py::arg("offset_per_bucket_feature"), py::arg("offset_per_bucket"),
                py::arg("bucket_feature_of_row"), py::arg("num_nodes"), py::arg("local_size"),
                py::arg("node"), py::arg("local_rank"), py::arg("num_features"),
                py::arg("lookup_keys"), py::arg("remote_keys"), py::arg("node_keys") = py::none(),
                py::arg("node_weights") = py::none());
  py_module.def("hier_output_dist_proxy_plan", &proxy_plan_test,
                "One proxy's steps of a call across nodes on plain tensors, which it writes as "
                "its buffers of the window, and its count messages, by name",
                py::arg("keys"), py::arg("weights"), py::arg("bucket_feature_of_occ"),
                py::arg("requester_start"), py::arg("skip"), py::arg("tables"),
                py::arg("need_remote"), py::arg("own_total"), py::arg("remote_map"),
                py::arg("key_stage"), py::arg("proxy_header"), py::arg("node_dedup"),
                py::arg("sizes"));
  py_module.def(
      "hier_output_dist_rail_tables", &rail_tables_test,
      "One rank's rail layout and status publish vector on the count messages it got, "
      "and given its rail_keys, node_keys and owner_offsets its owner tables and put lists, "
      "by name",
      py::arg("messages"), py::arg("own_lengths"), py::arg("own_offsets"), py::arg("chunks"),
      py::arg("node_chunk_start"), py::arg("node_chunks"), py::arg("uniq_off"),
      py::arg("uniq_per_bucket_feature"), py::arg("owner_offsets") = py::none(),
      py::arg("rail_keys") = py::none(), py::arg("node_keys") = py::none(),
      py::arg("node_weights") = py::none(), py::arg("sizes"));
  py_module.def("hier_output_dist_requester_tables", &requester_tables_test,
                "One requester's tables across nodes from its node's proxies' remote_map and "
                "headers, by name",
                py::arg("remote_maps"), py::arg("proxy_headers"),
                py::arg("offset_per_owner_feature"), py::arg("bucket_feature_of_row"),
                py::arg("offset_per_bucket_feature"), py::arg("offset_per_bucket"),
                py::arg("destination_base"), py::arg("sizes"));
  py_module.def(
      "hier_output_dist_clock", &clock_stamp,
      "Writes this GPU's %globaltimer (ns) into out[index] on the current stream, from one "
      "thread (benchmark tracing: clock offsets between GPUs); raises in a build without "
      "DYNAMICEMB_HIER_TRACE=1",
      py::arg("out"), py::arg("index"));
}

}  // namespace hier_output_dist
}  // namespace dyn_emb
