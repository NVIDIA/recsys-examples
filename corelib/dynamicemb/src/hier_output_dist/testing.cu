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
// Test-only bindings of the hierarchical dist. Some run the planner's steps
// outside an op, on plain tensors: part 1, the node layout, and one rank's
// proxy, rail and requester steps across nodes. Others are views into an op
// and its plans that only tests and the benchmark read. Last, a hook that
// makes the next construction fail, and the GPU clock the benchmark aligns
// timelines by (a build with DYNAMICEMB_HIER_TRACE=1).
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "hier_output_dist/common/formats.h"
#include "hier_output_dist/hier_output_dist.h"
#include "hier_output_dist/host/common.h"
#include "hier_output_dist/host/layout.h"
#include "hier_output_dist/kernels.h"
#include "hier_output_dist/plan/node.h"
#include "hier_output_dist/plan/part_one.h"
#include "hier_output_dist/plan/proxy.h"
#include "hier_output_dist/plan/put_list.h"
#include "hier_output_dist/plan/rail.h"
#include "hier_output_dist/plan/requester.h"

namespace py = pybind11;

namespace dyn_emb {
namespace hier_output_dist {
namespace {

// Part 1's partition of a KJT (partition_keys), by name, with each owner's
// feature offsets relative to its block (feature_start [world_size]
// [num_features + 1]), which the bucketize gives. dist_types are the rules'
// codes on the host; their copy to the GPU does not wait for the device
// (the tests run the planner under the sync debug mode "error").
std::map<std::string, at::Tensor> partition_test(
    at::Tensor values, c10::optional<at::Tensor> offsets, c10::optional<at::Tensor> lengths,
    int64_t batch_size, c10::optional<at::Tensor> weights, at::Tensor block_sizes,
    std::vector<int> dist_types, int64_t world_size, c10::optional<at::Tensor> input_offsets,
    int64_t input_batch_size) {
  c10::cuda::CUDAGuard guard(values.device());
  PlanInput input;
  input.values = values;
  input.offsets = offsets;
  input.lengths = lengths;
  input.batch_size = batch_size;
  input.weights = weights;
  input.input_offsets = input_offsets;
  input.input_batch_size = input_batch_size;
  const OwnerRules rules{at::tensor(dist_types, at::kInt)
                             .pin_memory()
                             .to(values.options().dtype(at::kInt), /*non_blocking=*/true),
                         block_sizes, dist_types};
  const Partition part = partition_keys(input, rules, world_size);
  const int64_t num_features = (int64_t)dist_types.size();
  std::map<std::string, at::Tensor> named{
      {"keys", part.keys},
      {"inv", part.inv},
      {"row_cell", part.row_cell},
      {"cell_start", part.cell_start},
      {"block_start", part.block_start},
      {"feature_start", (part.cell_start.unfold(0, num_features + 1, num_features) -
                         part.block_start.narrow(0, 0, world_size).unsqueeze(1))
                            .reshape({-1})},
      {"input_length_per_key", part.input_length_per_key}};
  if (part.key_weights.defined()) named["weights"] = part.key_weights;
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
// layout (node_layout) on the given node_counts, as node_count_exchange
// would give them, by name (requester_start and need_remote across nodes).
// On one node, with the owner's node_keys (and node_weights) after every
// requester pushed, also its KJT and src (one_node_owner).
std::map<std::string, at::Tensor> node_layout_test(
    at::Tensor node_counts, at::Tensor cell_start, at::Tensor block_start, at::Tensor row_cell,
    int64_t num_nodes, int64_t local_size, int64_t node, int64_t local_rank, int64_t num_features,
    int64_t lookup_keys, int64_t remote_keys, c10::optional<at::Tensor> node_keys,
    c10::optional<at::Tensor> node_weights) {
  c10::cuda::CUDAGuard guard(cell_start.device());
  const int world_size = (int)(num_nodes * local_size);
  const Topo topo{world_size,
                  (int)local_size,
                  (int)num_nodes,
                  (int)(local_size + num_nodes - 1),
                  (int)(node * local_size + local_rank),
                  (int)local_rank,
                  (int)node};
  const NodeLayout layout = node_layout(node_counts, cell_start, block_start, topo, num_features,
                                        lookup_keys, remote_keys);
  std::map<std::string, at::Tensor> named{
      {"count_vector", node_count_vector(cell_start, topo, num_features)},
      {"own_cells", layout.own_cells},
      {"own_starts", layout.own_starts},
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
        one_node_owner(layout, row_cell, *node_keys,
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

// The offsets of the small tables of an op's calls (Meta), by name.
std::map<std::string, int64_t> op_meta_offsets(const HierOutputDist &op) {
  const Layout layout(op.config());
  std::map<std::string, int64_t> offsets;
  for (const auto &field : meta_fields(layout.meta_at)) offsets[field.first] = field.second;
  return offsets;
}

// The sizes of a call across nodes (PlanSizes) by name: num_nodes,
// local_size, node, local_rank, num_features, lookup_keys, remote_keys,
// num_proxy_rows, partial_rows, chunk_rows, record_bytes; rounds_cap follows.
PlanSizes plan_sizes(const std::map<std::string, int64_t> &named) {
  const int num_nodes = (int)named.at("num_nodes"), local_size = (int)named.at("local_size"),
            node = (int)named.at("node"), local_rank = (int)named.at("local_rank");
  PlanSizes sizes;
  sizes.topo =
      Topo{num_nodes * local_size,         local_size, num_nodes, local_size + num_nodes - 1,
           node * local_size + local_rank, local_rank, node};
  sizes.num_features = named.at("num_features");
  sizes.lookup_keys = named.at("lookup_keys");
  sizes.remote_keys = named.at("remote_keys");
  sizes.num_proxy_rows = named.at("num_proxy_rows");
  sizes.rounds_cap = rounds_for(sizes.remote_keys, sizes.num_proxy_rows);
  sizes.partial_rows = named.at("partial_rows");
  sizes.chunk_rows = named.at("chunk_rows");
  sizes.record_bytes = named.at("record_bytes");
  TORCH_CHECK(
      num_nodes >= 2 && num_nodes <= kMaxNodes && local_size >= 1 && local_size <= kMaxLocalRanks &&
          node >= 0 && node < num_nodes && local_rank >= 0 && local_rank < local_size &&
          sizes.num_features >= 1 && sizes.num_features <= kMaxFeatures && sizes.lookup_keys >= 1 &&
          sizes.lookup_keys < INT32_MAX && sizes.remote_keys >= 1 &&
          sizes.remote_keys < INT32_MAX && sizes.num_proxy_rows >= min_proxy_rows_for(num_nodes) &&
          sizes.partial_rows >= 1 && sizes.chunk_rows % kTaskRows == 0 && sizes.chunk_rows > 0 &&
          (sizes.record_bytes == 8 || sizes.record_bytes == 16),
      "hier_output_dist: the sizes of a call across nodes");
  return sizes;
}

// One proxy's steps of a call across nodes (proxy_keys,
// proxy_rounds_and_slots, proxy_push_order) on its remote part (keys,
// weights, cells: what node_keys, node_weights and remote_cells hold past
// lookup_keys) and what the node layout says of it (requester_start, skip,
// need_remote, own_cells), and its count message (count_message), by name.
// It writes remote_map, key_stage and proxy_hdr as the steps write its
// windows. Also the node-unique key of each occurrence (key_of_occurrence),
// for the test's reference.
std::map<std::string, at::Tensor> proxy_plan_test(
    at::Tensor keys, c10::optional<at::Tensor> weights, at::Tensor cells,
    at::Tensor requester_start, at::Tensor skip, at::Tensor tables, at::Tensor need_remote,
    at::Tensor own_cells, at::Tensor remote_map, at::Tensor key_stage, at::Tensor proxy_hdr,
    bool node_dedup, std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(keys.device());
  const PlanSizes sizes = plan_sizes(named_sizes);
  const int64_t entries = sizes.remote_keys, local_size = sizes.topo.local_size,
                num_nodes = sizes.topo.num_nodes, num_features = sizes.num_features,
                rounds_cap = sizes.rounds_cap;
  TORCH_CHECK(
      is_cuda_array(keys, at::kLong, entries) &&
          (!weights.has_value() || is_cuda_array(*weights, at::kFloat, entries)) &&
          is_cuda_array(cells, at::kShort, entries) &&
          is_cuda_array(requester_start, at::kLong, local_size + 1) &&
          is_cuda_array(skip, at::kBool, 1) && is_cuda_array(tables, at::kInt, num_features) &&
          is_cuda_array(need_remote, at::kLong, 1) &&
          is_cuda_array(own_cells, at::kLong, local_size * num_features * local_size) &&
          is_cuda_array(remote_map, at::kInt, 3 * entries) &&
          is_cuda_array(key_stage, at::kLong,
                        (num_nodes - 1) * rounds_cap * 2 + entries * sizes.record_bytes / 8 + 2) &&
          is_cuda_array(proxy_hdr, at::kInt,
                        4 + 4 * rounds_cap * num_nodes + 4 * local_size * (rounds_cap + 1)),
      "hier_output_dist: proxy_plan's inputs");
  const ProxyInput input{keys,
                         weights.has_value() ? *weights : at::Tensor(),
                         cells,
                         requester_start,
                         skip.reshape({}),
                         tables};
  const ProxyWindows windows{remote_map.view({3, entries}), key_stage, proxy_hdr};
  ProxyKeys proxy = proxy_keys(input, sizes, node_dedup);
  const at::Tensor key_of_occurrence =
      at::empty_like(proxy.key_at).index_copy_(0, proxy.order.to(at::kLong), proxy.key_at);
  ProxyRounds rounds = proxy_rounds_and_slots(proxy, requester_start, windows, sizes);
  const ProxyPlan plan =
      proxy_push_order(std::move(proxy), std::move(rounds), requester_start, windows, sizes);
  return {{"num_rounds", plan.num_rounds},
          {"occurrence_count", plan.occurrence_count},
          {"cell_uniq", plan.cell_uniq},
          {"uniq_off", plan.uniq_off},
          {"uniq_out", plan.uniq_out},
          {"round_table", plan.round_table},
          {"stage_slice", plan.stage_slice},
          {"push_start", plan.push_start},
          {"reducer_start", plan.reducer_start},
          {"reducer_entries", plan.reducer_entries},
          {"message", count_message(need_remote.reshape({}),
                                    own_cells.view({local_size, num_features * local_size}),
                                    plan.num_rounds, plan.cell_uniq, sizes)},
          {"key_of_occurrence", key_of_occurrence}};
}

// One rank's rail steps of a call on its rail's count messages (int64
// [num_nodes][message], as rail_count_exchange gives them), by name: the
// rail layout (rail_layout), what the rank publishes to its node
// (status_publish), and as proxy its backward put list (proxy_put_list, on
// its round_table and cell_uniq). Given its rail_keys (int64) and the own part
// of its node_keys (and node_weights) once the rail's proxies and its node's
// requesters put their keys, also its owner tables (owner_tables).
std::map<std::string, at::Tensor> rail_tables_test(
    at::Tensor messages, at::Tensor own_cells, at::Tensor own_starts, at::Tensor round_table,
    at::Tensor cell_uniq, c10::optional<at::Tensor> rail_keys, c10::optional<at::Tensor> node_keys,
    c10::optional<at::Tensor> node_weights, std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(messages.device());
  const PlanSizes sizes = plan_sizes(named_sizes);
  const int64_t local_size = sizes.topo.local_size, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, rounds_cap = sizes.rounds_cap,
                lookup_keys = sizes.lookup_keys;
  const int64_t own_columns = num_features * local_size;
  TORCH_CHECK(
      is_cuda_array(messages, at::kLong, num_nodes * (2 + (num_nodes + 1) * num_features)) &&
          is_cuda_array(own_cells, at::kLong, local_size * own_columns) &&
          is_cuda_array(own_starts, at::kLong, local_size * own_columns) &&
          is_cuda_array(round_table, at::kInt, rounds_cap * num_nodes * 4) &&
          is_cuda_array(cell_uniq, at::kLong, num_nodes * num_features) &&
          (!rail_keys.has_value() ||
           (is_cuda_array(
                *rail_keys, at::kLong,
                (lookup_keys * sizes.record_bytes + (num_nodes - 1) * rounds_cap * 16) / 8) &&
            node_keys.has_value() && is_cuda_array(*node_keys, at::kLong, lookup_keys))) &&
          (!node_weights.has_value() || is_cuda_array(*node_weights, at::kFloat, lookup_keys)),
      "hier_output_dist: rail_tables' inputs");
  const at::Tensor other_nodes = other_nodes_of(sizes.topo, messages.options());
  const RailLayout rail =
      rail_layout(messages.view({num_nodes, -1}), own_cells.view({local_size, own_columns}),
                  other_nodes, sizes);
  const PutList list = proxy_put_list(round_table.view({rounds_cap, num_nodes, 4}), cell_uniq,
                                      rail.rail_cell_start, sizes);
  std::map<std::string, at::Tensor> named{{"put_dst", rail.put_dst},
                                          {"put_bytes", rail.put_bytes},
                                          {"put_skip", rail.put_skip},
                                          {"rail_cell_start", rail.rail_cell_start},
                                          {"kjt_lengths", rail.kjt_lengths},
                                          {"kjt_offsets", rail.kjt_offsets},
                                          {"num_recv", rail.num_recv},
                                          {"length_per_key", rail.length_per_key},
                                          {"slice_in", rail.slice_in},
                                          {"uniq_in", rail.uniq_in},
                                          {"my_cell_starts", rail.my_cell_starts},
                                          {"needs", rail.needs},
                                          {"publish", status_publish(rail)},
                                          {"bwd_put_list", list.bwd_put_list},
                                          {"bwd_put_start", list.bwd_put_start},
                                          {"round_table", list.round_table},
                                          {"reduce_task_starts", list.reduce_task_starts},
                                          {"num_chunks", list.num_chunks}};
  if (rail_keys.has_value()) {
    const OwnerTables owner = owner_tables(
        rail, own_starts.view({local_size, own_columns}), *node_keys,
        node_weights.has_value() ? *node_weights : at::Tensor(), *rail_keys, other_nodes, sizes);
    named["fwd_put_list"] = owner.fwd_put_list;
    named["kjt_values"] = owner.kjt_values;
    if (owner.kjt_weights.defined()) named["kjt_weights"] = owner.kjt_weights;
  }
  return named;
}

// One requester's tables of a call across nodes (requester_tables), by name,
// from what its node's proxies published (remote_maps: int32 [local_size][3]
// [remote_keys]; proxy_hdrs: int32 [local_size][proxy header words, proxy.h])
// and where its node's owners put its cells (owner_cell_start, as
// node_status_exchange gives it). The two gathers read them as peer_gather
// would, on the word lists of remote_rows and header_gather; an item a
// gather skips holds -1.
std::map<std::string, at::Tensor> requester_tables_test(
    at::Tensor remote_maps, at::Tensor proxy_hdrs, at::Tensor owner_cell_start, at::Tensor row_cell,
    at::Tensor cell_start, at::Tensor block_start, at::Tensor destination_base,
    std::map<std::string, int64_t> named_sizes) {
  c10::cuda::CUDAGuard guard(row_cell.device());
  const PlanSizes sizes = plan_sizes(named_sizes);
  const int64_t local_size = sizes.topo.local_size, num_nodes = sizes.topo.num_nodes,
                num_features = sizes.num_features, rounds_cap = sizes.rounds_cap,
                cells = sizes.topo.world_size * num_features;
  const int64_t proxy_hdr_words =
      4 + 4 * rounds_cap * num_nodes + 4 * local_size * (rounds_cap + 1);
  TORCH_CHECK(is_cuda_array(remote_maps, at::kInt, local_size * 3 * sizes.remote_keys) &&
                  is_cuda_array(proxy_hdrs, at::kInt, local_size * proxy_hdr_words) &&
                  is_cuda_array(owner_cell_start, at::kLong, local_size * num_features) &&
                  row_cell.is_cuda() && row_cell.is_contiguous() &&
                  row_cell.scalar_type() == at::kInt &&
                  is_cuda_array(cell_start, at::kLong, cells + 1) &&
                  is_cuda_array(block_start, at::kLong, sizes.topo.world_size + 1) &&
                  is_cuda_array(destination_base, at::kLong, cells + 1),
              "hier_output_dist: requester_tables' inputs");
  // peer_gather: out[k][item] = word first_word[item] + k * word_stride of
  // its peer's region
  auto gather = [](const at::Tensor &regions, const PeerWords &items, int64_t words,
                   int64_t word_stride) {
    const at::Tensor first =
        items.peer.clamp_min(0).to(at::kLong) * regions.size(1) + items.first_word;
    const at::Tensor word =
        at::arange(words, first.options()).view({words, 1}) * word_stride + first.view({1, -1});
    return at::where(items.peer.ge(0), regions.view({-1}).take(word), -1);
  };
  const at::Tensor words =
      gather(remote_maps.view({local_size, -1}),
             remote_rows(row_cell, block_start, destination_base, sizes), 3, sizes.remote_keys);
  const at::Tensor headers =
      gather(proxy_hdrs.view({local_size, -1}), header_gather(sizes, row_cell.options()), 1, 0)
          .view({local_size, header_words(sizes)});
  const RequesterTables tables =
      requester_tables(words, headers, owner_cell_start.view({local_size, num_features}), row_cell,
                       cell_start, block_start, sizes);
  return {{"src", tables.src},
          {"dst", tables.dst},
          {"push_order", tables.push_order},
          {"push_order_off", tables.push_order_off},
          {"push_tables", tables.push_tables},
          {"proxy_rounds", tables.proxy_rounds},
          {"proxy_round_tables", tables.proxy_round_tables},
          {"fwd_task_starts", tables.fwd_task_starts},
          {"bwd_task_starts", tables.bwd_task_starts},
          {"heavy_groups", tables.heavy_groups},
          {"group_first_sub", tables.group_first_sub},
          {"group_of_sub", tables.group_of_sub},
          {"group_subs", tables.group_subs}};
}

}  // namespace

void bind_testing(py::module &py_module) {
  py_module.def("hier_output_dist_plan_tensors", &Plan::tensors,
                "The device tensors a plan holds, by name", py::arg("plan"));
  py_module.def("hier_output_dist_op_meta_offsets", &op_meta_offsets,
                "Offsets (int64 words) of the small tables of an op's calls, by Meta field",
                py::arg("op"));
  py_module.def("hier_output_dist_live_views", &HierOutputDist::live_views,
                "Views of an op's lookup_rows alive", py::arg("op"));
  py_module.def("hier_output_dist_lookup_grad_live", &HierOutputDist::lookup_grad_live,
                "Gradient views of an op's lookup_rows alive (those its backward handed out)",
                py::arg("op"));
  py_module.def("hier_output_dist_fail_next_construction", &fail_next_construction,
                "The next construction of an op in this process fails to allocate",
                py::arg("fail_alloc"));
  py_module.def("hier_output_dist_partition", &partition_test,
                "Part 1's keys partitioned by owner, their cells and offsets, and the input's "
                "per-feature totals, by name",
                py::arg("values"), py::arg("offsets"), py::arg("lengths"), py::arg("batch_size"),
                py::arg("weights"), py::arg("block_sizes"), py::arg("dist_types"),
                py::arg("world_size"), py::arg("input_offsets"), py::arg("input_batch_size"));
  py_module.def("hier_output_dist_fanout", &fanout_test, "Index dedup's fan-out of part 1, by name",
                py::arg("reverse_indices"), py::arg("inv"));
  py_module.def("hier_output_dist_node_layout", &node_layout_test,
                "One rank's count vector and node layout on given node counts, and on one node "
                "with its node_keys its KJT and src, by name",
                py::arg("node_counts"), py::arg("cell_start"), py::arg("block_start"),
                py::arg("row_cell"), py::arg("num_nodes"), py::arg("local_size"), py::arg("node"),
                py::arg("local_rank"), py::arg("num_features"), py::arg("lookup_keys"),
                py::arg("remote_keys"), py::arg("node_keys") = py::none(),
                py::arg("node_weights") = py::none());
  py_module.def("hier_output_dist_proxy_plan", &proxy_plan_test,
                "One proxy's steps of a call across nodes on plain tensors, which it writes as "
                "its windows, and its count message, by name",
                py::arg("keys"), py::arg("weights"), py::arg("cells"), py::arg("requester_start"),
                py::arg("skip"), py::arg("tables"), py::arg("need_remote"), py::arg("own_cells"),
                py::arg("remote_map"), py::arg("key_stage"), py::arg("proxy_hdr"),
                py::arg("node_dedup"), py::arg("sizes"));
  py_module.def("hier_output_dist_rail_tables", &rail_tables_test,
                "One rank's rail layout, status publish vector and backward put list on its "
                "rail's count messages, and given its rail_keys and node_keys its owner tables, "
                "by name",
                py::arg("messages"), py::arg("own_cells"), py::arg("own_starts"),
                py::arg("round_table"), py::arg("cell_uniq"), py::arg("rail_keys") = py::none(),
                py::arg("node_keys") = py::none(), py::arg("node_weights") = py::none(),
                py::arg("sizes"));
  py_module.def("hier_output_dist_requester_tables", &requester_tables_test,
                "One requester's tables across nodes from its node's proxies' remote_map and "
                "headers, by name",
                py::arg("remote_maps"), py::arg("proxy_hdrs"), py::arg("owner_cell_start"),
                py::arg("row_cell"), py::arg("cell_start"), py::arg("block_start"),
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
