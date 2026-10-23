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

// ---------------------------------------------------------------------------
// Python binding for the cache's placement: which key is in which node, who is
// holding it, and which node an incoming key takes.
//
// The class owns each table's bucket chains, node array and reclaim queue, and
// nothing else. The rows themselves belong to the caller — one tensor per table,
// addressed by node id — and so does everything about where they come from and go
// to. What this exposes is what the per-step orchestration needs, in the order it
// needs it: see docs/lightweight_cache_design.md, 6.5 and 17.
//
// Per-table kernels are launched table by table over that table's slice of the
// input, which the caller describes with offsets — the position in the keys array
// at which each table starts, with the total at the end. The kernels that take a
// table id per key are launched once over everything.
//
// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------
//
// Every call runs on the caller's current stream, which is how the pipeline puts
// allocate on the prefetch stream and release on the backward one. The reclaim
// queue needs the two ordered around the tail snapshot (reclaim_queue.h): an
// allocate waits for every release issued before it, and a release waits for the
// snapshot of every allocate issued before it. Two events do that, one recorded
// after each release and one after each snapshot, and each call waits on the
// other's before it launches. With one stream they cost nothing.
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "../torch_utils.h"
#include "kernels.cuh"
#include "types.h"

namespace py = pybind11;

namespace embedding_cache {
namespace {

using MapStore =
    HashMapStorage<Node, Murmur3Hash64, static_cast<int32_t>(-1), TorchAllocator>;
using Map = MapStore::Map;
using QStore = ReclaimQueueStorage<int32_t, static_cast<int32_t>(-1), TorchAllocator>;
using Queue = QStore::Queue;
using Slot = AllocSlot<int64_t, int32_t>;

constexpr int32_t kNone = Map::kNoNode;

inline cudaStream_t current_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

inline void check_cuda_tensor(const at::Tensor& t, const char* name,
                              at::ScalarType type) {
  TORCH_CHECK(t.defined() && t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
}

class EmbeddingCacheCore {
 public:
  explicit EmbeddingCacheCore(std::vector<int64_t> capacities)
      : capacities_(std::move(capacities)) {
    TORCH_CHECK(!capacities_.empty(), "at least one table is needed");
    device_ = at::cuda::current_device();
    const c10::cuda::CUDAGuard guard(device_);
    const cudaStream_t stream = current_stream();
    const TorchAllocator alloc(device_, stream);

    for (const int64_t c : capacities_) {
      TORCH_CHECK(c > 0, "every table needs a positive capacity");
      maps_.emplace_back(new MapStore(c, 0, alloc, stream));
      queues_.emplace_back(new QStore(c, alloc, stream));
      init_nodes_kernel<Node><<<grid_for(c), kBlock, 0, stream>>>(
          maps_.back()->nodes(), c);
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    }
    upload_views();

    DEMB_CUDA_CHECK(
        cudaEventCreateWithFlags(&release_done_, cudaEventDisableTiming));
    DEMB_CUDA_CHECK(
        cudaEventCreateWithFlags(&snapshot_done_, cudaEventDisableTiming));
  }

  ~EmbeddingCacheCore() {
    cudaEventDestroy(release_done_);
    cudaEventDestroy(snapshot_done_);
  }

  EmbeddingCacheCore(const EmbeddingCacheCore&) = delete;
  EmbeddingCacheCore& operator=(const EmbeddingCacheCore&) = delete;

  int64_t num_tables() const { return static_cast<int64_t>(maps_.size()); }

  // The eviction guard, in the form the backing's insert takes: a device pointer
  // to the per-table maps, and how many there are. It stays valid as long as this
  // object does.
  std::pair<int64_t, int64_t> guard() const {
    return {reinterpret_cast<int64_t>(d_views_.data_ptr()), num_tables()};
  }
  int64_t capacity(int64_t table) const { return capacities_.at(table); }

  // Every table back to a fresh, empty cache. Not for use while a step is in
  // flight.
  void reset() {
    const c10::cuda::CUDAGuard guard(device_);
    DEMB_CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t t = 0; t < maps_.size(); ++t) {
      maps_[t]->clear_buckets();
      maps_[t]->clear_links();
      init_nodes_kernel<Node><<<grid_for(capacities_[t]), kBlock, 0,
                                maps_[t]->stream()>>>(maps_[t]->nodes(),
                                                      capacities_[t]);
      queues_[t]->reset();
    }
    DEMB_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // -- the per-step calls -----------------------------------------------------

  // For each key, the node holding it in its table — pinned, one reference taken —
  // or -1. Keys are unique within a table and grouped by table.
  at::Tensor lookup_pin(const at::Tensor& keys,
                        const std::vector<int64_t>& offsets) {
    check_cuda_tensor(keys, "keys", at::kLong);
    check_offsets(offsets, keys.numel());
    const c10::cuda::CUDAGuard guard(keys.device());
    const cudaStream_t stream = current_stream();
    const int64_t n_total = keys.numel();

    // The kernel compacts misses on the way out. Nothing here uses them — the
    // caller derives its misses from the -1s, after deciding which get a node —
    // but the kernel writes them, so they need somewhere to go.
    const auto opts_i64 = keys.options();
    at::Tensor nodes = at::empty({n_total}, opts_i64.dtype(at::kInt));
    at::Tensor miss_keys = at::empty({n_total}, opts_i64);
    at::Tensor miss_src = at::empty({n_total}, opts_i64);
    at::Tensor miss_count = at::zeros({num_tables()}, opts_i64);

    for (int64_t t = 0; t < num_tables(); ++t) {
      const int64_t b = offsets[t], n = offsets[t + 1] - offsets[t];
      if (n == 0) continue;
      lookup_pin_kernel<Map, Node><<<grid_for(n), kBlock, 0, stream>>>(
          maps_[t]->map(), maps_[t]->nodes(), keys.data_ptr<int64_t>() + b, n,
          nodes.data_ptr<int32_t>() + b, miss_keys.data_ptr<int64_t>() + b,
          miss_src.data_ptr<int64_t>() + b,
          reinterpret_cast<unsigned long long*>(miss_count.data_ptr<int64_t>()) +
              t);
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return nodes;
  }

  // Take a node for each key, evicting an idle one if it has to.
  //
  // `keys` are the ones to bind, grouped by table, and `positions` says where each
  // one's answer goes in `nodes` (the array lookup_pin returned), which is written
  // in place. Returns, per key and in the same order: the key the node held, if
  // it held one; whether it did; and the node it now is. A node of -1 means the
  // queue ran dry — every node of that table is held by a step in flight — which
  // is a capacity error and the caller's to report.
  //
  // The nodes are left LOADING and referenced once. Rows are moved by the caller,
  // and publish() is what makes them hits.
  std::tuple<at::Tensor, at::Tensor, at::Tensor> allocate(
      const at::Tensor& keys, const at::Tensor& positions,
      const std::vector<int64_t>& offsets, at::Tensor nodes) {
    check_cuda_tensor(keys, "keys", at::kLong);
    check_cuda_tensor(positions, "positions", at::kLong);
    check_cuda_tensor(nodes, "nodes", at::kInt);
    TORCH_CHECK(positions.numel() == keys.numel(),
                "positions must say where each key's answer goes");
    check_offsets(offsets, keys.numel());
    const c10::cuda::CUDAGuard guard(keys.device());
    const cudaStream_t stream = current_stream();
    const int64_t m_total = keys.numel();

    const auto opts_i64 = keys.options();
    at::Tensor victim_keys = at::empty({m_total}, opts_i64);
    at::Tensor has_victim = at::empty({m_total}, opts_i64.dtype(at::kByte));
    at::Tensor new_nodes = at::empty({m_total}, opts_i64.dtype(at::kInt));
    if (m_total == 0) return {victim_keys, has_victim, new_nodes};

    std::vector<int64_t> counts(num_tables());
    for (int64_t t = 0; t < num_tables(); ++t) counts[t] = offsets[t + 1] - offsets[t];
    const at::Tensor d_counts =
        at::tensor(counts, at::TensorOptions().dtype(at::kLong)).to(keys.device());
    at::Tensor slots = at::empty(
        {m_total * static_cast<int64_t>(sizeof(Slot))}, opts_i64.dtype(at::kByte));

    // The phase boundary between pushing and popping the queue: see the header.
    DEMB_CUDA_CHECK(cudaStreamWaitEvent(stream, release_done_, 0));
    for (int64_t t = 0; t < num_tables(); ++t) {
      if (counts[t] > 0) queues_[t]->take_snapshot(stream);
    }
    DEMB_CUDA_CHECK(cudaEventRecord(snapshot_done_, stream));

    for (int64_t t = 0; t < num_tables(); ++t) {
      const int64_t b = offsets[t], n = counts[t];
      if (n == 0) continue;
      allocate_kernel<Queue, Map, Node><<<grid_for(n), kBlock, 0, stream>>>(
          queues_[t]->queue(), maps_[t]->map(), maps_[t]->nodes(),
          keys.data_ptr<int64_t>() + b, positions.data_ptr<int64_t>() + b,
          reinterpret_cast<const unsigned long long*>(d_counts.data_ptr<int64_t>()) +
              t,
          reinterpret_cast<Slot*>(slots.data_ptr<uint8_t>()) + b,
          nodes.data_ptr<int32_t>());
    }
    unpack_slots_kernel<Map><<<grid_for(m_total), kBlock, 0, stream>>>(
        reinterpret_cast<const Slot*>(slots.data_ptr<uint8_t>()), m_total,
        victim_keys.data_ptr<int64_t>(), has_victim.data_ptr<uint8_t>(),
        new_nodes.data_ptr<int32_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return {victim_keys, has_victim, new_nodes};
  }

  // LOADING -> RESIDENT for the nodes allocate returned, once their rows are
  // written.
  void publish(const at::Tensor& new_nodes, const std::vector<int64_t>& offsets) {
    check_cuda_tensor(new_nodes, "new_nodes", at::kInt);
    check_offsets(offsets, new_nodes.numel());
    const c10::cuda::CUDAGuard guard(new_nodes.device());
    const cudaStream_t stream = current_stream();
    for (int64_t t = 0; t < num_tables(); ++t) {
      const int64_t b = offsets[t], n = offsets[t + 1] - offsets[t];
      if (n == 0) continue;
      publish_nodes_kernel<Map, Node><<<grid_for(n), kBlock, 0, stream>>>(
          maps_[t]->nodes(), new_nodes.data_ptr<int32_t>() + b, n);
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Give back the reference each step took, node by node; a node of -1 holds none.
  void release(const at::Tensor& nodes, const std::vector<int64_t>& offsets) {
    check_cuda_tensor(nodes, "nodes", at::kInt);
    check_offsets(offsets, nodes.numel());
    const c10::cuda::CUDAGuard guard(nodes.device());
    const cudaStream_t stream = current_stream();

    DEMB_CUDA_CHECK(cudaStreamWaitEvent(stream, snapshot_done_, 0));
    for (int64_t t = 0; t < num_tables(); ++t) {
      const int64_t b = offsets[t], n = offsets[t + 1] - offsets[t];
      if (n == 0) continue;
      release_kernel<Queue, Node><<<grid_for(n), kBlock, 0, stream>>>(
          queues_[t]->queue(), maps_[t]->nodes(), nodes.data_ptr<int32_t>() + b,
          n);
    }
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    DEMB_CUDA_CHECK(cudaEventRecord(release_done_, stream));
  }

  // -- looking at and taking things out of the cache ---------------------------

  // The node holding each key, in the table given for it, or -1. Pins nothing.
  at::Tensor find_nodes(const at::Tensor& keys, const at::Tensor& table_ids) {
    check_cuda_tensor(keys, "keys", at::kLong);
    check_cuda_tensor(table_ids, "table_ids", at::kLong);
    TORCH_CHECK(keys.numel() == table_ids.numel());
    const c10::cuda::CUDAGuard guard(keys.device());
    at::Tensor out = at::empty({keys.numel()}, keys.options().dtype(at::kInt));
    if (keys.numel() == 0) return out;
    find_nodes_kernel<Map><<<grid_for(keys.numel()), kBlock, 0,
                             current_stream()>>>(
        views_ptr(), keys.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
        keys.numel(), out.data_ptr<int32_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }

  // Drop the bindings find_nodes found. Nothing is written back.
  void invalidate(const at::Tensor& nodes, const at::Tensor& table_ids) {
    check_cuda_tensor(nodes, "nodes", at::kInt);
    check_cuda_tensor(table_ids, "table_ids", at::kLong);
    TORCH_CHECK(nodes.numel() == table_ids.numel());
    if (nodes.numel() == 0) return;
    const c10::cuda::CUDAGuard guard(nodes.device());
    invalidate_kernel<Map><<<grid_for(nodes.numel()), kBlock, 0,
                             current_stream()>>>(
        views_ptr(), nodes.data_ptr<int32_t>(), table_ids.data_ptr<int64_t>(),
        nodes.numel());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }

  // Every resident node of one table, with its key, and how many nodes still hold
  // a reference. For a checkpoint, run with the pipeline drained.
  std::tuple<at::Tensor, at::Tensor, int64_t> resident(int64_t table) {
    TORCH_CHECK(table >= 0 && table < num_tables(), "no such table");
    const c10::cuda::CUDAGuard guard(device_);
    const cudaStream_t stream = current_stream();
    const int64_t c = capacities_[table];
    const auto opts = at::TensorOptions().device(at::kCUDA, device_);

    at::Tensor slots = at::empty({c * static_cast<int64_t>(sizeof(Slot))},
                                 opts.dtype(at::kByte));
    at::Tensor counters = at::zeros({2}, opts.dtype(at::kLong));
    auto* count = reinterpret_cast<unsigned long long*>(counters.data_ptr<int64_t>());
    resident_slots_kernel<Map, Node><<<grid_for(c), kBlock, 0, stream>>>(
        maps_[table]->nodes(), c, reinterpret_cast<Slot*>(slots.data_ptr<uint8_t>()),
        count, count + 1);
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();

    const at::Tensor h = counters.to(at::kCPU);  // the one synchronisation
    const int64_t n = h.data_ptr<int64_t>()[0];
    const int64_t referenced = h.data_ptr<int64_t>()[1];

    at::Tensor keys = at::empty({n}, opts.dtype(at::kLong));
    at::Tensor flags = at::empty({n}, opts.dtype(at::kByte));
    at::Tensor nodes = at::empty({n}, opts.dtype(at::kInt));
    if (n > 0) {
      unpack_slots_kernel<Map><<<grid_for(n), kBlock, 0, stream>>>(
          reinterpret_cast<const Slot*>(slots.data_ptr<uint8_t>()), n,
          keys.data_ptr<int64_t>(), flags.data_ptr<uint8_t>(),
          nodes.data_ptr<int32_t>());
      DEMB_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {nodes, keys, referenced};
  }

  // How many nodes of a table are resident.
  int64_t num_resident(int64_t table) {
    return std::get<0>(resident(table)).numel();
  }

  // For a test, with nothing in flight: how many entries the reclaim queue holds,
  // and how many nodes say they have one. The two have to be equal. A node that
  // says so with no entry is one the queue will never hand out again, which is
  // what going wrong between a pop and a push looks like.
  std::pair<int64_t, int64_t> queue_audit(int64_t table) {
    TORCH_CHECK(table >= 0 && table < num_tables(), "no such table");
    const c10::cuda::CUDAGuard guard(device_);
    DEMB_CUDA_CHECK(cudaDeviceSynchronize());
    const auto nodes = maps_[table]->download_nodes();
    int64_t flagged = 0;
    for (const Node& n : nodes) flagged += n.state.get().in_ring() ? 1 : 0;
    return {queues_[table]->size(), flagged};
  }

 private:
  void check_offsets(const std::vector<int64_t>& offsets, int64_t total) const {
    TORCH_CHECK(static_cast<int64_t>(offsets.size()) == num_tables() + 1,
                "offsets needs one entry per table, and the total at the end");
    TORCH_CHECK(offsets.front() == 0 && offsets.back() == total,
                "offsets must run from 0 to the number of keys");
    for (size_t t = 0; t + 1 < offsets.size(); ++t) {
      TORCH_CHECK(offsets[t] <= offsets[t + 1], "offsets must not decrease");
    }
  }

  // The tables' maps as the array the fused kernels take.
  void upload_views() {
    std::vector<Map> views;
    for (const auto& m : maps_) views.push_back(m->map());
    const auto bytes = static_cast<int64_t>(views.size() * sizeof(Map));
    d_views_ = at::empty({bytes}, at::TensorOptions()
                                      .dtype(at::kByte)
                                      .device(at::kCUDA, device_));
    DEMB_CUDA_CHECK(cudaMemcpy(d_views_.data_ptr(), views.data(), bytes,
                               cudaMemcpyHostToDevice));
  }
  const Map* views_ptr() const {
    return reinterpret_cast<const Map*>(d_views_.data_ptr());
  }

  std::vector<int64_t> capacities_;
  int device_ = 0;
  std::vector<std::unique_ptr<MapStore>> maps_;
  std::vector<std::unique_ptr<QStore>> queues_;
  at::Tensor d_views_;
  cudaEvent_t release_done_ = nullptr;
  cudaEvent_t snapshot_done_ = nullptr;
};

}  // namespace
}  // namespace embedding_cache

void bind_embedding_cache(py::module& m) {
  using embedding_cache::EmbeddingCacheCore;
  py::class_<EmbeddingCacheCore>(m, "EmbeddingCacheCore")
      .def(py::init<std::vector<int64_t>>(), py::arg("capacities"))
      .def("num_tables", &EmbeddingCacheCore::num_tables)
      .def("guard", &EmbeddingCacheCore::guard)
      .def("capacity", &EmbeddingCacheCore::capacity, py::arg("table"))
      .def("reset", &EmbeddingCacheCore::reset)
      .def("lookup_pin", &EmbeddingCacheCore::lookup_pin, py::arg("keys"),
           py::arg("offsets"))
      .def("allocate", &EmbeddingCacheCore::allocate, py::arg("keys"),
           py::arg("positions"), py::arg("offsets"), py::arg("nodes"))
      .def("publish", &EmbeddingCacheCore::publish, py::arg("new_nodes"),
           py::arg("offsets"))
      .def("release", &EmbeddingCacheCore::release, py::arg("nodes"),
           py::arg("offsets"))
      .def("find_nodes", &EmbeddingCacheCore::find_nodes, py::arg("keys"),
           py::arg("table_ids"))
      .def("invalidate", &EmbeddingCacheCore::invalidate, py::arg("nodes"),
           py::arg("table_ids"))
      .def("resident", &EmbeddingCacheCore::resident, py::arg("table"))
      .def("num_resident", &EmbeddingCacheCore::num_resident, py::arg("table"))
      .def("queue_audit", &EmbeddingCacheCore::queue_audit, py::arg("table"));
}
