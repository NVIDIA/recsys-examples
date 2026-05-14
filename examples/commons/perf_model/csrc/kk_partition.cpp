// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Karmarkar-Karp k-way partitioning in C++ — drop-in replacement for the
// pure-Python implementation in `partitioner.py`.  The whole compute path
// releases the GIL so the main Python thread can keep submitting CUDA
// kernels while the algorithm runs in a background ThreadPoolExecutor.
//
// Output is bit-for-bit identical to the Python version (same tie-breaking
// rules) so it can be swapped in without changing downstream behaviour.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct Set {
    int64_t sum = 0;
    std::vector<std::pair<int64_t, int64_t>> items;  // (idx, val)

    void add(int64_t idx, int64_t val) {
        items.emplace_back(idx, val);
        sum += val;
    }

    void merge_from(Set& other) {
        items.reserve(items.size() + other.items.size());
        for (auto& it : other.items) {
            items.push_back(it);
            sum += it.second;
        }
    }

    // Matches Python `Set.__lt__`:
    //   if sum != other.sum: return sum < other.sum
    //   if len(items) != len(other.items): return len(items) < len(other.items)
    //   return items < other.items   # lexicographic
    bool operator<(const Set& other) const {
        if (sum != other.sum) return sum < other.sum;
        if (items.size() != other.items.size())
            return items.size() < other.items.size();
        return items < other.items;
    }
    bool operator>(const Set& other) const { return other < *this; }
};

struct State {
    int k;
    std::vector<Set> sets;  // maintained in *descending* order (sets[0] largest)

    explicit State(int k_) : k(k_), sets(k_) {}

    // ``items`` has length in [1, k]; element i goes into sets[i] (matching
    // Python init), then sets are sorted descending.
    void init_from(const std::vector<std::pair<int64_t, int64_t>>& items) {
        for (size_t i = 0; i < items.size(); ++i) {
            sets[i].add(items[i].first, items[i].second);
        }
        std::sort(sets.begin(), sets.end(), std::greater<Set>());
    }

    // Python `merge`: pair sets[i] ↔ other.sets[k-1-i], then resort descending.
    void merge_with(State& other) {
        for (int i = 0; i < k; ++i) {
            sets[i].merge_from(other.sets[k - 1 - i]);
        }
        std::sort(sets.begin(), sets.end(), std::greater<Set>());
    }

    int64_t spread() const { return sets.front().sum - sets.back().sum; }

    // Heap ordering. Python uses a min-heap (`heapq`) with `State.__lt__`
    // flipped so the state with the LARGEST spread is popped first:
    //   if spread != other.spread: return spread > other.spread
    //   return sets[0] > other.sets[0]
    //
    // ``std::priority_queue`` / ``std::push_heap`` give a max-heap based on
    // ``operator<``: the element where ``a < b`` is true for every other ``b``
    // gets popped LAST.  So define ``operator<`` such that "smaller" means
    // "lower priority" (popped later), which means we want LARGER spread
    // (and, on tie, larger ``sets[0]``) to compare as GREATER.
    bool operator<(const State& other) const {
        const int64_t s0 = spread();
        const int64_t s1 = other.spread();
        if (s0 != s1) return s0 < s1;
        return sets.front() < other.sets.front();
    }
};

std::vector<std::vector<int64_t>> karmarkar_karp_cpp(
    std::vector<int64_t> workloads,
    int k_partitions,
    bool equal_size) {
    // Release the GIL for the entire compute.  ``workloads`` was already
    // pickled in by pybind11 (when called across processes) or copied from a
    // Python list (when called in-process) before this point, so we do not
    // touch any Python object until we return.
    py::gil_scoped_release release;

    if (k_partitions <= 0) {
        throw std::invalid_argument("k_partitions must be > 0");
    }
    const size_t n = workloads.size();
    if (equal_size && (n % static_cast<size_t>(k_partitions) != 0)) {
        throw std::invalid_argument(
            "len(workloads) must be divisible by k_partitions when equal_size=True");
    }
    if (n == 0) {
        return std::vector<std::vector<int64_t>>(k_partitions);
    }

    // Match Python's ``sorted([(workload, i) for i, workload in enumerate(workloads)])``
    // — ascending by (workload, idx).  std::pair<int64_t,int64_t>::operator< is
    // lexicographic, so a plain std::sort on (workload, idx) does it.
    std::vector<std::pair<int64_t, int64_t>> sorted_workloads;
    sorted_workloads.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_workloads.emplace_back(workloads[i], static_cast<int64_t>(i));
    }
    std::sort(sorted_workloads.begin(), sorted_workloads.end());

    // Build initial heap of States.
    std::vector<State> heap;
    heap.reserve(equal_size ? n / k_partitions : n);

    if (equal_size) {
        std::vector<std::pair<int64_t, int64_t>> group;
        group.reserve(k_partitions);
        for (size_t off = 0; off < n; off += k_partitions) {
            group.clear();
            for (int i = 0; i < k_partitions; ++i) {
                const auto& [workload, idx] = sorted_workloads[off + i];
                // Python: items.append((idx, workload))  (note: (idx, workload), not (workload, idx))
                group.emplace_back(idx, workload);
            }
            State s(k_partitions);
            s.init_from(group);
            heap.push_back(std::move(s));
        }
    } else {
        std::vector<std::pair<int64_t, int64_t>> single(1);
        for (const auto& [workload, idx] : sorted_workloads) {
            single[0] = {idx, workload};
            State s(k_partitions);
            s.init_from(single);
            heap.push_back(std::move(s));
        }
    }
    std::make_heap(heap.begin(), heap.end());

    while (heap.size() > 1) {
        std::pop_heap(heap.begin(), heap.end());
        State s0 = std::move(heap.back());
        heap.pop_back();

        std::pop_heap(heap.begin(), heap.end());
        State s1 = std::move(heap.back());
        heap.pop_back();

        s0.merge_with(s1);
        heap.push_back(std::move(s0));
        std::push_heap(heap.begin(), heap.end());
    }

    // Extract partitions from the surviving state.
    State& final_state = heap.front();
    std::vector<std::vector<int64_t>> partitions(k_partitions);
    for (int i = 0; i < k_partitions; ++i) {
        auto& src = final_state.sets[i].items;
        auto& dst = partitions[i];
        dst.reserve(src.size());
        for (const auto& [idx, _val] : src) {
            dst.push_back(idx);
        }
    }
    return partitions;
}

}  // namespace

PYBIND11_MODULE(kk_cpu_ops, m) {
    m.doc() =
        "C++ Karmarkar-Karp k-way partitioning. Releases the GIL during compute "
        "so the main Python thread can keep submitting CUDA kernels.";
    m.def(
        "karmarkar_karp",
        &karmarkar_karp_cpp,
        py::arg("workloads"),
        py::arg("k_partitions"),
        py::arg("equal_size"),
        "Identical output to commons.perf_model.partitioner.karmarkar_karp "
        "(same tie-breaking rules), but with the GIL released for the entire "
        "compute.");
}
