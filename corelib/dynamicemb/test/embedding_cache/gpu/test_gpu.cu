// ---------------------------------------------------------------------------
// embedding_cache on a device.
//
// Everything the host tests cannot reach: whether these headers compile with
// nvcc at all, and whether the protocol survives real contention. The host
// build runs against a stand-in for the CUDA runtime that serialises every
// atomic, so it can show a protocol is right and cannot show it is race-free.
//
// No embedding rows here either. Each node gets one int64 standing in for its
// row; allocate writes the key into it, and every hit checks it reads its own
// key back. That is enough to catch the two failures that matter — a node bound
// to two keys, and a key bound to two nodes — and it needs neither a row pool
// nor a backing store, neither of which exists yet.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include "kernels.cuh"
#include "rows.cuh"

using namespace embedding_cache;

static int failures = 0;

// Under compute-sanitizer every memory access is instrumented, and the long runs
// below stretch from minutes to hours. EC_TEST_SCALE=n divides their step counts
// by n; unset, it is the full run. A scaled run checks the same properties on a
// shorter history, which is what a race detector needs: it looks at the accesses,
// and a few steps of them are the same accesses.
static int scaled(int n) {
  static const int divisor = [] {
    const char* e = std::getenv("EC_TEST_SCALE");
    const int v = e ? std::atoi(e) : 1;
    return v > 0 ? v : 1;
  }();
  return n / divisor > 0 ? n / divisor : 1;
}
#define CHECK(cond, ...)                          \
  do {                                            \
    if (!(cond)) {                                \
      std::printf("  FAIL: ");                    \
      std::printf(__VA_ARGS__);                   \
      std::printf("  (%s:%d)\n", __FILE__, __LINE__); \
      ++failures;                                 \
    }                                             \
  } while (0)

#define CU(call)                                                          \
  do {                                                                    \
    const cudaError_t e_ = (call);                                        \
    if (e_ != cudaSuccess) {                                              \
      std::printf("  CUDA FAIL: %s -> %s  (%s:%d)\n", #call,              \
                  cudaGetErrorString(e_), __FILE__, __LINE__);            \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// What the cache would define: the map's three fields from the base, the
// concurrency word from the other component.
struct Node : HashMapNode<int64_t, int32_t> {
  AtomicNodeState state;
};

using MapStore = HashMapStorage<Node>;
using Map = MapStore::Map;
using QStore = ReclaimQueueStorage<Map::NodeId, Map::kNoNode>;
using Queue = QStore::Queue;
using Slot = AllocSlot<Map::Key, Map::NodeId>;

static_assert(sizeof(Node) == 24, "key 8 + links 8 + state 4, padded to 24");
static_assert(Queue::kNoNode == Map::kNoNode, "one source of truth");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <typename T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    CU(cudaMalloc(&p, sizeof(T) * count));
    CU(cudaMemset(p, 0, sizeof(T) * count));
  }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  std::vector<T> get() const {
    std::vector<T> out(n);
    CU(cudaMemcpy(out.data(), p, sizeof(T) * n, cudaMemcpyDeviceToHost));
    return out;
  }
  void put(const std::vector<T>& v) {
    CU(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
  }
};

// Every node starts EMPTY and already in the queue: the queue is arange(C), so
// each node is sitting in it. HashMapStorage cannot do this — it does not know
// what a state word should hold — so it is the caller's, which here is us.
__global__ void init_states_kernel(Node* nodes, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    nodes[i].state.set(NodeState::make(NodeState::kEmpty, 0u,
                                       /*in_ring=*/true, /*used=*/false));
  }
}

// ---------------------------------------------------------------------------
// 1. NodeState under contention
// ---------------------------------------------------------------------------
//
// Many threads pin and release ONE node. Every pin that succeeds is matched by
// a release, so the count has to come back to where it started. A lost update
// in either direction shows up here and nowhere else.

__global__ void hammer_pin_release_kernel(Node* nodes, int32_t node, int rounds,
                                          unsigned long long* pins,
                                          unsigned long long* enqueues) {
  for (int r = 0; r < rounds; ++r) {
    if (nodes[node].state.try_pin()) {
      atomicAdd(pins, 1ull);
      if (nodes[node].state.release()) atomicAdd(enqueues, 1ull);
    }
  }
}

static void test_state_contention() {
  std::printf("=== NodeState under contention ===\n");
  const int64_t kNodes = 1;
  Dev<Node> nodes(kNodes);
  Dev<unsigned long long> pins(1), enqueues(1);

  std::vector<Node> h(1);
  h[0].key = 42;
  h[0].prev = h[0].next = Map::kNoNode;
  // Resident, one reference held by "the test", already in the queue.
  h[0].state.set(NodeState::make(NodeState::kResident, 1u, true, false));
  nodes.put(h);

  const int kBlocks = 64, kThreads = 256, kRounds = 64;
  hammer_pin_release_kernel<<<kBlocks, kThreads>>>(nodes.p, 0, kRounds, pins.p,
                                                   enqueues.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto after = nodes.get()[0].state.get();
  const auto npins = pins.get()[0];
  std::printf("  %llu pins by %d threads x %d rounds, refs back to %u\n",
              npins, kBlocks * kThreads, kRounds, after.refs());
  CHECK(npins == (unsigned long long)kBlocks * kThreads * kRounds,
        "every pin on a resident node should succeed");
  CHECK(after.refs() == 1u, "refs returned to its starting value, got %u",
        after.refs());
  CHECK(after.residency() == NodeState::kResident, "residency untouched");
  CHECK(after.used(), "pin recorded recency");
  CHECK(enqueues.get()[0] == 0ull,
        "the node never reached zero references, so nobody owned an enqueue");
}

// ---------------------------------------------------------------------------
// 2. try_claim hands one node to exactly one thread
// ---------------------------------------------------------------------------

__global__ void hammer_claim_kernel(Node* nodes, int64_t n,
                                    unsigned long long* wins) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n * 8; i += int64_t(gridDim.x) * blockDim.x) {
    NodeState was = NodeState::make(NodeState::kEmpty, 0u, false, false);
    if (nodes[i % n].state.try_claim(&was)) atomicAdd(&wins[i % n], 1ull);
  }
}

static void test_claim_exclusivity() {
  std::printf("=== try_claim exclusivity ===\n");
  const int64_t kNodes = 4096;
  Dev<Node> nodes(kNodes);
  Dev<unsigned long long> wins(kNodes);

  init_states_kernel<<<64, 256>>>(nodes.p, kNodes);
  CU(cudaDeviceSynchronize());

  hammer_claim_kernel<<<128, 256>>>(nodes.p, kNodes, wins.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto w = wins.get();
  int64_t bad = 0;
  for (int64_t i = 0; i < kNodes; ++i)
    if (w[i] != 1ull) ++bad;
  CHECK(bad == 0, "%lld of %lld nodes were claimed a number of times other "
                  "than once", (long long)bad, (long long)kNodes);
}

// ---------------------------------------------------------------------------
// 3. The queue, block-collective
// ---------------------------------------------------------------------------
//
// Drain it, then refill it, and check every node came back exactly once. This
// is also the first thing that runs the block-collective launch shape: a
// partial last wave is what would hang it.

__global__ void drain_kernel(Queue q, Node* nodes, int64_t n, int32_t* out) {
  __shared__ Queue::Scratch s;
  const uint64_t snap = q.snapshot_tail();
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    NodeState was = NodeState::make(NodeState::kEmpty, 0u, false, false);
    const int32_t got = q.pop(i < n, snap, s, [&](int32_t c) {
      return nodes[c].state.try_claim(&was);
    });
    if (i < n) out[i] = got;
  }
}

__global__ void refill_kernel(Queue q, Node* nodes, const int32_t* ids,
                              int64_t n) {
  __shared__ Queue::Scratch s;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    const int32_t node = (i < n) ? ids[i] : Queue::kNoNode;
    // Claimed nodes hold one reference; give it back so they go idle again.
    const bool owns = (node != Queue::kNoNode) && nodes[node].state.release();
    q.push(owns, node, s);
  }
}

static void test_queue_round_trip() {
  std::printf("=== reclaim queue, block-collective ===\n");
  // Deliberately not a power of two, and deliberately not a multiple of the
  // block size, so the last wave is partial.
  const int64_t kNodes = 3000;
  Dev<Node> nodes(kNodes);
  Dev<int32_t> got(kNodes);
  QStore qs(kNodes);

  init_states_kernel<<<64, 256>>>(nodes.p, kNodes);
  CU(cudaDeviceSynchronize());
  CHECK(qs.size() == kNodes, "a fresh queue holds every node, got %lld",
        (long long)qs.size());

  qs.take_snapshot(nullptr);
  drain_kernel<<<32, 256>>>(qs.queue(), nodes.p, kNodes, got.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto ids = got.get();
  std::vector<int> seen(kNodes, 0);
  int64_t dry = 0, dup = 0;
  for (int32_t id : ids) {
    if (id == Queue::kNoNode) { ++dry; continue; }
    if (seen[id]++) ++dup;
  }
  CHECK(dry == 0, "%lld pops came back empty from a full queue", (long long)dry);
  CHECK(dup == 0, "%lld nodes were handed out more than once", (long long)dup);
  CHECK(qs.size() == 0, "a drained queue is empty, got %lld",
        (long long)qs.size());

  refill_kernel<<<32, 256>>>(qs.queue(), nodes.p, got.p, kNodes);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());
  CHECK(qs.size() == kNodes, "everything went back, got %lld",
        (long long)qs.size());
}

// ---------------------------------------------------------------------------
// 4. The whole bookkeeping pipeline, many steps
// ---------------------------------------------------------------------------

__global__ void fill_stub_kernel(Node* nodes, const Slot* slots,
                                 const unsigned long long* count,
                                 int64_t* rows) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const Slot s = slots[i];
    if (s.node == Map::kNoNode) continue;
    rows[s.node] = s.key;   // the row, standing in for an embedding
    nodes[s.node].state.publish();
  }
}

__global__ void verify_kernel(const Node* nodes, const int64_t* keys,
                              const int32_t* node_ids, const int64_t* rows,
                              int64_t n, unsigned long long* wrong,
                              unsigned long long* dry) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int32_t node = node_ids[i];
    if (node == Map::kNoNode) { atomicAdd(dry, 1ull); continue; }
    if (rows[node] != keys[i] || nodes[node].key != keys[i]) {
      atomicAdd(wrong, 1ull);
    }
  }
}

static void test_pipeline(int steps, int64_t capacity, int64_t per_step,
                          int64_t key_space, unsigned seed) {
  std::printf("=== pipeline: %d steps, capacity %lld, %lld keys/step out of "
              "%lld ===\n",
              steps, (long long)capacity, (long long)per_step,
              (long long)key_space);

  MapStore map_store(capacity);
  QStore qs(capacity);
  Node* nodes = map_store.nodes();

  init_states_kernel<<<64, 256>>>(nodes, capacity);
  CU(cudaDeviceSynchronize());

  Dev<int64_t> rows(capacity);
  Dev<int64_t> keys(per_step), miss_keys(per_step), miss_src(per_step);
  Dev<int32_t> node_ids(per_step);
  Dev<Slot> slots(per_step);
  Dev<unsigned long long> miss_count(1), wrong(1), dry(1);

  std::mt19937_64 rng(seed);
  // Skewed, so the same keys come back and the cache is actually exercised.
  std::geometric_distribution<int64_t> hot(0.01);

  const int grid = grid_for(per_step);
  int64_t total_miss = 0;

  for (int step = 0; step < steps; ++step) {
    std::vector<int64_t> h_keys;
    h_keys.reserve(per_step);
    std::vector<char> present(key_space, 0);
    while ((int64_t)h_keys.size() < per_step) {
      const int64_t k = hot(rng) % key_space;
      if (present[k]) continue;  // unique within a step, as the cache requires
      present[k] = 1;
      h_keys.push_back(k);
    }
    keys.put(h_keys);
    CU(cudaMemset(miss_count.p, 0, sizeof(unsigned long long)));

    lookup_pin_kernel<Map, Node><<<grid, kBlock>>>(
        map_store.map(), nodes, keys.p, per_step, node_ids.p, miss_keys.p,
        miss_src.p, miss_count.p);
    CU(cudaGetLastError());

    qs.take_snapshot(nullptr);
    allocate_kernel<Queue, Map, Node><<<grid, kBlock>>>(
        qs.queue(), map_store.map(), nodes, miss_keys.p, miss_src.p,
        miss_count.p, slots.p, node_ids.p);
    CU(cudaGetLastError());

    fill_stub_kernel<<<grid, kBlock>>>(nodes, slots.p, miss_count.p, rows.p);
    CU(cudaGetLastError());

    verify_kernel<<<grid, kBlock>>>(nodes, keys.p, node_ids.p, rows.p, per_step,
                                    wrong.p, dry.p);
    CU(cudaGetLastError());

    release_kernel<Queue, Node><<<grid, kBlock>>>(qs.queue(), nodes,
                                                  node_ids.p, per_step);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    total_miss += (int64_t)miss_count.get()[0];
  }

  const auto n_wrong = wrong.get()[0];
  const auto n_dry = dry.get()[0];
  const double hit = 1.0 - double(total_miss) / double(steps * per_step);
  std::printf("  hit rate %.3f, %llu unserved, %llu wrong\n", hit, n_dry,
              n_wrong);

  CHECK(n_wrong == 0, "%llu keys resolved to a node holding something else",
        n_wrong);
  CHECK(map_store.check_bindings() == 0,
        "every node in a bucket hashes to that bucket");

  // No key may hold two nodes. check_bindings cannot see this — both nodes sit
  // in the right bucket — so it is checked here.
  const auto h_nodes = map_store.download_nodes();
  const auto h_heads = map_store.download_heads();
  std::map<int64_t, int> owner;
  int64_t dupes = 0;
  for (int64_t b = 0; b < map_store.num_buckets(); ++b) {
    for (int32_t cur = h_heads[b]; cur != Map::kNoNode;
         cur = h_nodes[cur].next) {
      if (!owner.emplace(h_nodes[cur].key, cur).second) ++dupes;
    }
  }
  CHECK(dupes == 0, "%lld keys are bound to more than one node",
        (long long)dupes);

  // At rest every node is idle, and every idle node is queued exactly once.
  int64_t queued = 0, referenced = 0;
  for (int64_t i = 0; i < capacity; ++i) {
    const NodeState s = h_nodes[i].state.get();
    if (s.has_refs()) ++referenced;
    if (s.in_ring()) ++queued;
  }
  CHECK(referenced == 0, "%lld nodes still hold a reference after every "
                         "release", (long long)referenced);
  std::printf("  %lld nodes flagged queued, queue holds %lld\n",
              (long long)queued, (long long)qs.size());
  CHECK(queued == qs.size(),
        "the in_queue flags and the queue itself disagree: %lld vs %lld",
        (long long)queued, (long long)qs.size());
}

// ---------------------------------------------------------------------------
// 5. A release overlapping an allocate
// ---------------------------------------------------------------------------
//
// The pipeline's one deliberate overlap: step N's release pushes the queue while
// step N+1's allocate pops it. The queue is only safe under that if no push can
// reach a slot a popper has yet to read, and the way to find out is to make it
// likely — a full queue, so any push wraps onto the oldest unread slot, and half
// the entries stale, so that pops are dropping entries while pushes land.
//
// Even nodes are held by a step that has not released yet. They were pinned after
// they were enqueued, so each has a stale entry. Odd nodes are idle. The popper
// asks for as many nodes as there are odd ones; the pusher releases the even
// ones. Whichever order a given node's drop and release fall in, the protocol
// has to come out consistent, and consistency is checkable at rest:
//
//   * a node is flagged in_ring exactly when it has one live entry
//   * no node has two
//   * every node a popper was handed is held, and has no entry
//
// A push that wrapped onto an unread slot shows up as a flagged node with no
// entry, and a popper that read the wrong slot as a node handed out that is also
// still queued. The snapshot is taken and synchronised before the pusher is
// launched, which is the ordering the queue asks of its caller; this checks the
// queue given that ordering, not the ordering.

__global__ void init_overlap_states_kernel(Node* nodes, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    nodes[i].state.set(NodeState::make(NodeState::kResident,
                                       (i % 2 == 0) ? 1u : 0u,
                                       /*in_ring=*/true, /*used=*/false));
  }
}

// One thread per even node: drop the reference, push if it owns the enqueue.
__global__ void release_even_kernel(Queue q, Node* nodes, int64_t n_even) {
  __shared__ Queue::Scratch s;
  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n_even;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    const int32_t node = (i < n_even) ? int32_t(2 * i) : Queue::kNoNode;
    const bool owns = (node != Queue::kNoNode) && nodes[node].state.release();
    q.push(owns, node, s);
  }
}

static void test_release_overlaps_allocate() {
  std::printf("=== a release overlapping an allocate ===\n");
  const int64_t kNodes = 1 << 18;
  const int64_t n_even = kNodes / 2;
  const int kIters = scaled(100);

  Dev<Node> nodes(kNodes);
  Dev<int32_t> got(n_even);
  QStore qs(kNodes);

  cudaStream_t pop_stream, push_stream;
  CU(cudaStreamCreateWithFlags(&pop_stream, cudaStreamNonBlocking));
  CU(cudaStreamCreateWithFlags(&push_stream, cudaStreamNonBlocking));
  const int grid = grid_for(n_even);

  int64_t bad_iters = 0;
  for (int it = 0; it < kIters; ++it) {
    qs.reset();
    init_overlap_states_kernel<<<64, 256>>>(nodes.p, kNodes);
    CU(cudaDeviceSynchronize());

    qs.take_snapshot(pop_stream);
    CU(cudaStreamSynchronize(pop_stream));

    drain_kernel<<<grid, kBlock, 0, pop_stream>>>(qs.queue(), nodes.p, n_even,
                                                  got.p);
    release_even_kernel<<<grid, kBlock, 0, push_stream>>>(qs.queue(), nodes.p,
                                                          n_even);
    CU(cudaGetLastError());
    CU(cudaStreamSynchronize(pop_stream));
    CU(cudaStreamSynchronize(push_stream));

    const auto h_nodes = nodes.get();
    const auto ids = got.get();
    const auto counters = qs.download_counters();
    const auto slots = qs.download_slots();
    const uint64_t mask = qs.queue().mask;

    std::vector<int> entries(kNodes, 0);
    for (uint64_t p = counters[0]; p < counters[1]; ++p) {
      const int32_t n = slots[p & mask];
      if (n < 0 || n >= kNodes) {
        entries.assign(kNodes, -1);  // an entry that names no node
        break;
      }
      ++entries[n];
    }

    int64_t wrong = 0;
    for (int64_t i = 0; i < kNodes; ++i) {
      const NodeState s = h_nodes[i].state.get();
      if (entries[i] < 0 || entries[i] > 1) ++wrong;               // duplicated
      else if (s.in_ring() != (entries[i] == 1)) ++wrong;          // flag != entry
    }
    std::vector<char> handed(kNodes, 0);
    for (int32_t id : ids) {
      if (id == Queue::kNoNode) { ++wrong; continue; }             // ran dry
      if (handed[id]++) ++wrong;                                   // handed twice
      const NodeState s = h_nodes[id].state.get();
      if (s.residency() != NodeState::kLoading || s.refs() != 1u) ++wrong;
      if (entries[id] != 0) ++wrong;                               // handed out and queued
    }
    if (wrong != 0) {
      ++bad_iters;
      if (bad_iters <= 3) {
        std::printf("  iteration %d: %lld inconsistencies\n", it,
                    (long long)wrong);
      }
    }
  }
  std::printf("  %d iterations, %lld inconsistent\n", kIters,
              (long long)bad_iters);
  CHECK(bad_iters == 0,
        "%lld iterations left the queue and the node flags disagreeing",
        (long long)bad_iters);

  CU(cudaStreamDestroy(pop_stream));
  CU(cudaStreamDestroy(push_stream));
}

// ---------------------------------------------------------------------------
// 6. Evicting by key, and the guard in front of it
// ---------------------------------------------------------------------------
//
// The backing evicts keys by its own score and the cache has no say, except that
// a key a step is using must not go. Two things are checked:
//
//   * the guard says no exactly for the keys whose nodes hold references
//   * find_nodes + invalidate take cached keys out cleanly — gone from the
//     chains, their nodes empty and still queued, every other binding untouched —
//     and the cache then carries on: the evicted keys miss and are bound again,
//     the empty nodes are taken with no victim to unlink
//
// The second is the one a wrong unlink would break. allocate treats an empty node
// as having nothing to remove, and remove() asserts on a node that is in no chain,
// so a node invalidate left in a chain, or allocate mistook for resident, stops
// the kernel here.
//
// These kernels take an array of per-table maps and a table id per key, so here
// they run with one table.

using Guard = EvictionGuard<Map>;

__global__ void guard_probe_kernel(Guard guard, const int64_t* keys,
                                   const int64_t* table_ids, int64_t n,
                                   char* out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = guard.may_evict(keys[i], table_ids[i]) ? 1 : 0;
  }
}

// One cache, one step at a time. acquire() looks up, allocates and fills; it
// leaves the step's references held until release(), which is what a step in
// flight looks like to the guard.
struct Harness {
  int64_t capacity, per_step;
  MapStore map_store;
  QStore qs;
  Node* nodes;
  Dev<int64_t> rows, keys, miss_keys, miss_src;
  Dev<int32_t> node_ids;
  Dev<Slot> slots;
  Dev<unsigned long long> miss_count, wrong, dry;
  Dev<Map> d_map;  // this one table, as the one-element array the kernels take
  int64_t held = 0;

  Harness(int64_t capacity_, int64_t per_step_)
      : capacity(capacity_), per_step(per_step_), map_store(capacity_),
        qs(capacity_), nodes(map_store.nodes()), rows(capacity_),
        keys(per_step_), miss_keys(per_step_), miss_src(per_step_),
        node_ids(per_step_), slots(per_step_), miss_count(1), wrong(1), dry(1),
        d_map(1) {
    init_states_kernel<<<64, 256>>>(nodes, capacity);
    CU(cudaDeviceSynchronize());
    d_map.put(std::vector<Map>{map_store.map()});
  }

  // Look the keys up, pin the hits, and bind a node to each miss. The misses are
  // left LOADING: what moves their rows, and publishes them, is the caller's.
  void lookup_and_allocate(const std::vector<int64_t>& h_keys) {
    held = static_cast<int64_t>(h_keys.size());
    const int grid = grid_for(held);
    keys.put(h_keys);
    CU(cudaMemset(miss_count.p, 0, sizeof(unsigned long long)));
    lookup_pin_kernel<Map, Node><<<grid, kBlock>>>(
        map_store.map(), nodes, keys.p, held, node_ids.p, miss_keys.p,
        miss_src.p, miss_count.p);
    qs.take_snapshot(nullptr);
    allocate_kernel<Queue, Map, Node><<<grid, kBlock>>>(
        qs.queue(), map_store.map(), nodes, miss_keys.p, miss_src.p,
        miss_count.p, slots.p, node_ids.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }

  // The above, then a stand-in for moving rows: each key's row holds its key.
  void acquire(const std::vector<int64_t>& h_keys) {
    lookup_and_allocate(h_keys);
    const int grid = grid_for(held);
    fill_stub_kernel<<<grid, kBlock>>>(nodes, slots.p, miss_count.p, rows.p);
    verify_kernel<<<grid, kBlock>>>(nodes, keys.p, node_ids.p, rows.p, held,
                                    wrong.p, dry.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }

  // A grid for a kernel with one thread per slot.
  int grid_slots() const { return grid_for(per_step); }

  void release() {
    release_kernel<Queue, Node><<<grid_for(held), kBlock>>>(qs.queue(), nodes,
                                                            node_ids.p, held);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    held = 0;
  }

  // Which of `h_keys` are cached: the node holding each, or kNoNode.
  std::vector<int32_t> find(const std::vector<int64_t>& h_keys) {
    Dev<int64_t> d_keys(h_keys.size()), d_tid(h_keys.size());  // tid: all zero
    Dev<int32_t> d_out(h_keys.size());
    d_keys.put(h_keys);
    find_nodes_kernel<Map><<<64, 256>>>(d_map.p, d_keys.p, d_tid.p,
                                        static_cast<int64_t>(h_keys.size()),
                                        d_out.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    return d_out.get();
  }

  void invalidate(const std::vector<int32_t>& found) {
    Dev<int32_t> d_nodes(found.size());
    Dev<int64_t> d_tid(found.size());
    d_nodes.put(found);
    invalidate_kernel<Map><<<64, 256>>>(d_map.p, d_nodes.p, d_tid.p,
                                        static_cast<int64_t>(found.size()));
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }

  // Every key bound to a node, found by walking the chains.
  std::vector<int64_t> bound_keys() {
    const auto h_nodes = map_store.download_nodes();
    const auto h_heads = map_store.download_heads();
    std::vector<int64_t> out;
    for (int64_t b = 0; b < map_store.num_buckets(); ++b) {
      for (int32_t cur = h_heads[b]; cur != Map::kNoNode; cur = h_nodes[cur].next) {
        out.push_back(h_nodes[cur].key);
      }
    }
    return out;
  }

  // The invariants of a cache at rest, for any test that has run it for a while.
  void check_at_rest(const char* name) {
    CHECK(map_store.check_bindings() == 0, "%s: bindings stay consistent", name);
    const auto h_nodes = map_store.download_nodes();
    const auto h_heads = map_store.download_heads();
    std::map<int64_t, int> owner;
    int64_t dupes = 0, queued = 0, referenced = 0;
    for (int64_t b = 0; b < map_store.num_buckets(); ++b) {
      for (int32_t cur = h_heads[b]; cur != Map::kNoNode; cur = h_nodes[cur].next) {
        if (!owner.emplace(h_nodes[cur].key, cur).second) ++dupes;
      }
    }
    for (int64_t i = 0; i < capacity; ++i) {
      const NodeState st = h_nodes[i].state.get();
      if (st.in_ring()) ++queued;
      if (st.has_refs()) ++referenced;
    }
    CHECK(dupes == 0, "%s: %lld keys are bound to more than one node", name,
          (long long)dupes);
    CHECK(referenced == 0, "%s: %lld nodes still hold a reference", name,
          (long long)referenced);
    CHECK(queued == qs.size(), "%s: flags and queue disagree: %lld vs %lld", name,
          (long long)queued, (long long)qs.size());
  }
};

static std::vector<int64_t> draw_keys(std::mt19937_64& rng,
                                      std::geometric_distribution<int64_t>& hot,
                                      int64_t n, int64_t space) {
  std::vector<int64_t> out;
  std::vector<char> present(space, 0);
  while ((int64_t)out.size() < n) {
    const int64_t k = hot(rng) % space;
    if (present[k]) continue;  // unique within a step, as the cache requires
    present[k] = 1;
    out.push_back(k);
  }
  return out;
}

static void test_evict_by_key() {
  std::printf("=== evicting by key, and the guard ===\n");
  const int64_t kCap = 2048, kPerStep = 256, kSpace = 4096;
  Harness h(kCap, kPerStep);

  std::mt19937_64 rng(7);
  std::geometric_distribution<int64_t> hot(0.01);

  for (int s = 0; s < scaled(80); ++s) {
    h.acquire(draw_keys(rng, hot, kPerStep, kSpace));
    h.release();
  }

  const auto cached = h.bound_keys();
  CHECK((int64_t)cached.size() > kPerStep,
        "the cache should be well populated, got %lld",
        (long long)cached.size());

  // --- the guard: hold some cached keys, as a step in flight would -----------
  {
    const std::vector<int64_t> held_keys(cached.begin(), cached.begin() + 64);
    h.acquire(held_keys);

    std::vector<int64_t> probe = cached;
    for (int64_t k = 0; k < 100; ++k) probe.push_back(kSpace + k);  // never cached
    std::vector<char> is_held_key(kSpace + 100, 0);
    for (int64_t k : held_keys) is_held_key[k] = 1;

    Dev<int64_t> d_probe(probe.size()), d_tid(probe.size());  // tid: all zero
    Dev<char> d_may(probe.size());
    d_probe.put(probe);
    guard_probe_kernel<<<64, 256>>>(Guard{h.d_map.p}, d_probe.p, d_tid.p,
                                    static_cast<int64_t>(probe.size()), d_may.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    const auto may = d_may.get();
    int64_t wrong = 0, vetoed = 0;
    for (size_t i = 0; i < probe.size(); ++i) {
      const bool expect = !is_held_key[probe[i]];
      if ((may[i] != 0) != expect) ++wrong;
      if (!may[i]) ++vetoed;
    }
    std::printf("  guard: %zu keys probed, %lld vetoed\n", probe.size(),
                (long long)vetoed);
    CHECK(wrong == 0,
          "%lld keys got the wrong answer from the guard: it must refuse "
          "exactly the keys whose nodes hold references",
          (long long)wrong);
    CHECK(vetoed == (int64_t)held_keys.size(),
          "expected %zu vetoes, got %lld", held_keys.size(), (long long)vetoed);

    h.release();
  }

  // --- evict by key: a third of the cached keys, and some never cached -------
  std::vector<int64_t> evict;
  std::set<int64_t> evicted_cached;
  for (size_t i = 0; i < cached.size(); i += 3) {
    evict.push_back(cached[i]);
    evicted_cached.insert(cached[i]);
  }
  for (int64_t k = 0; k < 50; ++k) evict.push_back(kSpace + 200 + k);

  const auto nodes_found = h.find(evict);
  const auto before = h.map_store.download_nodes();
  int64_t found = 0;
  for (size_t i = 0; i < evict.size(); ++i) {
    const bool is_cached = evicted_cached.count(evict[i]) != 0;
    CHECK((nodes_found[i] != Map::kNoNode) == is_cached,
          "find_nodes should locate exactly the cached keys (key %lld)",
          (long long)evict[i]);
    if (nodes_found[i] != Map::kNoNode) {
      ++found;
      CHECK(before[nodes_found[i]].key == evict[i],
            "the node found holds the key asked for");
    }
  }

  h.invalidate(nodes_found);
  std::printf("  evicted %lld cached keys of %zu asked for\n", (long long)found,
              evict.size());

  // Gone from the chains, and nothing else disturbed.
  for (int32_t n : h.find(evict)) {
    CHECK(n == Map::kNoNode, "an evicted key must not resolve to a node");
  }
  std::vector<int64_t> kept;
  for (int64_t k : cached)
    if (!evicted_cached.count(k)) kept.push_back(k);
  int64_t lost = 0;
  const auto kept_nodes = h.find(kept);
  for (size_t i = 0; i < kept.size(); ++i) {
    if (kept_nodes[i] == Map::kNoNode) ++lost;
  }
  CHECK(lost == 0, "%lld keys that were not evicted lost their node",
        (long long)lost);
  CHECK(h.map_store.check_bindings() == 0,
        "every node in a bucket still hashes to that bucket");

  {
    const auto after = h.map_store.download_nodes();
    int64_t bad_state = 0, queued = 0;
    for (int32_t n : nodes_found) {
      if (n == Map::kNoNode) continue;
      const NodeState st = after[n].state.get();
      if (st.residency() != NodeState::kEmpty || st.has_refs() || !st.in_ring() ||
          st.used()) {
        ++bad_state;
      }
    }
    for (int64_t i = 0; i < kCap; ++i)
      if (after[i].state.get().in_ring()) ++queued;
    CHECK(bad_state == 0,
          "%lld invalidated nodes are not empty, idle, queued and without "
          "recency",
          (long long)bad_state);
    CHECK(queued == h.qs.size(),
          "invalidating must not change the queue: %lld flagged vs %lld held",
          (long long)queued, (long long)h.qs.size());
  }

  // --- and the cache carries on ----------------------------------------------
  //
  // The evicted keys are asked for again first, so they miss and take nodes; then
  // ordinary traffic. Empty nodes get popped along the way, with no victim.
  std::vector<int64_t> again;
  for (int64_t k : evicted_cached) {
    again.push_back(k);
    if ((int64_t)again.size() == kPerStep) break;
  }
  h.acquire(again);
  h.release();
  for (int s = 0; s < scaled(60); ++s) {
    h.acquire(draw_keys(rng, hot, kPerStep, kSpace));
    h.release();
  }

  CHECK(h.wrong.get()[0] == 0ull, "%llu keys resolved to a node holding something else",
        h.wrong.get()[0]);
  CHECK(h.dry.get()[0] == 0ull, "%llu keys went unserved", h.dry.get()[0]);
  h.check_at_rest("after evicting and carrying on");
}

// ---------------------------------------------------------------------------
// 7. Rows: the whole cycle, with values — one table, then several
// ---------------------------------------------------------------------------
//
// Every step "trains" what it touched: each key's row gets +1. A key starts as a
// known function of itself, so after any run of steps, evictions and reloads,
// its latest value is exactly start + (number of steps that touched it) — and
// "latest" means the cache row if it is cached, the backing row if it is not.
// A lost write-back, a stale load, a row overwritten before it was copied out,
// or a row published before it was initialised all show up as a count that does
// not add up. The values are small integers, exact in float.
//
// The backing is a plain buffer with one row per key, and which keys have a
// history in it is kept on the host: a key gets one by being written back, and
// loses it by being evicted from the system. The row numbers the kernels take are
// worked out on the host between kernels, which is the position the integration
// is in too — the backing finds them with its own kernels.
//
// With several tables the same key value appears in each, with a different start
// value per table and a different row width, so a lookup, a node or a row that
// crossed tables reads as a wrong value.

using FRows = Rows<float>;

// New keys' rows, written in place. This is what the embedding initialiser and
// the optimizer do in the integration. `tag` is the table's offset.
__global__ void init_fresh_kernel(FRows cache, float tag, const Slot* slots,
                                  const unsigned long long* count,
                                  const int64_t* new_row) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = static_cast<int>(tid % kLanesPerRow);
  const int64_t groups = int64_t(gridDim.x) * blockDim.x / kLanesPerRow;
  for (int64_t s = tid / kLanesPerRow; s < n; s += groups) {
    const Slot slot = slots[s];
    if (slot.node == Map::kNoNode || new_row[s] >= 0) continue;
    for (int j = lane; j < cache.row_elems; j += kLanesPerRow) {
      cache.row(slot.node)[j] = tag + 1000.0f * slot.key + j;
    }
  }
}

// What a step does to what it touched.
__global__ void train_kernel(FRows cache, const int32_t* node_ids, int64_t n) {
  const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int lane = static_cast<int>(tid % kLanesPerRow);
  const int64_t groups = int64_t(gridDim.x) * blockDim.x / kLanesPerRow;
  for (int64_t i = tid / kLanesPerRow; i < n; i += groups) {
    const int32_t node = node_ids[i];
    if (node == Map::kNoNode) continue;
    for (int j = lane; j < cache.row_elems; j += kLanesPerRow) {
      cache.row(node)[j] += 1.0f;
    }
  }
}

// One table with its row pool, its backing rows, and the host's model of both.
struct ValueTable {
  const int64_t per_step, space;
  const int row_elems;
  const float tag;
  Harness h;
  Dev<float> d_pool, d_backing;
  Dev<int64_t> d_new_row, d_victim_row;
  std::vector<char> history;      // does the backing hold this key's row
  std::vector<int64_t> touches;   // steps since the key was last new
  int64_t total_miss = 0, total_victims = 0, total_loaded = 0;

  ValueTable(int64_t cap, int64_t per_step_, int64_t space_, int row_elems_,
             float tag_)
      : per_step(per_step_), space(space_), row_elems(row_elems_), tag(tag_),
        h(cap, per_step_), d_pool(cap * row_elems_),
        d_backing(space_ * row_elems_), d_new_row(per_step_),
        d_victim_row(per_step_), history(space_, 0), touches(space_, 0) {}

  FRows pool() const { return FRows{d_pool.p, row_elems}; }
  FRows backing() const { return FRows{d_backing.p, row_elems}; }
  float expected(int64_t key, int j) const {
    return tag + 1000.0f * key + j + float(touches[key]);
  }
  // The key has left the system: no history, and whatever it was is gone.
  void forget(int64_t key) {
    history[key] = 0;
    touches[key] = 0;
  }

  void step(const std::vector<int64_t>& step_keys) {
    const int warp_grid = grid_for(per_step * kLanesPerRow);
    h.lookup_and_allocate(step_keys);
    const int64_t n_miss = static_cast<int64_t>(h.miss_count.get()[0]);
    const auto slots = h.slots.get();

    std::vector<int64_t> new_row(per_step, -1), victim_row(per_step, -1);
    for (int64_t s = 0; s < n_miss; ++s) {
      CHECK(slots[s].node != Map::kNoNode,
            "the cache is large enough to serve every miss");
      new_row[s] = history[slots[s].key] ? slots[s].key : -1;
      victim_row[s] = slots[s].has_victim ? slots[s].victim_key : -1;
    }
    d_new_row.put(new_row);
    d_victim_row.put(victim_row);

    writeback_kernel<Map, float><<<warp_grid, kBlock>>>(
        pool(), backing(), h.slots.p, h.miss_count.p, d_victim_row.p);
    fill_kernel<Map, float><<<warp_grid, kBlock>>>(
        pool(), backing(), h.slots.p, h.miss_count.p, d_new_row.p);
    init_fresh_kernel<<<warp_grid, kBlock>>>(pool(), tag, h.slots.p,
                                             h.miss_count.p, d_new_row.p);
    publish_kernel<Map, Node><<<h.grid_slots(), kBlock>>>(h.nodes, h.slots.p,
                                                          h.miss_count.p);
    train_kernel<<<warp_grid, kBlock>>>(pool(), h.node_ids.p,
                                        static_cast<int64_t>(step_keys.size()));
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    h.release();

    // The host's model of what that did.
    for (int64_t s = 0; s < n_miss; ++s) {
      if (slots[s].has_victim) {
        history[slots[s].victim_key] = 1;
        ++total_victims;
      }
      if (new_row[s] >= 0) ++total_loaded;
    }
    for (int64_t k : step_keys) ++touches[k];
    total_miss += n_miss;
  }

  // A checkpoint: every resident row written back. The row numbers are looked up
  // on the host, as they are for victims. Returns how many nodes held a
  // reference, which must be none.
  int64_t flush() {
    const int64_t cap = h.capacity;
    Dev<Slot> d_slots(cap);
    Dev<unsigned long long> d_count(1), d_referenced(1);
    Dev<int64_t> d_row(cap);
    resident_slots_kernel<Map, Node><<<grid_for(cap), kBlock>>>(
        h.nodes, cap, d_slots.p, d_count.p, d_referenced.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());

    const int64_t n = static_cast<int64_t>(d_count.get()[0]);
    const auto slots = d_slots.get();
    std::vector<int64_t> row(cap, -1);
    for (int64_t s = 0; s < n; ++s) row[s] = slots[s].victim_key;
    d_row.put(row);

    writeback_kernel<Map, float><<<grid_for(cap * kLanesPerRow), kBlock>>>(
        pool(), backing(), d_slots.p, d_count.p, d_row.p);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    for (int64_t s = 0; s < n; ++s) history[slots[s].key] = 1;
    return static_cast<int64_t>(d_referenced.get()[0]);
  }

  // After a flush the backing alone holds the latest of every key touched — or
  // of just `only`, when given.
  int64_t check_backing(const char* when,
                        const std::vector<int64_t>* only = nullptr) {
    const auto backing_rows = d_backing.get();
    std::set<int64_t> wanted;
    if (only) wanted.insert(only->begin(), only->end());
    int64_t bad = 0, checked = 0;
    for (int64_t k = 0; k < space; ++k) {
      if (touches[k] == 0 || (only && !wanted.count(k))) continue;
      ++checked;
      if (!history[k]) { ++bad; continue; }
      for (int j = 0; j < row_elems; ++j) {
        if (backing_rows[k * row_elems + j] != expected(k, j)) { ++bad; break; }
      }
    }
    std::printf("  %s: %lld keys, read from the backing only, %lld wrong\n", when,
                (long long)checked, (long long)bad);
    CHECK(bad == 0, "%lld keys are behind in the backing (%s)", (long long)bad,
          when);
    return bad;
  }

  // The latest value of every key touched, from wherever it is.
  int64_t check_all(const char* when) {
    const auto cache_rows = d_pool.get();
    const auto backing_rows = d_backing.get();
    std::vector<int64_t> all_keys;
    for (int64_t k = 0; k < space; ++k)
      if (touches[k] > 0) all_keys.push_back(k);
    const auto where = h.find(all_keys);
    int64_t bad = 0, from_cache = 0;
    for (size_t i = 0; i < all_keys.size(); ++i) {
      const int64_t k = all_keys[i];
      const float* row;
      if (where[i] != Map::kNoNode) {
        row = &cache_rows[where[i] * row_elems];
        ++from_cache;
      } else {
        if (!history[k]) { ++bad; continue; }  // neither cached nor written back
        row = &backing_rows[k * row_elems];
      }
      for (int j = 0; j < row_elems; ++j) {
        if (row[j] != expected(k, j)) { ++bad; break; }
      }
    }
    std::printf("  %s: %zu keys touched, %lld read from the cache, %lld wrong\n",
                when, all_keys.size(), (long long)from_cache, (long long)bad);
    CHECK(bad == 0, "%lld keys do not hold start + touches (%s)", (long long)bad,
          when);
    return bad;
  }
};

// All the tables' maps and row pools as the arrays the fused kernels take.
struct Fused {
  std::vector<ValueTable*> tables;
  Dev<Map> maps;
  Dev<FRows> rows;

  explicit Fused(const std::vector<ValueTable*>& t)
      : tables(t), maps(t.size()), rows(t.size()) {
    std::vector<Map> m;
    std::vector<FRows> r;
    for (ValueTable* v : t) {
      m.push_back(v->h.map_store.map());
      r.push_back(v->pool());
    }
    maps.put(m);
    rows.put(r);
  }
};

using KeyOf = std::pair<int, int64_t>;  // (table, key)

// What the guard says about each (table, key), in the order given.
static std::vector<char> probe_guard(Fused& f, const std::vector<KeyOf>& list) {
  std::vector<int64_t> keys, tids;
  for (const KeyOf& e : list) {
    tids.push_back(e.first);
    keys.push_back(e.second);
  }
  Dev<int64_t> d_keys(keys.size()), d_tid(keys.size());
  Dev<char> d_may(keys.size());
  d_keys.put(keys);
  d_tid.put(tids);
  guard_probe_kernel<<<64, 256>>>(Guard{f.maps.p}, d_keys.p, d_tid.p,
                                  static_cast<int64_t>(keys.size()), d_may.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());
  return d_may.get();
}

// The order the cache runs when the backing evicts: find, fetch, invalidate. The
// list is cached keys from any tables in any order, as the backing hands them
// over. What is fetched must be the latest value; afterwards each key starts over.
// Returns how many fetched rows were wrong.
static int64_t evict_and_fetch(Fused& f, const std::vector<KeyOf>& list) {
  const int64_t n = static_cast<int64_t>(list.size());
  std::vector<int64_t> keys, tids, offset;
  int64_t total = 0;
  for (const KeyOf& e : list) {
    keys.push_back(e.second);
    tids.push_back(e.first);
    offset.push_back(total);
    total += f.tables[e.first]->row_elems;
  }
  Dev<int64_t> d_keys(n), d_tid(n), d_off(n);
  Dev<int32_t> d_nodes(n);
  Dev<float> d_out(total);
  d_keys.put(keys);
  d_tid.put(tids);
  d_off.put(offset);

  find_nodes_kernel<Map><<<64, 256>>>(f.maps.p, d_keys.p, d_tid.p, n, d_nodes.p);
  fetch_rows_kernel<Map, float><<<grid_for(n * kLanesPerRow), kBlock>>>(
      f.rows.p, d_nodes.p, d_tid.p, n, d_out.p, d_off.p);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());

  const auto nodes_found = d_nodes.get();
  const auto fetched = d_out.get();
  int64_t bad = 0;
  for (int64_t i = 0; i < n; ++i) {
    const ValueTable& t = *f.tables[list[i].first];
    if (nodes_found[i] == Map::kNoNode) { ++bad; continue; }
    for (int j = 0; j < t.row_elems; ++j) {
      if (fetched[offset[i] + j] != t.expected(list[i].second, j)) {
        ++bad;
        break;
      }
    }
  }

  invalidate_kernel<Map><<<64, 256>>>(f.maps.p, d_nodes.p, d_tid.p, n);
  CU(cudaGetLastError());
  CU(cudaDeviceSynchronize());
  for (const KeyOf& e : list) f.tables[e.first]->forget(e.second);
  return bad;
}

static void test_value_cycle() {
  std::printf("=== rows: the whole cycle, with values ===\n");
  // Smaller than the working set, on purpose: a cache that holds every key it
  // sees never evicts, and then neither direction of the move is exercised.
  ValueTable t(384, 128, 3000, 8, 0.0f);
  Fused f({&t});

  std::mt19937_64 rng(11);
  std::geometric_distribution<int64_t> hot(0.004);

  for (int s = 0; s < scaled(150); ++s) t.step(draw_keys(rng, hot, t.per_step, t.space));
  std::printf("  %lld misses, %lld victims written back, %lld rows loaded\n",
              (long long)t.total_miss, (long long)t.total_victims,
              (long long)t.total_loaded);
  CHECK(t.total_victims > 0 && t.total_loaded > 0,
        "the run has to exercise both directions");
  t.check_all("after the first run of steps");

  // A flush with a step still in flight must notice: hold some cached keys, count.
  {
    const auto cached_now = t.h.bound_keys();
    const std::vector<int64_t> held_keys(cached_now.begin(), cached_now.begin() + 20);
    t.h.lookup_and_allocate(held_keys);  // all hits: pins, binds nothing
    const int64_t referenced = t.flush();
    CHECK(referenced == 20, "flush should count the 20 nodes still referenced, "
                            "got %lld", (long long)referenced);
    // And it must still have written them: a hot key that never leaves the cache
    // is the one whose backing copy is furthest behind.
    t.check_backing("held keys, flushed while referenced", &held_keys);
    t.h.release();
  }
  CHECK(t.flush() == 0, "with every step released, no node holds a reference");
  t.check_backing("after a flush");

  // Evict some cached keys from the system, taking their values off, then ask for
  // them again so they come back as new keys.
  std::vector<KeyOf> evict;
  const auto cached = t.h.bound_keys();
  for (size_t i = 0; i < cached.size(); i += 4) evict.push_back({0, cached[i]});
  const int64_t bad = evict_and_fetch(f, evict);
  std::printf("  fetched %zu rows off nodes about to be invalidated, %lld wrong\n",
              evict.size(), (long long)bad);
  CHECK(bad == 0, "%lld fetched rows are not the latest value", (long long)bad);

  std::vector<int64_t> again;
  for (const KeyOf& e : evict) {
    again.push_back(e.second);
    if ((int64_t)again.size() == t.per_step) break;
  }
  t.step(again);
  for (int s = 0; s < scaled(100); ++s) t.step(draw_keys(rng, hot, t.per_step, t.space));
  t.check_all("after evicting and more steps");
  t.h.check_at_rest("value cycle");
}

// ---------------------------------------------------------------------------
// 8. Several tables
// ---------------------------------------------------------------------------
//
// Three tables with different capacities, row widths and key traffic, the same
// key values in all of them. Everything per table — lookup, allocate, the moves,
// release — runs per table, on the kernels the single table uses; what runs over
// all of them at once is what the backing's eviction needs: the guard, find,
// fetch and invalidate, fed lists that mix the tables.

static void test_multi_table() {
  std::printf("=== several tables ===\n");
  const int64_t kSpace = 600;
  const int kTables = 3;
  const int64_t caps[kTables] = {160, 256, 96};
  const int64_t per_step[kTables] = {48, 64, 32};
  const int widths[kTables] = {4, 8, 12};

  std::vector<std::unique_ptr<ValueTable>> owned;
  std::vector<ValueTable*> tables;
  for (int i = 0; i < kTables; ++i) {
    owned.emplace_back(new ValueTable(caps[i], per_step[i], kSpace, widths[i],
                                      100000.0f * i));
    tables.push_back(owned.back().get());
  }
  Fused f(tables);

  std::mt19937_64 rng(23);
  std::geometric_distribution<int64_t> hot(0.01);
  auto all_tables_step = [&]() {
    for (ValueTable* t : tables) t->step(draw_keys(rng, hot, t->per_step, kSpace));
  };

  // Not scaled below a floor: a table that never evicts exercises neither
  // direction of the move, and the check below says so.
  for (int s = 0; s < std::max(scaled(120), 60); ++s) all_tables_step();
  for (int i = 0; i < kTables; ++i) {
    std::printf("  table %d: %lld misses, %lld victims, %lld loaded\n", i,
                (long long)tables[i]->total_miss,
                (long long)tables[i]->total_victims,
                (long long)tables[i]->total_loaded);
    CHECK(tables[i]->total_victims > 0 && tables[i]->total_loaded > 0,
          "table %d has to exercise both directions", i);
    tables[i]->check_all("after the first run");
  }

  // --- the guard over a list that mixes the tables ---------------------------
  //
  // Hold some cached keys in every table; the others are idle. The same key value
  // is cached in several tables, held in one and idle in another, so an answer
  // taken from the wrong table is wrong.
  {
    std::vector<std::set<int64_t>> held(kTables);
    std::vector<std::vector<int64_t>> cached(kTables);
    for (int i = 0; i < kTables; ++i) {
      cached[i] = tables[i]->h.bound_keys();
      std::vector<int64_t> keys(cached[i].begin(), cached[i].begin() + 24);
      held[i].insert(keys.begin(), keys.end());
      tables[i]->h.lookup_and_allocate(keys);  // all hits: pins, binds nothing
    }
    std::vector<KeyOf> probe;
    size_t longest = 0;
    for (int i = 0; i < kTables; ++i) longest = std::max(longest, cached[i].size());
    for (size_t k = 0; k < longest; ++k) {  // interleave the tables
      for (int i = 0; i < kTables; ++i)
        if (k < cached[i].size()) probe.push_back({i, cached[i][k]});
    }
    for (int i = 0; i < kTables; ++i)
      for (int64_t k = 0; k < 30; ++k) probe.push_back({i, kSpace + k});

    const auto may = probe_guard(f, probe);
    int64_t wrong = 0, vetoed = 0;
    for (size_t i = 0; i < probe.size(); ++i) {
      const bool expect = !held[probe[i].first].count(probe[i].second);
      if ((may[i] != 0) != expect) ++wrong;
      if (!may[i]) ++vetoed;
    }
    std::printf("  guard: %zu (table, key) pairs probed, %lld vetoed\n",
                probe.size(), (long long)vetoed);
    CHECK(wrong == 0, "%lld pairs got the wrong answer from the guard",
          (long long)wrong);
    CHECK(vetoed == 24 * kTables, "expected %d vetoes, got %lld", 24 * kTables,
          (long long)vetoed);
    for (ValueTable* t : tables) t->h.release();
  }

  // --- evict a mixed list, taking the values off ------------------------------
  {
    std::vector<std::vector<int64_t>> cached(kTables);
    size_t longest = 0;
    for (int i = 0; i < kTables; ++i) {
      cached[i] = tables[i]->h.bound_keys();
      longest = std::max(longest, cached[i].size());
    }
    std::vector<KeyOf> evict;
    for (size_t k = 0; k < longest; k += 3) {  // interleaved, a third of each
      for (int i = 0; i < kTables; ++i)
        if (k < cached[i].size()) evict.push_back({i, cached[i][k]});
    }
    const int64_t bad = evict_and_fetch(f, evict);
    std::printf("  fetched %zu rows across %d tables, %lld wrong\n", evict.size(),
                kTables, (long long)bad);
    CHECK(bad == 0, "%lld fetched rows are not the latest value", (long long)bad);
  }

  for (int s = 0; s < scaled(80); ++s) all_tables_step();
  for (int i = 0; i < kTables; ++i) {
    tables[i]->check_all("after evicting and more steps");
    tables[i]->h.check_at_rest("multi table");
    CHECK(tables[i]->flush() == 0, "table %d: no node should hold a reference", i);
    tables[i]->check_backing("after a flush");
  }
}

int main() {
  int dev = 0;
  CU(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CU(cudaGetDeviceProperties(&prop, dev));
  std::printf("device: %s, sm_%d%d, %d SMs\n\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  test_state_contention();
  test_claim_exclusivity();
  test_queue_round_trip();
  test_release_overlaps_allocate();
  test_evict_by_key();
  test_value_cycle();
  test_multi_table();
  // Capacity below the working set on purpose: eviction has to actually happen.
  test_pipeline(/*steps=*/200, /*capacity=*/2048, /*per_step=*/512,
                /*key_space=*/8192, /*seed=*/1234);
  test_pipeline(/*steps=*/50, /*capacity=*/600, /*per_step=*/500,
                /*key_space=*/4096, /*seed=*/99);

  if (failures == 0) {
    std::printf("\nall GPU checks passed\n");
    return 0;
  }
  std::printf("\n%d check(s) failed\n", failures);
  return 1;
}
