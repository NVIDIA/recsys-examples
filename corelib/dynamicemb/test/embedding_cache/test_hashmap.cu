// Exercise of HashMap. Needs nvcc to build and a GPU to run.
//
// One thread per operation, so this validates structure — lookup,
// head/middle/tail removal, collisions, churn, the key<->node binding — and not
// the bucket lock under contention. That still needs a test of its own.
#include <cstdio>
#include <limits>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "hashmap.h"

using namespace embedding_cache;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

// What the cache would define. The map's three fields come from the base; the
// rest is whatever the other components need, with `state` standing in for the
// concurrency word. The point is that HashMap neither knows nor cares that it
// is there — and that the key and node-id types are stated once, on the base,
// so HashMapStorage<Node> needs no second telling.
struct Node : HashMapNode<int64_t, int32_t> {
  uint32_t state;
};

using Storage = HashMapStorage<Node>;
using Map = Storage::Map;

// ---------------------------------------------------------------------------
// Reaching the device from a host test
// ---------------------------------------------------------------------------
//
// lookup, insert and remove are __device__. This file used to call them
// directly, because it was built against a host stand-in for the CUDA runtime
// where __device__ expanded to nothing. It is a real .cu now, so each one needs
// a launch. One block, one thread: these are structural tests — chains,
// removal, collisions, churn — and none of them is about concurrency. The
// bucket lock under contention is still unverified here, as it always was.

__global__ void k_insert(Map m, int32_t node, int64_t key) {
  m.insert(node, key);
}
__global__ void k_remove(Map m, int32_t node) { m.remove(node); }
__global__ void k_lookup(Map m, int64_t key, int32_t* out) {
  *out = m.lookup(key);
}

static void d_insert(Map m, int32_t node, int64_t key) {
  k_insert<<<1, 1>>>(m, node, key);
  cudaDeviceSynchronize();
}
static void d_remove(Map m, int32_t node) {
  k_remove<<<1, 1>>>(m, node);
  cudaDeviceSynchronize();
}
static int32_t d_lookup(Map m, int64_t key) {
  // One device word for the whole process, deliberately never freed: it is
  // reached from every check in the file and outlives all of them.
  static int32_t* out = [] {
    int32_t* p = nullptr;
    cudaMalloc(&p, sizeof(int32_t));
    return p;
  }();
  k_lookup<<<1, 1>>>(m, key, out);
  int32_t host = Storage::kNoNode;
  cudaMemcpy(&host, out, sizeof(int32_t), cudaMemcpyDeviceToHost);
  return host;
}

// Corrupt one node from the host, to check that the sweep notices. The node
// array is device memory, so this is a read-modify-write and not a store —
// another thing that stopped being true when this became a .cu.
template <typename F>
static void poke_node(Storage& s, int32_t node, F mutate) {
  Node n{};
  cudaMemcpy(&n, s.nodes() + node, sizeof(Node), cudaMemcpyDeviceToHost);
  mutate(n);
  cudaMemcpy(s.nodes() + node, &n, sizeof(Node), cudaMemcpyHostToDevice);
}

constexpr uint32_t kStateMarker = 0xABCDu;

// Stamp the field that belongs to another component. The storage allocates the
// node array but cannot initialise this — it has no idea what a state word
// should hold — so the owner of that field does it, through nodes(). Every
// later check that the marker survived is a check that the map stayed out of
// fields that are not its own.
template <typename StorageT>
static void mark_states(StorageT& s) {
  auto staged = s.download_nodes();
  for (auto& n : staged) n.state = kStateMarker;
  cudaMemcpy(s.nodes(), staged.data(), sizeof(Node) * staged.size(),
             cudaMemcpyHostToDevice);
}

// Wraps CudaBuffer and tallies what passes through, to show that the storage
// routes every allocation to the allocator it was given — and, just as much,
// that it does not allocate what is not its to allocate.
struct CountingAllocator {
  struct Buffer {
    Buffer() = default;
    explicit Buffer(size_t bytes) : inner(bytes) {
      if (bytes) {
        ++allocations;
        ++live;
        CountingAllocator::bytes += bytes;
      }
    }
    ~Buffer() {
      if (inner.data()) --live;
    }
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&& o) {
      if (this != &o) {
        if (inner.data()) --live;
        inner = std::move(o.inner);
      }
      return *this;
    }
    void* data() const { return inner.data(); }
    size_t bytes() const { return inner.bytes(); }
    CudaBuffer inner;
  };

  Buffer allocate(size_t n) const { return Buffer(n); }

  static void reset_counters() { allocations = bytes = live = 0; }
  static size_t allocations;
  static size_t bytes;
  static size_t live;
};
size_t CountingAllocator::allocations = 0;
size_t CountingAllocator::bytes = 0;
size_t CountingAllocator::live = 0;

using CountedStorage =
    HashMapStorage<Node, Murmur3Hash64, int32_t(-1), CountingAllocator>;

// Walk a bucket, checking the links agree in both directions and there is no
// cycle. Returns the nodes in order.
static std::vector<int32_t> walk(const Storage& s, int64_t bucket, bool* ok) {
  const auto heads = s.download_heads();
  const auto nodes = s.download_nodes();
  std::vector<int32_t> out;
  *ok = true;
  int32_t n = heads[static_cast<size_t>(bucket)];
  int32_t expect_prev = Storage::kNoNode;
  int guard = 0;
  while (n != Storage::kNoNode) {
    if (nodes[static_cast<size_t>(n)].prev != expect_prev) *ok = false;
    out.push_back(n);
    expect_prev = n;
    n = nodes[static_cast<size_t>(n)].next;
    if (++guard > static_cast<int>(s.capacity()) + 1) {  // a cycle would hang
      *ok = false;
      break;
    }
  }
  return out;
}

static bool all_buckets_consistent(const Storage& s) {
  for (int64_t b = 0; b < s.num_buckets(); ++b) {
    bool ok = false;
    walk(s, b, &ok);
    if (!ok) return false;
  }
  return true;
}

// Keys that share a bucket, found by construction rather than by luck.
static std::vector<int64_t> colliding_keys(const Map& map, int count) {
  std::vector<int64_t> out;
  const int64_t target = map.bucket_of(1);
  for (int64_t k = 1; static_cast<int>(out.size()) < count && k < 4000000; ++k) {
    if (map.bucket_of(k) == target) out.push_back(k);
  }
  return out;
}

int main() {
  // --- the hash matches the rest of the repository ------------------------
  //
  // Expected values from dynamicemb/scored_hashtable.py's murmur3_hash_64bits,
  // the same finalizer table_operation/types.cuh and
  // sparse_block_bucketize_features.cu use. If these diverge, keys land in
  // different buckets here than everywhere else — harmless on its own, but it
  // means the four copies have drifted, which is worth knowing.
  {
    const struct {
      int64_t key;
      uint64_t want;
    } cases[] = {
        {0LL, UINT64_C(0x0000000000000000)},
        {1LL, UINT64_C(0xb456bcfc34c2cb2c)},
        {2LL, UINT64_C(0x3abf2a20650683e7)},
        {42LL, UINT64_C(0x810879608e4259cc)},
        {1000LL, UINT64_C(0xaf002c114878da41)},
        {-1LL, UINT64_C(0x64b5720b4b825f21)},
        {123456789LL, UINT64_C(0x8f7c29206384f886)},
        {1099511627776LL, UINT64_C(0xe80f632ed2f232f6)},
    };
    Murmur3Hash64 h;
    for (const auto& c : cases) {
      CHECK(h(c.key) == c.want, "matches the repository's murmur3_hash_64bits");
    }
  }

  // --- device properties are reachable -------------------------------------
  //
  // HashMapStorage::grid_for is private and stays that way — the grid is a
  // launch detail, not part of anything a caller decides. What is worth
  // checking here is only that the singleton it reads answers at all, since
  // linking utils.cpp into this build is the one dependency embedding_cache has
  // on the wider tree. The values below come from the shim, not a device.
  {
    const auto& p = dyn_emb::DeviceProp::getDeviceProp();
    CHECK(p.num_sms > 0, "DeviceProp reports an SM count");
    CHECK(p.max_thread_per_sm >= 1024,
          "and threads per SM, which grid_for's static_assert leans on being "
          "at least the maximum block size");
  }

  // --- the map is two pointers and a count --------------------------------
  {
    CHECK(sizeof(Map) == 2 * sizeof(void*) + sizeof(int64_t),
          "the map is exactly the bucket heads, the node pool and the count");
    CHECK(std::is_trivially_copyable<Map>::value,
          "the map is trivially copyable: it reaches kernels by value");
    CHECK((std::is_same<Map::Key, int64_t>::value),
          "the key type is deduced from the node base");
    CHECK((std::is_same<Map::NodeId, int32_t>::value),
          "the node id type is deduced from the node base too, so neither is "
          "stated twice");
  }

  // --- deriving costs the node nothing ------------------------------------
  {
    using Base = HashMapNode<int64_t, int32_t>;
    CHECK(sizeof(Base) == 16,
          "the base is key + prev + next with no tail padding, so a derived "
          "node loses nothing to a hole it cannot use");
    CHECK(sizeof(Node) == 24,
          "int64 key, two int32 links, an int32 state word, rounded to the "
          "8-byte alignment the key imposes");
    CHECK(std::is_trivially_copyable<Node>::value,
          "a derived node is still trivially copyable: cudaMemcpy of the node "
          "array depends on it");
    CHECK(std::is_trivially_default_constructible<Node>::value,
          "and still trivially default constructible: clear_links stages the "
          "array through a std::vector");
    CHECK((std::is_base_of<Base, Node>::value), "Node derives from the base");

    // The whole point of deriving rather than holding: the map's fields are
    // reached without naming the base.
    Node n{};
    n.key = 7;
    n.prev = 1;
    n.next = 2;
    n.state = 3;
    CHECK(n.key == 7 && n.prev == 1 && n.next == 2 && n.state == 3,
          "node.key, not node.chain.key");
  }

  // --- the storage owns the buckets and nothing else ----------------------
  {
    CountingAllocator::reset_counters();
    {
      CountedStorage s(100);
      CHECK(CountingAllocator::allocations == 2,
            "the bucket heads and the node array, and nothing else");
      // 256 buckets * 4  +  100 nodes * 24
      CHECK(CountingAllocator::bytes == 256u * 4 + 100u * sizeof(Node),
            "the allocator saw exactly the bytes the storage reports");
      CHECK(CountingAllocator::bytes == s.bytes(), "and bytes() agrees");
      CHECK(CountingAllocator::live == 2, "both still held");
    }
    CHECK(CountingAllocator::live == 0, "and both released on destruction");
  }

  // --- sizing and initial state -------------------------------------------
  {
    Storage s(100);
    mark_states(s);
    CHECK(s.capacity() == 100, "capacity kept as given");
    CHECK(s.num_buckets() == 256, "buckets default to next_pow2(2C)");
    CHECK(s.nodes() != nullptr, "the node array is allocated");

    for (int32_t h : s.download_heads()) {
      CHECK(h == Storage::kNoNode, "every bucket starts empty");
    }
    for (const auto& n : s.download_nodes()) {
      CHECK(n.prev == Storage::kNoNode && n.next == Storage::kNoNode,
            "every node starts unlinked — the links are all the map sets, and "
            "node.key is left meaningless until insert() writes it");
      CHECK(n.state == kStateMarker,
            "and the map left the fields that are not its own alone");
    }

    Storage explicit_buckets(100, 1000);
    CHECK(explicit_buckets.num_buckets() == 1024, "explicit count rounds up");

    const int64_t n = s.map().num_buckets;
    CHECK(n == s.num_buckets(), "the map and the storage agree on the count");
    CHECK((n & (n - 1)) == 0, "the count is 2^k, so masking spreads");
    for (int64_t k = -1000; k < 1000; ++k) {
      const int64_t b = s.map().bucket_of(k);
      CHECK(b >= 0 && b < s.num_buckets(), "bucket_of stays in range");
    }
  }

  // --- validation ----------------------------------------------------------
  {
    bool threw = false;
    try {
      Storage bad(0);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw, "zero capacity rejected");
  }

  // --- empty map -----------------------------------------------------------
  {
    Storage s(8);
    mark_states(s);
    CHECK(d_lookup(s.map(), 42) == Storage::kNoNode,
          "lookup in an empty map misses");
  }

  // --- bind, resolve, unbind ----------------------------------------------
  {
    Storage s(64);
    mark_states(s);
    Map map = s.map();

    for (int32_t n = 0; n < 20; ++n) d_insert(map, n, static_cast<int64_t>(1000 + n));

    for (int32_t n = 0; n < 20; ++n) {
      const int64_t k = static_cast<int64_t>(1000 + n);
      CHECK(d_lookup(map, k) == n, "every bound key resolves");
      // No key_of / holds on the map: the node array is the caller's, and
      // node.key is a public field. This is what those calls would have been.
      CHECK(s.download_nodes()[static_cast<size_t>(n)].key == k,
            "the node holds the key it was bound to");
    }
    CHECK(d_lookup(map, 999) == Storage::kNoNode, "an unbound key misses");
    CHECK(all_buckets_consistent(s), "links agree in both directions");
    CHECK(s.check_bindings() == 0, "every binding agrees with the buckets");

    for (const auto& n : s.download_nodes()) {
      CHECK(n.state == kStateMarker, "insert left the state word alone");
    }

    for (int32_t n = 0; n < 20; ++n) {
      d_remove(map, n);
      CHECK(d_lookup(map, static_cast<int64_t>(1000 + n)) == Storage::kNoNode,
            "an unbound key no longer resolves");
      CHECK(all_buckets_consistent(s), "chains stay consistent through removal");
    }
    for (int32_t h : s.download_heads()) {
      CHECK(h == Storage::kNoNode, "all buckets empty again");
    }
    for (const auto& n : s.download_nodes()) {
      CHECK(n.prev == Storage::kNoNode && n.next == Storage::kNoNode,
            "no dangling chain pointers");
    }
  }

  // --- rebinding a node ----------------------------------------------------
  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    d_insert(map, 0, 100);
    d_insert(map, 1, 200);

    d_remove(map, 0);
    d_insert(map, 0, 300);

    CHECK(d_lookup(map, 100) == Storage::kNoNode, "the old key no longer resolves");
    CHECK(d_lookup(map, 300) == 0, "the new key resolves to the same node");
    CHECK(s.download_nodes()[0].key == 300, "the binding followed");
    CHECK(d_lookup(map, 200) == 1, "the untouched node is unaffected");
    CHECK(all_buckets_consistent(s), "chains consistent after a rebind");
    CHECK(s.check_bindings() == 0, "bindings still agree");
  }

  // --- a key written behind the map's back is detected ---------------------
  //
  // Sharing the node type with other components means this is now expressible,
  // where it was unreachable when the map owned the key array. Detection is
  // what is left.
  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    d_insert(map, 0, 100);
    CHECK(s.check_bindings() == 0, "sound to begin with");

    poke_node(s, 0, [](Node& n) { n.key = 999; });  // no bucket move
    CHECK(s.check_bindings() == 1,
          "a key written without moving buckets is caught by the sweep");
  }

  // --- the sweep sees more than a mis-hashed key ---------------------------
  //
  // Driving it from the buckets rather than from the nodes is what makes these
  // two visible. The node-driven form asked "is this node where its key says it
  // should be", which cannot notice a chain disagreeing with itself, and needed
  // a reserved key value to know which nodes to skip.
  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    const auto ks = colliding_keys(map, 2);
    d_insert(map, 0, ks[0]);
    d_insert(map, 1, ks[1]);  // head = 1 -> 0
    CHECK(s.check_bindings() == 0, "sound to begin with");

    // forward link still says 1 -> 0
    poke_node(s, 0, [](Node& n) { n.prev = Storage::kNoNode; });
    CHECK(s.check_bindings() == 1, "a back pointer that disagrees is caught");
  }

  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    d_insert(map, 0, 100);
    poke_node(s, 0, [](Node& n) { n.next = 0; });  // points at itself
    CHECK(s.check_bindings() >= 1, "a cycle is reported, not walked forever");
  }

  // --- one bucket, long chain ---------------------------------------------
  {
    Storage s(64, 16);
    mark_states(s);
    Map map = s.map();
    const auto ks = colliding_keys(map, 6);
    CHECK(ks.size() == 6, "found six keys sharing a bucket");
    const int64_t b = map.bucket_of(ks[0]);

    for (size_t i = 0; i < ks.size(); ++i) d_insert(map, static_cast<int32_t>(i), ks[i]);

    bool ok = false;
    auto chain = walk(s, b, &ok);
    CHECK(ok, "chain links agree in both directions");
    CHECK(chain.size() == 6, "all six landed in one bucket");
    CHECK(chain[0] == 5, "insertion is at the head, so newest is first");
    for (size_t i = 0; i < ks.size(); ++i) {
      CHECK(d_lookup(map, ks[i]) == static_cast<int32_t>(i),
            "each colliding key resolves");
    }

    d_remove(map, 3);  // middle
    chain = walk(s, b, &ok);
    CHECK(ok && chain.size() == 5, "middle removal keeps the chain intact");
    CHECK(d_lookup(map, ks[3]) == Storage::kNoNode, "removed key gone");
    CHECK(d_lookup(map, ks[2]) == 2 && d_lookup(map, ks[4]) == 4,
          "its neighbours are still reachable");

    const int32_t head = chain[0];
    d_remove(map, head);
    chain = walk(s, b, &ok);
    CHECK(ok && chain.size() == 4, "head removal keeps the chain intact");

    const int32_t tail = chain.back();
    d_remove(map, tail);
    chain = walk(s, b, &ok);
    CHECK(ok && chain.size() == 3, "tail removal keeps the chain intact");

    while (!chain.empty()) {
      d_remove(map, chain[0]);
      chain = walk(s, b, &ok);
      CHECK(ok, "chain consistent while draining");
    }
    CHECK(s.download_heads()[static_cast<size_t>(b)] == Storage::kNoNode,
          "bucket empty");
  }

  // --- churn leaves no residue --------------------------------------------
  {
    Storage s(32, 16);
    mark_states(s);
    Map map = s.map();
    for (int round = 0; round < 50; ++round) {
      for (int32_t n = 0; n < 32; ++n) {
        d_insert(map, n, static_cast<int64_t>(round * 1000 + n));
      }
      CHECK(all_buckets_consistent(s), "consistent after a round of inserts");
      for (int32_t n = 0; n < 32; ++n) d_remove(map, n);
      CHECK(all_buckets_consistent(s), "consistent after a round of removals");
    }
    for (int32_t h : s.download_heads()) {
      CHECK(h == Storage::kNoNode, "no residue after churn");
    }
    for (const auto& n : s.download_nodes()) {
      CHECK(n.state == kStateMarker, "fifty rounds and the state word is intact");
    }
  }

  // --- clearing --------------------------------------------------------------
  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    d_insert(map, 0, 12345);
    CHECK(d_lookup(map, 12345) == 0, "bound before clearing");

    s.clear_buckets();
    s.clear_links();
    CHECK(d_lookup(map, 12345) == Storage::kNoNode, "clearing dropped the binding");
    CHECK(s.download_nodes()[0].prev == Storage::kNoNode &&
              s.download_nodes()[0].next == Storage::kNoNode,
          "and the links, which is the whole of what a clear can undo");
    CHECK(s.download_nodes()[0].key == 12345,
          "the key is left as it was: no sentinel to write, and nothing can "
          "reach a node that is in no bucket");
    CHECK(s.download_nodes()[0].state == kStateMarker,
          "while leaving the state word to its owner");
    CHECK(s.check_bindings() == 0,
          "and a cleared map has nothing in any bucket to disagree with");
  }

  // --- the lock is released ------------------------------------------------
  {
    Storage s(16);
    mark_states(s);
    Map map = s.map();
    d_insert(map, 0, 55);
    for (int32_t h : s.download_heads()) {
      CHECK(h != Map::kLocked, "no bucket left locked after an operation");
    }
    CHECK(Map::kLocked != Map::kNoNode, "the sentinels are distinct");
    CHECK(d_lookup(map, 55) == 0, "lookup works");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
