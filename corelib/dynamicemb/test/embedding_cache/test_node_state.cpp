// Exercise of NodeState.
//
// A .cpp, not a .cu, and that is the payoff of splitting the value type out of
// the concurrency layer: NodeState has no kernels and no atomics, so every bit
// of it can be checked with a plain compiler and no device. Built with -UNDEBUG
// so the assertions in the header are live, then re-built with -DNDEBUG so the
// compiled-out form is checked too — see run_tests.sh.
#include <cstdio>
#include <set>
#include <type_traits>
#include <vector>

#include "node_state.h"

using namespace embedding_cache;
using R = NodeState::Residency;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

static const R kAllResidencies[] = {NodeState::kEmpty, NodeState::kLoading,
                                    NodeState::kResident};

int main() {
  // --- the type is what a node can hold -----------------------------------
  {
    CHECK(sizeof(NodeState) == 4, "one word, no more");
    CHECK(std::is_trivially_copyable<NodeState>::value,
          "trivially copyable: the node array moves with cudaMemcpy");
    CHECK(std::is_trivially_default_constructible<NodeState>::value,
          "and trivially default constructible: it is staged through a vector");
    CHECK((std::is_same<NodeState::Raw, uint32_t>::value),
          "the raw type is the word a node declares");
  }

  // --- the fields tile the word, and none of them bleeds -------------------
  //
  // Setting one field must leave the other three alone. That is the whole
  // correctness question for a packed word, and it is exhaustible: three
  // residencies x in_ring x used x a set of refs that covers both ends and the
  // boundaries.
  {
    const uint32_t refs_cases[] = {0u, 1u, 2u, 12345u, NodeState::kMaxRefs - 1u,
                                   NodeState::kMaxRefs};
    for (R r : kAllResidencies) {
      for (uint32_t refs : refs_cases) {
        for (int ir = 0; ir < 2; ++ir) {
          for (int us = 0; us < 2; ++us) {
            const NodeState s = NodeState::make(r, refs, ir != 0, us != 0);
            CHECK(s.residency() == r, "residency round-trips");
            CHECK(s.refs() == refs, "refs round-trips");
            CHECK(s.in_ring() == (ir != 0), "in_ring round-trips");
            CHECK(s.used() == (us != 0), "used round-trips");
            CHECK(s.has_refs() == (refs != 0), "has_refs agrees with refs");
            CHECK(NodeState::from_raw(s.raw()) == s, "raw round-trips");

            // Each single-field change leaves the rest untouched.
            for (R r2 : kAllResidencies) {
              const NodeState t = s.with_residency(r2);
              CHECK(t.residency() == r2 && t.refs() == refs &&
                        t.in_ring() == s.in_ring() && t.used() == s.used(),
                    "with_residency touches only residency");
            }
            for (int v = 0; v < 2; ++v) {
              const NodeState t = s.with_in_ring(v != 0);
              CHECK(t.in_ring() == (v != 0) && t.residency() == r &&
                        t.refs() == refs && t.used() == s.used(),
                    "with_in_ring touches only in_ring");
              const NodeState u = s.with_used(v != 0);
              CHECK(u.used() == (v != 0) && u.residency() == r &&
                        u.refs() == refs && u.in_ring() == s.in_ring(),
                    "with_used touches only used");
            }
            for (uint32_t n : refs_cases) {
              const NodeState t = s.with_refs(n);
              CHECK(t.refs() == n && t.residency() == r &&
                        t.in_ring() == s.in_ring() && t.used() == s.used(),
                    "with_refs touches only refs");
            }
          }
        }
      }
    }
  }

  // --- distinct field settings give distinct words -------------------------
  //
  // A masking mistake that made two settings collide would pass every check
  // above, since each one only ever looks at the field it set.
  {
    std::set<uint32_t> seen;
    int collisions = 0;
    for (R r : kAllResidencies) {
      for (uint32_t refs : {0u, 1u, 7u, NodeState::kMaxRefs}) {
        for (int ir = 0; ir < 2; ++ir) {
          for (int us = 0; us < 2; ++us) {
            if (!seen.insert(NodeState::make(r, refs, ir != 0, us != 0).raw())
                     .second) {
              ++collisions;
            }
          }
        }
      }
    }
    CHECK(collisions == 0, "every distinct setting has its own word");
    CHECK(seen.size() == 3u * 4u * 2u * 2u, "and all of them were reachable");
  }

  // --- pin and release are inverses, and carry nothing across fields -------
  {
    for (R r : kAllResidencies) {
      for (int ir = 0; ir < 2; ++ir) {
        const NodeState s = NodeState::make(r, 5u, ir != 0, /*used=*/false);

        const NodeState p = s.pinned();
        CHECK(p.refs() == 6u, "pin adds a reference");
        CHECK(p.used(), "and sets used, which is what makes CLOCK free");
        CHECK(p.residency() == r && p.in_ring() == s.in_ring(),
              "pin leaves residency and in_ring alone");

        CHECK(p.released().refs() == 5u, "release takes one back");
        CHECK(p.released() == s.with_used(true),
              "pin then release differs from the original only in used");
      }
    }
  }

  // --- the boundaries of the refs field ------------------------------------
  //
  // The interesting ones: the increment must not carry up into `used`, and the
  // decrement must not borrow down from it.
  {
    const NodeState hi =
        NodeState::make(NodeState::kResident, NodeState::kMaxRefs - 1u,
                        /*in_ring=*/true, /*used=*/true);
    const NodeState hi_pinned = hi.pinned();
    CHECK(hi_pinned.refs() == NodeState::kMaxRefs,
          "refs reaches its maximum without wrapping");
    CHECK(hi_pinned.residency() == NodeState::kResident &&
              hi_pinned.in_ring() && hi_pinned.used(),
          "and the increment carried nothing into the fields above");

    const NodeState one =
        NodeState::make(NodeState::kResident, 1u, /*in_ring=*/false,
                        /*used=*/true);
    const NodeState zero = one.released();
    CHECK(zero.refs() == 0u && !zero.has_refs(), "release to zero");
    CHECK(zero.residency() == NodeState::kResident && !zero.in_ring() &&
              zero.used(),
          "and the decrement borrowed nothing from the fields above");
  }

  // --- residency values are the two low bits, in order --------------------
  {
    CHECK(NodeState::kEmpty == 0u && NodeState::kLoading == 1u &&
              NodeState::kResident == 2u,
          "the three residencies fit two bits");
    CHECK(NodeState::make(NodeState::kResident, 0, false, false).raw() ==
              0x80000000u,
          "residency occupies the top two bits");
    CHECK(NodeState::make(NodeState::kEmpty, 0, true, false).raw() ==
              0x10000000u,
          "in_ring is bit 28");
    CHECK(NodeState::make(NodeState::kEmpty, 0, false, true).raw() ==
              0x08000000u,
          "used is bit 27");
    CHECK(NodeState::make(NodeState::kEmpty, NodeState::kMaxRefs, false, false)
                  .raw() == 0x07FFFFFFu,
          "refs is the low 27 bits");
    CHECK((NodeState::make(NodeState::kResident, NodeState::kMaxRefs, true, true)
               .raw() &
           NodeState::kReservedMask) == 0u,
          "bit 29 stays reserved even with every field at its maximum");
  }

  // --- what a caller would build at initialisation -------------------------
  //
  // Not a method on NodeState: whether a fresh node is already in the ring is
  // the ring's business, not this component's. This is what the caller writes.
  {
    const NodeState fresh = NodeState::make(NodeState::kEmpty, 0u,
                                            /*in_ring=*/true, /*used=*/false);
    CHECK(fresh.residency() == NodeState::kEmpty && !fresh.has_refs() &&
              fresh.in_ring() && !fresh.used(),
          "a never-used node, sitting in a ring initialised to arange(C)");
  }

  // === AtomicNodeState ====================================================
  //
  // Single-threaded under the shim, which serialises every atomic. So what is
  // reachable here is the protocol — which states each operation accepts and
  // which it refuses, and what it leaves behind — and not one thing about
  // ordering or contention. Those need a GPU.
  {
    CHECK(sizeof(AtomicNodeState) == 4, "still one word");
    CHECK(std::is_trivially_copyable<AtomicNodeState>::value,
          "trivially copyable: a node travels through cudaMemcpy");
    CHECK(std::is_trivially_default_constructible<AtomicNodeState>::value,
          "and trivially default constructible");
  }

  // --- try_pin accepts RESIDENT and nothing else ---------------------------
  {
    for (R r : kAllResidencies) {
      AtomicNodeState a;
      a.set(NodeState::make(r, 0u, /*in_ring=*/true, /*used=*/false));
      const bool ok = a.try_pin();
      CHECK(ok == (r == NodeState::kResident),
            "only a resident node can be pinned");
      if (ok) {
        CHECK(a.get().refs() == 1u, "pin took a reference");
        CHECK(a.get().used(), "and recorded recency");
        CHECK(a.get().in_ring(),
              "and left in_ring alone — a pinned node keeps its ring entry, "
              "which the popper drops as stale");
      } else {
        CHECK(a.get() == NodeState::make(r, 0u, true, false),
              "a refused pin changes nothing");
      }
    }
  }

  // --- release, and who owns the enqueue -----------------------------------
  {
    // Last reference, not yet queued: this thread enqueues.
    AtomicNodeState a;
    a.set(NodeState::make(NodeState::kResident, 1u, /*in_ring=*/false,
                          /*used=*/true));
    CHECK(a.release(), "1 -> 0 with no ring entry owns the enqueue");
    CHECK(a.get().refs() == 0u, "and the count reached zero");
    CHECK(a.get().in_ring(), "with the bit set inside the same operation");
    CHECK(a.get().residency() == NodeState::kResident && a.get().used(),
          "residency and used untouched");

    // Last reference, already queued: the stale entry serves, nobody enqueues.
    AtomicNodeState b;
    b.set(NodeState::make(NodeState::kResident, 1u, /*in_ring=*/true, false));
    CHECK(!b.release(), "1 -> 0 with a ring entry already present does not");
    CHECK(b.get().refs() == 0u && b.get().in_ring(), "and leaves it in place");

    // Not the last reference.
    AtomicNodeState c;
    c.set(NodeState::make(NodeState::kResident, 3u, false, false));
    CHECK(!c.release(), "2 -> 1 does not enqueue");
    CHECK(c.get().refs() == 2u && !c.get().in_ring(), "nor set the bit");
  }

  // --- try_claim, and what it reports back ---------------------------------
  {
    for (R r : kAllResidencies) {
      AtomicNodeState a;
      a.set(NodeState::make(r, 0u, /*in_ring=*/true, /*used=*/true));
      NodeState was = NodeState::make(NodeState::kLoading, 0, false, false);
      const bool ok = a.try_claim(&was);
      const bool claimable =
          (r == NodeState::kEmpty || r == NodeState::kResident);
      CHECK(ok == claimable,
            "an unreferenced node is claimable when empty or resident, and "
            "not while loading");
      if (ok) {
        CHECK(was.residency() == r,
              "the reported state says whether there is a row to write back");
        CHECK(a.get().residency() == NodeState::kLoading,
              "claimed, and straight into the state nothing can pin or claim");
        CHECK(!a.try_pin(), "so a pin arriving now is refused");
        CHECK(a.get().refs() == 1u, "and held for the step that claimed it");
        CHECK(!a.get().used(), "and recency does not carry into a new key");
      } else {
        CHECK(a.get() == NodeState::make(r, 0u, /*in_ring=*/false,
                                         /*used=*/true),
              "a refused claim still consumes the queue entry, and changes "
              "nothing else");
      }
      CHECK(!a.get().in_ring(),
            "the entry is consumed either way — the pop that led here spent "
            "it, so the flag has to stop saying otherwise");
    }

    // A node hit again after being enqueued: the entry is stale.
    AtomicNodeState busy;
    busy.set(NodeState::make(NodeState::kResident, 1u, true, true));
    NodeState was = NodeState::make(NodeState::kEmpty, 0, false, false);
    CHECK(!busy.try_claim(&was), "a referenced node cannot be claimed");
    CHECK(busy.get() == NodeState::make(NodeState::kResident, 1u,
                                        /*in_ring=*/false, true),
          "and only in_queue moved");
  }

  // --- the leak this used to have -----------------------------------------
  //
  // Found by the first device run, and invisible until eviction pressure made
  // stale entries common. A node hit again after being enqueued has its entry
  // dropped by the popper; if the flag outlived the entry then release() would
  // decline to enqueue — an entry already exists, it would reason — and the
  // node would never return to the queue. Every node eventually ended up
  // flagged-but-absent and the cache ran out of slots it still owned.
  {
    AtomicNodeState n;
    NodeState was = NodeState::make(NodeState::kEmpty, 0, false, false);
    n.set(NodeState::make(NodeState::kResident, 1u, /*in_ring=*/true, false));

    CHECK(!n.try_claim(&was), "the entry is stale, so the claim is refused");
    CHECK(!n.get().in_ring(), "but the entry itself is spent");
    CHECK(n.release(),
          "so the release that follows is the one that owns the enqueue");
    CHECK(n.get().in_ring() && !n.get().has_refs(),
          "and the node is back on the queue rather than lost");
  }

  // --- a whole miss, end to end -------------------------------------------
  {
    AtomicNodeState a;
    a.set(NodeState::make(NodeState::kEmpty, 0u, /*in_ring=*/true, false));

    NodeState was = NodeState::make(NodeState::kResident, 0, false, false);
    CHECK(a.try_claim(&was), "claimed off the ring");
    CHECK(was.residency() == NodeState::kEmpty, "nothing to write back");
    CHECK(a.get().residency() == NodeState::kLoading &&
              a.get().refs() == 1u && !a.get().in_ring() && !a.get().used(),
          "claim alone leaves the node bound and held: nothing to set after");
    CHECK(!a.try_pin(), "a loading row cannot be handed to anybody");

    a.publish();
    CHECK(a.get().residency() == NodeState::kResident, "published");
    CHECK(a.get().refs() == 1u, "still held by the step that claimed it");

    CHECK(a.release(), "the step finishes and owns the enqueue");
    CHECK(a.get().refs() == 0u && a.get().in_ring(), "back on the ring");
    CHECK(a.get().residency() == NodeState::kResident,
          "and still a cache hit while it sits there");
    CHECK(a.try_pin(), "which the next hit takes");
  }

  // --- invalidate: an idle node whose key the backing evicted ---------------
  {
    AtomicNodeState a;
    a.set(NodeState::make(NodeState::kResident, 0u, /*in_ring=*/true,
                          /*used=*/true));
    a.invalidate();
    CHECK(a.get() == NodeState::make(NodeState::kEmpty, 0u, /*in_ring=*/true,
                                     /*used=*/false),
          "an idle resident node becomes empty, keeps its ring entry, and loses "
          "its recency");
    CHECK(!a.try_pin(), "and a hit can no longer land on it");

    NodeState was = NodeState::make(NodeState::kResident, 0, false, false);
    CHECK(a.try_claim(&was), "but an allocate can take it");
    CHECK(was.residency() == NodeState::kEmpty,
          "as an empty node, so there is nothing to write back");
  }

  if (failures == 0) {
    std::printf("all checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
