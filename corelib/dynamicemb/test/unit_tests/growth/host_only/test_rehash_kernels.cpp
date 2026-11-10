// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the REAL rehash kernels of src/table_operation/table_rows.cu (cut out by
// extract_rehash_kernels.py: table_carry_slot_state and table_carry_reclaimed_rows, with
// their wrappers) on the CPU, every "thread" one after the other, and compares the fused
// reclaimed-row carry with the collect + place pair it replaced
// (replaced_reclaimed_rows_kernels.inc) on random tables: identical key, digest and row
// arrays, and the properties that do not depend on the old kernels; and the edges of both
// kernels (a count of zero or below, a count that is too high or too low, an empty range, a
// range too small, a key without a slot, the wrappers' checks).
#include "kernel_stubs.h"
#include <random>
#include <map>
#include <set>

namespace dyn_emb {
#include "new_kernels.inc"
}
namespace oldimpl {
using dyn_emb::emulate_launch;
#include "replaced_reclaimed_rows_kernels.inc"
}

using at::Tensor;
static std::mt19937_64 rng(12345);
static int failures = 0;
#define EXPECT(c) do { if (!(c)) { ++failures; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)

template <class F> static bool throws(F f) {
  try { f(); } catch (const std::exception &e) { return true; }
  return false;
}

struct Maps {
  Tensor old_keys, old_rows, new_keys, new_digests, new_rows;
  int64_t cap, ob0, ob1, nb0, nb1, count;
  std::vector<int64_t> reclaimed_rows; // the rows the old reclaimed slots own
};

static Maps make(int64_t cap, int64_t old_buckets, int64_t new_buckets, int64_t ob0, int64_t nb0,
                 double p_live, double p_reclaimed, double p_reclaimed_no_row, int64_t key_pad) {
  Maps m;
  m.cap = cap; m.ob0 = ob0; m.ob1 = ob0 + old_buckets; m.nb0 = nb0; m.nb1 = nb0 + new_buckets;
  int64_t old_total = m.ob1 + 3, new_total = m.nb1 + 2;   // other tables around ours
  m.old_keys = Tensor::make<int64_t>({old_total, cap}, {cap + key_pad, 1});
  m.old_rows = Tensor::vec<int64_t>(old_total * cap);
  m.new_keys = Tensor::make<int64_t>({new_total, cap}, {cap + key_pad + 5, 1});
  m.new_digests = Tensor::make<uint8_t>({new_total, cap}, {cap + 9, 1});
  m.new_rows = Tensor::vec<int64_t>(new_total * cap);
  std::uniform_real_distribution<double> u(0, 1);
  int64_t *ok = m.old_keys.data_ptr<int64_t>(), *orow = m.old_rows.data_ptr<int64_t>();
  int64_t row = 0, live = 0;
  for (int64_t b = 0; b < old_total; ++b)
    for (int64_t i = 0; i < cap; ++i) {
      int64_t &k = ok[b * m.old_keys.stride(0) + i];
      double r = u(rng);
      orow[b * cap + i] = -1;
      bool ours = b >= m.ob0 && b < m.ob1;
      if (r < p_live) { k = (int64_t)(rng() >> 2); orow[b * cap + i] = row++; if (ours) ++live; }
      else if (r < p_live + p_reclaimed) {
        k = -2;
        if (u(rng) < p_reclaimed_no_row) orow[b * cap + i] = -1;
        else { orow[b * cap + i] = row++; if (ours) m.reclaimed_rows.push_back(orow[b * cap + i]); }
      } else k = -1;
    }
  m.count = (int64_t)m.reclaimed_rows.size();
  // the new table: as many live keys as the old one has, in random free slots
  int64_t *nk = m.new_keys.data_ptr<int64_t>(), *nr = m.new_rows.data_ptr<int64_t>();
  uint8_t *ndigest = m.new_digests.data_ptr<uint8_t>();
  for (int64_t b = 0; b < new_total; ++b)
    for (int64_t i = 0; i < cap; ++i) { nk[b * m.new_keys.stride(0) + i] = -1; ndigest[b * m.new_digests.stride(0) + i] = 0x11; nr[b * cap + i] = -1; }
  for (int64_t placed = 0; placed < live;) {
    int64_t s = rng() % (new_buckets * cap), b = m.nb0 + s / cap, i = s % cap;
    int64_t &k = nk[b * m.new_keys.stride(0) + i];
    if (k != -1) continue;
    k = (int64_t)(rng() >> 2); ndigest[b * m.new_digests.stride(0) + i] = 0x22; nr[b * cap + i] = 1000000 + placed++;
  }
  return m;
}

static Tensor clone(const Tensor &t) {
  Tensor c = t; c.mem = std::make_shared<std::vector<char>>(*t.mem); return c;
}
static bool same(const Tensor &a, const Tensor &b) { return *a.mem == *b.mem; }

static void run_old(Maps &m) {
  int64_t n = m.count;
  if (n > 0) {
    Tensor carried = oldimpl::table_collect_reclaimed_rows(m.old_keys, m.old_rows, m.ob0, m.ob1, m.cap, n);
    oldimpl::table_place_reclaimed_rows(m.new_keys, m.new_digests, m.new_rows, carried, m.nb0, m.nb1, m.cap);
  }
}
static void run_new(Maps &m, int64_t count_value) {
  Tensor count = Tensor::vec<int64_t>(1);
  *count.data_ptr<int64_t>() = count_value;
  dyn_emb::table_carry_reclaimed_rows(m.old_keys, m.old_rows, m.ob0, m.ob1, m.cap, m.new_keys,
                                      m.new_digests, m.new_rows, m.nb0, m.nb1, count);
}

static void test_e8_equals_collect_place() {
  struct Cfg { int64_t cap, ob, nb, ob0, nb0; double pl, pr, pn; };
  std::vector<Cfg> cfgs = {
      {16, 8, 16, 0, 0, 0.4, 0.3, 0.0}, {16, 8, 16, 3, 5, 0.3, 0.5, 0.1}, {128, 4, 8, 1, 2, 0.45, 0.2, 0.05},
      {16, 1, 2, 0, 0, 0.0, 0.0, 0.0},  // nothing at all
      {16, 3, 7, 2, 1, 0.1, 0.85, 0.0}, // almost all reclaimed
      {32, 17, 40, 0, 0, 0.35, 0.3, 0.0}};
  int cases = 0;
  for (auto c : cfgs)
    for (int rep = 0; rep < 8; ++rep) {
      Maps m = make(c.cap, c.ob, c.nb, c.ob0, c.nb0, c.pl, c.pr, c.pn, rep % 3);
      Maps a = m, b = m;
      a.new_keys = clone(m.new_keys); a.new_digests = clone(m.new_digests); a.new_rows = clone(m.new_rows);
      b.new_keys = clone(m.new_keys); b.new_digests = clone(m.new_digests); b.new_rows = clone(m.new_rows);
      run_old(a);
      run_new(b, b.count);
      EXPECT(same(a.new_keys, b.new_keys));
      EXPECT(same(a.new_digests, b.new_digests));
      EXPECT(same(a.new_rows, b.new_rows));
      // and the properties, independent of the old kernels
      std::multiset<int64_t> want(m.reclaimed_rows.begin(), m.reclaimed_rows.end()), got;
      const int64_t *nk = b.new_keys.data_ptr<int64_t>(), *nr = b.new_rows.data_ptr<int64_t>();
      const uint8_t *ndigest = b.new_digests.data_ptr<uint8_t>();
      for (int64_t bk = 0; bk < b.new_keys.size(0); ++bk)
        for (int64_t i = 0; i < c.cap; ++i)
          if (nk[bk * b.new_keys.stride(0) + i] == -2) {
            got.insert(nr[bk * c.cap + i]);
            EXPECT(ndigest[bk * b.new_digests.stride(0) + i] == 0xA7);
            EXPECT(bk >= b.nb0 && bk < b.nb1);
          }
      EXPECT(got == want);
      // nothing outside our range or on a live slot changed
      const int64_t *mk = m.new_keys.data_ptr<int64_t>();
      for (int64_t bk = 0; bk < m.new_keys.size(0); ++bk)
        for (int64_t i = 0; i < c.cap; ++i) {
          int64_t before = mk[bk * m.new_keys.stride(0) + i], after = nk[bk * b.new_keys.stride(0) + i];
          if (before != -1) EXPECT(after == before);
        }
      ++cases;
    }
  printf("e8 == collect+place: %d cases\n", cases);
}

static void test_e8_edges() {
  // count == 0 with reclaimed rows present: nothing is carried, nothing asserts (as before)
  Maps m = make(16, 4, 8, 0, 0, 0.3, 0.5, 0.0, 0);
  Tensor keys_before = clone(m.new_keys);
  run_new(m, 0);
  EXPECT(same(keys_before, m.new_keys));
  // a negative count likewise
  run_new(m, -3);
  EXPECT(same(keys_before, m.new_keys));
  // the counters promise more rows than there are reclaimed slots: the check asserts
  Maps m2 = make(16, 4, 8, 0, 0, 0.3, 0.4, 0.0, 0);
  EXPECT(throws([&] { run_new(m2, m2.count + 1); }));
  // ... or fewer: a thread takes an ordinal past the count
  Maps m3 = make(16, 4, 8, 0, 0, 0.3, 0.4, 0.0, 0);
  if (m3.count > 1) EXPECT(throws([&] { run_new(m3, m3.count - 1); }));
  // an empty old range returns before it launches anything (no zero-block launch)
  Maps m4 = make(16, 4, 8, 1, 0, 0.3, 0.4, 0.0, 0);
  Tensor count = Tensor::vec<int64_t>(1); *count.data_ptr<int64_t>() = 5;
  dyn_emb::table_carry_reclaimed_rows(m4.old_keys, m4.old_rows, 2, 2, 16, m4.new_keys, m4.new_digests,
                                      m4.new_rows, 0, 8, count);
  // a new range too small for the rows: the kernel asserts that it placed the row
  Maps m5 = make(16, 4, 1, 0, 0, 0.0, 0.9, 0.0, 0);
  // fill the single new bucket completely
  for (int64_t i = 0; i < 16; ++i) m5.new_keys.data_ptr<int64_t>()[i] = 5 + i;
  if (m5.count > 0) EXPECT(throws([&] { run_new(m5, m5.count); }));
  // wrapper checks
  Tensor bad_digests = Tensor::make<int64_t>({4, 16}, {16, 1});
  EXPECT(throws([&] {
    dyn_emb::table_carry_reclaimed_rows(m.old_keys, m.old_rows, 0, 4, 16, m.new_keys, bad_digests,
                                        m.new_rows, 0, 8, count);
  }));
  printf("e8 edges done\n");
}

struct Chunk { Tensor keys, slots, src_rows, src_ref, dst_rows, dst_ref; int64_t n, dst_base; };

static Chunk make_chunk(int64_t n, int64_t dst_cap, int64_t dst_base, double p_reserved) {
  Chunk c; c.n = n; c.dst_base = dst_base;
  c.keys = Tensor::vec<int64_t>(n); c.slots = Tensor::vec<int64_t>(n);
  c.src_rows = Tensor::vec<int64_t>(n); c.src_ref = Tensor::vec<int32_t>(n);
  c.dst_rows = Tensor::vec<int64_t>(dst_base + dst_cap); c.dst_ref = Tensor::vec<int32_t>(dst_base + dst_cap);
  std::fill(c.dst_rows.data_ptr<int64_t>(), c.dst_rows.data_ptr<int64_t>() + dst_base + dst_cap, -77);
  std::fill(c.dst_ref.data_ptr<int32_t>(), c.dst_ref.data_ptr<int32_t>() + dst_base + dst_cap, -88);
  std::vector<int64_t> slots(dst_cap);
  for (int64_t i = 0; i < dst_cap; ++i) slots[i] = i;
  std::shuffle(slots.begin(), slots.end(), rng);
  std::uniform_real_distribution<double> u(0, 1);
  int64_t used = 0;
  for (int64_t i = 0; i < n; ++i) {
    int64_t *k = c.keys.data_ptr<int64_t>() + i;
    if (u(rng) < p_reserved) { *k = -1 - (int64_t)(rng() % 4); c.slots.data_ptr<int64_t>()[i] = -1; }
    else {
      *k = (int64_t)(rng() >> 2) | ((rng() & 1) ? 0 : INT64_MIN + 0x10000); // negative keys below -4 are real too
      if (*k >= -4 && *k < 0) *k = 7;
      c.slots.data_ptr<int64_t>()[i] = slots[used++];
    }
    c.src_rows.data_ptr<int64_t>()[i] = 5000 + i;
    c.src_ref.data_ptr<int32_t>()[i] = (int32_t)(i % 7);
  }
  return c;
}

static void test_e7() {
  int cases = 0;
  for (int64_t n : {1, 5, 300, 4097}) for (int64_t base : {0, 128, 4096}) for (int mode = 0; mode < 4; ++mode) {
    bool rows = mode & 1, ref = mode & 2;
    Chunk c = make_chunk(n, n + 20, base, 0.3);
    std::optional<Tensor> sr, dr, sf, df;
    if (rows) { sr = c.src_rows; dr = c.dst_rows; }
    if (ref) { sf = c.src_ref; df = c.dst_ref; }
    Tensor dr_before = clone(c.dst_rows), df_before = clone(c.dst_ref);
    dyn_emb::table_carry_slot_state(c.keys, c.slots, base, sr, dr, sf, df);
    // reference
    std::vector<int64_t> er(dr_before.numel()); std::vector<int32_t> ef(df_before.numel());
    std::copy(dr_before.data_ptr<int64_t>(), dr_before.data_ptr<int64_t>() + er.size(), er.begin());
    std::copy(df_before.data_ptr<int32_t>(), df_before.data_ptr<int32_t>() + ef.size(), ef.begin());
    for (int64_t i = 0; i < n; ++i) {
      int64_t k = c.keys.data_ptr<int64_t>()[i], slot = c.slots.data_ptr<int64_t>()[i];
      bool reserved = (k >= -4 && k < 0);
      if (reserved) continue;
      if (rows) er[base + slot] = c.src_rows.data_ptr<int64_t>()[i];
      if (ref) ef[base + slot] = c.src_ref.data_ptr<int32_t>()[i];
    }
    EXPECT(std::equal(er.begin(), er.end(), c.dst_rows.data_ptr<int64_t>()));
    EXPECT(std::equal(ef.begin(), ef.end(), c.dst_ref.data_ptr<int32_t>()));
    ++cases;
  }
  // a real key without a slot is asserted; a reserved key without a slot is not
  Chunk c = make_chunk(10, 30, 0, 0.0);
  c.slots.data_ptr<int64_t>()[3] = -1;
  EXPECT(throws([&] { dyn_emb::table_carry_slot_state(c.keys, c.slots, 0, c.src_rows, c.dst_rows, c.src_ref, c.dst_ref); }));
  Chunk d = make_chunk(10, 30, 0, 0.0);
  d.keys.data_ptr<int64_t>()[3] = -3; d.slots.data_ptr<int64_t>()[3] = -1;
  EXPECT(!throws([&] { dyn_emb::table_carry_slot_state(d.keys, d.slots, 0, d.src_rows, d.dst_rows, d.src_ref, d.dst_ref); }));
  // the assertion alone (no row, no reference counter)
  EXPECT(throws([&] { dyn_emb::table_carry_slot_state(c.keys, c.slots, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt); }));
  EXPECT(!throws([&] { dyn_emb::table_carry_slot_state(d.keys, d.slots, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt); }));
  // wrapper checks: pairs come whole, lengths match, an empty chunk returns
  EXPECT(throws([&] { dyn_emb::table_carry_slot_state(d.keys, d.slots, 0, d.src_rows, std::nullopt, std::nullopt, std::nullopt); }));
  EXPECT(throws([&] { dyn_emb::table_carry_slot_state(d.keys, d.slots, 0, std::nullopt, std::nullopt, d.src_ref, std::nullopt); }));
  Tensor short_rows = Tensor::vec<int64_t>(3);
  EXPECT(throws([&] { dyn_emb::table_carry_slot_state(d.keys, d.slots, 0, short_rows, d.dst_rows, std::nullopt, std::nullopt); }));
  Tensor empty = Tensor::vec<int64_t>(0);
  dyn_emb::table_carry_slot_state(empty, empty, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  printf("e7: %d cases\n", cases);
}

int main() {
  test_e8_equals_collect_place();
  test_e8_edges();
  test_e7();
  printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
  return failures ? 1 : 0;
}
