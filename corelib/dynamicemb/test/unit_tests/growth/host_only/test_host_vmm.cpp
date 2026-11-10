// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Runs the REAL HostVMMTensor of src/vmm_tensor.cu (cut out by extract_host_class.py) on
// the CPU, against a fake CUDA runtime that records every registration: in-place
// growth, address stability, rollback on failure, the reservation and its halving, and
// what the readers of a buffer do while another thread grows it.
//
// usage: test_host_vmm [grow|rollback|reservation|ctor|concurrent|readers]...
// With no argument every test runs. The tests that set a resource limit (rollback,
// reservation) are for a plain build: a sanitizer maps address space of its own.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include "host_stubs.h"
#include "host_class.inc"

using dyn_emb::HostVMMTensor;
static int failures = 0;
#define EXPECT(c) do { if (!(c)) { ++failures; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)
template <class F> static std::string throws(F f) {
  try { f(); } catch (const std::exception &e) { return e.what(); }
  return "";
}
constexpr size_t PAGE = 4096;
constexpr auto F32 = at::kFloat;

// The permissions of the mapping that holds `addr` ("" if it is not mapped).
static std::string perms_of(const void *addr) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  while (std::getline(maps, line)) {
    uintptr_t lo, hi; char perms[8];
    if (sscanf(line.c_str(), "%lx-%lx %7s", &lo, &hi, perms) == 3 && a >= lo && a < hi) return perms;
  }
  return "";
}
static long vm_lck_kb() {
  std::ifstream st("/proc/self/status"); std::string l;
  while (std::getline(st, l)) if (l.rfind("VmLck:", 0) == 0) return atol(l.c_str() + 6);
  return -1;
}

static void test_grows_in_place() {
  g_cuda.reset();
  size_t floats_per_page = PAGE / 4;
  {
    HostVMMTensor t(floats_per_page, F32, 0, 64 * floats_per_page);   // 1 page now, 64 pages reserved
    void *base = t.data().ptr;
    EXPECT(t.allocated_bytes() == PAGE);
    EXPECT(t.reserved_bytes() == 64 * PAGE);
    EXPECT(g_cuda.register_calls.size() == 1);
    // a pattern in the first page survives every growth, at the same address
    float *f = static_cast<float *>(base);
    for (size_t i = 0; i < floats_per_page; ++i) f[i] = float(i);
    size_t total = floats_per_page;
    for (int step = 0; step < 20; ++step) {
      total += 1 + step * 100;                      // sub-page and multi-page steps
      t.extend(total);
      EXPECT(t.data().ptr == base);
      EXPECT(t.data().numel == (int64_t)total);
      EXPECT(t.allocated_bytes() >= total * 4);
      EXPECT(t.allocated_bytes() % PAGE == 0);
    }
    for (size_t i = 0; i < floats_per_page; ++i) EXPECT(f[i] == float(i));
    // every growth that needed a page registered ONLY the new tail; nothing was unregistered
    EXPECT(g_cuda.unregister_calls.empty());
    size_t covered = 0;
    char *expect_next = static_cast<char *>(base);
    for (auto &r : g_cuda.register_calls) { EXPECT(r.first == expect_next); expect_next += r.second; covered += r.second; }
    EXPECT(covered == t.allocated_bytes());
    // the tail pages are zero, readable and writable
    float *tail = f + floats_per_page;
    for (size_t i = 0; i < total - floats_per_page; ++i) { EXPECT(tail[i] == 0.f); tail[i] = 1.f; }
    EXPECT(perms_of(base).substr(0, 2) == "rw");
    // beyond the reservation: an error, nothing changed
    size_t before = t.allocated_bytes();
    std::string msg = throws([&] { t.extend(100 * floats_per_page); });
    EXPECT(msg.find("address space") != std::string::npos);
    EXPECT(t.allocated_bytes() == before && t.data().numel == (int64_t)total);
    EXPECT(t.data().ptr == base);
    t.extend(total + 3);   // and it is usable afterwards
    EXPECT(t.data().numel == (int64_t)total + 3);
    for (size_t i = 0; i < floats_per_page; ++i) EXPECT(f[i] == float(i));
  }
  // the destructor unregisters each piece once, by its own address, and gives the range back
  EXPECT(g_cuda.registered.empty());
  EXPECT(g_cuda.unregister_calls.size() == g_cuda.register_calls.size());
  printf("grows in place: %zu registrations\n", g_cuda.register_calls.size());
}

static void test_failures_roll_back() {
  size_t fpp = PAGE / 4;
  // (a) cudaHostRegister of the tail fails: the tail is address space again, the object is usable, a retry works
  {
    g_cuda.reset();
    HostVMMTensor t(fpp, F32, 0, 32 * fpp);
    char *base = static_cast<char *>(t.data().ptr);
    g_cuda.fail_register_at = g_cuda.calls;           // the next registration fails
    std::string msg = throws([&] { t.extend(4 * fpp); });
    EXPECT(msg.find("cudaHostRegister") != std::string::npos);
    EXPECT(t.allocated_bytes() == PAGE && t.data().numel == (int64_t)fpp);
    EXPECT(perms_of(base + PAGE) == "---p");                // inaccessible again
    EXPECT(perms_of(base).substr(0, 2) == "rw");            // the first page is untouched
    EXPECT(g_cuda.registered.size() == 1);
    g_cuda.fail_register_at = -1;
    t.extend(4 * fpp);
    EXPECT(t.allocated_bytes() == 4 * PAGE && perms_of(base + 3 * PAGE).substr(0, 2) == "rw");
  }
  // (b) the device pointer of the new piece is not contiguous: rolled back, the piece unregistered
  {
    g_cuda.reset();
    HostVMMTensor t(fpp, F32, 0, 32 * fpp);
    char *base = static_cast<char *>(t.data().ptr);
    g_cuda.device_offset_for_call = g_cuda.queries;    // the next device-pointer query is off by a page
    std::string msg = throws([&] { t.extend(3 * fpp); });
    EXPECT(msg.find("not supported") != std::string::npos);
    EXPECT(t.allocated_bytes() == PAGE);
    EXPECT(g_cuda.registered.size() == 1);              // only the first piece is still registered
    EXPECT(g_cuda.unregister_calls.size() == 1);        // the rejected tail, by its own address
    EXPECT(perms_of(base + PAGE) == "---p");
    g_cuda.device_offset_for_call = -1;
    t.extend(3 * fpp);
    EXPECT(t.allocated_bytes() == 3 * PAGE);
  }
  // (c) mlock fails (RLIMIT_MEMLOCK): rolled back, and the lock of the rest is intact.
  // A process that may lock memory whatever the limit (root, CAP_IPC_LOCK) is not
  // refused: nothing to see then.
  {
    g_cuda.reset();
    HostVMMTensor t(fpp, F32, 0, 256 * fpp);
    char *base = static_cast<char *>(t.data().ptr);
    long locked_kb = vm_lck_kb();
    struct rlimit old{}; getrlimit(RLIMIT_MEMLOCK, &old);
    struct rlimit tight{ (rlim_t)(locked_kb * 1024 + 2 * PAGE), old.rlim_max };
    setrlimit(RLIMIT_MEMLOCK, &tight);
    std::string msg = throws([&] { t.extend(64 * fpp); });   // 63 more pages: over the limit
    if (msg.empty()) {
      printf("RLIMIT_MEMLOCK is not enforced for this process: the mlock rollback is not exercised\n");
    } else {
      EXPECT(msg.find("mlock") != std::string::npos);
      EXPECT(t.allocated_bytes() == PAGE);
      EXPECT(perms_of(base + PAGE) == "---p");
      EXPECT(vm_lck_kb() == locked_kb);
      t.extend(2 * fpp);                                      // one more page fits
      EXPECT(t.allocated_bytes() == 2 * PAGE);
    }
    setrlimit(RLIMIT_MEMLOCK, &old);
  }
  EXPECT(g_cuda.registered.empty());
  printf("rollbacks done\n");
}

static void test_reservation() {
  g_cuda.reset();
  size_t fpp = PAGE / 4;
  size_t ram = getTotalPhysicalMemory();
  {  // the default: eight times what it needs, at least 1 GiB
    HostVMMTensor t(fpp, F32, 0);
    EXPECT(t.reserved_bytes() == (size_t(1) << 30));
    EXPECT(t.allocated_bytes() == PAGE);
  }
  {  // a reservation can be asked for by element count and is clamped to the installed memory
    HostVMMTensor t(fpp, F32, 0, size_t(1) << 62);
    EXPECT(t.reserved_bytes() == (ram + PAGE - 1) / PAGE * PAGE);
  }
  {  // a reservation below what the buffer needs is what it needs
    HostVMMTensor t(4 * fpp, F32, 0, 1);
    EXPECT(t.reserved_bytes() == 4 * PAGE);
    EXPECT(throws([&] { t.extend(5 * fpp); }).find("address space") != std::string::npos);
  }
  {  // the kernel refuses that much address space: it halves, down to a size that fits
    struct rlimit old{}; getrlimit(RLIMIT_AS, &old);
    std::ifstream st("/proc/self/statm"); long pages; st >> pages;
    struct rlimit lim{ (rlim_t)(pages * PAGE + (size_t(96) << 20)), old.rlim_max };
    setrlimit(RLIMIT_AS, &lim);
    HostVMMTensor t(fpp, F32, 0, size_t(1) << 38);            // asks for 1 TiB
    EXPECT(t.reserved_bytes() >= PAGE && t.reserved_bytes() < (size_t(96) << 20));
    size_t got = t.reserved_bytes();
    EXPECT(throws([&] { t.extend(got / 4 + fpp); }).find("address space") != std::string::npos);
    setrlimit(RLIMIT_AS, &old);
  }
  EXPECT(g_cuda.registered.empty());
  printf("reservation done\n");
}

static void test_constructor_failure_gives_the_range_back() {
  g_cuda.reset();
  size_t fpp = PAGE / 4;
  g_cuda.fail_register_at = 0;
  std::string msg = throws([&] { HostVMMTensor t(fpp, F32, 0, 16 * fpp); });
  EXPECT(msg.find("cudaHostRegister") != std::string::npos);
  EXPECT(g_cuda.registered.empty());
  g_cuda.fail_register_at = -1;
  printf("constructor failure ok\n");
}

static void test_concurrent_extend_and_data() {
  g_cuda.reset();
  size_t fpp = PAGE / 4;
  HostVMMTensor t(fpp, F32, 0, 600 * fpp);
  std::atomic<bool> stop{false};
  std::atomic<int> broken{0};
  // What a reader may rely on without a lock: the committed bytes cover the logical
  // size it has just seen, and neither ever goes back.
  std::thread reader([&] {
    int64_t last_numel = 0;
    size_t last_bytes = 0;
    while (!stop) {
      auto d = t.data();
      size_t bytes = t.allocated_bytes();
      if ((size_t)d.numel * 4 > bytes) ++broken;
      if (d.numel < last_numel || bytes < last_bytes) ++broken;
      if (t.logical_numel() < (size_t)d.numel) ++broken;
      size_t committed_numel = t.allocated_numel(); // read before the bytes: they only grow
      if (committed_numel * 4 > t.allocated_bytes()) ++broken;
      (void)t.reserved_bytes();
      last_numel = d.numel;
      last_bytes = bytes;
    }
  });
  for (size_t pages = 2; pages < 500; ++pages) t.extend(pages * fpp);
  stop = true;
  reader.join();
  EXPECT(broken == 0);
  EXPECT(t.allocated_bytes() == 499 * PAGE);
  printf("concurrency ok\n");
}

// A commit of a multi-GB tail takes seconds, and the thread that asks for data() or a
// size meanwhile does so with the GIL held: none of the readers may wait for the grower.
static void test_readers_do_not_wait_for_a_commit() {
  g_cuda.reset();
  size_t fpp = PAGE / 4;
  HostVMMTensor t(fpp, F32, 0, 64 * fpp);
  g_register_delay_ms = 400; // the registration of the tail is slow
  g_inside_register = false;
  std::atomic<bool> grown{false};
  std::thread grower([&] {
    t.extend(8 * fpp);
    grown = true;
  });
  while (!g_inside_register) std::this_thread::yield(); // the grower is inside the commit
  double worst_ms = 0;
  int reads = 0;
  while (!grown && reads < 20) {
    auto start = std::chrono::steady_clock::now();
    auto d = t.data();
    size_t bytes = t.allocated_bytes();
    size_t logical = t.logical_numel();
    size_t committed_numel = t.allocated_numel();
    (void)t.reserved_bytes();
    worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
    EXPECT((size_t)d.numel * 4 <= bytes && logical <= committed_numel);
    ++reads;
  }
  grower.join();
  g_register_delay_ms = 0;
  EXPECT(reads > 0);
  EXPECT(worst_ms < 100); // a reader that waits for the commit takes about 400 ms
  EXPECT(t.data().numel == (int64_t)(8 * fpp) && t.allocated_bytes() == 8 * PAGE);
  printf("worst read %.2f ms during a 400 ms commit (%d reads)\n", worst_ms, reads);
}

int main(int argc, char **argv) {
  struct Case {
    const char *name;
    void (*run)();
  };
  const Case cases[] = {
      {"grow", test_grows_in_place},
      {"rollback", test_failures_roll_back},
      {"reservation", test_reservation},
      {"ctor", test_constructor_failure_gives_the_range_back},
      {"concurrent", test_concurrent_extend_and_data},
      {"readers", test_readers_do_not_wait_for_a_commit},
  };
  for (const Case &c : cases) {
    bool wanted = argc < 2;
    for (int i = 1; i < argc; ++i) wanted |= std::string(argv[i]) == c.name;
    if (wanted) c.run();
  }
  printf(failures ? "FAILED (%d)\n" : "ALL PASSED\n", failures);
  return failures ? 1 : 0;
}
