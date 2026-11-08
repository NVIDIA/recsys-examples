// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal JVM-free smoke test for the JNI libraries. It dlopen()s each library, resolves the
// exported JNI symbols by their mangled names (which checks the package prefix) and drives them
// through a hand-built JNIEnv function table, so no JDK or Java classes are required.
//
// Build and run: see ../selftest.sh. Needs one visible GPU of a compiled-in architecture.
#include <jni.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct FakeBuffer { void* address; jlong capacity; };
std::string g_thrown;
FakeBuffer g_buffers[64];
int g_buffer_count = 0;

jclass JNICALL fake_find_class(JNIEnv*, const char* name) {
  static std::string last;
  last = name;
  return reinterpret_cast<jclass>(&last);
}
jint JNICALL fake_throw_new(JNIEnv*, jclass, const char* message) {
  g_thrown = message;
  return 0;
}
jobject JNICALL fake_new_direct_buffer(JNIEnv*, void* address, jlong capacity) {
  g_buffers[g_buffer_count] = {address, capacity};
  return reinterpret_cast<jobject>(&g_buffers[g_buffer_count++]);
}
jlongArray JNICALL fake_new_long_array(JNIEnv*, jsize length) {
  return reinterpret_cast<jlongArray>(new std::vector<jlong>(length));
}
void JNICALL fake_set_long_array_region(JNIEnv*, jlongArray array, jsize start, jsize length,
                                        const jlong* buffer) {
  auto* v = reinterpret_cast<std::vector<jlong>*>(array);
  for (jsize i = 0; i < length; ++i) (*v)[start + i] = buffer[i];
}

JNINativeInterface_ g_table;
JNIEnv_ g_env;

void init_env() {
  std::memset(&g_table, 0, sizeof g_table);
  g_table.FindClass = fake_find_class;
  g_table.ThrowNew = fake_throw_new;
  g_table.NewDirectByteBuffer = fake_new_direct_buffer;
  g_table.NewLongArray = fake_new_long_array;
  g_table.SetLongArrayRegion = fake_set_long_array_region;
  g_env.functions = &g_table;
}

int g_failures = 0;
#define CHECK(cond, what)                                                       \
  do {                                                                          \
    if (!(cond)) { std::printf("FAIL: %s\n", what); ++g_failures; }             \
    else { std::printf("ok:   %s\n", what); }                                   \
  } while (0)

void* open_lib(const std::string& dir, const char* name) {
  const std::string path = dir + "/" + name;
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle == nullptr) std::printf("dlopen %s failed: %s\n", name, dlerror());
  return handle;
}

template <typename F>
F sym(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (p == nullptr) { std::printf("missing symbol %s\n", name); ++g_failures; }
  return reinterpret_cast<F>(p);
}

constexpr const char* kPrefix = "Java_com_nvidia_fep_flink_";
std::string S(const char* cls, const char* method) {
  return std::string(kPrefix) + cls + "_" + method;
}

// ---------------------------------------------------------------- CudaDeviceBinding
void test_device(const std::string& dir) {
  void* lib = open_lib(dir, "libcudf_flink_device.so");
  if (!lib) { ++g_failures; return; }
  auto get = sym<jint (*)(JNIEnv*, jclass)>(lib, S("CudaDeviceBinding", "nativeGetDevice").c_str());
  auto set = sym<void (*)(JNIEnv*, jclass, jint)>(lib, S("CudaDeviceBinding", "nativeSetDevice").c_str());
  auto mem = sym<jlongArray (*)(JNIEnv*, jclass, jint)>(lib, S("CudaDeviceBinding", "nativeMemoryInfo").c_str());
  if (!get || !set || !mem) return;
  g_thrown.clear();
  set(&g_env, nullptr, 0);
  CHECK(g_thrown.empty() && get(&g_env, nullptr) == 0, "CudaDeviceBinding set/get device 0");
  auto* info = reinterpret_cast<std::vector<jlong>*>(mem(&g_env, nullptr, 0));
  CHECK(info && (*info)[1] > 0 && (*info)[0] > 0 && (*info)[0] <= (*info)[1],
        "CudaDeviceBinding memory info (free <= total, both > 0)");
}

// ---------------------------------------------------------------- LookupJoinKernel
void test_lookup(const std::string& dir) {
  void* lib = open_lib(dir, "libcudf_flink_lookup_join.so");
  if (!lib) { ++g_failures; return; }
  using Handle = jlong;
  auto create = sym<Handle (*)(JNIEnv*, jclass, jint, jint)>(lib, S("LookupJoinKernel", "nativeCreate").c_str());
  auto buf = [&](const char* m) {
    return sym<jobject (*)(JNIEnv*, jclass, Handle)>(lib, S("LookupJoinKernel", m).c_str());
  };
  auto dimKeys = buf("nativeDimensionKeysBuffer");
  auto dimKeyValid = buf("nativeDimensionKeyValidBuffer");
  auto dimPay = buf("nativeDimensionPayloadsBuffer");
  auto dimPayValid = buf("nativeDimensionPayloadValidBuffer");
  auto prKeys = buf("nativeProbeKeysBuffer");
  auto prKeyValid = buf("nativeProbeKeyValidBuffer");
  auto outPay = buf("nativeOutputPayloadsBuffer");
  auto outPayValid = buf("nativeOutputPayloadValidBuffer");
  auto outMatched = buf("nativeOutputMatchedBuffer");
  auto metrics = buf("nativeMetricsBuffer");
  auto rebuild = sym<void (*)(JNIEnv*, jclass, Handle, jint)>(lib, S("LookupJoinKernel", "nativeRebuild").c_str());
  auto probe = sym<void (*)(JNIEnv*, jclass, Handle, jint, jboolean, jint, jlong, jint)>(
      lib, S("LookupJoinKernel", "nativeProbe").c_str());
  auto close = sym<void (*)(JNIEnv*, jclass, Handle)>(lib, S("LookupJoinKernel", "nativeClose").c_str());
  if (!create || !rebuild || !probe || !close) return;

  constexpr int D = 1000, P = 4096;
  g_thrown.clear();
  const Handle h = create(&g_env, nullptr, D, P);
  CHECK(h != 0 && g_thrown.empty(), ("Lookup nativeCreate: " + g_thrown).c_str());
  if (h == 0) return;
  auto addr = [](jobject o) { return reinterpret_cast<FakeBuffer*>(o)->address; };
  auto* dk = static_cast<long long*>(addr(dimKeys(&g_env, nullptr, h)));
  auto* dkv = static_cast<std::uint8_t*>(addr(dimKeyValid(&g_env, nullptr, h)));
  auto* dp = static_cast<long long*>(addr(dimPay(&g_env, nullptr, h)));
  auto* dpv = static_cast<std::uint8_t*>(addr(dimPayValid(&g_env, nullptr, h)));
  auto* pk = static_cast<long long*>(addr(prKeys(&g_env, nullptr, h)));
  auto* pkv = static_cast<std::uint8_t*>(addr(prKeyValid(&g_env, nullptr, h)));
  auto* op = static_cast<long long*>(addr(outPay(&g_env, nullptr, h)));
  auto* opv = static_cast<std::uint8_t*>(addr(outPayValid(&g_env, nullptr, h)));
  auto* om = static_cast<std::uint8_t*>(addr(outMatched(&g_env, nullptr, h)));
  auto* met = static_cast<std::int32_t*>(addr(metrics(&g_env, nullptr, h)));

  // dimension: key = 10*i, payload = 7*i + 1; every 100th payload is NULL
  for (int i = 0; i < D; ++i) {
    dk[i] = 10LL * i; dkv[i] = 1; dp[i] = 7LL * i + 1; dpv[i] = (i % 100 != 0);
  }
  g_thrown.clear();
  rebuild(&g_env, nullptr, h, D);
  CHECK(g_thrown.empty(), ("Lookup nativeRebuild: " + g_thrown).c_str());

  // probe: key = 5*j (even j hit, odd j miss), every 17th key NULL
  for (int j = 0; j < P; ++j) { pk[j] = 5LL * j; pkv[j] = (j % 17 != 0); }
  g_thrown.clear();
  probe(&g_env, nullptr, h, P, JNI_FALSE, /*filter none*/ 0, 0, /*flags+payload*/ 3);
  CHECK(g_thrown.empty(), ("Lookup nativeProbe: " + g_thrown).c_str());
  int bad = 0, expected_matches = 0;
  for (int j = 0; j < P; ++j) {
    const bool valid = (j % 17 != 0);
    const long long key = 5LL * j;
    const bool hit = valid && key % 10 == 0 && key / 10 < D;
    const int i = static_cast<int>(key / 10);
    if (hit) ++expected_matches;
    if ((om[j] & 1) != (hit ? 1 : 0)) ++bad;
    else if (hit) {
      const bool pv = (i % 100 != 0);
      if (opv[j] != (pv ? 1 : 0) || (pv && op[j] != 7LL * i + 1)) ++bad;
    }
  }
  // Metrics layout: 6 x int64 then dimension_rows, probe_rows, match_count, device_status.
  const int match_count = met[2 * 6 + 2];
  CHECK(bad == 0, "Lookup probe results equal host reference (matched flag + payload + validity)");
  CHECK(match_count == expected_matches, "Lookup match_count equals host reference");

  g_thrown.clear();
  probe(&g_env, nullptr, h, P, JNI_TRUE, 0, 0, 3);
  CHECK(!g_thrown.empty(), "Lookup injected device error is reported as an exception");
  g_thrown.clear();
  close(&g_env, nullptr, h);
  CHECK(g_thrown.empty(), "Lookup nativeClose");
}

// ---------------------------------------------------------------- DenseGroupbyKernel
struct alignas(16) PackedInputRow {
  int group; int int_value; double metric; long long sequence; long long sent_nanos;
  std::uint32_t metric_valid, int_valid, sequence_valid, sent_nanos_valid;
  std::uint32_t filter0, filter1, filter2; std::int32_t weight;
};
static_assert(sizeof(PackedInputRow) == 64, "layout");
struct alignas(16) PackedAggregateRow {
  unsigned long long row_count, metric_count, int_count, long0_count, long1_count,
      filter0_count, filter1_count, filter2_count;
  double metric_sum;
  int min_int, max_int;
  long long int_sum, long0_sum, long1_sum, min_long0, max_long0, min_long1, max_long1;
  double min_metric, max_metric;
  unsigned long long metric_negative_zero_count;
};
static_assert(sizeof(PackedAggregateRow) == 160, "layout");

void test_groupby(const std::string& dir) {
  void* devbatch = open_lib(dir, "libcudf_flink_device_batch.so");
  void* lib = open_lib(dir, "libcudf_flink_groupby.so");
  if (!devbatch || !lib) { ++g_failures; return; }
  using Handle = jlong;
  auto create = sym<Handle (*)(JNIEnv*, jclass, jint, jint, jint, jint, jint, jint)>(
      lib, S("DenseGroupbyKernel", "nativeCreate").c_str());
  auto inBuf = sym<jobject (*)(JNIEnv*, jclass, Handle)>(lib, S("DenseGroupbyKernel", "nativeInputBuffer").c_str());
  auto outBuf = sym<jobject (*)(JNIEnv*, jclass, Handle)>(lib, S("DenseGroupbyKernel", "nativeOutputBuffer").c_str());
  auto exec = sym<void (*)(JNIEnv*, jclass, Handle, jint, jint)>(lib, S("DenseGroupbyKernel", "nativeExecute").c_str());
  auto close = sym<void (*)(JNIEnv*, jclass, Handle)>(lib, S("DenseGroupbyKernel", "nativeClose").c_str());
  if (!create || !inBuf || !outBuf || !exec || !close) return;

  constexpr int N = 10000, G = 37;
  constexpr int kRowCount = 1 << 0, kMetricCount = 1 << 1, kMetricSum = 1 << 8,
                kMinMetric = 1 << 18, kMaxMetric = 1 << 19;
  g_thrown.clear();
  const Handle h = create(&g_env, nullptr, N, kRowCount | kMetricCount | kMetricSum | kMinMetric | kMaxMetric,
                          -1, -1, -1, -1);
  CHECK(h != 0 && g_thrown.empty(), ("Groupby nativeCreate (device admitted): " + g_thrown).c_str());
  if (h == 0) return;
  auto* in = static_cast<PackedInputRow*>(reinterpret_cast<FakeBuffer*>(inBuf(&g_env, nullptr, h))->address);
  auto* out = static_cast<unsigned char*>(reinterpret_cast<FakeBuffer*>(outBuf(&g_env, nullptr, h))->address);
  std::vector<unsigned long long> cnt(G, 0), mcnt(G, 0);
  std::vector<double> sum(G, 0.0), mn(G, 1e300), mx(G, -1e300);
  for (int i = 0; i < N; ++i) {
    std::memset(&in[i], 0, sizeof(PackedInputRow));
    const int g = (i * 7) % G;
    in[i].group = g; in[i].metric = i * 0.5; in[i].weight = 1;
    in[i].metric_valid = (i % 11 != 0);
    cnt[g]++;
    if (in[i].metric_valid) {
      mcnt[g]++; sum[g] += i * 0.5; mn[g] = std::min(mn[g], i * 0.5); mx[g] = std::max(mx[g], i * 0.5);
    }
  }
  g_thrown.clear();
  exec(&g_env, nullptr, h, N, G);
  CHECK(g_thrown.empty(), ("Groupby nativeExecute: " + g_thrown).c_str());
  const auto* rows = reinterpret_cast<const PackedAggregateRow*>(out + 64);
  int bad = 0;
  for (int g = 0; g < G; ++g) {
    if (rows[g].row_count != cnt[g] || rows[g].metric_count != mcnt[g] ||
        rows[g].metric_sum != sum[g] || rows[g].min_metric != mn[g] || rows[g].max_metric != mx[g]) ++bad;
  }
  CHECK(bad == 0, "Groupby COUNT/SUM/MIN/MAX per group equal host reference");
  g_thrown.clear();
  close(&g_env, nullptr, h);
  CHECK(g_thrown.empty(), "Groupby nativeClose");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  init_env();
  test_device(dir);
  test_lookup(dir);
  test_groupby(dir);
  std::printf(g_failures == 0 ? "SELFTEST_PASS\n" : "SELFTEST_FAIL (%d)\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
