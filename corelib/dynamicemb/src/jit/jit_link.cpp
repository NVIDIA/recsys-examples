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

#include "jit_link.h"

#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nvJitLink.h>

namespace dyn_emb {
namespace {

struct EvictModule {
  CUmodule module = nullptr;
  CUfunction ovf = nullptr;
  CUfunction noovf = nullptr;
  CUfunction insert = nullptr;
};

// Source material for a custom score_function, kept device-agnostic so the
// module can be (re)linked per device. cc is the arch numba compiled the LTO-IR
// for; all devices in a (homogeneous) process share it.
struct RegInfo {
  std::vector<char> ltoir;
  std::vector<char> cust;
  int cc_major = 0;
  int cc_minor = 0;
};

std::mutex g_mu;
std::vector<char> g_lex_fatbin;                 // device-agnostic default fatbin
std::unordered_map<int64_t, RegInfo> g_reg;     // key != 0 -> registration source
// A CUmodule is bound to the context it was loaded in, so cache one module per
// (device, key). key 0 == default (Lex). This makes eviction correct when a
// single process drives more than one GPU (each device gets its own module).
std::map<std::pair<int, int64_t>, EvictModule> g_mod;

void cu_check(CUresult r, const char *what) {
  if (r != CUDA_SUCCESS) {
    const char *msg = nullptr;
    cuGetErrorString(r, &msg);
    throw std::runtime_error(std::string("dynamicemb jit_link: ") + what +
                             " failed: " + (msg ? msg : "unknown"));
  }
}

int current_device() {
  CUdevice d;
  cu_check(cuCtxGetDevice(&d), "cuCtxGetDevice");
  return static_cast<int>(d);
}

EvictModule load_from_image(const void *image, const char *ctx) {
  EvictModule m;
  cu_check(cuModuleLoadData(&m.module, image), ctx);
  // If any entry is missing (e.g. a stale/mismatched image), unload the module
  // we just loaded before propagating -- otherwise it leaks.
  try {
    cu_check(cuModuleGetFunction(&m.ovf, m.module, "dyn_emb_evict_entry_ovf"),
             "cuModuleGetFunction(ovf)");
    cu_check(cuModuleGetFunction(&m.noovf, m.module, "dyn_emb_evict_entry_noovf"),
             "cuModuleGetFunction(noovf)");
    cu_check(cuModuleGetFunction(&m.insert, m.module, "dyn_emb_insert_entry"),
             "cuModuleGetFunction(insert)");
  } catch (...) {
    cuModuleUnload(m.module);
    throw;
  }
  return m;
}

EvictModule link_custom(const RegInfo &r) {
  char arch[32];
  std::snprintf(arch, sizeof(arch), "-arch=sm_%d%d", r.cc_major, r.cc_minor);
  const char *opts[] = {arch, "-lto"};

  nvJitLinkHandle handle;
  auto jl_check = [&](nvJitLinkResult res, const char *what) {
    if (res != NVJITLINK_SUCCESS) {
      std::string log;
      size_t log_size = 0;
      if (nvJitLinkGetErrorLogSize(handle, &log_size) == NVJITLINK_SUCCESS &&
          log_size > 0) {
        log.resize(log_size);
        nvJitLinkGetErrorLog(handle, log.data());
      }
      throw std::runtime_error(std::string("dynamicemb jit_link: ") + what +
                               " failed: " + log);
    }
  };

  if (nvJitLinkCreate(&handle, 2, opts) != NVJITLINK_SUCCESS)
    throw std::runtime_error("dynamicemb jit_link: nvJitLinkCreate failed");
  // Destroy the linker handle on every exit path: jl_check throws on failure,
  // and load_from_image can throw too. jl_check reads the error log off `handle`
  // before it throws, so the handle is still live at that point.
  struct HandleGuard {
    nvJitLinkHandle &h;
    ~HandleGuard() { nvJitLinkDestroy(&h); }
  } handle_guard{handle};
  jl_check(nvJitLinkAddData(handle, NVJITLINK_INPUT_FATBIN, r.cust.data(),
                            r.cust.size(), "evict_custom"),
           "nvJitLinkAddData(custom fatbin)");
  jl_check(nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, r.ltoir.data(),
                            r.ltoir.size(), "user_score_fn"),
           "nvJitLinkAddData(user ltoir)");
  jl_check(nvJitLinkComplete(handle), "nvJitLinkComplete");
  size_t cubin_size = 0;
  jl_check(nvJitLinkGetLinkedCubinSize(handle, &cubin_size),
           "nvJitLinkGetLinkedCubinSize");
  std::vector<char> cubin(cubin_size);
  jl_check(nvJitLinkGetLinkedCubin(handle, cubin.data()),
           "nvJitLinkGetLinkedCubin");
  return load_from_image(cubin.data(), "cuModuleLoadData(custom cubin)");
}

// Build (or fetch cached) the module for (dev, key) in the CURRENT context.
// Caller must hold g_mu.
EvictModule &get_module_locked(int dev, int64_t key) {
  auto it = g_mod.find({dev, key});
  if (it != g_mod.end())
    return it->second;
  EvictModule m;
  if (key == 0) {
    if (g_lex_fatbin.empty())
      throw std::runtime_error("dynamicemb jit_link: default evict fatbin not "
                               "set (call demb_set_lex_fatbin first)");
    m = load_from_image(g_lex_fatbin.data(), "cuModuleLoadData(lex fatbin)");
  } else {
    auto r = g_reg.find(key);
    if (r == g_reg.end())
      throw std::runtime_error("dynamicemb jit_link: score_function key " +
                               std::to_string(key) + " not registered");
    m = link_custom(r->second);
  }
  return g_mod.emplace(std::make_pair(dev, key), m).first->second;
}

} // namespace

void demb_set_lex_fatbin(const void *lex, size_t lex_size) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Set-once: the default (Lex) fatbin bytes are device-agnostic; per-device
  // modules are loaded lazily from them. Ignore repeat calls so the stored bytes
  // never diverge. (Python's ensure_lex_fatbin_loaded() is already idempotent.)
  if (!g_lex_fatbin.empty())
    return;
  g_lex_fatbin.assign(static_cast<const char *>(lex),
                      static_cast<const char *>(lex) + lex_size);
}

void demb_register_score_function(int64_t key, const void *ltoir,
                                  size_t ltoir_size, const void *cust,
                                  size_t cust_size, int cc_major,
                                  int cc_minor) {
  if (key == 0)
    throw std::runtime_error("dynamicemb jit_link: score_function key 0 is "
                             "reserved for the default evictor");
  std::lock_guard<std::mutex> lk(g_mu);
  RegInfo &r = g_reg[key];
  if (r.ltoir.empty()) { // first registration of this key: record the source
    r.ltoir.assign(static_cast<const char *>(ltoir),
                   static_cast<const char *>(ltoir) + ltoir_size);
    r.cust.assign(static_cast<const char *>(cust),
                  static_cast<const char *>(cust) + cust_size);
    r.cc_major = cc_major;
    r.cc_minor = cc_minor;
  }
  // Eager-build for the current device so the link cost stays at table-creation
  // time (not the first forward). Other devices link lazily on first use.
  get_module_locked(current_device(), key);
}

CUfunction demb_get_evict_fn(int64_t key, bool overflow) {
  std::lock_guard<std::mutex> lk(g_mu);
  EvictModule &m = get_module_locked(current_device(), key);
  return overflow ? m.ovf : m.noovf;
}

CUfunction demb_get_insert_fn(int64_t key) {
  std::lock_guard<std::mutex> lk(g_mu);
  EvictModule &m = get_module_locked(current_device(), key);
  return m.insert;
}

void demb_launch_evict(CUfunction fn, EvictParams params, int64_t batch,
                       CUstream stream) {
  constexpr int BLOCK = 256;
  if (batch <= 0)
    return;
  unsigned grid = static_cast<unsigned>((batch + BLOCK - 1) / BLOCK);
  void *args[] = {&params};
  cu_check(cuLaunchKernel(fn, grid, 1, 1, BLOCK, 1, 1, 0, stream, args,
                          nullptr),
           "cuLaunchKernel(evict)");
}

} // namespace dyn_emb
