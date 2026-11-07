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
// The cost of a shared peer order on the rails (incast), with GIN puts as the
// hierarchical dist's kernels make them.
//
// The dist sends rows between nodes chunk by chunk, on each rail: every proxy
// has a stream of chunks, level after level, one chunk of each owner node in a
// level. Its agents take the streams in order, on every node at about the same
// pace. When every proxy takes the owner nodes of a level in the same order
// (node order), the proxies of a rail send to the same owner node at the same
// time: its NIC takes the puts of several proxies while the other NICs are
// idle. DeepEP turns its peers instead: each rank starts from a different peer.
// This benchmark sends such streams on every rail at once, with one-sided GIN
// puts (each adds 1 to a strong signal of the receiver), in both orders:
//   node      a sender on node p takes the other nodes in node order;
//   rotated   it takes p + 1, p + 2, ... mod nodes (each place of a level is a
//             permutation: every node receives one chunk).
// A sender has levels * (nodes - 1) places, a chunk each. It puts `window`
// places, waits until the chunks of those places that are its own have landed,
// and then the rail passes a barrier (signals) before the next window. Window
// 1 is the senders in full lock-step; a large window is a sender with a
// backlog of chunks, which spreads its puts over all peers.
//
// Build where DynamicEmb builds (NCCL 2.30.7 or newer, its device API):
//   nvcc -O3 -arch=sm_90 -o gin_rail_congestion gin_rail_congestion.cu -lnccl
// Run one process per GPU, across nodes, with srun (NCCL_CROSS_NIC=0, so GIN
// connects the rails):
//   srun -N <nodes> --ntasks-per-node=8 ./gin_rail_congestion <new file on a
//     shared file system> [chunk MiB, 2] [levels, 16] [windows, 1,2,3,0]
// The file passes NCCL's unique id from rank 0 to the others. Rank 0 prints,
// for each window and order, the time of the exchange (the slowest rank's) and
// the rate each GPU sends at. A window of 0 is every place at once.
#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t err = (x);                                                                \
    if (err != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)
#define NCCL_CHECK(x)                                                                     \
  do {                                                                                    \
    ncclResult_t res = (x);                                                               \
    if (res != ncclSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(res)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

constexpr int kMaxNodes = 64;

// The node a sender on `node` puts its chunk of place k of a level to.
__host__ __device__ int dest_of(int node, int nodes, int k, bool rotated) {
  if (rotated) return (node + 1 + k) % nodes;
  return k < node ? k : k + 1;  // the k-th of the other nodes, in node order
}

struct Params {
  int nodes, node, places, window;
  bool rotated;
  size_t chunk;
  uint64_t barriers;             // barrier rounds before this run
  uint64_t received[kMaxNodes];  // chunks from each sender node before this run
  unsigned long long *ns;        // [2]: the run's start and end (globaltimer)
};

__device__ unsigned long long now_ns() {
  unsigned long long ns;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
  return ns;
}

// One thread. Signal s (s < nodes) counts the chunks from sender node s;
// signal `nodes` counts the barrier rounds' signals of the rail peers. Window
// slot (i, s) of this rank's copy of the window holds the chunk that sender s
// puts at place i of a window; the last slot is the chunk a sender puts.
__global__ void bench_kernel(ncclDevComm dev, ncclWindow_t win, Params p) {
  if (threadIdx.x != 0) return;
  ncclGin gin(dev, 0);
  const ncclTeam rail = ncclTeamRail(dev);
  const int nodes = p.nodes, me = p.node;
  uint64_t received[kMaxNodes];
  for (int s = 0; s < nodes; ++s) received[s] = p.received[s];
  uint64_t barriers = p.barriers;
  const size_t source = (size_t)p.window * nodes * p.chunk;
  const unsigned long long start = now_ns();
  for (int first = 0; first < p.places; first += p.window) {
    const int end = min(first + p.window, p.places);
    for (int place = first; place < end; ++place) {
      const int dest = dest_of(me, nodes, place % (nodes - 1), p.rotated);
      const size_t slot = ((size_t)(place - first) * nodes + me) * p.chunk;
      gin.put(rail, dest, win, slot, win, source, p.chunk,
              ncclGin_StrongSignalAdd{(ncclGinSignal_t)me, 1});
    }
    for (int s = 0; s < nodes; ++s) {
      if (s == me) continue;
      for (int place = first; place < end; ++place)
        if (dest_of(s, nodes, place % (nodes - 1), p.rotated) == me) ++received[s];
      while (gin.readSignal((ncclGinSignal_t)s) < received[s]) {
      }
    }
    ++barriers;
    for (int peer = 0; peer < nodes; ++peer)
      if (peer != me) gin.signal(rail, peer, ncclGin_StrongSignalAdd{(ncclGinSignal_t)nodes, 1});
    while (gin.readSignal((ncclGinSignal_t)nodes) < barriers * (nodes - 1)) {
    }
  }
  gin.flush(ncclCoopThread{});
  p.ns[0] = start;
  p.ns[1] = now_ns();
}

static int env_int(const char *name) {
  const char *value = getenv(name);
  if (value == nullptr) {
    fprintf(stderr, "gin_rail_congestion: %s is not set (run it with srun)\n", name);
    exit(1);
  }
  return atoi(value);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <id file> [chunk MiB] [levels] [windows]\n", argv[0]);
    return 1;
  }
  const std::string id_file = argv[1];
  const double chunk_mib = argc > 2 ? atof(argv[2]) : 2.0;
  const int levels = argc > 3 ? atoi(argv[3]) : 16;
  const std::string window_list = argc > 4 ? argv[4] : "1,2,3,0";
  const int rank = env_int("SLURM_PROCID"), world = env_int("SLURM_NTASKS");
  const int local_rank = env_int("SLURM_LOCALID"), local_size = world / env_int("SLURM_NNODES");
  const int nodes = world / local_size, node = rank / local_size;
  if (nodes < 2 || nodes > kMaxNodes || nodes * local_size != world) {
    fprintf(stderr, "gin_rail_congestion: needs 2 to %d whole nodes\n", kMaxNodes);
    return 1;
  }
  CUDA_CHECK(cudaSetDevice(local_rank));

  // NCCL's unique id, from rank 0 through the file
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
    const std::string tmp = id_file + ".tmp";
    FILE *out = fopen(tmp.c_str(), "wb");
    if (out == nullptr || fwrite(&id, sizeof(id), 1, out) != 1 || fclose(out) != 0 ||
        rename(tmp.c_str(), id_file.c_str()) != 0) {
      fprintf(stderr, "gin_rail_congestion: cannot write %s\n", id_file.c_str());
      return 1;
    }
  } else {
    for (;;) {
      FILE *in = fopen(id_file.c_str(), "rb");
      if (in != nullptr) {
        const size_t got = fread(&id, sizeof(id), 1, in);
        fclose(in);
        if (got == 1) break;
      }
      usleep(100000);
    }
  }
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, world, id, rank));

  std::vector<int> windows;
  const int places = levels * (nodes - 1);
  for (size_t at = 0; at < window_list.size();) {
    const size_t comma = std::min(window_list.find(',', at), window_list.size());
    const int window = atoi(window_list.substr(at, comma - at).c_str());
    windows.push_back(window > 0 ? std::min(window, places) : places);
    at = comma + 1;
  }
  const int most_window = *std::max_element(windows.begin(), windows.end());
  const size_t chunk = (size_t)(chunk_mib * (1 << 20)) / 16 * 16;
  const size_t bytes = ((size_t)most_window * nodes + 1) * chunk;

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = 1;
  reqs.ginSignalCount = nodes + 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs.ginQueueDepth = 512;
  ncclDevComm dev;
  NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &dev));
  void *buffer;
  NCCL_CHECK(ncclMemAlloc(&buffer, bytes));
  CUDA_CHECK(cudaMemset(buffer, 1, bytes));
  ncclWindow_t win;
  NCCL_CHECK(ncclCommWindowRegister(comm, buffer, bytes, &win, NCCL_WIN_DEFAULT));
  unsigned long long *ns;
  double *took;
  CUDA_CHECK(cudaMalloc(&ns, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&took, sizeof(double)));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Params p{};
  p.nodes = nodes;
  p.node = node;
  p.places = places;
  p.chunk = chunk;
  p.ns = ns;
  constexpr int kWarmup = 3, kIterations = 10;
  if (rank == 0)
    printf(
        "gin_rail_congestion: %d nodes x %d GPUs, %d levels of %d chunks of %g MiB per sender, "
        "median of %d\n%7s %8s %9s %13s %13s\n",
        nodes, local_size, levels, nodes - 1, chunk_mib, kIterations, "window", "order", "ms",
        "GB/s per GPU", "most senders");
  for (int window : windows) {
    double ms[2] = {0, 0};
    for (int rotated = 0; rotated < 2; ++rotated) {
      p.window = window;
      p.rotated = rotated != 0;
      std::vector<double> times;
      for (int it = 0; it < kWarmup + kIterations; ++it) {
        NCCL_CHECK(ncclAllReduce(took, took, 1, ncclDouble, ncclMax, comm, stream));  // a barrier
        bench_kernel<<<1, 32, 0, stream>>>(dev, win, p);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long host_ns[2];
        CUDA_CHECK(cudaMemcpyAsync(host_ns, ns, sizeof(host_ns), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        double seconds = (host_ns[1] - host_ns[0]) * 1e-9;
        CUDA_CHECK(cudaMemcpyAsync(took, &seconds, sizeof(double), cudaMemcpyHostToDevice, stream));
        NCCL_CHECK(ncclAllReduce(took, took, 1, ncclDouble, ncclMax, comm, stream));
        CUDA_CHECK(cudaMemcpyAsync(&seconds, took, sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (it >= kWarmup) times.push_back(seconds);
        // the running totals: levels chunks from every other node, a barrier a window
        for (int s = 0; s < nodes; ++s)
          if (s != node) p.received[s] += levels;
        p.barriers += (places + window - 1) / window;
      }
      std::sort(times.begin(), times.end());
      const double seconds = times[times.size() / 2];
      ms[rotated] = seconds * 1e3;
      double most = 0;  // the mean, over places, of the most senders one node receives from
      for (int k = 0; k < nodes - 1; ++k) {
        std::vector<int> count(nodes, 0);
        for (int s = 0; s < nodes; ++s) ++count[dest_of(s, nodes, k, rotated != 0)];
        most += *std::max_element(count.begin(), count.end());
      }
      if (rank == 0)
        printf("%7d %8s %9.3f %13.1f %13.2f\n", window, rotated ? "rotated" : "node", seconds * 1e3,
               (double)places * chunk / seconds / 1e9, most / (nodes - 1));
    }
    if (rank == 0)
      printf("window %d: node order takes %.2fx the rotated order's time\n", window, ms[0] / ms[1]);
  }
  fflush(stdout);
  NCCL_CHECK(ncclCommWindowDeregister(comm, win));
  NCCL_CHECK(ncclDevCommDestroy(comm, &dev));
  NCCL_CHECK(ncclMemFree(buffer));
  NCCL_CHECK(ncclCommFinalize(comm));
  NCCL_CHECK(ncclCommDestroy(comm));
  if (rank == 0) unlink(id_file.c_str());
  return 0;
}
