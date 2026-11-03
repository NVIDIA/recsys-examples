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
// What NVLink delivers inside one node, per GPU and direction, with SM stores (the sender's kernel
// writes the peer's memory) and SM loads (the receiver's kernel reads it). A sending GPU moves one
// block to each other GPU, so the rates compare. In run order:
//   1. SM stores, all-to-all in rounds: in each round every GPU writes one block into one peer, the
//      next peer each round.
//   2. SM stores and SM loads with every peer at once: every GPU sending to every other (both
//      directions at once), one GPU sending (fan-out: a hot owner's rows read by every peer), and
//      one GPU receiving (fan-in: every peer storing a hot owner's gradient rows into it).
// One process drives all GPUs of the node through peer access.
//
//   nvcc -O3 -arch=sm_90 -o nvlink_ceiling nvlink_ceiling.cu
//   ./nvlink_ceiling [MB per block, default 64]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t err = (x);                                                                \
    if (err != cudaSuccess) {                                                             \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err)); \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

constexpr int kMaxGpus = 8;

// copy_kernel copies num_blocks blocks of `vecs` 16-byte vectors, dst[block] <- src[block]. Either
// side may be a peer's memory (remote dst: SM stores, remote src: SM loads). Each CTA copies the
// same slice of every block, CTA i starting at block i (mod num_blocks), so the CTAs cover all
// blocks, and their links, at once. Callers with a block per peer list each GPU's peers from its
// next one on (gpu + 1, gpu + 2, ... mod num_gpus), so the GPUs start on different peers.
struct Bufs {
  int4 *dst[kMaxGpus];
  const int4 *src[kMaxGpus];
  int num_blocks;  // blocks to copy (usually one per peer)
  size_t vecs;     // 16-byte vectors per block
};

__global__ void copy_kernel(Bufs bufs) {
  const size_t per_cta = (bufs.vecs + gridDim.x - 1) / gridDim.x;
  const size_t lo = blockIdx.x * per_cta, hi = min(bufs.vecs, lo + per_cta);
  for (int j = 0; j < bufs.num_blocks; ++j) {
    const int block = (j + (int)blockIdx.x) % bufs.num_blocks;
    int4 *dst = bufs.dst[block];
    const int4 *src = bufs.src[block];
    size_t i = lo + threadIdx.x;
    // 4 vectors in flight per thread
    for (; i + 3 * blockDim.x < hi; i += 4 * blockDim.x) {
      int4 v0 = __ldcg(src + i), v1 = __ldcg(src + i + blockDim.x);
      int4 v2 = __ldcg(src + i + 2 * blockDim.x), v3 = __ldcg(src + i + 3 * blockDim.x);
      __stcg(dst + i, v0);
      __stcg(dst + i + blockDim.x, v1);
      __stcg(dst + i + 2 * blockDim.x, v2);
      __stcg(dst + i + 3 * blockDim.x, v3);
    }
    for (; i < hi; i += blockDim.x) __stcg(dst + i, __ldcg(src + i));
  }
}

int main(int argc, char **argv) {
  const size_t mb_per_block = argc > 1 ? atol(argv[1]) : 64;
  const size_t bytes = mb_per_block << 20;
  int num_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
  num_gpus = std::min(num_gpus, kMaxGpus);
  printf("%d GPUs, %zu MB per block\n", num_gpus, mb_per_block);
  // per GPU: one send block per peer (local), one receive block per peer
  std::vector<std::vector<char *>> send(num_gpus, std::vector<char *>(num_gpus)),
      recv(num_gpus, std::vector<char *>(num_gpus));
  std::vector<cudaStream_t> streams(num_gpus);
  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    CUDA_CHECK(cudaSetDevice(gpu));
    CUDA_CHECK(cudaStreamCreateWithFlags(&streams[gpu], cudaStreamNonBlocking));
    for (int peer = 0; peer < num_gpus; ++peer) {
      if (peer != gpu) CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
      CUDA_CHECK(cudaMalloc(&send[gpu][peer], bytes));
      CUDA_CHECK(cudaMalloc(&recv[gpu][peer], bytes));
      CUDA_CHECK(cudaMemset(send[gpu][peer], gpu, bytes));
    }
  }
  auto sync_all = [&] {
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      CUDA_CHECK(cudaSetDevice(gpu));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  };
  // Times `issue(senders)` and prints GB/s per sending GPU (or per `per`): the bytes each sender
  // moves over NVLink over the wall time.
  auto timed = [&](const char *name, const std::vector<int> &senders, auto issue,
                   const char *per = "sending GPU") {
    sync_all();
    issue(senders);  // warm up
    sync_all();
    const int reps = 5;
    // host-side wall clock over all devices
    auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) issue(senders);
    sync_all();
    auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count() / reps;
    const double gbps = (double)bytes * (num_gpus - 1) / seconds / 1e9;
    printf("%-52s %8.1f GB/s per %s (%5.1f%% of 450)\n", name, gbps, per, gbps / 4.5);
    fflush(stdout);
  };
  std::vector<int> all(num_gpus), one{0};
  for (int gpu = 0; gpu < num_gpus; ++gpu) all[gpu] = gpu;
  char name[96];

  // 1. SM stores, all-to-all in rounds: in round `step` every GPU writes its block into the GPU
  // `step` after it.
  for (int ctas : {32, 66, 132, 264}) {
    auto rounds_sm_stores = [&](const std::vector<int> &) {
      for (int step = 1; step < num_gpus; ++step)
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          const int peer = (gpu + step) % num_gpus;
          Bufs bufs{};
          bufs.dst[0] = (int4 *)recv[peer][gpu];
          bufs.src[0] = (const int4 *)send[gpu][peer];
          bufs.num_blocks = 1;
          bufs.vecs = bytes / 16;
          CUDA_CHECK(cudaSetDevice(gpu));
          copy_kernel<<<ctas, 512, 0, streams[gpu]>>>(bufs);
        }
    };
    snprintf(name, sizeof(name), "SM stores (%d CTAs), all-to-all in rounds", ctas);
    timed(name, all, rounds_sm_stores);
  }

  // 2. Every peer at once: SM stores on the sending GPU (push) or SM loads on every receiving GPU
  // (pull).
  for (int ctas : {32, 66, 132, 264}) {
    for (int pull = 0; pull < 2; ++pull) {
      // each GPU of `senders` sends one block to every other GPU
      auto send_to_all = [&](const std::vector<int> &senders) {
        if (!pull) {
          for (int sender : senders) {
            Bufs bufs{};
            for (int j = 0; j < num_gpus - 1; ++j) {
              const int receiver = (sender + 1 + j) % num_gpus;
              bufs.dst[j] = (int4 *)recv[receiver][sender];
              bufs.src[j] = (const int4 *)send[sender][receiver];
            }
            bufs.num_blocks = num_gpus - 1;
            bufs.vecs = bytes / 16;
            CUDA_CHECK(cudaSetDevice(sender));
            copy_kernel<<<ctas, 512, 0, streams[sender]>>>(bufs);
          }
        } else {
          // every receiver loads its blocks from the senders
          for (int receiver = 0; receiver < num_gpus; ++receiver) {
            Bufs bufs{};
            int blocks = 0;
            for (int j = 1; j < num_gpus; ++j) {
              const int sender = (receiver + j) % num_gpus;
              if (std::find(senders.begin(), senders.end(), sender) == senders.end()) continue;
              bufs.dst[blocks] = (int4 *)recv[receiver][sender];
              bufs.src[blocks] = (const int4 *)send[sender][receiver];
              ++blocks;
            }
            if (blocks == 0) continue;
            bufs.num_blocks = blocks;
            bufs.vecs = bytes / 16;
            CUDA_CHECK(cudaSetDevice(receiver));
            copy_kernel<<<ctas, 512, 0, streams[receiver]>>>(bufs);
          }
        }
        CUDA_CHECK(cudaGetLastError());
      };
      // fan-in: every other GPU's block into GPU 0, stored by the senders (push) or loaded by
      // GPU 0 (pull)
      auto into_gpu0 = [&](const std::vector<int> &) {
        if (!pull) {
          for (int sender = 1; sender < num_gpus; ++sender) {
            Bufs bufs{};
            bufs.dst[0] = (int4 *)recv[0][sender];
            bufs.src[0] = (const int4 *)send[sender][0];
            bufs.num_blocks = 1;
            bufs.vecs = bytes / 16;
            CUDA_CHECK(cudaSetDevice(sender));
            copy_kernel<<<ctas, 512, 0, streams[sender]>>>(bufs);
          }
        } else {
          Bufs bufs{};
          for (int sender = 1; sender < num_gpus; ++sender) {
            bufs.dst[sender - 1] = (int4 *)recv[0][sender];
            bufs.src[sender - 1] = (const int4 *)send[sender][0];
          }
          bufs.num_blocks = num_gpus - 1;
          bufs.vecs = bytes / 16;
          CUDA_CHECK(cudaSetDevice(0));
          copy_kernel<<<ctas, 512, 0, streams[0]>>>(bufs);
        }
        CUDA_CHECK(cudaGetLastError());
      };
      const char *access = pull ? "loads" : "stores";
      snprintf(name, sizeof(name), "SM %s (%d CTAs), all-to-all", access, ctas);
      timed(name, all, send_to_all);
      snprintf(name, sizeof(name), "SM %s (%d CTAs), one GPU sending", access, ctas);
      timed(name, one, send_to_all);
      snprintf(name, sizeof(name), "SM %s (%d CTAs), one GPU receiving", access, ctas);
      timed(name, all, into_gpu0, "receiving GPU");
    }
  }
  return 0;
}
