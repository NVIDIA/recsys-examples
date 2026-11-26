// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <torch/library.h>

#include "table_operation/types.cuh"

namespace dyn_emb {
namespace {
// A warp handles adjacent 16-byte words of one embedding. The row mapping is
// local to the table; the mailbox always uses the common union-key order.
template <typename T, bool Merge>
__global__ void transfer(uint4 *table, uint4 *mailbox, const int64_t *rows,
                         const int64_t *slots, const int64_t *peer_rows,
                         uint8_t *buckets, int capacity, int64_t n, int width,
                         int64_t table_rows, int64_t mailbox_rows,
                         int64_t slot_count, const uint32_t *remote_sent,
                         uint32_t epoch) {
  if constexpr (Merge) {
    if (remote_sent) {
      // Pair with the producer's system-fenced stream write. The driver
      // wait schedules this wave, while the acquire publishes the payload
      // to the threads of this CTA before they read it.
      if (threadIdx.x == 0) {
        uint32_t seen;
        do {
          asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                       : "=r"(seen)
                       : "l"(remote_sent)
                       : "memory");
        } while (seen < epoch);
      }
      __syncthreads();
    }
  }
  const int lane = threadIdx.x % 32;
  for (int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       row < n; row += int64_t(gridDim.x) * blockDim.x / 32) {
    int64_t local_row = rows[row], wire_row = peer_rows ? peer_rows[row] : row;
    CUDA_KERNEL_ASSERT(local_row >= 0 && local_row < table_rows);
    CUDA_KERNEL_ASSERT(wire_row >= 0 && wire_row < mailbox_rows);
    CUDA_KERNEL_ASSERT(slots[row] >= 0 && slots[row] < slot_count);
    uint4 *local = table + local_row * width;
    uint4 *wire = mailbox + wire_row * width;
    for (int col = lane; col < width; col += 32) {
      uint4 x = local[col];
      if constexpr (Merge) {
        const uint4 y = wire[col];
        T a[16 / sizeof(T)], b[16 / sizeof(T)];
        memcpy(a, &x, 16);
        memcpy(b, &y, 16);
#pragma unroll
        for (int j = 0; j < 16 / sizeof(T); ++j)
          a[j] = T((float(a[j]) + float(b[j])) * 0.5f);
        memcpy(&x, a, 16);
        local[col] = x;
      } else {
        wire[col] = x;
      }
    }
    if (Merge && lane == 0) {
      int64_t slot = slots[row];
      LinearBucket<int64_t> bucket(buckets + (slot / capacity) * capacity * 17,
                                   capacity, 1);
      uint64_t ts;
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(ts));
      *bucket.scores(slot % capacity) = ts;
    }
  }
}

void check_transfer(const at::Tensor &table, const at::Tensor &mailbox,
                    const at::Tensor &rows, const at::Tensor &slots,
                    const at::Tensor &buckets,
                    const std::optional<at::Tensor> &peer_rows,
                    int64_t capacity) {
  TORCH_CHECK(table.is_cuda() && table.dim() == 2 && table.is_contiguous());
  TORCH_CHECK(table.scalar_type() == at::kFloat ||
              table.scalar_type() == at::kHalf ||
              table.scalar_type() == at::kBFloat16);
  for (const auto &t : {mailbox, rows, slots, buckets})
    TORCH_CHECK(t.is_contiguous() && t.device() == table.device() &&
                t.dim() == 1);
  TORCH_CHECK(mailbox.scalar_type() == at::kByte &&
              buckets.scalar_type() == at::kByte);
  TORCH_CHECK(slots.sizes() == rows.sizes() &&
              rows.scalar_type() == at::kLong &&
              slots.scalar_type() == at::kLong);
  if (peer_rows)
    TORCH_CHECK(peer_rows->device() == table.device() &&
                peer_rows->sizes() == rows.sizes() &&
                peer_rows->is_contiguous() &&
                peer_rows->scalar_type() == at::kLong);
  const int64_t bytes = table.size(1) * table.element_size();
  TORCH_CHECK(capacity > 0 && capacity <= INT_MAX && capacity % 8 == 0 &&
              buckets.numel() % (capacity * 17) == 0);
  TORCH_CHECK(bytes > 0 && bytes <= INT_MAX && bytes % 16 == 0 &&
              rows.numel() <= mailbox.numel() / bytes &&
              rows.numel() <= int64_t(INT_MAX) * 8);
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(table.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(mailbox.data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(buckets.data_ptr()) % alignof(uint64_t) == 0);
}

void launch_transfer(const at::Tensor &table, const at::Tensor &mailbox,
                     const at::Tensor &rows, const at::Tensor &slots,
                     const at::Tensor &buckets,
                     const std::optional<at::Tensor> &peer_rows,
                     int64_t capacity, bool merge,
                     const uint32_t *remote_sent = nullptr,
                     uint32_t epoch = 0) {
  const int64_t bytes = table.size(1) * table.element_size();
  if (!rows.numel())
    return;
  int64_t blocks = (rows.numel() + 7) / 8;
  if (remote_sent) {
    // Keep the copy engines fed while averaging the preceding waves. Launching
    // a full grid at every wave saturates HBM and stalls concurrent NVLink DMA.
    auto *properties = at::cuda::getDeviceProperties(table.get_device());
    blocks = std::min<int64_t>(blocks,
                               (3 * properties->multiProcessorCount + 1) / 2);
  }
  auto launch = [&](auto scalar, auto do_merge) {
    transfer<decltype(scalar), decltype(do_merge)::value>
        <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            static_cast<uint4 *>(table.data_ptr()),
            static_cast<uint4 *>(mailbox.data_ptr()), rows.data_ptr<int64_t>(),
            slots.data_ptr<int64_t>(),
            peer_rows ? peer_rows->data_ptr<int64_t>() : nullptr,
            buckets.data_ptr<uint8_t>(), capacity, rows.numel(), bytes / 16,
            table.size(0), mailbox.numel() / bytes, buckets.numel() / 17,
            remote_sent, epoch);
  };
  if (!merge)
    launch(float{}, std::false_type{});
  else if (table.scalar_type() == at::kFloat)
    launch(float{}, std::true_type{});
  else if (table.scalar_type() == at::kHalf)
    launch(at::Half{}, std::true_type{});
  else if (table.scalar_type() == at::kBFloat16)
    launch(at::BFloat16{}, std::true_type{});
  else
    TORCH_CHECK(false, "unsupported peer merge dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void peer_transfer(const at::Tensor &table, const at::Tensor &mailbox,
                   const at::Tensor &rows, const at::Tensor &slots,
                   const at::Tensor &buckets,
                   const std::optional<at::Tensor> &peer_rows,
                   std::tuple<int64_t, bool> config) {
  const auto [capacity, merge] = config;
  check_transfer(table, mailbox, rows, slots, buckets, peer_rows, capacity);
  c10::cuda::CUDAGuard guard(table.device());
  launch_transfer(table, mailbox, rows, slots, buckets, peer_rows, capacity,
                  merge);
}

// Test the exact pair of 32-bit stream operations on local memory before
// either rank enters the multi-GPU protocol. Some drivers disable them.
std::tuple<bool, bool> peer_capabilities(const at::Tensor &probe,
                                         int64_t peer_device) {
  TORCH_CHECK(probe.is_cuda() && probe.numel() == 1 &&
              probe.scalar_type() == at::kInt);
  c10::cuda::CUDAGuard guard(probe.device());
  int native_atomics = 0;
  const auto p2p = cudaDeviceGetP2PAttribute(
      &native_atomics, cudaDevP2PAttrNativeAtomicSupported, probe.get_device(),
      peer_device);
  if (p2p != cudaSuccess || !native_atomics)
    return {false, false};
  auto stream = at::cuda::getCurrentCUDAStream();
  auto flag = reinterpret_cast<CUdeviceptr>(probe.data_ptr<int>());
  if (cuStreamWriteValue32(stream, flag, 1, 0) != CUDA_SUCCESS)
    return {true, false};
  return {true, cuStreamWaitValue32(stream, flag, 1,
                                    CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS};
}

// The copy stream publishes its completed byte range to both peers. A merge
// wave waits for both source ranges before overwriting any local table rows.
// Driver stream memory ops retain their default system-scope memory barriers.
void stream_check(CUresult status) {
  if (status == CUDA_SUCCESS)
    return;
  const char *msg = nullptr;
  cuGetErrorString(status, &msg);
  TORCH_CHECK(false,
              "CUDA stream memory operation failed: ", msg ? msg : "unknown");
}

void peer_copy_merge(const at::Tensor &table, const at::Tensor &inbox,
                     const at::Tensor &peer, const at::Tensor &rows,
                     const at::Tensor &peer_rows, const at::Tensor &slots,
                     const at::Tensor &buckets, at::IntArrayRef ends,
                     std::tuple<int64_t, int64_t, bool> layout) {
  const auto [capacity, rows_per_copy, reverse] = layout;
  check_transfer(table, inbox, rows, slots, buckets, peer_rows, capacity);
  TORCH_CHECK(!ends.empty() && ends.back() == rows.numel() &&
              rows_per_copy > 0);
  TORCH_CHECK(table.dim() == 2 && table.is_contiguous() && table.is_cuda() &&
              table.size(0) >= rows.numel());
  TORCH_CHECK(inbox.device() == table.device() &&
              peer.device() == table.device() && inbox.is_contiguous() &&
              peer.is_contiguous() && inbox.scalar_type() == at::kByte &&
              peer.scalar_type() == at::kByte && inbox.sizes() == peer.sizes());
  const int64_t width = table.size(1) * table.element_size();
  TORCH_CHECK(rows.numel() > 0 && rows_per_copy <= INT64_MAX / width &&
              ends.size() == size_t(1 + (rows.numel() - 1) / rows_per_copy) &&
              ends.front() >= 0 && std::is_sorted(ends.begin(), ends.end()));
  TORCH_CHECK(inbox.numel() >= 8 &&
              rows.numel() <= (inbox.numel() - 8) / width &&
              reinterpret_cast<uintptr_t>(peer.data_ptr()) % 16 == 0);
  const int64_t nbytes = rows.numel() * width;
  c10::cuda::CUDAGuard guard(table.device());
  auto copy = at::cuda::getCurrentCUDAStream();
  auto compute = at::cuda::getStreamFromPool();
  // Do not observe a previous invocation's counters (or stale payload at a
  // new flag offset) before this invocation's zero and start barrier finish.
  at::cuda::CUDAEvent initialized;
  initialized.record(copy);
  initialized.block(compute);
  auto *out = peer.data_ptr<uint8_t>();
  const auto *src = static_cast<const uint8_t *>(table.const_data_ptr());
  auto *flags = inbox.data_ptr<uint8_t>() + nbytes;
  auto local_sent = reinterpret_cast<CUdeviceptr>(flags);
  auto remote_sent = local_sent + 4;
  const auto remote_flag = reinterpret_cast<CUdeviceptr>(out + nbytes + 4);
  int64_t begin = 0;
  for (size_t i = 0; i < ends.size(); ++i) {
    const auto end = ends[i];
    TORCH_CHECK(end >= begin && end <= rows.numel());
    auto first = (reverse ? ends.size() - i - 1 : i) * rows_per_copy * width;
    TORCH_CHECK(first < nbytes);
    auto bytes = std::min<int64_t>(rows_per_copy * width, nbytes - first);
    C10_CUDA_CHECK(cudaMemcpyAsync(out + first, src + first, bytes,
                                   cudaMemcpyDeviceToDevice, copy));
    stream_check(cuStreamWriteValue32(copy, remote_flag, i + 1, 0));
    stream_check(cuStreamWriteValue32(copy, local_sent, i + 1, 0));
    if (end > begin) {
      stream_check(cuStreamWaitValue32(compute, local_sent, i + 1,
                                       CU_STREAM_WAIT_VALUE_GEQ));
      stream_check(cuStreamWaitValue32(compute, remote_sent, i + 1,
                                       CU_STREAM_WAIT_VALUE_GEQ));
      c10::cuda::CUDAStreamGuard sg(compute);
      launch_transfer(table, inbox.narrow(0, 0, nbytes),
                      rows.slice(0, begin, end), slots.slice(0, begin, end),
                      buckets, peer_rows.slice(0, begin, end), capacity, true,
                      reinterpret_cast<const uint32_t *>(remote_sent), i + 1);
    }
    begin = end;
  }
  at::cuda::CUDAEvent completed;
  completed.record(compute);
  completed.block(copy);
}

TORCH_LIBRARY_FRAGMENT(dyn_emb, m) {
  m.def("replica_peer_capabilities(Tensor(a!) probe, int peer_device) -> "
        "(bool, bool)");
  m.def("replica_copy_merge_(Tensor(a!) table, Tensor(b!) inbox, Tensor(c!) "
        "peer, Tensor rows, Tensor peer_rows, Tensor slots, Tensor(d!) "
        "buckets, int[] ends, (int, int, bool) layout) -> ()");
  m.def("replica_peer_transfer_(Tensor(a!) table, Tensor(b!) mailbox, Tensor "
        "rows, Tensor slots, Tensor(c!) buckets, Tensor? peer_rows, (int, "
        "bool) config) -> ()");
}
TORCH_LIBRARY_IMPL(dyn_emb, CUDA, m) {
  m.impl("replica_peer_capabilities", TORCH_FN(peer_capabilities));
  m.impl("replica_copy_merge_", TORCH_FN(peer_copy_merge));
  m.impl("replica_peer_transfer_", TORCH_FN(peer_transfer));
}
} // namespace
} // namespace dyn_emb
