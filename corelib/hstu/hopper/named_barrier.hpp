/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// cutlass::arch::NamedBarrier::sync/arrive are only enabled Sm90 even though they work
// for Sm80 as well. We reimplement them here, enabled for both Sm90 and Sm80.

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
    static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, cutlass::arch::ReservedNamedBarriers reserved_named_barriers) {
    uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

enum class FwdNamedBarriers {
    QueryEmpty = 0,
    ValueEmpty = 1,
    TileCountSmemEmpty = 2,
    TileCountSmemFull = 3,
    WarpSchedulerWG1 = 4,
    WarpSchedulerWG2 = 5,
    WarpSchedulerWG3 = 6,
    ProducerWG = 7,
    AddRabWG1 = 8,
    AddRabWG2 = 9,
    AddRabWG3 = 10
};

enum class BwdNamedBarriers {
    QueryEmpty = 0,
    KVEmpty = 1,
    TileCountSmemEmpty = 2,
    TileCountSmemFull = 3,
    PdS = 4,
    // WarpSchedulerWG1 = 4,
    // WarpSchedulerWG2 = 5,
    // dQEmptyWG1 = 4,
    // dQEmptyWG2 = 5,
    // dSFull = 6,
    // dSEmptyWG1 = 7,
    // dSEmptyWG2 = 8,
    dQEmpty = 7,
    dQFull = 8,
    AddRabWG1 = 9,
    AddRabWG2 = 10,
    AddRabWG3 = 11
};

} // flash
