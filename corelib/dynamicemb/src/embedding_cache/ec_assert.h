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

// ---------------------------------------------------------------------------
// Debug-only checks on a caller's contract.
//
// Plain assert(), so NDEBUG is the switch, and the build defines it: see the
// -DNDEBUG in setup.py and CMakeLists.txt. Off in a normal build, on in one
// that drops that flag.
//
// It has to be off by default, and the reason is the failure mode rather than
// the cost. These fire in __device__ code, where an assert does not merely stop
// the thread: it aborts the kernel and leaves the CUDA context unusable, with
// every later call returning cudaErrorAssert. A training run that trips one is
// over. Someone already met this — table_operation/types.cuh:455 has a
// commented-out assert where the check was deleted rather than switched off.
//
// So these are for finding a bug, not for surviving one. Where a violation has
// to be caught in a real run, it needs a different mechanism — a host-side
// sweep such as HashMapStorage::check_bindings, or a counter — not this.
//
// The #ifndef lets a project with an assertion macro it prefers define
// EC_ASSERT before including this and keep it.
//
// Named with a prefix rather than reusing something generic: this header sits
// on the include path as ec_assert.h precisely so that neither the file name
// nor the macro can be captured by, or capture, anything else.
// ---------------------------------------------------------------------------

#pragma once

#include <cassert>

#ifndef EC_ASSERT
#define EC_ASSERT(cond) assert(cond)
#endif
