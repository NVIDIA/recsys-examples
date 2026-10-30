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
// Whether the build keeps the hierarchical dist's trace: DYNAMICEMB_HIER_TRACE=1
// at build time makes setup.py define DEMB_HIER_TRACE=1. Without it the trace
// compiles away.
#pragma once

#ifndef DEMB_HIER_TRACE
#define DEMB_HIER_TRACE 0
#endif

namespace dyn_emb {
namespace hier_output_dist {

constexpr bool kTrace = DEMB_HIER_TRACE != 0;

}  // namespace hier_output_dist
}  // namespace dyn_emb
