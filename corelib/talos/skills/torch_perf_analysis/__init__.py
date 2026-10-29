# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""torch_perf_analysis — global-hook NVTX instrumentation + nsys trace analysis.

No model wrapping needed — enable() once, then bracket the window and mark
each step:

    from talos.torch_perf_analysis import talos_profiler

    talos_profiler.enable()    # ONCE, before any worker thread is created

    for step, batch in enumerate(loader):
        if step == 50: talos_profiler.start()   # open capture window
        if step == 60: talos_profiler.stop()    # close it
        talos_profiler.step()                    # mark a step boundary
        loss = model(batch); loss.backward(); opt.step(); opt.zero_grad()

`enable()` pre-loads the C++ observer extension (JIT-compiled on first use,
~1 min, then cached) so the mid-training `start()` stays clean. It is
optional — `start()` loads it too.

Run under nsys with `--capture-range=cudaProfilerApi` so start()/stop()
bound the recorded region. Then drive the stats extractor on the trace:

    python3 -m talos.torch_perf_analysis.python.stats <profile.nsys-rep>
    # → writes .torch_perf_analysis/ (JSON evidence; --workspace to relocate)

Gated by TORCH_PERF_ANALYSIS_ENABLE (no-op when unset). Module hooks emit no
NVTX; they only feed the module path to the C++ observer, which is the sole
emitter: every outermost dispatched ATen op gets
`talos::<func>#<path>@<Class>!!file:line`. Only forward is
marked live; stats.py synthesizes the same label over each backward node's
`autograd::engine::evaluate_function` range via a seq join, so a backward
kernel attributes to the same module and source line as its forward
(direction, when needed, comes from that autograd range). Use `nvtx_range` to give a non-Module region
(an optimizer step, a free function) its own path + class.
"""

from .python.profiler import TalosProfiler, nvtx_range, talos_profiler

__all__ = ["TalosProfiler", "talos_profiler", "nvtx_range"]
