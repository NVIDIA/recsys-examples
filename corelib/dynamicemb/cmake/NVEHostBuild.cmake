# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pass through CMAKE_TOOLCHAIN_FILE: NVE 26.10's wheel builder has no CMAKE_ARGS.
set(NVE_DISABLE_PLUGINS ON CACHE BOOL "Use NVE's minimal feature set")
set(NVE_WITH_LIBNUMA OFF CACHE BOOL "Omit optional NUMA dependency")
set(NVE_WITH_LIBHWLOC OFF CACHE BOOL "Omit optional hwloc dependency")
set(NVE_ENABLE_NVTX OFF CACHE BOOL "Omit profiling instrumentation")
