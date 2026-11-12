// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions derived from Mega-Flink (https://github.com/cb521/Mega-Flink, Apache-2.0).
// Modified: device-admission macro generalised (see comment below); derived from Mega-Flink native/gpu_arch_compat.h.

#pragma once

#include <cuda_runtime.h>

#include <cstring>

// Device admission. Without any macro the original rule applies: only Hopper parts whose
// name contains "H20" are admitted. build.sh defines FEP_ADMITTED_DEVICE_MAJORS_MASK as a
// bit mask over the CUDA compute-capability majors that were compiled in (bit 8 for sm_89,
// bit 9 for sm_90a), so a build that targets L20 (sm_89) admits major 8, and a multi-arch
// build admits every major it contains. The older single-major macro is still honoured.
// Correctness tests must pass on a target before its results are accepted.
inline bool mega_flink_device_matches_native_build(
    const cudaDeviceProp& properties) {
#if defined(FEP_ADMITTED_DEVICE_MAJORS_MASK)
  return properties.major >= 0 && properties.major < 31 &&
      ((FEP_ADMITTED_DEVICE_MAJORS_MASK >> properties.major) & 1u) != 0u;
#elif defined(CUDF_FLINK_EXPERIMENTAL_DEVICE_MAJOR)
  return properties.major == CUDF_FLINK_EXPERIMENTAL_DEVICE_MAJOR;
#else
  return properties.major == 9 &&
      std::strstr(properties.name, "H20") != nullptr;
#endif
}
