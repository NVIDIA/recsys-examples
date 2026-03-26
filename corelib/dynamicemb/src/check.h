/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef CHECK_H
#define CHECK_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

#define CUDACHECK(cmd) DEMB_CUDA_CHECK(cmd)
#define CUDA_KERNEL_LAUNCH_CHECK() DEMB_CUDA_KERNEL_LAUNCH_CHECK()

// CUDA_ERROR_CHECK is used in HKV.
#ifndef CUDA_ERROR_CHECK && !defined(NDEMB_CUDA_ERROR_CHECK)
inline void __cudaCheckError(cudaError_t err, const char *file,
                             const int line) {
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "cudaCheckError() failed at " << file << ":" << line << " : "
       << cudaGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

#define DEMB_CUDA_KERNEL_LAUNCH_CHECK() DEMB_CUDA_CHECK(cudaGetLastError())
#define DEMB_CUDA_CHECK(cmd)                                                   \
  do {                                                                         \
    cudaError_t err = cmd;                                                     \
    __cudaCheckError(err, __FILE__, __LINE__);                                 \
  } while (0)

#else
#define DEMB_CUDA_CHECK(cmd) cmd
#define DEMB_CUDA_KERNEL_LAUNCH_CHECK()
#endif

#endif // CHECK_H
