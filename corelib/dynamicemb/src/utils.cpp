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

#include "utils.h"

namespace dyn_emb {

DeviceProp &DeviceProp::getDeviceProp(int device_id) {
  static DeviceProp device_prop(device_id);
  return device_prop;
}

DeviceProp::DeviceProp(int device_id) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  this->num_sms = device_prop.multiProcessorCount;
  this->warp_size = device_prop.warpSize;
  this->max_thread_per_sm = device_prop.maxThreadsPerMultiProcessor;
  this->max_thread_per_block = device_prop.maxThreadsPerBlock;
  this->total_threads = this->num_sms * this->max_thread_per_sm;
}

} // namespace dyn_emb