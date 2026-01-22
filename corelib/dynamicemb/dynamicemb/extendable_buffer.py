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

import abc

import torch
from dynamicemb_extensions import VMMTensor


class ExtendableBuffer(abc.ABC):
    @abc.abstractmethod
    def capacity(self) -> int:
        capacity: int
        return capacity

    @abc.abstractmethod
    def extend(self, capacity) -> None:
        pass

    @abc.abstractmethod
    def tensor(self) -> torch.Tensor:
        tensor: torch.Tensor
        return tensor


class VmmDeviceBuffer:
    def __init__(self, capacity, dtype, device: torch.device = None):
        device_id = device.index if device is not None else torch.cuda.current_device()

        self._capacity = capacity
        self._dtype = dtype
        self._device = device_id

        self.vmm_tensor = VMMTensor(capacity, dtype, device_id)

    def extend(self, capacity) -> None:
        torch.cuda.synchronize()
        self.vmm_tensor.extend(capacity)

    def tensor(self) -> torch.Tensor:
        return self.vmm_tensor.data()

    def capacity(self) -> int:
        return self.vmm_tensor.data().numel()
