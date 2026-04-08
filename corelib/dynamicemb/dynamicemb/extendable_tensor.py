# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
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
import platform
from typing import List, Tuple, Union

import torch
from dynamicemb_extensions import HostVMMTensor, VMMTensor


def _shape_to_numel(shape: Union[List[int], Tuple[int, ...]]) -> int:
    """Return total number of elements for a shape (list or tuple)."""
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _merge_extend_shape(
    current_shape: Tuple[int, ...],
    extend_shape: Union[List[int], Tuple[int, ...]],
    added_numel: int,
) -> Tuple[int, ...]:
    """Return new shape after extending. Prefer multidimensional when extend_shape matches trailing dims."""
    # Normalize to tuple so list vs tuple does not break equality (e.g. [D] != (D,)).
    ext = tuple(int(d) for d in extend_shape)
    if (
        len(ext) == len(current_shape)
        and len(current_shape) >= 1
        and ext[1:] == current_shape[1:]
    ):
        return (current_shape[0] + ext[0], *current_shape[1:])
    total_numel = _shape_to_numel(current_shape) + added_numel
    return (total_numel,)


class ExtendableBuffer(abc.ABC):
    """Base for host/device extendable buffers. Use is_device_buffer to distinguish
    device (HBM) from host; tensor.is_cuda can be True for host memory registered
    to CUDA address space and is not reliable for this."""

    @abc.abstractmethod
    def num_bytes(self) -> int:
        """Logical buffer size in bytes (user-visible tensor storage)."""

    @abc.abstractmethod
    def allocated_bytes(self) -> int:
        """Bytes actually mapped/backed by the native buffer (page-aligned, >= num_bytes())."""

    @abc.abstractmethod
    def extend(self, shape: Union[List[int], Tuple[int, ...]]) -> None:
        """Extend buffer by the number of elements given by product(shape)."""

    @abc.abstractmethod
    def tensor(self) -> torch.Tensor:
        """Return the buffer as a tensor (multidimensional when possible)."""

    def numel(self) -> int:
        """Return current number of elements (num_bytes() // element_size)."""
        return self.num_bytes() // self.element_size

    @property
    def element_size(self) -> int:
        """Bytes per element for the buffer dtype."""
        return torch.empty(0, dtype=self._dtype).element_size()

    def is_device_buffer(self) -> bool:
        """True if storage is GPU device memory (HBM); False if host memory.
        Prefer this over tensor().is_cuda when host may be CUDA-registered."""
        return False


class DeviceExtendableBuffer(ExtendableBuffer):
    """Extendable buffer in device (HBM) memory. shape is (list or tuple) initial dimensions."""

    def __init__(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        dtype: torch.dtype,
        device: torch.device = None,
    ):
        device_id = device.index if device is not None else torch.cuda.current_device()
        numel = _shape_to_numel(shape)
        if numel == 0:
            raise ValueError("shape must have positive total size")
        self._shape = tuple(int(d) for d in shape)
        self._dtype = dtype
        self._device = device
        self.vmm_tensor = VMMTensor(numel, dtype, device_id)

    def is_device_buffer(self) -> bool:
        return True

    def tensor(self) -> torch.Tensor:
        """Return buffer as tensor with logical shape (size from C++)."""
        return self.vmm_tensor.data().view(self._shape)

    def num_bytes(self) -> int:
        return self.vmm_tensor.data().numel() * self.element_size

    def allocated_bytes(self) -> int:
        return int(self.vmm_tensor.allocated_bytes())

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def extend(self, shape: Union[List[int], Tuple[int, ...]]) -> None:
        numel_add = _shape_to_numel(shape)
        if numel_add <= 0:
            return
        torch.cuda.synchronize()
        new_logical_numel = _shape_to_numel(self._shape) + numel_add
        self.vmm_tensor.extend(new_logical_numel)
        self._shape = _merge_extend_shape(self._shape, shape, numel_add)


class HostExtendableBuffer(ExtendableBuffer):
    """Extendable buffer in host memory (UVM). shape is (list or tuple) initial dimensions."""

    def __init__(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        dtype: torch.dtype,
        device: torch.device,
    ):
        if platform.system() != "Linux":
            raise RuntimeError("Only support extendable host buffer on Linux platform.")
        device_id = device.index if device is not None else torch.cuda.current_device()
        numel = _shape_to_numel(shape)
        if numel == 0:
            raise ValueError("shape must have positive total size")
        self._shape = tuple(int(d) for d in shape)
        self._dtype = dtype
        self._device = device
        self.vmm_tensor = HostVMMTensor(numel, dtype, device_id)

    def extend(self, shape: Union[List[int], Tuple[int, ...]]) -> None:
        numel_add = _shape_to_numel(shape)
        if numel_add <= 0:
            return
        torch.cuda.synchronize()
        new_logical_numel = _shape_to_numel(self._shape) + numel_add
        self.vmm_tensor.extend(new_logical_numel)
        self._shape = _merge_extend_shape(self._shape, shape, numel_add)

    def tensor(self) -> torch.Tensor:
        """Return buffer as tensor with logical shape (size from C++)."""
        return self.vmm_tensor.data().view(self._shape)

    def num_bytes(self) -> int:
        return self.vmm_tensor.data().numel() * self.element_size

    def allocated_bytes(self) -> int:
        return int(self.vmm_tensor.allocated_bytes())

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape
