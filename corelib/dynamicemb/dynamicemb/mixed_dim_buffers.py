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

"""The value rows of tables that differ in embedding dimension.

A module merges tables whose embedding dimensions (and so the widths of their rows,
with the optimizer state) differ. Each table has a buffer of its own, ``[rows, value_dim]``
for its own ``value_dim``, and the kernels that read and write rows are given the whole
set at once: the base pointers of the buffers, the dimensions of every table, the widest
of them. :class:`MixedDimBuffers` keeps those together, so nothing outside has to keep
them in step -- when a buffer grows its pointer is refreshed here.

What an optimizer needs to know about the rows -- where each table's rows are, how wide the
embedding and the whole row are -- is :class:`BuffersView`, taken from :attr:`MixedDimBuffers.view`.
The optimizer works on rows through it; the buffers themselves know nothing of optimizers.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch
from dynamicemb.extendable_tensor import (
    DeviceExtendableBuffer,
    ExtendableBuffer,
    FixedDeviceBuffer,
    HostExtendableBuffer,
)
from dynamicemb.types import CopyMode
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value


@dataclass(frozen=True)
class BuffersView:
    """A non-owning view of the buffers of :class:`MixedDimBuffers`, for the optimizer
    (the base pointers go stale when a buffer grows).

    A row is the embedding (``emb_dims[t]`` wide for table ``t``) followed by the optimizer
    state, ``value_dims[t]`` wide in all. In a padded buffer (the output of ``load_padded``) the
    state of every row starts at ``max_emb_dim``, whatever the table. ``ptrs`` are the base
    pointers of the buffers, valid until one of them grows.
    """

    ptrs: torch.Tensor
    dtype: torch.dtype
    value_dims: torch.Tensor
    emb_dims: torch.Tensor
    max_emb_dim: int
    max_value_dim: int
    all_dims_vec4: bool


class MixedDimBuffers:
    """One ``[rows, value_dim]`` buffer per table, and what the row kernels need of them.

    ``kind`` says where the rows live and whether they can grow: ``"device"`` and ``"host"``
    buffers are extendable (``grow_to``), ``"fixed"`` ones are a device pool of a size
    set at construction (a cache's).
    """

    def __init__(
        self,
        emb_dims: Sequence[int],
        value_dims: Sequence[int],
        row_capacities: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        kind: str,
    ):
        if not len(emb_dims) == len(value_dims) == len(row_capacities):
            raise ValueError("one embedding dim, value dim and row count per table")
        if kind not in ("device", "host", "fixed"):
            raise ValueError(f"kind is 'device', 'host' or 'fixed', not {kind!r}")
        self.dtype = dtype
        self.device = device
        self.kind = kind
        self.num_tables = len(emb_dims)

        self.emb_dims_list: List[int] = [int(d) for d in emb_dims]
        self.value_dims_list: List[int] = [int(d) for d in value_dims]
        self.max_emb_dim = max(self.emb_dims_list)
        self.max_value_dim = max(self.value_dims_list)
        # Rows of every table are read in units of 4 elements when all of them allow it.
        self.all_dims_vec4 = all(d % 4 == 0 for d in self.emb_dims_list) and all(
            v % 4 == 0 for v in self.value_dims_list
        )
        # The dims as the kernels take them: on the device, and one copy on the host.
        self.emb_dims = torch.tensor(self.emb_dims_list, dtype=torch.int64, device=device)
        self.emb_dims_host = torch.tensor(self.emb_dims_list, dtype=torch.int64)
        self.value_dims = torch.tensor(self.value_dims_list, dtype=torch.int64, device=device)

        make = {
            "device": DeviceExtendableBuffer,
            "host": HostExtendableBuffer,
            "fixed": FixedDeviceBuffer,
        }[kind]
        self.buffers: List[ExtendableBuffer] = [
            make((int(rows), vd), dtype, device)
            for rows, vd in zip(row_capacities, self.value_dims_list)
        ]
        self.ptrs = self._base_pointers()

    # -- view ---------------------------------------------------------------

    @property
    def view(self) -> BuffersView:
        return BuffersView(
            self.ptrs,
            self.dtype,
            self.value_dims,
            self.emb_dims,
            self.max_emb_dim,
            self.max_value_dim,
            self.all_dims_vec4,
        )

    def _base_pointers(self) -> torch.Tensor:
        return torch.tensor(
            [b.tensor().data_ptr() for b in self.buffers],
            dtype=torch.int64,
            device=self.device,
        )

    @property
    def on_device(self) -> bool:
        return self.buffers[0].is_device_buffer()

    def tensor(self, table: int) -> torch.Tensor:
        return self.buffers[table].tensor()

    def num_rows(self, table: int) -> int:
        return int(self.buffers[table].tensor().size(0))

    def grow_to(self, table: int, num_rows: int) -> bool:
        """Grow a table's buffer to at least ``num_rows`` rows (a target, not an increment);
        nothing happens if it has that many. Returns whether it grew; its pointer, and with
        it ``ptrs``, is refreshed. A ``"fixed"`` buffer raises if it would have to grow."""
        have = self.num_rows(table)
        if num_rows <= have:
            return False
        self.buffers[table].extend((num_rows - have, self.value_dims_list[table]))
        self.ptrs.copy_(self._base_pointers())
        return True

    def emb_dim(self, table: int) -> int:
        return self.emb_dims_list[table]

    def value_dim(self, table: int) -> int:
        return self.value_dims_list[table]

    # -- rows -----------------------------------------------------------------

    def load_padded(
        self, rows: torch.Tensor, table_ids: torch.Tensor, copy_mode: CopyMode
    ) -> torch.Tensor:
        """Rows of the given (row, table) pairs, padded to the widest table: the embedding
        only for ``CopyMode.EMBEDDING``, the whole value row otherwise. Each row is valid
        only up to its own table's width; the columns after it are uninitialized."""
        if copy_mode == CopyMode.EMBEDDING:
            width, kernel = self.max_emb_dim, _load_emb
        else:
            width, kernel = self.max_value_dim, _load_value
        out = torch.empty(rows.numel(), width, dtype=self.dtype, device=self.device)
        if rows.numel() > 0:
            kernel(
                self.ptrs,
                rows,
                table_ids,
                out,
                self.value_dims,
                self.emb_dims,
                self.max_emb_dim,
                self.all_dims_vec4,
            )
        return out

    def store_padded(
        self, rows: torch.Tensor, table_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write padded value rows (each as wide as the widest table) to (row, table) pairs;
        only each table's own width of a row is written."""
        if values.dim() == 1:
            values = values.unsqueeze(1)
        _store_value(
            self.ptrs,
            rows,
            table_ids,
            values.to(self.dtype),
            self.value_dims,
            self.emb_dims,
            self.max_emb_dim,
            self.all_dims_vec4,
        )

    def load_exact(self, rows: torch.Tensor, table: int) -> torch.Tensor:
        """The whole value rows of one table at exactly its width, no padding: ``[n, value_dim]``."""
        out = torch.empty(
            rows.numel(), self.value_dims_list[table], dtype=self.dtype, device=self.device
        )
        if rows.numel() > 0:
            _load_contiguous(
                self.ptrs,
                rows,
                table,
                out,
                self.value_dims,
                self.emb_dims,
                self.max_emb_dim,
                self.all_dims_vec4,
            )
        return out

    def store_exact(self, rows: torch.Tensor, table: int, values: torch.Tensor) -> None:
        """Write the whole value rows of one table at exactly its width, ``[n, value_dim]``."""
        if rows.numel() == 0:
            return
        if values.dim() == 1:
            values = values.unsqueeze(1)
        _store_contiguous(
            self.ptrs,
            rows,
            table,
            values.to(self.dtype),
            self.value_dims,
            self.emb_dims,
            self.max_emb_dim,
            self.all_dims_vec4,
        )
