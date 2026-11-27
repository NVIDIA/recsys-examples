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

What an optimizer or a kernel needs to know about the rows is a view made from them, on
demand, so it is never out of date: :meth:`MixedDimBuffers.flat_tables_view` for kernels that
read and write the tables' own buffers (where each table's rows are, how wide the embedding and
the whole row are, the element type), and :meth:`MixedDimBuffers.padded_buffer_view` for kernels
that work on a padded buffer (the output of ``load_to_padded``), which needs only the dimensions.
The buffers themselves know nothing of optimizers.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from dynamicemb.state.extendable_buffer import (
    DeviceExtendableBuffer,
    ExtendableBuffer,
    FixedDeviceBuffer,
    HostExtendableBuffer,
)
from dynamicemb.types import CopyMode
from dynamicemb.utils import encode_table_values
from dynamicemb_extensions import FlatTablesView, PaddedBufferView
from dynamicemb_extensions import load_from_flat_table_contiguous as _load_contiguous
from dynamicemb_extensions import load_from_flat_table_emb as _load_emb
from dynamicemb_extensions import load_from_flat_table_value as _load_value
from dynamicemb_extensions import store_to_flat_table_contiguous as _store_contiguous
from dynamicemb_extensions import store_to_flat_table_value as _store_value


@dataclass(frozen=True)
class PaddedBuffer:
    """A batch of padded rows -- the output of ``load_to_padded`` and the input of
    ``store_from_padded``, whatever tables they come from -- and the dims that describe it.

    A row is the embedding, padded to ``max_emb_dim``, followed by the optimizer state, in
    ``max_value_dim`` columns altogether; when the two widths are equal there is no state.
    ``values`` is ``[num_rows, max_value_dim]``, and ``view`` carries what is not in its
    shape: each table's own embedding dimension (``view.emb_dims``, on the device) and the
    two maxima. This is the one place that knows where the parts of a row are.

    The buffer lives for a batch; ``view`` is built from the tables' configuration and
    shared by every buffer of the batch (see ``empty``).
    """

    values: torch.Tensor
    view: PaddedBufferView

    def __post_init__(self):
        if self.values.dim() != 2 or self.values.size(1) != self.view.max_value_dim:
            raise ValueError(
                f"values must be [rows, {self.view.max_value_dim}], "
                f"not {list(self.values.shape)}"
            )

    @classmethod
    def empty(
        cls,
        view: PaddedBufferView,
        num_rows: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PaddedBuffer":
        """Uninitialized padded rows."""
        return cls(
            torch.empty(num_rows, view.max_value_dim, dtype=dtype, device=device), view
        )

    @property
    def max_emb_dim(self) -> int:
        return self.view.max_emb_dim

    @property
    def max_value_dim(self) -> int:
        return self.view.max_value_dim

    @property
    def has_state(self) -> bool:
        return self.view.max_value_dim != self.view.max_emb_dim

    def emb(self) -> torch.Tensor:
        """The embedding columns, as a writable view."""
        return self.values[:, : self.view.max_emb_dim]

    def state(self) -> torch.Tensor:
        """The optimizer-state columns, as a writable view. Slicing, not indexing, so that
        writing to it writes to ``values``."""
        return self.values[:, self.view.max_emb_dim :]


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
        self.emb_dims = torch.tensor(
            self.emb_dims_list, dtype=torch.int64, device=device
        )
        self.emb_dims_host = torch.tensor(self.emb_dims_list, dtype=torch.int64)
        self.value_dims = torch.tensor(
            self.value_dims_list, dtype=torch.int64, device=device
        )

        make = {
            "device": DeviceExtendableBuffer,
            "host": HostExtendableBuffer,
            "fixed": FixedDeviceBuffer,
        }[kind]
        self._buffers: List[ExtendableBuffer] = [
            make((int(rows), vd), dtype, device)
            for rows, vd in zip(row_capacities, self.value_dims_list)
        ]
        self.ptrs = self._base_pointers()

    # -- views --------------------------------------------------------------

    def flat_tables_view(self) -> FlatTablesView:
        """The tables' own buffers as the kernels take them. A row is the embedding
        (``emb_dims[t]`` wide for table ``t``) followed by the optimizer state, ``value_dims[t]``
        wide in all. Made on demand from the current pointers; ``ptrs`` is refreshed in place
        when a buffer grows, so a view taken earlier sees the new pointers too."""
        return FlatTablesView(
            self.ptrs,
            self.value_dims,
            self.emb_dims,
            self.max_emb_dim,
            self.max_value_dim,
            self.all_dims_vec4,
            self.dtype,
        )

    def padded_buffer_view(self) -> PaddedBufferView:
        """A padded buffer of these tables -- ``[n, max_value_dim]`` rows, each the embedding
        padded to ``max_emb_dim`` and followed by the optimizer state, as ``load_to_padded``
        returns -- seen by the dimensions alone."""
        return PaddedBufferView(
            self.emb_dims, self.max_emb_dim, self.max_value_dim, self.all_dims_vec4
        )

    def _base_pointers(self) -> torch.Tensor:
        return torch.tensor(
            [b.tensor().data_ptr() for b in self._buffers],
            dtype=torch.int64,
            device=self.device,
        )

    def _on_device_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Row numbers as the kernels take them; a key map hands them back on the
        host after a dump."""
        return rows.to(device=self.device, dtype=torch.int64)

    @property
    def on_device(self) -> bool:
        return self._buffers[0].is_device_buffer()

    def tensor(self, table: int) -> torch.Tensor:
        return self._buffers[table].tensor()

    def num_rows(self, table: int) -> int:
        return int(self._buffers[table].tensor().size(0))

    def num_bytes(self, table: int) -> int:
        """The size of a table's buffer in bytes, rows and optimizer state both."""
        return self._buffers[table].num_bytes()

    def grow_to(self, table: int, num_rows: int) -> bool:
        """Grow a table's buffer to at least ``num_rows`` rows (a target, not an increment);
        nothing happens if it has that many. Returns whether it grew; its pointer, and with
        it ``ptrs``, is refreshed. A ``"fixed"`` buffer raises if it would have to grow.
        """
        have = self.num_rows(table)
        if num_rows <= have:
            return False
        self._buffers[table].extend((num_rows - have, self.value_dims_list[table]))
        self.ptrs.copy_(self._base_pointers())
        return True

    def emb_dim(self, table: int) -> int:
        return self.emb_dims_list[table]

    def value_dim(self, table: int) -> int:
        return self.value_dims_list[table]

    # -- rows -----------------------------------------------------------------

    def load_to_padded(
        self, rows: torch.Tensor, table_ids: torch.Tensor, copy_mode: CopyMode
    ) -> torch.Tensor:
        """Rows of the given (row, table) pairs, padded to the widest table: the embedding
        only for ``CopyMode.EMBEDDING`` (``max_emb_dim`` columns), the whole value row
        otherwise (``max_value_dim`` columns). A value row is laid out as the embedding,
        padded to ``max_emb_dim``, followed by the optimizer state; columns a table does
        not use are uninitialized. The widest row needs ``max_emb_dim`` plus the largest
        optimizer width to fit in ``max_value_dim``."""
        if copy_mode == CopyMode.EMBEDDING:
            width, kernel = self.max_emb_dim, _load_emb
        else:
            width, kernel = self.max_value_dim, _load_value
        rows = self._on_device_rows(rows)
        out = torch.empty(rows.numel(), width, dtype=self.dtype, device=self.device)
        if rows.numel() > 0:
            kernel(self.flat_tables_view(), rows, table_ids, out)
        return out

    def store_from_padded(
        self, rows: torch.Tensor, table_ids: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Write padded value rows (each as wide as the widest table, in the layout
        ``load_to_padded`` returns) to (row, table) pairs; only the columns a table uses are
        written."""
        if values.dim() == 1:
            values = values.unsqueeze(1)
        _store_value(
            self.flat_tables_view(),
            self._on_device_rows(rows),
            table_ids,
            encode_table_values(values, self.dtype),
        )

    def load(self, table: int, rows: torch.Tensor) -> torch.Tensor:
        """The whole value rows of one table at exactly its width, no padding: ``[n, value_dim]``."""
        rows = self._on_device_rows(rows)
        out = torch.empty(
            rows.numel(),
            self.value_dims_list[table],
            dtype=self.dtype,
            device=self.device,
        )
        if rows.numel() > 0:
            _load_contiguous(self.flat_tables_view(), rows, table, out)
        return out

    def store(self, table: int, rows: torch.Tensor, values: torch.Tensor) -> None:
        """Write the whole value rows of one table at exactly its width, ``[n, value_dim]``."""
        if rows.numel() == 0:
            return
        if values.dim() == 1:
            values = values.unsqueeze(1)
        _store_contiguous(
            self.flat_tables_view(),
            self._on_device_rows(rows),
            table,
            encode_table_values(values, self.dtype),
        )

    def gather_emb_by_entry(
        self,
        rows: torch.Tensor,
        table_ids: torch.Tensor,
        entry_of_row: torch.Tensor,
        out: torch.Tensor,
        out_rows: Optional[torch.Tensor] = None,
    ) -> None:
        """Into row ``i`` of ``out`` (``[entry_of_row.numel(), columns]``, of the buffers'
        dtype; row ``out_rows[i]`` (int32) with ``out_rows``) the embedding of row
        ``rows[entry_of_row[i]]`` of table ``table_ids[entry_of_row[i]]``. Rows whose
        index is negative are left as they are.

        This is ``load_to_padded`` (embedding only) followed by ``gather_embedding``
        over ``entry_of_row``, in one kernel and without the intermediate buffer."""
        _load_emb(self.flat_tables_view(), rows, table_ids, out, entry_of_row, out_rows)
