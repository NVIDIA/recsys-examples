# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense forward operators used by TorchRec's batched embedding lookup.

DynamicEmb owns backward/optimizer state, so only the forward operators are
needed. Borrow the input's flat storage for one call; cache immutable metadata,
never weights or a batch. Strides that are whole embedding rows can be expressed
as a row-index multiplier without packing or exposing optimizer columns.
"""

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

from .dynamicemb_config import DynamicEmbPoolingMode

OUTPUT_TYPES = {
    torch.float32: SparseType.FP32.as_int(),
    torch.float16: SparseType.FP16.as_int(),
    torch.bfloat16: SparseType.BF16.as_int(),
}


class DenseLookup:
    def __init__(self, dim, pooling_mode, weighted, device):
        self._dim = dim
        self._pooling = {
            DynamicEmbPoolingMode.NONE: PoolingMode.NONE.value,
            DynamicEmbPoolingMode.SUM: PoolingMode.SUM.value,
            DynamicEmbPoolingMode.MEAN: PoolingMode.MEAN.value,
        }[pooling_mode]
        if pooling_mode == DynamicEmbPoolingMode.NONE:
            name = "dense_embedding_nobag_codegen_forward_unweighted_cuda"
        elif weighted:
            name = "dense_embedding_codegen_forward_weighted_cuda"
        else:
            name = "dense_embedding_codegen_forward_unweighted_cuda"
        self._operation = getattr(torch.ops.fbgemm, name)
        self._weighted = weighted
        self._weights_offsets = torch.zeros(1, device=device, dtype=torch.int64)
        self._d_offsets = (
            torch.tensor([0, dim], device=device, dtype=torch.int32)
            if pooling_mode != DynamicEmbPoolingMode.NONE
            else None
        )
        self._device_index = self._weights_offsets.device.index
        stream = torch.cuda.current_stream(self._device_index)
        self._streams = {stream.cuda_stream}
        self._ready = torch.cuda.Event()
        self._ready.record(stream)

    def forward(self, weights, indices, offsets, output_dtype, sample_weights):
        stream = torch.cuda.current_stream(self._device_index)
        if stream.cuda_stream not in self._streams:
            stream.wait_event(self._ready)
            # Allocator stream registration persists for this allocation. It
            # also protects metadata if the adapter is dropped with work queued.
            self._weights_offsets.record_stream(stream)
            if self._d_offsets is not None:
                self._d_offsets.record_stream(stream)
            self._streams.add(stream.cuda_stream)

        dim, stride, rows = self._dim, weights.stride(0), weights.size(0)
        # Exclude the last row's padding/moments. The resulting view also works
        # for a sliced matrix whose storage ends at its final embedding column.
        count = (rows - 1) * stride + dim if rows else 0
        flat = weights.as_strided((count,), (1,))
        if stride != dim:
            indices = indices * (stride // dim)
        if offsets.dtype != indices.dtype:
            offsets = offsets.to(indices.dtype)
        dtype = OUTPUT_TYPES[output_dtype]
        if self._d_offsets is None:
            return self._operation(
                flat, self._weights_offsets, dim, indices, offsets, dtype, False
            )
        args = (
            flat,
            self._weights_offsets,
            self._d_offsets,
            dim,
            dim,
            indices,
            offsets,
            self._pooling,
        )
        if self._weighted:
            return self._operation(*args, sample_weights, dtype, False)
        return self._operation(*args, dtype, False)
