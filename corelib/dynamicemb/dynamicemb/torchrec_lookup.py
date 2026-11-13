# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TorchRec lookup over the rows resolved by DynamicEmb.

TorchRec owns gather/pooling; DynamicEmb keeps admission, storage and its fused
optimizer. The weights address the current compact unique rows, with an fp32
workspace for pooling, including initialized rejected rows. No hash table is
accessed here. Keeping this adapter outside the
nn.Module tree also keeps these temporary weight views out of checkpoints and
external optimizers.
"""

from itertools import accumulate

import torch
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import JaggedTensor

from .dynamicemb_config import DynamicEmbPoolingMode


class TorchRecLookup:
    def __init__(self, dims, feature_table_map, pooling_mode, device):
        self._feature_num = len(feature_table_map)
        self._uniform = len(set(dims)) == 1
        # A uniform-D workspace is one row-addressable matrix, even when its
        # rows belong to different persistent tables. One synthetic feature
        # lets TorchRec gather all rows / pool all bags in one call. The real
        # table ids and feature layout remain with DynamicEmb's backward.
        self._dims = [dims[0]] if self._uniform else dims
        self._feature_table_map = [0] if self._uniform else feature_table_map
        self._pooling_mode = pooling_mode
        self._device = device
        self._feature_names = [
            f"feature_{i}" for i in range(len(self._feature_table_map))
        ]
        self._modules = {}

    def _module(self, weighted):
        if weighted not in self._modules:
            pooled = self._pooling_mode != DynamicEmbPoolingMode.NONE
            config_type = EmbeddingBagConfig if pooled else EmbeddingConfig
            configs = [
                config_type(
                    name=f"table_{t}",
                    embedding_dim=dim,
                    num_embeddings=1,
                    feature_names=[
                        self._feature_names[f]
                        for f, table in enumerate(self._feature_table_map)
                        if table == t
                    ],
                    init_fn=torch.nn.init.zeros_,
                    **(
                        {
                            "pooling": PoolingType.MEAN
                            if self._pooling_mode == DynamicEmbPoolingMode.MEAN
                            else PoolingType.SUM
                        }
                        if pooled
                        else {}
                    ),
                )
                for t, dim in enumerate(self._dims)
            ]
            # nn.Embedding/EmbeddingBag initialize their weights before TorchRec
            # applies init_fn. Build on meta so a cold lookup (including after
            # checkpoint restore) does not advance the caller's CUDA/CPU RNG.
            module = (
                EmbeddingBagCollection(
                    configs, is_weighted=weighted, device=torch.device("meta")
                )
                if pooled
                else EmbeddingCollection(configs, device=torch.device("meta"))
            )
            module.to_empty(device=self._device)
            with torch.no_grad():
                for weight in module.parameters():
                    weight.zero_()
                    weight.requires_grad_(False)
            self._modules[weighted] = module
        return self._modules[weighted]

    def __call__(
        self,
        unique_embs,
        batch,
        offsets,
        output_dtype,
        pooling_weights=None,
        *,
        indices=None,
    ):
        # Normally indices address the compact unique workspace. A single HBM
        # table/cache can instead bind its persistent buffer and physical rows.
        indices = batch.reverse_indices if indices is None else indices
        module = self._module(pooling_weights is not None)
        pooled = self._pooling_mode != DynamicEmbPoolingMode.NONE
        # DynamicEmb accumulates pooling in fp32, independently of the stored
        # embedding and output dtypes. Native EmbeddingBag returns its weight
        # dtype and requires sample weights to have that same dtype. Promote
        # only this temporary embedding workspace; optimizer rows stay intact.
        if pooled:
            unique_embs = unique_embs.float()
        embeddings = module.embedding_bags if pooled else module.embeddings
        table_offsets = (
            [0, unique_embs.size(0)]
            if self._uniform
            else [0] + list(accumulate(batch.unique_sizes))
        )
        placeholders = [
            (embedding.weight.data, embedding.num_embeddings)
            for embedding in embeddings.values()
        ]
        for t, (embedding, dim) in enumerate(zip(embeddings.values(), self._dims)):
            # Native embedding/embedding_bag accept strided weights. In mixed-D
            # groups each row occupies max_D elements, with padding after dim.
            weight = unique_embs[table_offsets[t] : table_offsets[t + 1], :dim]
            # The cached parameters are frozen and outside the model tree.
            # Rebind their data rather than allocating/registering a Parameter
            # for every batch; DynamicEmb's custom backward owns all updates.
            embedding.weight.data = weight
            embedding.num_embeddings = weight.size(0)
        if self._uniform:
            count = indices.numel()
            lengths = (
                offsets[1:] - offsets[:-1]
                if pooled
                else torch.full(
                    (1,), count, dtype=torch.int64, device=unique_embs.device
                )
            )
            # This synthetic feature already has its JT representation. Pass
            # the public cache so native lookup does not split it or reconstruct
            # offsets with another GPU scan/cat (pooling uses our input offsets).
            jt = JaggedTensor(
                values=indices,
                lengths=lengths,
                offsets=offsets if pooled else None,
                weights=pooling_weights,
            )
            # These lengths are known on the host; supplying KJT metadata
            # avoids a device-to-host length reduction in native to_dict().
            features = KeyedJaggedTensor(
                keys=self._feature_names,
                values=indices,
                offsets=offsets if pooled else None,
                lengths=lengths,
                weights=pooling_weights,
                stride=offsets.numel() - 1 if pooled else 1,
                length_per_key=[count],
                offset_per_key=[0, count],
                jt_dict={self._feature_names[0]: jt},
            )
        else:
            starts = torch.tensor(table_offsets[:-1], device=unique_embs.device)
            indices = (
                batch.reverse_indices
                - starts[batch.unique_table_ids[batch.reverse_indices]]
            )
            features = KeyedJaggedTensor(
                keys=self._feature_names,
                values=indices,
                offsets=offsets,
                weights=pooling_weights,
            )
        try:
            result = module(features)
            if pooled:
                values = result.values()
                if self._uniform:
                    batch_size = (offsets.numel() - 1) // self._feature_num
                    values = (
                        values.view(self._feature_num, batch_size, self._dims[0])
                        .transpose(0, 1)
                        .reshape(batch_size, self._feature_num * self._dims[0])
                    )
                return values.to(output_dtype)
            return result[self._feature_names[0]].values().to(output_dtype)
        finally:
            # The table owns persistent weights. Do not retain a whole batch
            # workspace in the cached TorchRec module after lookup.
            for embedding, (weight, size) in zip(embeddings.values(), placeholders):
                embedding.weight.data = weight
                embedding.num_embeddings = size
