# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

# pyre-strict
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Load exportable ops for dynamicemb
# ---------------------------------------------------------------------------


_ops_loaded: bool = False
_ops_load_attempted: bool = False


def _load_inference_emb_ops() -> bool:
    """Load `inference_emb_ops.so` once and return whether it succeeded."""
    global _ops_loaded, _ops_load_attempted

    if _ops_loaded:
        return True
    if _ops_load_attempted:
        return False

    _ops_load_attempted = True
    _DYNAMICEMB_OPS_LIB_DIR = os.getenv("DYNAMICEMB_OPS_LIB_DIR", "")
    lib_path = os.path.join(_DYNAMICEMB_OPS_LIB_DIR, "inference_emb_ops.so")
    try:
        torch.ops.load_library(lib_path)
        print(f"[INFO] Loaded inference_emb_ops.so from {lib_path}")
        _ops_loaded = True
    except Exception as _e:
        if not os.path.exists(lib_path):
            print(
                f"[ERROR] inference_emb_ops.so not found at {_DYNAMICEMB_OPS_LIB_DIR}."
            )
        raise RuntimeError(f"[WARN] Failed to load {lib_path}: {_e}")

    return _ops_loaded


# Load operators before register fake ops.
# isort: off
_load_inference_emb_ops()  # registers torch.ops.INFERENCE_EMB.* before import dynamicemb
import dynamicemb.index_range_meta as _index_range_meta  # noqa: F401 – registers fake impls for torch.export
import dynamicemb.lookup_meta as _lookup_meta  # noqa: F401 – registers fake impls for torch.export

import hstu_cuda_ops  # noqa: F401 – registers torch.ops.hstu_cuda_ops.*
import commons.ops.cuda_ops.fake_hstu_cuda_ops  # noqa: F401 – registers fake impls for torch.export

# isort: on


# ---------------------------------------------------------------------------
# ExportableEmbedding Module
# ---------------------------------------------------------------------------


from configs import InferenceEmbeddingConfig
from dynamicemb import (
    DynamicEmbInitializerArgs,
    DynamicEmbInitializerMode,
    DynamicEmbTableOptions,
)
from dynamicemb.exportable_tables import InferenceEmbeddingTable
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class ExportableEmbedding(torch.nn.Module):
    """
    ExportableEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
        embedding_backend (EmbeddingBackend): Embedding collection backend.
    """

    def __init__(
        self,
        embedding_configs: List[InferenceEmbeddingConfig],
        feature_to_index: Optional[Dict[str, int]] = None,
    ):
        super(ExportableEmbedding, self).__init__()

        self._embedding_configs = embedding_configs

        self.table_options = [
            DynamicEmbTableOptions(
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                dim=config.dim,
                max_capacity=config.vocab_size,
                local_hbm_for_values=0,
                bucket_capacity=128,
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.NORMAL,
                ),
                training=False,
            )
            for config in embedding_configs
        ]
        self.pooling_mode = -1
        self.table_names = [config.table_name for config in embedding_configs]
        self.feature_table_map = []
        self.feature_to_index = {} if feature_to_index is None else feature_to_index
        last_index = 0
        for config in embedding_configs:
            for fea_name in config.feature_names:
                self.feature_table_map.append(self.table_names.index(config.table_name))
                self.feature_to_index[fea_name] = last_index
                last_index += 1
        self.num_features = len(self.feature_to_index)

        self._embedding_table = InferenceEmbeddingTable(
            self.table_options,
            self.pooling_mode,
            self.table_names,
            self.feature_table_map,
            device=torch.device("cuda"),
        )

    def load_checkpoint(self, checkpoint_dir, model_state_dict=None):
        if checkpoint_dir is None:
            return

        embedding_table_dir = os.path.join(
            checkpoint_dir,
            "dynamicemb_module",
            "model._embedding_collection._model_parallel_embedding_collection",
        )

        self._embedding_table.load(
            embedding_table_dir, self.table_names
        )  # load from dynamic embedding checkpoint format
        print(f"[INFO] Loaded embedding tables from {embedding_table_dir}")

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        reduce_lengths = torch.ops.hstu_cuda_ops.lengths_reduce_dim1(
            kjt.lengths(), self.num_features
        )
        offsets = torch.zeros(
            (reduce_lengths.shape[0] + 1,),
            dtype=reduce_lengths.dtype,
            device=reduce_lengths.device,
        )
        offsets[1:] = torch.cumsum(reduce_lengths, dim=0)
        total_embeddings = self._embedding_table(kjt.values(), offsets)
        split_embeddings = torch.ops.hstu_cuda_ops.split_by_lengths(
            total_embeddings, reduce_lengths, self.num_features
        )
        split_lengths = torch.ops.hstu_cuda_ops.lengths_splits(
            kjt.lengths(), self.num_features
        )
        embeddings = {}
        for k in kjt.keys():
            embeddings[k] = JaggedTensor(
                values=split_embeddings[self.feature_to_index[k]],
                lengths=split_lengths[self.feature_to_index[k]],
            )
        return embeddings


def get_exportable_embedding(
    embedding_configs: List[InferenceEmbeddingConfig],
    feature_to_index: Optional[Dict[str, int]] = None,
):
    return ExportableEmbedding(
        embedding_configs,
        feature_to_index,
    )
