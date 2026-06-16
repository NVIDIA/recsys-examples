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
from dataclasses import dataclass

# pyre-strict
from typing import Dict, List, Optional, Union

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


from commons.modules.embedding import ShardedEmbedding, ShardedEmbeddingConfig
from dynamicemb.exportable_tables import (
    InferenceEmbeddingCollection,
    create_inference_embedding_collection,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


# Register fake impl for nve_ops::embedding_lookup
@torch.library.register_fake("nve_ops::embedding_lookup", allow_override=True)
def _f(keys, layer_id):
    ctx = torch.library.get_ctx()
    return keys.new_empty((keys.size(0), ctx.new_dynamic_size()), dtype=torch.float32)


@dataclass
class InferenceEmbeddingConfig(ShardedEmbeddingConfig):
    use_dynamic: bool = False

    def __init__(
        self, config: ShardedEmbeddingConfig, use_dynamic: Optional[bool] = None
    ):
        self.feature_names = config.feature_names
        self.table_name = config.table_name
        self.vocab_size = config.vocab_size
        self.dim = config.dim
        self.sharding_type = config.sharding_type
        self.use_dynamic = (
            (self.sharding_type == "model_parallel")
            if use_dynamic is None
            else use_dynamic
        )


class ExportableEmbedding(torch.nn.Module):
    """
    ExportableEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
        embedding_backend (EmbeddingBackend): Embedding collection backend.
    """

    def __init__(
        self,
        static_embedding_configs: Union[
            List[EmbeddingConfig], List[InferenceEmbeddingConfig]
        ],
        dynamic_embedding_configs: Union[
            List[EmbeddingConfig], List[InferenceEmbeddingConfig]
        ],
        static_embedding_collection: Optional[InferenceEmbeddingCollection] = None,
        dynamic_embedding_collection: Optional[InferenceEmbeddingCollection] = None,
    ):
        super(ExportableEmbedding, self).__init__()
        assert (
            len({type(config) for config in static_embedding_configs}) <= 1
        ), "All embedding configs should be of the same type, either EmbeddingConfig or RecsysEmbeddingConfig"
        assert (
            len({type(config) for config in dynamic_embedding_configs}) <= 1
        ), "All embedding configs should be of the same type, either EmbeddingConfig or RecsysEmbeddingConfig"

        for idx, config in enumerate(static_embedding_configs):
            if isinstance(config, InferenceEmbeddingConfig):
                static_embedding_configs[idx] = EmbeddingConfig(
                    feature_names=config.feature_names,
                    name=config.table_name,
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.dim,
                )
        for idx, config in enumerate(dynamic_embedding_configs):
            if isinstance(config, InferenceEmbeddingConfig):
                dynamic_embedding_configs[idx] = EmbeddingConfig(
                    feature_names=config.feature_names,
                    name=config.table_name,
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.dim,
                )

        self._static_embedding_configs = static_embedding_configs
        self._dynamic_embedding_configs = dynamic_embedding_configs

        if static_embedding_collection is None:
            static_embedding_collection = create_inference_embedding_collection(
                self._static_embedding_configs, pooling_mode=-1, use_dynamic=False
            )
        if dynamic_embedding_collection is None:
            dynamic_embedding_collection = create_inference_embedding_collection(
                self._dynamic_embedding_configs, pooling_mode=-1, use_dynamic=True
            )
        self._static_embedding_collection = static_embedding_collection
        self._dynamic_embedding_collection = dynamic_embedding_collection

        self._feature_names = (
            self._static_embedding_collection.feature_names_
            + self._dynamic_embedding_collection.feature_names_
        )
        # per collection feature name to index mapping for splitting the output embeddings
        self._static_feature_to_index = {
            fea_name: idx
            for idx, fea_name in enumerate(
                self._static_embedding_collection.feature_names_
            )
        }
        self._dynamic_feature_to_index = {
            fea_name: idx
            for idx, fea_name in enumerate(
                self._dynamic_embedding_collection.feature_names_
            )
        }

        self._has_uninitialized_input_dist = True
        self._need_features_permute = False

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
    ) -> None:
        features_order: List[int] = []
        for f in self._feature_names:
            features_order.append(input_feature_names.index(f))
        self._features_order = features_order
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(features_order, device=device, dtype=torch.int32),
            persistent=False,
        )
        if self._features_order != list(range(len(input_feature_names))):
            print(
                f"[WARNING] The input feature order {input_feature_names} is different from the order in the model."
            )
            print(
                f"          Permuting the input features to match the model order may degrade performance."
            )
            print(
                f"          Consider changing the input feature order to {self._feature_names} to avoid unnecessary permutation."
            )
            self._need_features_permute = True
        self._has_uninitialized_input_dist = False

    def load_checkpoint(
        self,
        checkpoint_dir: Optional[str],
        model_state_dict: Dict[str, torch.Tensor],
        static_module_name: Optional[str] = None,
        dynamic_module_name: Optional[str] = None,
    ) -> None:
        if checkpoint_dir is None:
            return

        if static_module_name is None:
            static_module_name = (
                "_embedding_collection._data_parallel_embedding_collection"
            )
        if dynamic_module_name is None:
            dynamic_module_name = (
                "_embedding_collection._model_parallel_embedding_collection"
            )

        param_name = (
            static_module_name
            + ".embeddings."
            + "/".join(self._static_embedding_collection.table_names_)
            + "_weights"
        )
        if param_name not in model_state_dict:
            raise ValueError(
                "Cannot find static embedding table weights from model_state_dict"
            )
        static_emb_table_weights = model_state_dict[param_name].view(
            -1, self._static_embedding_collection.emb_dim_
        )
        self._static_embedding_collection.load_from_embedding_table(
            static_emb_table_weights
        )

        self._dynamic_embedding_collection.load_from_dynamicemb_file(
            os.path.join(
                checkpoint_dir, "dynamicemb_module", "model." + dynamic_module_name
            ),
            self._dynamic_embedding_collection.table_names_,
        )
        print(f"[INFO] Loaded embedding tables from {checkpoint_dir}")

    def get_embedding_dict(self, features, lengths, use_dynamic=False):
        feature_to_index = (
            self._dynamic_feature_to_index
            if use_dynamic
            else self._static_feature_to_index
        )
        num_features = len(feature_to_index)
        embedding_collection = (
            self._dynamic_embedding_collection
            if use_dynamic
            else self._static_embedding_collection
        )

        reduce_lengths = torch.ops.hstu_cuda_ops.lengths_reduce_dim1(
            lengths, num_features
        )
        offsets = torch.zeros(
            (reduce_lengths.shape[0] + 1,),
            dtype=reduce_lengths.dtype,
            device=reduce_lengths.device,
        )
        offsets[1:] = torch.cumsum(reduce_lengths, dim=0)

        total_embeddings = embedding_collection(features, offsets)
        torch._check(total_embeddings.size(1) == embedding_collection.emb_dim_)
        split_embeddings = torch.ops.hstu_cuda_ops.split_by_lengths(
            total_embeddings, reduce_lengths, num_features
        )
        split_lengths = lengths.view(num_features, -1)
        embeddings = dict()
        for k in feature_to_index.keys():
            embeddings[k] = JaggedTensor(
                values=split_embeddings[feature_to_index[k]],
                lengths=split_lengths[feature_to_index[k]],
            )
        return embeddings

    # @output_nvtx_hook(nvtx_tag="InferenceEmbedding", hook_tensor_attr_name="_values")
    def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Forward pass of the sharded embedding module.

        Args:
            kjt (`KeyedJaggedTensor <https://pytorch.org/torchrec/concepts.html#keyedjaggedtensor>`): The input tokens.

        Returns:
            `Dict[str, JaggedTensor <https://pytorch.org/torchrec/concepts.html#jaggedtensor>]`: The output embeddings.
        """

        # Step.1 create input distribution and permute features if necessary
        if not torch.compiler.is_compiling():
            if self._has_uninitialized_input_dist:
                self._create_input_dist(kjt.keys(), kjt.values().device)
        else:
            assert (
                not self._has_uninitialized_input_dist
            ), "Input distribution should have been initialized before compilation."

        # if self._need_features_permute:
        #

        # Step.2 split features and lengths by embedding collections
        # Using cpp bind to avoid dynamic shape tracing failure in torch.export
        """Equivalent torch code:
            kjt = kjt.permute(
                self._features_order,
                self._features_order_tensor,
            )
            lengths_per_collections = torch.split(kjt.lengths(), self._num_features_in_collection, dim=0)
            split_sizes = torch.cat(
                [ lengths.sum() for lengths in self._num_features_in_collection ], dim=0
            ).tolist()  # `tolist` causing D2H transfer
            features_per_collections = torch.split(all_features, split_sizes, dim=0)
        """
        num_static_fea = len(self._static_embedding_collection.feature_names_)
        num_dynamic_fea = len(self._dynamic_embedding_collection.feature_names_)
        jagged_offsets = torch.zeros(
            (kjt.lengths().shape[0] + 1,),
            dtype=kjt.lengths().dtype,
            device=kjt.lengths().device,
        )
        jagged_offsets[1:] = torch.cumsum(kjt.lengths(), dim=0)
        (
            static_features,
            dynamic_features,
            static_features_lengths,
            dynamic_features_lengths,
        ) = torch.ops.hstu_cuda_ops.permute_and_split(
            kjt.values(),
            kjt.lengths(),
            jagged_offsets,
            num_static_fea,
            num_dynamic_fea,
            self._features_order,
        )

        # Step.4 perform embedding lookup for each collection and split the output embeddings
        static_embs = self.get_embedding_dict(
            static_features, static_features_lengths, use_dynamic=False
        )
        dynamic_embs = self.get_embedding_dict(
            dynamic_features, dynamic_features_lengths, use_dynamic=True
        )
        return {**static_embs, **dynamic_embs}


def get_exportable_embedding(
    embedding_configs: List[InferenceEmbeddingConfig],
    static_embedding_collection: Optional[InferenceEmbeddingCollection] = None,
    dynamic_embedding_collection: Optional[InferenceEmbeddingCollection] = None,
):
    static_embedding_configs, dynamic_embedding_configs = [], []
    for config in embedding_configs:
        if config.use_dynamic:
            dynamic_embedding_configs.append(config)
        else:
            static_embedding_configs.append(config)
    return ExportableEmbedding(
        static_embedding_configs,
        dynamic_embedding_configs,
        static_embedding_collection,
        dynamic_embedding_collection,
    )


def apply_inference_sparse(
    training_embedding: ShardedEmbedding,
) -> InferenceEmbeddingCollection:
    return ExportableEmbedding(
        training_embedding._model_parallel_embedding_collection.embedding_configs,
        training_embedding._data_parallel_embedding_collection.embedding_configs,
        training_embedding._data_parallel_embedding_collection,
        training_embedding._model_parallel_embedding_collection,
    )
