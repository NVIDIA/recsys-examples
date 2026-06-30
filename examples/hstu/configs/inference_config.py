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
from dataclasses import dataclass
from enum import Enum, unique
import os
from typing import List, Optional

import torch

from .hstu_config import HSTUPreprocessingConfig, PositionEncodingConfig


@unique
class EmbeddingBackend(Enum):
    """
    Enum class representing different embedding backends (for inference).

    Attributes:
      TORCHREC: Represents the TorchRec backend.
      DYNAMICEMB: Represents the DynamicEmb backend.
      NVEMB: Represents the NV-Embeddings backend.
    """

    TORCHREC = "TorchRec"
    DYNAMICEMB = "DynamicEmb"
    NVEMB = "NVEmb"


@dataclass
class InferenceEmbeddingConfig:
    """
    Configuration for inference embeddings with dynamic option.
    Args:
        feature_names (List[str]): The name of the features in this embedding.
        table_name (str): The name of the table.
        vocab_size (int): The size of the vocabulary.
        dim (int): The dimension size of the embeddings.
        use_dynamicemb (bool): The option for dynamic embedding.
    """

    feature_names: List[str]
    table_name: str
    vocab_size: int
    dim: int
    use_dynamicemb: bool


@dataclass
class InferenceHSTUConfig:
    """
    InferenceHSTUConfig is a configuration data class for the inference HSTU model.

    Args:
        hidden_size (int): The hidden states dimension size.
        num_layers (int): Number of attention layers.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        max_batch_size (int): The maximum batch size for the inference input.
        max_seq_len (int): The upper bound of sequence length for each sequence in the inference batch.
        layernorm_epsilon (float): Epsilon value for normalization.
        bf16 (bool): Whether to inference in bfloat16.
        fp16 (bool): Whether to inference in float16.

        learnable_input_layernorm (bool): Whether to have input layernorm weights.
        residual (bool): Whether to add residual connection.
        is_causal (bool):Whether the attention is causal.
        target_group_size (int):  The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking).
        position_encoding_config (PositionEncodingConfig, optional): Position embedding config.
        hstu_preprocessing_config (HSTUPreprocessingConfig, optional): HSTU preprocessing config.
        contextual_max_seqlen (int): The (maximum) length of contextual features.
        embedding_backend (EmbeddingBackend, optional): Embedding backend to use.
    """

    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    layernorm_epsilon: float = 1e-5
    bf16: bool = True
    fp16: bool = False

    learnable_input_layernorm: bool = True
    residual: bool = True
    is_causal: bool = True
    target_group_size: int = 1
    position_encoding_config: Optional[PositionEncodingConfig] = None
    hstu_preprocessing_config: Optional[HSTUPreprocessingConfig] = None
    contextual_max_seqlen: int = 0
    scaling_seqlen: int = -1
    embedding_backend: Optional[EmbeddingBackend] = None
    export_mode: bool = False

    def __post_init__(self):
        assert self.is_causal
        assert self.target_group_size == 1


def get_inference_hstu_config(
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    head_dim: int,
    max_batch_size: int,
    max_seq_len: int,
    norm_epsilon=1e-5,
    dtype: torch.dtype = torch.bfloat16,
    learnable_input_layernorm: bool = True,
    residual: bool = True,
    is_causal: bool = True,
    target_group_size: int = 1,
    position_encoding_config: Optional[PositionEncodingConfig] = None,
    contextual_max_seqlen: int = 0,
    scaling_seqlen: int = -1,
    embedding_backend=None,
    export_mode: Optional[bool] = None,
) -> InferenceHSTUConfig:
    """
    Create the HSTU configuration.

    Args:
        hidden_size (int): The hidden dimension size.
        num_layers (int): Number of attention layers.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Number of key-value channels (per attention head).
        max_batch_size (int): The maximum batch size for the inference input.
        max_seq_len (int): The upper bound of sequence length for each sequence in the inference batch.
        norm_epsilon (float, optional): Epsilon value for normalization. Defaults to 1e-5.
        dtype (torch.dtype): Data type (e.g., torch.float16).
        learnable_input_layernorm (bool, optional): Whether to have input layernorm weights. Defaults to True.
        residual (bool, optional): Whether to add residual connection. Defaults to True.
        is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        target_group_size (int, optional): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
        position_encoding_config (Optional[PositionEncodingConfig], optional): Position embedding config. Defaults to None.
        contextual_max_seqlen (int, optional): The (maximum) length of contextual features.
    Returns:
        HSTUConfig: The HSTU configuration object.
    """
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    if export_mode is None:
        export_mode = os.getenv("HSTU_INFERENCE_EXPORT_MODE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return InferenceHSTUConfig(  # type: ignore
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_attention_heads,
        head_dim=head_dim,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        layernorm_epsilon=norm_epsilon,
        bf16=is_bf16,
        fp16=is_fp16,
        learnable_input_layernorm=learnable_input_layernorm,
        residual=residual,
        is_causal=is_causal,
        target_group_size=target_group_size,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=contextual_max_seqlen,
        scaling_seqlen=scaling_seqlen,
        embedding_backend=embedding_backend,
        export_mode=export_mode,
    )
