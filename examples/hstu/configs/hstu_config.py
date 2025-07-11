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
from typing import Optional

import torch

try:
    from megatron.core import parallel_state
    from megatron.core.transformer import TransformerConfig
except:
    parallel_state = None

    class TransformerConfig:  # type: ignore
        pass


@unique
class HSTULayerType(Enum):
    """
    Enum class representing different HSTU layer types.

    Attributes:
      FUSED: Represents the fused type. The fused layer is scheduleable and pipelineable
      NATIVE: Represents the non-fused type.
    """

    FUSED = "FUSED"
    NATIVE = "NATIVE"


@unique
class KernelBackend(Enum):
    """
    Enum class representing different kernel backends.

    Attributes:
      TRITON: Represents the TRITON backend.
      PYTORCH: Represents the PYTORCH backend.
      CUTLASS: Represents the CUTLASS backend.
    """

    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    CUTLASS = "CUTLASS"


@dataclass
class PositionEncodingConfig:
    """
    Configuration data class for position encoding settings.

    Args:
      num_position_buckets: The number of buckets used for position encoding.
      num_time_buckets: The number of buckets used for time encoding.
      use_time_encoding: A boolean flag indicating whether time encoding should be used.

    """

    num_position_buckets: int
    num_time_buckets: int
    use_time_encoding: bool


@dataclass
class HSTUConfig(TransformerConfig):
    """
    HSTUConfig is a configuration data class for the HSTU model, inheriting from TransformerConfig.

    Args:
      position_encoding_config (PositionEncodingConfig): Position embedding config. Defaults to None.
      is_causal (bool): Indicates if the model is causal. Defaults to True.
      enable_relative_attention_bias (bool): Flag to enable relative attention bias. Defaults to False.
      kernel_backend (KernelBackend): Backend for kernel operations. Defaults to KernelBackend.CUTLASS.
      target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
      learnable_input_layernorm (bool): Flag to enable learnable input layernorm. Defaults to True.
      residual (bool): Flag to enable residual connection. Defaults to True.
      async_wgrad (bool): Flag to enable async wgrad. Defaults to False.
      async_wgrad_stream (torch.cuda.Stream): Stream for async wgrad. Defaults to None.
      async_wgrad_event (torch.cuda.Event): Event for async wgrad. Defaults to None.
      recompute_input_layernorm (bool): Flag to enable recompute input layernorm. Defaults to False.
      recompute_input_silu (bool): Flag to enable recompute input silu. Defaults to False.
    """

    position_encoding_config: Optional[PositionEncodingConfig] = None
    is_causal: bool = True
    enable_relative_attention_bias: bool = False

    kernel_backend: KernelBackend = KernelBackend.CUTLASS
    hstu_layer_type: HSTULayerType = HSTULayerType.FUSED

    target_group_size: int = 1
    learnable_input_layernorm: bool = True
    # whether to add residual connection
    residual: bool = True
    # whether to use async wgrad
    async_wgrad: bool = False
    async_wgrad_stream: Optional[torch.cuda.Stream] = None
    async_wgrad_event: Optional[torch.cuda.Event] = None
    # whether to recompute input layernorm
    recompute_input_layernorm: bool = False
    recompute_input_silu: bool = False
    # whether is the inference mode
    is_inference: bool = False

    def __post_init__(self):
        super().__post_init__()


def get_hstu_config(
    hidden_size,
    kv_channels,
    num_attention_heads,
    num_layers,
    dtype,
    position_encoding_config: Optional[PositionEncodingConfig] = None,
    hidden_dropout=0.2,
    norm_epsilon=1e-5,
    is_causal: bool = True,
    kernel_backend: KernelBackend = KernelBackend.CUTLASS,
    target_group_size: int = 1,
    hstu_layer_type: HSTULayerType = HSTULayerType.FUSED,
    learnable_input_layernorm: bool = True,
    residual: bool = True,
    async_wgrad: bool = False,
    recompute_input_layernorm: bool = False,
    recompute_input_silu: bool = False,
    is_inference: bool = False,
) -> HSTUConfig:
    """
    Create the HSTU configuration.

    Args:
        hidden_size (int): The hidden dimension size.
        kv_channels (int): Number of key-value channels (per attention head).
        num_attention_heads (int): Number of attention heads.
        num_layers (int): Number of attention layers.
        dtype (torch.dtype): Data type (e.g., torch.float16).
        position_encoding_config (Optional[PositionEncodingConfig], optional): Position embedding config. Defaults to None.
        hidden_dropout (float, optional): Dropout rate for hidden layers. Defaults to 0.2.
        norm_epsilon (float, optional): Epsilon value for normalization. Defaults to 1e-5.
        is_causal (bool, optional): Whether the attention is causal. Defaults to False.
        kernel_backend (KernelBackend, optional): Backend for kernel operations. Defaults to KernelBackend.CUTLASS.
        target_group_size (int, optional): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
        hstu_layer_type (HSTULayerType, optional): The type of HSTU layer. Defaults to HSTULayerType.FUSED.
        learnable_input_layernorm (bool, optional): Whether to use learnable input layernorm. Defaults to True.
        residual (bool, optional): Whether to add residual connection. Defaults to True.
        async_wgrad (bool, optional): Whether to use async wgrad. Defaults to False.
        recompute_input_layernorm (bool, optional): Whether to recompute input layernorm. Defaults to False.
        recompute_input_silu (bool, optional): Whether to recompute input silu. Defaults to False.
        is_inference (bool, optional): Whether is the inference mode
    Returns:
        HSTUConfig: The HSTU configuration object.
    """
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    if async_wgrad:
        async_wgrad_stream = torch.cuda.Stream()
        async_wgrad_event = torch.cuda.Event(enable_timing=False)
    else:
        async_wgrad_stream = None
        async_wgrad_event = None
    return HSTUConfig(  # type: ignore
        position_encoding_config=position_encoding_config,
        hidden_size=hidden_size,
        kv_channels=kv_channels,
        num_attention_heads=num_attention_heads,
        hidden_dropout=hidden_dropout,
        layernorm_epsilon=norm_epsilon,
        num_layers=num_layers,
        bf16=is_bf16,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size()
        if not is_inference
        else 1,
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size()
        if not is_inference
        else 1,
        context_parallel_size=parallel_state.get_pipeline_model_parallel_world_size()
        if not is_inference
        else 1,
        fp16=is_fp16,
        is_causal=is_causal,
        kernel_backend=kernel_backend,
        target_group_size=target_group_size,
        hstu_layer_type=hstu_layer_type,
        learnable_input_layernorm=learnable_input_layernorm,
        residual=residual,
        async_wgrad=async_wgrad,
        async_wgrad_stream=async_wgrad_stream,
        async_wgrad_event=async_wgrad_event,
        recompute_input_layernorm=recompute_input_layernorm,
        recompute_input_silu=recompute_input_silu,
    )
