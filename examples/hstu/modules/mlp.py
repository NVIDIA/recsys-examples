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
from functools import wraps
from typing import List, Optional

import torch

try:
    from commons.utils.nvtx_op import output_nvtx_hook
    from megatron.core.transformer.module import MegatronModule

    BaseModule = MegatronModule
except:

    def output_nvtx_hook(nvtx_tag):
        def decorator(module):
            @wraps(module)
            def forward(*args, **kwags):
                return module(*args, **kwags)

            return forward

        return decorator

    BaseModule = torch.nn.Module
from modules.utils import init_mlp_weights_optional_bias


class MLP(BaseModule):  # type: ignore
    """
    Multi-Layer Perceptron (MLP) module wrapper for processing jagged data.

    Args:
        in_size (int): The input size.
        layer_sizes (List[int]): The sizes of the layers.
        bias (bool, optional): Whether to include bias in the layers. Defaults to True.
        activation (Union[str, Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]], optional): The activation function. Defaults to torch.relu.
        device (Optional[torch.device], optional): The device to use. Defaults to None.
        dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        activation: str = "relu",
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if BaseModule is torch.nn.Module:
            super().__init__()
        else:
            super(BaseModule, self).__init__()

        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "gelu":
            activation_fn = torch.nn.GELU
        else:
            raise ValueError(f"Activation function {activation} not supported")

        layers = []
        for i in range(len(layer_sizes)):
            layers.extend(
                [
                    torch.nn.Linear(
                        layer_sizes[i - 1] if i > 0 else in_size,
                        layer_sizes[i],
                        bias=bias,
                        device=device,
                        dtype=dtype,
                    ),
                    activation_fn()
                    if i < len(layer_sizes) - 1
                    else torch.nn.Identity(),
                ]
            )

        self._mlp = torch.nn.Sequential(*layers)
        self._mlp.apply(init_mlp_weights_optional_bias)

    @output_nvtx_hook(nvtx_tag="mlp")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        assert input.dim() == 2, "Tensor must be 2-dimensional"
        return self._mlp(input)
