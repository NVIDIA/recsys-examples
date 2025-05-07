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
from typing import Callable, List, Optional, Union

import torch
from modules.jagged_module import JaggedData
from modules.utils import init_mlp_weights_optional_bias
from torch import nn

class MLP(nn.Module):
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
        bias: bool = True,
        activation: Union[
            str,
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        from torchrec.modules.mlp import MLP as torchrec_MLP

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_size, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=layer_sizes[-1]),
        ).apply(init_mlp_weights_optional_bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP module.

        Args:
            jd (JaggedData): The input jagged data.

        Returns:
            JaggedData: The output jagged data.
        """
        assert input.dim() == 2, "Tensor must be 2-dimensional"
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=True
        ):
            return self._mlp(input)
