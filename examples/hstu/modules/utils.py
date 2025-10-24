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

import torch
import warnings

try:
    import transformer_engine.pytorch as te
    use_te = True
except:
    warnings.warn("transformer_engine.pytorch is not installed, FP8 mixed precision will not be supported")
    use_te = False

def init_mlp_weights_optional_bias(
    m: torch.nn.Module,
) -> None:
    """
    Initialize the weights of a linear layer and optionally the bias.

    Args:
        m: The module to initialize.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # Always initialize bias to zero.
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def convert_te_linear_to_torch_linear(m: torch.nn.Module) -> torch.nn.Module:
    """
    Convert a Transformer Engine Linear layer to a PyTorch Linear layer.
    Copies weights and biases from the TE layer to the new PyTorch layer.

    Args:
        m: The module to convert. If not a TE Linear layer, returns unchanged.
    
    Returns:
        torch.nn.Linear if m is a TE Linear layer, otherwise returns m unchanged.
    """
    if not use_te:
        return m
    
    # Check if this is a Transformer Engine Linear layer
    if not isinstance(m, te.Linear):
        return m
    
    # Create new PyTorch Linear layer with same dimensions
    new_layer = torch.nn.Linear(
        m.in_features, 
        m.out_features, 
        bias=m.bias is not None, 
        device=m.weight.device, 
        dtype=m.weight.dtype
    )
    
    # Copy weights and bias
    with torch.no_grad():
        new_layer.weight.copy_(m.weight)
        if m.bias is not None and new_layer.bias is not None:
            new_layer.bias.copy_(m.bias)
    
    return new_layer