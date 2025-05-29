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


def torch_addmm_silu_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    silu: bool = False,
) -> torch.Tensor:
    """
    compute z = silu(y + x @ w); silu is optional
    """
    z = torch.addmm(y, x, w)
    if silu:
        silu_z = torch.nn.functional.silu(z)
    else:
        silu_z = None
    return z, silu_z
