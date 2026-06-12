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
import torch.nn.functional as F


def torch_addmm_silu_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    silu: bool = False,
    keep_unfused_out: bool = True,
    out: torch.Tensor | None = None,
    silu_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    compute z = silu(y + x @ w); silu is optional
    """
    z = torch.addmm(y, x, w)
    if out is not None:
        out.copy_(z)
        z_out = out
    elif keep_unfused_out:
        z_out = z
    else:
        z_out = silu_out

    if silu:
        silu_z = F.silu(z)
        if silu_out is not None:
            silu_out.copy_(silu_z)
            silu_z = silu_out
    else:
        silu_z = None
    return z_out, silu_z
