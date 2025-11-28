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
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from commons.utils.logger import print_rank_0
from dynamicemb.dump_load import DynamicEmbDump as dynamic_emb_save
from dynamicemb.dump_load import DynamicEmbLoad as dynamic_emb_load
from megatron.core.distributed import DistributedDataParallel
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import Float16Module
from torch import nn
from torchrec.distributed.model_parallel import DistributedModelParallel


def get_unwrapped_module(module: nn.Module) -> nn.Module:
    """
    Unwraps module wrapped by DMP, DDP, or Float16Module.
    """
    while (
        isinstance(module, DistributedModelParallel)
        or isinstance(module, Float16Module)
        or isinstance(module, DistributedDataParallel)
    ):
        if isinstance(module, DistributedModelParallel):
            module = module._dmp_wrapped_module
        else:
            module = module.module
    return module


def save(
    path: str,
    module: nn.Module,
    dense_optimizer: Optional[MegatronOptimizer] = None,
    include_optim_state=True,
):
    """
    Save the module and optimizer state to the given path.

    Args:
        path (str): The path to save the state.
        module (nn.Module): The module to save.
        dense_optimizer (Optional[MegatronOptimizer], optional): The optimizer to save. Defaults to None.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.

    Raises:
        FileExistsError: If the path does not exist or the save file already exists.
    """
    unwrapped_module = get_unwrapped_module(module)

    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist.")
    save_dir = os.path.join(path, "dynamicemb_module")
    os.makedirs(save_dir, exist_ok=True)
    print_rank_0(f"dynamic module save dir {save_dir}")
    dynamic_emb_save(save_dir, unwrapped_module, optim=include_optim_state)

    save_dir = os.path.join(path, "torch_module")
    print_rank_0(f"torch module save dir {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": unwrapped_module.state_dict(),
            "optimizer_state_dict": dense_optimizer.state_dict()
            if dense_optimizer
            else None,
        },
        os.path.join(save_dir, "model.{}.pth".format(dist.get_rank())),
    )


def load(
    path: str,
    module: nn.Module,
    dense_optimizer: Optional[MegatronOptimizer] = None,
    include_optim_state=True,
):
    """
    Load the module and optimizer state from the given path.

    Args:
        path (str): The path to load the state from.
        module (nn.Module): The module to load the state into.
        dense_optimizer (Optional[MegatronOptimizer], optional): The optimizer to load the state into. Defaults to None.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.
    """
    dist.barrier(device_ids=[torch.cuda.current_device()])
    unwrapped_module = get_unwrapped_module(module)

    save_dir = os.path.join(path, "dynamicemb_module")
    dynamic_emb_load(save_dir, unwrapped_module, optim=include_optim_state)

    save_path = os.path.join(
        path, "torch_module", "model.{}.pth".format(dist.get_rank())
    )
    state_dict = torch.load(save_path, weights_only=False)
    unwrapped_module.load_state_dict(state_dict["model_state_dict"])
    if dense_optimizer and state_dict["optimizer_state_dict"]:
        dense_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
