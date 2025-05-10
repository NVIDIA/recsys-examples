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
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from dataclasses import dataclass
from functools import partial  # pylint: disable-unused-import
from typing import List, Tuple, Union, cast

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from commons.utils.logging import print_rank_0
from configs import RankingConfig
from megatron.core.optimizer import get_megatron_optimizer
from model import get_ranking_model
from utils import (
    DistributedDataParallelArgs,
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    TrainerArgs,
    create_embedding_config,
    create_hstu_config,
    create_optimizer_config,
    get_data_loader,
    get_dataset_and_embedding_args,
    maybe_load_ckpts,
    train,
)

# pyre-strict
import logging
import os
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin

import torch
import torchrec
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer

from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ShardedTensor
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter


TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[List[int]] = cast(List[List[int]], None)
    prediction_head_act_type: Union[str, List[str]] = "relu"
    prediction_head_bias: Union[bool, List[bool]] = True
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu"
            ], "prediction_head_act_type should be in ['relu']"


parser = argparse.ArgumentParser(
    description="Distributed GR Arguments", allow_abbrev=False
)
parser.add_argument("--gin-config-file", type=str)
args = parser.parse_args()
gin.parse_config_file(args.gin_config_file)
trainer_args = TrainerArgs()
dataset_args, embedding_args = get_dataset_and_embedding_args()
network_args = NetworkArgs()
optimizer_args = OptimizerArgs()
ddp_args = DistributedDataParallelArgs()
tp_args = TensorModelParallelArgs()


def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    ams_grad: bool,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    ams_grad: bool,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory

def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class(
            optimizer_name="SGD",
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            ams_grad=False,
            momentum=0.9,
            learning_rate=1e-3,
        )
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class(
            optimizer_name="SGD",
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            ams_grad=False,
            momentum=0.9,
            learning_rate=1e-3,
        )
    )

    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
    )
    print('model.named_parameters()', list(model.named_parameters()))
    # Create keyed optimizer
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        print('k', k, v, v.requires_grad)
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_opt_factory,
                ),
            )
        )
    
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer

def create_ranking_config() -> RankingConfig:
    ranking_args = RankingArgs()

    return RankingConfig(
        embedding_configs=[
            create_embedding_config(network_args.hidden_size, arg, optimizer_args)
            for arg in embedding_args
        ],
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        eval_metrics=ranking_args.eval_metrics,
    )


def main():
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    device = torch.device("cuda")

    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    hstu_config = create_hstu_config(network_args)
    task_config = create_ranking_config()
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)
    print(model)
    model, optimzier = make_optimizer_and_shard(model, device)

    train_dataloader, test_dataloader = get_data_loader(
        "ranking", dataset_args, trainer_args
    )
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    # maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    train(
        model,
        trainer_args,
        train_dataloader,
        test_dataloader,
        optimzier,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
