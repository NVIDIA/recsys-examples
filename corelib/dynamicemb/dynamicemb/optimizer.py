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

import abc
import copy
import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch  # usort:skip
from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (
    # dynamic_emb_sgd,
    dynamic_emb_sgd_with_pointer,
    # dynamic_emb_adam,
    dynamic_emb_adam_with_pointer,
    # dynamic_emb_adagrad,
    dynamic_emb_adagrad_with_pointer,
    # dynamic_emb_rowwise_adagrad,
    dynamic_emb_rowwise_adagrad_with_pointer,
)


@dataclass
class OptimizerArgs:
    stochastic_rounding: bool = True
    gradient_clipping: bool = False
    max_gradient: float = 1.0
    max_norm: float = 0.0
    learning_rate: float = 0.01
    eps: float = 1.0e-8
    initial_accumulator_value: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    weight_decay_mode: int = 0
    eta: float = 0.001
    momentum: float = 0.9
    counter_halflife: int = -1
    adjustment_iter: int = -1
    adjustment_ub: float = 1.0
    learning_rate_mode: int = -1
    grad_sum_decay: int = -1
    tail_id_threshold: float = 0
    is_tail_id_thresh_ratio: int = 0
    total_hash_size: int = 0
    weight_norm_coefficient: float = 0
    lower_bound: float = 0
    regularization_mode: int = 0


@enum.unique
class EmbOptimType(enum.Enum):
    SGD = "sgd"  # uses non-deterministic updates (atomicAdd(..)) with duplicate ids
    EXACT_SGD = (
        "exact_sgd"  # uses deterministic updates (via sorting + segment reduction)
    )
    LAMB = "lamb"
    ADAM = "adam"
    # exact/dedup: gradients to the same row are applied with coalesce then apply
    # together, instead of applied in sequence (approx).
    EXACT_ADAGRAD = "exact_adagrad"
    EXACT_ROWWISE_ADAGRAD = "exact_row_wise_adagrad"
    LARS_SGD = "lars_sgd"
    PARTIAL_ROWWISE_ADAM = "partial_row_wise_adam"
    PARTIAL_ROWWISE_LAMB = "partial_row_wise_lamb"
    ROWWISE_ADAGRAD = "row_wise_adagrad"
    SHAMPOO = "shampoo"  # not currently supported for sparse embedding tables
    MADGRAD = "madgrad"
    EXACT_ROWWISE_WEIGHTED_ADAGRAD = "exact_row_wise_weighted_adagrad"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


def string_to_opt_type(optimizer_str: str) -> EmbOptimType:
    try:
        return EmbOptimType(optimizer_str)
    except ValueError:
        raise ValueError(f"'{optimizer_str}' is not a valid EmbOptimType.")


def get_required_arg(args: Dict[str, Any], key: str) -> Any:
    if key not in args:
        raise ValueError(
            f"Input args does not contain required optimizer argument: {key}"
        )
    return args[key]


class BaseDynamicEmbeddingOptimizerV2(abc.ABC):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        self._opt_args: OptimizerArgs = copy.deepcopy(opt_args)

    @abc.abstractmethod
    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        ...

    @abc.abstractmethod
    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor, # pointers to embeddng + optimizer states
    ) -> None:
        ...

    @abc.abstractmethod
    def get_opt_args(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def set_opt_args(self, args: Dict[str, Any]) -> None:
        ...

    @abc.abstractmethod
    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        pass

    def set_learning_rate(self, new_lr) -> None:
        self._opt_args.learning_rate = new_lr
        return

    def get_initial_optim_states(self) -> float:
        return self._opt_args.initial_accumulator_value

    def set_initial_optim_states(self, value: float) -> None:
        self._opt_args.initial_accumulator_value = value
        return


class SGDDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass
        # lr = self._opt_args.learning_rate
        # dynamic_emb_sgd(
        #     grads.size(0),
        #     grads,
        #     embs,
        #     lr,
        # )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor, # pointers to embeddng + optimizer states
        value_type,
    ) -> None:
        lr = self._opt_args.learning_rate
        dynamic_emb_sgd_with_pointer(
            grads,
            value_ptr,
            value_type,
            lr,
        )

    def get_opt_args(self):
        ret_args = {"lr": self._opt_args.learning_rate}
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return 0

class AdamDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)
        self._iterations: int = 0
    
    def _step(self):
        self._iterations += 1

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass
        # assert states is not None
        # self._step()

        # lr = self._opt_args.learning_rate
        # beta1 = self._opt_args.beta1
        # beta2 = self._opt_args.beta2
        # weight_decay = self._opt_args.weight_decay
        # eps = self._opt_args.eps

        # dynamic_emb_adam(
        #     grads.size(0),
        #     grads,
        #     embs,
        #     states,
        #     lr,
        #     beta1,
        #     beta2,
        #     eps,
        #     weight_decay,
        #     self._iterations,
        # )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor, # pointers to embeddng + optimizer states
        value_type,
    ) -> None:

        self._step()

        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_adam_with_pointer(
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            self._iterations,
        )

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "iters": self._iterations,
            "beta1": self._opt_args.beta1,
            "beta2": self._opt_args.beta2,
            "eps": self._opt_args.eps,
            "weight_decay": self._opt_args.weight_decay,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._iterations = get_required_arg(args, "iters")
        self._opt_args.beta1 = get_required_arg(args, "beta1")
        self._opt_args.beta2 = get_required_arg(args, "beta2")
        self._opt_args.eps = get_required_arg(args, "eps")
        self._opt_args.weight_decay = get_required_arg(args, "weight_decay")
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return emb_dim * 2


class AdaGradDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass
        # lr = self._opt_args.learning_rate
        # eps = self._opt_args.eps

        # dynamic_emb_adagrad(
        #     grads.size(0),
        #     grads,
        #     embs,
        #     states,
        #     lr,
        #     eps,
        # )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor, # pointers to embeddng + optimizer states
        value_type,
    ) -> None:

        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_adagrad_with_pointer(
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
            "initial_accumulator_value": self._opt_args.initial_accumulator_value,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")
        initial_value = get_required_arg(args, "initial_accumulator_value")
        self._opt_args.initial_accumulator_value = initial_value
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return emb_dim


class RowWiseAdaGradDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        emb_dtype: torch.dtype,
    ) -> None:
        super().__init__(opt_args)

        DTYPE_NUM_BYTES: Dict[torch.dtype, int] = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        self._optim_state_dim = 16 // DTYPE_NUM_BYTES[emb_dtype]

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass
        # lr = self._opt_args.learning_rate
        # eps = self._opt_args.eps

        # dynamic_emb_rowwise_adagrad(
        #     grads.size(0),
        #     grads,
        #     embs,
        #     states,
        #     lr,
        #     eps,
        # )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor, # pointers to embeddng + optimizer states
        value_type,
    ) -> None:

        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_rowwise_adagrad_with_pointer(
            grads.size(0),
            grads,
            value_ptr,
            value_type,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
            "initial_accumulator_value": self._opt_args.initial_accumulator_value,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")
        initial_value = get_required_arg(args, "initial_accumulator_value")
        self._opt_args.initial_accumulator_value = initial_value
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return self._optim_state_dim
