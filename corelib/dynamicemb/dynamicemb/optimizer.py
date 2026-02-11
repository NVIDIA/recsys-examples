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
from typing import Any, Dict

import torch  # usort:skip
from dynamicemb_extensions import (
    OptimizerType,
    adagrad_update_for_combined_table,
    adam_update_for_combined_table,
    rowwise_adagrad_for_combined_table,
    sgd_update_for_combined_table,
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


def convert_optimizer_type(optimizer: EmbOptimType) -> OptimizerType:
    if optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        return OptimizerType.RowWiseAdaGrad
    elif optimizer == EmbOptimType.SGD or optimizer == EmbOptimType.EXACT_SGD:
        return OptimizerType.SGD
    elif optimizer == EmbOptimType.ADAM:
        return OptimizerType.Adam
    elif optimizer == EmbOptimType.EXACT_ADAGRAD:
        return OptimizerType.AdaGrad
    else:
        raise ValueError(
            f"Not supported optimizer type ,optimizer type = {optimizer} {type(optimizer)} {optimizer.value}."
        )


class BaseDynamicEmbeddingOptimizer(abc.ABC):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        self._opt_args: OptimizerArgs = copy.deepcopy(opt_args)

    @abc.abstractmethod
    def fused_update_with_index(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        dev_table: torch.Tensor,
        uvm_table: torch.Tensor,
    ) -> None:
        ...

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Apply optimizer step on standalone (grads, values) tensors.

        values layout: [N, emb_dim + optstate_dim].  Treated as a
        contiguous dev_table with identity indices so the existing
        fused_update_with_index kernels can be reused.
        """
        n = grads.size(0)
        indices = torch.arange(n, device=grads.device, dtype=torch.int64)
        self.fused_update_with_index(grads, indices, values, None)

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

    def set_learning_rate(self, new_lr) -> None:
        self._opt_args.learning_rate = new_lr
        return

    def get_initial_optim_states(self) -> float:
        return self._opt_args.initial_accumulator_value

    def set_initial_optim_states(self, value: float) -> None:
        self._opt_args.initial_accumulator_value = value
        return

    def step(self) -> None:
        pass

    def need_gradient_clipping(self) -> bool:
        return self._opt_args.gradient_clipping

    def clip_gradient(self, grads) -> None:
        grads.clamp_(
            min=-1 * self._opt_args.max_gradient, max=self._opt_args.max_gradient
        )


class SGDDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

    def fused_update_with_index(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        dev_table: torch.Tensor,
        uvm_table: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        sgd_update_for_combined_table(
            grads,
            indices,
            dev_table,
            uvm_table,
            lr,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "sgd",
            "lr": self._opt_args.learning_rate,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        return

    def get_state_dim(self, emb_dim: int) -> int:
        """
        Get the state dim.
        """
        return 0


class AdamDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)
        self._iterations: int = 0

    def step(self):
        self._iterations += 1

    def fused_update_with_index(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        dev_table: torch.Tensor,
        uvm_table: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        adam_update_for_combined_table(
            grads,
            indices,
            dev_table,
            uvm_table,
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
            "opt_type": "adam",
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


class AdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

    def fused_update_with_index(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        dev_table: torch.Tensor,
        uvm_table: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        adagrad_update_for_combined_table(
            grads,
            indices,
            dev_table,
            uvm_table,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "exact_adagrad",
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


class RowWiseAdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
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

    def fused_update_with_index(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        dev_table: torch.Tensor,
        uvm_table: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        rowwise_adagrad_for_combined_table(
            grads,
            indices,
            dev_table,
            uvm_table,
            state_dim,
            lr,
            eps,
        )

    def get_opt_args(self):
        ret_args = {
            "opt_type": "exact_row_wise_adagrad",
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
