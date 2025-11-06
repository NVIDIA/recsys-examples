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
from typing import Any, Dict, List, Union

import torch  # usort:skip
from dynamicemb.dynamicemb_config import *
from dynamicemb_extensions import (  # dynamic_emb_sgd,; dynamic_emb_adam,; dynamic_emb_adagrad,; dynamic_emb_rowwise_adagrad,
    dynamic_emb_adagrad_fused,
    dynamic_emb_adagrad_with_pointer,
    dynamic_emb_adagrad_with_table,
    dynamic_emb_adam_fused,
    dynamic_emb_adam_with_pointer,
    dynamic_emb_adam_with_table,
    dynamic_emb_rowwise_adagrad_fused,
    dynamic_emb_rowwise_adagrad_with_pointer,
    dynamic_emb_rowwise_adagrad_with_table,
    dynamic_emb_sgd_fused,
    dynamic_emb_sgd_with_pointer,
    dynamic_emb_sgd_with_table,
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
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        self._opt_args: OptimizerArgs = copy.deepcopy(opt_args)
        self._table_options: List[DynamicEmbTableOptions] = copy.deepcopy(table_options)

        self._hashtables: List[DynamicEmbTable] = hashtables
        self._num_tables: int = len(self._hashtables)

        self._state_dict: Dict[str, List[DynamicEmbTable]] = {}
        self._table_state_map: Dict[DynamicEmbTable, int] = {}

        for i, ht in enumerate(self._hashtables):
            self._table_state_map[ht] = i

    def get_state_by_name(self, state_name: str) -> Union[List[DynamicEmbTable], None]:
        """
        Get the state from the state dictionary.
        """
        return self._state_dict.get(state_name, None)

    def get_state(self) -> Union[Dict[str, List[DynamicEmbTable]], None]:
        """
        Get the state from the state dictionary.
        """
        return self._state_dict

    def state_names(self) -> List[str]:
        """
        Get a list of all state names in the state dictionary.
        """
        return list(self._state_dict.keys())

    def table_state_map(self) -> Dict[DynamicEmbTable, int]:
        """
        Get a list of all state names in the state dictionary.
        """
        return self._table_state_map

    def set_learning_rate(self, new_lr) -> None:
        self._opt_args.learning_rate = new_lr
        return

    def _create_tables(self, states: List[DynamicEmbTable]) -> None:
        for i, table_option in enumerate(self._table_options):
            states.append(create_dynamicemb_table(table_option))

    @abc.abstractmethod
    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        ...

    @abc.abstractmethod
    def get_opt_args(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def set_opt_args(self, args: Dict[str, Any]) -> None:
        ...


class SGDDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._hashtables:
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in hashtables in class {self.__class__.__name__}."
                )

        lr = self._opt_args.learning_rate
        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            grad = grads[i]
            indice = indices[i]
            num_indice = indice.shape[0]
            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_sgd_with_table(
                ht,
                num_indice,
                indice,
                grad,
                lr,
                weight_dtype,
            )

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "opt_type": "exact_sgd",
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        return


class AdamDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        self._iterations: int = 0
        self._state_dict["m"] = []
        self._state_dict["v"] = []

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )
        self._iterations += 1
        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]

            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_adam_with_table(
                ht,
                num_indice,
                indice,
                grad,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                self._iterations,
                weight_dtype,
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


class AdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        self._state_dict["Gt"] = hashtables

        for table in hashtables:
            table.set_initial_optstate(self._opt_args.initial_accumulator_value)

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]

            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_adagrad_with_table(
                ht, num_indice, indice, grad, lr, eps, weight_dtype
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
        for table in self._state_dict["Gt"]:
            table.set_initial_optstate(initial_value)
        return


class RowWiseAdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        self._state_dict["Gt"] = hashtables

        for table in hashtables:
            table.set_initial_optstate(self._opt_args.initial_accumulator_value)

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps
        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]

            weight_dtype = torch_to_dyn_emb(table_option.embedding_dtype)

            dynamic_emb_rowwise_adagrad_with_table(
                ht, num_indice, indice, grad, lr, eps, weight_dtype
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
        for table in self._state_dict["Gt"]:
            table.set_initial_optstate(initial_value)
        return


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
    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        ...

    @abc.abstractmethod
    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
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

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        dynamic_emb_sgd_fused(
            grads,
            values,
            lr,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
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


class AdamDynamicEmbeddingOptimizerV2(BaseDynamicEmbeddingOptimizerV2):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)
        self._iterations: int = 0

    def step(self):
        self._iterations += 1

    def update(
        self,
        grads: torch.Tensor,
        embs: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> None:
        pass
        # assert states is not None

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

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        dynamic_emb_adam_fused(
            grads,
            values,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            self._iterations,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type,
    ) -> None:
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

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        dynamic_emb_adagrad_fused(
            grads,
            values,
            lr,
            eps,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
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

    def fused_update(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        self.get_state_dim(emb_dim)

        dynamic_emb_rowwise_adagrad_fused(
            grads,
            values,
            lr,
            eps,
        )

    def fused_update_with_pointer(
        self,
        grads: torch.Tensor,
        value_ptr: torch.Tensor,  # pointers to embeddng + optimizer states
        value_type,
    ) -> None:
        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        emb_dim = grads.size(1)
        state_dim = self.get_state_dim(emb_dim)

        dynamic_emb_rowwise_adagrad_with_pointer(
            grads,
            value_ptr,
            value_type,
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
