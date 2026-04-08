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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch  # usort:skip
from dynamicemb.utils import DTYPE_NUM_BYTES, torch_to_dyn_emb
from dynamicemb_extensions import (
    adagrad_update_for_flat_table,
    adagrad_update_for_padded_buffer,
    adam_update_for_flat_table,
    adam_update_for_padded_buffer,
    rowwise_adagrad_for_flat_table,
    rowwise_adagrad_for_padded_buffer,
    sgd_update_for_flat_table,
    sgd_update_for_padded_buffer,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType


def get_optimizer_state_dim(
    optimizer_type: EmbOptimType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Optimizer state elements per row (same rules as FBGEMM fused table value layout).

    ``dtype`` is only required for ``EXACT_ROWWISE_ADAGRAD`` (fixed 16-byte rowwise state
    in embedding dtype units). Callers that know the embedding dtype may pass it for any
    optimizer; it is ignored except for rowwise Adagrad.
    """
    if optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        if dtype is None:
            raise ValueError(
                "dtype is required when optimizer_type is EmbOptimType.EXACT_ROWWISE_ADAGRAD."
            )
        return 16 // DTYPE_NUM_BYTES[dtype]
    if optimizer_type == EmbOptimType.ADAM:
        return dim * 2
    if optimizer_type == EmbOptimType.EXACT_ADAGRAD:
        return dim
    return 0


def get_optimizer_ckpt_state_dim(
    optimizer_type: EmbOptimType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """Optimizer state elements per row stored in checkpoint files.

    Rowwise Adagrad keeps a wider fused layout at runtime (see
    :func:`get_optimizer_state_dim`) but only one accumulator scalar per row is
    needed in checkpoints; load pads back to the runtime width.
    """
    if optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
        return 1
    return get_optimizer_state_dim(optimizer_type, dim, dtype)


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


class BaseDynamicEmbeddingOptimizer(abc.ABC):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        self._opt_args: OptimizerArgs = copy.deepcopy(opt_args)

    @abc.abstractmethod
    def fused_update_for_flat_table(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        max_emb_dim: int,
        all_dims_vec4: bool,
        table_dtype: torch.dtype,
    ) -> None:
        ...

    @abc.abstractmethod
    def update_for_padded_buffer(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
        emb_dim: int,
        value_dim: int,
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

    def get_ckpt_state_dim(self, emb_dim: int) -> int:
        """Optimizer state width in checkpoint files (may be smaller than runtime)."""
        return self.get_state_dim(emb_dim)

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

    def update_for_padded_buffer(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
        emb_dim: int,
        value_dim: int,
    ) -> None:
        sgd_update_for_padded_buffer(
            grads,
            values,
            emb_dim,
            value_dim,
            self._opt_args.learning_rate,
        )

    def fused_update_for_flat_table(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        max_emb_dim: int,
        all_dims_vec4: bool,
        table_dtype: torch.dtype,
    ) -> None:
        sgd_update_for_flat_table(
            grads,
            indices,
            table_ptrs,
            table_ids,
            table_value_dims,
            table_emb_dims,
            max_emb_dim,
            all_dims_vec4,
            self._opt_args.learning_rate,
            torch_to_dyn_emb(table_dtype).value,
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
        return get_optimizer_state_dim(EmbOptimType.SGD, emb_dim)


class AdamDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)
        self._iterations: int = 0

    def step(self):
        self._iterations += 1

    def update_for_padded_buffer(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
        emb_dim: int,
        value_dim: int,
    ) -> None:
        adam_update_for_padded_buffer(
            grads,
            values,
            emb_dim,
            value_dim,
            self._opt_args.learning_rate,
            self._opt_args.beta1,
            self._opt_args.beta2,
            self._opt_args.eps,
            self._opt_args.weight_decay,
            self._iterations,
        )

    def fused_update_for_flat_table(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        max_emb_dim: int,
        all_dims_vec4: bool,
        table_dtype: torch.dtype,
    ) -> None:
        adam_update_for_flat_table(
            grads,
            indices,
            table_ptrs,
            table_ids,
            table_value_dims,
            table_emb_dims,
            self._opt_args.learning_rate,
            self._opt_args.beta1,
            self._opt_args.beta2,
            self._opt_args.eps,
            self._opt_args.weight_decay,
            self._iterations,
            max_emb_dim,
            all_dims_vec4,
            torch_to_dyn_emb(table_dtype).value,
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
        return get_optimizer_state_dim(EmbOptimType.ADAM, emb_dim)


class AdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
    ) -> None:
        super().__init__(opt_args)

    def update_for_padded_buffer(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
        emb_dim: int,
        value_dim: int,
    ) -> None:
        adagrad_update_for_padded_buffer(
            grads,
            values,
            emb_dim,
            value_dim,
            self._opt_args.learning_rate,
            self._opt_args.eps,
        )

    def fused_update_for_flat_table(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        max_emb_dim: int,
        all_dims_vec4: bool,
        table_dtype: torch.dtype,
    ) -> None:
        adagrad_update_for_flat_table(
            grads,
            indices,
            table_ptrs,
            table_ids,
            table_value_dims,
            table_emb_dims,
            self._opt_args.learning_rate,
            self._opt_args.eps,
            max_emb_dim,
            all_dims_vec4,
            torch_to_dyn_emb(table_dtype).value,
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
        return get_optimizer_state_dim(EmbOptimType.EXACT_ADAGRAD, emb_dim)


class RowWiseAdaGradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        emb_dtype: torch.dtype,
    ) -> None:
        super().__init__(opt_args)
        self._emb_dtype = emb_dtype

    def update_for_padded_buffer(
        self,
        grads: torch.Tensor,
        values: torch.Tensor,
        emb_dim: int,
        value_dim: int,
    ) -> None:
        rowwise_adagrad_for_padded_buffer(
            grads,
            values,
            emb_dim,
            value_dim,
            self._opt_args.learning_rate,
            self._opt_args.eps,
        )

    def fused_update_for_flat_table(
        self,
        grads: torch.Tensor,
        indices: torch.Tensor,
        table_ptrs: torch.Tensor,
        table_ids: torch.Tensor,
        table_value_dims: torch.Tensor,
        table_emb_dims: torch.Tensor,
        max_emb_dim: int,
        all_dims_vec4: bool,
        table_dtype: torch.dtype,
    ) -> None:
        rowwise_adagrad_for_flat_table(
            grads,
            indices,
            table_ptrs,
            table_ids,
            table_value_dims,
            table_emb_dims,
            self._opt_args.learning_rate,
            self._opt_args.eps,
            max_emb_dim,
            all_dims_vec4,
            torch_to_dyn_emb(table_dtype).value,
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
        return get_optimizer_state_dim(
            EmbOptimType.EXACT_ROWWISE_ADAGRAD, emb_dim, self._emb_dtype
        )

    def get_ckpt_state_dim(self, emb_dim: int) -> int:
        return get_optimizer_ckpt_state_dim(
            EmbOptimType.EXACT_ROWWISE_ADAGRAD, emb_dim, self._emb_dtype
        )


def truncate_optimizer_states_for_checkpoint(
    optimizer: BaseDynamicEmbeddingOptimizer,
    emb_dim: int,
    opt_states_runtime: torch.Tensor,
) -> torch.Tensor:
    """Slice runtime optimizer states to the width written in checkpoint files."""
    ckpt_dim = optimizer.get_ckpt_state_dim(emb_dim)
    if ckpt_dim == 0:
        return opt_states_runtime
    n = opt_states_runtime.size(1)
    if n == ckpt_dim:
        return opt_states_runtime
    if n < ckpt_dim:
        raise ValueError(
            f"Runtime optimizer state width {n} is less than checkpoint width {ckpt_dim}."
        )
    return opt_states_runtime[:, :ckpt_dim].contiguous()


def pad_optimizer_states_from_checkpoint(
    optimizer: BaseDynamicEmbeddingOptimizer,
    emb_dim: int,
    opt_states_from_file: torch.Tensor,
    initial_accumulator_value: float,
    values_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Expand checkpoint optimizer states to the runtime fused value width."""
    runtime_dim = optimizer.get_state_dim(emb_dim)
    file_dim = opt_states_from_file.size(1)
    if runtime_dim == 0:
        return opt_states_from_file
    if file_dim == runtime_dim:
        return opt_states_from_file.to(dtype=values_dtype)
    if file_dim > runtime_dim:
        return opt_states_from_file[:, :runtime_dim].contiguous().to(dtype=values_dtype)
    out = torch.full(
        (opt_states_from_file.size(0), runtime_dim),
        initial_accumulator_value,
        dtype=values_dtype,
        device=device,
    )
    out[:, :file_dim] = opt_states_from_file.to(dtype=values_dtype)
    return out
