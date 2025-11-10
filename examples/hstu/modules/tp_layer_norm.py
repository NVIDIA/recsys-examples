import megatron.core.parallel_state as parallel_state
import torch
from megatron.core.tensor_parallel.mappings import all_to_all_hp2sp, all_to_all_sp2hp
from ops.collective_ops import (
    gather_along_first_dim,
    gather_along_last_dim,
    split_along_last_dim,
)
from ops.pt_ops.pt_norm_mul_dropout import pytorch_norm_mul_dropout
from ops.triton_ops.triton_layer_norm import triton_layer_norm
from ops.triton_ops.triton_norm_mul_dropout import triton_norm_mul_dropout


def _divide_with_exception(x, y):
    if x % y == 0:
        return x // y
    else:
        raise ValueError(f"x {x} is not divisible by y {y}")


# TODO: to add customized TP autograd function where we can handle the tensor memory allocation and deallocation
class TPLayerNorm(torch.nn.Module):
    """
    This is a TP LayerNorm.

    In the forward stage: we need to allgather the activations across TP ranks to compute the mean and variance.
    In the backward stage: we need to allgather the gradients of the mean and variance to compute the gradients of the weights and bias.

    Note that we do not support the gradient of the weights and bias.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        *,
        trainable=True,
        gather_output=False,
        sequence_parallel=False,
    ):
        super().__init__()

        # TODO: use duplicated weight and bias to avoid allgather
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._tp_size = tp_size
        self._tp_pg = parallel_state.get_tensor_model_parallel_group()
        self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self._hidden_size = hidden_size
        self._sequence_parallel = sequence_parallel
        # no need to broadcast weight and bias because they are initialized the same on all TP ranks
        self.weight = (
            torch.nn.Parameter(torch.ones(self._hidden_size)) if trainable else None
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros(self._hidden_size)) if trainable else None
        )
        self.eps = eps
        self.gather_output = gather_output

    def forward(self, x):
        """
        x: [batch_size, hidden_size]
        """
        weight = self.weight
        bias = self.bias
        # allgather the activations
        compute_x = gather_along_last_dim(x, self._tp_pg)
        # we use triton layer norm such that compute_x can be of different dtype from weight/bias
        normed_x = triton_layer_norm(compute_x, weight=weight, bias=bias, eps=self.eps)
        if not self.gather_output:
            normed_x = split_along_last_dim(normed_x, self._tp_pg)

        return normed_x


class TPLayerNormMulDropout(torch.nn.Module):

    """
    Similar to RowParallelLinear, but the input is always parallel along the last dimension.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        dropout_ratio=0.0,
        *,
        input_is_parallel=True,
        sequence_parallel=False,
        trainable=True,
        gather_output=False,
        fusion=True,
    ):
        super().__init__()
        assert (
            input_is_parallel
        ), "input_is_parallel(hidden dim) must be True for TPLayerNormMulDropout currently"
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._tp_size = tp_size
        self._tp_pg = parallel_state.get_tensor_model_parallel_group()
        self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self._hidden_size = hidden_size
        self._dropout_ratio = dropout_ratio
        self._norm_mul_dropout_func = (
            pytorch_norm_mul_dropout if not fusion else triton_norm_mul_dropout
        )
        # no need to broadcast weight and bias because they are initialized the same on all TP ranks
        self.weight = (
            torch.nn.Parameter(torch.ones(self._hidden_size)) if trainable else None
        )
        self.bias = (
            torch.nn.Parameter(torch.zeros(self._hidden_size)) if trainable else None
        )
        self._sequence_parallel = sequence_parallel
        if self._sequence_parallel and trainable:
            self.weight.sequence_parallel = sequence_parallel
            self.bias.sequence_parallel = sequence_parallel
        self.eps = eps
        self.gather_output = gather_output

    def forward(self, x, u):
        """
        x: [T, hidden_size_per_partition]
        u: [T, hidden_size_per_partition]

        When sp is on, we actually need to an all2all communication among TP ranks.
        Otherwise, we need a allgather communication.

        """
        if u.dim() == 3:
            u = u.contiguous().view(u.size(0), -1)
        weight = self.weight
        bias = self.bias
        if self._sequence_parallel:
            # TODO: latest mcore supports specifying the group.
            compute_x = all_to_all_hp2sp(
                x
            )  # [T, hidden_size_per_partition] -> [ T / tp_size, hidden_size]
            compute_u = all_to_all_hp2sp(
                u
            )  # [T, hidden_size_per_partition] -> [ T / tp_size, hidden_size]
        else:
            compute_x = gather_along_last_dim(x, self._tp_pg)  # [ T, hidden_size]
            compute_u = gather_along_last_dim(u, self._tp_pg)  # [ T, hidden_size]
        # we use triton layer norm such that compute_x can be of different dtype from weight/bias
        # TODO: The activation is allgathered, we should ensure the dropout behavior is consistent across TP ranks.
        normed_x = self._norm_mul_dropout_func(
            compute_x,
            compute_u,
            weight,
            bias,
            self.eps,
            self._dropout_ratio,
            training=self.training,
        )  # [ T or T / tp_size, hidden_size]
        normed_x_this_rank = normed_x
        if not self.gather_output and not self._sequence_parallel:
            normed_x_this_rank = split_along_last_dim(
                normed_x, self._tp_pg
            )  # [ T, hidden_size_per_partition]
        if not self.gather_output and self._sequence_parallel:
            normed_x_this_rank = all_to_all_sp2hp(
                normed_x
            )  # [ T, hidden_size_per_partition]
        if self.gather_output and self._sequence_parallel:
            normed_x_this_rank = gather_along_first_dim(
                normed_x, self._tp_pg
            )  # [ T, hidden_size]
        return normed_x_this_rank
