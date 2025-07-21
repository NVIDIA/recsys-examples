import megatron.core.parallel_state as parallel_state
import torch
from ops.collective_ops import gather_along_last_dim, split_along_last_dim
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
    This is a TP LayerNorm. Weights and bias can be shared or sharded across TP ranks according to the shard_weight boolean argument.

    In the forward stage: we need to allgather the activations across TP ranks to compute the mean and variance (allgather weights/bias if shard_weight is True).
    In the backward stage: we need to allgather the gradients of the mean and variance to compute the gradients of the weights and bias.

    Note that we do not support the gradient of the weights and bias.
    """

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        *,
        trainable=True,
        shard_weight=False,
        gather_output=False,
    ):
        super().__init__()

        # TODO: use duplicated weight and bias to avoid allgather
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._tp_size = tp_size
        self._tp_pg = parallel_state.get_tensor_model_parallel_group()
        self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self._hidden_size = hidden_size
        self._shard_weight = shard_weight
        if shard_weight:
            hidden_size_per_partition = _divide_with_exception(hidden_size, tp_size)
            self.weight = (
                torch.nn.Parameter(torch.ones(hidden_size_per_partition))
                if trainable
                else None
            )
            self.bias = (
                torch.nn.Parameter(torch.zeros(hidden_size_per_partition))
                if trainable
                else None
            )
        else:
            # no need to broadcast weight and bias because they are initialized the same on all TP ranks
            self.weight = (
                torch.nn.Parameter(torch.ones(self._hidden_size)) if trainable else None
            )
            self.bias = (
                torch.nn.Parameter(torch.zeros(self._hidden_size))
                if trainable
                else None
            )
        self.eps = eps
        self.gather_output = gather_output

    def forward(self, x):
        """
        x: [batch_size, hidden_size]
        """
        weight = self.weight
        bias = self.bias
        if self._shard_weight:
            weight = gather_along_last_dim(self.weight, self._tp_pg)
            bias = gather_along_last_dim(self.bias, self._tp_pg)

        # allgather the activations
        full_x = gather_along_last_dim(x, self._tp_pg)
        # we use triton layer norm such that full_x can be of different dtype from weight/bias
        normed_x = triton_layer_norm(full_x, weight=weight, bias=bias, eps=self.eps)
        if not self.gather_output:
            normed_x = split_along_last_dim(normed_x, self._tp_pg)

        return normed_x


class TPLayerNormMulDropout(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        dropout_ratio=0.0,
        *,
        trainable=True,
        shard_weight=False,
        gather_output=False,
        fusion=True,
    ):
        super().__init__()

        # TODO: use duplicated weight and bias to avoid allgather
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self._tp_size = tp_size
        self._tp_pg = parallel_state.get_tensor_model_parallel_group()
        self._tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self._hidden_size = hidden_size
        self._shard_weight = shard_weight
        self._dropout_ratio = dropout_ratio
        self._norm_mul_dropout_func = (
            pytorch_norm_mul_dropout if not fusion else triton_norm_mul_dropout
        )
        if shard_weight:
            hidden_size_per_partition = _divide_with_exception(hidden_size, tp_size)
            self.weight = (
                torch.nn.Parameter(torch.ones(hidden_size_per_partition))
                if trainable
                else None
            )
            self.bias = (
                torch.nn.Parameter(torch.zeros(hidden_size_per_partition))
                if trainable
                else None
            )
        else:
            # no need to broadcast weight and bias because they are initialized the same on all TP ranks
            self.weight = (
                torch.nn.Parameter(torch.ones(self._hidden_size)) if trainable else None
            )
            self.bias = (
                torch.nn.Parameter(torch.zeros(self._hidden_size))
                if trainable
                else None
            )
        self.eps = eps
        self.gather_output = gather_output

    def forward(self, x, u):
        """
        x: [batch_size, hidden_size]
        """
        weight = self.weight
        bias = self.bias
        if self._shard_weight:
            weight = gather_along_last_dim(self.weight, self._tp_pg)
            bias = gather_along_last_dim(self.bias, self._tp_pg)

        # allgather the activations
        full_x = gather_along_last_dim(x, self._tp_pg)
        full_u = gather_along_last_dim(u, self._tp_pg)
        # we use triton layer norm such that full_x can be of different dtype from weight/bias
        # TODO: The activation is allgathered, we should ensure the dropout behavior is consistent across TP ranks.
        normed_x = self._norm_mul_dropout_func(
            full_x,
            full_u,
            weight,
            bias,
            self.eps,
            self._dropout_ratio,
            training=self.training,
        )
        if not self.gather_output:
            normed_x = split_along_last_dim(normed_x, self._tp_pg)
        return normed_x
