import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class BaseBatch(Pipelineable):
    """
    All tensors must share a same batch size.
    """

    features: KeyedJaggedTensor  # KJT 格式
    batch_size: int  # local batch size
    feature_to_max_seqlen: Dict[str, int]

    contextual_feature_names: List[str] = field(default_factory=list)
    # when labels is a tensor, it means the labels can be reshaped to [actual_batch_size, ...] and select along the batch dimension.
    labels: Union[KeyedJaggedTensor, torch.Tensor] = None
    actual_batch_size: Optional[int] = None  # in case of padding.

    def __post_init__(self):
        """数据验证"""
        if len(set(self.features.keys())) != len(list(self.features.keys())):
            raise ValueError(f"duplicate features keys {list(self.features.keys())}")
        assert isinstance(self.contextual_feature_names, list)
        assert isinstance(self.batch_size, int) and self.batch_size > 0
        self.actual_batch_size = (
            self.batch_size
            if self.actual_batch_size is None
            else self.actual_batch_size
        )

    def _apply_to_tensors_or_kjt(
        self, tensor_fn: Callable, *args, inplace: bool = False, **kwargs
    ) -> "BaseBatch":
        """
        Apply the specified function to all Tensors and KeyedJaggedTensors in the Batch.

        Args:
            tensor_fn: The function to apply to Tensor/KJT.
            *args, **kwargs: Arguments to pass to tensor_fn.
            inplace: Whether to operate in-place (such as record_stream)
                    - True: Do not create a new object; modify in-place and return None.
                    - False: Create a new object and return it.

        Returns:
            If inplace=False, returns a new Batch object; otherwise returns None.
        """
        batch_fields = fields(self)

        if inplace:
            for f in batch_fields:
                field_value = getattr(self, f.name)

                if field_value is None:
                    continue
                if isinstance(field_value, (torch.Tensor, KeyedJaggedTensor)):
                    tensor_fn(field_value, *args, **kwargs)
            return self

        else:
            new_kwargs: Dict[str, Any] = {}
            for f in batch_fields:
                field_name = f.name
                field_value = getattr(self, field_name)
                if field_value is None:
                    new_kwargs[field_name] = None
                    continue
                if isinstance(field_value, (torch.Tensor, KeyedJaggedTensor)):
                    new_kwargs[field_name] = tensor_fn(field_value, *args, **kwargs)
                else:
                    new_kwargs[field_name] = self._copy_field(field_value)
            return self.__class__(**new_kwargs)

    @staticmethod
    def _copy_field(value: Any) -> Any:
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        if isinstance(value, (list, dict, tuple, set)):
            return copy.deepcopy(value)
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def to(self, device: torch.device, non_blocking: bool = False) -> "BaseBatch":
        return self._apply_to_tensors_or_kjt(
            lambda t: t.to(device=device, non_blocking=non_blocking), inplace=False
        )

    def record_stream(self, stream: torch.cuda.Stream):
        self._apply_to_tensors_or_kjt(lambda t: t.record_stream(stream), inplace=True)

    def pin_memory(self) -> "BaseBatch":
        return self._apply_to_tensors_or_kjt(lambda t: t.pin_memory(), inplace=False)

    # select along the batch dimension
    # keyed_jagged_index_select_dim1(values, lengths, offsets, indices, batch_size, weights=None, selected_lengths_sum=None)
    # refer to https://github.com/pytorch/FBGEMM/blob/ca965328/fbgemm_gpu/fbgemm_gpu/docs/sparse_ops.py#L252-L260
    def index_select(self, indices: torch.Tensor) -> "BaseBatch":
        def index_select_dense_tensor(
            tensor: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            return (
                tensor.reshape(self.actual_batch_size, -1)
                .index_select(dim=0, index=indices)
                .reshape(-1)
            )

        def index_select_kjt(
            features: KeyedJaggedTensor, indices: torch.Tensor
        ) -> KeyedJaggedTensor:
            batch_size = len(features.lengths()) // len(features.keys())
            output = torch.ops.fbgemm.keyed_jagged_index_select_dim1(
                features.values(),
                features.lengths(),
                features.offsets(),
                indices,
                batch_size,
                features.weights_or_none(),
            )
            values = output[0]
            lengths = output[1]
            weights = output[2] if len(output) > 2 else None
            return KeyedJaggedTensor.from_lengths_sync(
                keys=features.keys(), values=values, lengths=lengths, weights=weights
            )

        def applier(t: Union[torch.Tensor, KeyedJaggedTensor]) -> Any:
            if isinstance(t, torch.Tensor):
                return index_select_dense_tensor(t, indices)
            elif isinstance(t, KeyedJaggedTensor):
                return index_select_kjt(t, indices)
            else:
                raise ValueError(f"Unsupported type: {type(t)}")

        return self._apply_to_tensors_or_kjt(
            applier,
            inplace=False,
        )


if __name__ == "__main__":
    batch_size = 10
    max_sequence_length = 10
    feature_names = ["feature1", "feature2"]
    feature_lengths = torch.randint(1, max_sequence_length, (batch_size * 2,)).cuda()
    feature_values = torch.randint(0, 100000, (feature_lengths.sum().item(),)).cuda()
    label_lengths = torch.randint(1, 20, (batch_size,)).cuda()
    label_values = torch.arange(label_lengths.sum().item(), device=torch.device("cuda"))
    labels = KeyedJaggedTensor.from_lengths_sync(
        keys=["label"],
        values=label_values,
        lengths=label_lengths,
    )
    features = KeyedJaggedTensor.from_lengths_sync(
        keys=feature_names,
        values=feature_values,
        lengths=feature_lengths.view(-1),
    )
    batch = BaseBatch(
        features=features,
        batch_size=batch_size,
        feature_to_max_seqlen={
            "feature1": max_sequence_length,
            "feature2": max_sequence_length,
        },
        labels=labels,
    )
    indices = torch.tensor([0, 2, 9]).cuda()
    selected_batch = batch.index_select(indices)
