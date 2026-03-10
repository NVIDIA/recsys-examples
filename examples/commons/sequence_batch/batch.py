import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable


@dataclass
class BaseBatch(Pipelineable):
    """
    All tensors must share a same batch size to better support index select and allgather.
    For incomplete batch, all dense tensors and KJT offsets/lengths are padded to the batch size.
    """

    features: KeyedJaggedTensor
    batch_size: int  # local batch size
    feature_to_max_seqlen: Dict[str, int]

    contextual_feature_names: List[str] = field(default_factory=list)
    # KJT should be jagged as well.
    labels: Optional[KeyedJaggedTensor] = None
    actual_batch_size: Optional[int] = None

    def __post_init__(self):
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
            If inplace=False, returns a new Batch object; otherwise returns self.
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
    def index_select(
        self, indices: torch.Tensor, oob_filter: bool = True
    ) -> "BaseBatch":
        # TODO@junzhang, we presume the dense tensor is padded to the batch size.
        def index_select_dense_tensor(
            tensor: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            return (
                tensor.reshape(self.batch_size, -1)
                .index_select(dim=0, index=indices)
                .reshape(-1)
            )

        # KJT index select guarantees the KJT batchsize is not changed(always padded). It's a must because we have KJT ALLGATHER.
        def index_select_kjt(
            features: KeyedJaggedTensor, indices: torch.Tensor
        ) -> KeyedJaggedTensor:
            batch_size = features.lengths().shape[0] // len(features.keys())
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

        if self.actual_batch_size != self.batch_size:
            # we use the actual batch size to filter the indices.
            actual_batch_size = indices[indices < self.actual_batch_size].numel()
        else:
            actual_batch_size = indices.numel()

        def applier(t: Union[torch.Tensor, KeyedJaggedTensor]) -> Any:
            if isinstance(t, torch.Tensor):
                return index_select_dense_tensor(t, indices)
            elif isinstance(t, KeyedJaggedTensor):
                return index_select_kjt(t, indices)
            else:
                raise ValueError(f"Unsupported type: {type(t)}")

        new_batch = self._apply_to_tensors_or_kjt(
            applier,
            inplace=False,
        )
        new_batch.actual_batch_size = actual_batch_size
        return new_batch
