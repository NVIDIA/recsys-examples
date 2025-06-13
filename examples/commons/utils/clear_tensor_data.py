import functools
from typing import Optional, Tuple

import torch


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def _resize_storage(t: torch.Tensor) -> None:
    t.untyped_storage().resize_(0)


def clear_tensor_data(
    *tensors: Tuple[Optional[torch.Tensor]], clear_storage: bool = False
) -> None:
    """
    Trick to deallocate tensor memory when delete operation does not
    release the tensor due to PyTorch override.

    Must be used carefully.
    """
    for t in tensors:
        if t is not None:
            if hasattr(t, "clear"):
                t.clear()  # type: ignore
            else:
                t.data = _empty_tensor()  # type: ignore
            if clear_storage:
                _resize_storage(t)
            del t
