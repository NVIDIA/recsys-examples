import functools
from typing import Optional, Tuple

import torch


@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor().cuda()


def clear_tensor_data(*tensors: Tuple[Optional[torch.Tensor]]) -> None:
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
            del t
