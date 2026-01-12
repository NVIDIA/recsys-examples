import torch
from typing import List, Optional

def get_device(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.device]:
    """
    Returns the device of the first non-None tensor in the list.
    """
    for t in tensors:
        if t is not None:
            return t.device
    return None

@torch.library.custom_op("dynamicemb::ir_emb_lookup", mutates_args={})
def ir_emb_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("dynamicemb::ir_emb_lookup")
def ir_emb_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    return [torch.empty(batch_size, dim, device=device) for dim in dims]
