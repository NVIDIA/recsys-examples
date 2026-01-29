import torch
from torch.utils.data import DataLoader


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return loader
