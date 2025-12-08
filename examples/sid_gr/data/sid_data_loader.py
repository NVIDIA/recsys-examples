import torch
from configs.sid_gin_config_args import DatasetArgs, TrainerArgs
from torch.utils.data import DataLoader

from .dataset import get_dataset


def _get_data_loader_from_dataset(
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


def get_data_loader(
    dataset_args: DatasetArgs,
    trainer_args: TrainerArgs,
):
    train_dataset = get_dataset(dataset_args, trainer_args, is_train_dataset=True)
    eval_dataset = get_dataset(dataset_args, trainer_args, is_train_dataset=False)

    train_loader = _get_data_loader_from_dataset(train_dataset)
    eval_loader = _get_data_loader_from_dataset(eval_dataset)

    return train_loader, eval_loader
