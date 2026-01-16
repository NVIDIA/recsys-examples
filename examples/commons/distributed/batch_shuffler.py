import torch


class BalancedBatchShuffler:
    # should be dp group.
    def __init__(
        self,
        batch_size: int,
        num_partitions: int,
        pg_group: torch.distributed.Group = torch.distributed.group.WORLD,
    ):
        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.pg_group = pg_group

    def shuffle(self, data: list[int]):
        return data
