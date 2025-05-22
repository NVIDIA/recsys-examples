from typing import Tuple

import torch

from .metric_modules import MultiClassificationTaskMetric


def get_multi_event_metric_module(
    num_classes: int,
    num_tasks: int,
    metric_types: Tuple[str, ...],
    comm_pg: torch.distributed.ProcessGroup = None,
):
    return MultiClassificationTaskMetric(
        num_classes=num_classes,
        number_of_tasks=num_tasks,
        metric_types=metric_types,
        process_group=comm_pg,
    )
