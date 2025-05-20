from typing import Tuple

import torch

from .metric_modules import MultiClassificationTaskMetric


def get_multi_event_metric_module(
    num_tasks: int,
    metric_types: Tuple[str, ...],
    comm_pg: torch.distributed.ProcessGroup = None,
):
    eval_metrics_modules = MultiClassificationTaskMetric(
        number_of_tasks=num_tasks,
        metric_types=metric_types,
        process_group=comm_pg,
    ).cuda()
    return eval_metrics_modules
