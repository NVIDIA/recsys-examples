import importlib
from typing import Any

from .data_loader import get_data_loader

_LAZY_MODULES = {
    "hstu_batch": ".hstu_batch",
    "hstu_random_dataset": ".hstu_random_dataset",
    "hstu_sequence_dataset": ".hstu_sequence_dataset",
    "random_inference_dataset": ".random_inference_dataset",
    "sid_random_dataset": ".sid_random_dataset",
    "sid_sequence_dataset": ".sid_sequence_dataset",
}

__all__ = [
    "get_data_loader",
    "hstu_random_dataset",
    "random_inference_dataset",
    "hstu_sequence_dataset",
    "hstu_batch",
    "sid_random_dataset",
    "sid_sequence_dataset",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, package=__name__)
    globals()[name] = module
    return module
