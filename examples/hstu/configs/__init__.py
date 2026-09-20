from . import hstu_config, inference_config
from .hstu_config import (
    HSTUConfig,
    HSTULayerType,
    HSTUPreprocessingConfig,
    KernelBackend,
    PositionEncodingConfig,
    get_hstu_config,
)
from .inference_config import (
    EmbeddingBackend,
    InferenceEmbeddingConfig,
    InferenceHSTUConfig,
    get_inference_hstu_config,
)


def __getattr__(name):
    """Load task schemas on demand without importing training dependencies."""
    # Inference configuration and tensor-only layer tests do not require the
    # training embedding stack. Load task/embedding schemas only when requested.
    if name in ("task_config", "RankingConfig", "RetrievalConfig"):
        from importlib import import_module

        module = import_module(".task_config", __name__)
        value = module if name == "task_config" else getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "hstu_config",
    "inference_config",
    "task_config",
    "ConfigType",
    "PositionEncodingConfig",
    "HSTUPreprocessingConfig",
    "HSTUConfig",
    "get_hstu_config",
    "RankingConfig",
    "RetrievalConfig",
    "KernelBackend",
    "HSTULayerType",
    "EmbeddingBackend",
    "InferenceEmbeddingConfig",
    "InferenceHSTUConfig",
    "get_inference_hstu_config",
]
