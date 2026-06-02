"""Qwen3 GR model metadata."""

from gr_inference.gr_models.qwen3.config import Qwen3GRConfig
from gr_inference.gr_models.qwen3.layers import (
    Qwen3LayerOps,
    Qwen3RMSNorm,
    Qwen3SingleLayerPrefill,
    TorchQwen3LayerOps,
    apply_qwen3_rope,
    flashinfer_call_counts,
    reset_flashinfer_call_counts,
)
from gr_inference.gr_models.qwen3.model import Qwen3GRModel
from gr_inference.gr_models.qwen3.variants import (
    DEFAULT_QWEN3_MODEL_ID,
    DEFAULT_QWEN3_VARIANT,
    KNOWN_QWEN3_VARIANTS,
    Qwen3VariantSpec,
    get_qwen3_variant,
    identify_qwen3_variant,
    resolve_qwen3_model_dir,
)
from gr_inference.gr_models.qwen3.weights import (
    Qwen3HFAdapter,
    QwenLayerWeightNames,
    QwenModelWeightNames,
    materialize_qwen3_checkpoint,
)

__all__ = [
    "Qwen3GRConfig",
    "Qwen3GRModel",
    "Qwen3HFAdapter",
    "Qwen3LayerOps",
    "Qwen3RMSNorm",
    "Qwen3SingleLayerPrefill",
    "Qwen3VariantSpec",
    "QwenLayerWeightNames",
    "QwenModelWeightNames",
    "TorchQwen3LayerOps",
    "DEFAULT_QWEN3_VARIANT",
    "DEFAULT_QWEN3_MODEL_ID",
    "KNOWN_QWEN3_VARIANTS",
    "apply_qwen3_rope",
    "flashinfer_call_counts",
    "get_qwen3_variant",
    "identify_qwen3_variant",
    "materialize_qwen3_checkpoint",
    "reset_flashinfer_call_counts",
    "resolve_qwen3_model_dir",
]
