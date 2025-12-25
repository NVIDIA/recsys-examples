from dataclasses import dataclass

import torch
from megatron.core.transformer import TransformerConfig


@dataclass
class GPTConfig(TransformerConfig):
    def __post_init__(self):
        super().__post_init__()


def get_gpt_config(
    hidden_size: int,
    kv_channels: int,
    num_attention_heads: int,
    num_layers: int,
    dtype: torch.dtype,
    normalization: str = "LayerNorm",  # "LayerNorm" or "rmsnorm"
    norm_epsilon: float = 1e-5,
    hidden_dropout=0.0,
    tensor_model_parallel_size: int = 1,
) -> GPTConfig:
    """
    normalization: { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                    type of normalization applied.
    """
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    return GPTConfig(  # type: ignore
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        hidden_dropout=hidden_dropout,
        attention_dropout=hidden_dropout,  # TODO?
        layernorm_epsilon=norm_epsilon,
        bf16=is_bf16,
        fp16=is_fp16,
        tensor_model_parallel_size=tensor_model_parallel_size,
    )
