from dataclasses import dataclass

import torch
from megatron.core.transformer import TransformerConfig


@dataclass
class GPTConfig(TransformerConfig):
    def __post_init__(self):
        super().__post_init__()


def get_gpt_config(
    hidden_size: int,
    kv_channels,
    num_attention_heads,
    num_layers,
    dtype,
    norm_epsilon: float = 1e-5,
    hidden_dropout=0.2,
) -> GPTConfig:
    is_bf16 = dtype == torch.bfloat16
    is_fp16 = dtype == torch.float16
    return GPTConfig(  # type: ignore
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        hidden_dropout=hidden_dropout,
        layernorm_epsilon=norm_epsilon,
        bf16=is_bf16,
        fp16=is_fp16,
    )
