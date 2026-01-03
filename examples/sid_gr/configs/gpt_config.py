from dataclasses import dataclass
from enum import IntFlag, auto

import torch
from megatron.core.transformer import TransformerConfig


class BOSMode(IntFlag):
    HISTORY = auto()
    CANDIDATE = auto()
    ALWAYS = HISTORY | CANDIDATE


@dataclass
class GPTConfig(TransformerConfig):
    bos_token_mode: BOSMode = BOSMode.CANDIDATE

    def __post_init__(self):
        super().__post_init__()


def get_gpt_config(
    hidden_size: int,
    kv_channels: int,
    num_attention_heads: int,
    num_layers: int,
    dtype: torch.dtype,
    normalization: str = "RMSNorm",  # "LayerNorm" or "RMSNorm"
    norm_epsilon: float = 1e-5,
    hidden_dropout=0.0,
    tensor_model_parallel_size: int = 1,
    loss_on_history: bool = False,
) -> GPTConfig:
    """
    normalization: { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                    type of normalization applied.
    """
    bos_token_mode = BOSMode.CANDIDATE
    if loss_on_history:
        bos_token_mode |= BOSMode.HISTORY
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
        normalization=normalization,
        bos_token_mode=bos_token_mode,
    )
