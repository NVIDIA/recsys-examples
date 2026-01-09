from typing import List

import torch
from commons.distributed.sharding import make_optimizer_and_shard
from commons.modules.embedding import ShardedEmbeddingConfig
from commons.optimizer import OptimizerParam
from configs.gpt_config import get_gpt_config
from model.gpt_model import SIDGRModel
from model.mcore_model_specs import get_gpt_decoder_block_spec


def create_sid_gr_model_and_optimizer(
    dtype: torch.dtype,
    hidden_size: int,
    num_attention_heads: int,
    kv_channels: int,
    num_layers: int,
    num_hierarchies: int,
    codebook_embedding_config: ShardedEmbeddingConfig,
    codebook_sizes: List[int],
    should_add_sep_token: bool = False,
    optimizer_type_str: str = "adam",
    pipeline_type: str = "none",
    device: torch.device = None,
):
    decoder_config = get_gpt_config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        kv_channels=kv_channels,
        num_layers=num_layers,
        dtype=dtype,
        normalization="LayerNorm",
        norm_epsilon=1e-5,
        hidden_dropout=0.0,
        tensor_model_parallel_size=1,
        loss_on_history=False,
    )
    # thd + causal + TE
    transformer_decoder_layer_spec = get_gpt_decoder_block_spec(
        decoder_config,
        use_transformer_engine=False,
        arbitrary_attention_mask=True,
    )

    sid_gr_model = SIDGRModel(
        decoder_config=decoder_config,
        codebook_embedding_config=codebook_embedding_config,
        codebook_sizes=codebook_sizes,
        num_hierarchies=num_hierarchies,
        transformer_decoder_layer_spec=transformer_decoder_layer_spec,
        should_add_sep_token=should_add_sep_token,
        top_k_for_generation=10,
        eval_metrics=("HitRate@2", "NDCG@10"),
        share_lm_head_across_hierarchies=False,
    )

    optimizer_param = OptimizerParam(
        optimizer_str=optimizer_type_str,
        learning_rate=1e-3 if optimizer_type_str == "adam" else 1e-1,
        adam_beta1=0.5,  # larger beta1 for better debugging!
        adam_beta2=0.999,
        adam_eps=1e-8,
        weight_decay=0.0,  # decay is off for better debugging
    )

    model_train, dense_optimizer = make_optimizer_and_shard(
        sid_gr_model,
        config=decoder_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        pipeline_type=pipeline_type,
        device=device,
    )

    return model_train, dense_optimizer
