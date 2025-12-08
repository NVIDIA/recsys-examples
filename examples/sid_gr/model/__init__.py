from typing import List

from commons.modules.embedding import ShardedEmbeddingConfig
from megatron.core.transformer import TransformerConfig

from .gpt_model import SIDGRModel
from .mcore_model_specs import get_gpt_decoder_block_spec

__all__ = ["get_sid_gr_model"]


def get_sid_gr_model(
    decoder_config: TransformerConfig,
    codebook_embedding_config: ShardedEmbeddingConfig,
    codebook_sizes: List[int],
    num_hierarchies: int,
    max_history_length: int,
) -> SIDGRModel:
    sid_gr_model = SIDGRModel(
        decoder_config=decoder_config,
        codebook_embedding_config=codebook_embedding_config,
        codebook_sizes=codebook_sizes,
        num_hierarchies=num_hierarchies,
        transformer_decoder_layer_spec=get_gpt_decoder_block_spec(
            decoder_config, use_transformer_engine=True
        ),
        max_history_length=max_history_length,
        should_add_sep_token=True,
    )

    return sid_gr_model
