from typing import List, Optional, Tuple

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
    normalization: Optional[str] = None,
    top_k_for_generation: int = 10,
    eval_metrics: Tuple[str, ...] = (),
    share_lm_head_across_hierarchies: bool = True,
) -> SIDGRModel:
    sid_gr_model = SIDGRModel(
        decoder_config=decoder_config,
        codebook_embedding_config=codebook_embedding_config,
        codebook_sizes=codebook_sizes,
        num_hierarchies=num_hierarchies,
        transformer_decoder_layer_spec=get_gpt_decoder_block_spec(
            # padding + arbitrary attention mask + Megatron-Core
            decoder_config,
            use_transformer_engine=False,
            arbitrary_attention_mask=True,
            normalization=normalization,
        ),
        should_add_sep_token=False,
        top_k_for_generation=top_k_for_generation,
        eval_metrics=eval_metrics,
        share_lm_head_across_hierarchies=share_lm_head_across_hierarchies,
    )

    return sid_gr_model
