from typing import List

import commons.utils as init
import pytest
import torch
from commons.modules.embedding import ShardedEmbeddingConfig
from commons.ops.length_to_offsets import length_to_complete_offsets
from data.gpt_sid_batch import FeatureConfig, GPTSIDBatch
from tests.test_utils import create_sid_gr_model_and_optimizer


def generate_batches(
    batchsize: int,
    num_batches: int,
    max_sequence_length: int,
    codebook_sizes: List[int],
    combined_history_feature_name: str,
    combined_candidate_feature_name: str,
    contextual_feature_names: List[str],
):
    codebook_sizes = torch.tensor(codebook_sizes)
    num_hierarchies = len(codebook_sizes)
    cum_sum_codebook_size = length_to_complete_offsets(codebook_sizes)
    max_item_ids = cum_sum_codebook_size[1:]
    min_item_ids = cum_sum_codebook_size[:-1]
    raw_hist_sid_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_sid_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    raw_feature_configs = [
        FeatureConfig(
            feature_names=raw_hist_sid_names,
            max_item_ids=max_item_ids,
            min_item_ids=min_item_ids,
            max_sequence_length=max_sequence_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_sid_names,
            max_item_ids=max_item_ids,
            min_item_ids=min_item_ids,
            max_sequence_length=1,  # candidate sid is a single sid
            is_jagged=False,
        ),
    ]
    return [
        GPTSIDBatch.random(
            batch_size=batchsize,
            feature_configs=raw_feature_configs,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            combined_history_feature_name=combined_history_feature_name,
            combined_candidate_feature_name=combined_candidate_feature_name,
            contextual_feature_names=contextual_feature_names,
            device=torch.cuda.current_device(),
        )
        for _ in range(num_batches)
    ]


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [128])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("max_sequence_length", [128])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128, 128], [256, 256, 256]])
def test_model_smoke(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_sequence_length,
    codebook_sizes,
):
    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)  # tp1
    init.set_random_seed(1234)
    history_sid_feature_name = "hist_sids"  # all sids are combined into this feature.
    candidate_sid_feature_name = "cand_sids"  # all sids are combined into this feature.
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[history_sid_feature_name, candidate_sid_feature_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )
    batchsize = 128
    num_batches = 10
    batches = generate_batches(
        batchsize=batchsize,
        num_batches=num_batches,
        max_sequence_length=max_sequence_length,
        codebook_sizes=codebook_sizes,
        combined_history_feature_name=history_sid_feature_name,
        combined_candidate_feature_name=candidate_sid_feature_name,
        contextual_feature_names=[],
    )
    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            max_sequence_length=max_sequence_length,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
        )
        optimizer.reload_model_params()

        for batch in batches:
            batch.to(torch.cuda.current_device())
            output = model(batch)
            # each sequence corresponds to one loss.
            loss, logits = output
            assert (
                loss.shape[0]
                == batch.features[batch.candidate_feature_name].offsets()[-1]
            )
            assert output is not None
            loss.sum().backward()
