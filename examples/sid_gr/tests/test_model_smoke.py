from typing import List

import commons.utils as init
import pytest
import torch
from commons.checkpoint import get_unwrapped_module
from commons.modules.embedding import ShardedEmbeddingConfig
from commons.ops.length_to_offsets import length_to_complete_offsets
from datasets.gpt_sid_batch import FeatureConfig, GPTSIDBatch
from tests.test_utils import create_sid_gr_model_and_optimizer


def generate_batches(
    batchsize: int,
    num_batches: int,
    max_history_length: int,
    max_candidate_length: int,
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
            max_sequence_length=max_history_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_sid_names,
            max_item_ids=max_item_ids,
            min_item_ids=min_item_ids,
            max_sequence_length=max_candidate_length,
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
@pytest.mark.parametrize("max_history_length", [128])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128, 128], [256, 256, 256]])
def test_model_smoke(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_history_length,
    codebook_sizes,
):
    # we now only support max_candidate_length = 1 for now
    max_candidate_length = 1
    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)  # tp1
    init.set_random_seed(1234)
    history_sid_feature_name = "hist_sids"
    candidate_sid_feature_name = "cand_sids"
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
        max_history_length=max_history_length,
        max_candidate_length=max_candidate_length,
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


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [128])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("max_history_length", [128])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128, 128], [256, 256, 256]])
def test_model_decoder_step(
    dtype,
    hidden_size,
    num_attention_heads,
    kv_channels,
    num_layers,
    max_history_length,
    codebook_sizes,
):
    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(1234)
    history_sid_feature_name = "hist_sids"
    candidate_sid_feature_name = "cand_sids"
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[history_sid_feature_name, candidate_sid_feature_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )
    batch_size = 1
    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
        )
        optimizer.reload_model_params()
        model = get_unwrapped_module(model)
        # inference mode
        model.eval()
        for i in range(10):
            history_hiddens = torch.randn(
                batch_size,
                max_history_length,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
            history_offsets = (
                torch.arange(
                    0,
                    batch_size + 1,
                    device=torch.cuda.current_device(),
                    dtype=torch.long,
                )
                * max_history_length
            )
            history_max_seqlen = max_history_length

            # each history corresponds to one candidate
            candidates_hiddens = torch.randn(
                batch_size,
                num_hierarchies,
                hidden_size,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )
            candidate_offsets = (
                torch.arange(
                    0,
                    batch_size + 1,
                    device=torch.cuda.current_device(),
                    dtype=torch.long,
                )
                * num_hierarchies
            )
            candidate_max_seqlen = num_hierarchies

            input_hidden_states = torch.cat(
                [history_hiddens, candidates_hiddens], dim=1
            )
            input_offsets = history_offsets + candidate_offsets
            input_max_seqlen = history_max_seqlen + candidate_max_seqlen

            # decoding in one shot
            output = model.decoder_step(
                input_hidden_states.view(-1, hidden_size),
                input_offsets,
                input_max_seqlen,
            ).view(batch_size, input_max_seqlen, -1)
            candidates_logits = output.view(batch_size, input_max_seqlen, -1)[
                :, history_max_seqlen:, :
            ]
            ref_probs = []
            for h in range(num_hierarchies):
                mlp = model._decoder_mlp[h]
                tuple_or_tensor = mlp(candidates_logits[:, h, :])
                logits = (
                    tuple_or_tensor[0]
                    if isinstance(tuple_or_tensor, tuple)
                    else tuple_or_tensor
                )
                ref_probs.append(torch.nn.functional.softmax(logits.float(), dim=-1))

            # decoding one by one
            for h in range(1, num_hierarchies + 1):
                # h = num_hierarchies
                # h = num_hierarchies
                prefix_candidates_hiddens = candidates_hiddens[:, :h, :]
                prefix_candidate_offsets = (
                    torch.arange(
                        0,
                        batch_size + 1,
                        device=torch.cuda.current_device(),
                        dtype=torch.long,
                    )
                    * h
                )
                prefix_candidate_max_seqlen = h
                prefix_input_hidden_states = torch.cat(
                    [history_hiddens, prefix_candidates_hiddens], dim=1
                )
                prefix_input_offsets = history_offsets + prefix_candidate_offsets
                prefix_input_max_seqlen = (
                    history_max_seqlen + prefix_candidate_max_seqlen
                )

                prefix_output = model.decoder_step(
                    prefix_input_hidden_states.view(-1, hidden_size),
                    prefix_input_offsets,
                    prefix_input_max_seqlen,
                ).view(batch_size, prefix_input_max_seqlen, -1)
                prefix_candidates_logits = prefix_output.view(
                    batch_size, prefix_input_max_seqlen, -1
                )[:, history_max_seqlen:, :]
                for hh in range(0, h):
                    ref_prob = ref_probs[hh]
                    tuple_or_tensor = model._decoder_mlp[hh](
                        prefix_candidates_logits[:, hh, :]
                    )
                    prob = (
                        tuple_or_tensor[0]
                        if isinstance(tuple_or_tensor, tuple)
                        else tuple_or_tensor
                    )
                    prob = prob.float().softmax(dim=-1)
                    this_sorted_prob, this_sorted_indices = torch.sort(
                        prob, dim=-1, descending=True
                    )
                    sorted_ref_prob, sorted_ref_indices = torch.sort(
                        ref_prob, dim=-1, descending=True
                    )
                    # top 10?
                    the_same_order = (
                        sorted_ref_indices[0:10] == this_sorted_indices[0:10]
                    ).all()
                    # import pdb; pdb.set_trace()
                    assert the_same_order
                    # assert torch.allclose(ref_prob, prob, atol=1e-4)
