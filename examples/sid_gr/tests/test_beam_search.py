import pytest
import torch
from beam_search.beam_search import BeamSearch


@pytest.mark.parametrize("batchsize", [10, 20, 50])
@pytest.mark.parametrize("beam_width", [10, 20, 50])
@pytest.mark.parametrize("codebook_sizes", [[100, 100, 100]])
def test_beam_search_smoke(batchsize, beam_width, codebook_sizes):
    num_hierarchies = len(codebook_sizes)
    beam_search = BeamSearch(beam_width, num_hierarchies, codebook_sizes)
    topk_prev_step = 1
    for i in range(num_hierarchies):
        log_probs = torch.randn(
            batchsize,
            topk_prev_step,
            codebook_sizes[i],
            device=torch.cuda.current_device(),
        )

        beam_search.propagate(log_probs)

        topk_prev_step = beam_width
        import pdb

        pdb.set_trace()


@pytest.mark.parametrize("batchsize", [10, 20, 50])
@pytest.mark.parametrize("codebook_sizes", [[100, 100, 100]])
def test_beam_search_top1(batchsize, codebook_sizes):
    """
    top1 means no beam search, only the top1 candidate is selected.
    """
    beam_width = 1
    num_hierarchies = len(codebook_sizes)
    beam_search = BeamSearch(beam_width, num_hierarchies, codebook_sizes)
    accu_log_probs = torch.zeros(batchsize, device=torch.cuda.current_device())
    sids = torch.empty(
        batchsize, 0, device=torch.cuda.current_device(), dtype=torch.long
    )
    for i in range(num_hierarchies):
        log_probs = torch.randn(
            batchsize, 1, codebook_sizes[i], device=torch.cuda.current_device()
        )
        beam_search.propagate(log_probs)
        accu_log_probs = accu_log_probs.unsqueeze(-1) + log_probs.view(batchsize, -1)
        accu_log_probs, current_sids = torch.max(accu_log_probs, dim=-1)
        # select the max prob candidate for each batch
        sids = torch.cat([sids, current_sids.unsqueeze(-1)], dim=-1)
        torch.equal(beam_search.get_sids().view(-1), sids.view(-1))
