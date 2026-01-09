import pytest
import torch
from beam_search.beam_search import BeamSearch


@pytest.mark.parametrize("batchsize", [10, 20, 50])
@pytest.mark.parametrize("beam_width", [10, 20, 50])
@pytest.mark.parametrize("codebook_sizes", [[100, 100, 100]])
def test_beam_search_sanity_check(batchsize, beam_width, codebook_sizes):
    num_hierarchies = len(codebook_sizes)
    beam_search = BeamSearch(
        beam_width, num_hierarchies, codebook_sizes, record_history=True
    )
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
    # check the childrens' prefix should be from parent
    for i in range(1, num_hierarchies):
        # shape [batchsize, cur_beam, i + 1]
        current_sids = beam_search.history_topk_sids[i]
        # shape [batchsize, par_beam, i]
        parent_sids = beam_search.history_topk_sids[i - 1]
        current_sids_depth = current_sids.shape[-1]
        parent_sids_depth = parent_sids.shape[-1]
        assert (
            parent_sids_depth + 1 == current_sids_depth
        ), "current_sids_depth should be parent_sids_depth + 1"
        current_slice = current_sids[:, :, :parent_sids_depth]  # [B, cur_beam, K]
        parent_slice = parent_sids  # [B, par_beam, K]

        # [batchsize, cur_beam, 1, K] == [batchsize, 1, par_beam, K]
        is_in = current_slice.unsqueeze(2) == parent_slice.unsqueeze(
            1
        )  # [B, cur_beam, par_beam, K]
        in_any_parent = is_in.any(dim=2)  # [B, cur_beam, K]
        assert torch.all(in_any_parent)


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
