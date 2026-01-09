import pytest
import torch
from modules.eval_metrics import DistributedRetrievalHitRate, DistributedRetrievalRecall
from torchmetrics.retrieval import RetrievalHitRate, RetrievalRecall

ref_metric_dict = {
    "hr": RetrievalHitRate,
    "recall": RetrievalRecall,
}

sid_metric_dict = {
    "hr": DistributedRetrievalHitRate,
    "recall": DistributedRetrievalRecall,
}


@pytest.mark.parametrize("eval_metric", ["Recall@10", "HR@10", "HR@20"])
@pytest.mark.parametrize("batch_size", [512, 1024, 2, 1])
@pytest.mark.parametrize("num_candidates", [100, 200, 5000])
def test_sid_retrieval_evaluator(
    eval_metric,
    batch_size,
    num_candidates,
):
    device = torch.device("cuda")

    for i in range(10):
        log_probs = torch.randn(batch_size, num_candidates, device=device)
        target = torch.zeros(
            batch_size, num_candidates, dtype=torch.bool, device=device
        )
        # set one target to True
        col_indices = torch.randint(0, num_candidates, (batch_size,), device=device)
        row_indices = torch.arange(batch_size, device=device)
        target[row_indices, col_indices] = True
        indexes = (
            torch.arange(batch_size, device=log_probs.device)
            .unsqueeze(-1)
            .expand(-1, num_candidates)
        )

        metric_name, top_k = eval_metric.split("@")
        metric_name = metric_name.lower()

        always_hit_batch_id = torch.randint(0, batch_size, (1,), device=device)
        log_probs[always_hit_batch_id, col_indices[always_hit_batch_id]] = (
            log_probs[always_hit_batch_id].max() + 0.1
        )
        top_k = int(top_k)
        ref_metric = ref_metric_dict[metric_name](top_k=top_k)
        # without sync and cache
        sid_metric = sid_metric_dict[metric_name](
            top_k=top_k, sync_on_compute=False, compute_with_cache=False
        ).cuda()

        sid_metric(log_probs, target, indexes=indexes)
        ref_metric(log_probs, target, indexes=indexes)

    sid_results = sid_metric.compute()
    ref_results = ref_metric.compute()
    assert torch.equal(sid_results, ref_results)
