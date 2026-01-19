from commons.distributed.batch_shuffler import BaseTaskBalancedBatchShuffler
from commons.perf_model.task_estimator import HSTUAttentionTask
from hstu.datasets.utils import Batch


class HASTUBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
    def __init__(
        self,
        num_heads: int = 1,
        head_dim: int = 1,
        action_interleaved: bool = True,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.action_interleaved = action_interleaved
        self.task = HSTUAttentionTask()

    def get_workloads(self, batch: Batch, *args, **kwargs):
        seqlen = batch.features[batch.item_feature_name].lengths()
        # for ranking, we have action interleaved with item, so we need to multiply the seqlen by 2
        if self.action_interleaved:
            seqlen = seqlen * 2
        return self.task.get_workloads(seqlen, self.num_heads, self.head_dim)
