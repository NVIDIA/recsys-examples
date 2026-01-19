from commons.distributed.batch_shuffler import BaseTaskBalancedBatchShuffler
from commons.perf_model.task_estimator import SelfAttentionTask
from sid_gr.datasets.gpt_sid_batch import GPTSIDBatch


class SIDGRBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
    def __init__(
        self,
        num_heads: int = 1,
        head_dim: int = 1,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.task = SelfAttentionTask()

    def get_workloads(self, batch: GPTSIDBatch, *args, **kwargs):
        return self.task.get_workloads(
            batch.features[batch.history_feature_name].lengths(),
            self.num_heads,
            self.head_dim,
        )
