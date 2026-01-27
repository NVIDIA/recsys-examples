from commons.datasets.gpt_sid_batch import GPTSIDBatch
from commons.distributed.batch_shuffler import BaseTaskBalancedBatchShuffler
from commons.distributed.batch_shuffler_factory import register_batch_shuffler
from commons.perf_model.task_estimator import SelfAttentionTask


@register_batch_shuffler("sid_gr")
class SIDGRBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
    """
    SID-GR specific batch shuffler for load balancing based on self-attention workload.

    This shuffler calculates workloads based on history sequence length and
    standard self-attention complexity.
    """

    def __init__(
        self,
        num_heads: int = 1,
        head_dim: int = 1,
    ):
        """
        Initialize SID-GR batch shuffler.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.task = SelfAttentionTask()

    def get_workloads(self, batch: GPTSIDBatch, *args, **kwargs):
        """
        Calculate workloads based on self-attention complexity.

        Args:
            batch: Input batch containing history features

        Returns:
            Tensor of workload values for each sample in the batch
        """
        return self.task.get_workloads(
            batch.features[batch.history_feature_name].lengths(),
            self.num_heads,
            self.head_dim,
        )
