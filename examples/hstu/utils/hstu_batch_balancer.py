from commons.datasets.hstu_batch import HSTUBatch
from commons.distributed.batch_shuffler import BaseTaskBalancedBatchShuffler
from commons.distributed.batch_shuffler_factory import register_batch_shuffler
from commons.perf_model.task_estimator import HSTUAttentionTask


@register_batch_shuffler("hstu")
class HASTUBalancedBatchShuffler(BaseTaskBalancedBatchShuffler):
    """
    HSTU-specific batch shuffler for load balancing based on attention workload.

    This shuffler calculates workloads based on sequence length and HSTU attention
    complexity, optionally accounting for action-item interleaving.
    """

    def __init__(
        self,
        num_heads: int = 1,
        head_dim: int = 1,
        action_interleaved: bool = True,
    ):
        """
        Initialize HSTU batch shuffler.

        Args:
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            action_interleaved: Whether actions are interleaved with items (doubles seqlen)
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.action_interleaved = action_interleaved
        self.task = HSTUAttentionTask()

    def get_workloads(self, batch: HSTUBatch, *args, **kwargs):
        """
        Calculate workloads based on HSTU attention complexity.

        Args:
            batch: Input batch containing item features

        Returns:
            Tensor of workload values for each sample in the batch
        """
        seqlen = batch.features[batch.item_feature_name].lengths()
        # for ranking, we have action interleaved with item, so we need to multiply the seqlen by 2
        if self.action_interleaved:
            seqlen = seqlen * 2
        return self.task.get_workloads(seqlen, self.num_heads, self.head_dim)
