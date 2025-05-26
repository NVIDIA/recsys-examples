from configs import HSTUConfig, RankingConfig, RetrievalConfig
from model.ranking_gr import RankingGR
from model.retrieval_gr import RetrievalGR

from . import ranking_gr, retrieval_gr

__all__ = ["ranking_gr", "retrieval_gr"]


def get_ranking_model(
    hstu_config: HSTUConfig,
    task_config: RankingConfig,
) -> RankingGR:
    """
    Get a ranking model.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RankingConfig): The ranking task configuration.

    Returns:
        RankingGR: The ranking model.
    """
    assert isinstance(task_config, RankingConfig), "please provide a ranking config"
    return RankingGR(hstu_config=hstu_config, task_config=task_config)


def get_retrieval_model(
    hstu_config: HSTUConfig,
    task_config: RetrievalConfig,
) -> RetrievalGR:
    """
    Get a retrieval model.

    Args:
        hstu_config (HSTUConfig): The HSTU configuration.
        task_config (RetrievalConfig): The retrieval task configuration.

    Returns:
        RetrievalGR: The retrieval model.
    """
    assert isinstance(task_config, RetrievalConfig), "please provide a retrieval config"
    return RetrievalGR(hstu_config=hstu_config, task_config=task_config)
