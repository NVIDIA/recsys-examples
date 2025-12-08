from typing import List

from commons.modules.embedding import ShardedEmbeddingConfig

from .sid_gin_config_args import DatasetArgs, EmbeddingArgs, NetworkArgs


def create_embedding_config(
    hidden_size: int, embedding_args: EmbeddingArgs
) -> ShardedEmbeddingConfig:
    return ShardedEmbeddingConfig(
        feature_names=embedding_args.feature_names,
        table_name=embedding_args.table_name,
        vocab_size=embedding_args.item_vocab_size_or_capacity,
        dim=hidden_size,
        sharding_type=embedding_args.sharding_type,
    )


def create_embedding_configs(
    dataset_args: DatasetArgs,
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> List[ShardedEmbeddingConfig]:
    raise NotImplementedError("create_embedding_configs is not implemented yet")
    # if (
    #     network_args.item_embedding_dim <= 0
    #     or network_args.contextual_embedding_dim <= 0
    # ):
    #     return [
    #         create_embedding_config(network_args.hidden_size, arg)
    #         for arg in embedding_args
    #     ]
    # if isinstance(dataset_args, DatasetArgs):
    #     from preprocessor import get_common_preprocessors

    #     common_preprocessors = get_common_preprocessors()
    #     dp = common_preprocessors[dataset_args.dataset_name]
    #     item_feature_name = dp._item_feature_name
    #     contextual_feature_names = dp._contextual_feature_names
    #     action_feature_name = dp._action_feature_name

    # embedding_configs = []
    # for arg in embedding_args:
    #     if (
    #         item_feature_name in arg.feature_names
    #         or action_feature_name in arg.feature_names
    #     ):
    #         emb_config = create_embedding_config(network_args.item_embedding_dim, arg)
    #     else:
    #         if len(set(arg.feature_names) & set(contextual_feature_names)) != len(
    #             arg.feature_names
    #         ):
    #             raise ValueError(
    #                 f"feature name {arg.feature_name} not match with contextual feature names {contextual_feature_names}"
    #             )
    #         emb_config = create_embedding_config(
    #             network_args.contextual_embedding_dim, arg
    #         )
    #     embedding_configs.append(emb_config)
    # return embedding_configs
