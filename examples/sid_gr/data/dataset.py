from configs.sid_gin_config_args import DatasetArgs, DatasetType, TrainerArgs

from .gpt_sid_batch import FeatureConfig
from .in_memory_random_dataset import InMemoryRandomDataset
from .preprocessor import get_common_preprocessors


def get_dataset(
    dataset_args: DatasetArgs,
    trainer_args: TrainerArgs,
    is_train_dataset: bool,
):
    max_history_length = dataset_args.max_history_length
    num_hierarchies = dataset_args.num_hierarchies
    codebook_sizes = dataset_args.codebook_sizes
    assert (
        len(codebook_sizes) == num_hierarchies
    ), "codebook_sizes should have the same length as num_hierarchies"
    if dataset_args.dataset_type == DatasetType.InMemoryRandomDataset:
        # we need to use feature configs to generate random data
        feature_configs = []
        raw_hist_sid_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
        raw_cand_sid_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
        # history sid features
        feature_configs.append(
            FeatureConfig(
                feature_names=raw_hist_sid_names + raw_cand_sid_names,
                max_item_ids=[codebook_sizes[i] for i in range(num_hierarchies)]
                + [codebook_sizes[i] for i in range(num_hierarchies)],
                max_history_length=max_history_length,
                is_jagged=True,
            )
        )
        # candidate sid features
        feature_configs.append(
            FeatureConfig(
                feature_names=raw_cand_sid_names,
                max_item_ids=[codebook_sizes[i] for i in range(num_hierarchies)],
                max_history_length=1,
                is_jagged=True,
            )
        )
        # no contextual
        return InMemoryRandomDataset.get_dataset(
            batch_size=trainer_args.train_batch_size
            if is_train_dataset
            else trainer_args.eval_batch_size,
            feature_configs=feature_configs,
            raw_hist_sid_names=raw_hist_sid_names,
            raw_cand_sid_names=raw_cand_sid_names,
            combined_history_feature_name="hist_sids",
            combined_candidate_feature_name="cand_sids",
            contextual_feature_names=[],
            num_generated_batches=1,
            num_batches=trainer_args.max_train_iters
            if is_train_dataset
            else trainer_args.max_eval_iters,
        )
    elif dataset_args.dataset_type == DatasetType.DiskSequenceDataset:
        dataset_name = dataset_args.dataset_name
        common_preprocessor = get_common_preprocessors(
            dataset_args.sequence_features_data_path
        )[dataset_name]
        common_preprocessor.sequence_is_sid
        num_hierarchies = common_preprocessor.num_hierarchies

        common_preprocessor.raw_sequence_feature_name

        common_preprocessor.contextual_feature_names
        # a raw sequence is split into history and candidate sequences
        common_preprocessor.history_feature_name
        common_preprocessor.candidate_feature_name
        common_preprocessor.item_id_to_sid_mapping_path
        raise NotImplementedError("DiskSequenceDataset is not implemented yet")
    else:
        raise ValueError(f"Invalid dataset type: {dataset_args.dataset_type}")
