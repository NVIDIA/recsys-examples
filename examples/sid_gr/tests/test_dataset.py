import pytest
import torch
from data.disk_sequence_dataset import DiskSequenceDataset
from data.gpt_sid_batch import FeatureConfig, GPTSIDBatch
from tqdm import tqdm


@pytest.mark.parametrize("batch_size", [128, 256, 512])
def test_batch(batch_size):
    feature_configs = [
        FeatureConfig(
            feature_names=[
                "hist_sid_0",
                "hist_sid_1",
                "hist_sid_2",
                "hist_sid_3",
                "timestamp",
            ],
            max_item_ids=[128, 128, 128, 128, 100000],
            min_item_ids=[0, 0, 0, 0, 0],
            max_history_length=128,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=["cand_sid_0", "cand_sid_1", "cand_sid_2", "cand_sid_3"],
            max_item_ids=[128, 128, 128, 128],
            min_item_ids=[0, 0, 0, 0],
            max_history_length=128,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=[
                "contextual_0",
                "contextual_1",
            ],
            max_item_ids=[
                4,
                100,
            ],
            min_item_ids=[
                0,
                0,
            ],
            max_history_length=4,
            is_jagged=False,
        ),
    ]
    raw_hist_sid_names = ["hist_sid_0", "hist_sid_1", "hist_sid_2", "hist_sid_3"]
    raw_cand_sid_names = ["cand_sid_0", "cand_sid_1", "cand_sid_2", "cand_sid_3"]
    contextual_feature_names = ["contextual_0", "contextual_1"]
    batch = GPTSIDBatch.random(
        batch_size=batch_size,
        feature_configs=feature_configs,
        raw_hist_sid_names=raw_hist_sid_names,
        raw_cand_sid_names=raw_cand_sid_names,
        contextual_feature_names=contextual_feature_names,
        combined_history_feature_name="hist_sids",
        combined_candidate_feature_name="cand_sids",
        device=torch.cuda.current_device(),
    )
    assert all(
        hist_sid_name not in batch.features.keys()
        for hist_sid_name in raw_hist_sid_names
    ), "history sid feature names should not be in the batch features"
    assert all(
        cand_sid_name not in batch.features.keys()
        for cand_sid_name in raw_cand_sid_names
    ), "candidate sid feature names should not be in the batch features"
    assert (
        "hist_sids" in batch.features.keys()
    ), "history sids feature name should be in the batch features"
    assert (
        "cand_sids" in batch.features.keys()
    ), "candidate sids feature name should be in the batch features"
    assert (
        batch.features["hist_sids"].lengths().numel() == batch_size
    ), "history sids feature length should be 128"
    assert (
        batch.features["cand_sids"].lengths().numel() == batch_size
    ), "candidate sids feature length should be 128"


@pytest.mark.parametrize("batch_size", [128, 256, 512])
@pytest.mark.parametrize("max_history_seqlen", [64, 128, 256])
def test_disk_sequence_dataset(
    batch_size,
    max_history_seqlen,
):
    num_hierarchies = 4
    disk_sequence_dataset = DiskSequenceDataset(
        raw_sequence_data_path="./tmp_data/amzn/beauty/training/22363.parquet",
        item_id_to_sid_mapping_tensor_path="./tmp_data/amzn/beauty/item-sid-mapping.pt",
        batch_size=batch_size,
        max_history_seqlen=max_history_seqlen,
        max_candidate_seqlen=1,
        raw_sequence_feature_name="sequence_data",
        num_hierarchies=num_hierarchies,
        codebook_sizes=[256, 256, 256, 256],
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=1234,
        is_train_dataset=True,
        deduplicate_sid_across_hierarchy=True,
    )
    num_batches = len(disk_sequence_dataset)
    for idx, batch in enumerate(
        tqdm(
            disk_sequence_dataset,
            total=num_batches,
            desc="Testing disk sequence dataset",
        )
    ):
        for key in batch.features.keys():
            assert (
                batch.features[key].lengths().numel() == batch_size
            ), f"length of {key} should be {batch_size}"
        if idx < len(disk_sequence_dataset) - 1:
            assert (
                batch.labels.view(-1, num_hierarchies).shape[0] == batch_size
            ), f"labels should be {batch_size}"


def test_sid_data_loader():
    rank = 0
    world_size = 1
    from configs.sid_gin_config_args import DatasetArgs, TrainerArgs
    from data.sid_data_loader import get_train_and_test_data_loader

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    dataset_args = DatasetArgs(
        dataset_name="amzn/beauty",
        max_history_length=128,
        dataset_type_str="disk_sequence_dataset",
        sequence_features_training_data_path="./tmp_data/amzn/beauty/training/22363.parquet",
        sequence_features_testing_data_path="./tmp_data/amzn/beauty/testing/22363.parquet",
        item_to_sid_mapping_path="./tmp_data/amzn/beauty/item-sid-mapping.pt",
        shuffle=False,
        num_hierarchies=4,
        codebook_sizes=[256, 256, 256, 256],
    )
    trainer_args = TrainerArgs(
        train_batch_size=128,
        eval_batch_size=128,
        max_train_iters=1000,
        max_eval_iters=100,
        seed=1234,
    )

    train_loader, eval_loader = get_train_and_test_data_loader(
        dataset_args, trainer_args
    )
    for idx, batch in enumerate(
        tqdm(train_loader, total=len(train_loader), desc="Testing train loader")
    ):
        for key in batch.features.keys():
            assert (
                batch.features[key].lengths().numel() == trainer_args.train_batch_size
            ), f"length of {key} should be {dataset_args.train_batch_size}"
        if idx < len(train_loader) - 1:
            assert (
                batch.labels.view(-1, dataset_args.num_hierarchies).shape[0]
                == trainer_args.train_batch_size
            ), f"labels should be {trainer_args.train_batch_size}"

    for idx, batch in enumerate(
        tqdm(eval_loader, total=len(eval_loader), desc="Testing eval loader")
    ):
        for key in batch.features.keys():
            assert (
                batch.features[key].lengths().numel() == trainer_args.eval_batch_size
            ), f"length of {key} should be {trainer_args.eval_batch_size}"
