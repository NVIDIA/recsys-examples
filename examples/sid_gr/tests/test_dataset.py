import pytest
import torch
from data.gpt_sid_batch import FeatureConfig, GPTSIDBatch


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
