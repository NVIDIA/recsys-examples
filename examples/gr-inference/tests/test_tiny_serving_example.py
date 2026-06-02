import importlib.util
from argparse import Namespace

import pytest


def test_tiny_serving_example_run_demo() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    from run_tiny_serving import run_demo

    summary = run_demo(
        Namespace(
            layers=1,
            context_len=8,
            decode_steps=2,
            beam_width=2,
            vocab_size=32,
            requests=2,
            warmup=True,
            max_batch_size=2,
            batched_decode=True,
            return_beam_details=True,
        )
    )

    assert summary["responses"] == 2
    assert summary["first_request_id"] == "req-0"
    assert len(summary["first_token_ids"]) == 2
    assert summary["first_metadata"]["total_ms"] >= 0.0
    assert summary["scheduler_status"]["finished"] == 2
    assert summary["scheduler_status"]["assembled_batches"] == 1
    assert summary["scheduler_status"]["avg_batch_size"] == 2.0
    assert summary["scheduler_metrics"]["processed_requests"] == 2
    assert summary["batch_history"][0]["tensor_batch_compatible"]
    assert summary["first_metadata"]["decode_batch_plan"][0]["size"] == 2
    assert summary["engine_status"]["max_beam_width"] == 2
    assert summary["engine_status"]["enable_batched_decode"] is True
    assert summary["first_metadata"]["batched_decode"] is True
    assert summary["first_metadata"]["batched_decode_steps"] == 2
    assert summary["first_metadata"]["batched_beam_path_steps"] == 3
    assert len(summary["first_metadata"]["beam_details"][0]["token_ids"]) == 3
    assert len(summary["first_metadata"]["beam_details"][0]["token_logprobs"]) == 3
    assert summary["first_metadata"]["beam_details"][0]["logprob_type"] == "token_logsoftmax"
    assert summary["first_metadata"]["beam_details"][0]["score_type"] == (
        "beam_score_logprob_cumulative"
    )
    assert summary["engine_status"]["beam_score_mode"] == "logprob"
