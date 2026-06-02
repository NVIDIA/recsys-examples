import importlib.util

import pytest


def test_decode_batch_planner_groups_by_beam_width() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_kv import ContextKV
    from gr_inference.gr_runtime import GRGenerationState, PrefillResult
    from gr_inference.gr_serving import GRDecodeBatchPlanner

    context_kv = ContextKV(
        torch.empty(1, 1, 4, 1, 8),
        torch.empty(1, 1, 4, 1, 8),
    )
    prefill = PrefillResult(
        logits=torch.randn(1, 4, 16),
        context_kv=context_kv,
    )
    gen_a = GRGenerationState.from_prefill(
        request_id="a",
        prefill=prefill,
        max_decode_steps=1,
        max_beam_width=4,
        fixed_beam_width=2,
    )
    gen_b = GRGenerationState.from_prefill(
        request_id="b",
        prefill=prefill,
        max_decode_steps=1,
        max_beam_width=4,
        fixed_beam_width=2,
    )

    batches = GRDecodeBatchPlanner().plan((gen_a, gen_b), step=0)

    assert len(batches) == 1
    assert batches[0].beam_width == 2
    assert batches[0].size == 2
    assert batches[0].metadata()["request_ids"] == ["a", "b"]

