import importlib.util

import pytest


def test_batched_beam_path_appends_initial_selection() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import select_initial_topk_batched

    logits = torch.tensor(
        [
            [[0.0, 5.0, 1.0, 3.0]],
            [[4.0, 0.0, 2.0, 6.0]],
        ]
    )
    selection = select_initial_topk_batched(logits, beam_width=2)
    path = BatchedBeamPath.create(
        batch_size=2,
        max_decode_steps=2,
        max_beam_width=2,
    )

    path.append(selection)

    assert path.batch_size == 2
    assert path.steps_done == 1
    assert path.active_widths() == (2, 2)
    assert path.token_trace(batch_idx=0, beam=0) == (1,)
    assert path.token_trace(batch_idx=1, beam=0) == (3,)


def test_batched_beam_path_rejects_batch_mismatch() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import select_initial_topk_batched

    selection = select_initial_topk_batched(torch.randn(2, 4), beam_width=2)
    path = BatchedBeamPath.create(
        batch_size=1,
        max_decode_steps=2,
        max_beam_width=2,
    )

    with pytest.raises(ValueError, match="batch_size"):
        path.append(selection)
