import importlib.util

import pytest


def test_compact_batched_beam_kv_history_gathers_shrunk_ancestors() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch

    from gr_inference.gr_kv import BatchedBeamPath, BeamKV
    from gr_inference.gr_runtime import (
        BatchedBeamSelection,
        compact_batched_beam_kv_history,
        needs_batched_beam_kv_history_compaction,
    )

    beam_kv = BeamKV(
        torch.arange(1 * 1 * 3 * 4 * 1 * 1, dtype=torch.float32).view(1, 1, 3, 4, 1, 1),
        torch.arange(100, 100 + 1 * 1 * 3 * 4 * 1 * 1, dtype=torch.float32).view(
            1,
            1,
            3,
            4,
            1,
            1,
        ),
    )
    path = BatchedBeamPath.create(
        batch_size=1,
        max_decode_steps=3,
        max_beam_width=4,
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((10, 11, 12, 13),),
            scores=((4.0, 3.0, 2.0, 1.0),),
            parent_beams=((0, 0, 0, 0),),
        )
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((20, 21),),
            scores=((5.0, 4.0),),
            parent_beams=((3, 1),),
        )
    )

    assert needs_batched_beam_kv_history_compaction(
        path,
        decode_nums=2,
        active_beam_width=2,
    )

    compact = compact_batched_beam_kv_history(
        beam_kv,
        path,
        decode_nums=2,
        active_beam_width=2,
    )

    assert compact.key_shape == beam_kv.key_shape
    assert compact.value_shape == beam_kv.value_shape
    assert torch.equal(compact.key[0, 0, 0, :2, 0, 0], torch.tensor([3.0, 1.0]))
    assert torch.equal(compact.value[0, 0, 0, :2, 0, 0], torch.tensor([103.0, 101.0]))


def test_beam_kv_history_compaction_not_needed_for_prefix_ancestors() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    from gr_inference.gr_kv import BatchedBeamPath
    from gr_inference.gr_runtime import (
        BatchedBeamSelection,
        needs_batched_beam_kv_history_compaction,
    )

    path = BatchedBeamPath.create(
        batch_size=1,
        max_decode_steps=3,
        max_beam_width=4,
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((10, 11, 12, 13),),
            scores=((4.0, 3.0, 2.0, 1.0),),
            parent_beams=((0, 0, 0, 0),),
        )
    )
    path.append(
        BatchedBeamSelection(
            token_ids=((20, 21),),
            scores=((5.0, 4.0),),
            parent_beams=((1, 0),),
        )
    )

    assert not needs_batched_beam_kv_history_compaction(
        path,
        decode_nums=2,
        active_beam_width=2,
    )
