import pytest
from gr_inference.gr_kv import BeamPath


def test_beam_path_traces_tokens_through_parents() -> None:
    path = BeamPath(max_decode_steps=3, max_beam_width=4)

    path.append(parent_beams=[0, 0], token_ids=[10, 11], scores=[0.5, 0.4])
    path.append(parent_beams=[1, 0, 1], token_ids=[20, 21, 22], scores=[0.7, 0.6, 0.3])
    path.append(parent_beams=[0, 2], token_ids=[30, 31], scores=[0.9, 0.8])

    assert path.steps_done == 3
    assert path.active_beam_width == 2
    assert path.token_trace(beam=0) == (11, 20, 30)
    assert path.token_trace(beam=1) == (11, 22, 31)
    assert path.score(beam=0) == 0.9


def test_beam_path_rejects_parent_outside_previous_width() -> None:
    path = BeamPath(max_decode_steps=3, max_beam_width=4)
    path.append(parent_beams=[0, 0], token_ids=[10, 11], scores=[0.5, 0.4])

    with pytest.raises(ValueError, match="outside previous width"):
        path.append(parent_beams=[2], token_ids=[20], scores=[0.7])


def test_beam_path_rejects_decode_step_overflow() -> None:
    path = BeamPath(max_decode_steps=1, max_beam_width=4)
    path.append(parent_beams=[0], token_ids=[10], scores=[0.5])

    with pytest.raises(ValueError, match="max_decode_steps"):
        path.append(parent_beams=[0], token_ids=[11], scores=[0.4])
