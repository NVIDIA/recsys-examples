import pytest

from gr_inference.gr_scheduler import (
    FixedBeamPolicy,
    ScheduledBeamPolicy,
    ScoreMarginBeamPolicy,
)


def test_fixed_beam_policy() -> None:
    policy = FixedBeamPolicy(beam_width=4)

    assert policy.width_for_step(0) == 4
    assert policy.width_for_step(3) == 4


def test_scheduled_beam_policy_reuses_latest_width() -> None:
    policy = ScheduledBeamPolicy({0: 4, 2: 2, 4: 1})

    assert policy.width_for_step(0) == 4
    assert policy.width_for_step(1) == 4
    assert policy.width_for_step(2) == 2
    assert policy.width_for_step(3) == 2
    assert policy.width_for_step(4) == 1


def test_scheduled_beam_policy_requires_step_zero() -> None:
    with pytest.raises(ValueError, match="step 0"):
        ScheduledBeamPolicy({1: 2})


def test_score_margin_beam_policy_shrinks_from_observed_scores() -> None:
    policy = ScoreMarginBeamPolicy(
        max_beam_width=4,
        min_beam_width=1,
        score_margin=0.5,
    )

    assert policy.width_for_step(0) == 4
    assert policy.observe_scores(0, (0.0, -0.1, -0.7, -2.0)) == 2
    assert policy.width_for_step(1) == 2
    assert policy.observe_scores(1, (0.0, -10.0)) == 1
    assert policy.width_for_step(2) == 1


def test_score_margin_beam_policy_can_allow_expansion() -> None:
    policy = ScoreMarginBeamPolicy(
        max_beam_width=4,
        min_beam_width=1,
        score_margin=1.0,
        monotonic_shrink=False,
    )

    policy.observe_scores(0, (0.0, -10.0, -20.0, -30.0))
    policy.observe_scores(1, (0.0, -0.1, -0.2, -0.3))

    assert policy.width_for_step(1) == 1
    assert policy.width_for_step(2) == 4
