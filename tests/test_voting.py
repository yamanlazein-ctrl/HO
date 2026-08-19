"""Unit tests for TemporalVoter — pure logic."""

from __future__ import annotations

import pytest

from inference.behavior.voting import (
    TemporalVoter,
    VoteObservation,
    VotingConfig,
)


def test_empty_voter_zero_score():
    v = TemporalVoter()
    assert v.score() == 0.0
    assert v.decide().decision == "REVERT"


def test_full_littering_evidence_confirms():
    v = TemporalVoter(VotingConfig(confirm_threshold=5.0))
    t = 0.0
    v.add(VoteObservation(timestamp=t, object_separated=True))
    v.add(VoteObservation(timestamp=t + 0.1, object_downward=True))
    v.add(VoteObservation(timestamp=t + 0.2, object_stationary=True))
    v.add(VoteObservation(timestamp=t + 0.3, object_low=True))
    v.add(VoteObservation(timestamp=t + 0.4, person_away=True, no_regrasp=True))
    r = v.decide()
    # 1 + 1 + 1.5 + 1 + 1.5 + 1 = 7.0 >= 5
    assert r.decision == "CONFIRM"
    assert r.score == pytest.approx(7.0, abs=0.01)


def test_regrasp_strong_negative_reverts():
    """A re-grasp observation is a strong negative; should drop below revert threshold."""
    v = TemporalVoter(VotingConfig(confirm_threshold=5.0, revert_threshold=1.5, w_regrasp=-3.0))
    t = 0.0
    v.add(VoteObservation(timestamp=t, object_separated=True))
    v.add(VoteObservation(timestamp=t + 0.1, object_downward=True))
    v.add(VoteObservation(timestamp=t + 0.2, regrasp=True))  # -3
    # score = 1 + 1 - 3 = -1 → clamped to 0 → REVERT
    r = v.decide()
    assert r.decision == "REVERT"


def test_person_returned_reverts():
    v = TemporalVoter(VotingConfig(confirm_threshold=5.0, revert_threshold=1.5))
    t = 0.0
    v.add(VoteObservation(timestamp=t, object_separated=True, object_downward=True, object_stationary=True, person_away=True))
    # 1+1+1.5+1.5 = 5.0 → CONFIRM borderline
    assert v.decide().decision == "CONFIRM"
    # now person returns
    v.add(VoteObservation(timestamp=t + 0.3, person_returned=True))  # -3
    r = v.decide()
    assert r.decision == "REVERT"


def test_decay_reduces_score_over_time():
    """Score fades by decay_per_second for each second of no new evidence."""
    v = TemporalVoter(VotingConfig(confirm_threshold=5.0, decay_per_second=2.0, window_seconds=100.0))
    v.add(VoteObservation(timestamp=0.0, object_separated=True, object_downward=True, object_stationary=True))
    # 1 + 1 + 1.5 = 3.5
    assert v.score(now_ts=0.0) == pytest.approx(3.5, abs=0.01)
    # 1 second later, decay 2.0 → 1.5
    assert v.score(now_ts=1.0) == pytest.approx(1.5, abs=0.01)
    # 2 seconds later → 0 (clamped)
    assert v.score(now_ts=3.0) == 0.0


def test_window_prunes_old_observations():
    v = TemporalVoter(VotingConfig(window_seconds=2.0))
    v.add(VoteObservation(timestamp=0.0, object_separated=True))
    v.add(VoteObservation(timestamp=5.0, object_downward=True))
    # the 0.0 obs should be pruned when we score at 5.0
    s = v.score(now_ts=5.0)
    assert s == pytest.approx(1.0, abs=0.01)  # only the downward one remains


def test_score_clamped_to_max():
    v = TemporalVoter(VotingConfig(max_score=4.0))
    for i in range(20):
        v.add(VoteObservation(timestamp=float(i), object_separated=True, person_away=True))
    assert v.score() <= 4.0


def test_reset_clears_observations():
    v = TemporalVoter()
    v.add(VoteObservation(timestamp=0.0, object_separated=True))
    assert v.observations == 1
    v.reset()
    assert v.observations == 0
    assert v.score() == 0.0


def test_hold_between_thresholds():
    """Score between revert and confirm → HOLD."""
    v = TemporalVoter(VotingConfig(confirm_threshold=5.0, revert_threshold=1.5))
    v.add(VoteObservation(timestamp=0.0, object_separated=True, object_downward=True))
    # 1 + 1 = 2.0 → between 1.5 and 5.0
    r = v.decide()
    assert r.decision == "HOLD"
