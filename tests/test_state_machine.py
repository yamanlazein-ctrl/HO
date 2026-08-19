"""Unit tests for the Temporal Littering State Machine — pure logic."""

from __future__ import annotations

import pytest

from inference.behavior.state_machine import (
    LitterState,
    LitteringStateMachine,
    Observation,
    StateMachineConfig,
)


def _fsm(**kw) -> LitteringStateMachine:
    return LitteringStateMachine(StateMachineConfig(**kw))


# ---------------- Forward path ----------------

def test_unknown_to_holding():
    fsm = _fsm()
    r = fsm.step(Observation(timestamp=0.0, person_present=True, object_present=True, hand_near_object=True))
    assert r.state == LitterState.HOLDING
    assert r.reason == "hand near object"


def test_full_forward_path_to_suspicious():
    """Drive the canonical forward sequence and reach SUSPICIOUS."""
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, ground_dwell=0.1, away_dwell=0.1)
    t = 0.0
    # HOLDING
    fsm.step(Observation(timestamp=t, hand_near_object=True))
    assert fsm.state == LitterState.HOLDING
    t += 0.2
    # RELEASE
    fsm.step(Observation(timestamp=t, hand_near_object=False, object_moving_down=True))
    assert fsm.state == LitterState.RELEASE
    t += 0.2
    # OBJECT_ON_GROUND
    fsm.step(Observation(timestamp=t, object_stationary=True, object_low=True, hand_near_object=False))
    assert fsm.state == LitterState.OBJECT_ON_GROUND
    t += 0.2
    # PERSON_AWAY
    fsm.step(Observation(timestamp=t, person_moving_away=True, object_stationary=True, object_low=True))
    assert fsm.state == LitterState.PERSON_AWAY
    t += 0.2
    # SUSPICIOUS (handed to voting)
    fsm.step(Observation(timestamp=t, person_moving_away=True, object_stationary=True, object_low=True))
    assert fsm.state == LitterState.SUSPICIOUS


def test_release_requires_hold_dwell():
    """A instantaneous grab-release should not advance to RELEASE."""
    fsm = _fsm(hold_dwell=0.5)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    assert fsm.state == LitterState.HOLDING
    # immediately release — too soon
    fsm.step(Observation(timestamp=0.1, hand_near_object=False, object_moving_down=True))
    assert fsm.state == LitterState.HOLDING  # stayed


# ---------------- Reversion paths (the contribution) ----------------

def test_regrasp_reverts_release_to_holding():
    """Person re-grasps within regrasp_window → back to HOLDING (NOT littering)."""
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, regrasp_window=2.0)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, hand_near_object=False, object_moving_down=True))
    assert fsm.state == LitterState.RELEASE
    # re-grasp quickly
    r = fsm.step(Observation(timestamp=0.5, hand_near_object=True, person_re_grasped=True))
    assert r.state == LitterState.HOLDING
    assert "re-grasped" in r.reason


def test_person_picks_object_back_up_reverts_ground_to_holding():
    """Object on ground, then person retrieves it → HOLDING (NOT littering)."""
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, ground_dwell=0.1)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, hand_near_object=False, object_moving_down=True))
    fsm.step(Observation(timestamp=0.4, object_stationary=True, object_low=True))
    assert fsm.state == LitterState.OBJECT_ON_GROUND
    # person picks it back up
    r = fsm.step(Observation(timestamp=0.6, hand_near_object=True))
    assert r.state == LitterState.HOLDING
    assert r.reason == "object retrieved from ground"


def test_put_down_not_littered_reverts_to_normal():
    """
    Critical scenario: person places object on ground and STAYS (doesn't
    leave). After abandon_window with no movement away → NORMAL.
    """
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, ground_dwell=0.1, abandon_window=1.0)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, object_stationary=True, object_low=True, hand_near_object=False))
    assert fsm.state == LitterState.OBJECT_ON_GROUND
    # person stays put, no moving away
    r = fsm.step(Observation(timestamp=1.5, object_stationary=True, object_low=True, person_moving_away=False))
    assert r.state == LitterState.NORMAL
    assert "put down" in r.reason
    assert r.reverted is True


def test_person_returned_reverts_person_away_to_ground():
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, ground_dwell=0.1, away_dwell=0.5)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, hand_near_object=False, object_moving_down=True))
    fsm.step(Observation(timestamp=0.4, object_stationary=True, object_low=True))
    fsm.step(Observation(timestamp=0.6, person_moving_away=True))
    assert fsm.state == LitterState.PERSON_AWAY
    # person comes back before away_dwell completes
    r = fsm.step(Observation(timestamp=0.7, person_returned=True))
    assert r.state == LitterState.OBJECT_ON_GROUND


def test_suspicious_decays_to_normal():
    """SUSPICIOUS with no voting confirmation decays to NORMAL."""
    fsm = _fsm(hold_dwell=0.1, release_dwell=0.1, ground_dwell=0.1, away_dwell=0.1, suspicious_decay=1.0)
    # ... reach SUSPICIOUS
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, hand_near_object=False, object_moving_down=True))
    fsm.step(Observation(timestamp=0.4, object_stationary=True, object_low=True))
    fsm.step(Observation(timestamp=0.6, person_moving_away=True))
    fsm.step(Observation(timestamp=0.8, person_moving_away=True))
    assert fsm.state == LitterState.SUSPICIOUS
    # wait past decay
    r = fsm.step(Observation(timestamp=2.5, person_moving_away=False, object_stationary=True))
    assert r.state == LitterState.NORMAL
    assert r.reverted is True


def test_both_tracks_lost_reverts_to_normal():
    fsm = _fsm(track_lost_timeout=1.0)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    assert fsm.state == LitterState.HOLDING
    # both tracks disappear
    fsm.step(Observation(timestamp=0.5, person_present=False, object_present=False))
    # not yet — timeout not reached (need two consecutive lost obs spanning timeout)
    # Actually our impl checks elapsed between consecutive both-lost obs.
    fsm.step(Observation(timestamp=1.6, person_present=False, object_present=False))
    assert fsm.state == LitterState.NORMAL


# ---------------- Voting hook ----------------

def test_confirm_from_voting():
    fsm = _fsm()
    fsm._s.state = LitterState.SUSPICIOUS
    fsm._s.suspicious_at = 0.0
    r = fsm.confirm_from_voting(timestamp=1.0, score=6.0, threshold=5.0)
    assert r.state == LitterState.LITTERING_CONFIRMED
    assert r.confirmed is True


def test_voting_below_threshold_stays_suspicious():
    fsm = _fsm()
    fsm._s.state = LitterState.SUSPICIOUS
    fsm._s.suspicious_at = 0.0
    r = fsm.confirm_from_voting(timestamp=1.0, score=2.0, threshold=5.0)
    assert r.state == LitterState.SUSPICIOUS
    assert r.confirmed is False


def test_force_revert():
    fsm = _fsm()
    fsm._s.state = LitterState.SUSPICIOUS
    r = fsm.force_revert(timestamp=2.0, reason="low score sustained")
    assert r.state == LitterState.NORMAL
    assert r.reverted is True


# ---------------- Terminal behavior ----------------

def test_confirmed_is_terminal():
    fsm = _fsm()
    fsm._s.state = LitterState.LITTERING_CONFIRMED
    r = fsm.step(Observation(timestamp=99.0, hand_near_object=True))
    assert r.state == LitterState.LITTERING_CONFIRMED


def test_reacquire_after_normal():
    fsm = _fsm()
    fsm._s.state = LitterState.NORMAL
    r = fsm.step(Observation(timestamp=1.0, hand_near_object=True, person_present=True, object_present=True))
    assert r.state == LitterState.HOLDING


def test_history_recorded():
    fsm = _fsm(hold_dwell=0.1)
    fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    fsm.step(Observation(timestamp=0.2, hand_near_object=False, object_moving_down=True))
    h = fsm.history()
    assert len(h) == 2
    assert h[0][1] == LitterState.HOLDING
    assert h[1][1] == LitterState.RELEASE
