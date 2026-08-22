"""
Temporal Littering State Machine — 🔴 core contribution.

The central claim of the project: littering is not a single-frame event,
it is a *temporal sequence*. This module encodes that sequence as a
deterministic finite-state machine (FSM) per (person, object) pair, with
explicit forward transitions toward LITTERING_CONFIRMED and explicit
*reversion* edges back to NORMAL when the behavioral evidence unwinds.

States
------
    UNKNOWN             initial / no association yet
    INTERACTING         person near object, not yet holding
    HOLDING             object close to hand (wrist keypoint within hold_radius)
    RELEASE             object separated from hand and moving (downward/away)
    OBJECT_ON_GROUND    object became stationary, low on the frame
    PERSON_AWAY         person moved away from object after ground contact
    LITTERING_CONFIRMED terminal positive (emit event)
    NORMAL              terminal negative / reversion target
    SUSPICIOUS          non-terminal holding area while voting is inconclusive

Forward path
------------
    UNKNOWN → INTERACTING → HOLDING → RELEASE → OBJECT_ON_GROUND
            → PERSON_AWAY → (voting) → LITTERING_CONFIRMED

Reversion paths (the part the prior project lacked)
----------------------------------------------------
    RELEASE            → HOLDING          (re-grasp within regrasp_window)
    OBJECT_ON_GROUND   → HOLDING          (person picked object back up)
    OBJECT_ON_GROUND   → NORMAL           (no person movement within abandon_window)
    PERSON_AWAY        → OBJECT_ON_GROUND (person came back)
    SUSPICIOUS         → NORMAL           (voting score decayed below revert_threshold)
    any                → NORMAL           (track lost for longer than track_lost_timeout)

Why this is defensible academically
-----------------------------------
The FSM separates *observation* (the per-frame inputs fed via `Observation`)
from *decision* (state transitions). Observations are noisy; the FSM only
advances when a condition holds for a configured dwell time, and it can
always revert. This is what makes the system a *behavior analyzer* rather
than a single-frame classifier — directly addressing the "is this just
heuristics?" question from the committee.

The module is pure logic: it takes a dataclass `Observation` and returns a
new state + reason. No CV imports. Fully unit-testable without a camera.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Optional


class LitterState(str, enum.Enum):
    UNKNOWN = "UNKNOWN"
    INTERACTING = "INTERACTING"
    HOLDING = "HOLDING"
    RELEASE = "RELEASE"
    OBJECT_ON_GROUND = "OBJECT_ON_GROUND"
    PERSON_AWAY = "PERSON_AWAY"
    SUSPICIOUS = "SUSPICIOUS"
    LITTERING_CONFIRMED = "LITTERING_CONFIRMED"
    NORMAL = "NORMAL"

    @property
    def is_terminal(self) -> bool:
        return self in (LitterState.LITTERING_CONFIRMED, LitterState.NORMAL)


@dataclass
class Observation:
    """
    Per-frame evidence for one (person, object) pair.

    All booleans are produced upstream by the association + pose modules.
    The FSM does not re-derive them; it consumes them. This isolation is
    what lets us unit-test the FSM with synthetic observation streams.

    Fields
    ------
    timestamp: float         wall-clock seconds
    person_present: bool     person track still alive
    object_present: bool     object track still alive
    hand_near_object: bool   wrist keypoint within hold_radius of object
    object_moving_down: bool object centroid moved downward this step
    object_stationary: bool  object speed below stationary_speed px/s
    object_low: bool         object centroid in lower fraction of frame (ground band)
    person_moving_away: bool person centroid receding from object
    person_re_grasped: bool  hand_near_object became true again after a release
    person_returned: bool    person came back near object after moving away
    """

    timestamp: float
    person_present: bool = True
    object_present: bool = True
    hand_near_object: bool = False
    object_moving_down: bool = False
    object_stationary: bool = False
    object_low: bool = False
    person_moving_away: bool = False
    person_re_grasped: bool = False
    person_returned: bool = False


@dataclass
class TransitionResult:
    """Outcome of one FSM step."""

    state: LitterState
    reason: str
    timestamp: float
    # Whether this step produced a terminal confirmation (for the pipeline to emit)
    confirmed: bool = False
    # Whether this step produced a reversion to NORMAL (for telemetry / FP analysis)
    reverted: bool = False


@dataclass
class StateMachineConfig:
    """
    Tunable thresholds. Defaults are conservative; the evaluation harness
    (evaluation/metrics.py) sweeps these when reporting per-scenario F1.

    Times are in seconds. Speeds/positions are unitless here — the
    association module maps px into these booleans.
    """

    hold_dwell: float = 0.25          # min time in HOLDING before a RELEASE counts
    release_dwell: float = 0.20       # min time in RELEASE before advancing
    ground_dwell: float = 0.30        # min time OBJECT_ON_GROUND must hold
    away_dwell: float = 0.30          # min time PERSON_AWAY must hold
    regrasp_window: float = 1.5       # if re-grasp happens within this, revert RELEASE→HOLDING
    abandon_window: float = 3.0       # if person doesn't leave within this after ground, → NORMAL
    track_lost_timeout: float = 2.0   # both tracks lost for this long → NORMAL
    suspicious_decay: float = 2.0     # time in SUSPICIOUS without forward progress → NORMAL


@dataclass
class _PairState:
    """Internal mutable per-pair tracking."""

    state: LitterState = LitterState.UNKNOWN
    entered_at: float = 0.0
    last_obs: Optional[Observation] = None
    # timestamp of the most recent RELEASE entry (for regrasp_window)
    released_at: Optional[float] = None
    # timestamp of most recent OBJECT_ON_GROUND entry (for abandon_window)
    grounded_at: Optional[float] = None
    # timestamp we entered SUSPICIOUS (for decay)
    suspicious_at: Optional[float] = None
    history: list = field(default_factory=list)


class LitteringStateMachine:
    """
    One FSM per (person_id, object_id) pair. The pipeline owns a dict of
    these keyed by the pair and feeds observations as frames arrive.

    Example
    -------
    >>> fsm = LitteringStateMachine()
    >>> fsm.step(Observation(timestamp=0.0, hand_near_object=True))
    >>> fsm.step(Observation(timestamp=0.5, hand_near_object=True,
    ...                      object_moving_down=True))   # → RELEASE
    >>> fsm.state
    <LitterState.RELEASE: ...>
    """

    def __init__(self, config: Optional[StateMachineConfig] = None) -> None:
        self.config = config or StateMachineConfig()
        self._s = _PairState()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> LitterState:
        return self._s.state

    @property
    def entered_at(self) -> float:
        return self._s.entered_at

    def history(self) -> list:
        """List of (timestamp, state, reason) tuples for debugging/eval."""
        return list(self._s.history)

    def reset(self, timestamp: float = 0.0) -> None:
        self._s = _PairState(state=LitterState.UNKNOWN, entered_at=timestamp)

    def step(self, obs: Observation) -> TransitionResult:
        """Advance the FSM by one observation. Returns the transition result."""
        prev = self._s.state
        new_state, reason = self._transition(obs)

        # Track-loss reversion has highest priority and is handled here.
        if self._both_tracks_lost(obs):
            new_state, reason = LitterState.NORMAL, "both tracks lost"
        elif self._s.state == LitterState.LITTERING_CONFIRMED:
            # terminal — stay
            new_state, reason = LitterState.LITTERING_CONFIRMED, "terminal"
        elif self._s.state == LitterState.NORMAL:
            # terminal negative; a brand-new holding can restart from UNKNOWN
            if obs.hand_near_object and obs.person_present and obs.object_present:
                new_state, reason = LitterState.HOLDING, "re-acquired after NORMAL"

        # Apply transition + bookkeeping
        if new_state != prev:
            self._enter(new_state, obs)
        else:
            self._s.last_obs = obs

        confirmed = new_state == LitterState.LITTERING_CONFIRMED and prev != LitterState.LITTERING_CONFIRMED
        reverted = new_state == LitterState.NORMAL and prev != LitterState.NORMAL
        result = TransitionResult(
            state=new_state, reason=reason, timestamp=obs.timestamp,
            confirmed=confirmed, reverted=reverted,
        )
        self._s.history.append((obs.timestamp, new_state, reason))
        # cap history to avoid unbounded growth in long sessions
        if len(self._s.history) > 2000:
            self._s.history = self._s.history[-1000:]
        return result

    # ------------------------------------------------------------------ #
    # Transition logic
    # ------------------------------------------------------------------ #
    def _transition(self, obs: Observation) -> tuple:
        s = self._s.state
        cfg = self.config
        dwell = obs.timestamp - self._s.entered_at

        if s == LitterState.UNKNOWN:
            if obs.hand_near_object and obs.person_present and obs.object_present:
                return LitterState.HOLDING, "hand near object"
            if obs.person_present and obs.object_present:
                return LitterState.INTERACTING, "person & object co-present"
            return LitterState.UNKNOWN, "no signal"

        if s == LitterState.INTERACTING:
            if obs.hand_near_object:
                return LitterState.HOLDING, "grasped"
            return LitterState.INTERACTING, "still interacting"

        if s == LitterState.HOLDING:
            # Reversion: if object gone or person gone → NORMAL
            if not (obs.person_present and obs.object_present):
                return LitterState.NORMAL, "track lost while holding"
            if not obs.hand_near_object and obs.object_moving_down:
                if dwell >= cfg.hold_dwell:
                    return LitterState.RELEASE, "released & moving down"
            if not obs.hand_near_object and obs.object_stationary and obs.object_low:
                # dropped straight to ground without clear release arc
                if dwell >= cfg.hold_dwell:
                    return LitterState.OBJECT_ON_GROUND, "dropped to ground"
            return LitterState.HOLDING, "still holding"

        if s == LitterState.RELEASE:
            # Reversion: re-grasp within window → back to HOLDING.
            # A REAL re-grasp means the object is held again AND has stopped
            # falling. Right after a release the object is still near the
            # hand for a few frames while it accelerates downward; treating
            # that as a grasp would cancel every real throw (found by the
            # real-video probe). While object_moving_down, it is flight.
            if obs.hand_near_object and obs.person_re_grasped and not obs.object_moving_down:
                if self._s.released_at is None or (obs.timestamp - self._s.released_at) <= cfg.regrasp_window:
                    return LitterState.HOLDING, "re-grasped (not littering)"
            if obs.object_stationary and obs.object_low:
                if dwell >= cfg.release_dwell:
                    return LitterState.OBJECT_ON_GROUND, "object settled on ground"
            # If object starts moving down again keep counting release dwell
            return LitterState.RELEASE, "in flight"

        if s == LitterState.OBJECT_ON_GROUND:
            # Reversion: person picks it back up → HOLDING.
            # A true re-grasp from ground means the object leaves the ground
            # band (it's lifted with the hand). The just-landed debris is
            # briefly still settling (speed may still be above threshold) and
            # sits right beside the hand in the throwing pose — treating that
            # as "retrieved" would cancel every real throw. The synthetic
            # probe's arm is frozen in the holding pose forever, exacerbating
            # this; on a real camera the arm swings away naturally. An object
            # that is still in the ground band is NOT being lifted.
            if obs.hand_near_object and not obs.object_low:
                return LitterState.HOLDING, "object retrieved from ground"

            # Reversion: person never leaves → NORMAL (put down, not littered)
            if self._s.grounded_at is not None and (obs.timestamp - self._s.grounded_at) >= cfg.abandon_window \
                    and not obs.person_moving_away:
                return LitterState.NORMAL, "person stayed (put down, not littered)"
            if obs.person_moving_away:
                if dwell >= cfg.ground_dwell:
                    return LitterState.PERSON_AWAY, "person leaving object"
            return LitterState.OBJECT_ON_GROUND, "on ground"

        if s == LitterState.PERSON_AWAY:
            # Reversion: person came back → OBJECT_ON_GROUND
            if obs.person_returned:
                return LitterState.OBJECT_ON_GROUND, "person returned"
            if dwell >= cfg.away_dwell:
                # hand off to voting; if voting not yet decided, go SUSPICIOUS
                return LitterState.SUSPICIOUS, "handed to temporal voting"
            return LitterState.PERSON_AWAY, "person away"

        if s == LitterState.SUSPICIOUS:
            # Decay: no forward progress → NORMAL
            if self._s.suspicious_at is not None and (obs.timestamp - self._s.suspicious_at) >= cfg.suspicious_decay:
                return LitterState.NORMAL, "suspicion decayed (no confirmation)"
            return LitterState.SUSPICIOUS, "awaiting voting"

        if s == LitterState.LITTERING_CONFIRMED:
            return LitterState.LITTERING_CONFIRMED, "terminal confirmed"
        if s == LitterState.NORMAL:
            if obs.hand_near_object and obs.person_present and obs.object_present:
                return LitterState.HOLDING, "re-acquired"
            return LitterState.NORMAL, "terminal normal"

        # Safety net
        return LitterState.UNKNOWN, "unhandled"

    # ------------------------------------------------------------------ #
    # External voting hook
    # ------------------------------------------------------------------ #
    def confirm_from_voting(self, timestamp: float, score: float, threshold: float) -> TransitionResult:
        """
        Called by the voting module when SUSPICIOUS. If score >= threshold,
        transition to LITTERING_CONFIRMED; else if below a revert floor,
        go NORMAL. Otherwise stay SUSPICIOUS.

        Returns the resulting TransitionResult.
        """
        prev = self._s.state
        if score >= threshold:
            self._enter(LitterState.LITTERING_CONFIRMED, Observation(timestamp=timestamp))
            self._s.history.append((timestamp, LitterState.LITTERING_CONFIRMED, f"voting score={score:.2f}"))
            return TransitionResult(LitterState.LITTERING_CONFIRMED, f"voting confirm {score:.2f}", timestamp, confirmed=True)
        # stay suspicious; voting module can drive decay too
        self._s.history.append((timestamp, LitterState.SUSPICIOUS, f"voting score={score:.2f} (below threshold)"))
        return TransitionResult(LitterState.SUSPICIOUS, f"voting hold {score:.2f}", timestamp)

    def force_revert(self, timestamp: float, reason: str = "external revert") -> TransitionResult:
        """Allow the voting module / pipeline to revert to NORMAL (e.g. low score sustained)."""
        self._enter(LitterState.NORMAL, Observation(timestamp=timestamp))
        self._s.history.append((timestamp, LitterState.NORMAL, reason))
        return TransitionResult(LitterState.NORMAL, reason, timestamp, reverted=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _enter(self, new_state: LitterState, obs: Observation) -> None:
        prev = self._s.state
        self._s.state = new_state
        self._s.entered_at = obs.timestamp
        self._s.last_obs = obs
        if new_state == LitterState.RELEASE and prev != LitterState.RELEASE:
            self._s.released_at = obs.timestamp
        if new_state == LitterState.OBJECT_ON_GROUND and prev != LitterState.OBJECT_ON_GROUND:
            self._s.grounded_at = obs.timestamp
        if new_state == LitterState.SUSPICIOUS and prev != LitterState.SUSPICIOUS:
            self._s.suspicious_at = obs.timestamp

    def _both_tracks_lost(self, obs: Observation) -> bool:
        if obs.person_present or obs.object_present:
            return False
        # need a last_obs to know how long both have been gone
        last = self._s.last_obs
        if last is None:
            return False
        # if the previous obs already had both gone and timeout exceeded
        if not last.person_present and not last.object_present:
            if obs.timestamp - last.timestamp >= self.config.track_lost_timeout:
                return True
        return False
