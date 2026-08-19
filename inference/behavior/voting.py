"""
Temporal Voting / Confidence Scoring — 🔴 core contribution.

The state machine reaches SUSPICIOUS when the behavioral sequence looks
like a littering event. Voting then aggregates multiple *independent
observations* over a time window into a single score, and confirms
LITTERING only if the score crosses a threshold **and stays above it**.

This is the answer to "is this just a single-frame heuristic?" — the
decision is the product of N observations weighted by their reliability,
with a decay so that stale evidence does not accumulate.

Scoring
-------
Each observation contributes a signed weight:
  +w_object_separated   object clearly separated from hand
  +w_object_downward     object moved downward (gravity-like arc)
  +w_object_stationary   object became stationary (landed)
  +w_object_low          object in the ground band of the frame
  +w_person_away         person moved away after release
  +w_no_regrasp          no re-grasp observed within regrasp_window
  -w_regrasp             re-grasp observed (strong negative)
  -w_person_returned     person came back (strong negative)

The score is clamped to [0, max_score] and decays linearly with time
since the last contributing observation (so a stalled sequence fades).

Decision
--------
  score >= confirm_threshold  → CONFIRM (state machine → LITTERING_CONFIRMED)
  score <  revert_threshold   → REVERT (state machine → NORMAL)
  otherwise                   → HOLD (stay SUSPICIOUS; keep observing)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class VoteObservation:
    """One scored observation, timestamped."""

    timestamp: float
    object_separated: bool = False
    object_downward: bool = False
    object_stationary: bool = False
    object_low: bool = False
    person_away: bool = False
    no_regrasp: bool = False
    regrasp: bool = False
    person_returned: bool = False


@dataclass
class VotingConfig:
    # Weights
    w_object_separated: float = 1.0
    w_object_downward: float = 1.0
    w_object_stationary: float = 1.5
    w_object_low: float = 1.0
    w_person_away: float = 1.5
    w_no_regrasp: float = 1.0
    w_regrasp: float = -4.0
    w_person_returned: float = -4.0

    # Decision thresholds (score is a weighted sum, capped)
    confirm_threshold: float = 5.0
    revert_threshold: float = 1.5
    max_score: float = 10.0

    # Memory window & decay
    window_seconds: float = 5.0     # ignore observations older than this
    decay_per_second: float = 0.5   # score fades by this much per second of no new evidence


@dataclass
class VoteResult:
    score: float
    decision: str  # "CONFIRM" | "REVERT" | "HOLD"
    contributing: int  # how many observations contributed


class TemporalVoter:
    """
    Accumulates VoteObservation events and produces a score + decision.

    Used by the pipeline when the FSM is in SUSPICIOUS. The voter is
    independent of the FSM's internal counters — it is a second opinion
    grounded in the same raw signals, which is what makes confirmation
    robust.
    """

    def __init__(self, config: Optional[VotingConfig] = None) -> None:
        self.config = config or VotingConfig()
        self._obs: Deque[VoteObservation] = deque()
        self._last_score: float = 0.0
        self._last_obs_ts: Optional[float] = None

    def add(self, obs: VoteObservation) -> None:
        self._obs.append(obs)
        self._last_obs_ts = obs.timestamp
        self._prune(obs.timestamp)

    def _prune(self, now_ts: float) -> None:
        cutoff = now_ts - self.config.window_seconds
        while self._obs and self._obs[0].timestamp < cutoff:
            self._obs.popleft()

    def score(self, now_ts: Optional[float] = None) -> float:
        """
        Compute the current score at ``now_ts``. Applies decay since the
        most recent contributing observation.
        """
        if now_ts is None:
            now_ts = self._obs[-1].timestamp if self._obs else 0.0
        self._prune(now_ts)
        if not self._obs:
            self._last_score = 0.0
            return 0.0

        raw = 0.0
        c = self.config
        for o in self._obs:
            if o.object_separated: raw += c.w_object_separated
            if o.object_downward: raw += c.w_object_downward
            if o.object_stationary: raw += c.w_object_stationary
            if o.object_low: raw += c.w_object_low
            if o.person_away: raw += c.w_person_away
            if o.no_regrasp: raw += c.w_no_regrasp
            if o.regrasp: raw += c.w_regrasp
            if o.person_returned: raw += c.w_person_returned

        # decay since last obs
        if self._last_obs_ts is not None:
            idle = max(0.0, now_ts - self._last_obs_ts)
            raw -= self.config.decay_per_second * idle

        score = max(0.0, min(self.config.max_score, raw))
        self._last_score = score
        return score

    def decide(self, now_ts: Optional[float] = None) -> VoteResult:
        s = self.score(now_ts)
        if s >= self.config.confirm_threshold:
            return VoteResult(score=s, decision="CONFIRM", contributing=len(self._obs))
        if s < self.config.revert_threshold:
            return VoteResult(score=s, decision="REVERT", contributing=len(self._obs))
        return VoteResult(score=s, decision="HOLD", contributing=len(self._obs))

    def reset(self) -> None:
        self._obs.clear()
        self._last_score = 0.0
        self._last_obs_ts = None

    @property
    def observations(self) -> int:
        return len(self._obs)
