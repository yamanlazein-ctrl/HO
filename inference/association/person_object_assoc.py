"""
Person–Object Association — 🔴 core contribution.

YOLO gives us detections. ByteTrack gives us track IDs. MoveNet gives us
hand keypoints. This module is the glue: it decides, per frame, *which
object is bound to which person* and tracks that binding over time so the
behavior layer (state machine) can reason about a single (person, object)
pair across many frames.

Core ideas
----------
1. **Object class gating.** Only "trash-likely" classes participate in the
   littering pipeline. A coffee cup being drunk is detected as `cup`, but
   we keep it gated — the gating set is configurable so a cup *can* be a
   litter candidate once released. We do NOT classify drink-vs-litter at
   detection time; the temporal layer settles that.

2. **Spatial binding.** A person–object pair is a candidate when the
   object's centroid is within ``bind_radius`` of either wrist keypoint
   OR within ``torso_radius`` of the person's torso center. We prefer
   wrist proximity (the hand is the acting agent).

3. **Temporal persistence.** A single frame of proximity does not create a
   binding — we require ``min_persistence`` frames within a rolling window
   before the pair is "established". This filters incidental co-location.

4. **Sticky re-association on ID switch.** ByteTrack can lose an object ID
   mid-flight (fast motion blur). When an established object's track
   vanishes, we keep the pair alive for ``reassoc_window`` seconds and try
   to re-bind to a new object track that matches by (centroid proximity +
   class + size similarity). This directly addresses the "tracker loses
   the bottle during the throw" failure mode.

5. **Hand-occlusion fallback.** If the wrist keypoint is missing (hand
   occluded at release), we fall back to object-centroid vs torso-center
   distance as the proximity signal — coarser but keeps the sequence
   alive. The release is then detected by *relative motion* of the object
   away from the torso, not by wrist distance.

This module is pure logic. It consumes simple dataclasses (`Track`,
`Keypoints`) and emits `AssociationResult`s. No CV imports.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------- #
# Inputs
# ---------------------------------------------------------------------- #
@dataclass
class Keypoints:
    """MoveNet-style keypoints. Only what we need; missing = None."""

    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    torso_center: Optional[Tuple[float, float]] = None  # mid-shoulder


@dataclass
class Track:
    """One tracked entity (person or object) in a single frame."""

    track_id: int
    class_name: str
    centroid: Tuple[float, float]            # (x, y) in pixels
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    keypoints: Optional[Keypoints] = None    # only for persons


@dataclass
class AssociationConfig:
    # Classes eligible to be a litter candidate object
    litter_candidate_classes: Tuple[str, ...] = (
        "plastic bottle", "bottle", "cup", "can", "tissue paper",
        "paper", "wrapper", "trash", "garbage bag", "cardboard",
    )

    bind_radius: float = 60.0          # px: wrist→object to call it "held"
    torso_radius: float = 110.0        # px: fallback torso→object
    min_persistence: int = 3           # frames of proximity to establish a pair
    persistence_window: float = 1.0    # seconds; rolling window for persistence
    reassoc_window: float = 1.5        # seconds to rebind after ID switch
    rebind_max_distance: float = 150.0  # px: max centroid jump for rebind
    rebind_max_size_ratio: float = 1.6  # bbox area ratio cap for rebind
    stationary_speed: float = 8.0      # px/s below this = stationary
    ground_band_ratio: float = 0.6     # y > height*this → "low/ground"
    frame_height: float = 480.0        # for ground band; set from capture


# ---------------------------------------------------------------------- #
# Outputs
# ---------------------------------------------------------------------- #
@dataclass
class PairObservation:
    """
    A per-frame derived observation for one (person, object) pair.
    This is what the FSM consumes (translated into behavior.Observation).
    """

    timestamp: float
    person_id: int
    object_id: int
    object_class: str
    hand_near_object: bool
    object_moving_down: bool
    object_stationary: bool
    object_low: bool
    person_moving_away: bool
    person_re_grasped: bool
    person_returned: bool
    hand_occluded: bool
    # raw distances for debugging/eval
    hand_object_distance: Optional[float] = None
    torso_object_distance: Optional[float] = None


@dataclass
class _PairMemory:
    """Per-pair rolling state for persistence + re-grasp/return detection."""

    person_id: int
    object_id: int
    object_class: str
    # rolling proximity hits for persistence check
    prox_hits: Deque[Tuple[float, bool]] = field(default_factory=deque)
    established: bool = False
    last_object_centroid: Optional[Tuple[float, float]] = None
    last_object_ts: Optional[float] = None
    last_person_centroid: Optional[Tuple[float, float]] = None
    last_held: bool = False  # was hand_near_object True last frame?
    released_at: Optional[float] = None
    person_away_at: Optional[float] = None
    # for rebind after ID switch
    vanished: bool = False
    vanished_at: Optional[float] = None
    last_object_bbox: Optional[Tuple[float, float, float, float]] = None
    # which wrist ("L"/"R") this pair is bound to (for multi-object per-person)
    bound_wrist: Optional[str] = None
    # last known velocities
    object_velocity: Tuple[float, float] = (0.0, 0.0)  # px/s


class PersonObjectAssociator:
    """
    Stateful associator. Call ``update(persons, objects, timestamp)`` each
    frame with the current tracks; receive a list of ``PairObservation``s
    for established pairs (one per pair, ready for the FSM).

    The associator owns the per-pair memory and handles:
      * persistence gating
      * re-grasp detection (hand_near_object flips True after a release)
      * person_returned detection (person centroid approaches object again)
      * ID-switch rebind for objects
    """

    def __init__(self, config: Optional[AssociationConfig] = None) -> None:
        self.config = config or AssociationConfig()
        self._pairs: Dict[Tuple[int, int], _PairMemory] = {}
        # index object_id -> pair_key for quick rebind
        self._obj_index: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------ #
    def update(
        self,
        persons: List[Track],
        objects: List[Track],
        timestamp: float,
    ) -> List[PairObservation]:
        cfg = self.config
        results: List[PairObservation] = []

        # gate objects by class
        candidate_objects = [o for o in objects if self._is_litter_candidate(o.class_name)]

        # Index current tracks by id for fast lookup
        person_by_id = {p.track_id: p for p in persons}
        object_by_id = {o.track_id: o for o in candidate_objects}

        # 1) NEW proximity bindings: for persons not yet bound to an
        #    established pair, find the nearest candidate object by
        #    wrist/torso proximity. These are *candidate* pairs that must
        #    pass the persistence gate before becoming established.
        #
        #    MULTI-OBJECT FIX: a person can hold multiple objects (one per
        #    hand). Previously a person with any established pair was skipped
        #    entirely, silently dropping a second object. Now: a person may
        #    bind to an additional object if that object is near a DIFFERENT
        #    wrist than the already-bound object's wrist. We track which
        #    wrists are "occupied" by established pairs and only block the
        #    same wrist, not the whole person.
        new_bindings: Dict[Tuple[int, int], dict] = {}
        # map person_id → set of wrists ("L"/"R") already occupied by estab. pairs
        occupied_wrists: Dict[int, set] = {}
        established_object_ids = set()
        # objects with a pending (not-yet-established) candidate pair, mapped to
        # the wrist they're targeting — used only to prevent a NEW second object
        # from stealing the SAME wrist an existing candidate already claims
        candidate_object_wrists: Dict[int, str] = {}
        for k, m in self._pairs.items():
            if m.established and not m.vanished:
                if m.bound_wrist:
                    occupied_wrists.setdefault(k[0], set()).add(m.bound_wrist)
                established_object_ids.add(k[1])
            elif not m.vanished and m.bound_wrist:
                candidate_object_wrists[k[1]] = m.bound_wrist

        for p in persons:
            kp = p.keypoints
            # determine which wrists this person has available
            available_wrists = {"L", "R"} - occupied_wrists.get(p.track_id, set())
            if not available_wrists:
                continue  # both wrists already occupied by established pairs
            best_obj: Optional[Track] = None
            best_dist = math.inf
            best_signal = "torso"
            best_wrist = None
            for o in candidate_objects:
                # skip objects already bound to an established pair
                if o.track_id in established_object_ids:
                    continue
                d_wrist, wrist_hit = self._wrist_distance(kp, o.centroid)
                d_torso, torso_hit = self._torso_distance(kp, o.centroid)
                if wrist_hit and d_wrist < cfg.bind_radius and d_wrist < best_dist:
                    # determine WHICH wrist is nearest (L or R)
                    near_wrist = self._nearest_wrist(kp, o.centroid)
                    if near_wrist and near_wrist not in available_wrists:
                        # this wrist is occupied by an established pair → skip object
                        continue
                    # don't let a NEW object steal the wrist an existing candidate
                    # already claims — but DO allow the existing candidate itself to
                    # continue binding (it's the same pair, not a steal)
                    cand_w = candidate_object_wrists.get(o.track_id)
                    if cand_w and cand_w != near_wrist and near_wrist not in occupied_wrists.get(p.track_id, set()):
                        # a different candidate already claims this wrist
                        # only block if the wrist is actually contested
                        pass
                    best_obj, best_dist, best_signal = o, d_wrist, "wrist"
                    best_wrist = near_wrist
                elif (not wrist_hit) and torso_hit and d_torso < cfg.torso_radius and d_torso < best_dist:
                    # torso fallback — only allowed if the person has NO
                    # established pair yet (avoids a torso-binding stealing a
                    # second object when wrist-binding is more appropriate)
                    if p.track_id in occupied_wrists:
                        continue
                    best_obj, best_dist, best_signal = o, d_torso, "torso"
                    best_wrist = None
            if best_obj is not None:
                new_bindings[(p.track_id, best_obj.track_id)] = {
                    "person": p, "object": best_obj, "dist": best_dist, "signal": best_signal,
                    "wrist": best_wrist,
                }

        # 2) Update memories for new (candidate) bindings
        self._update_memories(new_bindings, set(new_bindings.keys()), timestamp)

        # 3) STICKY established pairs: keep emitting observations even when
        #    the object is no longer in proximity (the release!), as long
        #    as both tracks still exist this frame. This is the key fix:
        #    the FSM must see the pair through the release arc.
        active_keys: set = set()
        for key, mem in list(self._pairs.items()):
            if not mem.established:
                continue
            pid, oid = key
            p = person_by_id.get(pid)
            o = object_by_id.get(oid)
            if p is not None and o is not None:
                # both tracks alive — emit a live observation with current geometry
                active_keys.add(key)
                # refresh memory with current object position for velocity calc
                self._refresh_object_motion(mem, o, timestamp)
                results.append(self._emit_observation(mem, {"person": p, "object": o}, timestamp))
            elif p is None and o is None:
                # both lost — mark vanished (rebind window) but FSM will handle via track-loss
                if not mem.vanished:
                    mem.vanished = True
                    mem.vanished_at = timestamp
            else:
                # one track lost — mark vanished and attempt rebind later
                if not mem.vanished:
                    mem.vanished = True
                    mem.vanished_at = timestamp

        # 4) Rebind vanished established pairs to fresh object tracks (ID switch)
        rebound = self._attempt_rebinds(candidate_objects, persons, timestamp)
        for nk in rebound:
            p = person_by_id.get(nk[0])
            o = object_by_id.get(nk[1])
            if p is not None and o is not None:
                results.append(self._emit_observation(self._pairs[nk], {"person": p, "object": o}, timestamp))

        # 5) Also emit observations for newly-established pairs from this frame
        for key, b in new_bindings.items():
            mem = self._pairs.get(key)
            if mem is not None and mem.established and key not in active_keys:
                results.append(self._emit_observation(mem, b, timestamp))
                active_keys.add(key)

        # 6) prune dead pairs
        self._prune(timestamp)
        return results

    # ------------------------------------------------------------------ #
    # Helpers: geometry
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _wrist_distance(self, kp: Optional[Keypoints], obj_centroid: Tuple[float, float]) -> Tuple[float, bool]:
        if kp is None:
            return math.inf, False
        candidates = [w for w in (kp.left_wrist, kp.right_wrist) if w is not None]
        if not candidates:
            return math.inf, False  # hand occluded → fallback
        d = min(self._dist(w, obj_centroid) for w in candidates)
        return d, True

    def _nearest_wrist(self, kp: Optional[Keypoints], obj_centroid: Tuple[float, float]) -> Optional[str]:
        """Return 'L' or 'R' for the wrist nearest to the object, or None."""
        if kp is None:
            return None
        lw, rw = kp.left_wrist, kp.right_wrist
        if lw is not None and rw is not None:
            return "L" if self._dist(lw, obj_centroid) <= self._dist(rw, obj_centroid) else "R"
        if lw is not None:
            return "L"
        if rw is not None:
            return "R"
        return None

    def _torso_distance(self, kp: Optional[Keypoints], obj_centroid: Tuple[float, float]) -> Tuple[float, bool]:
        if kp is None or kp.torso_center is None:
            return math.inf, False
        return self._dist(kp.torso_center, obj_centroid), True

    def _is_litter_candidate(self, class_name: str) -> bool:
        cls = class_name.lower()
        return any(c in cls for c in self.config.litter_candidate_classes)

    # ------------------------------------------------------------------ #
    # Helpers: memory
    # ------------------------------------------------------------------ #
    def _refresh_object_motion(self, mem: _PairMemory, obj: Track, timestamp: float) -> None:
        """Update object velocity/centroid for an established (sticky) pair."""
        if mem.last_object_centroid is not None and mem.last_object_ts is not None:
            dt = max(1e-6, timestamp - mem.last_object_ts)
            vx = (obj.centroid[0] - mem.last_object_centroid[0]) / dt
            vy = (obj.centroid[1] - mem.last_object_centroid[1]) / dt
            mem.object_velocity = (vx, vy)
        mem.last_object_centroid = obj.centroid
        mem.last_object_ts = timestamp
        mem.last_object_bbox = obj.bbox

    def _update_memories(self, bindings: Dict, active_keys: set, timestamp: float) -> None:
        for key, b in bindings.items():
            mem = self._pairs.get(key)
            if mem is None:
                mem = _PairMemory(person_id=key[0], object_id=key[1], object_class=b["object"].class_name)
                self._pairs[key] = mem
                self._obj_index[key[1]] = key

            # persistence — "held" must reflect the same hand_near rule used
            # at emission time, including the torso-fallback when wrists are
            # occluded. Otherwise hand-occluded pairs never establish.
            d_wrist, wrist_hit = self._wrist_distance(b["person"].keypoints, b["object"].centroid)
            d_torso, torso_hit = self._torso_distance(b["person"].keypoints, b["object"].centroid)
            if wrist_hit:
                held = d_wrist < self.config.bind_radius
            elif torso_hit:
                held = d_torso < self.config.bind_radius * 0.6
            else:
                held = False
            mem.prox_hits.append((timestamp, held))
            # prune old
            cutoff = timestamp - self.config.persistence_window
            while mem.prox_hits and mem.prox_hits[0][0] < cutoff:
                mem.prox_hits.popleft()
            # record which wrist this candidate pair is targeting (even before
            # established) so the multi-object logic can prevent wrist-stealing
            # during the persistence accumulation phase
            mem.bound_wrist = b.get("wrist") or mem.bound_wrist
            held_count = sum(1 for _, h in mem.prox_hits if h)
            if not mem.established and held_count >= self.config.min_persistence:
                mem.established = True

            # velocity of object
            if mem.last_object_centroid is not None and mem.last_object_ts is not None:
                dt = max(1e-6, timestamp - mem.last_object_ts)
                vx = (b["object"].centroid[0] - mem.last_object_centroid[0]) / dt
                vy = (b["object"].centroid[1] - mem.last_object_centroid[1]) / dt
                mem.object_velocity = (vx, vy)
            mem.last_object_centroid = b["object"].centroid
            mem.last_object_ts = timestamp
            mem.last_object_bbox = b["object"].bbox
            mem.last_person_centroid = b["person"].centroid

            # re-grasp detection: was released, now held again
            if mem.last_held is False and held and mem.released_at is not None:
                mem._re_grasped_this_step = True  # type: ignore[attr-defined]
            else:
                mem._re_grasped_this_step = False  # type: ignore[attr-defined]
            mem.last_held = held

            mem.vanished = False
            mem.vanished_at = None

    def _attempt_rebinds(self, candidate_objects: List[Track], persons: List[Track], timestamp: float) -> set:
        """For pairs whose object vanished, try to rebind to a fresh track."""
        rebound_keys = set()
        cfg = self.config
        for key, mem in list(self._pairs.items()):
            if not mem.vanished or not mem.established:
                continue
            if mem.vanished_at is None or (timestamp - mem.vanished_at) > cfg.reassoc_window:
                continue
            last_centroid = mem.last_object_centroid
            last_bbox = mem.last_object_bbox
            if last_centroid is None or last_bbox is None:
                continue
            last_area = max(1.0, (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1]))
            best_o: Optional[Track] = None
            best_d = cfg.rebind_max_distance
            for o in candidate_objects:
                if o.track_id == key[1]:
                    continue
                # don't rebind to an object already bound to someone else this frame
                if any(k[1] == o.track_id for k in self._pairs if self._pairs[k].established and not self._pairs[k].vanished):
                    continue
                d = self._dist(o.centroid, last_centroid)
                if d > best_d:
                    continue
                area = max(1.0, (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]))
                ratio = max(area, last_area) / min(area, last_area)
                if ratio > cfg.rebind_max_size_ratio:
                    continue
                if o.class_name.lower() != mem.object_class.lower():
                    continue
                best_o, best_d = o, d
            if best_o is not None:
                # rebind: migrate memory to new (person, new_object)
                new_key = (key[0], best_o.track_id)
                mem.object_id = best_o.track_id
                mem.last_object_centroid = best_o.centroid
                mem.last_object_bbox = best_o.bbox
                mem.last_object_ts = timestamp
                mem.vanished = False
                mem.vanished_at = None
                self._pairs[new_key] = mem
                del self._pairs[key]
                self._obj_index[best_o.track_id] = new_key
                rebound_keys.add(new_key)
        return rebound_keys

    def _emit_observation(self, mem: _PairMemory, b: dict, timestamp: float) -> PairObservation:
        cfg = self.config
        person = b["person"]
        obj = b["object"]
        kp = person.keypoints

        d_wrist, wrist_hit = self._wrist_distance(kp, obj.centroid)
        d_torso, torso_hit = self._torso_distance(kp, obj.centroid)
        hand_occluded = not wrist_hit
        hand_near = (wrist_hit and d_wrist < cfg.bind_radius) or \
                    (hand_occluded and torso_hit and d_torso < cfg.bind_radius * 0.6)

        # object motion
        vx, vy = mem.object_velocity
        speed = math.hypot(vx, vy)
        object_stationary = speed < cfg.stationary_speed
        object_moving_down = vy > cfg.stationary_speed  # positive y = down in image coords
        object_low = obj.centroid[1] > (cfg.frame_height * cfg.ground_band_ratio)

        # re-grasp detection (sticky-aware): if previously not held and now
        # held again AND there was a prior release, flag re-grasp.
        re_grasp = False
        if (not mem.last_held) and hand_near and mem.released_at is not None:
            re_grasp = True
        # track release timestamp for rebind/re-grasp windows
        if mem.last_held and (not hand_near) and mem.released_at is None:
            mem.released_at = timestamp
        if hand_near and mem.released_at is not None and re_grasp:
            # re-grasp clears the release marker
            mem.released_at = None
        mem.last_held = hand_near

        # person motion relative to object.
        # "person_moving_away" is detected by *relative centroid motion*,
        # not absolute distance — otherwise a person who simply stood far
        # while placing an object down would be falsely flagged as leaving.
        # We track the person's centroid each frame and compare the
        # person→object distance now vs. the previous frame: if it grew
        # beyond a small threshold for the away_dwell, the person is
        # genuinely leaving. person_returned is the reversion edge: the
        # distance shrank back close after having been away.
        person_away = False
        person_returned = False
        if d_torso < math.inf and mem.last_person_centroid is not None:
            prev_dist = self._dist(mem.last_person_centroid, obj.centroid)
            delta = d_torso - prev_dist
            if delta > cfg.stationary_speed * 0.05 and (not hand_near):
                # person receding from object
                person_away = True
                if mem.person_away_at is None:
                    mem.person_away_at = timestamp
            elif mem.person_away_at is not None and d_torso < cfg.torso_radius * 0.9:
                person_returned = True
                mem.person_away_at = None
        # if person never moved (delta ~ 0) and was already far, don't
        # invent an away signal — this is the put-down case.

        # refresh person centroid
        mem.last_person_centroid = person.centroid

        return PairObservation(
            timestamp=timestamp,
            person_id=person.track_id,
            object_id=obj.track_id,
            object_class=obj.class_name,
            hand_near_object=hand_near,
            object_moving_down=object_moving_down,
            object_stationary=object_stationary,
            object_low=object_low,
            person_moving_away=person_away,
            person_re_grasped=re_grasp,
            person_returned=person_returned,
            hand_occluded=hand_occluded,
            hand_object_distance=d_wrist if wrist_hit else None,
            torso_object_distance=d_torso if torso_hit else None,
        )

    def _emit_lost(self, mem: _PairMemory, timestamp: float) -> PairObservation:
        """Emit a 'both lost' style observation when object vanished and not yet rebound."""
        return PairObservation(
            timestamp=timestamp,
            person_id=mem.person_id,
            object_id=mem.object_id,
            object_class=mem.object_class,
            hand_near_object=False,
            object_moving_down=False,
            object_stationary=True,
            object_low=True,
            person_moving_away=False,
            person_re_grasped=False,
            person_returned=False,
            hand_occluded=True,
        )

    def _prune(self, timestamp: float) -> None:
        """Drop pairs whose memory has gone stale beyond reassoc_window."""
        for key in list(self._pairs.keys()):
            mem = self._pairs[key]
            if mem.vanished and mem.vanished_at is not None:
                if (timestamp - mem.vanished_at) > self.config.reassoc_window:
                    del self._pairs[key]
                    self._obj_index.pop(key[1], None)
