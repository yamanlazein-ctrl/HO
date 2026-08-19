"""Unit tests for PersonObjectAssociator — pure logic, synthetic tracks."""

from __future__ import annotations

import pytest

from inference.association.person_object_assoc import (
    AssociationConfig,
    Keypoints,
    PersonObjectAssociator,
    Track,
)


def _person(pid: int, cx: float, cy: float, kp: Keypoints = None) -> Track:
    return Track(track_id=pid, class_name="person", centroid=(cx, cy),
                 bbox=(cx - 40, cy - 80, cx + 40, cy + 80), keypoints=kp)


def _obj(oid: int, cls: str, cx: float, cy: float, w: float = 30, h: float = 30) -> Track:
    return Track(track_id=oid, class_name=cls, centroid=(cx, cy),
                 bbox=(cx - w, cy - h, cx + w, cy + h), keypoints=None)


def _kp(lw=None, rw=None, tc=(100, 100)) -> Keypoints:
    return Keypoints(left_wrist=lw, right_wrist=rw, torso_center=tc)


# ---------------- class gating ----------------

def test_non_litter_class_ignored():
    assoc = PersonObjectAssociator()
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    obj = _obj(2, "person", 120, 110)  # not a litter class
    out = assoc.update([p], [obj], timestamp=0.0)
    assert out == []  # nothing established


def test_litter_class_attaches():
    cfg = AssociationConfig(min_persistence=2, bind_radius=60.0)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "plastic bottle", 120, 110)
    # frame 1: proximity recorded but not yet established
    out = assoc.update([p], [bottle], timestamp=0.0)
    assert out == []
    # frame 2: established
    out = assoc.update([p], [bottle], timestamp=0.05)
    assert len(out) == 1
    assert out[0].person_id == 1 and out[0].object_id == 2
    assert out[0].hand_near_object is True


# ---------------- hand-occlusion fallback ----------------

def test_hand_occluded_falls_back_to_torso():
    cfg = AssociationConfig(min_persistence=2, bind_radius=60.0, torso_radius=110.0, frame_height=480)
    assoc = PersonObjectAssociator(cfg)
    # wrist missing, object near torso
    p = _person(1, 100, 100, _kp(lw=None, rw=None, tc=(100, 100)))
    bottle = _obj(2, "bottle", 105, 100)  # ~5px from torso
    assoc.update([p], [bottle], timestamp=0.0)
    out = assoc.update([p], [bottle], timestamp=0.05)
    assert len(out) == 1
    assert out[0].hand_occluded is True
    # because torso dist (5) < bind_radius*0.6 (36), still considered "held"
    assert out[0].hand_near_object is True


# ---------------- ID-switch rebind ----------------

def test_id_switch_rebinds_to_matching_object():
    """
    ByteTrack loses the bottle mid-throw (id 2 vanishes) and re-creates it
    as id 5 nearby, same class, similar size. The associator should rebind.
    """
    cfg = AssociationConfig(
        min_persistence=2, bind_radius=60.0, rebind_max_distance=150.0,
        reassoc_window=2.0, frame_height=480,
    )
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "plastic bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)  # established
    assert (1, 2) in assoc._pairs

    # bottle vanishes (only person this frame)
    assoc.update([p], [], timestamp=0.1)
    # bottle reappears as id 5 within rebind distance
    bottle2 = _obj(5, "plastic bottle", 125, 115)
    out = assoc.update([p], [bottle2], timestamp=0.15)
    # rebind should have happened
    assert any(o.object_id == 5 for o in out) or (1, 5) in assoc._pairs


def test_rebind_rejects_wrong_class():
    cfg = AssociationConfig(min_persistence=2, bind_radius=60.0, rebind_max_distance=200.0,
                            reassoc_window=2.0, frame_height=480)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "plastic bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)
    assoc.update([p], [], timestamp=0.1)
    cup = _obj(5, "cup", 125, 115)  # different class
    out = assoc.update([p], [cup], timestamp=0.15)
    # should NOT rebind bottle memory to a cup
    assert not any(o.object_id == 5 and o.object_class == "cup" for o in out) or len(out) == 0


# ---------------- re-grasp detection ----------------

def test_re_grasp_flag_when_hand_returns_after_release():
    cfg = AssociationConfig(min_persistence=2, bind_radius=60.0, frame_height=480)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "bottle", 120, 110)
    # establish held
    assoc.update([p], [bottle], timestamp=0.0)
    out1 = assoc.update([p], [bottle], timestamp=0.05)
    assert out1[0].hand_near_object is True
    # release: move bottle away from wrist
    bottle_far = _obj(2, "bottle", 300, 300)
    out2 = assoc.update([p], [bottle_far], timestamp=0.1)
    assert out2[0].hand_near_object is False
    # re-grasp: bottle back near wrist
    bottle_back = _obj(2, "bottle", 122, 112)
    out3 = assoc.update([p], [bottle_back], timestamp=0.15)
    assert out3[0].hand_near_object is True
    assert out3[0].person_re_grasped is True


# ---------------- object_low / stationary ----------------

def test_object_low_classification():
    cfg = AssociationConfig(min_persistence=2, frame_height=480, ground_band_ratio=0.6, bind_radius=60.0)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    # establish held (2 frames)
    bottle = _obj(2, "bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)
    # now bottle near ground band (y=400 > 480*0.6=288), still tracked
    bottle_ground = _obj(2, "bottle", 120, 400)
    out = assoc.update([p], [bottle_ground], timestamp=0.1)
    assert len(out) >= 1
    assert out[0].object_low is True


def test_object_stationary_when_not_moving():
    cfg = AssociationConfig(min_persistence=2, stationary_speed=10.0, bind_radius=60.0, frame_height=480)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)  # established
    # same position next frame → zero velocity → stationary
    out = assoc.update([p], [bottle], timestamp=1.05)
    assert out[0].object_stationary is True
    assert out[0].object_moving_down is False


def test_object_moving_down():
    cfg = AssociationConfig(min_persistence=2, stationary_speed=10.0, bind_radius=60.0, frame_height=480)
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110)))
    bottle = _obj(2, "bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)  # established held
    # bottle moved down 200px in 0.1s → vy=2000 px/s (release arc)
    bottle_down = _obj(2, "bottle", 120, 310)
    out = assoc.update([p], [bottle_down], timestamp=0.15)
    assert len(out) >= 1
    assert out[0].object_moving_down is True
    assert out[0].object_stationary is False
