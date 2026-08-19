"""
Correctness tests from the research-backed validation audit.

These tests verify whether the engine handles scenarios that the research
literature (ETRI 2019, AIDM-Strat 2022) flags as critical:

  - Multi-object: one person interacting with TWO objects simultaneously
    (AIDM-Strat uses DeepSORT to track multiple garbage bags; our associator
    must not silently drop the second object).
  - Tracking failure recovery: a bottle that receives a NEW track ID
    mid-release (ByteTrack ID switch on fast motion / occlusion). The
    associator's rebind logic must recover the relationship or the FSM
    will never see the release→ground sequence.

These are NOT synthetic-track injection tests (those already exist in
test_association.py / test_pipeline_integration.py). These directly
exercise the associator's sticky-binding + rebind logic with the
specific multi-object and ID-switch scenarios.
"""
from __future__ import annotations

import pytest

from inference.association.person_object_assoc import (
    AssociationConfig,
    Keypoints,
    PersonObjectAssociator,
    Track,
)


def _person(pid: int, cx: float, cy: float, kp: Keypoints = None) -> Track:
    return Track(pid, "person", (cx, cy), (cx - 40, cy - 80, cx + 40, cy + 80), kp)


def _obj(oid: int, cls: str, cx: float, cy: float, w=25, h=25) -> Track:
    return Track(oid, cls, (cx, cy), (cx - w, cy - h, cx + w, cy + h), None)


def _kp(lw=None, rw=None, tc=(100, 100)) -> Keypoints:
    return Keypoints(left_wrist=lw, right_wrist=rw, torso_center=tc)


# =========================================================================== #
# MULTI-OBJECT: one person + two objects simultaneously
# The audit flagged: persons_with_established may prevent a person from
# binding a second object. This test determines whether that is a real bug.
# =========================================================================== #
def test_one_person_two_objects_both_associated():
    """A person holding a bottle in one hand and a cup in the other should
    establish associations with BOTH objects, not just the first."""
    cfg = AssociationConfig(
        min_persistence=2, bind_radius=60.0, frame_height=480, torso_radius=110.0,
    )
    assoc = PersonObjectAssociator(cfg)
    t = 0.0

    # person with both wrists near different objects
    p = _person(1, 100, 100, _kp(lw=(120, 110), rw=(180, 110), tc=(100, 100)))
    bottle = _obj(10001, "plastic bottle", 120, 110)
    cup = _obj(10002, "cup", 180, 110)

    # frame 1: candidate proximity (not yet established)
    out1 = assoc.update([p], [bottle, cup], timestamp=t)
    assert out1 == [], "first frame should not be established yet"

    # frames 2–4: both should establish (persistence=2 needs 2 held frames each;
    # the cup starts 1 frame after the bottle because it's a new binding)
    t = 0.05
    for _ in range(3):
        assoc.update([p], [bottle, cup], timestamp=t)
        t += 0.05
    # THE QUESTION: does the associator establish BOTH pairs?
    object_ids_in_pairs = {k[1] for k in assoc._pairs.keys()}
    established_count = sum(1 for m in assoc._pairs.values() if m.established)
    print(f"pairs in store: {list(assoc._pairs.keys())} established={established_count}")
    # Document the finding honestly:
    if established_count >= 2:
        print("RESULT: both objects associated — multi-object WORKS")
    elif established_count == 1:
        print("RESULT: only ONE object established — MULTI-OBJECT BUG CONFIRMED")
    else:
        print(f"RESULT: {established_count} established (still accumulating persistence)")

    # ASSERT: the multi-object fix should establish both
    assert established_count >= 2, (
        f"multi-object fix failed: only {established_count} object(s) established, "
        f"expected 2 (bottle on wrist L + cup on wrist R)"
    )


def test_one_person_two_objects_documented_behavior():
    """Companion test that documents the EXPECTED behavior vs the ACTUAL.

    If the multi-object bug exists, this test captures it precisely so the
    limitation is documented, not hidden.
    """
    cfg = AssociationConfig(
        min_persistence=2, bind_radius=60.0, frame_height=480, torso_radius=110.0,
    )
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110), rw=(180, 110), tc=(100, 100)))
    bottle = _obj(10001, "plastic bottle", 120, 110)
    cup = _obj(10002, "cup", 180, 110)

    assoc.update([p], [bottle, cup], timestamp=0.0)
    for t in [0.05, 0.10, 0.15]:
        assoc.update([p], [bottle, cup], timestamp=t)

    # With the multi-object fix, a person can hold two objects (one per wrist).
    pairs_for_person_1 = [k for k in assoc._pairs.keys() if k[0] == 1]
    established = sum(1 for k in pairs_for_person_1 if assoc._pairs[k].established)
    print(f"pairs for person 1: {len(pairs_for_person_1)} established={established}")
    assert len(pairs_for_person_1) == 2, (
        f"multi-object: expected 2 pairs, got {len(pairs_for_person_1)}"
    )
    assert established == 2, (
        f"multi-object: expected 2 established, got {established}"
    )


# =========================================================================== #
# TRACKING FAILURE RECOVERY: bottle gets a NEW track ID mid-release
# ByteTrack can switch IDs on fast motion / occlusion. The associator's
# rebind logic (reassoc_window) must recover the relationship, or the FSM
# will never see release→ground for the bottle.
# =========================================================================== #
def test_id_switch_mid_release_rebind():
    """Person holds bottle (id 10001) → releases → bottle ID switches to 10005
    mid-flight (ByteTrack fragmentation). The associator should rebind the
    pair to the new object ID within reassoc_window so the FSM continues
    tracking the same physical bottle."""
    cfg = AssociationConfig(
        min_persistence=2, bind_radius=60.0, frame_height=480,
        reassoc_window=2.0, rebind_max_distance=150.0,
    )
    assoc = PersonObjectAssociator(cfg)
    t = 0.0

    # establish holding with bottle id 10001
    p = _person(1, 100, 100, _kp(lw=(120, 110), tc=(100, 100)))
    bottle = _obj(10001, "plastic bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=t)
    t += 0.05
    out = assoc.update([p], [bottle], timestamp=t)
    assert any(o.object_id == 10001 for o in out), "bottle 10001 not established"
    assert (1, 10001) in assoc._pairs

    # release: bottle moves down, then ID switches to 10005 (ByteTrack lost it)
    t += 0.05
    bottle_far = _obj(10001, "plastic bottle", 120, 250)  # still id 10001, moving down
    assoc.update([p], [bottle_far], timestamp=t)
    t += 0.05
    # now the bottle reappears with a NEW id 10005 (ID switch) near where 10001 was
    bottle_new_id = _obj(10005, "plastic bottle", 125, 260)
    out = assoc.update([p], [bottle_new_id], timestamp=t)

    # THE QUESTION: did the rebind recover the relationship?
    has_rebound = (1, 10005) in assoc._pairs
    print(f"rebind to new id 10005: {'YES — recovery works' if has_rebound else 'NO — TRACKING FAILURE BUG'}")
    # the rebind should migrate the memory to (1, 10005)
    assert has_rebound, (
        "ID-switch rebind FAILED: the pair (1,10001) vanished and was not "
        "rebound to (1,10005). The FSM would lose the bottle mid-release."
    )


def test_id_switch_wrong_class_no_rebind():
    """A bottle that switches to a 'cup' ID should NOT rebind (different class)."""
    cfg = AssociationConfig(
        min_persistence=2, bind_radius=60.0, frame_height=480,
        reassoc_window=2.0, rebind_max_distance=150.0,
    )
    assoc = PersonObjectAssociator(cfg)
    p = _person(1, 100, 100, _kp(lw=(120, 110), tc=(100, 100)))
    bottle = _obj(10001, "plastic bottle", 120, 110)
    assoc.update([p], [bottle], timestamp=0.0)
    assoc.update([p], [bottle], timestamp=0.05)
    # release + ID switch to a cup (wrong class)
    assoc.update([p], [_obj(10001, "plastic bottle", 120, 250)], timestamp=0.10)
    cup_new = _obj(10005, "cup", 125, 260)
    out = assoc.update([p], [cup_new], timestamp=0.15)
    assert (1, 10005) not in assoc._pairs or all(
        o.object_class != "cup" for o in out
    ), "should not rebind bottle memory to a cup"
