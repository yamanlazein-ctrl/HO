"""Unit tests for CircularFrameBuffer — pure logic, no OpenCV required."""

from __future__ import annotations

import time

import pytest

from inference.capture.circular_buffer import CircularFrameBuffer, BufferedFrame


def _frame(idx: int):
    """A cheap fake frame object with a .shape attribute (mimics ndarray)."""
    class FakeFrame:
        shape = (480, 640, 3)
        def __repr__(self):
            return f"<FakeFrame {idx}>"
    return FakeFrame()


def test_push_and_len():
    buf = CircularFrameBuffer(window_seconds=10.0, max_frames=100)
    assert len(buf) == 0
    t0 = 1000.0
    for i in range(5):
        buf.push(_frame(i), timestamp=t0 + i * 0.1, frame_index=i)
    assert len(buf) == 5
    assert buf.total_pushed == 5


def test_time_based_eviction():
    """Frames older than window_seconds must be dropped on push."""
    buf = CircularFrameBuffer(window_seconds=2.0, max_frames=1000)
    t0 = 2000.0
    # push 10 frames spanning 0..0.9s — all within window
    for i in range(10):
        buf.push(_frame(i), timestamp=t0 + i * 0.1)
    assert len(buf) == 10
    # now push a frame 3s later — old frames should evict (cutoff = t0+3-2 = t0+1)
    buf.push(_frame(99), timestamp=t0 + 3.0)
    # frames with ts < t0+1 are gone; only ts >= t0+1 survive (frame index 10..19 none, plus new)
    assert len(buf) == 1
    assert buf.current_duration() == 0.0  # single frame


def test_window_get_before_event():
    buf = CircularFrameBuffer(window_seconds=10.0, max_frames=1000)
    t0 = 5000.0
    for i in range(20):
        buf.push(_frame(i), timestamp=t0 + i * 0.1, frame_index=i)
    # event at t0 + 2.0s, ask for 1s before
    event_ts = t0 + 2.0
    pre = buf.get_window(before_ts=event_ts, seconds=1.0)
    # frames pushed: index 0..19 at ts t0+0.0 .. t0+1.9
    # window [t0+1.0, t0+2.0] => indices 10..19
    indices = [f.frame_index for f in pre]
    assert indices[0] == 10
    assert indices[-1] == 19
    # timestamps within [event_ts-1, event_ts]
    assert all(event_ts - 1.0 <= f.timestamp <= event_ts + 1e-9 for f in pre)


def test_get_around_event():
    buf = CircularFrameBuffer(window_seconds=20.0, max_frames=2000)
    t0 = 0.0
    for i in range(40):
        buf.push(_frame(i), timestamp=t0 + i * 0.1, frame_index=i)
    event_ts = t0 + 2.0  # frame 20
    window = buf.get_around(event_ts, pre_seconds=1.0, post_seconds=1.0)
    # ts in [1.0, 3.0] => frame_index 10..30
    indices = [f.frame_index for f in window]
    assert indices[0] == 10
    assert indices[-1] == 30
    assert len(indices) == 21


def test_snapshot_is_copy():
    buf = CircularFrameBuffer(window_seconds=10.0, max_frames=100)
    for i in range(5):
        buf.push(_frame(i), timestamp=10.0 + i * 0.1)
    snap = buf.snapshot()
    assert len(snap.frames) == 5
    assert snap.duration_seconds() == pytest.approx(0.4, abs=1e-6)
    # mutating buffer after snapshot must not change the snapshot
    buf.push(_frame(99), timestamp=10.5)
    assert len(snap.frames) == 5  # unchanged


def test_max_frames_ceiling():
    """Even with a huge time window, max_frames caps memory."""
    buf = CircularFrameBuffer(window_seconds=1000.0, max_frames=5)
    for i in range(20):
        buf.push(_frame(i), timestamp=10.0 + i * 0.001)
    assert len(buf) == 5
    # the last 5 should be there (deque drops from left)
    assert buf.snapshot().frames[-1].frame_index == 19


def test_invalid_args():
    with pytest.raises(ValueError):
        CircularFrameBuffer(window_seconds=0)
    with pytest.raises(ValueError):
        CircularFrameBuffer(window_seconds=-1)
    with pytest.raises(ValueError):
        CircularFrameBuffer(window_seconds=10, max_frames=1)


def test_empty_buffer_window():
    buf = CircularFrameBuffer(window_seconds=5.0)
    assert buf.get_window(before_ts=10.0, seconds=2.0) == []
    assert buf.get_around(event_ts=10.0) == []
    assert buf.snapshot().duration_seconds() == 0.0
