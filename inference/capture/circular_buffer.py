"""
Circular Frame Buffer — 🔴 core contribution.

A time-based ring buffer that always retains the last N seconds of frames
in memory. When a littering event is confirmed, we can assemble the
evidence clip as:  pre_event (T-3s) + event + post_event (T+3s) without
ever having started recording late.

Design notes
------------
* Time-based, not frame-count based. The contract is "keep the last
  ``window_seconds`` seconds", so behavior is stable across FPS changes
  (30fps, 15fps, ...). Internally we cap by a max-frames ceiling to bound
  memory, but the eviction policy is timestamp-driven.
* Each entry stores the frame, its wall-clock timestamp, and the source
  frame index. We do NOT decode/encode here — frames are stored as raw
  numpy arrays (BGR, as OpenCV delivers them).
* Thread-safe: a single producer (capture loop) pushes frames, and the
  evidence manager may snapshot the buffer at event time. A
  ``threading.Lock`` guards the deque.
* Memory-conscious: optional ``jpeg_encode`` flag stores frames as encoded
  JPEG bytes instead of raw arrays (≈10x smaller, ~ms encode cost). For
  evidence assembly we decode back. Default OFF for fidelity; turn ON for
  long-running demos on RAM-constrained laptops.

The buffer is pure logic — no OpenCV import at module load. It accepts
``np.ndarray`` but only checks ``hasattr(frame, 'shape')`` so it imports
cleanly even without numpy installed (tests inject simple stubs).
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple


@dataclass
class BufferedFrame:
    """A single frame in the ring buffer."""

    frame: Any  # np.ndarray (BGR) or JPEG bytes when encoded
    timestamp: float  # wall-clock seconds (time.time())
    frame_index: int  # monotonic source index from the capture loop
    width: int = 0
    height: int = 0
    encoded: bool = False  # True if `frame` is JPEG bytes


@dataclass
class BufferSnapshot:
    """A point-in-time copy of the buffer contents for evidence assembly."""

    frames: List[BufferedFrame] = field(default_factory=list)
    taken_at: float = 0.0

    def time_span(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (first_ts, last_ts) of the snapshot, or (None,None) if empty."""
        if not self.frames:
            return None, None
        return self.frames[0].timestamp, self.frames[-1].timestamp

    def duration_seconds(self) -> float:
        first, last = self.time_span()
        if first is None or last is None:
            return 0.0
        return last - first


class CircularFrameBuffer:
    """
    Time-based ring buffer keeping the last ``window_seconds`` of frames.

    Usage
    -----
    >>> buf = CircularFrameBuffer(window_seconds=6.0)
    >>> buf.push(frame, timestamp=time.time(), frame_index=0)
    >>> snapshot = buf.snapshot()           # copy for evidence
    >>> pre = buf.get_window(before_ts=ev_ts, seconds=3.0)
    """

    def __init__(
        self,
        window_seconds: float = 6.0,
        max_frames: int = 600,
        jpeg_encode: bool = False,
        jpeg_quality: int = 85,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if max_frames < 2:
            raise ValueError("max_frames must be >= 2")
        self.window_seconds = float(window_seconds)
        self.max_frames = int(max_frames)
        self.jpeg_encode = jpeg_encode
        self.jpeg_quality = int(jpeg_quality)

        self._buf: Deque[BufferedFrame] = collections.deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self._total_pushed = 0

    # ------------------------------------------------------------------ #
    # Production side
    # ------------------------------------------------------------------ #
    def push(self, frame: Any, timestamp: Optional[float] = None, frame_index: Optional[int] = None) -> None:
        """Add a frame to the buffer. Timestamp defaults to now."""
        if timestamp is None:
            timestamp = time.time()
        if frame_index is None:
            frame_index = self._total_pushed

        stored = frame
        encoded = False
        w = h = 0
        if self.jpeg_encode and hasattr(frame, "shape"):
            # Lazy import — only when encoding is requested AND numpy/cv2 exist.
            try:
                import cv2  # type: ignore

                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    stored = jpg.tobytes()
                    encoded = True
                    h, w = frame.shape[:2]
            except Exception:
                # Fall back to raw frame if encode fails — never drop a frame.
                stored = frame
        elif hasattr(frame, "shape"):
            try:
                h, w = frame.shape[:2]
            except Exception:
                pass

        entry = BufferedFrame(
            frame=stored,
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            width=int(w),
            height=int(h),
            encoded=encoded,
        )

        with self._lock:
            self._buf.append(entry)
            self._total_pushed += 1
            self._evict_locked(timestamp)

    def _evict_locked(self, now_ts: float) -> None:
        """Drop frames older than window_seconds. Caller must hold the lock."""
        cutoff = now_ts - self.window_seconds
        # deque: pop from left while too old
        while self._buf and self._buf[0].timestamp < cutoff:
            self._buf.popleft()

    # ------------------------------------------------------------------ #
    # Consumption side
    # ------------------------------------------------------------------ #
    def snapshot(self) -> BufferSnapshot:
        """Return a point-in-time copy of all currently buffered frames."""
        with self._lock:
            frames = list(self._buf)
        return BufferSnapshot(frames=frames, taken_at=time.time())

    def get_window(
        self,
        *,
        before_ts: Optional[float] = None,
        after_ts: Optional[float] = None,
        seconds: Optional[float] = None,
    ) -> List[BufferedFrame]:
        """
        Return frames in a time window.

        Common case for evidence: ``get_window(before_ts=event_ts, seconds=3.0)``
        returns the 3 seconds immediately *before* the event timestamp.

        If ``before_ts`` is None, uses the latest available timestamp.
        If ``seconds`` is None, returns the whole buffer.
        """
        with self._lock:
            frames = list(self._buf)
        if not frames:
            return []

        if seconds is None:
            return frames

        if before_ts is None:
            before_ts = frames[-1].timestamp

        start = before_ts - float(seconds)
        return [f for f in frames if start <= f.timestamp <= before_ts]

    def get_around(self, event_ts: float, pre_seconds: float = 3.0, post_seconds: float = 3.0) -> List[BufferedFrame]:
        """
        Return frames within [event_ts - pre, event_ts + post].

        Used to assemble the full evidence clip (pre + event + post).
        Frames are returned in chronological order.
        """
        start = event_ts - pre_seconds
        end = event_ts + post_seconds
        with self._lock:
            frames = [f for f in self._buf if start <= f.timestamp <= end]
        return frames

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def current_duration(self) -> float:
        """Return how many seconds of content are currently in the buffer."""
        with self._lock:
            if len(self._buf) < 2:
                return 0.0
            return self._buf[-1].timestamp - self._buf[0].timestamp

    @property
    def total_pushed(self) -> int:
        return self._total_pushed

    def is_healthy(self) -> bool:
        """True if the buffer has at least ~1s of content (heuristic for startup)."""
        return self.current_duration() >= 1.0
