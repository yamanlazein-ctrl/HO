#!/usr/bin/env python3
"""
Camera Discovery & Validation Service
=====================================

Cross-platform (Windows/Linux) camera enumeration that REALLY probes
each device instead of assuming index 0 is valid.

For each candidate index (0..9) it:
  - Opens the device with cv2.VideoCapture
  - Reads frames and measures actual FPS
  - Checks whether frames are *changing* (frame-diff)
  - Estimates brightness (mean pixel value)
  - Detects BLACK / FROZEN / UNAVAILABLE states

The module imports cleanly even when OpenCV is not installed — cv2 is
imported lazily inside the functions that need it.  This makes the file
safe to import in any environment (CI, sandbox without camera drivers,
etc.) and the actual probing only happens when ``discover_cameras()``
is called.

Usage
-----
    # As a module
    from scripts.camera_discovery import discover_cameras
    cameras = discover_cameras()
    for cam in cameras:
        print(cam)

    # From the command line
    PYTHONPATH=. python3 scripts/camera_discovery.py
    PYTHONPATH=. python3 scripts/camera_discovery.py --max-idx 15 --probe-frames 20
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

__all__ = [
    "CameraStatus",
    "CameraInfo",
    "discover_cameras",
    "print_report",
    "main",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class CameraStatus(str, Enum):
    """High-level status for a probed camera device."""

    LIVE = "LIVE"                # opens, reads frames, frames change
    FROZEN = "FROZEN"            # opens, reads frames, but frames never change
    BLACK = "BLACK"              # opens, reads frames, but frames are pure black
    UNAVAILABLE = "UNAVAILABLE"  # opens but read() fails (busy / no signal)
    NOT_FOUND = "NOT_FOUND"      # device does not even open (index out of range)


@dataclass
class CameraInfo:
    """Result of probing a single camera index."""

    index: int
    status: CameraStatus
    opened: bool = False
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    brightness: Optional[float] = None
    frame_freshness: Optional[bool] = None  # True if frames are changing
    error: Optional[str] = None
    probe_frames: int = 0
    backend: Optional[str] = None

    def resolution_str(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"

    def __str__(self) -> str:
        parts = [
            f"device {self.index:>2}  status={self.status.value:<12}",
            f"resolution={self.resolution_str()}",
        ]
        if self.fps is not None:
            parts.append(f"fps={self.fps:.1f}")
        if self.brightness is not None:
            parts.append(f"brightness={self.brightness:.1f}")
        if self.frame_freshness is not None:
            parts.append(f"fresh={self.frame_freshness}")
        if self.backend:
            parts.append(f"backend={self.backend}")
        if self.error:
            parts.append(f"error={self.error}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_cv2():
    """Lazy import of cv2. Raises ImportError with a clear message if absent."""
    try:
        import cv2  # noqa: F401
        return cv2
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "OpenCV (cv2) is required for camera discovery but is not installed. "
            "Install it with:  pip install opencv-python-headless"
        ) from exc


def _import_numpy():
    """Lazy import of numpy."""
    import numpy as np
    return np


def _open_capture(index: int, cv2):
    """Open a VideoCapture on the given index in a cross-platform way.

    On Windows we try DSHOW and MSMF explicitly so OpenCV does not default
    to FFMPEG for live webcam capture (which produces the warning
    'OpenCV should be configured with libavdevice to open a camera device').
    On Linux we rely on V4L2.
    """
    backends_to_try = []
    if sys.platform == "win32":
        backends_to_try = [
            getattr(cv2, "CAP_DSHOW", None),
            getattr(cv2, "CAP_MSMF", None),
        ]
    elif sys.platform.startswith("linux"):
        backends_to_try = [
            getattr(cv2, "CAP_V4L2", None),
            getattr(cv2, "CAP_GSTREAMER", None),
        ]

    for backend in backends_to_try:
        if backend is None:
            continue
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap, cv2.__dict__.get(
                    "_backend_name_" + str(backend), f"backend-{backend}"
                )
            cap.release()
        except Exception:
            pass

    # Default fallback
    try:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, None
    except Exception:
        pass

    return None, "open failed"


def _backend_name(cv2, cap) -> Optional[str]:
    """Try to determine a human-readable backend name."""
    try:
        bid = int(cap.get(cv2.CAP_PROP_BACKEND))
        # Map common backend IDs to names
        names = {
            getattr(cv2, "CAP_V4L2", -1): "V4L2",
            getattr(cv2, "CAP_V4L", -1): "V4L",
            getattr(cv2, "CAP_MSMF", -1): "MSMF",
            getattr(cv2, "CAP_DSHOW", -1): "DSHOW",
            getattr(cv2, "CAP_GSTREAMER", -1): "GStreamer",
            getattr(cv2, "CAP_FFMPEG", -1): "FFmpeg",
            getattr(cv2, "CAP_ANY", -1): "ANY",
        }
        return names.get(bid, f"backend-{bid}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core discovery
# ---------------------------------------------------------------------------


def _probe_device(
    index: int,
    probe_frames: int,
    max_idx: int,
) -> CameraInfo:
    """Probe a single camera index and return a CameraInfo dataclass."""
    cv2 = _import_cv2()
    np = _import_numpy()

    info = CameraInfo(index=index, status=CameraStatus.NOT_FOUND, probe_frames=probe_frames)

    cap, _ = _open_capture(index, cv2)
    if cap is None:
        info.status = CameraStatus.NOT_FOUND
        info.error = "VideoCapture.open() returned False — no device at this index"
        return info

    info.opened = True
    info.backend = _backend_name(cv2, cap)

    # Read resolution from caps (may be 0 if device reports nothing)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception:
        w, h = 0, 0
    info.width = w if w > 0 else None
    info.height = h if h > 0 else None

    # ------------------------------------------------------------------
    # Read probe_frames frames; measure FPS, brightness, frame-diff.
    # ------------------------------------------------------------------
    frames_read = 0
    brightness_sum = 0.0
    frames_are_changing = False
    prev_gray: Optional["np.ndarray"] = None  # type: ignore
    first_frame_ok = False
    read_failed = False

    start = time.monotonic()

    for _ in range(probe_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            read_failed = True
            break

        frames_read += 1
        first_frame_ok = True

        # Brightness (mean of all channels)
        try:
            brightness_sum += float(np.mean(frame))
        except Exception:
            pass

        # Frame-diff to detect changing content
        try:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, prev_gray)
                diff_mean = float(np.mean(diff))
                if diff_mean > 1.5:  # pixels change > ~1.5 on average
                    frames_are_changing = True
            prev_gray = gray
        except Exception:
            pass

    elapsed = time.monotonic() - start

    # Release the capture before computing final status
    cap.release()

    # --- Determine status -------------------------------------------------
    if not first_frame_ok:
        # Opens but can't read even a single frame → busy / unavailable
        info.status = CameraStatus.UNAVAILABLE
        info.error = "Device opens but read() fails — busy or no signal"
        return info

    # Actual FPS from real timing
    if elapsed > 0 and frames_read > 0:
        info.fps = round(frames_read / elapsed, 2)

    if frames_read > 0:
        info.brightness = round(brightness_sum / frames_read, 2)

    info.frame_freshness = frames_are_changing

    # BLACK detection — near-zero brightness across all frames
    avg_brightness = info.brightness or 0.0
    if avg_brightness < 5.0:
        info.status = CameraStatus.BLACK
        return info

    # FROZEN detection — read succeeds, brightness is fine, but frames don't change
    # Only flag frozen if we read at least 3 frames and never saw a change
    if frames_read >= 3 and not frames_are_changing:
        info.status = CameraStatus.FROZEN
        return info

    info.status = CameraStatus.LIVE
    return info


def discover_cameras(
    max_idx: int = 9,
    probe_frames: int = 10,
) -> List[CameraInfo]:
    """Enumerate and validate camera devices.

    Probes indices 0..*max_idx* inclusive.  For each openable device it
    reads *probe_frames* frames and measures real properties.

    Parameters
    ----------
    max_idx : int
        Highest camera index to probe (default 9).
    probe_frames : int
        Number of frames to read per device for FPS / brightness /
        freshness measurement (default 10).

    Returns
    -------
    list[CameraInfo]
        One entry per probed index.  Indices that don't open are
        included with status ``NOT_FOUND`` so the caller can see the
        full scan.
    """
    results: List[CameraInfo] = []
    for index in range(max_idx + 1):
        try:
            info = _probe_device(index, probe_frames=probe_frames, max_idx=max_idx)
        except Exception as exc:
            info = CameraInfo(
                index=index,
                status=CameraStatus.NOT_FOUND,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(info)

        # Early-exit optimisation: if we hit several consecutive NOT_FOUND
        # indices, the remaining ones are almost certainly also absent.
        # We still report them in the summary but stop probing.
        if (
            index >= 2
            and len(results) >= 3
            and all(r.status == CameraStatus.NOT_FOUND for r in results[-3:])
        ):
            # Fill remaining indices as NOT_FOUND without probing
            for remaining in range(index + 1, max_idx + 1):
                results.append(
                    CameraInfo(
                        index=remaining,
                        status=CameraStatus.NOT_FOUND,
                        error="not probed (3 consecutive not-found; scan stopped early)",
                    )
                )
            break

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(cameras: List[CameraInfo], verbose: bool = False) -> None:
    """Print a structured human-readable report of discovered cameras."""

    print("")
    print("=" * 72)
    print("  CAMERA DISCOVERY REPORT")
    print("=" * 72)
    print("")

    found = [c for c in cameras if c.status != CameraStatus.NOT_FOUND]
    live = [c for c in cameras if c.status == CameraStatus.LIVE]
    frozen = [c for c in cameras if c.status == CameraStatus.FROZEN]
    black = [c for c in cameras if c.status == CameraStatus.BLACK]
    unavail = [c for c in cameras if c.status == CameraStatus.UNAVAILABLE]
    not_found = [c for c in cameras if c.status == CameraStatus.NOT_FOUND]

    print(f"  Probed indices : 0..{cameras[-1].index if cameras else 'N/A'}")
    print(f"  Total probed   : {len(cameras)}")
    print(f"  Devices found  : {len(found)}")
    print(f"  LIVE           : {len(live)}")
    print(f"  FROZEN         : {len(frozen)}")
    print(f"  BLACK          : {len(black)}")
    print(f"  UNAVAILABLE    : {len(unavail)}")
    print(f"  NOT_FOUND      : {len(not_found)}")
    print("")

    # Per-device detail
    print("-" * 72)
    print(f"  {'Idx':<4} {'Status':<14} {'Resolution':<12} {'FPS':<7} "
          f"{'Bright':<7} {'Fresh':<6} {'Backend':<10} Detail")
    print("-" * 72)

    for cam in cameras:
        status = cam.status.value
        res = cam.resolution_str()
        fps = f"{cam.fps:.1f}" if cam.fps is not None else "-"
        bright = f"{cam.brightness:.1f}" if cam.brightness is not None else "-"
        fresh = str(cam.frame_freshness) if cam.frame_freshness is not None else "-"
        backend = cam.backend or "-"
        detail = cam.error or ""
        print(f"  {cam.index:<4} {status:<14} {res:<12} {fps:<7} "
              f"{bright:<7} {fresh:<6} {backend:<10} {detail}")

    print("-" * 72)
    print("")

    # Recommendations
    if live:
        print("  RECOMMENDED CAMERA (first LIVE):")
        c = live[0]
        print(f"    index={c.index}  resolution={c.resolution_str()}  "
              f"fps={c.fps}  brightness={c.brightness}")
        print(f"    Set CAMERA_DEVICE_INDEX={c.index} in .env")
    elif found:
        print("  No LIVE camera found.  Devices exist but have issues:")
        for c in found:
            print(f"    device {c.index}: {c.status.value} — {c.error or 'see above'}")
    else:
        print("  No camera devices found.")
        print("  • Connect a USB webcam or iPhone (via Camo/Iriun virtual camera).")
        if sys.platform.startswith("linux"):
            print("  • On Linux, check: ls /dev/video*  and ensure your user is in the 'video' group.")
            print("  • In WSL2 there are no real cameras unless /dev/video* is passed through.")
        elif sys.platform == "win32":
            print("  • On Windows, check Privacy settings → Allow apps to access your camera.")
        print("  • Re-run this script after connecting a camera.")
    print("")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover and validate camera devices (real OpenCV probing)."
    )
    parser.add_argument(
        "--max-idx",
        type=int,
        default=9,
        help="Highest camera index to probe (default 9).",
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=10,
        help="Frames to read per device for measurement (default 10).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output.",
    )
    args = parser.parse_args(argv)

    # Lazy import check — give a clean message if cv2 is missing
    try:
        _import_cv2()
    except ImportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Probing camera indices 0..{args.max_idx} "
          f"({args.probe_frames} frames each)...")

    cameras = discover_cameras(
        max_idx=args.max_idx,
        probe_frames=args.probe_frames,
    )

    print_report(cameras, verbose=args.verbose)

    # Exit code: 0 if at least one LIVE camera, 1 if devices found but none live,
    # 2 if no devices at all or cv2 missing
    live = [c for c in cameras if c.status == CameraStatus.LIVE]
    found = [c for c in cameras if c.status != CameraStatus.NOT_FOUND]
    if live:
        return 0
    if found:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
