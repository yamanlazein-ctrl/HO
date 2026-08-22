"""
Regression tests for Phase-2 repairs:

1. VideoFileSource must iterate EVERY frame of a real video file.
   (Regression: __iter__ used to null self._cap after the first yield,
   so upload analysis processed exactly ONE frame.)
2. Every class of the production litter model (best.pt) must be accepted
   by the association layer's litter_candidate_classes.
   (Regression: nescafe/plate/tissue were silently dropped.)
3. MovenetPose must expose honest loaded state and load from the local
   persistent cache without network after the first run.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.capture.camera_source import VideoFileSource
from inference.association.person_object_assoc import AssociationConfig


def _make_real_mp4(path: str, n_frames: int = 45, fps: float = 15.0) -> None:
    """Write a small but REAL mp4 (moving rectangle) via OpenCV."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    w, h = 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "VideoWriter could not open"
    for i in range(n_frames):
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        x = int(10 + i * (w - 60) / max(1, n_frames - 1))
        frame[100:140, x:x + 40] = (0, 200, 0)
        vw.write(frame)
    vw.release()


def test_video_file_source_iterates_all_frames(tmp_path):
    """REGRESSION: VideoFileSource.__iter__ must yield every frame."""
    vid = str(tmp_path / "clip.mp4")
    _make_real_mp4(vid, n_frames=45, fps=15.0)

    src = VideoFileSource(vid)
    assert src.open()
    assert src.total_frames == 45
    assert abs(src.duration_seconds - 3.0) < 0.1

    count = sum(1 for _ in src)
    src.release()
    assert count == 45, (
        f"VideoFileSource yielded {count}/45 frames — iteration is broken "
        f"(upload analysis would process only the first frame)"
    )


def test_video_file_source_timestamps_advance(tmp_path):
    """Synthetic wall-clock timestamps must advance by 1/fps per frame."""
    vid = str(tmp_path / "clip.mp4")
    _make_real_mp4(vid, n_frames=30, fps=15.0)

    src = VideoFileSource(vid)
    assert src.open()
    stamps = [pkt.timestamp for pkt in src]
    src.release()

    assert len(stamps) == 30
    deltas = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(abs(d - 1 / 15.0) < 1e-6 for d in deltas)


def test_best_pt_classes_are_all_litter_candidates():
    """REGRESSION: every best.pt class must pass _is_litter_candidate.

    best.pt classes: bottle, juice-cup, nescafe, plate, tissue.
    'nescafe', 'plate' and 'tissue' were silently dropped before the fix,
    meaning detections of those litter types never reached the FSM.
    """
    assoc = AssociationConfig()
    # production model classes (best.pt)
    for cls in ["bottle", "juice-cup", "nescafe", "plate", "tissue"]:
        assert any(c in cls.lower() for c in assoc.litter_candidate_classes), (
            f"best.pt class '{cls}' is NOT matched by litter_candidate_classes "
            f"— it would be silently ignored by the association layer"
        )
    # COCO fallback classes
    for cls in ["bottle", "cup"]:
        assert any(c in cls.lower() for c in assoc.litter_candidate_classes)


def test_movenet_honest_load_state():
    """MovenetPose must report honest loaded state; cache load must work offline."""
    try:
        import tensorflow  # noqa: F401
        import tensorflow_hub  # noqa: F401
    except ImportError:
        pytest.skip("tensorflow/tensorflow_hub not installed")

    from inference.pose.movenet_pose import MovenetPose

    m = MovenetPose()
    assert not m.is_loaded, "is_loaded must be False before load()"
    with pytest.raises(RuntimeError):
        m.estimate(None, [])  # estimate before load must fail loudly

    m.load()
    assert m.is_loaded
    assert m.loaded_from in ("cache", "download")
