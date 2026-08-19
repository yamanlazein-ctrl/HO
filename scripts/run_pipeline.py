#!/usr/bin/env python3
"""
Run the live inference pipeline against the iPhone camera (Camo/Iriun).

Usage:
    python scripts/run_pipeline.py --source camo --device 0 --buffer 6
    python scripts/run_pipeline.py --source file --video path/clip.mp4
    python scripts/run_pipeline.py --source camo --post-backend http://localhost:8000/api/events

This script is the live entry point. It wires the real CV components
(YOLO, ByteTrack, MoveNet) which require the heavy deps installed on
the laptop — not in the sandbox. The core logic (buffer, association,
FSM, voting, evidence) is already unit-tested without these.

Note on FPS: capture runs at full camera FPS; the heavy analysis runs
at --analysis-fps (default 10). Evidence clips are assembled from the
full-FPS buffer, so they stay smooth even when analysis is throttled.
"""

from __future__ import annotations

import argparse
import sys
import time

# allow running from repo root
sys.path.insert(0, ".")

from inference.capture.camera_source import CameraSource, VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig


def build_tracks_real(frame, tracked, movenet, tracker, frame_index):
    """
    REAL detection→tracking→pose→Track adapter.

    ``tracked`` is the output of ``YoloDetector.track()`` — a list of
    TrackedDetection objects carrying STABLE ByteTrack ids (persist=True
    keeps the tracker state between calls, so the same physical entity
    keeps the same id across frames).

    Steps:
      1. record tracked detections into the TrackStore (history),
      2. run MoveNet pose ONLY on person crops (lazy — biggest CPU save),
      3. convert to namespaced Track objects for the association engine.

    This replaces the old build_tracks() which assigned fake per-frame
    ids (i+1) and broke the entire temporal layer.
    """
    # 1) record history
    tracker.update(tracked, frame_index)

    # 2) person crops for pose (lazy: only persons, only this frame)
    person_tracked = [t for t in tracked if t.is_person]
    person_bboxes = [t.bbox for t in person_tracked]
    pose_results = movenet.estimate(frame, person_bboxes) if person_tracked else []

    # map namespaced person id → keypoints
    kp_by_ns = {}
    for td, pr in zip(person_tracked, pose_results):
        ns_id = tracker.namespace(td.track_id, is_person=True)
        kp_by_ns[ns_id] = pr.keypoints if pr else None

    # 3) convert to Track objects (namespaced ids, stable across frames)
    persons, objects = tracker.to_tracks(tracked, keypoints_by_person_ns=kp_by_ns)
    return persons, objects


def build_tracks(frame, detections, yolo, movenet, tracker_ns, namespace_offset):
    """DEPRECATED shim — kept only for backwards compatibility with old
    call sites. The live pipeline now uses build_tracks_real() with
    detector.track() output. Do NOT use this in production: it assigned
    fake per-frame ids and was the root cause of the audit's P0 finding.
    """
    raise RuntimeError(
        "build_tracks() is deprecated — it assigned fake per-frame ids. "
        "Use build_tracks_real() with YoloDetector.track() output instead."
    )


def main():
    ap = argparse.ArgumentParser(description="Run AI Littering Detection pipeline")
    ap.add_argument("--source", choices=["camo", "file"], default="camo")
    ap.add_argument("--device", type=int, default=0, help="OpenCV device index (Camo/Iriun)")
    ap.add_argument("--video", type=str, default="", help="path for --source file")
    ap.add_argument("--buffer", type=float, default=6.0, help="circular buffer window (s)")
    ap.add_argument("--analysis-fps", type=float, default=10.0)
    ap.add_argument("--pre", type=float, default=3.0)
    ap.add_argument("--post", type=float, default=3.0)
    ap.add_argument("--camera-id", type=str, default="cam-01")
    ap.add_argument("--post-backend", type=str, default="", help="FastAPI URL to POST events")
    ap.add_argument("--show", action="store_true", help="display live frame (dev)")
    args = ap.parse_args()

    cfg = PipelineConfig(
        buffer_seconds=args.buffer,
        analysis_fps=args.analysis_fps,
        pre_seconds=args.pre,
        post_seconds=args.post,
        camera_id=args.camera_id,
        post_backend_url=args.post_backend or None,
    )
    pipe = InferencePipeline(cfg)

    # source
    if args.source == "camo":
        src = CameraSource(device_index=args.device, target_fps=30)
    else:
        src = VideoFileSource(args.video)
    if not src.open():
        print(f"ERROR: cannot open source ({args.source})", file=sys.stderr)
        sys.exit(1)

    # CV components (lazy-loaded; needs laptop deps)
    detector = YoloDetector()
    detector.load()
    print(f"YOLO loaded. Litter classes: {detector.litter_classes}")
    movenet = MovenetPose()
    movenet.load()
    tracker_ns = BytetrackTracker()
    tracker_ns.load()

    print(f"Running. buffer={args.buffer}s analysis_fps={args.analysis_fps} pre/post={args.pre}/{args.post}s")
    print("Press Ctrl+C to stop.")

    _latest_frame = [None]  # mutable holder so the lambda can see updates

    # Register the frame source with the backend's MJPEG stream router so the
    # dashboard LiveCamera component receives real frames instead of the
    # "WAITING FOR CAMERA" placeholder. This is optional — if the backend
    # isn't importable, the stream simply stays on the placeholder.
    try:
        from backend.routers.stream import register_frame_source
        register_frame_source(lambda: _latest_frame[0])
        print("Frame source registered with backend stream router.")
    except Exception:
        print("Backend stream router not available — dashboard will show 'WAITING FOR CAMERA'.")

    try:
        import cv2  # type: ignore
        has_cv2 = True
    except Exception:
        has_cv2 = False

    frame_count = 0
    t0 = time.time()
    try:
        for pkt in src:
            _latest_frame[0] = pkt.frame  # expose to the backend stream router
            # REAL ByteTrack path: detector.track() returns TrackedDetection
            # objects with stable ids (persist=True keeps tracker state).
            tracked = detector.track(pkt.frame, persist=True)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker_ns, frame_count,
            )
            events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
            for ev in events:
                print(f"[{ev.event_timestamp:.2f}] 🚨 LITTERING CONFIRMED — {ev.object_type} "
                      f"person={ev.person_track_id} object={ev.object_track_id} conf={ev.confidence:.2f}")
            if args.show and has_cv2:
                cv2.imshow("littering", pkt.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
            if frame_count % 100 == 0:
                st = pipe.stats()
                fps = frame_count / max(1e-6, time.time() - t0)
                print(f"[stats] {st} capture_fps={fps:.1f}")
                # push live metrics to the backend /api/status so the dashboard
                # status bar reflects the real AI engine + camera state
                pipe.push_status(pkt.timestamp, capture_fps=fps)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        src.release()
        if args.show and has_cv2:
            cv2.destroyAllWindows()
        print(f"Total confirmed events: {len(pipe.events)}")


if __name__ == "__main__":
    main()
