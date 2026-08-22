"""
End-to-end video upload through the REAL backend API.

Exercises the corrected code path end-to-end:
  real mp4  ->  /api/analysis/upload  ->  background job (VideoFileSource
   ->  YOLO + ByteTrack + MoveNet -> Association -> FSM -> Voting -> Evidence)
  ->  direct DB event + evidence rows  ->  /api/evidence/* playback

No fake detectors, no synthetic tracks, no label-forcing — the AI decides.

This file mirrors the test_backend_api.py harness (in-memory SQLite +
FastAPI TestClient with the get_db dependency overridden) so it runs
without PostgreSQL.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pytest

# IMPORTANT: set DATABASE_URL to in-memory SQLite BEFORE importing the backend,
# exactly as test_backend_api.py does (same poolclass=StaticPool pattern).
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import models
from backend.database import Base, get_db
from backend.main import app
from backend.routers.evidence import EVIDENCE_STORE


def _make_real_probe_mp4(path: str) -> str:
    """Build the @30fps positive probe video (real photograph derived)
    at *path* via the same builder used by scripts/probe_positive_video.py.
    """
    REAL_IMAGE = os.path.join(
        os.path.dirname(__file__), "real_video", "person_bottle2.jpg"
    )
    img = cv2.imread(REAL_IMAGE)
    H, W = img.shape[:2]
    from ultralytics import YOLO

    m = YOLO("yolov8n.pt")
    res = m(img, verbose=False)[0]
    bottle_box = None
    for b in res.boxes:
        if m.names[int(b.cls[0])] == "bottle" and bottle_box is None:
            bottle_box = tuple(map(int, b.xyxy[0].tolist()))
    assert bottle_box is not None, "person_bottle2.jpg must contain a bottle"

    bx1, by1, bx2, by2 = bottle_box
    tpad = 6
    patch = img[max(0, by1 - tpad):min(H, by2 + tpad),
                max(0, bx1 - tpad):min(W, bx2 + tpad)].copy()
    ph, pw = patch.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[max(0, by1 - 4):min(H, by2 + 4), max(0, bx1 - 4):min(W, bx2 + 4)] = 255
    blanked = cv2.inpaint(img.copy(), mask, 7, cv2.INPAINT_TELEA)

    out_W, out_H = 1280, 720
    scale = (out_H - 40) / H
    img_s = cv2.resize(img, (int(W * scale), int(H * scale)))
    blanked_s = cv2.resize(blanked, (int(W * scale), int(H * scale)))
    ph_s, pw_s = max(8, int(ph * scale)), max(8, int(pw * scale))
    patch_s = cv2.resize(patch, (pw_s, ph_s))

    fps = 30.0
    n_frames, hold_end, arc_end = 105, 30, 60
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_W, out_H))
    start_x = 60
    end_x = out_W - img_s.shape[1] - 60
    hand_x = start_x + int((bx1 + tpad) * scale) + pw_s // 2
    hand_y = 20 + int((by1 + tpad) * scale) + ph_s // 2
    ground_cx = min(out_W - pw_s - 10, hand_x + 120)
    ground_cy = min(int(out_H * 0.88), out_H - ph_s // 2 - 2)

    for i in range(n_frames):
        frame = np.full((out_H, out_W, 3), 40, dtype=np.uint8)
        frame[int(out_H * 0.75):, :] = (52, 110, 60)
        t01 = i / (n_frames - 1)
        ix = int(start_x + t01 * (end_x - start_x))
        if i < hold_end:
            h_, w_ = img_s.shape[:2]
            frame[20:20 + h_, ix:ix + w_] = img_s
        else:
            h_, w_ = blanked_s.shape[:2]
            frame[20:20 + h_, ix:ix + w_] = blanked_s
        if hold_end <= i < arc_end:
            u = (i - hold_end) / float(arc_end - hold_end - 1)
            cx = int(hand_x + u * (ground_cx - hand_x))
            cy = int(hand_y + (u ** 1.5) * (ground_cy - hand_y))
            tx, ty = cx - pw_s // 2, cy - ph_s // 2
            tx = min(max(0, tx), out_W - pw_s); ty = min(max(0, ty), out_H - ph_s)
            frame[ty:ty + ph_s, tx:tx + pw_s] = patch_s
        elif i >= arc_end:
            tx, ty = ground_cx - pw_s // 2, ground_cy - ph_s // 2
            frame[ty:ty + ph_s, tx:tx + pw_s] = patch_s
        vw.write(frame)
    vw.release()
    return path


@pytest.fixture()
def _analysis_client(tmp_path):
    """TestClient harness for the analysis upload job (separate in-memory DB)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    SessionLocal = sessionmaker(
        bind=engine, autocommit=False, autoflush=False, future=True
    )
    Base.metadata.create_all(bind=engine)

    def _override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    # Background task also reads get_session_factory — redirect it too
    from backend import database

    old_factory = database.get_session_factory if hasattr(database, "get_session_factory") else None
    database.get_session_factory = lambda: SessionLocal  # type: ignore[assignment]

    with TestClient(app) as c:
        yield c

    # teardown
    app.dependency_overrides.pop(get_db, None)
    if old_factory is not None:
        database.get_session_factory = old_factory  # type: ignore[assignment]


def test_upload_real_video_end_to_end_through_api(_analysis_client, tmp_path):
    """Upload a REAL mp4 derived from the real photograph through the
    REAL /api/analysis backend and verify the production pipeline ran the
    right way end-to-end — SAME pipeline as live cameras.

    This test asserts what IS verifiable from a real upload:
      * the uploaded file is accepted and validated (201)
      * a VideoAnalysisJob row is created
      * VideoFileSource iterates EVERY frame (not the old 1-frame bug)
      * the real AI pipeline does run (persons/objects detected, timeline
        populated, report written) and the AI independently decides
        confirmed vs not — no forcing either way
      * if the AI does confirm, the evidence lifecycle is executed in
        the correct order and the evidence is retrievable via the API
    """
    # Build a real injected video from the real photograph right now
    vid_path = str(tmp_path / "probe_upload.mp4")
    _make_real_probe_mp4(vid_path)
    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert total == 105

    # Upload via the real backend
    with open(vid_path, "rb") as f:
        resp = _analysis_client.post(
            "/api/analysis/upload",
            files={"file": ("probe_upload.mp4", f, "video/mp4")},
        )
    assert resp.status_code == 201, resp.text
    job = resp.json()
    job_id = job["id"]
    assert job["original_filename"] == "probe_upload.mp4"
    assert job["status"] in ("queued", "processing", "completed", "failed")

    # Wait for the background job to finish (pipeline: YOLO+ByteTrack+MoveNet
    # across 105 frames at ~30fps analysis — takes ~20-40s with real TF/ULY).
    deadline = time.time() + 120.0
    job = None
    while time.time() < deadline:
        r = _analysis_client.get(f"/api/analysis/jobs/{job_id}")
        assert r.status_code == 200, r.text
        job = r.json()
        if job["status"] not in ("queued", "processing"):
            break
        time.sleep(0.4)
    assert job is not None
    assert job["status"] == "completed", f"job {job_id} -> {job['status']}: {job.get('error_message')}"

    # CRITICAL REGRESSION: every frame must have been processed (the old
    # iterator bug yielded exactly 1 frame).
    assert job["processed_frames"] == 105, (
        f"processed {job['processed_frames']}/105 frames — iteration is broken"
    )
    assert job["total_frames"] == 105
    # Real AI must have detected persons (and possibly objects) in real frames
    assert job["persons_detected"] >= 1, "AI saw NO persons in a video with a real person"
    assert job["report_json"] is not None, "no diagnostic report written"
    report = json.loads(job["report_json"])
    assert report["processed_frames"] == 105
    assert "diagnosis" in report
    assert report["diagnosis"]["yolo_person"] == "PASS"
    assert "timeline" in report
    assert "confirmed_events" in report  # the AI independently decided
    assert "persisted_event_ids" in report

    # If any events were confirmed, the full evidence lifecycle MUST have
    # happened in the correct order: finalize -> verify -> create event
    # -> store files -> retrievable via /api/evidence/*.
    confirmed = report["confirmed_events"]
    persisted = report.get("persisted_event_ids", [])
    if int(confirmed) > 0:
        assert len(persisted) == int(confirmed), (
            f"AI confirmed {confirmed} but only {len(persisted)} DB events persisted"
        )
        for eid in persisted:
            r_ev = _analysis_client.get(f"/api/evidence/{eid}")
            assert r_ev.status_code == 200, r_ev.text
            rows = r_ev.json()
            assert len(rows) >= 1
            row = rows[0]
            assert row["image_path"], "snapshot path missing"
            # Snapshot must be fetchable
            snap = _analysis_client.get(f"/api/evidence/file/{row['image_path']}")
            assert snap.status_code == 200, snap.text
            assert int(snap.headers.get("content-length") or len(snap.content)) > 100
            # If video was written, it must also be fetchable
            if row.get("video_path"):
                vid_resp = _analysis_client.get(f"/api/evidence/file/{row['video_path']}")
                assert vid_resp.status_code == 200, vid_resp.text
                assert int(vid_resp.headers.get("content-length") or len(vid_resp.content)) > 100
            # Convenience routes
            r_snap = _analysis_client.get(f"/api/evidence/event/{eid}/snapshot")
            assert r_snap.status_code == 200, r_snap.text
