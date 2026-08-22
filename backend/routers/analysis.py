"""Video Analysis Router — upload video files and run the real production AI pipeline."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import SessionLocal, get_db

router = APIRouter(prefix="/analysis", tags=["analysis"])
log = logging.getLogger("ai_littering.analysis")

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploaded_videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _run_video_analysis_job(job_id: int):
    """Background task running the REAL production AI pipeline on the uploaded video."""
    from backend.database import get_db
    # Use the configured get_db generator (or dependency override if testing)
    db_gen = get_db()
    db: Session = next(db_gen)
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        try:
            next(db_gen, None)
        except Exception:
            pass
        return

    try:
        job.status = "processing"
        db.commit()

        # Import AI pipeline components
        from inference.capture.camera_source import VideoFileSource
        from inference.detection.yolo_detector import YoloDetector
        from inference.pose.movenet_pose import MovenetPose
        from inference.tracking.bytetrack_tracker import BytetrackTracker
        from inference.pipeline import InferencePipeline, PipelineConfig
        from scripts.run_pipeline import build_tracks_real

        # Ensure Camera record exists for "Uploaded Video"
        cam = db.query(models.Camera).filter(models.Camera.name == f"Video: {job.original_filename}").first()
        if not cam:
            cam = models.Camera(
                name=f"Video: {job.original_filename}",
                location="Uploaded File Analysis",
                status="active"
            )
            db.add(cam)
            db.commit()
            db.refresh(cam)

        source = VideoFileSource(job.file_path)
        if not source.open():
            job.status = "failed"
            job.error_message = f"Could not open video file {job.file_path}"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.close()
            return

        job.total_frames = source.total_frames
        job.fps = source.fps
        job.duration_sec = source.duration_seconds
        db.commit()

        # Load models
        detector = YoloDetector()
        detector.load()
        tracker = BytetrackTracker()
        tracker.load()
        movenet = MovenetPose()
        movenet.load()

        pipeline_cfg = PipelineConfig(
            buffer_seconds=6.0,
            analysis_fps=10.0,
            camera_id=str(cam.id),
            # The job runs INSIDE the backend process and persists events +
            # evidence directly to the DB below (finalize -> verify -> create
            # event -> store files). An HTTP self-call would be fragile and
            # is unnecessary here. Live-camera mode keeps the HTTP path.
            post_backend_url=None,
        )
        pipe = InferencePipeline(pipeline_cfg)

        t_start = time.time()
        frame_idx = 0
        detected_persons_set = set()
        detected_objects_set = set()
        history_timeline = []

        for pkt in source:
            tracked = detector.track(pkt.frame, persist=True)
            persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker, frame_idx)

            for p in persons:
                detected_persons_set.add(p.track_id)
            for o in objects:
                detected_objects_set.add(f"{o.track_id}:{o.class_name}")

            events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)

            # Record timeline progression
            current_fsm_state = "UNKNOWN"
            for fsm in pipe._fsms.values():
                if fsm.state.name != "UNKNOWN":
                    current_fsm_state = fsm.state.name
                    break

            if current_fsm_state != "UNKNOWN" and (not history_timeline or history_timeline[-1]["state"] != current_fsm_state):
                history_timeline.append({
                    "timestamp": round(pkt.timestamp - (source._start_ts or pkt.timestamp), 2),
                    "frame": frame_idx,
                    "state": current_fsm_state
                })

            frame_idx += 1
            if frame_idx % 15 == 0:
                job.processed_frames = frame_idx
                job.persons_detected = len(detected_persons_set)
                job.objects_detected = len(detected_objects_set)
                job.events_count = len(pipe.events)
                elapsed = max(1e-6, time.time() - t_start)
                job.processing_fps = round(frame_idx / elapsed, 1)
                db.commit()

        source.release()

        # ------------------------------------------------------------------
        # Persist confirmed candidates: finalize -> verify -> create event
        # -> store evidence files -> (retrievable via /api/evidence/*).
        # The pipeline already finalized + verified each artifact's files
        # during the frame loop; pipe.finalized_artifacts holds only the
        # artifacts whose snapshot/video exist and are non-empty.
        # ------------------------------------------------------------------
        from backend.routers.evidence import EVIDENCE_STORE

        created_event_ids = []
        for ev in pipe.events:
            art = pipe.finalized_artifacts.get(ev.event_id)
            if art is None:
                log.warning(
                    "Job %s: event %s has no verified evidence artifact — "
                    "skipping DB persistence", job.id, ev.event_id
                )
                continue
            snap_path = art.snapshot_path
            vid_path = art.video_path
            snap_ok = snap_path and os.path.exists(snap_path) and os.path.getsize(snap_path) > 0
            vid_ok = vid_path and os.path.exists(vid_path) and os.path.getsize(vid_path) > 0
            if not snap_ok:
                log.error("Job %s: snapshot missing/empty for %s", job.id, ev.event_id)
                continue

            # 1) create the backend Event row
            db_event = models.Event(
                camera_id=cam.id,
                person_track_id=str(ev.person_track_id),
                object_track_id=str(ev.object_track_id),
                object_type=ev.object_type,
                confidence=float(ev.confidence),
                timestamp=datetime.now(timezone.utc),
                status="confirmed",
            )
            db.add(db_event)
            db.commit()
            db.refresh(db_event)
            created_event_ids.append(db_event.id)

            # 2) copy evidence files into the backend evidence store and
            #    create the Evidence row so the dashboard can play them.
            target_dir = EVIDENCE_STORE / str(db_event.id)
            target_dir.mkdir(parents=True, exist_ok=True)
            rel_snapshot = rel_video = None
            if snap_ok:
                dst_snap = target_dir / f"snapshot_{db_event.id}.jpg"
                shutil.copyfile(snap_path, dst_snap)
                rel_snapshot = str(dst_snap.relative_to(EVIDENCE_STORE))
            if vid_ok:
                dst_vid = target_dir / f"evidence_{db_event.id}.mp4"
                shutil.copyfile(vid_path, dst_vid)
                rel_video = str(dst_vid.relative_to(EVIDENCE_STORE))
            db_evidence = models.Evidence(
                event_id=db_event.id,
                image_path=rel_snapshot,
                video_path=rel_video,
                duration_sec=art.duration_seconds,
            )
            db.add(db_evidence)
            db.commit()
            log.info(
                "Job %s: persisted event #%d (%s) with evidence%s",
                job.id, db_event.id, ev.object_type,
                " (snapshot+video)" if (snap_ok and vid_ok) else " (snapshot only)",
            )

        # Finalize
        elapsed = max(1e-6, time.time() - t_start)
        job.processed_frames = frame_idx
        job.processing_fps = round(frame_idx / elapsed, 1)
        job.persons_detected = len(detected_persons_set)
        job.objects_detected = len(detected_objects_set)
        job.events_count = len(pipe.events)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)

        # Build diagnostic report
        report = {
            "source": job.original_filename,
            "duration_sec": round(source.duration_seconds, 2),
            "total_frames": source.total_frames,
            "processed_frames": frame_idx,
            "processing_fps": job.processing_fps,
            "persons_count": len(detected_persons_set),
            "objects_count": len(detected_objects_set),
            "confirmed_events": len(pipe.events),
            "persisted_event_ids": created_event_ids,
            "timeline": history_timeline,
            "diagnosis": {
                "yolo_person": "PASS" if detected_persons_set else "FAIL",
                "yolo_object": "PASS" if detected_objects_set else "FAIL",
                "tracking": "PASS" if tracker.store_size > 0 else "FAIL",
                "association": "PASS" if pipe._fsms else "FAIL",
                "littering_candidate": "CONFIRMED" if pipe.events else "NO_CANDIDATE_DETECTED"
            }
        }
        job.report_json = json.dumps(report)
        db.commit()
        log.info("Completed analysis job %s with %d events", job.id, len(pipe.events))

    except Exception as e:
        log.exception("Analysis job %s failed: %s", job_id, e)
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass


@router.post("/upload", response_model=schemas.VideoAnalysisJobOut, status_code=status.HTTP_201_CREATED)
async def upload_video_for_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a real video file (.mp4, .avi, .mov, .mkv) for real AI pipeline analysis."""
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: .mp4, .avi, .mov, .mkv, .webm"
        )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp_str}_{file.filename or 'video.mp4'}"
    target_path = UPLOAD_DIR / safe_name

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    target_path.write_bytes(content)

    job = models.VideoAnalysisJob(
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_path=str(target_path),
        status="queued",
        processed_frames=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Launch real analysis job in background
    background_tasks.add_task(_run_video_analysis_job, job.id)

    return job


@router.get("/jobs", response_model=schemas.VideoAnalysisJobListOut)
def list_analysis_jobs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all video analysis jobs with pagination."""
    q = db.query(models.VideoAnalysisJob).order_by(models.VideoAnalysisJob.id.desc())
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return schemas.VideoAnalysisJobListOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=schemas.VideoAnalysisJobOut)
def get_analysis_job(job_id: int, db: Session = Depends(get_db)):
    """Retrieve details, progress, and diagnostic report for a specific analysis job."""
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return job
