"""Live system + AI-engine status router.

Exposes ``GET /api/status`` which the dashboard's top status bar polls. The
status is backed by a module-level singleton that the inference pipeline
updates as it runs via ``set_status(...)``. Until the pipeline pushes real
metrics, the singleton reports a HONEST default state: the AI engine is
**offline** (YOLO model not loaded) and the camera is **offline** (no source
registered). We never fake an "online" engine or a real camera feed.

The helper functions ``set_status()`` / ``get_status()`` are importable by the
pipeline (``run_pipeline.py``) so it can push live metrics from the real
inference loop without coupling itself to FastAPI.

The router itself imports ONLY stdlib + fastapi — no ``ultralytics`` /
``torch`` / ``cv2`` — so the backend stays importable in environments without
those heavy packages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from sqlalchemy.orm import Session

from backend import models
from backend.database import get_db
from fastapi import Depends

router = APIRouter(prefix="/status", tags=["status"])

# --------------------------------------------------------------------------- #
# Module-level status singleton
#
# The pipeline owns the "truth": model-loaded flag, detection FPS, camera FPS,
# buffer fill level, etc. The status router just snapshots whatever the
# pipeline last pushed. Default state is HONEST: engine offline, camera
# offline, all metrics None/zero. ``set_status`` merges partial updates so the
# pipeline can push whichever fields it knows about without resetting others.
# --------------------------------------------------------------------------- #
_lock = Lock()

_DEFAULT_STATUS: Dict[str, Any] = {
    "system_online": True,  # if this endpoint responds, the backend IS up
    "ai_engine": {
        "status": "offline",
        "model_loaded": False,
        "classes": [],
    },
    "camera": {
        "status": "offline",
        "fps": None,
        "resolution": None,
        "source": None,
    },
    "processing": {
        "fps": None,
        "latency_ms": None,
        "analysis_fps": None,
    },
    "buffer": {
        "window_seconds": 0.0,
        "frames_buffered": 0,
        "buffer_duration": 0.0,
    },
    "events_today": 0,
    "active_cameras": 0,
    "updated_at": None,
}

_status: Dict[str, Any] = {k: (v.copy() if isinstance(v, dict) else v) for k, v in _DEFAULT_STATUS.items()}


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Recursively merge ``src`` into ``dst`` in place (one level of nesting)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def get_status() -> Dict[str, Any]:
    """Return a shallow copy of the current live status snapshot.

    Safe to call from any thread; the returned dict is a copy so callers can't
    mutate the singleton indirectly.
    """
    with _lock:
        return {
            k: (v.copy() if isinstance(v, dict) else v)
            for k, v in _status.items()
        }


def set_status(**updates: Any) -> None:
    """Push a partial status update from the inference pipeline.

    Only the supplied keys are merged; everything else is preserved. Example::

        set_status(
            ai_engine={"status": "online", "model_loaded": True,
                       "classes": ["person", "bottle", "can"]},
            camera={"status": "online", "fps": 30.0, "resolution": "1920x1080",
                    "source": "iPhone"},
            processing={"fps": 28.5, "latency_ms": 35.0, "analysis_fps": 30.0},
            buffer={"window_seconds": 20.0, "frames_buffered": 600,
                    "buffer_duration": 20.0},
        )
    """
    with _lock:
        _deep_merge(_status, updates)
        _status["updated_at"] = datetime.now(timezone.utc).isoformat()


def reset_status() -> None:
    """Reset the singleton back to the honest default (engine/camera offline)."""
    with _lock:
        for k, v in _DEFAULT_STATUS.items():
            _status[k] = v.copy() if isinstance(v, dict) else v


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.get("")
def get_system_status(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return the live AI-engine + system status for the dashboard status bar.

    ``system_online`` is always ``true`` here — if this endpoint responds at
    all, the backend process is up. The remaining fields reflect whatever the
    inference pipeline last pushed via ``set_status()``; if the pipeline has
    not registered any metrics yet, the engine and camera are reported as
    ``offline`` honestly.

    ``events_today`` and ``active_cameras`` are queried live from the database
    so they stay correct even when no pipeline is pushing status.
    """
    snapshot = get_status()

    # Live DB-backed counts — always available even without a running pipeline.
    start_of_today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    events_today: int = (
        db.query(models.Event)
        .filter(models.Event.timestamp >= start_of_today)
        .count()
    )
    active_cameras: int = (
        db.query(models.Camera)
        .filter(models.Camera.status == "active")
        .count()
    )

    snapshot["system_online"] = True
    snapshot["events_today"] = events_today
    snapshot["active_cameras"] = active_cameras
    snapshot["updated_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot
