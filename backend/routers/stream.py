"""MJPEG live-camera preview streaming router.

Exposes ``GET /api/cameras/{camera_id}/stream`` — a real MJPEG-over-HTTP stream
using FastAPI ``StreamingResponse`` with the
``multipart/x-mixed-replace; boundary=frame`` content type. Browsers can point
an ``<img>`` tag at this URL directly and see a live feed.

Design — HONEST placeholders, not fakes
---------------------------------------
The real frame source (an iPhone via Camo / a local webcam) requires OpenCV +
a physical camera, which does not exist in the sandbox/CI environment. So the
generator:

* If a frame source has been registered via ``register_frame_source()``
  (the real pipeline — ``run_pipeline.py`` — calls this at startup), each
  yielded JPEG is the latest real BGR frame from the camera.
* If NO frame source is registered, the generator renders a solid-colour
  placeholder JPEG with the ``camera_id`` + "WAITING FOR CAMERA" text drawn on
  it via ``cv2``. This is honest: the route and multipart format are real, but
  the stream visibly says "WAITING FOR CAMERA" until the iPhone is connected.

We never fabricate detection boxes or fake a live feed. ``cv2`` is imported
LAZILY inside the generator so that importing this module (and therefore
``backend.main``) does not require OpenCV to be installed.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend import models
from backend.database import get_db

router = APIRouter(prefix="/cameras", tags=["stream"])

# MJPEG multipart boundary used in the Content-Type header and between frames.
_BOUNDARY = "frame"

# --------------------------------------------------------------------------- #
# Frame-source registration
#
# ``register_frame_source(callable)`` lets the real pipeline push the latest
# BGR frame. The callable must return either a numpy ndarray (BGR HxWx3) or
# ``None`` (no frame available yet). It is invoked once per streamed JPEG.
# --------------------------------------------------------------------------- #
_frame_source: Optional[Callable[[], object]] = None


def register_frame_source(source: Callable[[], object]) -> None:
    """Register a callable that returns the latest BGR frame (ndarray | None).

    Called by the inference pipeline at startup so the stream endpoint serves
    real camera frames instead of the waiting placeholder.
    """
    global _frame_source
    _frame_source = source


def unregister_frame_source() -> None:
    """Remove the registered frame source (revert to the waiting placeholder)."""
    global _frame_source
    _frame_source = None


def get_frame_source() -> Optional[Callable[[], object]]:
    """Return the currently registered frame source, or ``None``."""
    return _frame_source


# --------------------------------------------------------------------------- #
# Placeholder frame rendering (lazy cv2 import)
# --------------------------------------------------------------------------- #
def _render_placeholder_jpeg(camera_id: int) -> bytes:
    """Render a solid-colour JPEG with 'WAITING FOR CAMERA' + camera_id text.

    Uses ``cv2`` (imported lazily so the backend stays importable without it).
    Returns raw JPEG bytes ready to embed in the MJPEG multipart stream.
    """
    import cv2  # lazy — keeps backend importable without OpenCV
    import numpy as np  # lazy for the same reason

    width, height = 640, 360
    # Solid dark blue background.
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 60)  # BGR

    text_main = "WAITING FOR CAMERA"
    text_cam = f"Camera {camera_id}"

    # Main heading
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_main, thickness_main = 1.1, 2
    (tw, th), _ = cv2.getTextSize(text_main, font, scale_main, thickness_main)
    x = (width - tw) // 2
    y = height // 2 - 10
    cv2.putText(frame, text_main, (x, y), font, scale_main, (255, 255, 255), thickness_main, cv2.LINE_AA)

    # Camera id subtitle
    scale_cam, thickness_cam = 0.8, 1
    (tw2, th2), _ = cv2.getTextSize(text_cam, font, scale_cam, thickness_cam)
    x2 = (width - tw2) // 2
    y2 = y + 35
    cv2.putText(frame, text_cam, (x2, y2), font, scale_cam, (180, 180, 180), thickness_cam, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        # Fallback: tiny minimal JPEG if encoding somehow fails.
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\xff\xc9\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11"
            b"\x00\xff\xcc\x00\x06\x00\x01\x01\x01\x00\xff\xd9"
        )
    return buf.tobytes()


def _frame_to_jpeg(frame: object) -> Optional[bytes]:
    """Encode a BGR ndarray frame to JPEG bytes. Returns ``None`` on failure."""
    import cv2  # lazy

    if frame is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:  # pragma: no cover — defensive; frame may be stale/bad
        return None


# --------------------------------------------------------------------------- #
# MJPEG generator
# --------------------------------------------------------------------------- #
def _mjpeg_generator(
    camera_id: int,
    fps_cap: float = 10.0,
    max_frames: Optional[int] = None,
) -> "Generator[bytes, None, None]":
    """Yield successive MJPEG multipart frames as raw bytes.

    Each part is::

        --frame\r\n
        Content-Type: image/jpeg\r\n
        Content-Length: <n>\r\n
        \r\n
        <jpeg bytes>\r\n

    When a real frame source is registered, its latest frame is encoded and
    served. Otherwise the waiting placeholder is served. ``fps_cap`` throttles
    the loop so a disconnected client / slow reader doesn't spin the CPU.

    ``max_frames`` (optional) stops the generator after that many frames,
    which makes the response finite. A browser <img> never sets it (the stream
    runs forever); a test client sets it to a small number so the synchronous
    TestClient can collect the body without hanging on an infinite generator.
    """
    delay = 1.0 / fps_cap if fps_cap > 0 else 0.0
    source = get_frame_source()
    emitted = 0

    while True:
        if max_frames is not None and emitted >= max_frames:
            break

        jpeg: Optional[bytes] = None

        if source is not None:
            try:
                frame = source()
            except Exception:  # pragma: no cover — source raised; degrade to placeholder
                frame = None
            jpeg = _frame_to_jpeg(frame) if frame is not None else None

        if jpeg is None:
            # No real frame source or source returned None -> honest placeholder.
            jpeg = _render_placeholder_jpeg(camera_id)

        part = (
            f"--{_BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode("ascii") + jpeg + b"\r\n"
        yield part
        emitted += 1

        if delay > 0:
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@router.get(
    "/{camera_id}/stream",
    response_class=StreamingResponse,
)
def stream_camera(
    camera_id: int,
    max_frames: Optional[int] = None,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Stream an MJPEG live preview for the given camera.

    Returns a ``StreamingResponse`` with ``multipart/x-mixed-replace`` content
    type and the ``frame`` boundary. If the camera doesn't exist, 404. If no
    frame source is registered (the normal sandbox case), the stream renders a
    "WAITING FOR CAMERA" placeholder JPEG — the route and stream format are
    real, but there is no live camera to show.

    ``max_frames`` (query param, optional) caps the number of frames emitted
    before the stream ends. Browsers never set it (infinite live feed); it
    exists so synchronous test clients can collect a bounded response without
    hanging on an infinite generator.
    """
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    return StreamingResponse(
        _mjpeg_generator(camera_id, max_frames=max_frames),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
