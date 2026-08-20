"""Pydantic v2 schemas for request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class CameraBase(BaseModel):
    name: str = Field(..., max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    status: str = Field("active", max_length=64)


class CameraCreate(CameraBase):
    pass


class CameraOut(CameraBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
class EventBase(BaseModel):
    camera_id: int
    person_track_id: Optional[str] = Field(None, max_length=64)
    object_track_id: Optional[str] = Field(None, max_length=64)
    object_type: str = Field(..., max_length=64)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    status: str = Field("confirmed", max_length=64)
    timestamp: Optional[datetime] = None


class EventCreate(EventBase):
    """Payload used by the inference engine to report a confirmed littering event."""

    pass


class EventOut(EventBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class EvidenceOut(BaseModel):
    id: int
    event_id: int
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    duration_sec: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class EvidenceUploadResponse(BaseModel):
    """Returned after a snapshot + video upload."""

    evidence: EvidenceOut
    snapshot_filename: Optional[str] = None
    video_filename: Optional[str] = None


# --------------------------------------------------------------------------- #
# User
# --------------------------------------------------------------------------- #
class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
class StatisticsOut(BaseModel):
    total_events: int = 0
    events_today: int = 0
    per_object_type: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Generic paginated response for events
# --------------------------------------------------------------------------- #
class EventListOut(BaseModel):
    items: List[EventOut]
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Video Analysis Job Schemas
# --------------------------------------------------------------------------- #
class VideoAnalysisJobBase(BaseModel):
    filename: str
    original_filename: str
    status: str = "queued"
    duration_sec: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: int = 0
    fps: Optional[float] = None
    processing_fps: Optional[float] = None
    events_count: int = 0
    persons_detected: int = 0
    objects_detected: int = 0
    report_json: Optional[str] = None
    error_message: Optional[str] = None


class VideoAnalysisJobOut(VideoAnalysisJobBase):
    id: int
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class VideoAnalysisJobListOut(BaseModel):
    items: List[VideoAnalysisJobOut]
    total: int
    limit: int
    offset: int
