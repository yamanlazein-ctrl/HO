"""Statistics router — aggregate metrics over littering events."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("", response_model=schemas.StatisticsOut)
def get_statistics(db: Session = Depends(get_db)) -> schemas.StatisticsOut:
    """Return aggregate littering statistics.

    - ``total_events``: all confirmed/reported events.
    - ``events_today``: events created since 00:00 UTC today.
    - ``per_object_type``: count of events grouped by ``object_type``.
    - ``avg_confidence``: mean confidence across all events.
    """
    total_events: int = db.query(models.Event).count()

    start_of_today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    events_today: int = (
        db.query(models.Event)
        .filter(models.Event.timestamp >= start_of_today)
        .count()
    )

    per_object_rows = (
        db.query(models.Event.object_type, func.count(models.Event.id))
        .group_by(models.Event.object_type)
        .all()
    )
    per_object_type = {obj_type: int(count) for obj_type, count in per_object_rows}

    avg_confidence: float = (
        db.query(func.avg(models.Event.confidence)).scalar() or 0.0
    )

    return schemas.StatisticsOut(
        total_events=total_events,
        events_today=events_today,
        per_object_type=per_object_type,
        avg_confidence=round(float(avg_confidence), 4),
    )
