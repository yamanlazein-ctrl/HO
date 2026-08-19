"""Event router — list (paginated), get one, create (inference reports)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=schemas.EventListOut)
def list_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.EventListOut:
    """Return a paginated list of littering events (newest first)."""
    query = db.query(models.Event).order_by(models.Event.id.desc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return schemas.EventListOut(
        items=[schemas.EventOut.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{event_id}", response_model=schemas.EventOut)
def get_event(event_id: int, db: Session = Depends(get_db)) -> models.Event:
    """Return a single event by id (including related evidence via schema)."""
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return event


@router.post(
    "",
    response_model=schemas.EventOut,
    status_code=status.HTTP_201_CREATED,
)
def create_event(
    event: schemas.EventCreate, db: Session = Depends(get_db)
) -> models.Event:
    """Create a confirmed littering event.

    Typically called by the inference engine once its voting/state machine
    confirms a littering act. The referenced ``camera_id`` must exist.
    """
    camera = (
        db.query(models.Camera).filter(models.Camera.id == event.camera_id).first()
    )
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera {event.camera_id} does not exist",
        )

    data = event.model_dump(exclude_unset=True)
    if data.get("timestamp") is None:
        data["timestamp"] = datetime.now(timezone.utc)

    db_event = models.Event(**data)
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event
