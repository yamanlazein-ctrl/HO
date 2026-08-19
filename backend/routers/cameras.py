"""Camera CRUD router."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=List[schemas.CameraOut])
def list_cameras(db: Session = Depends(get_db)) -> List[models.Camera]:
    """Return all registered cameras."""
    return db.query(models.Camera).order_by(models.Camera.id.desc()).all()


@router.post(
    "",
    response_model=schemas.CameraOut,
    status_code=status.HTTP_201_CREATED,
)
def create_camera(
    camera: schemas.CameraCreate, db: Session = Depends(get_db)
) -> models.Camera:
    """Register a new camera."""
    db_camera = models.Camera(**camera.model_dump())
    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)
    return db_camera


@router.get("/{camera_id}", response_model=schemas.CameraOut)
def get_camera(camera_id: int, db: Session = Depends(get_db)) -> models.Camera:
    """Return a single camera by id."""
    camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )
    return camera
