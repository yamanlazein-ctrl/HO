"""FastAPI application entrypoint for the AI Littering Detection backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import create_all
from backend.routers import cameras, events, evidence, statistics, status, stream

app = FastAPI(
    title="AI Littering Detection API",
    description=(
        "Backend API for the AI-Based CCTV Littering Detection and Evidence "
        "System. Exposes cameras, littering events, evidence uploads, and "
        "aggregate statistics."
    ),
    version="1.0.0",
)

# CORS — allow all origins for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routers (mounted under /api)
# --------------------------------------------------------------------------- #
app.include_router(cameras.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(statistics.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(stream.router, prefix="/api")


# --------------------------------------------------------------------------- #
# Lifespan / startup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on application startup."""
    create_all()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["health"])
def health() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}
