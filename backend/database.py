"""Database configuration for the AI Littering Detection backend.

Provides a lazily-created SQLAlchemy 2.0 engine, a session factory, a declarative
``Base`` class, a FastAPI ``get_db`` dependency, and a ``create_all`` helper that
imports the models and creates every table.

The engine is created lazily (on first attribute access) so that simply importing
this module — and therefore ``backend.main`` — never touches the database. This
keeps the package importable in environments without a running PostgreSQL server.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql://litter:litter@localhost:5432/littering"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


# The engine is created lazily so importing the package never connects.
# ``connect_args`` / ``pool_pre_ping`` keep things friendly for dev Postgres.
_engine = None
_SessionLocal: "sessionmaker | None" = None


def _build_engine():
    """Construct and cache the engine + session factory on first use."""
    global _engine, _SessionLocal
    if _engine is None:
        # SQLite (used in some tests) needs check_same_thread=False.
        connect_args = (
            {"check_same_thread": False}
            if DATABASE_URL.startswith("sqlite")
            else {}
        )
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args=connect_args,
            future=True,
        )
        _SessionLocal = sessionmaker(
            bind=_engine,
            autocommit=False,
            autoflush=False,
            class_=Session,
            future=True,
        )
    return _engine


@property
def _lazy_engine(self):  # pragma: no cover - unused, placeholder for clarity
    raise RuntimeError("Use get_engine() instead")


def get_engine():
    """Return the cached engine, creating it on first call."""
    return _build_engine()


def get_session_factory() -> sessionmaker:
    """Return the cached session factory, creating the engine if needed."""
    _build_engine()
    assert _SessionLocal is not None  # noqa: S101 - set by _build_engine
    return _SessionLocal


# Backwards-compatible module-level attributes that trigger lazy creation.
def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def create_all() -> None:
    """Create all tables registered on ``Base``.

    Imports the models package so that every table is mapped on ``Base.metadata``
    before issuing ``CREATE TABLE``. Safe to call at application startup.
    """
    # Import here to avoid a circular import at module load time.
    from backend import models  # noqa: F401 - registers tables on Base

    Base.metadata.create_all(bind=get_engine())
