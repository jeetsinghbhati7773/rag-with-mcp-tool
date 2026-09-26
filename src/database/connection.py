from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Dict, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    """Returns singleton SQLAlchemy engine instance with pool_pre_ping enabled."""
    global _engine
    if _engine is None:
        try:
            url = settings.database_url
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )
            logger.info("SQLAlchemy engine initialized for PostgreSQL database.")
            # Automatically create tables if they don't exist
            try:
                from src.database.base import Base
                import src.database.models  # Ensure all models are imported
                Base.metadata.create_all(bind=_engine)
                logger.info("SQLAlchemy Base metadata tables verified/created successfully.")
            except Exception as tbl_err:
                logger.warning("Could not auto-create Base metadata tables: %s", tbl_err)
        except Exception as e:
            logger.warning("Could not initialize PostgreSQL engine: %s", e)
            _engine = None
    return _engine


def get_session_factory():
    """Returns singleton sessionmaker instance."""
    global _SessionLocal
    if _SessionLocal is None:
        eng = get_engine()
        if eng is not None:
            _SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    return _SessionLocal


def SessionLocal() -> Optional[Session]:
    """Factory helper returning a new database session."""
    factory = get_session_factory()
    return factory() if factory else None


@contextmanager
def get_db() -> Generator[Optional[Session], None, None]:
    """Context manager yielding a database session with automatic cleanup and rollback on failure."""
    factory = get_session_factory()
    if factory is None:
        yield None
        return

    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error("Database transaction rolled back due to error: %s", e)
        raise
    finally:
        session.close()


def database_health_check() -> Dict[str, Any]:
    """
    Executes database health check query `SELECT 1` on PostgreSQL.
    Returns health dictionary: {"connected": bool, "message": str}.
    """
    try:
        eng = get_engine()
        if eng is None:
            return {"connected": False, "message": "PostgreSQL engine not configured or offline."}

        with eng.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                return {"connected": True, "message": "PostgreSQL Connected"}
    except Exception as e:
        logger.debug("PostgreSQL health check failed: %s", e)

    return {"connected": False, "message": "⚠️ PostgreSQL Unavailable (Check credentials & database status)"}
