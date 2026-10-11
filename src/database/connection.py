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


try:
    import streamlit as st
    _cache_resource = st.cache_resource
except Exception:
    def _cache_resource(func):
        return func


@_cache_resource
def _init_engine(url: str):
    """Initializes and caches the SQLAlchemy engine."""
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    # Automatically create tables if they don't exist
    try:
        from src.database.base import Base
        import src.database.models  # Ensure all models are imported
        Base.metadata.create_all(bind=engine)
        logger.info("SQLAlchemy Base metadata tables verified/created successfully.")
    except Exception as tbl_err:
        logger.warning("Could not auto-create Base metadata tables: %s", tbl_err)
    return engine


_last_db_error: Optional[str] = None


def get_engine():
    """Returns cached SQLAlchemy engine instance with pool_pre_ping enabled."""
    global _engine, _last_db_error
    try:
        url = settings.database_url
        if not url:
            _last_db_error = "DATABASE_URL is not set."
            return None
        _engine = _init_engine(url)
        return _engine
    except Exception as e:
        _last_db_error = str(e)
        logger.warning("Could not initialize PostgreSQL engine: %s", e)
        return None


def get_session_factory():
    """Returns singleton sessionmaker instance."""
    global _SessionLocal
    if _SessionLocal is None:
        eng = get_engine()
        if eng is not None:
            _SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, expire_on_commit=False)
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
    Returns health dictionary: {"connected": bool, "message": str, "error": Optional[str]}.
    """
    global _last_db_error
    try:
        eng = get_engine()
        if eng is None:
            err = _last_db_error or "PostgreSQL engine not configured or offline."
            return {"connected": False, "message": "PostgreSQL engine offline", "error": err}

        with eng.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            if result == 1:
                _last_db_error = None
                return {"connected": True, "message": "PostgreSQL Connected", "error": None}
    except Exception as e:
        _last_db_error = str(e)
        logger.warning("PostgreSQL health check failed: %s", e)
        return {"connected": False, "message": "PostgreSQL Unavailable", "error": str(e)}

    return {"connected": False, "message": "PostgreSQL Unavailable", "error": _last_db_error or "Connection check failed"}
