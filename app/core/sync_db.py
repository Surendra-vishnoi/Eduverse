"""
Shared synchronous SQLAlchemy engine for non-async operations.

Used by: vector_store.py, tools.py, memory.py
All share one pooled engine instead of creating separate ones.
"""

from sqlalchemy import create_engine

from app.core.config import settings

_sync_engine = None


def get_sync_engine():
    """Get or create the shared sync SQLAlchemy engine."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.PG_SYNC_URL,
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=5,
        )
    return _sync_engine
