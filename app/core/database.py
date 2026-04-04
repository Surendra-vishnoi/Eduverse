"""
Async database engine and session management.

Uses PostgreSQL via asyncpg with direct connection to Supabase
(bypasses the Supavisor pooler). Direct connection supports all
asyncpg features including prepared statements.
"""

import ssl
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.core.config import settings

# ── SSL context for Supabase ─────────────────────────────────────────
# Supabase requires TLS but uses its own CA (not in Python's default trust
# store). asyncpg raises SSLCertVerificationError without these two lines.
# The connection is still fully encrypted — only CA verification is skipped.
# To enable full verify-full mode: download prod-ca-2021.crt from the
# Supabase dashboard and pass cafile= to ssl.create_default_context().
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=8,
    max_overflow=12,
    pool_recycle=300,
    connect_args={
        "ssl": _ssl_context,                       # Supabase: TLS required
        "command_timeout": 180,                    # asyncpg client-side timeout (3 min)
        "server_settings": {
            "statement_timeout": "180000",          # 3 min server-side (Supabase default is low)
        },
    },
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that provides a DB session with auto-commit/rollback."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections on shutdown."""
    await engine.dispose()
