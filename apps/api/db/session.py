from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings


_engine = None
SessionLocal = sessionmaker(
    autoflush=False,
    autocommit=False,
)


def _database_url() -> str:
    url = settings.database_url.strip().strip("'").strip('"')
    if url.startswith("postgresql://"):
        # psycopg 3 is installed in the API environment.
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine

    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it in the deployment environment."
        )

    _engine = create_engine(
        _database_url(),
        pool_pre_ping=True,
    )
    SessionLocal.configure(bind=_engine)
    return _engine


def get_session() -> Session:
    """Create a database session, initializing the engine lazily."""
    _get_engine()
    return SessionLocal()


def get_db() -> Generator[Session, None, None]:
    db = get_session()
    try:
        yield db
    finally:
        db.close()
