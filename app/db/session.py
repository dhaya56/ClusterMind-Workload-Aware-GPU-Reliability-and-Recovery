from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.health import DatabaseHealth


class DatabaseRuntimeError(RuntimeError):
    """Raised when the database runtime cannot safely complete an operation."""


class DatabaseRuntime:
    """Own the SQLAlchemy engine, connection pool, and session factory."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = self._create_engine(settings)
        self._session_factory = sessionmaker(
            bind=self._engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )
        self._disposed = False

    @staticmethod
    def _create_engine(settings: Settings) -> Engine:
        return create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout_seconds,
            pool_recycle=settings.database_pool_recycle_seconds,
            connect_args={
                "connect_timeout": settings.database_connect_timeout_seconds,
                "application_name": settings.otel_service_name,
            },
            hide_parameters=True,
        )

    @property
    def engine(self) -> Engine:
        self._ensure_active()
        return self._engine

    @property
    def safe_url(self) -> str:
        return make_url(self._settings.database_url).render_as_string(
            hide_password=True
        )

    @property
    def pool_status(self) -> str:
        self._ensure_active()
        return self._engine.pool.status()

    def _ensure_active(self) -> None:
        if self._disposed:
            raise DatabaseRuntimeError("Database runtime has been disposed")

    def new_session(self) -> Session:
        """Return one new non-shared Session for the current unit of work."""

        self._ensure_active()
        return self._session_factory()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Commit on success, rollback on failure, and always close the Session."""

        session = self.new_session()
        try:
            with session.begin():
                yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read_session(self) -> Iterator[Session]:
        """Provide a read-oriented Session and roll back any accidental changes."""

        session = self.new_session()
        try:
            yield session
        finally:
            if session.in_transaction():
                session.rollback()
            session.close()

    def dependency(self) -> Generator[Session, None, None]:
        """FastAPI dependency that never commits implicitly."""

        with self.read_session() as session:
            yield session

    def health(self) -> DatabaseHealth:
        """Execute a bounded readiness query without leaking connection details."""

        self._ensure_active()
        started_at = perf_counter()
        try:
            with self._engine.connect() as connection:
                row = connection.execute(
                    text(
                        "SELECT current_database(), "
                        "current_setting('server_version')"
                    )
                ).one()
            return DatabaseHealth(
                status="healthy",
                latency_ms=round((perf_counter() - started_at) * 1000.0, 3),
                database=str(row[0]),
                server_version=str(row[1]),
            )
        except SQLAlchemyError as exc:
            return DatabaseHealth(
                status="unavailable",
                latency_ms=round((perf_counter() - started_at) * 1000.0, 3),
                error_type=type(exc).__name__,
            )

    def execute_scalar(self, statement: str) -> Any:
        """Execute a trusted internal scalar query for diagnostics and tests."""

        if not statement.strip():
            raise ValueError("statement cannot be blank")
        self._ensure_active()
        try:
            with self._engine.connect() as connection:
                return connection.execute(text(statement)).scalar_one()
        except SQLAlchemyError as exc:
            raise DatabaseRuntimeError("Database scalar query failed") from exc

    def dispose(self) -> None:
        """Close pooled connections. Repeated disposal is safe."""

        if self._disposed:
            return
        self._engine.dispose()
        self._disposed = True

    def __enter__(self) -> "DatabaseRuntime":
        self._ensure_active()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.dispose()
