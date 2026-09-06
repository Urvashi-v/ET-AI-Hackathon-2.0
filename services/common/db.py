"""PostgreSQL access.

One async connection pool per process, opened on startup and closed on
shutdown. Postgres carries four responsibilities in this system:

* the relational record store (documents, jobs, assets, work orders, ...);
* the chunk store with page/offset provenance;
* the **lexical index** -- a real Okapi BM25 built over a custom tokenizer that
  keeps industrial tags intact (see ``services/retrieval/lexical.py``); and
* the **dense index** -- pgvector, once an embedding provider is configured.

Keeping all four in one engine is a deliberate scope decision recorded in
docs/adr/0001-technology-choices.md: it removes two services and one class of
demo-day failure, and nothing in the corpus sizes we target needs more.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from services.common.config import get_settings
from services.common.errors import DependencyUnavailable
from services.common.logging import get_logger

log = get_logger(__name__)

_pool: AsyncConnectionPool | None = None


async def open_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    settings = get_settings()
    _pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn,
        min_size=settings.postgres_pool_min,
        max_size=settings.postgres_pool_max,
        open=False,
        kwargs={"row_factory": dict_row, "autocommit": False},
        name="brain-pg",
    )
    await _pool.open(wait=True, timeout=30)
    log.info(
        "postgres.pool_opened",
        host=settings.postgres_host,
        database=settings.postgres_db,
        min_size=settings.postgres_pool_min,
        max_size=settings.postgres_pool_max,
    )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("postgres.pool_closed")


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise DependencyUnavailable("PostgreSQL pool is not open.")
    return _pool


@asynccontextmanager
async def connection() -> AsyncIterator[psycopg.AsyncConnection[DictRow]]:
    """Transactional connection. Commits on success, rolls back on error.

    The pool is opened with ``dict_row``, so every cursor yields mappings. The
    annotation says so explicitly; without it every ``row["column"]`` in the
    codebase is an unchecked access on ``tuple | dict``.
    """
    pool = get_pool()
    try:
        async with pool.connection() as conn:
            yield cast("psycopg.AsyncConnection[DictRow]", conn)
    except psycopg.OperationalError as exc:  # pragma: no cover - infra failure
        log.error("postgres.operational_error", error=str(exc))
        raise DependencyUnavailable("PostgreSQL is unreachable.") from exc


Params = Sequence[Any] | dict[str, Any] | None


async def fetch_all(sql: str, params: Params = None) -> list[DictRow]:
    async with connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)  # type: ignore[arg-type]
        return list(await cur.fetchall())


async def fetch_one(sql: str, params: Params = None) -> DictRow | None:
    async with connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)  # type: ignore[arg-type]
        return await cur.fetchone()


async def execute(sql: str, params: Params = None) -> int:
    async with connection() as conn, conn.cursor() as cur:
        await cur.execute(sql, params)  # type: ignore[arg-type]
        return cur.rowcount


async def ping() -> dict[str, Any]:
    """Health probe. Reports the real state, including which extensions exist."""
    try:
        async with connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT version() AS version")
            row = await cur.fetchone()
            await cur.execute(
                "SELECT extname FROM pg_extension WHERE extname = ANY(%s)",
                (["vector", "pg_trgm"],),
            )
            extensions = sorted(r["extname"] for r in await cur.fetchall())
            await cur.execute(
                "SELECT count(*)::int AS n FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            table_row = await cur.fetchone()
            tables = int(table_row["n"]) if table_row else 0
            version = str(row["version"]) if row else ""
        return {
            "status": "up",
            "version": version.split(" on ")[0],
            "extensions": extensions,
            "public_tables": tables,
        }
    except Exception as exc:
        return {"status": "down", "error": type(exc).__name__, "message": str(exc)[:200]}
