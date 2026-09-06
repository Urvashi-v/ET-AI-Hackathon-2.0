"""Schema initialisation for both stores.

Run as ``python -m services.common.migrate``, and automatically on API startup.

PostgreSQL migrations are ordinary ``.sql`` files applied in filename order and
recorded in ``schema_migrations`` with a checksum, so a file that changes after
being applied is reported rather than silently ignored. Neo4j constraints are
``IF NOT EXISTS`` and are re-applied every run -- the graph has no migration
table because its DDL is declaratively idempotent.

``${VAR}`` placeholders in SQL are substituted from a whitelist of settings
(currently only ``EMBEDDING_DIM``); this is not general templating and no
user-supplied value ever reaches it.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

from services.common import bus, db, graph
from services.common.config import get_settings
from services.common.logging import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "database" / "migrations"
CYPHER_DIR = REPO_ROOT / "database" / "cypher"

_PLACEHOLDER = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _substitutions() -> dict[str, str]:
    settings = get_settings()
    return {"EMBEDDING_DIM": str(settings.embedding_dim)}


def render(sql: str) -> str:
    subs = _substitutions()

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in subs:
            raise ValueError(f"Unknown migration placeholder ${{{key}}}")
        return subs[key]

    return _PLACEHOLDER.sub(replace, sql)


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def apply_sql_migrations() -> list[dict[str, Any]]:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        log.warning("migrate.no_sql_migrations", directory=str(MIGRATIONS_DIR))
        return []

    results: list[dict[str, Any]] = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        rendered = render(raw)
        checksum = _checksum(rendered)

        async with db.connection() as conn, conn.cursor() as cur:
            # The bookkeeping table is created by 001; before that runs the
            # lookup must tolerate its absence.
            await cur.execute("SELECT to_regclass('public.schema_migrations') AS t")
            row = await cur.fetchone()
            applied: dict[str, Any] | None = None
            if row and row["t"]:
                await cur.execute(
                    "SELECT checksum FROM schema_migrations WHERE filename = %s", (path.name,)
                )
                applied = await cur.fetchone()

            if applied and applied["checksum"] == checksum:
                results.append({"file": path.name, "status": "already_applied"})
                continue
            if applied and applied["checksum"] != checksum:
                # Never silently re-run a changed migration over live data.
                results.append(
                    {
                        "file": path.name,
                        "status": "checksum_mismatch",
                        "detail": "file changed after it was applied; create a new migration instead",
                    }
                )
                log.error("migrate.checksum_mismatch", file=path.name)
                continue

            await cur.execute(rendered)  # type: ignore[arg-type]
            await cur.execute(
                "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s) "
                "ON CONFLICT (filename) DO UPDATE SET checksum = EXCLUDED.checksum, "
                "applied_at = now()",
                (path.name, checksum),
            )
        results.append({"file": path.name, "status": "applied"})
        log.info("migrate.sql_applied", file=path.name)
    return results


async def apply_cypher_scripts() -> list[dict[str, Any]]:
    files = sorted(CYPHER_DIR.glob("*.cypher"))
    results: list[dict[str, Any]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        outcomes = await graph.run_script(text)
        results.append({"file": path.name, "statements": len(outcomes), "status": "applied"})
        log.info("migrate.cypher_applied", file=path.name, statements=len(outcomes))
    return results


async def run() -> dict[str, Any]:
    """Apply both schemas. Safe to call on every startup."""
    sql = await apply_sql_migrations()
    cypher = await apply_cypher_scripts()
    failed = [r for r in sql if r["status"] == "checksum_mismatch"]
    return {
        "status": "ok" if not failed else "degraded",
        "postgres": sql,
        "neo4j": cypher,
    }


async def _main() -> int:
    configure_logging()
    await db.open_pool()
    await graph.open_driver()
    await bus.open_client()
    try:
        report = await run()
        for entry in report["postgres"]:
            log.info("migrate.result", store="postgres", **entry)
        for entry in report["neo4j"]:
            log.info("migrate.result", store="neo4j", **entry)
        return 0 if report["status"] == "ok" else 1
    finally:
        await db.close_pool()
        await graph.close_driver()
        await bus.close_client()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(_main()))
