"""Neo4j access -- the knowledge graph.

The graph exists to answer the questions no vector index can: multi-hop
traversal ("what is downstream of P-101B?"), aggregation across time and sites,
and the compliance gap, which is literally the *absence* of an edge.

Every write goes through :func:`write` inside an explicit transaction, and the
Cypher itself lives in ``database/cypher/`` or in the graph writer so that it is
reviewable as a unit rather than scattered through f-strings.
"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from services.common.config import get_settings
from services.common.errors import DependencyUnavailable
from services.common.logging import get_logger

log = get_logger(__name__)

_driver: AsyncDriver | None = None


async def open_driver() -> AsyncDriver:
    global _driver
    if _driver is not None:
        return _driver
    settings = get_settings()
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        max_connection_lifetime=3600,
        connection_acquisition_timeout=30,
    )
    await _driver.verify_connectivity()
    log.info("neo4j.driver_opened", uri=settings.neo4j_uri, database=settings.neo4j_database)
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        log.info("neo4j.driver_closed")


def get_driver() -> AsyncDriver:
    if _driver is None:
        raise DependencyUnavailable("Neo4j driver is not open.")
    return _driver


async def read(cypher: str, **params: Any) -> list[dict[str, Any]]:
    driver = get_driver()
    settings = get_settings()
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, **params)  # type: ignore[arg-type]
            return [record.data() async for record in result]
    except ServiceUnavailable as exc:  # pragma: no cover - infra failure
        raise DependencyUnavailable("Neo4j is unreachable.") from exc


async def write(cypher: str, **params: Any) -> list[dict[str, Any]]:
    driver = get_driver()
    settings = get_settings()
    try:
        async with driver.session(database=settings.neo4j_database) as session:

            async def _tx(tx: Any) -> list[dict[str, Any]]:
                result = await tx.run(cypher, **params)
                return [record.data() async for record in result]

            return await session.execute_write(_tx)
    except ServiceUnavailable as exc:  # pragma: no cover - infra failure
        raise DependencyUnavailable("Neo4j is unreachable.") from exc


def split_statements(cypher_text: str) -> list[str]:
    """Split a ``.cypher`` file into statements.

    Quote-aware, because a naive ``text.split(";")`` cuts through any semicolon
    inside a string literal and hands Neo4j two syntactically broken halves --
    which is easy to hit, since provenance strings routinely contain one.

    Handles single and double quoted strings, backtick-quoted identifiers,
    backslash escapes, ``//`` line comments and ``/* */`` block comments.
    """
    statements: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    index = 0
    length = len(cypher_text)

    while index < length:
        char = cypher_text[index]

        if quote:
            buffer.append(char)
            if char == "\\" and index + 1 < length:
                buffer.append(cypher_text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue

        if char in "'\"`":
            quote = char
            buffer.append(char)
            index += 1
            continue

        if cypher_text.startswith("//", index):
            end = cypher_text.find("\n", index)
            index = length if end == -1 else end + 1
            buffer.append("\n")
            continue

        if cypher_text.startswith("/*", index):
            end = cypher_text.find("*/", index + 2)
            index = length if end == -1 else end + 2
            continue

        if char == ";":
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    trailing = "".join(buffer).strip()
    if trailing:
        statements.append(trailing)
    return statements


async def run_script(cypher_text: str) -> list[dict[str, Any]]:
    """Run a multi-statement ``.cypher`` file.

    Constraint and index creation is ``IF NOT EXISTS`` and every seed uses
    ``MERGE`` on a natural key, so the whole script is idempotent and safe to
    re-run on every startup.
    """
    statements = split_statements(cypher_text)
    outcomes: list[dict[str, Any]] = []
    for stmt in statements:
        try:
            await write(stmt)
            outcomes.append({"statement": _summarise(stmt), "status": "ok"})
        except Neo4jError as exc:
            outcomes.append(
                {"statement": _summarise(stmt), "status": "error", "message": exc.message}
            )
            log.error(
                "neo4j.script_statement_failed", statement=_summarise(stmt), error=exc.message
            )
            raise
    return outcomes


def _summarise(stmt: str) -> str:
    flat = " ".join(stmt.split())
    return flat[:90] + ("..." if len(flat) > 90 else "")


async def ping() -> dict[str, Any]:
    """Health probe reporting real counts, not a hard-coded 'ok'."""
    try:
        rows = await read("CALL db.labels() YIELD label " "RETURN collect(label) AS labels")
        labels = rows[0]["labels"] if rows else []
        counts = await read(
            "MATCH (n) WITH count(n) AS nodes "
            "CALL { MATCH ()-[r]->() RETURN count(r) AS rels } "
            "RETURN nodes, rels"
        )
        constraints = await read("SHOW CONSTRAINTS YIELD name RETURN count(name) AS n")
        indexes = await read("SHOW INDEXES YIELD name RETURN count(name) AS n")
        return {
            "status": "up",
            "labels": sorted(labels),
            "nodes": counts[0]["nodes"] if counts else 0,
            "relationships": counts[0]["rels"] if counts else 0,
            "constraints": constraints[0]["n"] if constraints else 0,
            "indexes": indexes[0]["n"] if indexes else 0,
        }
    except Exception as exc:
        return {"status": "down", "error": type(exc).__name__, "message": str(exc)[:200]}
