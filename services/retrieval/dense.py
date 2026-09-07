"""Dense retrieval over pgvector.

Runs only when an embedding provider is configured. When it is not, the leg
reports ``provider_not_configured`` with the environment variables required and
contributes nothing to fusion -- the pipeline still returns real results from the
lexical and graph legs. There is no random-vector or keyword-emulation fallback:
a dense leg that silently degrades into something else makes every retrieval
metric meaningless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from services.common import db
from services.common.config import get_settings
from services.common.logging import get_logger
from services.common.schemas import CapabilityState
from services.ingest.embeddings import _to_pgvector, embedding_capability, get_embedding_provider

log = get_logger(__name__)


@dataclass(slots=True)
class DenseResult:
    state: CapabilityState
    rows: list[dict[str, Any]] = field(default_factory=list)
    detail: str | None = None
    required_env: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0


async def search(question: str, *, top_k: int = 50) -> DenseResult:
    started = time.perf_counter()
    state, detail, required_env = embedding_capability()
    if state is not CapabilityState.AVAILABLE:
        return DenseResult(
            state=state,
            detail=detail,
            required_env=required_env,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    provider = get_embedding_provider()
    try:
        vector = await provider.embed_query(question)
    except Exception as exc:
        log.error("dense.embed_query_failed", error=str(exc))
        return DenseResult(
            state=CapabilityState.ERROR,
            detail=f"Query embedding failed: {type(exc).__name__}: {str(exc)[:200]}",
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    if not vector:
        return DenseResult(
            state=CapabilityState.NOT_CONFIGURED,
            detail="Embedding provider returned no vector.",
            required_env=required_env,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    settings = get_settings()
    rows = await db.fetch_all(
        """
        SELECT c.chunk_id,
               c.doc_id,
               c.text,
               c.section_path,
               c.page_from,
               c.chunk_kind,
               c.data_class::text AS chunk_data_class,
               d.title,
               d.doc_type::text   AS doc_type,
               d.data_class::text AS doc_data_class,
               d.source_system,
               d.is_current,
               1 - (e.embedding <=> %(vec)s::vector) AS score
          FROM chunk_embeddings e
          JOIN document_chunks c ON c.chunk_id = e.chunk_id
          JOIN documents d       ON d.doc_id   = e.doc_id
         WHERE e.model = %(model)s
         ORDER BY e.embedding <=> %(vec)s::vector
         LIMIT %(top_k)s
        """,
        {"vec": _to_pgvector(vector), "model": settings.active_embedding_model, "top_k": top_k},
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["retriever"] = "dense"

    return DenseResult(
        state=CapabilityState.AVAILABLE,
        rows=rows,
        detail=f"model={settings.active_embedding_model}",
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


async def indexed_chunk_count() -> int:
    row = await db.fetch_one("SELECT count(*)::int AS n FROM chunk_embeddings")
    return int(row["n"]) if row else 0
