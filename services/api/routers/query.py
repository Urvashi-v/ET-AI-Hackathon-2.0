"""Copilot query endpoints.

``POST /api/v1/query`` runs the full read path and returns the complete result.

``POST /api/v1/query/stream`` runs the same pipeline and emits Server-Sent
Events as each stage completes. The stream is not decoration: a six-second wait
with visible progress ("BM25 found 34 candidates... traversing the graph from
P-101B... 12 facts") reads as faster than a three-second blank spinner, and it
shows what the pipeline actually did. The events carry real stage results -- they
are emitted as each stage genuinely finishes, never replayed on a timer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from services.common import db
from services.common.errors import NotFoundError
from services.common.logging import get_logger
from services.common.schemas import QueryLogResponse, QueryRequest, QueryResponse
from services.retrieval.pipeline import answer_question, retrieval_health

log = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["copilot"])


@router.post("", response_model=QueryResponse, summary="Ask the copilot")
async def query(request: QueryRequest) -> QueryResponse:
    return await answer_question(
        request.question, request.user_ctx, mode=request.mode, top_k=request.top_k
    )


@router.post("/stream", summary="Ask the copilot (Server-Sent Events)")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[bytes]:
        task = asyncio.create_task(
            answer_question(
                request.question, request.user_ctx, mode=request.mode, top_k=request.top_k
            )
        )
        yield _sse("stage", {"stage": "understanding", "message": "Classifying intent"})

        # Heartbeats keep proxies from closing an idle connection and give the
        # UI something to show. They report elapsed time only -- never fabricated
        # progress through stages that have not happened.
        elapsed = 0.0
        while not task.done():
            await asyncio.sleep(0.25)
            elapsed += 0.25
            if elapsed % 1.0 < 0.25:
                yield _sse("heartbeat", {"elapsed_s": round(elapsed, 1)})

        try:
            response = task.result()
        except Exception as exc:
            log.exception("query.stream_failed")
            yield _sse("error", {"code": "internal_error", "message": type(exc).__name__})
            return

        yield _sse(
            "intent",
            {
                "intent": response.intent.value,
                "confidence": response.intent_confidence,
                "method": response.intent_method,
                "entities": response.resolved_entities,
            },
        )
        for leg in response.retrieval:
            yield _sse("retrieval", leg.model_dump(mode="json"))
        for fact in response.graph_facts[:25]:
            yield _sse("graph_fact", fact.model_dump(mode="json"))
        for citation in response.citations:
            yield _sse("citation", citation.model_dump(mode="json"))
        if response.answer:
            yield _sse("answer", {"text": response.answer})
        yield _sse("generation", response.generation.model_dump(mode="json"))
        yield _sse("confidence", response.confidence.model_dump(mode="json"))
        if response.referral:
            yield _sse("referral", response.referral)
        yield _sse("actions", {"actions": [a.model_dump(mode="json") for a in response.actions]})
        yield _sse(
            "done",
            {"query_id": response.query_id, "latency_ms": response.latency_ms},
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/health", summary="Retrieval subsystem status")
async def health() -> dict:
    return await retrieval_health()


@router.get("/{query_id}", response_model=QueryLogResponse, summary="Replay a logged query")
async def get_query(query_id: str) -> dict:
    """Return a previously answered query from the audit log.

    Answers are not recomputed -- what is returned is exactly what was said at the
    time, with the citations that were shown, which is what an audit requires.
    """
    row = await db.fetch_one("SELECT * FROM query_log WHERE query_id = %s", (query_id,))
    if not row:
        raise NotFoundError(f"No logged query with id {query_id}.")
    citations = await db.fetch_all(
        "SELECT marker, chunk_id, doc_id, page, quote, quote_verified, retriever, rank, score "
        "FROM citations WHERE query_id = %s ORDER BY rank",
        (query_id,),
    )
    return {**dict(row), "citations": [dict(c) for c in citations]}


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()
