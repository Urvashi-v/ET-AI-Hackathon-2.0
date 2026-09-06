"""System event stream.

Ingestion is normally the boring part of a demo. Streamed as graph growth --
documents classified, entities resolved, new nodes attaching to existing ones --
it becomes the part that shows the substrate being built.

``GET /api/v1/events/stream`` is a Server-Sent Events feed backed by Redis
pub/sub. Every event on it was published by a real pipeline stage. The stream
opens by replaying recent history from a capped Redis stream, so a page that
loads mid-job still sees what happened, then follows live.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from services.common import bus, db
from services.common.logging import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/events", tags=["events"])

_HEARTBEAT_SECONDS = 15


@router.get("", summary="Recent system events")
async def recent(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    stored = await db.fetch_all(
        "SELECT event_id, event_type, doc_id, job_id, payload, occurred_at "
        "FROM events ORDER BY occurred_at DESC LIMIT %s",
        (limit,),
    )
    return {
        "persisted": [dict(r) for r in stored],
        "bus_recent": await bus.recent_events(limit),
    }


@router.get("/stream", summary="Live event stream (Server-Sent Events)")
async def stream(request: Request, replay: int = Query(20, ge=0, le=200)) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[bytes]:
        client = bus.get_client()
        pubsub = client.pubsub()
        await pubsub.subscribe(bus.EVENT_CHANNEL)
        try:
            if replay:
                history = await bus.recent_events(replay)
                for event in reversed(history):
                    yield _sse(event.get("type", "event"), {**event, "_replayed": True})
            yield _sse("ready", {"channel": bus.EVENT_CHANNEL, "replayed": replay})

            idle = 0.0
            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    idle += 1.0
                    if idle >= _HEARTBEAT_SECONDS:
                        idle = 0.0
                        yield _sse("heartbeat", {})
                    continue
                idle = 0.0
                try:
                    event = json.loads(message["data"])
                except (TypeError, json.JSONDecodeError):
                    continue
                yield _sse(event.get("type", "event"), event)
        except asyncio.CancelledError:  # pragma: no cover - client disconnect
            raise
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(bus.EVENT_CHANNEL)
                await pubsub.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()
