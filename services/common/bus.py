"""Redis client, reliable work queue, and the event bus.

Three responsibilities, one connection pool:

**Queue.** Ingestion is asynchronous because a single 400-page scanned drawing
must never block a query. The queue is a reliable-handoff pattern: ``BLMOVE``
atomically moves a job from the pending list to a per-worker processing list,
and the job is only removed once the worker acknowledges it. A worker that dies
mid-job leaves its entry in the processing list, and :func:`reclaim_stale` puts
it back. That is the property a plain ``BRPOP`` does not have.

**Event bus.** The write path emits ``graph.changed`` events; the proactive
engine and the dashboard's live ingestion panel consume them. Pub/sub for live
subscribers plus a capped stream so a page that loads late still sees recent
history.

**Cache.** Query-level caching lives here too, keyed by normalised question.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis

from services.common.config import get_settings
from services.common.errors import DependencyUnavailable
from services.common.logging import get_logger

log = get_logger(__name__)

_client: aioredis.Redis | None = None

EVENT_CHANNEL = "events:brain"
EVENT_STREAM = "events:brain:recent"
EVENT_STREAM_MAXLEN = 500


async def open_client() -> aioredis.Redis:
    global _client
    if _client is not None:
        return _client
    settings = get_settings()
    _client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=10,
        health_check_interval=30,
    )
    await _client.ping()
    log.info("redis.client_opened")
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        log.info("redis.client_closed")


def get_client() -> aioredis.Redis:
    if _client is None:
        raise DependencyUnavailable("Redis client is not open.")
    return _client


# ---------------------------------------------------------------------------
# Reliable queue
# ---------------------------------------------------------------------------


def _processing_key(worker: str) -> str:
    settings = get_settings()
    return f"{settings.ingest_queue_name}:processing:{worker}"


async def enqueue(payload: dict[str, Any], *, queue: str | None = None) -> int:
    """Push a job. Returns the resulting queue depth."""
    client = get_client()
    q = queue or get_settings().ingest_queue_name
    body = json.dumps({**payload, "_enqueued_at": time.time()}, separators=(",", ":"))
    return int(await client.lpush(q, body))


async def dequeue(
    worker: str, *, timeout_s: int = 5, queue: str | None = None
) -> dict[str, Any] | None:
    """Blocking reliable pop. The job stays in the worker's processing list
    until :func:`ack` removes it."""
    client = get_client()
    q = queue or get_settings().ingest_queue_name
    raw = await client.blmove(
        q, _processing_key(worker), timeout=timeout_s, src="RIGHT", dest="LEFT"
    )
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.error("queue.undecodable_job", raw=raw[:200])
        await client.lrem(_processing_key(worker), 1, raw)
        return None
    payload["_raw"] = raw
    return payload


async def ack(worker: str, payload: dict[str, Any]) -> None:
    client = get_client()
    raw = payload.get("_raw")
    if raw:
        await client.lrem(_processing_key(worker), 1, raw)


async def nack(worker: str, payload: dict[str, Any], *, requeue: bool = True) -> None:
    """Return a job to the pending queue (or drop it) after a failure."""
    client = get_client()
    raw = payload.get("_raw")
    if not raw:
        return
    q = get_settings().ingest_queue_name
    async with client.pipeline(transaction=True) as pipe:
        pipe.lrem(_processing_key(worker), 1, raw)
        if requeue:
            pipe.lpush(q, raw)
        await pipe.execute()


async def reclaim_stale(worker: str) -> int:
    """Move this worker's abandoned in-flight jobs back onto the queue.

    Called at worker start-up: a container restart in the middle of a long OCR
    job must not lose the job.
    """
    client = get_client()
    q = get_settings().ingest_queue_name
    moved = 0
    while await client.rpoplpush(_processing_key(worker), q):
        moved += 1
    if moved:
        log.warning("queue.reclaimed_stale_jobs", worker=worker, count=moved)
    return moved


async def queue_depth(queue: str | None = None) -> int:
    client = get_client()
    return int(await client.llen(queue or get_settings().ingest_queue_name))


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


async def publish_event(event_type: str, payload: dict[str, Any]) -> None:
    """Publish to both pub/sub (live subscribers) and a capped stream (replay)."""
    client = get_client()
    event = {"type": event_type, "ts": time.time(), **payload}
    body = json.dumps(event, separators=(",", ":"), default=str)
    async with client.pipeline(transaction=False) as pipe:
        pipe.publish(EVENT_CHANNEL, body)
        pipe.xadd(EVENT_STREAM, {"body": body}, maxlen=EVENT_STREAM_MAXLEN, approximate=True)
        await pipe.execute()


async def recent_events(count: int = 50) -> list[dict[str, Any]]:
    client = get_client()
    entries = await client.xrevrange(EVENT_STREAM, count=count)
    out: list[dict[str, Any]] = []
    for entry_id, fields in entries:
        try:
            event = json.loads(fields["body"])
            event["_id"] = entry_id
            out.append(event)
        except (KeyError, json.JSONDecodeError):
            continue
    return out


async def ping() -> dict[str, Any]:
    try:
        client = get_client()
        info = await client.info("server")
        return {
            "status": "up",
            "version": info.get("redis_version"),
            "queue_depth": await queue_depth(),
        }
    except Exception as exc:
        return {"status": "down", "error": type(exc).__name__, "message": str(exc)[:200]}
