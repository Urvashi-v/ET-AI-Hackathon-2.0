"""Background ingestion worker.

Runs as its own container so that a 400-page scanned drawing can never block a
query. Consumes the reliable Redis queue: a job is only removed from the
worker's in-flight list once it has been acknowledged, and jobs abandoned by a
crashed worker are reclaimed at start-up.

Failure policy is explicit rather than accidental:

* a document that fails is recorded as failed **on the job**, with the error,
  and the remaining documents still process -- one bad file must not lose a
  batch;
* a job that fails wholesale is nacked once and retried, then marked ``failed``
  with the reason, so it is never retried forever;
* the job's terminal status distinguishes ``succeeded``, ``partial`` (some
  documents failed) and ``failed``, because "it worked" and "most of it worked"
  are different facts.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import time
from typing import Any

from services.common import bus, db, graph
from services.common.config import get_settings
from services.common.logging import configure_logging, get_logger
from services.common.schemas import DataClass
from services.ingest import storage
from services.ingest.pipeline import DocumentResult, ingest_file

log = get_logger(__name__)

_MAX_ATTEMPTS = 2
_shutdown = asyncio.Event()


def worker_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


async def process_job(job: dict[str, Any]) -> None:
    """Process one ingestion job: N files through the write path."""
    job_id = job["job_id"]
    data_class = DataClass(job.get("data_class", DataClass.REAL_SOURCE_DOCUMENT.value))
    source_system = job.get("source_system", "unknown")
    files: list[dict[str, Any]] = job.get("files", [])

    started = time.perf_counter()
    await db.execute(
        "UPDATE ingestion_jobs SET status = 'running', started_at = now(), stage = %s "
        "WHERE job_id = %s",
        ("processing", job_id),
    )
    await bus.publish_event("ingest.started", {"job_id": job_id, "file_count": len(files)})

    results: list[DocumentResult] = []
    for index, entry in enumerate(files, start=1):
        await db.execute(
            "UPDATE ingestion_jobs SET stage = %s WHERE job_id = %s",
            (f"document {index}/{len(files)}: {entry.get('original_filename', '')}"[:200], job_id),
        )
        blob = storage.StoredBlob(
            content_hash=entry["content_hash"],
            path=entry["path"],
            byte_size=entry["byte_size"],
            original_filename=entry["original_filename"],
            extension=entry["extension"],
            mime_type=entry["mime_type"],
        )
        try:
            result = await ingest_file(
                job_id=job_id,
                blob=blob,
                data_class=data_class,
                source_system=source_system,
                source_path=entry.get("source_path"),
            )
        except Exception as exc:
            log.exception("ingest.document_failed", filename=blob.original_filename)
            result = DocumentResult(
                doc_id=None,
                filename=blob.original_filename,
                status="failed",
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
            )
        results.append(result)

        await bus.publish_event(
            "ingest.document",
            {
                "job_id": job_id,
                "index": index,
                "total": len(files),
                **result.to_dict(),
            },
        )

    processed = sum(1 for r in results if r.status == "ingested")
    duplicates = sum(1 for r in results if r.status == "duplicate")
    failed = sum(1 for r in results if r.status == "failed")
    status = "succeeded" if failed == 0 else ("partial" if processed or duplicates else "failed")

    await db.execute(
        """
        UPDATE ingestion_jobs SET
            status = %s,
            processed_count = %s,
            failed_count = %s,
            skipped_count = %s,
            chunks_created = %s,
            mentions_created = %s,
            entities_created = %s,
            edges_created = %s,
            stage = NULL,
            stage_report = %s,
            error = %s,
            finished_at = now()
        WHERE job_id = %s
        """,
        (
            status,
            processed,
            failed,
            duplicates,
            sum(r.chunks for r in results),
            sum(r.mentions for r in results),
            sum(r.assets for r in results),
            sum(r.edges for r in results),
            json.dumps({"documents": [r.to_dict() for r in results]}),
            "; ".join(r.error for r in results if r.error)[:1000] or None,
            job_id,
        ),
    )
    await bus.publish_event(
        "ingest.finished",
        {
            "job_id": job_id,
            "status": status,
            "processed": processed,
            "duplicates": duplicates,
            "failed": failed,
            "elapsed_s": round(time.perf_counter() - started, 2),
        },
    )
    log.info(
        "ingest.job_finished",
        job_id=job_id,
        status=status,
        processed=processed,
        duplicates=duplicates,
        failed=failed,
        elapsed_s=round(time.perf_counter() - started, 2),
    )


async def run_worker() -> None:
    configure_logging()
    settings = get_settings()
    name = worker_name()
    log.info("worker.starting", worker=name, queue=settings.ingest_queue_name)

    await db.open_pool()
    await graph.open_driver()
    await bus.open_client()
    await bus.reclaim_stale(name)

    idle_logged = False
    try:
        while not _shutdown.is_set():
            job = await bus.dequeue(name, timeout_s=5)
            if job is None:
                if not idle_logged:
                    log.debug("worker.idle", worker=name)
                    idle_logged = True
                continue
            idle_logged = False

            attempts = int(job.get("_attempts", 0)) + 1
            try:
                await process_job(job)
                await bus.ack(name, job)
            except Exception as exc:
                log.exception("worker.job_failed", job_id=job.get("job_id"), attempt=attempts)
                if attempts < _MAX_ATTEMPTS:
                    job["_attempts"] = attempts
                    await bus.nack(name, job, requeue=True)
                else:
                    await bus.nack(name, job, requeue=False)
                    await db.execute(
                        "UPDATE ingestion_jobs SET status = 'failed', error = %s, "
                        "finished_at = now() WHERE job_id = %s",
                        (f"{type(exc).__name__}: {str(exc)[:500]}", job.get("job_id")),
                    )
    finally:
        log.info("worker.stopping", worker=name)
        await db.close_pool()
        await graph.close_driver()
        await bus.close_client()


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except NotImplementedError:  # Windows
            signal.signal(sig, lambda *_: _shutdown.set())


async def _main() -> None:
    _install_signal_handlers(asyncio.get_running_loop())
    await run_worker()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
