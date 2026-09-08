"""Ingestion endpoints.

``POST /api/v1/ingest`` accepts either a multipart upload or a JSON body naming
paths the API container can read (used for the curated corpus under ``data/``).
Both paths do the same three things synchronously -- validate, store
content-addressed, create a durable job record -- and then hand the work to the
background worker. The response says exactly which files were accepted, which
were rejected and why, and which were duplicates of documents already ingested.

``GET /api/v1/ingest/{job_id}`` returns real progress from the job record: per
stage, per document, including the stages that could not run and the environment
variables that would enable them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from services.common import bus, db
from services.common.errors import FileValidationError, NotFoundError, ValidationError
from services.common.ids import document_id
from services.common.ids import job_id as new_job_id
from services.common.logging import get_logger
from services.common.schemas import (
    AcceptedFile,
    CapabilityState,
    DataClass,
    IngestJobResponse,
    IngestPathRequest,
    IngestResponse,
    JobStatus,
    ReviewItem,
    StageReport,
)
from services.ingest import storage

log = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

REPO_ROOT = Path(__file__).resolve().parents[3]
#: Filesystem ingestion is restricted to these roots. A path outside them is
#: rejected: the endpoint must not become an arbitrary-file-read primitive.
ALLOWED_ROOTS = (REPO_ROOT / "data",)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit documents for ingestion (multipart upload)",
)
async def ingest_upload(
    files: list[UploadFile] = File(..., description="Documents to ingest"),
    data_class: DataClass = Form(
        ...,
        description="Provenance class of these files. Required: every ingested document "
        "must declare whether it is a real source document or synthetic test data.",
    ),
    source_system: str = Form("upload"),
    submitted_by: str | None = Form(None),
) -> IngestResponse:
    if len(files) > 500:
        raise ValidationError("At most 500 files may be submitted in one request.")

    job_id = new_job_id()
    accepted: list[dict[str, Any]] = []
    reported: list[AcceptedFile] = []
    duplicates = 0

    for upload in files:
        raw = await upload.read()
        try:
            blob = storage.store_bytes(upload.filename or "unnamed", raw)
        except FileValidationError as exc:
            reported.append(
                AcceptedFile(
                    filename=storage.safe_filename(upload.filename or "unnamed"),
                    byte_size=len(raw),
                    accepted=False,
                    reason=exc.message,
                )
            )
            continue

        existing = await db.fetch_one(
            "SELECT doc_id FROM documents WHERE content_hash = %s", (blob.content_hash,)
        )
        if existing:
            duplicates += 1
            reported.append(
                AcceptedFile(
                    filename=blob.original_filename,
                    byte_size=blob.byte_size,
                    accepted=False,
                    reason="Identical content has already been ingested.",
                    content_hash=blob.content_hash,
                    duplicate_of=existing["doc_id"],
                )
            )
            continue

        accepted.append(_blob_entry(blob, source_path=None))
        reported.append(
            AcceptedFile(
                filename=blob.original_filename,
                byte_size=blob.byte_size,
                accepted=True,
                content_hash=blob.content_hash,
            )
        )

    return await _create_job(
        job_id=job_id,
        source="upload",
        source_system=source_system,
        submitted_by=submitted_by,
        data_class=data_class,
        accepted=accepted,
        reported=reported,
        duplicates=duplicates,
    )


@router.post(
    "/paths",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit documents already present on a readable path",
)
async def ingest_paths(request: IngestPathRequest) -> IngestResponse:
    job_id = new_job_id()
    accepted: list[dict[str, Any]] = []
    reported: list[AcceptedFile] = []
    duplicates = 0

    for raw_path in request.paths:
        for path in _expand(raw_path, recursive=request.recursive):
            try:
                _assert_within_allowed_roots(path)
                blob = storage.store_path(path)
            except FileValidationError as exc:
                reported.append(
                    AcceptedFile(
                        filename=path.name, byte_size=0, accepted=False, reason=exc.message
                    )
                )
                continue

            existing = await db.fetch_one(
                "SELECT doc_id FROM documents WHERE content_hash = %s", (blob.content_hash,)
            )
            if existing:
                duplicates += 1
                reported.append(
                    AcceptedFile(
                        filename=blob.original_filename,
                        byte_size=blob.byte_size,
                        accepted=False,
                        reason="Identical content has already been ingested.",
                        content_hash=blob.content_hash,
                        duplicate_of=existing["doc_id"],
                    )
                )
                continue

            accepted.append(_blob_entry(blob, source_path=str(path)))
            reported.append(
                AcceptedFile(
                    filename=blob.original_filename,
                    byte_size=blob.byte_size,
                    accepted=True,
                    content_hash=blob.content_hash,
                )
            )

    return await _create_job(
        job_id=job_id,
        source=request.source.value,
        source_system=request.source_system,
        submitted_by=request.submitted_by,
        data_class=request.data_class,
        accepted=accepted,
        reported=reported,
        duplicates=duplicates,
    )


@router.get("/{job_id}", response_model=IngestJobResponse, summary="Ingestion job progress")
async def get_job(job_id: str) -> IngestJobResponse:
    row = await db.fetch_one("SELECT * FROM ingestion_jobs WHERE job_id = %s", (job_id,))
    if not row:
        raise NotFoundError(f"No ingestion job with id {job_id}.")

    documents = await db.fetch_all(
        """
        SELECT doc_id, original_filename, title,
               doc_type::text   AS doc_type,
               doc_type_confidence, doc_type_method,
               data_class::text AS data_class,
               page_count, has_text_layer, byte_size, created_at,
               parser, ocr_engine, ocr_mean_confidence, ocr_word_count,
               vector_objects, is_drawing, tables_found,
               processing_ms, chunk_count, mention_count,
               graph_nodes_created, graph_edges_created, ingest_error
          FROM documents
         WHERE ingest_job_id = %s
         ORDER BY created_at
        """,
        (job_id,),
    )
    reviews = await db.fetch_all(
        "SELECT r.review_id, r.kind, r.subject, r.doc_id, r.confidence, r.detail, r.created_at "
        "FROM review_queue r JOIN documents d ON d.doc_id = r.doc_id "
        "WHERE d.ingest_job_id = %s AND r.resolved_at IS NULL "
        "ORDER BY r.created_at DESC LIMIT 100",
        (job_id,),
    )

    totals = await db.fetch_one(
        """
        SELECT coalesce(sum(processing_ms), 0)::int       AS processing_ms,
               coalesce(sum(page_count), 0)::int          AS pages,
               coalesce(sum(graph_nodes_created), 0)::int AS graph_nodes
          FROM documents WHERE ingest_job_id = %s
        """,
        (job_id,),
    )
    extraction_stats = await db.fetch_one(
        """
        SELECT count(*)::int                                   AS total,
               count(*) FILTER (WHERE quote_verified)::int     AS verified
          FROM extractions e
          JOIN documents d ON d.doc_id = e.doc_id
         WHERE d.ingest_job_id = %s
        """,
        (job_id,),
    )

    stage_report = row.get("stage_report") or {}
    stage_reports = _collapse_stage_reports(stage_report)

    return IngestJobResponse(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        source=row["source"],
        source_system=row["source_system"],
        stage=row["stage"],
        file_count=row["file_count"],
        processed=row["processed_count"],
        failed=row["failed_count"],
        skipped_duplicates=row["skipped_count"],
        chunks_created=row["chunks_created"],
        mentions_created=row["mentions_created"],
        entities_created=row["entities_created"],
        edges_created=row["edges_created"],
        stage_reports=stage_reports,
        documents=[dict(d) for d in documents],
        review_queue=[
            ReviewItem(
                review_id=r["review_id"],
                kind=r["kind"],
                subject=r["subject"],
                doc_id=r["doc_id"],
                confidence=r["confidence"],
                detail=r["detail"] or {},
                created_at=r["created_at"],
            )
            for r in reviews
        ],
        pages=int(totals["pages"]) if totals else 0,
        graph_nodes_created=int(totals["graph_nodes"]) if totals else 0,
        processing_ms=int(totals["processing_ms"]) if totals else 0,
        duration_ms=_duration_ms(row),
        extractions_total=int(extraction_stats["total"]) if extraction_stats else 0,
        extractions_verified=int(extraction_stats["verified"]) if extraction_stats else 0,
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


@router.get("", summary="Recent ingestion jobs")
async def list_jobs(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    rows = await db.fetch_all(
        "SELECT job_id, status::text AS status, source, source_system, file_count, "
        "processed_count, failed_count, skipped_count, chunks_created, entities_created, "
        "created_at, finished_at FROM ingestion_jobs ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    return {"items": [dict(r) for r in rows], "queue_depth": await bus.queue_depth()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _duration_ms(row: dict[str, Any]) -> int | None:
    """Wall-clock duration of the job, when it has actually finished.

    Distinct from the summed per-document processing time: a job also spends
    time queued and moving between documents, and reporting one as the other
    would overstate throughput.
    """
    started, finished = row.get("started_at"), row.get("finished_at")
    if not started or not finished:
        return None
    return int((finished - started).total_seconds() * 1000)


def _blob_entry(blob: storage.StoredBlob, *, source_path: str | None) -> dict[str, Any]:
    return {
        "content_hash": blob.content_hash,
        "path": blob.path,
        "byte_size": blob.byte_size,
        "original_filename": blob.original_filename,
        "extension": blob.extension,
        "mime_type": blob.mime_type,
        "source_path": source_path,
        "expected_doc_id": document_id(blob.content_hash),
    }


async def _create_job(
    *,
    job_id: str,
    source: str,
    source_system: str,
    submitted_by: str | None,
    data_class: DataClass,
    accepted: list[dict[str, Any]],
    reported: list[AcceptedFile],
    duplicates: int,
) -> IngestResponse:
    await db.execute(
        "INSERT INTO ingestion_jobs (job_id, status, source, source_system, submitted_by, "
        "file_count, skipped_count) VALUES (%s, 'queued', %s, %s, %s, %s, %s)",
        (job_id, source, source_system, submitted_by, len(accepted), duplicates),
    )

    if accepted:
        await bus.enqueue(
            {
                "job_id": job_id,
                "data_class": data_class.value,
                "source_system": source_system,
                "files": accepted,
            }
        )
    else:
        # Nothing to do is a terminal state, not a job left hanging in 'queued'.
        await db.execute(
            "UPDATE ingestion_jobs SET status = 'succeeded', finished_at = now(), "
            "stage_report = %s WHERE job_id = %s",
            (json.dumps({"documents": []}), job_id),
        )

    rejected = sum(1 for f in reported if not f.accepted) - duplicates
    log.info(
        "ingest.job_created",
        job_id=job_id,
        accepted=len(accepted),
        rejected=rejected,
        duplicates=duplicates,
        data_class=data_class.value,
    )
    return IngestResponse(
        job_id=job_id,
        status=JobStatus.QUEUED if accepted else JobStatus.SUCCEEDED,
        accepted=len(accepted),
        rejected=max(0, rejected),
        duplicates=duplicates,
        files=reported,
        queue_depth=await bus.queue_depth(),
        poll=f"/api/v1/ingest/{job_id}",
    )


#: Files that describe a corpus rather than belonging to it. Ingesting a
#: provenance manifest as a document would create an asset-less "document" whose
#: content is a list of hashes, and would inflate the corpus count.
_NON_DOCUMENT_FILENAMES = {"manifest.json", ".gitkeep", ".ds_store", "thumbs.db"}


def _expand(raw_path: str, *, recursive: bool) -> list[Path]:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / raw_path).resolve()
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            p
            for p in candidate.glob(pattern)
            if p.is_file() and p.name.lower() not in _NON_DOCUMENT_FILENAMES
        )
    return []


def _assert_within_allowed_roots(path: Path) -> None:
    resolved = path.resolve()
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return
        except (ValueError, OSError):
            continue
    raise FileValidationError(
        "Path is outside the permitted ingestion roots.",
        detail={"allowed_roots": [str(r) for r in ALLOWED_ROOTS]},
    )


def _collapse_stage_reports(stage_report: dict[str, Any]) -> list[StageReport]:
    """Aggregate per-document stage outcomes into one report per stage.

    The worst state observed wins, so a stage that failed on any document is not
    reported as available because it succeeded on the others.
    """
    severity = {
        CapabilityState.AVAILABLE: 0,
        CapabilityState.DISABLED: 1,
        CapabilityState.NOT_CONFIGURED: 2,
        CapabilityState.NOT_IMPLEMENTED: 3,
        CapabilityState.ERROR: 4,
    }
    collapsed: dict[str, StageReport] = {}
    for document in stage_report.get("documents", []):
        for stage in document.get("stages", []):
            try:
                state = CapabilityState(stage["state"])
            except ValueError:
                continue
            name = stage["stage"]
            current = collapsed.get(name)
            if current is None:
                collapsed[name] = StageReport(
                    stage=name,
                    state=state,
                    detail=stage.get("detail"),
                    required_env=stage.get("required_env", []),
                    items=stage.get("items", 0),
                )
                continue
            current.items += stage.get("items", 0)
            if severity[state] > severity[current.state]:
                current.state = state
                current.detail = stage.get("detail")
                current.required_env = stage.get("required_env", [])
    return list(collapsed.values())
