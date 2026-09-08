"""Source document access: the other half of a citation.

A citation that cannot be opened is an assertion, not evidence. These endpoints
are what make "[C1]" clickable:

* ``GET /documents``               -- the corpus, with provenance and currency
* ``GET /documents/{doc_id}``      -- metadata, revision lineage, chunk index
* ``GET /documents/{doc_id}/chunks/{chunk_id}`` -- the exact extracted span
* ``GET /documents/{doc_id}/page/{page}.png``   -- the rendered source page
* ``GET /documents/{doc_id}/raw``  -- the original file as uploaded

The page render is the one that closes the loop for a field engineer: the answer
says the seal failed, the citation says page 2 of the incident report, and the
render shows page 2 of the incident report with the cited text boxed on it. At
that point the operator is reading the source, not trusting the system.

Blobs are addressed by ``doc_id`` and resolved through the database, never by a
client-supplied path. That is deliberate: a path parameter that reaches the
filesystem is a directory-traversal bug waiting to happen, and the content-
addressed store means there is no need for one.
"""

from __future__ import annotations

import io
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Response
from fastapi.responses import FileResponse

from services.common import db
from services.common.logging import get_logger
from services.ingest import storage

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

#: Rendering scale for page images. 2.0 gives roughly 144 dpi, which is legible
#: for scanned reports without producing multi-megabyte PNGs over the wire.
_RENDER_SCALE = 2.0

#: Rendered pages are immutable: a document is content-addressed, so the same
#: doc_id can never have different page content. Long cache lifetimes are safe.
_IMMUTABLE = "public, max-age=86400, immutable"


@router.get("", summary="Documents in the corpus")
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    doc_type: str | None = None,
    current_only: bool = False,
) -> dict[str, Any]:
    where: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if doc_type:
        where.append("doc_type = %(doc_type)s::document_type")
        params["doc_type"] = doc_type
    if current_only:
        where.append("is_current")
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    total_row = await db.fetch_one(f"SELECT count(*)::int AS n FROM documents {clause}", params)
    rows = await db.fetch_all(
        f"""
        SELECT doc_id, title, doc_type::text AS doc_type, doc_type_confidence,
               data_class::text AS data_class, source_system, original_filename,
               mime_type, byte_size, page_count, has_text_layer, revision,
               issued_on, revised_on, superseded_by, valid_from, valid_to,
               is_current, parser, ocr_engine, is_drawing, tables_found,
               chunk_count, mention_count, created_at
          FROM documents
          {clause}
         ORDER BY created_at DESC
         LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    )
    return {
        "total": int(total_row["n"]) if total_row else 0,
        "limit": limit,
        "offset": offset,
        "items": rows,
    }


@router.get("/{doc_id}", summary="One document, with its revision lineage")
async def get_document(doc_id: str) -> dict[str, Any]:
    row = await db.fetch_one(
        """
        SELECT doc_id, content_hash, title, doc_type::text AS doc_type,
               doc_type_confidence, doc_type_method, data_class::text AS data_class,
               source_system, source_path, original_filename, mime_type, byte_size,
               page_count, has_text_layer, revision, issued_on, revised_on,
               superseded_by, valid_from, valid_to, is_current, licence_note,
               parser, ocr_engine, ocr_mean_confidence, ocr_word_count,
               is_drawing, tables_found, processing_ms, chunk_count, mention_count,
               metadata, created_at, updated_at
          FROM documents WHERE doc_id = %s
        """,
        (doc_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No document {doc_id}")

    # The revision chain, both directions. An engineer reading a procedure needs
    # to know it has been superseded far more urgently than they need any of its
    # content, so it is returned with the document rather than behind a link.
    supersedes = await db.fetch_all(
        """
        SELECT doc_id, title, revision, revised_on, is_current
          FROM documents WHERE superseded_by = %s ORDER BY revised_on DESC NULLS LAST
        """,
        (doc_id,),
    )
    superseded_by = None
    if row.get("superseded_by"):
        superseded_by = await db.fetch_one(
            "SELECT doc_id, title, revision, revised_on, is_current FROM documents "
            "WHERE doc_id = %s",
            (row["superseded_by"],),
        )

    chunks = await db.fetch_all(
        """
        SELECT chunk_id, ordinal, chunk_kind, section_path, page_from, page_to,
               data_class::text AS data_class, token_count,
               length(text) AS char_count, left(text, 240) AS preview
          FROM document_chunks WHERE doc_id = %s ORDER BY ordinal
        """,
        (doc_id,),
    )
    return {
        "document": row,
        "revision_chain": {"supersedes": supersedes, "superseded_by": superseded_by},
        "chunks": chunks,
    }


@router.get("/{doc_id}/chunks/{chunk_id}", summary="One extracted span, in full")
async def get_chunk(doc_id: str, chunk_id: str) -> dict[str, Any]:
    """The exact text a citation points at, with its geometry.

    ``bbox`` comes from the parser's word geometry, so a viewer can draw the box
    over the rendered page. It is null for chunks from formats that have no
    geometry -- CSV rows, Markdown -- and that is reported rather than guessed.
    """
    row = await db.fetch_one(
        """
        SELECT c.chunk_id, c.doc_id, c.ordinal, c.chunk_kind, c.section_path,
               c.context_header, c.text, c.page_from, c.page_to, c.bbox,
               c.data_class::text AS data_class, c.extraction_method,
               c.token_count, length(c.text) AS char_count,
               d.title, d.doc_type::text AS doc_type, d.is_current,
               d.original_filename, d.page_count
          FROM document_chunks c
          JOIN documents d ON d.doc_id = c.doc_id
         WHERE c.chunk_id = %s AND c.doc_id = %s
        """,
        (chunk_id, doc_id),
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No chunk {chunk_id} in {doc_id}")

    mentions = await db.fetch_all(
        """
        SELECT m.mention_id, m.surface_form, m.normalised, m.tag_kind,
               m.char_start, m.char_end, m.page, m.extractor,
               m.extractor_confidence, m.resolved_asset_id, m.resolution_score,
               m.resolution_action, m.needs_review, a.canonical_tag
          FROM mentions m
          LEFT JOIN assets a ON a.asset_id = m.resolved_asset_id
         WHERE m.chunk_id = %s ORDER BY m.char_start
        """,
        (chunk_id,),
    )
    return {"chunk": row, "mentions": mentions}


@router.get("/{doc_id}/page/{page}.png", summary="Rendered source page")
async def render_page(
    doc_id: str,
    page: int = Path(ge=1, description="1-based page number"),
) -> Response:
    """Render one PDF page to PNG, straight from the stored original.

    Rendered on demand rather than at ingest: page images are large, most are
    never viewed, and pypdfium2 renders a page in tens of milliseconds. Storing
    them would trade a lot of disk for latency nobody notices.
    """
    row = await db.fetch_one(
        "SELECT blob_path, mime_type, page_count, original_filename FROM documents "
        "WHERE doc_id = %s",
        (doc_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No document {doc_id}")
    if (row.get("mime_type") or "") != "application/pdf":
        raise HTTPException(
            status_code=415,
            detail=(
                f"{row['original_filename']} is {row.get('mime_type') or 'of unknown type'}, "
                "not a PDF. Page rendering applies to PDFs; use the chunk endpoint for "
                "the extracted text of other formats."
            ),
        )
    total = int(row.get("page_count") or 0)
    if page < 1 or (total and page > total):
        raise HTTPException(
            status_code=404, detail=f"Page {page} is out of range; the document has {total} page(s)"
        )

    path = storage.resolve_blob(row["blob_path"])
    if not path.exists():
        raise HTTPException(
            status_code=410,
            detail="The stored original is no longer present on disk. Re-ingest the document.",
        )

    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(path))
        try:
            bitmap = pdf[page - 1].render(scale=_RENDER_SCALE)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        finally:
            pdf.close()
    except HTTPException:
        raise
    except Exception as exc:
        log.error("documents.render_failed", doc_id=doc_id, page=page, error=str(exc))
        raise HTTPException(
            status_code=500, detail=f"Page render failed: {type(exc).__name__}"
        ) from exc

    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": _IMMUTABLE},
    )


@router.get("/{doc_id}/raw", summary="The original file as ingested")
async def get_raw(doc_id: str) -> FileResponse:
    """The byte-for-byte original.

    Served as an attachment with the stored filename. The path comes from the
    database, never from the request, so the endpoint cannot be pointed at a file
    outside the blob store.
    """
    row = await db.fetch_one(
        "SELECT blob_path, mime_type, original_filename FROM documents WHERE doc_id = %s",
        (doc_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No document {doc_id}")
    path = storage.resolve_blob(row["blob_path"])
    if not path.exists():
        raise HTTPException(
            status_code=410,
            detail="The stored original is no longer present on disk. Re-ingest the document.",
        )
    return FileResponse(
        path,
        media_type=row.get("mime_type") or "application/octet-stream",
        filename=row["original_filename"],
    )
