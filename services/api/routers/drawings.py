"""Drawing access: detections, overlays and asset locations.

What makes a P&ID useful to software is not that it can be displayed — a PDF
viewer does that. It is that "where is P-101B?" has an answer in coordinates,
and that the answer is the same node the incident report and the CMMS export
resolve to.

Three endpoints, one question each:

* ``GET /drawings``                     — which documents have been digitised
* ``GET /drawings/{doc_id}/detections``  — everything found on a page, with boxes
* ``GET /drawings/locate/{asset_tag}``   — where does this asset appear, on any sheet

Page images come from the documents router, which already renders PDF pages. The
overlay is drawn client-side from these coordinates rather than burned into the
image, so the same render serves every highlight and nothing has to be
re-rasterised when the selection changes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.common import db
from services.common.logging import get_logger
from services.ingest.pid import capability as pid_capability

log = get_logger(__name__)
router = APIRouter(prefix="/drawings", tags=["drawings"])


@router.get("", summary="Documents with digitised drawings")
async def list_drawings() -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT d.doc_id, d.title, d.doc_type::text AS doc_type, d.page_count,
               d.data_class::text AS data_class, d.is_current, d.revision,
               d.doc_number, d.original_filename, d.mime_type,
               count(det.detection_id)::int                                   AS detections,
               count(det.linked_asset_id)::int                                AS linked,
               count(DISTINCT det.normalised) FILTER (WHERE det.kind = 'tag') AS distinct_tags
          FROM documents d
          LEFT JOIN drawing_detections det ON det.doc_id = d.doc_id
         WHERE d.is_drawing OR d.doc_type = 'pid'
         GROUP BY d.doc_id
         ORDER BY d.title
        """
    )
    return {
        "total": len(rows),
        "items": rows,
        # Which detectors ran and which are declared unimplemented. Shown so an
        # empty overlay reads as "this detector is not built" rather than "this
        # drawing is empty".
        "detectors": pid_capability(),
    }


@router.get("/{doc_id}/detections", summary="Everything detected on a drawing page")
async def get_detections(
    doc_id: str,
    page: int = Query(1, ge=1),
    kind: str | None = Query(None, description="tag | instrument_bubble | line_segment"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
) -> dict[str, Any]:
    document = await db.fetch_one(
        "SELECT doc_id, title, page_count, data_class::text AS data_class, mime_type "
        "FROM documents WHERE doc_id = %s",
        (doc_id,),
    )
    if not document:
        raise HTTPException(status_code=404, detail=f"No document {doc_id}")

    rows = await db.fetch_all(
        """
        SELECT det.detection_id, det.kind, det.text, det.normalised,
               det.x0, det.y0, det.x1, det.y1, det.page_width, det.page_height,
               det.method, det.confidence, det.linked_asset_id,
               det.properties, det.data_class::text AS data_class,
               a.canonical_tag, a.class_label
          FROM drawing_detections det
          LEFT JOIN assets a ON a.asset_id = det.linked_asset_id
         WHERE det.doc_id = %(doc_id)s
           AND det.page = %(page)s
           AND (%(kind)s::text IS NULL OR det.kind = %(kind)s::text)
           AND det.confidence >= %(min_conf)s
         ORDER BY det.kind, det.confidence DESC
        """,
        {"doc_id": doc_id, "page": page, "kind": kind, "min_conf": min_confidence},
    )

    connections = await db.fetch_all(
        """
        SELECT c.connection_id, c.from_detection, c.to_detection, c.method, c.confidence,
               f.normalised AS from_tag, t.normalised AS to_tag
          FROM drawing_connections c
          JOIN drawing_detections f ON f.detection_id = c.from_detection
          JOIN drawing_detections t ON t.detection_id = c.to_detection
         WHERE c.doc_id = %s AND c.page = %s
        """,
        (doc_id, page),
    )

    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    # Page geometry is taken from a detection rather than recomputed, so the
    # overlay is scaled by exactly the frame the boxes were measured in.
    geometry = next(({"width": r["page_width"], "height": r["page_height"]} for r in rows), None)

    return {
        "document": document,
        "page": page,
        "page_geometry": geometry,
        "counts": by_kind,
        "linked": sum(1 for r in rows if r["linked_asset_id"]),
        "unlinked_tags": sorted(
            {r["normalised"] for r in rows if r["kind"] == "tag" and not r["linked_asset_id"]}
        ),
        "detections": rows,
        "connections": connections,
        "detectors": pid_capability(),
        "page_image": f"/api/v1/documents/{doc_id}/page/{page}.png",
    }


@router.get("/locate/{asset_tag}", summary="Where an asset appears on any drawing")
async def locate_asset(asset_tag: str) -> dict[str, Any]:
    """The endpoint behind "click an asset, highlight it on the drawing".

    Returns every appearance across every digitised sheet, not just the first.
    A pump appears on its P&ID, its isometric and its layout drawing, and
    answering with one of them silently picks for the reader.
    """
    rows = await db.fetch_all(
        """
        SELECT det.detection_id, det.doc_id, det.page, det.kind, det.text, det.normalised,
               det.x0, det.y0, det.x1, det.y1, det.page_width, det.page_height,
               det.method, det.confidence,
               d.title AS doc_title, d.doc_number, d.revision, d.is_current,
               d.data_class::text AS data_class
          FROM drawing_detections det
          JOIN documents d ON d.doc_id = det.doc_id
          LEFT JOIN assets a ON a.asset_id = det.linked_asset_id
         WHERE upper(coalesce(a.canonical_tag, det.normalised)) = upper(%s)
         ORDER BY d.title, det.page, det.confidence DESC
        """,
        (asset_tag,),
    )
    if not rows:
        return {
            "asset_tag": asset_tag.upper(),
            "appearances": [],
            "detail": (
                f"{asset_tag.upper()} was not detected on any digitised drawing. Either no "
                "drawing showing it has been ingested, or the tag on the sheet did not "
                "resolve to this asset."
            ),
        }

    for row in rows:
        row["page_image"] = f"/api/v1/documents/{row['doc_id']}/page/{row['page']}.png"
    return {
        "asset_tag": asset_tag.upper(),
        "appearances": rows,
        "drawings": len({r["doc_id"] for r in rows}),
        "detail": None,
    }
