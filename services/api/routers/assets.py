"""Asset endpoints.

``GET /api/v1/assets`` lists the canonical entity layer with its linkage
counters. Those counters are the platform's own headline metric: an asset
evidenced by more than one document, from more than one source system, is a
concrete instance of two systems that never talked to each other now being
joined.

``GET /api/v1/assets/{asset_id}`` returns the full dossier: every tag variant
that resolved to this asset (with the score and the reason), every document that
describes it, its maintenance, incident and inspection history, and its
siblings -- which is where the duty/standby pair shows up as a link rather than
a merge.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from services.common import db, graph
from services.common.errors import NotFoundError
from services.common.schemas import (
    AssetDetailResponse,
    AssetDocument,
    AssetListResponse,
    AssetSummary,
    CapabilityState,
    CapabilityStatus,
    DataClass,
    DocumentType,
)

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("", response_model=AssetListResponse, summary="List canonical assets")
async def list_assets(
    q: str | None = Query(None, description="Substring match on the canonical tag"),
    class_code: str | None = Query(None, description="Equipment class code, e.g. P"),
    tag_kind: str | None = Query(None, description="equipment | instrument | line | kks"),
    multi_document_only: bool = Query(
        False, description="Only assets evidenced by more than one document"
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> AssetListResponse:
    filters: list[str] = []
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if q:
        filters.append("canonical_tag ILIKE %(q)s")
        params["q"] = f"%{q}%"
    if class_code:
        filters.append("class_code = %(class_code)s")
        params["class_code"] = class_code.upper()
    if tag_kind:
        filters.append("tag_kind = %(tag_kind)s")
        params["tag_kind"] = tag_kind
    if multi_document_only:
        filters.append("document_count > 1")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    total_row = await db.fetch_one(f"SELECT count(*)::int AS n FROM assets {where}", params)
    rows = await db.fetch_all(
        f"""
        SELECT asset_id, canonical_tag, tag_kind, class_code, class_label, description,
               functional_location, site, data_class::text AS data_class,
               mention_count, document_count, source_system_count, updated_at
          FROM assets {where}
         ORDER BY document_count DESC, mention_count DESC, canonical_tag
         LIMIT %(limit)s OFFSET %(offset)s
        """,
        params,
    )
    return AssetListResponse(
        total=int(total_row["n"]) if total_row else 0,
        limit=limit,
        offset=offset,
        items=[_summary(r) for r in rows],
    )


@router.get("/stats", summary="Entity-layer statistics (calculated metrics)")
async def asset_stats() -> dict[str, Any]:
    """Knowledge-graph linkage completeness, computed from stored rows.

    Every value here is a ``calculated_metric``: a deterministic aggregate over
    the mentions and documents actually ingested. Nothing is estimated.
    """
    totals = await db.fetch_one(
        """
        SELECT count(*)::int                                              AS total_assets,
               count(*) FILTER (WHERE document_count > 1)::int            AS multi_document_assets,
               count(*) FILTER (WHERE source_system_count > 1)::int       AS cross_system_assets,
               coalesce(sum(mention_count), 0)::int                       AS total_mentions
          FROM assets
        """
    )
    resolution = await db.fetch_one(
        """
        SELECT count(*)::int                                                  AS mentions,
               count(*) FILTER (WHERE resolved_asset_id IS NOT NULL)::int     AS resolved,
               count(*) FILTER (WHERE needs_review)::int                      AS needs_review
          FROM mentions
        """
    )
    by_class = await db.fetch_all(
        "SELECT coalesce(class_label, 'unclassified') AS class_label, count(*)::int AS n "
        "FROM assets GROUP BY 1 ORDER BY n DESC LIMIT 20"
    )
    counts: dict[str, Any] = dict(totals) if totals else {}
    mentions = int(resolution["mentions"]) if resolution else 0
    resolved = int(resolution["resolved"]) if resolution else 0
    total_assets = int(totals["total_assets"]) if totals else 0

    return {
        "data_class": DataClass.CALCULATED_METRIC.value,
        "total_assets": total_assets,
        "total_mentions": int(counts.get("total_mentions", 0)),
        "multi_document_assets": int(counts.get("multi_document_assets", 0)),
        "cross_system_assets": int(counts.get("cross_system_assets", 0)),
        "multi_document_pct": round(
            100.0 * counts.get("multi_document_assets", 0) / total_assets, 1
        )
        if total_assets
        else 0.0,
        "cross_system_pct": round(100.0 * counts.get("cross_system_assets", 0) / total_assets, 1)
        if total_assets
        else 0.0,
        "mention_resolution_rate_pct": round(100.0 * resolved / mentions, 1) if mentions else 0.0,
        "mentions_needing_review": int(resolution["needs_review"]) if resolution else 0,
        "by_class": [dict(r) for r in by_class],
    }


@router.get("/{asset_id}", response_model=AssetDetailResponse, summary="Full asset dossier")
async def get_asset(asset_id: str) -> AssetDetailResponse:
    # Accept either the internal id or the canonical tag: a technician has the
    # tag on the nameplate in front of them, not a database key.
    row = await db.fetch_one(
        """
        SELECT asset_id, canonical_tag, tag_kind, class_code, class_label, description,
               functional_location, site, data_class::text AS data_class,
               mention_count, document_count, source_system_count, updated_at
          FROM assets
         WHERE asset_id = %(key)s OR upper(canonical_tag) = upper(%(key)s)
        """,
        {"key": asset_id},
    )
    if not row:
        raise NotFoundError(
            f"No asset matching '{asset_id}'.",
            detail={
                "hint": "Assets exist only once a document mentioning them has been ingested. "
                "Check /api/v1/assets for what the corpus contains."
            },
        )

    tag_variants = await db.fetch_all(
        """
        SELECT DISTINCT ON (surface_form)
               surface_form, normalised, extractor, resolution_score, resolution_method,
               resolution_action::text AS resolution_action, needs_review, doc_id
          FROM mentions
         WHERE resolved_asset_id = %s
         ORDER BY surface_form, resolution_score DESC NULLS LAST
        """,
        (row["asset_id"],),
    )
    documents = await db.fetch_all(
        """
        SELECT d.doc_id, d.title, d.doc_type::text AS doc_type,
               d.data_class::text AS data_class, d.source_system, d.revision,
               d.issued_on, d.is_current, count(m.mention_id)::int AS mention_count
          FROM documents d
          JOIN mentions m ON m.doc_id = d.doc_id
         WHERE m.resolved_asset_id = %s
         GROUP BY d.doc_id
         ORDER BY d.issued_on DESC NULLS LAST, d.title
        """,
        (row["asset_id"],),
    )
    work_orders = await db.fetch_all(
        "SELECT wo_id, wo_type, status, description, as_found, coded_failure_mode, "
        "opened_on, closed_on, downtime_hours, cost, source_system, "
        "data_class::text AS data_class FROM work_orders WHERE asset_id = %s "
        "ORDER BY opened_on DESC NULLS LAST LIMIT 100",
        (row["asset_id"],),
    )
    incidents = await db.fetch_all(
        "SELECT incident_id, title, occurred_on, severity, event_type, immediate_cause, "
        "root_cause, investigation_status, source_system, data_class::text AS data_class "
        "FROM incidents WHERE asset_id = %s ORDER BY occurred_on DESC NULLS LAST LIMIT 100",
        (row["asset_id"],),
    )
    inspections = await db.fetch_all(
        "SELECT inspection_id, cml_id, method, inspected_on, thickness_mm, min_required_mm, "
        "inspector, finding, source_system, data_class::text AS data_class "
        "FROM inspections WHERE asset_id = %s ORDER BY inspected_on DESC NULLS LAST LIMIT 200",
        (row["asset_id"],),
    )

    siblings: list[dict[str, Any]] = []
    graph_status = CapabilityStatus(capability="knowledge_graph", state=CapabilityState.AVAILABLE)
    try:
        sibling_rows = await graph.read(
            "MATCH (e:Equipment {canonical_tag: $tag})-[r:SIBLING_OF]->(s:Equipment) "
            "RETURN s.canonical_tag AS canonical_tag, s.class_label AS class_label, "
            "r.reason AS reason, r.confidence AS confidence",
            tag=row["canonical_tag"],
        )
        siblings = [dict(s) for s in sibling_rows]
    except Exception as exc:
        graph_status = CapabilityStatus(
            capability="knowledge_graph",
            state=CapabilityState.ERROR,
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
        )

    return AssetDetailResponse(
        asset=_summary(row),
        tag_variants=[dict(v) for v in tag_variants],
        documents=[
            AssetDocument(
                doc_id=d["doc_id"],
                title=d["title"],
                doc_type=DocumentType(d["doc_type"]),
                data_class=DataClass(d["data_class"]),
                source_system=d["source_system"],
                revision=d["revision"],
                issued_on=d["issued_on"],
                is_current=d["is_current"],
                mention_count=d["mention_count"],
            )
            for d in documents
        ],
        work_orders=[dict(w) for w in work_orders],
        incidents=[dict(i) for i in incidents],
        inspections=[dict(i) for i in inspections],
        siblings=siblings,
        graph_available=graph_status,
    )


def _summary(row: dict[str, Any]) -> AssetSummary:
    return AssetSummary(
        asset_id=row["asset_id"],
        canonical_tag=row["canonical_tag"],
        tag_kind=row["tag_kind"],
        class_code=row["class_code"],
        class_label=row["class_label"],
        description=row["description"],
        functional_location=row["functional_location"],
        site=row["site"],
        data_class=DataClass(row["data_class"]),
        mention_count=row["mention_count"],
        document_count=row["document_count"],
        source_system_count=row["source_system_count"],
        updated_at=row["updated_at"],
    )
