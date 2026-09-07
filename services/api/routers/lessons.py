"""Lessons learned: precedent for a new event.

One endpoint, two ways in:

* ``POST /lessons`` — describe an event and get matching history.
* ``GET  /lessons/incidents`` — the incident register itself, since a panel that
  shows matches also needs to show what it is matching against.

The matching is explained rather than scored. Every result carries the signals
that produced it and a sentence per signal, because the alternative — a bare
percentage — is either taken on faith or ignored, and both are worse than a
figure the reader can check.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from services.agents import lessons as lessons_agent
from services.common import db
from services.common.logging import get_logger
from services.common.schemas import LessonsRequest

log = get_logger(__name__)
router = APIRouter(prefix="/lessons", tags=["reliability"])


@router.post("", summary="Find historical incidents resembling a new event")
async def find_lessons(request: LessonsRequest) -> dict[str, Any]:
    result = await lessons_agent.find_precedents(
        description=request.description,
        asset_tag=request.asset_tag,
        exclude_incident_id=request.exclude_incident_id,
        limit=request.limit,
    )
    return {
        "query": {
            "description": request.description,
            "asset_tag": request.asset_tag,
        },
        **result.to_dict(),
    }


@router.get("/incidents", summary="The incident register")
async def list_incidents(
    asset_tag: str | None = None,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT i.incident_id, i.doc_id, i.title, i.occurred_on, i.severity,
               i.immediate_cause, i.root_cause, i.investigation_status,
               i.raw_asset_tag, i.metadata, i.data_class::text AS data_class,
               a.canonical_tag AS asset_tag
          FROM incidents i
          LEFT JOIN assets a ON a.asset_id = i.asset_id
         WHERE (%(tag)s::text IS NULL OR upper(a.canonical_tag) = upper(%(tag)s::text))
         ORDER BY i.occurred_on DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"tag": asset_tag, "limit": limit},
    )
    items = []
    for row in rows:
        meta = row.pop("metadata", None) or {}
        row["corrective_actions"] = meta.get("corrective_actions") or []
        row["referenced_incidents"] = meta.get("referenced_incidents") or []
        row["referenced_procedures"] = meta.get("referenced_procedures") or []
        # Which expected fields the source report never stated. Surfaced rather
        # than hidden: "this investigation records no root cause" is a finding
        # about the plant's reporting, not a defect in the extractor.
        row["fields_missing"] = meta.get("fields_missing") or []
        row["evidence_chunks"] = meta.get("evidence_chunks") or []
        items.append(row)
    return {"total": len(items), "items": items}


@router.get("/incidents/{incident_id}", summary="One incident, in full")
async def get_incident(incident_id: str) -> dict[str, Any]:
    row = await db.fetch_one(
        """
        SELECT i.*, a.canonical_tag AS asset_tag
          FROM incidents i
          LEFT JOIN assets a ON a.asset_id = i.asset_id
         WHERE upper(i.incident_id) = upper(%s)
        """,
        (incident_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No incident {incident_id}")

    meta = row.get("metadata") or {}
    # An incident's own precedent: what this one resembles, excluding itself.
    cause = " ".join(filter(None, [row.get("root_cause"), row.get("immediate_cause")]))
    precedent = await lessons_agent.find_precedents(
        description=cause or row.get("title") or "",
        asset_tag=row.get("asset_tag"),
        exclude_incident_id=row["incident_id"],
        limit=5,
    )
    return {
        "incident": {k: v for k, v in row.items() if k != "metadata"},
        "corrective_actions": meta.get("corrective_actions") or [],
        "referenced_incidents": meta.get("referenced_incidents") or [],
        "referenced_procedures": meta.get("referenced_procedures") or [],
        "fields_missing": meta.get("fields_missing") or [],
        "evidence_chunks": meta.get("evidence_chunks") or [],
        "precedent": precedent.to_dict(),
    }
