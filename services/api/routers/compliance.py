"""Compliance and audit endpoints.

A compliance gap is literally a missing edge, which is why this feature belongs
on a graph. Three checks are distinguished, because they need different evidence
and carry different remediation:

``coverage gap``   a requirement with no control that satisfies it
``evidence gap``   a control exists, but its proof is missing or older than the
                   requirement's own frequency
``content gap``    a control exists and is inadequate -- the procedure covers the
                   general topic but omits a specific obligation

The first two are graph and date queries and run today. The third requires
natural-language inference over procedure text and is capability-gated: without
a provider it is reported as not configured rather than approximated by keyword
overlap, because "your document does not mention safety" is not a compliance
finding.

Requirement provenance is reported on every response. A compliance claim is only
as good as the traceability of the requirement it is checked against, so the
response states how many requirements in scope carry verbatim text and how many
are demo paraphrases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from fastapi import HTTPException

from services.agents import compliance as compliance_agent
from services.common import db
from services.common.logging import get_logger
from services.common.schemas import (
    CapabilityState,
    CapabilityStatus,
    ComplianceGap,
    ComplianceRequest,
    ComplianceResponse,
    DataClass,
)

log = get_logger(__name__)
router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.post("", response_model=ComplianceResponse, summary="Scan a scope for compliance gaps")
async def scan(request: ComplianceRequest) -> ComplianceResponse:
    scope = request.scope
    filters: list[str] = []
    params: dict[str, Any] = {}
    if scope.standard:
        filters.append("source_standard ILIKE %(standard)s")
        params["standard"] = f"%{scope.standard}%"
    if scope.asset_tag:
        filters.append(
            "(applies_to_class IS NULL OR applies_to_class = ("
            " SELECT class_code FROM assets WHERE upper(canonical_tag) = upper(%(asset_tag)s)))"
        )
        params["asset_tag"] = scope.asset_tag
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    requirements = await db.fetch_all(
        f"""
        SELECT req_id, source_standard, clause, obligation_text, modality,
               applies_to_class, frequency_months, testable_by, effective_from,
               text_status, provenance_note, data_class::text AS data_class
          FROM requirements {where}
         ORDER BY source_standard, clause
        """,
        params,
    )

    provenance: dict[str, int] = {}
    for row in requirements:
        provenance[row["text_status"]] = provenance.get(row["text_status"], 0) + 1

    if not requirements:
        return ComplianceResponse(
            scope=scope,
            status=CapabilityStatus(
                capability="compliance_scan",
                state=CapabilityState.NOT_CONFIGURED,
                detail=(
                    "No atomised requirements are loaded, so there is nothing to scan against. "
                    "Load a requirement set with `python scripts/load_requirements.py` "
                    "(see data/requirements/README.md for provenance rules) and re-run."
                ),
            ),
            requirements_in_scope=0,
            satisfied=0,
            partial=0,
            gaps=[],
            coverage_pct=None,
            requirement_provenance=provenance,
            evaluated_at=datetime.now(UTC),
        )

    # --- coverage and evidence gaps ----------------------------------------
    # The gap IS the absence of an edge. This is the real query: for each
    # requirement, is there a Control that SATISFIES it, and does that control
    # have Evidence proving it that is newer than the requirement's own
    # frequency? Controls and Evidence are asserted into the graph when a
    # procedure or record documenting them is ingested.
    control_coverage = await _control_coverage([row["req_id"] for row in requirements])

    gaps: list[ComplianceGap] = []
    for row in requirements:
        state = control_coverage.get(row["req_id"])
        if state and state["has_control"] and not state["evidence_stale"]:
            continue

        if state and state["has_control"]:
            gap_type: Any = "evidence_stale" if state["latest_evidence"] else "evidence_missing"
            detail = (
                f"A control satisfies this requirement, but its most recent evidence is "
                f"dated {state['latest_evidence']} and the requirement calls for proof every "
                f"{row['frequency_months']} months."
                if state["latest_evidence"]
                else "A control satisfies this requirement, but no evidence proving it exists."
            )
        else:
            gap_type = "no_control"
            detail = (
                "No control in the knowledge graph satisfies this requirement. Controls are "
                "asserted when a procedure or record documenting them is ingested."
            )

        gaps.append(
            ComplianceGap(
                req_id=row["req_id"],
                source_standard=row["source_standard"],
                clause=row["clause"],
                obligation_text=row["obligation_text"],
                modality=row["modality"],
                gap_type=gap_type,
                severity="high" if row["modality"] == "shall" else "medium",
                asset_tag=scope.asset_tag,
                latest_evidence=state["latest_evidence"] if state else None,
                detail=detail,
                data_class=DataClass.CALCULATED_METRIC,
            )
        )

    satisfied = len(requirements) - len(gaps)
    coverage_pct = round(100.0 * satisfied / len(requirements), 1) if requirements else None

    return ComplianceResponse(
        scope=scope,
        status=CapabilityStatus(
            capability="compliance_scan",
            state=CapabilityState.AVAILABLE,
            detail=(
                "Coverage-gap detection ran over the loaded requirement set. "
                "Content-gap detection (does the procedure text actually entail the "
                "obligation?) requires a generation provider and did not run; those "
                "requirements are reported as coverage gaps, not as satisfied."
            ),
        ),
        requirements_in_scope=len(requirements),
        satisfied=satisfied,
        partial=0,
        gaps=gaps[:200],
        coverage_pct=coverage_pct,
        requirement_provenance=provenance,
        evaluated_at=datetime.now(UTC),
    )


async def _control_coverage(req_ids: list[str]) -> dict[str, dict[str, Any]]:
    """For each requirement: is it satisfied by a control, and is its evidence current?

    Runs against Neo4j. Requirements with no incoming ``SATISFIES`` edge come
    back absent from the map, which the caller reports as a coverage gap.
    """
    if not req_ids:
        return {}
    from services.common import graph

    rows = await graph.read(
        """
        UNWIND $req_ids AS rid
        OPTIONAL MATCH (r:Requirement {req_id: rid})
        OPTIONAL MATCH (ctrl:Control)-[:SATISFIES]->(r)
        OPTIONAL MATCH (ev:Evidence)-[:PROVES]->(ctrl)
        WITH rid,
             r,
             count(DISTINCT ctrl) AS controls,
             max(ev.date) AS latest_evidence
        RETURN rid                                        AS req_id,
               controls > 0                               AS has_control,
               latest_evidence                            AS latest_evidence,
               coalesce(r.frequency_months, 12)           AS frequency_months
        """,
        req_ids=req_ids,
    )

    coverage: dict[str, dict[str, Any]] = {}
    today = datetime.now(UTC).date()
    for row in rows:
        latest = row["latest_evidence"]
        latest_date = latest.to_native() if hasattr(latest, "to_native") else latest
        stale = True
        if latest_date is not None:
            months = int(row["frequency_months"] or 12)
            age_days = (today - latest_date).days
            stale = age_days > months * 30
        coverage[row["req_id"]] = {
            "has_control": bool(row["has_control"]),
            "latest_evidence": latest_date,
            "evidence_stale": stale,
        }
    return coverage


@router.get("/requirements", summary="Loaded requirements with their provenance")
async def list_requirements(standard: str | None = None, limit: int = 200) -> dict[str, Any]:
    limit = max(1, min(limit, 1000))
    params: dict[str, Any] = {"limit": limit}
    where = ""
    if standard:
        where = "WHERE source_standard ILIKE %(standard)s"
        params["standard"] = f"%{standard}%"
    rows = await db.fetch_all(
        f"""
        SELECT req_id, source_standard, clause, obligation_text, modality,
               applies_to_class, frequency_months, testable_by, effective_from,
               text_status, provenance_note, data_class::text AS data_class
          FROM requirements {where}
         ORDER BY source_standard, clause
         LIMIT %(limit)s
        """,
        params,
    )
    by_standard = await db.fetch_all(
        "SELECT source_standard, count(*)::int AS n, "
        "count(*) FILTER (WHERE text_status = 'verbatim')::int AS verbatim "
        "FROM requirements GROUP BY 1 ORDER BY n DESC"
    )
    return {
        "total": len(rows),
        "items": [dict(r) for r in rows],
        "by_standard": [dict(r) for r in by_standard],
        "provenance_note": (
            "text_status distinguishes requirement text reproduced verbatim from a public "
            "source from text paraphrased for demonstration. Only 'verbatim' requirements "
            "should be used to assert a regulatory position."
        ),
    }


@router.post("/evidence-package", summary="Generate an audit evidence package")
async def evidence_package(request: ComplianceRequest) -> dict[str, Any]:
    """Audit evidence assembly.

    Not implemented in this build, and reported as such. An evidence package is
    an audit artefact: every entry must carry its source document id and hash,
    page reference, extraction method and confidence, and whether a human
    attested it. Emitting a document that looks like an audit package but is not
    traceable to that standard would be worse than emitting nothing.
    """
    return {
        "status": CapabilityStatus(
            capability="compliance_evidence_package",
            state=CapabilityState.NOT_IMPLEMENTED,
            detail=(
                "Evidence-package generation is not implemented in this build. The schema it "
                "will draw on exists (documents.content_hash, citations.quote_verified, "
                "requirements.text_status, data_class on every row), so every package entry "
                "will be traceable to a source page and labelled as machine inference or "
                "human attestation."
            ),
        ).model_dump(),
        "scope": request.scope.model_dump(),
        "package_url": None,
    }


# ---------------------------------------------------------------------------
# Evidence-backed evaluation (Day 4)
# ---------------------------------------------------------------------------


@router.get("/evaluate", summary="Evaluate requirements against stored evidence")
async def evaluate_requirements(
    asset_tag: str | None = None,
    standard: str | None = None,
) -> dict[str, Any]:
    """Per-requirement findings, each with the evidence behind the verdict.

    Distinct from ``POST /compliance``, which reports coverage gaps over the
    requirement set. This answers the narrower and more useful question: for this
    asset, which obligations can be shown to be met, which are demonstrably not,
    and which cannot be decided from what the system holds.
    """
    result = await compliance_agent.evaluate(asset_tag=asset_tag, standard=standard)
    return {
        "scope": {"asset_tag": asset_tag, "standard": standard},
        "status": {
            "capability": "compliance_evaluation",
            "state": "available",
            "detail": (
                "Requirements are evaluated per testability mode. Only evidence-document and "
                "graph-state obligations can be decided from stored records; procedure-text "
                "obligations return a candidate control for human verification, and "
                "permit-record obligations require a permit system that is not connected. "
                "coverage_pct_of_decidable is computed over decidable requirements only."
            ),
        },
        **result,
    }


@router.get("/requirements/{req_id}", summary="One requirement, with its provenance")
async def get_requirement(req_id: str) -> dict[str, Any]:
    row = await db.fetch_one(
        "SELECT * FROM requirements WHERE upper(req_id) = upper(%s)", (req_id,)
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"No requirement {req_id}")
    return {"requirement": row}
