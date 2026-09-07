"""Compliance evaluation: a requirement is satisfied only when evidence says so.

The claim "94% compliant" is the easiest number in this entire system to
fabricate and the most dangerous to get wrong, because it is read by people
deciding whether to intervene. So nothing here is asserted without a record
behind it, and the four testability modes are evaluated differently because they
can be:

``evidence_document``
    A record must exist, be about this asset, and be recent enough for the
    requirement's own stated frequency. This is a database query and its answer
    is a fact: the inspection either happened within twelve months or it did not.

``graph_state``
    A structural condition — "documents describing an asset shall be revised
    when an approved change alters it" is the *absence of an edge* after an MOC.
    Also a fact, and the kind only a graph can answer.

``procedure_text``
    A procedure must contain the obligation. Retrieval can find the candidate
    procedure and the cross-encoder can say how well it matches, but neither can
    confirm that a specific clause is *satisfied* by a specific paragraph. So
    this mode never returns "satisfied": it returns the candidate control with
    its evidence and asks a human. A similarity score is a search result, not a
    compliance finding.

``permit_record_field``
    Needs the permit-to-work system, which is not connected. Reported as
    requiring integration rather than assumed compliant or assumed failing.

Consequently ``coverage_pct`` counts only requirements this system can actually
decide. Requirements it cannot decide are reported separately and are never
quietly counted as passing — which is exactly how compliance dashboards come to
show green over an unexamined estate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from services.common import db, graph
from services.common.logging import get_logger

log = get_logger(__name__)

#: A control found by retrieval must reach this cross-encoder score before it is
#: even offered as a candidate. Below it, the procedure is not plausibly about
#: the obligation and offering it wastes a reviewer's time.
MIN_CONTROL_SCORE = 0.45

#: Concepts the inspection register can actually evidence. An obligation that
#: names none of them is not testable against inspection records, whatever else
#: is true of it.
_RECORD_CONCEPTS = re.compile(
    r"\b(inspect\w*|examin\w*|calibrat\w*|test\w*|survey\w*|thickness|monitor\w*|"
    r"measur\w*|readings?|condition)\b",
    re.I,
)

#: Grace period on a frequency-based requirement. An inspection due at twelve
#: months and done at twelve months and three days is late, not a breach, and
#: flagging it as one trains people to ignore the flag.
_GRACE_DAYS = 30


@dataclass(slots=True)
class RequirementFinding:
    req_id: str
    source_standard: str
    clause: str | None
    obligation_text: str
    modality: str | None
    testable_by: str
    text_status: str
    applies_to_class: str | None = None
    frequency_months: int | None = None

    #: satisfied | gap | needs_verification | not_evaluable
    state: str = "not_evaluable"
    reason: str = ""
    asset_tag: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    controls: list[dict[str, Any]] = field(default_factory=list)
    due_by: str | None = None
    owner: str | None = None

    @property
    def decidable(self) -> bool:
        """Whether this system can decide the requirement at all.

        The distinction ``coverage_pct`` rests on. A requirement needing a permit
        system that is not connected is neither satisfied nor breached — it is
        unexamined, and averaging it into a percentage either way is a lie.
        """
        return self.state in ("satisfied", "gap")

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "source_standard": self.source_standard,
            "clause": self.clause,
            "obligation_text": self.obligation_text,
            "modality": self.modality,
            "testable_by": self.testable_by,
            "text_status": self.text_status,
            "applies_to_class": self.applies_to_class,
            "frequency_months": self.frequency_months,
            "state": self.state,
            "reason": self.reason,
            "asset_tag": self.asset_tag,
            "evidence": self.evidence,
            "controls": self.controls,
            "due_by": self.due_by,
            "owner": self.owner,
        }


async def evaluate(
    *,
    asset_tag: str | None = None,
    standard: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Evaluate every requirement in scope against stored evidence."""
    today = today or date.today()
    requirements = await _requirements(standard)
    assets = await _assets_in_scope(asset_tag)

    findings: list[RequirementFinding] = []
    for req in requirements:
        applicable = _applicable_assets(req, assets)
        if req["applies_to_class"] and not applicable:
            findings.append(
                _finding(
                    req,
                    state="not_evaluable",
                    reason=(
                        f"Applies to equipment class '{req['applies_to_class']}', and no asset "
                        "of that class is in scope. Not counted as passing or failing."
                    ),
                )
            )
            continue

        for asset in applicable or [None]:
            findings.append(await _evaluate_one(req, asset, today))

    decidable = [f for f in findings if f.decidable]
    satisfied = [f for f in decidable if f.state == "satisfied"]
    gaps = [f for f in findings if f.state == "gap"]
    needs_verification = [f for f in findings if f.state == "needs_verification"]
    not_evaluable = [f for f in findings if f.state == "not_evaluable"]

    provenance: dict[str, int] = {}
    for req in requirements:
        provenance[req["text_status"]] = provenance.get(req["text_status"], 0) + 1

    return {
        "requirements_loaded": len(requirements),
        "assets_in_scope": [a["canonical_tag"] for a in assets],
        "findings_total": len(findings),
        "satisfied": len(satisfied),
        "gaps": len(gaps),
        "needs_verification": len(needs_verification),
        "not_evaluable": len(not_evaluable),
        # Only over what could be decided. Stated in the field name so a reader
        # cannot mistake it for coverage over the whole estate.
        "coverage_pct_of_decidable": (
            round(100.0 * len(satisfied) / len(decidable), 1) if decidable else None
        ),
        "decidable_count": len(decidable),
        "requirement_provenance": provenance,
        "findings": [f.to_dict() for f in findings],
    }


# ---------------------------------------------------------------------------
# Per-mode evaluation
# ---------------------------------------------------------------------------


async def _evaluate_one(
    req: dict[str, Any], asset: dict[str, Any] | None, today: date
) -> RequirementFinding:
    mode = req["testable_by"]
    if mode == "evidence_document":
        return await _by_evidence_document(req, asset, today)
    if mode == "graph_state":
        return await _by_graph_state(req, asset)
    if mode == "procedure_text":
        return await _by_procedure_text(req, asset)
    if mode == "permit_record_field":
        return _finding(
            req,
            asset=asset,
            state="not_evaluable",
            reason=(
                "Testable only against permit-to-work records. No permit system is connected, "
                "so this requirement is unexamined — neither satisfied nor breached. "
                "Set CMMS_BASE_URL / CMMS_API_KEY to enable."
            ),
        )
    return _finding(
        req,
        asset=asset,
        state="not_evaluable",
        reason=f"No evaluator implemented for testability mode '{mode}'.",
    )


async def _by_evidence_document(
    req: dict[str, Any], asset: dict[str, Any] | None, today: date
) -> RequirementFinding:
    """A record must exist, cover this asset, and be recent enough.

    The only mode that can return ``satisfied`` from data alone, because it is
    the only one whose obligation is a fact about whether something happened.
    """
    if asset is None:
        return _finding(
            req,
            state="not_evaluable",
            reason="Requires an asset in scope to check records against.",
        )

    # Does this system hold the *kind* of record this obligation is about?
    #
    # Without this check the evaluator reported three false gaps: "every
    # dangerous part of machinery shall be securely fenced" was failed against a
    # pump because the pump had no ultrasonic thickness readings. The absence of
    # a thickness survey says nothing about guarding. The obligation is real and
    # unmet-as-far-as-we-know, but that is "unexamined", not "breached" -- and a
    # compliance report that cries breach on category errors is one nobody reads.
    if not _RECORD_CONCEPTS.search(req["obligation_text"]):
        return _finding(
            req,
            asset=asset,
            state="not_evaluable",
            reason=(
                "This obligation is evidenced by a record type the system does not hold "
                "(no inspection, examination, calibration or test record corresponds to it). "
                "Reported as unexamined rather than as a breach."
            ),
        )

    rows = await db.fetch_all(
        """
        SELECT inspection_id, doc_id, method, inspected_on, cml_id, thickness_mm,
               min_required_mm, inspector, finding, data_class::text AS data_class
          FROM inspections
         WHERE asset_id = %s
         ORDER BY inspected_on DESC NULLS LAST
         LIMIT 25
        """,
        (asset["asset_id"],),
    )
    if not rows:
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason=(
                f"No inspection record of any kind exists for {asset['canonical_tag']}, so "
                "there is nothing to demonstrate this requirement was met."
            ),
        )

    latest = next((r for r in rows if r["inspected_on"]), None)
    evidence = [_inspection_evidence(r) for r in rows[:5]]

    if req["frequency_months"] is None:
        # An obligation with no stated interval is satisfied by the existence of
        # a record. "Readings shall be recorded against fixed CMLs" is met by
        # readings existing against CMLs, whenever they were taken.
        has_cml = any(r["cml_id"] for r in rows)
        if "cml" in req["obligation_text"].lower() or "condition monitoring" in req[
            "obligation_text"
        ].lower():
            if not has_cml:
                return _finding(
                    req,
                    asset=asset,
                    state="gap",
                    reason="Inspection records exist but none is recorded against a fixed CML.",
                    evidence=evidence,
                )
        return _finding(
            req,
            asset=asset,
            state="satisfied",
            reason=(
                f"{len(rows)} inspection record(s) on file for {asset['canonical_tag']}"
                + (f", most recent {latest['inspected_on']}" if latest else "")
                + "."
            ),
            evidence=evidence,
        )

    if latest is None:
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason="Inspection records exist but none carries a date, so currency cannot be shown.",
            evidence=evidence,
        )

    due = _add_months(latest["inspected_on"], req["frequency_months"])
    overdue_by = (today - due).days
    if overdue_by > _GRACE_DAYS:
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason=(
                f"Last inspection {latest['inspected_on']}; the requirement's "
                f"{req['frequency_months']}-month interval fell due {due.isoformat()}, "
                f"{overdue_by} days ago."
            ),
            evidence=evidence,
            due_by=due.isoformat(),
        )
    return _finding(
        req,
        asset=asset,
        state="satisfied",
        reason=(
            f"Last inspection {latest['inspected_on']} by {latest['inspector'] or 'unrecorded'}; "
            f"next due {due.isoformat()} on the {req['frequency_months']}-month interval."
        ),
        evidence=evidence,
        due_by=due.isoformat(),
    )


_STALE_AFTER_CHANGE = """
MATCH (m:MOC)-[c:CHANGED]->(e:Equipment)
WHERE $tag IS NULL OR e.canonical_tag = $tag
OPTIONAL MATCH (d:Document)-[:DESCRIBES]->(e)
WITH m, e, d
WHERE d IS NOT NULL
  AND (d.valid_from IS NULL OR m.approved_on IS NULL OR d.valid_from < m.approved_on)
RETURN m.moc_id AS moc_id, m.approved_on AS approved_on, m.change_desc AS change_desc,
       e.canonical_tag AS asset_tag,
       collect({doc_id: d.doc_id, title: d.title, revision: d.revision,
                valid_from: d.valid_from, is_current: d.is_current}) AS stale_documents
"""


async def _by_graph_state(req: dict[str, Any], asset: dict[str, Any] | None) -> RequirementFinding:
    """Structural obligations — the ones that are the absence of an edge.

    Only the change-control obligation is implemented. The others in the loaded
    set ("inspection intervals shall be set from a risk assessment", "corrective
    action effectiveness shall be reviewed") need records this corpus does not
    contain, and are reported as unexamined rather than assumed.
    """
    text = req["obligation_text"].lower()
    tag = asset["canonical_tag"] if asset else None

    if "revised when" in text or ("document" in text and "change" in text):
        rows = await graph.read(_STALE_AFTER_CHANGE, tag=tag)
        if not rows:
            return _finding(
                req,
                asset=asset,
                state="satisfied",
                reason=(
                    "No approved change was found that alters an asset whose describing "
                    "documents predate it."
                ),
            )
        offenders = [
            {
                "moc_id": r["moc_id"],
                "approved_on": _iso(r["approved_on"]),
                "change": r["change_desc"],
                "asset_tag": r["asset_tag"],
                "stale_documents": [
                    {**d, "valid_from": _iso(d.get("valid_from"))} for d in r["stale_documents"]
                ],
            }
            for r in rows
        ]
        doc_count = sum(len(o["stale_documents"]) for o in offenders)
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason=(
                f"{len(offenders)} approved change(s) altered equipment whose describing "
                f"documents were not revised afterwards: {doc_count} document(s) still predate "
                "the change."
            ),
            evidence=offenders,
        )

    return _finding(
        req,
        asset=asset,
        state="not_evaluable",
        reason=(
            "This obligation is testable against graph state, but the records it depends on "
            "(risk assessments, corrective-action effectiveness reviews) are not present in "
            "the corpus. Reported as unexamined."
        ),
    )


async def _by_procedure_text(
    req: dict[str, Any], asset: dict[str, Any] | None
) -> RequirementFinding:
    """Find the procedure that ought to contain this obligation.

    Uses the same retrieval stack the copilot uses, and stops short of the
    conclusion. Retrieval can establish "SOP-4412 section 3 is what this clause
    is about"; it cannot establish "and it satisfies the clause". The result is
    therefore always ``needs_verification`` with the candidate and its evidence
    — a starting point for a reviewer rather than a verdict.
    """
    from services.retrieval import lexical, rerank

    query = req["obligation_text"]
    try:
        rows = await lexical.search(query, top_k=12)
    except Exception as exc:
        log.error("compliance.control_search_failed", req_id=req["req_id"], error=str(exc))
        return _finding(
            req, asset=asset, state="not_evaluable", reason=f"Control search failed: {exc!s:.120}"
        )

    procedure_rows = [r for r in rows if r.get("doc_type") in ("sop", "manual", "permit")]
    if not procedure_rows:
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason=(
                "No procedure in the corpus matches this obligation, so no control was found "
                "that could satisfy it."
            ),
        )

    result = await rerank.get_reranker().rerank(query, [r["text"] for r in procedure_rows])
    ordered = (
        [procedure_rows[i] for i in result.order] if result.order else procedure_rows
    )
    scores = (
        [result.normalised(i) for i in result.order]
        if result.order
        else [0.0] * len(procedure_rows)
    )

    controls = [
        {
            "doc_id": row["doc_id"],
            "chunk_id": row["chunk_id"],
            "doc_title": row.get("title"),
            "section_path": row.get("section_path"),
            "page": row.get("page_from"),
            "snippet": (row.get("text") or "")[:400],
            "match_score": round(score, 4),
            "is_current": row.get("is_current"),
            "data_class": row.get("doc_data_class"),
        }
        for row, score in zip(ordered[:3], scores[:3], strict=False)
        if score >= MIN_CONTROL_SCORE
    ]
    if not controls:
        return _finding(
            req,
            asset=asset,
            state="gap",
            reason=(
                "Procedures were searched and none matched this obligation closely enough to "
                f"be offered as a control (best score below {MIN_CONTROL_SCORE})."
            ),
        )

    superseded = [c for c in controls if c["is_current"] is False]
    note = ""
    if superseded:
        note = (
            f" Note: {len(superseded)} candidate control(s) sit in a superseded revision, "
            "which would not satisfy the requirement even if the text matched."
        )
    return _finding(
        req,
        asset=asset,
        state="needs_verification",
        reason=(
            f"{len(controls)} candidate control(s) found by retrieval. Whether the text "
            "satisfies the obligation is a judgement this system does not make — a reviewer "
            "must confirm it." + note
        ),
        controls=controls,
    )


# ---------------------------------------------------------------------------
# Loading and helpers
# ---------------------------------------------------------------------------


async def _requirements(standard: str | None) -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT req_id, source_standard, clause, obligation_text, modality,
               applies_to_class, frequency_months, testable_by, text_status,
               provenance_note, effective_from, data_class::text AS data_class
          FROM requirements
         WHERE (%(std)s::text IS NULL OR source_standard = %(std)s::text)
         ORDER BY source_standard, clause, req_id
        """,
        {"std": standard},
    )


async def _assets_in_scope(asset_tag: str | None) -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT asset_id, canonical_tag, class_code, class_label, functional_location
          FROM assets
         WHERE (%(tag)s::text IS NULL OR upper(canonical_tag) = upper(%(tag)s::text))
         ORDER BY canonical_tag
        """,
        {"tag": asset_tag},
    )


def _applicable_assets(
    req: dict[str, Any], assets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Which assets a requirement actually bites on.

    A pressure-vessel clause evaluated against a pump is not a gap, it is a
    category error, and counting it as one is how compliance dashboards produce
    hundreds of meaningless findings.
    """
    if not req["applies_to_class"]:
        return assets[:1] if assets else []
    return [a for a in assets if a.get("class_code") == req["applies_to_class"]]


def _finding(
    req: dict[str, Any],
    *,
    asset: dict[str, Any] | None = None,
    state: str,
    reason: str,
    evidence: list[dict[str, Any]] | None = None,
    controls: list[dict[str, Any]] | None = None,
    due_by: str | None = None,
) -> RequirementFinding:
    return RequirementFinding(
        req_id=req["req_id"],
        source_standard=req["source_standard"],
        clause=req.get("clause"),
        obligation_text=req["obligation_text"],
        modality=req.get("modality"),
        testable_by=req["testable_by"],
        text_status=req["text_status"],
        applies_to_class=req.get("applies_to_class"),
        frequency_months=req.get("frequency_months"),
        state=state,
        reason=reason,
        asset_tag=asset["canonical_tag"] if asset else None,
        evidence=evidence or [],
        controls=controls or [],
        due_by=due_by,
    )


def _inspection_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "inspection_record",
        "inspection_id": row["inspection_id"],
        "doc_id": row.get("doc_id"),
        "method": row.get("method"),
        "inspected_on": _iso(row.get("inspected_on")),
        "cml_id": row.get("cml_id"),
        "thickness_mm": float(row["thickness_mm"]) if row.get("thickness_mm") else None,
        "min_required_mm": float(row["min_required_mm"]) if row.get("min_required_mm") else None,
        "inspector": row.get("inspector"),
        "finding": row.get("finding"),
        "data_class": row.get("data_class"),
    }


def _add_months(start: date, months: int) -> date:
    """Calendar-month arithmetic, clamped to the end of a short month."""
    year = start.year + (start.month - 1 + months) // 12
    month = (start.month - 1 + months) % 12 + 1
    day = min(start.day, [31, 29 if _leap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return date(year, month, day)


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "iso_format"):  # neo4j Date
        return value.iso_format()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
